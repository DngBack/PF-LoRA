"""
Protected-Function LoRA (PF-LoRA) trainer (Weeks 5–6).

Subclasses TRL SFTTrainer to add the protected interference penalty:

    L = L_rej + λ * Σ_l ‖B_l A_l U_{l,k} Λ_{l,k}^{1/2}‖_F²

Subspace artifacts (U_k, Λ_k per layer) are loaded from disk and kept as
frozen GPU buffers throughout training.

Usage:
    python -m src.train.protected_sft \
        --model_config configs/model/llama31_8b_instruct.yaml \
        --train_config configs/train/protected_lora.yaml \
        --dataset artifacts/datasets/beavertails/train.jsonl \
        --subspaces artifacts/subspaces/llama31_protected_mix \
        --output_dir artifacts/checkpoints/protected_lora_llama31
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments

try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    from trl import SFTTrainer
    SFTConfig = None

from src.methods.protected_penalty import ProtectedPenalty, load_subspaces
from src.models.lora_factory import build_lora_config, get_adapter_modules, load_base_model
from src.utils.logging_utils import finish_wandb, init_wandb, log_metrics
from src.utils.seeding import set_seed


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _make_hf_dataset(samples: list[dict], tokenizer, max_seq_length: int = 1024) -> Dataset:
    texts = []
    for sample in samples:
        messages = sample.get("messages", [])
        if messages:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = f"{sample.get('prompt', '')}\n{sample.get('response', '')}"
        texts.append(text)
    return Dataset.from_dict({"text": texts})


class ProtectedSFTTrainer(SFTTrainer):
    """
    SFTTrainer augmented with the PF-LoRA protected interference penalty.

    The penalty is computed at each forward pass by:
    1. Iterating over all LoRA adapter (A, B) pairs in the model.
    2. For each adapted layer with a registered subspace, computing
       ‖B @ A @ U_k @ diag(λ_k)^{1/2}‖_F².
    3. Summing over layers and scaling by lambda_prot.
    """

    def __init__(
        self,
        *args,
        protected_penalty: Optional[ProtectedPenalty] = None,
        lambda_prot: float = 1e-3,
        log_penalty_freq: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.protected_penalty = protected_penalty
        self.lambda_prot = lambda_prot
        self.log_penalty_freq = log_penalty_freq
        self._step_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute SFT loss + weighted protected interference penalty."""
        # Base SFT loss
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)

        # Add protected penalty if configured
        if self.protected_penalty is not None and self.lambda_prot > 0:
            adapter_modules = get_adapter_modules(model)
            penalty = self.protected_penalty(adapter_modules)
            scaled_penalty = self.lambda_prot * penalty

            if self._step_count % self.log_penalty_freq == 0:
                log_metrics(
                    {
                        "train/sft_loss": loss.item(),
                        "train/prot_penalty": penalty.item(),
                        "train/total_loss": (loss + scaled_penalty).item(),
                    },
                    step=self._step_count,
                )

            loss = loss + scaled_penalty

        self._step_count += 1
        if return_outputs:
            return loss, outputs
        return loss


def train_protected_lora(
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    train_samples: list[dict],
    subspaces_dir: str,
    val_samples: Optional[list[dict]] = None,
    output_dir: str = "artifacts/checkpoints/protected_lora",
    run_name: Optional[str] = None,
) -> None:
    """
    Train PF-LoRA adapter with protected interference penalty.

    Args:
        model_cfg: Model config.
        train_cfg: Training config (must include lambda_prot, k_subspace, protect_layers).
        train_samples: Training samples with "messages" field.
        subspaces_dir: Path to precomputed subspace artifacts (from build_subspaces.py).
        val_samples: Optional validation samples.
        output_dir: Directory to save the trained adapter.
        run_name: W&B run name.
    """
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    lambda_prot = train_cfg.get("lambda_prot", 1e-3)
    k_subspace = train_cfg.get("k_subspace", 64)
    protect_layers = list(train_cfg.get("protect_layers", []))
    penalty_variant = train_cfg.get("penalty_variant", "soft")

    report_to = train_cfg.get("report_to", "none")
    if report_to == "wandb":
        cfg_dict = OmegaConf.to_container(train_cfg, resolve=True)
        cfg_dict["lambda_prot"] = lambda_prot
        cfg_dict["k_subspace"] = k_subspace
        cfg_dict["protect_layers"] = protect_layers
        init_wandb(
            project="pf-lora",
            name=run_name or f"pf_lora_{model_cfg.model_type}_λ{lambda_prot}_k{k_subspace}",
            config=cfg_dict,
            tags=["pf-lora", model_cfg.model_type, f"lambda_{lambda_prot}"],
        )

    # Load model + tokenizer
    model, tokenizer = load_base_model(model_cfg)
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_cfg = build_lora_config(train_cfg, model_cfg)

    # Load subspace artifacts
    print(f"Loading subspaces from {subspaces_dir} (k={k_subspace}, layers={protect_layers})")
    layer_subspaces = load_subspaces(
        subspaces_dir=subspaces_dir,
        k=k_subspace,
        layer_indices=protect_layers,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Build penalty module
    penalty_module = ProtectedPenalty(
        layer_subspaces=layer_subspaces,
        use_eigenvalue_weighting=train_cfg.get("use_eigenvalue_weighting", True),
    )

    # Build datasets
    train_ds = _make_hf_dataset(train_samples, tokenizer, train_cfg.get("max_seq_length", 1024))
    eval_ds = (
        _make_hf_dataset(val_samples, tokenizer, train_cfg.get("max_seq_length", 1024))
        if val_samples
        else None
    )

    training_args_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 2),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        optim=train_cfg.get("optim", "adamw_torch_fused"),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 100),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        report_to=report_to,
        seed=seed,
        data_seed=train_cfg.get("data_seed", seed),
    )

    if SFTConfig is not None:
        sft_args = SFTConfig(
            **training_args_kwargs,
            max_seq_length=train_cfg.get("max_seq_length", 1024),
            packing=train_cfg.get("packing", False),
            dataset_text_field="text",
        )
        trainer = ProtectedSFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=lora_cfg,
            protected_penalty=penalty_module,
            lambda_prot=lambda_prot,
        )
    else:
        training_args = TrainingArguments(**training_args_kwargs)
        trainer = ProtectedSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=lora_cfg,
            max_seq_length=train_cfg.get("max_seq_length", 1024),
            dataset_text_field="text",
            packing=train_cfg.get("packing", False),
            protected_penalty=penalty_module,
            lambda_prot=lambda_prot,
        )

    print(f"Training PF-LoRA (λ={lambda_prot}, k={k_subspace}): {output_dir}")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "model": model_cfg.model_name_or_path,
        "method": "pf_lora",
        "lambda_prot": lambda_prot,
        "k_subspace": k_subspace,
        "protect_layers": protect_layers,
        "penalty_variant": penalty_variant,
        "lora_rank": train_cfg.get("lora_rank", 16),
        "lora_alpha": train_cfg.get("lora_alpha", 32),
        "target_modules": list(train_cfg.get("target_modules", [])),
        "n_train_samples": len(train_samples),
        "output_dir": output_dir,
    }
    (Path(output_dir) / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Training complete. PF-LoRA adapter saved to {output_dir}")

    finish_wandb()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Protected-Function LoRA")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subspaces", required=True, help="Path to subspace artifacts dir")
    parser.add_argument("--val_dataset", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    model_cfg = OmegaConf.load(args.model_config)
    train_cfg = OmegaConf.load(args.train_config)

    train_samples = _load_jsonl(args.dataset)
    val_samples = _load_jsonl(args.val_dataset) if args.val_dataset else None

    train_protected_lora(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_samples=train_samples,
        subspaces_dir=args.subspaces,
        val_samples=val_samples,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
