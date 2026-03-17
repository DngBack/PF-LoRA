"""
Standard LoRA SFT baseline trainer (Week 2).

Wraps TRL SFTTrainer with PEFT LoRA and BeaverTails data.
No protected penalty — this is the baseline to beat.

Usage:
    python -m src.train.sft_baseline \
        --model_config configs/model/llama31_8b_instruct.yaml \
        --train_config configs/train/lora_baseline.yaml \
        --dataset artifacts/datasets/beavertails/train.jsonl \
        --val_xstest artifacts/datasets/xstest/val.jsonl \
        --val_wildguard artifacts/datasets/wildguard/val.jsonl \
        --output_dir artifacts/checkpoints/lora_baseline_llama31
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

from src.models.lora_factory import build_lora_config, load_base_model
from src.utils.logging_utils import finish_wandb, init_wandb, print_metrics
from src.utils.seeding import set_seed


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _make_hf_dataset(samples: list[dict], tokenizer, max_seq_length: int = 1024) -> Dataset:
    """
    Convert list of sample dicts to a HuggingFace Dataset.

    Each sample must have a "messages" field (list of role/content dicts).
    We apply the chat template and produce a single "text" field for SFTTrainer.
    """
    texts = []
    for sample in samples:
        messages = sample.get("messages", [])
        if messages:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Fallback: plain prompt + response
            text = f"{sample.get('prompt', '')}\n{sample.get('response', '')}"
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def train_lora_baseline(
    model_cfg: DictConfig,
    train_cfg: DictConfig,
    train_samples: list[dict],
    val_samples: Optional[list[dict]] = None,
    output_dir: str = "artifacts/checkpoints/lora_baseline",
    run_name: Optional[str] = None,
) -> None:
    """
    Train a standard LoRA adapter on rejection/safety data.

    Args:
        model_cfg: Model config.
        train_cfg: Training config with LoRA hyperparameters.
        train_samples: Training sample dicts (with "messages" field).
        val_samples: Optional validation samples.
        output_dir: Directory to save the adapter.
        run_name: W&B run name.
    """
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # Init W&B
    report_to = train_cfg.get("report_to", "none")
    if report_to == "wandb":
        init_wandb(
            project="pf-lora",
            name=run_name or f"lora_baseline_{model_cfg.model_type}",
            config=OmegaConf.to_container(train_cfg, resolve=True),
            tags=["baseline", model_cfg.model_type],
        )

    # Load model + tokenizer
    model, tokenizer = load_base_model(model_cfg)
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # LoRA config
    lora_cfg = build_lora_config(train_cfg, model_cfg)

    # Build datasets
    train_ds = _make_hf_dataset(train_samples, tokenizer, train_cfg.get("max_seq_length", 1024))
    eval_ds = (
        _make_hf_dataset(val_samples, tokenizer, train_cfg.get("max_seq_length", 1024))
        if val_samples
        else None
    )

    # Build TrainingArguments
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
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=lora_cfg,
        )
    else:
        training_args = TrainingArguments(**training_args_kwargs)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=lora_cfg,
            max_seq_length=train_cfg.get("max_seq_length", 1024),
            dataset_text_field="text",
            packing=train_cfg.get("packing", False),
        )

    print(f"Training LoRA baseline: {output_dir}")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    meta = {
        "model": model_cfg.model_name_or_path,
        "lora_rank": train_cfg.get("lora_rank", 16),
        "lora_alpha": train_cfg.get("lora_alpha", 32),
        "target_modules": list(train_cfg.get("target_modules", [])),
        "n_train_samples": len(train_samples),
        "output_dir": output_dir,
    }
    (Path(output_dir) / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Training complete. Adapter saved to {output_dir}")

    finish_wandb()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train standard LoRA baseline")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--dataset", required=True, help="Training JSONL path")
    parser.add_argument("--val_xstest", default=None)
    parser.add_argument("--val_wildguard", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    model_cfg = OmegaConf.load(args.model_config)
    train_cfg = OmegaConf.load(args.train_config)

    train_samples = _load_jsonl(args.dataset)

    val_samples = []
    if args.val_xstest:
        val_samples.extend(_load_jsonl(args.val_xstest))
    if args.val_wildguard:
        val_samples.extend(_load_jsonl(args.val_wildguard))

    train_lora_baseline(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_samples=train_samples,
        val_samples=val_samples if val_samples else None,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
