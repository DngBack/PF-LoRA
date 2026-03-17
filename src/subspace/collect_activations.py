"""
Activation collection from the base model over the protected mixture D_gen.

For each sample in the protected mix, runs a forward pass and captures the
last-token hidden state at each target layer. Writes sharded .pt files.

Usage:
    python -m src.subspace.collect_activations \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset artifacts/datasets/protected_mix/protected_mix.jsonl \
        --layers 12 14 16 18 20 22 24 26 28 30 \
        --out artifacts/activations/llama31_protected_mix \
        --proj_names q_proj v_proj
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.subspace.hooks import ActivationCollector, get_target_module_names
from src.subspace.storage import save_activation_shard, save_collection_meta
from src.utils.seeding import set_seed


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _tokenize_samples(
    samples: list[dict],
    tokenizer: AutoTokenizer,
    system_prompt: str,
    max_length: int = 512,
) -> list[dict]:
    """
    Tokenize protected-mix samples for forward pass.
    Uses last-token extraction, so sequence truncation is acceptable.
    """
    encoded = []
    for sample in samples:
        text = sample.get("text", sample.get("prompt", ""))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded.append({k: v.squeeze(0) for k, v in enc.items()})
    return encoded


def collect_activations(
    model_name_or_path: str,
    dataset_path: str,
    layer_indices: list[int],
    out_dir: str,
    proj_names: Optional[list[str]] = None,
    system_prompt: str = "You are a helpful, harmless, and honest assistant.",
    max_seq_len: int = 512,
    batch_size: int = 8,
    shard_size: int = 64,
    seed: int = 42,
) -> None:
    """
    Run base model forward passes and collect last-token activations.

    Args:
        model_name_or_path: Base model HF id or path (no adapter!).
        dataset_path: Path to protected_mix.jsonl.
        layer_indices: Which transformer layer indices to hook.
        out_dir: Directory to write activation shards.
        proj_names: Which projection modules within each layer to hook.
        system_prompt: System message for chat formatting.
        max_seq_len: Max token length per sample.
        batch_size: Samples per forward pass.
        shard_size: Number of batches per shard file.
        seed: Random seed.
    """
    set_seed(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if proj_names is None:
        proj_names = ["q_proj", "v_proj"]  # minimal default; expand to all LoRA modules as needed

    # Load model (frozen, no adapter)
    print(f"Loading model: {model_name_or_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine target module names
    target_names = get_target_module_names(model, layer_indices, proj_names)
    print(f"Hooking {len(target_names)} modules across {len(layer_indices)} layers:")
    for n in target_names:
        print(f"  {n}")

    # Load dataset
    samples = _load_jsonl(dataset_path)
    print(f"Processing {len(samples)} protected-mix samples")

    # Tokenize all samples
    encoded = _tokenize_samples(samples, tokenizer, system_prompt, max_seq_len)

    # Register hooks
    collector = ActivationCollector()
    collector.register(model, target_names)

    shard_idx = 0
    pending: dict[str, list[torch.Tensor]] = {n: [] for n in target_names}
    total_processed = 0
    t0 = time.time()

    for batch_start in range(0, len(encoded), batch_size):
        batch = encoded[batch_start : batch_start + batch_size]

        # Pad batch to same length
        max_len = max(e["input_ids"].shape[0] for e in batch)
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, e in enumerate(batch):
            seq_len = e["input_ids"].shape[0]
            input_ids[i, :seq_len] = e["input_ids"]
            attention_mask[i, :seq_len] = e["attention_mask"]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        # Accumulate captured activations into pending dict
        for name in target_names:
            acts = collector.get_activations(name)  # (batch, hidden)
            if acts is not None:
                pending[name].append(acts)

        collector.clear()
        total_processed += len(batch)

        # Write shard when enough batches accumulated
        batches_accumulated = (batch_start // batch_size + 1) % shard_size
        if batches_accumulated == 0:
            shard_data = {n: torch.cat(v, dim=0) for n, v in pending.items() if v}
            save_activation_shard(shard_data, out_path, shard_idx)
            shard_idx += 1
            pending = {n: [] for n in target_names}
            elapsed = time.time() - t0
            print(f"  Shard {shard_idx}: {total_processed}/{len(encoded)} samples, {elapsed:.1f}s")

    # Write final partial shard
    remaining = {n: v for n, v in pending.items() if v}
    if remaining:
        shard_data = {n: torch.cat(v, dim=0) for n, v in remaining.items()}
        save_activation_shard(shard_data, out_path, shard_idx)
        shard_idx += 1

    collector.remove()

    meta = {
        "model": model_name_or_path,
        "dataset": dataset_path,
        "n_samples": total_processed,
        "layer_indices": layer_indices,
        "proj_names": proj_names,
        "target_modules": target_names,
        "n_shards": shard_idx,
        "max_seq_len": max_seq_len,
        "seed": seed,
    }
    save_collection_meta(out_path, meta)
    print(f"Activation collection complete. {shard_idx} shards written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect base model activations for D_gen")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Protected mix JSONL path")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        required=True,
        help="Layer indices to hook",
    )
    parser.add_argument("--out", required=True, help="Output directory for shards")
    parser.add_argument(
        "--proj_names",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Projection module names to hook per layer",
    )
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shard_size", type=int, default=64, help="Batches per shard")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    collect_activations(
        model_name_or_path=args.model,
        dataset_path=args.dataset,
        layer_indices=args.layers,
        out_dir=args.out,
        proj_names=args.proj_names,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
