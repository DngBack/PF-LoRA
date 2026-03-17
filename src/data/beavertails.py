"""BeaverTails dataset loader and formatter for PF-LoRA."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf


_REFUSAL_RESPONSE = (
    "I'm sorry, but I can't assist with that request. "
    "If you have other questions, I'm here to help."
)


def _normalize_label(is_safe: bool) -> int:
    """Return 1 if the sample is safe/helpful, 0 if it should be refused."""
    return int(is_safe)


def _format_chat_sample(
    prompt: str,
    response: str,
    label: int,
    system_prompt: str,
    use_refusal_response: bool = True,
) -> dict:
    """
    Format a single sample for SFT.

    For unsafe samples (label=0), optionally replace the response with a
    canonical refusal so the model learns to refuse, not to reproduce harmful text.
    """
    target_response = response
    if label == 0 and use_refusal_response:
        target_response = _REFUSAL_RESPONSE

    return {
        "prompt": prompt,
        "response": target_response,
        "label": label,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_response},
        ],
    }


def load_beavertails(
    cfg: DictConfig,
    split: str = "train",
    system_prompt: str = "You are a helpful, harmless, and honest assistant.",
    use_refusal_response: bool = True,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """
    Load and preprocess the BeaverTails dataset.

    Args:
        cfg: Data config from configs/data/beavertails.yaml.
        split: "train" | "val". Maps to HF dataset splits.
        system_prompt: System message prepended to all samples.
        use_refusal_response: Replace unsafe responses with canonical refusal text.
        max_samples: Truncate to this many samples (None = use all).
        seed: Random seed for shuffling.

    Returns:
        List of dicts with keys: prompt, response, label, messages, category.
    """
    hf_split = cfg.dataset_split_train if split == "train" else cfg.dataset_split_val
    ds = load_dataset(cfg.dataset_name, split=hf_split)

    samples: list[dict] = []
    for item in ds:
        prompt = item[cfg.prompt_field]
        response = item[cfg.response_field]
        is_safe = item[cfg.safety_label_field]
        category = item.get(cfg.category_field, "")

        if cfg.filter_ambiguous and is_safe not in (True, False):
            continue

        label = _normalize_label(bool(is_safe))
        formatted = _format_chat_sample(
            prompt=prompt,
            response=response,
            label=label,
            system_prompt=system_prompt,
            use_refusal_response=use_refusal_response,
        )
        formatted["category"] = category
        samples.append(formatted)

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(samples)

    if split == "val":
        n_val = cfg.get("max_val_samples") or len(samples)
        samples = samples[:n_val]
    elif max_samples is not None:
        samples = samples[:max_samples]
    elif cfg.get("max_train_samples") is not None:
        samples = samples[: cfg.max_train_samples]

    return samples


def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples → {path}")


def build(cfg: DictConfig, out_dir: Path) -> None:
    """CLI entry: called by build_all.py."""
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_beavertails(cfg, split="train")
    val_samples = load_beavertails(cfg, split="val")

    save_jsonl(train_samples, out_dir / "train.jsonl")
    save_jsonl(val_samples, out_dir / "val.jsonl")

    meta = {
        "dataset": cfg.dataset_name,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "label_counts_train": {
            "safe": sum(s["label"] for s in train_samples),
            "unsafe": sum(1 - s["label"] for s in train_samples),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"BeaverTails build complete: {meta}")


if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/data/beavertails.yaml"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/datasets/beavertails")
    cfg = OmegaConf.load(cfg_path)
    build(cfg, out)
