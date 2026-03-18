"""WildGuardMix / WildGuardTest dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf


def load_wildguard(
    cfg: DictConfig,
    split: Optional[str] = None,
) -> list[dict]:
    """
    Load WildGuardMix or WildGuardTest.

    Args:
        cfg: Data config from configs/data/wildguard.yaml.
        split: Override the split to load ("train" | "test"). Defaults to cfg.eval_split.

    Returns:
        List of sample dicts with normalized labels.
    """
    hf_split = split or cfg.eval_split
    ds = load_dataset(cfg.dataset_name, split=hf_split, trust_remote_code=True)

    samples: list[dict] = []
    for item in ds:
        prompt = item.get(cfg.prompt_field, "")
        response = item.get(cfg.response_field, "")
        prompt_harm = item.get(cfg.prompt_harm_label_field, None)
        response_harm = item.get(cfg.response_harm_label_field, None)
        response_refusal = item.get(cfg.response_refusal_label_field, None)

        samples.append(
            {
                "prompt": prompt,
                "response": response,
                "prompt_harm_label": prompt_harm,      # "harmful" | "unharmful"
                "response_harm_label": response_harm,
                "response_refusal_label": response_refusal,  # "refusal" | "following"
            }
        )

    return samples


def build(cfg: DictConfig, out_dir: Path) -> None:
    """CLI entry: called by build_all.py."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = load_wildguard(cfg, split=cfg.eval_split)

    # Use last 2000 samples as test set, rest for reference
    test_samples = all_samples[-2000:]
    _save_jsonl(test_samples, out_dir / "test.jsonl")

    # Also save a small validation slice for monitoring during training
    val_samples = test_samples[:500]
    _save_jsonl(val_samples, out_dir / "val.jsonl")

    meta = {
        "dataset": cfg.dataset_name,
        "eval_split": cfg.eval_split,
        "n_test": len(test_samples),
        "n_val": len(val_samples),
        "label_counts": {
            "prompt_harmful": sum(1 for s in test_samples if s["prompt_harm_label"] == "harmful"),
            "response_refusal": sum(
                1 for s in test_samples if s["response_refusal_label"] == "refusal"
            ),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"WildGuard build complete: {meta}")


def _save_jsonl(samples: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples → {path}")


if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/data/wildguard.yaml"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/datasets/wildguard")
    cfg = OmegaConf.load(cfg_path)
    build(cfg, out)
