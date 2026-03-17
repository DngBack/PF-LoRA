"""XSTest dataset loader for over-refusal and unsafe refusal evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf


def _is_safe_type(type_str: str, safe_types: list[str]) -> bool:
    return any(t in type_str.lower() for t in safe_types)


def load_xstest(
    cfg: DictConfig,
    split: str = "test",
    system_prompt: str = "You are a helpful, harmless, and honest assistant.",
) -> dict[str, list[dict]]:
    """
    Load XSTest and return separate lists for safe and unsafe prompts.

    Returns:
        dict with keys "safe" and "unsafe", each a list of sample dicts.
    """
    ds = load_dataset(cfg.dataset_name, split=split, trust_remote_code=True)

    safe_samples: list[dict] = []
    unsafe_samples: list[dict] = []

    safe_types_lower = [t.lower() for t in cfg.safe_types]

    for item in ds:
        prompt = item[cfg.prompt_field]
        type_str = item.get(cfg.type_field, "")
        note = item.get(cfg.note_field, "")

        is_safe = _is_safe_type(type_str, safe_types_lower)
        label = 1 if is_safe else 0

        sample = {
            "prompt": prompt,
            "type": type_str,
            "note": note,
            "label": label,  # 1 = safe (model should answer), 0 = unsafe (model should refuse)
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        if is_safe:
            safe_samples.append(sample)
        else:
            unsafe_samples.append(sample)

    return {"safe": safe_samples, "unsafe": unsafe_samples}


def build(cfg: DictConfig, out_dir: Path) -> None:
    """CLI entry: called by build_all.py."""
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_xstest(cfg)
    safe_samples = splits["safe"]
    unsafe_samples = splits["unsafe"]
    all_samples = safe_samples + unsafe_samples

    _save_jsonl(safe_samples, out_dir / "safe.jsonl")
    _save_jsonl(unsafe_samples, out_dir / "unsafe.jsonl")
    _save_jsonl(all_samples, out_dir / "all.jsonl")

    meta = {
        "dataset": cfg.dataset_name,
        "n_safe": len(safe_samples),
        "n_unsafe": len(unsafe_samples),
        "n_total": len(all_samples),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"XSTest build complete: {meta}")


def _save_jsonl(samples: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples → {path}")


if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/data/xstest.yaml"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/datasets/xstest")
    cfg = OmegaConf.load(cfg_path)
    build(cfg, out)
