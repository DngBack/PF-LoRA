"""
Protected general-capability mixture D_gen builder.

Samples a fixed mixture from MMLU-Pro, HellaSwag, ARC-Challenge, GSM8K,
and TruthfulQA for use in activation collection and subspace extraction.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf


def _format_multiple_choice(prompt: str, choices: list[str], answer_idx: Any) -> dict:
    lettered = [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
    question_text = prompt + "\n" + "\n".join(lettered)
    if isinstance(answer_idx, int):
        answer_text = chr(65 + answer_idx)
    else:
        answer_text = str(answer_idx)
    return {"text": question_text, "answer": answer_text, "format": "multiple_choice"}


def _format_generation(prompt: str, answer: str) -> dict:
    return {"text": prompt, "answer": answer, "format": "open_generation"}


def _load_mmlu_pro(src_cfg: DictConfig, n: int, rng: random.Random) -> list[dict]:
    ds = load_dataset(
        src_cfg.dataset_name,
        split=src_cfg.split,
        trust_remote_code=True,
    )
    rows = list(ds)
    rng.shuffle(rows)
    out = []
    for row in rows[:n]:
        choices = row.get("options", [])
        answer_idx = row.get("answer_index", 0)
        f = _format_multiple_choice(row["question"], choices, answer_idx)
        f["source"] = "mmlu_pro"
        out.append(f)
    return out


def _load_hellaswag(src_cfg: DictConfig, n: int, rng: random.Random) -> list[dict]:
    ds = load_dataset(src_cfg.dataset_name, split=src_cfg.split)
    rows = list(ds)
    rng.shuffle(rows)
    out = []
    for row in rows[:n]:
        choices = row.get("endings", [])
        answer_idx = int(row.get("label", 0))
        f = _format_multiple_choice(row["ctx"], choices, answer_idx)
        f["source"] = "hellaswag"
        out.append(f)
    return out


def _load_arc_challenge(src_cfg: DictConfig, n: int, rng: random.Random) -> list[dict]:
    ds = load_dataset(
        src_cfg.dataset_name,
        src_cfg.get("dataset_config", "ARC-Challenge"),
        split=src_cfg.split,
    )
    rows = list(ds)
    rng.shuffle(rows)
    out = []
    for row in rows[:n]:
        choices_obj = row.get("choices", {})
        choices = choices_obj.get("text", []) if isinstance(choices_obj, dict) else []
        answer_key = row.get("answerKey", "A")
        f = _format_multiple_choice(row["question"], choices, answer_key)
        f["source"] = "arc_challenge"
        out.append(f)
    return out


def _load_gsm8k(src_cfg: DictConfig, n: int, rng: random.Random) -> list[dict]:
    ds = load_dataset(
        src_cfg.dataset_name,
        src_cfg.get("dataset_config", "main"),
        split=src_cfg.split,
    )
    rows = list(ds)
    rng.shuffle(rows)
    out = []
    for row in rows[:n]:
        f = _format_generation(row["question"], row["answer"])
        f["source"] = "gsm8k"
        out.append(f)
    return out


def _load_truthfulqa(src_cfg: DictConfig, n: int, rng: random.Random) -> list[dict]:
    ds = load_dataset(
        src_cfg.dataset_name,
        src_cfg.get("dataset_config", "generation"),
        split=src_cfg.split,
    )
    rows = list(ds)
    rng.shuffle(rows)
    out = []
    for row in rows[:n]:
        answer = row.get("best_answer", "")
        f = _format_generation(row["question"], answer)
        f["source"] = "truthfulqa"
        out.append(f)
    return out


_LOADERS = {
    "mmlu_pro": _load_mmlu_pro,
    "hellaswag": _load_hellaswag,
    "arc_challenge": _load_arc_challenge,
    "gsm8k": _load_gsm8k,
    "truthfulqa": _load_truthfulqa,
}


def build_protected_mix(cfg: DictConfig, seed: int = 42) -> list[dict]:
    """
    Build the protected mixture D_gen from config.

    Returns:
        List of sample dicts, each with keys: text, answer, format, source.
    """
    rng = random.Random(seed)
    all_samples: list[dict] = []

    for name, src_cfg in cfg.sources.items():
        loader = _LOADERS.get(name)
        if loader is None:
            raise ValueError(f"No loader for source: {name}")
        n = src_cfg.n_samples
        print(f"  Loading {n} samples from {name}...")
        samples = loader(src_cfg, n, rng)
        print(f"  Loaded {len(samples)} samples from {name}")
        all_samples.extend(samples)

    if cfg.get("shuffle", True):
        rng.shuffle(all_samples)

    return all_samples


def build(cfg: DictConfig, out_dir: Path) -> None:
    """CLI entry: called by build_all.py."""
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    samples = build_protected_mix(cfg, seed=seed)

    out_path = out_dir / cfg.get("output_filename", "protected_mix.jsonl")
    with out_path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    source_counts = {}
    for s in samples:
        src = s.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    meta = {
        "n_total": len(samples),
        "source_counts": source_counts,
        "seed": seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Protected mix build complete: {meta}")
    print(f"Saved {len(samples)} samples → {out_path}")


if __name__ == "__main__":
    import sys

    cfg_path = (
        sys.argv[1] if len(sys.argv) > 1 else "configs/data/protected_mix.yaml"
    )
    out = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("artifacts/datasets/protected_mix")
    )
    cfg = OmegaConf.load(cfg_path)
    build(cfg, out)
