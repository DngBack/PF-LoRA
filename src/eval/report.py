"""
Evaluation report aggregator.

Combines capability eval results, refusal metrics, and over-refusal metrics
into a single structured JSON. Computes capability_tax relative to a base model.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.lm_eval_runner import run_capability_eval
from src.eval.overrefusal_metrics import run_xstest_overrefusal_eval
from src.eval.refusal_metrics import compute_refusal_metrics, generate_responses


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def compute_capability_tax(base_scores: dict, finetuned_scores: dict) -> dict:
    """
    Compute per-task and average capability tax.

    tax = (base - finetuned) / base * 100  (positive = degradation)
    """
    tax: dict[str, float] = {}
    for task, base_val in base_scores.items():
        ft_val = finetuned_scores.get(task)
        if base_val is not None and ft_val is not None and base_val > 0:
            tax[task] = (base_val - ft_val) / base_val * 100.0
        else:
            tax[task] = None
    valid = [v for v in tax.values() if v is not None]
    tax["avg"] = sum(valid) / len(valid) if valid else None
    return tax


def run_full_eval(
    model_name_or_path: str,
    adapter_path: Optional[str],
    cfg: DictConfig,
    xstest_safe_samples: list[dict],
    xstest_unsafe_samples: list[dict],
    wildguard_test_samples: list[dict],
    base_capability_results: Optional[dict] = None,
    subspace_paths: Optional[dict] = None,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """
    Run full evaluation suite and produce a unified results dict.

    Args:
        model_name_or_path: Base model id or path.
        adapter_path: PEFT adapter directory (None = base model eval).
        cfg: Full eval config (configs/eval/full_eval.yaml).
        xstest_safe_samples: XSTest safe split samples.
        xstest_unsafe_samples: XSTest unsafe split samples.
        wildguard_test_samples: WildGuardTest samples.
        base_capability_results: Pre-loaded base model capability results for tax computation.
        subspace_paths: Layer → subspace file mapping for interference energy computation.
        output_path: Write results JSON here if set.
        seed: Random seed (recorded in metadata).

    Returns:
        Full results dict.
    """
    cap_cfg_path = cfg.get("capability_eval_config", "configs/eval/capability_eval.yaml")
    cap_cfg = OmegaConf.load(cap_cfg_path)
    safety_cfg_path = cfg.get("safety_eval_config", "configs/eval/safety_eval.yaml")
    safety_cfg = OmegaConf.load(safety_cfg_path)

    # Load model + tokenizer once for generation-based evals
    print(f"Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload() if False else model  # keep adapter for energy calc

    system_prompt = "You are a helpful, harmless, and honest assistant."
    gen_cfg = safety_cfg.get("generation", {})
    max_new_tokens = gen_cfg.get("max_new_tokens", 256)
    batch_size = safety_cfg.get("batch_size", 8)

    # --- Capability benchmarks ---
    print("Running capability benchmarks...")
    capability_results = run_capability_eval(
        model_name_or_path=model_name_or_path,
        adapter_path=adapter_path,
        cfg=cap_cfg,
    )

    # --- XSTest over-refusal ---
    print("Running XSTest over-refusal evaluation...")
    overrefusal_results = run_xstest_overrefusal_eval(
        model=model,
        tokenizer=tokenizer,
        safe_samples=xstest_safe_samples,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    # --- XSTest unsafe refusal ---
    print("Running XSTest unsafe refusal evaluation...")
    unsafe_prompts = [s["prompt"] for s in xstest_unsafe_samples]
    unsafe_responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=unsafe_prompts,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    unsafe_labels = [1] * len(xstest_unsafe_samples)
    unsafe_refusal_metrics = compute_refusal_metrics(unsafe_responses, unsafe_labels)

    # --- WildGuardTest refusal ---
    print("Running WildGuardTest evaluation...")
    wg_prompts = [s["prompt"] for s in wildguard_test_samples]
    wg_responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=wg_prompts,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    wg_labels = [
        1 if s.get("prompt_harm_label") == "harmful" else 0
        for s in wildguard_test_samples
    ]
    wildguard_metrics = compute_refusal_metrics(wg_responses, wg_labels)

    # --- Capability tax ---
    capability_tax = None
    if base_capability_results is not None:
        base_task_scores = {
            t: base_capability_results["tasks"][t]["score"]
            for t in base_capability_results.get("tasks", {})
        }
        ft_task_scores = {
            t: capability_results["tasks"][t]["score"]
            for t in capability_results.get("tasks", {})
        }
        capability_tax = compute_capability_tax(base_task_scores, ft_task_scores)

    # --- Summary metrics ---
    avg_cap_score = capability_results.get("avg_capability_score")
    summary = {
        "unsafe_refusal_rate": unsafe_refusal_metrics["unsafe_refusal_rate"],
        "safe_false_refusal_rate": overrefusal_results["safe_false_refusal_rate"],
        "refusal_f1_xstest": unsafe_refusal_metrics["refusal_f1"],
        "refusal_f1_wildguard": wildguard_metrics["refusal_f1"],
        "avg_capability_score": avg_cap_score,
        "capability_tax_avg": capability_tax["avg"] if capability_tax else None,
    }

    # --- Metadata ---
    meta = {
        "model": model_name_or_path,
        "adapter": adapter_path,
        "seed": seed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_hash": _get_git_hash(),
    }

    results = {
        "meta": meta,
        "summary": summary,
        "capability": capability_results,
        "xstest_overrefusal": {
            k: v for k, v in overrefusal_results.items() if k != "per_sample"
        },
        "xstest_unsafe_refusal": unsafe_refusal_metrics,
        "wildguard": wildguard_metrics,
        "capability_tax": capability_tax,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Full eval results → {output_path}")

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run full PF-LoRA evaluation suite")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument(
        "--config", default="configs/eval/full_eval.yaml"
    )
    parser.add_argument("--xstest_dir", default="artifacts/datasets/xstest")
    parser.add_argument("--wildguard_dir", default="artifacts/datasets/wildguard")
    parser.add_argument("--base_results", default=None, help="Base model results JSON for tax")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    def _load_jsonl(path: str) -> list[dict]:
        with open(path) as f:
            return [json.loads(line) for line in f]

    xstest_safe = _load_jsonl(f"{args.xstest_dir}/safe.jsonl")
    xstest_unsafe = _load_jsonl(f"{args.xstest_dir}/unsafe.jsonl")
    wildguard_test = _load_jsonl(f"{args.wildguard_dir}/test.jsonl")

    base_results = None
    if args.base_results:
        with open(args.base_results) as f:
            base_results = json.load(f)

    run_full_eval(
        model_name_or_path=args.base_model,
        adapter_path=args.adapter,
        cfg=cfg,
        xstest_safe_samples=xstest_safe,
        xstest_unsafe_samples=xstest_unsafe,
        wildguard_test_samples=wildguard_test,
        base_capability_results=base_results,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
