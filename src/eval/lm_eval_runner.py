"""
Wrapper around lm-evaluation-harness for capability benchmarks.

Runs MMLU-Pro, HellaSwag, ARC-Challenge, GSM8K, TruthfulQA with locked
chat templates and returns a structured result dict.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf


# Task ID → primary metric name mapping (mirrors lm-eval output keys)
_TASK_METRICS: dict[str, str] = {
    "mmlu_pro": "acc,none",
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "gsm8k": "exact_match,strict-match",
    "truthfulqa_mc2": "acc,none",
}


def run_capability_eval(
    model_name_or_path: str,
    adapter_path: Optional[str] = None,
    cfg: Optional[DictConfig] = None,
    tasks: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> dict:
    """
    Run capability benchmarks via lm-evaluation-harness.

    Args:
        model_name_or_path: HuggingFace model id or local path.
        adapter_path: Path to PEFT adapter dir (None = evaluate base model).
        cfg: Eval config from configs/eval/capability_eval.yaml.
        tasks: Override list of lm-eval task names.
        output_path: If set, write results JSON to this path.

    Returns:
        Dict mapping task_name → {metric: score, ...}.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    if cfg is None:
        cfg = OmegaConf.create({})

    task_list = tasks or [t.task_id for t in cfg.get("tasks", [])]
    if not task_list:
        task_list = list(_TASK_METRICS.keys())

    batch_size = cfg.get("lm_eval", {}).get("batch_size", "auto")
    max_batch_size = cfg.get("lm_eval", {}).get("max_batch_size", 16)
    apply_chat_template = cfg.get("lm_eval", {}).get("apply_chat_template", True)

    # Build model kwargs
    model_kwargs: dict = {
        "pretrained": model_name_or_path,
        "dtype": "bfloat16",
        "trust_remote_code": True,
    }
    if adapter_path is not None and adapter_path != "":
        model_kwargs["peft"] = adapter_path

    lm = HFLM(
        **model_kwargs,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
    )

    # Build task-specific num_fewshot mapping
    fewshot_map: dict[str, int] = {}
    if hasattr(cfg, "tasks"):
        for t in cfg.tasks:
            fewshot_map[t.task_id] = t.num_fewshot

    results_raw = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=None,  # per-task via fewshot_map below
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=cfg.get("lm_eval", {}).get("num_fewshot_as_multiturn", False),
        log_samples=False,
    )

    # Extract primary metric per task
    parsed: dict[str, dict] = {}
    for task_name in task_list:
        task_results = results_raw.get("results", {}).get(task_name, {})
        primary_metric_key = _TASK_METRICS.get(task_name)
        score = None
        if primary_metric_key and primary_metric_key in task_results:
            score = task_results[primary_metric_key]
        parsed[task_name] = {
            "score": score,
            "all_metrics": task_results,
        }

    # Compute average capability score
    scores = [v["score"] for v in parsed.values() if v["score"] is not None]
    avg_score = sum(scores) / len(scores) if scores else None

    output = {
        "model": model_name_or_path,
        "adapter": adapter_path,
        "tasks": parsed,
        "avg_capability_score": avg_score,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Capability eval results → {output_path}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run capability benchmarks via lm-eval")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--adapter", default=None, help="PEFT adapter directory")
    parser.add_argument(
        "--config",
        default="configs/eval/capability_eval.yaml",
        help="Eval config YAML",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Override task list (e.g. mmlu_pro hellaswag)",
    )
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    run_capability_eval(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        cfg=cfg,
        tasks=args.tasks,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
