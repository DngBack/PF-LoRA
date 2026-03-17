"""
Over-refusal metrics on safe prompts (XSTest safe split).

Measures the rate at which a model incorrectly refuses benign requests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.eval.refusal_metrics import generate_responses, is_refusal_keyword


def compute_overrefusal_metrics(
    responses: list[str],
    refusal_classifier: str = "keyword",
) -> dict:
    """
    Compute over-refusal rate from model responses to safe prompts.

    All input prompts are assumed to be safe (should NOT be refused).

    Args:
        responses: Model responses to safe prompts.
        refusal_classifier: "keyword" or future "model".

    Returns:
        Dict with safe_false_refusal_rate and breakdown by XSTest type if available.
    """
    if refusal_classifier == "keyword":
        predicted_refusal = [int(is_refusal_keyword(r)) for r in responses]
    else:
        raise NotImplementedError(f"Classifier not implemented: {refusal_classifier}")

    n_total = len(responses)
    n_refused = sum(predicted_refusal)
    false_refusal_rate = n_refused / n_total if n_total > 0 else 0.0

    return {
        "safe_false_refusal_rate": false_refusal_rate,
        "n_safe_prompts": n_total,
        "n_incorrectly_refused": n_refused,
        "n_correctly_answered": n_total - n_refused,
    }


def run_xstest_overrefusal_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    safe_samples: list[dict],
    system_prompt: str = "You are a helpful, harmless, and honest assistant.",
    max_new_tokens: int = 256,
    batch_size: int = 8,
    refusal_classifier: str = "keyword",
    device: Optional[str] = None,
) -> dict:
    """
    Run over-refusal evaluation on XSTest safe prompts end-to-end.

    Args:
        model: Loaded model (optionally PEFT-wrapped).
        tokenizer: Corresponding tokenizer.
        safe_samples: List of XSTest safe-split sample dicts (each has "prompt").
        system_prompt: System message.
        max_new_tokens: Max tokens per response.
        batch_size: Generation batch size.
        refusal_classifier: Classification method.
        device: Device override.

    Returns:
        Dict with metrics + per-sample predictions.
    """
    prompts = [s["prompt"] for s in safe_samples]
    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
    )

    metrics = compute_overrefusal_metrics(responses, refusal_classifier=refusal_classifier)

    # Attach per-sample predictions for inspection
    per_sample = []
    for sample, response in zip(safe_samples, responses):
        per_sample.append(
            {
                "prompt": sample["prompt"],
                "type": sample.get("type", ""),
                "response": response,
                "predicted_refusal": int(is_refusal_keyword(response)),
            }
        )

    # Compute per-type breakdown if type info available
    type_breakdown: dict[str, dict] = {}
    for s in per_sample:
        t = s.get("type", "unknown")
        if t not in type_breakdown:
            type_breakdown[t] = {"n_total": 0, "n_refused": 0}
        type_breakdown[t]["n_total"] += 1
        type_breakdown[t]["n_refused"] += s["predicted_refusal"]
    for t, counts in type_breakdown.items():
        counts["false_refusal_rate"] = (
            counts["n_refused"] / counts["n_total"] if counts["n_total"] > 0 else 0.0
        )

    return {
        **metrics,
        "type_breakdown": type_breakdown,
        "per_sample": per_sample,
    }
