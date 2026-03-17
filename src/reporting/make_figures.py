"""
Paper figure generator (Week 8).

Produces three core figures:
  1. Pareto frontier: refusal quality vs capability preservation
  2. Over-refusal tradeoff: safe_false_refusal_rate vs unsafe_refusal_rate
  3. Spectrum plot: imported from src/subspace/analyze.py

Usage:
    python -m src.reporting.make_figures \
        --inputs results/logs/*.json \
        --out_dir results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


_METHOD_STYLES = {
    "base": {"marker": "^", "color": "gray", "linestyle": "none", "zorder": 3},
    "lora_baseline": {"marker": "s", "color": "steelblue", "linestyle": "none", "zorder": 3},
    "pf_lora": {"marker": "o", "color": "crimson", "linestyle": "none", "zorder": 4},
    "l2_reg": {"marker": "D", "color": "olivedrab", "linestyle": "none", "zorder": 3},
    "ewc": {"marker": "P", "color": "darkorange", "linestyle": "none", "zorder": 3},
    "replay": {"marker": "X", "color": "mediumpurple", "linestyle": "none", "zorder": 3},
}

_DEFAULT_STYLE = {"marker": "o", "color": "black", "linestyle": "none", "zorder": 2}


def _load_results(paths: list[str]) -> list[dict]:
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return results


def _get_style(method_name: str) -> dict:
    for key, style in _METHOD_STYLES.items():
        if key in method_name.lower():
            return style
    return _DEFAULT_STYLE


def plot_pareto_frontier(
    results: list[dict],
    method_names: Optional[list[str]] = None,
    out_path: Optional[str] = None,
    annotate: bool = True,
) -> plt.Figure:
    """
    Plot the Pareto frontier: refusal_f1 (x) vs avg_capability_score (y).

    Better Pareto = high refusal quality AND high capability.
    PF-LoRA should appear in the upper-right vs standard LoRA.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, result in enumerate(results):
        summary = result.get("summary", {})
        name = (
            method_names[i]
            if method_names and i < len(method_names)
            else result.get("meta", {}).get("adapter", f"run_{i}")
        )

        x = summary.get("refusal_f1_wildguard") or summary.get("refusal_f1_xstest")
        y = summary.get("avg_capability_score")

        if x is None or y is None:
            print(f"  [SKIP] {name}: missing refusal_f1 or avg_capability_score")
            continue

        style = _get_style(name)
        ax.scatter(
            x, y,
            marker=style["marker"],
            color=style["color"],
            s=120,
            zorder=style["zorder"],
            label=name,
            edgecolors="black",
            linewidths=0.5,
        )

        if annotate:
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Refusal F1 (WildGuardTest)", fontsize=12)
    ax.set_ylabel("Avg Capability Score", fontsize=12)
    ax.set_title("Pareto Frontier: Refusal Quality vs Capability", fontsize=13)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Ideal corner annotation
    ax.annotate(
        "ideal",
        xy=(1.0, 1.0),
        xytext=(-15, -15),
        textcoords="offset points",
        fontsize=8,
        color="green",
        arrowprops=dict(arrowstyle="->", color="green", lw=1.0),
    )

    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Pareto figure saved → {out_path}")

    return fig


def plot_overrefusal_tradeoff(
    results: list[dict],
    method_names: Optional[list[str]] = None,
    out_path: Optional[str] = None,
    annotate: bool = True,
) -> plt.Figure:
    """
    Plot over-refusal tradeoff:
        X = safe_false_refusal_rate (lower is better)
        Y = unsafe_refusal_rate (higher is better)

    Ideal point is (0, 1): never refuses safe prompts, always refuses unsafe ones.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, result in enumerate(results):
        summary = result.get("summary", {})
        name = (
            method_names[i]
            if method_names and i < len(method_names)
            else result.get("meta", {}).get("adapter", f"run_{i}")
        )

        x = summary.get("safe_false_refusal_rate")
        y = summary.get("unsafe_refusal_rate")

        if x is None or y is None:
            print(f"  [SKIP] {name}: missing refusal metrics")
            continue

        style = _get_style(name)
        ax.scatter(
            x, y,
            marker=style["marker"],
            color=style["color"],
            s=120,
            zorder=style["zorder"],
            label=name,
            edgecolors="black",
            linewidths=0.5,
        )

        if annotate:
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    # Mark ideal point
    ax.scatter(0.0, 1.0, marker="*", color="green", s=200, zorder=5, label="ideal")

    ax.set_xlabel("Safe False Refusal Rate (↓ better)", fontsize=12)
    ax.set_ylabel("Unsafe Refusal Rate (↑ better)", fontsize=12)
    ax.set_title("Over-Refusal Tradeoff (XSTest)", fontsize=13)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Over-refusal figure saved → {out_path}")

    return fig


def plot_capability_tax_bars(
    results: list[dict],
    method_names: Optional[list[str]] = None,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of capability tax per method across benchmarks.
    """
    tasks = ["mmlu_pro", "hellaswag", "arc_challenge", "gsm8k", "truthfulqa_mc2"]
    n_methods = len(results)
    x = np.arange(len(tasks))
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, result in enumerate(results):
        tax = result.get("capability_tax", {}) or {}
        name = (
            method_names[i]
            if method_names and i < len(method_names)
            else result.get("meta", {}).get("adapter", f"run_{i}")
        )

        values = [tax.get(t) for t in tasks]
        has_any = any(v is not None for v in values)
        if not has_any:
            continue

        bars = [v if v is not None else 0.0 for v in values]
        offset = (i - n_methods / 2 + 0.5) * width
        style = _get_style(name)
        ax.bar(
            x + offset,
            bars,
            width=width,
            label=name,
            color=style["color"],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Benchmark", fontsize=12)
    ax.set_ylabel("Capability Tax (%) ↓ better", fontsize=12)
    ax.set_title("Per-benchmark Capability Tax", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Capability tax figure saved → {out_path}")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--inputs", nargs="+", required=True, help="Result JSON files")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument(
        "--subspaces_dir",
        default=None,
        help="If set, also generate spectrum plot from subspace artifacts",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(args.inputs)

    plot_pareto_frontier(
        results=results,
        method_names=args.names,
        out_path=str(out_dir / "pareto.png"),
    )

    plot_overrefusal_tradeoff(
        results=results,
        method_names=args.names,
        out_path=str(out_dir / "overrefusal_tradeoff.png"),
    )

    plot_capability_tax_bars(
        results=results,
        method_names=args.names,
        out_path=str(out_dir / "capability_tax_bars.png"),
    )

    if args.subspaces_dir:
        from src.subspace.analyze import analyze_subspaces
        analyze_subspaces(
            subspaces_dir=args.subspaces_dir,
            out_dir=str(out_dir / "spectrum"),
        )

    plt.close("all")
    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
