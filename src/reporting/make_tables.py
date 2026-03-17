"""
Paper table generator (Week 8).

Reads multiple result JSON files (produced by src/eval/report.py) and
generates:
  - main_table.csv: base vs LoRA baseline vs PF-LoRA
  - ablation_table.csv: all ablation runs

Usage:
    python -m src.reporting.make_tables \
        --inputs results/logs/base_llama31.json \
                 results/logs/lora_baseline_llama31.json \
                 results/logs/protected_llama31.json \
        --out_dir results/tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


# Columns in the main results table, in order
_MAIN_COLUMNS = [
    "method",
    "model",
    "unsafe_refusal_rate",
    "safe_false_refusal_rate",
    "refusal_f1_xstest",
    "refusal_f1_wildguard",
    "mmlu_pro",
    "hellaswag",
    "arc_challenge",
    "gsm8k",
    "truthfulqa_mc2",
    "avg_capability_score",
    "capability_tax_avg",
]

_TASK_NAME_MAP = {
    "mmlu_pro": "mmlu_pro",
    "hellaswag": "hellaswag",
    "arc_challenge": "arc_challenge",
    "gsm8k": "gsm8k",
    "truthfulqa_mc2": "truthfulqa_mc2",
}


def _extract_row(result: dict, method_name: Optional[str] = None) -> dict:
    """Extract a flat row dict from a full result JSON."""
    meta = result.get("meta", {})
    summary = result.get("summary", {})
    cap = result.get("capability", {})
    cap_tasks = cap.get("tasks", {})
    cap_tax = result.get("capability_tax", {}) or {}

    row = {
        "method": method_name or meta.get("adapter", meta.get("model", "unknown")),
        "model": meta.get("model", ""),
        "unsafe_refusal_rate": summary.get("unsafe_refusal_rate"),
        "safe_false_refusal_rate": summary.get("safe_false_refusal_rate"),
        "refusal_f1_xstest": summary.get("refusal_f1_xstest"),
        "refusal_f1_wildguard": summary.get("refusal_f1_wildguard"),
        "avg_capability_score": summary.get("avg_capability_score"),
        "capability_tax_avg": summary.get("capability_tax_avg"),
    }

    # Per-task benchmark scores
    for task_key, col_name in _TASK_NAME_MAP.items():
        task_data = cap_tasks.get(task_key, {})
        row[col_name] = task_data.get("score")

    return row


def build_main_table(
    result_files: list[str],
    method_names: Optional[list[str]] = None,
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the main results table from a list of result JSON files.

    Args:
        result_files: Paths to result JSON files (base, lora_baseline, pf_lora, ...).
        method_names: Optional display names for each file (same order as result_files).
        out_path: If set, save CSV here.

    Returns:
        DataFrame with one row per method.
    """
    rows = []
    for i, rf in enumerate(result_files):
        with open(rf) as f:
            result = json.load(f)
        name = method_names[i] if method_names and i < len(method_names) else None
        rows.append(_extract_row(result, method_name=name))

    df = pd.DataFrame(rows)
    # Reorder and keep only defined columns that exist
    available = [c for c in _MAIN_COLUMNS if c in df.columns]
    df = df[available]

    # Round floats to 4 decimal places
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(4)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Main table saved → {out_path}")

    return df


def build_ablation_table(
    result_files: list[str],
    method_names: Optional[list[str]] = None,
    out_path: Optional[str] = None,
    sort_by: str = "avg_capability_score",
) -> pd.DataFrame:
    """
    Build the ablation table. Same format as main table but includes all ablation runs.

    Args:
        result_files: Paths to ablation result JSON files.
        method_names: Optional display names.
        out_path: If set, save CSV here.
        sort_by: Column to sort rows by (descending).

    Returns:
        DataFrame sorted by sort_by column.
    """
    df = build_main_table(result_files, method_names=method_names)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Ablation table saved → {out_path}")

    return df


def print_table(df: pd.DataFrame, title: str = "") -> None:
    """Pretty-print a table to stdout."""
    if title:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    try:
        from rich.table import Table
        from rich.console import Console

        table = Table(title=title, show_lines=True)
        for col in df.columns:
            table.add_column(col, style="cyan" if col == "method" else "white")
        for _, row in df.iterrows():
            table.add_row(*[str(v) if v is not None else "-" for v in row.values])
        Console().print(table)
    except ImportError:
        print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper result tables")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Result JSON files (space-separated)",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for CSVs")
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Display names for each input file (same order)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="If set, also generate ablation table (uses all inputs as ablation runs)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_df = build_main_table(
        result_files=args.inputs,
        method_names=args.names,
        out_path=str(out_dir / "main_table.csv"),
    )
    print_table(main_df, title="Main Results")

    if args.ablation:
        ablation_df = build_ablation_table(
            result_files=args.inputs,
            method_names=args.names,
            out_path=str(out_dir / "ablation_table.csv"),
        )
        print_table(ablation_df, title="Ablation Results")


if __name__ == "__main__":
    main()
