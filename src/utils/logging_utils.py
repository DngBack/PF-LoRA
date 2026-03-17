"""Experiment tracking and logging utilities."""

from __future__ import annotations

import json
import os
from typing import Any, Optional


def init_wandb(
    project: str = "pf-lora",
    name: Optional[str] = None,
    config: Optional[dict] = None,
    tags: Optional[list[str]] = None,
    disabled: bool = False,
) -> Any:
    """
    Initialize a Weights & Biases run.

    Args:
        project: W&B project name.
        name: Run name (auto-generated if None).
        config: Hyperparameter dict to log.
        tags: List of tag strings.
        disabled: Set True to disable W&B (useful for local debug).

    Returns:
        wandb.Run object, or None if W&B is disabled.
    """
    if disabled or os.environ.get("WANDB_DISABLED", "false").lower() == "true":
        return None
    try:
        import wandb
        run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags or [],
            reinit=True,
        )
        return run
    except ImportError:
        print("wandb not installed; skipping experiment tracking")
        return None


def log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    """Log metrics to W&B if available."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


def finish_wandb() -> None:
    """Finish the current W&B run."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


def log_table(name: str, data: list[dict], step: Optional[int] = None) -> None:
    """Log a list of dicts as a W&B table."""
    try:
        import wandb
        if wandb.run is not None and data:
            table = wandb.Table(columns=list(data[0].keys()))
            for row in data:
                table.add_data(*row.values())
            wandb.log({name: table}, step=step)
    except ImportError:
        pass


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """Pretty-print a metrics dict to stdout."""
    pad = prefix + " " if prefix else ""
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {pad}{k}: {v:.4f}")
        else:
            print(f"  {pad}{k}: {v}")
