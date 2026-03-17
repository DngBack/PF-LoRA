"""
Subspace analysis and visualization.

Generates spectrum decay plots and explained-variance curves from
precomputed subspace artifacts. Used for Week 4 deliverables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_singular_value_decay(
    eigenvalues: torch.Tensor,
    layer_name: str,
    out_path: Optional[str] = None,
    top_n: int = 128,
) -> plt.Figure:
    """
    Plot the singular value (eigenvalue) decay spectrum for a single layer.

    Args:
        eigenvalues: 1D tensor of eigenvalues in descending order.
        layer_name: Label for the plot title.
        out_path: If set, save figure to this path.
        top_n: Number of leading eigenvalues to plot.

    Returns:
        Matplotlib Figure.
    """
    vals = eigenvalues[:top_n].float().numpy()
    indices = np.arange(1, len(vals) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Linear scale
    axes[0].plot(indices, vals, linewidth=1.5, color="steelblue")
    axes[0].set_xlabel("Component index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title(f"{layer_name} — Eigenvalue spectrum (linear)")
    axes[0].grid(True, alpha=0.3)

    # Log scale
    axes[1].semilogy(indices, vals + 1e-12, linewidth=1.5, color="darkorange")
    axes[1].set_xlabel("Component index")
    axes[1].set_ylabel("Eigenvalue (log scale)")
    axes[1].set_title(f"{layer_name} — Eigenvalue spectrum (log)")
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved spectrum plot → {out_path}")

    return fig


def plot_explained_variance_vs_k(
    summary: dict[str, dict],
    out_path: Optional[str] = None,
    highlight_layers: Optional[list[str]] = None,
) -> plt.Figure:
    """
    Plot explained variance ratio vs k for all layers.

    Args:
        summary: Dict of {layer_name: {k: explained_variance_ratio}} from subspace_summary.json.
        out_path: If set, save figure here.
        highlight_layers: Optional subset of layers to draw in bold.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for layer_name, kv_dict in summary.items():
        ks = sorted(int(k) for k in kv_dict.keys())
        evs = [kv_dict[str(k)] for k in ks]
        label = layer_name.split(".")[-3] + "." + layer_name.split(".")[-1]  # short label
        is_highlight = highlight_layers and layer_name in highlight_layers
        ax.plot(
            ks, evs,
            marker="o",
            linewidth=2.0 if is_highlight else 0.8,
            alpha=0.9 if is_highlight else 0.5,
            label=label if is_highlight else None,
        )

    ax.axhline(0.90, color="red", linestyle="--", linewidth=1.0, label="90% threshold")
    ax.axhline(0.95, color="orange", linestyle="--", linewidth=1.0, label="95% threshold")
    ax.set_xlabel("k (number of protected components)")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Protected subspace: explained variance vs k")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    if highlight_layers:
        ax.legend(loc="lower right")

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved explained variance plot → {out_path}")

    return fig


def analyze_subspaces(
    subspaces_dir: str,
    out_dir: Optional[str] = None,
    top_n_spectrum: int = 128,
) -> None:
    """
    Run full analysis on a subspace artifacts directory and save figures.

    Args:
        subspaces_dir: Path to subspace artifacts (output of build_subspaces.py).
        out_dir: Where to save figures (defaults to subspaces_dir/analysis).
        top_n_spectrum: How many leading eigenvalues to show in spectrum plot.
    """
    sp_path = Path(subspaces_dir)
    if out_dir is None:
        out_dir = str(sp_path / "analysis")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load summary
    summary_file = sp_path / "subspace_summary.json"
    if not summary_file.exists():
        print(f"No summary file found at {summary_file}")
        return

    summary = json.loads(summary_file.read_text())

    # 1) Explained variance vs k (all layers)
    plot_explained_variance_vs_k(
        summary=summary,
        out_path=str(out_path / "explained_variance_vs_k.png"),
    )

    # 2) Spectrum plots for each layer (using largest k available)
    layers_dir = sp_path / "layers"
    for layer_name in summary.keys():
        ks = sorted(int(k) for k in summary[layer_name].keys())
        if not ks:
            continue
        max_k = ks[-1]
        stem = f"{layer_name.replace('.', '_')}_k{max_k}"
        pt_path = layers_dir / f"{stem}.pt"
        if not pt_path.exists():
            continue

        data = torch.load(pt_path, map_location="cpu")
        lambda_k = data.get("lambda_k")
        if lambda_k is None:
            continue

        safe_name = layer_name.replace(".", "_")
        plot_singular_value_decay(
            eigenvalues=lambda_k,
            layer_name=layer_name,
            out_path=str(out_path / f"spectrum_{safe_name}.png"),
            top_n=min(top_n_spectrum, len(lambda_k)),
        )

    plt.close("all")
    print(f"Analysis complete. Figures saved to {out_path}")


if __name__ == "__main__":
    import sys

    subspaces_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/subspaces/llama31_protected_mix"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    analyze_subspaces(subspaces_dir, out_dir)
