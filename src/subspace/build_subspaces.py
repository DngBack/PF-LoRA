"""
Subspace build orchestrator (Week 4).

Reads activation shards, computes covariance per layer, runs eigendecomposition,
and saves U_k and Λ_k artifacts for each (layer, k) combination.

Usage:
    python -m src.subspace.build_subspaces \
        --activations_dir artifacts/activations/llama31_protected_mix \
        --k_list 32 64 128 256 \
        --out artifacts/subspaces/llama31_protected_mix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.subspace.sketch import build_covariance_from_shards
from src.subspace.storage import load_activation_shards, load_collection_meta
from src.subspace.svd import compute_protected_subspace, save_subspace


def build_subspaces(
    activations_dir: str,
    k_list: list[int],
    out_dir: str,
    svd_method: str = "eigh",
    regularize: float = 1e-6,
) -> dict:
    """
    Build protected subspace artifacts for all layers and k values.

    Args:
        activations_dir: Directory of activation shards (from collect_activations.py).
        k_list: List of k values to compute subspaces for.
        out_dir: Output directory for subspace artifacts.
        svd_method: "eigh" | "randomized".
        regularize: Covariance regularization (added to diagonal).

    Returns:
        Summary dict of {layer_name: {k: explained_variance_ratio}}.
    """
    acts_path = Path(activations_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    meta = load_collection_meta(acts_path)
    target_modules = meta.get("target_modules", [])
    if not target_modules:
        # Infer from first shard
        first_shard = next(load_activation_shards(acts_path), None)
        if first_shard:
            target_modules = list(first_shard.keys())

    if not target_modules:
        raise RuntimeError(f"No target modules found in {activations_dir}")

    print(f"Building subspaces for {len(target_modules)} layers, k={k_list}")

    summary: dict = {}

    for layer_name in target_modules:
        print(f"\nLayer: {layer_name}")

        # Determine hidden dim from first available activation
        dim = None
        for shard in load_activation_shards(acts_path, layer_name=layer_name):
            t = shard.get(layer_name)
            if t is not None:
                dim = t.shape[-1]
                break

        if dim is None:
            print(f"  [SKIP] No activations for {layer_name}")
            continue

        print(f"  hidden_dim={dim}")

        # Build covariance
        cov, n_samples = build_covariance_from_shards(
            shard_loader=load_activation_shards(acts_path, layer_name=layer_name),
            layer_name=layer_name,
            dim=dim,
            regularize=regularize,
        )
        print(f"  Covariance built: {n_samples} samples, trace={cov.trace():.4f}")

        summary[layer_name] = {}
        layer_out_dir = out_path / "layers"

        for k in k_list:
            actual_k = min(k, dim)
            subspace = compute_protected_subspace(
                covariance=cov,
                k=actual_k,
                method=svd_method,
            )
            save_subspace(
                subspace=subspace,
                out_path=layer_out_dir,
                layer_name=layer_name,
                k=actual_k,
                n_samples=n_samples,
            )
            ev = subspace["explained_variance_ratio"]
            summary[layer_name][k] = ev
            print(f"  k={actual_k}: explained_variance={ev:.4f}")

    # Save overall summary
    summary_path = out_path / "subspace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSubspace artifacts saved to {out_path}")
    print(f"Summary: {summary_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build protected subspace artifacts")
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--k_list", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--svd_method",
        choices=["eigh", "randomized"],
        default="eigh",
        help="'eigh' for exact (recommended for dim<=4096), 'randomized' for larger",
    )
    parser.add_argument("--regularize", type=float, default=1e-6)
    args = parser.parse_args()

    build_subspaces(
        activations_dir=args.activations_dir,
        k_list=args.k_list,
        out_dir=args.out,
        svd_method=args.svd_method,
        regularize=args.regularize,
    )


if __name__ == "__main__":
    main()
