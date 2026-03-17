"""
Protected interference penalty module — the core of PF-LoRA.

Computes:
    R_prot = Σ_l ‖ B_l A_l U_{l,k} Λ_{l,k}^{1/2} ‖_F²

where (A_l, B_l) are the LoRA factor matrices for adapted layer l,
and (U_{l,k}, Λ_{l,k}) are the precomputed protected subspace artifacts.

This is added to the SFT loss as λ * R_prot in protected_sft.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.subspace.svd import load_subspace


class ProtectedPenalty(nn.Module):
    """
    Computes the truncated protected interference penalty.

    Given a dict of {layer_name: (A, B)} LoRA matrices and preloaded
    subspace tensors (U_k, Λ_k) per layer, computes:

        penalty = Σ_l ‖ B_l @ A_l @ U_{l,k} @ diag(Λ_{l,k})^{1/2} ‖_F²

    Only layers that have a registered subspace contribute to the penalty;
    LoRA modules on unprotected layers are left unconstrained.
    """

    def __init__(
        self,
        layer_subspaces: dict[str, dict],
        use_eigenvalue_weighting: bool = True,
    ) -> None:
        """
        Args:
            layer_subspaces: Dict mapping layer_name →
                             {"U_k": (dim, k) tensor, "lambda_k": (k,) tensor}.
                             Tensors should already be on the correct device.
            use_eigenvalue_weighting: If True, scale columns of U_k by √Λ_k
                                      (gives exact interference energy).
                                      If False, treat all protected directions equally.
        """
        super().__init__()
        self.use_eigenvalue_weighting = use_eigenvalue_weighting

        # Register subspace tensors as buffers (moved to device with model, but frozen)
        for layer_name, sub in layer_subspaces.items():
            safe_name = layer_name.replace(".", "__")
            self.register_buffer(f"U_k_{safe_name}", sub["U_k"])
            self.register_buffer(f"lambda_k_{safe_name}", sub["lambda_k"])

        # Store layer names for iteration
        self._layer_names: list[str] = list(layer_subspaces.keys())

    def _get_subspace(self, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        safe_name = layer_name.replace(".", "__")
        U_k = getattr(self, f"U_k_{safe_name}")
        lambda_k = getattr(self, f"lambda_k_{safe_name}")
        return U_k, lambda_k

    def forward(self, adapter_modules: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Compute the protected penalty summed over all adapted layers.

        Args:
            adapter_modules: Dict of {layer_name: (A, B)} where:
                - A: (r, d_in) — LoRA down-projection matrix
                - B: (d_out, r) — LoRA up-projection matrix

        Returns:
            Scalar penalty tensor (differentiable w.r.t. A and B).
        """
        device = next(self.buffers()).device
        total_penalty = torch.tensor(0.0, device=device, requires_grad=True)

        for layer_name in self._layer_names:
            if layer_name not in adapter_modules:
                continue

            A, B = adapter_modules[layer_name]  # A: (r, d_in), B: (d_out, r)
            U_k, lambda_k = self._get_subspace(layer_name)  # U_k: (d_in, k), λ_k: (k,)

            # LoRA update: ΔW = B @ A  →  (d_out, d_in)
            # Projected update on subspace: B @ A @ U_k  →  (d_out, k)
            projected = B @ (A @ U_k)  # (d_out, k)

            if self.use_eigenvalue_weighting:
                # Scale columns by sqrt(λ_k): ‖B A U_k Λ_k^{1/2}‖_F²
                sqrt_lambda = lambda_k.sqrt().unsqueeze(0)  # (1, k)
                projected = projected * sqrt_lambda

            # ‖projected‖_F² = sum of squared entries
            layer_penalty = projected.pow(2).sum()
            total_penalty = total_penalty + layer_penalty

        return total_penalty


def load_subspaces(
    subspaces_dir: str,
    k: int,
    layer_indices: list[int],
    device: str = "cuda",
    proj_names: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Load precomputed subspace artifacts for given layer indices and k.

    This function scans the subspaces_dir/layers/ directory for files
    matching the naming convention produced by build_subspaces.py.

    Args:
        subspaces_dir: Root subspace artifacts directory.
        k: Number of components to load.
        layer_indices: Which layer indices were protected.
        device: Load tensors to this device.
        proj_names: Filter to specific projection names (None = load all found).

    Returns:
        Dict mapping full layer_name → {"U_k": tensor, "lambda_k": tensor}.
    """
    sp_path = Path(subspaces_dir) / "layers"
    if not sp_path.exists():
        raise FileNotFoundError(f"Subspace layers dir not found: {sp_path}")

    layer_subspaces: dict[str, dict] = {}

    # Scan all .pt files in layers dir
    for pt_file in sorted(sp_path.glob(f"*_k{k}.pt")):
        stem = pt_file.stem  # e.g. "model__layers__12__self_attn__q_proj_k64"
        # Reconstruct layer name: replace double underscores back to dots
        layer_name = stem[: stem.rfind(f"_k{k}")].replace("__", ".")

        # Filter by layer index
        parts = layer_name.split(".")
        layer_idx_matches = any(str(idx) in parts for idx in layer_indices)
        if not layer_idx_matches:
            continue

        # Filter by projection name
        if proj_names and parts[-1] not in proj_names:
            continue

        data = torch.load(pt_file, map_location=device)
        layer_subspaces[layer_name] = {
            "U_k": data["U_k"].to(device),
            "lambda_k": data["lambda_k"].to(device),
        }

    if not layer_subspaces:
        raise RuntimeError(
            f"No subspace files found in {sp_path} for k={k}, layers={layer_indices}"
        )

    print(f"Loaded {len(layer_subspaces)} protected subspaces (k={k}) onto {device}")
    return layer_subspaces
