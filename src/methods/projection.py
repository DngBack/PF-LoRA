"""
Hard gradient projection ablation.

Instead of a soft penalty, this ablation zeroes out the LoRA gradient
components that lie in the protected subspace directions.

Mathematically: at each step, after computing gradients, we project
the gradient of each (A, B) pair to the subspace orthogonal to U_{l,k}:

    grad_A ← grad_A - U_{l,k} (U_{l,k}^T grad_A^T)^T
              = grad_A - (grad_A U_{l,k}) U_{l,k}^T

This is a hard constraint (no learning in protected directions) vs the
soft penalty which only discourages but allows some use.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def project_out_protected_subspace(
    grad: torch.Tensor,
    U_k: torch.Tensor,
    transpose: bool = False,
) -> torch.Tensor:
    """
    Remove the component of a gradient tensor that lies in the protected subspace.

    For matrix A of shape (r, d_in):
        grad_A ← grad_A - (grad_A @ U_k) @ U_k^T

    For matrix B of shape (d_out, r), protection is applied through A,
    so B gradients are typically left unmodified.

    Args:
        grad: Gradient tensor (r, d_in) or (d_out, r).
        U_k: Protected subspace basis (d_in, k) — columns are eigenvectors.
        transpose: If True, apply projection to the transposed input dimension.

    Returns:
        Projected gradient with protected component removed.
    """
    if transpose:
        # For B: we don't directly project B gradients in the hard variant
        return grad

    # grad: (r, d_in), U_k: (d_in, k)
    # proj_component: (r, k) = grad @ U_k
    proj = grad @ U_k          # (r, k)
    # Remove: grad -= proj @ U_k^T
    return grad - proj @ U_k.t()


class HardProjectionHook:
    """
    Registers backward hooks on LoRA A matrices to zero out protected gradient directions.

    Activated as an ablation baseline to compare against the soft penalty variant.
    """

    def __init__(
        self,
        layer_subspaces: dict[str, dict],
    ) -> None:
        """
        Args:
            layer_subspaces: Same format as ProtectedPenalty:
                             {layer_name: {"U_k": (dim, k), "lambda_k": (k,)}}.
        """
        self.layer_subspaces = layer_subspaces
        self._hooks: list = []

    def register(self, model: nn.Module) -> None:
        """Register gradient hooks on LoRA A matrices for protected layers."""
        for name, module in model.named_modules():
            if name not in self.layer_subspaces:
                continue
            if not (hasattr(module, "lora_A") and "default" in module.lora_A):
                continue

            U_k = self.layer_subspaces[name]["U_k"]
            A_param = module.lora_A["default"].weight  # (r, d_in)

            def make_hook(u_k):
                def hook_fn(grad):
                    return project_out_protected_subspace(grad, u_k)
                return hook_fn

            h = A_param.register_hook(make_hook(U_k))
            self._hooks.append(h)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()
