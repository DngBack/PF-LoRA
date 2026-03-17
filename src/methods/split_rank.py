"""
Guarded / Free rank split LoRA (ablation, Week 6).

Splits the total LoRA rank budget into two parts:
  - guarded_rank: small rank adapter whose output is constrained to lie
    in the complement of the protected subspace (does not touch protected dirs)
  - free_rank:    remaining rank adapter with no constraint (free to learn alignment)

The total effective LoRA update is:
    ΔW = B_free A_free + B_guarded A_guarded

where A_guarded is initialized orthogonal to U_{l,k} via QR or direct projection.

This is an *architectural* ablation as opposed to the *penalty* ablation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _init_orthogonal_to_subspace(
    r: int,
    d_in: int,
    U_k: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Initialize an (r, d_in) matrix whose rows are orthogonal to the columns of U_k.

    Strategy: sample random rows and then project out U_k components.
    This ensures initial ΔW_guarded has zero protected interference.

    Args:
        r: Rank (number of rows).
        d_in: Input dimension (column count = hidden size).
        U_k: Protected subspace basis (d_in, k).
        dtype: Output dtype.

    Returns:
        Orthogonally initialized A matrix (r, d_in).
    """
    A = torch.randn(r, d_in, dtype=dtype)
    # Project out U_k component: A ← A - (A @ U_k) @ U_k^T
    proj = A @ U_k  # (r, k)
    A = A - proj @ U_k.t()
    # Normalize rows
    norms = A.norm(dim=1, keepdim=True).clamp(min=1e-8)
    A = A / norms * (1.0 / (r ** 0.5))
    return A


class SplitRankLoRALayer(nn.Module):
    """
    A single LoRA layer with guarded + free rank split.

    Forward output added to base weight output:
        ΔW x = B_free A_free x + B_guarded A_guarded x
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        guarded_rank: int,
        free_rank: int,
        U_k: torch.Tensor,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.guarded_rank = guarded_rank
        self.free_rank = free_rank
        self.total_rank = guarded_rank + free_rank
        scaling = lora_alpha / self.total_rank

        # Free adapter (standard LoRA init)
        self.A_free = nn.Parameter(
            torch.randn(free_rank, d_in, dtype=dtype) * (1.0 / free_rank ** 0.5)
        )
        self.B_free = nn.Parameter(torch.zeros(d_out, free_rank, dtype=dtype))

        # Guarded adapter (init orthogonal to protected subspace)
        A_guarded_init = _init_orthogonal_to_subspace(guarded_rank, d_in, U_k.float())
        self.A_guarded = nn.Parameter(A_guarded_init.to(dtype))
        self.B_guarded = nn.Parameter(torch.zeros(d_out, guarded_rank, dtype=dtype))

        self.scaling = scaling
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Store U_k as buffer (non-trainable)
        self.register_buffer("U_k", U_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined LoRA delta applied to input x.

        Args:
            x: Input tensor (..., d_in).

        Returns:
            Delta tensor (..., d_out).
        """
        x_dropped = self.dropout(x)
        delta_free = (x_dropped @ self.A_free.t()) @ self.B_free.t()
        delta_guarded = (x_dropped @ self.A_guarded.t()) @ self.B_guarded.t()
        return self.scaling * (delta_free + delta_guarded)

    def get_guarded_penalty(self) -> torch.Tensor:
        """
        Compute the protected interference of the guarded adapter.

        Ideally this stays near zero; can be added to loss as a small regularizer
        to maintain the guarded property during training.
        """
        projected = self.B_guarded @ (self.A_guarded @ self.U_k)  # (d_out, k)
        return projected.pow(2).sum()
