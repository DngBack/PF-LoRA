"""
Incremental covariance sketching for large hidden-state collections.

Computes the sample covariance C = (1/N) Σ h_i h_i^T incrementally,
without loading all activations into memory simultaneously.
"""

from __future__ import annotations

import torch


class CovarianceSketch:
    """
    Online covariance accumulator.

    Maintains running sum-of-outer-products and sample count.
    Supports chunked updates and returns the accumulated covariance matrix.
    """

    def __init__(self, dim: int, dtype: torch.dtype = torch.float32) -> None:
        """
        Args:
            dim: Hidden dimension (size of each activation vector).
            dtype: Accumulator dtype. float32 or float64 for numerical stability.
        """
        self.dim = dim
        self.dtype = dtype
        self._accumulator = torch.zeros(dim, dim, dtype=dtype)
        self._n = 0

    def update(self, activations: torch.Tensor) -> None:
        """
        Incorporate a batch of activation vectors.

        Args:
            activations: Tensor of shape (batch, dim).
        """
        if activations.dim() != 2 or activations.shape[1] != self.dim:
            raise ValueError(
                f"Expected shape (N, {self.dim}), got {activations.shape}"
            )
        h = activations.to(self.dtype).cpu()
        # C += h^T h  (using einsum for clarity; mm is equivalent)
        self._accumulator += torch.mm(h.t(), h)
        self._n += h.shape[0]

    def get_covariance(self, regularize: float = 1e-6) -> torch.Tensor:
        """
        Return the sample covariance matrix C = (1/N) Σ h_i h_i^T.

        Args:
            regularize: Add ε * I to improve numerical stability before eigendecomp.

        Returns:
            Covariance matrix of shape (dim, dim).
        """
        if self._n == 0:
            raise RuntimeError("No samples accumulated yet")
        cov = self._accumulator / self._n
        if regularize > 0:
            cov += regularize * torch.eye(self.dim, dtype=self.dtype)
        return cov

    @property
    def n_samples(self) -> int:
        return self._n

    def reset(self) -> None:
        self._accumulator.zero_()
        self._n = 0


def build_covariance_from_shards(
    shard_loader,  # iterable yielding {layer_name: tensor}
    layer_name: str,
    dim: int,
    dtype: torch.dtype = torch.float32,
    regularize: float = 1e-6,
) -> tuple[torch.Tensor, int]:
    """
    Build covariance from a streaming shard loader for a single layer.

    Args:
        shard_loader: Iterable of shard dicts.
        layer_name: Which layer's activations to use.
        dim: Hidden dimension.
        dtype: Accumulator dtype.
        regularize: Regularization for numerical stability.

    Returns:
        (covariance_matrix, n_samples) tuple.
    """
    sketch = CovarianceSketch(dim=dim, dtype=dtype)
    for shard in shard_loader:
        acts = shard.get(layer_name)
        if acts is not None:
            sketch.update(acts)
    cov = sketch.get_covariance(regularize=regularize)
    return cov, sketch.n_samples
