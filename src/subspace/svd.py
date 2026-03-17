"""
Eigendecomposition of layer covariance matrices to extract the protected subspace.

Computes U_{l,k} and Λ_{l,k} from the activation covariance Σ_l such that:
    Σ_l ≈ U_{l,k} Λ_{l,k} U_{l,k}^T

Supports both full eigendecomposition (torch.linalg.eigh) and
randomized SVD (scikit-learn) for efficiency with large hidden dims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch


def compute_protected_subspace(
    covariance: torch.Tensor,
    k: int,
    method: str = "eigh",
    n_oversampling: int = 10,
    n_power_iter: int = 2,
) -> dict:
    """
    Decompose a covariance matrix and return the top-k protected subspace.

    Args:
        covariance: Symmetric PSD covariance matrix (dim, dim).
        k: Number of principal components to retain.
        method: "eigh" (exact, full) or "randomized" (approximate, faster for large dim).
        n_oversampling: Extra samples for randomized SVD (only used if method="randomized").
        n_power_iter: Power iterations for randomized SVD.

    Returns:
        Dict with:
            - U_k: (dim, k) matrix of top-k eigenvectors
            - lambda_k: (k,) vector of top-k eigenvalues
            - explained_variance_ratio: scalar
            - total_variance: scalar
            - k: int
    """
    dim = covariance.shape[0]
    k = min(k, dim)

    if method == "eigh":
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
        # Reverse to get descending order (largest first)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)
        # Clamp negative eigenvalues (numerical noise)
        eigenvalues = eigenvalues.clamp(min=0.0)

    elif method == "randomized":
        from sklearn.utils.extmath import randomized_svd
        import numpy as np

        cov_np = covariance.float().numpy()
        # Covariance is symmetric PSD, so SVD = eigendecomp
        U, S, Vt = randomized_svd(
            cov_np,
            n_components=k,
            n_oversamples=n_oversampling,
            n_iter=n_power_iter,
            random_state=42,
        )
        eigenvalues = torch.from_numpy(S.astype(np.float32))
        eigenvectors = torch.from_numpy(U.astype(np.float32))
        eigenvalues = eigenvalues.clamp(min=0.0)
        # randomized_svd already returns top-k, so we don't slice below
        return {
            "U_k": eigenvectors,          # (dim, k)
            "lambda_k": eigenvalues,       # (k,)
            "explained_variance_ratio": float(eigenvalues.sum() / covariance.trace()),
            "total_variance": float(covariance.trace()),
            "k": k,
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'eigh' or 'randomized'.")

    U_k = eigenvectors[:, :k]        # (dim, k)
    lambda_k = eigenvalues[:k]        # (k,)

    total_variance = eigenvalues.sum().item()
    explained = lambda_k.sum().item()
    explained_ratio = explained / total_variance if total_variance > 0 else 0.0

    return {
        "U_k": U_k,
        "lambda_k": lambda_k,
        "explained_variance_ratio": explained_ratio,
        "total_variance": total_variance,
        "k": k,
    }


def save_subspace(
    subspace: dict,
    out_path: Path,
    layer_name: str,
    k: int,
    n_samples: int,
) -> None:
    """
    Save a subspace artifact to disk.

    Saves:
        - <layer_name>_k<k>.pt  — dict with U_k, lambda_k
        - <layer_name>_k<k>_meta.json — explained variance, k, n_samples
    """
    out_path.mkdir(parents=True, exist_ok=True)
    stem = f"{layer_name.replace('.', '_')}_k{k}"

    torch.save(
        {"U_k": subspace["U_k"], "lambda_k": subspace["lambda_k"]},
        out_path / f"{stem}.pt",
    )

    meta = {
        "layer_name": layer_name,
        "k": k,
        "n_samples": n_samples,
        "explained_variance_ratio": subspace["explained_variance_ratio"],
        "total_variance": subspace["total_variance"],
    }
    (out_path / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))


def load_subspace(
    subspaces_dir: Path,
    layer_name: str,
    k: int,
    device: str = "cpu",
) -> dict:
    """
    Load a saved subspace artifact.

    Returns:
        Dict with U_k (dim, k) and lambda_k (k,) tensors on device.
    """
    stem = f"{layer_name.replace('.', '_')}_k{k}"
    pt_path = subspaces_dir / f"{stem}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Subspace file not found: {pt_path}")
    data = torch.load(pt_path, map_location=device)
    return data
