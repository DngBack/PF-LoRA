"""
Activation shard writer and reader.

Saves activation tensors per layer in sharded .pt files to avoid
OOM when processing large datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import torch


_SHARD_PREFIX = "activations_layer"
_META_FILE = "collection_meta.json"


def save_activation_shard(
    activations: dict[str, torch.Tensor],
    out_dir: Path,
    shard_idx: int,
) -> None:
    """
    Save one shard of activations (one batch worth of samples).

    Args:
        activations: Dict mapping layer_name → (batch, hidden_dim) tensor.
        out_dir: Output directory.
        shard_idx: Shard index for file naming.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{shard_idx:05d}.pt"
    torch.save(activations, shard_path)


def load_activation_shards(
    out_dir: Path,
    layer_name: Optional[str] = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """
    Yield activation shards one at a time (memory-friendly).

    Args:
        out_dir: Directory containing shard files.
        layer_name: If set, yield only the tensor for this layer.

    Yields:
        Dict of layer_name → tensor (or single tensor if layer_name set).
    """
    shard_paths = sorted(out_dir.glob("shard_*.pt"))
    for path in shard_paths:
        data = torch.load(path, map_location="cpu")
        if layer_name is not None:
            if layer_name in data:
                yield {layer_name: data[layer_name]}
        else:
            yield data


def load_all_activations_for_layer(
    out_dir: Path,
    layer_name: str,
) -> torch.Tensor:
    """
    Load and concatenate all shard activations for a single layer.

    Args:
        out_dir: Shard directory.
        layer_name: Target layer name.

    Returns:
        Tensor of shape (N_total_samples, hidden_dim).
    """
    tensors = []
    for shard in load_activation_shards(out_dir, layer_name=layer_name):
        t = shard.get(layer_name)
        if t is not None:
            tensors.append(t)
    if not tensors:
        raise FileNotFoundError(f"No activations found for layer '{layer_name}' in {out_dir}")
    return torch.cat(tensors, dim=0)


def save_collection_meta(
    out_dir: Path,
    meta: dict,
) -> None:
    (out_dir / _META_FILE).write_text(json.dumps(meta, indent=2))


def load_collection_meta(out_dir: Path) -> dict:
    meta_path = out_dir / _META_FILE
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())
