"""
Forward hooks for capturing hidden activations at LoRA target layers.

Registers hooks on specific linear modules to intercept their input tensors
(the hidden states h_l(x) that LoRA acts on).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


class ActivationCollector:
    """
    Manages forward hooks on a set of named modules.

    For each target module, captures the *input* activation
    (the tensor that gets multiplied by the weight, i.e. h_l(x)).
    Only the last-token position is stored to minimize memory.
    """

    def __init__(self) -> None:
        self._hooks: list = []
        # layer_name → list of captured last-token activations (one per batch)
        self.captured: dict[str, list[torch.Tensor]] = {}

    def register(self, model: nn.Module, layer_names: list[str]) -> None:
        """
        Register hooks on the specified submodule names.

        Args:
            model: The model to hook.
            layer_names: Dot-separated module names (as returned by named_modules).
        """
        name_set = set(layer_names)
        for name, module in model.named_modules():
            if name in name_set:
                self.captured[name] = []
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str) -> Callable:
        def hook_fn(module, inputs, output):
            # inputs[0]: (batch, seq_len, hidden_dim) or (batch*seq_len, hidden_dim)
            x = inputs[0].detach()
            if x.dim() == 3:
                # Take last non-padding token: use the final token position
                last_token = x[:, -1, :]  # (batch, hidden_dim)
            elif x.dim() == 2:
                # Already flattened (batch*seq, hidden)
                # We approximate: take every seq_len-th row (last token per sample)
                last_token = x  # caller must decide how to pool
            else:
                last_token = x.reshape(x.shape[0], -1)

            self.captured[name].append(last_token.cpu())

        return hook_fn

    def remove(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_activations(self, name: str) -> Optional[torch.Tensor]:
        """Return stacked activations for a layer. Shape: (N, hidden_dim)."""
        if name not in self.captured or not self.captured[name]:
            return None
        return torch.cat(self.captured[name], dim=0)

    def clear(self) -> None:
        """Clear stored activations without removing hooks."""
        for k in self.captured:
            self.captured[k] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


def get_target_module_names(
    model: nn.Module,
    layer_indices: list[int],
    proj_names: list[str],
) -> list[str]:
    """
    Build the list of fully-qualified module names for target LoRA projections
    at specified layer indices.

    Args:
        model: The model.
        layer_indices: Which transformer layer indices to target (0-indexed).
        proj_names: Projection module names within each layer (e.g. ["q_proj", "v_proj"]).

    Returns:
        List of dot-separated module name strings.
    """
    target_names: list[str] = []
    for layer_idx in layer_indices:
        for name, module in model.named_modules():
            # Match patterns like "model.layers.12.self_attn.q_proj"
            parts = name.split(".")
            if (
                str(layer_idx) in parts
                and parts[-1] in proj_names
            ):
                # Verify the layer index is in the right position
                # (handle both "layers.12" and "h.12" style naming)
                for i, p in enumerate(parts):
                    if p == str(layer_idx) and i > 0:
                        target_names.append(name)
                        break
    return sorted(set(target_names))
