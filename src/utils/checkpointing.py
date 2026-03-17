"""Adapter checkpoint save/load utilities."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def save_adapter_checkpoint(
    model,
    output_dir: str | Path,
    tokenizer=None,
    metadata: Optional[dict] = None,
    tag: Optional[str] = None,
) -> Path:
    """
    Save a PEFT adapter to disk with optional metadata.

    Args:
        model: PEFT model (or plain model).
        output_dir: Base directory; a timestamped subdirectory is created.
        tokenizer: Optional tokenizer to save alongside the adapter.
        metadata: Extra metadata dict to write as meta.json.
        tag: Optional tag appended to checkpoint directory name.

    Returns:
        Path to the checkpoint directory.
    """
    output_dir = Path(output_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    ckpt_dir = output_dir / f"checkpoint_{timestamp}{suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(ckpt_dir)
    else:
        import torch
        torch.save(model.state_dict(), ckpt_dir / "model_state_dict.pt")

    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)

    if metadata:
        (ckpt_dir / "meta.json").write_text(json.dumps(metadata, indent=2))

    # Also write a "latest" symlink for convenience
    latest_link = output_dir / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(ckpt_dir.name)
    except OSError:
        pass

    print(f"Checkpoint saved → {ckpt_dir}")
    return ckpt_dir


def load_adapter_checkpoint(
    base_model,
    adapter_dir: str | Path,
) -> Any:
    """
    Load a PEFT adapter onto a base model.

    Args:
        base_model: Already-loaded base model (without adapter).
        adapter_dir: Path to saved PEFT adapter directory.

    Returns:
        PeftModel with adapter loaded.
    """
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    if (adapter_dir / "latest").is_symlink():
        adapter_dir = adapter_dir / "latest"

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    print(f"Adapter loaded from {adapter_dir}")
    return model


def get_latest_checkpoint(output_dir: str | Path) -> Optional[Path]:
    """Return path to the most recent checkpoint in output_dir."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    latest = output_dir / "latest"
    if latest.is_symlink():
        return latest.resolve()
    checkpoints = sorted(output_dir.glob("checkpoint_*"))
    return checkpoints[-1] if checkpoints else None
