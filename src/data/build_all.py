"""
Data preparation entry point.

Usage:
    python -m src.data.build_all --config configs/data/beavertails.yaml --out artifacts/datasets/beavertails
    python -m src.data.build_all --config configs/data/xstest.yaml --out artifacts/datasets/xstest
    python -m src.data.build_all --config configs/data/wildguard.yaml --out artifacts/datasets/wildguard
    python -m src.data.build_all --config configs/data/protected_mix.yaml --out artifacts/datasets/protected_mix
    python -m src.data.build_all --all  # builds all four datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.data import beavertails, xstest, wildguard, protected_mix as protected_mix_mod


_DATASET_BUILDERS = {
    "beavertails": (beavertails.build, "configs/data/beavertails.yaml", "artifacts/datasets/beavertails"),
    "xstest": (xstest.build, "configs/data/xstest.yaml", "artifacts/datasets/xstest"),
    "wildguard": (wildguard.build, "configs/data/wildguard.yaml", "artifacts/datasets/wildguard"),
    "protected_mix": (
        protected_mix_mod.build,
        "configs/data/protected_mix.yaml",
        "artifacts/datasets/protected_mix",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PF-LoRA datasets")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a single dataset YAML config"
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output directory for the dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(_DATASET_BUILDERS.keys()),
        default=None,
        help="Which dataset to build (required unless --all is set)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Build all four datasets"
    )
    args = parser.parse_args()

    if args.all:
        for name, (builder_fn, default_cfg, default_out) in _DATASET_BUILDERS.items():
            print(f"\n{'='*60}")
            print(f"Building: {name}")
            print(f"{'='*60}")
            cfg = OmegaConf.load(default_cfg)
            builder_fn(cfg, Path(default_out))
        return

    if args.config is None and args.dataset is None:
        parser.error("Provide --config/--out or --dataset or --all")

    if args.dataset is not None:
        builder_fn, default_cfg, default_out = _DATASET_BUILDERS[args.dataset]
        cfg_path = args.config or default_cfg
        out_dir = Path(args.out or default_out)
    else:
        # Infer dataset type from config path
        cfg_path = args.config
        out_dir = Path(args.out) if args.out else None
        cfg_stem = Path(cfg_path).stem
        if cfg_stem not in _DATASET_BUILDERS:
            raise ValueError(
                f"Cannot infer dataset type from config '{cfg_path}'. "
                f"Use --dataset to specify one of: {list(_DATASET_BUILDERS.keys())}"
            )
        builder_fn, _, default_out = _DATASET_BUILDERS[cfg_stem]
        if out_dir is None:
            out_dir = Path(default_out)

    cfg = OmegaConf.load(cfg_path)
    builder_fn(cfg, out_dir)


if __name__ == "__main__":
    main()
