#!/usr/bin/env bash
# Week 1 — Prepare all datasets
# Builds: BeaverTails, XSTest, WildGuard, Protected Mix
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "================================================================"
echo " PF-LoRA: Data Preparation (Script 01)"
echo "================================================================"

echo ""
echo "--- BeaverTails ---"
python -m src.data.build_all \
  --config configs/data/beavertails.yaml \
  --out artifacts/datasets/beavertails

echo ""
echo "--- XSTest ---"
python -m src.data.build_all \
  --config configs/data/xstest.yaml \
  --out artifacts/datasets/xstest

echo ""
echo "--- WildGuard ---"
python -m src.data.build_all \
  --config configs/data/wildguard.yaml \
  --out artifacts/datasets/wildguard

echo ""
echo "--- Protected Mix (D_gen) ---"
python -m src.data.build_all \
  --config configs/data/protected_mix.yaml \
  --out artifacts/datasets/protected_mix

echo ""
echo "================================================================"
echo " Data preparation complete."
echo " Check artifacts/datasets/ for outputs."
echo "================================================================"
