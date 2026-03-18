#!/usr/bin/env bash
# Week 4 — Build protected subspace artifacts (U_k, Λ_k) from activations
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODEL_SHORT=${1:-"Qwen2.5-1.5B-Instruct"}
ACTIVATIONS_DIR="artifacts/activations/${MODEL_SHORT}_protected_mix"
OUT="artifacts/subspaces/${MODEL_SHORT}_protected_mix"

echo "================================================================"
echo " PF-LoRA: Build Protected Subspaces (Script 05)"
echo " Activations: $ACTIVATIONS_DIR"
echo " Output: $OUT"
echo "================================================================"

python -m src.subspace.build_subspaces \
  --activations_dir "$ACTIVATIONS_DIR" \
  --k_list 32 64 128 256 \
  --out "$OUT" \
  --svd_method eigh \
  --regularize 1e-6

echo ""
echo "--- Generating spectrum analysis figures ---"
python -m src.subspace.analyze \
  "$OUT" \
  "${OUT}/analysis"

echo ""
echo "================================================================"
echo " Subspace build complete."
echo " Artifacts: $OUT"
echo " Spectrum plots: ${OUT}/analysis"
echo "================================================================"
