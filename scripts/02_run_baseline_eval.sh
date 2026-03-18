#!/usr/bin/env bash
# Week 1 — Run base model evaluation (no fine-tuning)
# Produces baseline scores for all capability and safety benchmarks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
MODEL_SHORT=${MODEL##*/}
OUT="results/logs/base_${MODEL_SHORT}.json"

echo "================================================================"
echo " PF-LoRA: Base Model Evaluation (Script 02)"
echo " Model: $MODEL"
echo " Output: $OUT"
echo "================================================================"

python -m src.eval.report \
  --base_model "$MODEL" \
  --adapter "" \
  --config configs/eval/full_eval.yaml \
  --xstest_dir artifacts/datasets/xstest \
  --wildguard_dir artifacts/datasets/wildguard \
  --out "$OUT"

echo ""
echo "================================================================"
echo " Base eval complete. Results: $OUT"
echo "================================================================"
