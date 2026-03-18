#!/usr/bin/env bash
# Week 7/8 — Run full evaluation suite on a trained adapter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODEL_CONFIG=${1:-"configs/model/qwen25_15b_instruct.yaml"}
ADAPTER_DIR=${2:-"artifacts/checkpoints/protected_lora_qwen25_lambda1e-3_k64"}
MODEL_SHORT="qwen25"

BASE_MODEL=$(python -c "import yaml; c=yaml.safe_load(open('$MODEL_CONFIG')); print(c['model_name_or_path'])")
ADAPTER_NAME=$(basename "$ADAPTER_DIR")
OUT="results/logs/${ADAPTER_NAME}.json"

echo "================================================================"
echo " PF-LoRA: Full Evaluation Suite (Script 07)"
echo " Base model: $BASE_MODEL"
echo " Adapter: $ADAPTER_DIR"
echo " Output: $OUT"
echo "================================================================"

python -m src.eval.report \
  --base_model "$BASE_MODEL" \
  --adapter "$ADAPTER_DIR" \
  --config configs/eval/full_eval.yaml \
  --xstest_dir artifacts/datasets/xstest \
  --wildguard_dir artifacts/datasets/wildguard \
  --base_results "results/logs/base_${MODEL_SHORT}.json" \
  --out "$OUT"

echo ""
echo "================================================================"
echo " Full evaluation complete. Results: $OUT"
echo "================================================================"
