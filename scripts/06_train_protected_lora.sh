#!/usr/bin/env bash
# Week 5/6 — Train Protected-Function LoRA
# Supports lambda sweep via environment variable overrides.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODEL_CONFIG=${1:-"configs/model/llama31_8b_instruct.yaml"}
TRAIN_CONFIG=${2:-"configs/train/protected_lora.yaml"}
MODEL_SHORT="llama31"
SUBSPACES="artifacts/subspaces/Llama-3.1-8B-Instruct_protected_mix"

# PF-LoRA hyperparameter sweep
# To run a single config: set LAMBDA and K_SUBSPACE before calling this script
LAMBDA=${LAMBDA:-"1e-3"}
K_SUBSPACE=${K_SUBSPACE:-"64"}

OUTPUT_DIR="artifacts/checkpoints/protected_lora_${MODEL_SHORT}_lambda${LAMBDA}_k${K_SUBSPACE}"
RUN_NAME="pf_lora_${MODEL_SHORT}_lambda${LAMBDA}_k${K_SUBSPACE}"

echo "================================================================"
echo " PF-LoRA: Protected LoRA Training (Script 06)"
echo " Model config: $MODEL_CONFIG"
echo " Lambda: $LAMBDA | k: $K_SUBSPACE"
echo " Subspaces: $SUBSPACES"
echo " Output: $OUTPUT_DIR"
echo "================================================================"

python -m src.train.protected_sft \
  --model_config "$MODEL_CONFIG" \
  --train_config "$TRAIN_CONFIG" \
  --dataset artifacts/datasets/beavertails/train.jsonl \
  --subspaces "$SUBSPACES" \
  --val_dataset artifacts/datasets/wildguard/val.jsonl \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME"

echo ""
echo "--- Evaluating trained PF-LoRA adapter ---"
BASE_MODEL=$(python -c "import yaml; c=yaml.safe_load(open('$MODEL_CONFIG')); print(c['model_name_or_path'])")
python -m src.eval.report \
  --base_model "$BASE_MODEL" \
  --adapter "$OUTPUT_DIR" \
  --config configs/eval/full_eval.yaml \
  --xstest_dir artifacts/datasets/xstest \
  --wildguard_dir artifacts/datasets/wildguard \
  --base_results "results/logs/base_${MODEL_SHORT}.json" \
  --out "results/logs/protected_lora_${MODEL_SHORT}_lambda${LAMBDA}_k${K_SUBSPACE}.json"

echo ""
echo "================================================================"
echo " PF-LoRA training complete."
echo " Adapter: $OUTPUT_DIR"
echo "================================================================"

# --- Lambda sweep convenience block ---
# Uncomment to run all lambda values:
# for LAMBDA in 1e-4 1e-3 1e-2 1e-1; do
#   LAMBDA=$LAMBDA K_SUBSPACE=64 bash scripts/06_train_protected_lora.sh
# done
