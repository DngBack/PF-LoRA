#!/usr/bin/env bash
# Week 2 — Train standard LoRA baseline on BeaverTails
# Sweeps over rank=[8,16,32] and module sets=[attn, attn+mlp]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODEL_CONFIG=${1:-"configs/model/llama31_8b_instruct.yaml"}
TRAIN_CONFIG=${2:-"configs/train/lora_baseline.yaml"}
MODEL_SHORT="llama31"

echo "================================================================"
echo " PF-LoRA: Standard LoRA Baseline Training (Script 03)"
echo " Model config: $MODEL_CONFIG"
echo " Train config: $TRAIN_CONFIG"
echo "================================================================"

# Default run (r=16, attention only — from lora_baseline.yaml)
OUTPUT_DIR="artifacts/checkpoints/lora_baseline_${MODEL_SHORT}_r16"
echo ""
echo "--- Training: r=16, attention only ---"
python -m src.train.sft_baseline \
  --model_config "$MODEL_CONFIG" \
  --train_config "$TRAIN_CONFIG" \
  --dataset artifacts/datasets/beavertails/train.jsonl \
  --val_xstest artifacts/datasets/xstest/all.jsonl \
  --val_wildguard artifacts/datasets/wildguard/val.jsonl \
  --output_dir "$OUTPUT_DIR" \
  --run_name "lora_baseline_${MODEL_SHORT}_r16"

echo ""
echo "--- Evaluating: r=16 baseline ---"
python -m src.eval.report \
  --base_model "$(python -c "import yaml; c=yaml.safe_load(open('$MODEL_CONFIG')); print(c['model_name_or_path'])")" \
  --adapter "$OUTPUT_DIR" \
  --config configs/eval/full_eval.yaml \
  --xstest_dir artifacts/datasets/xstest \
  --wildguard_dir artifacts/datasets/wildguard \
  --base_results "results/logs/base_${MODEL_SHORT}.json" \
  --out "results/logs/lora_baseline_${MODEL_SHORT}_r16.json"

echo ""
echo "================================================================"
echo " LoRA baseline training complete."
echo " Best adapter: $OUTPUT_DIR"
echo "================================================================"
