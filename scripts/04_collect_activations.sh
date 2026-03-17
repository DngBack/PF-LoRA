#!/usr/bin/env bash
# Week 3 — Collect base model activations on D_gen (protected mix)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODEL=${1:-"meta-llama/Llama-3.1-8B-Instruct"}
MODEL_SHORT=${MODEL##*/}
DATASET="artifacts/datasets/protected_mix/protected_mix.jsonl"
OUT="artifacts/activations/${MODEL_SHORT}_protected_mix"

echo "================================================================"
echo " PF-LoRA: Activation Collection (Script 04)"
echo " Model: $MODEL"
echo " Dataset: $DATASET"
echo " Output: $OUT"
echo "================================================================"

python -m src.subspace.collect_activations \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --layers 12 14 16 18 20 22 24 26 28 30 \
  --proj_names q_proj k_proj v_proj o_proj \
  --out "$OUT" \
  --max_seq_len 512 \
  --batch_size 8 \
  --shard_size 64 \
  --seed 42

echo ""
echo "================================================================"
echo " Activation collection complete."
echo " Shards written to: $OUT"
echo "================================================================"
