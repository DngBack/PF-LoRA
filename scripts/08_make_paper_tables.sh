#!/usr/bin/env bash
# Week 8 — Generate paper tables and figures from all evaluation results
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODEL_SHORT="llama31"
SUBSPACES="artifacts/subspaces/Llama-3.1-8B-Instruct_protected_mix"

echo "================================================================"
echo " PF-LoRA: Paper Tables & Figures (Script 08)"
echo "================================================================"

# --- Main results table ---
echo ""
echo "--- Generating main results table ---"
python -m src.reporting.make_tables \
  --inputs \
    results/logs/base_${MODEL_SHORT}.json \
    results/logs/lora_baseline_${MODEL_SHORT}_r16.json \
    results/logs/protected_lora_${MODEL_SHORT}_lambda1e-3_k64.json \
  --names \
    "Base Model" \
    "LoRA Baseline (r=16)" \
    "PF-LoRA (λ=1e-3, k=64)" \
  --out_dir results/tables

# --- Ablation table (all protected LoRA runs) ---
echo ""
echo "--- Generating ablation table ---"
ABLATION_FILES=$(ls results/logs/protected_lora_${MODEL_SHORT}_*.json 2>/dev/null | tr '\n' ' ')
if [ -n "$ABLATION_FILES" ]; then
  python -m src.reporting.make_tables \
    --inputs $ABLATION_FILES \
    --out_dir results/tables \
    --ablation
else
  echo "  No ablation result files found. Run script 06 with multiple lambda/k values first."
fi

# --- Paper figures ---
echo ""
echo "--- Generating paper figures ---"
ALL_RESULTS=$(ls results/logs/*.json 2>/dev/null | tr '\n' ' ')
if [ -n "$ALL_RESULTS" ]; then
  python -m src.reporting.make_figures \
    --inputs $ALL_RESULTS \
    --out_dir results/figures \
    --subspaces_dir "$SUBSPACES"
else
  echo "  No result files found in results/logs/. Run evaluations first."
fi

echo ""
echo "================================================================"
echo " Done. Outputs:"
echo "   Tables:  results/tables/"
echo "   Figures: results/figures/"
echo "================================================================"
