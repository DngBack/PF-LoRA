# PF-LoRA: Protected-Function LoRA for Near-Zero-Tax Rejection Alignment

Parameter-efficient fine-tuning for safety alignment that explicitly preserves
general capabilities by penalizing interference on a protected activation subspace.

See [docs/mini_paper_protected_function_lora.md](docs/mini_paper_protected_function_lora.md) for the full method description.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## 8-Week Experiment Pipeline

```
scripts/01_prepare_data.sh          # Week 1: download & build all datasets
scripts/02_run_baseline_eval.sh     # Week 1: evaluate base model (no fine-tuning)
scripts/03_train_lora_baseline.sh   # Week 2: train standard LoRA baseline
scripts/04_collect_activations.sh   # Week 3: collect D_gen hidden activations
scripts/05_build_subspaces.sh       # Week 4: compute U_k, Λ_k per layer
scripts/06_train_protected_lora.sh  # Week 5: train PF-LoRA with penalty
scripts/07_run_full_eval.sh         # Week 7/8: full evaluation suite
scripts/08_make_paper_tables.sh     # Week 8: generate paper tables + figures
```

## Directory Structure

```
PF-LoRA/
├── configs/             YAML hyperparameter configs
│   ├── model/           model loading configs (llama31, mistral7b)
│   ├── data/            dataset configs (beavertails, xstest, wildguard, protected_mix)
│   ├── train/           training configs (lora_baseline, protected_lora, ablation)
│   └── eval/            evaluation configs (safety, capability, full)
├── scripts/             bash scripts 01–08
├── src/
│   ├── data/            dataset loaders and builders
│   ├── eval/            lm-eval wrapper, refusal metrics, report aggregator
│   ├── models/          LoRA factory
│   ├── train/           SFT baseline + PF-LoRA protected trainer
│   ├── methods/         protected_penalty, projection (ablation), split_rank (ablation)
│   ├── subspace/        activation hooks, covariance sketch, SVD, subspace analysis
│   ├── reporting/       table and figure generators
│   └── utils/           seeding, logging, checkpointing
├── artifacts/           gitignored: datasets, activations, subspaces, checkpoints
├── results/             tables, figures, eval logs
└── notebooks/
    ├── debug_data.ipynb          inspect all four datasets
    ├── inspect_spectrum.ipynb    analyze eigenvalue spectra (Week 4)
    └── analyze_tradeoff.ipynb    compare methods, Pareto plots
```

## Core Method

The PF-LoRA training objective:

```
L = L_rej  +  λ * Σ_l ‖ B_l A_l U_{l,k} Λ_{l,k}^{1/2} ‖_F²
```

where `(U_{l,k}, Λ_{l,k})` are the top-k eigenvectors/eigenvalues of the
activation covariance of layer `l` on the protected capability distribution `D_gen`.

## Key Configs

| Config | Key parameters |
|--------|---------------|
| `configs/train/lora_baseline.yaml` | `lora_rank=16, lora_alpha=32, lr=1e-4` |
| `configs/train/protected_lora.yaml` | `lambda_prot=1e-3, k_subspace=64, protect_layers=[12..30]` |
| `configs/train/ablation.yaml` | 12 ablation variants (lambda sweep, k sweep, hard projection, split rank) |

## Datasets

| Dataset | Role |
|---------|------|
| BeaverTails | Rejection/alignment training (333k QA pairs) |
| XSTest | Over-refusal evaluation (250 safe + 200 unsafe prompts) |
| WildGuardTest | Refusal F1 + harmfulness evaluation |
| MMLU-Pro + HellaSwag + ARC + GSM8K + TruthfulQA | Protected capability audit |
