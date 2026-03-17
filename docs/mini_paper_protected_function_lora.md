# Protected-Function LoRA for Near-Zero-Tax Rejection Alignment

## Abstract
Parameter-efficient fine-tuning (PEFT) is widely used to align large language models (LLMs) for refusal and safety behavior. However, a persistent practical issue is **alignment tax**: after fine-tuning for rejection or safety compliance, the model often becomes overly conservative and suffers measurable degradation on benign general-capability tasks such as knowledge QA, reasoning, and truthful response generation. We study this problem through a functional perspective. Instead of treating forgetting as generic weight drift, we define a **protected general-capability distribution** and penalize LoRA updates according to the hidden-state interference they induce on that distribution. Concretely, for each adapted layer, we estimate a low-rank protected activation subspace from a benign capability mixture, then regularize the LoRA update in proportion to its energy on that subspace. This yields a practical training objective that is memory-efficient, compatible with standard PEFT stacks, and admits an interpretable upper bound: the increase in expected hidden perturbation on the protected distribution is controlled by the protected-subspace penalty plus the spectral tail omitted by truncation. We propose **Protected-Function LoRA (PF-LoRA)**, derive its main guarantee, and outline an experimental plan on BeaverTails, XSTest, WildGuard, MMLU-Pro, HellaSwag, ARC-Challenge, GSM8K, and TruthfulQA. The goal is not merely safer refusal, but **high refusal quality with low capability tax**.

## 1. Introduction
Fine-tuning LLMs for safety and refusal is now standard deployment practice. A model should decline genuinely harmful or disallowed requests while remaining helpful on benign prompts. In practice, however, conventional supervised fine-tuning or LoRA-based alignment often introduces an undesirable tradeoff: improved refusal on unsafe prompts is accompanied by **over-refusal** on safe prompts and measurable degradation on general tasks. We refer to this degradation as **alignment tax**.

Existing PEFT methods are parameter-efficient, but they do not explicitly protect the model’s pretrained functional behavior on benign data. Classical anti-forgetting methods from continual learning, such as EWC-style regularization, operate in parameter space and often rely on approximations that discard the geometry of hidden representations. Orthogonality-inspired LoRA methods mitigate task interference, but they do not directly formalize the deployment objective relevant here: preserving a target distribution of benign capabilities while learning rejection behavior.

This paper proposes a simple principle: if alignment tax is manifested as unwanted perturbation of hidden representations on benign inputs, then training should directly penalize the **expected hidden interference** induced by the adapter on a protected benign distribution. We operationalize this principle using a truncated protected subspace per adapted layer, extracted from activation statistics of a curated general-capability mixture. The resulting method, PF-LoRA, preserves the most energetic directions of benign behavior while leaving substantial residual capacity for learning rejection.

Our contributions are:

1. We formalize **alignment tax control** as a protected-function preservation problem for PEFT.
2. We define a layerwise **protected interference energy** and show that it upper-bounds hidden perturbation on benign inputs.
3. We propose a practical **low-rank protected-subspace regularizer** compatible with LoRA.
4. We outline a realistic evaluation protocol that simultaneously measures refusal quality, over-refusal, and general-capability preservation.

## 2. Problem Setup
Let a pretrained LLM with parameters $\theta_0$ be adapted using LoRA. For a target linear module at layer $l$, the base weight is $W_l^0 \in \mathbb{R}^{d'_l \times d_l}$, and the LoRA update is

$$
\Delta W_l = B_l A_l,
$$

where $A_l \in \mathbb{R}^{r_l \times d_l}$, $B_l \in \mathbb{R}^{d'_l \times r_l}$, and $r_l \ll d_l$.

We consider two distributions:

- $\mathcal{D}_{\text{rej}}$: rejection/alignment training distribution, containing prompts that should be refused or safely redirected, plus benign prompts that should still be answered.
- $\mathcal{D}_{\text{gen}}$: protected benign capability distribution, representing the behaviors we do not wish to degrade.

For an input $x \sim \mathcal{D}_{\text{gen}}$, let $h_l(x) \in \mathbb{R}^{d_l}$ denote the hidden input to layer $l$. The LoRA-induced hidden perturbation at that layer is

$$
\delta_l(x) = \Delta W_l h_l(x).
$$

The core objective is to learn rejection behavior on $\mathcal{D}_{\text{rej}}$ while keeping $\delta_l(x)$ small in expectation over $\mathcal{D}_{\text{gen}}$.

## 3. Protected Interference Energy
We define the protected activation covariance of layer $l$ as

$$
\Sigma_l = \mathbb{E}_{x \sim \mathcal{D}_{\text{gen}}}[h_l(x) h_l(x)^\top].
$$

The **protected interference energy** of a LoRA update at layer $l$ is

$$
\mathcal{I}_l(\Delta W_l)
:= \mathbb{E}_{x \sim \mathcal{D}_{\text{gen}}}\|\Delta W_l h_l(x)\|_2^2.
$$

Using the covariance definition,

$$
\mathcal{I}_l(\Delta W_l) = \operatorname{Tr}(\Delta W_l \Sigma_l \Delta W_l^\top).
$$

This quantity has a direct interpretation: it is the average squared hidden perturbation induced by the adapter on the protected benign distribution. If this quantity is small across adapted layers, the adapter is functionally close to the base model on benign inputs.

## 4. Low-Rank Protected Subspace Approximation
Directly storing or decomposing the full covariance $\Sigma_l$ may be expensive for large hidden dimensions. We therefore approximate it by its top-$k_l$ eigensubspace:

$$
\Sigma_l \approx U_{l,k} \Lambda_{l,k} U_{l,k}^\top,
$$

where $U_{l,k} \in \mathbb{R}^{d_l \times k_l}$ contains the principal directions and $\Lambda_{l,k}$ is diagonal with the top eigenvalues.

This approximation can be obtained efficiently from a protected activation corpus using streaming covariance sketches or randomized SVD. The storage cost becomes $O(d_l k_l)$ instead of $O(d_l^2)$.

We then define the truncated protected penalty

$$
\mathcal{R}_{\text{prot}} = \sum_l \|B_l A_l U_{l,k} \Lambda_{l,k}^{1/2}\|_F^2.
$$

This penalty is exactly the interference energy restricted to the dominant protected subspace.

## 5. Training Objective
Let $\mathcal{L}_{\text{rej}}$ denote the supervised alignment loss on $\mathcal{D}_{\text{rej}}$. Our training objective is

$$
\mathcal{L}_{\text{PF-LoRA}} = \mathcal{L}_{\text{rej}} + \lambda \sum_l \|B_l A_l U_{l,k} \Lambda_{l,k}^{1/2}\|_F^2,
$$

where $\lambda > 0$ controls the tradeoff between alignment and capability preservation.

Intuitively, the second term discourages the adapter from changing directions that are heavily used by benign capability behavior. Unlike hard orthogonality constraints, this is a **soft budget**: the optimizer may still exploit protected directions if the gain in rejection loss is sufficiently large, but such interference is explicitly penalized.

An equivalent constrained form is

$$
\min_{\Delta} \; \mathcal{L}_{\text{rej}}(\Delta)
\quad \text{subject to} \quad
\sum_l \|B_l A_l U_{l,k} \Lambda_{l,k}^{1/2}\|_F^2 \le \tau,
$$

which interprets preservation as an auditable tax budget.

## 6. Main Theoretical Result
The key question is whether truncating to the top-$k_l$ protected directions still controls total interference. Let

$$
\Sigma_l = U_{l,k} \Lambda_{l,k} U_{l,k}^\top + \Sigma_l^{(\perp)}.
$$

Then:

### Theorem 1 (Truncated Protected-Interference Bound)
For any layer $l$ and LoRA update $\Delta W_l$,

$$
\mathcal{I}_l(\Delta W_l)
\le
\|\Delta W_l U_{l,k} \Lambda_{l,k}^{1/2}\|_F^2
+ \|\Delta W_l\|_2^2 \operatorname{Tr}(\Sigma_l^{(\perp)}).
$$

#### Proof sketch
Expand

$$
\mathcal{I}_l(\Delta W_l)=\operatorname{Tr}(\Delta W_l \Sigma_l \Delta W_l^\top)
= \operatorname{Tr}(\Delta W_l U_{l,k}\Lambda_{l,k}U_{l,k}^\top \Delta W_l^\top)
+ \operatorname{Tr}(\Delta W_l \Sigma_l^{(\perp)} \Delta W_l^\top).
$$

The first term is exactly

$$
\|\Delta W_l U_{l,k}\Lambda_{l,k}^{1/2}\|_F^2.
$$

For the second term, using $\Sigma_l^{(\perp)} \succeq 0$ and standard trace inequalities,

$$
\operatorname{Tr}(\Delta W_l \Sigma_l^{(\perp)} \Delta W_l^\top)
\le \|\Delta W_l\|_2^2 \operatorname{Tr}(\Sigma_l^{(\perp)}).
$$

Combining the two parts yields the result.

This theorem says the full benign interference is controlled by two factors:
1. the protected penalty on the retained principal benign directions, and
2. the spectral tail omitted by truncation.

Hence, if the benign activation spectrum is sufficiently concentrated, low-rank protection is theoretically justified.

## 7. Functional Drift Corollary
Suppose the remainder of the network from layer $l$ to the final logits is $L_l$-Lipschitz with respect to the layer output. Then the average logit drift on $\mathcal{D}_{\text{gen}}$ can be bounded by the accumulated hidden interference. Informally,

$$
\mathbb{E}_{x \sim \mathcal{D}_{\text{gen}}}\|f_{\theta_0 + \Delta}(x) - f_{\theta_0}(x)\|_2
\le \sum_l L_l \sqrt{\mathcal{I}_l(\Delta W_l)}.
$$

Therefore, reducing protected interference energy reduces output drift on benign inputs, which in turn supports smaller degradation on general-capability tasks.

## 8. Why This Avoids the Main Failure Modes
A natural criticism is that enforcing orthogonality to benign features may overly restrict learning capacity, causing underfitting on rejection tasks. PF-LoRA addresses this in three ways.

First, the constraint is **soft**, not exact. The model may still use protected directions when necessary.

Second, only a **top-$k$ protected subspace** is penalized. If the hidden dimension is large and $k$ is moderate, substantial residual capacity remains for alignment.

Third, the objective targets **functionally meaningful directions** derived from benign activations rather than uniformly penalizing all weight changes. This is more targeted than generic $L_2$ regularization and more geometrically faithful than diagonal importance weighting.

## 9. Experimental Plan
We propose the following evaluation design.

### Training data
- **BeaverTails** as the primary rejection/alignment training set.
- Optional mixture with additional benign prompts to stabilize helpful behavior.

### Safety and refusal evaluation
- **XSTest** to measure both unsafe refusal and over-refusal on safe prompts.
- **WildGuardTest** for refusal quality and harmfulness-related evaluation.

### Protected capability evaluation
- **MMLU-Pro / MMLU** for broad academic knowledge.
- **HellaSwag** for commonsense inference.
- **ARC-Challenge** for science reasoning.
- **GSM8K** for mathematical reasoning.
- **TruthfulQA** for truthfulness and false-belief susceptibility.

### Models
- Main model: Llama-3.1-8B-Instruct.
- Secondary model: Mistral-7B-Instruct.

### Baselines
- Base model without fine-tuning.
- Standard LoRA.
- LoRA + $L_2$ regularization.
- LoRA + diagonal EWC-style penalty.
- LoRA + benign replay mixture.

### Metrics
1. **Unsafe refusal rate**.
2. **Safe false refusal rate**.
3. **Refusal F1**.
4. **Average protected capability score** across benign benchmarks.
5. **Capability tax**, defined as the post-training score drop relative to the base model.
6. **Protected interference energy** estimated empirically on a held-out benign set.

### Main expected result
PF-LoRA should trace out a better Pareto frontier than standard LoRA: at similar refusal quality, it should incur lower over-refusal and lower capability tax.

## 10. Limitations
This work has several limitations.

First, the guarantee depends on the choice of the protected benign distribution. If $\mathcal{D}_{\text{gen}}$ is poorly chosen, the protected subspace may fail to capture important capabilities.

Second, the theory controls hidden interference rather than exact task accuracy. While functional drift provides a principled surrogate, the final benchmark relationship remains empirical.

Third, extracting protected subspaces still requires nontrivial engineering for large models, especially when adapting many layers.

Finally, if rejection alignment genuinely requires altering some benign directions, a strong protection budget may trade off too aggressively against alignment performance.

## 11. Conclusion
We presented PF-LoRA, a principled approach to reducing alignment tax in rejection fine-tuning. The core idea is simple: preserve the dominant hidden directions used by benign general capability, and penalize LoRA updates according to how much they perturb those directions. This yields a low-rank, memory-efficient regularizer with a clean interference interpretation and a truncated-spectrum theoretical guarantee. More broadly, the paper advocates a shift in perspective: alignment should not only be evaluated by how often a model refuses unsafe requests, but also by how well it preserves the capabilities users actually want to keep.

## References
- Kirkpatrick et al. *Overcoming catastrophic forgetting in neural networks.* PNAS, 2017.
- Wang et al. *BeaverTails: Towards improved safety alignment of LLMs.* 2023.
- Röttger et al. *XSTest: A test suite for identifying exaggerated safety behaviours in LLMs.* 2023.
- Han et al. *WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and refusal.* 2024.
- Hendrycks et al. *Measuring massive multitask language understanding.* 2020.
- Wang et al. *MMLU-Pro: A more robust and challenging multi-task language understanding benchmark.* 2024.
- Zellers et al. *HellaSwag: Can a machine really finish your sentence?* 2019.
- Clark et al. *Think you have solved question answering? Try ARC.* 2018.
- Cobbe et al. *Training verifiers to solve math word problems.* 2021.
- Lin et al. *TruthfulQA: Measuring how models mimic human falsehoods.* 2021.

