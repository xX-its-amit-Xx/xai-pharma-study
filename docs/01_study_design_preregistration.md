# Paper 1 — Preregistration & study design

*Phase 1 deliverable. Frozen before any results are inspected. Version 1.0, 2026-05-21.*
*Any deviation from this document during execution will be logged in `docs/deviations.md` with rationale.*

> **Working title.** *Can you trust the explanation? A reliability audit of feature
> attributions for ADMET and toxicity prediction.*

## 1. Background & objective
Post-hoc attributions (SHAP, LIME, Integrated Gradients) are increasingly used to
provide the "mechanistic interpretation" that regulators (OECD/EMA) expect from
QSAR/ADMET models. This study measures whether those attributions are reliable
enough for that role, on real endpoints, and how reliability changes under the
out-of-distribution (scaffold) conditions of real deployment. We do **not**
propose a new method here; we audit existing ones. (A new method is Paper 2,
gated on this paper's completion.)

## 2. Hypotheses (confirmatory)
Each hypothesis has a pre-specified test, effect-size measure, and falsification
condition. All comparisons use the random-null reference.

- **H1 — Faithfulness is not guaranteed over the null.**
  A non-trivial fraction (pre-registered: ≥ 20%) of (endpoint × representation ×
  model × method) cells are *not* significantly more faithful than a content-matched
  random attribution baseline (paired test across molecules, BH-corrected).
  *Falsified if* < 20% of cells fail to beat the null.

- **H2 — Reliability degrades out-of-distribution.**
  Faithfulness and stability are *worse* under scaffold split than under random
  split for the same model, on average (paired across endpoints).
  *Primary test:* one-sided Wilcoxon signed-rank over endpoints on the
  scaffold-minus-random difference, separately for faithfulness and stability.
  *Falsified if* no significant degradation (α = 0.05) for either metric.

- **H3 — Explainers disagree, heterogeneously, and safety-critical endpoints are
  not more self-consistent.**
  Mean pairwise SHAP/LIME/IG rank agreement is low (pre-registered descriptive
  threshold: median Spearman < 0.5 across endpoints) and the toxicity endpoints
  (hERG, DILI, Ames, LD50) do not show higher agreement than the others (Mann–
  Whitney, two-sided). *Reported as descriptive + test; not falsified, characterized.*

- **H4 — Sanity-check failures occur in practice.**
  A non-trivial fraction (≥ 15%) of (model × method) combinations fail the Adebayo
  model-randomization sanity check (similarity above threshold, see §6.4).
  *Falsified if* < 15% fail.

- **H5 — Representation dominates method.**
  Variance in reliability metrics is explained more by molecular representation
  (descriptors / ECFP / GNN-embedding) than by attribution method.
  *Test:* mixed-effects / ANOVA variance decomposition (η² for representation vs
  method factors). *Reported as effect sizes.*

## 3. Design
Factorial, fully crossed where computationally feasible:

| Factor | Levels |
| --- | --- |
| **Endpoint** (dataset) | 22 TDC ADMET datasets (subset justified in §4 if compute-bound) |
| **Representation** | (a) RDKit 2D descriptors (~200), (b) ECFP4/Morgan 2048-bit, (c) learned GNN embedding |
| **Model** | RandomForest, HistGradientBoosting, MLP (descriptors/ECFP); GIN/GCN (graph) |
| **Attribution method** | SHAP, LIME, Integrated Gradients (gradient models only), **random null** (always) |
| **Split** | scaffold (primary) and random (contrast for H2) |
| **Reliability metric** | faithfulness, stability, agreement, model-randomization sanity |

Primary unit of analysis: one (endpoint, representation, model, method, split)
*cell*; within a cell, metrics are computed over a fixed held-out molecule sample.

## 4. Datasets
TDC ADMET benchmark group (scaffold splits provided). Full set of 22 (sizes
475–13,130). **Compute-scaling rule (pre-registered):** if the full factorial is
infeasible within budget, we retain *all 4 toxicity endpoints* (hERG, DILI, Ames,
LD50) plus a stratified sample covering each ADME category (≥ 1 regression and ≥ 1
classification per category), for ≥ 12 endpoints total. Selection is by dataset
size/category coverage, decided **before** running attributions, and logged.

## 5. Models & training
- Hyperparameters: light, fixed, documented defaults (no leakage from test). No
  per-endpoint tuning beyond a small fixed grid on the validation split; the
  *reliability* questions are conditional on a competently-trained model, not on
  SOTA accuracy. We report each model's predictive performance (TDC metric) so
  reliability is interpreted only for models that actually learned the endpoint
  (pre-registered floor: models below a trivial-baseline margin are excluded from
  reliability claims and reported separately).
- Reproducibility: fixed seeds, environment lockfile, all artifacts hashed.

## 6. Metrics (operational definitions)
All implemented in the `xai-eval-harness` (already built and tested) and extended
as needed; every metric references the random null.

### 6.1 Faithfulness
ROAR-style. Primary: `mask_and_repredict` (mean-imputation reference) AOPC-style
area between the degradation curve and baseline; **normalized against the random
null** (per Normalized-AOPC critique). Secondary on a subset: `remove_and_retrain`
(true ROAR) to confirm the cheap proxy tracks the expensive ground truth
(report rank correlation between the two). Higher = more faithful.

### 6.2 Stability
Local Lipschitz estimate under L∞ feature perturbation within a pre-set ε
(separately calibrated per representation; for descriptors/ECFP, ε on standardized
features; documented). Report mean and worst-case sensitivity. Lower = more stable.

### 6.3 Agreement
Per-molecule Kendall τ, Spearman ρ, and top-k Jaccard between method pairs;
aggregate mean **and variance** across molecules; endpoint-level matrices.

### 6.4 Model-randomization sanity (Adebayo)
Similarity (|rank corr|) between attributions on the trained model vs a randomized
model. For torch models (MLP/GNN): true weight reinitialization incl. cascading.
For tree/non-parametric: label-permutation refit analogue (limitation stated
explicitly). Pre-registered fail threshold: similarity ≥ 0.5. Lower = better.

## 7. Statistical analysis plan
- **Within-cell uncertainty:** bootstrap (1,000 resamples) over molecules for each
  metric → 95% CIs.
- **Null comparison (H1):** paired bootstrap / permutation test of method vs random
  within cell; one-sided (method > null faithfulness).
- **OOD effect (H2):** Wilcoxon signed-rank over endpoints, one-sided.
- **Endpoint contrasts (H3):** Mann–Whitney (toxicity vs non-toxicity).
- **Variance decomposition (H5):** linear mixed model with random intercepts for
  endpoint; fixed effects for representation and method; report η²/ω².
- **Multiple comparisons:** Benjamini–Hochberg FDR at q = 0.05 across the full
  family of cell-level tests; corrected and uncorrected both reported.
- **Effect sizes always reported** alongside p-values; we privilege effect sizes
  and CIs over significance stars.

## 8. What would make us report a *negative* / null result proudly
If attributions turn out broadly reliable (H1/H4 falsified, no OOD degradation),
that is a publishable, decision-relevant reassurance and we will report it as the
headline. The study is designed to be informative under either outcome.

## 9. Threats to validity (and mitigations)
- *Faithfulness-by-masking is off-distribution.* → cross-check with retrain ROAR;
  report both; mean-imputation reference; normalized to null.
- *Tree-model "randomization" via label permutation is an analogue, not weight
  reinit.* → stated as a limitation; torch models give the true test; conclusions
  about sanity failures separated by model class.
- *Representation confounds.* → representation is an explicit factor (H5), not a
  nuisance.
- *Single split realization.* → repeat over ≥ 3 seeds for splits where size allows;
  report variability.
- *Compute.* → pre-registered scaling rule (§4) chosen before results.

## 10. Deliverables
A reproducible pipeline, a results database (per-cell metrics + CIs), the figure
set (§ to be enumerated in P5), and a manuscript. Reproducibility package:
environment lock, seeds, data-access scripts (TDC is programmatic), and a one-command
rerun.

## 11. Out of scope (explicitly)
- Proposing a new attribution method (→ Paper 2).
- Wet-lab validation of mechanistic claims.
- Foundation-model / LLM-based property predictors (possible extension, not core).
