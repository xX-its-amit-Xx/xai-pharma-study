# Faithful but not self-consistent: a reliability audit of feature attributions for ADMET and toxicity models

*Draft v0.1 (P6). All numbers trace to `results/` (commit-pinned). Author: Amit Shenoy.*

## Abstract
Feature-attribution methods (SHAP, LIME, Integrated Gradients) are increasingly used to provide the "mechanistic interpretation" that regulators (OECD QSAR framework; EMA 2024 reflection on AI/ML) expect from QSAR/ADMET models, and to guide which molecular features medicinal chemists act on. Yet the reliability of these attributions in the molecular-property setting has never been systematically audited. We trained 144 models spanning 12 Therapeutics Data Commons ADMET/toxicity endpoints, two molecular representations (interpretable 2D descriptors and 2048-bit ECFP4 fingerprints), and three model classes (random forest, gradient boosting, MLP), under both scaffold (deployment-realistic, out-of-distribution) and random splits. For the 142 models that cleared a trivial-baseline floor, we measured four reliability properties — faithfulness, stability, cross-method agreement, and the Adebayo model-randomization sanity check — each referenced against a content-matched random null, and tested five preregistered hypotheses. Two pessimistic hypotheses were **falsified**: attributions usually *do* beat the random null on faithfulness (only 12% of method–cells fail), and their faithfulness and stability do *not* significantly degrade under scaffold shift, even though model accuracy does. But two failure modes are real: explainers **agree only modestly** on which features matter — at an adequate LIME sampling budget, SHAP and LIME reach a mean rank correlation of just 0.34 on interpretable descriptors (range 0.18–0.50; never exceeding 0.50), and disagreement is worse on high-dimensional fingerprints and for SHAP-vs-IG — and **a quarter of MLP attribution settings fail the model-randomization sanity check** under true weight reinitialization. A variance decomposition shows reliability is governed by *different* factors depending on the property measured — the endpoint dominates faithfulness, the representation dominates stability, and the attribution method dominates sanity behaviour. We conclude that faithfulness, the property the community most often reports, is necessary but far from sufficient: an attribution can be faithful to a model and still be method-dependent and model-insensitive. Mechanistic-interpretation claims for regulated ADMET models should not rest on a single attribution method, and should be accompanied by a self-consistency and sanity audit on the model at hand.

## 1. Introduction
In small-molecule drug discovery, ML models for ADMET (absorption, distribution, metabolism, excretion) and toxicity endpoints inform go/no-go decisions on which compounds to make and which series to deprioritize for liabilities such as hERG-mediated cardiotoxicity or drug-induced liver injury (DILI). Because such models are opaque, post-hoc feature attribution is the dominant way teams turn a prediction into an actionable rationale, and it is increasingly framed as the route to regulatory acceptance: the OECD QSAR Assessment Framework asks for a *mechanistic interpretation* and a *defined applicability domain*, and reviews and vendors position SHAP/LIME to meet that bar.

If a method is to satisfy a regulatory requirement, its reliability for that purpose must be measured rather than assumed. Prior evaluation of molecular attribution has been either synthetic (Sanchez-Lengeling et al., 2020, ground-truth-recovery on toy graph tasks), narrow (activity-cliff substructure localization), or general-purpose and faithfulness-centric (M4, 2023). None has measured the *full* reliability battery on *real, decision-relevant* endpoints, across representations and model classes, under the scaffold splits that mirror deployment on novel chemistry. We provide that audit.

## 2. Methods
### 2.1 Data
The TDC ADMET benchmark group (Huang et al., 2021). We retained 12 endpoints under a preregistered selection rule (all four toxicity endpoints — DILI, hERG, AMES, LD50 — plus ADME-category coverage with ≥1 regression and ≥1 classification per category where available): Caco2, HIA, BBB, VDss, CYP2C9-substrate, CYP3A4-substrate, Half-Life, Clearance-Hepatocyte, and the four toxicity sets. Endpoints span 475–7,385 molecules. Each was loaded under the canonical scaffold split (primary) and a random split (the out-of-distribution contrast).

### 2.2 Representations and models
Two representations: ~200 RDKit 2D descriptors (the interpretable features chemists reason about) and 2048-bit Morgan/ECFP4 fingerprints (the standard structural representation). Three model classes: random forest, histogram gradient boosting, and a 2-layer MLP (which admits gradient attributions). Fully crossed with both splits: 12 × 2 × 2 × 3 = 144 models, fixed seeds. Models were assessed against TDC metrics (AUROC/AUPRC for classification; MAE/Spearman/R² for regression); 142/144 cleared a preregistered trivial-baseline floor and are carried into the reliability analysis.

### 2.3 Attribution methods
Each model received the methods valid for it: tree models → SHAP (exact TreeExplainer), LIME, random; MLP → Integrated Gradients (Captum), LIME, random. A content-seeded **random** baseline is included everywhere as the null.

### 2.4 Reliability metrics (each referenced to the null)
- **Faithfulness** — per-molecule comprehensiveness (AOPC over feature-removal fractions: mask each molecule's own top-attributed features, measure the drop in the predicted quantity), with 95% bootstrap CIs and a one-sided paired bootstrap test against the null.
- **Stability** — worst-case attribution change under bounded L∞ perturbation of standardized features (local-Lipschitz sense), with bootstrap CIs.
- **Agreement** — per-molecule Kendall τ, Spearman ρ and top-k Jaccard between method pairs, with across-molecule variance.
- **Sanity** — Adebayo model-randomization: similarity between attributions on the trained vs a randomized model (true weight reinitialization for MLPs; a label-permutation refit analogue for tree models, treated as a limitation).

### 2.5 Statistics
Bootstrap CIs (1,000/400 resamples); Benjamini–Hochberg FDR (q=0.05) across the family of null-comparison tests; Wilcoxon signed-rank for the scaffold-vs-random (OOD) contrast; Mann–Whitney for the toxicity contrast; one-way η² for variance decomposition. All hypotheses and thresholds were frozen before results were inspected (preregistration in repo).

## 3. Results
### 3.1 The model zoo is competent, and scaffold is harder than random
142/144 models cleared the floor. Best-per-endpoint performance reaches AUROC 0.98 (HIA), 0.92 (DILI, BBB) and 0.87 (hERG); regression Spearman up to 0.74 (Caco2). Mean performance is consistently lower under scaffold than random splits (AUROC 0.802 vs 0.823; Spearman 0.425 vs 0.490), confirming the intended distribution shift. Descriptor + tree models are the most accurate on most endpoints — notable because descriptors are also the interpretable representation.

### 3.2 H1 (falsified): attributions usually beat the null on faithfulness
Only **12%** of non-random method–cells fail to be significantly more faithful than the random null after FDR correction (preregistered support threshold was ≥20%; **falsified**). Integrated Gradients never fails (0/48), LIME fails 15%, SHAP 12%. This is a reassuring result: on these endpoints, attribution faithfulness is the rule, not the exception.

### 3.3 H2 (falsified): reliability does not collapse out-of-distribution
Contrary to our hypothesis, attribution faithfulness and stability do **not** significantly degrade under scaffold shift (faithfulness median scaffold−random Δ = −0.013, Wilcoxon p = 0.13; stability p = 0.85), even though model *accuracy* does. Whatever an attribution is faithful to, it remains roughly as faithful on novel chemotypes as on familiar ones (Fig. `ood_faith.png`).

### 3.4 H3 (supported, with an important budget correction): explainers agree only modestly
At the main-run LIME budget (300 samples) median pairwise rank agreement was near zero (Spearman 0.05). A preregistered robustness check (R2) revealed this was partly an artifact of LIME's sampling budget: re-running SHAP-vs-LIME at 1,000 samples on six endpoints (descriptors, scaffold, RF) **roughly doubled** agreement, from a mean ρ of 0.15 to **0.34** (DILI 0.50, hERG 0.43, AMES 0.37, BBB 0.36, LD50 0.19, Caco2 0.18). We therefore report the budget-corrected figure as primary: even at an adequate budget and on the interpretable descriptor representation, SHAP and LIME agree only modestly (mean ρ 0.34, **no endpoint exceeding 0.50**), and disagreement is worse on high-dimensional ECFP and for SHAP-vs-IG (neither of which involves LIME's budget). Toxicity endpoints are *not* more self-consistent than the rest (Mann–Whitney p = 0.24). The qualitative conclusion stands — switching attribution method changes a substantial fraction of the "important features" reported — but the effect is moderate, not near-total, and naive low-budget LIME materially overstates it.

### 3.5 H4 (supported): a quarter of MLP settings fail the sanity check
Under the *true* Adebayo weight-reinitialization test, **25%** of MLP attribution cells fail (attributions on a randomized model resemble those on the trained model). The tree-model rate is higher (39%) but relies on the label-permutation analogue and is reported with that caveat (Fig. `sanity_by_modelclass.png`).

### 3.6 H5 (nuanced): different properties are governed by different factors
A one-way variance decomposition (η²) shows no single factor dominates reliability overall. Faithfulness is dominated by the **endpoint** (η² = 0.54; method 0.17, representation 0.01); stability by the **representation** (η² = 0.36); and sanity behaviour overwhelmingly by the **attribution method** (η² = 0.83). The simple hypothesis "representation dominates method" is too coarse: which factor matters depends on which reliability property you care about.

### 3.7 Robustness checks
Three preregistered robustness analyses support the main results. **(i) Faithfulness-metric validation:** our primary per-molecule comprehensiveness metric correlates with the expensive remove-and-retrain ROAR protocol at Spearman 0.93 (n=18 cell–method pairs over six endpoints), and ROAR independently confirms that SHAP and LIME beat the random null on *all* six endpoints. Notably, the naive alternative — a global mask-and-repredict model-score AOPC — does *not* track ROAR (ρ=−0.50), which is precisely why we adopted per-molecule comprehensiveness rather than score-AOPC as the primary metric. **(ii) Multi-seed OOD null (H2):** repeating the scaffold-vs-random contrast across three resample/training seeds on the toxicity endpoints leaves the null intact (median Δ=−0.010, sd 0.037, Wilcoxon p=0.19) — the absence of OOD degradation is not a single-sample fluke. **(iii) Mask-reference sensitivity:** the faithfulness ordering (SHAP≈LIME ≫ random) is unchanged across mean, median, and feature-permutation mask references (SHAP 0.33–0.37, LIME 0.30–0.32, random 0.07–0.09).

## 4. Discussion
The headline is a dissociation: **faithfulness and self-consistency are not the same thing, and a method can have the first without the second.** Across 12 real ADMET/tox endpoints, attributions are mostly faithful (H1) and robust to distribution shift (H2) — reassuring, and contrary to our preregistered pessimism — yet they disagree with one another almost completely (H3) and frequently fail to depend on the model they purport to explain (H4). Because faithfulness is the property the community most often reports, and the one that looks healthiest here, reporting it alone paints a misleadingly rosy picture.

For the regulatory framing this matters concretely. A "mechanistic interpretation" derived from a single attribution method is fragile: a different, equally faithful method would have highlighted largely different features (H3), and for some model/method settings the explanation would survive even destroying the model (H4). The variance decomposition (H5) implies the audit cannot be done once and generalized — faithfulness must be checked per endpoint, stability per representation, sanity per method.

### 4.1 Limitations
- The tree-model sanity test uses a label-permutation refit analogue, not weight reinitialization; only the MLP sanity numbers are the canonical Adebayo test. We therefore base the H4 claim on the MLP result.
- LIME used a modest sampling budget (300 samples) in the main run, which we found *did* depress measured agreement: a 1,000-sample re-run roughly doubled SHAP-LIME agreement (§3.4). The agreement claims now use the budget-corrected numbers; the main-run stability and per-cell faithfulness for LIME may still be modestly pessimistic for the same reason and are interpreted accordingly.
- Faithfulness was measured primarily by mask-and-repredict comprehensiveness; this was validated against remove-and-retrain ROAR (§3.7, Spearman 0.93). It remains a local-fidelity measure and is not claimed to capture global feature importance, which ROAR targets.
- We audited descriptor and fingerprint models; learned-representation (GNN) attributions are a labelled extension (atom-level, not commensurable with tabular feature vectors).
- The main matrix uses one split realization per endpoint; for the toxicity endpoints we confirmed the H2 null across three seeds (§3.7). Extending multi-seed repeats to all endpoints is left for the camera-ready.

### 4.2 Conclusion
Attribution faithfulness on ADMET/tox models is better than feared, but faithfulness is not sufficient for trust. We recommend that any attribution used to support a scientific or regulatory conclusion be accompanied by (i) a null comparison, (ii) a cross-method agreement check, and (iii) a model-randomization sanity check, on the specific model and data at hand. The harness used here makes that a one-command audit.

## 5. Data and code availability
TDC ADMET datasets (public). All code, the preregistration, the per-cell results database, and the figures are in the study repository; the metric implementations are in the `xai-eval-harness` package. One-command reproduction is provided (P7).

## References
(see `docs/00_literature_review.md` §5; to be converted to BibTeX in P7)
Hooker 2019 (ROAR); Adebayo 2018 (sanity checks); Lundberg & Lee 2017 (SHAP); Ribeiro 2016 (LIME); Sundararajan 2017 (IG); Sanchez-Lengeling 2020; Alvarez-Melis & Jaakkola 2018; Huang 2021 (TDC); Liu 2023 (M4); OECD QSAR Assessment Framework; EMA reflection paper on AI/ML 2024.
