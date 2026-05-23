# Anticipated reviewer critiques & responses (iteration register)

Severity: **H** (could sink the paper), **M** (must address), **L** (acknowledge).
Status: planned / running / DONE.

## Paper 1 — the audit
| # | Critique | Sev | Response / action | Status |
|---|---|---|---|---|
| 1 | "The Disagreement Problem (Krishna et al. 2022) already showed SHAP/LIME disagree — H3 not novel." | H | Cite it; differentiate: ours is the *regulatory ADMET decision context*, the *full trust battery* (5 axes, not just disagreement), *null-referenced*, *under scaffold/OOD*; disagreement is one axis and not the headline. Add Related Work. | DONE |
| 2 | "No GNNs — the dominant modern molecular model class is absent." | H | **DONE.** GIN extension (§3.8): AUROC 0.79-0.86 on 4 endpoints; occlusion attributions beat null 3/4; all pass true weight-reinit sanity. | DONE |
| 3 | "Tree sanity test is label-permutation, not Adebayo weight-reinit; H4 leans on MLP only." | M | **DONE/strengthened.** H4 based on MLP; GNN extension adds a *second* true-weight-reinit model class (all 4 pass). Tree caveat stated. | DONE |
| 4 | "Single split realization for most endpoints." | M | Multi-seed confirmed for the 4 toxicity endpoints (§3.7). Extending to all 12 is straightforward but compute-heavy; **scoped to camera-ready**, flagged honestly. | partial |
| 5 | "Main agreement matrix used LIME@300; you showed that depresses agreement." | M | **Addressed via the 6-endpoint LIME@1000 robustness**; H3 reports the budget-corrected numbers as primary (§3.4, §3.7). Full-matrix @1000 re-run scoped to camera-ready. | partial |
| 6 | "Only comprehensiveness for faithfulness; add sufficiency / a second metric." | M | **DONE.** Sufficiency preserves the ordering SHAP>LIME>random (§3.7 (iv)). | DONE |
| 7 | "H2 surprising — maybe scaffold isn't a strong shift here." | M | **DONE.** Shift is real (1.2x avg) but faithfulness change doesn't track it (Spearman -0.11) — H2 confirmed, not a weak-shift artifact (§3.7 (v)). | DONE |

## Paper 2 — the certificate
| # | Critique | Sev | Response / action | Status |
|---|---|---|---|---|
| A | "Your certificate is just model confidence (conf/margin are features)." | H | **CONFIRMED (honest).** Confidence-only within-cell AUROC 0.680 vs full 0.694 (+0.014); dropping attr_l2 gives 0.699. The certificate is largely reducible to model confidence; explanation-specific signals add ~nothing. Paper reframed around this — now a cautionary result, not a framework-sells claim. | DONE |
| B | "Attribution-magnitude predicting comprehensiveness is circular — both scale with attribution size." | H | **Partially confirmed.** within-cell Spearman(attr_l2, faith) 0.456 -> partial 0.215 controlling for confidence: some independent signal survives but it is confidence-entangled. Reported transparently. | DONE |
| C | "Prior work aggregates explanations to resolve disagreement (Krishna; 'consensus as training objective'). You ignore it." | M | Position: we *test* whether consensus predicts per-instance faithfulness and find it does NOT — a direct, contrarian empirical contribution. Add Related Work. | DONE |
| D | "This is selective prediction relabeled." | M | Position: selective prediction abstains on low-confidence *predictions*; we abstain on low-reliability *explanations*, and show (critique A) it is not reducible to confidence. | DONE |
| E | "AUROC 0.69 is modest." | L | Frame as triage, report abstention lift (the deployable quantity) with CIs; honest about modesty. | DONE |
| F | "Hard masking ignores per-feature importance (soft-erasure critique)." | L | Acknowledge; cite Normalized-AOPC / soft-erasure; note as future work. | DONE |

## Iteration rounds
- **R1 (this):** register; positioning/Related Work for both; Paper 2 ablation (A,B); Paper 2 figures.
- **R2:** Paper 1 GNN extension (#2), multi-seed all endpoints (#4), LIME@1000 main agreement (#5),
  sufficiency metric (#6), shift-magnitude (#7).
- **R3:** integrate, re-draft, final self-review.

## R4 (post-Paper-3) — new critiques + external wet-lab validation
| # | Critique | Sev | Response / action | Status |
|---|---|---|---|---|
| R4-1 | "Paper 3 D2's Spearman=1.0 is on n=3 methods; broaden the method set." | H | **DONE - changed the finding.** With 6 methods (IG / saliency / grad*input / SmoothGrad / occlusion / random) all 4 gradient methods cluster at near-identical mask-faithfulness (0.16-0.18) but only occlusion recovers chemistry. Spearman(faith, recovery) = -0.09 (p=0.87). The harness has a real boundary: it certifies above-null mask-faithfulness but cannot distinguish chemistry-faithful methods from any-reasons-the-model-uses faithful methods. Paper 3 D2 and abstract reframed honestly. | DONE |
| R4-2 | "No external chemistry validation - everything is self-consistency." | H | **DONE.** PAINS/BRENK alert overlap on tox-trained GIN. AMES Δ=+0.075 [+0.005, +0.148] (significant chemistry consistency); hERG/DILI CIs include 0 (not significant). Reported as honest *partial* validation; explained by PAINS/BRENK being reactive/mutagenic-skewed. Paper 1 §3.8. | DONE |
| R4-3 | "Does Paper 2's 'mostly confidence' finding generalize beyond molecules?" | M | Cheap test not yet executed; not pursued in R4. **Camera-ready backlog.** | scoped |
| R4-4 | "Why only GIN? GNN attribution likely architecture-dependent." | M | One architecture sweep (GCN/GAT) on the Paper 3 benchmark would strengthen generality. **Scoped to camera-ready** — Paper 3 result already substantive on GIN with the broader method set. | scoped |
| R4-5 | "No GNNExplainer / PGExplainer / SubgraphX comparison in Paper 3." | M | Add at least one graph-native attribution method (e.g., GNNExplainer). | scoped (heavy) |
| R4-6 | "Reproducibility: no Dockerfile / exact env." | L | The requirements.txt + REPRODUCE.md cover it; Dockerfile is camera-ready polish. | scoped |
