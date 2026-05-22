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
