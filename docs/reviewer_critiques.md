# Anticipated reviewer critiques & responses (iteration register)

Severity: **H** (could sink the paper), **M** (must address), **L** (acknowledge).
Status: planned / running / DONE.

## Paper 1 — the audit
| # | Critique | Sev | Response / action | Status |
|---|---|---|---|---|
| 1 | "The Disagreement Problem (Krishna et al. 2022) already showed SHAP/LIME disagree — H3 not novel." | H | Cite it; differentiate: ours is the *regulatory ADMET decision context*, the *full trust battery* (5 axes, not just disagreement), *null-referenced*, *under scaffold/OOD*; disagreement is one axis and not the headline. Add Related Work. | DONE |
| 2 | "No GNNs — the dominant modern molecular model class is absent." | H | Add a GNN learned-representation extension (PyG available) on a few endpoints; report within-representation. (Round 2) | planned |
| 3 | "Tree sanity test is label-permutation, not Adebayo weight-reinit; H4 leans on MLP only." | M | Already caveated; base H4 on MLP. Add discussion; note as scoped limitation. | DONE |
| 4 | "Single split realization for most endpoints." | M | Multi-seed confirmed for toxicity (§3.7); extend to all endpoints. (Round 2) | planned |
| 5 | "Main agreement matrix used LIME@300; you showed that depresses agreement." | M | Re-run the main agreement at LIME@1000 (or report H3 primarily from the 1000-sample robustness numbers). (Round 2) | planned |
| 6 | "Only comprehensiveness for faithfulness; add sufficiency / a second metric." | M | Add sufficiency; show ordering robust. (Round 2) | planned |
| 7 | "H2 surprising — maybe scaffold isn't a strong shift here. Quantify the shift." | M | Measure distribution-shift magnitude (train/test embedding distance) per endpoint; correlate with any reliability change. (Round 2) | planned |

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
