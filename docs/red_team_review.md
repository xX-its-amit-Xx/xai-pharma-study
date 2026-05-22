# Internal red-team review (P7)

A deliberately adversarial self-critique of Paper 1, with the response/mitigation
for each objection. Goal: find the reasons a tough reviewer would reject, and either
fix them or state them honestly.

## R1. "Your faithfulness metric (mask-and-repredict) is off-distribution; masking
features creates inputs the model never saw, so the drop measures brittleness, not
importance."
- **Status: partially valid, mitigation in progress.** This is the standard critique
  of comprehensiveness. Mitigation: (a) we reference every value to a random null, so
  systematic off-distribution effects partly cancel; (b) a remove-and-retrain (ROAR)
  cross-check on a subset is planned to confirm the cheap proxy tracks the expensive
  ground truth. Until that lands, faithfulness claims are stated as "comprehensiveness-
  faithfulness," not unqualified faithfulness.

## R2. "Low cross-method agreement (H3) is just because LIME with 300 samples is noise."
- **Status: being tested directly.** `robustness_lime.py` re-runs SHAP-vs-LIME at 1000
  samples on 6 endpoints. If agreement stays low, H3 stands; if it rises substantially,
  we soften H3 to a budget-dependent claim. Result folded into the manuscript caveat.
  (Note: SHAP-vs-IG and the ECFP disagreement do not involve LIME budget at all, and
  are already low — so H3 does not rest solely on LIME.)

## R3. "Tree-model sanity test isn't the real Adebayo test."
- **Status: acknowledged; claim already restricted.** The H4 headline (25% failure)
  uses MLP true weight-reinitialization. The tree label-permutation number is reported
  separately and explicitly caveated. No claim rests on the tree sanity number.

## R4. "Single split seed; your OOD null result (H2) could be a fluke of one scaffold
partition."
- **Status: valid, planned.** Multi-seed split repeats (>=3) for the toxicity endpoints
  are a planned robustness check. H2 is a *null* result and we are appropriately cautious:
  we claim "no significant degradation was detected," not "there is none."

## R5. "12 endpoints, 2 representations — is this enough to generalize?"
- **Response:** the endpoints span all five ADMET categories and the four canonical
  toxicity sets; effect sizes (not just p-values) are reported; H5's variance
  decomposition explicitly quantifies endpoint-to-endpoint heterogeneity rather than
  hiding it. We do not claim universality; we claim a decision-relevant map.

## R6. "Comprehensiveness uses standardized-feature mean (0) as the mask reference;
results may depend on the reference."
- **Status: planned sensitivity.** Re-run a subset with median and feature-permutation
  references; report whether faithfulness ordering is stable.

## R7. "Are the models good enough that interpreting them is meaningful?"
- **Response:** yes — 142/144 cleared a preregistered trivial-baseline floor; best
  endpoints match TDC-leaderboard-competitive performance (e.g., DILI 0.92 AUROC). The
  two below-floor cells are excluded from reliability claims.

## Open items feeding the manuscript revision
- [x] Fold LIME-1000 robustness result into §4.1 and H3 (done; H3 softened to budget-sensitive).
- [x] ROAR subset cross-check (R1): primary comprehensiveness metric vs ROAR Spearman **0.93**;
      ROAR confirms SHAP/LIME beat null on all 6 endpoints. Naive score-AOPC rejected (-0.50).
      Folded into §3.7 + §4.1.
- [x] Multi-seed split repeats for toxicity endpoints (R4): H2 null holds (p=0.19). §3.7.
- [x] Mask-reference sensitivity (R6): ordering stable across mean/median/perm. §3.7.
- [ ] Convert references.bib into the manuscript body and finalize figure callouts (camera-ready).
- [ ] Extend multi-seed repeats to all 12 endpoints (camera-ready; toxicity done).
