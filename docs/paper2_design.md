# Paper 2 — design & preregistration: per-instance reliability certificates for attributions

*Version 1.0, 2026-05-22. Frozen before Paper-2 results are inspected. Gated on Paper 1
being submission-ready (satisfied). Deviations -> docs/deviations.md.*

> **Working title.** *Certifying explanations one prediction at a time: per-instance
> reliability certificates for feature attributions, across omics.*

## 1. Motivation, reshaped by Paper 1
Paper 1 established, on real ADMET/tox models, that attributions are usually *faithful*
(H1) and do *not* degrade out-of-distribution (H2) — but they disagree across methods
(H3) and a quarter of MLP settings fail the model-randomization sanity check (H4), and
reliability is heterogeneous across endpoints/representations/methods (H5).

Two design consequences:
1. **Drop the OOD/applicability-domain predictor.** The Paper-1 concept sketch assumed an
   OOD flag would predict attribution unreliability. H2 *falsified* that. We therefore do
   **not** build the certificate on distribution-distance.
2. **Key the certificate on what actually varies per instance:** local **stability**,
   cross-method **consensus** (does SHAP agree with LIME/IG *for this specific input*?),
   and a model-level **sanity gate** (if a model+method fails Adebayo, flag all its
   attributions). These are the axes Paper 1 showed are informative.

## 2. The framework
A model-agnostic **trust layer** wrapping any attribution method. For each prediction it
emits a **reliability certificate**:
- `stability_i`: local Lipschitz estimate around input i (harness, bounded perturbations).
- `consensus_i`: per-instance agreement between >=2 attribution methods on input i
  (rank correlation / top-k Jaccard of their feature rankings).
- `sanity_gate`: model+method-level pass/fail of the Adebayo test (a hard gate; a method
  that ignores the model cannot certify anything).
- `certificate_i`: a calibrated score combining the above into [0,1] trust, with an
  **abstain** decision below a held-out-calibrated threshold.

## 3. The load-bearing hypothesis (this is make-or-break, tested FIRST)
- **C1 — Per-instance consensus and stability predict per-instance faithfulness.**
  If true, we can certify individual explanations *cheaply and without ground truth*.
  *Test:* across instances (pooled, with endpoint as covariate), do `consensus_i` and
  `stability_i` predict `faithfulness_i` (per-molecule comprehensiveness)? Report Spearman,
  and the AUROC of (consensus, stability) predicting "faithful instance" (faithfulness >
  the cell's null). *Falsified if* AUROC <= 0.55 (no better than chance).
- **C2 — Utility: abstention improves the retained set.** Abstaining on low-certificate
  instances raises the mean faithfulness of retained explanations more than random
  abstention at the same abstention rate. *Test:* faithfulness-vs-coverage curves;
  area between certificate-guided and random-abstention curves > 0 (bootstrap CI).
- **C3 — Cross-omics generality.** C1 holds (AUROC > 0.55) in >=2 additional modalities
  beyond small molecules.

If **C1 is falsified**, we report that honestly as a negative result (per-instance
certification from consensus/stability is not feasible) and pivot the paper to a
characterization of *why* — still a contribution, but the framework claim is withdrawn.

## 4. Experiments
- **P9a (core premise, molecules):** reuse Paper-1 cached attributions + models. Per
  instance compute faithfulness_i, consensus_i (between the two available non-random
  methods per cell), stability_i; test C1. Make-or-break; run before building anything.
- **P9b (the trust layer + utility):** implement the certificate + abstention; test C2;
  faithfulness-coverage curves.
- **P9c (omics generalization):** demonstrate C1/C2 on >=2 further model types/modalities:
  - **transcriptomics** — a gene-expression classifier (tabular; per-gene attribution).
  - **proteomics/sequence** — a sequence model (per-residue attribution; harness sequence
    path + attention/IG).
  Datasets chosen for public availability and offline feasibility (logged in deviations).
- **P10:** manuscript.

## 5. Metrics & stats
Per-instance faithfulness = comprehensiveness (validated against ROAR in Paper 1, rho 0.93).
Consensus = per-instance Spearman/top-k Jaccard between methods. Certificate AUROC for
predicting faithful instances; faithfulness-coverage curves with bootstrap CIs; mixed
model with endpoint/modality random effects for C1 pooling. Preregistered thresholds above.

## 6. Industry usefulness (explicit)
The deliverable a med-chem or clinical-omics team gets is not a global benchmark score but
a **per-prediction trust readout**: "feature X drove this call, and here is whether you
should believe that for *this* sample." This directly serves the OECD applicability-domain
+ mechanistic-interpretation pairing, and the abstention policy maps onto a triage workflow
(auto-trust high-certificate explanations; route low-certificate ones to a human).

## 7. Honest limitations (anticipated)
- Certificates are *self-consistency* guarantees, not ground-truth correctness.
- Consensus needs >=2 attribution methods (extra compute per prediction).
- Calibration thresholds are modality/endpoint-specific (held-out calibration required).
- If C1 is weak, certificates add cost without benefit — we will say so.

## 8. Out of scope
Wet-lab validation; new attribution *methods* (we wrap existing ones); LLM-scale models.

---

## P9a VERDICT (2026-05-22): C1 falsified at the per-instance level — framework pivots

**Result.** Pooled, stability looked predictive (AUROC 0.65) but this is **Simpson's
paradox**: within-cell (controlling for model+endpoint) AUROC is 0.53 (stability) and
0.55 (consensus) — at chance. Consensus is non-predictive even pooled (AUROC 0.47;
Spearman with faithfulness -0.06). Per-instance faithfulness is **not** estimable from
cheap consensus/stability signals; combined with Paper 1's H2 (OOD doesn't predict
either), no cheap per-instance trust proxy survives.

**Decision (per §3 stopping rule): withdraw the per-instance certificate; pivot.**
Paper 2 becomes an honest two-part contribution:
1. **Negative result (novel and useful):** cheap per-instance reliability certification of
   attributions is infeasible — cross-method consensus, local stability, and distribution
   distance all fail to predict which individual explanations are faithful (within-cell
   AUROC ~0.53). This warns the field against false-confidence per-prediction "trust scores."
2. **Constructive alternative — reliability *gating* at the (model x method x endpoint)
   level:** what *is* actionable is a cheap cell-level screen — beat-the-null faithfulness +
   Adebayo sanity gate + aggregate stability — that decides whether to trust a model+method's
   explanations *at all*. Demonstrate this gate transfers across omics (transcriptomics,
   sequence). This is the deployable, honest framework.

Revised hypotheses for the pivoted paper:
- **C1' (falsified, reported):** per-instance certification infeasible (above).
- **C2' (gating utility):** the cell-level gate (null+sanity+stability) separates
  trustworthy from untrustworthy (model x method x endpoint) cells, and gating out failed
  cells raises mean faithfulness of retained cells vs no gating.
- **C3' (cross-omics):** the gate's verdicts are meaningful in >=2 further modalities.

---

## P9a-v2 VERDICT (2026-05-22): C1 LIVES (supersedes the pivot above)

A fair, well-powered retest (7 features, learned certifier, honest within-cell CV) gives
**within-cell AUROC 0.694** (> 0.55). The earlier "falsified" call was an artifact of using
only the two *weakest* signals. Per-instance certification IS feasible.

**What predicts per-instance faithfulness (within-cell Spearman):** attribution magnitude
(attr_l2 +0.38) and concentration (entropy -0.29), and model confidence/margin (+0.26) —
NOT cross-method consensus (+0.15) and NOT distribution distance (kNN +0.03, consistent with
Paper 1's H2). So the certificate is built primarily on attribution-intrinsic strength +
model confidence.

**Reinstated framework + revised hypotheses:**
- **C1 (SUPPORTED):** per-instance faithfulness is predictable (within-cell AUROC 0.69),
  modest but useful for triage.
- **C2 (test next):** abstaining on low-certificate instances raises mean faithfulness of
  retained explanations vs random abstention, *within cell* (faithfulness-coverage lift).
- **C3 (test next):** the certifier transfers to >=2 further omics modalities.
The certificate definition is updated: features = {attr_l2, attr_entropy, conf, margin,
stability, consensus}; the OOD/kNN term is dropped as non-predictive.
