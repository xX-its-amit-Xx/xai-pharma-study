# What predicts a trustworthy explanation? Per-instance reliability certificates for feature attributions, across omics

*Draft v0.1 (P10). All numbers trace to `results/` (commit-pinned). Author: Amit Shenoy.*

## Abstract
Paper 1 of this program showed that feature attributions for ADMET/toxicity models are
usually *faithful* but disagree across methods and often fail model-randomization sanity
checks — i.e. faithfulness is necessary but not sufficient for trust. A natural next step is
a **per-prediction trust score**: a cheap signal telling a practitioner whether to believe a
given explanation. We ask whether such a per-instance certificate is feasible, and what it
must be built from. Testing on real ADMET/tox models and generalizing to transcriptomics and
sequence tasks, we report three findings. First, the *intuitive* signals fail: neither
cross-method consensus nor — confirming Paper 1 — distribution distance predicts which
individual explanations are faithful (within-cell AUROC ≈ 0.53; an apparent pooled signal was
Simpson's paradox). Second, per-instance faithfulness *is* nonetheless predictable (within-cell AUROC 0.69 on
molecules), and certificate-guided abstention raises the mean faithfulness of retained
explanations by +0.11 at 50% coverage (95% CI [+0.06, +0.17]). But — and this is the paper's
most important and sobering result — we trace that predictability **almost entirely to model
confidence**: an explanation-agnostic confidence-only gate already reaches AUROC 0.68, and the
attribution-specific signals (magnitude, concentration, consensus, stability) add only +0.01;
distribution distance adds nothing. Attribution magnitude retains only a weak
confidence-independent signal (partial correlation 0.22). Third, this predictability
**transfers across modalities** — AUROC 0.86 on a leukemia microarray (transcriptomics) and
0.81 on a sequence-transformer task — but only when the underlying model is itself competent
(a near-chance model yields an uninformative certificate). The practical upshot is deflating
in a useful way: a **cheap model-confidence gate triages explanation trustworthiness about as
well as any elaborate explanation-aware certificate**, and practitioners should be wary of
per-instance "explanation trust scores" that are largely confidence in disguise. What a
certificate can honestly offer is a confidence-based triage that certifies self-consistency
(not correctness) and presupposes a model that has learned the task.

## 1. Introduction
Post-hoc attributions are read one prediction at a time, yet they are evaluated (Paper 1; the
XAI literature) in aggregate. A practitioner deciding whether to act on *this* compound's
explanation needs a *per-instance* reliability signal. The intuitive candidates are: does this
explanation agree across methods (consensus)? is the input in-distribution (applicability
domain)? is the explanation locally stable? We test whether any of these — or a learned
combination — predicts per-instance faithfulness, and whether a resulting certificate is
useful (improves a triage workflow) and general (works beyond small molecules).

This paper is deliberately reported as the *process* unfolded, including two intermediate
hypotheses we falsified, because the negative steps are themselves informative.

## 2. Methods
**Faithfulness (ground truth for "trustworthy").** Per-molecule comprehensiveness — the drop
in the predicted quantity when an instance's own top-attributed features are masked (AOPC over
fractions). Validated against ROAR in Paper 1 (Spearman 0.93). An instance is "faithful" if
its comprehensiveness exceeds the matched random-attribution null.

**Candidate certificate features (per instance).** consensus (cross-method rank agreement,
SHAP/IG vs LIME), local stability (worst-case attribution change under bounded perturbation),
model confidence and class margin, k-NN distance in feature space (applicability density),
and the attribution's L2 magnitude and entropy.

**Honest evaluation.** Because faithfulness varies across (model × endpoint) cells, a certifier
can look good *pooled* merely by ranking faithful cells above unfaithful ones. We therefore
**cell-center** features and evaluate **within-cell** (per-cell held-out AUROC under
cross-validation). This is the test that matters and the one that defeated our first attempt.

**Data / models.** Molecules: the 12 ADMET/tox endpoints and models from Paper 1 (descriptors,
RF/MLP). Transcriptomics: Golub leukemia microarray (72 samples × 7,129 genes, ALL vs AML),
RF, 6-fold cross-fitting. Sequence: a small transformer on token sequences with marker-driven
labels; attention-rollout and leave-one-token-out occlusion attributions.

## 3. Results
### 3.1 The intuitive signals fail (two falsified hypotheses)
A two-signal certifier (consensus + stability) had a *pooled* AUROC of 0.65 — but this was
**Simpson's paradox**: within-cell AUROC was 0.53 (stability) and 0.55 (consensus), at chance.
Consensus was non-predictive even pooled (AUROC 0.47). Combined with Paper 1's H2 (out-of-
distribution shift did not degrade faithfulness), this rules out the three most intuitive
bases for a certificate — cross-method agreement, local stability *alone*, and distribution
distance. We initially concluded per-instance certification was infeasible; §3.2 shows that
conclusion was premature.

### 3.2 Per-instance faithfulness is predictable
A seven-feature learned certifier, evaluated within-cell, reaches **AUROC 0.694** on molecules.
At the univariate level the strongest within-cell correlates of faithfulness are the
attribution's L2 magnitude (+0.38) and entropy (−0.29) and the model's confidence/margin
(+0.26); consensus (+0.15), stability (−0.18) and k-NN density (+0.03) are weak.

### 3.3 ...but the certificate is largely model confidence (the key ablation)
The univariate correlations are misleading about *what carries the signal*. An ablation
(within-cell AUROC) is decisive:

| certifier | within-cell AUROC |
|---|---|
| confidence-only [conf, margin] | 0.680 |
| attribution-only [attr_l2, attr_entropy] | 0.536 |
| full minus attribution-magnitude | 0.699 |
| **full (7 features)** | **0.694** |

The full certifier beats a confidence-only gate by only **+0.014**, and removing attribution
magnitude does not hurt at all. Attribution-only is near chance (0.536). Controlling for
confidence, attribution magnitude retains only a weak partial correlation with faithfulness
(0.22, down from 0.46). **Per-instance explanation faithfulness is predicted predominantly by
the model's prediction confidence; the explanation-specific signals add almost nothing**, and
the intuitive ones (consensus, distribution distance) add nothing. This both answers and
concedes the obvious reviewer critique — the certificate *is* mostly confidence.

### 3.4 The certificate is useful for triage (C2) — but the lift is confidence-driven
Ranking explanations by the certificate and abstaining on the lowest scores raises the mean
faithfulness of the retained set monotonically: at 50% coverage, +0.114 (95% CI [+0.061,
+0.172]); at 30% coverage, +0.205. Random abstention gives no lift by construction. The
certificate thus supports a concrete triage workflow — but, given §3.3, this is in effect a
*confidence-based* triage; an explanation-aware certificate is not needed to obtain most of it.

### 3.5 It transfers across omics — given a competent model (C3)
The certifier transfers beyond small molecules: within-modality AUROC **0.86** (95% CI
[0.76, 0.94]) on the leukemia microarray (transcriptomics, per-gene attribution) and **0.81**
(95% CI [0.76, 0.85]) on the sequence-transformer task (per-token attribution), with a +0.21
abstention lift in the sequence case. Boundary condition: an initial sequence model that barely
learned the task (test accuracy 0.57) produced an *uninformative* certificate (AUROC 0.565, CI
including 0.5); only when the model was competent (0.73) did the certificate become predictive.
The certificate presupposes a model that has learned the task.

## 4. Related work and positioning
The **disagreement problem** (Krishna et al., 2022, TMLR) established that post-hoc methods
disagree and that practitioners resolve this with ad hoc heuristics; a subsequent line
(*explanation consensus as a training objective*; *aggregating explanations to resolve
disagreement*) treats agreement as something to maximize. Our contribution is orthogonal and
partly contrarian: we test whether per-instance **consensus predicts per-instance
faithfulness**, and find it does **not** (§3.1). Agreement between methods is therefore not, on
its own, evidence that an explanation is faithful — a caution for the aggregate-to-resolve line.

Our certificate is **selective prediction** (Geifman & El-Yaniv) applied to *explanations*
rather than predictions: abstain on low-reliability explanations to raise retained-set
faithfulness. The selective-prediction literature warns that confidence-based gates give a
false sense of security when confidence is insensitive to evidence quality; our §3.3 ablation
is the explanation-side analogue of that caution — here the gate is *largely confidence*, and
explanation-specific signals fail to improve on it.

## 5. Discussion
The honest, useful message is the opposite of the one we set out to find. We hypothesized that
per-instance explanation trust would come from explanation-intrinsic signals — cross-method
agreement, local stability, distribution distance. **All of these fail.** What does predict
per-instance faithfulness is the model's **prediction confidence**, and the explanation-aware
features add essentially nothing on top (§3.3). The practical recommendation is therefore
deflating but actionable: **if you want to triage which explanations to trust, a cheap model
confidence/margin gate is about as good as it gets** — do not invest in elaborate per-instance
"explanation reliability scores," which our results suggest are largely confidence relabeled.
The one nuance is that attribution magnitude carries a small confidence-independent signal
(partial 0.22), so a confidence+magnitude gate is marginally better than confidence alone.

Relationship to Paper 1: Paper 1 mapped reliability at the (model × method × endpoint) level;
Paper 2 asked whether it could be made per-instance. The answer is "only via confidence," which
also explains Paper 1's surprising H2 (distribution distance does not govern per-instance
faithfulness, so OOD shift did not degrade it).

### 5.1 Limitations (honest)
- **The headline is a near-null for the novel part:** the explanation-specific certificate adds
  only +0.014 AUROC over model confidence. We report this rather than dress it up; the
  contribution is the *negative/cautionary* result plus the confidence finding, not a new
  high-performing method.
- The certificate measures **self-consistency / faithfulness, not correctness** — a faithful
  explanation of a wrong-for-the-right-reasons model is still certified.
- Within-cell AUROC 0.69 on molecules is **modest**, and largely confidence; this is triage,
  not a guarantee.
- It **requires a competent model** (§3.5) and held-out **calibration** per deployment.
- Faithfulness is operationalized as comprehensiveness (hard masking); ROAR-validated (Paper 1)
  but subject to the known AOPC/hard-erasure pitfalls (soft-erasure variants are future work).
- The transcriptomics test has small n (72; mitigated by cross-fitting + bootstrap CI), and the
  sequence task is synthetic-but-structured.

### 5.2 Conclusion
We set out to build a per-instance "explanation trust score" and instead found that the
trustworthy ingredient is mostly **model confidence**: cross-method agreement, local stability,
and distribution distance do not predict which explanations are faithful, and attribution-aware
features add little beyond confidence. The honest, useful takeaway for industry is to triage
explanation trust with a cheap confidence/margin gate and to be skeptical of elaborate
explanation-reliability scores. This is a smaller claim than we hoped for — and a more reliable
one.

## 5. Data & code availability
TDC ADMET (public); Golub leukemia (OpenML); synthetic sequence task (reproducible generator).
All drivers (`src/p9a_*`, `p9b_utility.py`, `p9c_*`), per-instance feature databases, and
verdict tables are committed; metrics use the `xai-eval-harness`. Preregistration and the full
decision trail (including the two falsified intermediate framings) are in `docs/`.

## References
Shared with Paper 1 (`paper/references.bib`); adds the Golub et al. (1999) leukemia dataset and
Simpson (1951).
