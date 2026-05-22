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
Simpson's paradox). Second, per-instance faithfulness *is* nonetheless predictable, but from
the attribution's own **magnitude and concentration** plus the **model's confidence**: a
seven-feature certifier reaches within-cell AUROC 0.69 on molecules, and certificate-guided
abstention raises the mean faithfulness of retained explanations by +0.11 at 50% coverage
(95% CI [+0.06, +0.17]). Third, the certificate **transfers across modalities** — AUROC 0.86
on a leukemia microarray (transcriptomics) and 0.81 on a sequence-transformer task — but only
when the underlying model is itself competent (a near-chance model yields an uninformative
certificate). The practical upshot is a model-agnostic trust layer that turns "feature X drove
this prediction" into "...and here is whether to believe it for *this* sample," with an
honest boundary: it certifies self-consistency, not correctness, and presupposes a model that
has actually learned the task.

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

### 3.2 Per-instance faithfulness is predictable — from attribution strength and model confidence
A seven-feature learned certifier, evaluated within-cell, reaches **AUROC 0.694** on molecules.
The carrying signals (within-cell Spearman with faithfulness) are the attribution's **L2
magnitude (+0.38)** and **entropy (−0.29)**, and the model's **confidence/margin (+0.26)** —
*not* consensus (+0.15), stability (−0.18), or k-NN density (+0.03). In words: a strong,
concentrated attribution from a confident prediction tends to be faithful; how much two methods
agree, and how in-distribution the input is, carry little per-instance information.

### 3.3 The certificate is useful for triage (C2)
Ranking explanations by the certificate and abstaining on the lowest scores raises the mean
faithfulness of the retained set monotonically: at 50% coverage, +0.114 (95% CI [+0.061,
+0.172]); at 30% coverage, +0.205. Random abstention gives no lift by construction. The
certificate thus supports a concrete workflow — auto-trust high-certificate explanations, route
low-certificate ones to a human.

### 3.4 It transfers across omics — given a competent model (C3)
The certifier transfers beyond small molecules: within-modality AUROC **0.86** (95% CI
[0.76, 0.94]) on the leukemia microarray (transcriptomics, per-gene attribution) and **0.81**
(95% CI [0.76, 0.85]) on the sequence-transformer task (per-token attribution), with a +0.21
abstention lift in the sequence case. Boundary condition: an initial sequence model that barely
learned the task (test accuracy 0.57) produced an *uninformative* certificate (AUROC 0.565, CI
including 0.5); only when the model was competent (0.73) did the certificate become predictive.
The certificate presupposes a model that has learned the task.

## 4. Discussion
The useful, slightly counter-intuitive message is that **what makes an explanation trustworthy
for a given prediction is mostly intrinsic to the attribution (its strength and focus) and to
the model's confidence — not the things practitioners reach for first** (does SHAP agree with
LIME? is the molecule in-domain?). This both delivers a deployable trust layer and warns
against two tempting-but-empty heuristics.

Relationship to Paper 1: Paper 1 mapped reliability at the (model × method × endpoint) level;
Paper 2 makes it *per-instance* and actionable, and explains the apparent paradox that OOD
shift didn't hurt faithfulness (because distribution distance simply isn't what governs
per-instance faithfulness).

### 4.1 Limitations (honest)
- The certificate measures **self-consistency / faithfulness, not correctness** — a faithful
  explanation of a wrong-for-the-right-reasons model is still certified.
- Within-cell AUROC 0.69 on molecules is **useful but modest**; this is triage, not a guarantee.
- It **requires a competent model** (§3.4) and held-out **calibration** of the abstention
  threshold per deployment.
- Faithfulness is operationalized as comprehensiveness; though ROAR-validated (Paper 1), it is
  a masking-based measure.
- The transcriptomics test has small n (72; mitigated by cross-fitting + bootstrap CI), and the
  sequence task is synthetic-but-structured.

### 4.2 Conclusion
Per-prediction explanation trust is estimable, cheaply and across modalities, but from
attribution strength/concentration and model confidence rather than the intuitive signals.
Shipped as a trust layer with an abstention policy, it converts an attribution into a
calibrated, triage-ready recommendation — provided the model underneath has genuinely learned.

## 5. Data & code availability
TDC ADMET (public); Golub leukemia (OpenML); synthetic sequence task (reproducible generator).
All drivers (`src/p9a_*`, `p9b_utility.py`, `p9c_*`), per-instance feature databases, and
verdict tables are committed; metrics use the `xai-eval-harness`. Preregistration and the full
decision trail (including the two falsified intermediate framings) are in `docs/`.

## References
Shared with Paper 1 (`paper/references.bib`); adds the Golub et al. (1999) leukemia dataset and
Simpson (1951).
