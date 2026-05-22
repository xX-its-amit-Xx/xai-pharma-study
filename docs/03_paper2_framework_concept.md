# Paper 2 — novel framework concept (GATED, forward-looking)

> **STATUS: NOT STARTED. Do not implement before Paper 1 reaches P6.**
> This is a thinking-ahead sketch only. Paper 1's empirical results are expected to
> reshape it. Recording it now so the program has direction, not so it constrains
> the science prematurely.

## The problem Paper 1 is likely to surface
Paper 1 will probably show (based on the literature and our pilot intuition) that
post-hoc attributions are (i) often only marginally above the random null on some
endpoints, (ii) representation-dependent, and (iii) less reliable out-of-
distribution — i.e., least trustworthy on novel chemotypes / novel samples, which
is exactly the regime where a chemist or clinician most wants a rationale. A flat
"feature X mattered" with no reliability qualifier is therefore dangerous.

## Candidate framework: **reliability-certified attributions**
A model-agnostic *trust layer* that wraps any attribution method and emits, per
prediction:
1. the attribution itself;
2. an **empirical stability certificate** (local Lipschitz bound estimated by the
   harness around that specific input);
3. a **faithfulness-calibrated confidence** (how much this attribution's implied
   feature ranking actually moves the model here);
4. an **applicability-domain / OOD flag** (distance to training manifold), which
   Paper 1 will have shown predicts attribution reliability;
5. an **abstain / down-weight** decision when (2)–(4) fall below calibrated
   thresholds — turning "here is the explanation" into "here is the explanation,
   and here is whether you should trust it for *this* case."

This directly answers the OECD applicability-domain + mechanistic-interpretation
pair, which is the industry/regulatory need, and is novel: certificates are
*per-instance and calibrated*, not a global benchmark score.

## Why it should generalize across omics
The trust layer operates on (model, input representation, attribution) — agnostic
to modality. Demonstrations planned across model *types* and omics:
- **small molecules** (Paper 1 carryover): per-substructure / per-descriptor.
- **transcriptomics / genomics** (e.g., gene-expression classifiers, tabular):
  per-gene attribution + per-sample reliability certificate; applicability domain =
  expression-manifold distance.
- **proteomics / sequence** (per-residue attribution on a sequence model): ties to
  the harness's sequence modality and attention-rollout path.
The unifying claim: *the reliability of an attribution is itself estimable and
should travel with the attribution, in any modality.*

## What we will be honest about (anticipated limitations)
- Certificates add compute (extra forward passes / local sampling).
- Stability/faithfulness certificates are *necessary not sufficient* for
  correctness — they bound self-consistency, not ground-truth mechanism.
- Calibration thresholds are domain-specific and must be set with held-out data.
- Across omics, applicability-domain estimation is itself hard and modality-specific.

## Open design questions (to resolve with Paper 1 in hand)
- Is per-instance faithfulness estimable cheaply enough for screening-scale use?
- Does the OOD flag actually predict attribution unreliability *quantitatively*
  (this is a *result* Paper 1 can supply, and a load-bearing assumption here)?
- What is the right abstention policy / utility model for a med-chem workflow?
