Dear bioRxiv editors,

We submit *Faithful but not self-consistent: a reliability audit of feature attributions for
ADMET and toxicity models* for posting as a preprint.

This is the first systematic audit of attribution reliability on the real, decision-relevant
ADMET/toxicity endpoints to which post-hoc methods (SHAP, LIME, Integrated Gradients) are
now being applied in a *regulatory* setting. The OECD QSAR Assessment Framework and the
European Medicines Agency's 2024 reflection paper on AI/ML both require a "mechanistic
interpretation" alongside a defined applicability domain; SHAP/LIME are widely positioned to
provide it. If a method is to satisfy a regulatory requirement, its reliability for that
purpose has to be measured rather than assumed. We measure it.

What is novel:
1. The full reliability battery (faithfulness, stability, cross-method agreement, model-
   randomization sanity) on real ADMET/toxicity endpoints, with every metric referenced to a
   hard random null and tested under deployment-realistic scaffold (out-of-distribution)
   splits.
2. Preregistered hypotheses, several of which we falsified by the data; we report those
   honestly rather than burying them.
3. A learned-representation GNN extension and an external chemistry-knowledge validation via
   the PAINS/BRENK alert libraries — the closest available proxy for historical wet-lab
   knowledge, with the honest finding that overlap is significant where the libraries capture
   the mechanism (mutagenicity) and at chance where they don't (target-/tissue-specific tox).

The headline is sobering and decision-relevant: attribution faithfulness on ADMET/tox models
is *better than feared* — but faithfulness is *not sufficient* for trust. The same
attributions that pass faithfulness checks disagree across methods (budget-corrected
SHAP–LIME mean rank correlation 0.34 on interpretable descriptors), and a quarter of MLP
attributions fail the canonical Adebayo weight-reinitialization sanity check. Regulatory
mechanistic-interpretation claims should not rest on a single attribution method.

All data are public (Therapeutics Data Commons); the code, preregistration, full numerical
claim audit (66/66 claims verified against source CSVs), every figure, and one-command
reproduction live at https://github.com/xX-its-amit-Xx/xai-pharma-study, with the metric
implementations in the companion repository https://github.com/xX-its-amit-Xx/xai-eval-harness.

This manuscript is part of a three-paper program; papers 2 and 3 are being submitted in
parallel and are mutually independent but share the open-source instrument. We have no
competing interests.

Sincerely,
Amit Shenoy
[affiliation, email]
