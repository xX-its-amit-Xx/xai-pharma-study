Dear bioRxiv editors,

We submit *What predicts a trustworthy explanation? Per-instance reliability certificates for
feature attributions, across omics* for posting as a preprint.

A practitioner reads an attribution one prediction at a time, yet the XAI literature
evaluates attributions in aggregate. This paper asks whether per-prediction reliability is
itself estimable, and what cheap signals it should be built from. The work is positioned
against (and partly contrarian to) the *Disagreement Problem* line (Krishna et al., TMLR 2022;
"consensus as a training objective"; "aggregating explanations to resolve disagreement") and
against classical selective prediction (Geifman & El-Yaniv, NeurIPS 2017).

The paper's most important result is sobering and honest. The intuitive explanation-specific
signals — cross-method consensus, local stability, distribution distance — *do not* predict
per-instance faithfulness (within-cell AUROC ≈ 0.53 after we caught and corrected a Simpson's
paradox in our own initial finding). Per-instance faithfulness *is* predictable (within-cell
AUROC 0.69), and certificate-guided abstention raises retained-set faithfulness by +0.11 at
50% coverage — but our ablation traces that predictability **almost entirely to model
confidence**: a confidence-only gate already reaches AUROC 0.68, and the
explanation-specific features add only +0.014. The honest practical message is: a cheap
model-confidence gate triages explanation trustworthiness about as well as any elaborate
explanation-aware certificate; the field should be skeptical of per-instance "explanation
trust scores" that are largely confidence in disguise.

We add a cross-omics demonstration (Golub leukemia transcriptomics, a sequence-transformer
task) showing the certifier transfers when the underlying model is itself competent, with an
honest boundary condition: a near-chance model yields an uninformative certificate.

This paper deliberately reports the *process* as it unfolded, including two intermediate
framings we falsified — because the negative steps are themselves informative. We believe
they make the final claim more reliable, not less.

All code and data are public; the numerical claim audit (66/66 verified) is in the
repository.

This manuscript is part of a three-paper program; papers 1 and 3 are being submitted in
parallel. We have no competing interests.

Sincerely,
Amit Shenoy
[affiliation, email]
