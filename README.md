# xai-pharma-study

A reproducible research program on the **reliability of model interpretability in
biotech/pharma**, built on the [`xai-eval-harness`](../xai-eval-harness).

This is a **two-paper program** (see [`docs/02_publication_plan.md`](docs/02_publication_plan.md)):

1. **Paper 1 — the audit.** *Can you trust the explanation?* A systematic reliability
   audit of post-hoc feature attributions (SHAP, LIME, Integrated Gradients) for
   ADMET/toxicity prediction across the 22 TDC endpoints, measuring faithfulness,
   stability, cross-method agreement, and model-randomization sanity — referenced
   against a hard random null, and tested under scaffold (out-of-distribution)
   splits that mirror real deployment on novel chemistry.
2. **Paper 2 — the framework (gated).** A novel, industry-useful interpretability
   framework demonstrated across omics model types. **Not started until Paper 1 is
   manuscript-complete.**

## Why this matters
Regulators (OECD QSAR framework, EMA 2024 AI/ML reflection) expect a *mechanistic
interpretation* and *applicability domain* for QSAR/ADMET models, and SHAP/LIME are
being positioned to satisfy that. If a method is used to meet a regulatory
requirement, its reliability for that purpose must be measured — not assumed.

## Scientific commitments
Preregistered hypotheses, random-null reference for every metric, effect sizes +
CIs over p-values, honest negative/reassuring results reported as headlines, full
reproducibility. See [`docs/01_study_design_preregistration.md`](docs/01_study_design_preregistration.md).

## Status
See [`docs/progress_log.md`](docs/progress_log.md) and the live task list. The
program advances in scheduled chunks; each chunk updates the log and arms the next.

## Layout
```
docs/    literature review, preregistration, publication plan, paper-2 concept, progress log
src/     pipeline code (featurization, model zoo, experiment driver) — built in P2+
data/    cached TDC datasets (gitignored)
results/ per-cell metrics database, figures, tables — produced in P4/P5
```

## Reproducing
Will be a one-command rerun once P2 lands. Depends on `xai-eval-harness`
(installed editable), `PyTDC`, `rdkit`, and the harness's ML stack.
