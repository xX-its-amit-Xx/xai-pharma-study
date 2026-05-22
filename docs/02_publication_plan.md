# Publication plan — two-paper program

*Living document. Version 1.0, 2026-05-21.*

## Program structure
- **Paper 1 — the audit (evaluation paper).** Reliability of post-hoc attributions
  for ADMET/tox prediction. Must be *complete* before Paper 2 begins (user
  directive, and good science: Paper 2's framework should be informed by Paper 1's
  empirical findings).
- **Paper 2 — the framework (method paper), GATED.** A novel interpretability
  framework, industry-useful, demonstrated across omics model types, with honest
  advantages/limitations. Design begins only after Paper 1 is submitted-ready.

## Phases (Paper 1)
| Phase | Deliverable | Gate to advance |
| --- | --- | --- |
| P0 ✅ | Literature review, novelty statement | gap is defensible & distinct from prior art |
| P1 ✅ | Preregistration / study design | hypotheses falsifiable, analysis plan fixed |
| P1b ✅ | This publication plan | venues + timeline + Paper 2 gate defined |
| P2 | Env + data pipeline (TDC, RDKit) + harness integration | **feasibility gate**: can load ≥12 endpoints, featurize, train a model, run all 4 metrics end-to-end on ≥1 endpoint |
| P3 | Trained model zoo + performance table | models clear the trivial-baseline floor |
| P4 | Full attribution × metric × dataset results database | matrix complete or scaling-rule applied & logged |
| P5 | Statistical analysis + figures + tables | every preregistered test executed |
| P6 | Manuscript draft (Nature-tier structure) | all claims trace to a result; limitations written |
| P7 | Internal red-team review, robustness checks, reproducibility package | one-command rerun reproduces headline numbers |

## Feasibility gate (P2) — explicit
Before committing to the full design we verify in-environment:
1. `PyTDC` installs and downloads an ADMET dataset.
2. `rdkit` installs; descriptors + Morgan fingerprints compute on real SMILES.
3. A descriptor-based RF/MLP trains and reaches non-trivial performance.
4. The existing harness runs all four metrics on one real endpoint end-to-end.
5. (Stretch) `torch_geometric` installs and a small GIN trains; if not feasible in
   budget, GNN representation is **descoped to a clearly-labelled extension** and
   the paper stands on descriptors + fingerprints (both heavily used in industry).
The gate result is logged; the design (esp. §4 scaling, §3 GNN factor) is updated
in `docs/deviations.md` accordingly. Honest scoping > overreach.

## Target venues (Paper 1), in priority order
1. **Nature Communications** / **Nature Machine Intelligence** — broad, high bar;
   fits "trust in AI for a high-stakes scientific domain." Realistic only if
   findings are clean and the regulatory framing lands.
2. **Journal of Cheminformatics** (open access, field-canonical) or **JCIM** —
   strong, very appropriate home; high acceptance probability for rigorous audits.
3. **NeurIPS / ICML Datasets & Benchmarks track** — if we package the pipeline as a
   reusable benchmark (the harness + TDC integration supports this).

We will write to the *science*, target (2)/(3) as the credible home, and only
escalate to (1) if results warrant. We will not inflate claims to chase (1).

## Honesty commitments (apply to both papers)
- Report effect sizes and CIs, not just p-values.
- Preregister before peeking; log every deviation.
- A null/reassuring result is reported as the headline if that is what we find.
- Every limitation in §9 of the prereg appears in the manuscript discussion.
- Full reproducibility: data scripts, seeds, env lock, one-command rerun.
- No cherry-picked example explanations as evidence of a general claim.

## Timeline & self-scheduling
This program is executed in chunks across sessions. At the end of each working
chunk the agent schedules the next chunk (durable cron) and updates the todo list
and `docs/progress_log.md`. Recurring schedules auto-expire after 7 days; the
agent re-arms as needed and surfaces status to the user. The user can interrupt,
redirect, or pause at any chunk boundary.

## Paper 2 gate (do not start before this is true)
Paper 2 design begins only when Paper 1 reaches **P6 (manuscript draft complete)**.
At that point the empirical findings (which methods/representations are reliable,
where they fail, OOD behaviour) become the *motivation and design constraints* for
the novel framework. See `docs/03_paper2_framework_concept.md` for the gated,
forward-looking concept (subject to revision by Paper 1's results).
