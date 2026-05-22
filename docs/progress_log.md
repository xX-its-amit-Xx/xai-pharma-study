# Progress log

Reverse-chronological. Each scheduled chunk appends an entry.

## 2026-05-21 — P0/P1/P1b complete (planning foundation)
- **P0 done:** literature review + novelty statement ([`00_literature_review.md`](00_literature_review.md)).
  Gap confirmed distinct from Sanchez-Lengeling 2020 (synthetic GNN ground truth),
  activity-cliff benchmark (narrow), and M4 (general, faithfulness-only). Novel axis:
  full reliability battery on real ADMET/tox endpoints under scaffold/OOD shift,
  with regulatory framing.
- **P1 done:** preregistration with 5 falsifiable hypotheses, operational metric
  definitions, statistical analysis plan, threats-to-validity ([`01_study_design_preregistration.md`](01_study_design_preregistration.md)).
- **P1b done:** two-paper publication plan, feasibility gate spec, venue strategy,
  Paper 2 gate ([`02_publication_plan.md`](02_publication_plan.md)).
- **Paper 2 concept** sketched but GATED ([`03_paper2_framework_concept.md`](03_paper2_framework_concept.md)).
- **Next chunk (P2):** feasibility gate — install PyTDC + rdkit, load an ADMET
  dataset, featurize (descriptors + Morgan), train a model, run all 4 harness
  metrics end-to-end on one real endpoint. Log result; adjust design if needed.
