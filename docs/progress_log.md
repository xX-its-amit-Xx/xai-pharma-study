# Progress log

Reverse-chronological. Each scheduled chunk appends an entry.

## 2026-05-21 22:45 EDT — P2 feasibility gate: PASS
- **Environment:** Python 3.14 only (no C compiler). `rdkit==2026.03.2` installs
  cleanly. **PyTDC's `numpy<2.0` pin cannot build on py3.14** → installed
  `PyTDC --no-deps` + pure-python deps (fuzzywuzzy, python-Levenshtein, requests);
  runtime works fine against numpy 2.4. Logged as deviation (data access, not data).
- **End-to-end validated** on real DILI (drug-induced liver injury), scaffold split
  (332/47/96): 217 RDKit 2D descriptors + RandomForest → **test AUROC 0.921**
  (competitive with TDC leaderboard). All four harness metrics ran.
- **Pilot signal (n=80 test mols):** SHAP faith +0.196 & LIME +0.180 both beat
  random null (+0.071); SHAP **fails** Adebayo sanity (sim 0.668) while LIME passes
  (0.136); SHAP-LIME Spearman 0.48, top-10 Jaccard 0.40.
- **Design impact:** core design holds. GNN/`torch_geometric` remains a stretch
  factor (feasibility checked early in P3); paper stands on descriptors + ECFP if
  GNN is descoped. Code: [`src/feasibility.py`](../src/feasibility.py).
- **Next chunk (P3):** build production featurization module (descriptors + Morgan
  ECFP, cached), dataset-selection per §4 scaling rule (all 4 tox + ADME coverage,
  ≥12 endpoints), model-zoo trainer (RF/HGB/MLP × representations × scaffold+random
  splits), and run training; emit a performance table + trivial-baseline floor check.

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
