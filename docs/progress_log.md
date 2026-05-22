# Progress log

Reverse-chronological. Each scheduled chunk appends an entry.

## 2026-05-22 — P4 descriptors chunk COMPLETE (ECFP running)
- `src/experiment.py` ran the descriptors representation: 216 (cell x method) rows in
  `results/reliability.csv` + pairwise `results/agreement.csv`. SHAP via fast
  TreeExplainer; method matrix trees=SHAP/LIME/random, MLP=IG/LIME/random.
- **Preliminary descriptors signal (NOT final — ECFP + P5 stats pending, do not
  over-interpret):**
  - beats-null rate (non-random methods): 0.86 → ~14% of method-cells don't beat the
    random null on faithfulness.
  - Adebayo sanity FAIL rate: 0.49 (mostly tree models via the label-permutation
    analogue; must be split by model class in P5 — MLP uses true weight reinit).
  - mean faithfulness: IG 0.65 > SHAP 0.40 > LIME 0.23 > random 0.13.
  - faith_vs_null nearly equal scaffold (0.24) vs random (0.23); stability scaffold
    0.39 vs random 0.43 → no large OOD degradation in descriptors so far (H2 to be
    tested properly with ECFP + paired tests in P5).
- ECFP chunk launched (background, resumable). Next: on completion, commit, then P5
  (stats: H1-H5 tests, BH-FDR, figures).

## 2026-05-21 ~23:10 EDT — P3 model zoo: COMPLETE
- Built `src/featurize.py` (cached RDKit 2D descriptors + 2048-bit Morgan/ECFP4),
  `src/data.py` (12-endpoint selection per §4: all 4 tox + ADME coverage; scaffold
  AND random splits), `src/models_torch.py` (picklable/reloadable MLP adapter),
  `src/train.py` (shared deterministic builders for P3+P4).
- **Trained 144 models** = 12 endpoints x {descriptors, ECFP} x {scaffold, random} x
  {RF, HGB, MLP}. **142/144 cleared the trivial-baseline floor** (2 ECFP cells below:
  CYP2C9-Sub/scaffold/RF, HalfLife/scaffold/HGB → excluded from reliability claims).
- **Scaffold is harder than random** (mean AUROC 0.802 vs 0.823; mean Spearman 0.425
  vs 0.490): the deployment-shift signal needed for H2 is present and modest.
- Best-per-endpoint: DILI 0.920, HIA 0.983, BBB 0.918, hERG 0.871, AMES 0.861 AUROC;
  Caco2 0.738, VDss 0.555 Spearman. **Descriptor+tree models dominate** — the
  interpretable representation is also the most accurate on most endpoints.
- Tables: `results/performance.csv`, `results/performance_summary.md`. Log:
  `results/train_log.txt`.
- **Decisions logged:** D1 (PyTDC --no-deps), D2 (GNN → labelled extension). See
  [`deviations.md`](deviations.md).
- **Next chunk (P4):** attribution x metric x dataset matrix. For each floor-clearing
  cell, compute SHAP/LIME/IG(+random) attributions on a fixed test-molecule sample,
  then faithfulness (normalized to null), stability, agreement, and Adebayo sanity;
  write a tidy per-cell results database to `results/reliability.csv` with bootstrap
  CIs. Mind compute: subsample explained molecules (prereg allows), cache attributions.

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
