# Progress log

Reverse-chronological. Each scheduled chunk appends an entry.

## 2026-05-22 — R4 iteration: external chemistry validation + Paper 3 reframed
- **R4-2 PAINS/BRENK external validation (Paper 1 §3.8).** Tox-trained GIN occlusion vs
  chemistry-curated alerts: AMES Δ=+0.075 [+0.005, +0.148] **significant**; hERG/DILI CIs include 0.
  Honest partial validation - chemistry consistency where the library captures the mechanism
  (mutagenicity), at chance where it doesn't (target-/tissue-specific tox). The closest thing to
  external historical-wet-lab validation available without new bench experiments.
- **R4-1 Paper 3 D2 with broader method set (CHANGED the finding).** With 6 methods on n=2184
  (mol, bit) pairs: all 4 gradient methods (IG, saliency, grad*input, SmoothGrad) recover at chance
  (~0.47-0.49); only occlusion clears chance (0.551). All 4 gradient methods + occlusion sit at
  near-identical mask-faithfulness (0.16-0.18). Spearman(harness-faithfulness, recovery) = -0.09
  (p=0.87). The previous n=3 Spearman=1.0 was a small-set artifact. The harness boundary is now
  precise: certifies above-null but cannot distinguish chemistry-faithful from any-faithful within
  the gradient family. Paper 3 abstract + §3.2-3.3 reframed.
- Net effect of R4: Paper 1 gains a real external-chemistry validation row; Paper 3's strongest
  claim (Spearman=1.0 harness validates against chemistry) is honestly downgraded to "agrees at
  small n, decorrelates on a broader method set" - with a sharp new finding (gradient methods are
  jointly mask-faithful but not chemistry-faithful). Reviewer critique register updated.

## 2026-05-22 — Paper 3 P12 + P13: all three preregistered hypotheses SUPPORTED
- n = 3,434 (mol, bit) pairs over 1,000 SMILES x 128 top Morgan bits, multi-task GIN.
- **D1**: occlusion 0.705 [0.697, 0.714] >> 0.6 bar; IG 0.497 [0.485, 0.508] (chance);
  random 0.492. Occlusion vs IG gap 0.21, CI-separated.
- **D2**: Spearman 1.000 between Paper-1 null-referenced faithfulness ordering and
  ground-truth recovery ordering. The harness is right about which method to prefer.
- **D3**: recovery robust across n_atoms / n_gt / frac_gt (|rho| <= 0.08).
- Headline methods finding: **gradient IG fails (chance) while occlusion succeeds (0.71) on
  identical model+molecules+bits** - cleanest ground-truth-validated example of gradient
  attribution failing on graph models that we are aware of.
- Manuscript drafted: `paper/manuscript_paper3.md`. Figure:
  `results/figures/p3_recovery_by_method.png`.

## 2026-05-22 — Paper 3 P11 feasibility: PASS, with an immediately-interesting finding
- Multi-task GIN learns to predict 64 top Morgan bits well (mean per-bit test AUROC 0.774).
- Ground-truth-atom recovery on n=298 (molecule, bit) pairs:
  - **IG = 0.494** (chance) - despite the GIN learning the bits
  - **Atom occlusion = 0.713** (clears 0.6 bar)
  - random baseline = 0.490
- Interpretation: the GIN *is* causally attending to the bit-defining atoms (occlusion finds them);
  IG fails to surface that on a 3-layer GIN + global_add_pool (gradients smear/cancel). This
  validates the benchmark - it distinguishes attribution methods using ground truth - and is
  itself a publishable methods finding.
- Files: `src/p11_feasibility.py`, `results/p11_feasibility.csv`.
- **Next:** P12 scale up (1000 mols, K=128 bits, IG + occlusion + random) + D2 cross-check
  with null-referenced faithfulness.

## 2026-05-22 — Review-response iteration (R1-R3) COMPLETE
- **R1:** positioned both papers vs the Disagreement Problem (Krishna 2022) + selective
  prediction; ran the Paper-2 ablation that **confirmed critique A** — the certificate is
  largely model confidence (full beats confidence-only by only +0.014); reframed Paper 2
  honestly around this; added figures + bib.
- **R2 (Paper 1):** **GNN extension** (GIN, 4 endpoints, AUROC 0.79-0.86; occlusion
  attributions beat null 3/4; all pass true weight-reinit sanity) — addresses critiques #2/#3.
  **Sufficiency** metric preserves SHAP>LIME>random (#6). **Shift-magnitude**: scaffold shift
  real (1.2x) but faithfulness change doesn't track it (rho -0.11), confirming H2 (#7).
- **Reproducibility fix:** results/ + figures were gitignored; now tracked (75 files in repo).
- **Honestly scoped to camera-ready:** multi-seed across all 12 endpoints (#4; toxicity done),
  full-matrix LIME@1000 (#5; 6-endpoint subset done).
- **R3:** figure manifests + reviewer-response notes added to both manuscripts; critique
  register (docs/reviewer_critiques.md) is the audit trail.
- **Net:** both manuscripts strengthened and made more honest; the single most important
  change is that Paper 2's central claim is now correctly stated as "explanation trust is
  mostly model confidence" rather than oversold.

## 2026-05-22 — Paper 2 P9c: C3 cross-omics SUPPORTED across 3 modalities
- **Molecules:** within-cell AUROC 0.69 (P9a-v2). **Transcriptomics** (leukemia microarray
  72x7129): AUROC 0.864 (95% CI 0.76-0.94). **Sequence** (transformer on token sequences):
  AUROC 0.807 (95% CI 0.76-0.85), C2 lift +0.213.
- **Model-competence caveat (a finding):** a first sequence run with a near-chance
  transformer (acc 0.57) gave an inconclusive certifier (AUROC 0.565, CI incl. 0.5). With a
  competently trained model (acc 0.73) the certificate transferred cleanly. The certificate
  is only meaningful when the model has actually learned the task — reported as a boundary
  condition, not hidden.
- All Paper 2 empirics complete (C1 feasible, C2 useful, C3 generalizes). Files:
  results/p9c_{transcriptomics,sequence}*.csv. **Next: P10 manuscript.**

## 2026-05-22 — Paper 2 P9a-v2: C1 LIVES (supersedes the pivot)
- Per user steer, gave per-instance certification a fair shot: 7 features + learned
  certifier + honest within-cell evaluation (cell-centered, per-cell held-out AUROC).
- **WITHIN-CELL AUROC = 0.694** (> 0.55 bar) over 22 testable cells. The 2-signal failure
  was because consensus/stability are the WEAK predictors. Strong predictors (within-cell
  Spearman vs faithfulness): **attr_l2 +0.38, attr_entropy -0.29, conf/margin +0.26**;
  stability -0.18, consensus +0.15, kNN/OOD density +0.03 (~useless, consistent with H2).
- **Per-instance certification is feasible** — but from the attribution's own magnitude/
  concentration and the model's confidence, NOT from cross-method agreement or distribution
  distance. Framework reinstated in this revised, evidence-based form.
- Files: results/p9a_v2_features.csv, results/p9a_v2_verdict.csv.
- **Next:** P9b build the certificate wrapper + abstention utility test (C2: within-cell
  faithfulness-coverage, cert vs random); then P9c cross-omics.

## 2026-05-22 — Paper 2 P9a: C1 FALSIFIED at instance level -> framework PIVOT
- Make-or-break premise test (1,440 instances over 12 endpoints x {RF, MLP}, descriptors).
- **Simpson's paradox caught:** pooled stability->faithful AUROC 0.65 looked good, but
  WITHIN-CELL AUROC is 0.53 (stability) / 0.55 (consensus) = chance. Consensus non-predictive
  even pooled (0.47). Per-instance faithfulness is NOT estimable from cheap signals.
- **Per the preregistered stopping rule, the per-instance certificate claim is WITHDRAWN.**
  Paper 2 pivots to: (1) an honest negative result (cheap per-instance attribution trust
  scores are infeasible — warns against false-confidence "confidence scores"); (2) a
  constructive cell-level reliability *gating* protocol (null + sanity + aggregate stability)
  that decides whether to trust a (model x method x endpoint) combination's explanations at
  all, demonstrated across omics. See docs/paper2_design.md "P9a VERDICT" section.
- Files: results/p9a_premise.csv, results/p9a_verdict.csv.
- **Next:** P9b' implement + evaluate the cell-level gate (C2'); P9c' cross-omics (C3').

## 2026-05-22 — P7 submission-final robustness COMPLETE; Paper 1 SUBMISSION-READY
- **R1 (ROAR cross-check):** primary per-molecule comprehensiveness vs expensive ROAR
  Spearman **0.93** (n=18); ROAR independently confirms SHAP & LIME beat the null on all
  6 tested endpoints. The naive global score-AOPC proxy does NOT track ROAR (-0.50),
  validating our choice of comprehensiveness. Folded into manuscript §3.7 + §4.1.
- **R4 (multi-seed H2):** scaffold-vs-random null holds across 3 seeds (median Δ -0.010,
  p=0.19). The no-OOD-degradation result is not a fluke.
- **R6 (mask-reference):** faithfulness ordering SHAP≈LIME >> random stable across
  mean/median/permutation references.
- Manuscript updated (§3.7 robustness, H1/H3/limitations); red_team_review items closed.
- **Paper 1 is submission-ready** (draft). Remaining camera-ready niceties: inline
  reference formatting, all-endpoint multi-seed. Files: paper/manuscript.md,
  results/robustness_{roar,multiseed,maskref}.csv.
- **Paper 2 gate satisfied and Paper 1 now submission-final** — awaiting user go-ahead
  to open Paper 2 (novel reliability-certified-attribution framework across omics).

## 2026-05-22 — P6 manuscript draft + P7 (partial): COMPLETE/IN-PROGRESS
- **P6:** `paper/manuscript.md` v0.1 — full structure, results tied to committed
  numbers, falsified hypotheses reported honestly. `paper/references.bib`.
- **P7 done:** reproducibility package (`requirements.txt`, `REPRODUCE.md`),
  adversarial self-review (`docs/red_team_review.md`), and the LIME-budget robustness
  check (`src/robustness_lime.py`, `results/robustness_lime.csv`).
- **Robustness correction (good science in action):** SHAP-LIME agreement roughly
  doubles from mean rho 0.15 (300 LIME samples) to **0.34 (1,000 samples)** — never
  exceeding 0.50. H3 corrected from "near-zero" to "modest, budget-sensitive";
  abstract + §3.4 + §4.1 updated to the budget-corrected numbers as primary. The
  qualitative disagreement finding stands; the magnitude was overstated at low budget.
- **P7 backlog (refinements, not blockers; tracked in red_team_review.md):** ROAR
  remove-and-retrain cross-check on a subset; multi-seed split repeats for toxicity
  endpoints; mask-reference sensitivity.
- **Milestone:** Paper 1 is DRAFT-COMPLETE (P6 reached) — the Paper 2 gate condition
  is now satisfied, though Paper 1 is not yet submission-final (P7 backlog remains).

## 2026-05-22 — P4 COMPLETE + P5 hypothesis verdicts
- Full matrix: 142 floor-clearing cells x methods = 426 rows in `results/reliability.csv`,
  pairwise `results/agreement.csv`. Analysis in `results/analysis/hypotheses_summary.md`,
  figures in `results/figures/`.
- **Preregistered verdicts (headline science):**
  - **H1 FALSIFIED (reassuring):** only **12%** of non-random method-cells fail to beat
    the random null on faithfulness (threshold was >=20%). IG never fails (0.00),
    LIME 0.15, SHAP 0.12. Attributions are faithful more often than we pessimistically
    pre-registered.
  - **H2 FALSIFIED (surprising):** no significant attribution-reliability degradation
    under scaffold/OOD split — faithfulness Wilcoxon p=0.13, stability p=0.85. Model
    *accuracy* dropped modestly OOD (P3) but attribution faithfulness/stability did not.
  - **H3 SUPPORTED (strong):** explainers barely agree on feature rankings — median
    pairwise Spearman 0.05 overall (descriptors 0.12, ECFP 0.03; SHAP-vs-LIME on
    descriptors 0.09). Toxicity endpoints not more self-consistent (p=0.24).
  - **H4 SUPPORTED:** sanity-check failures are real — **25%** of MLP cells fail the
    *true* Adebayo weight-reinit test (tree label-perm analogue 39%, caveated).
  - **H5 NUANCED (metric-dependent):** eta^2 shows endpoint dominates faithfulness
    (0.54), representation dominates stability (0.36), method dominates sanity (0.83).
    The simple "representation > method" hypothesis is too coarse.
- **Caveat to verify in P7:** LIME used num_samples=300 (compute budget); low budget
  may inflate LIME instability and depress agreement. Robustness re-run at 1000 planned.
- **Next chunk (P6):** manuscript draft (title, abstract, intro, methods, results tied
  to these numbers, discussion incl. the falsified hypotheses, limitations).

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
