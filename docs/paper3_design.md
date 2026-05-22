# Paper 3 — design & preregistration: fingerprint-distillation as a ground-truth attribution benchmark for molecular GNNs

*Version 1.0, 2026-05-22. Frozen before Paper-3 results are inspected. Deviations -> docs/deviations.md.*

> **Working title.** *Decoding the bits: fingerprint distillation as a real, cheminformatically meaningful
> ground-truth attribution benchmark for graph neural networks.*

## 1. Motivation
Papers 1 & 2 measured attribution reliability without access to ground truth — they used a random null
and self-consistency checks. The molecular-attribution literature lacks a non-synthetic, abundant
ground-truth benchmark: prior work either invents synthetic graph tasks (Sanchez-Lengeling et al. 2020)
or relies on narrow activity-cliff substructure recovery. Both are valuable but limited.

RDKit's Morgan fingerprint computation exposes a `bitInfo` dictionary giving, for each (molecule, bit),
the **exact atoms** that activated that bit. This is per-(molecule, bit) ground truth — deterministic,
cheap, and cheminformatically real. A GNN trained to predict ECFP bits from the molecular graph *must*
attend to those atoms in order to fire each bit. **Attribution methods on such a distilled GNN can
therefore be scored against the actual chemistry**, not just self-consistency.

## 2. Contributions (claimed)
1. A **non-synthetic ground-truth attribution benchmark** for molecular GNNs, derived from ECFP `bitInfo`.
   Thousands of per-instance ground-truth labels per molecule.
2. A **comparative ranking** of attribution methods (Integrated Gradients on node features, atom-
   occlusion, GNNExplainer) by how well they recover the chemistry-defined ground truth.
3. A **validation of Paper 1's null-referenced faithfulness instrument**: does the harness's
   faithfulness ordering on these GNNs match the ground-truth-recovery ordering? If yes, Paper 1's
   instrument is grounded; if no, the harness has blind spots that the chemistry exposes.
4. A practical byproduct: **per-molecule bit decoding**, useful to the chemistry literature on its own
   (the Morgan hash is opaque; attribution-on-distilled-GNN gives an empirical, learned decoding).

## 3. Hypotheses (confirmatory, preregistered)
- **D1 — Recovery is possible.** Faithfulness-aligned attribution methods (IG, occlusion) recover
  bitInfo ground-truth atoms substantially better than random. *Test:* mean per-(molecule, bit) AUROC
  of attribution scores predicting the binary "is this atom in the ground-truth set" label, pooled.
  *Falsified if* mean AUROC ≤ 0.6.
- **D2 — Faithfulness ranking matches ground-truth-recovery ranking.** The ordering by Paper 1's
  null-referenced faithfulness on this GNN matches the ordering by ground-truth recovery (Spearman
  rank correlation across methods × bits). *Falsified if* the orderings are uncorrelated (|ρ| < 0.3).
- **D3 — Some bits and molecules systematically defeat attribution.** Recovery is heterogeneous;
  certain bit types or molecular sizes have low recovery. *Reported as characterization, not falsified.*

## 4. Design
- **Data.** A pooled set of ~5,000 SMILES drawn from TDC's largest ADMET/tox sets (AMES, LD50, AqSol,
  CYP2C9_Veith) for chemical diversity. Scaffold split.
- **Featurization.** PyG `from_smiles` for graphs; Morgan/ECFP4 (radius=2, nBits=1024) with `bitInfo`.
- **Target.** The top K=128 most-frequently-firing bits across the training set. (Multi-task sigmoid.)
- **Model.** A 3-layer GIN with multi-output sigmoid head; trained to predict the 128-bit vector
  (binary cross-entropy). Test accuracy per bit should be high (this is a learnable function).
- **Attribution methods (per active bit).**
  - **IG** on node features, with `target=bit_idx` (Captum IntegratedGradients).
  - **Atom occlusion**: zero each atom's features in turn; record drop in the bit's output sigmoid.
  - **Random null**: random per-atom scores.
- **Ground truth.** Per (molecule, bit) where the bit fired: the set of atom indices in `bitInfo[bit]`
  (RDKit's exact, deterministic atom set).
- **Metrics.** For each (molecule, bit, method): top-k Jaccard and AUROC of attribution scores
  predicting ground-truth-atom membership. Aggregate by method; report per-bit and per-molecule
  variance. Compare to the harness's null-referenced per-bit faithfulness for D2.

## 5. Statistics
Bootstrap CIs across (molecule, bit) pairs; paired tests for method comparison; BH-FDR across the
family. Effect sizes (AUROC differences) over p-values.

## 6. Phases
- **P11 — Feasibility gate.** Confirm `bitInfo` works; train a small multi-task GIN on ~500 SMILES;
  run IG on a handful of active bits; check whether IG recovers ground-truth atoms on a smoke set.
  If recovery is at chance, the design needs revision.
- **P12 — Main experiment.** Full ~5,000 molecules; 128 bits; IG + occlusion + random; full metric
  matrix with bootstrap CIs.
- **P13 — Manuscript.** Methods/benchmark paper.

## 7. Honest limitations (anticipated)
- ECFP is a *known deterministic function*; the prediction task is by design uninteresting. The
  contribution is the attribution-recovery benchmark, not the prediction.
- ECFP has hash collisions; we use per-(molecule, bit) ground truth which is well-defined regardless.
- Recovery is a *necessary* condition for attribution validity, not sufficient (a method could recover
  ECFP atoms here but fail on downstream property-prediction GNNs). We discuss this scope.
- This is a benchmark/methods paper; we do not propose new attribution methods.

## 8. Out of scope
Downstream property prediction (Paper 1 territory); per-instance reliability scores (Paper 2);
wet-lab.
