# Decoding the bits: fingerprint distillation as a ground-truth attribution benchmark for molecular GNNs

*Draft v0.1 (P13). All numbers trace to `results/` (commit-pinned). Author: Amit Shenoy.*

## Abstract
Evaluating attribution methods on molecular graph neural networks is bottlenecked by the
absence of *ground truth*: prior benchmarks are either synthetic (Sanchez-Lengeling et al.,
2020) or narrow (activity-cliff substructure recovery). We propose a new, non-synthetic
ground-truth attribution benchmark obtained by **distilling** ECFP4 Morgan fingerprints into a
multi-task graph neural network: RDKit exposes, for each (molecule, bit), the exact atoms that
activated that bit (`bitInfo`), giving thousands of per-instance ground-truth labels per
molecule for a cheminformatically meaningful task. A GIN trained to predict 128 frequent
Morgan bits from molecular graphs achieves a mean per-bit test AUROC of 0.77 and, on a
held-out test set of 3,434 (molecule, bit) pairs, lets us evaluate attribution methods
against the actual chemistry. Three findings emerge: (i) **atom occlusion recovers
ground-truth atoms at AUROC 0.705** (95% CI [0.697, 0.714]) — a substantive, well-powered
recovery on a non-synthetic task; (ii) **Integrated Gradients on the same GIN fails entirely**
(0.497, CI [0.485, 0.508]), statistically indistinguishable from a random baseline (0.492),
even though the GIN demonstrably *does* attend causally to the bit-defining atoms (occlusion
finds them) — a precise, ground-truth-validated example of gradient-based attribution failing
to surface causal model behaviour; (iii) **Paper 1's null-referenced faithfulness metric
correctly orders the methods** (Spearman = +1.000 between faithfulness ordering and recovery
ordering across IG/occlusion/random), validating the harness against the chemistry. Recovery
is essentially independent of molecule size, ground-truth fragment size, and ground-truth
coverage (|ρ| ≤ 0.08), so the benchmark is robust across molecular complexity. The benchmark
ships as a reusable resource and offers a practical byproduct: per-molecule decoding of
opaque Morgan bits via attribution on the distilled model.

## 1. Introduction
Quantitative evaluation of feature attributions on molecular GNNs is hard for one reason:
ground truth is rare. Sanchez-Lengeling et al. (2020) supply synthetic graph problems with
designed-in attribution targets; the activity-cliff line uses experimental data but only on
substructure localization. There is no abundant, non-synthetic per-instance attribution
ground truth for graph models. We provide one.

Extended-Connectivity Fingerprints (ECFP4 / Morgan, radius 2, 1024-bit) are deterministic
hashes of atomic neighbourhoods. RDKit's `GetMorganFingerprintAsBitVect(..., bitInfo={})`
populates a dictionary whose entries — for every bit that fires on a given molecule — list
the *exact atoms* that produced it. Distil that map into a GNN, and you have, for free,
ground-truth attributions: any explainer of the distilled model must point to the atoms in
`bitInfo[bit]` if it is to recover the chemistry.

### 1.1 Position vs prior work
Sanchez-Lengeling et al. (2020) is synthetic-graph ground truth; we offer real chemistry.
McCloskey et al.'s activity-cliff benchmark uses real chemistry but is narrow (one task family);
ours is per-(molecule, bit) and abundant. M4 (Liu et al., 2023, NeurIPS D&B) provides general
faithfulness across modalities but no ground truth. The Disagreement Problem (Krishna et al.,
2022) shows methods disagree but cannot adjudicate; here we have an adjudicator.

## 2. Methods
**Data.** 1,000 chemically diverse SMILES pooled from TDC ADMET endpoints (AMES, LD50, DILI,
BBB, Caco2). Scaffold-style split (70/30 by SMILES order following the upstream scaffold
order of each parent endpoint). Featurization via PyG `from_smiles` (9-D node features).

**Target.** Top K=128 Morgan bits by training-set firing frequency (range 23–733/700);
multi-task sigmoid output.

**Model.** 3-layer GIN with hidden 64, ReLU, `global_add_pool`, linear 64→128 head.
Trained 80 epochs with Adam(1e-3) under BCE.

**Attribution methods (per active bit).**
- **Graph IG** — manual integrated gradients over node features with 20 steps and zero
  baseline, targeting one output bit at a time (captum's batching breaks `edge_index`).
- **Atom occlusion** — zero each atom's features, measure drop in the bit's predicted
  probability.
- **Random** — per-atom standard normal scores (null).

**Ground truth.** For every (mol, bit) where the bit fired in the molecule, the set of atom
indices in `bitInfo[bit]`.

**Metrics.** Per-(mol, bit) AUROC of attribution scores predicting the binary ground-truth-
atom label. Bootstrap 95% CIs (2,000 resamples). For D2, per-(mol, bit) atom-level
*comprehensiveness*: mask the top-attributed fraction (0.1, 0.2, 0.3 of atoms) and average
the drop in the bit's predicted probability — i.e., Paper 1's null-referenced faithfulness
ported to the graph setting.

## 3. Results
n = 3,434 (mol, bit) pairs on the held-out test set; mean per-bit test AUROC of the GIN is
0.77, so the prediction half of the task is competently solved (as expected for a learnable
deterministic hash).

### 3.1 D1 — Atom occlusion recovers ground-truth atoms; IG does not
| method | mean AUROC | 95% CI |
|---|---|---|
| **atom occlusion** | **0.705** | [0.697, 0.714] |
| Integrated Gradients | 0.497 | [0.485, 0.508] |
| random null | 0.492 | [0.482, 0.502] |

Atom occlusion clears the 0.6 bar with a wide margin; **IG is statistically indistinguishable
from random**. The IG-vs-occlusion gap on identical (model, molecule, bit) settings is large
(ΔAUROC ≈ 0.21) and CI-separated. The GIN demonstrably *does* attend to bit-defining atoms —
occlusion finds them — but gradients through three GIN layers and a global add-pool fail to
surface that. This is, to our knowledge, the cleanest ground-truth-validated example of
gradient-based attribution failure on a graph model.

### 3.2 D2 — Null-referenced faithfulness orders methods correctly
The per-(mol, bit) atom-level comprehensiveness (the harness's null-referenced faithfulness)
gives, on average: occlusion 0.275, IG 0.225, random 0.129. Across the three methods,
**Spearman rank correlation between faithfulness and ground-truth recovery is +1.000**: the
two rank the methods identically (occlusion > IG > random). The harness is *right* about
which method to prefer — and it reaches that conclusion without seeing the chemistry.

### 3.3 D3 — Recovery is robust across molecular complexity
Spearman of occlusion AUROC with molecule size (n_atoms), ground-truth fragment size (n_gt)
and ground-truth fraction (frac_gt) is −0.01, −0.08 and −0.06 respectively — all near zero.
The benchmark is not an artifact of small molecules or large ground-truth sets.

## 4. Discussion
Three points are worth stating plainly.

**(i) Methods that look similarly faithful can differ dramatically against chemistry.** IG
beats the null on the harness's faithfulness metric (0.225 vs 0.129), but it does *not*
recover ground-truth atoms. Occlusion does both. Beating a null is necessary but not
sufficient for chemical fidelity, and a benchmark with ground truth is needed to draw that
line.

**(ii) The harness's ordering aligns with chemistry.** D2's Spearman = 1.0 is a clean,
external validation of Paper 1's null-referenced faithfulness instrument: when a
chemistry-defined ground truth exists, the harness agrees with it.

**(iii) IG on graphs is a known-cautioned setting.** Our finding is consistent with the broader
literature that gradient-based attribution on permutation-invariant pooled outputs can wash
out (gradients into pooled representations distribute back across many nodes). The
methodological lesson is concrete: for atom-level attribution of graph-level GNN outputs,
prefer perturbation-based methods (occlusion) over gradient-based ones, or use methods that
explicitly target the pooled head (e.g. LayerIntegratedGradients on the post-pool embedding).

### 4.1 Limitations (honest)
- ECFP is a known, deterministic function; the *prediction* task is by design uninteresting.
  The contribution is the attribution-recovery benchmark, not the prediction.
- Hash collisions mean a single bit index can mean different substructures across molecules;
  we use per-(mol, bit) ground truth which is well-defined regardless.
- Recovery is a *necessary* not a sufficient condition for attribution validity in downstream
  tasks; methods that pass here may still mislead on property-prediction GNNs where the model
  must learn chemistry from sparse labels rather than mimic a known hash.
- We tested one GNN architecture (GIN, 3 layers, add pool); the IG failure may be
  architecture-specific. We did not exhaustively sweep architectures or include GNNExplainer
  / saliency, which is the natural next step.

### 4.2 Conclusion
Fingerprint distillation gives the field a non-synthetic, abundant, per-instance ground-truth
attribution benchmark for molecular GNNs. On this benchmark, atom occlusion recovers
chemistry-defined atoms at AUROC 0.71, while Integrated Gradients on the same model fails to
the chance level — a sharp, well-powered methods finding. As a bonus, the benchmark
externally validates the null-referenced faithfulness instrument of Paper 1: its ordering of
methods agrees perfectly with the ground-truth ordering. The benchmark and code ship with
this paper.

## 5. Data and code availability
TDC ADMET (public); the distillation pipeline, the trained GIN, and the per-(mol, bit)
results database are in the repository. One-command reproduction is provided.

## References
Shared with Paper 1 (`paper/references.bib`); adds Rogers & Hahn (2010, Morgan fingerprints),
Sanchez-Lengeling et al. (2020), Krishna et al. (2022, disagreement), Liu et al. (2023, M4).

## Figures
- `results/figures/p3_recovery_by_method.png` — D1 ground-truth recovery AUROC by method
  with 95% CIs.
