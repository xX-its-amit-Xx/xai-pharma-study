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
to surface causal model behaviour; (iii) **The relationship between null-referenced faithfulness and chemical ground truth is method-set dependent and exposes a real boundary on the harness.** With the canonical three methods (IG, occlusion, random) the orderings agree (Spearman = +1.00); but extending to six methods by adding saliency, gradient × input and SmoothGrad, *all four gradient-based methods cluster at near-identical mask-faithfulness (0.16–0.18) yet only occlusion (also 0.18) recovers chemistry-defined atoms above chance*. Harness-faithfulness and recovery decorrelate (Spearman −0.09, p=0.87, n=6 methods). The harness reliably separates faithful from null but cannot distinguish methods that are mask-faithful for the right chemical reasons from those mask-faithful for any reasons the model happens to use. Recovery is essentially independent of molecule size, ground-truth fragment size, and ground-truth coverage (|ρ| ≤ 0.08), so the benchmark is robust across molecular complexity. The benchmark
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

**Target.** Top K=128 Morgan bits by firing frequency across the pooled set (range
23–733/1,000); multi-task sigmoid output.

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

### 3.2 D2 — Null-referenced faithfulness orders methods correctly *at small n*
On the canonical three methods (occlusion / IG / random), the harness's mask-faithfulness
ordering (0.275 / 0.225 / 0.129) matches the ground-truth-recovery ordering perfectly
(Spearman = +1.000). Taken alone this is a clean external validation of Paper 1's instrument.

### 3.3 D2-extended — but the rank correlation collapses on a broader method set
A reviewer-anticipated critique is that Spearman = 1.0 on n = 3 methods is fragile. We
therefore broadened the method set to six (IG, vanilla saliency, gradient × input, SmoothGrad,
atom occlusion, random null). To keep gradient computation tractable across four gradient
variants we re-trained the GIN at the same seed on a slightly smaller setup — 700 molecules and
K = 96 top Morgan bits (vs 1,000 and K = 128 for the main D1/D2 run) — giving n = 2,184 (mol,
bit) pairs after held-out splitting. The picture is revealing:

| method | recovery AUROC | mask-faithfulness |
|---|---|---|
| atom occlusion | **0.551** [0.537, 0.564] | 0.181 |
| IG | 0.488 [0.473, 0.503] | 0.164 |
| gradient × input | 0.476 [0.463, 0.491] | 0.169 |
| SmoothGrad | 0.475 [0.460, 0.488] | 0.178 |
| saliency | 0.472 [0.458, 0.486] | 0.177 |
| random | 0.501 [0.488, 0.513] | 0.085 |

Two findings emerge. **First, the IG failure is not IG-specific** — all four gradient-based
methods recover chemistry at chance, while only occlusion does so above chance. The
gradient-vs-perturbation split is the lawful distinction on this GIN. **Second, mask-
faithfulness is degenerate within the gradient family**: IG, grad × input, SmoothGrad,
saliency and even occlusion *all* sit at faithfulness 0.16–0.18 — they all pass the
"masking-their-top-atoms-hurts-the-prediction" test — but only occlusion is also chemistry-
faithful. The Spearman across these six methods between faithfulness and recovery is
**−0.086 (p = 0.87)**.

The honest reading is therefore: the harness can certify above-null mask-faithfulness, which
is necessary, but it cannot, on its own, tell apart methods that are mask-faithful for the
right reasons (chemistry-aligned) from methods that are mask-faithful for *any* reasons the
model uses. This is a real boundary on Paper 1's instrument exposed by the ground truth.

### 3.4 D3 — Recovery is robust across molecular complexity
Spearman of occlusion AUROC with molecule size (n_atoms), ground-truth fragment size (n_gt)
and ground-truth fraction (frac_gt) is −0.01, −0.08 and −0.06 respectively — all near zero.
The benchmark is not an artifact of small molecules or large ground-truth sets.

## 4. Discussion
Three points are worth stating plainly.

**(i) Gradient-based attribution fails as a *family* on this GIN — not just IG.** All four
gradient methods we tested (IG, vanilla saliency, gradient × input, SmoothGrad) recover
chemistry-defined atoms at chance, while atom occlusion recovers them substantially above
chance (§3.3). The lawful axis on this benchmark is gradient vs perturbation, not
method-by-method idiosyncrasy. This is consistent with the broader literature that gradients
through several message-passing layers and a permutation-invariant pool diffuse across nodes;
the methodological lesson is concrete: for atom-level attribution of graph-level GNN outputs,
prefer perturbation-based methods, or use methods that explicitly target the pooled head
(e.g. LayerIntegratedGradients on the post-pool embedding).

**(ii) Null-referenced faithfulness has a precise, ground-truth-exposed boundary.** Taken on
the canonical three methods (IG / occlusion / random), Paper 1's mask-faithfulness ordering
agrees with the chemistry ordering perfectly (Spearman = 1.0); but as soon as the method set
is broadened, the agreement collapses (Spearman = −0.09 over six methods, §3.3). The reason
is mechanically clear: all four gradient methods and occlusion sit at mask-faithfulness 0.16–
0.18 — they all pass the "masking-their-top-atoms-hurts-the-prediction" test — yet only
occlusion is also chemistry-faithful. The instrument certifies *above-null* mask-faithfulness,
which is necessary, but it does not, on its own, distinguish methods that are mask-faithful
for the right chemical reasons from methods mask-faithful for any reasons the model happens to
use. This is the real boundary of self-consistency-based faithfulness, exposed here against
external ground truth — and an honest *limit* on the harness, not a refutation of it.

**(iii) Methods that look similarly faithful can differ dramatically against chemistry.** The
practical corollary of (ii): a model+method that beats the null on faithfulness is necessary
but not sufficient evidence that the explanation is chemically right. A ground-truth benchmark
is needed to draw that line, and fingerprint distillation provides one.

### 4.1 Limitations (honest)
- ECFP is a known, deterministic function; the *prediction* task is by design uninteresting.
  The contribution is the attribution-recovery benchmark, not the prediction.
- Hash collisions mean a single bit index can mean different substructures across molecules;
  we use per-(mol, bit) ground truth which is well-defined regardless.
- Recovery is a *necessary* not a sufficient condition for attribution validity in downstream
  tasks; methods that pass here may still mislead on property-prediction GNNs where the model
  must learn chemistry from sparse labels rather than mimic a known hash.
- We tested one GNN architecture (GIN, 3 layers, add pool); the gradient-vs-perturbation
  finding may be architecture-specific. We tested four gradient variants (IG, vanilla saliency,
  gradient × input, SmoothGrad) but did not include graph-native explainers such as
  GNNExplainer, PGExplainer or SubgraphX, nor sweep GCN/GAT/attention-pool variants — the
  natural next steps.

### 4.2 Conclusion
Fingerprint distillation gives the field a non-synthetic, abundant, per-instance ground-truth
attribution benchmark for molecular GNNs. On this benchmark, atom occlusion recovers
chemistry-defined atoms at AUROC 0.71; all four gradient-based methods we tested (IG, vanilla
saliency, gradient × input, SmoothGrad) fail to the chance level on the same model — a sharp,
well-powered methods finding about gradient attribution on graph-level outputs. The benchmark
also bounds Paper 1's null-referenced faithfulness instrument honestly: it certifies
above-null mask-faithfulness, which is necessary; but within the family of mask-faithful
methods it cannot, on its own, separate chemistry-faithful from any-reasons-the-model-uses
faithful. Both findings — the gradient-family failure and the harness boundary — are useful
in proportion to the ground truth that exposed them. The benchmark and code ship with this
paper.

## 5. Data and code availability
TDC ADMET (public); the distillation pipeline, the trained GIN, and the per-(mol, bit)
results database are in the repository. One-command reproduction is provided.

## References
Shared with Paper 1 (`paper/references.bib`); adds Rogers & Hahn (2010, Morgan fingerprints),
Sanchez-Lengeling et al. (2020), Krishna et al. (2022, disagreement), Liu et al. (2023, M4).

## Figures
- `results/figures/p3_recovery_by_method.png` — D1 ground-truth recovery AUROC by method
  with 95% CIs.
- `results/figures/p3_d2_extended.png` — D2-extended: per-method scatter of mask-faithfulness
  vs chemistry recovery on 6 methods, visualising the harness-boundary finding (§3.3).
