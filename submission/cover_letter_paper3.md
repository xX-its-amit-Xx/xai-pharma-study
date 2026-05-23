Dear bioRxiv editors,

We submit *Decoding the bits: fingerprint distillation as a ground-truth attribution
benchmark for molecular graph neural networks* for posting as a preprint.

Quantitative evaluation of feature attributions on molecular GNNs is bottlenecked by one
thing: ground truth is rare. Sanchez-Lengeling et al. (NeurIPS 2020) introduced synthetic
graph tasks with designed-in attribution targets; the activity-cliff line uses experimental
data but is narrow. We provide an abundant, non-synthetic alternative.

The construction is direct. Extended-Connectivity Fingerprints (ECFP4 / Morgan) are
deterministic hashes of atomic neighbourhoods, and RDKit's `bitInfo` exposes, for every
(molecule, bit), the exact atoms that activated that bit. Distil that map into a GNN, and
any explainer of the distilled model has, for free, per-(molecule, bit) ground-truth
attributions to be scored against.

Three findings emerge on this benchmark:

1. **Atom occlusion recovers chemistry-defined atoms** at AUROC 0.71 over 3,434 (mol, bit)
   pairs; **Integrated Gradients on the same GIN fails entirely** (0.50, indistinguishable
   from random) — a precise, ground-truth-validated example of gradient-based attribution
   failure on a graph model.
2. **The failure generalizes across the gradient family.** Adding vanilla saliency,
   gradient × input, and SmoothGrad: all four gradient methods recover at chance; only
   occlusion clears chance. The lawful split is gradient vs perturbation, not method
   idiosyncrasy.
3. **The relationship between null-referenced mask-faithfulness and chemistry is
   method-set dependent — exposing a real boundary on self-consistency-based faithfulness
   metrics.** On three methods, mask-faithfulness orders methods exactly as the chemistry
   ground truth does (Spearman = 1.0); on six methods the orderings decorrelate (Spearman =
   −0.09), because all gradient methods and occlusion sit at near-identical
   mask-faithfulness (0.16–0.18) yet only occlusion is also chemistry-faithful.

The benchmark is offered as a reusable resource — code, the trained GIN, the per-(mol, bit)
results database, and figures are public, with 100% of in-text numerical claims verified
against source CSVs.

This manuscript is part of a three-paper program. Paper 1 (audit) and Paper 2 (per-instance
certification) are being submitted in parallel; this paper validates Paper 1's instrument
against external ground truth and quantifies a precise limit of it. We have no competing
interests.

Sincerely,
Amit Shenoy
[affiliation, email]
