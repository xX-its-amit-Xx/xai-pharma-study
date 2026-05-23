# bioRxiv metadata — three submissions

Copy/paste fields directly into the bioRxiv submission form. Abstracts have been trimmed to
fit bioRxiv's character ceiling (~2,800 chars including spaces); the un-trimmed versions live
in the manuscripts.

---

## Paper 1

**Title.** *Faithful but not self-consistent: a reliability audit of feature attributions for
ADMET and toxicity models.*

**Running title.** *Auditing attribution reliability for ADMET.*

**Type of article.** New result.

**Subject category (primary).** Bioinformatics.
**Subject category (secondary).** Pharmacology and Toxicology.

**Keywords.** explainable AI; feature attribution; SHAP; LIME; Integrated Gradients; ADMET;
toxicity; QSAR; faithfulness; sanity checks; scaffold split; graph neural networks; PAINS.

**Authors.** Amit Shenoy [PLACEHOLDER: confirm and add affiliation/ORCID].
**Corresponding author email.** shenoy.am@husky.neu.edu [PLACEHOLDER: confirm].

**License.** CC-BY 4.0.

**Conflict-of-interest statement.** The authors declare no competing interests.

**Funding statement.** No external funding [PLACEHOLDER: update if applicable].

**Data and code availability.** All data are public (Therapeutics Data Commons). All code,
the preregistration, the per-cell results database, every figure, and the full numerical
claim audit are at https://github.com/xX-its-amit-Xx/xai-pharma-study. Metric implementations
are in the companion package https://github.com/xX-its-amit-Xx/xai-eval-harness.

**Abstract (bioRxiv-trimmed, ~2,400 chars).**
Feature-attribution methods (SHAP, LIME, Integrated Gradients) are increasingly used to
provide the "mechanistic interpretation" that regulators (OECD QSAR framework; EMA 2024
reflection on AI/ML) expect from QSAR/ADMET models. Yet their reliability in the
molecular-property setting has never been systematically audited. We trained 144 models
spanning 12 Therapeutics Data Commons ADMET/toxicity endpoints, two molecular representations
(2D descriptors and 2048-bit ECFP4 fingerprints), and three model classes (random forest,
gradient boosting, MLP), under both scaffold and random splits. For the 142 models that
cleared a preregistered trivial-baseline floor, we measured four reliability properties —
faithfulness, stability, cross-method agreement, and the Adebayo model-randomization sanity
check — each referenced to a content-matched random null, and tested five preregistered
hypotheses. Two pessimistic hypotheses were falsified: attributions usually do beat the random
null on faithfulness (only 12% of method-cells fail), and faithfulness and stability do not
significantly degrade under scaffold (OOD) shift, even though model accuracy does. But two
failure modes are real: explainers agree only modestly on which features matter (budget-
corrected SHAP-LIME mean rank correlation 0.34 on descriptors, not reaching above 0.50 on any
endpoint), and 25% of MLP attribution settings fail the model-randomization sanity check under
true weight reinitialization. A variance decomposition shows reliability is governed by
different factors depending on the property measured: the endpoint dominates faithfulness, the
representation dominates stability, and the attribution method dominates sanity behaviour.
We add a GNN extension (GIN with atom-occlusion attribution) that passes the canonical sanity
test on all four toxicity endpoints, and a partial external chemistry validation via
PAINS/BRENK alert overlap that is significant on AMES (Δ AUROC +0.075, 95% CI [+0.005,
+0.148]) and at chance on hERG/DILI — consistent with the reactivity skew of the alert
libraries. Faithfulness is necessary but not sufficient: mechanistic-interpretation claims for
regulated ADMET models should not rest on a single attribution method, and should be
accompanied by a self-consistency and sanity audit on the model at hand. The harness used here
makes that a one-command audit.

---

## Paper 2

**Title.** *What predicts a trustworthy explanation? Per-instance reliability certificates for
feature attributions, across omics.*

**Running title.** *Per-instance attribution certificates.*

**Type of article.** New result.

**Subject category (primary).** Bioinformatics.
**Subject category (secondary).** Systems Biology.

**Keywords.** explainable AI; selective prediction; attribution reliability; per-instance
certification; model confidence; transcriptomics; cross-modality.

**Authors / affiliation / email / ORCID.** [as Paper 1].

**License.** CC-BY 4.0.

**Conflict / funding / data availability.** [as Paper 1].

**Abstract (bioRxiv-trimmed).**
Paper 1 of this program showed feature attributions for ADMET/toxicity models are usually
faithful but disagree across methods and often fail model-randomization sanity checks —
faithfulness is necessary but not sufficient for trust. A natural next step is a per-prediction
trust score telling a practitioner whether to believe a given explanation. We ask whether such
a per-instance certificate is feasible. Testing on real ADMET/tox models and generalizing to
transcriptomics (Golub leukemia microarray) and a sequence-transformer task, we report three
findings. First, the intuitive signals fail: neither cross-method consensus nor distribution
distance predicts which individual explanations are faithful (within-cell AUROC ≈ 0.53; the
apparent pooled signal is Simpson's paradox). Second, per-instance faithfulness is predictable
(within-cell AUROC 0.69 on molecules) and certificate-guided abstention raises retained-set
faithfulness by +0.11 at 50% coverage (95% CI [+0.06, +0.17]) — but we trace that
predictability almost entirely to model confidence: an explanation-agnostic confidence-only
gate already reaches AUROC 0.68, and the attribution-specific signals add only +0.014;
distribution distance adds nothing; attribution magnitude retains only a weak
confidence-independent signal (partial correlation 0.22). Third, this predictability transfers
across modalities — AUROC 0.86 on the leukemia microarray and 0.81 on the sequence task — but
only when the underlying model is itself competent (a near-chance model yields an
uninformative certificate). The practical upshot is deflating in a useful way: a cheap
model-confidence gate triages explanation trustworthiness about as well as any elaborate
explanation-aware certificate, and practitioners should be wary of per-instance "explanation
trust scores" that are largely confidence in disguise.

---

## Paper 3

**Title.** *Decoding the bits: fingerprint distillation as a ground-truth attribution
benchmark for molecular graph neural networks.*

**Running title.** *Fingerprint-distillation attribution benchmark.*

**Type of article.** New result.

**Subject category (primary).** Bioinformatics.
**Subject category (secondary).** Pharmacology and Toxicology.

**Keywords.** graph neural networks; molecular machine learning; feature attribution;
fingerprint; ECFP; ground truth; benchmark; Integrated Gradients; occlusion.

**Authors / affiliation / email / ORCID.** [as Paper 1].

**License.** CC-BY 4.0.

**Conflict / funding / data availability.** [as Paper 1].

**Abstract (bioRxiv-trimmed).**
Evaluating attribution methods on molecular graph neural networks is bottlenecked by the
absence of ground truth: prior benchmarks are either synthetic or narrow. We propose a new
non-synthetic ground-truth attribution benchmark obtained by distilling ECFP4 Morgan
fingerprints into a multi-task GNN: RDKit exposes, for each (molecule, bit), the exact atoms
that activated that bit, giving thousands of per-instance ground-truth labels per molecule
for a cheminformatically meaningful task. A GIN trained to predict 128 frequent Morgan bits
reaches a mean per-bit test AUROC of 0.77 and, on a held-out test set of 3,434 (molecule, bit)
pairs, lets us evaluate attribution methods against the actual chemistry. Three findings
emerge. (i) Atom occlusion recovers ground-truth atoms at AUROC 0.705 (95% CI [0.697,
0.714]); Integrated Gradients on the same GIN fails entirely (0.497, indistinguishable from
random) — a ground-truth-validated example of gradient-based attribution failing to surface
causal model behaviour. (ii) Broadening the method set, all four gradient methods we tested
(IG, vanilla saliency, gradient × input, SmoothGrad) recover chemistry at chance, while only
occlusion succeeds — the failure is the gradient family, not IG specifically. (iii) The
relationship between mask-faithfulness and chemistry recovery is method-set dependent: with
the canonical three methods (IG/occlusion/random) the orderings agree (Spearman = +1.0); with
six methods they decorrelate (Spearman = −0.09), because gradient methods and occlusion all
sit at near-identical mask-faithfulness (0.16–0.18) yet only occlusion is also chemistry-
faithful — exposing a precise boundary on null-referenced faithfulness. Recovery is robust
across molecular complexity. The benchmark is offered as a reusable resource.
