# Literature review & novelty positioning

*Phase 0 deliverable. Last updated: 2026-05-21.*

## 1. Why interpretability reliability matters in pharma specifically

In small-molecule drug discovery, ML models for ADMET (absorption, distribution,
metabolism, excretion) and toxicity endpoints increasingly inform *go/no-go*
decisions: which compounds to synthesize, which series to deprioritize for
cardiotoxicity (hERG) or hepatotoxicity (DILI), and what to disclose to
regulators. Post-hoc feature attribution (SHAP, LIME, Integrated Gradients) is
the dominant way teams turn an opaque model into a "mechanistic" rationale a
medicinal chemist or regulator can act on.

Crucially, this is now a **regulatory** issue, not only a scientific one. The
OECD QSAR Assessment Framework requires, among five elements, a *defined
applicability domain* and a *mechanistic interpretation where possible*; the EMA's
2024 reflection on AI/ML references it. Reviews and vendors are explicitly
positioning SHAP/LIME as the way to satisfy the mechanistic-interpretation
requirement. **If a method is going to be used to satisfy a regulatory
requirement, its reliability for that purpose must be measured — not assumed.**
That measurement is what this study provides.

## 2. What has already been done

### 2.1 Attribution evaluation in molecular ML
- **Sanchez-Lengeling et al., NeurIPS 2020 — *Evaluating Attribution for Graph
  Neural Networks.*** Builds *synthetic* graph tasks with computable ground-truth
  attributions and measures how well attribution methods recover them for GNNs.
  Foundational, but (a) synthetic ground truth only, (b) GNNs only, (c) recovery
  of known structure, not deployment reliability.
- **Benchmarking Molecular Feature Attribution with Activity Cliffs, JCIM 2021
  (also chemRxiv 2021).** Uses maximum-common-substructure on experimentally
  determined activity cliffs as a real-data signal. Real but narrow: a single
  task family (activity cliffs / substructure localization).
- **A Perspective on Explanations of Molecular Prediction Models, JCTC 2022/23.**
  Conceptual perspective; argues attribution and counterfactual explanations need
  careful, quantitative evaluation. Calls for exactly this kind of study; does not
  run a systematic audit.

### 2.2 General faithfulness/attribution evaluation methodology
- **Hooker et al., NeurIPS 2019 — ROAR.** Remove-and-retrain protocol for
  faithfulness; the gold standard but expensive.
- **Adebayo et al., NeurIPS 2018 — Sanity Checks for Saliency Maps.** Model- and
  data-randomization tests; an explanation insensitive to model randomization is
  not explaining the model.
- **Alvarez-Melis & Jaakkola, 2018 — robustness/local-Lipschitz stability** of
  interpretability methods.
- **M4, NeurIPS 2023 Datasets & Benchmarks.** Unified faithfulness benchmark across
  metrics, modalities and models — general-purpose, *not* tied to the molecular
  decision context, and faithfulness-centric (not the full trust battery,
  no scaffold/OOD axis).
- **Normalized AOPC (arXiv 2024).** Shows naive faithfulness metrics are
  misleading without a random/baseline reference — validating our design choice to
  reference every metric against a hard random null.

### 2.3 OOD / scaffold robustness in molecular ML
- Scaffold splitting (Bemis–Murcko) is the field-standard way to simulate the
  realistic shift to novel chemotypes; TDC's ADMET group uses it for all 22
  datasets.
- **Evaluating ML Models for Molecular Property Prediction: Robustness on OOD Data,
  JCIM 2025** and related work study how *predictive accuracy* degrades under
  scaffold/cluster splits. They do **not** study how *attribution reliability*
  behaves under the same shift.

### 2.4 Data substrate
- **Therapeutics Data Commons (TDC), Huang et al., NeurIPS 2021.** The ADMET
  benchmark group: 22 curated datasets (6 absorption, 3 distribution, 6
  metabolism, 3 excretion, 4 toxicity), scaffold split, standardized metrics
  (MAE/Spearman for regression; AUROC/AUPRC for classification). Sizes 475–13,130.
  This is our real-data substrate; it is curated, citable, and decision-relevant.

## 3. The gap (precise)

No published work has, **on real decision-relevant ADMET/toxicity endpoints**:
1. measured the **full reliability battery** — faithfulness (vs a hard random
   null), local stability, cross-method agreement *and* model-randomization sanity
   checks — for the post-hoc methods (SHAP/LIME/IG) regulators are being told to
   trust;
2. across **multiple molecular representations** (interpretable 2D descriptors,
   ECFP fingerprints, and learned GNN embeddings) and model classes; and
3. **under scaffold (out-of-distribution) splits**, i.e. tested whether
   attribution reliability holds precisely where mechanistic interpretation is
   most needed — on novel chemotypes.

This is the union of three previously separate threads (molecular attribution
evaluation; the general trust battery; OOD robustness) applied to the regulatory
decision context that makes it matter.

## 4. Novelty statement (Paper 1)

> We present the first systematic *reliability audit* of post-hoc feature
> attributions for ADMET/toxicity prediction. Across 22 TDC endpoints, three
> molecular representations, multiple model classes, and four reliability metrics
> referenced against a hard random null, we quantify when and where attributions
> can be trusted — and show how reliability changes under the scaffold splits that
> mirror real deployment on novel chemistry. The result is a decision-relevant map
> of attribution trustworthiness for the exact setting in which it is now being
> used to satisfy regulatory mechanistic-interpretation requirements.

This is novel, falsifiable, uses only public data, has explicit controls, and is
designed to report honest (likely partly negative) findings.

## 5. Key references (to be expanded into BibTeX in P6)
- Hooker, Erhan, Kindermans, Kim. *A Benchmark for Interpretability Methods in DNNs* (ROAR). NeurIPS 2019.
- Adebayo et al. *Sanity Checks for Saliency Maps.* NeurIPS 2018.
- Lundberg & Lee. *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS 2017.
- Ribeiro, Singh, Guestrin. *"Why Should I Trust You?"* (LIME). KDD 2016.
- Sundararajan, Taly, Yan. *Axiomatic Attribution for Deep Networks* (IG). ICML 2017.
- Sanchez-Lengeling et al. *Evaluating Attribution for Graph Neural Networks.* NeurIPS 2020.
- Alvarez-Melis & Jaakkola. *On the Robustness of Interpretability Methods.* 2018.
- Huang et al. *Therapeutics Data Commons.* NeurIPS 2021 (Datasets & Benchmarks).
- Liu et al. *M4: A Unified XAI Benchmark for Faithfulness.* NeurIPS 2023 (D&B).
- *Normalized AOPC: Fixing Misleading Faithfulness Metrics.* arXiv 2024.
- Lavecchia. *Explainable AI in Drug Discovery.* WIREs Comput Mol Sci 2025.
- OECD QSAR Assessment Framework; EMA reflection paper on AI/ML in the medicinal product lifecycle (2024).

*Sources consulted are catalogued in `docs/00_sources.md` (to be added with verified URLs during P6 reference cleanup).*
