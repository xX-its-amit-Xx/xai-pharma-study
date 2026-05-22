# Deviations log

Every departure from the frozen preregistration (`01_study_design_preregistration.md`),
with rationale. Good science requires these be visible.

## D1 — Data access via PyTDC `--no-deps` (P2)
- **Prereg said:** use TDC (programmatic) for the ADMET benchmark group.
- **Deviation:** PyTDC pins `numpy<2.0,>=1.26.4`; on the Python 3.14 environment
  there is no numpy-1.26 wheel and no C compiler to build it, so a normal install
  fails. Installed `PyTDC --no-deps` plus its pure-python runtime deps
  (`fuzzywuzzy`, `python-Levenshtein`, `requests`); the data-loading code runs
  correctly against numpy 2.4.
- **Impact:** none on the science — the *datasets and scaffold splits are the TDC
  canonical ones* (verified: DILI loaded 332/47/96). Only the package install path
  changed. Recorded for reproducibility; the env lock in P7 will pin this.

## D2 — GNN descoped from the core factorial to a labelled extension (P3)
- **Prereg said:** representation factor includes a learned GNN embedding (GIN/GCN),
  flagged as a *stretch* factor pending `torch_geometric` feasibility (P1b gate).
- **Finding:** `torch_geometric==2.7.0` *is* installed and importable — so the GNN
  is technically feasible.
- **Decision:** keep the GNN as a **clearly-labelled learned-representation
  extension (P4b)**, not part of the core fully-crossed analysis. Rationale:
  1. GNN attributions are atom/edge-level and not directly commensurable with the
     217-dimensional descriptor or 2048-bit fingerprint feature vectors, so the
     cross-method agreement and representation-effect (H5) comparisons are cleanest
     and most interpretable on the two tabular representations.
  2. The two tabular representations (interpretable 2D descriptors vs structural
     ECFP) already instantiate the representation factor of H5.
  3. Industry/regulatory QSAR-ADMET practice still relies heavily on descriptor/FP +
     tree/MLP models; the core is decision-relevant on its own.
  4. Bounds compute/scope so the core matrix is complete and rigorous.
- **Honesty note:** this is a *scope* decision made before inspecting any reliability
  result, on methodological-commensurability grounds — not because of any outcome.
  If P4 lands with budget to spare, the GNN extension (atom-level faithfulness/
  stability/sanity, within-representation) will be added and reported separately.

## D3 — GNN extension implemented (reverses D2's descope, in revision)
- D2 had descoped the GNN to a labelled extension. During the review-response iteration (R2),
  reviewer critique #2 ("no GNNs") was judged sink-the-paper severity, so the extension was
  implemented: a GIN on molecular graphs (PyG) for the 4 safety-critical classification
  endpoints, with node-occlusion attributions and the true weight-reinitialization sanity test.
  See `src/gnn_extension.py`, `results/gnn_extension.csv`, manuscript §3.8. The GNN is reported
  as a within-representation extension (atom-level attributions remain non-commensurable with
  the tabular feature vectors, per D2's rationale), not folded into the cross-method matrix.
