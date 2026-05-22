"""Paper 2, P9a-v2: a FAIR, stronger test of per-instance certification (C1).

The 2-signal test failed within-cell. Here we give per-instance certification its best
shot: a richer feature set + a learned certifier + an honest within-cell evaluation
(cell-centered features and per-cell held-out AUROC), so between-cell structure cannot
inflate the result (the Simpson trap that fooled the pooled 2-signal AUROC).

Per-instance features:
  consensus    - cross-method rank agreement (SHAP/IG vs LIME) at this instance
  stability    - local worst-case attribution change (primary method)
  conf         - model confidence (classification: top-1 prob; regression: 0)
  margin       - top1-top2 class prob (classification; 0 for regression)
  knn_dist     - mean distance to 5 nearest training points (applicability density)
  attr_l2      - L2 norm of the attribution (signal strength)
  attr_entropy - entropy of |attribution| (diffuse vs concentrated)
Label: faithful_i = comprehensiveness_i > null_i.

Verdict: within-cell mean AUROC of the learned certifier > 0.55 -> C1 lives.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from xai_eval.explainers import build_explainer  # noqa: E402
from xai_eval.metrics import stability  # noqa: E402

import data as data_mod  # noqa: E402
from experiment import comprehensiveness  # noqa: E402
from p9a_premise import per_instance_consensus  # noqa: E402
from train import build_dataset, train_one  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
N = 60
PRIMARY = {"rf": "shap", "mlp": "ig"}
FEATURES = ["consensus", "stability", "conf", "margin", "knn_dist", "attr_l2", "attr_entropy"]


def instance_features(model, Xtr, Xe, a_prim, a_sec, task, mn):
    cons = per_instance_consensus(a_prim, a_sec)
    st = stability(build_explainer(PRIMARY[mn], **({"n_steps": 32} if mn == "mlp" else {})),
                   model, Xe, epsilon=0.1, n_perturb=4, seed=0)
    stab = np.array(st.per_sample_worst)
    proba = model.predict_proba(Xe)
    if task == "classification":
        srt = np.sort(proba, axis=1)
        conf = srt[:, -1]; margin = srt[:, -1] - srt[:, -2]
    else:
        conf = np.zeros(len(Xe)); margin = np.zeros(len(Xe))
    nn = NearestNeighbors(n_neighbors=5).fit(Xtr)
    knn_dist = nn.kneighbors(Xe)[0].mean(1)
    imp = np.abs(a_prim)
    attr_l2 = np.sqrt((a_prim ** 2).sum(1))
    p = imp / (imp.sum(1, keepdims=True) + 1e-12)
    attr_entropy = -(p * np.log(p + 1e-12)).sum(1)
    return dict(consensus=cons, stability=stab, conf=conf, margin=margin,
                knn_dist=knn_dist, attr_l2=attr_l2, attr_entropy=attr_entropy)


def main():
    rows = []
    rng = np.random.default_rng(0)
    for ep in data_mod.load_selected(seed=0):
        Xtr, ytr, Xte, yte, names = build_dataset(ep, "descriptors", "scaffold")
        for mn in ["rf", "mlp"]:
            model = train_one(ep.task, mn, Xtr, ytr, names)
            Xe = Xte[rng.choice(len(Xte), min(N, len(Xte)), replace=False)]
            a_prim = build_explainer(PRIMARY[mn], **({"n_steps": 32} if mn == "mlp" else {})).explain(model, Xe).values
            a_sec = build_explainer("lime", num_samples=500).explain(model, Xe).values
            a_rand = build_explainer("random").explain(model, Xe).values
            feats = instance_features(model, Xtr, Xe, a_prim, a_sec, ep.task, mn)
            faith = comprehensiveness(model, Xe, a_prim, ep.task)
            null = comprehensiveness(model, Xe, a_rand, ep.task)
            cell = f"{ep.name}-{mn}"
            for i in range(len(Xe)):
                rows.append({"cell": cell, "faith": faith[i], "faithful": int(faith[i] > null[i]),
                             **{k: feats[k][i] for k in FEATURES}})
            print(f"{cell:15s} faithful_rate={np.mean(faith>null):.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "p9a_v2_features.csv"), index=False)

    # cell-center features (within-cell signal only)
    Xc = df[FEATURES].copy()
    for c in FEATURES:
        Xc[c] = df.groupby("cell")[c].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    Xc = Xc.fillna(0.0).values
    y = df.faithful.values

    # honest within-cell evaluation: 5-fold CV on cell-centered features, AUROC per cell
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.zeros(len(df))
    for tr, te in skf.split(Xc, y):
        lr = LogisticRegression(max_iter=500).fit(Xc[tr], y[tr])
        oof[te] = lr.predict_proba(Xc[te])[:, 1]
    df["cert"] = oof
    cell_aucs = []
    for cell, g in df.groupby("cell"):
        if g.faithful.nunique() == 2 and len(g) >= 15:
            cell_aucs.append(roc_auc_score(g.faithful, g.cert))
    within_auc = float(np.mean(cell_aucs))
    pooled_centered_auc = roc_auc_score(y, oof)

    # per-feature within-cell Spearman with continuous faithfulness
    feat_rho = {}
    for c in FEATURES:
        rr = [spearmanr(g[c], g.faith).statistic for _, g in df.groupby("cell") if len(g) >= 15]
        feat_rho[c] = float(np.nanmean(rr))

    print("\n=== C1 (stronger, within-cell) ===")
    print(f"testable cells: {len(cell_aucs)}")
    print(f"WITHIN-CELL mean AUROC (learned certifier, 7 features) = {within_auc:.3f}")
    print(f"pooled AUROC on cell-centered features = {pooled_centered_auc:.3f}")
    print("per-feature within-cell Spearman vs faithfulness:")
    for c, r in sorted(feat_rho.items(), key=lambda kv: -abs(kv[1])):
        print(f"   {c:13s} {r:+.3f}")
    verdict = "C1 LIVES (per-instance certification feasible)" if within_auc > 0.55 \
        else "C1 FALSIFIED even with rich features (within-cell ~ chance)"
    print(f"\nPreregistered verdict (within-cell AUROC>0.55): {verdict}")
    pd.DataFrame([{"within_cell_auc": within_auc, "pooled_centered_auc": pooled_centered_auc,
                   "n_cells": len(cell_aucs), **{f"rho_{k}": v for k, v in feat_rho.items()}}]
                 ).to_csv(os.path.join(RES, "p9a_v2_verdict.csv"), index=False)


if __name__ == "__main__":
    main()
