"""Paper 2 R1: kill the two highest-severity reviewer critiques.

A) "The certificate is just model confidence." -> ablation: confidence-only vs full
   certifier within-cell AUROC. Full must add value beyond confidence.
B) "Attribution magnitude predicting comprehensiveness is circular." -> partial
   correlation of attr_l2 with faithfulness controlling for confidence (and for the
   other features), within cell. If attr_l2 carries signal independent of confidence,
   it is not merely circular with confidence; we also report it honestly if it shrinks.

Reads results/p9a_v2_features.csv.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from p9a_v2_certifier import FEATURES  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def cellcenter(df, cols):
    X = df[cols].copy()
    for c in cols:
        X[c] = df.groupby("cell")[c].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    return X.fillna(0.0).values


def within_cell_auc(df, cols):
    X = cellcenter(df, cols); y = df.faithful.values
    oof = np.zeros(len(df))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = LogisticRegression(max_iter=500).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    aucs = [roc_auc_score(g.faithful, oof[g.index]) for _, g in df.groupby("cell")
            if g.faithful.nunique() == 2 and len(g) >= 15]
    return float(np.mean(aucs)), aucs


def main():
    df = pd.read_csv(os.path.join(RES, "p9a_v2_features.csv")).reset_index(drop=True)

    print("=== Critique A: is the certificate just confidence? (within-cell AUROC) ===")
    feature_sets = {
        "confidence-only [conf,margin]": ["conf", "margin"],
        "attribution-only [attr_l2,attr_entropy]": ["attr_l2", "attr_entropy"],
        "no-attr-magnitude (drop attr_l2)": [f for f in FEATURES if f != "attr_l2"],
        "full (7 features)": FEATURES,
    }
    res = {}
    for name, cols in feature_sets.items():
        auc, _ = within_cell_auc(df, cols)
        res[name] = auc
        print(f"  {name:42s} AUROC = {auc:.3f}")
    delta = res["full (7 features)"] - res["confidence-only [conf,margin]"]
    print(f"  -> full beats confidence-only by {delta:+.3f} "
          f"({'NOT just confidence' if delta > 0.02 else 'reducible to confidence!'})")

    print("\n=== Critique B: is attr_l2 -> faithfulness circular with confidence? ===")
    # within-cell partial Spearman of attr_l2 with faith, controlling for conf+margin
    def residualize(target, controls):
        X = cellcenter(df, controls); yv = df[target].values
        return yv - LinearRegression().fit(X, yv).predict(X)
    from scipy.stats import spearmanr
    # cell-center target/predictor too
    df["_faith_c"] = df.groupby("cell").faith.transform(lambda v: v - v.mean())
    df["_attr_c"] = df.groupby("cell").attr_l2.transform(lambda v: v - v.mean())
    raw = spearmanr(df._attr_c, df._faith_c).statistic
    faith_resid = residualize("faith", ["conf", "margin"])
    attr_resid = residualize("attr_l2", ["conf", "margin"])
    partial = spearmanr(attr_resid, faith_resid).statistic
    print(f"  within-cell Spearman(attr_l2, faith)            = {raw:+.3f}")
    print(f"  partial (controlling conf+margin)               = {partial:+.3f}")
    print(f"  -> attr_l2 signal {'survives' if abs(partial) > 0.1 else 'mostly vanishes'} "
          f"after removing confidence (so it is {'not merely' if abs(partial)>0.1 else 'largely'} a confidence proxy)")

    pd.DataFrame([{**res, "full_minus_conf": delta,
                   "attr_faith_raw": raw, "attr_faith_partial": partial}]).to_csv(
        os.path.join(RES, "p9d_ablation.csv"), index=False)
    print("\nwrote results/p9d_ablation.csv")


if __name__ == "__main__":
    main()
