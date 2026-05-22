"""Paper 2, P9c (modality 2 of 3): transcriptomics generalization (hypothesis C3).

Golub leukemia microarray (72 samples x 7,129 genes, ALL vs AML) from OpenML — a genuine
high-dimensional omics task with per-gene attribution. We cross-fit (explain every sample
on a model trained without it) to use all 72 samples despite the small n, then test whether
the per-instance reliability certificate (P9a-v2) transfers: within-dataset AUROC and the
C2 abstention lift. Honest caveat: n=72 makes estimates noisy; reported with bootstrap CIs.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from xai_eval.explainers import build_explainer  # noqa: E402
from xai_eval.models.sklearn_adapter import SklearnAdapter  # noqa: E402

from experiment import comprehensiveness  # noqa: E402
from p9a_v2_certifier import FEATURES, instance_features  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    d = fetch_openml(name="leukemia", as_frame=False)
    X = np.asarray(d.data, float)
    y = (np.asarray(d.target) == "AML").astype(int)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    X = (X - mu) / sd
    names = [f"gene_{i}" for i in range(X.shape[1])]
    print(f"leukemia: X={X.shape}, AML rate={y.mean():.2f}")

    rows = []
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=0)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, ytr, Xte = X[tr], y[tr], X[te]
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, n_jobs=-1, random_state=0)
        model = SklearnAdapter.fit(rf, Xtr, ytr, feature_names=names)
        a_prim = build_explainer("shap", background_size=40).explain(model, Xte).values
        a_sec = build_explainer("lime", num_samples=300, background_size=min(60, len(Xtr))).explain(model, Xte).values
        a_rand = build_explainer("random").explain(model, Xte).values
        feats = instance_features(model, Xtr, Xte, a_prim, a_sec, "classification", "rf")
        faith = comprehensiveness(model, Xte, a_prim, "classification")
        null = comprehensiveness(model, Xte, a_rand, "classification")
        for i in range(len(Xte)):
            rows.append({"cell": "leukemia-rf", "faith": faith[i], "faithful": int(faith[i] > null[i]),
                         **{k: feats[k][i] for k in FEATURES}})
        print(f"fold {fold}: explained {len(Xte)}, faithful_rate={np.mean(faith>null):.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "p9c_transcriptomics.csv"), index=False)

    # within-dataset certifier AUROC (cell-centered features, 5-fold CV)
    Xc = df[FEATURES].copy()
    for c in FEATURES:
        Xc[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)
    Xc = Xc.fillna(0.0).values; yb = df.faithful.values
    oof = np.zeros(len(df))
    for trf, tef in StratifiedKFold(5, shuffle=True, random_state=0).split(Xc, yb):
        oof[tef] = LogisticRegression(max_iter=500).fit(Xc[trf], yb[trf]).predict_proba(Xc[tef])[:, 1]
    auc = roc_auc_score(yb, oof) if len(np.unique(yb)) == 2 else np.nan
    # bootstrap CI
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(df), len(df))
        if len(np.unique(yb[idx])) == 2:
            boot.append(roc_auc_score(yb[idx], oof[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # C2 lift @ 50% coverage within this dataset
    df["cert"] = oof
    gg = df.sort_values("cert", ascending=False)
    k = len(gg) // 2
    lift50 = gg.faith.iloc[:k].mean() - df.faith.mean()

    print(f"\n=== C3 transcriptomics ===")
    print(f"instances={len(df)}, faithful_rate={yb.mean():.2f}")
    print(f"within-dataset certifier AUROC = {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"C2 lift @ 50% coverage = {lift50:+.3f}")
    print("transfers" if auc > 0.55 else "does NOT clear 0.55 (small-n; report honestly)")
    pd.DataFrame([{"modality": "transcriptomics", "n": len(df), "auc": auc, "auc_lo": lo,
                   "auc_hi": hi, "lift50": lift50}]).to_csv(
        os.path.join(RES, "p9c_transcriptomics_verdict.csv"), index=False)


if __name__ == "__main__":
    main()
