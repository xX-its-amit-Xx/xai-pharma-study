"""Paper 2, P9c (modality 3 of 3): sequence generalization (hypothesis C3).

A protein/text-sequence-shaped task (harness sequence_imdb: integer token sequences with a
few label-driving marker tokens) and a small transformer. Per-token attributions:
attention rollout (primary) and leave-one-token-out occlusion (secondary); per-token random
null. Faithfulness = drop in predicted-class probability when the top-attributed tokens are
masked. We test whether the reliability certificate transfers to the sequence modality.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
HARNESS = "d:/Users/ashenoy00000/.windsurf/xai-eval-harness/src"
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)

from xai_eval.builders import build_model  # noqa: E402
from xai_eval.datasets import load_dataset  # noqa: E402
from xai_eval.explainers import build_explainer  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SEQ_FEATURES = ["consensus", "conf", "margin", "knn_dist", "attr_l2", "attr_entropy"]
FRACS = (0.1, 0.2, 0.3)


def occlusion(model, X, pad_id):
    """Leave-one-token-out attribution: prob drop when each token is masked."""
    base = model.predict_proba(X); pred = base.argmax(1); base_q = base[np.arange(len(X)), pred]
    attr = np.zeros(X.shape, float)
    for t in range(X.shape[1]):
        Xm = X.copy(); Xm[:, t] = pad_id
        q = model.predict_proba(Xm)[np.arange(len(X)), pred]
        attr[:, t] = base_q - q
    return attr


def seq_comprehensiveness(model, X, attr, pad_id):
    base = model.predict_proba(X); pred = base.argmax(1); base_q = base[np.arange(len(X)), pred]
    d = X.shape[1]; imp = np.abs(attr); drops = np.zeros(len(X)); rows = np.arange(len(X))[:, None]
    for frac in FRACS:
        k = max(1, int(round(frac * d)))
        topk = np.argpartition(-imp, kth=k - 1, axis=1)[:, :k]
        Xm = X.copy(); Xm[rows, topk] = pad_id
        q = model.predict_proba(Xm)[np.arange(len(X)), pred]
        drops += base_q - q
    return drops / len(FRACS)


def per_instance_consensus(a, b):
    a, b = np.abs(a), np.abs(b); out = np.zeros(len(a))
    for i in range(len(a)):
        if np.allclose(a[i], a[i][0]) or np.allclose(b[i], b[i][0]):
            out[i] = 0.0
        else:
            s = spearmanr(a[i], b[i]).statistic; out[i] = 0.0 if np.isnan(s) else s
    return out


def main():
    ds = load_dataset("sequence_imdb", n_samples=700, seq_len=40, vocab_size=400)
    n_tr = 500
    Xtr, ytr = ds.X[:n_tr], ds.y[:n_tr]
    Xte, yte = ds.X[n_tr:], ds.y[n_tr:]
    pad_id = 0  # CLS/aggregation token doubles as mask
    model = build_model("small_transformer", _bundle(ds, Xtr, ytr),
                        {"d_model": 32, "n_heads": 2, "n_layers": 2, "epochs": 15})
    acc = (model.predict(Xte) == yte).mean()
    print(f"sequence transformer test acc={acc:.3f}, test n={len(Xte)}")

    rollout = build_explainer("attention_rollout").explain(model, Xte).values
    occ = occlusion(model, Xte, pad_id)
    rng = np.random.default_rng(0); rand = rng.standard_normal(Xte.shape)

    faith = seq_comprehensiveness(model, Xte, rollout, pad_id)
    null = seq_comprehensiveness(model, Xte, rand, pad_id)
    proba = model.predict_proba(Xte); srt = np.sort(proba, 1)
    nn = NearestNeighbors(n_neighbors=5).fit(Xtr.astype(float))
    imp = np.abs(rollout); p = imp / (imp.sum(1, keepdims=True) + 1e-12)
    feats = {"consensus": per_instance_consensus(rollout, occ),
             "conf": srt[:, -1], "margin": srt[:, -1] - srt[:, -2],
             "knn_dist": nn.kneighbors(Xte.astype(float))[0].mean(1),
             "attr_l2": np.sqrt((rollout ** 2).sum(1)),
             "attr_entropy": -(p * np.log(p + 1e-12)).sum(1)}
    df = pd.DataFrame({"faith": faith, "faithful": (faith > null).astype(int), **feats})
    df.to_csv(os.path.join(RES, "p9c_sequence.csv"), index=False)

    Xc = df[SEQ_FEATURES].copy()
    for c in SEQ_FEATURES:
        Xc[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)
    Xc = Xc.fillna(0.0).values; yb = df.faithful.values
    if len(np.unique(yb)) < 2:
        print("only one faithfulness class; cannot compute AUROC"); return
    oof = np.zeros(len(df))
    for trf, tef in StratifiedKFold(5, shuffle=True, random_state=0).split(Xc, yb):
        oof[tef] = LogisticRegression(max_iter=500).fit(Xc[trf], yb[trf]).predict_proba(Xc[tef])[:, 1]
    auc = roc_auc_score(yb, oof)
    rng = np.random.default_rng(0)
    boot = [roc_auc_score(yb[i], oof[i]) for i in (rng.integers(0, len(df), len(df)) for _ in range(2000))
            if len(np.unique(yb[i])) == 2]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    df["cert"] = oof; gg = df.sort_values("cert", ascending=False); k = len(gg) // 2
    lift50 = gg.faith.iloc[:k].mean() - df.faith.mean()
    print(f"\n=== C3 sequence ===\ninstances={len(df)}, faithful_rate={yb.mean():.2f}")
    print(f"within-modality certifier AUROC = {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"C2 lift @ 50% coverage = {lift50:+.3f}")
    print("transfers" if auc > 0.55 else "does NOT clear 0.55")
    pd.DataFrame([{"modality": "sequence", "n": len(df), "auc": auc, "auc_lo": lo, "auc_hi": hi,
                   "lift50": lift50}]).to_csv(os.path.join(RES, "p9c_sequence_verdict.csv"), index=False)


def _bundle(ds, X, y):
    from copy import copy
    b = copy(ds); b.X = X; b.y = y; return b


if __name__ == "__main__":
    main()
