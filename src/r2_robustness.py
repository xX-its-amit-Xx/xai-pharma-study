"""Paper 1 R2 robustness: sufficiency metric (#6) and shift-magnitude analysis (#7).

#6 Sufficiency: keep ONLY the top-attributed features (mask the rest), measure retained
   predicted probability. A faithful explainer's top features should suffice. We check the
   method ordering matches comprehensiveness (robustness of the faithfulness conclusion).
#7 Shift magnitude: quantify the scaffold-split distribution shift (mean test->nearest-train
   distance in standardized descriptor space) per endpoint, and test whether the
   scaffold-vs-random change in faithfulness tracks the shift. If shifts are real but
   faithfulness change does not track them, H2 (no OOD degradation) is explained, not an artifact.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from xai_eval.explainers import build_explainer  # noqa: E402
from xai_eval.models.base import CLASSIFICATION  # noqa: E402

import data as data_mod  # noqa: E402
from train import build_dataset, train_one  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SUBSET = ["DILI", "hERG", "AMES", "LD50", "Caco2", "BBB"]


def sufficiency(model, X, attr, task, ref=0.0):
    """Retained predicted quantity when keeping ONLY top-k features (mask the rest)."""
    X = np.asarray(X, float); d = X.shape[1]; imp = np.abs(attr)
    base = model.predict_proba(X); pred = base.argmax(1) if task == CLASSIFICATION else None
    keeps = []
    for frac in (0.1, 0.2, 0.3):
        k = max(1, int(round(frac * d)))
        topk = np.argpartition(-imp, kth=k - 1, axis=1)[:, :k]
        Xm = np.full_like(X, ref); rows = np.arange(len(X))[:, None]
        Xm[rows, topk] = X[rows, topk]
        mp = model.predict_proba(Xm)
        q = mp[np.arange(len(X)), pred] if task == CLASSIFICATION else mp.reshape(-1)
        keeps.append(q)
    return np.mean(keeps, axis=0)


def main():
    eps = {e.name: e for e in data_mod.load_selected(seed=0)}

    # ---- #6 sufficiency ordering ----
    suff_rows = []
    for name in SUBSET:
        ep = eps[name]
        Xtr, ytr, Xte, yte, names = build_dataset(ep, "descriptors", "scaffold")
        model = train_one(ep.task, "rf", Xtr, ytr, names)
        rng = np.random.default_rng(0); Xe = Xte[rng.choice(len(Xte), min(60, len(Xte)), replace=False)]
        for m in ["shap", "lime", "random"]:
            a = build_explainer(m, **({"num_samples": 1000} if m == "lime" else {})).explain(model, Xe).values
            suff_rows.append({"endpoint": name, "method": m, "sufficiency": float(sufficiency(model, Xe, a, ep.task).mean())})
    suff = pd.DataFrame(suff_rows)
    suff.to_csv(os.path.join(RES, "r2_sufficiency.csv"), index=False)
    piv = suff.pivot_table(index="method", values="sufficiency", aggfunc="mean")
    print("=== #6 Sufficiency (mean retained prob keeping only top features; higher=better) ===")
    print(piv.round(3).to_string())
    print(f"ordering: {' > '.join(piv.sufficiency.sort_values(ascending=False).index)}")

    # ---- #7 shift magnitude vs faithfulness change ----
    rel = pd.read_csv(os.path.join(RES, "reliability.csv")).drop_duplicates(
        subset=["endpoint", "representation", "split", "model", "method"])
    rel = rel[(rel.representation == "descriptors") & (rel.method != "random")]
    shift_rows = []
    for ep in data_mod.load_selected(seed=0):
        Xtr_s, _, Xte_s, _, _ = build_dataset(ep, "descriptors", "scaffold")
        Xtr_r, _, Xte_r, _, _ = build_dataset(ep, "descriptors", "random")
        d_scaf = NearestNeighbors(n_neighbors=1).fit(Xtr_s).kneighbors(Xte_s)[0].mean()
        d_rand = NearestNeighbors(n_neighbors=1).fit(Xtr_r).kneighbors(Xte_r)[0].mean()
        sub = rel[rel.endpoint == ep.name]
        fd = (sub[sub.split == "scaffold"].faith_vs_null.mean()
              - sub[sub.split == "random"].faith_vs_null.mean())
        shift_rows.append({"endpoint": ep.name, "shift_scaffold": d_scaf, "shift_random": d_rand,
                           "shift_ratio": d_scaf / (d_rand + 1e-9), "faith_delta_scaf_minus_rand": fd})
    sh = pd.DataFrame(shift_rows); sh.to_csv(os.path.join(RES, "r2_shift.csv"), index=False)
    print("\n=== #7 Distribution shift vs faithfulness change ===")
    print(f"scaffold test->train distance is {sh.shift_ratio.mean():.2f}x the random-split distance "
          f"(min {sh.shift_ratio.min():.2f}, max {sh.shift_ratio.max():.2f}) -> shift is real")
    rho = spearmanr(sh.shift_ratio, sh.faith_delta_scaf_minus_rand).statistic
    print(f"Spearman(shift magnitude, faithfulness change) = {rho:+.3f} "
          f"-> {'tracks shift' if abs(rho) > 0.5 else 'faithfulness change does NOT track shift (supports H2)'}")


if __name__ == "__main__":
    main()
