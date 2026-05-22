"""Paper 2, P9b: abstention utility of the reliability certificate (hypothesis C2).

Given the per-instance certificate (P9a-v2), if a practitioner abstains on the
lowest-certificate explanations, does the mean faithfulness of the *retained* set rise
faster than under random abstention? Tested WITHIN cell (rank instances by certificate
inside each model x endpoint), so we measure genuine per-instance triage, not the trivial
strategy of preferring already-faithful cells.

Reads results/p9a_v2_features.csv (per-instance features + faith + cell).
"""

from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from p9a_v2_certifier import FEATURES  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(RES, "figures")
COVERAGES = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])


def oof_certificate(df):
    Xc = df[FEATURES].copy()
    for c in FEATURES:
        Xc[c] = df.groupby("cell")[c].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    Xc = Xc.fillna(0.0).values
    y = df.faithful.values
    oof = np.zeros(len(df))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xc, y):
        oof[te] = LogisticRegression(max_iter=500).fit(Xc[tr], y[tr]).predict_proba(Xc[te])[:, 1]
    return oof


def retained_mean_curve(g):
    """Per-cell: mean faithfulness of the top-coverage instances by certificate."""
    gg = g.sort_values("cert", ascending=False).reset_index(drop=True)
    n = len(gg)
    out = []
    for c in COVERAGES:
        k = max(1, int(round(c * n)))
        out.append(gg.faith.iloc[:k].mean())
    return np.array(out)


def main():
    df = pd.read_csv(os.path.join(RES, "p9a_v2_features.csv"))
    df["cert"] = oof_certificate(df)

    cert_curves, base = [], []
    for _, g in df.groupby("cell"):
        if len(g) < 20:
            continue
        cert_curves.append(retained_mean_curve(g))
        base.append(g.faith.mean())  # random abstention retains a representative subset
    cert_curves = np.array(cert_curves)
    base = np.array(base)
    mean_cert = cert_curves.mean(0)
    mean_base = base.mean()

    # lift at each coverage = certificate-retained mean - random (full-set) mean
    lift = mean_cert - mean_base
    # bootstrap CI on lift @ 50% coverage over cells
    rng = np.random.default_rng(0)
    idx50 = list(COVERAGES).index(0.5)
    per_cell_lift50 = cert_curves[:, idx50] - base
    boot = [per_cell_lift50[rng.integers(0, len(per_cell_lift50), len(per_cell_lift50))].mean()
            for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print("=== C2: abstention utility (within-cell) ===")
    print(f"cells: {len(cert_curves)}; full-set mean faithfulness = {mean_base:.3f}")
    print("coverage : cert-retained mean faithfulness (lift over random)")
    for c, m, l in zip(COVERAGES, mean_cert, lift):
        print(f"   {c:.1f}    : {m:.3f}  ({l:+.3f})")
    print(f"\nLift @ 50% coverage = {lift[idx50]:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    aulc = float(np.trapezoid(lift, -COVERAGES))
    verdict = "C2 SUPPORTED (certificate-guided abstention helps)" if lo > 0 \
        else "C2 not supported (CI includes 0)"
    print(f"Area under lift curve = {aulc:.3f}.  {verdict}")

    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(COVERAGES, mean_cert, "o-", label="certificate-guided abstention")
    ax.axhline(mean_base, ls="--", color="gray", label="random abstention (full-set mean)")
    ax.set_xlabel("coverage (fraction of explanations retained)")
    ax.set_ylabel("mean faithfulness of retained set")
    ax.set_title("C2: reliability-certificate triage (within-cell)")
    ax.invert_xaxis(); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "p2_abstention_utility.png"), dpi=120)
    pd.DataFrame({"coverage": COVERAGES, "cert_retained_faith": mean_cert, "lift": lift}).to_csv(
        os.path.join(RES, "p9b_utility.csv"), index=False)
    print("wrote results/p9b_utility.csv + figure")


if __name__ == "__main__":
    main()
