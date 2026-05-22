"""P7 submission-final robustness checks (R1, R4, R6 from the red-team review):

A) ROAR cross-check  — does cheap mask-and-repredict faithfulness track expensive
   remove-and-retrain (ROAR)? (validates R1)
B) Multi-seed H2     — is the "no OOD degradation" null stable across resample seeds?
C) Mask-reference    — is the faithfulness ordering stable under mean/median/permutation
   mask references? (R6)

Bounded subsets for compute. Writes results/robustness_{roar,multiseed,maskref}.csv.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from xai_eval.explainers import build_explainer  # noqa: E402
from xai_eval.metrics import faithfulness  # noqa: E402

import data as data_mod  # noqa: E402
from experiment import FRACS, comprehensiveness  # noqa: E402
from train import build_dataset, train_one  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
TOX = ["DILI", "hERG", "AMES", "LD50"]
SUBSET = TOX + ["BBB", "Caco2"]


def _eps():
    return {e.name: e for e in data_mod.load_selected(seed=0)}


def comp_ref(model, X, attr, task, ref_vec):
    """Comprehensiveness with an arbitrary per-feature mask reference vector."""
    X = np.asarray(X, float); d = X.shape[1]; imp = np.abs(attr)
    base = model.predict_proba(X)
    pred = base.argmax(1) if task == "classification" else None
    base_q = base[np.arange(len(X)), pred] if task == "classification" else base.reshape(-1)
    drops = np.zeros(len(X))
    for frac in FRACS:
        k = max(1, int(round(frac * d)))
        topk = np.argpartition(-imp, kth=k - 1, axis=1)[:, :k]
        Xm = X.copy(); rows = np.arange(len(X))[:, None]
        Xm[rows, topk] = np.asarray(ref_vec)[topk]
        mp = model.predict_proba(Xm)
        mq = mp[np.arange(len(X)), pred] if task == "classification" else mp.reshape(-1)
        drops += (base_q - mq) if task == "classification" else np.abs(base_q - mq) / (np.std(base_q) or 1)
    return drops / len(FRACS)


def roar_crosscheck(eps):
    rows = []
    for name in SUBSET:
        ep = eps[name]; task = ep.task
        Xtr, ytr, Xte, yte, names = build_dataset(ep, "descriptors", "scaffold")
        model = train_one(task, "rf", Xtr, ytr, names)
        rng = np.random.default_rng(0); Xe = Xte[rng.choice(len(Xte), min(60, len(Xte)), replace=False)]
        for mth in ["shap", "lime", "random"]:
            attr = build_explainer(mth, **({"num_samples": 1000} if mth == "lime" else {})).explain(model, Xe)
            cheap = faithfulness(model, Xe, yte[:len(Xe)], attr, mode="mask_and_repredict").faithfulness
            roar = faithfulness(model, Xtr, ytr, attr, mode="remove_and_retrain",
                                fractions=[0.0, 0.1, 0.2, 0.5, 1.0], seed=0).faithfulness
            rows.append({"endpoint": name, "method": mth, "cheap_faith": cheap, "roar_faith": roar})
            print(f"ROAR {name:6s} {mth:6s} cheap={cheap:+.3f} roar={roar:+.3f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "robustness_roar.csv"), index=False)
    rho = spearmanr(df.cheap_faith, df.roar_faith).statistic
    print(f"\nROAR cross-check: Spearman(cheap, roar) = {rho:.3f} over {len(df)} (cell,method) pairs")
    return rho


def multiseed_h2(eps):
    rows = []
    for name in TOX:
        ep = eps[name]; task = ep.task
        for seed in [0, 1, 2]:
            for split in ["scaffold", "random"]:
                Xtr, ytr, Xte, yte, names = build_dataset(ep, "descriptors", split)
                model = train_one(task, "rf", Xtr, ytr, names, seed=seed)
                rng = np.random.default_rng(seed); idx = rng.choice(len(Xte), min(60, len(Xte)), replace=False)
                Xe = Xte[idx]
                shap_a = build_explainer("shap").explain(model, Xe).values
                rnd_a = build_explainer("random").explain(model, Xe).values
                fv = comprehensiveness(model, Xe, shap_a, task).mean() - comprehensiveness(model, Xe, rnd_a, task).mean()
                rows.append({"endpoint": name, "seed": seed, "split": split, "faith_vs_null": fv})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "robustness_multiseed.csv"), index=False)
    piv = df.pivot_table(index=["endpoint", "seed"], columns="split", values="faith_vs_null")
    delta = (piv["scaffold"] - piv["random"]).dropna()
    stat, p = wilcoxon(piv["scaffold"], piv["random"], alternative="less") if len(delta) > 5 else (np.nan, np.nan)
    print(f"\nMulti-seed H2: scaffold-random faith_vs_null delta median={delta.median():+.3f} "
          f"(sd {delta.std():.3f}), Wilcoxon p={p:.3f} -> {'degraded' if p<0.05 else 'no sig degradation (null holds)'}")
    return float(delta.median()), float(p)


def mask_reference(eps):
    rows = []
    for name in SUBSET:
        ep = eps[name]; task = ep.task
        Xtr, ytr, Xte, yte, names = build_dataset(ep, "descriptors", "scaffold")
        model = train_one(task, "rf", Xtr, ytr, names)
        rng = np.random.default_rng(0); Xe = Xte[rng.choice(len(Xte), min(60, len(Xte)), replace=False)]
        refs = {"mean": np.zeros(Xtr.shape[1]), "median": np.median(Xtr, 0),
                "perm": Xtr[rng.integers(0, len(Xtr), Xtr.shape[1]), np.arange(Xtr.shape[1])]}
        for mth in ["shap", "lime", "random"]:
            attr = build_explainer(mth, **({"num_samples": 1000} if mth == "lime" else {})).explain(model, Xe).values
            for rname, rvec in refs.items():
                rows.append({"endpoint": name, "method": mth, "ref": rname,
                             "faith": float(comp_ref(model, Xe, attr, task, rvec).mean())})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "robustness_maskref.csv"), index=False)
    print("\nMask-reference: mean faithfulness by method x reference")
    print(df.pivot_table(index="method", columns="ref", values="faith").round(3).to_string())
    return df


def main():
    eps = _eps()
    print("=== R1: ROAR cross-check ==="); roar_crosscheck(eps)
    print("\n=== R4: multi-seed H2 ==="); multiseed_h2(eps)
    print("\n=== R6: mask-reference sensitivity ==="); mask_reference(eps)
    print("\nALL P7 ROBUSTNESS CHECKS COMPLETE")


if __name__ == "__main__":
    main()
