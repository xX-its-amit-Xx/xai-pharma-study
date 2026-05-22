"""Paper 2, P9a: the make-or-break premise test (hypothesis C1).

Does per-instance cross-method CONSENSUS (and local stability) predict per-instance
FAITHFULNESS? If yes, we can certify individual explanations cheaply without ground
truth. If no (AUROC <= 0.55), the certificate idea is withdrawn and reported as a
negative result.

For each cell (descriptors, scaffold; trees use SHAP+LIME, MLP uses IG+LIME):
  faith_i      = per-molecule comprehensiveness of the primary method (SHAP/IG)
  null_i       = comprehensiveness of the random baseline (same molecule)
  faithful_i   = faith_i > null_i
  consensus_i  = per-instance rank agreement between the two real methods
  stability_i  = per-instance worst attribution change of the primary method
Then: do consensus_i and stability_i predict faithful_i (pooled, with cell effects)?
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

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from xai_eval.explainers import build_explainer  # noqa: E402
from xai_eval.metrics import stability  # noqa: E402

import data as data_mod  # noqa: E402
from experiment import comprehensiveness  # noqa: E402
from train import build_dataset, train_one  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
N = 60
PRIMARY = {"rf": "shap", "hgb": "shap", "mlp": "ig"}
SECONDARY = "lime"


def per_instance_consensus(a, b):
    """Per-instance Spearman of |a| vs |b| over features. shape (n,) ."""
    a, b = np.abs(a), np.abs(b)
    out = np.zeros(len(a))
    for i in range(len(a)):
        if np.allclose(a[i], a[i][0]) or np.allclose(b[i], b[i][0]):
            out[i] = 0.0
        else:
            s = spearmanr(a[i], b[i]).statistic
            out[i] = 0.0 if np.isnan(s) else s
    return out


def main():
    eps = list(data_mod.load_selected(seed=0))
    rows = []
    rng = np.random.default_rng(0)
    for ep in eps:
        Xtr, ytr, Xte, yte, names = build_dataset(ep, "descriptors", "scaffold")
        for mn in ["rf", "mlp"]:
            model = train_one(ep.task, mn, Xtr, ytr, names)
            idx = rng.choice(len(Xte), min(N, len(Xte)), replace=False)
            Xe = Xte[idx]
            prim_name = PRIMARY[mn]
            a_prim = build_explainer(prim_name, **({"n_steps": 32} if prim_name == "ig" else {})).explain(model, Xe).values
            a_sec = build_explainer(SECONDARY, num_samples=500).explain(model, Xe).values
            a_rand = build_explainer("random").explain(model, Xe).values

            faith = comprehensiveness(model, Xe, a_prim, ep.task)
            null = comprehensiveness(model, Xe, a_rand, ep.task)
            consensus = per_instance_consensus(a_prim, a_sec)
            st = stability(build_explainer(prim_name, **({"n_steps": 32} if prim_name == "ig" else {})),
                           model, Xe, epsilon=0.1, n_perturb=4, seed=0)
            stab = np.array(st.per_sample_worst)
            for i in range(len(Xe)):
                rows.append({"endpoint": ep.name, "model": mn, "primary": prim_name,
                             "faith": faith[i], "null": null[i], "faithful": int(faith[i] > null[i]),
                             "consensus": consensus[i], "stability": stab[i]})
            print(f"{ep.name:11s} {mn:3s} faithful_rate={np.mean(faith>null):.2f} "
                  f"mean_consensus={consensus.mean():+.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "p9a_premise.csv"), index=False)

    # C1 tests, pooled
    y = df.faithful.values
    print("\n=== C1: do consensus / stability predict per-instance faithfulness? ===")
    print(f"n instances = {len(df)}, faithful rate = {y.mean():.2f}")
    if 0 < y.mean() < 1:
        auc_cons = roc_auc_score(y, df.consensus.values)
        auc_stab = roc_auc_score(y, -df.stability.values)  # lower instability -> more faithful
        X = np.column_stack([df.consensus.values, -df.stability.values])
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        lr = LogisticRegression().fit(Xs, y)
        auc_comb = roc_auc_score(y, lr.predict_proba(Xs)[:, 1])
        rho_c = spearmanr(df.consensus, df.faith).statistic
        rho_s = spearmanr(df.stability, df.faith).statistic
        print(f"AUROC consensus->faithful   = {auc_cons:.3f}")
        print(f"AUROC stability->faithful   = {auc_stab:.3f}")
        print(f"AUROC combined (logistic)   = {auc_comb:.3f}")
        print(f"Spearman(consensus, faith)  = {rho_c:+.3f}")
        print(f"Spearman(stability, faith)  = {rho_s:+.3f}")
        verdict = "C1 SUPPORTED (certificate feasible)" if auc_comb > 0.55 else "C1 FALSIFIED (withdraw certificate claim)"
        print(f"\nPreregistered verdict (AUROC>0.55): {verdict}")
        pd.DataFrame([{"auc_consensus": auc_cons, "auc_stability": auc_stab, "auc_combined": auc_comb,
                       "rho_consensus_faith": rho_c, "rho_stability_faith": rho_s,
                       "n": len(df), "faithful_rate": y.mean()}]).to_csv(
            os.path.join(RES, "p9a_verdict.csv"), index=False)


if __name__ == "__main__":
    main()
