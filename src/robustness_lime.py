"""P7 robustness: is the low SHAP-LIME agreement (H3) an artifact of LIME's sample
budget? Re-compute SHAP-vs-LIME agreement with LIME num_samples=1000 (vs 300 in P4)
on a representative subset and compare.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from xai_eval.explainers import build_explainer  # noqa: E402
from xai_eval.explainers.base import Attribution  # noqa: E402
from xai_eval.metrics import agreement  # noqa: E402

import data as data_mod  # noqa: E402
from train import build_dataset, train_one  # noqa: E402

SUBSET = ["DILI", "hERG", "AMES", "LD50", "Caco2", "BBB"]
RES = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    eps = {e.name: e for e in data_mod.load_selected(seed=0)}
    prev = pd.read_csv(os.path.join(RES, "agreement.csv")).drop_duplicates(
        subset=["endpoint", "representation", "split", "model", "method_a", "method_b"])
    rng = np.random.default_rng(0)
    rows = []
    for name in SUBSET:
        ep = eps[name]
        task = ep.task
        Xtr, ytr, Xte, yte, names = build_dataset(ep, "descriptors", "scaffold")
        model = train_one(task, "rf", Xtr, ytr, names)
        Xe = Xte[rng.choice(len(Xte), min(60, len(Xte)), replace=False)]
        shap_v = build_explainer("shap", background_size=40).explain(model, Xe).values
        lime1000 = build_explainer("lime", num_samples=1000, background_size=100).explain(model, Xe).values
        ag = agreement(Attribution(shap_v, method="shap"), Attribution(lime1000, method="lime"), top_k=10)
        old = prev[(prev.endpoint == name) & (prev.representation == "descriptors")
                   & (prev.split == "scaffold") & (prev.model == "rf")
                   & (prev.method_a.isin(["shap", "lime"])) & (prev.method_b.isin(["shap", "lime"]))]
        old_sp = float(old.spearman_mean.iloc[0]) if len(old) else np.nan
        rows.append({"endpoint": name, "spearman_lime300": old_sp,
                     "spearman_lime1000": ag.mean_spearman, "jaccard_lime1000": ag.mean_topk_jaccard})
        print(f"{name:7s} SHAP-LIME Spearman: 300-sample={old_sp:.3f} -> 1000-sample={ag.mean_spearman:.3f} "
              f"(Jaccard {ag.mean_topk_jaccard:.3f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "robustness_lime.csv"), index=False)
    print(f"\nmean Spearman 300={df.spearman_lime300.mean():.3f}  1000={df.spearman_lime1000.mean():.3f}")
    print("Conclusion: low agreement is", "ROBUST to LIME budget" if df.spearman_lime1000.mean() < 0.5
          else "partly a LIME-budget artifact")


if __name__ == "__main__":
    main()
