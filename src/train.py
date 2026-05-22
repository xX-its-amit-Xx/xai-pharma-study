"""Model-zoo trainer (P3) and the shared, deterministic model builders reused by P4.

Grid: {12 endpoints} x {descriptors, ecfp} x {scaffold, random} x {RF, HGB, MLP}.
Reliability questions are conditional on a competently trained model, so we record
each model's predictive performance and apply a preregistered trivial-baseline
floor; models below the floor are reported but excluded from reliability claims.

P4 imports ``build_dataset`` and ``train_one`` to reconstruct the *identical* model
on demand (fixed seeds + cached features), avoiding fragile serialization.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from xai_eval.models.base import CLASSIFICATION  # noqa: E402
from xai_eval.models.sklearn_adapter import SklearnAdapter  # noqa: E402

import data as data_mod  # noqa: E402
import featurize as feat  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
REPRESENTATIONS = ["descriptors", "ecfp"]
SPLITS = ["scaffold", "random"]
MODELS = ["rf", "hgb", "mlp"]
SEED = 0


def _standardize(train: np.ndarray, test: np.ndarray):
    mu, sd = train.mean(0), train.std(0)
    sd[sd == 0] = 1.0
    return (train - mu) / sd, (test - mu) / sd, mu, sd


def build_dataset(endpoint, representation: str, split: str):
    """Return standardized (Xtr, ytr, Xte, yte, feature_names) for one cell.

    Identical for P3 and P4 thanks to cached featurization + fixed ordering.
    """
    tr = endpoint.splits[split]["train"]
    te = endpoint.splits[split]["test"]
    Xtr_raw, mtr = feat.featurize(endpoint.name, split, "train", representation, tr["Drug"].tolist())
    Xte_raw, mte = feat.featurize(endpoint.name, split, "test", representation, te["Drug"].tolist())
    ytr = tr["Y"].to_numpy()[mtr]
    yte = te["Y"].to_numpy()[mte]
    if endpoint.task == CLASSIFICATION:
        ytr, yte = ytr.astype(int), yte.astype(int)
    else:
        ytr, yte = ytr.astype(float), yte.astype(float)
    Xtr, Xte, _, _ = _standardize(Xtr_raw, Xte_raw)
    return Xtr, ytr, Xte, yte, feat.feature_names(representation)


def train_one(task: str, model_name: str, Xtr, ytr, feature_names, seed: int = SEED):
    """Train one model and return a harness ModelAdapter (deterministic)."""
    if model_name in ("rf", "hgb"):
        if model_name == "rf":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            Est = RandomForestClassifier if task == CLASSIFICATION else RandomForestRegressor
            est = Est(n_estimators=300, max_depth=12, n_jobs=-1, random_state=seed)
        else:
            from sklearn.ensemble import (
                HistGradientBoostingClassifier,
                HistGradientBoostingRegressor,
            )

            Est = HistGradientBoostingClassifier if task == CLASSIFICATION else HistGradientBoostingRegressor
            est = Est(max_iter=300, max_depth=None, learning_rate=0.1, random_state=seed)
        return SklearnAdapter.fit(est, Xtr, ytr, feature_names=feature_names)

    # mlp (torch) -- enables Integrated Gradients in P4
    from models_torch import MLPSpec, make_mlp_adapter

    out_dim = 2 if task == CLASSIFICATION else 1
    spec = MLPSpec(in_dim=Xtr.shape[1], out_dim=out_dim, hidden=(128, 64), task=task)
    return make_mlp_adapter(spec, Xtr, ytr, feature_names=feature_names, seed=seed, epochs=120)


def _evaluate(task: str, model, Xte, yte) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import (
        average_precision_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
    )

    out = {"auroc": np.nan, "auprc": np.nan, "mae": np.nan, "r2": np.nan, "spearman": np.nan}
    if task == CLASSIFICATION:
        proba = model.predict_proba(Xte)[:, 1]
        out["auroc"] = float(roc_auc_score(yte, proba))
        out["auprc"] = float(average_precision_score(yte, proba))
        out["primary_metric"], out["primary_value"] = "AUROC", out["auroc"]
        out["cleared_floor"] = out["auroc"] > 0.55
    else:
        pred = np.asarray(model.predict(Xte)).reshape(-1)
        out["mae"] = float(mean_absolute_error(yte, pred))
        out["r2"] = float(r2_score(yte, pred))
        out["spearman"] = float(spearmanr(yte, pred).statistic)
        out["primary_metric"], out["primary_value"] = "Spearman", out["spearman"]
        out["cleared_floor"] = out["spearman"] > 0.1
    return out


def main() -> None:
    rows = []
    endpoints = list(data_mod.load_selected(seed=SEED))
    print(f"Loaded {len(endpoints)} endpoints")
    for ep in endpoints:
        for rep in REPRESENTATIONS:
            for split in SPLITS:
                Xtr, ytr, Xte, yte, names = build_dataset(ep, rep, split)
                for model_name in MODELS:
                    model = train_one(ep.task, model_name, Xtr, ytr, names)
                    metrics = _evaluate(ep.task, model, Xte, yte)
                    row = {"endpoint": ep.name, "category": ep.category, "task": ep.task,
                           "representation": rep, "split": split, "model": model_name,
                           "n_train": len(ytr), "n_test": len(yte), **metrics}
                    rows.append(row)
                    print(f"{ep.name:11s} {rep:11s} {split:8s} {model_name:3s} "
                          f"{metrics['primary_metric']}={metrics['primary_value']:.3f} "
                          f"{'OK' if metrics['cleared_floor'] else 'BELOW-FLOOR'}", flush=True)
    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS, "performance.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(df)} rows)")
    print(f"Cleared floor: {int(df['cleared_floor'].sum())}/{len(df)}")
    # quick summary by split
    for split in SPLITS:
        sub = df[df.split == split]
        cls = sub[sub.task == "classification"]
        reg = sub[sub.task == "regression"]
        print(f"  {split}: mean AUROC={cls.auroc.mean():.3f}, mean Spearman={reg.spearman.mean():.3f}")


if __name__ == "__main__":
    main()
