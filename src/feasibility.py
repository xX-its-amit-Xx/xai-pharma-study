"""P2 feasibility gate: real TDC endpoint -> RDKit features -> model -> all 4 metrics.

Confirms the whole Paper-1 pipeline runs end-to-end on one real endpoint before we
commit to the full factorial. Run: PYTHONPATH=../xai-eval-harness/src python src/feasibility.py
"""

from __future__ import annotations

import warnings

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestClassifier

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

from xai_eval.explainers import build_explainer  # noqa: E402
from xai_eval.metrics import (  # noqa: E402
    agreement,
    faithfulness,
    model_randomization_test,
    stability,
)
from xai_eval.models.sklearn_adapter import SklearnAdapter  # noqa: E402

_DESC = [(n, f) for n, f in Descriptors._descList]
_DESC_NAMES = [n for n, _ in _DESC]


def descriptors(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """RDKit 2D descriptors; returns (X, valid_mask)."""
    rows, mask = [], []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            mask.append(False)
            continue
        vals = []
        for _, fn in _DESC:
            try:
                vals.append(float(fn(m)))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
        mask.append(True)
    X = np.asarray(rows, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.asarray(mask)


def standardize(train: np.ndarray, *others: np.ndarray):
    mu, sd = train.mean(0), train.std(0)
    sd[sd == 0] = 1.0
    return tuple((a - mu) / sd for a in (train, *others))


def main() -> None:
    from tdc.single_pred import Tox

    split = Tox(name="DILI").get_split(method="scaffold")
    tr, te = split["train"], split["test"]

    Xtr_raw, mtr = descriptors(tr["Drug"].tolist())
    Xte_raw, mte = descriptors(te["Drug"].tolist())
    ytr = tr["Y"].to_numpy()[mtr].astype(int)
    yte = te["Y"].to_numpy()[mte].astype(int)
    Xtr, Xte = standardize(Xtr_raw, Xte_raw)
    print(f"DILI scaffold: train {Xtr.shape}, test {Xte.shape}, {len(_DESC_NAMES)} descriptors")
    print(f"label balance train={ytr.mean():.2f} test={yte.mean():.2f}")

    rf = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=0)
    model = SklearnAdapter.fit(rf, Xtr, ytr, feature_names=_DESC_NAMES)
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    print(f"test AUROC = {auc:.3f}  (TDC DILI leaderboard ~0.88-0.92; ours is a quick baseline)")

    n = min(80, len(Xte))
    Xe, ye = Xte[:n], yte[:n]
    attrs = {name: build_explainer(name).explain(model, Xe)
             for name in ["shap", "lime", "random"]}

    print("\n--- all four metrics on real DILI endpoint ---")
    rand_f = faithfulness(model, Xe, ye, attrs["random"]).faithfulness
    for name, attr in attrs.items():
        f = faithfulness(model, Xe, ye, attr)
        s = stability(build_explainer(name), model, Xe[:30], epsilon=0.1, n_perturb=6)
        sn = model_randomization_test(build_explainer(name), model, Xe, threshold=0.5)
        beat = "" if name == "random" else (" [beats null]" if f.faithfulness > rand_f else " [<= NULL]")
        print(f"{name:7s} faith={f.faithfulness:+.3f}{beat:12s} "
              f"stab(mean)={s.mean_sensitivity:.3f} sanity_sim={sn.similarity:.3f} "
              f"{'PASS' if sn.passed else 'FAIL'}")
    sl = agreement(attrs["shap"], attrs["lime"], top_k=10)
    print(f"\nSHAP vs LIME: Spearman {sl.mean_spearman:.3f}+/-{sl.std_spearman:.3f}, "
          f"top-10 Jaccard {sl.mean_topk_jaccard:.3f}")
    print("\nFEASIBILITY GATE: PASS")


if __name__ == "__main__":
    main()
