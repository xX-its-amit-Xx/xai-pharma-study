"""P4: the attribution x metric x dataset reliability matrix.

For every floor-clearing cell (endpoint x representation x split x model) we rebuild
the model deterministically, draw a fixed test-molecule sample, compute attributions
for the methods appropriate to that model, and score four reliability metrics, each
referenced against a content-matched random null.

Method-availability matrix (each model gets the methods that are valid for it):
    RF, HGB  -> SHAP (TreeExplainer, exact+fast), LIME, random
    MLP      -> Integrated Gradients (Captum), LIME, random

Primary faithfulness is per-molecule **comprehensiveness** (AOPC over fractions:
mask each molecule's own top-attributed features, measure the drop in the predicted
quantity), which is standard, supports bootstrap CIs, and enables a paired
method-vs-null test within each cell. Stability, agreement and the Adebayo sanity
check come from the harness. Results are written incrementally so partial progress
survives, and chunked by representation (CLI arg) to bound wall-clock per run.

Usage:  python src/experiment.py [descriptors|ecfp|all]
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
from xai_eval.metrics import agreement, model_randomization_test, stability  # noqa: E402
from xai_eval.models.base import CLASSIFICATION  # noqa: E402

import data as data_mod  # noqa: E402
from train import build_dataset, train_one  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
ATTR_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "attr_cache")
os.makedirs(ATTR_CACHE, exist_ok=True)

# Bounded, preregistration-consistent budget (logged in deviations if changed).
EXPLAINED_N = 60
STAB_N = 12
N_PERTURB = 4
EPS = 0.1
LIME_SAMPLES = 300
SHAP_BG = 40
TOPK = 10
FRACS = (0.05, 0.1, 0.2)
N_BOOT = 400
SEED = 0

METHODS = {"rf": ["shap", "lime", "random"],
           "hgb": ["shap", "lime", "random"],
           "mlp": ["ig", "lime", "random"]}


def _explainer(method: str):
    if method == "shap":
        return build_explainer("shap", background_size=SHAP_BG)
    if method == "lime":
        return build_explainer("lime", num_samples=LIME_SAMPLES, background_size=100)
    if method == "ig":
        return build_explainer("ig", n_steps=32)
    return build_explainer("random", seed=SEED)


def comprehensiveness(model, X, attr_vals, task, ref=0.0) -> np.ndarray:
    """Per-molecule faithfulness: AOPC of the drop when masking each row's own top
    features. Higher = more faithful. Returns shape (n,)."""
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    imp = np.abs(attr_vals)
    base = model.predict_proba(X)
    if task == CLASSIFICATION:
        pred = base.argmax(1)
        base_q = base[np.arange(len(X)), pred]
    else:
        base_q = base.reshape(-1)
    drops = np.zeros(len(X))
    for frac in FRACS:
        k = max(1, int(round(frac * d)))
        topk = np.argpartition(-imp, kth=k - 1, axis=1)[:, :k]
        Xm = X.copy()
        rows = np.arange(len(X))[:, None]
        Xm[rows, topk] = ref
        mp = model.predict_proba(Xm)
        if task == CLASSIFICATION:
            masked_q = mp[np.arange(len(X)), pred]
            drops += (base_q - masked_q)
        else:
            scale = np.std(base_q) or 1.0
            drops += np.abs(base_q - mp.reshape(-1)) / scale
    return drops / len(FRACS)


def _ci(vals: np.ndarray, rng) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=float)
    boot = [np.mean(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(N_BOOT)]
    return float(vals.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _paired_beats_null(method_v: np.ndarray, null_v: np.ndarray, rng) -> tuple[float, bool]:
    """One-sided paired bootstrap: P(mean(method - null) <= 0). beats_null if p<0.05."""
    diff = np.asarray(method_v) - np.asarray(null_v)
    boot = np.array([np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(N_BOOT)])
    p = float(np.mean(boot <= 0))
    return p, bool(p < 0.05)


def attr_for(cell_key, method, model, Xe):
    path = os.path.join(ATTR_CACHE, f"{cell_key}__{method}.npz")
    if os.path.exists(path):
        return np.load(path)["v"]
    v = _explainer(method).explain(model, Xe).values
    np.savez_compressed(path, v=v)
    return v


def run(representation_filter: str = "all") -> None:
    perf = pd.read_csv(os.path.join(RESULTS, "performance.csv"))
    rel_path = os.path.join(RESULTS, "reliability.csv")
    agr_path = os.path.join(RESULTS, "agreement.csv")
    done = set()
    if os.path.exists(rel_path):
        prev = pd.read_csv(rel_path)
        done = set(zip(prev.endpoint, prev.representation, prev.split, prev.model, prev.method))
    endpoints = {e.name: e for e in data_mod.load_selected(seed=SEED)}
    rng = np.random.default_rng(SEED)

    cells = perf[perf.cleared_floor].copy()
    if representation_filter != "all":
        cells = cells[cells.representation == representation_filter]

    for _, c in cells.iterrows():
        ep = endpoints[c.endpoint]
        rep, split, mn, task = c.representation, c.split, c.model, c.task
        ckey = f"{c.endpoint}__{rep}__{split}__{mn}"
        Xtr, ytr, Xte, yte, names = build_dataset(ep, rep, split)
        model = train_one(task, mn, Xtr, ytr, names)
        n = min(EXPLAINED_N, len(Xte))
        idx = rng.choice(len(Xte), size=n, replace=False)
        Xe = Xte[idx]

        methods = METHODS[mn]
        attrs = {m: attr_for(ckey, m, model, Xe) for m in methods}
        null_comp = comprehensiveness(model, Xe, attrs["random"], task)

        for m in methods:
            if (c.endpoint, rep, split, mn, m) in done:
                continue
            comp = comprehensiveness(model, Xe, attrs[m], task)
            f_mean, f_lo, f_hi = _ci(comp, rng)
            if m == "random":
                p_null, beats = np.nan, False
            else:
                p_null, beats = _paired_beats_null(comp, null_comp, rng)
            expl = _explainer(m)
            st = stability(expl, model, Xe[:STAB_N], epsilon=EPS, n_perturb=N_PERTURB, seed=SEED)
            s_mean, s_lo, s_hi = _ci(np.array(st.per_sample_worst), rng)
            sn = model_randomization_test(expl, model, Xe, threshold=0.5, seed=SEED)
            row = {"endpoint": c.endpoint, "category": c.category, "task": task,
                   "representation": rep, "split": split, "model": mn, "method": m,
                   "n_explained": n, "faith_mean": f_mean, "faith_lo": f_lo, "faith_hi": f_hi,
                   "null_faith_mean": float(null_comp.mean()),
                   "faith_vs_null": f_mean - float(null_comp.mean()),
                   "beats_null_p": p_null, "beats_null": beats,
                   "stab_mean": s_mean, "stab_lo": s_lo, "stab_hi": s_hi,
                   "stab_worst": st.worst_case_sensitivity,
                   "sanity_sim": sn.similarity, "sanity_passed": sn.passed,
                   "primary_perf": c.primary_value, "perf_metric": c.primary_metric}
            pd.DataFrame([row]).to_csv(rel_path, mode="a", header=not os.path.exists(rel_path), index=False)
            print(f"{c.endpoint:11s} {rep:11s} {split:8s} {mn:3s} {m:6s} "
                  f"faith={f_mean:+.3f}(null{null_comp.mean():+.3f}) stab={s_mean:.3f} "
                  f"sanity={sn.similarity:.3f}{'PASS' if sn.passed else 'FAIL'}", flush=True)

        # pairwise agreement among non-random methods (+ each vs random as reference)
        from xai_eval.explainers.base import Attribution
        amap = {m: Attribution(attrs[m], method=m) for m in methods}
        keys = methods
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                ag = agreement(amap[a], amap[b], top_k=TOPK)
                arow = {"endpoint": c.endpoint, "category": c.category, "representation": rep,
                        "split": split, "model": mn, "method_a": a, "method_b": b,
                        "kendall_mean": ag.mean_kendall, "kendall_std": ag.std_kendall,
                        "spearman_mean": ag.mean_spearman, "spearman_std": ag.std_spearman,
                        "jaccard_mean": ag.mean_topk_jaccard, "jaccard_std": ag.std_topk_jaccard,
                        "top_k": TOPK}
                pd.DataFrame([arow]).to_csv(agr_path, mode="a", header=not os.path.exists(agr_path), index=False)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
