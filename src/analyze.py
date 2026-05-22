"""P5: statistical analysis + figures for the preregistered hypotheses H1-H5.

Reads results/reliability.csv (+ agreement.csv), executes each preregistered test
(section 7 of the preregistration), applies Benjamini-Hochberg FDR to the family of
null-comparison tests, and writes summary tables to results/analysis/ and figures to
results/figures/. Privileges effect sizes + CIs over significance stars.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
ANA = os.path.join(RES, "analysis")
FIG = os.path.join(RES, "figures")
os.makedirs(ANA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
TOX = {"hERG", "DILI", "AMES", "LD50"}


def bh_fdr(pvals, q=0.05):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    crit = q * (np.arange(1, n + 1)) / n
    passed = ranked <= crit
    kmax = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(n, dtype=bool)
    if kmax:
        rej[order[:kmax]] = True
    return rej


def main():
    df = pd.read_csv(os.path.join(RES, "reliability.csv"))
    df = df.drop_duplicates(subset=["endpoint", "representation", "split", "model", "method"])
    nm = df[df.method != "random"].copy()
    lines = ["# P5 analysis — preregistered hypothesis tests", "",
             f"Reliability cells: {len(df)} ({df.representation.nunique()} reps, "
             f"{df.endpoint.nunique()} endpoints, methods {sorted(df.method.unique())})", ""]

    # ---- H1: faithfulness vs null ----
    rej = bh_fdr(nm.beats_null_p.fillna(1.0).values, q=0.05)
    nm["beats_null_fdr"] = rej
    frac_fail = float((~nm.beats_null_fdr).mean())
    lines += ["## H1 — faithfulness over the null",
              f"- non-random method-cells: {len(nm)}",
              f"- fraction NOT beating null (BH-FDR q=0.05): **{frac_fail:.2f}** "
              f"(prereg threshold for H1 support: >=0.20 -> {'SUPPORTED' if frac_fail>=0.2 else 'FALSIFIED'})",
              f"- raw (uncorrected) not-beating rate: {(~nm.beats_null).mean():.2f}",
              "- by method (fraction not beating null, FDR):"]
    for m, g in nm.groupby("method"):
        lines.append(f"    - {m}: {(~g.beats_null_fdr).mean():.2f} (n={len(g)})")

    # ---- H2: OOD degradation (scaffold vs random) ----
    lines += ["", "## H2 — reliability degrades out-of-distribution (scaffold vs random)"]
    key = ["endpoint", "representation", "model", "method"]
    piv = nm.pivot_table(index=key, columns="split", values=["faith_mean", "stab_mean"])
    for metric, better in [("faith_mean", "lower scaffold = worse"), ("stab_mean", "higher scaffold = worse")]:
        sub = piv[metric].dropna()
        if {"scaffold", "random"}.issubset(sub.columns) and len(sub) > 5:
            if metric == "faith_mean":
                diff = sub["scaffold"] - sub["random"]  # negative => degraded
                stat, p = wilcoxon(sub["scaffold"], sub["random"], alternative="less")
            else:
                diff = sub["scaffold"] - sub["random"]  # positive => degraded (less stable)
                stat, p = wilcoxon(sub["scaffold"], sub["random"], alternative="greater")
            lines.append(f"- {metric}: median scaffold-random delta = {diff.median():+.3f}, "
                         f"Wilcoxon p={p:.4f} ({'degraded' if p<0.05 else 'no sig. degradation'}) [{better}]")

    # ---- H3: cross-method agreement ----
    lines += ["", "## H3 — explainer disagreement"]
    if os.path.exists(os.path.join(RES, "agreement.csv")):
        ag = pd.read_csv(os.path.join(RES, "agreement.csv")).drop_duplicates(
            subset=["endpoint", "representation", "split", "model", "method_a", "method_b"])
        agnr = ag[(ag.method_a != "random") & (ag.method_b != "random")]
        med = agnr.spearman_mean.median()
        lines.append(f"- median pairwise Spearman (non-random pairs): **{med:.3f}** "
                     f"(prereg descriptive threshold <0.5 -> {'low agreement' if med<0.5 else 'not low'})")
        lines.append("- by representation (median Spearman / median top-k Jaccard):")
        for rep, g in agnr.groupby("representation"):
            lines.append(f"    - {rep}: Spearman {g.spearman_mean.median():.3f}, "
                         f"Jaccard {g.jaccard_mean.median():.3f}")
        lines.append("- by method-pair (median Spearman, descriptors only):")
        for (a, b), g in agnr[agnr.representation == "descriptors"].groupby(["method_a", "method_b"]):
            lines.append(f"    - {a} vs {b}: {g.spearman_mean.median():.3f} (n={len(g)})")
        agnr = agnr.assign(is_tox=agnr.endpoint.isin(TOX))
        tox, non = agnr[agnr.is_tox].spearman_mean, agnr[~agnr.is_tox].spearman_mean
        if len(tox) > 3 and len(non) > 3:
            u, p = mannwhitneyu(tox, non, alternative="two-sided")
            lines.append(f"- toxicity vs non-toxicity agreement: tox median {tox.median():.3f} "
                         f"vs {non.median():.3f}, Mann-Whitney p={p:.3f}")
        agnr.to_csv(os.path.join(ANA, "agreement_nonrandom.csv"), index=False)

    # ---- H4: sanity-check failures, split by model class ----
    lines += ["", "## H4 — model-randomization sanity failures (split by model class)"]
    nm = nm.assign(model_class=np.where(nm.model == "mlp", "mlp(true-reinit)", "tree(label-perm)"))
    for mc, g in nm.groupby("model_class"):
        fail = (~g.sanity_passed).mean()
        lines.append(f"- {mc}: fail rate {fail:.2f} (n={len(g)}) "
                     f"{'(>=0.15 -> support)' if fail>=0.15 else ''}")
    lines.append("  NOTE: tree fail rate uses the label-permutation analogue (limitation D-prereg §9); "
                 "the MLP rate is the true Adebayo weight-reinitialization test.")

    # ---- H5: representation vs method (eta^2 one-way) ----
    lines += ["", "## H5 — representation vs method (variance explained, eta^2)"]
    def eta2(frame, factor, y="faith_mean"):
        grand = frame[y].mean()
        ss_tot = ((frame[y] - grand) ** 2).sum()
        ss_b = sum(len(g) * (g[y].mean() - grand) ** 2 for _, g in frame.groupby(factor))
        return ss_b / ss_tot if ss_tot > 0 else np.nan
    for y in ["faith_mean", "stab_mean", "sanity_sim"]:
        e_rep = eta2(nm, "representation", y)
        e_meth = eta2(nm, "method", y)
        e_model = eta2(nm, "model", y)
        e_ep = eta2(nm, "endpoint", y)
        lines.append(f"- {y}: eta^2 representation={e_rep:.3f}, method={e_meth:.3f}, "
                     f"model={e_model:.3f}, endpoint={e_ep:.3f}")

    open(os.path.join(ANA, "hypotheses_summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    df.to_csv(os.path.join(ANA, "reliability_dedup.csv"), index=False)
    print("\n".join(lines))
    _figures(df, nm)
    print("\nWrote analysis to results/analysis/ and figures to results/figures/")


def _figures(df, nm):
    # Fig 1: faithfulness vs null by method
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["random", "lime", "shap", "ig"]
    data = [df[df.method == m].faith_mean.dropna() for m in order if m in df.method.unique()]
    labs = [m for m in order if m in df.method.unique()]
    ax.boxplot(data, labels=labs)
    ax.set_ylabel("per-molecule faithfulness (comprehensiveness)")
    ax.set_title("Faithfulness by attribution method (all endpoints)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "faith_by_method.png"), dpi=120); plt.close(fig)

    # Fig 2: scaffold vs random faithfulness (paired)
    fig, ax = plt.subplots(figsize=(5, 5))
    piv = nm.pivot_table(index=["endpoint", "representation", "model", "method"],
                         columns="split", values="faith_mean").dropna()
    if {"scaffold", "random"}.issubset(piv.columns):
        ax.scatter(piv["random"], piv["scaffold"], alpha=0.5, s=18)
        lim = [min(piv.min().min(), 0), piv.max().max()]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("faithfulness (random split)"); ax.set_ylabel("faithfulness (scaffold split)")
        ax.set_title("OOD effect: scaffold vs random")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "ood_faith.png"), dpi=120); plt.close(fig)

    # Fig 3: sanity similarity by model class
    fig, ax = plt.subplots(figsize=(6, 4))
    nm2 = nm.assign(mc=np.where(nm.model == "mlp", "mlp", "tree"))
    for i, (mc, g) in enumerate(nm2.groupby("mc")):
        ax.scatter(np.full(len(g), i) + np.random.uniform(-0.1, 0.1, len(g)), g.sanity_sim, alpha=0.4, s=14)
    ax.axhline(0.5, ls="--", c="red", label="fail threshold")
    ax.set_xticks(range(nm2.mc.nunique())); ax.set_xticklabels(sorted(nm2.mc.unique()))
    ax.set_ylabel("|rank corr| trained vs randomized"); ax.set_title("Adebayo sanity by model class")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG, "sanity_by_modelclass.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
