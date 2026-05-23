"""Regenerate every manuscript figure at publication quality.

Standards applied:
- 300 DPI output
- TrueType fonts embedded (pdf.fonttype=42, ps.fonttype=42)
- Arial/DejaVu Sans family for cross-platform consistency
- Color-blind-safe palette (Wong 2011 / Okabe-Ito-derived)
- Sized for single-column (~89 mm) or double-column (~183 mm) journals
- Consistent axis weights and spine treatment
- Saved as PNG (rasterized embed) and PDF (vector, preferred for journals)

Outputs to results/figures/pub/.
"""

from __future__ import annotations

import os
import warnings

import matplotlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Publication settings
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,        # embed TrueType
    "font.family": "DejaVu Sans",                  # cross-platform; Arial-like
    "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})

# Color-blind-safe palette (Okabe-Ito)
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "gray": "#999999",
      "sky": "#56B4E9", "yellow": "#F0E442"}

RES = os.path.join(os.path.dirname(__file__), "..", "results")
PUB = os.path.join(RES, "figures", "pub")
os.makedirs(PUB, exist_ok=True)

# Sizes (inches) for Nature-style columns
SINGLE = (3.5, 2.8)   # ~89 mm wide
DOUBLE = (7.2, 4.0)   # ~183 mm wide


def savefig(fig, name, both_formats=True):
    fig.savefig(os.path.join(PUB, f"{name}.png"), dpi=300, bbox_inches="tight")
    if both_formats:
        fig.savefig(os.path.join(PUB, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Paper 1 figures
# ============================================================================

def fig_p1_faith_by_method():
    """Per-method faithfulness across endpoints (uses reliability.csv comprehensiveness)."""
    df = pd.read_csv(os.path.join(RES, "reliability.csv")).drop_duplicates(
        subset=["endpoint", "representation", "split", "model", "method"])
    fig, ax = plt.subplots(figsize=SINGLE)
    order = ["random", "lime", "shap", "ig"]
    palette = {"random": CB["gray"], "lime": CB["orange"], "shap": CB["blue"], "ig": CB["green"]}
    data = [df[df.method == m].faith_mean.dropna() for m in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.2),
                    flierprops=dict(marker="o", markersize=2, markerfacecolor="black", alpha=0.4))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(palette[m]); patch.set_edgecolor("black"); patch.set_linewidth(0.8)
    ax.set_ylabel("per-molecule faithfulness")
    ax.set_xlabel("attribution method")
    ax.set_title("Paper 1 · Faithfulness by method (all cells)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    savefig(fig, "p1_fig1_faith_by_method")


def fig_p1_ood():
    """Scaffold vs random faithfulness — H2 visual (no OOD degradation)."""
    df = pd.read_csv(os.path.join(RES, "reliability.csv")).drop_duplicates(
        subset=["endpoint", "representation", "split", "model", "method"])
    nm = df[df.method != "random"]
    piv = nm.pivot_table(index=["endpoint", "representation", "model", "method"],
                         columns="split", values="faith_mean").dropna()
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.scatter(piv["random"], piv["scaffold"], s=14, alpha=0.55, color=CB["blue"], edgecolor="none")
    lim = [min(piv.min().min(), 0), piv.max().max() * 1.02]
    ax.plot(lim, lim, ls="--", color="black", linewidth=0.8, label="y = x (no shift)")
    ax.set_xlabel("faithfulness · random split")
    ax.set_ylabel("faithfulness · scaffold (OOD) split")
    ax.set_title("Paper 1 · H2 falsified — no OOD faithfulness collapse")
    ax.legend(frameon=False)
    savefig(fig, "p1_fig2_ood")


def fig_p1_sanity():
    """Adebayo sanity by model class — H4."""
    df = pd.read_csv(os.path.join(RES, "reliability.csv")).drop_duplicates(
        subset=["endpoint", "representation", "split", "model", "method"])
    nm = df[df.method != "random"].copy()
    nm["model_class"] = np.where(nm.model == "mlp", "MLP (true weight reinit)",
                                  "tree (label-perm)")
    fig, ax = plt.subplots(figsize=SINGLE)
    classes = sorted(nm.model_class.unique())
    rng = np.random.default_rng(0)
    for i, cls in enumerate(classes):
        g = nm[nm.model_class == cls]
        color = CB["green"] if "MLP" in cls else CB["red"]
        ax.scatter(np.full(len(g), i) + rng.uniform(-0.15, 0.15, len(g)),
                   g.sanity_sim, alpha=0.5, s=12, color=color, edgecolor="none")
    ax.axhline(0.5, ls="--", color=CB["red"], linewidth=0.9, label="fail threshold (0.5)")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=7)
    ax.set_ylabel(r"$|\rho|$ trained vs randomized model")
    ax.set_title("Paper 1 · H4 — Adebayo sanity by model class")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, "p1_fig3_sanity")


def fig_p1_external_alerts():
    """PAINS/BRENK alert overlap — external chemistry validation §3.8."""
    sm = pd.read_csv(os.path.join(RES, "r4_alert_overlap_summary.csv"))
    fig, ax = plt.subplots(figsize=SINGLE)
    xs = np.arange(len(sm))
    err = [[d - lo for d, lo in zip(sm.delta, sm.delta_lo)],
           [hi - d for d, hi in zip(sm.delta, sm.delta_hi)]]
    colors = [CB["green"] if lo > 0 else CB["gray"] for lo in sm.delta_lo]
    ax.bar(xs, sm.delta, yerr=err, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(sm.endpoint)
    ax.set_ylabel(r"$\Delta$ AUROC (attribution vs random)")
    ax.set_title("Paper 1 · External validation: PAINS/BRENK alert overlap")
    for i, (d, lo, hi) in enumerate(zip(sm.delta, sm.delta_lo, sm.delta_hi)):
        mark = "*" if lo > 0 else ""
        ax.text(i, hi + 0.01, mark, ha="center", fontsize=12)
    savefig(fig, "p1_fig4_external_alerts")


def fig_p1_gnn_extension():
    """GIN faithfulness + sanity — §3.9."""
    g = pd.read_csv(os.path.join(RES, "gnn_extension.csv"))
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
    xs = np.arange(len(g))
    ax = axes[0]
    ax.bar(xs - 0.2, g.faith, width=0.4, label="attribution", color=CB["green"],
           edgecolor="black", linewidth=0.6)
    ax.bar(xs + 0.2, g.null_faith, width=0.4, label="random null", color=CB["gray"],
           edgecolor="black", linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(g.endpoint)
    ax.set_ylabel("per-molecule faithfulness"); ax.set_title("GIN occlusion vs random null")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.bar(xs, g.sanity_sim, color=CB["blue"], edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, ls="--", color=CB["red"], linewidth=0.9, label="fail threshold")
    ax.set_xticks(xs); ax.set_xticklabels(g.endpoint)
    ax.set_ylabel(r"$|\rho|$ trained vs randomized GIN")
    ax.set_title("GIN sanity (true weight reinit) — all PASS")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Paper 1 · §3.9 GNN learned-representation extension")
    fig.tight_layout()
    savefig(fig, "p1_fig5_gnn_extension")


# ============================================================================
# Paper 2 figures
# ============================================================================

def fig_p2_ablation():
    """The critical Paper 2 figure — certificate is largely model confidence."""
    ab = pd.read_csv(os.path.join(RES, "p9d_ablation.csv")).iloc[0]
    labels = ["confidence-\nonly", "attribution-\nonly", "drop attr.\nmagnitude", "full\n(7 feats)"]
    keys = ["confidence-only [conf,margin]", "attribution-only [attr_l2,attr_entropy]",
            "no-attr-magnitude (drop attr_l2)", "full (7 features)"]
    vals = [ab[k] for k in keys]
    colors = [CB["blue"], CB["sky"], CB["purple"], CB["red"]]
    fig, ax = plt.subplots(figsize=SINGLE)
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, ls=":", color=CB["gray"], linewidth=0.7, label="chance (0.5)")
    ax.set_ylim(0.45, 0.75); ax.set_ylabel("within-cell AUROC")
    ax.set_title("Paper 2 · The certificate is mostly model confidence")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", fontsize=7)
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, "p2_fig1_ablation")


def fig_p2_abstention():
    """Abstention curve — C2."""
    u = pd.read_csv(os.path.join(RES, "p9b_utility.csv"))
    fig, ax = plt.subplots(figsize=SINGLE)
    base = float(u[u.coverage == 1.0].cert_retained_faith.iloc[0])
    ax.plot(u.coverage, u.cert_retained_faith, marker="o", color=CB["green"],
            markersize=4, linewidth=1.2, label="certificate-guided")
    ax.axhline(base, ls="--", color=CB["gray"], linewidth=0.9, label="random abstention")
    ax.invert_xaxis()
    ax.set_xlabel("coverage (fraction retained)")
    ax.set_ylabel("mean faithfulness of retained set")
    ax.set_title("Paper 2 · C2 — abstention utility (within-cell)")
    ax.legend(frameon=False, loc="upper right")
    savefig(fig, "p2_fig2_abstention")


def fig_p2_cross_modality():
    """Cross-omics transfer — C3."""
    rows = [("molecules\n(ADMET/tox)", 0.694, None, None)]
    for fname, lab in [("p9c_transcriptomics_verdict.csv", "transcriptomics\n(leukemia)"),
                       ("p9c_sequence_verdict.csv", "sequence\n(transformer)")]:
        p = os.path.join(RES, fname)
        if os.path.exists(p):
            r = pd.read_csv(p).iloc[0]
            rows.append((lab, r["auc"], r["auc_lo"], r["auc_hi"]))
    fig, ax = plt.subplots(figsize=SINGLE)
    xs = list(range(len(rows)))
    vals = [r[1] for r in rows]
    err = [[r[1] - (r[2] if r[2] is not None else r[1]) for r in rows],
           [(r[3] if r[3] is not None else r[1]) - r[1] for r in rows]]
    ax.bar(xs, vals, yerr=err, capsize=4, color=CB["green"], edgecolor="black", linewidth=0.6)
    ax.axhline(0.55, ls="--", color=CB["red"], linewidth=0.9, label="prereg. bar (0.55)")
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("certifier AUROC")
    ax.set_title("Paper 2 · C3 — transfer across modalities")
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    ax.legend(frameon=False)
    savefig(fig, "p2_fig3_cross_modality")


# ============================================================================
# Paper 3 figures
# ============================================================================

def fig_p3_recovery():
    """D1 ground-truth recovery by method (main 3-method, n=3434)."""
    df = pd.read_csv(os.path.join(RES, "p12_main.csv"))
    methods = [("ig_auc", "IG", CB["sky"]), ("occ_auc", "occlusion", CB["green"]),
               ("rand_auc", "random", CB["gray"])]
    rng = np.random.default_rng(0)
    def boot(v):
        v = np.asarray(v); b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
    fig, ax = plt.subplots(figsize=SINGLE)
    means, los, his = zip(*[boot(df[c]) for c, _, _ in methods])
    xs = np.arange(len(methods))
    colors = [c for _, _, c in methods]
    err = [[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]]
    ax.bar(xs, means, yerr=err, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, ls=":", color=CB["gray"], linewidth=0.7, label="chance")
    ax.axhline(0.6, ls="--", color=CB["red"], linewidth=0.9, label="prereg. bar (0.6)")
    ax.set_xticks(xs); ax.set_xticklabels([n for _, n, _ in methods])
    ax.set_ylim(0.40, 0.80); ax.set_ylabel("ground-truth atom recovery AUROC")
    ax.set_title(f"Paper 3 · D1 — chemistry recovery (n={len(df)})")
    for x, m in zip(xs, means):
        ax.text(x, m + 0.012, f"{m:.3f}", ha="center", fontsize=8)
    ax.legend(frameon=False)
    savefig(fig, "p3_fig1_d1_recovery")


def fig_p3_d2_extended():
    """D2 extended — scatter of recovery vs faithfulness across 6 methods."""
    d = pd.read_csv(os.path.join(RES, "r4_p3_d2_extended.csv"))
    fig, ax = plt.subplots(figsize=SINGLE)
    colors = {"occlusion": CB["green"], "IG": CB["sky"], "grad": CB["purple"],
              "grad*input": CB["orange"], "smoothgrad": CB["red"], "random": CB["gray"]}
    label_map = {"grad": "saliency", "grad*input": "grad×input",
                 "smoothgrad": "SmoothGrad", "IG": "IG", "occlusion": "occlusion",
                 "random": "random"}
    for _, r in d.iterrows():
        ax.scatter(r["faithfulness"], r["recovery"], s=80,
                   color=colors.get(r["method"], "black"), edgecolor="black", zorder=3)
        ax.annotate(label_map.get(r["method"], r["method"]),
                    (r["faithfulness"] + 0.003, r["recovery"] + 0.005),
                    fontsize=7)
    ax.axhline(0.5, ls=":", color=CB["gray"], linewidth=0.7, label="chance recovery")
    ax.axhline(0.6, ls="--", color=CB["red"], linewidth=0.9, alpha=0.5, label="prereg. bar (0.6)")
    ax.set_xlabel("null-referenced mask-faithfulness")
    ax.set_ylabel("ground-truth recovery AUROC")
    ax.set_title("Paper 3 · D2-ext — faithfulness ≠ chemistry "
                 r"(Spearman = −0.09, $p$ = 0.87, n = 6)")
    ax.set_xlim(0.06, 0.20); ax.set_ylim(0.45, 0.58)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    savefig(fig, "p3_fig2_d2_extended")


# ============================================================================
if __name__ == "__main__":
    print("Generating Paper 1 figures...")
    fig_p1_faith_by_method(); fig_p1_ood(); fig_p1_sanity()
    fig_p1_external_alerts(); fig_p1_gnn_extension()
    print("Generating Paper 2 figures...")
    fig_p2_ablation(); fig_p2_abstention(); fig_p2_cross_modality()
    print("Generating Paper 3 figures...")
    fig_p3_recovery(); fig_p3_d2_extended()
    print(f"\nAll figures written to {PUB}/")
    print(f"Files: {sorted(os.listdir(PUB))}")
