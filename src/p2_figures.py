"""Paper 2 figures: the ablation (key honest figure) and cross-modality transfer."""

from __future__ import annotations

import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)


def ablation_fig():
    ab = pd.read_csv(os.path.join(RES, "p9d_ablation.csv")).iloc[0]
    labels = ["confidence\nonly", "attribution\nonly", "full minus\nattr-magnitude", "full\n(7 feats)"]
    keys = ["confidence-only [conf,margin]", "attribution-only [attr_l2,attr_entropy]",
            "no-attr-magnitude (drop attr_l2)", "full (7 features)"]
    vals = [ab[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, vals, color=["#1f77b4", "#aec7e8", "#ff9896", "#d62728"])
    ax.axhline(0.5, ls=":", c="gray", label="chance")
    ax.set_ylim(0.45, 0.75); ax.set_ylabel("within-cell AUROC (predicting faithful instance)")
    ax.set_title("The certificate is largely model confidence\n(explanation-specific features add +0.01)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "p2_ablation_confidence.png"), dpi=120); plt.close(fig)


def cross_modality_fig():
    rows = [("molecules\n(ADMET/tox)", 0.694, None, None)]
    for f, lab in [("p9c_transcriptomics_verdict.csv", "transcriptomics\n(leukemia)"),
                   ("p9c_sequence_verdict.csv", "sequence\n(transformer)")]:
        p = os.path.join(RES, f)
        if os.path.exists(p):
            r = pd.read_csv(p).iloc[0]
            rows.append((lab, r["auc"], r["auc_lo"], r["auc_hi"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = range(len(rows))
    vals = [r[1] for r in rows]
    err = [[r[1] - (r[2] if r[2] is not None else r[1]) for r in rows],
           [(r[3] if r[3] is not None else r[1]) - r[1] for r in rows]]
    ax.bar(xs, vals, yerr=err, capsize=5, color="#2ca02c")
    ax.axhline(0.55, ls="--", c="red", label="preregistered bar (0.55)")
    ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("certifier AUROC")
    ax.set_title("Per-instance certificate transfers across modalities")
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "p2_cross_modality.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    ablation_fig()
    cross_modality_fig()
    print("wrote p2_ablation_confidence.png, p2_cross_modality.png")
