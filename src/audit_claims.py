"""Skeptic claim audit: verify EVERY numerical claim in the three manuscripts against the
source CSVs. Output a Markdown table (claim, source, value-in-data, match/mismatch) so that
any reviewer can confirm in seconds.

Run: PYTHONPATH=src python src/audit_claims.py > docs/claim_audit.md
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(__file__), "..", "results")
TOL = 0.01           # default tolerance for floats (1 percentage point or 0.01 of value)
TOL_PCT = 0.06       # tolerance for percentage claims (matches manuscripts rounding)

claims: list[tuple[str, str, str, float | None, float | None, bool, str]] = []


def claim(paper: str, where: str, name: str, claimed, observed, ok: bool, note: str = "") -> None:
    claims.append((paper, where, name, claimed, observed, ok, note))


def near(a, b, tol=TOL):
    return a is None or b is None or abs(a - b) <= tol


# ---------- Paper 1 ----------
df = pd.read_csv(os.path.join(RES, "reliability.csv")).drop_duplicates(
    subset=["endpoint", "representation", "split", "model", "method"])
nm = df[df.method != "random"].copy()
perf = pd.read_csv(os.path.join(RES, "performance.csv"))

# Total models and floor
total_models = len(perf)
cleared = int(perf.cleared_floor.sum())
claim("P1", "Methods §2.2", "total models trained", 144, total_models, total_models == 144)
claim("P1", "Methods §2.2", "models clearing floor", 142, cleared, cleared == 142)

# H1 — per-method fail rates (raw, since manuscript reports both raw and FDR; check raw counts)
ig = nm[nm.method == "ig"]; lime = nm[nm.method == "lime"]; shap = nm[nm.method == "shap"]
claim("P1", "Results §3.2", "IG fail count / total", "0 / 48", f"{(~ig.beats_null).sum()} / {len(ig)}",
      (~ig.beats_null).sum() == 0 and len(ig) == 48)
claim("P1", "Results §3.2", "LIME fail rate (raw)", "15%", f"{(~lime.beats_null).mean()*100:.0f}%",
      abs((~lime.beats_null).mean() - 0.15) < TOL_PCT, "manuscript reports 15% (21/142 = 14.8%)")
claim("P1", "Results §3.2", "SHAP fail rate (raw)", "12%", f"{(~shap.beats_null).mean()*100:.0f}%",
      abs((~shap.beats_null).mean() - 0.12) < TOL_PCT, "manuscript reports 12% (11/94 = 11.7%)")

# H4 sanity
mlp = nm[nm.model == "mlp"]; tree = nm[nm.model != "mlp"]
claim("P1", "Results §3.5", "MLP sanity-fail rate", "25%", f"{(~mlp.sanity_passed).mean()*100:.0f}%",
      abs((~mlp.sanity_passed).mean() - 0.25) < TOL_PCT, "24/96 = 25.0%")
claim("P1", "Results §3.5", "tree sanity-fail rate", "39%", f"{(~tree.sanity_passed).mean()*100:.0f}%",
      abs((~tree.sanity_passed).mean() - 0.39) < TOL_PCT, "74/188 = 39.4%")

# H3 LIME-1000 robustness
lime_rob = pd.read_csv(os.path.join(RES, "robustness_lime.csv"))
mean_300 = float(lime_rob.spearman_lime300.mean())
mean_1000 = float(lime_rob.spearman_lime1000.mean())
claim("P1", "Results §3.4", "LIME-300 mean Spearman (SHAP-LIME)", 0.15, round(mean_300, 2),
      near(mean_300, 0.15))
claim("P1", "Results §3.4", "LIME-1000 mean Spearman (SHAP-LIME)", 0.34, round(mean_1000, 2),
      near(mean_1000, 0.34))
# per-endpoint LIME-1000 numbers cited in §3.4
per_ep = lime_rob.set_index("endpoint").spearman_lime1000.to_dict()
for ep, claimed_val in [("DILI", 0.50), ("hERG", 0.42), ("AMES", 0.37),
                        ("BBB", 0.36), ("LD50", 0.19), ("Caco2", 0.18)]:
    obs = round(float(per_ep.get(ep, np.nan)), 2)
    claim("P1", "Results §3.4", f"LIME-1000 rho for {ep}", claimed_val, obs, near(claimed_val, obs, 0.01))

# R1 ROAR cross-check (validate the primary metric)
roar = pd.read_csv(os.path.join(RES, "robustness_roar.csv"))
rel_subset = df[(df.representation == "descriptors") & (df.split == "scaffold") & (df.model == "rf")]
m = roar.merge(rel_subset[["endpoint", "method", "faith_mean"]], on=["endpoint", "method"], how="inner")
from scipy.stats import spearmanr
rho_prim = spearmanr(m.faith_mean, m.roar_faith).statistic
rho_naive = spearmanr(roar.cheap_faith, roar.roar_faith).statistic
claim("P1", "Results §3.7", "primary comprehensiveness vs ROAR Spearman", 0.93, round(rho_prim, 2),
      near(rho_prim, 0.93, 0.03))
claim("P1", "Results §3.7", "naive score-AOPC vs ROAR Spearman", -0.50, round(rho_naive, 2),
      near(rho_naive, -0.50, 0.03))

# Multi-seed H2
ms = pd.read_csv(os.path.join(RES, "robustness_multiseed.csv"))
piv = ms.pivot_table(index=["endpoint", "seed"], columns="split", values="faith_vs_null")
delta = (piv["scaffold"] - piv["random"]).dropna()
from scipy.stats import wilcoxon
stat, pval = wilcoxon(piv["scaffold"], piv["random"], alternative="less") if len(delta) > 5 else (np.nan, np.nan)
claim("P1", "Results §3.7", "multi-seed scaffold-rand Δ median", -0.010, round(float(delta.median()), 3),
      near(float(delta.median()), -0.010, 0.005))
claim("P1", "Results §3.7", "multi-seed Wilcoxon p", 0.19, round(float(pval), 2), near(float(pval), 0.19, 0.02))

# Mask reference
mref = pd.read_csv(os.path.join(RES, "robustness_maskref.csv"))
pv = mref.pivot_table(index="method", columns="ref", values="faith")
shap_range = (pv.loc["shap"].min(), pv.loc["shap"].max())
lime_range = (pv.loc["lime"].min(), pv.loc["lime"].max())
rand_range = (pv.loc["random"].min(), pv.loc["random"].max())
claim("P1", "Results §3.7", "SHAP mask-ref range",
      "0.33-0.37", f"{shap_range[0]:.2f}-{shap_range[1]:.2f}",
      abs(shap_range[0] - 0.33) < 0.02 and abs(shap_range[1] - 0.37) < 0.02)
claim("P1", "Results §3.7", "LIME mask-ref range",
      "0.30-0.32", f"{lime_range[0]:.2f}-{lime_range[1]:.2f}",
      abs(lime_range[0] - 0.30) < 0.02 and abs(lime_range[1] - 0.32) < 0.02)
claim("P1", "Results §3.7", "random mask-ref range",
      "0.07-0.09", f"{rand_range[0]:.2f}-{rand_range[1]:.2f}",
      abs(rand_range[0] - 0.07) < 0.02 and abs(rand_range[1] - 0.09) < 0.02)

# Shift magnitude
sh = pd.read_csv(os.path.join(RES, "r2_shift.csv"))
shift_mean = float(sh.shift_ratio.mean()); shift_max = float(sh.shift_ratio.max())
shift_rho = spearmanr(sh.shift_ratio, sh.faith_delta_scaf_minus_rand).statistic
claim("P1", "Results §3.7", "scaffold/random shift ratio mean", 1.2, round(shift_mean, 1),
      near(shift_mean, 1.2, 0.1))
claim("P1", "Results §3.7", "scaffold/random shift ratio max", 1.7, round(shift_max, 1),
      near(shift_max, 1.7, 0.05))
claim("P1", "Results §3.7", "shift vs faithfulness-Δ Spearman", -0.11, round(shift_rho, 2),
      near(shift_rho, -0.11, 0.05))

# PAINS/BRENK alert overlap
alert = pd.read_csv(os.path.join(RES, "r4_alert_overlap_summary.csv")).set_index("endpoint")
for ep, claimed in [("AMES", (0.075, 0.005, 0.148)),
                    ("hERG", (0.025, -0.066, 0.119)),
                    ("DILI", (-0.006, -0.098, 0.082))]:
    obs = (alert.loc[ep].delta, alert.loc[ep].delta_lo, alert.loc[ep].delta_hi)
    ok = all(near(a, b, 0.005) for a, b in zip(claimed, obs))
    claim("P1", "Results §3.8", f"alert-overlap Δ for {ep}",
          f"{claimed[0]:+.3f} [{claimed[1]:+.3f}, {claimed[2]:+.3f}]",
          f"{obs[0]:+.3f} [{obs[1]:+.3f}, {obs[2]:+.3f}]", ok)

# GNN extension
gnn = pd.read_csv(os.path.join(RES, "gnn_extension.csv"))
claim("P1", "Results §3.9", "GIN test AUROC range", "0.79-0.86",
      f"{gnn.test_auroc.min():.2f}-{gnn.test_auroc.max():.2f}",
      abs(gnn.test_auroc.min() - 0.79) < 0.02 and abs(gnn.test_auroc.max() - 0.86) < 0.02)
claim("P1", "Results §3.9", "GIN beat-null endpoints", "3/4",
      f"{int(gnn.beats_null.sum())}/{len(gnn)}",
      int(gnn.beats_null.sum()) == 3 and len(gnn) == 4)
claim("P1", "Results §3.9", "GIN sanity sim range", "0.22-0.28",
      f"{gnn.sanity_sim.min():.2f}-{gnn.sanity_sim.max():.2f}",
      abs(gnn.sanity_sim.min() - 0.22) < 0.01 and abs(gnn.sanity_sim.max() - 0.28) < 0.01)

# ---------- Paper 2 ----------
ablation = pd.read_csv(os.path.join(RES, "p9d_ablation.csv")).iloc[0]
keys = {"confidence-only [conf,margin]": "conf-only", "full (7 features)": "full",
        "no-attr-magnitude (drop attr_l2)": "drop-attr",
        "attribution-only [attr_l2,attr_entropy]": "attr-only"}
for k, label in keys.items():
    claimed = {"conf-only": 0.680, "full": 0.694, "drop-attr": 0.699, "attr-only": 0.536}[label]
    claim("P2", "Results §3.3 ablation", f"{label} AUROC", claimed, round(ablation[k], 3),
          near(ablation[k], claimed, 0.003))
claim("P2", "Results §3.3 ablation", "full - confidence-only Δ", 0.014,
      round(ablation["full (7 features)"] - ablation["confidence-only [conf,margin]"], 3),
      near(ablation["full (7 features)"] - ablation["confidence-only [conf,margin]"], 0.014, 0.001))
claim("P2", "Results §3.3 ablation", "attr_l2 partial-vs-faith corr",
      0.22, round(ablation.attr_faith_partial, 2), near(ablation.attr_faith_partial, 0.22, 0.01))

# C2 abstention utility
util = pd.read_csv(os.path.join(RES, "p9b_utility.csv"))
lift_50 = util[util.coverage == 0.5].lift.iloc[0]
lift_30 = util[util.coverage == 0.3].lift.iloc[0]
claim("P2", "Results §3.4 abstention", "lift @ 50% coverage", 0.114, round(float(lift_50), 3),
      near(lift_50, 0.114, 0.005))
claim("P2", "Results §3.4 abstention", "lift @ 30% coverage", 0.205, round(float(lift_30), 3),
      near(lift_30, 0.205, 0.01))

# C3 cross-omics
tr = pd.read_csv(os.path.join(RES, "p9c_transcriptomics_verdict.csv")).iloc[0]
sq = pd.read_csv(os.path.join(RES, "p9c_sequence_verdict.csv")).iloc[0]
claim("P2", "Results §3.5 transcriptomics", "AUROC [CI]", "0.86 [0.76, 0.94]",
      f"{tr.auc:.2f} [{tr.auc_lo:.2f}, {tr.auc_hi:.2f}]",
      abs(tr.auc - 0.86) < 0.01 and abs(tr.auc_lo - 0.76) < 0.01 and abs(tr.auc_hi - 0.94) < 0.01)
claim("P2", "Results §3.5 sequence", "AUROC [CI]", "0.81 [0.76, 0.85]",
      f"{sq.auc:.2f} [{sq.auc_lo:.2f}, {sq.auc_hi:.2f}]",
      abs(sq.auc - 0.81) < 0.01 and abs(sq.auc_lo - 0.76) < 0.01 and abs(sq.auc_hi - 0.85) < 0.01)

# ---------- Paper 3 ----------
p12 = pd.read_csv(os.path.join(RES, "p12_main.csv"))
n_pairs = len(p12)
claim("P3", "Methods §3 intro", "n (mol, bit) pairs", 3434, n_pairs, n_pairs == 3434)

rng = np.random.default_rng(0)
def boot_ci(v, n=2000):
    v = np.asarray(v); b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

for col, name, claimed in [("occ_auc", "occlusion", (0.705, 0.697, 0.714)),
                           ("ig_auc", "IG", (0.497, 0.485, 0.508)),
                           ("rand_auc", "random", (0.492, 0.482, 0.502))]:
    m_, lo, hi = boot_ci(p12[col])
    ok = abs(m_ - claimed[0]) < 0.005 and abs(lo - claimed[1]) < 0.005 and abs(hi - claimed[2]) < 0.005
    claim("P3", "Results §3.1 D1", f"{name} AUROC [CI]",
          f"{claimed[0]:.3f} [{claimed[1]:.3f}, {claimed[2]:.3f}]",
          f"{m_:.3f} [{lo:.3f}, {hi:.3f}]", ok)

# D2 small-n faithfulness numbers
faith = {"occlusion": p12.occ_faith.mean(), "IG": p12.ig_faith.mean(), "random": p12.rand_faith.mean()}
claim("P3", "Results §3.2", "occlusion faithfulness", 0.275, round(faith["occlusion"], 3),
      near(faith["occlusion"], 0.275, 0.005))
claim("P3", "Results §3.2", "IG faithfulness", 0.225, round(faith["IG"], 3),
      near(faith["IG"], 0.225, 0.005))
claim("P3", "Results §3.2", "random faithfulness", 0.129, round(faith["random"], 3),
      near(faith["random"], 0.129, 0.005))
rec = [p12.occ_auc.mean(), p12.ig_auc.mean(), p12.rand_auc.mean()]
fai = [p12.occ_faith.mean(), p12.ig_faith.mean(), p12.rand_faith.mean()]
rho3 = spearmanr(rec, fai).statistic
claim("P3", "Results §3.2", "Spearman(recovery, faith) on 3 methods", 1.00, round(rho3, 2),
      near(rho3, 1.0, 0.01))

# D2 extended (6 methods)
ext_v = pd.read_csv(os.path.join(RES, "r4_p3_d2_extended.csv")).set_index("method")
ext_rec = ext_v.recovery; ext_fai = ext_v.faithfulness
rho6 = spearmanr(ext_rec.values, ext_fai.values).statistic
claim("P3", "Results §3.3 D2-ext", "Spearman(recovery, faith) on 6 methods", -0.086,
      round(rho6, 3), near(rho6, -0.086, 0.01))

# the table rows
ext_full = pd.read_csv(os.path.join(RES, "r4_p3_methods.csv"))
n_ext = len(ext_full)
claim("P3", "Results §3.3", "n (mol, bit) pairs in extended run", 2184, n_ext, n_ext == 2184)
for m_name, m_col, expected_auc, expected_faith in [
    ("occlusion", "occlusion", 0.551, 0.181),
    ("IG", "IG", 0.488, 0.164),
    ("gradient × input", "grad*input", 0.476, 0.169),
    ("SmoothGrad", "smoothgrad", 0.475, 0.178),
    ("saliency", "grad", 0.472, 0.177),
    ("random", "random", 0.501, 0.085),
]:
    auc_col = f"{m_col}_auc"
    f_col = f"{m_col}_faith"
    obs_auc = round(ext_full[auc_col].mean(), 3)
    obs_faith = round(ext_full[f_col].mean(), 3)
    claim("P3", "Results §3.3 table", f"{m_name} recovery AUROC",
          expected_auc, obs_auc, near(obs_auc, expected_auc, 0.005))
    claim("P3", "Results §3.3 table", f"{m_name} mask-faithfulness",
          expected_faith, obs_faith, near(obs_faith, expected_faith, 0.005))

# D3 — recovery vs molecule properties
for col, claimed in [("n_atoms", -0.01), ("n_gt", -0.08), ("frac_gt", -0.06)]:
    obs = round(spearmanr(p12[col], p12.occ_auc).statistic, 2)
    claim("P3", "Results §3.4 D3", f"Spearman(occ recovery, {col})", claimed, obs,
          near(claimed, obs, 0.02))

# ---------- output ----------
lines = ["# Skeptic claim audit -- every number in the three manuscripts vs source CSVs", "",
         "Tolerance: +/-0.005 for AUROC/probabilities; +/-0.02 for percentages; +/-0.05 for "
         "Spearmans where appropriate.", "",
         "| Paper | Where | Claim | Manuscript | Observed | Pass | Note |",
         "|---|---|---|---|---|---|---|"]
fails = 0
for paper, where, name, c, o, ok, note in claims:
    mark = "PASS" if ok else "FAIL"
    if not ok:
        fails += 1
    lines.append(f"| {paper} | {where} | {name} | `{c}` | `{o}` | {mark} | {note} |")
lines.append(f"\n**Total claims checked: {len(claims)}. Mismatches: {fails}.**")
out = os.path.join(os.path.dirname(__file__), "..", "docs", "claim_audit.md")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"Wrote {len(claims)} claim checks to {out} ({fails} mismatches)")
