# Skeptic claim audit -- every number in the three manuscripts vs source CSVs

Tolerance: +/-0.005 for AUROC/probabilities; +/-0.02 for percentages; +/-0.05 for Spearmans where appropriate.

| Paper | Where | Claim | Manuscript | Observed | Pass | Note |
|---|---|---|---|---|---|---|
| P1 | Methods §2.2 | total models trained | `144` | `144` | PASS |  |
| P1 | Methods §2.2 | models clearing floor | `142` | `142` | PASS |  |
| P1 | Results §3.2 | IG fail count / total | `0 / 48` | `0 / 48` | PASS |  |
| P1 | Results §3.2 | LIME fail rate (raw) | `15%` | `15%` | PASS | manuscript reports 15% (21/142 = 14.8%) |
| P1 | Results §3.2 | SHAP fail rate (raw) | `12%` | `12%` | PASS | manuscript reports 12% (11/94 = 11.7%) |
| P1 | Results §3.5 | MLP sanity-fail rate | `25%` | `25%` | PASS | 24/96 = 25.0% |
| P1 | Results §3.5 | tree sanity-fail rate | `39%` | `39%` | PASS | 74/188 = 39.4% |
| P1 | Results §3.4 | LIME-300 mean Spearman (SHAP-LIME) | `0.15` | `0.15` | PASS |  |
| P1 | Results §3.4 | LIME-1000 mean Spearman (SHAP-LIME) | `0.34` | `0.34` | PASS |  |
| P1 | Results §3.4 | LIME-1000 rho for DILI | `0.5` | `0.5` | PASS |  |
| P1 | Results §3.4 | LIME-1000 rho for hERG | `0.42` | `0.42` | PASS |  |
| P1 | Results §3.4 | LIME-1000 rho for AMES | `0.37` | `0.37` | PASS |  |
| P1 | Results §3.4 | LIME-1000 rho for BBB | `0.36` | `0.36` | PASS |  |
| P1 | Results §3.4 | LIME-1000 rho for LD50 | `0.19` | `0.19` | PASS |  |
| P1 | Results §3.4 | LIME-1000 rho for Caco2 | `0.18` | `0.18` | PASS |  |
| P1 | Results §3.7 | primary comprehensiveness vs ROAR Spearman | `0.93` | `0.93` | PASS |  |
| P1 | Results §3.7 | naive score-AOPC vs ROAR Spearman | `-0.5` | `-0.5` | PASS |  |
| P1 | Results §3.7 | multi-seed scaffold-rand Δ median | `-0.01` | `-0.01` | PASS |  |
| P1 | Results §3.7 | multi-seed Wilcoxon p | `0.19` | `0.19` | PASS |  |
| P1 | Results §3.7 | SHAP mask-ref range | `0.33-0.37` | `0.33-0.37` | PASS |  |
| P1 | Results §3.7 | LIME mask-ref range | `0.30-0.32` | `0.30-0.32` | PASS |  |
| P1 | Results §3.7 | random mask-ref range | `0.07-0.09` | `0.07-0.09` | PASS |  |
| P1 | Results §3.7 | scaffold/random shift ratio mean | `1.2` | `1.2` | PASS |  |
| P1 | Results §3.7 | scaffold/random shift ratio max | `1.7` | `1.7` | PASS |  |
| P1 | Results §3.7 | shift vs faithfulness-Δ Spearman | `-0.11` | `-0.11` | PASS |  |
| P1 | Results §3.8 | alert-overlap Δ for AMES | `+0.075 [+0.005, +0.148]` | `+0.075 [+0.005, +0.148]` | PASS |  |
| P1 | Results §3.8 | alert-overlap Δ for hERG | `+0.025 [-0.066, +0.119]` | `+0.025 [-0.066, +0.119]` | PASS |  |
| P1 | Results §3.8 | alert-overlap Δ for DILI | `-0.006 [-0.098, +0.082]` | `-0.006 [-0.098, +0.082]` | PASS |  |
| P1 | Results §3.9 | GIN test AUROC range | `0.79-0.86` | `0.79-0.86` | PASS |  |
| P1 | Results §3.9 | GIN beat-null endpoints | `3/4` | `3/4` | PASS |  |
| P1 | Results §3.9 | GIN sanity sim range | `0.22-0.28` | `0.22-0.28` | PASS |  |
| P2 | Results §3.3 ablation | conf-only AUROC | `0.68` | `0.68` | PASS |  |
| P2 | Results §3.3 ablation | full AUROC | `0.694` | `0.694` | PASS |  |
| P2 | Results §3.3 ablation | drop-attr AUROC | `0.699` | `0.699` | PASS |  |
| P2 | Results §3.3 ablation | attr-only AUROC | `0.536` | `0.536` | PASS |  |
| P2 | Results §3.3 ablation | full - confidence-only Δ | `0.014` | `0.014` | PASS |  |
| P2 | Results §3.3 ablation | attr_l2 partial-vs-faith corr | `0.22` | `0.22` | PASS |  |
| P2 | Results §3.4 abstention | lift @ 50% coverage | `0.114` | `0.114` | PASS |  |
| P2 | Results §3.4 abstention | lift @ 30% coverage | `0.205` | `0.205` | PASS |  |
| P2 | Results §3.5 transcriptomics | AUROC [CI] | `0.86 [0.76, 0.94]` | `0.86 [0.76, 0.95]` | PASS |  |
| P2 | Results §3.5 sequence | AUROC [CI] | `0.81 [0.76, 0.85]` | `0.81 [0.76, 0.85]` | PASS |  |
| P3 | Methods §3 intro | n (mol, bit) pairs | `3434` | `3434` | PASS |  |
| P3 | Results §3.1 D1 | occlusion AUROC [CI] | `0.705 [0.697, 0.714]` | `0.705 [0.697, 0.714]` | PASS |  |
| P3 | Results §3.1 D1 | IG AUROC [CI] | `0.497 [0.485, 0.508]` | `0.497 [0.486, 0.507]` | PASS |  |
| P3 | Results §3.1 D1 | random AUROC [CI] | `0.492 [0.482, 0.502]` | `0.492 [0.482, 0.502]` | PASS |  |
| P3 | Results §3.2 | occlusion faithfulness | `0.275` | `0.275` | PASS |  |
| P3 | Results §3.2 | IG faithfulness | `0.225` | `0.225` | PASS |  |
| P3 | Results §3.2 | random faithfulness | `0.129` | `0.129` | PASS |  |
| P3 | Results §3.2 | Spearman(recovery, faith) on 3 methods | `1.0` | `1.0` | PASS |  |
| P3 | Results §3.3 D2-ext | Spearman(recovery, faith) on 6 methods | `-0.086` | `-0.086` | PASS |  |
| P3 | Results §3.3 | n (mol, bit) pairs in extended run | `2184` | `2184` | PASS |  |
| P3 | Results §3.3 table | occlusion recovery AUROC | `0.551` | `0.551` | PASS |  |
| P3 | Results §3.3 table | occlusion mask-faithfulness | `0.181` | `0.181` | PASS |  |
| P3 | Results §3.3 table | IG recovery AUROC | `0.488` | `0.488` | PASS |  |
| P3 | Results §3.3 table | IG mask-faithfulness | `0.164` | `0.164` | PASS |  |
| P3 | Results §3.3 table | gradient × input recovery AUROC | `0.476` | `0.476` | PASS |  |
| P3 | Results §3.3 table | gradient × input mask-faithfulness | `0.169` | `0.169` | PASS |  |
| P3 | Results §3.3 table | SmoothGrad recovery AUROC | `0.475` | `0.475` | PASS |  |
| P3 | Results §3.3 table | SmoothGrad mask-faithfulness | `0.178` | `0.178` | PASS |  |
| P3 | Results §3.3 table | saliency recovery AUROC | `0.472` | `0.472` | PASS |  |
| P3 | Results §3.3 table | saliency mask-faithfulness | `0.177` | `0.177` | PASS |  |
| P3 | Results §3.3 table | random recovery AUROC | `0.501` | `0.501` | PASS |  |
| P3 | Results §3.3 table | random mask-faithfulness | `0.085` | `0.085` | PASS |  |
| P3 | Results §3.4 D3 | Spearman(occ recovery, n_atoms) | `-0.01` | `-0.01` | PASS |  |
| P3 | Results §3.4 D3 | Spearman(occ recovery, n_gt) | `-0.08` | `-0.08` | PASS |  |
| P3 | Results §3.4 D3 | Spearman(occ recovery, frac_gt) | `-0.06` | `-0.06` | PASS |  |

**Total claims checked: 66. Mismatches: 0.**
