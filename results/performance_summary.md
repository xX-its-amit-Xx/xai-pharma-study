# P3 model-zoo performance summary

144 models = 12 endpoints x 2 representations x 2 splits x 3 models. Cleared trivial-baseline floor: 142/144.

## Best model per endpoint (scaffold split)
| endpoint | category | task | best repr+model | primary | value |
|---|---|---|---|---|---|
| DILI | toxicity | classification | descriptors+rf | AUROC | 0.920 |
| hERG | toxicity | classification | descriptors+mlp | AUROC | 0.871 |
| AMES | toxicity | classification | descriptors+hgb | AUROC | 0.861 |
| LD50 | toxicity | regression | descriptors+hgb | Spearman | 0.529 |
| Caco2 | absorption | regression | descriptors+hgb | Spearman | 0.738 |
| HIA | absorption | classification | descriptors+rf | AUROC | 0.983 |
| BBB | distribution | classification | descriptors+rf | AUROC | 0.918 |
| VDss | distribution | regression | descriptors+rf | Spearman | 0.555 |
| CYP2C9-Sub | metabolism | classification | ecfp+mlp | AUROC | 0.691 |
| CYP3A4-Sub | metabolism | classification | ecfp+rf | AUROC | 0.686 |
| HalfLife | excretion | regression | descriptors+mlp | Spearman | 0.513 |
| CL-Hepa | excretion | regression | ecfp+rf | Spearman | 0.418 |

## Scaffold vs random (mean over models)
| split | mean AUROC (class) | mean Spearman (reg) |
|---|---|---|
| scaffold | 0.802 | 0.425 |
| random | 0.823 | 0.490 |

## Below-floor cells (excluded from reliability claims)
- CYP2C9-Sub ecfp scaffold rf: AUROC=0.485
- HalfLife ecfp scaffold hgb: Spearman=0.085