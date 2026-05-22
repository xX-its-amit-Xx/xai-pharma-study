# P5 analysis — preregistered hypothesis tests

Reliability cells: 426 (2 reps, 12 endpoints, methods ['ig', 'lime', 'random', 'shap'])

## H1 — faithfulness over the null
- non-random method-cells: 284
- fraction NOT beating null (BH-FDR q=0.05): **0.12** (prereg threshold for H1 support: >=0.20 -> FALSIFIED)
- raw (uncorrected) not-beating rate: 0.11
- by method (fraction not beating null, FDR):
    - ig: 0.00 (n=48)
    - lime: 0.15 (n=142)
    - shap: 0.12 (n=94)

## H2 — reliability degrades out-of-distribution (scaffold vs random)
- faith_mean: median scaffold-random delta = -0.013, Wilcoxon p=0.1263 (no sig. degradation) [lower scaffold = worse]
- stab_mean: median scaffold-random delta = +0.000, Wilcoxon p=0.8485 (no sig. degradation) [higher scaffold = worse]

## H3 — explainer disagreement
- median pairwise Spearman (non-random pairs): **0.054** (prereg descriptive threshold <0.5 -> low agreement)
- by representation (median Spearman / median top-k Jaccard):
    - descriptors: Spearman 0.115, Jaccard 0.096
    - ecfp: Spearman 0.033, Jaccard 0.059
- by method-pair (median Spearman, descriptors only):
    - ig vs lime: 0.146 (n=24)
    - shap vs lime: 0.094 (n=48)
- toxicity vs non-toxicity agreement: tox median 0.068 vs 0.053, Mann-Whitney p=0.240

## H4 — model-randomization sanity failures (split by model class)
- mlp(true-reinit): fail rate 0.25 (n=96) (>=0.15 -> support)
- tree(label-perm): fail rate 0.39 (n=188) (>=0.15 -> support)
  NOTE: tree fail rate uses the label-permutation analogue (limitation D-prereg §9); the MLP rate is the true Adebayo weight-reinitialization test.

## H5 — representation vs method (variance explained, eta^2)
- faith_mean: eta^2 representation=0.010, method=0.165, model=0.045, endpoint=0.535
- stab_mean: eta^2 representation=0.364, method=0.011, model=0.012, endpoint=0.098
- sanity_sim: eta^2 representation=0.018, method=0.832, model=0.053, endpoint=0.010