# Reproducing the study

One-command-ish reproduction of Paper 1 (the reliability audit). Deterministic
(fixed seeds); intermediate artifacts are cached so reruns are cheap.

## 1. Environment
```bash
git clone <this repo> && cd xai-pharma-study
python -m venv .venv && source .venv/bin/activate   # Python 3.11 recommended
pip install -r requirements.txt
pip install -e ../xai-eval-harness                  # metric implementations
# If PyTDC's numpy<2 pin conflicts with your env (e.g. Python 3.13+):
#   pip install --no-deps PyTDC && pip install fuzzywuzzy python-Levenshtein requests
```

## 2. Run the pipeline
```bash
export PYTHONPATH=../xai-eval-harness/src:src      # Windows: use ';' separator
python src/train.py                 # P3: trains 144 models -> results/performance.csv
python src/experiment.py descriptors# P4a: descriptors reliability matrix
python src/experiment.py ecfp       # P4b: ECFP reliability matrix -> results/reliability.csv
python src/analyze.py               # P5: H1-H5 tests + figures -> results/analysis, results/figures
python src/robustness_lime.py       # P7: LIME-budget robustness check
```

Total wall-clock on a CPU laptop: ~2 h (TDC downloads + 144 models + ~142 attribution
cells). `experiment.py` writes `results/reliability.csv` incrementally and skips
already-computed cells, so it is safe to interrupt and resume.

## 3. Outputs
| File | Produced by | Contents |
| --- | --- | --- |
| `results/performance.csv` | train.py | per-model predictive performance + floor flag |
| `results/reliability.csv` | experiment.py | per (cell × method) faithfulness/stability/sanity + CIs |
| `results/agreement.csv` | experiment.py | pairwise cross-method agreement |
| `results/analysis/hypotheses_summary.md` | analyze.py | the H1–H5 verdicts |
| `results/figures/*.png` | analyze.py | manuscript figures |
| `results/robustness_lime.csv` | robustness_lime.py | LIME-budget sensitivity of agreement |

## 4. Provenance
Preregistration: `docs/01_study_design_preregistration.md` (frozen pre-results).
Every deviation: `docs/deviations.md`. Phase-by-phase history: `docs/progress_log.md`.
