# bioRxiv submission package — read me first

This folder contains everything you need to submit the three papers to **bioRxiv**
(https://www.biorxiv.org/submit-a-manuscript). The papers are independent submissions sharing
a common code/data substrate.

## What's in here
```
submission/
  manuscript.docx              ← Paper 1 (audit) — main file to upload
  manuscript_paper2.docx       ← Paper 2 (certificate) — main file to upload
  manuscript_paper3.docx       ← Paper 3 (fingerprint distillation) — main file to upload
  figures/                     ← every figure as separate PNG (300 DPI) AND vector PDF
    paper1/                    ← Figures 1–5 for Paper 1
    paper2/                    ← Figures 1–3 for Paper 2
    paper3/                    ← Figures 1–2 for Paper 3
  cover_letter_paper1.md
  cover_letter_paper2.md
  cover_letter_paper3.md
  metadata.md                  ← title, abstract (bioRxiv-trimmed), category, authors, license
  CLAIM_AUDIT.md               ← every numerical claim verified vs source CSVs (66/66 pass)
  SUBMISSION_GUIDE.md          ← this file
```

## Before you submit — fill in these placeholders
The papers were drafted with placeholder author/affiliation/funding metadata. Edit each .docx
(and `metadata.md`) to insert:

1. **Author full name** — currently "Amit Shenoy"; confirm or update co-authors.
2. **Affiliation(s)** — institution, department, address.
3. **Corresponding author email** — likely `shenoy.am@husky.neu.edu`.
4. **ORCID** — if you have one, add to the author block.
5. **Funding statement** — e.g., "No external funding" / grant numbers.
6. **Competing-interests statement** — likely "The authors declare no competing interests."
7. **Acknowledgements** — optional.

These are the only things missing from a clean bioRxiv submission.

## Submission steps (bioRxiv web flow)
For each paper:

1. Log in at https://www.biorxiv.org/submit-a-manuscript.
2. **Type of article:** New result.
3. **Subject category:**
   - Paper 1 → *Bioinformatics* (primary), *Pharmacology and Toxicology* (secondary)
   - Paper 2 → *Bioinformatics* (primary), *Systems Biology* (secondary)
   - Paper 3 → *Bioinformatics* (primary)
4. **Title + abstract** — copy from `metadata.md` (abstract is already bioRxiv-trimmed to fit
   the ~2,800-character limit; if your draft abstract is longer, the trimmed version is the one
   to paste).
5. **Authors** — enter as in the manuscript; mark corresponding author.
6. **License** — recommend **CC-BY 4.0** (Attribution; broadest reuse, standard for preprints).
7. **Upload files:**
   - Main: `manuscript.docx` (or `manuscript_paper2.docx` / `manuscript_paper3.docx`).
   - Figures: from `figures/paperN/` — upload each PNG separately as a Figure file, with
     captions copied from the manuscript's Figures section.
8. **Cover letter:** copy/paste from `cover_letter_paperN.md` if bioRxiv prompts for one
   (optional).
9. **Conflict / funding / data availability statements** — copy from the manuscript or insert
   directly in the bioRxiv form.
10. **Data and code availability:** point to the public GitHub repo
    `https://github.com/xX-its-amit-Xx/xai-pharma-study` and the harness
    `https://github.com/xX-its-amit-Xx/xai-eval-harness`.
11. Preview, confirm, submit. bioRxiv typically posts within 24–48 h after light screening.

## Three papers, one program: how they relate
Treat each paper as a stand-alone submission, but make the cross-references explicit in your
cover letter:

- **Paper 1** is the empirical audit (the instrument validates itself against a hard null).
- **Paper 2** asks whether the audit can go *per-instance* (and answers honestly: only via
  model confidence).
- **Paper 3** introduces an external ground-truth attribution benchmark for GNNs (fingerprint
  distillation) and uses it to expose both a sharp methods finding (gradient attribution fails
  on graph-level outputs) and a real boundary of Paper 1's instrument.

Submit Paper 1 first; once it has a preprint DOI, the other cover letters and reference lists
can cite it. **Or** submit all three simultaneously — bioRxiv allows it — and cross-link by
DOI after the fact.

## Reproducibility
The bioRxiv reviewer (or any reader) can reproduce every number in the manuscripts by:
```
git clone https://github.com/xX-its-amit-Xx/xai-pharma-study
cd xai-pharma-study
# follow REPRODUCE.md
```
The `claim_audit.md` (also copied here as `CLAIM_AUDIT.md`) shows the manuscript-to-data
mapping for every number — 66/66 verified.

## After submission
- Add the bioRxiv DOI to the manuscript headers ("Cite as: 10.1101/...") and re-push the
  GitHub repo so the live record matches the preprint.
- Tag the repo at the submission commit: `git tag biorxiv-submission` and push the tag.
