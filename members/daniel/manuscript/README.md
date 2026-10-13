# Manuscript — COMPASS Phases I & II

Elsevier `elsarticle` manuscript describing the Phase I (bid-line covering) and
Phase II (shrinkage: qualifications, training, vacation) solution implemented in
`members/daniel/`.

## Files
- `document.tex` — the manuscript (`\documentclass[preprint,12pt]{elsarticle}`).
- `references.bib` — BibTeX; verified literature-review DOIs plus Gurobi/Xpress
  documentation and the competition problem statements / pseudocode / data spec.

## Build
Requires a TeX distribution with `elsarticle` (present at
`/usr/local/texlive/2026/texmf-dist/tex/latex/elsarticle/`) and BibTeX.

```bash
# from this directory
latexmk -pdf document.tex        # full compile incl. bibliography
# or manually
pdflatex document.tex && bibtex document && pdflatex document.tex && pdflatex document.tex
```

Build artifacts (`*.aux`, `*.bbl`, `*.log`, `*.pdf`, ...) are git-ignored.

## Numbers
The results in Section 7 come from real runs of the pipeline:

```bash
# random baseline (default; writes to outputs/)
python3 members/daniel/main.py --random --score --reconcile

# exact BLT with each solver (Xpress reads its licence from $XPRESS)
python3 members/daniel/main.py --exact-mip --solver gurobi --output-data-dir outputs/exact_gurobi --score --reconcile --log
XPRESS="/path/to/dir/with/xpauth.xpr" python3 members/daniel/main.py --exact-mip --solver xpress --output-data-dir outputs/exact_xpress --score --reconcile --log
```

| Training layer | Supply | Under | Over |
|---|---|---|---|
| Random | 83,885 | 4,503 | 6,136 |
| BLT — Gurobi | 83,888 | 4,209 | 5,845 |
| BLT — Xpress | 83,885 | 4,174 | 5,807 |

Regenerate these if the upstream roster (`outputs/Supply_Regular.csv` et al.) changes.
