# Mahoraga Paper

Contents of this folder:

- `mahoraga_baseline_paper.tex`: main LaTeX paper source.
- `references.bib`: BibTeX bibliography.
- `figures/`: figures copied from the audited official baseline outputs.

Factual sources used:

- Frozen official baseline: `D:/QuantMahoraga/baseline/mahoraga14_3_baseline`
- Official documentation: `D:/QuantMahoraga/baseline/mahoraga14_3_baseline/docs`
- Official outputs: `D:/QuantMahoraga/baseline/mahoraga14_3_baseline/outputs`
- Official audits: `D:/QuantMahoraga/baseline/mahoraga14_3_baseline/audit`

Recommended compilation:

```powershell
cd D:\QuantMahoraga\paper
latexmk -pdf mahoraga_baseline_paper.tex
```

Alternative with `pdflatex` + `bibtex`:

```powershell
cd D:\QuantMahoraga\paper
pdflatex mahoraga_baseline_paper.tex
bibtex mahoraga_baseline_paper
pdflatex mahoraga_baseline_paper.tex
pdflatex mahoraga_baseline_paper.tex
```

Note about this environment:

- In the review performed on `2026-05-04`, `pdflatex`, `xelatex`, and `latexmk` were not available, so the PDF could not be compiled locally.
- The content is prepared for compilation once a working TeX distribution is available on the system.
