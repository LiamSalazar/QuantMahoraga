# Paper de Mahoraga

Contenido de esta carpeta:

- `mahoraga_baseline_paper.tex`: fuente principal del paper en LaTeX.
- `references.bib`: bibliografia BibTeX.
- `figures/`: figuras copiadas desde los outputs oficiales auditados del baseline.

Fuente factual usada:

- Baseline oficial congelado: `D:/QuantMahoraga/baseline/mahoraga14_3_baseline`
- Documentacion oficial: `D:/QuantMahoraga/baseline/mahoraga14_3_baseline/docs`
- Outputs oficiales: `D:/QuantMahoraga/baseline/mahoraga14_3_baseline/outputs`
- Auditorias oficiales: `D:/QuantMahoraga/baseline/mahoraga14_3_baseline/audit`

Compilacion recomendada:

```powershell
cd D:\QuantMahoraga\paper
latexmk -pdf mahoraga_baseline_paper.tex
```

Alternativa con `pdflatex` + `bibtex`:

```powershell
cd D:\QuantMahoraga\paper
pdflatex mahoraga_baseline_paper.tex
bibtex mahoraga_baseline_paper
pdflatex mahoraga_baseline_paper.tex
pdflatex mahoraga_baseline_paper.tex
```

Nota sobre este entorno:

- En la revision realizada el `2026-05-04` no estaban disponibles `pdflatex`, `xelatex` ni `latexmk`, por lo que no se pudo compilar el PDF aqui.
- El contenido se dejo listo para compilar en cuanto exista una distribucion TeX funcional en el sistema.
