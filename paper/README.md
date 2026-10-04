# Paper source

**MMPostTrainBench: Benchmarking Autonomous Research for Multimodal Post-Training**

- [Download the original arXiv source ZIP](MMPostTrainBench_arxiv_source_20261003.zip)
- [Main LaTeX document](arxiv.tex)
- [Sections and tables](mile/)

The 27 source files are reproduced unchanged from the author-provided package
`MMPostTrainBench_arxiv_source_20261003.zip`. The archive is also preserved unchanged
for download. The entry point is `arxiv.tex`; figures, bibliography files, and
bundled style files are included.

With a suitable TeX installation, compile from this directory:

```sh
cd paper
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape arxiv.tex
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape arxiv.tex
```

The supplied `arxiv.bbl` contains the bibliography. File references were checked
for this upload; compilation has not been rerun in the upload environment.
