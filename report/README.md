# Six-page report and Overleaf source

`report.pdf` is the compiled English report. `main.tex` is the editable source, and `figures/` contains all four required PNG assets. No external images or bibliography files are needed to compile it.

For Overleaf, upload the supplied Overleaf ZIP as a new project, select `main.tex`, and use **XeLaTeX**. Uploading the whole ZIP preserves the `figures/` paths. If pasting the LaTeX manually instead, also upload the four images into a folder named `figures`.

The author is identified by the GitHub handle `lllingshen`; edit the line below the title if a legal name or student ID is required. The body is exactly six pages in the verified build. Changing fonts, compiler, or layout can affect pagination.

The report preserves the observed failures: fine-tuning improved the small validation AP but increased false positives in the apartment, and YOLOE still made errors. It describes ordinary fine-tuning, not LoRA. The YOLOE experiment used text prompts only.

Figures were redrawn from unchanged source photographs and stored model predictions. Crops are for presentation only; inference used the full images. Green/red outlines in selected-example figures reflect qualitative review and are not a full ground-truth annotation set.
