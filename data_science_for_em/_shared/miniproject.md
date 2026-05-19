# DSEM Miniproject — Definition & Rubric

**Weighting:** 40% of final grade  
**Workload:** ≈20 h (done entirely off lecture time)  
**Scope:** Individual or pairs (pairs must clearly divide contributions in the notebook).

---

## Overview

The miniproject demonstrates that you can apply the full data-science pipeline taught in this course to a real (or clearly-labelled synthetic) electron-microscopy or materials-characterisation dataset. The pipeline must include:

1. **Data** — describe provenance, size, format, any preprocessing steps.
2. **Model** — justify your choice; document hyperparameter decisions.
3. **Uncertainty** — provide calibrated error bars, a Bayesian posterior, or at minimum a held-out validation set with honest reporting of generalisation performance.
4. **Explainability** — at least one interpretability step (feature importance, saliency map, latent-space visualisation, sensitivity analysis, …) that answers a materials-science question.

The pipeline must be fully **reproducible**: a grader running `jupyter nbconvert --to notebook --execute your_notebook.ipynb` on a standard university Python environment must produce the same figures and numbers.

---

## Timeline

| Milestone | When | How |
|-----------|------|-----|
| Briefed + topic selection open | Week 1 lecture | — |
| **Proposal check-in** (≤1 page: dataset, task, planned pipeline) | ≈ Week 6 (off lecture time — submit to course email) | Written submission |
| **Progress check-in** (draft notebook, at least data + model sections complete) | ≈ Week 10 (off lecture time — submit to course email) | Written submission |
| **Final submission** | Exam period (exact date announced in Week 1) | Notebook + report PDF |

All check-ins are off lecture time (email/Moodle upload). There are no in-class presentation slots for the miniproject.

---

## Dataset / Task Menu

Choose **one** of the four options below. If you have a compelling alternative from your own research, email the instructor by the Week 6 deadline for approval.

---

### Option A — STEM/SEM Image Segmentation

**Task:** Segment grain boundaries or nanoparticles from real or synthetic microscopy images; report IoU, Dice, and how model confidence correlates with image quality.

**Suggested datasets / sources:**
- Synthetic Voronoi microstructures with perfect grain-boundary masks (generate in-notebook with `scipy.spatial.Voronoi` + Gaussian blur + Poisson noise — zero download required, explicitly labelled synthetic).
- NFFA-EUROPE open TEM/SEM datasets (requires an NFFA user account — use the synthetic fallback if access is unavailable): <https://www.nffa.eu/apply/data/>.
- Any SEM grain-boundary dataset you can export from your department's microscope (include a data-sharing statement).

**Deliverable:** A segmentation notebook (data generation or loading → lightweight CNN (≤3 conv layers) or a pretrained backbone (e.g. ResNet encoder) → IoU vs noise level curve → at least one Grad-CAM or saliency map on a failure case). A U-Net is an optional stretch goal for pairs with GPU access.

---

### Option B — Spectral Denoising & Phase Clustering

**Task:** Denoise low-count EELS, EDS, or Raman/XRD spectral stacks with PCA or a small autoencoder, then cluster the latent space into chemical phases; validate with a known reference spectrum.

**Suggested datasets / sources:**
- Synthetic low-count EELS spectra: Poisson-noisy Gaussian peaks at known positions (generate in-notebook — zero download required, explicitly labelled synthetic).
- `hyperspy` example datasets (installed in course environment): `hs.datasets.example_signals` (e.g. `hs.datasets.example_signals.EDS_TEM_Spectrum()`).
- Public EELS Atlas spectra from <https://eelsdb.eu> for reference validation.

**Deliverable:** A denoising + clustering notebook (spectra loading or generation → PCA/autoencoder → latent-space t-SNE/UMAP → phase map → comparison with ground-truth positions; uncertainty: reconstruction error distribution per phase).

---

### Option C — Materials Property Regression with Honest Validation

**Task:** Predict a scalar materials property (e.g. hardness, bandgap, corrosion rate) from tabular composition / processing descriptors; rigorously quantify aleatoric + epistemic uncertainty and flag out-of-distribution inputs.

**Suggested datasets / sources:**
- `matminer` datasets (`pip install matminer`): `from matminer.datasets import load_dataset; load_dataset("elastic_tensor_2015")` or similar; see <https://hackingmaterials.lbl.gov/matminer/dataset_summary.html>.
- Synthetic composition-property data: linear model + heteroscedastic noise (generate in-notebook — explicitly labelled synthetic, useful for benchmarking your uncertainty estimates against the known ground truth).
- Any tabular dataset from your group's experiments with ≥80 samples.

**Deliverable:** A regression notebook (EDA → feature engineering → cross-validated model (GP or gradient-boosted tree) → calibration plot (reliability diagram) → SHAP feature-importance plot answering "which descriptor matters most and why?").

---

### Option D — Small Imaging Inverse Problem

**Task:** Solve a 2-D deblurring problem or a limited-angle tomographic reconstruction on a provided or synthetic phantom; study how regularisation strength controls the bias–variance trade-off.

**Suggested datasets / sources:**
- Synthetic disc/bar phantom (generate in-notebook with `skimage.draw`; add Gaussian PSF blur and Poisson noise — zero download required, explicitly labelled synthetic). This is the recommended zero-download fallback and sufficient for full marks.
- Real electron-tomography tilt series from the EMPIAR database: <https://www.ebi.ac.uk/empiar/> (browse for suitable cryo-EM or STEM tomography entries; check file size before downloading).

**Deliverable:** A regularisation-study notebook (phantom generation or loading → forward model → Tikhonov / TV regularisation at ≥5 λ values → SSIM/PSNR vs λ curve → at least one uncertainty map from a Bayesian or ensemble approach; explain which λ a materials scientist should pick and why).

---

## Deliverables

1. **Notebook** (`miniproject_<surname>.ipynb`): self-contained, fully executable. All figures must be generated inside the notebook. Target runtime: ≤ 5 min on a laptop CPU for Options B, C, and D; image-segmentation submissions (Option A) using a small CNN should target ≤ 10 min (or use a pretrained backbone / Google Colab GPU).
2. **Report** (`miniproject_<surname>.pdf`): ≈4–6 pages (A4, ≥11 pt font, figures included in the page count). Sections: Introduction / Dataset & Preprocessing / Methods / Results & Uncertainty / Explainability / Conclusion. References in any consistent style.

Both files submitted as a single ZIP to Moodle by the deadline.

---

## Grading Rubric

| Criterion | Weight | Excellent | Satisfactory | Insufficient |
|-----------|--------|-----------|--------------|--------------|
| **Problem framing** (clear scientific question, justified dataset/task choice) | 15% | Crisp question, strong motivation tied to EM/materials context | Question stated but motivation thin | No clear question; dataset chosen arbitrarily |
| **Methodology** (model choice, implementation, reproducibility) | 25% | Appropriate model, well-motivated hyperparameters, notebook runs end-to-end | Reasonable model, minor reproducibility issues | Wrong model for task; notebook does not execute |
| **Uncertainty & validation** (calibrated errors, honest held-out performance) | 20% | Calibrated uncertainty; reliability diagram or credible intervals; no data leakage | Validation set used; some uncertainty quantification attempted | Only training-set metrics; no uncertainty |
| **Explainability** (interpretability step linked to a materials insight) | 15% | Saliency/SHAP/latent-space plot with a concrete materials-science conclusion | Visualisation present but conclusion superficial | No explainability step |
| **Reproducibility** (notebook executes without modification; data access documented) | 15% | One-command execution; synthetic fallback or data download script provided | Minor manual steps required | Notebook fails or data inaccessible |
| **Report quality** (clarity, structure, figures, references) | 10% | Well-structured, publication-quality figures, proper references | Readable; some figures or references missing | Hard to follow; no references |
| **Total** | **100%** | | | |
