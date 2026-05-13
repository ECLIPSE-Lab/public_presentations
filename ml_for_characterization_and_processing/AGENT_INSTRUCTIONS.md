# Agent Instructions: Slides for ML-PC

## Project Overview

This directory contains the RevealJS presentation slides for the university lecture "Machine Learning in Materials Processing & Characterization" (ML-PC) for SS26 at FAU Erlangen-Nürnberg.

The slides are built using Quarto and RevealJS, with a 1920x1080 resolution and a custom Eclipse Lab theme.

## Directory Structure

Each `unitXX_topic` folder is a self-contained Quarto project for one lecture unit (90 minutes, ~50 slides).

### Typical Unit Folder Contents
```
unitXX_topic/
├── 01_intro.qmd              # Main RevealJS slide deck
├── _metadata.yml             # Format configuration
├── custom.scss / custom.css  # Slide styling
├── title-slide.html / .scss  # Custom title slide template
├── eclipse_logo_small.png    # Footer logo
├── ref.bib                   # Unit-specific bibliography
├── unitN_plan.md             # Lecture plan (90-min structure)
└── unitN_content_50slides.md # Detailed slide content drafting file
```

## Slide Deck Conventions

### YAML Header
```yaml
title: |
  Machine Learning in Materials Processing & Characterization<br>Unit N: [Topic]
bibliography: ref.bib
author:
  - name: Prof. Dr. Philipp Pelz
    affiliation:
      - FAU Erlangen-Nürnberg
format:
  revealjs:
    width: 1920
    height: 1080
    template-partials:
      - title-slide.html
    css: custom.css
    theme: custom.scss
    slide-number: c/t
    logo: "eclipse_logo_small.png"
    footer: "© Philipp Pelz - Machine Learning in Materials Processing & Characterization"
```

### Slide Formatting
- Use `## NN. Slide Title` for numbered slides.
- Use `::: {.columns}` and `::: {.column width="50%"}` for side-by-side content.
- Cite using `[@citation_key]` linked to `ref.bib`.
- Ensure all images are placed in the unit folder or a subfolder (e.g., `images/`).

## Curriculum Mapping

Post-realignment (2026-05-13): ML-PC W7 (26.05.2026) is cancelled (public
holiday). The original Time-series unit is demoted to optional self-study
(`unit07_time_series_supplementary/`, marked `published: false` in
`units.yml`) and the schedule shifts down by one. Folder index now matches
delivered-lecture index.

- unit01_intro -> Week 1: What makes materials data special?
- unit02_physics_of_data -> Week 2: Physics of data formation
- unit03_data_quality -> Week 3: Data quality, labels, and leakage
- unit04_microstructure_representations -> Week 4: From classical microstructure metrics to learned representations
- unit05_unsupervised_learning -> Week 5: Unsupervised methods for materials — clustering & autoencoders
- unit06_transfer_learning -> Week 6: Data scarcity & transfer learning
- unit07_time_series_supplementary -> (not delivered; optional self-study after W7 cancellation)
- unit07_generalization_robustness -> Week 7 (delivered W8): Generalization, robustness, and process windows
- unit08_inverse_problems -> Week 8 (delivered W9): Inverse problems and process maps
- unit09_characterization_signals -> Week 9 (delivered W10): ML for characterization signals
- unit09b_transformers_for_materials -> Companion deck within Week 9: Transformers for materials (ViT, Flash Attention, Mamba)
- unit10_automation -> Week 10 (delivered W11): Automation in microscopy and characterization
- unit11_uncertainty_gp -> Week 11 (delivered W12): Uncertainty-aware regression & Gaussian Processes
- unit12_pinns -> Week 12 (delivered W13): Physics-informed and constrained ML
- unit13_reflection -> Week 13 (delivered W14): Integration, limits, and reflection

## Development Workflow
1.  Check `unitN_plan.md` for the pedagogical structure.
2.  Use `unitN_content_50slides.md` to draft detailed slide text and image placeholders.
3.  Populate `01_intro.qmd` with the content.
4.  Render and verify layout (1920x1080).
