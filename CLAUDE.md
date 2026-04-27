# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo purpose

Public Quarto site that hosts RevealJS slide decks for ECLIPSE Lab (FAU Erlangen-Nürnberg). Five content trees: `data_science_for_em/`, `mathematical_foundations_of_ai_and_ml/`, `materials_genomics/`, `ml_for_characterization_and_processing/`, `conference_talks/`. Each lecture *unit* under those trees is a self-contained Quarto sub-project rendered into a single deck. There is no application code — output of `quarto render` is the deliverable.

## Build / preview

- Python deps live in `.venv/`, declared in both `requirements.txt` (pip, used in CI) and `pyproject.toml` + `uv.lock` (uv, used locally). Install with `pip install -r requirements.txt` or `uv sync`.
- Render the whole site: `quarto render` from repo root. Output → `_site/`.
- Preview a single deck (faster):
  - Linux/macOS: `QUARTO_PYTHON=.venv/bin/python quarto preview <unit_path>/<deck>.qmd --no-browser --no-watch-inputs`
  - Windows: `run_quarto.bat` (currently hardcoded to `conference_talks/2025_MC/template.qmd`; edit before reuse).
- Deployment: pushes to `main` trigger `.github/workflows/quarto-deploy.yml`, which runs `quarto publish gh-pages`. The `static.yml` workflow then publishes the `gh-pages` branch to GitHub Pages. Don't run `quarto publish` locally — let CI own `gh-pages`.
- Top-level `_quarto.yml` sets `execute.enabled: false`, so notebook/Python cells do **not** run during site render. Individual decks override this in their YAML header when they need executed code (see `data_science_for_em/01_intro/template.qmd`).

## Slide deck anatomy

Each unit folder is structured to be Quarto-renderable on its own:

```
<unit>/
├── 01_intro.qmd              # main deck (most courses) — OR template.qmd (DSEM, conf talks)
├── _metadata.yml             # per-unit RevealJS overrides (menu/progress/search)
├── _quarto.yml               # OPTIONAL — only some units (e.g. DSEM/01_intro) override site config
├── custom.scss / custom.css  # Eclipse Lab theming
├── title-slide.html / .scss  # custom title-slide partial
├── eclipse_logo_small.png    # required by `logo:` in YAML
├── ref.bib                   # per-unit bibliography, cited via `[@key]`
└── images/, img/, *.png      # local assets
```

Standard YAML header (1920×1080, custom theme, footer with course name) — match the existing pattern in neighboring units rather than inventing one. See `ml_for_characterization_and_processing/AGENT_INSTRUCTIONS.md` for the canonical ML-PC header and `mathematical_foundations_of_ai_and_ml/01_learning_vs_data_analysis/01_intro.qmd` for the MFML pattern.

**Naming conventions differ between trees** — don't normalize them:
- `data_science_for_em/` and `conference_talks/` → unit folders contain `template.qmd`.
- `mathematical_foundations_of_ai_and_ml/`, `materials_genomics/` → unit folders contain `01_intro.qmd`.
- `ml_for_characterization_and_processing/` → folders prefixed `unitXX_` (not `XX_`); deck is `01_intro.qmd`.

The published URLs in `README.md` are the authoritative slug map; if you rename a folder, update `README.md` too.

## index.qmd is generated

`index.qmd` is rebuilt by `rebuild_index.py`, which scans the unit folders, reads each deck's YAML `title:`, and emits styled HTML cards. Do **not** hand-edit `index.qmd` for routine link changes — regenerate. Caveats before running:
- `base_dir` in `rebuild_index.py` is hardcoded to a Windows path (`c:\Users\braun\...`). Edit it to the current working dir before running on Linux.
- `update_index_with_notebooks.py` likewise has a hardcoded Linux gdrive path and *appends* a "Code Examples" section to whatever `index.qmd` exists. It is run after `rebuild_index.py`, not in place of it.

## Three-course teaching triad (SS26)

`mathematical_foundations_of_ai_and_ml/` (MFML), `materials_genomics/` (MG), and `ml_for_characterization_and_processing/` (ML-PC) are taught as a coordinated triad. MFML provides notation/foundations consumed by the other two. When editing one, check whether the change affects shared notation or topic boundaries. Realignment history for MG lives in `materials_genomics/REALIGNMENT_OLD_TO_NEW_MAPPING.md` and `BOOK_CHAPTER_MAPPING_UNITS3_13.md`; MG units 3–13 were renumbered to match the upstream `MaterialsGenomics/index.qmd` — preserve those slugs.

## Companion repo for executable code

PyTorch teaching notebooks for the triad live in a separate public repo `ECLIPSE-Lab/Ai4MatLectures` (package `ai4mat`, `from ai4mat.datasets import ...`). Slide decks **link out** to that site rather than embedding executable notebooks — there is no build coupling. See `project_ai4matlectures.md` for context.

## Things to leave alone

- `_site/`, `.quarto/`, `.venv/`, `.worktrees/` — build/tool output, gitignored.
- `_quarto_internal_scss_error.scss` (root and per-unit copies) — Quarto internal artifact.
- `complete_merge.ps1` — emergency Windows script that does `git reset --hard origin/main`. Never invoke without the user explicitly asking; it discards local work.
- `render_log.txt`, `render_log_final.txt` — leftover render output, not part of the build.
- `07.txt`, `08.txt`, `09.txt` — scratch transcripts in repo root.
