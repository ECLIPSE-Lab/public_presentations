# Unit 11 ↔ Unit 12 swap (GPs ↔ PINNs) — 2026-06-29

**Rationale:** Gaussian Processes are taught in **Unit 12 of MFML**; ML-PC is realigned so its
GP unit is also **Unit 12**. PINNs moves to **Unit 11**. Because the 07.07.2026 lecture
(Week 13) is cancelled/self-study, this makes **PINNs the live Week 12 lecture** and **GPs the
required Week 13 self-study** — consistent with GPs being delivered live in MFML.

## Before → After

| Topic | Folder (before) | Folder (after) | Title | Week | Deck file |
|-------|-----------------|----------------|-------|------|-----------|
| PINNs / physics-constrained ML | `unit12_pinns` | `unit11_pinns` | Unit 11 | Week 12 (live, 30.06) | `12_pinns.qmd` |
| Uncertainty / Gaussian Processes | `unit11_uncertainty_gp` | `unit12_uncertainty_gp` | Unit 12 | Week 13 (self-study, 07.07 cancelled) | `13_uncertainty_gp.qmd` |

Numbering convention preserved: folder/title `unitN` = sequential unit; deck-file & content/plan
prefix = **week number** (= folder N + 1 for units 8–13).

## Files changed

- **Folders renamed** + inner deck/`_files`/`unitNN_content_50slides.{md,html}`/`unitNN_plan.md`
  renamed to the new week prefixes.
- **Decks** `12_pinns.qmd`, `13_uncertainty_gp.qmd`: title, `## Recap` header, in-deck `## Continue`
  prev/next, and internal self/cross references (GP's "next deck" → "previous deck").
- **Navigation** `_prev_next.json` ×4: `unit10_transformers` (next), `unit11_pinns`,
  `unit12_uncertainty_gp`, `unit13_reflection` (prev). Plus the in-deck footers of the
  transformers (Unit 10) and reflection (Unit 13) decks.
- **Website** `MachineLearningForCharacterizationAndProcessing/index.qmd`: Week 12 ↔ Week 13 topic
  blocks (heading, slides link, bullets, exercise, self-study callout) swapped; date/cancellation
  lines kept with their calendar weeks; unit-group labels → "Unit IV — …and **Physics** (Weeks 10–12)",
  "Unit V — **Uncertainty**, Trust, and Synthesis (Weeks 13–14)"; topic-list order.
- **Week summaries** `week12_summary.md` (now PINNs) and `week13_summary.md` (now GPs) swapped.
- **AGENT_INSTRUCTIONS.md** curriculum-map lines for these two units.
- **Cross-references** across units 02, 03, 05, 06, 07, 08 (decks + 50-slide drafts) normalised so
  **GPs = Unit 12** and **PINNs = Unit 11** everywhere.

MFML `W12`/`W13` cross-course pointers in the decks were left unchanged (they reference the sibling
course's schedule, which is out of scope for this swap).

Backup of the pre-swap state: `/tmp/mlpc_swap_backup_2026-06-29.tar.gz`.
Generated `_freeze`/`_manuscript`/`.quarto` artifacts were not hand-edited; re-render to refresh.

## Public-presentations repo (git) — added 2026-06-30

This deck folder lives in the git repo `/home/philipp/projects/_public_presentations` (the
`SS26/_public_presentations` path is a symlink into it). Repo-level files updated for the swap:

- `index.qmd` — the ML-PC landing-page cards: Unit 11 → PINNs (`unit11_pinns/12_pinns.html`),
  Unit 12 → GPs (`unit12_uncertainty_gp/13_uncertainty_gp.html`).
- `units.yml` — the ML-PC manifest entries for num 11/12 (title/folder/deck_file/source_file).
- `README.md` — the ML-PC Unit 11/12 listing.

Not changed: `404.qmd`'s ML-PC block is stale from *before* the May realignment (references the
deleted `unit08_generalization_robustness`, archived `unit11_automation`, and old folders for
units 9/10/13/14) — a separate cleanup, left untouched. `rebuild_index.py` has a hardcoded Windows
`base_dir`, so `index.qmd` was hand-edited rather than regenerated. **`_site/` is git-tracked and
still stale — run `quarto render` before committing so the built HTML matches the renamed decks.**
