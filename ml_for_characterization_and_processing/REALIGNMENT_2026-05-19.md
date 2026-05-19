# ML-PC Realignment — 2026-05-19

**Goal:** Week 7 becomes time-series self-study (lecture self-study + Thursday exercise); Week 8 becomes Inverse problems & process maps (lecture delivered, Thursday exercise self-study due to Fronleichnam); the Generalization/robustness unit is dropped (duplicate content); Weeks 9–14 shift up by one with Week 14 freed as a buffer/review slot.

**Why:** Week 7 Tue 26.05 lecture cancelled (Pfingstdienstag); Week 8 Thu 04.06 exercise cancelled (Fronleichnam). User decision 2026-05-19: shift back half up by one, delete generalization deck, realign disk folders to week-aligned URLs.

---

## Target schedule (single source of truth)

| Week | Tue lecture | Thu exercise | Topic | Disk folder | Deck file |
|---|---|---|---|---|---|
| 7 | 26.05 **self-study** (Pfingstdienstag) | 28.05 exercise (delivered) | Time-series and process monitoring | `unit07_time_series` | `01_intro.html` |
| 8 | 02.06 lecture (delivered) | 04.06 **self-study** (Fronleichnam) | Inverse problems and process maps | `unit08_inverse_problems` | `09_inverse_problems.html` |
| 9 | 09.06 | 11.06 | ML for characterization signals | `unit09_characterization_signals` | `10_characterization_signals.html` |
| — | (companion within W9) | | Transformers for materials | `unit09b_transformers_for_materials` | `transformers_for_materials.html` |
| 10 | 16.06 | 18.06 | Automation in microscopy and characterization | `unit10_automation` | `11_automation.html` |
| 11 | 23.06 | 25.06 | Uncertainty-aware regression & Gaussian Processes | `unit11_uncertainty_gp` | `12_uncertainty_gp.html` |
| 12 | 30.06 | 02.07 | Physics-informed and constrained ML | `unit12_pinns` | `13_pinns.html` |
| 13 | 07.07 | 09.07 | Integration, limits, and reflection | `unit13_reflection` | `14_reflection.html` |
| 14 | 14.07 | 16.07 | **Buffer / review / project work** (no unit deck) | — | — |

**Deleted:** `unit07_generalization_robustness/` (was website Week 8).

**Key fact:** disk folders `unit08`–`unit13` are already week-aligned for the new schedule. The website `index.qmd` was written against an aborted scheme, so its W8–W14 slide links were already broken; this realignment fixes them.

---

## Open judgment calls (defaults chosen; flag on review)

1. Section headers in `index.qmd` regrouped: **Unit III — Learning from Processing Data (Weeks 7–8)**; **Unit IV — Characterization, Automation, Uncertainty (Weeks 9–11)**; **Unit V — Physics, Trust, Synthesis (Weeks 12–13)**; Week 14 standalone buffer.
2. Week 14 wording: "Buffer / review / mini-project work — no new material."

---

## Tasks

### Task 1: Safety backup of the deck being deleted
- [ ] `tar -czf /tmp/unit07_generalization_robustness_backup_2026-05-19.tar.gz -C _public_presentations/ml_for_characterization_and_processing unit07_generalization_robustness`
- [ ] Verify archive non-empty: `tar -tzf /tmp/...tar.gz | head`

### Task 2: Rename time-series folder
- [ ] `git`-free move: `mv unit07_time_series_supplementary unit07_time_series` (in the ML-PC presentation dir)
- [ ] Verify: `ls -d unit07_time_series` succeeds; `unit07_time_series_supplementary` gone

### Task 3: Delete generalization folder
- [ ] `rm -rf unit07_generalization_robustness` (backup exists from Task 1)
- [ ] Verify gone

### Task 4: Fix navigation `_prev_next.json` (3 files)
- [ ] `unit06_transfer_learning/_prev_next.json`: `next_url` → `../unit07_time_series/01_intro.html`; `next_title` → `Unit 07 — Time-series and process monitoring`
- [ ] `unit07_time_series/_prev_next.json`: `prev_url` → `../unit06_transfer_learning/06_transfer_learning.html`; `next_url` → `../unit08_inverse_problems/09_inverse_problems.html`; `next_title` → `Unit 08 — Inverse problems and process maps`
- [ ] `unit08_inverse_problems/_prev_next.json`: `prev_url` → `../unit07_time_series/01_intro.html`; `prev_title` → `Unit 07 — Time-series and process monitoring`
- [ ] (unit09 … unit13 chains already correct — verify only, no edit)

### Task 5: Rewrite `index.qmd` schedule (MachineLearningForCharacterizationAndProcessing/index.qmd)
- [ ] Week 7 block (~L319–341): change lecture label to `Tuesday, 26.05.2026 … (self-study — Pfingstdienstag) | Exercise: Thursday, 28.05.2026 …`; add a one-line "self-study lecture" note; slide URL stays `unit07_time_series/01_intro.html`
- [ ] Week 8 block (~L344–364): replace Generalization content with **Inverse problems and process maps** content (moved from current Week 9); exercise label → `Thursday, 04.06.2026 … (self-study — Fronleichnam)`; slide URL → `unit08_inverse_problems/09_inverse_problems.html`
- [ ] Week 9 (~L368): now **ML for characterization signals**; URL → `unit09_characterization_signals/10_characterization_signals.html` (+ note 9b transformers companion)
- [ ] Week 10: now **Automation**; URL → `unit10_automation/11_automation.html`
- [ ] Week 11: now **Uncertainty-aware regression & GP**; URL → `unit11_uncertainty_gp/12_uncertainty_gp.html`
- [ ] Week 12: now **Physics-informed and constrained ML**; URL → `unit12_pinns/13_pinns.html`
- [ ] Week 13: now **Integration, limits, and reflection**; URL → `unit13_reflection/14_reflection.html`
- [ ] Week 14: **Buffer / review / mini-project**, no slide link
- [ ] Update section dividers ("Unit III/IV/V — Weeks …") per judgment call 1
- [ ] Verify no remaining `unit07_generalization` / `unit08_generalization` / off-by-one URL strings in index.qmd

### Task 6: Reconcile `weekN_summary.md` (manuscript folder)
- [ ] Delete `week8_summary.md` (Generalization — dropped)
- [ ] Shift content: week9→week8, week10→week9, week11→week10, week12→week11, week13→week12, week14→week13 (retitle headings to match new week + topic)
- [ ] `week14_summary.md`: replace with a short "Week 14 — buffer/review" note (or delete; flag)

### Task 7: Update curriculum docs
- [ ] `_public_presentations/ml_for_characterization_and_processing/AGENT_INSTRUCTIONS.md`: rewrite Curriculum Mapping block to the new schedule; remove generalization line; note W14 buffer
- [ ] `MachineLearningForCharacterizationAndProcessing/AGENT_INSTRUCTIONS.md`: update the unit-folder list (drop `unit08_generalization_robustness`, fix `unit07_time_series`, correct the "1:1 to 14 weeks" mapping)

### Task 8: Sweep for stragglers
- [ ] `grep -rin "generalization_robustness\|time_series_supplementary" _public_presentations/ml_for_characterization_and_processing MachineLearningForCharacterizationAndProcessing` → expect only intentional prose mentions, no path/link refs
- [ ] Fix any deck internal cross-references that pointed at the deleted generalization unit (known: `unit08_inverse_problems/09_inverse_problems.qmd`, `unit06_transfer_learning/06_transfer_learning.qmd`, the renamed time-series deck) — soften to prose, don't link

### Task 9: Update memory
- [ ] Update `reference_mlpc_slides.md` with the post-2026-05-19 schedule + this realignment note path

---

## Verification (final)
- [ ] `ls -d unit07_time_series` exists; `unit07_time_series_supplementary` and `unit07_generalization_robustness` gone
- [ ] Nav chain unbroken: unit06 → unit07_time_series → unit08_inverse_problems → unit09_characterization_signals → unit09b → unit10_automation → unit11_uncertainty_gp → unit12_pinns → unit13_reflection (each prev/next resolves to an existing folder+file)
- [ ] Every slide URL in index.qmd resolves to an existing disk folder+deck file
- [ ] No dangling references to generalization as a scheduled unit
