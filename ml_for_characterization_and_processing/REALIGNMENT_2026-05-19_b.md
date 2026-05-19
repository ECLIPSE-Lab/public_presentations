# ML-PC Realignment — 2026-05-19 (follow-up b)

**Goal:** Skip the Reinforcement-Learning / Automation lecture and instead teach
transformer applications in characterization. Archive `unit10_automation`;
promote the former Week 9 companion deck `unit09b_transformers_for_materials`
to the standalone Week 10 slot.

**Why:** Course scope decision — RL/autonomous-lab control is out; transformer
methods (ViT, Flash Attention, attention on 4D-STEM / LPBF stacks) are in.

---

## Changes

| Before | After |
|---|---|
| Week 9: characterization signals + **unit09b** transformers companion | Week 9: characterization signals (no companion) |
| Week 10: **unit10_automation** (Automation / RL) | Week 10: **unit10_transformers_for_materials** (promoted from unit09b) |
| unit10_automation published | `_archive/unit10_automation/` (Quarto-excluded, kept in git history) |

Weeks 11–13 unchanged (uncertainty/GP, PINNs, reflection); Week 14 still buffer.
No cascade renumber — this is a 1:1 swap of the Week 10 slot plus removal of the
Week 9 companion status.

## Files touched

- **Filesystem (git mv):** `unit10_automation` → `_archive/unit10_automation/`;
  `unit09b_transformers_for_materials` → `unit10_transformers_for_materials/`
  (deck file kept: `transformers_for_materials.qmd`).
- **Nav `_prev_next.json`:** `unit09_characterization_signals` (next → unit10_transformers),
  `unit10_transformers_for_materials` (unit_num 09b→10, next → unit11_uncertainty_gp),
  `unit11_uncertainty_gp` (prev → unit10_transformers).
- **Deck internals:** transformers deck title `Unit 9b` → `Unit 10`; its footer
  Next → Unit 11; stale "supplementary time-series deck" → "Week 7 time-series
  deck"; `unit09_characterization_signals` footer Next → Unit 10 transformers;
  `unit11_uncertainty_gp` footer Previous → Unit 10 transformers.
- **Website (`MachineLearningForCharacterizationAndProcessing/index.qmd`):**
  removed the Week 9 companion line; replaced the Week 10 Automation block with
  Transformers content + new slide URL; section header → "Characterization,
  Transformers, and Uncertainty".
- **`week10_summary.md`:** rewritten Automation → Transformers.
- **Curriculum docs:** both `AGENT_INSTRUCTIONS.md` files updated.

## Reversal

`git mv _archive/unit10_automation unit10_automation` and restore the Week 10
block from git history; revert the nav/footers.
