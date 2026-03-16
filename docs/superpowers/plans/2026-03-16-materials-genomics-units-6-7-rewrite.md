# Materials Genomics Units 6-7 Rewrite Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Materials Genomics Units 6 and 7 with full-content lecture decks in the style of the MFML presentations.

**Architecture:** Treat each unit as a self-contained lecture package with three synchronized artifacts: the deck (`01_intro.qmd`), substantive companion notes (`unitN_content_50slides.md`), and a short instructor guide (`unitN_plan.md`). Rewrite Unit 6 first, verify and commit it, then repeat the same pattern for Unit 7 so each unit remains reviewable and independently usable.

**Tech Stack:** Quarto, RevealJS, Markdown, BibTeX citations, existing Materials Genomics slide theme and assets.

---

## Chunk 1: Unit 6 Rewrite

### Task 1: Rebuild Unit 6 around a real lecture narrative

**Files:**
- Modify: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/01_intro.qmd`
- Modify: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/unit6_content_50slides.md`
- Modify: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/unit6_plan.md`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/mathematical_foundations_of_ai_and_ml/02_linear_algebra_pca_svd/01_intro.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/docs/superpowers/specs/2026-03-16-materials-genomics-units-6-7-redesign.md`

- [ ] **Step 1: Re-read the current Unit 6 files and identify scaffold sections to delete**

Run:
```bash
sed -n '1,260p' /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/01_intro.qmd
sed -n '1,260p' /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/unit6_content_50slides.md
sed -n '1,260p' /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/unit6_plan.md
```

Expected: the current files still contain scaffold-style prompts that should be replaced rather than lightly edited.

- [ ] **Step 2: Rewrite `unit6_plan.md` as a concise instructor guide**

Replace the prompt scaffold with a short plan containing:

- lecture arc
- approximate timing by block
- must-cover concepts
- optional cuts if time runs short
- next-unit bridge into regression/generalization

- [ ] **Step 3: Rewrite `unit6_content_50slides.md` into substantive lecture notes**

Write actual content notes covering:

- motivation for local atomic environments
- descriptor requirements and invariances
- coordination, bond geometry, and Voronoi-style descriptors
- SOAP and ACSF concepts
- aggregation to material-level features
- defects, noise, transferability, and failure modes
- one worked materials case

- [ ] **Step 4: Rewrite `01_intro.qmd` as a full lecture deck**

Replace scaffold slides with approximately 30-40 content-rich slides that include:

- numbered slide titles
- actual definitions and explanatory prose
- equations where needed
- concrete materials examples
- explicit failure modes and transition logic

- [ ] **Step 5: Render Unit 6**

Run:
```bash
quarto render /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/01_intro.qmd
```

Expected: render exits with code 0 and produces an updated `01_intro.html`.

- [ ] **Step 6: Spot-check the rendered deck structure**

Run:
```bash
rg -n "^## " /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/01_intro.qmd
```

Expected: a coherent sequence of full-content slides, not prompt-style stubs.

- [ ] **Step 7: Commit Unit 6**

Run:
```bash
git add /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/01_intro.qmd /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/unit6_content_50slides.md /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/unit6_plan.md
git commit -m "docs: rewrite materials genomics unit 6"
```

Expected: one commit containing only Unit 6 source changes.

## Chunk 2: Unit 7 Rewrite

### Task 2: Rebuild Unit 7 around trustworthy materials regression practice

**Files:**
- Modify: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/01_intro.qmd`
- Modify: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/unit7_content_50slides.md`
- Modify: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/unit7_plan.md`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/docs/superpowers/specs/2026-03-16-materials-genomics-units-6-7-redesign.md`

- [ ] **Step 1: Re-read the current Unit 7 files and mark scaffold sections for replacement**

Run:
```bash
sed -n '1,260p' /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/01_intro.qmd
sed -n '1,260p' /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/unit7_content_50slides.md
sed -n '1,260p' /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/unit7_plan.md
```

Expected: current files still reflect scaffold language and need full rewrite.

- [ ] **Step 2: Rewrite `unit7_plan.md` as an instructor guide**

Replace the prompt scaffold with:

- lecture arc
- timing by block
- must-cover diagnostics and validation concepts
- optional cuts
- bridge into the neural-network unit

- [ ] **Step 3: Rewrite `unit7_content_50slides.md` into substantive lecture notes**

Write actual content notes covering:

- regression setup for materials targets
- linear and regularized baselines
- nonlinear baselines and benchmarking
- metrics and what they hide
- grouped splits and OOD evaluation
- learning curves, residual analysis, and leakage
- trust criteria for surrogate deployment

- [ ] **Step 4: Rewrite `01_intro.qmd` as a full lecture deck**

Replace scaffold slides with approximately 30-40 content-rich slides that include:

- numbered slide titles
- definitions, equations, and interpretation
- materials-specific validation examples
- comparison of random vs grouped splits
- trust and failure-mode slides

- [ ] **Step 5: Render Unit 7**

Run:
```bash
quarto render /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/01_intro.qmd
```

Expected: render exits with code 0 and produces an updated `01_intro.html`.

- [ ] **Step 6: Spot-check the rendered deck structure**

Run:
```bash
rg -n "^## " /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/01_intro.qmd
```

Expected: a coherent sequence of full-content slides, not prompt-style stubs.

- [ ] **Step 7: Commit Unit 7**

Run:
```bash
git add /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/01_intro.qmd /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/unit7_content_50slides.md /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/unit7_plan.md
git commit -m "docs: rewrite materials genomics unit 7"
```

Expected: one commit containing only Unit 7 source changes.

## Chunk 3: Final Verification And Handoff

### Task 3: Verify the rewrite state and prepare integration

**Files:**
- Modify: none required
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/`

- [ ] **Step 1: Check git status for the rewrite branch**

Run:
```bash
git status --short
```

Expected: only known pre-existing Unit 2 dirt and untracked render artifacts remain, or a cleaner state if those artifacts were not regenerated.

- [ ] **Step 2: Summarize the new lecture shape**

Run:
```bash
rg -n "^## " /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/06_local_atomic_envs/01_intro.qmd /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/07_regression_and_generalization_in_materials_data/01_intro.qmd
```

Expected: slide outlines show fully rewritten lecture arcs for both units.

- [ ] **Step 3: Prepare the branch for finishing workflow**

No code change. Confirm that Unit 6 and Unit 7 commits exist and that the branch can now go through the finishing workflow.

- [ ] **Step 4: Use the finishing workflow**

Invoke `superpowers:finishing-a-development-branch` after the implementation work and verification are complete.
