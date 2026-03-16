# Materials Genomics Unit 8 Rewrite Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite Materials Genomics Unit 8 as an application-focused lecture on neural surrogates for materials-property prediction.

**Architecture:** Keep the existing three-file unit structure, but replace the current scaffold content with a synchronized set of substantive artifacts: a full lecture deck in `01_intro.qmd`, companion lecture notes in `unit8_content_50slides.md`, and a concise instructor guide in `unit8_plan.md`. The lecture should assume MFML neural-network prerequisites and focus on materials-genomics application decisions, benchmark design, failure modes, and trust.

**Tech Stack:** Quarto, RevealJS, Markdown, existing Materials Genomics slide theme and assets.

---

## Chunk 1: Unit 8 Rewrite

### Task 1: Replace the Unit 8 scaffold with an application-focused lecture package

**Files:**
- Modify: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/01_intro.qmd`
- Modify: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/unit8_content_50slides.md`
- Modify: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/unit8_plan.md`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/mathematical_foundations_of_ai_and_ml/04_neural_networks_activations/01_intro.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/mathematical_foundations_of_ai_and_ml/05_backpropagation_gradient_flow/01_intro.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/mathematical_foundations_of_ai_and_ml/06_loss_landscapes_optimization/01_intro.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/mathematical_foundations_of_ai_and_ml/07_generalization_bias_variance/01_intro.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/docs/superpowers/specs/2026-03-16-materials-genomics-unit-8-application-redesign.md`

- [ ] **Step 1: Re-read the current Unit 8 source files and identify scaffold sections to replace**

Run:
```bash
sed -n '1,260p' /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/01_intro.qmd
sed -n '1,260p' /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/unit8_content_50slides.md
sed -n '1,220p' /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/unit8_plan.md
```

Expected: current files are scaffold-style and overemphasize generic neural-network topics already covered in MFML.

- [ ] **Step 2: Rewrite `unit8_plan.md` as a concise instructor guide**

Replace the scaffold with a guide containing:

- lecture arc
- timing by block
- must-cover application points
- optional cuts
- explicit bridge into Unit 9 representation learning

- [ ] **Step 3: Rewrite `unit8_content_50slides.md` into substantive lecture notes**

Write actual content notes covering:

- when an MLP is justified over ridge and RF on fixed materials features
- dataset-regime and effective-sample-size issues in materials data
- single-target and multi-target property prediction
- surrogate role relative to expensive simulation
- grouped evaluation and fair benchmark comparison
- domain shift, extrapolation, false confidence, and "when not to use an NN"

- [ ] **Step 4: Rewrite `01_intro.qmd` as a full lecture deck**

Replace scaffold slides with approximately 30-35 content-rich slides that:

- assume MFML neural-network prerequisites
- avoid generic NN-intro material
- focus on materials-genomics application choices and failure modes
- use concise equations only where they support surrogate-model decisions
- maintain the established MFML-style lecture density

- [ ] **Step 5: Render Unit 8**

Run:
```bash
quarto render /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/01_intro.qmd
```

Expected: render exits with code 0 and produces an updated `01_intro.html`.

- [ ] **Step 6: Spot-check the rendered deck structure**

Run:
```bash
rg -n "^## " /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/01_intro.qmd
```

Expected: the outline shows a coherent application-focused lecture rather than generic deep-learning introduction slides.

- [ ] **Step 7: Commit Unit 8**

Run:
```bash
git add /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/01_intro.qmd /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/unit8_content_50slides.md /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/unit8_plan.md
git commit -m "docs: rewrite materials genomics unit 8"
```

Expected: one commit containing only Unit 8 source changes.

## Chunk 2: Verification And Handoff

### Task 2: Verify the branch state after the Unit 8 rewrite

**Files:**
- Modify: none required
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/`

- [ ] **Step 1: Check git status**

Run:
```bash
git status --short
```

Expected: Unit 8 source changes are committed; only pre-existing Unit 2 dirt and untracked HTML outputs may remain.

- [ ] **Step 2: Summarize the new Unit 8 lecture shape**

Run:
```bash
rg -n "^## " /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/_public_presentations/.worktrees/mg-folder-rename/materials_genomics/08_neural_networks_for_materials_properties/01_intro.qmd
```

Expected: the outline reflects a materials-surrogate application unit rather than an NN-foundations lecture.

- [ ] **Step 3: Prepare for the finishing workflow**

No file change. Confirm that the Unit 8 commit exists and that the branch is ready either for continuation to Unit 9 or for an integration decision later.
