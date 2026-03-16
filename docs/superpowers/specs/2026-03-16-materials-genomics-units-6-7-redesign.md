# Materials Genomics Units 6-7 Redesign

Date: 2026-03-16
Repo: `_public_presentations`
Branch: `mg-folder-rename`
Scope: Replace the current scaffold-style slide decks for Materials Genomics Units 6 and 7 with full-content lecture decks in the style of the existing MFML presentations.

## Context

The current Materials Genomics decks for Unit 6 (`Local Atomic Environments`) and Unit 7 (`Regression and Generalization in Materials Data`) were drafted as topic scaffolds. They contain prompts, placeholders, and chapter anchors, but not enough actual lecture content to support a 90-minute class in the style already established by the Mathematical Foundations of AI and ML lecture.

The target style is the existing MFML reveal.js decks:

- dense but teachable slides
- explicit definitions and equations
- clear concept progression
- worked examples and failure modes
- slides that can stand on their own as lecture artifacts

Units 2-5 are out of scope and will be handled by a colleague.

## Goals

- Replace Unit 6 and Unit 7 `01_intro.qmd` files with full-content lecture decks.
- Match the pedagogical style and density of the MFML decks rather than extending the current MG scaffold style.
- Synthesize lecture content from the cited book chapters and materials-ML references instead of merely pointing to them.
- Keep the lecture pacing appropriate for about 90 minutes per unit.
- Preserve the existing folder structure and Quarto/reveal.js setup.

## Non-Goals

- Reworking Units 2-5.
- Redesigning the overall reveal.js theme or lecture template.
- Converting the full course into a different artifact structure.
- Committing rendered HTML artifacts unless that becomes an explicit publishing requirement.

## Chosen Approach

Use full-content decks of roughly 30-40 slides per unit, with each slide doing one concrete teaching job. This is the closest match to the MFML reference decks and the best fit for a 90-minute lecture.

Alternatives considered:

1. Chapter-faithful higher-slide-count decks.
   This improves topical coverage but tends to produce thinner slides and a weaker lecture narrative.
2. Highly compressed 20-25 slide decks.
   This can work live, but it leaves too little explicit content on the slides for reuse and student review.

## Content Standard

Every retained or new slide must contain substantive lecture material. Slides should not say "explain X" or "apply Y." Instead they should include some combination of:

- a definition, principle, or claim
- an equation or formal object when needed
- a materials-focused example
- a diagnostic, limitation, or failure mode
- a clear transition to the next concept

The slide deck should read like a lecture manuscript in slide form, not like an outline for a future lecture.

## Unit 6 Design

### Teaching Arc

Unit 6 will be rebuilt around the following arc:

1. Why local atomic environments are needed in materials representations.
2. What a physically meaningful descriptor must satisfy.
3. Simple geometric local descriptors: neighbors, coordination, bond lengths, bond angles, Voronoi-based ideas.
4. Descriptor invariance requirements: translation, rotation, permutation, and continuity.
5. SOAP and ACSF-style descriptors as principled atom-centered feature constructions.
6. Aggregating local descriptors into material-level features for downstream prediction.
7. Transferability, defects, noise sensitivity, and common failure modes.
8. One or more worked materials examples showing how local environments drive a prediction or structure discrimination task.

### Expected Output

- A coherent lecture deck in `materials_genomics/06_local_atomic_envs/01_intro.qmd`
- Companion notes in `unit6_content_50slides.md` rewritten as actual content notes
- `unit6_plan.md` reduced to an instructor map and sequencing guide

## Unit 7 Design

### Teaching Arc

Unit 7 will be rebuilt around the following arc:

1. Regression as supervised prediction for materials targets.
2. Target choice and what different targets imply for modeling difficulty.
3. Linear baselines, least squares, and regularized regression.
4. Nonlinear baselines and why benchmarking against them matters.
5. Metrics and what they do or do not reveal.
6. Train/validation/test design for materials datasets.
7. Grouped splits, chemistry-aware splits, and out-of-distribution evaluation.
8. Learning curves, residual analysis, and leakage diagnostics.
9. Practical trust criteria for surrogate models used in discovery workflows.
10. A worked benchmark-style comparison that demonstrates how random splits can overstate performance.

### Expected Output

- A coherent lecture deck in `materials_genomics/07_regression_and_generalization_in_materials_data/01_intro.qmd`
- Companion notes in `unit7_content_50slides.md` rewritten as actual content notes
- `unit7_plan.md` reduced to an instructor map and sequencing guide

## File Strategy

### Authoritative Content

`01_intro.qmd` will be the primary teaching artifact and contain the actual lecture content.

### Companion Notes

`unit*_content_50slides.md` will remain as companion content files, but they will no longer be prompt scaffolds. They will summarize the actual lecture content and can be used as planning or revision notes even if the final deck is not exactly 50 slides.

### Instructor Plan

`unit*_plan.md` will become concise instructor-facing guidance:

- lecture arc
- time allocation
- key takeaways
- optional cuts if time runs short

## Verification

For each unit:

- render the deck with Quarto
- confirm the deck builds without errors
- inspect enough output to ensure slide structure is intact

Rendered HTML and `_files` directories should remain untracked unless there is a separate publishing need.

## Risks

- The current repository contains older scaffold content that can bias the rewrite toward placeholders. The rewrite should replace, not lightly edit, that style.
- Source synthesis can become too chapter-faithful and lose lecture flow. The narrative must be optimized for teaching first.
- Slide counts can drift upward if every subtopic gets isolated. Dense explanatory slides are preferable to many thin slides.

## Implementation Notes

- Work only in Units 6 and 7.
- Leave existing Unit 2 dirty work untouched for now.
- Treat previous Unit 6 and 7 scaffold commits as superseded by the rewrite.
- Follow the existing MFML formatting and tone rather than inventing a new slide idiom.
