# Materials Genomics Content Planning Design

**Date:** 2026-03-10
**Scope:** The 13-unit Materials Genomics slide-deck backbone, reconciled against `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`
**Primary workspace:** `/home/philipp/projects/_public_presentations/materials_genomics`

## Goal

Create a disciplined production workflow for the full Materials Genomics course that turns the syllabus into unit-by-unit lecture content. Each unit must produce the same five outputs:

1. a chapter extraction list from the existing syllabus and book mapping,
2. a cross-book summary with `neuer-machine-learning-for-engineers` as the main source,
3. a 90-minute lecture strategy plus 50-slide content strategy,
4. a short website summary update for the teaching site,
5. an updated todo state that marks what is done and what remains.

## Constraints

- The syllabus source of truth is `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`.
- The existing slide-deck structure in `/home/philipp/projects/_public_presentations/materials_genomics` should be reused rather than replaced.
- Book content is stored as markdown/qmd files under `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26`.
- Book priority is:
  1. `neuer-machine-learning-for-engineers`
  2. `sandfeld-materials-data-science`
  3. `mcclarren-machine-learning-for-engineers`
  4. `bishop-pattern-recognition-and-machine-learning-2006`
  5. `murphy-machine-learning-a-probabilistic-perspective-2012`
- The work should respect the current per-unit folder layout and existing `unit*_plan.md` / `unit*_content_50slides.md` files.

## Existing structure to reuse

- Unit folders already exist for Units 1-13 under `/home/philipp/projects/_public_presentations/materials_genomics`.
- Existing helper files already capture part of the mapping problem:
  - `/home/philipp/projects/_public_presentations/materials_genomics/BOOK_CHAPTER_MAPPING_UNITS3_13.md`
  - `/home/philipp/projects/_public_presentations/materials_genomics/REALIGNMENT_OLD_TO_NEW_MAPPING.md`
- The teaching website currently stores the course outline and short summaries in `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`.

## Normalization rule

The teaching site currently presents a week-by-week syllabus, while the slide repository already follows a 13-unit content backbone.

Planning should therefore normalize to the existing 13-unit deck structure:

1. `01_intro`
2. `02_crystal_structure_fundamentals`
3. `03_materials_databases`
4. `04_classical_descriptors`
5. `05_graph_based_rep`
6. `06_local_atomic_envs`
7. `07_regression_and_generalization_in_materials_data`
8. `08_neural_networks_for_materials_properties`
9. `09_representation_learning_and_feature_discovery`
10. `10_latent_spaces_of_materials`
11. `11_clustering_vs_discovery_in_materials_spaces`
12. `12_uncertainty_aware_discovery_and_gaussian_processes`
13. `13_constraints_trust_and_integration_outlook`

This means the plan must explicitly document where the current syllabus is more fine-grained than the deck structure, especially in the early simulation/stability topics and the late-course synthesis topics.

## Design decision

Use the slide-deck repository as the execution hub and keep the teaching website as the publication target.

This keeps planning artifacts close to the per-unit slide folders, avoids duplicating tracking logic, and isolates planning from the already-dirty teaching repository until a unit is ready for summary updates.

## Deliverable model

For each of the 13 units, the workflow will follow the same template:

1. Identify exact source chapters from the five book folders.
2. Read the required book markdown files and produce one synthesized unit summary.
3. Convert the summary into:
   - a 90-minute lecture arc,
   - a 50-slide teaching scaffold,
   - lecture emphasis, examples, and likely failure modes.
4. Update the unit summary on the teaching website.
5. Mark the unit state in a shared tracker.

## File strategy

The planning layer will add three artifacts:

- This design document in `docs/superpowers/specs/`
- One implementation plan in `docs/superpowers/plans/`
- One master todo tracker in `/home/philipp/projects/_public_presentations/materials_genomics/`

Execution should then modify the existing per-unit files rather than invent new destinations.

## Execution chunks

The work is large enough to chunk by dependency and topic coherence:

- Chunk 1: Units 1-4
- Chunk 2: Units 5-7
- Chunk 3: Units 8-10
- Chunk 4: Units 11-13

This keeps each execution segment reviewable and lets progress continue even if later units need topic refinement.

## Success criteria

The plan is successful if it enables a future execution pass to complete each unit without ambiguity about:

- which chapter files to read,
- which existing course files to update,
- how to prioritize books,
- what “done” means for lecture content,
- how to track progress across all 13 units.
