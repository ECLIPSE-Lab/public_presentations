# Materials Genomics Content Plan Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full Materials Genomics lecture-content workflow for the existing 13-unit deck structure, including chapter extraction, cross-book summaries, 90-minute lecture strategy, 50-slide strategy, website summary updates, and todo-state tracking.

**Architecture:** Use the existing unit folders in `/home/philipp/projects/_public_presentations/materials_genomics` as the execution backbone. Track course-wide progress in one master todo file, update each unit's planning/content files in place, and publish concise unit summaries back into `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`.

**Tech Stack:** Quarto markdown (`.qmd` / `.md`), git, shell tooling (`rg`, `sed`, `find`), existing book markdown exports in `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26`

---

## Chunk 1: Course-wide setup and tracking

### Task 1: Build the shared tracker and source map

**Files:**
- Create: `/home/philipp/projects/_public_presentations/materials_genomics/MATERIALS_GENOMICS_MASTER_TODO.md`
- Reference: `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`
- Reference: `/home/philipp/projects/_public_presentations/materials_genomics/BOOK_CHAPTER_MAPPING_UNITS3_13.md`
- Reference: `/home/philipp/projects/_public_presentations/materials_genomics/REALIGNMENT_OLD_TO_NEW_MAPPING.md`

- [ ] **Step 1: Reconcile the teaching syllabus with the 13-unit deck structure**

Run: `sed -n '1,260p' /home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`
Expected: the week-by-week syllabus with current summary text.

Document the normalization rule explicitly:
- the teaching site is week-based,
- the slide decks are 13 content units,
- early simulation/stability topics are absorbed into the existing early deck units,
- the final units keep the current late-course backbone already present in the slide repo.

- [ ] **Step 2: Align each canonical content unit to its slide-deck folder**

Create a canonical table with:
- unit number,
- syllabus title,
- slide folder,
- unit plan file,
- unit content file,
- website target section.

Expected output destination: `/home/philipp/projects/_public_presentations/materials_genomics/MATERIALS_GENOMICS_MASTER_TODO.md`

- [ ] **Step 3: Record the shared five-step workflow**

Add the required workflow once at the top of the tracker:
1. extract chapters,
2. summarize across books,
3. design 90-minute lecture and 50-slide strategy,
4. update website summary,
5. update todo state and hand control back.

- [ ] **Step 4: Add one checklist block for each of the 13 canonical units**

Each checklist block must contain:
- exact unit title,
- folder path,
- five requested tasks,
- status markers for `not started`, `in progress`, `done`,
- a notes line for sources/issues.

- [ ] **Step 5: Verify the tracker is complete**

Run: `rg -n "^## Unit|^- \[ \]" /home/philipp/projects/_public_presentations/materials_genomics/MATERIALS_GENOMICS_MASTER_TODO.md`
Expected: 13 canonical unit headers and a repeated checklist structure under each.

## Chunk 2: Units 1-4

### Task 2: Units 1-4 chapter extraction and unit-plan hardening

**Files:**
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/01_intro/unit1_plan.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/02_crystal_structure_fundamentals/unit2_plan.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/03_materials_databases/unit3_plan.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/04_classical_descriptors/unit4_plan.md`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/01-data-as-the-basis-of-models.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/03-data-preprocessing.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/04-supervised-learning.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/05-unsupervised-learning.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/sandfeld-materials-data-science/markdown/04-part-i-introduction-and-foundations.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/sandfeld-materials-data-science/markdown/06-part-iii-classical-machine-learning.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/mcclarren-machine-learning-for-engineers/markdown/01-the-landscape-of-machine-learning-supervised-and-unsupervised-learning-optimization-and-other-topics.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/mcclarren-machine-learning-for-engineers/markdown/02-linear-models-for-regression-and-classification.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/mcclarren-machine-learning-for-engineers/markdown/04-finding-structure-within-a-data-set-data-reduction-and-clustering.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/bishop-pattern-recognition-and-machine-learning-2006/markdown/05-introduction.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/bishop-pattern-recognition-and-machine-learning-2006/markdown/07-linear-models-for-regression.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/bishop-pattern-recognition-and-machine-learning-2006/markdown/16-continuous-latent-variables.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/murphy-machine-learning-a-probabilistic-perspective-2012/markdown/08-introduction.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/murphy-machine-learning-a-probabilistic-perspective-2012/markdown/14-linear-regression.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/murphy-machine-learning-a-probabilistic-perspective-2012/markdown/19-latent-linear-models.qmd`

- [ ] **Step 1: Add exact chapter lists to Units 1-4**

For each of `unit1_plan.md` through `unit4_plan.md`, add a `Required chapter files` section with exact source paths grouped by book priority.

- [ ] **Step 2: Add synthesis instructions to Units 1-4**

In each unit plan file, add a `Cross-book summary target` section that states:
- main ideas to extract,
- what must come from Neuer,
- what secondary books should contribute,
- what should be excluded to avoid overloading a 5th-semester BSc audience.

- [ ] **Step 3: Add lecture timing and slide allocation guidance**

For each unit, define:
- 90-minute timing blocks,
- target slide counts per block summing to roughly 50,
- one concrete materials example,
- one common failure mode or misconception,
- one exercise handoff.

- [ ] **Step 4: Add website summary update instructions**

For each unit, specify the exact heading in `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd` that will receive the revised summary and list the new summary points to include.

- [ ] **Step 5: Verify Units 1-4 plans are execution-ready**

Run: `rg -n "Required chapter files|Cross-book summary target|Website summary update" /home/philipp/projects/_public_presentations/materials_genomics/{01_intro,02_crystal_structure_fundamentals,03_materials_databases,04_classical_descriptors}/unit*_plan.md`
Expected: each of the four files contains all three required sections.

### Task 3: Units 1-4 50-slide scaffolds and website summaries

**Files:**
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/01_intro/unit1_content_50slides.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/02_crystal_structure_fundamentals/unit2_content_50slides.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/03_materials_databases/unit3_content_50slides.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/04_classical_descriptors/unit4_content_50slides.md`
- Modify: `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/MATERIALS_GENOMICS_MASTER_TODO.md`

- [ ] **Step 1: Rewrite the slide scaffolds from chapter-backed summaries**

For each unit content file, ensure the summary is based on the extracted book chapters rather than generic placeholders.

- [ ] **Step 2: Check the 50-slide scaffold quality**

Each content file must include:
- a book-backed summary,
- source anchors,
- core equations/objects if applicable,
- a coherent 50-slide progression without filler “advanced note” placeholders unless they are justified.

- [ ] **Step 3: Update the website summaries for Units 1-4**

Revise the corresponding week/unit summaries in `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd` to match the new synthesis.

- [ ] **Step 4: Update tracker status for Units 1-4**

Mark completed checklist items in `MATERIALS_GENOMICS_MASTER_TODO.md` and add brief notes on sources used.

- [ ] **Step 5: Verify rendered-text consistency**

Run: `rg -n "Summary:" /home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`
Expected: the file contains updated summary blocks for the units completed in this chunk.

## Chunk 3: Units 5-7

### Task 4: Units 5-7 chapter extraction, synthesis, and lecture plans

**Files:**
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/05_graph_based_rep/unit5_plan.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/06_local_atomic_envs/unit6_plan.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/07_regression_and_generalization_in_materials_data/unit7_plan.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/05_graph_based_rep/unit5_content_50slides.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/06_local_atomic_envs/unit6_content_50slides.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/07_regression_and_generalization_in_materials_data/unit7_content_50slides.md`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/04-supervised-learning.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/05-unsupervised-learning.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/06-physics-informed-learning.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/sandfeld-materials-data-science/markdown/06-part-iii-classical-machine-learning.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/sandfeld-materials-data-science/markdown/07-part-iv-artificial-neural-networks-and-deep-learning.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/mcclarren-machine-learning-for-engineers/markdown/02-linear-models-for-regression-and-classification.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/mcclarren-machine-learning-for-engineers/markdown/05-feed-forward-neural-networks.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/mcclarren-machine-learning-for-engineers/markdown/08-unsupervised-learning-with-neural-networks-autoencoders.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/bishop-pattern-recognition-and-machine-learning-2006/markdown/07-linear-models-for-regression.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/bishop-pattern-recognition-and-machine-learning-2006/markdown/09-neural-networks.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/murphy-machine-learning-a-probabilistic-perspective-2012/markdown/14-linear-regression.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/murphy-machine-learning-a-probabilistic-perspective-2012/markdown/35-deep-learning.qmd`

- [ ] **Step 1: Add exact source chapter lists for Units 5-7**

Record unit-specific file lists under `Required chapter files` in each `unit*_plan.md`.

- [ ] **Step 2: Add synthesized lecture intent for Units 5-7**

For each unit plan, state:
- the central representational or regression question,
- which concepts are prerequisite from MFML,
- which parts are materials-specific and therefore need domain examples.

- [ ] **Step 3: Align slide scaffolds with the unit plans**

For each of the three `unit*_content_50slides.md` files, rewrite the top summary and slide flow so it reflects the selected chapters and avoids generic placeholder expansions.

- [ ] **Step 4: Update website summary targets**

Add or rewrite the corresponding summaries in `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd` for Weeks 5-8 as appropriate.

- [ ] **Step 5: Update tracker status and verify**

Run: `rg -n "Unit 5|Unit 6|Unit 7|done|in progress" /home/philipp/projects/_public_presentations/materials_genomics/MATERIALS_GENOMICS_MASTER_TODO.md`
Expected: the tracker reflects current progress for all three units.

## Chunk 4: Units 8-10

### Task 5: Units 8-10 chapter extraction, synthesis, and lecture plans

**Files:**
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/08_neural_networks_for_materials_properties/unit8_plan.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/09_representation_learning_and_feature_discovery/unit9_plan.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/10_latent_spaces_of_materials/unit10_plan.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/08_neural_networks_for_materials_properties/unit8_content_50slides.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/09_representation_learning_and_feature_discovery/unit9_content_50slides.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/10_latent_spaces_of_materials/unit10_content_50slides.md`
- Modify: `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/MATERIALS_GENOMICS_MASTER_TODO.md`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/04-supervised-learning.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/05-unsupervised-learning.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/mcclarren-machine-learning-for-engineers/markdown/05-feed-forward-neural-networks.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/mcclarren-machine-learning-for-engineers/markdown/08-unsupervised-learning-with-neural-networks-autoencoders.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/bishop-pattern-recognition-and-machine-learning-2006/markdown/09-neural-networks.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/bishop-pattern-recognition-and-machine-learning-2006/markdown/16-continuous-latent-variables.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/murphy-machine-learning-a-probabilistic-perspective-2012/markdown/19-latent-linear-models.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/murphy-machine-learning-a-probabilistic-perspective-2012/markdown/35-deep-learning.qmd`

- [ ] **Step 1: Add required chapter files to Units 8-10**

Ensure each unit plan includes the exact qmd file paths and a clear priority order with Neuer first.

- [ ] **Step 2: Add summary objectives and time budgets**

For each unit plan, specify:
- 3-5 synthesis questions the reading must answer,
- 90-minute sequencing,
- target slide distribution across concept, example, and exercise-briefing sections.

- [ ] **Step 3: Rewrite the content scaffolds for Units 8-10**

Replace generic slide filler with chapter-backed content and make the cross-unit transitions explicit:
- Unit 8 -> property prediction,
- Unit 9 -> feature discovery,
- Unit 10 -> latent materials spaces.

- [ ] **Step 4: Update the teaching-site summaries**

Revise the related summaries in `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`.

- [ ] **Step 5: Verify**

Run: `rg -n "Required chapter files|90-minute|50-slide|Website summary update" /home/philipp/projects/_public_presentations/materials_genomics/{08_neural_networks_for_materials_properties,09_representation_learning_and_feature_discovery,10_latent_spaces_of_materials}/unit*_plan.md`
Expected: the required planning sections exist in all three files.

## Chunk 5: Units 11-13

### Task 6: Units 11-13 chapter extraction, synthesis, and lecture plans

**Files:**
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/11_clustering_vs_discovery_in_materials_spaces/unit11_plan.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/12_uncertainty_aware_discovery_and_gaussian_processes/unit12_plan.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/13_constraints_trust_and_integration_outlook/unit13_plan.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/11_clustering_vs_discovery_in_materials_spaces/unit11_content_50slides.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/12_uncertainty_aware_discovery_and_gaussian_processes/unit12_content_50slides.md`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/13_constraints_trust_and_integration_outlook/unit13_content_50slides.md`
- Modify: `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`
- Modify: `/home/philipp/projects/_public_presentations/materials_genomics/MATERIALS_GENOMICS_MASTER_TODO.md`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/02-mathematical-description-of-data.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/05-unsupervised-learning.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/06-physics-informed-learning.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/neuer-machine-learning-for-engineers/markdown/07-explainability.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/mcclarren-machine-learning-for-engineers/markdown/04-finding-structure-within-a-data-set-data-reduction-and-clustering.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/mcclarren-machine-learning-for-engineers/markdown/08-unsupervised-learning-with-neural-networks-autoencoders.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/bishop-pattern-recognition-and-machine-learning-2006/markdown/13-mixture-models-and-em.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/bishop-pattern-recognition-and-machine-learning-2006/markdown/16-continuous-latent-variables.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/murphy-machine-learning-a-probabilistic-perspective-2012/markdown/18-mixture-models-and-the-em-algorithm.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/murphy-machine-learning-a-probabilistic-perspective-2012/markdown/22-gaussian-processes.qmd`
- Reference: `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/murphy-machine-learning-a-probabilistic-perspective-2012/markdown/32-clustering.qmd`

- [ ] **Step 1: Add exact source chapter files to Units 11-13**

Populate the required reading lists with clear primary vs secondary roles.

- [ ] **Step 2: Add lecture synthesis questions and constraints**

For each plan file, define:
- what discovery claim is being examined,
- what uncertainty/trust limit should be foregrounded,
- what should remain conceptual rather than overly mathematical for 5th-semester students.

- [ ] **Step 3: Rewrite the slide scaffolds for Units 11-13**

Ensure the slide sequences distinguish:
- clustering from discovery,
- uncertainty from mere variance,
- trustworthy constrained modeling from hype.

- [ ] **Step 4: Update the teaching-site summaries**

Revise the final unit summaries in `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`.

- [ ] **Step 5: Update the tracker and verify**

Run: `rg -n "Unit 11|Unit 12|Unit 13" /home/philipp/projects/_public_presentations/materials_genomics/MATERIALS_GENOMICS_MASTER_TODO.md`
Expected: all late-course units show clear status and notes.

## Chunk 6: Final verification and handoff

### Task 7: Verify the course-level planning state

**Files:**
- Verify: `/home/philipp/projects/_public_presentations/materials_genomics/MATERIALS_GENOMICS_MASTER_TODO.md`
- Verify: `/home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`
- Verify: `/home/philipp/projects/_public_presentations/materials_genomics/*/unit*_plan.md`
- Verify: `/home/philipp/projects/_public_presentations/materials_genomics/*/unit*_content_50slides.md`

- [ ] **Step 1: Check that all 13 units have the required sections**

Run: `rg -n "Required chapter files|Cross-book summary target|Website summary update" /home/philipp/projects/_public_presentations/materials_genomics/*/unit*_plan.md`
Expected: every canonical unit plan is reported.

- [ ] **Step 2: Check that all 13 content files are chapter-backed**

Run: `rg -n "Book-backed content summary|Source anchors used|50-slide scaffold" /home/philipp/projects/_public_presentations/materials_genomics/*/unit*_content_50slides.md`
Expected: every canonical unit content file is reported.

- [ ] **Step 3: Check that the teaching-site summaries were updated**

Run: `rg -n "Summary:" /home/philipp/projects/_teaching/MaterialsGenomics/index.qmd`
Expected: updated summary blocks exist for the completed units.

- [ ] **Step 4: Check tracker completeness**

Run: `rg -n "^## Unit" /home/philipp/projects/_public_presentations/materials_genomics/MATERIALS_GENOMICS_MASTER_TODO.md`
Expected: 13 unit sections.

- [ ] **Step 5: Prepare handoff note**

Summarize:
- completed units,
- units still open,
- book chapters actually used,
- website sections changed,
- any unresolved dependency on MFML or ML-PC coordination.

Plan complete and saved to `docs/superpowers/plans/2026-03-10-materials-genomics-content-plan.md`. Ready to execute?
