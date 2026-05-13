# MG Realignment — 2026-05-13

Second pass on the SS26 schedule. The actual Week 5 lecture (12.05.2026)
covered Hiemer §7.4 (Monte Carlo) and §8.1–8.3 (continuum mechanics:
balance laws, FDM, FEM/FV), not the originally planned "classical
descriptors → learned representations". The remainder of the semester
is replanned around 8 sessions (one week cancelled, 26.05.2026 public
holiday).

## New unit folder + renumbering

- **Inserted**: `05_monte_carlo_and_continuum` between Units 04 and the
  former Unit 05. Contains a 50-slide deck plus `unit5_plan.md` and
  `unit5_content_50slides.md`.
- **Renumbered**: every folder from the old `05` upward shifted by +1
  so that folder index now equals lecture week number.

| Old folder | New folder | Maps to week |
|---|---|---|
| `05_graph_based_rep` | `06_graph_based_rep` | Week 6 |
| `06_local_atomic_envs` | `07_local_atomic_envs` | Week 8 (no Week 7) |
| `07_regression_and_generalization_in_materials_data` | `08_regression_and_generalization_in_materials_data` | Week 9 |
| `08_neural_networks_for_materials_properties` | `09_neural_networks_for_materials_properties` | Week 10 |
| `09_representation_learning_and_feature_discovery` | `10_representation_learning_and_feature_discovery` | Week 11 (merged with latent) |
| `10_latent_spaces_of_materials` | `11_latent_spaces_of_materials` | merged into Week 11 |
| `11_clustering_vs_discovery_in_materials_spaces` | `12_clustering_vs_discovery_in_materials_spaces` | repurposed for Week 12 generative |
| `12_uncertainty_aware_discovery_and_gaussian_processes` | `13_uncertainty_aware_discovery_and_gaussian_processes` | Week 13 |
| `13_constraints_trust_and_integration_outlook` | `14_constraints_trust_and_integration_outlook` | Week 14 |

`_prev_next.json` and the prev-next markdown block in each `01_intro.qmd`
were rewritten to match the new numbering. The `unitN_plan.md` and
`unitN_content_50slides.md` files were also renamed so the local
filename matches the new unit number.

## New schedule (Weeks 5–14)

8 sessions remain after Week 5 because the 26.05.2026 lecture (Week 7)
is cancelled for a public holiday.

| Week | Date | Topic | Folder |
|---|---|---|---|
| 5 | 12.05.2026 | Monte Carlo sampling & continuum mechanics | `05_monte_carlo_and_continuum` |
| 6 | 19.05.2026 | Graph-based crystal reps (+ classical descriptor recap) | `06_graph_based_rep` |
| 7 | 26.05.2026 | **Cancelled** — public holiday | — |
| 8 | 02.06.2026 | Local atomic environments + universal MLIPs (MACE-MP-0, M3GNet, CHGNet) | `07_local_atomic_envs` |
| 9 | 09.06.2026 | Regression & generalization in materials data | `08_regression_and_generalization_in_materials_data` |
| 10 | 16.06.2026 | Neural networks for materials properties | `09_neural_networks_for_materials_properties` |
| 11 | 23.06.2026 | Representation learning + latent spaces (merged) | `10_representation_learning_and_feature_discovery` (absorbs `11_latent_spaces_of_materials`) |
| 12 | 30.06.2026 | Generative models & inverse design (MatterGen, DiffCSP, CrystaLLM, FlowMM) | `12_generative_models_and_inverse_design` |
| 13 | 07.07.2026 | Uncertainty-aware discovery: GPs, deep ensembles, active learning | `13_uncertainty_aware_discovery_and_gaussian_processes` |
| 14 | 14.07.2026 | Physical constraints, limits, outlook | `14_constraints_trust_and_integration_outlook` |

## Editorial decisions behind the replan

1. **Classical descriptors compressed** into a 10–15-minute recap inside
   Week 6 (graph reps). Magpie / matminer / RDF still appear as baselines
   in modern matbench leaderboards, but they are no longer where the
   field lives.
2. **Local atomic envs merged with universal MLIPs** in Week 8 because
   SOAP / ACSF are now best taught as the *bridge* to MACE-family
   equivariant message passing rather than as a destination.
3. **Representation learning and latent spaces merged** (Week 11)
   because the original split produced two thin sessions on
   overlapping autoencoder / latent-variable material.
4. **Clustering as standalone discovery dropped**. Modern materials
   discovery is graph model + uncertainty + generative; cluster-based
   triage is folded into the Week 13 active-learning session.
5. **Generative models added** as Week 12. MatterGen (2023), DiffCSP,
   CrystaLLM, FlowMM are the headline 2023–2025 development and were
   missing from the curriculum.
6. **GPs broadened to UQ + active learning** (Week 13). GPs remain the
   small-data reference, but deep ensembles and evidential learning
   now dominate at production scale; active-learning loops are
   the framing tool.
7. **Week 14 kept** as the synthesis/limits/outlook lecture; topic list
   updated to include equivariance and the foundation-model era.

## Completed follow-ups (same-day)

All four originally-pending follow-ups were closed in the same pass:

- Folder `12_clustering_vs_discovery_in_materials_spaces` was rewritten
  with a 50-slide deck on generative models & inverse design
  (CDVAE → DiffCSP → MatterGen → FlowMM → CrystaLLM, plus the
  candidate-filtering funnel) and renamed to
  `12_generative_models_and_inverse_design`. The unit12 plan /
  content scaffold were rewritten in the same step. Prev/next links
  in neighbouring folders 11 and 13 were updated.
- Folder `11_latent_spaces_of_materials` is now marked **supplementary
  reading**. The frontmatter title carries a `(supplementary)` suffix
  and a banner slide explains that the Week 11 lecture is delivered
  from folder 10 and that this deck is an optional deeper dive.
- `BOOK_CHAPTER_MAPPING_UNITS3_13.md` was archived as
  `BOOK_CHAPTER_MAPPING.md.old` and replaced by a new
  `BOOK_CHAPTER_MAPPING.md` covering Units 5–14 with the new topics
  and citations (MatterGen, DiffCSP, FlowMM, GNoME, MACE-MP-0, etc.).
- `AGENT_INSTRUCTIONS.md` was rewritten to reflect the
  post-2026-05-13 folder layout, slide style families, and the known
  caveat that in-deck cross-references in older scaffold-style decks
  still cite pre-renumbering unit numbers.

## In-deck cross-reference cleanup (completed same-day)

A second pass swept every Unit-number cross-reference in the body prose
of folders 06–11 and 13–14:

- "Unit N" and "Units N–M" forms — bumped by +1 when N ≥ 5, ranges
  handled element-wise.
- "MG U N" / "MG Unit N" shorthand — bumped on the same rule.
- Bare "U N" cross-references (e.g. "to U13 next week",
  "Forward link to U13") — bumped via a second-pass regex that
  protects MFML / ML-PC prefixes through fixed-width negative
  lookbehinds.
- Previously-bumped self-references ("Today — Unit N in one line",
  frontmatter titles) were reverted before the comprehensive pass to
  avoid double-bumping, then ended up at the correct new value after
  the +1 sweep.
- "MFML W*N*" / "MFML U*N*" / "ML-PC U*N*" / "ML-PC W*N*" references
  were preserved unchanged (parallel-track unit numbers are owned by
  the other two courses).
- Frontmatter and prev/next blocks were excluded from both passes via
  zone splitting so the previously-fixed navigation metadata stayed
  intact.

The supplementary banner in
`11_latent_spaces_of_materials/01_intro.qmd` was added post-renumbering
and its unit references had to be reverted manually after the bumper
incorrectly incremented them. The banner now correctly points students
at Unit 10 (representation learning) and Unit 12 (generative models).

## Remaining lecturer cleanup

- Folder 11's body prose still describes Week 11 in terms of
  "clustering / discovery" (the original Unit 12 framing). The
  numbering is consistent, but a few sentences mix "U12 partitions $z$"
  (now generative models, not clustering) with clustering content. A
  banner slide flags this; the body prose will need light editing
  during lecture prep if the supplementary deck is offered to students.
- "Last unit covered X" phrases in some decks (e.g. folder 09 saying
  "Last unit you built a crystal graph" while the bumped reference is
  now "MG U6") were authored relative to a different unit-ordering
  intuition than today's schedule. Numbering is correct; narrative
  framing may need a sentence-level rewrite by the lecturer.
