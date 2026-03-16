# Materials Genomics Unit 8 Application Redesign

Date: 2026-03-16
Repo: `_public_presentations`
Branch: `mg-folder-rename`
Scope: Replace the current scaffold-style Unit 8 deck with an application-focused lecture on neural surrogates for materials-property prediction that assumes neural-network fundamentals have already been covered in MFML.

## Context

The current Unit 8 files in `materials_genomics/08_neural_networks_for_materials_properties/` are scaffold notes rather than a teachable lecture. Their topic selection also drifts toward generic neural-network introduction material such as activation and optimization concepts that already belong to the Mathematical Foundations of AI and ML lecture sequence.

MFML already covers:

- neural-network architectures and activations
- backpropagation and gradient flow
- loss landscapes and optimization
- generalization, regularization, and trust

Materials Genomics Unit 8 should therefore not repeat those topics as primary content. Instead, it should build on them and focus on how feed-forward neural surrogates are used, evaluated, and limited in materials workflows.

## Goals

- Rewrite Unit 8 as a full-content lecture deck in the same teaching style as the MFML decks and the rewritten MG Units 6 and 7.
- Keep the unit focused on feed-forward neural surrogates on fixed materials representations.
- Emphasize materials-specific application decisions rather than generic neural-network fundamentals.
- Position Unit 8 as the bridge from classical baselines to later units on learned representations.

## Non-Goals

- Re-teaching perceptrons, activation taxonomies, universal approximation, or generic backpropagation derivations.
- Covering graph neural networks, autoencoders, latent models, or feature-learning architectures in depth.
- Redesigning the slide framework or reveal.js theme.

## Chosen Approach

Use a 30-35 slide full-content deck that treats neural networks as one more model class in a materials benchmark stack. The lecture should assume MFML background and spend its time on:

- when an MLP is scientifically justified
- how neural surrogates behave in small and medium materials datasets
- how they compare fairly to ridge and random-forest baselines
- where they fail and how to judge trust

## Teaching Arc

Unit 8 should follow this arc:

1. Why move from linear and tree baselines to a neural surrogate at all.
2. What changes when the input is composition-only, descriptor-based, or structure-enriched fixed features.
3. Architecture choices for limited-data materials regimes without re-deriving neural-network basics.
4. Single-target versus multi-target materials prediction.
5. Neural surrogates as approximations to expensive simulation pipelines rather than generic black boxes.
6. Effective sample size, near-duplicate compounds, and grouped evaluation logic.
7. Honest comparison to ridge and random forest under identical grouped splits.
8. Failure modes: extrapolation, weak splits, domain shift, false confidence, and overclaiming.
9. Criteria for when not to use a neural surrogate.
10. Bridge to Unit 9, where the representation itself becomes learned.

## Content Standard

Each slide must contain substantive materials-genomics teaching content. Slides should include:

- a concrete modeling decision or scientific claim
- a materials-focused example or benchmark framing
- equations only where they support application choices
- explicit failure modes and trust criteria

Slides should not read like generic deep-learning notes. Neural-network fundamentals may appear only as short reminders in service of application-focused points.

## File Strategy

### Authoritative Deck

`materials_genomics/08_neural_networks_for_materials_properties/01_intro.qmd` becomes the full lecture deck.

### Companion Notes

`unit8_content_50slides.md` becomes substantive content notes aligned with the actual lecture, even if the final deck uses fewer than 50 slides.

### Instructor Guide

`unit8_plan.md` becomes a concise instructor-facing lecture map with timing, must-cover points, optional cuts, and transition notes.

## Verification

- Render `01_intro.qmd` with Quarto.
- Confirm that the output builds without errors.
- Inspect the slide outline to ensure the deck is application-focused and no longer scaffold-based.

## Risks

- The rewrite can drift back into generic ML content because the unit title contains "neural networks."
- The rewrite can underuse the MFML prerequisite and end up duplicating foundation material.
- The unit can become too implementation-heavy unless the focus remains on scientific decision-making in materials workflows.

## Implementation Notes

- Leave Units 2-5 untouched.
- Preserve the current Unit 6 and Unit 7 rewrite standard.
- Keep the bridge to Unit 9 explicit: Unit 8 still assumes fixed representations.
