# Unit 6 Plan — Materials Genomics

> **Realignment note (2026-05-09):** Following the MG SS26 realignment
> (`../REALIGNMENT_2026-05-09.md`), this unit absorbs the classical
> compositional and pair-distribution descriptors that were previously
> sketched for old U4 (Magpie, matminer feature library, RDF / partial
> RDF, structure-aggregated coordination statistics). They are inserted
> at the front of the unit as the simplest descriptor family — one that
> needs only a formula or a structure file, no neighborhood construction.
> They motivate the rest of the unit: composition-only and globally-pooled
> descriptors are surprisingly strong baselines, but they are blind to
> the local motifs that control most structure-sensitive properties.
> The triad context — MFML covers ML pipeline mechanics, ML-PC covers
> experimental data quality — means MG U6 stays focused on
> materials-specific representation choices.

## Unit title
Local Atomic Environments

## Lecture arc
- Composition-only descriptors as the simplest baseline: Magpie elemental statistics, matminer feature library. Cheap, interpretable, formula-only — set the bar.
- Globally-pooled structural descriptors: RDF / partial RDF, structure-averaged coordination statistics. Use positions but discard local identity.
- Why these baselines fail for motif-sensitive properties — the motivation for going local.
- What a valid local descriptor must preserve: translation, rotation, permutation, continuity, and chemical identity.
- Simple local-geometric summaries: coordination number, bond lengths, bond angles, and Voronoi coordination.
- Richer descriptors: ACSF and SOAP as systematic encodings of local neighborhoods.
- Pooling local environments into material-level features for supervised learning.
- Failure modes: cutoff sensitivity, periodic-image mistakes, polymorph aliasing, defects, and long-range physics.
- Bridge to Unit 7: once a material is represented as a vector, regression quality depends on split design and generalization logic.

## Timing guide for 90 minutes
- 0-8 min: motivation and recap from graph-based representations
- 8-22 min: composition-only and globally-pooled descriptors (Magpie, matminer, RDF, coordination stats) as the baseline tier
- 22-32 min: why baselines fail; what counts as a local environment; invariance discipline
- 32-50 min: geometric local descriptors and Voronoi ideas
- 50-68 min: ACSF and SOAP intuition
- 68-82 min: pooling, transferability, and defect examples
- 82-90 min: summary and handoff to regression/generalization

## Must-cover concepts
- Composition-only descriptor families: Magpie elemental statistics, matminer feature library
- Globally-pooled structural descriptors: RDF, partial RDF, structure-aggregated coordination
- Why these baselines are surprisingly strong yet motif-blind
- Local versus global structural representation
- Neighbor construction under periodic boundary conditions
- Descriptor invariances and continuity
- Coordination and bond-geometry features
- SOAP and ACSF at conceptual-operational level
- Pooling local descriptors to a material-level vector
- Failure modes caused by cutoffs, parser errors, and missing long-range physics

## Optional cuts if time runs short
- Keep SOAP as a conceptual descriptor and skip the finer basis-expansion discussion.
- Shorten the comparison between Voronoi and radial-cutoff neighborhoods.
- Collapse the final worked example into the summary slide.

## Exercise handoff
- Build local environments for a small crystal set.
- Compare coordination statistics with one richer descriptor family.
- Pool local descriptors into a material-level vector.
- Document one failure mode and one mitigation.

## Bridge to Unit 7
Unit 7 starts where this unit ends: after choosing a representation, the central question becomes whether a regression result reflects genuine structure-property learning or an artifact of split design, leakage, or narrow chemistry coverage.
