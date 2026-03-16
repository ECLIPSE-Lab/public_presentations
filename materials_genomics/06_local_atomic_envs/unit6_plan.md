# Unit 6 Plan — Materials Genomics

## Unit title
Local Atomic Environments

## Lecture arc
- Why materials ML needs atom-centered structure information instead of only one global fingerprint.
- What a valid local descriptor must preserve: translation, rotation, permutation, continuity, and chemical identity.
- Simple environment summaries: coordination number, bond lengths, bond angles, and Voronoi coordination.
- Richer descriptors: ACSF and SOAP as systematic encodings of local neighborhoods.
- Pooling local environments into material-level features for supervised learning.
- Failure modes: cutoff sensitivity, periodic-image mistakes, polymorph aliasing, defects, and long-range physics.
- Bridge to Unit 7: once a material is represented as a vector, regression quality depends on split design and generalization logic.

## Timing guide for 90 minutes
- 0-10 min: motivation and recap from graph-based representations
- 10-25 min: what counts as a local environment and why invariances matter
- 25-45 min: geometric descriptors and Voronoi ideas
- 45-65 min: ACSF and SOAP intuition
- 65-80 min: pooling, transferability, and defect examples
- 80-90 min: summary and handoff to regression/generalization

## Must-cover concepts
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
