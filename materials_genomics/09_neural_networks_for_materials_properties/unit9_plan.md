# Unit 8 Plan — Materials Genomics

## Unit title
Neural Networks for Materials Properties

## Lecture arc
- Motivate MLP surrogates as one model class among several benchmark baselines for fixed materials features.
- Focus on when descriptor-based materials problems actually justify a neural surrogate.
- Compare composition-only, descriptor-based, and structure-enriched fixed-feature settings.
- Cover practical design choices for limited-data materials tasks: architecture scale, single-target versus multi-target outputs, and benchmark fairness.
- Center the unit on materials-specific failure modes: weak splits, domain shift, extrapolation, false confidence, and overclaiming.
- End with explicit criteria for when an NN is not the right tool and bridge into Unit 9, where representations become learned.

## Timing guide for 90 minutes
- 0-10 min: recap from Unit 7 and why another model class enters the benchmark stack
- 10-25 min: where MLP surrogates fit in materials workflows
- 25-45 min: dataset regimes, architecture choices, and target setups
- 45-65 min: fair benchmark design and comparison to ridge/RF
- 65-80 min: failure modes, extrapolation, domain shift, and trust
- 80-90 min: summary and transition to representation learning

## Must-cover concepts
- Fixed-representation neural surrogates versus learned-representation models
- When nonlinear surrogates are justified over simpler baselines
- Effective sample size in materials datasets with many near-neighbors
- Single-target and multi-target materials property prediction
- Grouped evaluation and fair baseline comparison
- Domain shift across chemistry families and databases
- Limits of calibration and trust for neural surrogates

## Optional cuts if time runs short
- Keep multi-target learning to one concise slide.
- Collapse cross-database shift and unseen-chemistry shift into one broader failure slide.
- Shorten the detailed comparison between multiple materials workflow types.

## Exercise handoff
- Build one descriptor-based neural surrogate benchmark.
- Compare it fairly to ridge and random forest on the same grouped split.
- Report not only scalar metrics but also one failure pattern under shift.
- Conclude whether the NN is justified for that task.

## Bridge to Unit 9
Unit 8 still assumes that the representation is fixed before training. Unit 9 changes the focus: the model is no longer only learning a predictor, but also learning or refining the representation itself.
