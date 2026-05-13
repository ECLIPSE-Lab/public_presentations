# Materials Genomics Unit 8 — Neural Networks for Materials Properties

## Core teaching claim
In Materials Genomics, feed-forward neural networks are not introduced as a new mathematical foundation. That foundation already exists in MFML. Here they are treated as a practical surrogate-model class that must earn its place against ridge regression and tree ensembles under realistic materials-data constraints.

## 1. Why this unit exists after Unit 7

Unit 7 established the discipline of honest benchmarking: grouped splits, residual analysis, and skepticism toward leaderboard scores. Unit 8 adds one more model family to that benchmark stack. The point is not that neural networks are automatically better. The point is that some materials-property relations are nonlinear enough that a flexible surrogate may capture signal that simpler baselines miss.

This means the unit begins with a methodological question: when is an MLP a scientifically justified next step?

## 2. Neural surrogates on fixed materials features

This unit assumes the input representation is already fixed:
- composition vectors
- tabular descriptors derived from crystal structure
- pooled local-environment features
- other engineered or precomputed materials features

That is an important boundary. We are not yet learning the representation itself. We are only replacing the downstream predictor.

## 3. Why not jump directly to a neural network

There are several reasons not to start with an MLP:
- materials datasets are often small relative to model flexibility
- engineered descriptors may already linearize much of the useful signal
- neural surrogates introduce more hyperparameters and more unstable training outcomes
- higher flexibility makes weak validation more dangerous

Therefore, the burden of proof lies with the neural surrogate.

## 4. When an MLP is plausible

An MLP becomes plausible when:
- the target depends on nonlinear interactions between descriptors
- simpler baselines underfit in a systematic way
- the dataset is large enough, or rich enough, to support added flexibility
- the use case tolerates the added complexity and reduced interpretability

The decision is empirical, but it should be guided by the data regime and the scientific task.

## 5. Composition-only versus structure-enriched inputs

The same MLP architecture can behave very differently depending on the input:
- with composition-only features, the model may mostly interpolate within chemistry families
- with structure-enriched descriptors, it may capture more of the local structural signal
- with poor descriptors, extra nonlinearity may simply fit noise

So the network does not rescue a weak representation automatically.

## 6. Neural surrogates as approximations to expensive computation

In materials workflows, the most natural role for an MLP is often as a surrogate for something expensive:
- DFT property prediction
- high-throughput screening filters
- approximate inverse models for process-property tasks

This perspective matters because the relevant comparison is not only error minimization. It is speed, robustness, transferability, and whether the surrogate fails safely.

## 7. Architecture choice in limited-data regimes

For descriptor-based materials tasks, the main architecture decision is often modest:
- how many hidden layers
- how wide they should be
- whether the output is single-target or multi-target

Very deep networks are often unjustified here. Limited data and fixed descriptors usually favor shallow or moderately deep MLPs over large architectures.

## 8. Capacity should match dataset regime

- If the dataset contains only a few thousand materials and many are closely related, large capacity is dangerous.
- If the descriptor is already high-level and physically informed, a small network may be enough.
- If the target is heterogeneous and the descriptor misses key physics, even a larger network may not help.

Capacity should therefore match effective sample size, not just raw row count.

## 9. Effective sample size is smaller than it looks

Materials datasets often contain many close relatives:
- compounds in the same chemistry family
- polymorphs with minor variations
- near-duplicate structures from the same workflow

A dataset with ten thousand rows may behave statistically more like a much smaller dataset once these correlations are accounted for. This is why neural models can appear well-supported while actually seeing very little genuinely independent information.

## 10. Single-target and multi-target prediction

- A single-target model predicts one property such as band gap or modulus.
- A multi-target model predicts several coupled properties at once.

Multi-target learning is useful when targets share physical drivers, but harmful when weakly related targets force the network to compromise. The relevant question is whether shared structure exists in the outputs, not whether multitask learning sounds modern.

## 11. Example: coupled elastic properties

For bulk, shear, and Young's moduli, a shared hidden representation may help because the targets are related mechanically. For loosely related targets such as band gap and ionic conductivity, a shared network may create negative transfer.

The lesson is simple: multi-target learning is a materials hypothesis, not a generic default.

## 12. Target scaling and problem framing

- Some materials targets span narrow ranges; others span orders of magnitude.
- Log transforms, normalization, or family-specific scaling can change optimization behavior dramatically.
- The choice should be driven by target physics and interpretability, not only by training convenience.

A neural surrogate is sensitive to these framing decisions because it fits the transformed target, not the raw scientific quantity in the abstract.

## 13. Benchmark discipline carries over from Unit 7

Everything from Unit 7 still applies:
- the split defines the claim
- the baseline comparison must be fair
- the metric must match the scientific use case
- residual structure matters more than one scalar score

Unit 8 does not relax those rules. It makes them stricter, because flexible models exploit weak protocols more easily.

## 14. Comparing MLPs to ridge and random forest

The correct benchmark is not "neural net versus nothing." It is:
- ridge as the strong linear baseline
- random forest as a robust nonlinear non-neural baseline
- MLP as the flexible fixed-feature neural baseline

If the MLP wins only on a weak split or only by a negligible margin, its scientific value is limited.

## 15. Grouped evaluation matters even more here

If similar compounds appear in both train and test sets, an MLP may appear remarkably strong because it interpolates smoothly between near neighbors. Under grouped chemistry-aware or prototype-aware splits, the same model may collapse.

That collapse is not a failure of the metric. It is evidence that the random-split claim was too optimistic.

## 16. Example: band-gap benchmark under two splits

Imagine an MLP trained on descriptor vectors for semiconductor compounds:
- under a random split, the test MAE may look excellent
- under a grouped split by chemistry family, the same network can lose much of its apparent advantage

This is the standard cautionary tale for neural surrogates in materials datasets.

## 17. Domain shift across databases

Neural surrogates can also fail when trained and tested on different databases because:
- DFT functionals differ
- relaxation criteria differ
- curation rules differ
- chemical coverage differs

What looked like a good surrogate inside one database may partly be a surrogate for that database's conventions.

## 18. Extrapolation beyond seen chemistry

MLPs are powerful interpolators, but they are rarely trustworthy extrapolators.

In materials discovery this matters because the practical question is often: what happens on a chemistry family we have not seen before? The answer is usually much less favorable than within-domain validation suggests.

## 19. False confidence is a central risk

Neural surrogates often output a precise number even when the input lies far outside training support. That creates false confidence.

The scientific danger is not only being wrong. It is being wrong while looking certain enough to guide expensive experiments or simulations in the wrong direction.

## 20. Calibration is limited for plain MLP surrogates

Without additional uncertainty machinery, an MLP typically gives point predictions rather than calibrated uncertainty estimates. This means:
- a small residual on average does not imply reliable confidence
- badly shifted inputs may still produce smooth, plausible-looking outputs
- uncertainty awareness must be handled explicitly, not assumed

This is one reason Unit 12 exists later in the course.

## 21. Overfitting can hide behind weak splits

With fixed descriptor inputs, an MLP may memorize subtle family-specific correlations that look predictive only because train and test remain too similar. This is particularly dangerous when the effective sample size is small but the nominal feature dimension is high.

In such settings, "the network learned a complex nonlinear relationship" may really mean "the network fit a narrow local pattern."

## 22. When not to use a neural surrogate

Do not prefer an MLP when:
- ridge or RF already performs just as well under the relevant split
- the dataset is too small or too correlated
- interpretability is central and the accuracy gain is negligible
- the deployment domain is strongly shifted from training support

Complexity needs a scientific payoff.

## 23. A good use case for MLPs

A good use case looks like this:
- a descriptor-rich dataset with enough diversity
- evidence that linear and tree baselines miss structured signal
- a grouped validation protocol aligned with the intended deployment claim
- a practical need for fast nonlinear surrogate predictions

Then an MLP is not just fashionable; it is justified.

## 24. Hybrid value of physically informed descriptors

One strength of Materials Genomics is that fixed descriptors can already encode physical judgment. An MLP on top of these features can then model nonlinear interactions without throwing domain knowledge away.

This hybrid setup is often stronger than either extreme:
- purely manual linear modeling
- fully end-to-end learning without enough data

## 25. Example: pooled local environments to neural surrogate

Suppose Unit 6 produced pooled local-environment descriptors and Unit 7 showed that ridge captures only part of the band-gap signal. A small MLP may then be a natural next baseline because it can model interactions among motif statistics without requiring a full graph model.

This is exactly the kind of progression the course should make explicit.

## 26. Example: when RF still wins

If the dataset is small, tabular, and irregular, a random forest may remain the better choice:
- less sensitive to scaling
- robust in low-data regimes
- easier to tune under strict validation

This slide matters because it prevents the unit from becoming a pro-NN sales pitch.

## 27. Model selection should match deployment

Ask what the surrogate will actually be used for:
- ranking candidates inside a known family
- screening across new chemistries
- replacing expensive simulation inside an optimization loop

The same MLP can be acceptable for the first use case and irresponsible for the second.

## 28. Trust checklist for neural surrogates

Before trusting a descriptor-based MLP, check:
- does it beat strong baselines under the right split?
- does the gain persist across chemistry families?
- do residuals remain acceptable for important cases?
- is there evidence against leakage and database-specific artifacts?
- is the deployment domain close enough to training support?

This checklist is more important than one extra decimal in MAE.

## 29. Unit summary

The central message of Unit 8 is:
- MLPs are one candidate surrogate class for fixed materials representations
- they should be judged against strong simpler baselines
- effective sample size and split design dominate many apparent gains
- domain shift, extrapolation, and false confidence are the real scientific risks
- complexity is justified only when it survives honest evaluation

## 30. Bridge to Unit 9

Unit 8 still assumes that the representation is fixed before training. Unit 9 changes the modeling problem itself: instead of only learning a predictor on top of descriptors, we ask how the representation can be learned or refined from data.
