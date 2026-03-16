# Materials Genomics Unit 6 — Local Atomic Environments

## Core teaching claim
Many structure-property relations in materials are controlled locally: by the atoms around a site, their distances, their angular arrangement, and their chemistry. A good local descriptor compresses this neighborhood into a machine-readable object without discarding the physics that makes the neighborhood meaningful.

## 1. Why local environments exist as a representation layer

Global descriptors often average away the very motif that matters. Two materials can have similar stoichiometry and symmetry labels but very different local polyhedra, and those local polyhedra can control migration barriers, magnetic exchange, catalytic activity, or defect formation energies. Local atomic environments therefore sit between simple hand-built descriptors and full learned graph representations.

The representation goal is not to reproduce the entire crystal exactly. It is to preserve the part of structure that is predictive for a target property while keeping the representation stable enough for statistical learning.

## 2. What is a local atomic environment

For atom `i`, a local environment contains:
- the central species `Z_i`
- neighboring species `Z_j`
- relative positions `r_ij`
- optionally angles and higher-order relations among neighbors

The neighborhood is usually defined either by a radial cutoff `r_c` or by a geometric partition such as a Voronoi tessellation. In periodic crystals, the environment must be built with periodic images, otherwise atoms near the unit-cell boundary receive an unphysical neighborhood.

## 3. Descriptor requirements

A useful local descriptor should satisfy:
- translation invariance: shifting the whole crystal changes nothing
- rotation invariance or equivariance: orientation in the simulation cell should not change the scientific meaning
- permutation invariance: reordering identical neighbors must not change the descriptor
- continuity: a small displacement in atomic positions should induce a small change in the descriptor
- chemical sensitivity: the descriptor must distinguish species and local chemistry

If one of these properties fails, learning becomes brittle. A model may then react to file ordering, cell orientation, or parser quirks instead of physical structure.

## 4. Simple local descriptors

### Coordination number
The simplest environment summary is the coordination number

`N_i(r_c) = sum_j 1[r_ij < r_c]`

It counts how many neighbors surround atom `i` inside a chosen cutoff. This is useful because many motifs are already distinguished by coordination: tetrahedral, octahedral, square-planar, and so on.

The weakness is equally clear: the count ignores how neighbors are arranged. A tetrahedral site and a distorted square-planar site may both have coordination four.

### Bond-length statistics
A stronger local summary keeps the list or histogram of distances `{r_ij}`. Mean distance, variance, and histogram peaks can distinguish compressed from expanded environments and reveal disorder or strain.

### Bond-angle statistics
Including angles `theta_jik` adds local geometry. Octahedral, tetrahedral, and close-packed motifs become much easier to separate once angle distributions are included.

## 5. Voronoi environments

Instead of fixing a radial cutoff, one can define neighbors geometrically through a Voronoi tessellation. Two atoms are neighbors if their Voronoi cells share a face. This approach adapts to local density and often reduces arbitrary cutoff decisions.

The gain is robustness to moderate density variation. The cost is sensitivity in noisy or highly distorted structures, where very small Voronoi faces may create questionable neighbors. In practice, people often combine Voronoi logic with additional face-area or distance thresholds.

## 6. Why invariance matters

Suppose the same crystal is written in two CIF files with different axis conventions. If the descriptor changes strongly under rigid rotation, a model sees two different inputs for one physical structure. This is not data augmentation; it is representation failure.

This is why raw Cartesian coordinates are rarely used directly in classical pipelines. The descriptor must encode structure, not coordinate-system accidents.

## 7. ACSF idea

Atom-centered symmetry functions (ACSF) build local descriptors from carefully designed radial and angular basis functions. A typical radial term has the form

`G_i^rad = sum_j exp[-eta (r_ij - R_s)^2] f_c(r_ij)`

with a cutoff function `f_c` that smoothly goes to zero at `r_c`.

Angular terms add neighbor triplets and therefore carry shape information. By evaluating many such functions with different hyperparameters, one obtains a descriptor vector that is invariant and differentiable.

ACSF is attractive because each feature has a clear design purpose. The drawback is that feature design is manual and parameter choices can be tedious.

## 8. SOAP idea

SOAP, short for Smooth Overlap of Atomic Positions, replaces discrete neighbor lists by a smooth local density around each atom:

`rho_i(r) = sum_j exp(-|r - r_ij|^2 / (2 sigma^2))`

This density is expanded in radial basis functions and spherical harmonics. Instead of comparing raw coefficients, SOAP uses rotationally invariant combinations known as the power spectrum. The similarity between two environments is then computed through a normalized kernel.

Conceptually, SOAP says: two environments are similar if their smeared neighbor densities overlap strongly after accounting for rotation. This usually yields a smooth and expressive descriptor, especially for subtle distortions.

## 9. SOAP versus ACSF

ACSF is explicit, hand-designed, and often easier to interpret feature by feature. SOAP is more systematic and usually better at capturing fine geometric similarity, but it is more abstract and can be computationally heavier.

A practical rule:
- start with simpler descriptors if interpretability and speed matter most
- prefer SOAP-like descriptors when small geometric distortions matter scientifically

## 10. Pooling from atoms to materials

Local descriptors are atom-level objects `phi_i`. Many supervised tasks, however, predict one target per material, not per atom. We therefore need a pooling rule that turns `{phi_i}` into one material-level vector `Phi`.

Common choices:
- mean pooling: represents the average environment
- histograms or moments: preserve some distributional information
- species-resolved pooling: keep separate summaries per element

Pooling is not a technical afterthought. It encodes a scientific assumption about whether a property depends on a typical site, a rare motif, or the full environment distribution.

## 11. Defects and rare motifs

Local descriptors are especially useful when a property depends on a defect or on a rare local motif. A vacancy changes the coordination and bond geometry only near a few sites; a global stoichiometric descriptor may barely notice. A local representation can isolate the changed environments directly.

This is why defect classification, local phase discrimination, and atom-wise property models benefit so much from environment-based features.

## 12. Cutoff radius as a modeling choice

The cutoff `r_c` decides what the model is allowed to see. If it is too small, chemically relevant neighbors disappear. If it is too large, the descriptor becomes expensive and may mix local chemistry with weak long-range clutter.

This means `r_c` is part of the scientific model, not just a preprocessing hyperparameter. It should be chosen with chemistry and target property in mind.

## 13. Periodic images and parser failure

In crystals, atoms near a cell boundary interact with image atoms across the boundary. If the neighbor construction ignores periodic images, the computed coordination can be catastrophically wrong. This creates fake low-coordination sites and injects structured noise into the dataset.

One of the first audits for any local-descriptor pipeline is therefore simple: inspect a few boundary atoms and verify that their neighbors are physically plausible.

## 14. Transferability and aliasing

A local descriptor is transferable only if similar descriptor values correspond to genuinely similar chemistry across structure families. Problems arise when very different materials share nearly identical local motifs. This is descriptor aliasing.

For example, a local octahedral environment may appear in compounds with very different long-range connectivity. If the target depends on band dispersion or framework connectivity, a purely local descriptor can overstate similarity.

## 15. Long-range limits

Not every materials property is local. Band topology, extended magnetic order, charge transport pathways, and elastic anisotropy may depend on crystal-wide structure. In those cases, local descriptors are still useful but incomplete. They must be augmented by global descriptors or replaced by richer graph or field-based models.

This is a useful teaching boundary: local environments are powerful because they are physically meaningful, not because they are universally sufficient.

## 16. Worked example: distinguishing tetrahedral and octahedral chemistry

Consider a mixed oxide dataset. If we only record composition, two compounds may look similar. If we compute local descriptors, tetrahedrally coordinated cations separate from octahedrally coordinated ones almost immediately. Bond-angle features or SOAP similarities sharpen that separation further.

A downstream classifier can then use the local motif distribution to distinguish structure families or to predict a property correlated with local coordination.

## 17. Worked example: defect-sensitive hardness proxy

Suppose we build a simple hardness proxy from local bond density and coordination statistics. A vacancy locally stretches bonds and reduces coordination. The pooled material vector changes only slightly, but the tail of the environment distribution changes a lot. This tells us that average pooling may hide defect information, while histogram-based pooling can preserve it.

That example motivates an important modeling lesson: the pooling strategy should match the scientific mechanism.

## 18. Comparison with graph representations

Graph neural networks also start from local neighborhoods, but they learn how to aggregate them rather than fixing the descriptor by hand. Classical local descriptors therefore remain valuable for three reasons:
- they are interpretable
- they work well in small-data regimes
- they provide strong baselines and hybrid inputs

In practice, local descriptors are often the baseline a learned representation must beat.

## 19. Quality checklist before downstream learning

Before using local descriptors in regression or classification, check:
- are periodic images handled correctly?
- is the cutoff scientifically justified?
- do distortions or relaxation noise change the descriptor smoothly?
- does the pooling rule preserve the relevant motif statistics?
- does the split strategy prevent similar structures from leaking across train and test?

These checks matter as much as the choice between ACSF and SOAP.

## 20. Unit summary

The central message of Unit 6 is:
- local descriptors compress the neighborhood of an atom into a physically meaningful feature vector
- good descriptors respect invariance, continuity, and chemistry
- ACSF and SOAP are principled ways to move beyond simple counts and histograms
- pooling converts atom-level information into material-level inputs
- failure often comes from preprocessing and scientific mismatch, not only from model choice

## 21. Bridge to Unit 7

Once each material is represented by a vector, the question changes. We no longer ask how to encode structure, but whether a predictive model actually generalizes. Unit 7 therefore turns from representation to evaluation: baselines, split design, leakage, out-of-distribution behavior, and trust.
