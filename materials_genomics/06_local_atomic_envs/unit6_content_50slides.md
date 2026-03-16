# Materials Genomics Unit 6 — 50-Slide Teaching Scaffold (book-backed)

## Book-backed content summary (for this unit)
- Local vs global representation of crystal information
- Coordination number and neighbor shells
- Bond-length and bond-angle distributions
- Voronoi tessellation intuition for local structure
- SOAP descriptor concept and kernel view
- ACSF and related atom-centered features
- Symmetry functions and rotational invariance
- Environment fingerprints for phase discrimination
- Aggregating local descriptors to material-level vectors
- Histogram and moment summaries of environments
- Local environments for defect characterization
- Sensitivity to noise in atomic positions

## Source anchors used
- Sandfeld 2.2
- Neuer 6.2
- Neuer 6.3
- McClarren Ch4
- Bishop kernels/feature maps

## Essential equations / objects (lecture must-include)
- Coordination number $N_i(r_c)$ under a chosen cutoff radius
- Local environment descriptor $\phi_i$ and pooled material vector $\Phi=\text{pool}(\{\phi_i\})$
- SOAP-style similarity as a normalized kernel between local environments
- Grouped train/validation/test split by chemistry or prototype family
- Periodic-neighbor construction under minimum-image or cell-expansion logic

## 50-slide scaffold

1. **Title: Local Atomic Environments**
- Frame the unit in the end-to-end materials discovery workflow and state the decision problems it addresses.
- Applied anchor: compute coordination distribution.
- Book anchor: [Sandfeld 2.2].
2. **Learning objectives and expected outputs**
- State measurable outcomes (what students can explain, implement, and critique by the end of the unit).
- Applied anchor: SOAP vector extraction.
- Book anchor: [Neuer 6.2].
3. **Recap from previous unit and dependency map**
- Reconnect prerequisite concepts from earlier units and make dependency assumptions explicit.
- Applied anchor: cluster local motifs.
- Book anchor: [Neuer 6.3].
4. **Why this unit matters for materials discovery**
- Motivate with a realistic failure/success scenario from materials discovery practice.
- Applied anchor: aggregate motif histograms.
- Book anchor: [McClarren Ch4].
5. **Reading map and chapter anchors**
- Map slide blocks to the key book chapters so students can pre-read and post-review effectively.
- Applied anchor: regression with motif features.
- Book anchor: [Bishop kernels/feature maps].
6. **Local vs global representation of crystal information**
- Explain **local vs global representation of crystal information** using one concrete materials example and one common failure mode.
- Applied anchor: compute coordination distribution.
- Book anchor: [Sandfeld 2.2].
7. **Coordination number and neighbor shells**
- Compare **coordination number and neighbor shells** using one concrete materials example and one common failure mode.
- Applied anchor: SOAP vector extraction.
- Book anchor: [Neuer 6.2].
8. **Bond-length and bond-angle distributions**
- Diagnose **bond-length and bond-angle distributions** using one concrete materials example and one common failure mode.
- Applied anchor: cluster local motifs.
- Book anchor: [Neuer 6.3].
9. **Voronoi tessellation intuition for local structure**
- Apply **voronoi tessellation intuition for local structure** using one concrete materials example and one common failure mode.
- Applied anchor: aggregate motif histograms.
- Book anchor: [McClarren Ch4].
10. **SOAP descriptor concept and kernel view**
- Define **soap descriptor concept and kernel view** using one concrete materials example and one common failure mode.
- Applied anchor: regression with motif features.
- Book anchor: [Bishop kernels/feature maps].
11. **ACSF and related atom-centered features**
- Explain **acsf and related atom-centered features** using one concrete materials example and one common failure mode.
- Applied anchor: compute coordination distribution.
- Book anchor: [Sandfeld 2.2].
12. **Symmetry functions and rotational invariance**
- Compare **symmetry functions and rotational invariance** using one concrete materials example and one common failure mode.
- Applied anchor: SOAP vector extraction.
- Book anchor: [Neuer 6.2].
13. **Environment fingerprints for phase discrimination**
- Diagnose **environment fingerprints for phase discrimination** using one concrete materials example and one common failure mode.
- Applied anchor: cluster local motifs.
- Book anchor: [Neuer 6.3].
14. **Aggregating local descriptors to material-level vectors**
- Apply **aggregating local descriptors to material-level vectors** using one concrete materials example and one common failure mode.
- Applied anchor: aggregate motif histograms.
- Book anchor: [McClarren Ch4].
15. **Histogram and moment summaries of environments**
- Define **histogram and moment summaries of environments** using one concrete materials example and one common failure mode.
- Applied anchor: regression with motif features.
- Book anchor: [Bishop kernels/feature maps].
16. **Local environments for defect characterization**
- Explain **local environments for defect characterization** using one concrete materials example and one common failure mode.
- Applied anchor: compute coordination distribution.
- Book anchor: [Sandfeld 2.2].
17. **Sensitivity to noise in atomic positions**
- Compare **sensitivity to noise in atomic positions** using one concrete materials example and one common failure mode.
- Applied anchor: SOAP vector extraction.
- Book anchor: [Neuer 6.2].
18. **Preprocessing choices: normalization and cutoff selection**
- Diagnose **preprocessing choices: normalization and cutoff selection** using one concrete materials example and one common failure mode.
- Applied anchor: cluster local motifs.
- Book anchor: [Neuer 6.3].
19. **Computational scaling of local descriptor extraction**
- Apply **computational scaling of local descriptor extraction** using one concrete materials example and one common failure mode.
- Applied anchor: aggregate motif histograms.
- Book anchor: [McClarren Ch4].
20. **Interpretability benefits of local descriptors**
- Define **interpretability benefits of local descriptors** using one concrete materials example and one common failure mode.
- Applied anchor: regression with motif features.
- Book anchor: [Bishop kernels/feature maps].
21. **Comparing SOAP-like descriptors to graph embeddings**
- Explain **comparing soap-like descriptors to graph embeddings** using one concrete materials example and one common failure mode.
- Applied anchor: compute coordination distribution.
- Book anchor: [Sandfeld 2.2].
22. **Using local environments as hybrid model inputs**
- Compare **using local environments as hybrid model inputs** using one concrete materials example and one common failure mode.
- Applied anchor: SOAP vector extraction.
- Book anchor: [Neuer 6.2].
23. **Kernel methods on local descriptors**
- Diagnose **kernel methods on local descriptors** using one concrete materials example and one common failure mode.
- Applied anchor: cluster local motifs.
- Book anchor: [Neuer 6.3].
24. **Transferability across crystal families**
- Apply **transferability across crystal families** using one concrete materials example and one common failure mode.
- Applied anchor: aggregate motif histograms.
- Book anchor: [McClarren Ch4].
25. **OOD detection via environment distribution shifts**
- Define **ood detection via environment distribution shifts** using one concrete materials example and one common failure mode.
- Applied anchor: regression with motif features.
- Book anchor: [Bishop kernels/feature maps].
26. **Failure mode: nonphysical neighborhoods from parser errors**
- Explain **failure mode: nonphysical neighborhoods from parser errors** using one concrete materials example and one common failure mode.
- Applied anchor: compute coordination distribution.
- Book anchor: [Sandfeld 2.2].
27. **Failure mode: aliasing environments across polymorphs**
- Compare **failure mode: aliasing environments across polymorphs** using one concrete materials example and one common failure mode.
- Applied anchor: SOAP vector extraction.
- Book anchor: [Neuer 6.2].
28. **Descriptor compression and PCA for environment vectors**
- Diagnose **descriptor compression and pca for environment vectors** using one concrete materials example and one common failure mode.
- Applied anchor: cluster local motifs.
- Book anchor: [Neuer 6.3].
29. **Clustering local motifs before supervised modeling**
- Apply **clustering local motifs before supervised modeling** using one concrete materials example and one common failure mode.
- Applied anchor: aggregate motif histograms.
- Book anchor: [McClarren Ch4].
30. **Uncertainty estimates from environment variability**
- Define **uncertainty estimates from environment variability** using one concrete materials example and one common failure mode.
- Applied anchor: regression with motif features.
- Book anchor: [Bishop kernels/feature maps].
31. **Case: local motifs vs hardness proxy**
- Explain **case: local motifs vs hardness proxy** using one concrete materials example and one common failure mode.
- Applied anchor: compute coordination distribution.
- Book anchor: [Sandfeld 2.2].
32. **Case: identifying coordination-driven anomalies**
- Compare **case: identifying coordination-driven anomalies** using one concrete materials example and one common failure mode.
- Applied anchor: SOAP vector extraction.
- Book anchor: [Neuer 6.2].
33. **Case: structure family separation in environment space**
- Diagnose **case: structure family separation in environment space** using one concrete materials example and one common failure mode.
- Applied anchor: cluster local motifs.
- Book anchor: [Neuer 6.3].
34. **Cutoff radius as a scientific modeling choice**
- Show how changing $r_c$ changes coordination counts, neighbor identity, and descriptor stability.
- Applied anchor: coordination histogram under multiple cutoffs.
- Book anchor: [Neuer 6.2].
35. **Periodic images and minimum-image conventions**
- Explain how periodic boundary handling changes local neighborhoods in crystals.
- Applied anchor: neighbor list across cell boundaries.
- Book anchor: [Sandfeld 2.2].
36. **Defects, vacancies, and distorted local motifs**
- Use a defect example to show where local descriptors are more sensitive than global summaries.
- Applied anchor: vacancy-induced coordination change.
- Book anchor: [Neuer 6.3].
37. **When local descriptors fail**
- Discuss long-range physics, charge transfer, and global topology as limits of strictly local fingerprints.
- Applied anchor: same local motif, different bulk context.
- Book anchor: [McClarren Ch4].
38. **Choosing between coordination stats, ACSF, and SOAP**
- Compare the three by interpretability, cost, and transferability.
- Applied anchor: side-by-side descriptor choice table.
- Book anchor: [Bishop kernels/feature maps].
39. **Pooling local descriptors to one material vector**
- Mean, histogram, and attention-like pooling as distinct scientific assumptions.
- Applied anchor: pooled representation for a small crystal family.
- Book anchor: [Sandfeld 2.2].
40. **Environment-space visualization**
- Use PCA or UMAP conceptually to inspect motif clusters without overselling the picture.
- Applied anchor: motif cluster plot.
- Book anchor: [Neuer 6.2].
41. **Quality checklist for environment features**
- Verify parser quality, periodicity, cutoff, normalization, and split strategy before modeling.
- Applied anchor: pre-regression audit checklist.
- Book anchor: [Neuer 6.3].
42. **Bridge to Unit 7 regression**
- Show how local-environment vectors become supervised learning inputs for property prediction.
- Applied anchor: local descriptor to target table.
- Book anchor: [McClarren Ch4].
43. **Exercise setup and dataset definition**
- Define dataset, split protocol, deliverables, and one explicit failure-analysis requirement.
- Applied anchor: coordination-plus-SOAP exercise brief.
- Book anchor: [Sandfeld 2.2].
44. **Exercise task 1: build the neighborhood pipeline**
- Construct neighbors, choose cutoffs, and compute one simple local descriptor family.
- Applied anchor: coordination distribution.
- Book anchor: [Neuer 6.2].
45. **Exercise task 2: compute a richer fingerprint**
- Add SOAP or ACSF-like features and document the computational tradeoff.
- Applied anchor: SOAP vector extraction.
- Book anchor: [Bishop kernels/feature maps].
46. **Exercise task 3: aggregate to material level**
- Pool local environments to a single vector and justify the pooling rule.
- Applied anchor: motif histogram aggregation.
- Book anchor: [Sandfeld 2.2].
47. **Exercise task 4: compare stable vs noisy structures**
- Compare descriptor robustness under perturbation, relaxation noise, or a defect.
- Applied anchor: noisy structure ablation.
- Book anchor: [Neuer 6.3].
48. **Exercise task 5: short failure analysis**
- Identify one descriptor failure mode and propose an evidence-backed mitigation.
- Applied anchor: aliasing across polymorphs.
- Book anchor: [McClarren Ch4].
49. **Exam-oriented key statements**
- Summarize high-yield statements in concise written-exam style with definitions and caveats.
- Applied anchor: aggregate motif histograms.
- Book anchor: [McClarren Ch4].
50. **Summary, next-unit bridge, and references**
- Consolidate the unit into a checklist: concepts, pitfalls, and decisions for next-unit transfer.
- Applied anchor: regression with motif features.
- Book anchor: [Bishop kernels/feature maps].
