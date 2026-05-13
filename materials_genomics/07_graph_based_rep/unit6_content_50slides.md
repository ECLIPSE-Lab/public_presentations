# Materials Genomics Unit 5 — 50-Slide Teaching Scaffold (book-backed)

## Book-backed content summary (for this unit)
- Why crystals map naturally to graph objects
- Node, edge, and global attributes in crystal graphs
- Periodic boundary conditions in graph construction
- Neighbor cutoff choices and physics implications
- Distance/angle encoding for bonding environments
- Invariance and equivariance requirements
- Message passing intuition for materials graphs
- How CGCNN represents local interactions
- How MEGNet adds global state variables
- SchNet-like continuous-filter intuition
- Pooling/readout choices for property prediction
- Graph depth vs over-smoothing tradeoff

## Source anchors used
- Sandfeld 2.2
- Neuer 4.5.1–4.5.4
- McClarren Ch8
- Bishop Ch5
- Murphy representation basics

## Essential equations / objects (lecture must-include)
- $\hat{\theta}=\arg\min_\theta \frac{1}{N}\sum_i \ell(f_\theta(x_i),y_i)+\lambda\Omega(\theta)$
- Train/validation/test with grouped split by chemistry/prototype
- Generalization gap: $R_{test}-R_{train}$
- Uncertainty decomposition: aleatoric + epistemic
- Acquisition objective (conceptual): exploration vs exploitation

## 50-slide scaffold

1. **Title: Graph-Based Crystal Representations**
- Frame the unit in the end-to-end materials discovery workflow and state the decision problems it addresses.
- Applied anchor: build graph from CIF.
- Book anchor: [Sandfeld 2.2].
2. **Learning objectives and expected outputs**
- State measurable outcomes (what students can explain, implement, and critique by the end of the unit).
- Applied anchor: compare two cutoff radii.
- Book anchor: [Neuer 4.5.1–4.5.4].
3. **Recap from previous unit and dependency map**
- Reconnect prerequisite concepts from earlier units and make dependency assumptions explicit.
- Applied anchor: descriptor MLP vs CGCNN.
- Book anchor: [McClarren Ch8].
4. **Why this unit matters for materials discovery**
- Motivate with a realistic failure/success scenario from materials discovery practice.
- Applied anchor: readout pooling ablation.
- Book anchor: [Bishop Ch5].
5. **Reading map and chapter anchors**
- Map slide blocks to the key book chapters so students can pre-read and post-review effectively.
- Applied anchor: prototype-wise generalization split.
- Book anchor: [Murphy representation basics].
6. **Why crystals map naturally to graph objects**
- Explain **why crystals map naturally to graph objects** using one concrete materials example and one common failure mode.
- Applied anchor: build graph from CIF.
- Book anchor: [Sandfeld 2.2].
7. **Node, edge, and global attributes in crystal graphs**
- Compare **node, edge, and global attributes in crystal graphs** using one concrete materials example and one common failure mode.
- Applied anchor: compare two cutoff radii.
- Book anchor: [Neuer 4.5.1–4.5.4].
8. **Periodic boundary conditions in graph construction**
- Diagnose **periodic boundary conditions in graph construction** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor MLP vs CGCNN.
- Book anchor: [McClarren Ch8].
9. **Neighbor cutoff choices and physics implications**
- Apply **neighbor cutoff choices and physics implications** using one concrete materials example and one common failure mode.
- Applied anchor: readout pooling ablation.
- Book anchor: [Bishop Ch5].
10. **Distance/angle encoding for bonding environments**
- Define **distance/angle encoding for bonding environments** using one concrete materials example and one common failure mode.
- Applied anchor: prototype-wise generalization split.
- Book anchor: [Murphy representation basics].
11. **Invariance and equivariance requirements**
- Explain **invariance and equivariance requirements** using one concrete materials example and one common failure mode.
- Applied anchor: build graph from CIF.
- Book anchor: [Sandfeld 2.2].
12. **Message passing intuition for materials graphs**
- Compare **message passing intuition for materials graphs** using one concrete materials example and one common failure mode.
- Applied anchor: compare two cutoff radii.
- Book anchor: [Neuer 4.5.1–4.5.4].
13. **How CGCNN represents local interactions**
- Diagnose **how cgcnn represents local interactions** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor MLP vs CGCNN.
- Book anchor: [McClarren Ch8].
14. **How MEGNet adds global state variables**
- Apply **how megnet adds global state variables** using one concrete materials example and one common failure mode.
- Applied anchor: readout pooling ablation.
- Book anchor: [Bishop Ch5].
15. **SchNet-like continuous-filter intuition**
- Define **schnet-like continuous-filter intuition** using one concrete materials example and one common failure mode.
- Applied anchor: prototype-wise generalization split.
- Book anchor: [Murphy representation basics].
16. **Pooling/readout choices for property prediction**
- Explain **pooling/readout choices for property prediction** using one concrete materials example and one common failure mode.
- Applied anchor: build graph from CIF.
- Book anchor: [Sandfeld 2.2].
17. **Graph depth vs over-smoothing tradeoff**
- Compare **graph depth vs over-smoothing tradeoff** using one concrete materials example and one common failure mode.
- Applied anchor: compare two cutoff radii.
- Book anchor: [Neuer 4.5.1–4.5.4].
18. **Data efficiency compared with descriptor MLP baselines**
- Diagnose **data efficiency compared with descriptor mlp baselines** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor MLP vs CGCNN.
- Book anchor: [McClarren Ch8].
19. **Handling variable-size crystal inputs**
- Apply **handling variable-size crystal inputs** using one concrete materials example and one common failure mode.
- Applied anchor: readout pooling ablation.
- Book anchor: [Bishop Ch5].
20. **Graph construction reproducibility and determinism**
- Define **graph construction reproducibility and determinism** using one concrete materials example and one common failure mode.
- Applied anchor: prototype-wise generalization split.
- Book anchor: [Murphy representation basics].
21. **Edge feature engineering beyond distances**
- Explain **edge feature engineering beyond distances** using one concrete materials example and one common failure mode.
- Applied anchor: build graph from CIF.
- Book anchor: [Sandfeld 2.2].
22. **Incorporating composition priors into graph models**
- Compare **incorporating composition priors into graph models** using one concrete materials example and one common failure mode.
- Applied anchor: compare two cutoff radii.
- Book anchor: [Neuer 4.5.1–4.5.4].
23. **Training stability and batch construction issues**
- Diagnose **training stability and batch construction issues** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor MLP vs CGCNN.
- Book anchor: [McClarren Ch8].
24. **Computational cost on large datasets**
- Apply **computational cost on large datasets** using one concrete materials example and one common failure mode.
- Applied anchor: readout pooling ablation.
- Book anchor: [Bishop Ch5].
25. **Interpretability in graph models (attention/sensitivity)**
- Define **interpretability in graph models (attention/sensitivity)** using one concrete materials example and one common failure mode.
- Applied anchor: prototype-wise generalization split.
- Book anchor: [Murphy representation basics].
26. **Failure mode: shortcut learning from cell-size proxies**
- Explain **failure mode: shortcut learning from cell-size proxies** using one concrete materials example and one common failure mode.
- Applied anchor: build graph from CIF.
- Book anchor: [Sandfeld 2.2].
27. **Failure mode: cutoff choice causing nonphysical neighbors**
- Compare **failure mode: cutoff choice causing nonphysical neighbors** using one concrete materials example and one common failure mode.
- Applied anchor: compare two cutoff radii.
- Book anchor: [Neuer 4.5.1–4.5.4].
28. **Transferability across chemistry families**
- Diagnose **transferability across chemistry families** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor MLP vs CGCNN.
- Book anchor: [McClarren Ch8].
29. **OOD behavior for unseen prototypes**
- Apply **ood behavior for unseen prototypes** using one concrete materials example and one common failure mode.
- Applied anchor: readout pooling ablation.
- Book anchor: [Bishop Ch5].
30. **Baseline comparison protocol: descriptor vs graph**
- Define **baseline comparison protocol: descriptor vs graph** using one concrete materials example and one common failure mode.
- Applied anchor: prototype-wise generalization split.
- Book anchor: [Murphy representation basics].
31. **Evaluation metrics for regression and ranking**
- Explain **evaluation metrics for regression and ranking** using one concrete materials example and one common failure mode.
- Applied anchor: build graph from CIF.
- Book anchor: [Sandfeld 2.2].
32. **Uncertainty with graph ensembles (preview)**
- Compare **uncertainty with graph ensembles (preview)** using one concrete materials example and one common failure mode.
- Applied anchor: compare two cutoff radii.
- Book anchor: [Neuer 4.5.1–4.5.4].
33. **Case: bandgap from crystal graph**
- Diagnose **case: bandgap from crystal graph** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor MLP vs CGCNN.
- Book anchor: [McClarren Ch8].
34. **Case: formation energy from graph representation**
- Apply **case: formation energy from graph representation** using one concrete materials example and one common failure mode.
- Applied anchor: readout pooling ablation.
- Book anchor: [Bishop Ch5].
35. **Case: elasticity prediction under limited data**
- Define **case: elasticity prediction under limited data** using one concrete materials example and one common failure mode.
- Applied anchor: prototype-wise generalization split.
- Book anchor: [Murphy representation basics].
36. **How graph embeddings feed Unit 9 representation learning**
- Explain **how graph embeddings feed unit 9 representation learning** using one concrete materials example and one common failure mode.
- Applied anchor: build graph from CIF.
- Book anchor: [Sandfeld 2.2].
37. **How this unit connects to local environments in Unit 6**
- Compare **how this unit connects to local environments in unit 6** using one concrete materials example and one common failure mode.
- Applied anchor: compare two cutoff radii.
- Book anchor: [Neuer 4.5.1–4.5.4].
38. **Advanced note: Graph-Based Crystal Representations concept extension 33**
- Diagnose **advanced note: graph-based crystal representations concept extension 33** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor MLP vs CGCNN.
- Book anchor: [McClarren Ch8].
39. **Advanced note: Graph-Based Crystal Representations concept extension 34**
- Apply **advanced note: graph-based crystal representations concept extension 34** using one concrete materials example and one common failure mode.
- Applied anchor: readout pooling ablation.
- Book anchor: [Bishop Ch5].
40. **Advanced note: Graph-Based Crystal Representations concept extension 35**
- Define **advanced note: graph-based crystal representations concept extension 35** using one concrete materials example and one common failure mode.
- Applied anchor: prototype-wise generalization split.
- Book anchor: [Murphy representation basics].
41. **Advanced note: Graph-Based Crystal Representations concept extension 36**
- Explain **advanced note: graph-based crystal representations concept extension 36** using one concrete materials example and one common failure mode.
- Applied anchor: build graph from CIF.
- Book anchor: [Sandfeld 2.2].
42. **Advanced note: Graph-Based Crystal Representations concept extension 37**
- Compare **advanced note: graph-based crystal representations concept extension 37** using one concrete materials example and one common failure mode.
- Applied anchor: compare two cutoff radii.
- Book anchor: [Neuer 4.5.1–4.5.4].
43. **Advanced note: Graph-Based Crystal Representations concept extension 38**
- Diagnose **advanced note: graph-based crystal representations concept extension 38** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor MLP vs CGCNN.
- Book anchor: [McClarren Ch8].
44. **Advanced note: Graph-Based Crystal Representations concept extension 39**
- Apply **advanced note: graph-based crystal representations concept extension 39** using one concrete materials example and one common failure mode.
- Applied anchor: readout pooling ablation.
- Book anchor: [Bishop Ch5].
45. **Exercise setup and dataset definition**
- Define dataset, split protocol, and expected deliverables before any coding begins.
- Applied anchor: prototype-wise generalization split.
- Book anchor: [Murphy representation basics].
46. **Exercise task 1 (pipeline core)**
- Implement the core pipeline component with reproducible settings and documented assumptions.
- Applied anchor: build graph from CIF.
- Book anchor: [Sandfeld 2.2].
47. **Exercise task 2 (comparison/ablation)**
- Run an ablation/comparison under identical validation protocol and interpret differences.
- Applied anchor: compare two cutoff radii.
- Book anchor: [Neuer 4.5.1–4.5.4].
48. **Exercise task 3 (failure analysis)**
- Perform structured failure analysis and propose one evidence-backed mitigation.
- Applied anchor: descriptor MLP vs CGCNN.
- Book anchor: [McClarren Ch8].
49. **Exam-oriented key statements**
- Summarize high-yield statements in concise written-exam style with definitions and caveats.
- Applied anchor: readout pooling ablation.
- Book anchor: [Bishop Ch5].
50. **Summary, next-unit bridge, and references**
- Consolidate the unit into a checklist: concepts, pitfalls, and decisions for next-unit transfer.
- Applied anchor: prototype-wise generalization split.
- Book anchor: [Murphy representation basics].

