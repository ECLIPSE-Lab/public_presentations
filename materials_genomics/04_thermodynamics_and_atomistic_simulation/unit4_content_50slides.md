# Materials Genomics Unit 4 — 50-Slide Teaching Scaffold (book-backed)

## Book-backed content summary (for this unit)
- Descriptor purpose: encode chemistry/structure into ML-ready vectors
- Composition descriptors: elemental statistics and stoichiometric moments
- Magpie-style feature families and what they capture
- matminer descriptor ecosystem and practical tradeoffs
- Structure descriptors: radial/angle summaries and coordination stats
- Process descriptors as context variables in PSPP pipelines
- Physical invariances required in descriptor design
- Scaling and normalization across heterogeneous feature groups
- Interpretability strengths of hand-crafted features
- Where classical descriptors fail: nonlocal interactions
- Feature sparsity and high-dimensional curse in materials spaces
- Correlation and multicollinearity in descriptor tables

## Source anchors used
- Sandfeld 2.1–2.2
- Neuer 5.5
- McClarren Ch5
- Murphy Ch1
- Bishop 12.1–12.3

## Essential equations / objects (lecture must-include)
- $\hat{\theta}=\arg\min_\theta \frac{1}{N}\sum_i \ell(f_\theta(x_i),y_i)+\lambda\Omega(\theta)$
- Train/validation/test with grouped split by chemistry/prototype
- Generalization gap: $R_{test}-R_{train}$
- Uncertainty decomposition: aleatoric + epistemic
- Acquisition objective (conceptual): exploration vs exploitation

## 50-slide scaffold

1. **Title: From Classical Descriptors to Learned Representations**
- Frame the unit in the end-to-end materials discovery workflow and state the decision problems it addresses.
- Applied anchor: matminer composition baseline.
- Book anchor: [Sandfeld 2.1–2.2].
2. **Learning objectives and expected outputs**
- State measurable outcomes (what students can explain, implement, and critique by the end of the unit).
- Applied anchor: descriptor ablation table.
- Book anchor: [Neuer 5.5].
3. **Recap from previous unit and dependency map**
- Reconnect prerequisite concepts from earlier units and make dependency assumptions explicit.
- Applied anchor: feature scaling comparison.
- Book anchor: [McClarren Ch5].
4. **Why this unit matters for materials discovery**
- Motivate with a realistic failure/success scenario from materials discovery practice.
- Applied anchor: L1-selected sparse descriptor set.
- Book anchor: [Murphy Ch1].
5. **Reading map and chapter anchors**
- Map slide blocks to the key book chapters so students can pre-read and post-review effectively.
- Applied anchor: baseline vs learned embedding comparison.
- Book anchor: [Bishop 12.1–12.3].
6. **Descriptor purpose: encode chemistry/structure into ML-ready vectors**
- Explain **descriptor purpose: encode chemistry/structure into ml-ready vectors** using one concrete materials example and one common failure mode.
- Applied anchor: matminer composition baseline.
- Book anchor: [Sandfeld 2.1–2.2].
7. **Composition descriptors: elemental statistics and stoichiometric moments**
- Compare **composition descriptors: elemental statistics and stoichiometric moments** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor ablation table.
- Book anchor: [Neuer 5.5].
8. **Magpie-style feature families and what they capture**
- Diagnose **magpie-style feature families and what they capture** using one concrete materials example and one common failure mode.
- Applied anchor: feature scaling comparison.
- Book anchor: [McClarren Ch5].
9. **matminer descriptor ecosystem and practical tradeoffs**
- Apply **matminer descriptor ecosystem and practical tradeoffs** using one concrete materials example and one common failure mode.
- Applied anchor: L1-selected sparse descriptor set.
- Book anchor: [Murphy Ch1].
10. **Structure descriptors: radial/angle summaries and coordination stats**
- Define **structure descriptors: radial/angle summaries and coordination stats** using one concrete materials example and one common failure mode.
- Applied anchor: baseline vs learned embedding comparison.
- Book anchor: [Bishop 12.1–12.3].
11. **Process descriptors as context variables in PSPP pipelines**
- Explain **process descriptors as context variables in pspp pipelines** using one concrete materials example and one common failure mode.
- Applied anchor: matminer composition baseline.
- Book anchor: [Sandfeld 2.1–2.2].
12. **Physical invariances required in descriptor design**
- Compare **physical invariances required in descriptor design** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor ablation table.
- Book anchor: [Neuer 5.5].
13. **Scaling and normalization across heterogeneous feature groups**
- Diagnose **scaling and normalization across heterogeneous feature groups** using one concrete materials example and one common failure mode.
- Applied anchor: feature scaling comparison.
- Book anchor: [McClarren Ch5].
14. **Interpretability strengths of hand-crafted features**
- Apply **interpretability strengths of hand-crafted features** using one concrete materials example and one common failure mode.
- Applied anchor: L1-selected sparse descriptor set.
- Book anchor: [Murphy Ch1].
15. **Where classical descriptors fail: nonlocal interactions**
- Define **where classical descriptors fail: nonlocal interactions** using one concrete materials example and one common failure mode.
- Applied anchor: baseline vs learned embedding comparison.
- Book anchor: [Bishop 12.1–12.3].
16. **Feature sparsity and high-dimensional curse in materials spaces**
- Explain **feature sparsity and high-dimensional curse in materials spaces** using one concrete materials example and one common failure mode.
- Applied anchor: matminer composition baseline.
- Book anchor: [Sandfeld 2.1–2.2].
17. **Correlation and multicollinearity in descriptor tables**
- Compare **correlation and multicollinearity in descriptor tables** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor ablation table.
- Book anchor: [Neuer 5.5].
18. **Filter, wrapper, and embedded feature selection strategies**
- Diagnose **filter, wrapper, and embedded feature selection strategies** using one concrete materials example and one common failure mode.
- Applied anchor: feature scaling comparison.
- Book anchor: [McClarren Ch5].
19. **Feature leakage introduced during preprocessing**
- Apply **feature leakage introduced during preprocessing** using one concrete materials example and one common failure mode.
- Applied anchor: L1-selected sparse descriptor set.
- Book anchor: [Murphy Ch1].
20. **Target-aware descriptor engineering pitfalls**
- Define **target-aware descriptor engineering pitfalls** using one concrete materials example and one common failure mode.
- Applied anchor: baseline vs learned embedding comparison.
- Book anchor: [Bishop 12.1–12.3].
21. **Data regime analysis: small-data vs medium-data descriptor choices**
- Explain **data regime analysis: small-data vs medium-data descriptor choices** using one concrete materials example and one common failure mode.
- Applied anchor: matminer composition baseline.
- Book anchor: [Sandfeld 2.1–2.2].
22. **When to keep descriptor baselines as scientific controls**
- Compare **when to keep descriptor baselines as scientific controls** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor ablation table.
- Book anchor: [Neuer 5.5].
23. **Transition criteria to learned representations**
- Diagnose **transition criteria to learned representations** using one concrete materials example and one common failure mode.
- Applied anchor: feature scaling comparison.
- Book anchor: [McClarren Ch5].
24. **Representation learning objective: task performance + transferability**
- Apply **representation learning objective: task performance + transferability** using one concrete materials example and one common failure mode.
- Applied anchor: L1-selected sparse descriptor set.
- Book anchor: [Murphy Ch1].
25. **Autoencoder intuition as nonlinear descriptor learner**
- Define **autoencoder intuition as nonlinear descriptor learner** using one concrete materials example and one common failure mode.
- Applied anchor: baseline vs learned embedding comparison.
- Book anchor: [Bishop 12.1–12.3].
26. **Latent feature semantics vs black-box embeddings**
- Explain **latent feature semantics vs black-box embeddings** using one concrete materials example and one common failure mode.
- Applied anchor: matminer composition baseline.
- Book anchor: [Sandfeld 2.1–2.2].
27. **Descriptor robustness under domain shift**
- Compare **descriptor robustness under domain shift** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor ablation table.
- Book anchor: [Neuer 5.5].
28. **Descriptor uncertainty and error propagation**
- Diagnose **descriptor uncertainty and error propagation** using one concrete materials example and one common failure mode.
- Applied anchor: feature scaling comparison.
- Book anchor: [McClarren Ch5].
29. **Ablation studies for descriptor family contribution**
- Apply **ablation studies for descriptor family contribution** using one concrete materials example and one common failure mode.
- Applied anchor: L1-selected sparse descriptor set.
- Book anchor: [Murphy Ch1].
30. **Interpreting feature importance under correlated inputs**
- Define **interpreting feature importance under correlated inputs** using one concrete materials example and one common failure mode.
- Applied anchor: baseline vs learned embedding comparison.
- Book anchor: [Bishop 12.1–12.3].
31. **Domain knowledge injection into descriptor construction**
- Explain **domain knowledge injection into descriptor construction** using one concrete materials example and one common failure mode.
- Applied anchor: matminer composition baseline.
- Book anchor: [Sandfeld 2.1–2.2].
32. **Computational cost of descriptor pipelines**
- Compare **computational cost of descriptor pipelines** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor ablation table.
- Book anchor: [Neuer 5.5].
33. **Reproducible featurization templates and data cards**
- Diagnose **reproducible featurization templates and data cards** using one concrete materials example and one common failure mode.
- Applied anchor: feature scaling comparison.
- Book anchor: [McClarren Ch5].
34. **Case: descriptor baseline for bandgap prediction**
- Apply **case: descriptor baseline for bandgap prediction** using one concrete materials example and one common failure mode.
- Applied anchor: L1-selected sparse descriptor set.
- Book anchor: [Murphy Ch1].
35. **Case: descriptor baseline for stability screening**
- Define **case: descriptor baseline for stability screening** using one concrete materials example and one common failure mode.
- Applied anchor: baseline vs learned embedding comparison.
- Book anchor: [Bishop 12.1–12.3].
36. **Failure mode: over-engineering without split discipline**
- Explain **failure mode: over-engineering without split discipline** using one concrete materials example and one common failure mode.
- Applied anchor: matminer composition baseline.
- Book anchor: [Sandfeld 2.1–2.2].
37. **How this unit feeds graph-based representations in Unit 5**
- Compare **how this unit feeds graph-based representations in unit 5** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor ablation table.
- Book anchor: [Neuer 5.5].
38. **Advanced note: From Classical Descriptors to Learned Representations concept extension 33**
- Diagnose **advanced note: from classical descriptors to learned representations concept extension 33** using one concrete materials example and one common failure mode.
- Applied anchor: feature scaling comparison.
- Book anchor: [McClarren Ch5].
39. **Advanced note: From Classical Descriptors to Learned Representations concept extension 34**
- Apply **advanced note: from classical descriptors to learned representations concept extension 34** using one concrete materials example and one common failure mode.
- Applied anchor: L1-selected sparse descriptor set.
- Book anchor: [Murphy Ch1].
40. **Advanced note: From Classical Descriptors to Learned Representations concept extension 35**
- Define **advanced note: from classical descriptors to learned representations concept extension 35** using one concrete materials example and one common failure mode.
- Applied anchor: baseline vs learned embedding comparison.
- Book anchor: [Bishop 12.1–12.3].
41. **Advanced note: From Classical Descriptors to Learned Representations concept extension 36**
- Explain **advanced note: from classical descriptors to learned representations concept extension 36** using one concrete materials example and one common failure mode.
- Applied anchor: matminer composition baseline.
- Book anchor: [Sandfeld 2.1–2.2].
42. **Advanced note: From Classical Descriptors to Learned Representations concept extension 37**
- Compare **advanced note: from classical descriptors to learned representations concept extension 37** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor ablation table.
- Book anchor: [Neuer 5.5].
43. **Advanced note: From Classical Descriptors to Learned Representations concept extension 38**
- Diagnose **advanced note: from classical descriptors to learned representations concept extension 38** using one concrete materials example and one common failure mode.
- Applied anchor: feature scaling comparison.
- Book anchor: [McClarren Ch5].
44. **Advanced note: From Classical Descriptors to Learned Representations concept extension 39**
- Apply **advanced note: from classical descriptors to learned representations concept extension 39** using one concrete materials example and one common failure mode.
- Applied anchor: L1-selected sparse descriptor set.
- Book anchor: [Murphy Ch1].
45. **Exercise setup and dataset definition**
- Define dataset, split protocol, and expected deliverables before any coding begins.
- Applied anchor: baseline vs learned embedding comparison.
- Book anchor: [Bishop 12.1–12.3].
46. **Exercise task 1 (pipeline core)**
- Implement the core pipeline component with reproducible settings and documented assumptions.
- Applied anchor: matminer composition baseline.
- Book anchor: [Sandfeld 2.1–2.2].
47. **Exercise task 2 (comparison/ablation)**
- Run an ablation/comparison under identical validation protocol and interpret differences.
- Applied anchor: descriptor ablation table.
- Book anchor: [Neuer 5.5].
48. **Exercise task 3 (failure analysis)**
- Perform structured failure analysis and propose one evidence-backed mitigation.
- Applied anchor: feature scaling comparison.
- Book anchor: [McClarren Ch5].
49. **Exam-oriented key statements**
- Summarize high-yield statements in concise written-exam style with definitions and caveats.
- Applied anchor: L1-selected sparse descriptor set.
- Book anchor: [Murphy Ch1].
50. **Summary, next-unit bridge, and references**
- Consolidate the unit into a checklist: concepts, pitfalls, and decisions for next-unit transfer.
- Applied anchor: baseline vs learned embedding comparison.
- Book anchor: [Bishop 12.1–12.3].

