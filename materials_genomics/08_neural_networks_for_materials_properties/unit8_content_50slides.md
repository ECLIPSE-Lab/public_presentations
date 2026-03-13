# Materials Genomics Unit 8 — 50-Slide Teaching Scaffold (book-backed)

## Book-backed content summary (for this unit)
- Neuron model, activations, and expressive nonlinear mappings
- Network architecture choices for tabular/materials features
- Depth/width tradeoff for moderate-size materials datasets
- Initialization and optimization stability
- Backpropagation flow and gradient diagnostics
- Learning-rate schedules and convergence behavior
- Batch size, normalization, and regularization controls
- Dropout, weight decay, and early stopping roles
- Loss choices for regression and multi-target setups
- Multi-task learning for coupled materials properties
- Neural surrogate vs DFT cost-accuracy tradeoff
- Data augmentation options for materials representations

## Source anchors used
- Neuer 4.5.1/4.5.3/4.5.4/4.5.5
- McClarren Ch8
- Bishop 5.1–5.3
- Murphy deep model caveats
- Sandfeld domain context

## Essential equations / objects (lecture must-include)
- $\hat{\theta}=\arg\min_\theta \frac{1}{N}\sum_i \ell(f_\theta(x_i),y_i)+\lambda\Omega(\theta)$
- Train/validation/test with grouped split by chemistry/prototype
- Generalization gap: $R_{test}-R_{train}$
- Uncertainty decomposition: aleatoric + epistemic
- Acquisition objective (conceptual): exploration vs exploitation

## 50-slide scaffold

1. **Title: Neural Networks for Materials Properties**
- Frame the unit in the end-to-end materials discovery workflow and state the decision problems it addresses.
- Applied anchor: MLP on descriptor table.
- Book anchor: [Neuer 4.5.1/4.5.3/4.5.4/4.5.5].
2. **Learning objectives and expected outputs**
- State measurable outcomes (what students can explain, implement, and critique by the end of the unit).
- Applied anchor: learning-rate sweep.
- Book anchor: [McClarren Ch8].
3. **Recap from previous unit and dependency map**
- Reconnect prerequisite concepts from earlier units and make dependency assumptions explicit.
- Applied anchor: early stopping vs none.
- Book anchor: [Bishop 5.1–5.3].
4. **Why this unit matters for materials discovery**
- Motivate with a realistic failure/success scenario from materials discovery practice.
- Applied anchor: dropout+weight-decay ablation.
- Book anchor: [Murphy deep model caveats].
5. **Reading map and chapter anchors**
- Map slide blocks to the key book chapters so students can pre-read and post-review effectively.
- Applied anchor: NN vs RF baseline.
- Book anchor: [Sandfeld domain context].
6. **Neuron model, activations, and expressive nonlinear mappings**
- Explain **neuron model, activations, and expressive nonlinear mappings** using one concrete materials example and one common failure mode.
- Applied anchor: MLP on descriptor table.
- Book anchor: [Neuer 4.5.1/4.5.3/4.5.4/4.5.5].
7. **Network architecture choices for tabular/materials features**
- Compare **network architecture choices for tabular/materials features** using one concrete materials example and one common failure mode.
- Applied anchor: learning-rate sweep.
- Book anchor: [McClarren Ch8].
8. **Depth/width tradeoff for moderate-size materials datasets**
- Diagnose **depth/width tradeoff for moderate-size materials datasets** using one concrete materials example and one common failure mode.
- Applied anchor: early stopping vs none.
- Book anchor: [Bishop 5.1–5.3].
9. **Initialization and optimization stability**
- Apply **initialization and optimization stability** using one concrete materials example and one common failure mode.
- Applied anchor: dropout+weight-decay ablation.
- Book anchor: [Murphy deep model caveats].
10. **Backpropagation flow and gradient diagnostics**
- Define **backpropagation flow and gradient diagnostics** using one concrete materials example and one common failure mode.
- Applied anchor: NN vs RF baseline.
- Book anchor: [Sandfeld domain context].
11. **Learning-rate schedules and convergence behavior**
- Explain **learning-rate schedules and convergence behavior** using one concrete materials example and one common failure mode.
- Applied anchor: MLP on descriptor table.
- Book anchor: [Neuer 4.5.1/4.5.3/4.5.4/4.5.5].
12. **Batch size, normalization, and regularization controls**
- Compare **batch size, normalization, and regularization controls** using one concrete materials example and one common failure mode.
- Applied anchor: learning-rate sweep.
- Book anchor: [McClarren Ch8].
13. **Dropout, weight decay, and early stopping roles**
- Diagnose **dropout, weight decay, and early stopping roles** using one concrete materials example and one common failure mode.
- Applied anchor: early stopping vs none.
- Book anchor: [Bishop 5.1–5.3].
14. **Loss choices for regression and multi-target setups**
- Apply **loss choices for regression and multi-target setups** using one concrete materials example and one common failure mode.
- Applied anchor: dropout+weight-decay ablation.
- Book anchor: [Murphy deep model caveats].
15. **Multi-task learning for coupled materials properties**
- Define **multi-task learning for coupled materials properties** using one concrete materials example and one common failure mode.
- Applied anchor: NN vs RF baseline.
- Book anchor: [Sandfeld domain context].
16. **Neural surrogate vs DFT cost-accuracy tradeoff**
- Explain **neural surrogate vs dft cost-accuracy tradeoff** using one concrete materials example and one common failure mode.
- Applied anchor: MLP on descriptor table.
- Book anchor: [Neuer 4.5.1/4.5.3/4.5.4/4.5.5].
17. **Data augmentation options for materials representations**
- Compare **data augmentation options for materials representations** using one concrete materials example and one common failure mode.
- Applied anchor: learning-rate sweep.
- Book anchor: [McClarren Ch8].
18. **Handling imbalance in property distributions**
- Diagnose **handling imbalance in property distributions** using one concrete materials example and one common failure mode.
- Applied anchor: early stopping vs none.
- Book anchor: [Bishop 5.1–5.3].
19. **Extrapolation risk in neural models**
- Apply **extrapolation risk in neural models** using one concrete materials example and one common failure mode.
- Applied anchor: dropout+weight-decay ablation.
- Book anchor: [Murphy deep model caveats].
20. **Model calibration and confidence diagnostics**
- Define **model calibration and confidence diagnostics** using one concrete materials example and one common failure mode.
- Applied anchor: NN vs RF baseline.
- Book anchor: [Sandfeld domain context].
21. **Interpretability in neural property models**
- Explain **interpretability in neural property models** using one concrete materials example and one common failure mode.
- Applied anchor: MLP on descriptor table.
- Book anchor: [Neuer 4.5.1/4.5.3/4.5.4/4.5.5].
22. **Training reproducibility: seeds and deterministic ops**
- Compare **training reproducibility: seeds and deterministic ops** using one concrete materials example and one common failure mode.
- Applied anchor: learning-rate sweep.
- Book anchor: [McClarren Ch8].
23. **Hyperparameter optimization under strict validation**
- Diagnose **hyperparameter optimization under strict validation** using one concrete materials example and one common failure mode.
- Applied anchor: early stopping vs none.
- Book anchor: [Bishop 5.1–5.3].
24. **Failure mode: data leakage hidden in preprocessing**
- Apply **failure mode: data leakage hidden in preprocessing** using one concrete materials example and one common failure mode.
- Applied anchor: dropout+weight-decay ablation.
- Book anchor: [Murphy deep model caveats].
25. **Failure mode: overfitting with tiny effective data**
- Define **failure mode: overfitting with tiny effective data** using one concrete materials example and one common failure mode.
- Applied anchor: NN vs RF baseline.
- Book anchor: [Sandfeld domain context].
26. **Failure mode: unstable training due to poor scaling**
- Explain **failure mode: unstable training due to poor scaling** using one concrete materials example and one common failure mode.
- Applied anchor: MLP on descriptor table.
- Book anchor: [Neuer 4.5.1/4.5.3/4.5.4/4.5.5].
27. **Case: shallow MLP baseline from descriptors**
- Compare **case: shallow mlp baseline from descriptors** using one concrete materials example and one common failure mode.
- Applied anchor: learning-rate sweep.
- Book anchor: [McClarren Ch8].
28. **Case: training dynamics under different learning rates**
- Diagnose **case: training dynamics under different learning rates** using one concrete materials example and one common failure mode.
- Applied anchor: early stopping vs none.
- Book anchor: [Bishop 5.1–5.3].
29. **Case: regularization sweep and generalization gap**
- Apply **case: regularization sweep and generalization gap** using one concrete materials example and one common failure mode.
- Applied anchor: dropout+weight-decay ablation.
- Book anchor: [Murphy deep model caveats].
30. **Comparing NN to non-NN baselines honestly**
- Define **comparing nn to non-nn baselines honestly** using one concrete materials example and one common failure mode.
- Applied anchor: NN vs RF baseline.
- Book anchor: [Sandfeld domain context].
31. **When not to use NNs in materials tasks**
- Explain **when not to use nns in materials tasks** using one concrete materials example and one common failure mode.
- Applied anchor: MLP on descriptor table.
- Book anchor: [Neuer 4.5.1/4.5.3/4.5.4/4.5.5].
32. **Hybrid models with physically informed features**
- Compare **hybrid models with physically informed features** using one concrete materials example and one common failure mode.
- Applied anchor: learning-rate sweep.
- Book anchor: [McClarren Ch8].
33. **Readiness criteria before moving to representation learning**
- Diagnose **readiness criteria before moving to representation learning** using one concrete materials example and one common failure mode.
- Applied anchor: early stopping vs none.
- Book anchor: [Bishop 5.1–5.3].
34. **Connection to Unit 9 feature discovery**
- Apply **connection to unit 9 feature discovery** using one concrete materials example and one common failure mode.
- Applied anchor: dropout+weight-decay ablation.
- Book anchor: [Murphy deep model caveats].
35. **Connection to Unit 10 latent embeddings**
- Define **connection to unit 10 latent embeddings** using one concrete materials example and one common failure mode.
- Applied anchor: NN vs RF baseline.
- Book anchor: [Sandfeld domain context].
36. **Exercise: train and diagnose a small neural surrogate**
- Explain **exercise: train and diagnose a small neural surrogate** using one concrete materials example and one common failure mode.
- Applied anchor: MLP on descriptor table.
- Book anchor: [Neuer 4.5.1/4.5.3/4.5.4/4.5.5].
37. **Exam checklist: credible NN claim in materials science**
- Compare **exam checklist: credible nn claim in materials science** using one concrete materials example and one common failure mode.
- Applied anchor: learning-rate sweep.
- Book anchor: [McClarren Ch8].
38. **Advanced note: Neural Networks for Materials Properties concept extension 33**
- Diagnose **advanced note: neural networks for materials properties concept extension 33** using one concrete materials example and one common failure mode.
- Applied anchor: early stopping vs none.
- Book anchor: [Bishop 5.1–5.3].
39. **Advanced note: Neural Networks for Materials Properties concept extension 34**
- Apply **advanced note: neural networks for materials properties concept extension 34** using one concrete materials example and one common failure mode.
- Applied anchor: dropout+weight-decay ablation.
- Book anchor: [Murphy deep model caveats].
40. **Advanced note: Neural Networks for Materials Properties concept extension 35**
- Define **advanced note: neural networks for materials properties concept extension 35** using one concrete materials example and one common failure mode.
- Applied anchor: NN vs RF baseline.
- Book anchor: [Sandfeld domain context].
41. **Advanced note: Neural Networks for Materials Properties concept extension 36**
- Explain **advanced note: neural networks for materials properties concept extension 36** using one concrete materials example and one common failure mode.
- Applied anchor: MLP on descriptor table.
- Book anchor: [Neuer 4.5.1/4.5.3/4.5.4/4.5.5].
42. **Advanced note: Neural Networks for Materials Properties concept extension 37**
- Compare **advanced note: neural networks for materials properties concept extension 37** using one concrete materials example and one common failure mode.
- Applied anchor: learning-rate sweep.
- Book anchor: [McClarren Ch8].
43. **Advanced note: Neural Networks for Materials Properties concept extension 38**
- Diagnose **advanced note: neural networks for materials properties concept extension 38** using one concrete materials example and one common failure mode.
- Applied anchor: early stopping vs none.
- Book anchor: [Bishop 5.1–5.3].
44. **Advanced note: Neural Networks for Materials Properties concept extension 39**
- Apply **advanced note: neural networks for materials properties concept extension 39** using one concrete materials example and one common failure mode.
- Applied anchor: dropout+weight-decay ablation.
- Book anchor: [Murphy deep model caveats].
45. **Exercise setup and dataset definition**
- Define dataset, split protocol, and expected deliverables before any coding begins.
- Applied anchor: NN vs RF baseline.
- Book anchor: [Sandfeld domain context].
46. **Exercise task 1 (pipeline core)**
- Implement the core pipeline component with reproducible settings and documented assumptions.
- Applied anchor: MLP on descriptor table.
- Book anchor: [Neuer 4.5.1/4.5.3/4.5.4/4.5.5].
47. **Exercise task 2 (comparison/ablation)**
- Run an ablation/comparison under identical validation protocol and interpret differences.
- Applied anchor: learning-rate sweep.
- Book anchor: [McClarren Ch8].
48. **Exercise task 3 (failure analysis)**
- Perform structured failure analysis and propose one evidence-backed mitigation.
- Applied anchor: early stopping vs none.
- Book anchor: [Bishop 5.1–5.3].
49. **Exam-oriented key statements**
- Summarize high-yield statements in concise written-exam style with definitions and caveats.
- Applied anchor: dropout+weight-decay ablation.
- Book anchor: [Murphy deep model caveats].
50. **Summary, next-unit bridge, and references**
- Consolidate the unit into a checklist: concepts, pitfalls, and decisions for next-unit transfer.
- Applied anchor: NN vs RF baseline.
- Book anchor: [Sandfeld domain context].

