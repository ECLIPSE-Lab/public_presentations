# Materials Genomics Unit 9 — 50-Slide Teaching Scaffold (book-backed)

## Book-backed content summary (for this unit)
- Engineered vs learned features: objective-level comparison
- Representation learning goals in materials tasks
- Embedding quality criteria: separability, transferability, robustness
- Self-supervised intuition for materials data (conceptual)
- Contrastive objectives and positive-pair design
- Reconstruction objectives and bottleneck effects
- Feature discovery through hidden-layer analysis
- Latent factors linked to chemistry and structure
- Embedding drift across datasets and domains
- Task-agnostic vs task-specific representations
- Transfer learning from one property to another
- Few-shot adaptation with pretrained embeddings

## Source anchors used
- Neuer 5.5
- McClarren Ch5
- Bishop 12.3
- Sandfeld 2.2
- Murphy latent variable intuition

## Essential equations / objects (lecture must-include)
- $\hat{\theta}=\arg\min_\theta \frac{1}{N}\sum_i \ell(f_\theta(x_i),y_i)+\lambda\Omega(\theta)$
- Train/validation/test with grouped split by chemistry/prototype
- Generalization gap: $R_{test}-R_{train}$
- Uncertainty decomposition: aleatoric + epistemic
- Acquisition objective (conceptual): exploration vs exploitation

## 50-slide scaffold

1. **Title: Representation Learning and Feature Discovery**
- Frame the unit in the end-to-end materials discovery workflow and state the decision problems it addresses.
- Applied anchor: embedding probe classifier.
- Book anchor: [Neuer 5.5].
2. **Learning objectives and expected outputs**
- State measurable outcomes (what students can explain, implement, and critique by the end of the unit).
- Applied anchor: transfer to new target.
- Book anchor: [McClarren Ch5].
3. **Recap from previous unit and dependency map**
- Reconnect prerequisite concepts from earlier units and make dependency assumptions explicit.
- Applied anchor: descriptor+embedding fusion.
- Book anchor: [Bishop 12.3].
4. **Why this unit matters for materials discovery**
- Motivate with a realistic failure/success scenario from materials discovery practice.
- Applied anchor: t-SNE diagnostic panel.
- Book anchor: [Sandfeld 2.2].
5. **Reading map and chapter anchors**
- Map slide blocks to the key book chapters so students can pre-read and post-review effectively.
- Applied anchor: few-shot adaptation experiment.
- Book anchor: [Murphy latent variable intuition].
6. **Engineered vs learned features: objective-level comparison**
- Explain **engineered vs learned features: objective-level comparison** using one concrete materials example and one common failure mode.
- Applied anchor: embedding probe classifier.
- Book anchor: [Neuer 5.5].
7. **Representation learning goals in materials tasks**
- Compare **representation learning goals in materials tasks** using one concrete materials example and one common failure mode.
- Applied anchor: transfer to new target.
- Book anchor: [McClarren Ch5].
8. **Embedding quality criteria: separability, transferability, robustness**
- Diagnose **embedding quality criteria: separability, transferability, robustness** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor+embedding fusion.
- Book anchor: [Bishop 12.3].
9. **Self-supervised intuition for materials data (conceptual)**
- Apply **self-supervised intuition for materials data (conceptual)** using one concrete materials example and one common failure mode.
- Applied anchor: t-SNE diagnostic panel.
- Book anchor: [Sandfeld 2.2].
10. **Contrastive objectives and positive-pair design**
- Define **contrastive objectives and positive-pair design** using one concrete materials example and one common failure mode.
- Applied anchor: few-shot adaptation experiment.
- Book anchor: [Murphy latent variable intuition].
11. **Reconstruction objectives and bottleneck effects**
- Explain **reconstruction objectives and bottleneck effects** using one concrete materials example and one common failure mode.
- Applied anchor: embedding probe classifier.
- Book anchor: [Neuer 5.5].
12. **Feature discovery through hidden-layer analysis**
- Compare **feature discovery through hidden-layer analysis** using one concrete materials example and one common failure mode.
- Applied anchor: transfer to new target.
- Book anchor: [McClarren Ch5].
13. **Latent factors linked to chemistry and structure**
- Diagnose **latent factors linked to chemistry and structure** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor+embedding fusion.
- Book anchor: [Bishop 12.3].
14. **Embedding drift across datasets and domains**
- Apply **embedding drift across datasets and domains** using one concrete materials example and one common failure mode.
- Applied anchor: t-SNE diagnostic panel.
- Book anchor: [Sandfeld 2.2].
15. **Task-agnostic vs task-specific representations**
- Define **task-agnostic vs task-specific representations** using one concrete materials example and one common failure mode.
- Applied anchor: few-shot adaptation experiment.
- Book anchor: [Murphy latent variable intuition].
16. **Transfer learning from one property to another**
- Explain **transfer learning from one property to another** using one concrete materials example and one common failure mode.
- Applied anchor: embedding probe classifier.
- Book anchor: [Neuer 5.5].
17. **Few-shot adaptation with pretrained embeddings**
- Compare **few-shot adaptation with pretrained embeddings** using one concrete materials example and one common failure mode.
- Applied anchor: transfer to new target.
- Book anchor: [McClarren Ch5].
18. **Probing embeddings with linear/readout tests**
- Diagnose **probing embeddings with linear/readout tests** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor+embedding fusion.
- Book anchor: [Bishop 12.3].
19. **Ablation: which input channels drive embedding quality**
- Apply **ablation: which input channels drive embedding quality** using one concrete materials example and one common failure mode.
- Applied anchor: t-SNE diagnostic panel.
- Book anchor: [Sandfeld 2.2].
20. **Regularization effects on learned feature geometry**
- Define **regularization effects on learned feature geometry** using one concrete materials example and one common failure mode.
- Applied anchor: few-shot adaptation experiment.
- Book anchor: [Murphy latent variable intuition].
21. **Failure mode: embeddings collapse to shortcuts**
- Explain **failure mode: embeddings collapse to shortcuts** using one concrete materials example and one common failure mode.
- Applied anchor: embedding probe classifier.
- Book anchor: [Neuer 5.5].
22. **Failure mode: representation overfits training chemistry**
- Compare **failure mode: representation overfits training chemistry** using one concrete materials example and one common failure mode.
- Applied anchor: transfer to new target.
- Book anchor: [McClarren Ch5].
23. **Failure mode: transfer fails under structural shift**
- Diagnose **failure mode: transfer fails under structural shift** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor+embedding fusion.
- Book anchor: [Bishop 12.3].
24. **Visualization pitfalls in embedding spaces**
- Apply **visualization pitfalls in embedding spaces** using one concrete materials example and one common failure mode.
- Applied anchor: t-SNE diagnostic panel.
- Book anchor: [Sandfeld 2.2].
25. **t-SNE/UMAP as diagnostics, not proof of discovery**
- Define **t-sne/umap as diagnostics, not proof of discovery** using one concrete materials example and one common failure mode.
- Applied anchor: few-shot adaptation experiment.
- Book anchor: [Murphy latent variable intuition].
26. **Quantitative evaluation of representation usefulness**
- Explain **quantitative evaluation of representation usefulness** using one concrete materials example and one common failure mode.
- Applied anchor: embedding probe classifier.
- Book anchor: [Neuer 5.5].
27. **Hybrid pipelines: descriptors + learned embeddings**
- Compare **hybrid pipelines: descriptors + learned embeddings** using one concrete materials example and one common failure mode.
- Applied anchor: transfer to new target.
- Book anchor: [McClarren Ch5].
28. **Interpretability vs performance tradeoff**
- Diagnose **interpretability vs performance tradeoff** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor+embedding fusion.
- Book anchor: [Bishop 12.3].
29. **Uncertainty in embedding-based predictions**
- Apply **uncertainty in embedding-based predictions** using one concrete materials example and one common failure mode.
- Applied anchor: t-SNE diagnostic panel.
- Book anchor: [Sandfeld 2.2].
30. **Case: descriptor baseline vs learned embedding model**
- Define **case: descriptor baseline vs learned embedding model** using one concrete materials example and one common failure mode.
- Applied anchor: few-shot adaptation experiment.
- Book anchor: [Murphy latent variable intuition].
31. **Case: transfer to unseen chemistry subset**
- Explain **case: transfer to unseen chemistry subset** using one concrete materials example and one common failure mode.
- Applied anchor: embedding probe classifier.
- Book anchor: [Neuer 5.5].
32. **Case: feature attribution in latent feature models**
- Compare **case: feature attribution in latent feature models** using one concrete materials example and one common failure mode.
- Applied anchor: transfer to new target.
- Book anchor: [McClarren Ch5].
33. **Readiness for latent-space modeling in Unit 10**
- Diagnose **readiness for latent-space modeling in unit 10** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor+embedding fusion.
- Book anchor: [Bishop 12.3].
34. **How representation choices affect clustering in Unit 11**
- Apply **how representation choices affect clustering in unit 11** using one concrete materials example and one common failure mode.
- Applied anchor: t-SNE diagnostic panel.
- Book anchor: [Sandfeld 2.2].
35. **How representation choices affect uncertainty in Unit 12**
- Define **how representation choices affect uncertainty in unit 12** using one concrete materials example and one common failure mode.
- Applied anchor: few-shot adaptation experiment.
- Book anchor: [Murphy latent variable intuition].
36. **Exercise: compare engineered and learned pipelines**
- Explain **exercise: compare engineered and learned pipelines** using one concrete materials example and one common failure mode.
- Applied anchor: embedding probe classifier.
- Book anchor: [Neuer 5.5].
37. **Exam checklist: defend representation choice with evidence**
- Compare **exam checklist: defend representation choice with evidence** using one concrete materials example and one common failure mode.
- Applied anchor: transfer to new target.
- Book anchor: [McClarren Ch5].
38. **Advanced note: Representation Learning and Feature Discovery concept extension 33**
- Diagnose **advanced note: representation learning and feature discovery concept extension 33** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor+embedding fusion.
- Book anchor: [Bishop 12.3].
39. **Advanced note: Representation Learning and Feature Discovery concept extension 34**
- Apply **advanced note: representation learning and feature discovery concept extension 34** using one concrete materials example and one common failure mode.
- Applied anchor: t-SNE diagnostic panel.
- Book anchor: [Sandfeld 2.2].
40. **Advanced note: Representation Learning and Feature Discovery concept extension 35**
- Define **advanced note: representation learning and feature discovery concept extension 35** using one concrete materials example and one common failure mode.
- Applied anchor: few-shot adaptation experiment.
- Book anchor: [Murphy latent variable intuition].
41. **Advanced note: Representation Learning and Feature Discovery concept extension 36**
- Explain **advanced note: representation learning and feature discovery concept extension 36** using one concrete materials example and one common failure mode.
- Applied anchor: embedding probe classifier.
- Book anchor: [Neuer 5.5].
42. **Advanced note: Representation Learning and Feature Discovery concept extension 37**
- Compare **advanced note: representation learning and feature discovery concept extension 37** using one concrete materials example and one common failure mode.
- Applied anchor: transfer to new target.
- Book anchor: [McClarren Ch5].
43. **Advanced note: Representation Learning and Feature Discovery concept extension 38**
- Diagnose **advanced note: representation learning and feature discovery concept extension 38** using one concrete materials example and one common failure mode.
- Applied anchor: descriptor+embedding fusion.
- Book anchor: [Bishop 12.3].
44. **Advanced note: Representation Learning and Feature Discovery concept extension 39**
- Apply **advanced note: representation learning and feature discovery concept extension 39** using one concrete materials example and one common failure mode.
- Applied anchor: t-SNE diagnostic panel.
- Book anchor: [Sandfeld 2.2].
45. **Exercise setup and dataset definition**
- Define dataset, split protocol, and expected deliverables before any coding begins.
- Applied anchor: few-shot adaptation experiment.
- Book anchor: [Murphy latent variable intuition].
46. **Exercise task 1 (pipeline core)**
- Implement the core pipeline component with reproducible settings and documented assumptions.
- Applied anchor: embedding probe classifier.
- Book anchor: [Neuer 5.5].
47. **Exercise task 2 (comparison/ablation)**
- Run an ablation/comparison under identical validation protocol and interpret differences.
- Applied anchor: transfer to new target.
- Book anchor: [McClarren Ch5].
48. **Exercise task 3 (failure analysis)**
- Perform structured failure analysis and propose one evidence-backed mitigation.
- Applied anchor: descriptor+embedding fusion.
- Book anchor: [Bishop 12.3].
49. **Exam-oriented key statements**
- Summarize high-yield statements in concise written-exam style with definitions and caveats.
- Applied anchor: t-SNE diagnostic panel.
- Book anchor: [Sandfeld 2.2].
50. **Summary, next-unit bridge, and references**
- Consolidate the unit into a checklist: concepts, pitfalls, and decisions for next-unit transfer.
- Applied anchor: few-shot adaptation experiment.
- Book anchor: [Murphy latent variable intuition].

