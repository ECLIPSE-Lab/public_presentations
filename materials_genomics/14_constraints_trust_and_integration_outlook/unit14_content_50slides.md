# Materials Genomics Unit 13 — 50-Slide Teaching Scaffold (book-backed)

## Book-backed content summary (for this unit)
- Why constraints are essential in scientific ML
- Hard vs soft physical constraints in optimization
- Constraint examples: charge neutrality, stability bounds, symmetry
- Penalty-based constrained learning
- Physics-informed loss design concepts
- Hybrid model architecture: mechanistic + data-driven
- Constraint violation diagnostics and monitoring
- Explainability levels for different stakeholders
- Sensitivity analysis for trust building
- Causality caution: association vs mechanism
- Model cards and uncertainty statements for deployment
- Reproducibility package: data, code, splits, environment

## Source anchors used
- Neuer 6.1–6.3
- Neuer Ch7
- McClarren Ch11+12
- Murphy model selection
- Bishop model limits

## Essential equations / objects (lecture must-include)
- $\hat{\theta}=\arg\min_\theta \frac{1}{N}\sum_i \ell(f_\theta(x_i),y_i)+\lambda\Omega(\theta)$
- Train/validation/test with grouped split by chemistry/prototype
- Generalization gap: $R_{test}-R_{train}$
- Uncertainty decomposition: aleatoric + epistemic
- Acquisition objective (conceptual): exploration vs exploitation

## 50-slide scaffold

1. **Title: Physical Constraints, Trust, and Integration Outlook**
- Frame the unit in the end-to-end materials discovery workflow and state the decision problems it addresses.
- Applied anchor: constraint penalty ablation.
- Book anchor: [Neuer 6.1–6.3].
2. **Learning objectives and expected outputs**
- State measurable outcomes (what students can explain, implement, and critique by the end of the unit).
- Applied anchor: model card template.
- Book anchor: [Neuer Ch7].
3. **Recap from previous unit and dependency map**
- Reconnect prerequisite concepts from earlier units and make dependency assumptions explicit.
- Applied anchor: sensitivity analysis report.
- Book anchor: [McClarren Ch11+12].
4. **Why this unit matters for materials discovery**
- Motivate with a realistic failure/success scenario from materials discovery practice.
- Applied anchor: failed recommendation post-mortem.
- Book anchor: [Murphy model selection].
5. **Reading map and chapter anchors**
- Map slide blocks to the key book chapters so students can pre-read and post-review effectively.
- Applied anchor: deployment readiness checklist.
- Book anchor: [Bishop model limits].
6. **Why constraints are essential in scientific ML**
- Explain **why constraints are essential in scientific ml** using one concrete materials example and one common failure mode.
- Applied anchor: constraint penalty ablation.
- Book anchor: [Neuer 6.1–6.3].
7. **Hard vs soft physical constraints in optimization**
- Compare **hard vs soft physical constraints in optimization** using one concrete materials example and one common failure mode.
- Applied anchor: model card template.
- Book anchor: [Neuer Ch7].
8. **Constraint examples: charge neutrality, stability bounds, symmetry**
- Diagnose **constraint examples: charge neutrality, stability bounds, symmetry** using one concrete materials example and one common failure mode.
- Applied anchor: sensitivity analysis report.
- Book anchor: [McClarren Ch11+12].
9. **Penalty-based constrained learning**
- Apply **penalty-based constrained learning** using one concrete materials example and one common failure mode.
- Applied anchor: failed recommendation post-mortem.
- Book anchor: [Murphy model selection].
10. **Physics-informed loss design concepts**
- Define **physics-informed loss design concepts** using one concrete materials example and one common failure mode.
- Applied anchor: deployment readiness checklist.
- Book anchor: [Bishop model limits].
11. **Hybrid model architecture: mechanistic + data-driven**
- Explain **hybrid model architecture: mechanistic + data-driven** using one concrete materials example and one common failure mode.
- Applied anchor: constraint penalty ablation.
- Book anchor: [Neuer 6.1–6.3].
12. **Constraint violation diagnostics and monitoring**
- Compare **constraint violation diagnostics and monitoring** using one concrete materials example and one common failure mode.
- Applied anchor: model card template.
- Book anchor: [Neuer Ch7].
13. **Explainability levels for different stakeholders**
- Diagnose **explainability levels for different stakeholders** using one concrete materials example and one common failure mode.
- Applied anchor: sensitivity analysis report.
- Book anchor: [McClarren Ch11+12].
14. **Sensitivity analysis for trust building**
- Apply **sensitivity analysis for trust building** using one concrete materials example and one common failure mode.
- Applied anchor: failed recommendation post-mortem.
- Book anchor: [Murphy model selection].
15. **Causality caution: association vs mechanism**
- Define **causality caution: association vs mechanism** using one concrete materials example and one common failure mode.
- Applied anchor: deployment readiness checklist.
- Book anchor: [Bishop model limits].
16. **Model cards and uncertainty statements for deployment**
- Explain **model cards and uncertainty statements for deployment** using one concrete materials example and one common failure mode.
- Applied anchor: constraint penalty ablation.
- Book anchor: [Neuer 6.1–6.3].
17. **Reproducibility package: data, code, splits, environment**
- Compare **reproducibility package: data, code, splits, environment** using one concrete materials example and one common failure mode.
- Applied anchor: model card template.
- Book anchor: [Neuer Ch7].
18. **Failure taxonomy for materials ML systems**
- Diagnose **failure taxonomy for materials ml systems** using one concrete materials example and one common failure mode.
- Applied anchor: sensitivity analysis report.
- Book anchor: [McClarren Ch11+12].
19. **Ethical and safety dimensions in accelerated discovery**
- Apply **ethical and safety dimensions in accelerated discovery** using one concrete materials example and one common failure mode.
- Applied anchor: failed recommendation post-mortem.
- Book anchor: [Murphy model selection].
20. **What ML can discover vs what requires experiment**
- Define **what ml can discover vs what requires experiment** using one concrete materials example and one common failure mode.
- Applied anchor: deployment readiness checklist.
- Book anchor: [Bishop model limits].
21. **Human-in-the-loop validation workflows**
- Explain **human-in-the-loop validation workflows** using one concrete materials example and one common failure mode.
- Applied anchor: constraint penalty ablation.
- Book anchor: [Neuer 6.1–6.3].
22. **Integration with high-throughput and lab pipelines**
- Compare **integration with high-throughput and lab pipelines** using one concrete materials example and one common failure mode.
- Applied anchor: model card template.
- Book anchor: [Neuer Ch7].
23. **Decision governance under uncertainty and constraints**
- Diagnose **decision governance under uncertainty and constraints** using one concrete materials example and one common failure mode.
- Applied anchor: sensitivity analysis report.
- Book anchor: [McClarren Ch11+12].
24. **When to reject model recommendations**
- Apply **when to reject model recommendations** using one concrete materials example and one common failure mode.
- Applied anchor: failed recommendation post-mortem.
- Book anchor: [Murphy model selection].
25. **Robustness checks before deployment**
- Define **robustness checks before deployment** using one concrete materials example and one common failure mode.
- Applied anchor: deployment readiness checklist.
- Book anchor: [Bishop model limits].
26. **Case: constrained surrogate for stability-aware screening**
- Explain **case: constrained surrogate for stability-aware screening** using one concrete materials example and one common failure mode.
- Applied anchor: constraint penalty ablation.
- Book anchor: [Neuer 6.1–6.3].
27. **Case: explainability report for candidate shortlist**
- Compare **case: explainability report for candidate shortlist** using one concrete materials example and one common failure mode.
- Applied anchor: model card template.
- Book anchor: [Neuer Ch7].
28. **Case: post-mortem of failed discovery recommendation**
- Diagnose **case: post-mortem of failed discovery recommendation** using one concrete materials example and one common failure mode.
- Applied anchor: sensitivity analysis report.
- Book anchor: [McClarren Ch11+12].
29. **Balancing performance, interpretability, and cost**
- Apply **balancing performance, interpretability, and cost** using one concrete materials example and one common failure mode.
- Applied anchor: failed recommendation post-mortem.
- Book anchor: [Murphy model selection].
30. **Long-term maintenance of discovery models**
- Define **long-term maintenance of discovery models** using one concrete materials example and one common failure mode.
- Applied anchor: deployment readiness checklist.
- Book anchor: [Bishop model limits].
31. **Domain drift and model update policy**
- Explain **domain drift and model update policy** using one concrete materials example and one common failure mode.
- Applied anchor: constraint penalty ablation.
- Book anchor: [Neuer 6.1–6.3].
32. **Auditability and traceability requirements**
- Compare **auditability and traceability requirements** using one concrete materials example and one common failure mode.
- Applied anchor: model card template.
- Book anchor: [Neuer Ch7].
33. **Integration outlook with autonomous labs**
- Diagnose **integration outlook with autonomous labs** using one concrete materials example and one common failure mode.
- Applied anchor: sensitivity analysis report.
- Book anchor: [McClarren Ch11+12].
34. **Course-wide synthesis: from data to trusted decisions**
- Apply **course-wide synthesis: from data to trusted decisions** using one concrete materials example and one common failure mode.
- Applied anchor: failed recommendation post-mortem.
- Book anchor: [Murphy model selection].
35. **Capstone framing for students**
- Define **capstone framing for students** using one concrete materials example and one common failure mode.
- Applied anchor: deployment readiness checklist.
- Book anchor: [Bishop model limits].
36. **Exercise: constrained model mini-project**
- Explain **exercise: constrained model mini-project** using one concrete materials example and one common failure mode.
- Applied anchor: constraint penalty ablation.
- Book anchor: [Neuer 6.1–6.3].
37. **Exam checklist: trust argument with evidence**
- Compare **exam checklist: trust argument with evidence** using one concrete materials example and one common failure mode.
- Applied anchor: model card template.
- Book anchor: [Neuer Ch7].
38. **Advanced note: Physical Constraints, Trust, and Integration Outlook concept extension 33**
- Diagnose **advanced note: physical constraints, trust, and integration outlook concept extension 33** using one concrete materials example and one common failure mode.
- Applied anchor: sensitivity analysis report.
- Book anchor: [McClarren Ch11+12].
39. **Advanced note: Physical Constraints, Trust, and Integration Outlook concept extension 34**
- Apply **advanced note: physical constraints, trust, and integration outlook concept extension 34** using one concrete materials example and one common failure mode.
- Applied anchor: failed recommendation post-mortem.
- Book anchor: [Murphy model selection].
40. **Advanced note: Physical Constraints, Trust, and Integration Outlook concept extension 35**
- Define **advanced note: physical constraints, trust, and integration outlook concept extension 35** using one concrete materials example and one common failure mode.
- Applied anchor: deployment readiness checklist.
- Book anchor: [Bishop model limits].
41. **Advanced note: Physical Constraints, Trust, and Integration Outlook concept extension 36**
- Explain **advanced note: physical constraints, trust, and integration outlook concept extension 36** using one concrete materials example and one common failure mode.
- Applied anchor: constraint penalty ablation.
- Book anchor: [Neuer 6.1–6.3].
42. **Advanced note: Physical Constraints, Trust, and Integration Outlook concept extension 37**
- Compare **advanced note: physical constraints, trust, and integration outlook concept extension 37** using one concrete materials example and one common failure mode.
- Applied anchor: model card template.
- Book anchor: [Neuer Ch7].
43. **Advanced note: Physical Constraints, Trust, and Integration Outlook concept extension 38**
- Diagnose **advanced note: physical constraints, trust, and integration outlook concept extension 38** using one concrete materials example and one common failure mode.
- Applied anchor: sensitivity analysis report.
- Book anchor: [McClarren Ch11+12].
44. **Advanced note: Physical Constraints, Trust, and Integration Outlook concept extension 39**
- Apply **advanced note: physical constraints, trust, and integration outlook concept extension 39** using one concrete materials example and one common failure mode.
- Applied anchor: failed recommendation post-mortem.
- Book anchor: [Murphy model selection].
45. **Exercise setup and dataset definition**
- Define dataset, split protocol, and expected deliverables before any coding begins.
- Applied anchor: deployment readiness checklist.
- Book anchor: [Bishop model limits].
46. **Exercise task 1 (pipeline core)**
- Implement the core pipeline component with reproducible settings and documented assumptions.
- Applied anchor: constraint penalty ablation.
- Book anchor: [Neuer 6.1–6.3].
47. **Exercise task 2 (comparison/ablation)**
- Run an ablation/comparison under identical validation protocol and interpret differences.
- Applied anchor: model card template.
- Book anchor: [Neuer Ch7].
48. **Exercise task 3 (failure analysis)**
- Perform structured failure analysis and propose one evidence-backed mitigation.
- Applied anchor: sensitivity analysis report.
- Book anchor: [McClarren Ch11+12].
49. **Exam-oriented key statements**
- Summarize high-yield statements in concise written-exam style with definitions and caveats.
- Applied anchor: failed recommendation post-mortem.
- Book anchor: [Murphy model selection].
50. **Summary, next-unit bridge, and references**
- Consolidate the unit into a checklist: concepts, pitfalls, and decisions for next-unit transfer.
- Applied anchor: deployment readiness checklist.
- Book anchor: [Bishop model limits].

