# Materials Genomics Unit 7 — 50-Slide Teaching Scaffold (book-backed)

## Book-backed content summary (for this unit)
- Regression as empirical risk minimization for materials targets
- Choice of target variable: bandgap, E_hull, modulus, conductivity
- Linear baseline and least-squares geometry refresher
- Regularized regression: ridge, lasso, elastic net
- Tree and ensemble baselines for nonlinear patterns
- Bias-variance decomposition in materials datasets
- Overfitting signatures in small-to-medium data
- Cross-validation under grouped chemistry splits
- Train/val/test protocols for fair model comparison
- Metric selection: MAE, RMSE, R2, rank metrics
- Error calibration vs point accuracy distinction
- Residual analysis by chemistry family

## Source anchors used
- Neuer 4.2.2
- Neuer 4.5.9
- McClarren Ch4+6
- Bishop 3.1–3.3
- Murphy model selection

## Essential equations / objects (lecture must-include)
- $\hat{\theta}=\arg\min_\theta \frac{1}{N}\sum_i \ell(f_\theta(x_i),y_i)+\lambda\Omega(\theta)$
- Closed-form ridge view: $(X^\top X+\lambda I)^{-1}X^\top y$ conceptually
- Train/validation/test with grouped split by chemistry/prototype
- Generalization gap: $R_{test}-R_{train}$
- Learning curve: error as a function of training-set size
- Residual plot by chemistry family or structure prototype

## 50-slide scaffold

1. **Title: Regression and Generalization in Materials Data**
- Frame the unit in the end-to-end materials discovery workflow and state the decision problems it addresses.
- Applied anchor: bandgap regression benchmark.
- Book anchor: [Neuer 4.2.2].
2. **Learning objectives and expected outputs**
- State measurable outcomes (what students can explain, implement, and critique by the end of the unit).
- Applied anchor: grouped split by composition.
- Book anchor: [Neuer 4.5.9].
3. **Recap from previous unit and dependency map**
- Reconnect prerequisite concepts from earlier units and make dependency assumptions explicit.
- Applied anchor: nested CV study.
- Book anchor: [McClarren Ch4+6].
4. **Why this unit matters for materials discovery**
- Motivate with a realistic failure/success scenario from materials discovery practice.
- Applied anchor: residuals by prototype.
- Book anchor: [Bishop 3.1–3.3].
5. **Reading map and chapter anchors**
- Map slide blocks to the key book chapters so students can pre-read and post-review effectively.
- Applied anchor: learning curve comparison.
- Book anchor: [Murphy model selection].
6. **Regression as empirical risk minimization for materials targets**
- Explain **regression as empirical risk minimization for materials targets** using one concrete materials example and one common failure mode.
- Applied anchor: bandgap regression benchmark.
- Book anchor: [Neuer 4.2.2].
7. **Choice of target variable: bandgap, E_hull, modulus, conductivity**
- Compare **choice of target variable: bandgap, e_hull, modulus, conductivity** using one concrete materials example and one common failure mode.
- Applied anchor: grouped split by composition.
- Book anchor: [Neuer 4.5.9].
8. **Linear baseline and least-squares geometry refresher**
- Diagnose **linear baseline and least-squares geometry refresher** using one concrete materials example and one common failure mode.
- Applied anchor: nested CV study.
- Book anchor: [McClarren Ch4+6].
9. **Regularized regression: ridge, lasso, elastic net**
- Apply **regularized regression: ridge, lasso, elastic net** using one concrete materials example and one common failure mode.
- Applied anchor: residuals by prototype.
- Book anchor: [Bishop 3.1–3.3].
10. **Tree and ensemble baselines for nonlinear patterns**
- Define **tree and ensemble baselines for nonlinear patterns** using one concrete materials example and one common failure mode.
- Applied anchor: learning curve comparison.
- Book anchor: [Murphy model selection].
11. **Bias-variance decomposition in materials datasets**
- Explain **bias-variance decomposition in materials datasets** using one concrete materials example and one common failure mode.
- Applied anchor: bandgap regression benchmark.
- Book anchor: [Neuer 4.2.2].
12. **Overfitting signatures in small-to-medium data**
- Compare **overfitting signatures in small-to-medium data** using one concrete materials example and one common failure mode.
- Applied anchor: grouped split by composition.
- Book anchor: [Neuer 4.5.9].
13. **Cross-validation under grouped chemistry splits**
- Diagnose **cross-validation under grouped chemistry splits** using one concrete materials example and one common failure mode.
- Applied anchor: nested CV study.
- Book anchor: [McClarren Ch4+6].
14. **Train/val/test protocols for fair model comparison**
- Apply **train/val/test protocols for fair model comparison** using one concrete materials example and one common failure mode.
- Applied anchor: residuals by prototype.
- Book anchor: [Bishop 3.1–3.3].
15. **Metric selection: MAE, RMSE, R2, rank metrics**
- Define **metric selection: mae, rmse, r2, rank metrics** using one concrete materials example and one common failure mode.
- Applied anchor: learning curve comparison.
- Book anchor: [Murphy model selection].
16. **Error calibration vs point accuracy distinction**
- Explain **error calibration vs point accuracy distinction** using one concrete materials example and one common failure mode.
- Applied anchor: bandgap regression benchmark.
- Book anchor: [Neuer 4.2.2].
17. **Residual analysis by chemistry family**
- Compare **residual analysis by chemistry family** using one concrete materials example and one common failure mode.
- Applied anchor: grouped split by composition.
- Book anchor: [Neuer 4.5.9].
18. **Heteroscedastic target noise handling**
- Diagnose **heteroscedastic target noise handling** using one concrete materials example and one common failure mode.
- Applied anchor: nested CV study.
- Book anchor: [McClarren Ch4+6].
19. **Feature-target leakage diagnostics**
- Apply **feature-target leakage diagnostics** using one concrete materials example and one common failure mode.
- Applied anchor: residuals by prototype.
- Book anchor: [Bishop 3.1–3.3].
20. **Model selection with nested validation**
- Define **model selection with nested validation** using one concrete materials example and one common failure mode.
- Applied anchor: learning curve comparison.
- Book anchor: [Murphy model selection].
21. **Hyperparameter search scope and stopping rules**
- Explain **hyperparameter search scope and stopping rules** using one concrete materials example and one common failure mode.
- Applied anchor: bandgap regression benchmark.
- Book anchor: [Neuer 4.2.2].
22. **Learning curves for sample-efficiency diagnosis**
- Compare **learning curves for sample-efficiency diagnosis** using one concrete materials example and one common failure mode.
- Applied anchor: grouped split by composition.
- Book anchor: [Neuer 4.5.9].
23. **Ablation of feature families and representations**
- Diagnose **ablation of feature families and representations** using one concrete materials example and one common failure mode.
- Applied anchor: nested CV study.
- Book anchor: [McClarren Ch4+6].
24. **OOD generalization to unseen compositions**
- Apply **ood generalization to unseen compositions** using one concrete materials example and one common failure mode.
- Applied anchor: residuals by prototype.
- Book anchor: [Bishop 3.1–3.3].
25. **OOD generalization to unseen structure prototypes**
- Define **ood generalization to unseen structure prototypes** using one concrete materials example and one common failure mode.
- Applied anchor: learning curve comparison.
- Book anchor: [Murphy model selection].
26. **Domain shift across databases or calculation settings**
- Explain **domain shift across databases or calculation settings** using one concrete materials example and one common failure mode.
- Applied anchor: bandgap regression benchmark.
- Book anchor: [Neuer 4.2.2].
27. **Interpretability checks for regression models**
- Compare **interpretability checks for regression models** using one concrete materials example and one common failure mode.
- Applied anchor: grouped split by composition.
- Book anchor: [Neuer 4.5.9].
28. **Trust criteria before deploying surrogate models**
- Diagnose **trust criteria before deploying surrogate models** using one concrete materials example and one common failure mode.
- Applied anchor: nested CV study.
- Book anchor: [McClarren Ch4+6].
29. **Case: compare linear, RF, shallow NN on same split**
- Apply **case: compare linear, rf, shallow nn on same split** using one concrete materials example and one common failure mode.
- Applied anchor: residuals by prototype.
- Book anchor: [Bishop 3.1–3.3].
30. **Case: random split vs grouped split performance gap**
- Define **case: random split vs grouped split performance gap** using one concrete materials example and one common failure mode.
- Applied anchor: learning curve comparison.
- Book anchor: [Murphy model selection].
31. **Case: model collapse on unseen chemistry**
- Explain **case: model collapse on unseen chemistry** using one concrete materials example and one common failure mode.
- Applied anchor: bandgap regression benchmark.
- Book anchor: [Neuer 4.2.2].
32. **Common anti-pattern: leaderboard optimization only**
- Compare **common anti-pattern: leaderboard optimization only** using one concrete materials example and one common failure mode.
- Applied anchor: grouped split by composition.
- Book anchor: [Neuer 4.5.9].
33. **Uncertainty-aware regression preview**
- Diagnose **uncertainty-aware regression preview** using one concrete materials example and one common failure mode.
- Applied anchor: nested CV study.
- Book anchor: [McClarren Ch4+6].
34. **Random split vs grouped split**
- Contrast the apparent score gain from random splits with the more realistic performance from grouped chemistry-aware evaluation.
- Applied anchor: side-by-side split comparison.
- Book anchor: [Neuer 4.5.9].
35. **Learning curves as sample-efficiency diagnostics**
- Use train/test curves to decide whether the next bottleneck is data, features, or model class.
- Applied anchor: learning curve comparison.
- Book anchor: [Murphy model selection].
36. **Residual structure matters more than one scalar metric**
- Inspect residuals by chemistry family, target scale, and hard examples.
- Applied anchor: residuals by prototype.
- Book anchor: [Bishop 3.1–3.3].
37. **When a low-complexity model should win**
- Explain why simple baselines are scientifically valuable and often preferable in small-data settings.
- Applied anchor: ridge beating a deeper model.
- Book anchor: [McClarren Ch4+6].
38. **OOD behavior in unseen chemistry**
- Show how a model can look strong in-domain and collapse when compositions shift.
- Applied anchor: hold-out chemistry family test.
- Book anchor: [Neuer 4.2.2].
39. **OOD behavior in unseen structure prototypes**
- Separate chemistry shift from structure shift in evaluation design.
- Applied anchor: prototype-holdout comparison.
- Book anchor: [Neuer 4.5.9].
40. **Trust checks before using a surrogate**
- Require baseline comparison, residual sanity checks, and a credible split before acting on predictions.
- Applied anchor: surrogate deployment checklist.
- Book anchor: [McClarren Ch4+6].
41. **Bridge to Unit 8 neural surrogates**
- Show what neural models must beat and why benchmark discipline carries forward.
- Applied anchor: linear/RF/NN comparison table.
- Book anchor: [Bishop 3.1–3.3].
42. **Bridge to Unit 12 uncertainty-aware screening**
- Connect generalization failure directly to the need for uncertainty-aware decision rules later.
- Applied anchor: confident-but-wrong predictor example.
- Book anchor: [Murphy model selection].
43. **Exercise setup and dataset definition**
- Define one target property, one feature set, one grouped split protocol, and fixed evaluation metrics.
- Applied anchor: bandgap regression benchmark.
- Book anchor: [Neuer 4.2.2].
44. **Exercise task 1: build the baseline benchmark**
- Train linear and regularized models first and record all preprocessing assumptions.
- Applied anchor: ridge/lasso benchmark.
- Book anchor: [Neuer 4.5.9].
45. **Exercise task 2: compare one nonlinear baseline**
- Add a tree-based or shallow neural model under the exact same split and compare fairly.
- Applied anchor: RF versus ridge comparison.
- Book anchor: [McClarren Ch4+6].
46. **Exercise task 3: inspect learning curves and residuals**
- Use error trends and residual plots to diagnose underfitting, overfitting, and family-specific failure.
- Applied anchor: learning curve plus residual dashboard.
- Book anchor: [Bishop 3.1–3.3].
47. **Exercise task 4: analyze one generalization failure**
- Identify one chemistry or prototype family where performance degrades sharply.
- Applied anchor: chemistry-family failure case.
- Book anchor: [Murphy model selection].
48. **Exercise task 5: propose one mitigation**
- Suggest one improvement grounded in evidence: more data, better split design, new features, or a new model.
- Applied anchor: mitigation memo.
- Book anchor: [Neuer 4.2.2].
49. **Exam-oriented key statements**
- Summarize high-yield statements in concise written-exam style with definitions and caveats.
- Applied anchor: residuals by prototype.
- Book anchor: [Bishop 3.1–3.3].
50. **Summary, next-unit bridge, and references**
- Consolidate the unit into a checklist: concepts, pitfalls, and decisions for next-unit transfer.
- Applied anchor: learning curve comparison.
- Book anchor: [Murphy model selection].
