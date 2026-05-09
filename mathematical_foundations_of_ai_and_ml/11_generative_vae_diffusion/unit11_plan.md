# Unit 11 Plan — Mathematical Foundations of AI & ML

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: Units 1--10 (supervised learning, linear algebra, optimization, PCA/SVD, neural networks, autoencoders)
- Assumed: ERM, gradient descent, latent representations from AE unit, basic probability
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)

## Learning objectives (Unit 11)
By the end of the unit, students can:
1. Explain why unsupervised learning is essential when labels are unavailable (process data, sensor streams).
2. State the K-Means objective and derive the assignment-update algorithm as coordinate descent on the distortion cost.
3. Formulate a Gaussian Mixture Model as a latent variable model with discrete hidden cluster indicators.
4. Derive the E-step (responsibilities) and M-step (parameter updates) of EM for GMMs.
5. Explain why EM maximizes a lower bound on the observed-data log-likelihood and why convergence is guaranteed to a local optimum.
6. Compare hard clustering (K-Means) and soft clustering (GMM/EM) and recognize K-Means as a special case of EM.
7. Describe K-Medoids as a robust alternative and Bernoulli mixtures for binary data.
8. Apply clustering concepts to engineering tasks: image segmentation, vector quantization, regime identification.

## Book-aligned content mapping (priority order)
1. Murphy (2012): Ch. 11.1--11.5 (LVMs, mixture models, EM algorithm, lower bound, model selection).
2. Neuer (2024): Ch. 5.3 (K-Means algorithm, cluster variance objective, batch K-Means).
3. McClarren (2021): Ch. 4.3 (K-Means clustering, loss function, choosing K).
4. Bishop (2006): Ch. 9.2--9.4 (GMM, EM derivation, lower bound view) -- referenced but not primary source.

## 90-minute structure
- 0--10 min: Unsupervised paradigm recap and motivation (why labels are missing in engineering)
- 10--25 min: K-Means algorithm -- objective, assignment-update cycle, convergence, initialization
- 25--40 min: From hard to soft -- GMM as probabilistic clustering, latent variables, responsibilities
- 40--55 min: EM algorithm -- E-step, M-step, auxiliary function, closed-form updates for GMM
- 55--70 min: Lower bound maximization -- Jensen's inequality, monotonic convergence, EM as bound optimization
- 70--80 min: Variants and extensions -- K-Medoids, Bernoulli mixtures, model selection for K
- 80--88 min: Engineering applications -- image segmentation, vector quantization, regime identification
- 88--90 min: Summary + exercise handoff

## Exercise (90 min)
- Implement K-Means from scratch in NumPy on 2D synthetic Gaussian clusters
- Implement EM for a 2-component GMM: compute responsibilities, update parameters, track log-likelihood
- Compare K-Means hard assignments vs GMM soft assignments on overlapping clusters
- Bonus: compare clustering-based embeddings to autoencoder embeddings from Unit 10

## Assessment alignment
- Written exam prep: EM derivation steps, responsibility formula, K-Means objective, lower bound argument.
- Every unit introduces 3--5 exam-relevant "must-know" statements.

## Slide production notes
- Style: follow data_science_for_em revealjs pattern
- Include structured recap + glossary slide
- Add citation placeholders for each reused concept/figure
