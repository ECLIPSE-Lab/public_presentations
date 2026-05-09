# Unit 2 Plan — Mathematical Foundations of AI & ML

## Unit title
Linear Algebra for Machine Learning (from geometry to optimization)

## Audience/profile
- 5th semester undergrad (KI-Materialtechnologie)
- Assumed: basic LA + SVD familiarity
- Goal: unify notation and intuition for all downstream ML lectures

## Learning objectives
By the end of Unit 2 students can:
1. Interpret vectors, matrices, and linear maps geometrically for ML.
2. Explain rank/nullspace/conditioning and why they matter for learning.
3. Use SVD/eigendecomposition to reason about data structure and model stability.
4. Connect least-squares, projections, and regularization in one framework.
5. Distinguish lecture-essential LA concepts from exercise implementation work.

## 90-min structure
- 0–10: recap + notation contract
- 10–30: vector spaces, basis, linear maps, projections
- 30–50: matrix properties (rank, nullspace, conditioning)
- 50–70: eigendecomposition, SVD, PCA link
- 70–85: least-squares geometry + regularization lens
- 85–90: summary + exercise handoff

## Exercise (90 min)
- projection and reconstruction in NumPy
- compute SVD and low-rank approximations
- solve least squares with normal equations / `lstsq`
- compare ill-conditioned vs well-conditioned systems

## Book mapping (priority)
1. Neuer: model and data-mapping perspective, uncertainty-aware interpretation
2. Sandfeld: data geometry and ML pipeline relevance
3. McClarren: linear models for engineering tasks
4. Murphy/Bishop: model selection + probabilistic framing links
