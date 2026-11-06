// Small dense linear algebra for the explainers. Matrices are row-major Float64Arrays.

// Symmetric eigen-decomposition by cyclic Jacobi rotations. A is n x n (not modified).
// Returns {values, vectors} sorted by decreasing eigenvalue; vectors[k] is the k-th eigenvector.
export function eigSym(A, n, sweeps = 60) {
  const a = Float64Array.from(A), v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sw = 0; sw < sweeps; sw++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const th = (a[q * n + q] - a[p * n + p]) / (2 * apq);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {                       // A <- A J
        const akp = a[k * n + p], akq = a[k * n + q];
        a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {                       // A <- Jᵀ A
        const apk = a[p * n + k], aqk = a[q * n + k];
        a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {                       // V <- V J
        const vkp = v[k * n + p], vkq = v[k * n + q];
        v[k * n + p] = c * vkp - s * vkq; v[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[j * n + j] - a[i * n + i]);
  return {
    values: idx.map((i) => a[i * n + i]),
    vectors: idx.map((i) => Float64Array.from({ length: n }, (_, k) => v[k * n + i])),
  };
}

// Solve the n x n system A x = b by Gaussian elimination with partial pivoting.
export function solve(A, b, n) {
  const M = Float64Array.from(A), x = Float64Array.from(b);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[p * n + c])) p = r;
    if (p !== c) { for (let k = 0; k < n; k++) [M[c * n + k], M[p * n + k]] = [M[p * n + k], M[c * n + k]]; [x[c], x[p]] = [x[p], x[c]]; }
    const d = M[c * n + c] || 1e-300;
    for (let r = c + 1; r < n; r++) {
      const f = M[r * n + c] / d;
      if (!f) continue;
      for (let k = c; k < n; k++) M[r * n + k] -= f * M[c * n + k];
      x[r] -= f * x[c];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let sum = x[r];
    for (let k = r + 1; k < n; k++) sum -= M[r * n + k] * x[k];
    x[r] = sum / (M[r * n + r] || 1e-300);
  }
  return x;
}

// Gram matrix XᵀX and Xᵀy for X with m rows and n columns (row-major).
export function normalEq(X, y, m, n, w) {
  const G = new Float64Array(n * n), r = new Float64Array(n);
  for (let i = 0; i < m; i++) {
    const wi = w ? w[i] : 1;
    for (let a = 0; a < n; a++) {
      const xa = X[i * n + a] * wi;
      r[a] += xa * y[i];
      for (let b = a; b < n; b++) G[a * n + b] += xa * X[i * n + b];
    }
  }
  for (let a = 0; a < n; a++) for (let b = 0; b < a; b++) G[a * n + b] = G[b * n + a];
  return { G, r };
}

// Least squares: argmin_w ||X w - y||² (optionally weighted), via the normal equations.
export function lstsq(X, y, m, n, w) {
  const { G, r } = normalEq(X, y, m, n, w);
  return solve(G, r, n);
}

// Non-negative least squares (Lawson-Hanson active set), for small n.
export function nnls(X, y, m, n, iters = 200) {
  const { G, r } = normalEq(X, y, m, n);
  const x = new Float64Array(n), P = new Array(n).fill(false);
  const grad = () => Float64Array.from({ length: n }, (_, j) => { let s = r[j]; for (let k = 0; k < n; k++) s -= G[j * n + k] * x[k]; return s; });
  for (let it = 0; it < iters; it++) {
    const g = grad();
    let j = -1, best = 1e-10 * (1 + Math.max(...r.map(Math.abs)));
    for (let k = 0; k < n; k++) if (!P[k] && g[k] > best) { best = g[k]; j = k; }
    if (j < 0) break;
    P[j] = true;
    for (;;) {
      const idx = P.map((p, k) => (p ? k : -1)).filter((k) => k >= 0), q = idx.length;
      const Gs = new Float64Array(q * q), rs = new Float64Array(q);
      idx.forEach((a, i) => { rs[i] = r[a]; idx.forEach((b, k) => { Gs[i * q + k] = G[a * n + b]; }); });
      const z = solve(Gs, rs, q);
      if ([...z].every((v) => v > 0)) { x.fill(0); idx.forEach((a, i) => { x[a] = z[i]; }); break; }
      let alpha = Infinity;
      idx.forEach((a, i) => { if (z[i] <= 0) alpha = Math.min(alpha, x[a] / (x[a] - z[i])); });
      idx.forEach((a, i) => { x[a] += alpha * (z[i] - x[a]); if (x[a] <= 1e-12) { x[a] = 0; P[a] = false; } });
    }
  }
  return x;
}

export const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export const norm = (a) => Math.sqrt(dot(a, a));

// ---------------------------------------------------------------------------------------------
// NMF by multiplicative updates (Lee & Seung 2001). X is N x D (row-major, ≥ 0), W is N x K, H is K x D.
// loss "fro": ||X − WH||²_F (Gaussian); loss "kl": generalised KL divergence (Poisson likelihood).
// nmfStep() does one update of H and one of W in place and returns the loss.
// ---------------------------------------------------------------------------------------------
export function nmfInit(X, N, D, K, seed = 1) {
  let s = seed >>> 0;
  const u = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  let m = 0;
  for (let i = 0; i < N * D; i++) m += X[i];
  m /= N * D;
  const sc = Math.sqrt(m / K);
  return { W: Float64Array.from({ length: N * K }, () => sc * (0.2 + u())), H: Float64Array.from({ length: K * D }, () => sc * (0.2 + u())) };
}

export function nmfStep(X, W, H, N, D, K, loss = "kl") {
  const EPS = 1e-12, WH = new Float64Array(N * D);
  const prod = () => {
    WH.fill(0);
    for (let i = 0; i < N; i++) for (let k = 0; k < K; k++) {
      const w = W[i * K + k];
      if (!w) continue;
      const o = i * D, hk = k * D;
      for (let j = 0; j < D; j++) WH[o + j] += w * H[hk + j];
    }
  };
  prod();
  if (loss === "kl") {
    // H <- H ⊙ Wᵀ(X / WH) / Wᵀ1
    const num = new Float64Array(K * D), den = new Float64Array(K);
    for (let i = 0; i < N; i++) for (let k = 0; k < K; k++) {
      const w = W[i * K + k]; den[k] += w;
      const o = i * D, hk = k * D;
      for (let j = 0; j < D; j++) num[hk + j] += w * X[o + j] / (WH[o + j] + EPS);
    }
    for (let k = 0; k < K; k++) for (let j = 0; j < D; j++) H[k * D + j] *= num[k * D + j] / (den[k] + EPS);
    prod();
    // W <- W ⊙ (X / WH) Hᵀ / 1Hᵀ
    const hs = new Float64Array(K);
    for (let k = 0; k < K; k++) for (let j = 0; j < D; j++) hs[k] += H[k * D + j];
    for (let i = 0; i < N; i++) for (let k = 0; k < K; k++) {
      let a = 0;
      const o = i * D, hk = k * D;
      for (let j = 0; j < D; j++) a += X[o + j] / (WH[o + j] + EPS) * H[hk + j];
      W[i * K + k] *= a / (hs[k] + EPS);
    }
  } else {
    // H <- H ⊙ WᵀX / WᵀWH ;  W <- W ⊙ XHᵀ / WHHᵀ
    const WtX = new Float64Array(K * D), WtW = new Float64Array(K * K);
    for (let i = 0; i < N; i++) for (let k = 0; k < K; k++) {
      const w = W[i * K + k], o = i * D, hk = k * D;
      for (let j = 0; j < D; j++) WtX[hk + j] += w * X[o + j];
      for (let l = 0; l < K; l++) WtW[k * K + l] += w * W[i * K + l];
    }
    for (let k = 0; k < K; k++) for (let j = 0; j < D; j++) {
      let d = 0;
      for (let l = 0; l < K; l++) d += WtW[k * K + l] * H[l * D + j];
      H[k * D + j] *= WtX[k * D + j] / (d + EPS);
    }
    const HHt = new Float64Array(K * K);
    for (let k = 0; k < K; k++) for (let l = 0; l < K; l++) { let a = 0; for (let j = 0; j < D; j++) a += H[k * D + j] * H[l * D + j]; HHt[k * K + l] = a; }
    for (let i = 0; i < N; i++) {
      const o = i * D;
      for (let k = 0; k < K; k++) {
        let a = 0, d = 0;
        for (let j = 0; j < D; j++) a += X[o + j] * H[k * D + j];
        for (let l = 0; l < K; l++) d += W[i * K + l] * HHt[l * K + k];
        W[i * K + k] *= a / (d + EPS);
      }
    }
  }
  prod();
  let L = 0;
  for (let t = 0; t < N * D; t++) {
    const x = X[t], m = WH[t] + EPS;
    L += loss === "kl" ? (x > 0 ? x * Math.log(x / m) : 0) - x + m : (x - m) ** 2;
  }
  return L;
}

// PCA of a small data matrix X (N x D) through the D x D covariance. Returns the mean, the
// eigenvalues (variances) and the eigenvectors (eigenspectra), largest first.
export function pca(X, N, D) {
  const mu = new Float64Array(D);
  for (let i = 0; i < N; i++) for (let j = 0; j < D; j++) mu[j] += X[i * D + j] / N;
  const C = new Float64Array(D * D), row = new Float64Array(D);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < D; j++) row[j] = X[i * D + j] - mu[j];
    for (let a = 0; a < D; a++) { const ra = row[a]; if (!ra) continue; for (let b = a; b < D; b++) C[a * D + b] += ra * row[b]; }
  }
  for (let a = 0; a < D; a++) for (let b = a; b < D; b++) { C[a * D + b] /= N - 1; C[b * D + a] = C[a * D + b]; }
  const { values, vectors } = eigSym(C, D);
  return { mean: mu, values, vectors };
}
