// Validate group (Act 6), on real data of ten specimens (STEM-Bank-26, one spectrum image each):
//   split-leak : predict the thickness t/λ (log-ratio from the low-loss EELS) from the ADF intensity
//                (and the plasmon energy). Random pixel split versus GroupKFold by specimen.
//   prep-leak  : supervised feature selection before the split. 1000 pure-noise features, real
//                targets: selecting on all data gives R² > 0 on noise; selecting inside the fold not.
import { register, T, SERIES, fmt, clamp, rng, h, s, row, col, card, controls, readout, slider, toggle, button, legend, chart } from "./core.js";
import { loadData } from "./data.js";
import { lstsq } from "./la.js";

if (!document.getElementById("xp-val-style")) {
  document.head.append(h("style", { id: "xp-val-style" }, `
.xp-val-side { width: 640px; display: flex; flex-direction: column; gap: 14px; }
.xp-val-side .xp-readout { margin-top: 0; line-height: 1.5; }
.xp-src { font-size: 17px; color: #64748b; margin-top: 4px; }
.xp-big2 { font-family: Outfit, sans-serif; font-size: 64px; font-weight: 800; line-height: 1; }
`));
}
const src = (html) => h("div", { class: "xp-src", html });
const SRC = "Data: ten spectrum images (one per session) from STEM-Bank-26 (Hollyer &amp; Stach 2026), CC BY 4.0; 400 pixels each.";
const waiting = (el) => { const w = h("div", { class: "xp-caption" }, "loading data …"); el.append(w); return w; };
export const SPEC_COL = ["#60a5fa", "#fb923c", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#22d3ee", "#f87171", "#a3e635", "#e2e8f0"];

// Folds: random (shuffled pixels) or grouped (whole specimens).
function folds(n, g, kind, K = 5, seed = 1) {
  const f = new Int32Array(n);
  if (kind === "group") {
    for (let i = 0; i < n; i++) f[i] = g[i] % K;                     // 10 specimens, 5 folds: 2 specimens per fold
  } else {
    const R = rng(seed), idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(R.u() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    idx.forEach((i, r) => { f[i] = r % K; });
  }
  return f;
}
const r2 = (y, p) => { const m = y.reduce((a, b) => a + b, 0) / y.length; let ss = 0, st = 0; y.forEach((v, i) => { ss += (v - p[i]) ** 2; st += (v - m) ** 2; }); return 1 - ss / st; };
const mae = (y, p) => y.reduce((a, v, i) => a + Math.abs(v - p[i]), 0) / y.length;

// Cross-validated predictions. The scaler is fitted inside each fold (training rows only).
function cvPredict(X, y, f, model, k) {
  const n = y.length, d = X[0].length, pred = new Float64Array(n), K = Math.max(...f) + 1;
  for (let fold = 0; fold < K; fold++) {
    const tr = [], te = [];
    for (let i = 0; i < n; i++) (f[i] === fold ? te : tr).push(i);
    const mu = Array(d).fill(0), sd = Array(d).fill(0);
    tr.forEach((i) => X[i].forEach((v, j) => { mu[j] += v / tr.length; }));
    tr.forEach((i) => X[i].forEach((v, j) => { sd[j] += (v - mu[j]) ** 2 / tr.length; }));
    const Z = (i) => X[i].map((v, j) => (v - mu[j]) / (Math.sqrt(sd[j]) || 1));
    if (model === "linear") {
      const A = new Float64Array(tr.length * (d + 1)), b = new Float64Array(tr.length);
      tr.forEach((i, r) => { const z = Z(i); A[r * (d + 1)] = 1; z.forEach((v, j) => { A[r * (d + 1) + j + 1] = v; }); b[r] = y[i]; });
      const w = lstsq(A, b, tr.length, d + 1);
      te.forEach((i) => { const z = Z(i); pred[i] = w[0] + z.reduce((a, v, j) => a + v * w[j + 1], 0); });
    } else {
      const Ztr = tr.map(Z);
      te.forEach((i) => {
        const z = Z(i), best = [];
        Ztr.forEach((q, r) => {
          let dd = 0; for (let j = 0; j < d; j++) dd += (q[j] - z[j]) ** 2;
          if (best.length < k || dd < best[best.length - 1][0]) { best.push([dd, y[tr[r]]]); best.sort((a, c) => a[0] - c[0]); if (best.length > k) best.pop(); }
        });
        pred[i] = best.reduce((a, q) => a + q[1], 0) / best.length;
      });
    }
  }
  return pred;
}

// ---------------------------------------------------------------------------------------------
// split-leak
// ---------------------------------------------------------------------------------------------
register("split-leak", (el) => {
  const wait = waiting(el);
  loadData("thick").then((d) => {
    wait.remove();
    const n = d.t.length, g = Array.from(d.g), y = Array.from(d.t), G = d.specs.length;
    let split = "random", model = "knn", k = 5, feats = "both";
    const cA = chart({ width: 620, height: 520, x: { domain: [0, 1], label: "ADF intensity (detector units, ×10⁶)" }, y: { domain: [-0.2, 2.2], label: "t/λ  (from low-loss EELS)" }, margin: { l: 90 } });
    const cP = chart({ width: 560, height: 520, x: { domain: [-0.2, 2.2], label: "true t/λ" }, y: { domain: [-0.2, 2.2], label: "predicted t/λ (test folds)" }, margin: { l: 90 } });
    cP.setDomain();
    const big = h("div", { class: "xp-big2" }), out = readout();
    const adf = Array.from(d.adf, (v) => v / 1e6), amax = Math.max(...adf);
    cA.setDomain([0, amax * 1.05], null);
    // scatter of the data, coloured by specimen (drawn once)
    for (let s2 = 0; s2 < G; s2++) cA.dots(adf.map((a, i) => [a, y[i]]).filter((_, i) => g[i] === s2), { r: 2.6, fill: SPEC_COL[s2], opacity: 0.55, stroke: "none" });
    function draw() {
      const X = adf.map((a, i) => (feats === "both" ? [a, d.ep[i]] : [a]));
      const f = folds(n, g, split === "random" ? "random" : "group");
      const p = cvPredict(X, y, f, model, k);
      cP.clear();
      cP.line([[-0.2, -0.2], [2.2, 2.2]], { stroke: T.faint, width: 2, dash: "6 6" });
      for (let s2 = 0; s2 < G; s2++) cP.dots(y.map((v, i) => [v, p[i]]).filter((_, i) => g[i] === s2), { r: 2.6, fill: SPEC_COL[s2], opacity: 0.55, stroke: "none" });
      const R2 = r2(y, Array.from(p)), M = mae(y, Array.from(p));
      big.innerHTML = `R² = <span style="color:${R2 > 0.3 ? T.truth : T.thread}">${fmt(R2, 2)}</span>`;
      // per-specimen R² under the current split (how well does it work for each new specimen?)
      out.set(`${split === "random" ? "random 5-fold over pixels" : "GroupKFold: whole specimens held out"} · ${model === "knn" ? `${k}-NN` : "linear"} · ${feats === "both" ? "ADF + E<sub>p</sub>" : "ADF only"}<br>` +
        `MAE = <b>${fmt(M, 3)}</b> t/λ &nbsp; (spread of t/λ: ${fmt(Math.sqrt(y.reduce((a, v) => a + (v - y.reduce((q, w) => q + w, 0) / n) ** 2, 0) / n), 2)})<br>` +
        `within one specimen, ADF and t/λ correlate with r = 0.63–0.99;<br>across specimens the detector units differ`);
    }
    const sp = toggle({ label: "split", options: [{ value: "random", label: "random pixels" }, { value: "group", label: "by specimen (GroupKFold)" }], value: split, onChange: (v) => { split = v; draw(); } });
    const mt = toggle({ label: "model", options: [{ value: "linear", label: "linear" }, { value: "knn", label: "k-NN" }], value: model, onChange: (v) => { model = v; draw(); } });
    const ks = slider({ label: "k", min: 1, max: 30, step: 1, value: k, digits: 0, onInput: (v) => { k = Math.round(v); draw(); } });
    const ft = toggle({ label: "features", options: [{ value: "adf", label: "ADF" }, { value: "both", label: "ADF + plasmon E<sub>p</sub>" }], value: feats, onChange: (v) => { feats = v; draw(); } });
    el.append(controls(sp, mt, ks, ft),
      row(col(cA.el, src(SRC)), col(cP.el), h("div", { class: "xp-val-side", style: { width: "560px" } }, big, card(out),
        legend(d.specs.map((q, i) => [SPEC_COL[i], q.name]).slice(0, 10)))));
    draw();
  });
});

// ---------------------------------------------------------------------------------------------
// prep-leak
// ---------------------------------------------------------------------------------------------
register("prep-leak", (el) => {
  const wait = waiting(el);
  loadData("thick").then((d) => {
    wait.remove();
    // 80 random pixels of the ten specimens (real targets); the features are pure noise
    const R0 = rng(11), N = 80, P = 1000;
    const pick = Array.from({ length: N }, () => Math.floor(R0.u() * d.t.length));
    const y = pick.map((i) => d.t[i]);
    let seed = 1, kSel = 10, inside = false, Xn;
    const noise = () => { const R = rng(seed); Xn = Array.from({ length: N }, () => Float64Array.from({ length: P }, () => R.normal())); };
    const corrSel = (rows, kk) => {
      const m = rows.length, ym = rows.reduce((a, i) => a + y[i], 0) / m, sc = new Float64Array(P);
      for (let j = 0; j < P; j++) {
        let xm = 0; rows.forEach((i) => { xm += Xn[i][j]; }); xm /= m;
        let sxy = 0, sxx = 0; rows.forEach((i) => { const a = Xn[i][j] - xm; sxy += a * (y[i] - ym); sxx += a * a; });
        sc[j] = Math.abs(sxy) / Math.sqrt(sxx || 1);
      }
      return Array.from(sc.keys()).sort((a, b) => sc[b] - sc[a]).slice(0, kk);
    };
    const cP = chart({ width: 620, height: 560, x: { domain: [-0.2, 2.2], label: "true t/λ" }, y: { domain: [-0.6, 2.6], label: "predicted t/λ (test folds)" }, margin: { l: 90 } });
    cP.setDomain();
    const big = h("div", { class: "xp-big2" }), out = readout();
    function draw() {
      const K = 5, f = folds(N, [], "random", K, 7), p = new Float64Array(N);
      const all = inside ? null : corrSel(Array.from({ length: N }, (_, i) => i), kSel);
      for (let fold = 0; fold < K; fold++) {
        const tr = [], te = [];
        for (let i = 0; i < N; i++) (f[i] === fold ? te : tr).push(i);
        const S = inside ? corrSel(tr, kSel) : all, q = S.length + 1;
        const A = new Float64Array(tr.length * q), b = new Float64Array(tr.length);
        tr.forEach((i, r) => { A[r * q] = 1; S.forEach((j, c) => { A[r * q + c + 1] = Xn[i][j]; }); b[r] = y[i]; });
        const w = lstsq(A, b, tr.length, q);
        te.forEach((i) => { p[i] = w[0] + S.reduce((a, j, c) => a + w[c + 1] * Xn[i][j], 0); });
      }
      const R2 = r2(y, Array.from(p));
      cP.clear(); cP.line([[-0.2, -0.2], [2.2, 2.2]], { stroke: T.faint, width: 2, dash: "6 6" });
      cP.dots(y.map((v, i) => [v, clamp(p[i], -0.6, 2.6)]), { r: 6, fill: inside ? T.truth : T.thread, opacity: 0.8, stroke: "none" });
      big.innerHTML = `R² = <span style="color:${inside ? T.truth : T.thread}">${fmt(R2, 2)}</span>`;
      out.set(`${N} pixels, real t/λ; <b>${P} features of pure noise</b><br>` +
        `select the ${kSel} features most correlated with t/λ <b>${inside ? "inside each training fold" : "on all 80 pixels, then cross-validate"}</b><br>` +
        (inside ? `honest: noise cannot predict thickness (R² ≤ 0)` : `<span class="bad">leak: the test pixels helped to choose the features</span>`));
    }
    const ks = slider({ label: "selected features k", min: 1, max: 40, step: 1, value: kSel, digits: 0, onInput: (v) => { kSel = Math.round(v); draw(); } });
    const it = toggle({ label: "selection", options: [{ value: "all", label: "before the split (all data)" }, { value: "in", label: "inside each fold (Pipeline)" }], value: "all", onChange: (v) => { inside = v === "in"; draw(); } });
    el.append(controls(it, ks, button("new noise", () => { seed++; noise(); draw(); })),
      row(cP.el, h("div", { class: "xp-val-side" }, big, card(out), src("Targets: real t/λ of 80 pixels (STEM-Bank-26, CC BY 4.0). Features: Gaussian noise. 5-fold CV, linear regression."))));
    noise(); draw();
  });
});
