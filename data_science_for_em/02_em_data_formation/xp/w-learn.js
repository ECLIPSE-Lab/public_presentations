// Act 1 "Learn": what a model f_θ learns from EM data.
//   learning-types : supervised (logistic classifier), unsupervised (k-means), self-supervised (Noise2Noise)
//   bias-variance  : polynomial vs physical model of a carbon K-edge; error on training and new data
//   data-zoo       : the tensors of EM (HAADF, 4D-STEM, EELS/EDS, tilt series), their shapes and sizes
//   box-spectrum   : white-box / grey-box / black-box models, from Bragg's law to foundation models
// All data are synthetic. Fixed seeds give the same default view on every load.
import { register, T, rng, linspace, mean, clamp, fmt, nelderMead, h, s, row, col, card, controls, caption,
  readout, slider, toggle, button, legend, chart, imageCanvas, sup } from "./core.js";

// One <style> element for the widget-specific classes of this file.
function css(id, text) {
  if (document.getElementById(id)) return;
  document.head.append(h("style", { id }, text));
}

// =============================================================================================
// 1. learning-types
// Two "diffraction features" per pattern: the counts in spot g₁ and in spot g₂. Phase A has a
// strong g₂, phase B a strong g₁. The specimen thickness t scales both spots (t = 0.55…1.45),
// and the counts are Poisson. 80 patterns, 40 per phase, about 6 % overlap.
// Supervised: logistic regression on standardised features, full-batch gradient descent, η = 0.4.
// Unsupervised: Lloyd's k-means from fixed start centroids in the low-count corner.
// Self-supervised: Noise2Noise. Clean image λ(x) = 0.15 + columns (2.6 and 1.0 counts peak,
// σ = 1.3 px) on an 8 px square lattice, 64 x 64 px. Each frame is Poisson(λ). The mean of N
// frames has RMSE √(λ̄/N) against λ: the noise averages out because E[frame] = λ.
// =============================================================================================
const LT_N = 80, LT_DOM = [0, 66];
function phaseData(seed = 11) {
  const r = rng(seed), pts = [];
  for (let i = 0; i < LT_N; i++) {
    const lab = i % 2, t = 0.55 + 0.9 * r.u(), m = lab ? [36, 22] : [22, 36];
    pts.push({ x: r.poisson(t * m[0]), y: r.poisson(t * m[1]), lab });
  }
  return pts;
}

const N2N_N = 64, N2N_MAX = 256;
function cleanLattice(n = N2N_N, a = 8, sig = 1.3) {
  const im = new Float64Array(n * n).fill(0.15);
  for (let p = 0; p < n / a; p++) for (let q = 0; q < n / a; q++) {
    const cx = a / 2 + a * q, cy = a / 2 + a * p, amp = (p + q) % 2 ? 1.0 : 2.6;
    for (let y = Math.max(0, cy - 5); y < Math.min(n, cy + 5); y++)
      for (let x = Math.max(0, cx - 5); x < Math.min(n, cx + 5); x++)
        im[y * n + x] += amp * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig * sig));
  }
  return im;
}

register("learning-types", (el, api) => {
  css("xp-learning-types-css", `
    .xp-learning-types-body { height: 560px; }
    .xp-learning-types-body > div { height: 100%; }
    .xp-learning-types-ctl { display: flex; gap: 18px; align-items: center; min-height: 52px; }`);
  const P = phaseData(), Np = P.length;
  const xs = P.map((p) => p.x), ys = P.map((p) => p.y);
  const mx = mean(xs), my = mean(ys);
  const sx = Math.sqrt(mean(xs.map((v) => (v - mx) ** 2))), sy = Math.sqrt(mean(ys.map((v) => (v - my) ** 2)));
  const AX = { x: { domain: LT_DOM, label: "feature 1: counts in spot g₁" }, y: { domain: LT_DOM, label: "feature 2: counts in spot g₂" } };
  const CL = [T.gauss, T.pois, T.violet, T.pink];
  let mode = "Supervised";

  // ---------- supervised: logistic regression by gradient descent ----------
  const sc = chart({ width: 820, height: 540, ...structuredClone(AX) });
  const lc = chart({ width: 780, height: 540, x: { domain: [0, 300], label: "gradient-descent step" },
    y: { domain: [0, 0.9], label: "training loss (log-loss)" } });
  const W0 = [0.6, 0.6, 0.3];
  let w, step, hist, training = false;
  const evalLR = () => {
    let L = 0, acc = 0; const g = [0, 0, 0], wrong = [];
    for (const p of P) {
      const z1 = (p.x - mx) / sx, z2 = (p.y - my) / sy, q = 1 / (1 + Math.exp(-(w[0] * z1 + w[1] * z2 + w[2])));
      L -= p.lab ? Math.log(q) : Math.log(1 - q);
      if ((q > 0.5) === !!p.lab) acc++; else wrong.push(p);
      g[0] += (q - p.lab) * z1; g[1] += (q - p.lab) * z2; g[2] += q - p.lab;
    }
    return { L: L / Np, acc: acc / Np, g: g.map((v) => v / Np), wrong };
  };
  const resetSup = () => { w = W0.slice(); step = 0; training = false; hist = [[0, evalLR().L]]; drawSup(); };
  function drawSup() {
    const e = evalLR();
    sc.clear();
    const G = 22, d = (LT_DOM[1] - LT_DOM[0]) / G;                      // probability shading
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      const x = LT_DOM[0] + (j + 0.5) * d, y = LT_DOM[0] + (i + 0.5) * d;
      const q = 1 / (1 + Math.exp(-(w[0] * (x - mx) / sx + w[1] * (y - my) / sy + w[2])));
      sc.rect(x - d / 2, y - d / 2, x + d / 2, y + d / 2, { fill: q > 0.5 ? T.pois : T.gauss, opacity: 0.32 * Math.abs(q - 0.5) });
    }
    const a = w[0] / sx, b = w[1] / sy, c = w[2] - w[0] * mx / sx - w[1] * my / sy;   // a x + b y + c = 0
    const t = linspace(-20, 90, 2);
    sc.line(Math.abs(b) > Math.abs(a) ? t.map((x) => [x, -(a * x + c) / b]) : t.map((y) => [-(b * y + c) / a, y]), { stroke: T.thread, width: 5 });
    sc.dots(P.filter((p) => !p.lab), { fill: T.gauss, r: 8 });
    sc.dots(P.filter((p) => p.lab), { fill: T.pois, r: 8 });
    sc.dots(e.wrong, { fill: "none", stroke: T.thread, strokeWidth: 3, r: 14 });
    sc.note("phase A", { corner: "tl", fill: T.gauss, weight: 700 });
    sc.note("phase B", { corner: "br", fill: T.pois, weight: 700 });
    lc.clear();
    lc.line(hist, { stroke: T.thread, width: 4 });
    lc.dots([hist[hist.length - 1]], { fill: T.thread, r: 9 });
    lc.note(`accuracy ${fmt(100 * e.acc, 0)} %`, { corner: "tr", size: 26, weight: 700 });
    lc.note(`log-loss ${fmt(e.L, 3)}`, { corner: "tr", line: 1, size: 22, fill: T.muted });
    if (mode === "Supervised") ro.set(step === 0
      ? `Untrained guess: training accuracy <span class="bad">${fmt(100 * e.acc, 0)} %</span>. ${e.wrong.length} patterns are on the wrong side (red rings). Press <b>train</b>.`
      : `Step ${step}: training accuracy <span class="${e.acc > 0.9 ? "good" : "warn"}">${fmt(100 * e.acc, 0)} %</span> · log-loss ${fmt(e.L, 3)} · ${e.wrong.length} patterns still wrong (overlap of the two phases).`);
    return e;
  }
  const supCtl = h("div", { class: "xp-learning-types-ctl" },
    button("train", () => { if (step >= 300) resetSup(); training = true; }, "primary"), button("reset", resetSup),
    legend([[T.gauss, "phase A"], [T.pois, "phase B"], [T.thread, "decision boundary f(x) = 0.5"]]));
  const supView = row(sc.el, lc.el);

  // ---------- unsupervised: k-means ----------
  const kc = chart({ width: 820, height: 540, ...structuredClone(AX) });
  const wc = chart({ width: 780, height: 540, x: { domain: [0, 14], label: "k-means iteration" },
    y: { domain: [4000, 100000], type: "log", label: "within-cluster sum of squares (counts²)", ticks: [5000, 10000, 20000, 50000, 100000], tickFormat: (v) => `${v / 1000}k` } });
  const INIT = [[10, 8], [16, 6], [8, 14], [20, 12]];
  const km = { K: 2 };
  const nearest = (C) => P.map((p) => { let bi = 0, bd = Infinity; C.forEach((c, j) => { const d = (p.x - c[0]) ** 2 + (p.y - c[1]) ** 2; if (d < bd) { bd = d; bi = j; } }); return bi; });
  const wcss = (C, as) => P.reduce((t, p, i) => t + (p.x - C[as[i]][0]) ** 2 + (p.y - C[as[i]][1]) ** 2, 0);
  const resetKm = () => {
    Object.assign(km, { C: INIT.slice(0, km.K).map((c) => c.slice()), from: null, to: null, t: 1, iter: 0, running: false, done: false, trail: [] });
    km.as = nearest(km.C); km.hist = [[0, wcss(km.C, km.as)]]; km.trail = km.C.map((c) => [c.slice()]);
    drawKm();
  };
  const kmStep = () => {                                         // update centroids, then tween to them
    if (km.done || km.t < 1) return;
    km.from = km.C.map((c) => c.slice());
    km.to = km.C.map((c, j) => { const m = P.filter((p, i) => km.as[i] === j); return m.length ? [mean(m.map((p) => p.x)), mean(m.map((p) => p.y))] : c.slice(); });
    km.t = 0;
  };
  const kmArrive = () => {                                       // tween finished: assign points again
    km.C = km.to; km.iter++;
    const as = nearest(km.C), same = as.every((a, i) => a === km.as[i]);
    km.as = as; km.hist.push([km.iter, wcss(km.C, km.as)]);
    km.C.forEach((c, j) => km.trail[j].push(c.slice()));
    if (same) { km.done = true; km.running = false; }
    drawKm();
    if (km.running) kmStep();
  };
  function drawKm() {
    kc.clear();
    for (let j = 0; j < km.K; j++) kc.dots(P.filter((p, i) => km.as[i] === j), { fill: CL[j], r: 8 });
    km.trail.forEach((tr, j) => kc.line([...tr, km.C[j]], { stroke: CL[j], width: 2, dash: "4 6" }));
    km.C.forEach((c, j) => {
      const X = kc.x(c[0]), Y = kc.y(c[1]);
      kc.gTop.append(s("path", { d: `M${X - 16},${Y - 16}L${X + 16},${Y + 16}M${X - 16},${Y + 16}L${X + 16},${Y - 16}`, stroke: "#0b0f19", "stroke-width": 11, "stroke-linecap": "round" }),
        s("path", { d: `M${X - 16},${Y - 16}L${X + 16},${Y + 16}M${X - 16},${Y + 16}L${X + 16},${Y - 16}`, stroke: CL[j], "stroke-width": 6, "stroke-linecap": "round" }));
    });
    kc.note("× = cluster centre", { corner: "tr", fill: T.muted });
    wc.clear();
    wc.line(km.hist, { stroke: T.thread, width: 4 });
    wc.dots(km.hist, { fill: T.thread, r: 7 });
    const W = km.hist[km.hist.length - 1][1];
    wc.note(`K = ${km.K}: WCSS ${fmt(W / 1000, 1)}k counts²`, { corner: "tr", size: 24, weight: 700 });
    if (mode !== "Unsupervised") return;
    let tail;
    if (!km.done) tail = `Each <b>step</b> moves a centre to the mean of its points.`;
    else if (km.K === 2) {
      const m = km.as.filter((a, i) => a === P[i].lab).length / Np;
      tail = `<span class="good">Converged:</span> the clusters match the hidden phase labels for <b>${fmt(100 * Math.max(m, 1 - m), 0)} %</b> of patterns.`;
    } else tail = `<span class="warn">Converged, but K = ${km.K} splits a phase.</span> WCSS always falls with K.`;
    ro.set(`Iteration ${km.iter} · WCSS <b>${fmt(W, 0)}</b> counts². ${tail}`);
  }
  const kSl = slider({ label: "clusters K", min: 2, max: 4, value: 2, step: 1, digits: 0, onInput: (v) => { km.K = Math.round(v); resetKm(); } });
  const kmCtl = h("div", { class: "xp-learning-types-ctl" }, kSl,
    button("step", () => { if (km.done) resetKm(); km.running = false; kmStep(); }),
    button("run", () => { if (km.done) resetKm(); km.running = true; kmStep(); }, "primary"),
    button("reset", resetKm));
  const kmView = row(kc.el, wc.el);

  // ---------- self-supervised: Noise2Noise ----------
  const clean = cleanLattice(), n2 = N2N_N * N2N_N, lamBar = mean(clean), vmax = Math.max(...clean) * 1.1;
  const r = rng(23), frameA = Float64Array.from(clean, (v) => r.poisson(v));
  const targets = new Uint8Array(N2N_MAX * n2), rmse = [NaN], acc = new Float64Array(n2);
  let nReady = 0;                                                // 1 M Poisson draws: 32 frames per chunk, so that mount stays fast
  const more = (redraw = true) => {
    for (const end = Math.min(N2N_MAX, nReady + 32); nReady < end; nReady++) {
      let e = 0;
      for (let i = 0; i < n2; i++) { const c = targets[nReady * n2 + i] = r.poisson(clean[i]); acc[i] += c; e += (acc[i] / (nReady + 1) - clean[i]) ** 2; }
      rmse.push(Math.sqrt(e / n2));
    }
    if (redraw && mode === "Self-supervised") drawN2N();
    if (nReady < N2N_MAX) setTimeout(more, 0);
  };
  more(false);
  const IM = 280;
  const imA = imageCanvas({ n: N2N_N, size: IM, label: "noisy frame A (input)" });
  const imB = imageCanvas({ n: N2N_N, size: IM, label: "noisy frame B (target)" });
  const imM = imageCanvas({ n: N2N_N, size: IM, label: "mean of N noisy targets" });
  const imC = imageCanvas({ n: N2N_N, size: IM, label: "clean (hidden, only to score)" });
  imA.draw(frameA, 0, vmax); imB.draw(targets.subarray(0, n2), 0, vmax); imC.draw(clean, 0, vmax);
  imC.canvas.style.opacity = 0.75; imC.canvas.style.borderColor = T.truth;
  const rc = chart({ width: 470, height: 520, x: { domain: [1, 256], type: "log", label: "N targets", ticks: [1, 4, 16, 64, 256] },
    y: { domain: [0.03, 1.5], type: "log", label: "RMSE vs clean (counts/px)", ticks: [0.03, 0.1, 0.3, 1] }, margin: { l: 84 } });
  const Ns = Array.from({ length: N2N_MAX }, (_, i) => i + 1);
  let nN = 16;
  function drawN2N() {
    const nn = Math.min(nN, nReady), mimg = new Float64Array(n2);
    for (let k = 0; k < nn; k++) for (let i = 0; i < n2; i++) mimg[i] += targets[k * n2 + i];
    for (let i = 0; i < n2; i++) mimg[i] /= nn;
    imM.draw(mimg, 0, vmax); imM.setLabel(`mean of <b>N = ${nn}</b> noisy targets`);
    rc.clear();
    rc.line(Ns.map((k) => [k, Math.sqrt(lamBar / k)]), { stroke: T.pois, width: 3, dash: "8 6" });
    rc.line(Ns.slice(0, nReady).map((k) => [k, rmse[k]]), { stroke: T.thread, width: 4 });
    rc.dots([[nn, rmse[nn]]], { fill: T.thread, r: 10, stroke: "#fff" });
    rc.note("√(λ̄/N): Poisson", { corner: "tr", fill: T.pois, size: 20 });
    rc.note("measured", { corner: "tr", line: 1, fill: T.thread, size: 20 });
    if (mode !== "Self-supervised") return;
    if (nn < nN) ro.set(`computing noisy targets… ${nReady} of ${N2N_MAX}`);
    else ro.set(`Mean of N = ${nN} targets: RMSE <b>${fmt(rmse[nN], 3)}</b> counts/px vs clean ` +
      `(one frame: ${fmt(rmse[1], 3)}) → <span class="good">${fmt(rmse[1] / rmse[nN], 1)}× lower</span>; 1/√N predicts ${fmt(Math.sqrt(nN), 1)}×. ` +
      `Dose ${fmt(lamBar, 2)} e⁻/px per frame.`);
  }
  const nSl = slider({ label: "N noisy targets", min: 1, max: N2N_MAX, value: nN, log: true, digits: 0,
    format: (v) => String(Math.round(v)), onInput: (v) => { nN = Math.round(v); drawN2N(); } });
  const n2nCtl = h("div", { class: "xp-learning-types-ctl" }, nSl);
  const n2nView = row(imA, imB, imM, imC, rc.el);
  for (const b of [imA, imB, imM, imC]) b.style.marginTop = "70px";

  // ---------- frame: mode toggle, scenes, readout, caption ----------
  const CAP = {
    "Supervised": "Supervised: labels (x, y) → learn f(x) = y. Here x = two spot intensities, y = the phase that an expert gave.",
    "Unsupervised": "Unsupervised: x only → find groups. No labels. The same idea sorts EELS spectra or diffraction patterns into phases.",
    "Self-supervised": "Self-supervised (Noise2Noise): train f(A) ≈ B with an L2 loss. The noise is zero-mean → the expected target is the clean image.",
  };
  const views = { "Supervised": [supView, supCtl], "Unsupervised": [kmView, kmCtl], "Self-supervised": [n2nView, n2nCtl] };
  const ro = readout(), cap = caption("");
  const ctlBox = h("div", {}), body = h("div", { class: "xp-learning-types-body" });
  const setMode = (m) => {
    mode = m;
    body.replaceChildren(views[m][0]); ctlBox.replaceChildren(views[m][1]); cap.innerHTML = CAP[m];
    ({ "Supervised": drawSup, "Unsupervised": drawKm, "Self-supervised": drawN2N })[m]();
  };
  const tg = toggle({ options: [
    { value: "Supervised", label: "Supervised", color: T.thread }, { value: "Unsupervised", label: "Unsupervised", color: T.thread },
    { value: "Self-supervised", label: "Self-supervised", color: T.thread }], value: mode, onChange: setMode });
  el.append(card(controls(tg, ctlBox), body, ro, cap));

  api.animate((dt) => {
    if (training) {
      for (let k = 0; k < 2 && step < 300; k++) {
        const e = evalLR();
        for (let i = 0; i < 3; i++) w[i] -= 0.4 * e.g[i];
        step++; hist.push([step, evalLR().L]);
      }
      if (step >= 300) training = false;
      if (mode === "Supervised") drawSup();
    }
    if (km.t < 1) {
      km.t = Math.min(1, km.t + dt / 0.6);
      const e = 0.5 - 0.5 * Math.cos(Math.PI * km.t);
      km.C = km.from.map((c, j) => [c[0] + e * (km.to[j][0] - c[0]), c[1] + e * (km.to[j][1] - c[1])]);
      if (km.t >= 1) kmArrive(); else if (mode === "Unsupervised") drawKm();
    }
  });
  resetSup(); resetKm(); setMode(mode);
});

// =============================================================================================
// 2. bias-variance (ported from the OJS widgets of the SS26 deck, git history)
// Carbon K-edge: background A (E/250)^−r plus an edge B·step(E − 284 eV, w = 1.5 eV)·(E/284)^−3,
// true parameters A = 100, r = 3, B = 40 counts. 60 channels from 250 to 330 eV, Poisson counts.
// Polynomial fits (Legendre basis, least squares by QR) of degree 0…15 versus the 3-parameter
// physical model (Nelder-Mead). Errors are averages over 200 pairs of (training, new) data sets.
// =============================================================================================
const BV_E = linspace(250, 330, 24), BV_Et = linspace(250, 330, 300), BV_TRUE = [100, 3, 40], BV_REPS = 200;
const toZ = (E) => (E - 290) / 40;
function edgeModel(E, [A, r, B], e0 = 284.0, w = 1.5) {
  return A * (E / 250) ** -r + B / (1 + Math.exp(-(E - e0) / w)) * (E / e0) ** -3;
}
function fitEdge(y) {
  const loss = (p) => BV_E.reduce((t, E, i) => t + (edgeModel(E, p) - y[i]) ** 2, 0);
  return nelderMead(loss, [90, 2.5, 30], [10, 0.3, 5], 2000);
}
function lstsq(A, y) {                                          // Householder QR, A is an array of rows
  const m = A.length, n = A[0].length, R = A.map((r) => r.slice()), b = y.slice();
  for (let j = 0; j < n; j++) {
    let norm = 0;
    for (let i = j; i < m; i++) norm += R[i][j] ** 2;
    norm = Math.sqrt(norm);
    const v = new Array(m).fill(0);
    for (let i = j; i < m; i++) v[i] = R[i][j];
    v[j] -= R[j][j] > 0 ? -norm : norm;
    let vv = 0;
    for (let i = j; i < m; i++) vv += v[i] ** 2;
    if (vv === 0) continue;
    for (let k = j; k < n; k++) {
      let t = 0;
      for (let i = j; i < m; i++) t += v[i] * R[i][k];
      t *= 2 / vv;
      for (let i = j; i < m; i++) R[i][k] -= t * v[i];
    }
    let t = 0;
    for (let i = j; i < m; i++) t += v[i] * b[i];
    t *= 2 / vv;
    for (let i = j; i < m; i++) b[i] -= t * v[i];
  }
  const x = new Array(n).fill(0);
  for (let j = n - 1; j >= 0; j--) {
    let t = b[j];
    for (let k = j + 1; k < n; k++) t -= R[j][k] * x[k];
    x[j] = t / R[j][j];
  }
  return x;
}
function legendre(z, deg) {
  const P = [1, z];
  for (let k = 1; k < deg; k++) P.push(((2 * k + 1) * z * P[k] - k * P[k - 1]) / (k + 1));
  return P.slice(0, deg + 1);
}
function polyfit(z, y, deg, zEval) {
  const c = lstsq(z.map((t) => legendre(t, deg)), y);
  return zEval.map((t) => legendre(t, deg).reduce((a, p, i) => a + p * c[i], 0));
}

// Error curves, computed once per page in chunks of 10 data-set pairs (≈ 0.5 s in total).
let BV_CURVES = null, BV_JOB = null;
function bvCurves(onProgress) {
  if (BV_CURVES) return Promise.resolve(BV_CURVES);
  if (BV_JOB) { BV_JOB.listeners.push(onProgress); return BV_JOB.promise; }
  const listeners = [onProgress];
  const promise = new Promise((resolve) => {
    const r = rng(7), truth = BV_E.map((E) => edgeModel(E, BV_TRUE)), z = BV_E.map(toZ);
    const tr = new Array(16).fill(0), te = new Array(16).fill(0);
    let tePhys = 0, k = 0;
    const chunk = () => {
      for (const end = Math.min(BV_REPS, k + 10); k < end; k++) {
        const y1 = truth.map(r.poisson), y2 = truth.map(r.poisson);
        for (let d = 0; d < 16; d++) {
          const f = polyfit(z, y1, d, z);
          tr[d] += mean(f.map((v, i) => (v - y1[i]) ** 2)) / BV_REPS;
          te[d] += mean(f.map((v, i) => (v - y2[i]) ** 2)) / BV_REPS;
        }
        const p = fitEdge(y1);
        tePhys += mean(BV_E.map((E, i) => (edgeModel(E, p) - y2[i]) ** 2)) / BV_REPS;
      }
      listeners.forEach((f) => f && f(k / BV_REPS));
      if (k < BV_REPS) setTimeout(chunk, 0);
      else { BV_CURVES = { tr, te, tePhys }; resolve(BV_CURVES); }
    };
    setTimeout(chunk, 0);
  });
  BV_JOB = { promise, listeners };
  return promise;
}

register("bias-variance", (el, api) => {
  let degree = 15, seed = 3, curves = null, progress = 0;
  const L = chart({ width: 960, height: 520, x: { domain: [250, 330], label: "energy loss (eV)" }, y: { domain: [0, 150], label: "counts per channel" } });
  const R = chart({ width: 720, height: 520, x: { domain: [-0.5, 15.5], label: "model complexity: polynomial degree", ticks: [0, 3, 6, 9, 12, 15] },
    y: { domain: [15, 260], type: "log", label: "mean squared error (counts²)", ticks: [20, 50, 100, 200] }, margin: { l: 90 } });
  const ro = readout();
  function drawLeft() {
    const r = rng(seed), y = BV_E.map((E) => r.poisson(edgeModel(E, BV_TRUE)));
    const poly = polyfit(BV_E.map(toZ), y, degree, BV_Et.map(toZ)), phys = fitEdge(y);
    L.clear();
    L.line(BV_Et.map((E) => [E, edgeModel(E, BV_TRUE)]), { stroke: "rgba(226,232,240,0.28)", width: 10 });
    L.dots(BV_E.map((E, i) => [E, y[i]]), { fill: T.data, r: 5 });
    L.line(BV_Et.map((E) => [E, edgeModel(E, phys)]), { stroke: T.truth, width: 3.5, dash: "10 7" });
    L.line(BV_Et.map((E, i) => [E, clamp(poly[i], -50, 250)]), { stroke: T.pois, width: 4 });
    L.note(`polynomial, degree ${degree}`, { corner: "tr", fill: T.pois, weight: 700, size: 22 });
    L.note("physical edge model (3 parameters)", { corner: "tr", line: 1, fill: T.truth });
    L.text(284, 6, "C K-edge, 284 eV", { anchor: "middle", fill: T.muted, size: 18 });
  }
  function drawRight() {
    R.clear();
    R.ruleX(degree, { stroke: T.thread, width: 3, dash: null });
    if (!curves) {
      R.note(`computing… ${fmt(100 * progress, 0)} %`, { corner: "tl", fill: T.warn, size: 24 });
      ro.set(`Degree ${degree}. Computing the average errors over ${BV_REPS} synthetic data sets…`);
      return;
    }
    const D = Array.from({ length: 16 }, (_, d) => d), best = curves.te.indexOf(Math.min(...curves.te));
    R.ruleY(curves.tePhys, { stroke: T.truth, width: 3, dash: "10 7" });
    R.line(D.map((d) => [d, curves.tr[d]]), { stroke: T.gauss, width: 3.5 });
    R.dots(D.map((d) => [d, curves.tr[d]]), { fill: T.gauss, r: 5 });
    R.line(D.map((d) => [d, curves.te[d]]), { stroke: T.pois, width: 3.5 });
    R.dots(D.map((d) => [d, curves.te[d]]), { fill: T.pois, r: 5 });
    R.dots([[degree, curves.tr[degree]]], { fill: T.gauss, r: 11, stroke: "#fff", strokeWidth: 3, top: true });
    R.dots([[degree, curves.te[degree]]], { fill: T.pois, r: 11, stroke: "#fff", strokeWidth: 3, top: true });
    R.note("error on new data", { corner: "tr", fill: T.pois });
    R.note("error on training data", { corner: "tr", line: 1, fill: T.gauss });
    R.note("physical model, new data", { corner: "tr", line: 2, fill: T.truth });
    const verdict = degree < best ? `<span class="warn">too simple: high bias</span>` : degree > best + 2
      ? `<span class="bad">too flexible: high variance</span>` : `<span class="good">near the best polynomial</span>`;
    ro.set(`Degree ${degree} → ${verdict}. Error on new data <b>${fmt(curves.te[degree], 0)}</b> counts², on training data ${fmt(curves.tr[degree], 0)} ` +
      `(best polynomial: degree ${best}, ${fmt(curves.te[best], 0)}; physical model: <b>${fmt(curves.tePhys, 0)}</b>). Averages over ${BV_REPS} data sets.`);
  }
  const sl = slider({ label: "polynomial degree", min: 0, max: 15, value: degree, step: 1, digits: 0,
    onInput: (v) => { degree = Math.round(v); drawLeft(); drawRight(); } });
  const nb = button("new data set", () => { seed += 1; drawLeft(); });
  el.append(card(controls(sl, nb, legend([[T.data, "data (Poisson counts)"], [T.pois, "polynomial fit"], [T.truth, "physical model fit", "dash"], ["rgba(226,232,240,0.4)", "true curve"]])),
    row(L.el, R.el), ro));
  drawLeft(); drawRight();
  bvCurves((p) => { progress = p; if (!curves) drawRight(); }).then((c) => { curves = c; drawRight(); });
});

// =============================================================================================
// 3. data-zoo
// Sizes: bytes = product of the shape × 2 (uint16) or 4 (float32). 1 MB = 10⁶ bytes.
// HAADF Z-contrast: I ∝ Z^n with n ≈ 1.6…2.0 (Rutherford-like scattering, screened: n < 2).
// Tilt series: N tilts spread evenly over ±70°, so the step is 140°/(N − 1).
// =============================================================================================
const ZOO_RAM = 16e9;
function fmtBytes(b) {
  return b >= 1e12 ? `${fmt(b / 1e12, 2)} TB` : b >= 1e9 ? `${fmt(b / 1e9, 1)} GB` : b >= 1e6 ? `${fmt(b / 1e6, 1)} MB` : `${fmt(b / 1e3, 0)} kB`;
}

// Oblique projection: x to the right, y down, z into the page (up and to the right).
const OBL = [0.62, 0.42];
const proj = (ox, oy) => (x, y, z = 0) => [ox + x + z * OBL[0], oy + y - z * OBL[1]];
const poly = (P, pts, attrs) => s("polygon", { points: pts.map((p) => P(...p).map((v) => v.toFixed(1)).join(",")).join(" "), ...attrs });
const lineP = (P, a, b, attrs) => { const [x1, y1] = P(...a), [x2, y2] = P(...b); return s("line", { x1, y1, x2, y2, ...attrs }); };
const label = (x, y, str, attrs = {}) => s("text", { x, y, fill: T.ink, "font-size": 20, ...attrs }, str);
function cuboid(P, w, hh, d, fill = "rgba(96,165,250,0.18)") {
  const st = { stroke: T.muted, "stroke-width": 1.5 };
  return [poly(P, [[0, 0, 0], [w, 0, 0], [w, 0, d], [0, 0, d]], { fill, ...st }),
    poly(P, [[w, 0, 0], [w, hh, 0], [w, hh, d], [w, 0, d]], { fill, ...st }),
    poly(P, [[0, 0, 0], [w, 0, 0], [w, hh, 0], [0, hh, 0]], { fill, ...st })];
}
function arrow(x1, y1, x2, y2, txt, { color = T.ink, dx = 0, dy = 0, anchor = "middle" } = {}) {
  const a = Math.atan2(y2 - y1, x2 - x1), hx = (t) => x2 - 14 * Math.cos(a + t), hy = (t) => y2 - 14 * Math.sin(a + t);
  return s("g", {}, s("line", { x1, y1, x2, y2, stroke: color, "stroke-width": 2.5 }),
    s("path", { d: `M${x2},${y2}L${hx(0.4)},${hy(0.4)}L${hx(-0.4)},${hy(-0.4)}Z`, fill: color }),
    txt ? label((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, txt, { fill: color, "text-anchor": anchor }) : null);
}
// A small diffraction pattern: central disc and 6 Bragg discs, rotated by `rot`.
function dp(cx, cy, R, rot, glow = 1) {
  const g = s("g", {}, s("circle", { cx, cy, r: R * 0.32, fill: "#f8fafc", opacity: 0.95 }));
  for (let k = 0; k < 6; k++) {
    const a = rot + k * Math.PI / 3;
    g.append(s("circle", { cx: cx + 0.7 * R * Math.cos(a), cy: cy + 0.7 * R * Math.sin(a), r: R * 0.13, fill: T.pois, opacity: 0.45 + 0.5 * glow * ((k % 2) ? 0.6 : 1) }));
  }
  return g;
}
const latticeVal = (x, y) => 0.12 + 0.88 * Math.exp(-((x % 1 - 0.5) ** 2 + (y % 1 - 0.5) ** 2) / 0.035);

const ZOO = {
  "HAADF image": {
    shape: (v) => [["ny", v.scan], ["nx", v.scan]],
    sliders: ["scan"],
    facts: ["each pixel counts electrons that scattered to high angles (annular detector), one number per probe position",
      "Poisson shot noise, plus scan distortion and drift", "U-Net for denoising, atom finding and segmentation"],
  },
  "4D-STEM": {
    shape: (v) => [["ny", v.scan], ["nx", v.scan], ["ky", v.det], ["kx", v.det]],
    sliders: ["scan", "det"],
    facts: ["each detector pixel counts electrons at one scattering angle k, for every probe position",
      "Poisson, often less than 1 e⁻ per detector pixel", "CNN per diffraction pattern (strain, orientation); ptychography as a physics model"],
  },
  "EELS/EDS spectrum image": {
    shape: (v) => [["ny", v.scan], ["nx", v.scan], ["E", v.energy]],
    sliders: ["scan", "energy"],
    facts: ["each channel counts electrons with energy loss E (EELS) or X-ray photons of energy E (EDS)",
      "Poisson; EDS is very sparse (most channels hold 0 or 1 counts)", "PCA / NMF to unmix phases; 1-D CNN per spectrum"],
  },
  "tilt series": {
    shape: (v) => [["θ", v.tilts], ["ny", v.scan], ["nx", v.scan]],
    sliders: ["tilts", "scan"],
    facts: ["each pixel is a projection: the signal summed along the beam path at tilt θ",
      "Poisson at a low dose per tilt; no data beyond ±70° (missing wedge)", "3-D U-Net for denoising and reconstruction (e.g. cryo-CARE)"],
  },
};

register("data-zoo", (el, api) => {
  css("xp-data-zoo-css", `
    .xp-data-zoo-left { width: 820px; height: 640px; }
    .xp-data-zoo-right { width: 860px; gap: 16px; }
    .xp-data-zoo-shape { font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 32px; color: #fff; }
    .xp-data-zoo-shape .ax { color: var(--thread); }
    .xp-data-zoo-facts { display: grid; grid-template-columns: 9.6em 1fr; gap: 6px 14px; font-size: 20px; }
    .xp-data-zoo-facts .k { color: var(--muted); }
    .xp-data-zoo-sl { display: flex; flex-direction: column; gap: 12px; min-height: 104px; }`);
  const v = { scan: 256, det: 256, energy: 2048, tilts: 71, dtype: "uint16", n: 1.7 };
  let tab = "4D-STEM";
  const bytesPer = () => (v.dtype === "uint16" ? 2 : 4);
  const sk = h("div", {}), zBox = h("div", {}), left = h("div", { class: "xp-data-zoo-left" }, sk, zBox), shapeEl = h("div", { class: "xp-data-zoo-shape" });
  const ro = readout(), facts = h("div", { class: "xp-data-zoo-facts" }), slBox = h("div", { class: "xp-data-zoo-sl" });
  const fmtInt = (x) => String(Math.round(x));
  const SL = {
    scan: slider({ label: "scan size (px per side)", min: 64, max: 1024, value: v.scan, log: true, step: 16, format: fmtInt, onInput: (x) => { v.scan = Math.round(x); update(); } }),
    det: slider({ label: "detector (px per side)", min: 64, max: 512, value: v.det, log: true, step: 16, format: fmtInt, onInput: (x) => { v.det = Math.round(x); update(); } }),
    energy: slider({ label: "energy channels", min: 256, max: 4096, value: v.energy, log: true, step: 64, format: fmtInt, onInput: (x) => { v.energy = Math.round(x); update(); } }),
    tilts: slider({ label: "tilts over ±70°", min: 31, max: 181, value: v.tilts, step: 2, format: fmtInt, onInput: (x) => { v.tilts = Math.round(x); update(); } }),
  };
  const dt = toggle({ label: "dtype", options: ["uint16", "float32"], value: v.dtype, onChange: (x) => { v.dtype = x; update(); } });

  // Z-contrast bar chart (HAADF tab only).
  const ELEM = [["C", 6], ["Si", 14], ["Fe", 26], ["Au", 79]];
  const zc = chart({ width: 820, height: 280, x: { domain: [-0.6, 3.6], ticks: [0, 1, 2, 3], tickFormat: (i) => ELEM[i] ? `${ELEM[i][0]} (Z = ${ELEM[i][1]})` : "", grid: false },
    y: { domain: [1, 2000], type: "log", label: "I / I(C)" }, margin: { b: 40, t: 14 } });
  const nSl = slider({ label: "Z-contrast exponent n", min: 1.6, max: 2.0, value: v.n, digits: 2, onInput: (x) => { v.n = x; update(); } });
  function drawZ() {
    zc.clear();
    ELEM.forEach(([el2, Z], i) => {
      const rel = (Z / 6) ** v.n;
      zc.bars([[i - 0.32, i + 0.32, rel]], { fill: i === 3 ? T.warn : T.data, opacity: 0.85 });
      zc.text(i, rel, `×${fmt(rel, rel < 10 ? 1 : 0)}`, { anchor: "middle", dy: -10, size: 20 });
    });
    zc.note(`HAADF intensity ∝ Zⁿ, n = ${fmt(v.n, 2)}`, { corner: "tl", fill: T.muted });
  }

  // Sketches of each tensor (SVG, 820 px wide).
  function sketch() {
    const H = tab === "HAADF image" ? 336 : 640;
    const g = s("svg", { width: 820, height: H, viewBox: `0 0 820 ${H}`, class: "xp-chart" });
    if (tab === "HAADF image") {
      const N = 32, S = 256, x0 = 70, y0 = 24, c = S / N;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const val = latticeVal(j / 5.33, i / 5.33), gv = Math.round(255 * val);
        g.append(s("rect", { x: x0 + j * c, y: y0 + i * c, width: c + 0.5, height: c + 0.5, fill: `rgb(${gv},${gv},${gv})` }));
      }
      g.append(s("rect", { x: x0, y: y0, width: S, height: S, fill: "none", stroke: T.muted, "stroke-width": 1.5 }),
        s("rect", { x: x0 + 18 * c, y: y0 + 8 * c, width: 2 * c, height: 2 * c, fill: "none", stroke: T.thread, "stroke-width": 3 }),
        arrow(x0, y0 + S + 22, x0 + S, y0 + S + 22, `x: nx = ${v.scan}`, { dy: 26 }),
        arrow(x0 - 22, y0, x0 - 22, y0 + S, "", {}),
        label(x0 - 34, y0 + S / 2, `y: ny = ${v.scan}`, { transform: `rotate(-90 ${x0 - 34} ${y0 + S / 2})`, "text-anchor": "middle" }));
      // probe and annular detector
      const px = 600, py = 30;
      g.append(s("path", { d: `M${px - 40},${py}L${px},${py + 90}L${px + 40},${py}Z`, fill: "rgba(244,63,94,0.25)", stroke: T.thread }),
        s("line", { x1: px - 70, x2: px + 70, y1: py + 90, y2: py + 90, stroke: T.muted, "stroke-width": 4 }),
        s("path", { d: `M${px},${py + 90}L${px - 150},${py + 230}M${px},${py + 90}L${px + 150},${py + 230}`, stroke: T.pois, "stroke-width": 2, "stroke-dasharray": "6 5" }),
        s("ellipse", { cx: px, cy: py + 235, rx: 170, ry: 30, fill: "none", stroke: T.pois, "stroke-width": 16, opacity: 0.8 }),
        s("ellipse", { cx: px, cy: py + 235, rx: 70, ry: 12, fill: T.bg, stroke: T.faint }),
        label(px + 50, py + 30, "probe", { fill: T.thread }), label(px + 85, py + 105, "specimen", { fill: T.muted }),
        label(px, py + 296, "annular detector: 1 number per probe position", { "text-anchor": "middle", fill: T.pois, "font-size": 19 }),
        s("line", { x1: x0 + 20 * c, y1: y0 + 9 * c, x2: px - 20, y2: py + 90, stroke: T.thread, "stroke-dasharray": "4 5" }));
    } else if (tab === "4D-STEM") {
      const N = 6, c = 64, x0 = 40, y0 = 120, hi = [2, 4];
      g.append(label(x0, y0 - 46, `scan grid (ny, nx) = (${v.scan}, ${v.scan})`, { fill: T.ink }),
        label(x0, y0 - 18, "one diffraction pattern per probe position", { fill: T.muted, "font-size": 18 }));
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const grain = i + j > 5 ? 0.52 : 0;
        g.append(s("rect", { x: x0 + j * c, y: y0 + i * c, width: c - 3, height: c - 3, rx: 4, fill: "#05070d", stroke: T.faint }),
          dp(x0 + j * c + c / 2 - 1.5, y0 + i * c + c / 2 - 1.5, c * 0.42, grain, 0.6 + 0.4 * ((i * 7 + j * 3) % 5) / 4));
      }
      const hx = x0 + hi[1] * c, hy = y0 + hi[0] * c, bx = 500, by = 150, B = 250;
      g.append(s("rect", { x: hx - 2, y: hy - 2, width: c + 1, height: c + 1, rx: 5, fill: "none", stroke: T.thread, "stroke-width": 4 }),
        s("path", { d: `M${hx + c},${hy}L${bx},${by}M${hx + c},${hy + c - 3}L${bx},${by + B}`, stroke: T.thread, "stroke-dasharray": "5 5", "stroke-width": 2 }),
        s("rect", { x: bx, y: by, width: B, height: B, rx: 10, fill: "#05070d", stroke: T.thread, "stroke-width": 3 }),
        dp(bx + B / 2, by + B / 2, B * 0.42, 0, 1),
        arrow(bx, by + B + 26, bx + B, by + B + 26, `kx: ${v.det} px`, { dy: 28 }),
        arrow(bx + B + 24, by, bx + B + 24, by + B, ""),
        label(bx + B + 40, by + B / 2, `ky: ${v.det} px`, { transform: `rotate(90 ${bx + B + 40} ${by + B / 2})`, "text-anchor": "middle" }),
        label(bx, by - 16, "detector frame (ky, kx)", { fill: T.ink }),
        arrow(x0, y0 + N * c + 18, x0 + N * c - 3, y0 + N * c + 18, `x: nx = ${v.scan}`, { dy: 28 }),
        label(x0, y0 + N * c + 92, "grain boundary: the pattern rotates", { fill: T.muted, "font-size": 18 }));
    } else if (tab === "EELS/EDS spectrum image") {
      const W = 280, D = 260, x0 = 60, y0 = 300, P = proj(x0, y0), N = 8, c = W / N, col = 6;
      g.append(...cuboid(P, W, W, D));
      for (let k = 1; k < N; k++) g.append(lineP(P, [k * c, 0, 0], [k * c, W, 0], { stroke: T.faint }), lineP(P, [0, k * c, 0], [W, k * c, 0], { stroke: T.faint }),
        lineP(P, [k * c, 0, 0], [k * c, 0, D], { stroke: T.faint }));
      g.append(poly(P, [[col * c, 0, 0], [(col + 1) * c, 0, 0], [(col + 1) * c, 0, D], [col * c, 0, D]], { fill: T.thread, opacity: 0.85 }),
        poly(P, [[col * c, 0, 0], [(col + 1) * c, 0, 0], [(col + 1) * c, c, 0], [col * c, c, 0]], { fill: T.thread, opacity: 0.85 }));
      const [ax, ay] = P(W, W, 0), [bx2, by2] = P(W, W, D), [lx, ly] = P(0, W, 0);
      g.append(arrow(lx, ly + 24, ax, ay + 24, `x: nx = ${v.scan}`, { dy: 28 }),
        arrow(ax + 18, ay + 10, bx2 + 18, by2 + 10, `E: ${v.energy} channels`, { dx: 22, dy: 26, anchor: "start" }),
        arrow(x0 - 20, y0, x0 - 20, y0 + W, ""),
        label(x0 - 30, y0 + W / 2, `y: ny = ${v.scan}`, { transform: `rotate(-90 ${x0 - 30} ${y0 + W / 2})`, "text-anchor": "middle" }));
      // the spectrum of the red pixel, pulled out
      const sx0 = 520, sy0 = 40, SW = 290, SH = 170, Es = linspace(0, 1, 120);
      const spec = (e) => 0.12 + 0.75 * Math.exp(-6 * e) + (e > 0.35 ? 0.35 * Math.exp(-5 * (e - 0.35)) : 0) + (e > 0.7 ? 0.18 * Math.exp(-6 * (e - 0.7)) : 0);
      const [cx, cy] = P((col + 0.5) * c, 0, D);
      g.append(s("path", { d: `M${cx},${cy}L${sx0},${sy0 + SH}`, stroke: T.thread, "stroke-dasharray": "5 5", "stroke-width": 2 }),
        s("rect", { x: sx0, y: sy0, width: SW, height: SH, rx: 8, fill: "rgba(5,7,13,0.7)", stroke: T.faint }),
        s("path", { d: Es.map((e, i) => `${i ? "L" : "M"}${(sx0 + 10 + e * (SW - 20)).toFixed(1)},${(sy0 + SH - 12 - spec(e) * (SH - 30)).toFixed(1)}`).join(""), fill: "none", stroke: T.thread, "stroke-width": 3 }),
        label(sx0 + SW / 2, sy0 + SH + 26, "one spectrum: counts vs E", { "text-anchor": "middle", fill: T.ink }),
        label(sx0 + 0.35 * SW + 14, sy0 + 62, "edge", { fill: T.muted, "font-size": 18 }),
        label(sx0 + 0.7 * SW, sy0 + 98, "edge", { fill: T.muted, "font-size": 18 }));
    } else {
      // tilt geometry: beam directions over ±70°, missing wedge beyond
      const cx = 200, cy = 330, R = 170, rad = (d) => d * Math.PI / 180;
      const wedge = (a0, a1) => { const p = (a) => [cx + 1.2 * R * Math.sin(rad(a)), cy - 1.2 * R * Math.cos(rad(a))];
        return `M${cx},${cy}L${p(a0).join(",")}A${1.2 * R},${1.2 * R} 0 0 1 ${p(a1).join(",")}Z`; };
      g.append(s("path", { d: wedge(70, 110), fill: T.warn, opacity: 0.16 }), s("path", { d: wedge(-110, -70), fill: T.warn, opacity: 0.16 }),
        label(cx, cy + 1.2 * R + 40, "missing wedge: |θ| > 70°", { "text-anchor": "middle", fill: T.warn }));
      for (const a of [-70, -35, 0, 35, 70]) {
        const ux = Math.sin(rad(a)), uy = -Math.cos(rad(a)), tx = -uy, ty = ux;
        const ex = cx - ux * R, ey = cy - uy * R;                    // detector on the far side
        g.append(s("line", { x1: cx + ux * R, y1: cy + uy * R, x2: cx - ux * (R - 16), y2: cy - uy * (R - 16), stroke: a === 0 ? T.thread : T.faint, "stroke-width": 2, "stroke-dasharray": "6 5" }),
          arrow(cx + ux * R, cy + uy * R, cx + ux * (R - 40), cy + uy * (R - 40), "", { color: a === 0 ? T.thread : T.muted }),
          s("line", { x1: ex - tx * 30, y1: ey - ty * 30, x2: ex + tx * 30, y2: ey + ty * 30, stroke: a === 0 ? T.thread : T.muted, "stroke-width": 6, "stroke-linecap": "round" }),
          label(cx + ux * (R + 26), cy + uy * (R + 26) + 6, `${a > 0 ? "+" : ""}${a}°`, { "text-anchor": "middle", fill: T.muted, "font-size": 18 }));
      }
      g.append(s("ellipse", { cx, cy, rx: 48, ry: 30, fill: T.truth, opacity: 0.55 }), label(cx, cy + 6, "specimen", { "text-anchor": "middle", "font-size": 17, fill: "#0b0f19" }),
        label(cx, 70, "beam", { "text-anchor": "middle", fill: T.thread }), label(cx, cy + R + 34, "detector", { "text-anchor": "middle", fill: T.thread, "font-size": 18 }));
      // the stack of projections
      const K = 9, x0 = 500, y0 = 320, S = 180, dz = 13;
      for (let k = K - 1; k >= 0; k--) {
        const X = x0 + k * dz * 1.2, Y = y0 - k * dz;
        g.append(s("rect", { x: X, y: Y, width: S, height: S, rx: 4, fill: k ? "rgba(52,211,153,0.10)" : "rgba(52,211,153,0.25)", stroke: k ? T.faint : T.ink, "stroke-width": k ? 1.2 : 2.5 }));
      }
      g.append(s("ellipse", { cx: x0 + S / 2, cy: y0 + S / 2, rx: 60, ry: 34, fill: T.truth, opacity: 0.55 }),
        label(x0, y0 + S + 30, "θ = −70° (first projection)", { fill: T.ink, "font-size": 19 }),
        arrow(x0 - 14, y0, x0 - 14 + (K - 1) * dz * 1.2, y0 - (K - 1) * dz, ""),
        label(x0 - 14, y0 - (K - 1) * dz - 50, `θ: ${v.tilts} tilts, step ${fmt(140 / (v.tilts - 1), 2)}°`, { fill: T.ink }),
        label(x0 - 14, y0 - (K - 1) * dz - 22, "last projection: θ = +70°", { fill: T.muted, "font-size": 18 }),
        arrow(x0, y0 + S + 52, x0 + S, y0 + S + 52, `x: nx = ${v.scan}`, { dy: 28 }));
    }
    return g;
  }

  function update() {
    const Z = ZOO[tab], sh = Z.shape(v), n = sh.reduce((p, [, k]) => p * k, 1), bytes = n * bytesPer();
    shapeEl.innerHTML = `(${sh.map(([a]) => `<span class="ax">${a}</span>`).join(", ")}) = (${sh.map(([, k]) => k).join(", ")})`;
    sk.replaceChildren(sketch());
    if (tab === "HAADF image") drawZ();
    const fit = bytes <= ZOO_RAM ? `<span class="good">one scan fits in 16 GB of RAM</span>`
      : `<span class="bad">one scan is ${fmt(bytes / ZOO_RAM, bytes > 1.6e11 ? 0 : 1)}× larger than 16 GB of RAM</span>`;
    ro.set(`${fmt(n / 1e6, n < 1e7 ? 2 : 0)} M values × ${bytesPer()} bytes = <b>${fmtBytes(bytes)}</b> per scan.<br>` +
      `A 10-scan session = <b>${fmtBytes(10 * bytes)}</b> · ${fit}.`);
    if (tab === "HAADF image") ro.innerHTML += ` Au is <b>${fmt((79 / 6) ** v.n, 0)}×</b> brighter than C.`;
  }
  function setTab(t) {
    tab = t;
    const Z = ZOO[t];
    slBox.replaceChildren(...Z.sliders.map((k) => SL[k]));
    zBox.replaceChildren(...(t === "HAADF image" ? [zc.el, nSl] : []));
    facts.replaceChildren(...["one pixel", "noise model", "typical ML"].flatMap((k, i) => [h("span", { class: "k" }, k), h("span", {}, Z.facts[i])]));
    update();
  }
  const tabs = toggle({ options: Object.keys(ZOO), value: tab, onChange: setTab });
  const right = col(h("div", { class: "xp-slider-label" }, "shape of one data set"), shapeEl, slBox, dt, ro, facts);
  right.classList.add("xp-data-zoo-right");
  el.append(card(controls(tabs), row(left, right)));
  setTab(tab);
});

// =============================================================================================
// 4. box-spectrum
// White box: all physics, no training data. Black box: all learned, physics only in the
// architecture. Scores (0…1) are a teaching scale, not measurements. The Bragg and Beer–Lambert
// readouts are computed: λ(200 kV) = h / √(2 m₀ e U (1 + e U / 2 m₀ c²)) = 2.51 pm.
// =============================================================================================
const BOX_LAMBDA = (() => {
  const hP = 6.62607015e-34, m0 = 9.1093837e-31, e = 1.602176634e-19, c = 299792458, U = 200e3;
  return hP / Math.sqrt(2 * m0 * e * U * (1 + e * U / (2 * m0 * c * c)));
})();
const BOX = [
  { name: "Bragg's law", pos: 0.03, row: 0, anchor: "left", phys: 1.0, interp: 1.0, data: 0.03, flex: 0.05, params: "0",
    text: () => { const d = 2.08e-10, th = Math.asin(BOX_LAMBDA / (2 * d));
      return `Bragg's law: λ = ${fmt(BOX_LAMBDA * 1e12, 2)} pm at 200 kV and Cu (111), d = 2.08 Å → Bragg angle θ = <b>${fmt(th * 1e3, 2)} mrad</b>. No training data, but it knows only elastic Bragg scattering.`; } },
  { name: "Beer–Lambert law", pos: 0.13, row: 1, phys: 0.95, interp: 0.95, data: 0.08, flex: 0.1, params: "1 (inelastic mean free path λᵢₙ)",
    text: () => `Beer–Lambert: I/I₀ = exp(−t/λᵢₙ). At t/λᵢₙ = 0.8, <b>${fmt(100 * Math.exp(-0.8), 0)} %</b> of the electrons pass without inelastic scattering. One parameter per material.` },
  { name: "EELS edge fit with physical cross-sections", pos: 0.25, row: 0, phys: 0.85, interp: 0.9, data: 0.15, flex: 0.2, params: "3 (A, r, N)",
    text: () => `EELS edge fit: background A·E⁻ʳ plus N·σ_K(E) from atomic physics. <b>3 parameters</b> from one spectrum, each with an error bar.` },
  { name: "Gaussian process with physics kernel", pos: 0.41, row: 1, phys: 0.65, interp: 0.7, data: 0.35, flex: 0.45, params: "≈ 5 hyperparameters + the data",
    text: () => `Gaussian process: the kernel builds in smoothness and symmetry. It learns the rest from ≈ 50 measurements and gives <b>an error bar for each prediction</b>.` },
  { name: "learned hardening law (crystal plasticity)", pos: 0.52, row: 0, phys: 0.55, interp: 0.55, data: 0.45, flex: 0.55, params: "≈ 10² weights",
    text: () => `Crystal plasticity keeps the slip systems and the stress balance. A small network learns <b>only the hardening law</b> τ_c(γ) from the data.` },
  { name: "PINN (physics-informed NN)", pos: 0.64, row: 1, phys: 0.4, interp: 0.4, data: 0.55, flex: 0.7, params: "≈ 10⁴ weights",
    text: () => `PINN: the loss adds the PDE residual to the data misfit. Physics is <b>a soft constraint</b>: the network can still break it where data are sparse.` },
  { name: "CNN phase classifier", pos: 0.83, row: 1, phys: 0.12, interp: 0.15, data: 0.8, flex: 0.85, params: "≈ 10⁶ weights",
    text: () => `CNN phase classifier: 97 % accuracy, but <b>it cannot say</b> if it learned lattice geometry or a detector artefact. Needs ≈ 10⁴ labelled patterns.` },
  { name: "foundation model", pos: 0.97, row: 0, anchor: "right", phys: 0.03, interp: 0.05, data: 1.0, flex: 1.0, params: "≈ 10⁹ weights",
    text: () => `Foundation model: trained on millions of images, it adapts to many tasks. <b>Nobody can list what it learned</b>, so check it on data with a known answer.` },
];

register("box-spectrum", (el, api) => {
  css("xp-box-spectrum-css", `
    .xp-box-spectrum-track { position: relative; width: 1700px; height: 270px; }
    .xp-box-spectrum-bar { position: absolute; left: 0; right: 0; top: 112px; height: 46px; border-radius: 12px;
      background: linear-gradient(90deg, #f8fafc 0%, #94a3b8 50%, #0b0f19 100%); border: 1px solid rgba(255,255,255,0.25); }
    .xp-box-spectrum-track .xp-chip { position: absolute; font-size: 20px; white-space: nowrap; }
    .xp-box-spectrum-stem { position: absolute; width: 2px; background: rgba(148,163,184,0.6); }
    .xp-box-spectrum-mark { position: absolute; top: 96px; width: 0; height: 0; }
    .xp-box-spectrum-mark::before { content: ""; position: absolute; left: -7px; top: 0; width: 14px; height: 78px; border-radius: 7px;
      background: var(--thread); box-shadow: 0 0 0 3px #0b0f19, 0 0 18px rgba(244,63,94,0.8); }
    .xp-box-spectrum-end { position: absolute; top: 120px; font-size: 20px; font-weight: 600; pointer-events: none; }
    .xp-box-spectrum-meter { display: grid; grid-template-columns: 9.5em 360px 5em; align-items: center; gap: 14px; font-size: 21px; }
    .xp-box-spectrum-meter .bg { height: 22px; border-radius: 11px; background: rgba(148,163,184,0.18); overflow: hidden; }
    .xp-box-spectrum-meter .fg { height: 100%; border-radius: 11px; background: var(--thread); transition: width 0.4s; }
    .xp-box-spectrum-stack { display: flex; width: 760px; height: 64px; border-radius: 14px; overflow: hidden; font-size: 21px; font-weight: 700; }
    .xp-box-spectrum-stack > div { display: flex; align-items: center; justify-content: center; transition: width 0.4s; white-space: nowrap; overflow: hidden; color: #0b0f19; }`);
  const W = 1700;
  let sel = 6, pos = BOX[sel].pos;
  const track = h("div", { class: "xp-box-spectrum-track" }, h("div", { class: "xp-box-spectrum-bar" }),
    h("div", { class: "xp-box-spectrum-end", style: { left: "110px", color: "#0b0f19" } }, "white box: physics only"),
    h("div", { class: "xp-box-spectrum-end", style: { left: `${W / 2}px`, transform: "translateX(-50%)", color: "#0b0f19" } }, "grey box"),
    h("div", { class: "xp-box-spectrum-end", style: { right: "110px", color: "#f8fafc" } }, "black box: learned from data"));
  const chips = BOX.map((b, i) => {
    const X = b.pos * W, top = b.row ? 214 : 8;
    track.append(h("div", { class: "xp-box-spectrum-stem", style: { left: `${X - 1}px`, top: b.row ? "158px" : "50px", height: b.row ? "56px" : "62px" } }));
    const c = h("div", { class: "xp-chip", style: { left: `${X}px`, top: `${top}px`,
      transform: b.anchor === "left" ? "translateX(-14px)" : b.anchor === "right" ? "translateX(calc(-100% + 14px))" : "translateX(-50%)" } }, b.name);
    c.addEventListener("click", () => choose(i));
    track.append(c);
    return c;
  });
  const mark = h("div", { class: "xp-box-spectrum-mark" });
  track.append(mark);

  const stPhys = h("div", { style: { background: T.truth } }), stData = h("div", { style: { background: T.data } });
  const meter = (name) => {
    const fg = h("div", { class: "fg" }), val = h("span", {});
    const elm = h("div", { class: "xp-box-spectrum-meter" }, h("span", { class: "xp-slider-label" }, name), h("div", { class: "bg" }, fg), val);
    elm.set = (x) => { fg.style.width = `${100 * x}%`; val.textContent = x < 0.34 ? "low" : x < 0.67 ? "medium" : "high"; };
    return elm;
  };
  const mI = meter("interpretability"), mD = meter("data needed"), mF = meter("flexibility");
  const params = h("div", { class: "xp-box-spectrum-meter" }, h("span", { class: "xp-slider-label" }, "free parameters"), h("b", {}));
  const ro = readout("", "big");
  const leftCol = col(h("div", { class: "xp-slider-label" }, "physics built in vs learned from data"), h("div", { class: "xp-box-spectrum-stack" }, stPhys, stData),
    legend([[T.truth, "physics built in"], [T.data, "learned from data"]]));
  el.append(card(track, row(leftCol, col(mI, mD, mF, params)), ro, caption("Click an example. Further right: more flexible, but more data needed and less to read out of the model.")));

  function choose(i) {
    sel = i;
    const b = BOX[i];
    chips.forEach((c, k) => c.classList.toggle("on", k === i));
    stPhys.style.width = `${100 * b.phys}%`; stData.style.width = `${100 * (1 - b.phys)}%`;
    stPhys.textContent = b.phys >= 0.15 ? `physics ${fmt(100 * b.phys, 0)} %` : "";
    stData.textContent = 1 - b.phys >= 0.15 ? `learned ${fmt(100 * (1 - b.phys), 0)} %` : "";
    mI.set(b.interp); mD.set(b.data); mF.set(b.flex);
    params.lastChild.textContent = b.params;
    ro.set(b.text());
  }
  const place = () => { mark.style.left = `${pos * W}px`; };
  api.animate((dt) => {
    const target = BOX[sel].pos;
    if (Math.abs(target - pos) < 1e-4) return;
    pos += (target - pos) * Math.min(1, dt * 8);
    place();
  });
  choose(sel); place();
});
