// Generalise group (Act 4), on real EELS pre-edge spectra (CeVOx region, 30 pixels):
//   poly-bg   : polynomial background of degree d in log-log space, fitted in 470-496 eV and checked
//               in 497-508 eV (the channels next to the edge, where the background is extrapolated).
//   bias-var  : the same fit on 30 real pixels: the spread of the fits is the variance, the offset of
//               their mean from the 30-pixel mean spectrum is the bias.
import { register, T, fmt, h, s, row, col, card, controls, readout, slider, toggle, legend, chart } from "./core.js";
import { loadData } from "./data.js";
import { lstsq } from "./la.js";

if (!document.getElementById("xp-gen-style")) {
  document.head.append(h("style", { id: "xp-gen-style" }, `
.xp-gen-side { width: 640px; display: flex; flex-direction: column; gap: 14px; }
.xp-gen-side .xp-readout { margin-top: 0; line-height: 1.5; }
.xp-src { font-size: 17px; color: #64748b; margin-top: 4px; }
`));
}
const src = (html) => h("div", { class: "xp-src", html });
const SRC_CEVO3 = "Data: CeVO₃ on Al₂O₃, STEM-Bank-26 (Hollyer &amp; Stach 2026), CC BY 4.0.";
const waiting = (el) => { const w = h("div", { class: "xp-caption" }, "loading data …"); el.append(w); return w; };
const TRAIN = [470, 496], VAL = [496.5, 508], DMAX = 8;

function setup(d) {
  const [ny, nx, ne] = d.cube.shape, E = Array.from(d.E);
  const px = [];
  for (let y = 26; y < 32; y++) for (let x = 33; x < 38; x++) px.push(Float64Array.from({ length: ne }, (_, k) => Math.max(1e-3, d.cube[(y * nx + x) * ne + k] * d.scale)));
  const mean = Float64Array.from({ length: ne }, (_, k) => px.reduce((a, p) => a + p[k], 0) / px.length);
  const lx = E.map(Math.log), tr = E.map((e) => e >= TRAIN[0] && e <= TRAIN[1]), va = E.map((e) => e >= VAL[0] && e <= VAL[1]);
  const xs = lx.filter((_, i) => tr[i]), m = xs.reduce((a, b) => a + b, 0) / xs.length, sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
  const u = lx.map((v) => (v - m) / sd);                               // standardised feature (Act 2)
  const fit = (spec, deg) => {
    const I = E.map((_, i) => i).filter((i) => tr[i]), k = deg + 1, X = new Float64Array(I.length * k), y = new Float64Array(I.length);
    I.forEach((i, r) => { for (let j = 0; j < k; j++) X[r * k + j] = u[i] ** j; y[r] = Math.log(spec[i]); });
    const c = lstsq(X, y, I.length, k);
    return E.map((_, i) => { let v = 0; for (let j = 0; j < k; j++) v += c[j] * u[i] ** j; return v; });   // log I
  };
  const mse = (spec, pred, mask) => { let a = 0, n = 0; E.forEach((_, i) => { if (mask[i]) { a += (Math.log(spec[i]) - pred[i]) ** 2; n++; } }); return a / n; };
  return { E, px, mean, tr, va, fit, mse };
}

// ---------------------------------------------------------------------------------------------
// poly-bg
// ---------------------------------------------------------------------------------------------
register("poly-bg", (el) => {
  const wait = waiting(el);
  loadData("preedge").then((d) => {
    wait.remove();
    const S = setup(d), E = S.E;
    let deg = 1, which = "pixel";
    const cS = chart({ width: 1080, height: 470, x: { domain: [463, 545], label: "energy loss (eV)" }, y: { domain: [0, 1], label: "log I" }, margin: { l: 96 } });
    const cE = chart({ width: 600, height: 400, x: { domain: [-0.3, DMAX + 0.3], label: "polynomial degree d", ticks: [0, 1, 2, 3, 4, 5, 6, 7, 8] }, y: { domain: [1e-5, 1e2], type: "log", label: "mean squared error (log I)" }, margin: { l: 96 } });
    cE.setDomain();
    const out = readout();
    function draw() {
      const spec = which === "pixel" ? S.px[7] : S.mean;
      const ly = E.map((_, i) => Math.log(spec[i])), vis = E.map((e, i) => i).filter((i) => E[i] <= 545);
      const lo = Math.min(...vis.map((i) => ly[i])), hi = Math.max(...vis.map((i) => ly[i]));
      cS.setDomain(null, [lo - 0.15, hi + 0.15]); cS.clear();
      cS.rect(TRAIN[0], lo - 0.15, TRAIN[1], hi + 0.15, { fill: T.truth, opacity: 0.12 });
      cS.rect(VAL[0], lo - 0.15, VAL[1], hi + 0.15, { fill: T.warn, opacity: 0.14 });
      cS.text((TRAIN[0] + TRAIN[1]) / 2, hi + 0.08, "train", { fill: T.truth, anchor: "middle", size: 20 });
      cS.text((VAL[0] + VAL[1]) / 2, hi + 0.08, "validation", { fill: T.warn, anchor: "middle", size: 20 });
      cS.ruleX(513, { stroke: T.thread }); cS.text(514, lo - 0.08, "V-L₃ edge", { fill: T.thread, size: 19 });
      cS.line(vis.map((i) => [E[i], ly[i]]), { stroke: "#fff", width: 2.5 });
      const p = S.fit(spec, deg);
      cS.line(vis.map((i) => [E[i], p[i]]), { stroke: T.gauss, width: 3.5, dash: "10 6" });
      // error curves
      cE.clear();
      const trE = [], vaE = [];
      for (let k = 0; k <= DMAX; k++) { const q = S.fit(spec, k); trE.push([k, S.mse(spec, q, S.tr)]); vaE.push([k, S.mse(spec, q, S.va)]); }
      cE.line(trE, { stroke: T.truth, width: 3.5 }); cE.dots(trE, { r: 6, fill: T.truth });
      cE.line(vaE, { stroke: T.warn, width: 3.5 }); cE.dots(vaE, { r: 6, fill: T.warn });
      cE.ruleX(deg, { stroke: T.thread });
      const best = vaE.reduce((a, b) => (b[1] < a[1] ? b : a));
      out.set(`degree <b>${deg}</b>: train MSE ${trE[deg][1].toExponential(1)}, validation MSE <b>${vaE[deg][1].toExponential(1)}</b><br>` +
        `best validation degree: <b class="good">${best[0]}</b> &nbsp;${deg === 1 ? "(the power law A·E⁻ʳ)" : ""}<br>` +
        `the training error falls with every degree; the validation error does not`);
    }
    const ds = slider({ label: "degree d", min: 0, max: DMAX, step: 1, value: deg, digits: 0, onInput: (v) => { deg = Math.round(v); draw(); } });
    const wt = toggle({ label: "data", options: [{ value: "pixel", label: "one pixel" }, { value: "mean", label: "mean of 30 pixels" }], value: which, onChange: (v) => { which = v; draw(); } });
    el.append(controls(ds, wt), row(col(cS.el, src(SRC_CEVO3 + " CeVOₓ region; fit of log I as a polynomial in standardised log E.")),
      h("div", { class: "xp-gen-side" }, cE.el, card(out), legend([["#fff", "spectrum"], [T.gauss, "fitted background", "dash"], [T.truth, "train error"], [T.warn, "validation error"]]))));
    draw();
  });
});

// ---------------------------------------------------------------------------------------------
// bias-var
// ---------------------------------------------------------------------------------------------
register("bias-var", (el) => {
  const wait = waiting(el);
  loadData("preedge").then((d) => {
    wait.remove();
    const S = setup(d), E = S.E, N = S.px.length;
    const truth = E.map((_, i) => Math.log(S.mean[i]));
    const dec = [];
    for (let k = 0; k <= DMAX; k++) {
      const P = S.px.map((p) => S.fit(p, k));
      let b2 = 0, v = 0, n = 0;
      E.forEach((_, i) => {
        if (!S.va[i]) return;
        const mu = P.reduce((a, q) => a + q[i], 0) / N;
        b2 += (mu - truth[i]) ** 2; v += P.reduce((a, q) => a + (q[i] - mu) ** 2, 0) / N; n++;
      });
      dec.push({ k, P, b2: b2 / n, v: v / n });
    }
    let deg = 1;
    const cF = chart({ width: 1000, height: 560, x: { domain: [463, 525], label: "energy loss (eV)" }, y: { domain: [0, 1], label: "log I" }, margin: { l: 96 } });
    const cD = chart({ width: 640, height: 420, x: { domain: [-0.5, DMAX + 0.5], label: "polynomial degree d", ticks: [0, 1, 2, 3, 4, 5, 6, 7, 8] }, y: { domain: [1e-6, 1e4], type: "log", label: "error at the validation channels" }, margin: { l: 96 } });
    cD.setDomain();
    const out = readout();
    function draw() {
      const D = dec[deg], vis = E.map((e, i) => i).filter((i) => E[i] <= 525);
      const lo = Math.min(...vis.map((i) => truth[i])), hi = Math.max(...vis.map((i) => truth[i]));
      cF.setDomain(null, [lo - 0.25, hi + 0.25]); cF.clear();
      cF.rect(S.E.find((e) => e >= 496.5), lo - 0.25, 508, hi + 0.25, { fill: T.warn, opacity: 0.12 });
      D.P.forEach((q, j) => cF.line(vis.map((i) => [E[i], q[i]]), { stroke: T.gauss, width: 1.6, opacity: 0.45 }));
      const mu = vis.map((i) => D.P.reduce((a, q) => a + q[i], 0) / N);
      cF.line(vis.map((i, j) => [E[i], mu[j]]), { stroke: T.thread, width: 4, dash: "10 6" });
      cF.line(vis.map((i) => [E[i], truth[i]]), { stroke: "#fff", width: 3 });
      cD.clear();
      const bw = 0.26;
      cD.bars(dec.map((q) => [q.k - bw, q.k, Math.max(q.b2, 1.1e-6)]), { fill: T.pink, opacity: 0.85 });
      cD.bars(dec.map((q) => [q.k, q.k + bw, Math.max(q.v, 1.1e-6)]), { fill: T.gauss, opacity: 0.85 });
      cD.line(dec.map((q) => [q.k, q.b2 + q.v]), { stroke: T.warn, width: 3 });
      cD.ruleX(deg, { stroke: T.thread });
      const best = dec.reduce((a, b) => (b.b2 + b.v < a.b2 + a.v ? b : a));
      out.set(`degree <b>${deg}</b>: bias² <b style="color:${T.pink}">${D.b2.toExponential(1)}</b> &nbsp; variance <b style="color:${T.gauss}">${D.v.toExponential(1)}</b><br>` +
        `expected error = bias² + variance (+ noise) &nbsp; best: degree <b class="good">${best.k}</b><br>` +
        `${N} real pixels = ${N} repeated measurements of the same background`);
    }
    const ds = slider({ label: "degree d", min: 0, max: DMAX, step: 1, value: deg, digits: 0, onInput: (v) => { deg = Math.round(v); draw(); } });
    el.append(controls(ds), row(col(cF.el, src(SRC_CEVO3 + " 30 pixels of the CeVOₓ region; truth = mean of the 30 pixels.")),
      h("div", { class: "xp-gen-side" }, cD.el, card(out), legend([[T.gauss, "fit to one pixel"], [T.thread, "mean of the fits", "dash"], ["#fff", "mean spectrum"], [T.pink, "bias²"], [T.warn, "bias² + variance"]]))));
    draw();
  });
});
