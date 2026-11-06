// Descend and Step groups (Acts 2-3), on real EELS data:
//   gd-bowl   : gradient descent on the power-law background fit (log I = b + m log E, CeVOx region,
//               470-508 eV). Features raw / centred / standardised change the shape of the bowl.
//               data-mode="gd" (one optimiser, learning-rate slider) | "race" (GD, momentum, Adam, SGD)
//   align     : aligning a Fe-L spectrum to the FeO standard by its energy shift: a non-convex loss.
import { register, T, fmt, clamp, rng, h, s, row, col, card, controls, readout, slider, toggle, button, legend, chart } from "./core.js";
import { loadData } from "./data.js";

if (!document.getElementById("xp-desc-style")) {
  document.head.append(h("style", { id: "xp-desc-style" }, `
.xp-desc-side { width: 700px; display: flex; flex-direction: column; gap: 14px; }
.xp-desc-side .xp-readout { margin-top: 0; line-height: 1.5; }
.xp-src { font-size: 17px; color: #64748b; margin-top: 4px; }
`));
}
const src = (html) => h("div", { class: "xp-src", html });
export const SRC_CEVO3 = "Data: CeVO₃ on Al₂O₃, STEM-Bank-26 (Hollyer &amp; Stach 2026), CC BY 4.0.";
const SRC_CYMES = "Data: Cymes, Burgess &amp; Stroud (2022), Zenodo 7439174, CC BY 4.0.";
const waiting = (el) => { const w = h("div", { class: "xp-caption" }, "loading data …"); el.append(w); return w; };

// pre-edge points of the CeVOx region (the same as Week 3): x = log E, y = log I
export function preedgePoints(d, lo = 470, hi = 508, box = [26, 32, 33, 38]) {
  const [ny, nx, ne] = d.cube.shape, E = Array.from(d.E), sp = new Float64Array(ne);
  let c = 0;
  for (let y = box[0]; y < box[1]; y++) for (let x = box[2]; x < box[3]; x++) { for (let k = 0; k < ne; k++) sp[k] += d.cube[(y * nx + x) * ne + k] * d.scale; c++; }
  const idx = E.map((e, i) => i).filter((i) => E[i] >= lo && E[i] <= hi);
  return { x: idx.map((i) => Math.log(E[i])), y: idx.map((i) => Math.log(sp[i] / c)), E: idx.map((i) => E[i]) };
}

// a 2-parameter least-squares problem L(w) = 1/(2n) Σ (y − w0 f0 − w1 f1)²
function problem(x, y, scaling) {
  const n = x.length, xm = x.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(x.reduce((a, b) => a + (b - xm) ** 2, 0) / n);
  const f = (v) => (scaling === "raw" ? v : scaling === "centred" ? v - xm : (v - xm) / sd);
  const F = x.map(f);
  let s00 = n, s01 = 0, s11 = 0, r0 = 0, r1 = 0;
  F.forEach((v, i) => { s01 += v; s11 += v * v; r0 += y[i]; r1 += v * y[i]; });
  const H = [s00 / n, s01 / n, s01 / n, s11 / n], det = H[0] * H[3] - H[1] * H[2];
  const wHat = [(H[3] * r0 / n - H[1] * r1 / n) / det, (H[0] * r1 / n - H[2] * r0 / n) / det];
  const tr = H[0] + H[3], disc = Math.sqrt(tr * tr / 4 - det), lmax = tr / 2 + disc, lmin = tr / 2 - disc;
  const loss = (w) => { let a = 0; F.forEach((v, i) => { a += (y[i] - w[0] - w[1] * v) ** 2; }); return a / (2 * n); };
  const grad = (w, idx) => { const I = idx || F.map((_, i) => i); let g0 = 0, g1 = 0;
    for (const i of I) { const r = y[i] - w[0] - w[1] * F[i]; g0 -= r; g1 -= r * F[i]; } return [g0 / I.length, g1 / I.length]; };
  // the same physical line (level b at the window centre, slope m) in these parameters
  const fromLine = (b, m) => (scaling === "raw" ? [b - m * xm, m] : scaling === "centred" ? [b, m] : [b, m * sd]);
  const toLine = (w) => (scaling === "raw" ? [w[0] + w[1] * xm, w[1]] : scaling === "centred" ? [w[0], w[1]] : [w[0], w[1] / sd]);
  return { n, F, H, wHat, lmax, lmin, kappa: lmax / lmin, loss, grad, fromLine, toLine, xm, sd, Lmin: loss(wHat) };
}

const OPT = {
  gd: { label: "GD", color: T.gauss },
  mom: { label: "momentum", color: T.warn },
  adam: { label: "Adam", color: T.truth },
  sgd: { label: "SGD (4 points, 0.3 η)", color: T.pink },
};
function stepper(kind, P, eta, seed = 3) {
  const R = rng(seed);
  let v = [0, 0], m = [0, 0], q = [0, 0], t = 0;
  return (w) => {
    t++;
    if (kind === "sgd") { const I = Array.from({ length: 4 }, () => Math.floor(R.u() * P.n)); const g = P.grad(w, I); return [w[0] - eta * g[0], w[1] - eta * g[1]]; }
    const g = P.grad(w);
    if (kind === "gd") return [w[0] - eta * g[0], w[1] - eta * g[1]];
    if (kind === "mom") { v = [0.9 * v[0] + g[0], 0.9 * v[1] + g[1]]; return [w[0] - eta * v[0], w[1] - eta * v[1]]; }
    // Adam: per-parameter step sizes
    const a = 0.02;
    m = [0.9 * m[0] + 0.1 * g[0], 0.9 * m[1] + 0.1 * g[1]]; q = [0.999 * q[0] + 0.001 * g[0] ** 2, 0.999 * q[1] + 0.001 * g[1] ** 2];
    const mh = m.map((v2) => v2 / (1 - 0.9 ** t)), qh = q.map((v2) => v2 / (1 - 0.999 ** t));
    return [w[0] - a * mh[0] / (Math.sqrt(qh[0]) + 1e-8), w[1] - a * mh[1] / (Math.sqrt(qh[1]) + 1e-8)];
  };
}

// ---------------------------------------------------------------------------------------------
// gd-bowl
// ---------------------------------------------------------------------------------------------
register("gd-bowl", (el, api) => {
  const wait = waiting(el);
  const mode = api.data.mode || "gd";
  loadData("preedge").then((d) => {
    wait.remove();
    const pts = preedgePoints(d);
    let scaling = mode === "race" ? "centred" : "raw", etaRel = 0.9, P, start, runs = {}, it = 0, timer = null;
    const NMAX = 3000;
    const cB = chart({ width: 640, height: 600, x: { domain: [0, 1], label: "w₁" }, y: { domain: [0, 1], label: "w₀" }, margin: { l: 96 } });
    const cL = chart({ width: 660, height: 330, x: { domain: [1, NMAX], type: "log", label: "iteration" }, y: { domain: [1e-6, 10], type: "log", label: "L(w) − L_min" }, margin: { l: 96 } });
    cL.setDomain();
    const imgEl = s("image", { preserveAspectRatio: "none", style: "image-rendering:pixelated" });
    cB.gPlot.append(imgEl);
    const gT = s("g"); cB.gTop.append(gT);
    const out = readout();
    function setup() {
      P = problem(pts.x, pts.y, scaling);
      // the start: a flat line 0.15 above the mean level (the same physical line for every scaling)
      const yb = pts.y.reduce((a, b) => a + b, 0) / pts.y.length;
      start = P.fromLine(yb + 0.15, 0);
      // plot window: around the minimum, wide enough for the start, in the parameters of this scaling
      const d1 = Math.max(Math.abs(start[1] - P.wHat[1]) * 1.6, 1e-3), d0 = Math.max(Math.abs(start[0] - P.wHat[0]) * 1.6, 0.05);
      cB.setDomain([P.wHat[1] - d1, P.wHat[1] + d1], [P.wHat[0] - d0, P.wHat[0] + d0]);
      // loss image (log bands)
      const G = 140, cv = document.createElement("canvas"); cv.width = G; cv.height = G;
      const ctx = cv.getContext("2d"), im = ctx.createImageData(G, G);
      const [x0, x1] = [P.wHat[1] - d1, P.wHat[1] + d1], [y0, y1] = [P.wHat[0] - d0, P.wHat[0] + d0];
      let lmax = 0; const vals = new Float64Array(G * G);
      for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { const w = [y1 - (y1 - y0) * (j + 0.5) / G, x0 + (x1 - x0) * (i + 0.5) / G]; vals[j * G + i] = Math.log(P.loss(w) - P.Lmin + 1e-9); lmax = Math.max(lmax, vals[j * G + i]); }
      const lmin = Math.log(1e-6);
      for (let k = 0; k < G * G; k++) {
        const t = clamp((vals[k] - lmin) / (lmax - lmin), 0, 1), band = Math.floor(t * 16) % 2 ? 0.85 : 1;
        im.data[4 * k] = band * (8 + 150 * t); im.data[4 * k + 1] = band * (12 + 110 * t); im.data[4 * k + 2] = band * (30 + 160 * t); im.data[4 * k + 3] = 255;
      }
      ctx.putImageData(im, 0, 0);
      const m = cB.m;
      Object.entries({ href: cv.toDataURL(), x: m.l, y: m.t, width: cB.width - m.l - m.r, height: cB.height - m.t - m.b }).forEach(([k, v]) => imgEl.setAttribute(k, v));
      reset();
    }
    function reset() {
      stop(); it = 0;
      const kinds = mode === "race" ? Object.keys(OPT) : ["gd"];
      const eta = etaRel * 2 / P.lmax;
      runs = Object.fromEntries(kinds.map((k) => [k, { w: start.slice(), path: [start.slice()], hist: [P.loss(start) - P.Lmin], step: stepper(k, P, k === "sgd" ? 0.3 * eta : eta), done: null }]));
      draw();
    }
    function stop() { if (timer) cancelAnimationFrame(timer); timer = null; }
    function run() {
      stop();
      const tick = () => {
        const per = it < 50 ? 1 : it < 300 ? 4 : 20;
        for (let r = 0; r < per && it < NMAX; r++) {
          it++;
          for (const [k, R] of Object.entries(runs)) {
            if (!Number.isFinite(R.w[0]) || Math.abs(R.w[1]) > 1e6) continue;
            R.w = R.step(R.w); const L = P.loss(R.w) - P.Lmin;
            R.hist.push(L); if (R.path.length < 4000) R.path.push(R.w.slice());
            if (R.done === null && L < 1e-4 * R.hist[0]) R.done = it;
          }
        }
        draw();
        if (it < NMAX) timer = requestAnimationFrame(tick); else timer = null;
      };
      timer = requestAnimationFrame(tick);
    }
    function draw() {
      gT.replaceChildren();
      const X = (w) => cB.x(w[1]), Y = (w) => cB.y(w[0]);
      gT.append(s("path", { d: `M${X(P.wHat) - 11},${Y(P.wHat) - 11} L${X(P.wHat) + 11},${Y(P.wHat) + 11} M${X(P.wHat) - 11},${Y(P.wHat) + 11} L${X(P.wHat) + 11},${Y(P.wHat) - 11}`, stroke: "#fff", "stroke-width": 3 }));
      cL.clear();
      for (const [k, R] of Object.entries(runs)) {
        const vis = R.path.filter((w) => Number.isFinite(w[0]) && Math.abs(X(w)) < 5000 && Math.abs(Y(w)) < 5000);
        gT.append(s("polyline", { points: vis.map((w) => `${X(w)},${Y(w)}`).join(" "), fill: "none", stroke: OPT[k].color, "stroke-width": 2.5, opacity: 0.95 }));
        const w = R.path[R.path.length - 1];
        if (Number.isFinite(w[0])) gT.append(s("circle", { cx: clamp(X(w), -20, 2000), cy: clamp(Y(w), -20, 2000), r: 8, fill: OPT[k].color, stroke: "#fff", "stroke-width": 2 }));
        cL.line(R.hist.map((v, i) => [i + 1, clamp(v, 1e-7, 1e3)]).filter((p, i) => i < 50 || i % 5 === 0), { stroke: OPT[k].color, width: 3 });
      }
      gT.append(s("circle", { cx: X(start), cy: Y(start), r: 7, fill: "none", stroke: "#fff", "stroke-width": 2 }));
      const g = runs.gd, eta = etaRel * 2 / P.lmax;
      const sum = (R) => (R.done ? `<b class="good">${R.done}</b> steps` : !Number.isFinite(R.w[0]) || R.hist[R.hist.length - 1] > 1e3 * R.hist[0] ? `<b class="bad">diverged</b>` : `not yet (${it})`);
      out.set(`features: <b>${scaling}</b> &nbsp; Hessian eigenvalues ${P.lmax.toExponential(1)}, ${P.lmin.toExponential(1)} → κ = <b>${P.kappa < 1e4 ? fmt(P.kappa, 0) : P.kappa.toExponential(1)}</b><br>` +
        `η = ${fmt(etaRel, 2)} × 2/λ<sub>max</sub> = ${eta.toExponential(2)}${etaRel > 1 ? ' &nbsp;<span class="bad">above the stability limit</span>' : ""}<br>` +
        (mode === "race" ? Object.entries(runs).map(([k, R]) => `<span style="color:${OPT[k].color}">${OPT[k].label}</span>: ${sum(R)}`).join(" &nbsp; ")
          : `GD to 10⁻⁴ of the start error: ${sum(g)} &nbsp; (theory: ${(() => { const q = Math.max(Math.abs(1 - eta * P.lmin), Math.abs(1 - eta * P.lmax)); return q >= 1 ? "never" : "≈ " + fmt(Math.log(1e4) / (-2 * Math.log(q)), 0); })()})`));
    }
    const st = toggle({ label: "features", options: [{ value: "raw", label: "raw log E" }, { value: "centred", label: "centred" }, { value: "std", label: "standardised" }], value: scaling, onChange: (v) => { scaling = v; setup(); } });
    const es = slider({ label: "η / (2/λ<sub>max</sub>)", min: 0.05, max: 1.1, value: etaRel, digits: 2, onInput: (v) => { etaRel = v; reset(); } });
    cB.el.style.cursor = "crosshair";
    cB.el.addEventListener("pointerdown", (e) => {
      const pt = cB.el.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; const q = pt.matrixTransform(cB.el.getScreenCTM().inverse());
      start = [cB.y.invert(q.y), cB.x.invert(q.x)]; reset();
    });
    const leg = mode === "race" ? legend(Object.values(OPT).map((o) => [o.color, o.label])) : legend([[T.gauss, "GD path"], ["#fff", "minimum ×"]]);
    el.append(controls(st, es, button("run", run), button("reset", reset), h("span", { class: "xp-slider-label" }, "click the bowl: new start")),
      row(col(cB.el), h("div", { class: "xp-desc-side" }, cL.el, card(out), leg, src(SRC_CEVO3 + " Fit of log I = w₀ + w₁·f(log E), 470–508 eV."))));
    setup();
  });
});

// ---------------------------------------------------------------------------------------------
// align: the loss of an energy shift is not convex
// ---------------------------------------------------------------------------------------------
register("align", (el) => {
  const wait = waiting(el);
  loadData("align").then((d) => {
    wait.remove();
    const E = Array.from(d.E), ref = Array.from(d.ref), y = Array.from(d.y);
    const shifted = (sh) => E.map((e) => { const q = e - sh, i = (q - E[0]) / (E[1] - E[0]), i0 = Math.floor(i); return i0 < 0 || i0 >= E.length - 1 ? 0 : ref[i0] + (i - i0) * (ref[i0 + 1] - ref[i0]); });
    const lossOf = (sh) => { const r = shifted(sh); let ry = 0, rr = 0; r.forEach((v, i) => { ry += v * y[i]; rr += v * v; }); const a = ry / (rr || 1); let L = 0; r.forEach((v, i) => { L += (y[i] - a * v) ** 2; }); return { L, a }; };
    const S = Array.from({ length: 401 }, (_, i) => -20 + i * 0.1), LS = S.map((v) => lossOf(v).L);
    const Lmax = Math.max(...LS);
    let s0 = -9, cur = s0, path = [], timer = null;
    const cLo = chart({ width: 900, height: 360, x: { domain: [-20, 20], label: "energy shift s (eV)" }, y: { domain: [0, Lmax * 1.05], label: "loss L(s)" }, margin: { l: 90 } });
    const cS = chart({ width: 900, height: 330, x: { domain: [690, 745], label: "energy loss (eV)" }, y: { domain: [-0.15, 1.1], label: "intensity" }, margin: { l: 90 } });
    cLo.setDomain(); cS.setDomain();
    const out = readout();
    function draw() {
      cLo.clear();
      cLo.line(S.map((v, i) => [v, LS[i]]), { stroke: T.data, width: 3 });
      if (path.length) cLo.line(path.map((v) => [v, lossOf(v).L]), { stroke: T.thread, width: 2, dash: "4 4" });
      cLo.dots([[cur, lossOf(cur).L]], { r: 11, fill: T.thread, stroke: "#fff" });
      cLo.dots([[s0, lossOf(s0).L]], { r: 7, fill: "none", stroke: "#fff" });
      const { a } = lossOf(cur);
      cS.clear();
      cS.line(E.map((e, i) => [e, y[i]]), { stroke: "#fff", width: 2.5 });
      cS.line(E.map((e, i) => [e, a * shifted(cur)[i]]), { stroke: T.thread, width: 2.5 });
      const iBest = LS.indexOf(Math.min(...LS)), best = S[iBest];
      out.set((path.length ? `start s₀ = ${fmt(s0, 1)} eV → GD ends at <b>s = ${fmt(cur, 2)} eV</b>` : `start s₀ = ${fmt(s0, 1)} eV: press <b>descend</b>`) + `<br>` +
        `global minimum: s = ${fmt(best, 1)} eV ${!path.length ? "" : Math.abs(cur - best) < 0.3 ? '<span class="good">✓ found</span>' : '<span class="bad">✗ stuck in a local minimum</span>'}<br>` +
        `the side minima: the L₂ peak of one spectrum sits on the L₃ peak of the other (13 eV apart)`);
    }
    function descend() {
      if (timer) cancelAnimationFrame(timer);
      cur = s0; path = [cur]; let k = 0;
      const tick = () => {
        for (let r = 0; r < 3; r++) { const g = (lossOf(cur + 0.02).L - lossOf(cur - 0.02).L) / 0.04; cur = clamp(cur - 0.25 * g / (Lmax / 20), -20, 20); path.push(cur); k++; }
        draw();
        if (k < 300 && Math.abs(lossOf(cur + 0.02).L - lossOf(cur - 0.02).L) > 1e-4) timer = requestAnimationFrame(tick);
      };
      timer = requestAnimationFrame(tick);
    }
    cLo.el.style.cursor = "crosshair";
    cLo.el.addEventListener("pointerdown", (e) => {
      const pt = cLo.el.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY; const q = pt.matrixTransform(cLo.el.getScreenCTM().inverse());
      s0 = clamp(cLo.x.invert(q.x), -20, 20); cur = s0; path = []; draw();
    });
    el.append(controls(button("descend", descend), h("span", { class: "xp-slider-label" }, "click the loss curve: new start s₀")),
      row(col(cLo.el, cS.el), h("div", { class: "xp-desc-side", style: { width: "640px" } }, card(out),
        legend([["#fff", "pyroxene Fe-L (measured)"], [T.thread, "FeO standard, shifted by s"]]), src(SRC_CYMES + " Loss: Σ (y − a·ref(E − s))², best scale a for each s."))));
    draw();
  });
});
