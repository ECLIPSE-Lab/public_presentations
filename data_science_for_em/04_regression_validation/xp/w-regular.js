// Regularise group (Act 5), on real spectra:
//   ridge-fe  : per-pixel fit of the Fe-L edge of a lunar pyroxene with three standards (Fe0, Fe2+,
//               Fe3+). Fe0 and Fe2+ are nearly collinear: their weights trade off (correlation −0.9).
//               Ridge (RᵀR + λI)⁻¹Rᵀy shrinks the unstable direction; the fit hardly changes.
//   lasso-eds : which elements are in an EDS spectrum? Non-negative lasso over a library of 78
//               element line patterns; the path shows the order in which elements enter.
import { register, T, fmt, clamp, h, s, row, col, card, controls, readout, slider, toggle, legend, chart, imageCanvas } from "./core.js";
import { loadData } from "./data.js";
import { solve } from "./la.js";

if (!document.getElementById("xp-reg-style")) {
  document.head.append(h("style", { id: "xp-reg-style" }, `
.xp-reg-side { width: 640px; display: flex; flex-direction: column; gap: 14px; }
.xp-reg-side .xp-readout { margin-top: 0; line-height: 1.5; }
.xp-src { font-size: 17px; color: #64748b; margin-top: 4px; }
.xp-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.xp-chips span { padding: 4px 12px; border-radius: 10px; font-size: 21px; font-weight: 600; }
`));
}
const src = (html) => h("div", { class: "xp-src", html });
const SRC_CYMES = "Data: Cymes, Burgess &amp; Stroud (2022), Zenodo 7439174, CC BY 4.0.";
const SRC_CEVO3 = "Data: CeVO₃ on Al₂O₃, STEM-Bank-26 (Hollyer &amp; Stach 2026), CC BY 4.0.";
const waiting = (el) => { const w = h("div", { class: "xp-caption" }, "loading data …"); el.append(w); return w; };

// ---------------------------------------------------------------------------------------------
// ridge-fe
// ---------------------------------------------------------------------------------------------
register("ridge-fe", (el) => {
  const wait = waiting(el);
  loadData("fel").then((d) => {
    wait.remove();
    const [ny, nx, D] = d.cube.shape, N = ny * nx, K = 3, keys = ["Fe0", "Fe2", "Fe3"];
    const R = keys.map((k) => Float64Array.from(d.refs[k]));
    const G = new Float64Array(9), RY = new Float64Array(N * 3), sig = new Float64Array(N);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { let v = 0; for (let j = 0; j < D; j++) v += R[a][j] * R[b][j]; G[a * 3 + b] = v; }
    for (let i = 0; i < N; i++) for (let a = 0; a < 3; a++) { let v = 0; for (let j = 0; j < D; j++) { const y = d.cube[i * D + j] * d.scale; v += R[a][j] * y; if (a === 0) sig[i] += y; } RY[i * 3 + a] = v; }
    const med = Float64Array.from(sig).sort()[N >> 1], good = Array.from(sig, (v) => v > med);
    const tr = (G[0] + G[4] + G[8]) / 3;
    let lam = 0, show = "Fe0", sel = Math.floor(ny * 0.55) * nx + Math.floor(nx * 0.5);
    const NAMES = { Fe0: "Fe⁰ (metal)", Fe2: "Fe²⁺", Fe3: "Fe³⁺", f3: "Fe³⁺ / ΣFe" };
    const fitAll = (l) => {
      const A = G.slice(); for (let a = 0; a < 3; a++) A[a * 4] += l * tr;
      const W = new Float64Array(N * 3);
      for (let i = 0; i < N; i++) { const w = solve(A.slice(), RY.slice(i * 3, i * 3 + 3), 3); W.set(w, i * 3); }
      return W;
    };
    const IS = 300, im = imageCanvas({ n: nx, ny, size: IS, label: "" });
    const mk = s("svg", { width: IS, height: IS * ny / nx, viewBox: `0 0 ${IS} ${IS * ny / nx}`, style: "position:absolute;left:0;bottom:0;pointer-events:none" });
    const mkR = s("rect", { width: IS / nx, height: IS / nx, fill: "none", stroke: T.thread, "stroke-width": 3 }); mk.append(mkR);
    im.style.position = "relative"; im.append(mk); im.canvas.style.cursor = "crosshair";
    const cS = chart({ width: 560, height: 470, x: { domain: [-30000, 50000], label: "weight of Fe⁰" }, y: { domain: [-20000, 50000], label: "weight of Fe²⁺" }, margin: { l: 100 } });
    cS.setDomain();
    const cF = chart({ width: 620, height: 260, x: { domain: [704, 724], label: "energy loss (eV)" }, y: { domain: [-300, 1500], label: "counts" }, margin: { l: 90 } });
    const out = readout();
    let W0 = fitAll(0);
    function draw() {
      const W = lam > 0 ? fitAll(lam) : W0;
      const val = (i) => (show === "f3" ? W[i * 3 + 2] / (W[i * 3] + W[i * 3 + 1] + W[i * 3 + 2]) : W[i * 3 + keys.indexOf(show)]);
      const v = Float64Array.from({ length: N }, (_, i) => (good[i] ? val(i) : NaN));
      const lo = show === "f3" ? 0 : -20000, hi = show === "f3" ? 0.6 : 40000;
      im.draw(v.map((x) => (Number.isFinite(x) ? x : lo)), lo, hi, "magma");
      im.setLabel(`${NAMES[show]} &nbsp;(${show === "f3" ? "0–0.6" : "−2·10⁴ … 4·10⁴"})`);
      mkR.setAttribute("x", (sel % nx) * IS / nx); mkR.setAttribute("y", Math.floor(sel / nx) * IS / nx);
      cS.clear();
      cS.ruleX(0, { stroke: T.faint }); cS.ruleY(0, { stroke: T.faint });
      const P = []; for (let i = 0; i < N; i++) if (good[i]) P.push([W[i * 3], W[i * 3 + 1]]);
      cS.dots(P, { r: 3.5, fill: T.gauss, opacity: 0.5, stroke: "none" });
      cS.dots([[W[sel * 3], W[sel * 3 + 1]]], { r: 9, fill: T.thread, stroke: "#fff" });
      // the fit of the selected pixel
      const y = Array.from({ length: D }, (_, j) => d.cube[sel * D + j] * d.scale), E = Array.from(d.E);
      const fit = E.map((_, j) => W[sel * 3] * R[0][j] + W[sel * 3 + 1] * R[1][j] + W[sel * 3 + 2] * R[2][j]);
      cF.setDomain(null, [Math.min(...y) - 50, Math.max(...y) * 1.1]); cF.clear();
      cF.line(E.map((e, j) => [e, y[j]]), { stroke: "#fff", width: 1.8 });
      cF.line(E.map((e, j) => [e, fit[j]]), { stroke: T.thread, width: 3 });
      // statistics over the good pixels
      const stat = (k) => { let m = 0, n = 0; for (let i = 0; i < N; i++) if (good[i]) { m += W[i * 3 + k]; n++; } m /= n; let v2 = 0; for (let i = 0; i < N; i++) if (good[i]) v2 += (W[i * 3 + k] - m) ** 2; return [m, Math.sqrt(v2 / n)]; };
      const [m0, s0] = stat(0), [m2, s2] = stat(1);
      let c = 0, n = 0; for (let i = 0; i < N; i++) if (good[i]) { c += (W[i * 3] - m0) * (W[i * 3 + 1] - m2); n++; } c /= n * s0 * s2;
      let rr = 0, yy = 0; for (let i = 0; i < N; i++) if (good[i]) for (let j = 0; j < D; j++) { const yv = d.cube[i * D + j] * d.scale, f = W[i * 3] * R[0][j] + W[i * 3 + 1] * R[1][j] + W[i * 3 + 2] * R[2][j]; rr += (yv - f) ** 2; yy += yv * yv; }
      out.set(`κ(R) = 15 &nbsp; λ = <b>${lam > 0 ? lam.toExponential(0) : "0"}</b> × tr(RᵀR)/3<br>` +
        `corr(Fe⁰, Fe²⁺ weights) over pixels: <b class="${c < -0.7 ? "bad" : ""}">${fmt(c, 2)}</b><br>` +
        `spread of the Fe⁰ weight: <b>${fmt(s0, 0)}</b> &nbsp; mean Fe⁰: ${fmt(m0, 0)}${m0 < 0 ? ' <span class="bad">(&lt; 0: unphysical)</span>' : ""}<br>` +
        `fit residual ‖y − Rw‖/‖y‖ = ${fmt(Math.sqrt(rr / yy), 4)}`);
    }
    im.canvas.addEventListener("pointerdown", (e) => {
      const r = im.canvas.getBoundingClientRect(); sel = clamp(Math.floor((e.clientY - r.top) / r.height * ny), 0, ny - 1) * nx + clamp(Math.floor((e.clientX - r.left) / r.width * nx), 0, nx - 1); draw();
    });
    const ls = slider({ label: "ridge λ", min: 1e-4, max: 1, log: true, value: 1e-4, format: (v) => (v <= 1.0001e-4 ? "0" : v.toExponential(0)), onInput: (v) => { lam = v <= 1.0001e-4 ? 0 : v; draw(); } });
    const mt = toggle({ label: "map", options: Object.entries(NAMES).map(([k, v]) => ({ value: k, label: v })), value: show, onChange: (v) => { show = v; draw(); } });
    el.append(controls(ls, mt), row(col(im, h("div", { style: { maxWidth: "300px" } }, src(SRC_CYMES + " Fe-L₂,₃ of a lunar pyroxene rim, 3×3 binned; brightest half of the pixels."))),
      col(cS.el), h("div", { class: "xp-reg-side", style: { width: "640px" } }, cF.el, card(out),
        legend([[T.gauss, "one pixel: (w<sub>Fe⁰</sub>, w<sub>Fe²⁺</sub>)"], [T.thread, "selected pixel / its fit"], ["#fff", "spectrum"]]))));
    draw();
  });
});

// ---------------------------------------------------------------------------------------------
// lasso-eds
// ---------------------------------------------------------------------------------------------
register("lasso-eds", (el) => {
  const wait = waiting(el);
  loadData("eds").then((d) => {
    wait.remove();
    const E = Array.from(d.E), D = E.length, L = d.lib, M = d.names.length, names = d.names;
    const TRUTH = new Set(d.truth), EXTRA = new Set(d.expected_extra);
    function prepare(key) {
      const y = Array.from(d[key]), w = y.map((v) => 1 / Math.sqrt(Math.max(v, 1)));   // Poisson weights (Week 2)
      // background basis, projected out: 1, E, E², exp(−E)
      const B = [E.map(() => 1), E, E.map((e) => e * e), E.map((e) => Math.exp(-e))].map((b) => b.map((v, j) => v * w[j]));
      // Gram–Schmidt on B
      const Q = [];
      for (const b of B) { let v = b.slice(); for (const q of Q) { const c = v.reduce((a, x, j) => a + x * q[j], 0); v = v.map((x, j) => x - c * q[j]); } const n = Math.hypot(...v); Q.push(v.map((x) => x / n)); }
      const proj = (v) => { let r = v.slice(); for (const q of Q) { const c = r.reduce((a, x, j) => a + x * q[j], 0); r = r.map((x, j) => x - c * q[j]); } return r; };
      const yp = proj(y.map((v, j) => v * w[j]));
      const X = [], nrm = [];
      for (let k = 0; k < M; k++) { const c = proj(Array.from({ length: D }, (_, j) => L[k * D + j] * w[j])); const n = Math.hypot(...c) || 1; nrm.push(n); X.push(c.map((v) => v / n)); }
      // non-negative lasso path by coordinate descent, warm starts
      const lmax = Math.max(...X.map((c) => c.reduce((a, v, j) => a + v * yp[j], 0)));
      const lams = Array.from({ length: 50 }, (_, i) => lmax * 10 ** (-3.5 * i / 49));
      let b = new Float64Array(M), r = yp.slice();
      const path = [];
      for (const lam of lams) {
        for (let sweep = 0; sweep < 200; sweep++) {
          let dmax = 0;
          for (let k = 0; k < M; k++) {
            const ck = X[k], g = ck.reduce((a, v, j) => a + v * r[j], 0) + b[k];
            const nb = Math.max(0, g - lam);
            if (nb !== b[k]) { const dlt = nb - b[k]; for (let j = 0; j < D; j++) r[j] -= dlt * ck[j]; dmax = Math.max(dmax, Math.abs(dlt)); b[k] = nb; }
          }
          if (dmax < 1e-6 * lmax) break;
        }
        path.push(Float64Array.from(b));
      }
      const order = []; path.forEach((p) => p.forEach((v, k) => { if (v > 0 && !order.includes(k)) order.push(k); }));
      return { y, lams, path, order, lmax, nrm, w };
    }
    const P = { total: prepare("total"), small: prepare("small") };
    let key = "total";
    const firstN = (p, n) => { let i = 0; while (i < 49 && p.path[i + 1].filter((v) => v > 0).length <= n) i++; return i; };
    let li = firstN(P.total, 5);
    const cP = chart({ width: 820, height: 470, x: { domain: [0, 49], label: "λ  (large → small)", ticks: [] }, y: { domain: [0, 1], label: "lasso weight (standardised)" }, margin: { l: 90 } });
    const cY = chart({ width: 820, height: 290, x: { domain: [0.2, 10.2], label: "X-ray energy (keV)" }, y: { domain: [1, 1e5], type: "log", label: "counts" }, margin: { l: 90 } });
    cY.setDomain();
    const chips = h("div", { class: "xp-chips" }), out = readout();
    const col3 = (k) => (TRUTH.has(names[k]) ? T.truth : EXTRA.has(names[k]) ? T.warn : T.thread);
    function draw() {
      const p = P[key], top = Math.max(...p.path[p.path.length - 1]);
      cP.setDomain(null, [0, top * 1.08]); cP.clear();
      cP.ruleX(li, { stroke: "#fff" });
      for (const k of p.order.slice(0, 14)) {
        cP.line(p.path.map((b, i) => [i, b[k]]), { stroke: col3(k), width: 3 });
        const iEnd = p.path.length - 1; cP.text(iEnd + 0.3, p.path[iEnd][k], names[k], { fill: col3(k), size: 17 });
      }
      const b = p.path[li], sel = p.order.filter((k) => b[k] > 0);
      chips.replaceChildren(...sel.map((k) => h("span", { style: { background: col3(k) + "33", color: col3(k), border: `1px solid ${col3(k)}` } }, names[k])));
      const ymax = Math.max(...p.y);
      cY.setDomain(null, [Math.max(1, Math.min(...p.y.filter((v) => v > 0)) * 0.7), ymax * 1.5]); cY.clear();
      cY.line(E.map((e, j) => [e, Math.max(p.y[j], 0.8)]), { stroke: "#fff", width: 2 });
      const used = [];
      for (const k of sel.slice(0, 10)) {
        // label at the strongest line above 0.9 keV if there is one (V-K, not V-L under O-K)
        const mx = Math.max(...Array.from({ length: D }, (_, j) => L[k * D + j]));
        let j0 = E.reduce((a, _, j) => (L[k * D + j] > L[k * D + a] ? j : a), 0);
        const hiJ = E.reduce((a, e, j) => (e > 0.9 && L[k * D + j] > (a < 0 ? 0 : L[k * D + a]) ? j : a), -1);
        if (hiJ >= 0 && L[k * D + hiJ] > 0.25 * mx) j0 = hiJ;
        const n = used.filter((u) => Math.abs(u - E[j0]) < 0.35).length; used.push(E[j0]);
        cY.text(E[j0], Math.max(p.y[j0], 1) * 1.35 * 2.2 ** n, names[k], { fill: col3(k), size: 17, anchor: "middle" });
      }
      const nT = sel.filter((k) => TRUTH.has(names[k])).length, nE = sel.filter((k) => EXTRA.has(names[k])).length, nF = sel.length - nT - nE;
      out.set(`λ = ${(p.lams[li] / p.lmax).toExponential(1)} × λ<sub>max</sub>: <b>${sel.length}</b> of ${names.length} elements selected<br>` +
        `<span style="color:${T.truth}">sample (O, Al, V, Ce): ${nT}</span> &nbsp; <span style="color:${T.warn}">expected extra (C film, Cu grid, Si): ${nE}</span> &nbsp; <span style="color:${T.thread}">other: ${nF}</span><br>` +
        `entry order: ${p.order.slice(0, 8).map((k) => names[k]).join(" → ")}`);
    }
    const ls = slider({ label: "λ", min: 0, max: 49, step: 1, value: li, format: (v) => `step ${Math.round(v)}`, onInput: (v) => { li = Math.round(v); draw(); } });
    const kt = toggle({ label: "spectrum", options: [{ value: "total", label: "all 6 400 pixels (371 k counts)" }, { value: "small", label: "5×5 pixels (1.6 k counts)" }], value: key, onChange: (v) => { key = v; li = firstN(P[key], 5); ls.set(li, false); draw(); } });
    el.append(controls(kt, ls), row(col(cP.el, cY.el), h("div", { class: "xp-reg-side", style: { width: "640px" } }, chips, card(out),
      legend([[T.truth, "in the sample"], [T.warn, "expected extra"], [T.thread, "not expected"]]), src(SRC_CEVO3 + " Library: line energies and weights from the exspy element database; Poisson-weighted, smooth background projected out."))));
    draw();
  });
});
