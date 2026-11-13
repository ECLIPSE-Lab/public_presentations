// Loss group (Act 1), on real 4D-STEM disk positions (ion-irradiated Al, Mills et al. 2023):
//   robust-fit  : fit the lattice p = c + h g1 + k g2 to the detected disks with MSE, MAE or Huber.
//                 One misdetected disk drags the MSE lattice and creates fake strain.
//   resid-hist  : the residuals of 21 000 real disk fits: a Gaussian core with heavy tails.
//                 The loss is the negative log-likelihood of the noise model.
import { register, T, fmt, clamp, h, s, row, col, card, controls, readout, slider, toggle, legend, chart, imageCanvas } from "./core.js";
import { loadData } from "./data.js";
import { solve } from "./la.js";

if (!document.getElementById("xp-loss-style")) {
  document.head.append(h("style", { id: "xp-loss-style" }, `
.xp-loss-side { width: 760px; display: flex; flex-direction: column; gap: 14px; }
.xp-loss-side .xp-readout { margin-top: 0; line-height: 1.5; }
.xp-src { font-size: 17px; color: #64748b; margin-top: 4px; }
.xp-loss-pat { position: relative; }
.xp-loss-pat svg .disk { cursor: grab; }
`));
}
const src = (html) => h("div", { class: "xp-src", html });
export const SRC_MILLS = "Data: ion-irradiated Al, [001], 4D-STEM; Mills et al., Acta Mater. 2023; Zenodo 7041997, CC BY 4.0.";
const waiting = (el) => { const w = h("div", { class: "xp-caption" }, "loading data …"); el.append(w); return w; };

export const LOSSES = {
  mse: { label: "MSE", color: T.gauss, rho: (r) => r * r / 2, w: () => 1 },
  mae: { label: "MAE", color: T.violet, rho: (r) => Math.abs(r), w: (r) => 1 / Math.max(Math.abs(r), 0.05) },
  huber: { label: "Huber", color: T.truth, rho: (r, d) => (Math.abs(r) <= d ? r * r / 2 : d * (Math.abs(r) - d / 2)), w: (r, d) => (Math.abs(r) <= d ? 1 : d / Math.abs(r)) },
};

// Lattice fit by iteratively reweighted least squares. p: [[x, y]], hk: [[h, k]]. Unknowns
// (cx, cy, g1x, g2x, g1y, g2y). The weight of a disk depends on the length of its 2-D residual.
export function fitLattice(p, hk, loss = "mse", delta = 1, iters = 60) {
  const n = p.length, L = LOSSES[loss];
  let wts = new Float64Array(n).fill(1), sol = null;
  for (let it = 0; it < (loss === "mse" ? 1 : iters); it++) {
    const G = new Float64Array(36), r = new Float64Array(6);
    for (let i = 0; i < n; i++) {
      const rows = [[1, 0, hk[i][0], hk[i][1], 0, 0], [0, 1, 0, 0, hk[i][0], hk[i][1]]];
      rows.forEach((x, c) => { for (let a = 0; a < 6; a++) { r[a] += wts[i] * x[a] * p[i][c]; for (let b = 0; b < 6; b++) G[a * 6 + b] += wts[i] * x[a] * x[b]; } });
    }
    sol = solve(G, r, 6);
    const res = p.map(([x, y], i) => Math.hypot(x - sol[0] - sol[2] * hk[i][0] - sol[3] * hk[i][1], y - sol[1] - sol[4] * hk[i][0] - sol[5] * hk[i][1]));
    wts = Float64Array.from(res, (v) => L.w(v, delta));
  }
  const res = p.map(([x, y], i) => [x - sol[0] - sol[2] * hk[i][0] - sol[3] * hk[i][1], y - sol[1] - sol[4] * hk[i][0] - sol[5] * hk[i][1]]);
  return { c: [sol[0], sol[1]], G: [sol[2], sol[3], sol[4], sol[5]], res, wts };
}

// Strain of a lattice G (pattern pixels, columns g1 g2 as x, y-down) against the reference Gr.
const inv2 = (m) => { const d = m[0] * m[3] - m[1] * m[2]; return [m[3] / d, -m[1] / d, -m[2] / d, m[0] / d]; };
const mul2 = (A, B) => [A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3], A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3]];
export function strainOf(G, Gr) {
  const fl = (m) => [m[0], -m[1], -m[2], m[3]];
  const A = mul2(fl(G), inv2(fl(Gr))), Ai = inv2(A), F = [Ai[0], Ai[2], Ai[1], Ai[3]];
  return { exx: F[0] - 1, eyy: F[3] - 1, exy: (F[1] + F[2]) / 2, rot: (F[2] - F[1]) / 2 };
}

// ---------------------------------------------------------------------------------------------
// robust-fit
// ---------------------------------------------------------------------------------------------
register("robust-fit", (el) => {
  const wait = waiting(el);
  loadData("disks").then((d) => {
    wait.remove();
    const Gr = [d.Gref[0][0], d.Gref[0][1], d.Gref[1][0], d.Gref[1][1]];
    const [ox, oy] = d.pat_origin, bin = d.pat_bin, PS = 560, sc = PS / 96;
    const P = (x, y) => [(x - ox) / bin * sc, (y - oy) / bin * sc], iP = (u, v) => [u / sc * bin + ox, v / sc * bin + oy];
    let ex = 0, loss = "huber", delta = 1.0, pts = null;
    const pat = imageCanvas({ n: 96, size: PS, label: "" });
    pat.classList.add("xp-loss-pat");
    const ov = s("svg", { width: PS, height: PS, viewBox: `0 0 ${PS} ${PS}`, style: "position:absolute;left:0;bottom:0;touch-action:none" });
    pat.append(ov);
    const cL = chart({ width: 560, height: 300, x: { domain: [-4, 4], label: "residual r (px)" }, y: { domain: [0, 4.2], label: "loss ρ(r)" }, margin: { l: 80 } });
    const out = readout();
    const reset = () => { const e = d.examples[ex]; pts = e.p.map((q) => q.slice()); };
    function draw() {
      const e = d.examples[ex];
      pat.draw(d.pat.subarray(ex * 9216, (ex + 1) * 9216), 0, 255, "gray");
      const fM = fitLattice(pts, e.hk, "mse"), fC = fitLattice(pts, e.hk, loss, delta);
      ov.replaceChildren();
      const lat = (f, col, dash) => {
        for (const [hh, kk] of [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1], [2, 0], [0, 2], [-2, 0], [0, -2], [2, 1], [1, 2], [-2, -1], [-1, -2], [2, -1], [-1, 2], [-2, 1], [1, -2]]) {
          const [u, v] = P(f.c[0] + f.G[0] * hh + f.G[1] * kk, f.c[1] + f.G[2] * hh + f.G[3] * kk);
          if (u < -20 || v < -20 || u > PS + 20 || v > PS + 20) continue;
          ov.append(s("circle", { cx: u, cy: v, r: (d.disk_r || 35) / bin * sc, fill: "none", stroke: col, "stroke-width": 2.5, "stroke-dasharray": dash, opacity: 0.9, "pointer-events": "none" }));
        }
      };
      if (loss !== "mse") lat(fM, T.gauss, "7 6");
      lat(fC, LOSSES[loss].color);
      pts.forEach(([x, y], i) => {
        const [u, v] = P(x, y), wi = fC.wts[i] / Math.max(...fC.wts), bad = Math.hypot(...fC.res[i]) > 2;
        const c = s("circle", { class: "disk", cx: u, cy: v, r: 9, fill: bad ? T.thread : T.warn, "fill-opacity": 0.35 + 0.65 * wi, stroke: "#fff", "stroke-width": 2 });
        c.addEventListener("pointerdown", (ev) => {
          c.setPointerCapture(ev.pointerId);
          const mv = (e2) => { const pt = ov.createSVGPoint(); pt.x = e2.clientX; pt.y = e2.clientY; const q = pt.matrixTransform(ov.getScreenCTM().inverse()); pts[i] = iP(clamp(q.x, 0, PS), clamp(q.y, 0, PS)); draw(); };
          const up = () => { c.removeEventListener("pointermove", mv); c.removeEventListener("pointerup", up); };
          c.addEventListener("pointermove", mv); c.addEventListener("pointerup", up);
        });
        ov.append(c);
      });
      pat.setLabel(`frame ${e.frame}: ${pts.length} detected disks`);
      // loss curves
      cL.clear();
      const rr = Array.from({ length: 161 }, (_, i) => -4 + i * 0.05);
      for (const [k, L] of Object.entries(LOSSES)) cL.line(rr.map((r) => [r, L.rho(r, delta)]), { stroke: L.color, width: k === loss ? 5 : 2.5, opacity: k === loss ? 1 : 0.55 });
      if (loss === "huber") { cL.ruleX(delta, { stroke: T.truth }); cL.ruleX(-delta, { stroke: T.truth }); }
      const sM = strainOf(fM.G, Gr), sC = strainOf(fC.G, Gr), worst = Math.max(...fM.res.map((r) => Math.hypot(...r)));
      const pc = (v) => fmt(100 * v, 2), dg = (v) => fmt(v * 180 / Math.PI, 2);
      out.set(`<b style="color:${T.gauss}">MSE</b>: ε<sub>xx</sub> ${pc(sM.exx)} %, ε<sub>yy</sub> ${pc(sM.eyy)} %, ε<sub>xy</sub> ${pc(sM.exy)} %, rot ${dg(sM.rot)}°<br>` +
        (loss !== "mse" ? `<b style="color:${LOSSES[loss].color}">${LOSSES[loss].label}</b>: ε<sub>xx</sub> ${pc(sC.exx)} %, ε<sub>yy</sub> ${pc(sC.eyy)} %, ε<sub>xy</sub> ${pc(sC.exy)} %, rot ${dg(sC.rot)}°<br>` +
          `fake strain from MSE: <b class="${Math.abs(sM.exx - sC.exx) + Math.abs(sM.eyy - sC.eyy) > 0.003 ? "bad" : "good"}">${pc(Math.max(Math.abs(sM.exx - sC.exx), Math.abs(sM.eyy - sC.eyy), Math.abs(sM.exy - sC.exy)))} %</b> &nbsp;(noise ≈ 0.17 %)<br>` : "") +
        `largest MSE residual: ${fmt(worst, 1)} px &nbsp;(good disks: ≈ 0.3 px)`);
    }
    const ext = toggle({ label: "frame", options: d.examples.map((e, i) => ({ value: i, label: i === d.examples.length - 1 ? "clean" : `#${e.frame}` })), value: 0, onChange: (v) => { ex = v; reset(); draw(); } });
    const lt = toggle({ label: "loss", options: Object.entries(LOSSES).map(([k, L]) => ({ value: k, label: L.label, color: L.color })), value: loss, onChange: (v) => { loss = v; draw(); } });
    const ds = slider({ label: "Huber δ", min: 0.2, max: 5, value: delta, digits: 1, unit: "px", onInput: (v) => { delta = v; draw(); } });
    el.append(controls(ext, lt, ds),
      row(col(pat, h("div", { style: { maxWidth: "560px" } }, src(SRC_MILLS + " Drag a disk to move a detection."))),
        h("div", { class: "xp-loss-side" }, cL.el, card(out),
          legend([[T.gauss, "MSE lattice", "dash"], [T.truth, "robust lattice"], [T.warn, "disk (opacity = weight)"], [T.thread, "outlier (|r| > 2 px)"]]))));
    reset(); draw();
  });
});

// ---------------------------------------------------------------------------------------------
// resid-hist
// ---------------------------------------------------------------------------------------------
register("resid-hist", (el) => {
  const wait = waiting(el);
  loadData("disks").then((d) => {
    wait.remove();
    let which = "residH", logy = true;
    const mk = (log) => chart({ width: 1040, height: 600, x: { domain: [-5, 5], label: "residual of one disk coordinate (px)" },
      y: log ? { domain: [1e-5, 3], type: "log", label: "probability density (log)" } : { domain: [0, 1.6], label: "probability density" }, margin: { l: 100 } });
    const cLog = mk(true), cLin = mk(false);
    cLog.setDomain(); cLin.setDomain();
    const host = h("div", {}, cLog.el);
    const cN = chart({ width: 560, height: 360, x: { domain: [-5, 5], label: "residual r (px)" }, y: { domain: [0, 12], label: "−log p(r)  = loss" }, margin: { l: 80 } });
    const out = readout();
    function draw() {
      const R = d[which], n = R.length, b = 0.1;
      const med = Float64Array.from(R).sort()[n >> 1];
      const ad = Float64Array.from(R, (v) => Math.abs(v - med)).sort(), sig = 1.4826 * ad[n >> 1];
      let lap = 0; for (const v of R) lap += Math.abs(v - med); lap /= n;     // Laplace scale (ML)
      const hist = new Float64Array(100);
      for (const v of R) { const k = Math.floor((v + 5) / b); if (k >= 0 && k < 100) hist[k]++; }
      const c = logy ? cLog : cLin;
      host.replaceChildren(c.el); c.clear();
      c.bars(Array.from(hist, (v, k) => [-5 + k * b, -5 + (k + 1) * b, Math.max(v / n / b, logy ? 1e-6 : 0)]), { fill: T.data, opacity: 0.55 });
      const xs = Array.from({ length: 401 }, (_, i) => -5 + i * 0.025);
      const gp = (x) => Math.exp(-0.5 * ((x - med) / sig) ** 2) / (sig * Math.sqrt(2 * Math.PI)), lp = (x) => Math.exp(-Math.abs(x - med) / lap) / (2 * lap);
      c.line(xs.map((x) => [x, Math.max(gp(x), 1e-7)]), { stroke: T.gauss, width: 3.5 });
      c.line(xs.map((x) => [x, Math.max(lp(x), 1e-7)]), { stroke: T.violet, width: 3.5, dash: "10 6" });
      const tail = (k) => R.reduce((a, v) => a + (Math.abs(v - med) > k * sig ? 1 : 0), 0) / n;
      const gTail = (k) => { let a = 0; for (let x = k; x < 12; x += 0.001) a += Math.exp(-0.5 * x * x) * 0.001; return 2 * a / Math.sqrt(2 * Math.PI); };
      // −log p for each model: the loss
      cN.clear();
      cN.line(xs.map((x) => [x, -Math.log(gp(x)) + Math.log(gp(med))]), { stroke: T.gauss, width: 3.5 });
      cN.line(xs.map((x) => [x, -Math.log(lp(x)) + Math.log(lp(med))]), { stroke: T.violet, width: 3.5, dash: "10 6" });
      out.set(`${n.toLocaleString("en")} residuals &nbsp; robust σ (from the median) = <b>${fmt(sig, 2)} px</b><br>` +
        `beyond 3σ: data <b>${fmt(100 * tail(3), 1)} %</b> &nbsp; Gaussian ${fmt(100 * gTail(3), 2)} %<br>` +
        `beyond 5σ: data <b class="bad">${fmt(100 * tail(5), 1)} %</b> &nbsp; Gaussian ${(100 * gTail(5)).toExponential(0)} %<br>` +
        `−log p of the Gaussian grows as r² (MSE); of the Laplace as |r| (MAE)`);
    }
    const wt = toggle({ label: "residuals of", options: [{ value: "residH", label: "Huber fit" }, { value: "resid", label: "MSE fit" }], value: which, onChange: (v) => { which = v; draw(); } });
    const yt = toggle({ label: "y axis", options: [{ value: "lin", label: "linear" }, { value: "log", label: "log" }], value: "log", onChange: (v) => { logy = v === "log"; draw(); } });
    el.append(controls(wt, yt), row(host, h("div", { class: "xp-loss-side", style: { width: "600px" } }, cN.el, card(out),
      legend([[T.data, "data"], [T.gauss, "Gaussian (σ robust)"], [T.violet, "Laplace", "dash"]]), src(SRC_MILLS + " Residuals of all frames, threshold 5 % of the brightest disk."))));
    draw();
  });
});
