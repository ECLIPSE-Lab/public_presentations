// PCA group (Act 4) and the residual test of Act 6, on real data:
//   pca-cloud : two edge signals per pixel (EELS V-L, Ce-M); the variance along a direction v; PC1 = max
//   eds-pca   : PCA of the CeVO3 EDS spectrum image (80 x 80 x 250, ≈ 58 counts per pixel)
//               data-mode = "components" | "denoise" | "weight" | "residual"
//   drift     : a ZLP spectrum image with real energy drift: PC1 = the derivative of the peak
import { register, T, fmt, clamp, h, s, row, col, card, controls, readout, slider, toggle, button, legend, chart, imageCanvas } from "./core.js";
import { loadData } from "./data.js";
import { eelsCube, prange, SRC_CEVO3, SRC_CYMES } from "./w-vector.js";

if (!document.getElementById("xp-pca-style")) {
  document.head.append(h("style", { id: "xp-pca-style" }, `
.xp-pca-side { display: flex; flex-direction: column; gap: 12px; }
.xp-pca-side .xp-readout { margin-top: 0; line-height: 1.5; }
.xp-click canvas { cursor: crosshair; }
.xp-src2 { font-size: 17px; color: #64748b; }
`));
}
const src = (html) => h("div", { class: "xp-src2", html });
const waiting = (el) => { const w = h("div", { class: "xp-caption" }, "loading data …"); el.append(w); return w; };
const sym = (a) => { const [lo, hi] = prange(a, 0.5); const m = Math.max(Math.abs(lo), Math.abs(hi)); return [-m, m]; };

// clickable image with a marker; onPick(index)
function pickImage(nx, ny, size, label, onPick) {
  const im = imageCanvas({ n: nx, ny, size, label });
  const mark = s("svg", { width: size, height: size * ny / nx, style: "position:absolute;left:0;top:0;pointer-events:none" });
  const stack = h("div", { style: { position: "relative" } });
  im.insertBefore(stack, im.canvas); stack.append(im.canvas, mark);
  im.classList.add("xp-click");
  im.mark = (i, color = "#fff") => {
    if (i < 0) { mark.replaceChildren(); return; }
    const sz = size / nx, y = Math.floor(i / nx), x = i % nx;
    mark.replaceChildren(s("rect", { x: x * sz - 3, y: y * sz - 3, width: sz + 6, height: sz + 6, fill: "none", stroke: color, "stroke-width": 3 }));
  };
  im.overlay = (els) => mark.append(...els);
  if (onPick) im.canvas.addEventListener("pointerdown", (e) => {
    const r = im.canvas.getBoundingClientRect();
    const x = clamp(Math.floor((e.clientX - r.left) / r.width * nx), 0, nx - 1), y = clamp(Math.floor((e.clientY - r.top) / r.height * ny), 0, ny - 1);
    onPick(y * nx + x);
  });
  return im;
}

// ---------------------------------------------------------------------------------------------
// pca-cloud. Each pixel is a point (V-L signal, Ce-M signal). Turn the direction v: the variance of
// the projections vᵀx traces a curve over the angle; its maximum is PC1, the minimum PC2 (⊥ PC1).
// The projections themselves form a map: the PC score map.
// ---------------------------------------------------------------------------------------------
register("pca-cloud", (el) => {
  const wait = waiting(el);
  loadData("cevo3_eels").then((d) => {
    wait.remove();
    const cu = eelsCube(d), { nx, ny } = cu;
    const a = cu.edgeMap(512, 530, [478, 508]), b = cu.edgeMap(878, 910, [840, 875]);
    const idx = [...a.keys()].filter((i) => !cu.vacuum[i]);
    let ang = 20, stdz = true;
    const cP = chart({ width: 700, height: 640, x: { domain: [-3, 3], label: "V-L₂,₃ signal" }, y: { domain: [-3, 3], label: "Ce-M₄,₅ signal" } });
    const cV = chart({ width: 620, height: 300, x: { domain: [0, 180], label: "direction of v (degrees)" }, y: { domain: [0, 1], label: "Var(vᵀx)" } });
    const im = pickImage(nx, ny, 300, "score map vᵀx");
    const out = readout();
    function prep() {                                            // centre; standardise each axis, or one common scale
      const mean = (f) => idx.reduce((p, i) => p + f[i], 0) / idx.length;
      const sd = (f, m) => Math.sqrt(idx.reduce((p, i) => p + (f[i] - m) ** 2, 0) / idx.length);
      const ma = mean(a), mb = mean(b), sa = sd(a, ma), sb = sd(b, mb), c = Math.max(sa, sb);
      const [ka, kb] = stdz ? [sa, sb] : [c, c];
      return idx.map((i) => [(a[i] - ma) / ka, (b[i] - mb) / kb]);
    }
    function update() {
      const P = prep(), n = P.length;
      let sxx = 0, syy = 0, sxy = 0;
      for (const [x, y] of P) { sxx += x * x; syy += y * y; sxy += x * y; }
      sxx /= n - 1; syy /= n - 1; sxy /= n - 1;
      const tr = sxx + syy, det = sxx * syy - sxy * sxy, l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr / 2 - Math.sqrt(tr * tr / 4 - det);
      const pc = Math.atan2(l1 - sxx, sxy) * 180 / Math.PI, pc1 = ((pc % 180) + 180) % 180;
      const varAt = (deg) => { const t = deg * Math.PI / 180, c = Math.cos(t), sn = Math.sin(t); return c * c * sxx + 2 * c * sn * sxy + sn * sn * syy; };
      const R = 3.2;
      cP.setDomain([-R, R], [-R, R]); cP.clear();
      cP.dots(P, { r: 3.5, fill: T.data, opacity: 0.45 });
      const t = ang * Math.PI / 180, v = [Math.cos(t), Math.sin(t)];
      // ellipse (2 standard deviations) and principal axes
      const tp = pc1 * Math.PI / 180, e1 = [Math.cos(tp), Math.sin(tp)], e2 = [-Math.sin(tp), Math.cos(tp)];
      cP.line(Array.from({ length: 101 }, (_, k) => { const q = k / 100 * 2 * Math.PI; return [2 * Math.sqrt(l1) * Math.cos(q) * e1[0] + 2 * Math.sqrt(l2) * Math.sin(q) * e2[0], 2 * Math.sqrt(l1) * Math.cos(q) * e1[1] + 2 * Math.sqrt(l2) * Math.sin(q) * e2[1]]; }), { stroke: T.warn, width: 2, dash: "6 6" });
      cP.line([[-R * v[0], -R * v[1]], [R * v[0], R * v[1]]], { stroke: T.thread, width: 4 });
      cP.line([[0, 0], [2 * Math.sqrt(l1) * e1[0], 2 * Math.sqrt(l1) * e1[1]]], { stroke: T.truth, width: 3, dash: "3 5" });
      cP.text(R * v[0] * 0.9, R * v[1] * 0.9, "v", { fill: T.thread, size: 30, weight: 700, dx: 10 });
      const curve = Array.from({ length: 181 }, (_, k) => [k, varAt(k)]);
      cV.setDomain([0, 180], [0, l1 * 1.15]); cV.clear();
      cV.line(curve, { stroke: T.gauss, width: 3 });
      cV.ruleX(pc1, { stroke: T.truth }); cV.ruleX((pc1 + 90) % 180, { stroke: T.faint });
      cV.dots([[ang, varAt(ang)]], { r: 9, fill: T.thread });
      cV.text(pc1, l1 * 1.08, "PC1", { fill: T.truth, anchor: "middle", size: 20, weight: 700 });
      // score map
      const m = new Float64Array(nx * ny);
      idx.forEach((i, k) => { m[i] = P[k][0] * v[0] + P[k][1] * v[1]; });
      const [lo, hi] = sym(m);
      im.draw(m, lo, hi, "div");
      out.set(`Var(vᵀx) at ${fmt(ang, 0)}° = <b>${fmt(varAt(ang), 2)}</b><br>` +
        `PC1 at ${fmt(pc1, 0)}°: λ₁ = <span class="good">${fmt(l1, 2)}</span> &nbsp; PC2: λ₂ = ${fmt(l2, 2)}<br>` +
        `PC1 carries <b>${fmt(100 * l1 / (l1 + l2), 0)} %</b> of the variance`);
    }
    const sl = slider({ label: "direction of v", min: 0, max: 180, value: ang, digits: 0, unit: "°", onInput: (v) => { ang = v; update(); } });
    const st = toggle({ label: "axes", options: [{ value: "z", label: "standardised" }, { value: "raw", label: "raw units" }], value: "z", onChange: (v) => { stdz = v === "z"; update(); } });
    el.append(controls(sl, st), row(cP.el, h("div", { class: "xp-pca-side" }, cV.el, row(im, card(out)),
      legend([[T.thread, "direction v"], [T.truth, "PC1", "dash"], [T.warn, "2σ ellipse", "dash"]]), src(SRC_CEVO3 + " One point = one pixel (vacuum masked)."))));
    update();
  });
});

// ---------------------------------------------------------------------------------------------
// eds-pca. Real EDS spectrum image with Poisson counts. PCA through the precomputed SVD (12 scores
// and loadings kept), either of the raw counts or of the Poisson-weighted counts X / (√g √h).
// ---------------------------------------------------------------------------------------------
const LINES = {
  Al: { label: "Al-K", lo: 1.44, hi: 1.54 },
  V: { label: "V-Kα", lo: 4.90, hi: 5.00 },
  Ce: { label: "Ce-Lβ", lo: 5.22, hi: 5.30 },
  Cu: { label: "Cu-Kα (grid)", lo: 8.0, hi: 8.1 },
};

function edsModel(d) {
  const [ny, nx, D] = d.cube.shape, N = ny * nx, E = Array.from(d.E), C = d.cube;
  function variant(which) {
    const p = which === "raw" ? d.raw : which === "wtd" ? d.wtd : d.planted;
    const K = p.V.shape[0], G = which === "raw" ? null : which === "wtd" ? d.G : p.G, H = which === "raw" ? null : which === "wtd" ? d.H : p.H;
    const sv = Array.from(p.sv), tot = sv.reduce((a, b) => a + b * b, 0);
    const data = (i, j) => {
      let x = C[i * D + j];
      if (which === "pt") { const k = ptIndex.get(i); if (k !== undefined) x += d.planted.add[k * D + j]; }
      return x;
    };
    const ptIndex = new Map(which === "pt" ? Array.from(d.planted.idx).map((i, k) => [i, k]) : []);
    // reconstruction of pixel i with K components, in counts
    const recon = (i, Kc, out = new Float64Array(D)) => {
      for (let j = 0; j < D; j++) {
        let v = p.mean[j];
        for (let k = 0; k < Kc; k++) v += p.T[i * K + k] * p.V[k * D + j];
        out[j] = G ? v * G[i] * H[j] : v;
      }
      return out;
    };
    const pearson = (Kc) => {                                      // Σ (x − x̂)² / x̂ / D per pixel
      const m = new Float64Array(N), r = new Float64Array(D);
      for (let i = 0; i < N; i++) {
        recon(i, Kc, r);
        let a = 0;
        for (let j = 0; j < D; j++) { const mu = Math.max(r[j], 0.05), x = data(i, j); a += (x - mu) ** 2 / mu; }
        m[i] = a / D;
      }
      return m;
    };
    return { K, sv, frac: sv.map((v) => v * v / tot), recon, pearson, data, score: (k) => Float64Array.from({ length: N }, (_, i) => p.T[i * K + k]),
      loading: (k) => Float64Array.from({ length: D }, (_, j) => p.V[k * D + j] * (H ? H[j] : 1)) };
  }
  const lineMap = (lo, hi, f) => {
    const j0 = E.findIndex((e) => e >= lo), j1x = E.findIndex((e) => e >= hi), j1 = j1x < 0 ? D : j1x, m = new Float64Array(N);
    for (let i = 0; i < N; i++) { let a = 0; for (let j = j0; j < j1; j++) a += f(i, j); m[i] = a; }
    return m;
  };
  const total = lineMap(E[0], E[D - 1] + 1, (i, j) => C[i * D + j]);
  return { ny, nx, D, N, E, C, variant, lineMap, total };
}

register("eds-pca", (el, api) => {
  const wait = waiting(el);
  const mode = api.data.mode || "denoise";
  loadData("cevo3_eds").then((d) => {
    wait.remove();
    const M = edsModel(d), { nx, ny, D, E } = M;
    let which = mode === "residual" ? "pt" : mode === "weight" ? "wtd" : "raw", Kc = 4, comp = 1, pix = 20 * nx + 40, line = "Ce", showTruth = false;
    let V = M.variant(which);
    const cScree = chart({ width: 620, height: 340, x: { domain: [0.7, 60], type: "log", label: "component k" }, y: { domain: [1e-3, 1], type: "log", label: "variance fraction" }, title: "scree" });
    const cSpec = chart({ width: 1000, height: 340, x: { domain: [0.2, 10.2], label: "X-ray energy (keV)" }, y: { domain: [0, 1] } });
    const big = mode === "residual" ? 430 : 380;
    const imA = pickImage(nx, ny, big, " ", (i) => { pix = i; update(); });
    const imB = pickImage(nx, ny, big, " ", (i) => { pix = i; update(); });
    const out = readout();

    function scree() {
      cScree.clear();
      const n = 60;
      if (mode === "weight") {
        const R = M.variant("raw"), W = M.variant("wtd");
        cScree.line(R.frac.slice(0, n).map((v, k) => [k + 1, v]), { stroke: T.gauss, width: 3 });
        cScree.dots(R.frac.slice(0, n).map((v, k) => [k + 1, v]), { r: 4, fill: T.gauss });
        cScree.line(W.frac.slice(0, n).map((v, k) => [k + 1, v]), { stroke: T.pois, width: 3 });
        cScree.dots(W.frac.slice(0, n).map((v, k) => [k + 1, v]), { r: 4, fill: T.pois });
        const floor = W.frac.slice(20, 60).reduce((a, b) => a + b, 0) / 40;
        cScree.ruleY(floor, { stroke: T.pois, dash: "4 6" });
        cScree.note("raw counts", { corner: "tr", fill: T.gauss }); cScree.note("Poisson-weighted", { corner: "tr", line: 1, fill: T.pois });
      } else {
        cScree.line(V.frac.slice(0, n).map((v, k) => [k + 1, v]), { stroke: T.data, width: 3 });
        cScree.dots(V.frac.slice(0, n).map((v, k) => [k + 1, v]), { r: 4.5, fill: T.data });
        cScree.dots(V.frac.slice(0, Kc).map((v, k) => [k + 1, v]), { r: 7, fill: T.thread });
        cScree.ruleX(Kc + 0.5, { stroke: T.thread });
      }
    }

    function update() {
      scree();
      const [lo0, hi0] = prange(M.total, 0.5);
      if (mode === "components" || mode === "weight") {
        const k = comp - 1, sc = V.score(k), ld = V.loading(k);
        const [a0, a1] = sym(sc);
        imA.draw(sc, a0, a1, "div"); imA.setLabel(`score map of PC${comp}`);
        imB.draw(M.total, lo0, hi0, "gray"); imB.setLabel("total counts per pixel");
        const lmax = Math.max(...ld.map(Math.abs));
        cSpec.setDomain(null, [-1.1 * lmax, 1.1 * lmax]); cSpec.clear();
        cSpec.ruleY(0, { stroke: T.faint });
        cSpec.line(E.map((e, j) => [e, ld[j]]), { stroke: comp <= Kc ? T.thread : T.data, width: 2.5 });
        cSpec.note(`loading (eigenspectrum) of PC${comp}${mode === "weight" ? (which === "raw" ? " · raw" : " · weighted, un-weighted for display") : ""}`, { corner: "tl" });
        for (const L of Object.values(LINES)) cSpec.text((L.lo + L.hi) / 2, -0.95 * lmax, L.label.split("-")[0], { fill: T.muted, size: 18, anchor: "middle" });
        out.set(`PC${comp}: <b>${fmt(100 * V.frac[k], 2)} %</b> of the variance &nbsp; (noise floor ≈ ${fmt(100 * V.frac[30], 2)} %)<br>` +
          `scores and loadings are signed: <span class="warn">blue / red</span> = below / above the mean spectrum`);
      } else if (mode === "denoise") {
        const L = LINES[line];
        const raw = M.lineMap(L.lo, L.hi, (i, j) => M.C[i * D + j]);
        const den = (() => { const m = new Float64Array(M.N), r = new Float64Array(D), j0 = E.findIndex((e) => e >= L.lo), j1 = E.findIndex((e) => e >= L.hi);
          for (let i = 0; i < M.N; i++) { V.recon(i, Kc, r); let a = 0; for (let j = j0; j < j1; j++) a += r[j]; m[i] = a; } return m; })();
        const [l0, l1] = prange(den, 0.5);
        imA.draw(raw, l0, l1, "magma"); imA.setLabel(`${L.label}: raw counts`);
        imB.draw(den, l0, l1, "magma"); imB.setLabel(`${L.label}: PCA, K = ${Kc}`);
        imA.mark(pix); imB.mark(pix);
        const x = Float64Array.from({ length: D }, (_, j) => M.C[pix * D + j]), r = V.recon(pix, Kc);
        const ymax = Math.max(4, ...x) * 1.1;
        cSpec.setDomain(null, [-0.5, ymax]); cSpec.clear();
        cSpec.rect(L.lo, -0.5, L.hi, ymax, { fill: T.warn, opacity: 0.15 });
        cSpec.bars(E.map((e, j) => [e - 0.02, e + 0.02, x[j]]), { fill: T.data, opacity: 0.6 });
        cSpec.line(E.map((e, j) => [e, r[j]]), { stroke: T.thread, width: 3 });
        cSpec.note(`pixel (${Math.floor(pix / nx)}, ${pix % nx}): ${x.reduce((a, b) => a + b, 0)} counts in 250 channels`, { corner: "tr" });
        out.set(`one pixel has ≈ 58 counts: most channels are 0 or 1 (grey bars)<br>` +
          `K = ${Kc} components ${which === "raw" ? "(raw PCA)" : "(weighted PCA)"} pool the counts of all 6 400 pixels (red line)`);
      } else if (mode === "residual") {
        const P = V.pearson(Kc), [r0, r1] = [0.3, 2.2];
        imA.draw(P, r0, r1, "magma"); imA.setLabel(`Pearson residual Σ(x − x̂)²/x̂ / D at K = ${Kc}`);
        imB.draw(M.total, lo0, hi0, "gray"); imB.setLabel("total counts (what you see)");
        const pts = Array.from(d.planted.idx), ptm = pts.reduce((a, i) => a + P[i], 0) / pts.length;
        const med = Float64Array.from(P).sort()[P.length >> 1];
        imA.mark(-1);
        if (showTruth) {
          const sz = big / nx;
          imA.overlay([[30, 25], [55, 45], [40, 62]].map(([y, x]) => s("circle", { cx: (x + 1) * sz, cy: (y + 1) * sz, r: 3 * sz, fill: "none", stroke: T.truth, "stroke-width": 3 })));
        }
        cSpec.clear();
        const ptSpec = new Float64Array(D); pts.forEach((i) => { for (let j = 0; j < D; j++) ptSpec[j] += (V.data(i, j) - V.recon(i, Kc)[j]) / pts.length; });
        const ymax = Math.max(1, ...ptSpec.map(Math.abs)) * 1.2;
        cSpec.setDomain(null, [-ymax, ymax]); cSpec.ruleY(0, { stroke: T.faint });
        if (showTruth) cSpec.line(E.map((e, j) => [e, ptSpec[j]]), { stroke: T.truth, width: 2.5 });
        cSpec.note(showTruth ? "mean residual spectrum of the 12 planted pixels: Pt M (2.05 keV), Pt L (9.44 keV)" : "where is the hidden phase? Look at the residual map, then press reveal", { corner: "tl" });
        out.set(`scree after K = 4: looks like the noise floor<br>` +
          `planted pixels: residual <span class="${ptm > 1.4 * med ? "bad" : "good"}">${fmt(ptm, 2)}</span> vs median ${fmt(med, 2)} &nbsp; (pure Poisson noise: ≈ 1 at high counts)`);
      }
    }
    const Ks = slider({ label: "keep K", min: 1, max: 12, value: Kc, step: 1, digits: 0, onInput: (v) => { Kc = Math.round(v); update(); } });
    const Cs = slider({ label: "component", min: 1, max: 8, value: comp, step: 1, digits: 0, onInput: (v) => { comp = Math.round(v); update(); } });
    const Wt = toggle({ label: "PCA of", options: [{ value: "raw", label: "raw counts" }, { value: "wtd", label: "Poisson-weighted" }], value: which,
      onChange: (v) => { which = v; V = M.variant(v); update(); } });
    const Lt = toggle({ label: "map", options: Object.entries(LINES).map(([k, L]) => ({ value: k, label: L.label })), value: line, onChange: (v) => { line = v; update(); } });
    const reveal = button("reveal the planted truth", () => { showTruth = !showTruth; update(); });
    const ctl = mode === "components" ? [Cs, Wt] : mode === "weight" ? [Wt, Cs] : mode === "denoise" ? [Ks, Wt, Lt] : [Ks, reveal];
    const note = mode === "residual" ? "Planted truth: 12 pixels (0.2 %) of the real data received ≈ 10 extra Pt counts each (Poisson). " : "";
    el.append(controls(...ctl),
      row(col(cScree.el, card(out)), col(cSpec.el, row(imA, imB))),
      src(note + SRC_CEVO3 + " EDS: 80 × 80 pixels × 250 channels (40 eV)."));
    update();
  });
});

// ---------------------------------------------------------------------------------------------
// drift. Real ZLP spectrum image (81 x 60, binned 2 x 2) with an energy drift of ±0.5 eV over the
// scan. I(E − δ) ≈ I(E) − δ I'(E): PCA of the raw stack finds −I'(E) as PC1, and the PC1 score map
// is the drift map. After alignment, that component is gone.
// ---------------------------------------------------------------------------------------------
register("drift", (el) => {
  const wait = waiting(el);
  loadData("zlp").then((d) => {
    wait.remove();
    const [ny, nx] = d.shape, E = Array.from(d.E);
    let which = "raw";
    const imS = imageCanvas({ n: nx, ny, size: 250, label: "measured ZLP shift δ" });
    const imT = imageCanvas({ n: nx, ny, size: 250, label: "PC1 score map" });
    const cL = chart({ width: 960, height: 440, x: { domain: [-2.5, 3], label: "energy loss (eV)" }, y: { domain: [-1, 1], label: "PC1 loading" } });
    const cS = chart({ width: 560, height: 330, x: { domain: [0.7, 20], type: "log", label: "component k" }, y: { domain: [1e-6, 1], type: "log", label: "σₖ² / total as recorded" }, title: "scree" });
    const out = readout();
    const sh = d.shift, [s0, s1] = sym(sh);
    imS.draw(sh, s0, s1, "div");
    function update() {
      const p = d[which], v = Array.from(p.V.slice(0, E.length)), dm = Array.from(d.dmean);
      const nv = Math.hypot(...v), nd = Math.hypot(...dm), cos = v.reduce((a, b, j) => a + b * dm[j], 0) / (nv * nd);
      const T1 = p.T1, [t0, t1] = sym(T1);
      imT.draw(T1, t0, t1, "div");
      cL.setDomain(null, [-1.15 * Math.max(...v.map(Math.abs)), 1.15 * Math.max(...v.map(Math.abs))]); cL.clear();
      cL.ruleY(0, { stroke: T.faint });
      const fitc = v.reduce((a, b, j) => a + b * dm[j], 0) / (nd * nd);   // best multiple of dI/dE
      if (which === "raw") cL.line(E.map((e, j) => [e, fitc * dm[j]]), { stroke: T.truth, width: 3, dash: "8 6" });
      cL.line(E.map((e, j) => [e, v[j]]), { stroke: T.thread, width: 3 });
      const mean = Array.from(d.mean), mm = Math.max(...mean);
      cL.line(E.map((e, j) => [e, mean[j] / mm * 0.9 * Math.max(...v.map(Math.abs))]), { stroke: T.faint, width: 2 });
      const sv = Array.from(p.sv), tot = sv.reduce((a, b) => a + b * b, 0);
      const svr = Array.from(d.raw.sv), totr = svr.reduce((a, b) => a + b * b, 0);
      cS.setDomain(null, [1e-6, 1]); cS.clear();                // both in units of the variance as recorded
      cS.line(svr.map((x, k) => [k + 1, x * x / totr]), { stroke: T.gauss, width: 3, opacity: which === "raw" ? 1 : 0.4 });
      cS.dots(svr.map((x, k) => [k + 1, x * x / totr]), { r: 4, fill: T.gauss, opacity: which === "raw" ? 1 : 0.4 });
      if (which === "aligned") {
        cS.line(sv.map((x, k) => [k + 1, x * x / totr]), { stroke: T.truth, width: 3 });
        cS.dots(sv.map((x, k) => [k + 1, x * x / totr]), { r: 5, fill: T.truth });
      }
      cS.note("as recorded", { corner: "tr", fill: T.gauss });
      if (which === "aligned") cS.note("aligned", { corner: "tr", line: 1, fill: T.truth });
      let r = 0; { const a = Array.from(T1), b = Array.from(sh), ma = a.reduce((p_, q) => p_ + q, 0) / a.length, mb = b.reduce((p_, q) => p_ + q, 0) / b.length;
        r = a.reduce((p_, q, i) => p_ + (q - ma) * (b[i] - mb), 0) / Math.sqrt(a.reduce((p_, q) => p_ + (q - ma) ** 2, 0) * b.reduce((p_, q) => p_ + (q - mb) ** 2, 0)); }
      out.set(`cos(PC1 loading, dI/dE) = <b>${fmt(Math.abs(cos), 2)}</b> &nbsp; corr(PC1 map, shift map) = <span class="${Math.abs(r) > 0.8 ? "bad" : "good"}">${fmt(r, 2)}</span><br>` +
        `total variance: <b>${which === "raw" ? "100 %" : fmt(100 * tot / totr, 2) + " %"}</b> of the recorded stack` +
        (which === "raw" ? `; PC1 alone: <b>${fmt(100 * sv[0] ** 2 / tot, 0)} %</b>` : ` (alignment removed ${fmt(totr / tot, 0)}×)`));
    }
    const tg = toggle({ label: "stack", options: [{ value: "raw", label: "as recorded" }, { value: "aligned", label: "aligned to the ZLP" }], value: which, onChange: (v) => { which = v; update(); } });
    el.append(controls(tg), row(col(cL.el, legend([[T.thread, "PC1 loading"], [T.truth, "c · dI/dE (best multiple)", "dash"], [T.faint, "mean ZLP (scaled)"]]), card(out)),
      h("div", { class: "xp-pca-side" }, row(imS, imT), cS.el)), src(SRC_CYMES + " ZLP spectrum image, 81 × 60 pixels (2 × 2 binned), 0.05 eV channels."));
    update();
  });
});
