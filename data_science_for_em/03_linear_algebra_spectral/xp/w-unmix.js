// Unmix group (Act 5) and rotational ambiguity (Act 6), on the real CeVO3 / Al2O3 EDS data:
//   simplex  : pixels as barycentric coordinates of three endmember spectra (from KL-NMF)
//              data-mode = "mixing" (drag a point: mixture spectrum) | "ambiguity" (enlarge the triangle)
//   nmf-live : multiplicative-update NMF running in the browser (Frobenius or KL loss)
//   sped-nmf : the same factorisation on 4D-STEM / SPED diffraction patterns of a twinned nanowire
import { register, T, fmt, clamp, h, s, row, col, card, controls, readout, slider, toggle, button, legend, chart, imageCanvas } from "./core.js";
import { loadData } from "./data.js";
import { nmfInit, nmfStep } from "./la.js";
import { SRC_CEVO3 } from "./w-vector.js";

if (!document.getElementById("xp-unmix-style")) {
  document.head.append(h("style", { id: "xp-unmix-style" }, `
.xp-unmix-side { display: flex; flex-direction: column; gap: 12px; }
.xp-unmix-side .xp-readout { margin-top: 0; line-height: 1.5; }
.xp-src3 { font-size: 17px; color: #64748b; }
.xp-sped-card { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 10px 12px; border-radius: 16px;
  background: rgba(20,25,40,0.72); border: 2px solid transparent; cursor: pointer; }
.xp-sped-card.on { border-color: #f43f5e; }
`));
}
const src = (html) => h("div", { class: "xp-src3", html });
const waiting = (el) => { const w = h("div", { class: "xp-caption" }, "loading data …"); el.append(w); return w; };
const PH = [{ name: "C film + Cu grid", c: T.violet }, { name: "Al₂O₃", c: T.gauss }, { name: "CeVOₓ", c: T.pois }];

// 2 x 2 binned counts (40 x 40 x D) from the 80 x 80 cube
function binned(d) {
  const [ny, nx, D] = d.cube.shape, n2 = (ny / 2) * (nx / 2), X = new Float64Array(n2 * D);
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const o = ((y >> 1) * (nx / 2) + (x >> 1)) * D, i = (y * nx + x) * D;
    for (let j = 0; j < D; j++) X[o + j] += d.cube[i + j];
  }
  return { X, N: n2, D, ny: ny / 2, nx: nx / 2 };
}

// ---------------------------------------------------------------------------------------------
// simplex. Every pixel spectrum, normalised to unit sum, is fitted as y ≈ Σ a_k m_k with Σ a_k = 1
// (least squares on the affine hull of the three endmembers m_k). The a_k are barycentric
// coordinates: the pixel is a point in a triangle whose corners are the pure endmembers.
// Ambiguity: the triangle with corners c + s (m_k − c) explains every pixel exactly as well, with
// abundances a'_k = (a_k − 1/3)/s + 1/3. Small s: abundances < 0. Large s: spectra < 0.
// ---------------------------------------------------------------------------------------------
register("simplex", (el, api) => {
  const wait = waiting(el);
  const mode = api.data.mode || "mixing";
  loadData("cevo3_eds").then((d) => {
    wait.remove();
    const { X, N, D, ny, nx } = binned(d), E = Array.from(d.E), M = d.simplex.M;
    const m = [0, 1, 2].map((k) => Float64Array.from({ length: D }, (_, j) => M[k * D + j]));
    // least squares with a3 = 1 − a1 − a2: y − m3 ≈ a1 (m1 − m3) + a2 (m2 − m3)
    const d1 = m[0].map((v, j) => v - m[2][j]), d2 = m[1].map((v, j) => v - m[2][j]);
    let g11 = 0, g12 = 0, g22 = 0;
    for (let j = 0; j < D; j++) { g11 += d1[j] * d1[j]; g12 += d1[j] * d2[j]; g22 += d2[j] * d2[j]; }
    const det = g11 * g22 - g12 * g12;
    const A = new Float64Array(N * 3), Y = new Float64Array(N * D), tot = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let t = 0; for (let j = 0; j < D; j++) t += X[i * D + j];
      tot[i] = t;
      let b1 = 0, b2 = 0;
      for (let j = 0; j < D; j++) { const y = X[i * D + j] / t - m[2][j]; Y[i * D + j] = X[i * D + j] / t; b1 += d1[j] * y; b2 += d2[j] * y; }
      const a1 = (g22 * b1 - g12 * b2) / det, a2 = (g11 * b2 - g12 * b1) / det;
      A[i * 3] = a1; A[i * 3 + 1] = a2; A[i * 3 + 2] = 1 - a1 - a2;
    }
    // fit error of the affine model (the same for every enclosing triangle)
    let err = 0, nrm = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < D; j++) {
      const f = A[i * 3] * m[0][j] + A[i * 3 + 1] * m[1][j] + A[i * 3 + 2] * m[2][j], y = Y[i * D + j];
      err += (y - f) ** 2; nrm += y * y;
    }
    const relErr = Math.sqrt(err / nrm);

    let sc = 1, pt = [0.3, 0.3, 0.4], hidePure = false;
    const PURE = 0.2;                                           // "no pure pixels": keep pixels with all a_k > PURE
    const keep = (i) => tot[i] > 60 && (!hidePure || (A[i * 3] > PURE && A[i * 3 + 1] > PURE && A[i * 3 + 2] > PURE));
    const cbar = E.map((_, j) => (m[0][j] + m[1][j] + m[2][j]) / 3);
    const minSpec = (sv) => { let mn = Infinity; for (let k = 0; k < 3; k++) for (let j = 0; j < D; j++) mn = Math.min(mn, cbar[j] + sv * (m[k][j] - cbar[j])); return mn; };
    const negFrac = (sv) => { let n = 0, c = 0; for (let i = 0; i < N; i++) if (keep(i)) { c++; for (let k = 0; k < 3; k++) if ((A[i * 3 + k] - 1 / 3) / sv + 1 / 3 < -0.1) { n++; break; } } return n / Math.max(1, c); };
    const TOL_S = -0.003, TOL_A = 0.05;                          // noise tolerances: 1 % of the largest peak; 5 % of the pixels
    function band() {
      let lo = null, hi = null;
      for (let sv = 0.3; sv <= 2.2001; sv += 0.01) if (negFrac(sv) <= TOL_A && minSpec(sv) >= TOL_S) { if (lo === null) lo = sv; hi = sv; }
      return [lo, hi];
    }
    let B = band();
    const W = 620, H = 560, cx = W / 2, R = 240, cy = 310;
    const VX = [[cx, cy - R], [cx - R * Math.sin(Math.PI / 3) * 1.0, cy + R / 2], [cx + R * Math.sin(Math.PI / 3), cy + R / 2]];
    const toXY = (a) => [a[0] * VX[0][0] + a[1] * VX[1][0] + a[2] * VX[2][0], a[0] * VX[0][1] + a[1] * VX[1][1] + a[2] * VX[2][1]];
    const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "xp-chart", style: "overflow:hidden;border-radius:16px;background:rgba(5,8,15,0.55);touch-action:none" });
    const gPts = s("g"), gTri = s("g"), gTop = s("g");
    svg.append(gPts, gTri, gTop);
    const dots = [];
    for (let i = 0; i < N; i++) {
      const [x, y] = toXY([A[i * 3], A[i * 3 + 1], A[i * 3 + 2]]);
      dots.push(gPts.appendChild(s("circle", { cx: x, cy: y, r: 2.6, fill: T.data, opacity: 0.45 })));
    }
    const cS = chart({ width: 1000, height: 330, x: { domain: [0.2, 10.2], label: "X-ray energy (keV)" }, y: { domain: [-0.01, 0.08], label: "fraction of counts" } });
    const ims = [0, 1, 2].map((k) => imageCanvas({ n: nx, ny, size: 205, label: PH[k].name }));
    const out = readout();

    function abund(i, k) { return (A[i * 3 + k] - 1 / 3) / sc + 1 / 3; }
    function update() {
      // triangle of the current endmembers (scaled about the centroid)
      const C = [cx, (VX[0][1] + VX[1][1] + VX[2][1]) / 3], Vs = VX.map(([x, y]) => [C[0] + sc * (x - C[0]), C[1] + sc * (y - C[1])]);
      dots.forEach((dt, i) => dt.setAttribute("opacity", keep(i) || tot[i] <= 60 && !hidePure ? 0.45 : 0.06));
      const tri = (f) => VX.map(([x, y]) => [C[0] + f * (x - C[0]), C[1] + f * (y - C[1])]);
      const bandEls = mode === "ambiguity" && B[0] !== null ? [s("path", { d: `M${tri(B[1]).map((p) => p.join(",")).join("L")}Z M${tri(B[0]).map((p) => p.join(",")).join("L")}Z`,
        fill: "rgba(52,211,153,0.10)", "fill-rule": "evenodd", stroke: T.truth, "stroke-width": 1.5, "stroke-dasharray": "4 6" })] : [];
      gTri.replaceChildren(...bandEls,
        s("path", { d: `M${VX.map((p) => p.join(",")).join("L")}Z`, fill: "none", stroke: T.faint, "stroke-width": 2, "stroke-dasharray": "6 6" }),
        s("path", { d: `M${Vs.map((p) => p.join(",")).join("L")}Z`, fill: "rgba(244,63,94,0.06)", stroke: T.thread, "stroke-width": 3 }),
        ...Vs.map(([x, y], k) => s("circle", { cx: x, cy: y, r: 12, fill: PH[k].c, stroke: "#fff", "stroke-width": 2 })),
        ...Vs.map(([x, y], k) => s("text", { x: x + (k === 0 ? 18 : k === 1 ? -14 : 14), y: y + (k === 0 ? 6 : 34), fill: PH[k].c, "font-size": 22, "font-weight": 700,
          "text-anchor": k === 1 ? "end" : "start", class: "xp-ctext" }, PH[k].name)));
      // endmember spectra
      const ms = m.map((mk) => mk.map((v, j) => (m[0][j] + m[1][j] + m[2][j]) / 3 + sc * (v - (m[0][j] + m[1][j] + m[2][j]) / 3)));
      cS.clear(); cS.ruleY(0, { stroke: T.faint });
      let minS = Infinity;
      if (mode === "mixing") {
        const mix = E.map((_, j) => pt[0] * ms[0][j] + pt[1] * ms[1][j] + pt[2] * ms[2][j]);
        ms.forEach((mk, k) => cS.line(E.map((e, j) => [e, mk[j]]), { stroke: PH[k].c, width: 2, opacity: 0.45 }));
        cS.line(E.map((e, j) => [e, mix[j]]), { stroke: "#fff", width: 3 });
        // nearest real pixel
        let best = 0, bd = Infinity;
        for (let i = 0; i < N; i++) { const dd = (A[i * 3] - pt[0]) ** 2 + (A[i * 3 + 1] - pt[1]) ** 2 + (A[i * 3 + 2] - pt[2]) ** 2; if (dd < bd && tot[i] > 150) { bd = dd; best = i; } }
        cS.bars(E.map((e, j) => [e - 0.02, e + 0.02, Y[best * D + j]]), { fill: T.warn, opacity: 0.5 });
        gTop.replaceChildren(s("circle", { cx: toXY(pt)[0], cy: toXY(pt)[1], r: 13, fill: "#fff", stroke: T.thread, "stroke-width": 4, style: "cursor:grab", class: "xp-mixpt" }));
        out.set(`a = (${pt.map((v) => fmt(v, 2)).join(", ")}), &nbsp; Σ a<sub>k</sub> = 1<br>` +
          `mixture y = Σ a<sub>k</sub> m<sub>k</sub> (white) vs the nearest real pixel (yellow bars)<br>` +
          `affine fit of all 1 600 pixels: ‖Y − AM‖ / ‖Y‖ = <b>${fmt(100 * relErr, 1)} %</b> (mostly Poisson noise)`);
      } else {
        ms.forEach((mk, k) => {
          minS = Math.min(minS, ...mk);
          cS.line(E.map((e, j) => [e, mk[j]]), { stroke: PH[k].c, width: 2.5 });
          const neg = E.map((e, j) => [e, mk[j] < 0 ? mk[j] : NaN]);
          cS.line(neg, { stroke: T.thread, width: 6 });
        });
        const fa = negFrac(sc), okS = minS >= TOL_S, okA = fa <= TOL_A;
        out.set(`endmembers c + s (m<sub>k</sub> − c), &nbsp; s = <b>${fmt(sc, 2)}</b> &nbsp; fit error <b>${fmt(100 * relErr, 1)} %</b> for every s<br>` +
          `pixels with an abundance &lt; −0.1: <span class="${okA ? "good" : "bad"}">${fmt(100 * fa, 1)} %</span> &nbsp; ` +
          `smallest spectrum value: <span class="${okS ? "good" : "bad"}">${fmt(minS * 1000, 1)}·10⁻³</span><br>` +
          `feasible band (green): <b>s ∈ [${B[0] === null ? "–" : fmt(B[0], 2)}, ${B[1] === null ? "–" : fmt(B[1], 2)}]</b> ${B[0] !== null && B[1] - B[0] < 0.1 ? "<span class='good'>≈ unique</span>" : "<span class='bad'>many equally good answers</span>"}`);
      }
      ims.forEach((im, k) => {
        const a = Float64Array.from({ length: N }, (_, i) => (!keep(i) ? 0 : abund(i, k)));
        im.draw(a, 0, 1, "magma");
      });
    }
    if (mode === "mixing") {
      svg.addEventListener("pointerdown", (e) => {
        const mv = (ev) => {
          const p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
          const q = p.matrixTransform(svg.getScreenCTM().inverse());
          // barycentric coordinates of q
          const [[x1, y1], [x2, y2], [x3, y3]] = VX, dt = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
          let a1 = ((y2 - y3) * (q.x - x3) + (x3 - x2) * (q.y - y3)) / dt, a2 = ((y3 - y1) * (q.x - x3) + (x1 - x3) * (q.y - y3)) / dt;
          a1 = clamp(a1, 0, 1); a2 = clamp(a2, 0, 1 - a1);
          pt = [a1, a2, 1 - a1 - a2]; update();
        };
        mv(e);
        svg.setPointerCapture(e.pointerId);
        const up = () => { svg.removeEventListener("pointermove", mv); svg.removeEventListener("pointerup", up); };
        svg.addEventListener("pointermove", mv); svg.addEventListener("pointerup", up);
      });
    }
    const sl = slider({ label: "scale the triangle: s", min: 0.3, max: 1.6, value: 1, digits: 2, onInput: (v) => { sc = v; update(); } });
    const pureT = toggle({ label: "pixels", options: [{ value: "all", label: "all" }, { value: "mixed", label: "no pure pixels (all aₖ > 0.2)" }], value: "all",
      onChange: (v) => { hidePure = v === "mixed"; B = band(); update(); } });
    const lg = legend(PH.map((p) => [p.c, p.name]).concat(mode === "ambiguity" ? [[T.thread, "negative counts"]] : [["#fff", "mixture Σ aₖmₖ"]]));
    el.append(mode === "ambiguity" ? controls(sl, pureT) : controls(h("span", { class: "xp-slider-label" }, "drag inside the triangle")),
      row(col(svg, src(SRC_CEVO3 + " 40 × 40 pixels (2 × 2 binned); endmembers from KL-NMF.")),
        h("div", { class: "xp-unmix-side" }, cS.el, lg, row(...ims), card(out))));
    update();
  });
});

// ---------------------------------------------------------------------------------------------
// nmf-live. Lee–Seung multiplicative updates on the real counts (1 600 pixels x 250 channels),
// two iterations per frame. "new start" keeps the previous result as thin lines: different starts
// can end in different, equally good factorisations.
// ---------------------------------------------------------------------------------------------
register("nmf-live", (el, api) => {
  const wait = waiting(el);
  loadData("cevo3_eds").then((d) => {
    wait.remove();
    const { X, N, D, ny, nx } = binned(d), E = Array.from(d.E);
    let K = 3, loss = "kl", seed = 1, it = 0, running = true, W, H, hist = [], prev = null;
    const cS = chart({ width: 900, height: 380, x: { domain: [0.2, 10.2], label: "X-ray energy (keV)" }, y: { domain: [1e-4, 0.5], type: "log", label: "endmember spectrum (unit area)" } });
    const cL = chart({ width: 520, height: 300, x: { domain: [1, 1000], type: "log", label: "iteration" }, y: { domain: [1, 10], type: "log", label: "loss" } });
    const mapsBox = h("div", { class: "xp-row", style: { gap: "12px", justifyContent: "flex-start" } });
    const out = readout();
    let ims = [];
    function start(keep) {
      if (keep && H) prev = { H: Float64Array.from(H), K };
      ({ W, H } = nmfInit(X, N, D, K, seed)); it = 0; hist = []; running = true;
      ims = Array.from({ length: K }, (_, k) => imageCanvas({ n: nx, ny, size: K > 3 ? 175 : 220, label: `component ${k + 1}` }));
      mapsBox.replaceChildren(...ims);
    }
    function draw() {
      const cols = [T.violet, T.gauss, T.pois, T.truth, T.pink];
      cS.clear();
      if (prev) for (let k = 0; k < prev.K; k++) {
        let sm = 0; for (let j = 0; j < D; j++) sm += prev.H[k * D + j];
        cS.line(E.map((e, j) => [e, Math.max(1e-4, prev.H[k * D + j] / sm)]), { stroke: "#94a3b8", width: 1.5, opacity: 0.6 });
      }
      for (let k = 0; k < K; k++) {
        let sm = 0; for (let j = 0; j < D; j++) sm += H[k * D + j];
        cS.line(E.map((e, j) => [e, Math.max(1e-4, H[k * D + j] / sm)]), { stroke: cols[k], width: 2.5 });
        const a = Float64Array.from({ length: N }, (_, i) => W[i * K + k] * sm);
        let mx = 0; for (const v of a) mx = Math.max(mx, v);
        ims[k].draw(a, 0, mx, "magma");
        ims[k].setLabel(`<span style="color:${cols[k]}">■</span> component ${k + 1}`);
      }
      if (hist.length > 1) {
        const lo = Math.min(...hist), hi = Math.max(...hist);
        cL.setDomain([1, Math.max(10, hist.length)], [lo * 0.98, hi * 1.02]); cL.clear();
        cL.line(hist.map((v, i) => [i + 1, v]), { stroke: loss === "kl" ? T.pois : T.gauss, width: 3 });
      }
      out.set(`iteration <b>${it}</b> &nbsp; loss = <b>${hist.length ? hist[hist.length - 1].toExponential(4) : "–"}</b><br>` +
        `${loss === "kl" ? "KL divergence = Poisson NLL (up to a constant)" : "Frobenius ‖X − WH‖² = Gaussian NLL"}<br>` +
        `W ≥ 0 and H ≥ 0 at every step: updates only multiply`);
    }
    api.animate(() => {
      if (!running || !api.isActive()) return;
      for (let r = 0; r < 2; r++) { hist.push(nmfStep(X, W, H, N, D, K, loss)); it++; }
      if (it >= 1000) running = false;
      draw();
    });
    const Ks = slider({ label: "K", min: 2, max: 5, value: K, step: 1, digits: 0, onInput: (v) => { if (Math.round(v) !== K) { K = Math.round(v); prev = null; start(false); } } });
    const Lt = toggle({ label: "loss", options: [{ value: "kl", label: "KL (Poisson)", color: T.pois }, { value: "fro", label: "Frobenius (Gauss)", color: T.gauss }], value: loss,
      onChange: (v) => { loss = v; start(true); } });
    const go = button("⏯ run / pause", () => { running = !running; });
    const again = button("↻ new random start", () => { seed++; start(true); }, "primary");
    el.append(controls(Ks, Lt, go, again),
      row(col(cS.el, mapsBox), h("div", { class: "xp-unmix-side" }, cL.el, card(out),
        legend([["#94a3b8", "previous run"]]), src(SRC_CEVO3 + " 1 600 pixels × 250 channels, counts."))));
    start(false); draw();
    api.onShow(() => { running = true; });
  });
});

// ---------------------------------------------------------------------------------------------
// sped-nmf. NMF (K = 4) of 3 000 precession electron diffraction patterns of a twinned GaAs
// nanowire. The components are diffraction patterns, the abundances are maps: twin domains.
// ---------------------------------------------------------------------------------------------
register("sped-nmf", (el) => {
  const wait = waiting(el);
  loadData("sped").then((d) => {
    wait.remove();
    const [K, my, mx] = d.maps.shape;
    const vdf = imageCanvas({ n: 30, ny: 100, size: 96, label: "virtual ADF" });
    vdf.draw(d.vdf, 0, 255, "gray");
    const cards = [];
    for (let k = 0; k < K; k++) {
      const pat = imageCanvas({ n: 64, size: 250, label: "" });
      pat.draw(d.patterns.subarray(k * 4096, (k + 1) * 4096), 0, 255, "magma");
      const map = imageCanvas({ n: mx, ny: my, size: 96, label: "" });
      const a = d.maps.subarray(k * my * mx, (k + 1) * my * mx);
      let hi = 0; for (const v of a) hi = Math.max(hi, v);
      map.draw(a, 0, hi, "magma");
      const c = h("div", { class: "xp-sped-card" }, h("div", { class: "xp-slider-label" }, `component ${k + 1}`), row(pat, map));
      cards.push(c);
    }
    el.append(row(col(vdf), ...cards.slice(0, 2)), row(...cards.slice(2)),
      src("Data: twinned GaAs nanowire, SPED (pyxem demo data, Zenodo 15490547, CC BY 4.0; Martineau et al. 2019). NMF with K = 4 on √(counts)."));
  });
});
