// Vector and Matrix groups (Acts 1-2), all on real data:
//   projection : dot product and projection in 2-D; the axes are two channels of the real Fe-L spectra
//   fe-match   : a spectrum projected onto the span of real Fe2+ / Fe3+ standards (least squares)
//   datacube   : the CeVO3 / Al2O3 EELS spectrum image; one pixel = one row, one energy window = one column
//   lsq-line   : least squares by hand: squared residuals and the error bowl (real pre-edge points)
//   bg-fit     : the power-law background as a linear least-squares problem in log-log space
import { register, T, rng, fmt, clamp, h, s, row, col, card, controls, readout, slider, toggle, button, legend, chart, imageCanvas } from "./core.js";
import { loadData } from "./data.js";
import { lstsq, dot, norm } from "./la.js";

if (!document.getElementById("xp-vector-style")) {
  document.head.append(h("style", { id: "xp-vector-style" }, `
.xp-vec-side { width: 700px; display: flex; flex-direction: column; gap: 14px; }
.xp-vec-side .xp-readout { margin-top: 0; line-height: 1.5; }
.xp-vec-svg { border-radius: 16px; background: rgba(5,8,15,0.55); border: 1px solid rgba(255,255,255,0.09); touch-action: none; }
.xp-vec-svg .handle { cursor: grab; }
.xp-bars { display: grid; grid-template-columns: 170px 1fr 70px; gap: 6px 12px; align-items: center; font-size: 21px; }
.xp-bars .bar { height: 18px; border-radius: 9px; background: #475569; }
.xp-cube-img canvas { cursor: crosshair; }
.xp-src { font-size: 17px; color: #64748b; margin-top: 4px; }
`));
}
const src = (html) => h("div", { class: "xp-src", html });
export const SRC_CEVO3 = "Data: CeVO₃ on Al₂O₃, STEM-Bank-26 (Hollyer &amp; Stach 2026), CC BY 4.0.";
export const SRC_CYMES = "Data: Cymes, Burgess &amp; Stroud (2022), Zenodo 7439174, CC BY 4.0.";

const at = (E, y, e0) => { let i = 0; while (i < E.length - 1 && E[i + 1] <= e0) i++; return y[i]; };
const waiting = (el, txt = "loading data …") => { const w = h("div", { class: "xp-caption" }, txt); el.append(w); return w; };

// ---------------------------------------------------------------------------------------------
// projection. Two vectors in the plane of two energy channels: 708.8 eV (the Fe2+ L3 maximum) and
// 710.2 eV (the Fe3+ L3 maximum). a = Fe2+ standard, b = magnetite, both real. Drag the tips.
// ---------------------------------------------------------------------------------------------
register("projection", (el) => {
  const wait = waiting(el);
  loadData("fe_refs").then((d) => {
    wait.remove();
    const E = d.E, e1 = 708.8, e2 = 710.2;
    const two = (y) => [at(E, y, e1), at(E, y, e2)];
    const A0 = two(d.refs.Fe2), B0 = two(d.refs.Fe23);
    let a = A0.slice(), b = B0.slice();
    const W = 820, H = 640, M = 70, R = 1.15 * Math.max(...A0, ...B0, ...two(d.refs.Fe3));
    // one scale for both axes, so that right angles look like right angles
    const SC = Math.min(W - 2 * M, H - 2 * M) / R;
    const X = (v) => M + v * SC, Y = (v) => H - M - v * SC;
    const iX = (p) => (p - M) / SC, iY = (p) => (H - M - p) / SC;
    const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "xp-chart xp-vec-svg", style: "overflow:hidden" });
    svg.append(s("defs", {}, ...[["pa", T.gauss], ["pb", T.thread], ["pp", T.truth]].map(([id, c]) =>
      s("marker", { id: `xpv${id}`, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto" }, s("path", { d: "M0 0 L10 5 L0 10 Z", fill: c })))));
    svg.append(s("line", { x1: X(0), x2: X(R), y1: Y(0), y2: Y(0), stroke: T.axis, "stroke-width": 2 }),
      s("line", { x1: X(0), x2: X(0), y1: Y(0), y2: Y(R), stroke: T.axis, "stroke-width": 2 }),
      s("text", { x: X(R), y: Y(0) + 40, "text-anchor": "end", class: "xp-axlabel" }, "channel 1: intensity at 708.8 eV (Fe²⁺ L₃)"),
      s("text", { x: 0, y: 0, class: "xp-axlabel", "text-anchor": "middle", transform: `translate(${X(0) - 34} ${Y(R / 2)}) rotate(-90)` }, "channel 2: at 710.2 eV (Fe³⁺ L₃)"));
    const g = s("g");
    svg.append(g);
    const hA = s("circle", { r: 16, fill: T.gauss, class: "handle", stroke: "#fff", "stroke-width": 3 });
    const hB = s("circle", { r: 16, fill: T.thread, class: "handle", stroke: "#fff", "stroke-width": 3 });
    svg.append(hA, hB);
    const out = readout(), note = h("div", { class: "xp-caption", style: { fontSize: "22px" } });

    function draw() {
      const ah = [a[0] / norm(a), a[1] / norm(a)], c = dot(ah, b), p = [c * ah[0], c * ah[1]], r = [b[0] - p[0], b[1] - p[1]];
      const cos = dot(a, b) / (norm(a) * norm(b));
      g.replaceChildren();
      const ext = R * 1.6;
      g.append(s("line", { x1: X(-ext * ah[0]), y1: Y(-ext * ah[1]), x2: X(ext * ah[0]), y2: Y(ext * ah[1]), stroke: T.gauss, "stroke-width": 1.5, "stroke-dasharray": "4 8", opacity: 0.6 }));
      const arrow = (v, c, id, w = 6, dash) => s("line", { x1: X(0), y1: Y(0), x2: X(v[0]), y2: Y(v[1]), stroke: c, "stroke-width": w, "marker-end": id ? `url(#xpv${id})` : null, "stroke-dasharray": dash });
      g.append(arrow(a, T.gauss, "pa", 12));                          // a first (wide), the projection on top of it
      g.append(s("line", { x1: X(p[0]), y1: Y(p[1]), x2: X(b[0]), y2: Y(b[1]), stroke: T.pois, "stroke-width": 4, "stroke-dasharray": "9 7" }));
      // right-angle mark
      const q = 0.045 * R, rn = norm(r) || 1, rh = [r[0] / rn, r[1] / rn];
      const c1 = [p[0] - q * ah[0], p[1] - q * ah[1]], c2 = [c1[0] + q * rh[0], c1[1] + q * rh[1]], c3 = [p[0] + q * rh[0], p[1] + q * rh[1]];
      if (norm(r) > 2 * q) g.append(s("path", { d: `M${X(c1[0])},${Y(c1[1])} L${X(c2[0])},${Y(c2[1])} L${X(c3[0])},${Y(c3[1])}`, fill: "none", stroke: T.pois, "stroke-width": 2.5 }));
      g.append(arrow(p, T.truth, "pp", 5), arrow(b, T.thread, "pb"));
      const lab = (v, t, c, dy = -18) => g.append(s("text", { x: X(v[0]) + 14, y: Y(v[1]) + dy, fill: c, "font-size": 26, "font-weight": 700, class: "xp-ctext" }, t));
      lab(a, "a = Fe²⁺ (FeO)", T.gauss, 30); lab(b, "b = magnetite", T.thread);
      lab([p[0] / 2, p[1] / 2], "c â", T.truth, 36);
      hA.setAttribute("cx", X(a[0])); hA.setAttribute("cy", Y(a[1]));
      hB.setAttribute("cx", X(b[0])); hB.setAttribute("cy", Y(b[1]));
      out.set(`aᵀb = <b>${fmt(dot(a, b), 4)}</b> &nbsp; ‖a‖ = ${fmt(norm(a), 3)} &nbsp; ‖b‖ = ${fmt(norm(b), 3)}<br>` +
        `cos θ = aᵀb / (‖a‖‖b‖) = <b>${fmt(cos, 3)}</b> &nbsp; (θ = ${fmt(Math.acos(clamp(cos, -1, 1)) * 180 / Math.PI, 1)}°)<br>` +
        `score c = âᵀb = <span class="good">${fmt(c, 3)}</span> &nbsp; projection c â (green)<br>` +
        `residual r = b − c â: &nbsp; rᵀa = <b>${fmt(dot(r, a), 6)}</b> &nbsp; ⇒ r ⊥ a`);
    }
    function drag(handle, set) {
      handle.addEventListener("pointerdown", (e) => {
        handle.setPointerCapture(e.pointerId);
        const mv = (ev) => {
          const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
          const q = pt.matrixTransform(svg.getScreenCTM().inverse());
          set([clamp(iX(q.x), -0.1 * R, R), clamp(iY(q.y), -0.1 * R, R)]); draw();
        };
        const up = () => { handle.removeEventListener("pointermove", mv); handle.removeEventListener("pointerup", up); };
        handle.addEventListener("pointermove", mv); handle.addEventListener("pointerup", up);
      });
    }
    drag(hA, (v) => { a = v; }); drag(hB, (v) => { b = v; });
    const preset = toggle({ label: "b =", options: [{ value: "Fe23", label: "magnetite" }, { value: "Fe3", label: "Fe³⁺ glass" }, { value: "Fe0", label: "Fe metal" }, { value: "Fe2", label: "FeO" }],
      value: "Fe23", onChange: (k) => { a = A0.slice(); b = two(d.refs[k]); draw(); } });
    note.innerHTML = "A spectrum with only two channels is a vector in the plane. A real Fe-L spectrum has 301 channels here: the same formulas, in ℝ³⁰¹.";
    el.append(controls(preset), row(svg, h("div", { class: "xp-vec-side" }, card(out), card(note),
      legend([[T.gauss, "a"], [T.thread, "b"], [T.truth, "projection"], [T.pois, "residual", "dash"]]), src(SRC_CYMES))));
    draw();
  });
});

// ---------------------------------------------------------------------------------------------
// fe-match. y ≈ w₂·Fe²⁺ + w₃·Fe³⁺ (+ w₀·Fe⁰): least squares = projection of y onto the span of the
// references. Planted test first (a mixture of the two real standards, with noise): the projection
// recovers the fraction. Then real magnetite: Fe³⁺/ΣFe = 0.43, not 0.67, and the residual has
// structure. The references do not span magnetite (tetrahedral Fe³⁺ is missing).
// ---------------------------------------------------------------------------------------------
register("fe-match", (el) => {
  const wait = waiting(el);
  loadData("fe_refs").then((d) => {
    wait.remove();
    const E = Array.from(d.E), n = E.length, R = d.refs;
    let mode = "mix", alpha = 0.67, noise = 0.02, refset = "23";
    const rr = rng(5), noiseVec = E.map(() => rr.normal());
    const cS = chart({ width: 1040, height: 470, x: { domain: [700, 730], label: "energy loss (eV)" }, y: { domain: [-0.02, 0.2], label: "intensity (unit area)" } });
    const cR = chart({ width: 1040, height: 210, x: { domain: [700, 730] }, y: { domain: [-0.03, 0.03], label: "residual" }, margin: { b: 30 } });
    const out = readout(), bars = h("div", { class: "xp-bars" });
    const SETS = { 23: ["Fe2", "Fe3"], "023": ["Fe0", "Fe2", "Fe3"] };
    const NAME = { Fe0: "Fe⁰ (FeNi)", Fe2: "Fe²⁺ (FeO)", Fe3: "Fe³⁺ (glass)", Fe23: "magnetite" };
    const COL = { Fe0: T.violet, Fe2: T.gauss, Fe3: T.pois };
    function target() {
      if (mode === "real") return Array.from(R.Fe23);
      const sc = 0.15;
      return E.map((_, i) => (1 - alpha) * R.Fe2[i] + alpha * R.Fe3[i] + noise * sc * noiseVec[i]);
    }
    function update() {
      const y = target(), keys = SETS[refset], k = keys.length;
      const X = new Float64Array(n * k);
      for (let i = 0; i < n; i++) keys.forEach((key, j) => { X[i * k + j] = R[key][i]; });
      const w = lstsq(X, y, n, k);
      const fit = E.map((_, i) => keys.reduce((acc, key, j) => acc + w[j] * R[key][i], 0));
      const r = y.map((v, i) => v - fit[i]);
      cS.clear(); cR.clear();
      keys.forEach((key, j) => cS.line(E.map((e, i) => [e, w[j] * R[key][i]]), { stroke: COL[key], width: 3, dash: "8 6" }));
      cS.line(E.map((e, i) => [e, y[i]]), { stroke: "#fff", width: 3 });
      cS.line(E.map((e, i) => [e, fit[i]]), { stroke: T.thread, width: 4 });
      cR.ruleY(0, { stroke: T.faint });
      cR.line(E.map((e, i) => [e, r[i]]), { stroke: T.warn, width: 2.5 });
      const f3 = keys.indexOf("Fe3"), tot = w.reduce((p, q) => p + q, 0);
      const orth = keys.map((key, j) => { let s_ = 0; for (let i = 0; i < n; i++) s_ += X[i * k + j] * r[i]; return s_; });
      const ny = norm(y), cos = (u) => dot(u, y) / (norm(u) * ny);
      bars.replaceChildren(...["Fe0", "Fe2", "Fe3"].flatMap((key) => {
        const c = cos(Array.from(R[key]));
        return [h("span", { html: NAME[key] }), h("div", { class: "bar", style: { width: `${Math.max(0, (c - 0.7) / 0.3) * 100}%`, background: COL[key] } }), h("b", {}, fmt(c, 3))];
      }));
      const truth = mode === "mix" ? ` &nbsp; (planted: <b>${fmt(alpha, 2)}</b>)` : " &nbsp; (stoichiometry: <b>0.67</b>)";
      const ok = mode === "mix" ? Math.abs(w[f3] / tot - alpha) < 0.03 : false;
      out.set(`weights: ${keys.map((key, j) => `${NAME[key].split(" ")[0]} <b>${fmt(w[j], 3)}</b>`).join(" · ")}<br>` +
        `Fe³⁺ / ΣFe = <span class="${ok ? "good" : "bad"}">${fmt(w[f3] / tot, 2)}</span>${truth}<br>` +
        `‖r‖ / ‖y‖ = <b>${fmt(norm(r) / ny * 100, 1)} %</b> &nbsp; Xᵀr = (${orth.map((v) => fmt(v * 1e6, 1)).join(", ")})·10⁻⁶ ≈ 0`);
    }
    const unk = toggle({ label: "unknown y", options: [{ value: "mix", label: "planted mix" }, { value: "real", label: "real magnetite" }], value: mode,
      onChange: (v) => { mode = v; al.style.opacity = no.style.opacity = v === "mix" ? 1 : 0.35; update(); } });
    const ref = toggle({ label: "references", options: [{ value: "23", label: "Fe²⁺, Fe³⁺" }, { value: "023", label: "+ Fe⁰" }], value: refset, onChange: (v) => { refset = v; update(); } });
    const al = slider({ label: "Fe³⁺ fraction", min: 0, max: 1, value: alpha, digits: 2, onInput: (v) => { alpha = v; update(); } });
    const no = slider({ label: "noise", min: 0, max: 0.2, value: noise, digits: 2, onInput: (v) => { noise = v; update(); } });
    el.append(controls(unk, ref), controls(al, no),
      row(col(cS.el, cR.el, legend([["#fff", "y (unknown)"], [T.thread, "projection Xŵ"], [T.gauss, "w·Fe²⁺", "dash"], [T.pois, "w·Fe³⁺", "dash"], [T.warn, "residual r"]])),
        h("div", { class: "xp-vec-side", style: { width: "640px" } },
          card(h("div", { class: "xp-slider-label", style: { marginBottom: "8px" } }, "cos θ between y and each reference"), bars),
          card(out), src(SRC_CYMES))));
    update();
  });
});

// ---------------------------------------------------------------------------------------------
// datacube. Real EELS spectrum image, 40 x 40 pixels (2 x 2 binned, 1.25 nm) x 512 channels (1 eV).
// Click the map: that pixel's spectrum is one row of X (1600 x 512). Pick an edge: the energy
// window is a block of columns of X, and summing it gives a map.
// ---------------------------------------------------------------------------------------------
const WINDOWS = {
  V: { label: "V-L₂,₃", lo: 512, hi: 530, bg: [478, 508], c: T.thread },
  Ce: { label: "Ce-M₄,₅", lo: 878, hi: 910, bg: [840, 875], c: T.pois },
  all: { label: "all channels", lo: 463, hi: 975, c: T.data },
};
export function eelsCube(d) {
  const [ny, nx, ne] = d.cube.shape, E = Array.from(d.E), sc = d.scale, C = d.cube;
  const spec = (i) => { const o = i * ne, out = new Float64Array(ne); for (let k = 0; k < ne; k++) out[k] = C[o + k] * sc; return out; };
  const mapOf = (lo, hi) => {
    const m = new Float64Array(ny * nx), j0 = E.findIndex((e) => e >= lo), j1 = E.findIndex((e) => e >= hi);
    for (let i = 0; i < ny * nx; i++) { let a = 0; for (let k = j0; k < (j1 < 0 ? ne : j1); k++) a += C[i * ne + k]; m[i] = a * sc; }
    return m;
  };
  // edge signal per pixel: power law fitted in log-log space in the window bg, subtracted in [lo, hi)
  const tot = mapOf(E[0], E[ne - 1] + 1), med = Float64Array.from(tot).sort()[tot.length >> 1];
  const vacuum = tot.map((v) => v < 0.5 * med);             // thin / vacuum pixels: no reliable background fit
  const edgeMap = (lo, hi, bg) => {
    const m = new Float64Array(ny * nx), b0 = E.findIndex((e) => e >= bg[0]), b1 = E.findIndex((e) => e >= bg[1]);
    const j0 = E.findIndex((e) => e >= lo), j1 = E.findIndex((e) => e >= hi);
    for (let i = 0; i < ny * nx; i++) {
      let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (let k = b0; k < b1; k++) { const x = Math.log(E[k]), y = Math.log(Math.max(1e-9, C[i * ne + k] * sc)); n++; sx += x; sy += y; sxx += x * x; sxy += x * y; }
      const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx), icpt = (sy - slope * sx) / n;
      let a = 0;
      for (let k = j0; k < j1; k++) a += C[i * ne + k] * sc - Math.exp(icpt + slope * Math.log(E[k]));
      m[i] = vacuum[i] || slope > -0.5 || slope < -8 ? 0 : a;   // unphysical power law (scan-start pixels): mask
    }
    return m;
  };
  return { ny, nx, ne, E, spec, mapOf, edgeMap, vacuum };
}
const range = (a) => { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; };
export const prange = (a, p = 1) => { const b = Float64Array.from(a).sort(), n = b.length; return [b[Math.floor(n * p / 100)], b[Math.min(n - 1, Math.floor(n * (100 - p) / 100))]]; };

register("datacube", (el) => {
  const wait = waiting(el);
  loadData("cevo3_eels").then((d) => {
    wait.remove();
    const cu = eelsCube(d), { ny, nx, ne, E } = cu;
    let pix = 7 * nx + 35, win = "Ce", sub = true;
    const im = imageCanvas({ n: nx, ny, size: 520, label: " " });
    im.classList.add("xp-cube-img");
    const mark = s("svg", { width: 520, height: 520, style: "position:absolute;left:0;top:0;pointer-events:none" });
    const stack = h("div", { style: { position: "relative" } });
    im.insertBefore(stack, im.canvas);
    stack.append(im.canvas, mark);
    const cS = chart({ width: 1000, height: 420, x: { domain: [463, 975], label: "energy loss (eV)" }, y: { domain: [1e3, 1e5], type: "log", label: "intensity" } });
    const mat = s("svg", { width: 1000, height: 250, class: "xp-chart" });
    const out = readout();
    function drawMat() {
      const x0 = 210, y0 = 30, w = 700, hh = 190, j0 = E.findIndex((e) => e >= WINDOWS[win].lo), j1 = E.findIndex((e) => e >= WINDOWS[win].hi);
      const ry = y0 + (pix / (ny * nx)) * hh;
      mat.replaceChildren(
        s("rect", { x: x0, y: y0, width: w, height: hh, fill: "rgba(30,41,59,0.9)", stroke: T.axis, "stroke-width": 2 }),
        s("rect", { x: x0 + (j0 / ne) * w, y: y0, width: Math.max(3, ((j1 < 0 ? ne : j1) - j0) / ne * w), height: hh, fill: WINDOWS[win].c, opacity: 0.35 }),
        s("rect", { x: x0, y: ry - 3, width: w, height: 6, fill: "#fff" }),
        s("text", { x: x0 - 16, y: y0 + hh / 2 + 8, "text-anchor": "end", "font-size": 30, fill: "#fff", "font-family": "serif", "font-style": "italic" }, "X ="),
        s("text", { x: x0 + w / 2, y: y0 - 8, "text-anchor": "middle", class: "xp-tick" }, `${ne} columns = energy channels`),
        s("text", { x: x0 + w + 14, y: y0 + hh / 2, class: "xp-tick" }, `${ny * nx} rows`),
        s("text", { x: x0 + w + 14, y: y0 + hh / 2 + 24, class: "xp-tick" }, "= pixels"),
        s("text", { x: x0 + w + 14, y: ry + 7, fill: "#fff", "font-size": 20 }, `row ${pix}`));
    }
    function update() {
      const W = WINDOWS[win], m = sub && W.bg ? cu.edgeMap(W.lo, W.hi, W.bg) : cu.mapOf(W.lo, W.hi);
      const [lo, hi] = prange(m, 1);
      im.setLabel(sub && W.bg ? `${W.label}: edge signal after power-law background` : `${W.label}: raw sum over the window`);
      im.draw(m, lo, hi, "magma");
      const y = Math.floor(pix / nx), x = pix % nx, sz = 520 / nx;
      mark.replaceChildren(s("rect", { x: x * sz - 2, y: y * sz - 2, width: sz + 4, height: sz + 4, fill: "none", stroke: "#fff", "stroke-width": 3 }));
      const sp = cu.spec(pix), [slo, shi] = range(sp);
      cS.setDomain(null, [slo * 0.8, shi * 1.3]);
      cS.clear();
      for (const [k, w] of Object.entries(WINDOWS)) if (k !== "all") {
        cS.rect(w.lo, slo * 0.8, w.hi, shi * 1.3, { fill: w.c, opacity: k === win ? 0.3 : 0.08 });
        if (k === win && sub) cS.rect(w.bg[0], slo * 0.8, w.bg[1], shi * 1.3, { fill: T.truth, opacity: 0.14 });
      }
      cS.line(E.map((e, i) => [e, sp[i]]), { stroke: "#fff", width: 2 });
      for (const [k, w] of Object.entries(WINDOWS)) if (k !== "all") cS.text((w.lo + w.hi) / 2, shi * 1.15, w.label, { fill: w.c, anchor: "middle", size: 20, weight: 700 });
      drawMat();
      out.set(`pixel (${y}, ${x}) → row ${pix} of X: <b>X[${pix}, :]</b> is the spectrum (white)<br>` +
        `${W.label}: columns ${E.findIndex((e) => e >= W.lo)}–${E.findIndex((e) => e >= W.hi) - 1} → <b>X[:, j₀:j₁].sum(1).reshape(${ny}, ${nx})</b> is the map`);
    }
    im.canvas.addEventListener("pointerdown", (e) => {
      const r = im.canvas.getBoundingClientRect(), x = Math.floor((e.clientX - r.left) / r.width * nx), y = Math.floor((e.clientY - r.top) / r.height * ny);
      pix = clamp(y, 0, ny - 1) * nx + clamp(x, 0, nx - 1); update();
    });
    const wt = toggle({ label: "energy window", options: Object.entries(WINDOWS).map(([k, w]) => ({ value: k, label: w.label, color: w.c })), value: win, onChange: (v) => { win = v; update(); } });
    const st = toggle({ label: "map", options: [{ value: "raw", label: "raw window sum" }, { value: "sub", label: "background removed" }], value: "sub",
      onChange: (v) => { sub = v === "sub"; update(); } });
    el.append(controls(wt, st), row(col(im, h("div", { class: "xp-caption" }, "click a pixel · map = sum over the window"), src(SRC_CEVO3)), col(cS.el, mat, card(out))));
    update();
  });
});

// ---------------------------------------------------------------------------------------------
// lsq-line. Least squares by hand, before the matrix form. Real points: the pre-edge window of the
// CeVOx region in log-log space (every 3rd channel). Model y = b + m·u with u = log E − mean, so that
// the two unknowns do not trade off (the error bowl is round, not a long valley). Drag the two line
// handles or click into the bowl; "solve" moves to the bottom, where Σr = 0 and Σu·r = 0, i.e. Xᵀr = 0.
// ---------------------------------------------------------------------------------------------
register("lsq-line", (el) => {
  const wait = waiting(el);
  loadData("cevo3_eels").then((d) => {
    wait.remove();
    const cu = eelsCube(d), { ny, nx, ne, E } = cu;
    const sp = new Float64Array(ne); let cnt = 0;
    for (let y = 26; y < 32; y++) for (let x = 33; x < 38; x++) { const v = cu.spec(y * nx + x); for (let k = 0; k < ne; k++) sp[k] += v[k]; cnt++; }
    const idx = E.map((e, i) => i).filter((i) => E[i] >= 470 && E[i] <= 508).filter((i, k) => k % 3 === 0);
    const lx = idx.map((i) => Math.log(E[i])), ly = idx.map((i) => Math.log(sp[i] / cnt)), n = lx.length;
    const xm = lx.reduce((a, b) => a + b, 0) / n, u = lx.map((v) => v - xm);
    const suu = u.reduce((a, v) => a + v * v, 0);
    const bHat = ly.reduce((a, v) => a + v, 0) / n, mHat = u.reduce((a, v, i) => a + v * ly[i], 0) / suu;
    const sse = (b, m) => ly.reduce((a, y, i) => a + (y - b - m * u[i]) ** 2, 0), sMin = sse(bHat, mHat);
    const yr = Math.max(...ly) - Math.min(...ly), uMax = Math.max(...u);
    const DB = 0.9 * yr, DM = 0.9 * yr / uMax;
    const B0 = bHat + 0.12 * DB, M0 = mHat + 0.6 * DM;
    let b = B0, m = M0, anim = null;

    // left: the data and the line, with each residual drawn as a real square (area = r²)
    const pad = 0.006, ypad = 0.35 * yr;
    const cD = chart({ width: 860, height: 580, x: { domain: [lx[0] - pad, lx[n - 1] + pad], label: "log E   (E = energy loss in eV)" },
      y: { domain: [Math.min(...ly) - ypad, Math.max(...ly) + ypad], label: "log I" }, margin: { l: 96 } });
    cD.setDomain();
    const gL = s("g"); cD.gPlot.append(gL);
    const u0 = lx[0] - pad - xm, u1 = lx[n - 1] + pad - xm;
    const mkH = (c) => s("circle", { r: 15, fill: c, stroke: "#fff", "stroke-width": 3, style: "cursor:grab" });
    const hL = mkH(T.truth), hR = mkH(T.truth); cD.gTop.append(hL, hR);
    // right: the error bowl SSE(b, m)
    const cB = chart({ width: 560, height: 470, x: { domain: [mHat - DM, mHat + DM], label: "slope m  (= −r)" },
      y: { domain: [bHat - DB, bHat + DB], label: "level b" }, title: "sum of squared residuals: a bowl", margin: { l: 96 } });
    cB.setDomain();
    const pw = cB.width - cB.m.l - cB.m.r, ph = cB.height - cB.m.t - cB.m.b, G = 160;
    const cv = document.createElement("canvas"); cv.width = G; cv.height = G;
    const ctx = cv.getContext("2d"), img = ctx.createImageData(G, G);
    const lmax = Math.log(sse(bHat + DB, mHat + DM) / sMin);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      const mm = mHat - DM + 2 * DM * (i + 0.5) / G, bb = bHat + DB - 2 * DB * (j + 0.5) / G;
      const t = Math.sqrt(Math.min(1, Math.log(sse(bb, mm) / sMin) / lmax)), band = Math.floor(t * 12) % 2 ? 0.86 : 1;
      const o = 4 * (j * G + i);
      img.data[o] = band * (8 + 150 * t); img.data[o + 1] = band * (12 + 110 * t); img.data[o + 2] = band * (30 + 160 * t); img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    cB.gPlot.append(s("image", { href: cv.toDataURL(), x: cB.m.l, y: cB.m.t, width: pw, height: ph, preserveAspectRatio: "none", style: "image-rendering:pixelated" }));
    const gB = s("g"); cB.gTop.append(gB);
    const out = readout();

    function draw() {
      gL.replaceChildren();
      const r = ly.map((y, i) => y - b - m * u[i]);
      const pxPerY = Math.abs(cD.y(0) - cD.y(1));
      r.forEach((ri, i) => {
        const X = cD.x(lx[i]), Y0 = cD.y(ly[i]), Y1 = cD.y(b + m * u[i]), side = Math.abs(Y1 - Y0);
        gL.append(s("rect", { x: X, y: Math.min(Y0, Y1), width: side, height: side, fill: T.pois, opacity: 0.22, stroke: T.pois, "stroke-width": 1.5 }));
        gL.append(s("line", { x1: X, x2: X, y1: Y0, y2: Y1, stroke: T.pois, "stroke-width": 3 }));
      });
      gL.append(s("line", { x1: cD.x(u0 + xm), x2: cD.x(u1 + xm), y1: cD.y(b + m * u0), y2: cD.y(b + m * u1), stroke: T.truth, "stroke-width": 4 }));
      lx.forEach((x, i) => gL.append(s("circle", { cx: cD.x(x), cy: cD.y(ly[i]), r: 7, fill: "#fff" })));
      hL.setAttribute("cx", cD.x(u0 + xm)); hL.setAttribute("cy", cD.y(b + m * u0));
      hR.setAttribute("cx", cD.x(u1 + xm)); hR.setAttribute("cy", cD.y(b + m * u1));
      gB.replaceChildren(
        s("path", { d: `M${cB.x(mHat) - 12},${cB.y(bHat) - 12} L${cB.x(mHat) + 12},${cB.y(bHat) + 12} M${cB.x(mHat) - 12},${cB.y(bHat) + 12} L${cB.x(mHat) + 12},${cB.y(bHat) - 12}`, stroke: "#fff", "stroke-width": 3, opacity: 0.8 }),
        s("circle", { cx: cB.x(m), cy: cB.y(b), r: 11, fill: T.truth, stroke: "#fff", "stroke-width": 3 }));
      const S = sse(b, m), g1 = r.reduce((a, v) => a + v, 0), g2 = r.reduce((a, v, i) => a + u[i] * v, 0);
      const at = S < 1.0005 * sMin;
      out.set(`Σ rᵢ² = <b>${(S / sMin).toFixed(2)} ×</b> the minimum ${at ? "<span class='good'>✓</span>" : ""} &nbsp; r = −m = <b>${fmt(-m, 2)}</b><br>` +
        `Σ rᵢ = ${fmt(g1, 4)} &nbsp; &nbsp; Σ uᵢ rᵢ = ${fmt(g2, 5)}<br>` +
        (at ? `<span class="good">both 0: the bowl is flat ⇒ Xᵀr = 0 ⇒ XᵀX ŵ = Xᵀy</span>`
            : `at the bottom both sums are 0: the bowl is flat`));
    }
    const svgPt = (svg, ev) => { const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY; return pt.matrixTransform(svg.getScreenCTM().inverse()); };
    const invY = (py) => { const a = cD.y(0), c = cD.y(1); return (py - a) / (c - a); };
    function drag(hd, which) {
      hd.addEventListener("pointerdown", (e) => {
        stop(); hd.setPointerCapture(e.pointerId);
        const mv = (ev) => {
          const yv = invY(svgPt(cD.el, ev).y);
          const yl = which === "L" ? yv : b + m * u0, yrr = which === "R" ? yv : b + m * u1;
          m = (yrr - yl) / (u1 - u0); b = yl - m * u0; draw();
        };
        const up = () => { hd.removeEventListener("pointermove", mv); hd.removeEventListener("pointerup", up); };
        hd.addEventListener("pointermove", mv); hd.addEventListener("pointerup", up);
      });
    }
    drag(hL, "L"); drag(hR, "R");
    const bowlSet = (ev) => {
      const q = svgPt(cB.el, ev), a0 = cB.x(0), a1 = cB.x(1), c0 = cB.y(0), c1 = cB.y(1);
      m = clamp((q.x - a0) / (a1 - a0), mHat - DM, mHat + DM); b = clamp((q.y - c0) / (c1 - c0), bHat - DB, bHat + DB); draw();
    };
    cB.el.style.cursor = "crosshair"; cB.el.style.touchAction = "none";
    cB.el.addEventListener("pointerdown", (e) => { stop(); cB.el.setPointerCapture(e.pointerId); bowlSet(e);
      const up = () => { cB.el.removeEventListener("pointermove", bowlSet); cB.el.removeEventListener("pointerup", up); };
      cB.el.addEventListener("pointermove", bowlSet); cB.el.addEventListener("pointerup", up); });
    function stop() { if (anim) cancelAnimationFrame(anim); anim = null; }
    function solve() {
      stop();
      const step = () => {               // gradient descent with a fixed fraction: a visible slide to the bottom
        b += 0.12 * (bHat - b); m += 0.12 * (mHat - m);
        if (Math.abs(b - bHat) < 1e-5 && Math.abs(m - mHat) < 1e-4) { b = bHat; m = mHat; draw(); anim = null; return; }
        draw(); anim = requestAnimationFrame(step);
      };
      anim = requestAnimationFrame(step);
    }
    el.append(controls(button("go to the bottom of the bowl", solve), button("bad start", () => { stop(); b = B0; m = M0; draw(); }),
      h("span", { class: "xp-slider-label" }, "drag the two green handles, or click into the bowl")),
      row(col(cD.el, legend([["#fff", "data (log I)"], [T.truth, "line b + m·u"], [T.pois, "residual rᵢ and its square rᵢ²"]]),
          src(SRC_CEVO3 + " Pre-edge window 470–508 eV, every 3rd channel.")),
        h("div", { class: "xp-vec-side", style: { width: "600px" } }, cB.el, card(out))));
    draw();
  });
});

// ---------------------------------------------------------------------------------------------
// bg-fit. Power law I = A·E^(−r) in a pre-edge window: log I = log A − r log E is linear in the
// unknowns (log A, r), so the fit is a projection: X = [1, log E], y = log I, Xᵀ(y − Xŵ) = 0.
// Extrapolated under the edge and subtracted, it reveals the edge signal.
// ---------------------------------------------------------------------------------------------
const EDGES = {
  V: { label: "V-L₂,₃ + O-K", win: [470, 508], view: [465, 600], onset: 513 },
  Ce: { label: "Ce-M₄,₅", win: [835, 870], view: [800, 960], onset: 883 },
};
const PIXELS = { all: "sum of all pixels", ce: "Ce-rich particle", v: "CeVOₓ region", al: "Al₂O₃ support" };

register("bg-fit", (el) => {
  const wait = waiting(el);
  loadData("cevo3_eels").then((d) => {
    wait.remove();
    const cu = eelsCube(d), { ny, nx, ne, E } = cu;
    // regions: boxes in the 40 x 40 binned map (rows, cols)
    const BOX = { ce: [5, 10, 33, 38], v: [26, 32, 33, 38], al: [20, 26, 8, 14] };
    const regionSpec = (key) => {
      const out = new Float64Array(ne); let cnt = 0;
      for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        if (key !== "all") { const [y0, y1, x0, x1] = BOX[key]; if (y < y0 || y >= y1 || x < x0 || x >= x1) continue; }
        const sp = cu.spec(y * nx + x); for (let k = 0; k < ne; k++) out[k] += sp[k]; cnt++;
      }
      return out.map((v) => v / cnt);
    };
    let edge = "V", region = "v", lo = EDGES.V.win[0], wid = EDGES.V.win[1] - EDGES.V.win[0];
    const cS = chart({ width: 1080, height: 430, x: { domain: EDGES.V.view, label: "energy loss (eV)" }, y: { domain: [0, 1], label: "intensity" }, margin: { l: 104 } });
    const cE = chart({ width: 1080, height: 300, x: { domain: EDGES.V.view, label: "energy loss (eV)" }, y: { domain: [0, 1], label: "edge signal" }, margin: { l: 104 } });
    const cL = chart({ width: 560, height: 430, x: { domain: [6.1, 6.4], label: "log E" }, y: { domain: [7, 9], label: "log I" }, title: "the fit is a straight line here", margin: { l: 96 } });
    const out = readout();
    function update() {
      const ed = EDGES[edge], sp = regionSpec(region), v = ed.view;
      const idx = E.map((e, i) => i).filter((i) => E[i] >= lo && E[i] <= lo + wid);
      const m = idx.length, X = new Float64Array(m * 2), y = new Float64Array(m);
      idx.forEach((i, r) => { X[r * 2] = 1; X[r * 2 + 1] = Math.log(E[i]); y[r] = Math.log(sp[i]); });
      const w = lstsq(X, y, m, 2), A = Math.exp(w[0]), rr = -w[1];
      const res = Array.from(y).map((yy, r) => yy - (w[0] + w[1] * X[r * 2 + 1]));
      const o0 = res.reduce((p, q) => p + q, 0), o1 = res.reduce((p, q, r) => p + q * X[r * 2 + 1], 0);
      const bg = E.map((e) => A * e ** -rr);
      const vi = E.map((e, i) => i).filter((i) => E[i] >= v[0] && E[i] <= v[1]);
      const ymax = Math.max(...vi.map((i) => sp[i])), ymin = Math.min(...vi.map((i) => sp[i]));
      cS.setDomain(v, [0.85 * ymin, 1.05 * ymax]); cS.clear();
      cS.rect(lo, 0.85 * ymin, lo + wid, 1.05 * ymax, { fill: T.truth, opacity: 0.14 });
      cS.line(vi.map((i) => [E[i], sp[i]]), { stroke: "#fff", width: 2.5 });
      cS.line(vi.map((i) => [E[i], bg[i]]), { stroke: T.truth, width: 3, dash: "10 6" });
      cS.text(lo + wid / 2, 1.0 * ymax, "fit window", { fill: T.truth, anchor: "middle", size: 20 });
      const sig = vi.map((i) => [E[i], sp[i] - bg[i]]), smax = Math.max(...sig.map((p) => p[1]));
      cE.setDomain(v, [-0.1 * smax, 1.1 * smax]); cE.clear();
      cE.ruleY(0, { stroke: T.faint }); cE.ruleX(ed.onset, { stroke: T.thread });
      cE.area(sig, sig.map(([e]) => [e, 0]), { fill: T.thread, opacity: 0.25 });
      cE.line(sig, { stroke: T.thread, width: 2.5 });
      const lx = idx.map((i) => Math.log(E[i])), ly = Array.from(y);
      const pad = 0.02;
      cL.setDomain([Math.min(...lx) - pad, Math.max(...lx) + pad], [Math.min(...ly) - pad, Math.max(...ly) + pad]); cL.clear();
      cL.dots(lx.map((x, r) => [x, ly[r]]), { r: 4, fill: T.data });
      cL.line([[lx[0] - pad, w[0] + w[1] * (lx[0] - pad)], [lx[m - 1] + pad, w[0] + w[1] * (lx[m - 1] + pad)]], { stroke: T.truth, width: 3 });
      out.set(`X = [1, log E] (${m} × 2), y = log I &nbsp;→&nbsp; XᵀX ŵ = Xᵀy<br>` +
        `ŵ = (log A, −r): &nbsp; <b>r = ${fmt(rr, 2)}</b> &nbsp; A = ${A.toExponential(2)}<br>` +
        `Xᵀ(y − Xŵ) = (${o0.toExponential(1)}, ${o1.toExponential(1)}) ≈ 0: the residual is ⊥ to both columns`);
    }
    const et = toggle({ label: "edge", options: Object.entries(EDGES).map(([k, e]) => ({ value: k, label: e.label })), value: edge,
      onChange: (k) => { edge = k; lo = EDGES[k].win[0]; wid = EDGES[k].win[1] - EDGES[k].win[0]; sLo.set(lo, false); sW.set(wid, false); update(); } });
    const rt = toggle({ label: "region", options: Object.entries(PIXELS).map(([k, l]) => ({ value: k, label: l })), value: region, onChange: (k) => { region = k; update(); } });
    const sLo = slider({ label: "window start", min: 466, max: 940, value: lo, digits: 0, unit: "eV", onInput: (v) => { lo = v; update(); } });
    const sW = slider({ label: "width", min: 6, max: 60, value: wid, digits: 0, unit: "eV", onInput: (v) => { wid = v; update(); } });
    el.append(controls(et, rt), controls(sLo, sW),
      row(col(cS.el, cE.el), h("div", { class: "xp-vec-side", style: { width: "600px" } }, cL.el, card(out),
        legend([["#fff", "spectrum"], [T.truth, "A·E⁻ʳ", "dash"], [T.thread, "spectrum − background"]]), src(SRC_CEVO3))));
    update();
  });
});
