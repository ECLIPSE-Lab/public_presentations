// SVD group: what a matrix does to space (svd-2x2), and the best rank-k image (lowrank-image).
import { register, T, fmt, clamp, h, s, row, col, card, controls, readout, slider, toggle, button, legend, chart, imageCanvas } from "./core.js";
import { loadData } from "./data.js"; // eslint-disable-line no-unused-vars

if (!document.getElementById("xp-svd-style")) {
  document.head.append(h("style", { id: "xp-svd-style" }, `
.xp-svd-mat { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 22px; display: inline-grid; grid-template-columns: auto auto;
  gap: 2px 18px; padding: 4px 14px; border-left: 3px solid #94a3b8; border-right: 3px solid #94a3b8; border-radius: 8px; vertical-align: middle; }
.xp-svd-eq { display: flex; align-items: center; gap: 8px; font-size: 26px; flex-wrap: nowrap; }
.xp-svd-eq .lab { font-size: 18px; color: #94a3b8; text-align: center; margin-top: 4px; }
.xp-svd-side { width: 760px; display: flex; flex-direction: column; gap: 14px; }
.xp-svd-side .xp-slider input[type=range] { width: 250px; }
.xp-svd-stage { display: flex; gap: 10px; }
.xp-svd-stage span { padding: 6px 14px; border-radius: 10px; background: rgba(30,41,59,.9); color: #94a3b8; font-size: 20px; }
.xp-svd-stage span.on { background: #f43f5e; color: #fff; font-weight: 700; }
`));
}

// Closed-form SVD of a 2 x 2 matrix [[a, b], [c, d]] = R(phi) · diag(s1, s2) · R(theta).
// s2 carries the sign of det(A) (negative = reflection), so the three steps can be animated.
export function svd2(a, b, c, d) {
  const E = (a + d) / 2, F = (a - d) / 2, G = (c + b) / 2, H = (c - b) / 2;
  const Q = Math.hypot(E, H), R = Math.hypot(F, G);
  const a1 = Math.atan2(G, F), a2 = Math.atan2(H, E);
  return { s1: Q + R, s2: Q - R, theta: (a2 - a1) / 2, phi: (a2 + a1) / 2 };
}
const rot = (t) => [Math.cos(t), -Math.sin(t), Math.sin(t), Math.cos(t)];
const mul = (A, B) => [A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3], A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3]];
const app = (M, [x, y]) => [M[0] * x + M[1] * y, M[2] * x + M[3] * y];
const deg = (r) => r * 180 / Math.PI;

// ---------------------------------------------------------------------------------------------
// svd-2x2. A 2 x 2 matrix acts on a crystal lattice (atomic columns) and on the unit circle.
// The stage slider plays A x = U Σ Vᵀ x step by step: rotate (Vᵀ), stretch (Σ), rotate (U).
// EM reading: in 4D-STEM strain mapping, A maps the reference lattice vectors onto the measured
// ones. Its SVD gives the lattice rotation (U Vᵀ) and the principal stretches (σ1, σ2), i.e.
// the principal strains σ − 1. The presets are typical distortions of a scanned image.
// ---------------------------------------------------------------------------------------------
const PRESETS = {
  "strain": { label: "strain + rotation", A: (() => { const R = rot(6 * Math.PI / 180); return mul(R, mul(rot(0.5), mul([1.06, 0, 0, 0.97], rot(-0.5)))); })(),
    note: "Strained crystal: 6 % tension, 3 % compression, 6° lattice rotation. σ − 1 are the principal strains; U Vᵀ is the rotation." },
  "shear": { label: "scan shear", A: [1, 0.25, 0, 1],
    note: "Fast-scan drift shears a STEM image. A shear is not a stretch, but SVD writes it as rotate · stretch · rotate." },
  "iso": { label: "magnification error", A: [1.1, 0, 0, 1.1],
    note: "A 10 % calibration error stretches all directions equally: σ1 = σ2. Every direction is a singular direction." },
  "ill": { label: "nearly singular", A: [1, 0.9, 0.95, 0.9],
    note: "Two almost parallel columns: σ2 ≈ 0. The lattice collapses towards a line. κ = σ1/σ2 is large: Act 6." },
  "rank1": { label: "rank 1 (projection)", A: [0.5, 0.5, 0.5, 0.5],
    note: "σ2 = 0: every point lands on one line. A 2-D input becomes 1-D: information is lost and cannot be inverted." },
};

register("svd-2x2", (el, api) => {
  const W = 860, H = 760, SC = 120, CX = W / 2, CY = H / 2;
  let A = PRESETS.strain.A.slice(), stage = 3, playing = false;
  const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "xp-chart" });
  svg.append(s("rect", { x: 0, y: 0, width: W, height: H, rx: 16, fill: "rgba(5,8,15,0.55)", stroke: "rgba(255,255,255,0.09)" }));
  for (let k = -3; k <= 3; k++) {
    svg.append(s("line", { x1: CX + k * SC, x2: CX + k * SC, y1: 0, y2: H, stroke: T.grid }));
    svg.append(s("line", { y1: CY + k * SC, y2: CY + k * SC, x1: 0, x2: W, stroke: T.grid }));
  }
  const gGhost = s("g"), gLat = s("g"), gVec = s("g");
  svg.append(gGhost, gLat, gVec);
  const P = ([x, y]) => [CX + SC * x, CY - SC * y];

  const lattice = [];
  for (let i = -6; i <= 6; i++) for (let j = -6; j <= 6; j++) {
    const p = [0.5 * i, 0.5 * j];
    if (Math.hypot(...p) <= 3.05) lattice.push(p);
  }
  const circle = Array.from({ length: 121 }, (_, i) => [Math.cos(i / 120 * 2 * Math.PI), Math.sin(i / 120 * 2 * Math.PI)]);
  // ghost: the undistorted lattice
  for (const p of lattice) { const [x, y] = P(p); gGhost.append(s("circle", { cx: x, cy: y, r: 4, fill: T.faint })); }
  gGhost.append(s("path", { d: circle.map((p, i) => `${i ? "L" : "M"}${P(p).join(",")}`).join(""), fill: "none", stroke: T.faint, "stroke-width": 2, "stroke-dasharray": "6 6" }));

  const stageBox = h("div", { class: "xp-svd-stage" }, ["x", "Vᵀx: rotate", "ΣVᵀx: stretch", "UΣVᵀx: rotate"].map((t) => h("span", {}, t)));
  const eq = h("div");
  const info = readout("", "");
  const note = h("div", { class: "xp-caption", style: { fontSize: "23px", color: "#e2e8f0", minHeight: "3em" } });

  function current(t) {                              // the map applied at stage t in [0, 3]
    const { s1, s2, theta, phi } = svd2(...A);
    const u1 = clamp(t, 0, 1), u2 = clamp(t - 1, 0, 1), u3 = clamp(t - 2, 0, 1);
    return mul(rot(phi * u3), mul([1 + (s1 - 1) * u2, 0, 0, 1 + (s2 - 1) * u2], rot(theta * u1)));
  }
  function draw() {
    const M = current(stage), { s1, s2, theta, phi } = svd2(...A);
    gLat.replaceChildren(); gVec.replaceChildren();
    const cp = circle.map((p) => P(app(M, p)));
    gLat.append(s("path", { d: cp.map((p, i) => `${i ? "L" : "M"}${p.join(",")}`).join(""), fill: "rgba(244,63,94,0.10)", stroke: T.thread, "stroke-width": 3 }));
    for (const p of lattice) {
      const [x, y] = P(app(M, p));
      gLat.append(s("circle", { cx: x, cy: y, r: 7, fill: "#e2e8f0", opacity: 0.92 }));
    }
    // right singular vectors v1, v2 (columns of V = R(-theta)) carried along by the map
    const V = rot(-theta), v1 = [V[0], V[2]], v2 = [V[1], V[3]];
    for (const [v, c, lab] of [[v1, T.thread, "v₁"], [v2, T.gauss, "v₂"]]) {
      const [x1, y1] = P(app(M, v));
      gVec.append(s("line", { x1: CX, y1: CY, x2: x1, y2: y1, stroke: c, "stroke-width": 6, "stroke-linecap": "round" }),
        s("circle", { cx: x1, cy: y1, r: 9, fill: c }),
        s("text", { x: x1 + 14, y: y1 - 12, fill: c, "font-size": 26, "font-weight": 700, class: "xp-ctext" }, stage >= 2.99 ? (lab === "v₁" ? "σ₁u₁" : "σ₂u₂") : lab));
    }
    [...stageBox.children].forEach((e, i) => e.classList.toggle("on", Math.round(stage) === i));
    const S1 = Math.abs(s1), S2 = Math.abs(s2), flip = s2 < 0;
    const m = (vals) => h("div", { class: "xp-svd-mat" }, vals.map((v) => h("span", {}, fmt(v, 2))));
    const rotDeg = deg(phi + theta);
    eq.replaceChildren(h("div", { class: "xp-svd-eq" },
      h("div", {}, m(A), h("div", { class: "lab" }, "A")), "=",
      h("div", {}, m(rot(phi)), h("div", { class: "lab" }, `U: ${fmt(deg(phi), 0)}°`)), "·",
      h("div", {}, m([S1, 0, 0, flip ? -S2 : S2]), h("div", { class: "lab" }, "Σ")), "·",
      h("div", {}, m(rot(theta)), h("div", { class: "lab" }, `Vᵀ: ${fmt(deg(theta), 0)}°`))));
    const kappa = S2 > 1e-9 ? S1 / S2 : Infinity;
    info.set(`σ₁ = <b>${fmt(S1, 3)}</b> &nbsp; σ₂ = <b>${fmt(S2, 3)}</b> ${flip ? "<span class='warn'>(det &lt; 0: mirror)</span>" : ""}<br>` +
      `principal strains σ − 1: <b>${fmt(100 * (S1 - 1), 1)} %</b>, <b>${fmt(100 * (S2 - 1), 1)} %</b><br>` +
      `rotation U Vᵀ: <b>${fmt(rotDeg, 1)}°</b> &nbsp; area |det A| = σ₁σ₂ = <b>${fmt(S1 * S2, 3)}</b><br>` +
      `condition number κ = σ₁/σ₂ = <span class="${kappa > 20 ? "bad" : "good"}">${Number.isFinite(kappa) ? fmt(kappa, kappa < 10 ? 2 : 0) : "∞"}</span>`);
  }

  const sl = ["a₁₁", "a₁₂", "a₂₁", "a₂₂"].map((lab, i) => slider({ label: lab, min: -1.5, max: 1.5, value: A[i], digits: 2,
    onInput: (v) => { A[i] = v; note.textContent = "Any real matrix has an SVD — try to find one that is not rotate · stretch · rotate."; draw(); } }));
  const preset = toggle({ options: Object.entries(PRESETS).map(([k, p]) => ({ value: k, label: p.label })), value: "strain",
    onChange: (k) => { A = PRESETS[k].A.slice(); sl.forEach((x, i) => x.set(A[i], false)); note.textContent = PRESETS[k].note; stage = 0; playing = true; draw(); } });
  const stageSl = slider({ label: "step", min: 0, max: 3, value: 3, digits: 1, onInput: (v) => { stage = v; playing = false; draw(); } });
  const play = button("▶ play U Σ Vᵀ", () => { stage = 0; playing = true; }, "primary");
  note.textContent = PRESETS.strain.note;

  el.append(controls(preset),
    row(svg, h("div", { class: "xp-svd-side" },
      card(stageBox, h("div", { style: { display: "flex", gap: "20px", alignItems: "center", marginTop: "10px" } }, play, stageSl)),
      card(eq, info),
      card(h("div", { style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 20px" } }, sl), note),
      legend([[T.thread, "v₁ → σ₁u₁ (strongest direction)"], [T.gauss, "v₂ → σ₂u₂"]]))));
  draw();
  api.onShow(() => { stage = 0; playing = true; });
  api.animate((dt) => {
    if (!playing) return;
    stage = Math.min(3, stage + dt * 0.7);
    stageSl.set(stage, false); draw();
    if (stage >= 3) playing = false;
  });
});

// ---------------------------------------------------------------------------------------------
// lowrank-image. The real HAADF image (256 x 256, SrFeOₓ / LaAlO₃ interface) is a matrix X. Its
// truncated SVD X_k = Σ_{j≤k} σ_j u_j v_jᵀ is the best rank-k image (Eckart–Young), and the error
// ‖X − X_k‖_F equals √(Σ_{j>k} σ_j²) exactly. A crystal aligned with the scan is nearly low rank;
// the interface and the noise need the high-rank terms.
// ---------------------------------------------------------------------------------------------
register("lowrank-image", (el, api) => {
  const wait = h("div", { class: "xp-caption" }, "loading data …");
  el.append(wait);
  loadData("haadf").then((d) => {
    wait.remove();
    const n = d.img.shape[0], K = d.U.shape[1], sv = Array.from(d.sv), X = d.img;
    let k = 8, playing = false;
    const SZ = 400;
    const imO = imageCanvas({ n, size: SZ, label: "X: HAADF image (256 × 256)" });
    const imK = imageCanvas({ n, size: SZ, label: " " });
    const imR = imageCanvas({ n, size: SZ, label: "X − Xₖ (residual)" });
    const imT = imageCanvas({ n, size: SZ, label: " " });
    const cS = chart({ width: 820, height: 380, x: { domain: [1, n], type: "log", label: "index j" }, y: { domain: [sv[n - 1] * 0.8, sv[0] * 1.3], type: "log", label: "singular value σⱼ" } });
    const out = readout();
    const lo = -2.5, hi = 3;
    imO.draw(X, lo, hi, "gray");
    const Xk = new Float64Array(n * n), R = new Float64Array(n * n), Tk = new Float64Array(n * n);
    function update() {
      Xk.fill(0);
      for (let j = 0; j < k; j++) {
        const s_ = sv[j];
        for (let r = 0; r < n; r++) {
          const u = d.U[r * K + j] * s_;
          if (!u) continue;
          const o = r * n, vo = j * n;
          for (let c = 0; c < n; c++) Xk[o + c] += u * d.Vt[vo + c];
        }
      }
      let e2 = 0;
      for (let i = 0; i < n * n; i++) { R[i] = X[i] - Xk[i]; e2 += R[i] * R[i]; }
      const j = k - 1;
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) Tk[r * n + c] = sv[j] * d.U[r * K + j] * d.Vt[j * n + c];
      let tmax = 0; for (const v of Tk) tmax = Math.max(tmax, Math.abs(v));
      imK.draw(Xk, lo, hi, "gray"); imK.setLabel(`Xₖ: best rank-${k} image`);
      imR.draw(R, -1.5, 1.5, "div");
      imT.draw(Tk, -tmax, tmax, "div"); imT.setLabel(`term ${k} alone: σ${sub(k)} u${sub(k)} v${sub(k)}ᵀ`);
      let tail = 0, all = 0;
      for (let i = 0; i < n; i++) { all += sv[i] ** 2; if (i >= k) tail += sv[i] ** 2; }
      cS.clear();
      cS.area(sv.slice(0, k).map((v, i) => [i + 1, v]), sv.slice(0, k).map((_, i) => [i + 1, sv[n - 1] * 0.8]), { fill: T.thread, opacity: 0.25 });
      cS.line(sv.map((v, i) => [i + 1, v]), { stroke: "#fff", width: 3 });
      cS.ruleX(k, { stroke: T.thread, dash: "6 5" });
      cS.note(`kept: ${k} of ${n}`, { corner: "tr", fill: T.thread });
      const store = k * (2 * n + 1);
      out.set(`‖X − Xₖ‖<sub>F</sub> computed: <b>${fmt(Math.sqrt(e2), 3)}</b> &nbsp; √(Σ<sub>j&gt;k</sub> σⱼ²): <span class="good">${fmt(Math.sqrt(tail), 3)}</span><br>` +
        `variance kept: <b>${fmt(100 * (1 - tail / all), 1)} %</b> &nbsp; numbers stored: k(2n + 1) = <b>${store.toLocaleString("en-US")}</b> of ${(n * n).toLocaleString("en-US")} (${fmt(100 * store / (n * n), 1)} %)`);
    }
    const sub = (v) => String(v).split("").map((c) => "₀₁₂₃₄₅₆₇₈₉"[+c]).join("");
    const sl = slider({ label: "rank k", min: 1, max: K, value: k, step: 1, digits: 0, onInput: (v) => { k = Math.round(v); playing = false; update(); } });
    const play = button("▶ k = 1 … 96", () => { k = 1; playing = true; }, "primary");
    let acc = 0;
    api.animate((dt) => {
      if (!playing) return;
      acc += dt;
      if (acc < 0.12) return;
      acc = 0; k = Math.min(K, k + (k < 12 ? 1 : 4)); sl.set(k, false); update();
      if (k >= K) playing = false;
    });
    el.append(controls(sl, play), row(imO, imK, imR, imT),
      row(cS.el, h("div", { style: { width: "860px", display: "flex", flexDirection: "column", gap: "12px" } }, card(out),
        h("div", { class: "xp-caption", html: "Data: SrFeOₓ / LaAlO₃ interface, HAADF 200 kV, STEM-Bank-26 (Hollyer &amp; Stach 2026), CC BY 4.0." }))));
    update();
  });
});
