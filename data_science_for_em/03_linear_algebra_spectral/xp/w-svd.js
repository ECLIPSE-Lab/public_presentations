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

// ---------------------------------------------------------------------------------------------
// strain-4d. Real 4D-STEM strain map of ion-irradiated Al ([001], 40 x 40 positions; Mills et al.
// 2023, Zenodo 7041997, CC BY 4.0). Per position the disk positions were fitted by least squares,
// p = c + h g1 + k g2 (xp/prep/make_strain.py). Here, live: A = G G_ref⁻¹, F = A⁻ᵀ, SVD of F.
// Click a scan position: its pattern with the fitted lattice, and F drawn as an exaggerated ellipse.
// ---------------------------------------------------------------------------------------------
const STRAIN_SRC = "Data: ion-irradiated Al, [001], 4D-STEM; Mills et al., Acta Mater. 2023; Zenodo 7041997, CC BY 4.0.";
register("strain-4d", (el) => {
  const wait = h("div", { class: "xp-caption" }, "loading data …"); el.append(wait);
  loadData("strain4d").then((d) => {
    wait.remove();
    const N = 1600, n = 40, G = d.G, Gr = d.Gref, C = d.c, ok = d.ok;
    // physical frame: flip the pattern y axis (pixels go down), P = diag(1, −1)
    const inv = (m) => { const det = m[0] * m[3] - m[1] * m[2]; return [m[3] / det, -m[1] / det, -m[2] / det, m[0] / det]; };
    const flip = (m) => [m[0], -m[1], -m[2], m[3]];                       // P m P
    const tr = (m) => [m[0], m[2], m[1], m[3]];
    const GrInv = inv(flip([Gr[0], Gr[1], Gr[2], Gr[3]]));
    const Fs = [];
    for (let i = 0; i < N; i++) {
      const A = mul(flip([G[4 * i], G[4 * i + 1], G[4 * i + 2], G[4 * i + 3]]), GrInv);
      Fs.push(tr(inv(A)));                                                  // F = A⁻ᵀ
    }
    const avg = (i) => {                                                    // 3 x 3 mean of the valid neighbours
      const y = Math.floor(i / n), x = i % n, o = [0, 0, 0, 0]; let c = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx; if (yy < 0 || xx < 0 || yy >= n || xx >= n || !ok[yy * n + xx]) continue;
        const f = Fs[yy * n + xx]; for (let k = 0; k < 4; k++) o[k] += f[k]; c++;
      }
      return o.map((v) => v / Math.max(1, c));
    };
    let mapKey = "rot", smooth = false, sel = 4 * n + 16;          // start inside the rotated region (top)
    const Fof = (i) => (smooth ? avg(i) : Fs[i]);
    const MAPS = {
      bf: { label: "bright field", f: (i) => d.bf[i], cmap: "gray" },
      exx: { label: "ε<sub>xx</sub>", f: (i) => 100 * (Fof(i)[0] - 1), unit: "%" },
      eyy: { label: "ε<sub>yy</sub>", f: (i) => 100 * (Fof(i)[3] - 1), unit: "%" },
      exy: { label: "ε<sub>xy</sub>", f: (i) => 100 * (Fof(i)[1] + Fof(i)[2]) / 2, unit: "%" },
      rot: { label: "rotation", f: (i) => deg((Fof(i)[2] - Fof(i)[1]) / 2), unit: "°" },
      shear: { label: "max shear (σ₁−σ₂)/2", f: (i) => { const v = svd2(...Fof(i)); return 100 * (v.s1 - v.s2) / 2; }, unit: "%", cmap: "magma" },
    };
    const imM = imageCanvas({ n, size: 430, label: "" });
    const mk = s("svg", { width: 430, height: 430, viewBox: "0 0 430 430", style: "position:absolute;left:0;bottom:0;pointer-events:none" });
    const mkR = s("rect", { width: 430 / n * 2, height: 430 / n * 2, fill: "none", stroke: T.truth, "stroke-width": 3 }); mk.append(mkR);
    imM.style.position = "relative"; imM.append(mk); imM.canvas.style.cursor = "crosshair";
    const PS = 430, pat = imageCanvas({ n: 96, size: PS, label: "diffraction pattern + least-squares lattice" });
    const ov = s("svg", { width: PS, height: PS, viewBox: `0 0 ${PS} ${PS}`, style: "position:absolute;left:0;bottom:0;pointer-events:none" });
    pat.style.position = "relative"; pat.append(ov);
    const EW = 330, ex = s("svg", { width: EW, height: EW, viewBox: `0 0 ${EW} ${EW}`, class: "xp-chart", style: "border-radius:14px;background:rgba(5,8,15,0.55)" });
    const out = readout();
    const [ox, oy] = d.pat_origin, bin = d.pat_bin, sc = PS / 96;
    const P = (x, y) => [(x - ox) / bin * sc, (y - oy) / bin * sc];
    function drawMap() {
      const M = MAPS[mapKey], v = new Float64Array(N);
      for (let i = 0; i < N; i++) v[i] = ok[i] || mapKey === "bf" ? M.f(i) : NaN;
      const fin = Array.from(v).filter(Number.isFinite).sort((a, b) => a - b), q = (p) => fin[Math.floor(p * (fin.length - 1))];
      let lo, hi, cm = M.cmap || "div";
      if (cm === "div") { const m = Math.max(Math.abs(q(0.02)), Math.abs(q(0.98))); lo = -m; hi = m; } else { lo = q(0.01); hi = q(0.99); }
      imM.draw(v.map((x) => (Number.isFinite(x) ? x : (lo + hi) / 2)), lo, hi, cm);
      imM.setLabel(`${M.label}${M.unit ? ` &nbsp;(${cm === "div" ? "±" + fmt(hi, 2) : fmt(lo, 2) + "–" + fmt(hi, 2)} ${M.unit})` : ""}${smooth && mapKey !== "bf" ? " · 3×3 mean" : ""}`);
    }
    function drawSel() {
      const iy = Math.floor(sel / n) & ~1, ix = (sel % n) & ~1, i = iy * n + ix, k = (iy / 2) * 20 + ix / 2;
      mkR.setAttribute("x", ix * 430 / n); mkR.setAttribute("y", iy * 430 / n);
      pat.draw(d.pat.subarray(k * 9216, (k + 1) * 9216), 0, 255, "gray");
      ov.replaceChildren();
      const g = [G[4 * i], G[4 * i + 1], G[4 * i + 2], G[4 * i + 3]], c = [C[2 * i], C[2 * i + 1]];
      const at = (hh, kk) => P(c[0] + g[0] * hh + g[1] * kk, c[1] + g[2] * hh + g[3] * kk);
      for (const [hh, kk] of [[0, 0], ...d.hk]) {
        const [x, y] = at(hh, kk); if (x < 0 || y < 0 || x > PS || y > PS) continue;
        ov.append(s("circle", { cx: x, cy: y, r: 38 / bin * sc, fill: "none", stroke: T.truth, "stroke-width": 2, opacity: 0.85 }));
      }
      const [x0, y0] = at(0, 0);
      for (const [hh, kk, lab] of [[1, 0, "g₁"], [0, 1, "g₂"]]) {
        const [x1, y1] = at(hh, kk);
        ov.append(s("line", { x1: x0, y1: y0, x2: x1, y2: y1, stroke: T.thread, "stroke-width": 4 }),
          s("text", { x: x1 + 10, y: y1 - 10, fill: T.thread, "font-size": 24, "font-weight": 700, class: "xp-ctext" }, lab));
      }
      // F as an ellipse, exaggerated: I + X (F − I)
      const F = Fof(i), X = 100, Fx = [1 + X * (F[0] - 1), X * F[1], X * F[2], 1 + X * (F[3] - 1)];
      const v = svd2(...F), vx = svd2(...Fx), R0 = 90, cx = EW / 2, cy = EW / 2;
      const circ = [], ell = [];
      for (let t = 0; t <= 64; t++) { const a = 2 * Math.PI * t / 64, p = [Math.cos(a), Math.sin(a)], q = app(Fx, p);
        circ.push(`${cx + R0 * p[0]},${cy - R0 * p[1]}`); ell.push(`${cx + R0 * q[0]},${cy - R0 * q[1]}`); }
      const U = rot(vx.phi), ax = [[U[0], U[2]], [U[1], U[3]]];
      ex.replaceChildren(
        s("polyline", { points: circ.join(" "), fill: "none", stroke: T.faint, "stroke-width": 2, "stroke-dasharray": "6 6" }),
        s("polyline", { points: ell.join(" "), fill: "rgba(52,211,153,0.12)", stroke: T.truth, "stroke-width": 3 }),
        ...ax.map((u, j) => { const sv = j ? vx.s2 : vx.s1; return s("line", { x1: cx - R0 * sv * u[0], y1: cy + R0 * sv * u[1], x2: cx + R0 * sv * u[0], y2: cy - R0 * sv * u[1], stroke: j ? T.gauss : T.thread, "stroke-width": 3 }); }),
        s("text", { x: 12, y: 28, fill: T.muted, "font-size": 19 }, "F on the unit circle, strain × 100"));
      const fm = (m) => `<span class="xp-svd-mat"><span>${fmt(m[0], 4)}</span><span>${fmt(m[1], 4)}</span><span>${fmt(m[2], 4)}</span><span>${fmt(m[3], 4)}</span></span>`;
      out.set(`position (${ix}, ${iy}): ${d.n[i]} disks, fit rms ${fmt(d.rms[i], 2)} px<br>` +
        `F = A⁻ᵀ = ${fm(F)}<br>` +
        `SVD: <span style="color:${T.thread}">σ₁ − 1 = ${fmt(100 * (v.s1 - 1), 2)} %</span> &nbsp; <span style="color:${T.gauss}">σ₂ − 1 = ${fmt(100 * (v.s2 - 1), 2)} %</span><br>` +
        `rotation U Vᵀ = <b>${fmt(deg(v.phi + v.theta), 3)}°</b>${smooth ? " &nbsp;(3×3 mean)" : ""}`);
    }
    imM.canvas.addEventListener("pointerdown", (e) => {
      const r = imM.canvas.getBoundingClientRect(), x = Math.floor((e.clientX - r.left) / r.width * n), y = Math.floor((e.clientY - r.top) / r.height * n);
      sel = clamp(y, 0, n - 1) * n + clamp(x, 0, n - 1); drawSel();
    });
    const mt = toggle({ label: "map", options: Object.entries(MAPS).map(([k, m]) => ({ value: k, label: m.label })), value: mapKey, onChange: (k) => { mapKey = k; drawMap(); } });
    const st = toggle({ label: "pixels", options: [{ value: "1", label: "single" }, { value: "3", label: "3×3 mean" }], value: "1", onChange: (v) => { smooth = v === "3"; drawMap(); drawSel(); } });
    el.append(controls(mt, st),
      row(col(imM, h("div", { class: "xp-src" , style: { fontSize: "17px", color: "#64748b", maxWidth: "430px" }, html: STRAIN_SRC })), pat,
        h("div", { class: "xp-svd-side", style: { width: "620px" } }, row(ex, legend([[T.thread, "σ₁ axis"], [T.gauss, "σ₂ axis"], [T.faint, "unstrained", "dash"]])), card(out))));
    drawMap(); drawSel();
  });
});
