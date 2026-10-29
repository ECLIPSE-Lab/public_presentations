// Station 5, Loss: noise model → likelihood → loss.
//   likelihood  : maximum likelihood for 8 counts of one pixel; L = Π p(yᵢ | λ̂) and −log L vs λ̂
//   loss-curves : MSE vs Poisson NLL for one observed count; curvature = weight of the pixel
//   vst         : Anscombe variance-stabilising transform, with read noise; bias of the naive inverse
//   peak-fit    : a low-count peak fitted with four losses; bias and spread over 150 spectra
//   loss-picker : the loss cheat sheet as a decision tree; the chosen path is the red line
//   sparse-map  : MAP preview; Gaussian prior → L2 (ridge), Laplace prior → L1 (FISTA)
import { register, T, rng, poissonPmf, gaussPdf, linspace, logspace, mean, fmt, sup, nelderMead,
  h, s, row, col, card, controls, caption, readout, slider, toggle, button, legend, chart } from "./core.js";

// Widget-specific CSS (inserted once).
if (!document.getElementById("xp-loss-style")) document.head.append(h("style", { id: "xp-loss-style" }, `
.xp-loss-picker-node { cursor: pointer; }
.xp-loss-picker-node:hover rect { stroke: #fff; }
.xp-loss-picker-edge.on { stroke-dasharray: 1; animation: xp-loss-picker-draw 0.6s ease-out both; }
@keyframes xp-loss-picker-draw { from { stroke-dashoffset: 1; } to { stroke-dashoffset: 0; } }
.xp-loss-picker-leaf { font-size: 22px; line-height: 1.4; }
.xp-loss-picker-leaf h3 { margin: 0 0 12px 0; font-size: 34px; color: var(--thread); font-family: Outfit, sans-serif; }
.xp-loss-picker-leaf dt { color: var(--muted); font-size: 19px; margin-top: 12px; }
.xp-loss-picker-leaf dd { margin: 2px 0 0 0; }
.xp-loss-picker-leaf code { font-family: "JetBrains Mono", Consolas, monospace; font-size: 20px; color: #fde68a; }
.xp-peak-fit-note { font-size: 20px; color: var(--muted); }
`));

const SUB = "₀₁₂₃₄₅₆₇₈₉";
const sub = (n) => String(n).split("").map((c) => SUB[c] ?? c).join("");
const sci = (x, d = 1) => {                                  // 3.2 × 10⁻⁸
  if (!(x > 0)) return "0";
  let e = Math.floor(Math.log10(x)), m = x / 10 ** e;
  if (+m.toFixed(d) >= 10) { m /= 10; e += 1; }
  return `${m.toFixed(d)} × 10${sup(e)}`;
};
const signed = (v, d = 0) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v), d);
// Outer card: as wide as its content, at most 1760 px, centred.
const box = (...c) => { const e = card(...c); Object.assign(e.style, { width: "fit-content", maxWidth: "1760px", margin: "0 auto" }); return e; };
const sized = (w, ...c) => h("div", { class: "xp-col", style: { width: `${w}px` } }, ...c);

// =============================================================================================
// 1. likelihood — 8 counts of one pixel in 8 frames, y ~ Poisson(λ = 4), seed 7 by default.
// The model assigns p(yᵢ | λ̂) to each count (the stem heights). The likelihood is the product
// of the stems; −log L is a sum and has its minimum at λ̂ = ȳ for both models. The Gaussian model
// uses σ² = ȳ fixed, so its −log L is an exact parabola; the Poisson −log L is skewed.
// =============================================================================================
register("likelihood", (el) => {
  const N = 8, LAM_TRUE = 4, KMAX = 20, XD = [-0.6, 20.6];
  let seed = 7, y = [], ybar = 0, model = "Poisson", lam = 2;
  const sd = () => Math.sqrt(Math.max(ybar, 0.25));
  const p = (m, k, l) => (m === "Poisson" ? poissonPmf(k, l) : gaussPdf(k, l, sd()));
  const nll = (m, l) => y.reduce((a, k) => a - Math.log(p(m, k, l)), 0);
  const COL = { Poisson: T.pois, Gaussian: T.gauss };

  const top = chart({ width: 1130, height: 360, x: { domain: XD, label: "count k (e⁻ in this pixel, one frame)" },
    y: { domain: [0, 0.3], label: "p(k | λ̂)" }, title: "The model and the 8 observed counts" });
  const bot = chart({ width: 1130, height: 350, x: { domain: XD, label: "model rate λ̂ (e⁻)" },
    y: { domain: [0, 1], label: "−log L" }, title: "Negative log-likelihood −log L = −Σ log p(yᵢ | model)" });
  const bar = chart({ width: 560, height: 330, x: { domain: [0, 10], label: "−log p(yᵢ | λ̂)" },
    y: { domain: [7.6, -0.6], ticks: [0, 1, 2, 3, 4, 5, 6, 7], grid: false, tickFormat: (i) => `y${sub(i + 1)} = ${y[i]}` },
    margin: { l: 100 }, title: "Each count adds −log p to the loss" });
  const out = readout();
  const LG = linspace(0.15, XD[1], 400);

  function drawTop() {
    const col = COL[model];
    const peak = model === "Poisson" ? Math.max(...Array.from({ length: KMAX + 1 }, (_, k) => p(model, k, lam))) : gaussPdf(0, 0, sd());
    const ymax = Math.max(0.25, 1.15 * peak);
    top.setDomain(null, [0, ymax]);
    top.clear();
    if (model === "Poisson") top.bars(Array.from({ length: KMAX + 1 }, (_, k) => [k - 0.38, k + 0.38, p(model, k, lam)]), { fill: col, opacity: 0.5 });
    else {
      const ks = linspace(XD[0], XD[1], 300), pts = ks.map((k) => [k, p(model, k, lam)]);
      top.area(pts, ks.map((k) => [k, 0]), { fill: col, opacity: 0.22 });
      top.line(pts, { stroke: col, width: 3 });
    }
    top.ruleX(lam, { stroke: T.ink, width: 1.5, dash: "4 6" });
    const left = lam > 11;
    top.text(lam, ymax, `λ̂ = ${fmt(lam, 2)}`, { dx: left ? -8 : 8, dy: 22, fill: T.ink, size: 19, anchor: left ? "end" : "start" });
    const n = {}, seen = {};
    for (const k of y) n[k] = (n[k] || 0) + 1;
    for (const k of y) {                                     // equal counts sit side by side
      const j = (seen[k] = (seen[k] ?? -1) + 1), xk = k + (j - (n[k] - 1) / 2) * 0.22, pv = p(model, k, lam);
      top.line([[xk, 0], [xk, pv]], { stroke: T.thread, width: 4 });
      top.dots([[xk, pv]], { r: 5, fill: T.thread });
      top.dots([[xk, 0]], { r: 8, fill: T.data, stroke: T.bg, strokeWidth: 2, top: true });
    }
    top.note(model === "Poisson" ? "bars: Poisson pmf" : "curve: Gaussian, σ² = ȳ", { corner: "tr", fill: col });
    top.note("● observed count, stem = p(yᵢ | λ̂)", { corner: "tr", line: 1, fill: T.thread, size: 19 });
  }

  function drawBot() {
    const both = ["Poisson", "Gaussian"].map((m) => ({ m, v: LG.map((l) => nll(m, l)) }));
    const lo = Math.min(nll("Poisson", ybar), nll("Gaussian", ybar));
    bot.setDomain(null, [Math.floor(lo) - 3, Math.floor(lo) + 42]);
    bot.clear();
    for (const { m, v } of both.sort((a) => (a.m === model ? 1 : -1))) {
      const on = m === model;
      bot.line(LG.map((l, i) => [l, v[i]]), { stroke: COL[m], width: on ? 4.5 : 2.5, dash: on ? null : "8 7", opacity: on ? 1 : 0.55 });
    }
    const yTop = bot.y.invert(bot.m.t);
    bot.ruleX(ybar, { stroke: T.truth, width: 2.5, dash: "6 6" });
    bot.text(ybar, yTop, `minimum at λ̂ = ȳ = ${fmt(ybar, 2)}`, { fill: T.truth, dx: 10, dy: 24, size: 19 });
    bot.dots([[ybar, nll(model, ybar)]], { r: 7, fill: T.truth });
    const v = nll(model, lam);                               // above the plot: pin the dot to the top edge
    bot.dots([[lam, Math.min(v, yTop)]], { r: 10, fill: T.thread, stroke: "#fff", strokeWidth: 2, top: true });
    if (v > yTop) bot.text(lam, yTop, `↑ ${fmt(v, 1)}`, { dx: lam > 11 ? -16 : 16, dy: 22, fill: T.thread, size: 19, anchor: lam > 11 ? "end" : "start" });
    bot.note("Poisson", { corner: "br", line: 1, fill: T.pois, weight: model === "Poisson" ? 700 : 400 });
    bot.note("Gaussian (σ² = ȳ fixed)", { corner: "br", line: 0, fill: T.gauss, weight: model === "Gaussian" ? 700 : 400 });
  }

  function drawBar() {
    bar.setDomain();                                         // tick labels show the current counts
    bar.clear();
    y.forEach((k, i) => {
      const v = -Math.log(p(model, k, lam));
      bar.rect(0, i - 0.32, Math.min(v, 10), i + 0.32, { fill: T.thread, opacity: 0.8 });
      bar.text(Math.min(v, 10), i, v > 10 ? `${fmt(v, 1)} ▸` : fmt(v, 1), { dx: v > 10 ? -8 : 8, dy: 7, size: 18, anchor: v > 10 ? "end" : "start" });
    });
  }

  function update() {
    drawTop(); drawBot(); drawBar();
    const L = y.reduce((a, k) => a * p(model, k, lam), 1), nl = nll(model, lam), nmin = nll(model, ybar);
    const d = nl - nmin;
    out.set(`L = Π p(yᵢ | λ̂) = <b>${sci(L)}</b> (tiny → take −log)<br>` +
      `−log L = <b>${fmt(nl, 1)}</b> = Σ bars, ` +
      (d < 0.05 ? `<span class="good">at the minimum</span>` : `<span class="warn">+${fmt(d, 1)} above the minimum</span>`) +
      `<br>Minimum at λ̂ = ȳ = <span class="good">${fmt(ybar, 2)}</span> (true λ = ${LAM_TRUE})`);
  }

  const sl = slider({ label: "model rate λ̂", min: 0.2, max: 15, value: lam, digits: 2, unit: "e⁻", onInput: (v) => { lam = v; update(); } });
  const tg = toggle({ label: "model", value: model, onChange: (v) => { model = v; update(); },
    options: [{ value: "Poisson", label: "Poisson", color: T.pois }, { value: "Gaussian", label: "Gaussian (σ² = ȳ)", color: T.gauss }] });
  const nb = button("new data", () => { seed += 1; sample(); update(); });
  const sample = () => { const r = rng(seed); y = Array.from({ length: N }, () => r.poisson(LAM_TRUE)); ybar = mean(y); };
  sample();
  el.append(box(row(col(top.el, bot.el), sized(560, controls(sl, tg, nb), bar.el, out))));
  update();
});

// =============================================================================================
// 2. loss-curves — one pixel with observed count x. MSE (x − λ̂)² has curvature 2 for every x.
// The Poisson NLL λ̂ − x log λ̂ (shifted to 0 at its minimum λ̂ = x) has curvature x/λ̂² = 1/x at the
// minimum: the pull of a pixel on the fit scales with 1/variance = 1/x. For x = 0 the NLL is λ̂.
// =============================================================================================
register("loss-curves", (el) => {
  let x = 2;
  const LG = linspace(0.005, 16, 700), YMAX = 30;
  const mk = (name, formula, colr) => {                      // title with the formula in regular weight (hats render)
    const c = chart({ width: 650, height: 540, margin: { t: 46 },
      x: { domain: [0, 16], label: "predicted rate λ̂ (e⁻)" }, y: { domain: [0, YMAX], label: "loss (0 at the minimum)" } });
    c.el.append(s("text", { x: c.m.l, y: 30, class: "xp-ctitle" }, `${name}   `,
      s("tspan", { fill: colr, style: "font-weight:400" }, formula)));
    return c;
  };
  const cm = mk("MSE", "(x − λ̂)²", T.gauss), cp = mk("Poisson NLL", "λ̂ − x log λ̂ (shifted)", T.pois);
  const cw = chart({ width: 360, height: 230, title: "Weight of this pixel",
    x: { domain: [0, 1.25], label: "relative weight", ticks: [0, 0.5, 1] },
    y: { domain: [1.6, -0.6], ticks: [0, 1], tickFormat: (i) => ["MSE", "Poisson"][i], grid: false }, margin: { l: 100, b: 56 } });
  const cx = chart({ width: 360, height: 270, title: "Weight vs observed count", margin: { l: 60, b: 56 },
    x: { domain: [0, 15.5], label: "observed count x (e⁻)" }, y: { domain: [0, 1.35], ticks: [0, 0.5, 1] } });
  const out = readout();
  const XS = linspace(1, 15, 15);

  function panel(c, f, col, curv) {
    c.clear();
    c.ruleX(x, { stroke: T.data, width: 2, dash: "4 6" });
    c.line(LG.map((l) => [l, f(l)]), { stroke: col, width: 5 });
    if (curv > 0) c.line(LG.map((l) => [l, 0.5 * curv * (l - x) ** 2]), { stroke: "#fff", width: 2, dash: "7 7", opacity: 0.85 });
    c.dots([[x, 0]], { r: 10, fill: T.thread, stroke: "#fff", strokeWidth: 2, top: true });
  }

  function update() {
    const pmin = x > 0 ? x - x * Math.log(x) : 0;
    panel(cm, (l) => (x - l) ** 2, T.gauss, 2);
    panel(cp, (l) => l - x * Math.log(l) - pmin, T.pois, x > 0 ? 1 / x : 0);
    cm.note("curvature 2, for every x", { corner: "tr", fill: T.gauss });
    cm.note("dashed: parabola at the minimum", { corner: "tr", line: 1, fill: T.muted, size: 18 });
    cp.note(x > 0 ? `curvature 1/x = ${fmt(1 / x, 2)}` : "x = 0: linear penalty λ̂", { corner: "tr", fill: T.pois });
    cp.note(x > 0 ? "dashed: parabola at the minimum" : "no parabola: minimum at the edge", { corner: "tr", line: 1, fill: T.muted, size: 18 });
    cw.clear();
    cw.rect(0, -0.3, 1, 0.3, { fill: T.gauss });
    cw.text(1, 0, "1", { dx: -8, dy: 7, size: 19, anchor: "end", fill: T.bg });
    if (x > 0) {
      cw.rect(0, 0.7, 1 / x, 1.3, { fill: T.pois });
      const inside = 1 / x > 0.6;
      cw.text(1 / x, 1, `1/x = ${fmt(1 / x, 2)}`, { dx: inside ? -8 : 8, dy: 7, size: 19, anchor: inside ? "end" : "start", fill: inside ? T.bg : T.ink });
    } else cw.text(0, 1, "no minimum: constant pull", { dx: 8, dy: 7, size: 19, fill: T.pois });
    cx.clear();
    cx.line([[0, 1], [15.5, 1]], { stroke: T.gauss, width: 3 });
    cx.line(XS.map((v) => [v, 1 / v]), { stroke: T.pois, width: 3 });
    cx.dots(XS.map((v) => [v, 1 / v]), { r: 4, fill: T.pois });
    if (x > 0) cx.dots([[x, 1 / x]], { r: 9, fill: "none", stroke: T.thread, strokeWidth: 3, top: true });
    cx.note("MSE", { corner: "tr", line: 0, fill: T.gauss, size: 19 });
    cx.text(5, 0.2, "Poisson 1/x", { dx: 6, dy: -16, fill: T.pois, size: 19 });
    out.set(x > 0
      ? `Observed count x = <b>${x}</b>. Pull of this pixel on the fit (curvature at the minimum): MSE <b>2</b>, Poisson <b>1/x = ${fmt(1 / x, 2)}</b>.<br>` +
        `The Poisson variance is x, so the Poisson NLL weights each pixel by 1/x: <span class="good">bright pixels get less weight</span>, MSE weights all pixels the same.`
      : `Observed count x = <b>0</b>. The Poisson NLL is λ̂: a <span class="good">linear penalty</span> with slope 1 that pushes λ̂ to 0.<br>` +
        `MSE gives λ̂², which is flat near 0: <span class="warn">MSE barely pulls a dark pixel down</span>.`);
  }

  const sl = slider({ label: "observed count x", min: 0, max: 15, step: 1, value: x, digits: 0, unit: "e⁻", onInput: (v) => { x = v; update(); } });
  el.append(box(controls(sl), row(cm.el, cp.el, sized(360, cw.el, cx.el, caption("Weight = curvature at the minimum, MSE scaled to 1."))), out));
  update();
});

// =============================================================================================
// 3. vst — y = Poisson(λ) + N(0, σ²) in e⁻ (gain 1). Generalised Anscombe z = 2√(y + 3/8 + σ²)
// has std ≈ 1 for λ ≳ 4. The std curves come from exact sums over k (σ = 0) and a Gauss-Hermite-like
// grid over the read noise (σ > 0). The naive inverse (z̄/2)² − 3/8 − σ² is biased low at low λ.
// Histograms use 20 000 samples with a fixed seed.
// =============================================================================================
const GQ = (() => {                                          // 49-point grid for E over ξ ~ N(0, 1)
  const xs = linspace(-6, 6, 49), w = xs.map((v) => Math.exp(-v * v / 2)), sw = w.reduce((a, b) => a + b, 0);
  return { xs, ws: w.map((v) => v / sw) };
})();
const anscombe = (y, sig) => 2 * Math.sqrt(Math.max(0, y + 0.375 + sig * sig));
function vstMoments(lam, sig) {
  const kmax = Math.ceil(lam + 10 * Math.sqrt(lam) + 10);
  let m1 = 0, m2 = 0;
  for (let k = 0; k <= kmax; k++) {
    const p = poissonPmf(k, lam);
    if (p < 1e-15) continue;
    if (sig === 0) { const z = anscombe(k, 0); m1 += p * z; m2 += p * z * z; continue; }
    for (let j = 0; j < GQ.xs.length; j++) { const z = anscombe(k + sig * GQ.xs[j], sig), w = p * GQ.ws[j]; m1 += w * z; m2 += w * z * z; }
  }
  return { mean: m1, sd: Math.sqrt(Math.max(0, m2 - m1 * m1)), inv: (m1 / 2) ** 2 - 0.375 - sig * sig };
}

register("vst", (el) => {
  let lam = 2, sig = 0;
  const NS = 20000, LG = logspace(0.1, 50, 70), curves = new Map();
  const cr = chart({ width: 550, height: 510, title: "Raw signal y", x: { domain: [0, 1], label: "y (e⁻)" }, y: { domain: [0, 1], label: "probability density" } });
  const cz = chart({ width: 550, height: 510, title: "After the VST", x: { domain: [0, 1], label: "z = 2√(y + 3/8 + σ²)" }, y: { domain: [0, 1], label: "probability density" } });
  const cs = chart({ width: 550, height: 510, title: "Standard deviation vs λ",
    x: { domain: [0.1, 50], type: "log", label: "mean signal λ (e⁻)", ticks: [0.1, 0.3, 1, 3, 10, 30] },
    y: { domain: [0.2, 10], type: "log", label: "standard deviation", ticks: [0.3, 1, 3, 10] } });
  const out = readout();

  const uniform = (v, lo, hi, bw) => {                       // bins [x0, x1, count]
    const nb = Math.max(1, Math.round((hi - lo) / bw)), cnt = new Float64Array(nb);
    for (const t of v) { const b = Math.floor((t - lo) / bw); if (b >= 0 && b < nb) cnt[b]++; }
    return Array.from(cnt, (n, i) => [lo + i * bw, lo + (i + 1) * bw, n]);
  };
  const hist = (c, bins, xd, col) => {                       // draws densities, returns the peak density
    const dens = bins.map(([a, b, n]) => [a, b, n / (NS * (b - a))]), peak = Math.max(...dens.map((d) => d[2]));
    c.setDomain(xd, [0, 1.18 * peak]);
    c.clear();
    c.bars(dens, { fill: col, opacity: 0.6 });
    return peak;
  };
  const bracket = (c, m, sdv, yv, col, txt) => {             // ±1 std as a bar with end ticks
    c.line([[m - sdv, yv], [m + sdv, yv]], { stroke: col, width: 4 });
    for (const e of [m - sdv, m + sdv]) c.line([[e, yv * 0.96], [e, yv * 1.04]], { stroke: col, width: 4 });
    c.text(m + sdv, yv, txt, { dx: 10, dy: 7, size: 19, fill: col });
  };
  const curve = () => {
    if (!curves.has(sig)) curves.set(sig, LG.map((l) => vstMoments(l, sig).sd));
    return curves.get(sig);
  };

  function update() {
    const r = rng(5), ys = new Float64Array(NS), zs = new Float64Array(NS);
    for (let i = 0; i < NS; i++) { ys[i] = r.poisson(lam) + sig * r.normal(); zs[i] = anscombe(ys[i], sig); }
    const my = mean(ys), sy = Math.sqrt(mean(Array.from(ys, (v) => (v - my) ** 2)));
    const mz = mean(zs), sz = Math.sqrt(mean(Array.from(zs, (v) => (v - mz) ** 2)));
    const sth = Math.sqrt(lam + sig * sig);
    // raw histogram: integer bins without read noise
    const lo = sig === 0 ? -0.5 : Math.min(-0.5, Math.floor(lam - 4.5 * sth) - 0.5), hi = Math.max(5.5, Math.ceil(lam + 4.5 * sth) + 0.5);
    const yr = hist(cr, uniform(ys, lo, hi, sig === 0 ? 1 : Math.max(0.25, sth / 6)), [lo, hi], T.pois);
    bracket(cr, my, sy, 0.5 * yr, T.thread, "±1 std");
    cr.note(`std = ${fmt(sy, 2)} e⁻`, { corner: "tr", fill: T.pois });
    cr.note(`√(λ + σ²) = ${fmt(sth, 2)}`, { corner: "tr", line: 1, fill: T.muted, size: 19 });
    // z of integer counts: each count k fills [z(k − ½), z(k + ½)], so the z histogram has no comb
    let zb;
    if (sig === 0) {
      const kc = new Map();
      for (const v of ys) kc.set(v, (kc.get(v) || 0) + 1);
      zb = [...kc].map(([k, n]) => [anscombe(k - 0.5, 0), anscombe(k + 0.5, 0), n]);
    } else zb = uniform(zs, mz - 4.5, mz + 4.5, 0.2);
    const yz = hist(cz, zb, [mz - 4.5, mz + 4.5], T.gauss);
    bracket(cz, mz, sz, 0.5 * yz, T.thread, "±1 std");
    bracket(cz, mz, 1, 0.25 * yz, T.ink, "±1 (target)");
    cz.note(`std = ${fmt(sz, 2)}`, { corner: "tl", fill: T.gauss });
    const sv = curve(), m = vstMoments(lam, sig), bias = 100 * (m.inv / lam - 1);
    cs.clear();
    cs.ruleY(1, { stroke: T.muted, width: 2, dash: "6 6" });
    cs.line(LG.map((l) => [l, Math.sqrt(l + sig * sig)]), { stroke: T.pois, width: 4 });
    cs.line(LG.map((l, i) => [l, sv[i]]), { stroke: T.gauss, width: 4 });
    cs.ruleX(lam, { stroke: T.thread, width: 2, dash: "4 6" });
    cs.dots([[lam, sth], [lam, m.sd]], { r: 8, fill: T.thread, stroke: "#fff", strokeWidth: 2 });
    cs.note("raw y: √(λ + σ²)", { corner: "tl", fill: T.pois });
    cs.note("after VST: ≈ 1", { corner: "tl", line: 1, fill: T.gauss });
    const flat = Math.abs(m.sd - 1) < 0.05;
    out.set(`Raw std <b>${fmt(sy, 2)}</b> e⁻ ≈ √(λ + σ²): it grows with the signal. After the VST: std <b>${fmt(sz, 2)}</b> ` +
      (flat ? `<span class="good">≈ 1, stable</span>` : `<span class="warn">not yet 1 (needs λ ≳ 4)</span>`) + `.<br>` +
      `Naive inverse (z̄/2)² − 3/8${sig > 0 ? " − σ²" : ""} = ${fmt(m.inv, 2)} e⁻ for λ = ${fmt(lam, 2)}: bias ` +
      `<span class="${Math.abs(bias) > 5 ? "bad" : "good"}">${signed(bias, 1)} %</span>` +
      (Math.abs(bias) > 5 ? ` → at low counts use the exact NLL.` : ` → small at this dose.`));
  }

  const sl = slider({ label: "mean signal λ", min: 0.1, max: 50, value: lam, log: true, digits: 2, unit: "e⁻", onInput: (v) => { lam = v; update(); } });
  const ss = slider({ label: "read noise σ", min: 0, max: 5, step: 0.1, value: sig, digits: 1, unit: "e⁻", onInput: (v) => { sig = v; update(); } });
  el.append(box(controls(sl, ss, caption("y = Poisson(λ) + N(0, σ²), gain 1")), row(cr.el, cz.el, cs.el), out));
  update();
});

// =============================================================================================
// 4. peak-fit — a Gaussian peak (a = 1, width s = 0.05, background b = 0.05) on 64 channels,
// rate = dose · (a·exp(−(E − 0.5)²/2s²) + b). Each spectrum is fitted with four losses (Nelder-Mead,
// bounds as penalties, same as img/make_figures.py). Statistics: 150 spectra per dose, computed in
// chunks with setTimeout and cached per dose. Only the Poisson NLL stays unbiased at low dose.
// =============================================================================================
const PK_E = linspace(0, 1, 64), PK_TRUE = [1.0, 0.05, 0.05], PK_REPS = 150;
const KINDS = ["MSE", "weighted χ²", "Anscombe + MSE", "Poisson NLL"];
const PK_COL = [T.gauss, T.warn, T.violet, T.pois], PK_DASH = [null, "12 7", "4 6", null];
const PK_DOSES = [1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200];

const peakModel = ([a, b, w], dose) => PK_E.map((E) => dose * (a * Math.exp(-((E - 0.5) ** 2) / (2 * w * w)) + b));
const peakTotal = (q) => peakModel(q, 1).reduce((acc, v) => acc + v, 0);
function fitPeak(x, dose, kind) {
  const loss = (q) => {
    const a = Math.max(q[0], 0), b = Math.max(q[1], 1e-6), w = Math.min(Math.max(q[2], 0.01), 0.3);
    const pen = 1e6 * ((a - q[0]) ** 2 + (b - q[1]) ** 2 + (w - q[2]) ** 2);
    const lam = peakModel([a, b, w], dose);
    let L = 0;
    for (let i = 0; i < lam.length; i++) {
      if (kind === "MSE") L += (x[i] - lam[i]) ** 2 / dose ** 2;
      else if (kind === "weighted χ²") L += (x[i] - lam[i]) ** 2 / Math.max(x[i], 1) / dose;
      else if (kind === "Anscombe + MSE") L += (2 * Math.sqrt(x[i] + 3 / 8) - 2 * Math.sqrt(lam[i] + 3 / 8)) ** 2;
      else L += (lam[i] - x[i] * Math.log(Math.max(lam[i], 1e-12))) / dose;
    }
    return L + pen;
  };
  const q = nelderMead(loss, [1.2, 0.1, 0.06], [0.3, 0.05, 0.02], 1500);
  return [Math.max(q[0], 0), Math.max(q[1], 1e-6), Math.min(Math.max(q[2], 0.01), 0.3)];
}

// Statistics of the four fits at one dose, built up in chunks (same seeds as the old peakStats).
const PK_JOBS = new Map();
function peakJob(dose) {
  if (!PK_JOBS.has(dose)) PK_JOBS.set(dose, { dose, r: rng(13 + Math.round(dose * 10)), truth: peakModel(PK_TRUE, dose), i: 0, est: KINDS.map(() => []), out: null });
  return PK_JOBS.get(dose);
}
function peakStep(job, budgetMs) {
  const t0 = performance.now();
  while (job.i < PK_REPS && performance.now() - t0 < budgetMs) {
    const x = job.truth.map((v) => job.r.poisson(v));
    KINDS.forEach((k, j) => job.est[j].push(fitPeak(x, job.dose, k)));
    job.i++;
  }
  if (job.i < PK_REPS || job.out) return;
  const totTrue = peakTotal(PK_TRUE);
  job.out = KINDS.map((kind, j) => {
    const tot = job.est[j].map((q) => peakTotal(q) / totTrue), w = job.est[j].map((q) => q[2]);
    const mt = mean(tot), mw = mean(w);
    return { kind, bias: 100 * (mt - 1), se: 100 * Math.sqrt(mean(tot.map((v) => (v - mt) ** 2)) / PK_REPS),
      sd: 100 * Math.sqrt(mean(w.map((v) => (v - mw) ** 2))) / PK_TRUE[2] };
  });
  job.est = null;
}

register("peak-fit", (el, api) => {
  let dose = 5, draw = 0, timer = null, last = null;
  const snap = (v) => PK_DOSES.reduce((b, d) => (Math.abs(Math.log(d / v)) < Math.abs(Math.log(b / v)) ? d : b));
  const fd = (d) => fmt(d, d % 1 ? 1 : 0);
  const spec = chart({ width: 790, height: 530, title: "One spectrum and the four fits",
    x: { domain: [0, 1], label: "energy loss (a.u.)" }, y: { domain: [0, 1], label: "counts per channel" } });
  const cat = { domain: [3.6, -0.6], ticks: [0, 1, 2, 3], grid: false };
  const cb = chart({ width: 470, height: 530, title: "Bias of the peak area", margin: { l: 178 },
    x: { domain: [-25, 5], label: "bias of integrated counts (%)", tickCount: 3 }, y: { ...cat, tickFormat: (i) => KINDS[i] } });
  const cv = chart({ width: 400, height: 530, title: "Spread of the width", margin: { l: 24 },
    x: { domain: [0, 60], label: "relative std of width (%)", tickCount: 4 }, y: { ...cat, axis: false } });
  const out = readout();

  function drawSpec() {
    const r = rng(101 + draw), x = peakModel(PK_TRUE, dose).map((v) => r.poisson(v));
    const fits = KINDS.map((k) => fitPeak(x, dose, k)), tr = peakModel(PK_TRUE, dose);
    spec.setDomain(null, [0, 1.15 * Math.max(...x, ...tr, 1)]);
    spec.clear();
    const hw = 0.5 / 63;
    spec.bars(x.map((v, i) => [PK_E[i] - hw, PK_E[i] + hw, v]), { fill: T.data, opacity: 0.35 });
    fits.forEach((q, j) => spec.line(peakModel(q, dose).map((v, i) => [PK_E[i], v]), { stroke: PK_COL[j], width: 3.5, dash: PK_DASH[j] }));
    spec.line(tr.map((v, i) => [PK_E[i], v]), { stroke: T.truth, width: 3, dash: "2 7" });
    spec.note(`peak: ${fd(dose * 1.05)} e⁻ per channel`, { corner: "tl" });
  }

  const verdict = (st) => {
    const biased = st.filter((v) => Math.abs(v.bias) > Math.max(1, 2 * v.se));
    const fair = st.filter((v) => !biased.includes(v));
    const win = (fair.length ? fair : st).reduce((b, v) => (fair.length ? v.sd < b.sd : Math.abs(v.bias) < Math.abs(b.bias)) ? v : b);
    const ties = fair.filter((v) => v.sd <= 1.05 * win.sd);   // equal spread within 5 %
    return { biased, win, ties };
  };

  function drawBars() {
    const job = peakJob(dose), st = job.out || last;
    cb.clear(); cv.clear();
    if (st) {
      const { win } = verdict(st), op = job.out ? 0.9 : 0.25;
      const lo = Math.min(-5, 1.15 * Math.min(...st.map((v) => v.bias - 2 * v.se)));
      cb.setDomain([lo, Math.max(-0.4 * lo, 1.15 * Math.max(...st.map((v) => v.bias + 2 * v.se)))]);
      cv.setDomain([0, 1.3 * Math.max(...st.map((v) => v.sd))]);
      st.forEach((v, j) => {
        const hl = job.out && v === win ? { stroke: T.thread, "stroke-width": 4 } : {};
        cb.rect(0, j - 0.32, v.bias, j + 0.32, { fill: PK_COL[j], opacity: op, ...hl });
        cv.rect(0, j - 0.32, v.sd, j + 0.32, { fill: PK_COL[j], opacity: op, ...hl });
        if (!job.out) return;
        cb.line([[v.bias - 2 * v.se, j], [v.bias + 2 * v.se, j]], { stroke: T.ink, width: 2 });
        cb.text(Math.max(0, v.bias + 2 * v.se), j, `${signed(Math.abs(v.bias) < 0.5 ? 0 : v.bias, 0)} %`, { dx: 10, dy: 7, size: 19 });
        cv.text(v.sd, j, `${fmt(v.sd, 0)} %`, { dx: 10, dy: 7, size: 19 });
      });
      cb.ruleX(0, { stroke: T.ink, width: 2, dash: null });
    }
    if (!job.out) {
      const msg = `computing… ${Math.round(100 * job.i / PK_REPS)} %`;
      cb.note(msg, { corner: "bl", fill: T.warn }); cv.note(msg, { corner: "bl", fill: T.warn });
      out.set(`Fitting ${PK_REPS} spectra at dose ${fd(dose)} with the four losses…`);
      return;
    }
    last = job.out;
    const { biased, win, ties } = verdict(job.out);
    const list = biased.map((v) => `<b>${v.kind}</b> (${signed(v.bias, 0)} %)`).join(", ");
    const best = ties.length > 1 ? `${ties.map((v) => `<span class="good">${v.kind}</span>`).join(" and ")} tie: unbiased, width spread ${fmt(win.sd, 0)} %.`
      : `<span class="good">${win.kind}</span>, unbiased with the smallest width spread (${fmt(win.sd, 0)} %).`;
    out.set(`Dose ${fd(dose)}: ` + (biased.length ? `${list} <span class="bad">${biased.length > 1 ? "bias" : "biases"} the peak area</span>. ` : `<span class="good">no loss is biased</span>. `) +
      `Best: ${best}`);
  }

  // Work loop: the current dose first; then, while the slide is on screen, the other doses.
  function pump() {
    timer = null;
    const job = peakJob(dose);
    if (!job.out) { peakStep(job, 25); drawBars(); }
    else {
      if (!api.isActive()) return;
      const next = [...PK_DOSES].sort((a, b) => Math.abs(Math.log(a / dose)) - Math.abs(Math.log(b / dose))).find((d) => !peakJob(d).out);
      if (next === undefined) return;
      peakStep(peakJob(next), 25);
    }
    timer = setTimeout(pump, 0);
  }
  const kick = () => { if (!timer) timer = setTimeout(pump, 0); };

  const sl = slider({ label: "dose", min: 1, max: 200, value: dose, log: true, unit: "e⁻", format: (v) => fd(snap(v)),
    onInput: (v) => { const d = snap(v); if (d === dose) return; dose = d; drawSpec(); drawBars(); kick(); } });
  const nb = button("new spectrum", () => { draw += 1; drawSpec(); });
  const lg = legend([...KINDS.map((k, j) => [PK_COL[j], k, PK_DASH[j] ? (j === 1 ? "dash" : "dot") : null]), [T.truth, "true rate", "dot"]]);
  el.append(box(controls(sl, nb, lg), row(spec.el, cb.el, cv.el), out,
    h("div", { class: "xp-peak-fit-note" }, `Bars: ${PK_REPS} simulated spectra per dose, each fitted with the four losses; whiskers ±2 standard errors. Red frame: best loss.`)));
  drawSpec(); drawBars(); kick();
  api.onShow(kick);
});

// =============================================================================================
// 5. loss-picker — the cheat sheet as a decision tree. Click a question, an answer or a loss.
// Each leaf names the likelihood, the loss, a PyTorch class and an EM example. 8 leaves.
// =============================================================================================
const PICK = [
  { id: "q0", q: ["What do you", "predict?"], x: 560, y: 46 },
  { id: "q1", q: ["Does the detector", "count single e⁻?"], x: 300, y: 178, p: "q0", a: "pixel intensity", say: "pixel intensity" },
  { id: "ce", x: 700, y: 178, p: "q0", a: "class label", say: "class label", t: ["cross-entropy", "categorical"],
    name: "Cross-entropy", noise: "class c ~ Categorical(p̂₁ … p̂ₖ), p̂ = softmax(logits)", lik: "p(c | x) = p̂_c",
    loss: "−log p̂_c (summed over samples)", torch: "nn.CrossEntropyLoss()", ex: "Phase or orientation class of each 4D-STEM diffraction pattern." },
  { id: "bce", x: 960, y: 178, p: "q0", a: "mask", say: "mask", t: ["BCE / Dice", "Bernoulli"],
    name: "BCE / Dice", noise: "mask pixel m ~ Bernoulli(p̂)", lik: "p(m) = p̂ᵐ (1 − p̂)¹⁻ᵐ",
    loss: "−[m log p̂ + (1 − m) log(1 − p̂)]; add Dice 1 − 2|M∩M̂| / (|M| + |M̂|) for small objects",
    torch: "nn.BCEWithLogitsLoss() (+ a Dice term)", ex: "Segmentation of nanoparticles or defects; the foreground is often only a few % of the pixels." },
  { id: "pois", x: 120, y: 310, p: "q1", a: "yes", say: "counting detector", t: ["Poisson NLL", "Poisson"],
    name: "Poisson NLL", noise: "y ~ Poisson(λ)", lik: "p(y | λ) = λʸ exp(−λ) / y!",
    loss: "λ̂ − y log λ̂ (+ const.)", torch: "nn.PoissonNLLLoss(log_input=True)", ex: "Counting detector (4D-STEM, low-dose cryo-EM): 0 to 5 e⁻ per pixel, no read noise." },
  { id: "q2", q: ["Read noise comparable", "to the signal?"], x: 430, y: 310, p: "q1", a: "no", say: "integrating detector" },
  { id: "vst", x: 215, y: 442, p: "q2", a: "yes", say: "read noise ≈ signal", t: ["VST + MSE", "or Poisson–Gaussian"],
    name: "VST + MSE, or mixed Poisson–Gaussian NLL", noise: "y = Poisson(λ) + N(0, σ²)", lik: "p(y | λ) = Σₖ Poisson(k | λ) · N(y − k | 0, σ²)",
    loss: "(2√(y + 3/8 + σ²) − 2√(λ̂ + 3/8 + σ²))², or Gaussian NLL with variance λ̂ + σ²",
    torch: "nn.MSELoss() on z = 2√(y + 3/8 + σ²), or nn.GaussianNLLLoss() with var = λ̂ + σ²", ex: "Low-dose image on a CCD or CMOS camera with a few e⁻ read noise." },
  { id: "q3", q: ["Noise ≈ Gaussian.", "What about σ?"], x: 655, y: 442, p: "q2", a: "no", say: "small read noise, high dose" },
  { id: "mse", x: 130, y: 590, p: "q3", a: "same σ", say: "same σ everywhere", t: ["MSE", "Gaussian, same σ"],
    name: "MSE", noise: "y ~ N(μ, σ²), the same σ for all pixels", lik: "p(y | μ) ∝ exp(−(y − μ)² / 2σ²)",
    loss: "(y − μ̂)²", torch: "nn.MSELoss()", ex: "High-dose HAADF image from an integrating detector: σ is about the same everywhere." },
  { id: "wmse", x: 395, y: 590, p: "q3", a: "σᵢ known", say: "σᵢ known per pixel", t: ["weighted MSE", "Gaussian, σᵢ known"],
    name: "Weighted MSE", noise: "yᵢ ~ N(μᵢ, σᵢ²) with known σᵢ", lik: "p(yᵢ | μᵢ) ∝ exp(−(yᵢ − μᵢ)² / 2σᵢ²)",
    loss: "Σ (yᵢ − μ̂ᵢ)² / σᵢ²", torch: "((y − ŷ)**2 / var).mean() (no built-in class)", ex: "EELS spectrum after background subtraction; σᵢ from error propagation." },
  { id: "gnll", x: 660, y: 590, p: "q3", a: "σ from model", say: "σ predicted by the model", t: ["Gaussian NLL", "σ from the model"],
    name: "Gaussian NLL", noise: "y ~ N(μ̂, σ̂²), the model predicts μ̂ and σ̂", lik: "p(y | μ̂, σ̂) = N(y | μ̂, σ̂²)",
    loss: "(y − μ̂)² / 2σ̂² + log σ̂", torch: "nn.GaussianNLLLoss()", ex: "A denoiser that also outputs its own uncertainty map." },
  { id: "huber", x: 925, y: 590, p: "q3", a: "outliers", say: "outliers", t: ["Huber / L1", "Laplace"],
    name: "Huber / L1", noise: "heavy tails: y − μ ~ Laplace(0, b)", lik: "p(y | μ) ∝ exp(−|y − μ| / b)",
    loss: "|y − μ̂|, or Huber: quadratic near 0, linear for large errors", torch: "nn.L1Loss() or nn.HuberLoss(delta=…)",
    ex: "X-ray hits and hot pixels in a CCD frame: a few huge errors must not dominate the fit." },
];

register("loss-picker", (el) => {
  const W = 1110, H = 640, BW = 236, BH = 72, by = Object.fromEntries(PICK.map((n) => [n.id, n]));
  const isLeaf = (n) => !n.q;
  const pathTo = (id) => { const p = []; for (let n = by[id]; n; n = by[n.p]) p.unshift(n.id); return p; };
  const firstLeaf = (id) => isLeaf(by[id]) ? id : firstLeaf(PICK.find((n) => n.p === id).id);
  let leaf = "pois";

  const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, style: "display:block" });
  const gE = s("g"), gL = s("g"), gN = s("g");
  svg.append(gE, gL, gN);
  const panel = h("div", { class: "xp-loss-picker-leaf" }), out = readout();

  const pick = (id) => {
    const n = by[id];
    leaf = isLeaf(n) ? id : (pathTo(leaf).includes(id) ? leaf : firstLeaf(id));
    update();
  };

  function update() {
    const path = new Set(pathTo(leaf));
    gE.replaceChildren(); gL.replaceChildren(); gN.replaceChildren();
    for (const n of PICK) {
      if (n.p) {
        const p = by[n.p], on = path.has(n.id);
        const x1 = p.x, y1 = p.y + BH / 2, x2 = n.x, y2 = n.y - BH / 2, my = (y1 + y2) / 2;
        gE.append(s("path", { d: `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`, fill: "none", pathLength: 1,
          class: `xp-loss-picker-edge${on ? " on" : ""}`, stroke: on ? T.thread : T.faint, "stroke-width": on ? 7 : 2.5, "stroke-linecap": "round" }));
        const lx = x1 + 0.62 * (x2 - x1), ly = y1 + 0.55 * (y2 - y1), wd = 13 + 10 * n.a.length;
        const g = s("g", { class: "xp-loss-picker-node", onclick: () => pick(n.id) },
          s("rect", { x: lx - wd / 2, y: ly - 16, width: wd, height: 32, rx: 16, fill: on ? T.thread : "#1e293b", stroke: on ? T.thread : T.faint }),
          s("text", { x: lx, y: ly + 7, "text-anchor": "middle", "font-size": 19, fill: on ? "#fff" : T.muted, "font-weight": on ? 700 : 400 }, n.a));
        gL.append(g);
      }
      const on = path.has(n.id), sel = n.id === leaf, lf = isLeaf(n), lines = lf ? n.t : n.q;
      const g = s("g", { class: "xp-loss-picker-node", onclick: () => pick(n.id), transform: `translate(${n.x} ${n.y})` },
        s("rect", { x: -BW / 2, y: -BH / 2, width: BW, height: BH, rx: lf ? 36 : 12,
          fill: sel ? T.thread : lf ? "rgba(30,41,59,0.95)" : "rgba(15,23,42,0.95)",
          stroke: on ? T.thread : lf ? T.axis : T.faint, "stroke-width": on ? 3.5 : 1.5 }),
        s("text", { x: 0, y: -6, "text-anchor": "middle", "font-size": lf ? 22 : 20, "font-weight": lf ? 700 : 500, fill: sel ? "#fff" : lf ? T.ink : T.muted }, lines[0]),
        s("text", { x: 0, y: 20, "text-anchor": "middle", "font-size": lf ? 18 : 20, "font-weight": lf ? 400 : 500, fill: sel ? "#ffe4e6" : T.muted }, lines[1]));
      gN.append(g);
    }
    const L = by[leaf];
    panel.replaceChildren(h("h3", {}, L.name), h("dl", {},
      h("dt", {}, "Noise model"), h("dd", {}, L.noise),
      h("dt", {}, "Likelihood"), h("dd", {}, L.lik),
      h("dt", {}, "Loss = −log likelihood"), h("dd", { style: { color: "#fff", fontSize: "24px" } }, L.loss),
      h("dt", {}, "PyTorch"), h("dd", {}, h("code", {}, L.torch)),
      h("dt", {}, "EM example"), h("dd", {}, L.ex)));
    const steps = pathTo(leaf).slice(1, -1).map((id) => by[id].say).concat(by[leaf].say);
    out.set(`Path: ${steps.join(" → ")} → <span class="bad">${L.name}</span>`);
  }

  el.append(box(row(col(svg, caption("Click a question, an answer or a loss. The red line is your path through the cheat sheet.")),
    sized(560, out, panel))));
  update();
});

// =============================================================================================
// 6. sparse-map — MAP: a prior on x becomes a regulariser. Line profile of 20 Å (n = 200, 0.1 Å
// per pixel) over 8 atomic columns (two pairs only 0.8–0.9 Å apart). PSF: Gaussian, σ = 0.6 Å
// (FWHM 1.4 Å), peak 1. y = A x + N(0, σₙ²). Gaussian prior → ridge: (AᵀA + λI) x̂ = Aᵀy (Cholesky).
// Laplace prior with x ≥ 0 → ½‖Ax − y‖² + λ‖x‖₁, solved by FISTA: 300 iterations per update, then
// more in the animation loop until it converges. A column is "found" if a local maximum above 20 %
// of the largest value lies within ±0.3 Å.
// =============================================================================================
register("sparse-map", (el, api) => {
  const N = 200, DX = 0.1, SP = 6, KR = 24;
  const POS = [20, 46, 70, 78, 104, 130, 139, 170], HGT = [0.9, 0.55, 1.0, 0.7, 0.4, 0.85, 0.6, 1.0];
  const K = Array.from({ length: 2 * KR + 1 }, (_, i) => Math.exp(-((i - KR) ** 2) / (2 * SP * SP)));
  const conv = (x) => {                                      // A x; A is symmetric
    const o = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let acc = 0;
      for (let j = Math.max(0, i - KR), e = Math.min(N - 1, i + KR); j <= e; j++) acc += K[j - i + KR] * x[j];
      o[i] = acc;
    }
    return o;
  };
  const xt = new Float64Array(N);
  POS.forEach((p, i) => { xt[p] = HGT[i]; });
  const clean = conv(xt), LIP = K.reduce((a, b) => a + b, 0) ** 2;
  const AtA = (() => {                                       // AᵀA = A·A, column by column
    const M = new Float64Array(N * N), e = new Float64Array(N);
    for (let j = 0; j < N; j++) { e.fill(0); e[j] = 1; const c = conv(conv(e)); for (let i = 0; i < N; i++) M[i * N + j] = c[i]; }
    return M;
  })();

  let sigN = 0.05, lam = 0.1, seed = 3, y, xr, fs;
  const measure = () => { const r = rng(seed); y = clean.map((v) => v + sigN * r.normal()); };
  function ridge() {
    const L = new Float64Array(N * N), b = conv(y), z = new Float64Array(N), x = new Float64Array(N);
    for (let i = 0; i < N; i++) for (let j = 0; j <= i; j++) {
      let acc = AtA[i * N + j] + (i === j ? lam : 0);
      for (let k = 0; k < j; k++) acc -= L[i * N + k] * L[j * N + k];
      L[i * N + j] = i === j ? Math.sqrt(acc) : acc / L[j * N + j];
    }
    for (let i = 0; i < N; i++) { let acc = b[i]; for (let k = 0; k < i; k++) acc -= L[i * N + k] * z[k]; z[i] = acc / L[i * N + i]; }
    for (let i = N - 1; i >= 0; i--) { let acc = z[i]; for (let k = i + 1; k < N; k++) acc -= L[k * N + i] * x[k]; x[i] = acc / L[i * N + i]; }
    return x;
  }
  // FISTA state; warm start keeps the previous x when λ or the data change.
  const fistaReset = () => { fs = { x: fs ? fs.x : new Float64Array(N), v: fs ? Float64Array.from(fs.x) : new Float64Array(N), t: 1, it: 0, done: false }; };
  function fista(iters) {
    const g = 1 / LIP;
    for (let k = 0; k < iters && !fs.done; k++) {
      const r = conv(fs.v);
      for (let i = 0; i < N; i++) r[i] -= y[i];
      const gr = conv(r), xn = new Float64Array(N);
      let dx = 0, nx = 0;
      for (let i = 0; i < N; i++) { xn[i] = Math.max(0, fs.v[i] - g * (gr[i] + lam)); dx += (xn[i] - fs.x[i]) ** 2; nx += xn[i] ** 2; }
      const tn = (1 + Math.sqrt(1 + 4 * fs.t * fs.t)) / 2;
      for (let i = 0; i < N; i++) fs.v[i] = xn[i] + (fs.t - 1) / tn * (xn[i] - fs.x[i]);
      fs.x = xn; fs.t = tn; fs.it++;
      fs.conv = fs.it > 50 && dx < 1e-10 * (nx + 1e-12);
      if (fs.conv || fs.it >= 6000) fs.done = true;
    }
  }
  const score = (x) => {
    let mx = 0;
    for (const v of x) mx = Math.max(mx, v);
    const pk = [];
    for (let i = 1; i < N - 1; i++) if (x[i] > 0.2 * mx && x[i] >= x[i - 1] && x[i] > x[i + 1]) pk.push(i);
    const used = new Set();
    for (const p of POS) { const m = pk.find((q) => Math.abs(q - p) <= 3 && !used.has(q)); if (m !== undefined) used.add(m); }
    let e = 0, n = 0;
    for (let i = 0; i < N; i++) { e += (x[i] - xt[i]) ** 2; n += xt[i] ** 2; }
    return { found: used.size, extra: pk.length - used.size, err: Math.sqrt(e / n), pk };
  };

  const c = chart({ width: 1710, height: 545, x: { domain: [0, (N - 1) * DX], label: "position along the line profile (Å)" },
    y: { domain: [-0.25, 1.4], label: "intensity (a.u.)" } });
  const out = readout();
  const stems = (x, attrs, dotR, dotFill, ring) => {
    for (let i = 0; i < N; i++) if (x[i] > 1e-3) {
      c.line([[i * DX, 0], [i * DX, x[i]]], attrs);
      c.dots([[i * DX, x[i]]], ring ? { r: dotR, fill: "none", stroke: dotFill, strokeWidth: 3 } : { r: dotR, fill: dotFill });
    }
  };

  function draw() {
    const sr = score(xr), sl1 = score(fs.x);
    c.clear();
    c.ruleY(0, { stroke: T.axis, width: 1, dash: null });
    c.dots(Array.from(y, (v, i) => [i * DX, v]), { r: 3.5, fill: T.data, opacity: 0.6 });
    stems(xt, { stroke: T.truth, width: 2.5, dash: "5 5" }, 9, T.truth, true);
    c.line(Array.from(xr, (v, i) => [i * DX, v]), { stroke: T.gauss, width: 3.5 });
    stems(fs.x, { stroke: T.violet, width: 4 }, 5, T.violet);
    const tag = (r) => `${r.found} of 8 columns` + (r.extra ? `, ${r.extra} false peak${r.extra > 1 ? "s" : ""}` : "");
    c.note(`L2 (ridge): ${tag(sr)}`, { corner: "tl", fill: T.gauss, weight: 700 });
    c.note(`L1 (FISTA): ${tag(sl1)}`, { corner: "tl", line: 1, fill: T.violet, weight: 700 });
    c.note(fs.conv ? `FISTA converged after ${fs.it} iterations` : fs.done ? `FISTA stopped at ${fs.it} iterations` : `FISTA iteration ${fs.it}…`, { corner: "tl", line: 2, fill: T.muted, size: 18 });
    const cls = (r) => (r.found === 8 && !r.extra ? "good" : "bad");
    out.set(`Columns found: L2 <span class="${cls(sr)}">${sr.found} of 8</span>${sr.extra ? ` (+${sr.extra} false)` : ""}, ` +
      `L1 <span class="${cls(sl1)}">${sl1.found} of 8</span>${sl1.extra ? ` (+${sl1.extra} false)` : ""}. ` +
      `Error ‖x̂ − x‖ / ‖x‖: L2 <b>${fmt(sr.err, 2)}</b>, L1 <b>${fmt(sl1.err, 2)}</b>.<br>` +
      (sl1.found === 8 && !sl1.extra ? `<span class="good">L2 spreads intensity, L1 finds the columns.</span>`
        : sl1.found < 8 ? `<span class="warn">λ too large: L1 drops weak columns.</span>` : `<span class="warn">λ too small: L1 fits noise peaks.</span>`) +
      ` Gaussian prior → λ‖x‖₂², Laplace prior → λ‖x‖₁.`);
  }
  function solve() { xr = ridge(); fistaReset(); fista(300); draw(); }

  api.animate(() => { if (!fs.done) { fista(150); draw(); } });
  const sn = slider({ label: "noise σ", min: 0.005, max: 0.2, value: sigN, log: true, digits: 3, onInput: (v) => { sigN = v; measure(); solve(); } });
  const sl = slider({ label: "regularisation λ", min: 0.003, max: 3, value: lam, log: true, digits: 3, onInput: (v) => { lam = v; solve(); } });
  const nb = button("new noise", () => { seed += 1; measure(); solve(); });
  const lg = legend([[T.data, "measurement y = A x + noise", "dot"], [T.truth, "true columns", "dash"], [T.gauss, "L2 / ridge (Gaussian prior)"], [T.violet, "L1 / FISTA, x ≥ 0 (Laplace prior)"]]);
  el.append(box(controls(sn, sl, nb), lg, c.el, out));
  measure(); solve();
});
