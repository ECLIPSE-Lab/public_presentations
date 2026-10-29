// Core of the DSEM Week 2 explainers: theme, random numbers, DOM and SVG builders,
// controls, a small SVG chart, an image canvas, FFT, and the widget registry.
// No external dependencies: the deck runs offline in the lecture hall.

// ---------------------------------------------------------------------------------------------
// Theme. One colour has one meaning in every widget.
// ---------------------------------------------------------------------------------------------
export const T = {
  bg: "#0b0f19", panel: "rgba(20,25,40,0.72)", ink: "#e2e8f0", muted: "#94a3b8", faint: "#475569",
  grid: "rgba(148,163,184,0.16)", axis: "#64748b",
  thread: "#f43f5e",   // the red line of the lecture; also "the current choice"
  gauss: "#60a5fa",    // Gaussian noise, read noise, MSE
  pois: "#fb923c",     // Poisson noise, shot noise, Poisson NLL
  truth: "#34d399",    // ground truth, physical model
  warn: "#fbbf24",     // fixed-pattern noise, warnings, Nyquist limit
  pink: "#f472b6", violet: "#a78bfa", data: "#cbd5e1",
};
export const SERIES = [T.gauss, T.pois, T.truth, T.warn, T.pink, T.violet];

// ---------------------------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------------------------
export function rng(seed = 1) {
  let s = seed >>> 0;
  const u = () => {                                   // mulberry32, uniform in [0, 1)
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), s | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const normal = () => {                              // Marsaglia polar method
    if (spare !== null) { const v = spare; spare = null; return v; }
    let a, b, r;
    do { a = 2 * u() - 1; b = 2 * u() - 1; r = a * a + b * b; } while (r >= 1 || r === 0);
    const f = Math.sqrt(-2 * Math.log(r) / r);
    spare = b * f;
    return a * f;
  };
  const poisson = (lam) => {
    if (!(lam > 0)) return 0;
    if (lam < 30) {                                   // Knuth: multiply uniforms
      const L = Math.exp(-lam);
      let k = 0, p = 1;
      do { k++; p *= u(); } while (p > L);
      return k - 1;
    }
    // Hormann (1993), transformed rejection with squeeze (PTRS)
    const sl = Math.sqrt(lam), ll = Math.log(lam);
    const b = 0.931 + 2.53 * sl, a = -0.059 + 0.02483 * b;
    const ia = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
    for (;;) {
      const U = u() - 0.5, V = u(), us = 0.5 - Math.abs(U);
      const k = Math.floor((2 * a / us + b) * U + lam + 0.43);
      if (us >= 0.07 && V <= vr) return k;
      if (k < 0 || (us < 0.013 && V > us)) continue;
      if (Math.log(V) + Math.log(ia) - Math.log(a / (us * us) + b) <= -lam + k * ll - logFactorial(k)) return k;
    }
  };
  return { u, normal, poisson };
}

export function logFactorial(k) {
  if (k < 20) { let s = 0; for (let i = 2; i <= k; i++) s += Math.log(i); return s; }
  const n = k + 1;                                    // Stirling series for log Gamma(n)
  return (n - 0.5) * Math.log(n) - n + 0.5 * Math.log(2 * Math.PI) + 1 / (12 * n) - 1 / (360 * n ** 3);
}
export const poissonPmf = (k, lam) => Math.exp(k * Math.log(lam) - lam - logFactorial(k));
export const gaussPdf = (x, mu, sd) => Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));

export const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
export const logspace = (a, b, n) => linspace(Math.log10(a), Math.log10(b), n).map((t) => 10 ** t);
export const mean = (a) => { let s = 0; for (const v of a) s += v; return s / a.length; };
export const variance = (a) => { const m = mean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return s / a.length; };
export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const fmt = (x, d = 1) => Number.isFinite(x)
  ? x.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }) : "–";

// Nelder-Mead simplex minimiser.
export function nelderMead(f, x0, step, iters = 600) {
  const n = x0.length;
  let pts = [x0.slice()];
  for (let i = 0; i < n; i++) { const p = x0.slice(); p[i] += step[i]; pts.push(p); }
  let vals = pts.map(f);
  for (let it = 0; it < iters; it++) {
    const order = vals.map((v, i) => i).sort((a, b) => vals[a] - vals[b]);
    pts = order.map((i) => pts[i]); vals = order.map((i) => vals[i]);
    if (Math.abs(vals[n] - vals[0]) <= 1e-10 * (Math.abs(vals[0]) + 1e-12)) break;
    const c = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += pts[i][j] / n;
    const along = (t) => c.map((cj, j) => cj + t * (pts[n][j] - cj));
    const xr = along(-1), fr = f(xr);
    if (fr < vals[0]) {
      const xe = along(-2), fe = f(xe);
      if (fe < fr) { pts[n] = xe; vals[n] = fe; } else { pts[n] = xr; vals[n] = fr; }
    } else if (fr < vals[n - 1]) {
      pts[n] = xr; vals[n] = fr;
    } else {
      const xc = fr < vals[n] ? along(-0.5) : along(0.5), fc = f(xc);
      if (fc < Math.min(fr, vals[n])) { pts[n] = xc; vals[n] = fc; }
      else for (let i = 1; i <= n; i++) { pts[i] = pts[i].map((v, j) => pts[0][j] + 0.5 * (v - pts[0][j])); vals[i] = f(pts[i]); }
    }
  }
  return pts[vals.indexOf(Math.min(...vals))];
}

// In-place radix-2 FFT of complex arrays re, im (length a power of 2).
export function fft1(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? 2 : -2) * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

// |FFT| of an n x n real image (n a power of 2), mean removed, fftshifted (zero frequency at n/2).
export function fft2mag(img, n) {
  const re = Float64Array.from(img), im = new Float64Array(n * n), m = mean(img);
  for (let i = 0; i < n * n; i++) re[i] -= m;
  const r = new Float64Array(n), c = new Float64Array(n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) { r[x] = re[y * n + x]; c[x] = im[y * n + x]; }
    fft1(r, c);
    for (let x = 0; x < n; x++) { re[y * n + x] = r[x]; im[y * n + x] = c[x]; }
  }
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { r[y] = re[y * n + x]; c[y] = im[y * n + x]; }
    fft1(r, c);
    for (let y = 0; y < n; y++) { re[y * n + x] = r[y]; im[y * n + x] = c[y]; }
  }
  const out = new Float64Array(n * n), h = n >> 1;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++)
    out[((y + h) % n) * n + ((x + h) % n)] = Math.hypot(re[y * n + x], im[y * n + x]);
  return out;
}

// |DFT| for any n (direct, O(n^3)); zero frequency at floor(n/2). Use for n up to about 250.
export function dft2mag(img, n) {
  const cs = new Float64Array(n * n), sn = new Float64Array(n * n);
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { const t = -2 * Math.PI * ((a * b) % n) / n; cs[a * n + b] = Math.cos(t); sn[a * n + b] = Math.sin(t); }
  const m = mean(img), re1 = new Float64Array(n * n), im1 = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let kx = 0; kx < n; kx++) {
    let sr = 0, si = 0;
    for (let j = 0; j < n; j++) { const v = img[i * n + j] - m; sr += v * cs[kx * n + j]; si += v * sn[kx * n + j]; }
    re1[i * n + kx] = sr; im1[i * n + kx] = si;
  }
  const out = new Float64Array(n * n), h = Math.floor(n / 2);
  for (let kx = 0; kx < n; kx++) for (let ky = 0; ky < n; ky++) {
    let sr = 0, si = 0;
    for (let i = 0; i < n; i++) {
      const c = cs[ky * n + i], s = sn[ky * n + i], a = re1[i * n + kx], b = im1[i * n + kx];
      sr += a * c - b * s; si += a * s + b * c;
    }
    out[((ky + h) % n) * n + ((kx + h) % n)] = Math.hypot(sr, si);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// DOM and SVG builders
// ---------------------------------------------------------------------------------------------
const SVGNS = "http://www.w3.org/2000/svg";

function build(el, attrs, children) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) if (c !== null && c !== undefined && c !== false)
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}
// h("div", {class: "xp-row"}, child1, child2)
export const h = (tag, attrs, ...children) => build(document.createElement(tag), attrs, children);
// s("circle", {cx: 1, cy: 2, r: 3})
export const s = (tag, attrs, ...children) => build(document.createElementNS(SVGNS, tag), attrs, children);

// Layout helpers. Use them instead of inline flex styles.
export const row = (...c) => h("div", { class: "xp-row" }, ...c);
export const col = (...c) => h("div", { class: "xp-col" }, ...c);
export const card = (...c) => h("div", { class: "xp-card" }, ...c);
export const controls = (...c) => h("div", { class: "xp-controls" }, ...c);
export const caption = (html) => h("div", { class: "xp-caption", html });

// A line of text that the widget rewrites: const r = readout(); r.set("SNR = <b>3</b>")
export function readout(html = "", cls = "") {
  const el = h("div", { class: `xp-readout ${cls}`, html });
  el.set = (v) => { el.innerHTML = v; };
  return el;
}

// Return focus to the deck, so that the arrow keys move slides again after a drag.
const release = (e) => setTimeout(() => e.target.blur(), 0);

// slider({label: "dose", min: 1, max: 1e4, value: 50, log: true, digits: 0, unit: "e⁻/Å²", onInput: v => ...})
// Returns the wrapper element with .value (number) and .set(v).
export function slider({ label, min, max, value, step, log = false, digits = 2, unit = "", format, onInput }) {
  const N = 1000;
  const toPos = (v) => log ? N * Math.log(v / min) / Math.log(max / min) : N * (v - min) / (max - min);
  const fromPos = (p) => {
    let v = log ? min * (max / min) ** (p / N) : min + (max - min) * p / N;
    if (step) v = Math.round(v / step) * step;
    return clamp(v, min, max);
  };
  const input = h("input", { type: "range", min: 0, max: N, step: 1, value: toPos(value) });
  const out = h("span", { class: "xp-slider-value" });
  const show = (v) => { out.textContent = (format ? format(v) : fmt(v, digits)) + (unit ? ` ${unit}` : ""); };
  const el = h("label", { class: "xp-slider" }, h("span", { class: "xp-slider-label", html: label }), input, out);
  el.value = value;
  show(value);
  input.addEventListener("input", () => { el.value = fromPos(+input.value); show(el.value); onInput && onInput(el.value); });
  input.addEventListener("change", release);
  input.addEventListener("pointerup", release);
  el.set = (v, fire = true) => { el.value = v; input.value = toPos(v); show(v); if (fire && onInput) onInput(v); };
  return el;
}

// Segmented control: toggle({options: ["MSE", "Poisson"], value: "MSE", onChange: v => ...})
// options can be strings or {value, label, color}.
export function toggle({ options, value, onChange, label }) {
  const opts = options.map((o) => (typeof o === "string" ? { value: o, label: o } : o));
  const el = h("div", { class: "xp-toggle" }, label ? h("span", { class: "xp-slider-label", html: label }) : null);
  const btns = opts.map((o) => {
    const b = h("button", { type: "button", html: o.label });
    if (o.color) b.style.setProperty("--c", o.color);
    b.addEventListener("click", (e) => { el.set(o.value); release(e); });
    el.append(b);
    return b;
  });
  el.set = (v, fire = true) => {
    el.value = v;
    opts.forEach((o, i) => btns[i].classList.toggle("on", o.value === v));
    if (fire && onChange) onChange(v);
  };
  el.set(value, false);
  return el;
}

export function button(label, onClick, cls = "") {
  const b = h("button", { type: "button", class: `xp-button ${cls}`, html: label });
  b.addEventListener("click", (e) => { onClick(e); release(e); });
  return b;
}

// Coloured key: legend([[T.gauss, "MSE"], [T.pois, "Poisson NLL", "dash"]])
export function legend(items) {
  return h("div", { class: "xp-legend" }, items.map(([c, txt, style]) =>
    h("span", { class: "xp-key" },
      s("svg", { width: 34, height: 14 }, s("line", { x1: 2, x2: 32, y1: 7, y2: 7, stroke: c, "stroke-width": 4,
        "stroke-dasharray": style === "dash" ? "7 5" : style === "dot" ? "2 5" : null, "stroke-linecap": "round" })),
      h("span", { html: txt }))));
}

// ---------------------------------------------------------------------------------------------
// Chart: a small SVG plot with linear or log axes.
//   const c = chart({width: 760, height: 420, x: {domain: [0, 10], label: "dose"}, y: {domain: [0, 1], type: "log"}});
//   c.clear(); c.line([[0, 0], [1, 1]], {stroke: T.gauss}); c.dots(pts, {fill: T.data});
// All drawing methods take data coordinates. Call c.clear() before each redraw.
// ---------------------------------------------------------------------------------------------
function niceStep(span, count) {
  const raw = span / Math.max(1, count), p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
  return (f >= 7.5 ? 10 : f >= 3.5 ? 5 : f >= 1.5 ? 2 : 1) * p;
}
function makeScale(ax, r0, r1) {
  const [d0, d1] = ax.domain, log = ax.type === "log";
  const t = log ? (v) => Math.log10(v) : (v) => v;
  const a = t(d0), b = t(d1);
  const sc = (v) => r0 + (t(v) - a) / (b - a) * (r1 - r0);
  sc.invert = (p) => { const q = a + (p - r0) / (r1 - r0) * (b - a); return log ? 10 ** q : q; };
  sc.ticks = () => {
    if (ax.ticks) return ax.ticks;
    if (log) {
      const out = [];
      for (let e = Math.ceil(Math.log10(Math.min(d0, d1)) - 1e-9); e <= Math.floor(Math.log10(Math.max(d0, d1)) + 1e-9); e++) out.push(10 ** e);
      return out;
    }
    const lo = Math.min(d0, d1), hi = Math.max(d0, d1), st = niceStep(hi - lo, ax.tickCount || 5), out = [];
    for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + st * 1e-9; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : v);
    return out;
  };
  sc.fmt = ax.tickFormat || ((v) => {
    if (log) return v >= 1e4 || v < 1e-2 ? `10${sup(Math.round(Math.log10(v)))}` : String(+v.toPrecision(3));
    return String(+v.toPrecision(6));
  });
  return sc;
}
const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
export const sup = (n) => String(n).split("").map((c) => SUP[c] ?? c).join("");

let clipId = 0;
export function chart({ width = 760, height = 420, x, y, margin = {}, title, grid = true, bg }) {
  const m = { l: 76, r: 22, t: title ? 46 : 18, b: 62, ...margin };
  const el = s("svg", { class: "xp-chart", width, height, viewBox: `0 0 ${width} ${height}` });
  const id = `xpclip${clipId++}`;
  el.append(s("defs", {}, s("clipPath", { id }, s("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b }))));
  if (bg) el.append(s("rect", { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b, fill: bg }));
  const gAxes = s("g"), gPlot = s("g", { "clip-path": `url(#${id})` }), gTop = s("g");
  el.append(gAxes, gPlot, gTop);
  const c = { el, width, height, m, gPlot, gTop };

  c.setDomain = (xd, yd) => {
    if (xd) x.domain = xd;
    if (yd) y.domain = yd;
    c.x = makeScale(x, m.l, width - m.r);
    c.y = makeScale(y, height - m.b, m.t);
    gAxes.replaceChildren();
    const X0 = m.l, X1 = width - m.r, Y0 = height - m.b, Y1 = m.t;
    for (const v of c.x.ticks()) {
      const p = c.x(v);
      if (p < X0 - 1 || p > X1 + 1) continue;
      if (grid && x.grid !== false) gAxes.append(s("line", { x1: p, x2: p, y1: Y0, y2: Y1, stroke: T.grid }));
      gAxes.append(s("text", { x: p, y: Y0 + 26, class: "xp-tick", "text-anchor": "middle" }, c.x.fmt(v)));
    }
    for (const v of c.y.ticks()) {
      const p = c.y(v);
      if (p > Y0 + 1 || p < Y1 - 1) continue;
      if (grid && y.grid !== false) gAxes.append(s("line", { x1: X0, x2: X1, y1: p, y2: p, stroke: T.grid }));
      if (y.axis !== false) gAxes.append(s("text", { x: X0 - 10, y: p + 7, class: "xp-tick", "text-anchor": "end" }, c.y.fmt(v)));
    }
    gAxes.append(s("line", { x1: X0, x2: X1, y1: Y0, y2: Y0, stroke: T.axis, "stroke-width": 1.5 }));
    if (y.axis !== false) gAxes.append(s("line", { x1: X0, x2: X0, y1: Y0, y2: Y1, stroke: T.axis, "stroke-width": 1.5 }));
    if (x.label) gAxes.append(s("text", { x: (X0 + X1) / 2, y: height - 8, class: "xp-axlabel", "text-anchor": "middle" }, x.label));
    if (y.label) gAxes.append(s("text", { x: 0, y: 0, class: "xp-axlabel", "text-anchor": "middle",
      transform: `translate(20 ${(Y0 + Y1) / 2}) rotate(-90)` }, y.label));
    if (title) gAxes.append(s("text", { x: X0, y: 30, class: "xp-ctitle" }, title));
  };
  c.setDomain();

  const P = (p) => (Array.isArray(p) ? p : [p.x, p.y]);
  const ok = (v) => Number.isFinite(v);
  c.clear = () => { gPlot.replaceChildren(); gTop.replaceChildren(); };
  c.path = (pts) => {
    let d = "", pen = false;
    for (const p of pts) {
      const [a, b] = P(p), X = c.x(a), Y = c.y(b);
      if (!ok(X) || !ok(Y)) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${X.toFixed(2)},${Y.toFixed(2)}`; pen = true;
    }
    return d;
  };
  // Lines, areas and markers. `top: true` draws outside the clip region (for labels).
  c.line = (pts, { stroke = T.gauss, width: w = 3, dash, opacity, top } = {}) =>
    (top ? gTop : gPlot).appendChild(s("path", { d: c.path(pts), fill: "none", stroke, "stroke-width": w,
      "stroke-dasharray": dash, opacity, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  c.area = (upper, lower, { fill = T.gauss, opacity = 0.2 } = {}) => {
    const d = c.path(upper) + c.path([...lower].reverse()).replace(/^M/, "L") + "Z";
    return gPlot.appendChild(s("path", { d, fill, opacity, stroke: "none" }));
  };
  c.dots = (pts, { r = 6, fill = T.data, stroke, strokeWidth = 2, opacity, top } = {}) => {
    const g = s("g");
    for (const p of pts) {
      const [a, b] = P(p), X = c.x(a), Y = c.y(b);
      if (ok(X) && ok(Y)) g.append(s("circle", { cx: X, cy: Y, r, fill, stroke, "stroke-width": strokeWidth, opacity }));
    }
    return (top ? gTop : gPlot).appendChild(g);
  };
  // Histogram-style bars: items [x0, x1, height]; base is y = 0 (or the bottom of a log axis).
  c.bars = (items, { fill = T.data, opacity = 0.85, stroke } = {}) => {
    const g = s("g"), base = y.type === "log" ? c.y(Math.min(...y.domain)) : c.y(0);
    for (const [x0, x1, v] of items) {
      const X0 = c.x(x0), X1 = c.x(x1), Y = c.y(v);
      if (!ok(X0) || !ok(X1) || !ok(Y)) continue;
      g.append(s("rect", { x: Math.min(X0, X1), y: Math.min(Y, base), width: Math.max(0.5, Math.abs(X1 - X0)),
        height: Math.abs(base - Y), fill, opacity, stroke }));
    }
    return gPlot.appendChild(g);
  };
  c.rect = (x0, y0, x1, y1, attrs = {}) => gPlot.appendChild(s("rect", {
    x: Math.min(c.x(x0), c.x(x1)), y: Math.min(c.y(y0), c.y(y1)),
    width: Math.abs(c.x(x1) - c.x(x0)), height: Math.abs(c.y(y1) - c.y(y0)), ...attrs }));
  c.ruleX = (v, { stroke = T.muted, width: w = 2, dash = "6 6" } = {}) =>
    gPlot.appendChild(s("line", { x1: c.x(v), x2: c.x(v), y1: m.t, y2: height - m.b, stroke, "stroke-width": w, "stroke-dasharray": dash }));
  c.ruleY = (v, { stroke = T.muted, width: w = 2, dash = "6 6" } = {}) =>
    gPlot.appendChild(s("line", { x1: m.l, x2: width - m.r, y1: c.y(v), y2: c.y(v), stroke, "stroke-width": w, "stroke-dasharray": dash }));
  // Text at data coordinates.
  c.text = (a, b, str, { fill = T.ink, size = 20, anchor = "start", weight, dx = 0, dy = 0 } = {}) =>
    gTop.appendChild(s("text", { x: c.x(a) + dx, y: c.y(b) + dy, fill, "font-size": size, "text-anchor": anchor, "font-weight": weight, class: "xp-ctext" }, str));
  // Text at a corner of the plot area: corner is "tl", "tr", "bl" or "br"; line = 0, 1, 2, ... stacks labels.
  c.note = (str, { corner = "tl", line = 0, fill = T.ink, size = 20, weight } = {}) => {
    const right = corner[1] === "r", bottom = corner[0] === "b";
    const X = right ? width - m.r - 12 : m.l + 12, Y = bottom ? height - m.b - 12 - line * (size + 8) : m.t + size + 6 + line * (size + 8);
    return gTop.appendChild(s("text", { x: X, y: Y, fill, "font-size": size, "text-anchor": right ? "end" : "start", "font-weight": weight, class: "xp-ctext" }, str));
  };
  return c;
}

// ---------------------------------------------------------------------------------------------
// Image canvas with nearest-neighbour upscaling.
//   const im = imageCanvas({n: 128, size: 420}); im.draw(data, 0, 1, "gray");
// cmap: "gray", "magma" (good for FFT magnitudes), "div" (blue-white-red, symmetric data).
// ---------------------------------------------------------------------------------------------
const STOPS = {
  gray: [[0, 0, 0], [255, 255, 255]],
  magma: [[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]],
  div: [[59, 130, 246], [15, 23, 42], [244, 63, 94]],
};
const LUT = {};
function lut(name) {
  if (LUT[name]) return LUT[name];
  const st = STOPS[name] || STOPS.gray, out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (st.length - 1), k = Math.min(st.length - 2, Math.floor(t)), f = t - k;
    for (let c = 0; c < 3; c++) out[3 * i + c] = st[k][c] + f * (st[k + 1][c] - st[k][c]);
  }
  return (LUT[name] = out);
}
export function imageCanvas({ n, ny, size = 400, height, label }) {
  const nx = n, H = ny || n;
  const cv = h("canvas", { width: nx, height: H, class: "xp-image" });
  cv.style.width = `${size}px`; cv.style.height = `${height || size * H / nx}px`;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(nx, H);
  const el = h("div", { class: "xp-imagebox" }, label ? h("div", { class: "xp-imlabel", html: label }) : null, cv);
  el.canvas = cv;
  el.setLabel = (html) => { const l = el.querySelector(".xp-imlabel"); if (l) l.innerHTML = html; };
  el.draw = (data, vmin, vmax, cmap = "gray") => {
    const L = lut(cmap), sc = 255 / ((vmax - vmin) || 1);
    for (let i = 0; i < nx * H; i++) {
      const g = Math.max(0, Math.min(255, Math.round((data[i] - vmin) * sc)));
      img.data[4 * i] = L[3 * g]; img.data[4 * i + 1] = L[3 * g + 1]; img.data[4 * i + 2] = L[3 * g + 2]; img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  };
  return el;
}

// ---------------------------------------------------------------------------------------------
// Registry and life cycle.
//   register("dose-image", (el, api) => { ...build into el... });
// api.data     : the data-* attributes of the placeholder div (strings)
// api.animate(fn) : fn(dtSeconds, tSeconds) runs every frame while the slide is on screen
// api.onShow(fn)  : fn() runs each time the slide comes on screen (after the first mount too)
// api.isActive()  : true while the slide is on screen
// ---------------------------------------------------------------------------------------------
const REG = new Map();
export function register(name, mount) { REG.set(name, mount); }
export const registered = () => [...REG.keys()];

const live = [];
let rafOn = false, last = 0;
function loop(t) {
  const dt = Math.min(0.05, (t - last) / 1000 || 0); last = t;
  for (const w of live) if (w.active) for (const f of w.tickers) f(dt, t / 1000);
  requestAnimationFrame(loop);
}

export function mount(el) {
  if (el.__xp) return el.__xp;
  const name = el.dataset.xp, fn = REG.get(name);
  const w = { el, active: false, tickers: [], shows: [] };
  el.__xp = w;
  if (!fn) { el.append(h("div", { class: "xp-missing" }, `widget "${name}" is not registered`)); return w; }
  const api = {
    data: { ...el.dataset },
    animate: (f) => w.tickers.push(f),
    onShow: (f) => w.shows.push(f),
    isActive: () => w.active,
  };
  el.classList.add("xp-mounted");
  try { fn(el, api); } catch (err) { console.error(err); el.append(h("pre", { class: "xp-missing" }, `${name}: ${err.message}`)); }
  live.push(w);
  if (!rafOn) { rafOn = true; requestAnimationFrame(loop); }
  return w;
}

export function setActive(w, on) {
  const was = w.active;
  w.active = on;
  if (on && !was) for (const f of w.shows) f();
}
