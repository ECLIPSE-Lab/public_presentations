// Sampling group: the pixel grid. Nyquist in 1-D, aliasing of a 2-D lattice, Moiré of two
// lattices, and the detector point-spread function (PSF) with its MTF.
import {
  register, T, chart, slider, toggle, button, legend, readout, imageCanvas, card, row, col, controls, caption,
  h, s, rng, fmt, clamp, linspace, mean,
} from "./core.js";

// ---------------------------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------------------------
const CSS = `
.xp-alias2d-side { width: 456px; }
.xp-alias2d-side .xp-readout { font-size: 23px; }
.xp-moire-stage { position: relative; }
.xp-moire-stage svg { position: absolute; left: 0; top: 0; pointer-events: none; }
.xp-psf-mid { width: 210px; }
.xp-psf-mid .xp-caption { font-size: 19px; }
.xp-moire-key .xp-legend { flex-direction: column; gap: 4px; font-size: 19px; }
`;
function injectStyle() {
  if (document.getElementById("xp-sampling-style")) return;
  document.head.append(h("style", { id: "xp-sampling-style" }, CSS));
}

// Render an array to a data URL with the colour maps of imageCanvas (one offscreen canvas per size).
const RASTER = new Map();
function rasterURL(data, nx, ny, vmin, vmax, cmap) {
  const key = `${nx}x${ny}`;
  if (!RASTER.has(key)) RASTER.set(key, imageCanvas({ n: nx, ny }));
  const ic = RASTER.get(key);
  ic.draw(data, vmin, vmax, cmap);
  return ic.canvas.toDataURL();
}
// Place a raster in a chart: data-coordinate box [x0, x1] x [y0, y1]; row 0 is drawn at y0.
function chartImage(c, url, x0, x1, y0, y1) {
  const X0 = c.x(x0), X1 = c.x(x1), Y0 = c.y(y0), Y1 = c.y(y1);
  const img = s("image", { x: Math.min(X0, X1), y: Math.min(Y0, Y1), width: Math.abs(X1 - X0), height: Math.abs(Y1 - Y0),
    preserveAspectRatio: "none", style: "image-rendering: pixelated" });
  img.setAttribute("href", url);
  c.gPlot.append(img);
  if (Y0 > Y1) img.setAttribute("transform", `translate(0 ${Y0 + Y1}) scale(1 -1)`);   // row 0 at the bottom
  return img;
}
// Arrow-head marker in a chart's <defs>; returns "url(#id)".
let markerId = 0;
function arrowHead(c, color) {
  const id = `xp-samp-arrow${markerId++}`;
  c.el.querySelector("defs").append(s("marker", { id, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7,
    orient: "auto-start-reverse" }, s("path", { d: "M0,0 L10,5 L0,10 z", fill: color })));
  return `url(#${id})`;
}
const ring = (c, x, y, r, stroke, w = 3) => c.gTop.append(s("circle", { cx: c.x(x), cy: c.y(y), r, fill: "none", stroke, "stroke-width": w }));
const keyRing = (color, filled) => s("svg", { width: 26, height: 26 },
  s("circle", { cx: 13, cy: 13, r: filled ? 6 : 10, fill: filled ? color : "none", stroke: color, "stroke-width": 3 }));
const keyItem = (sym, txt) => h("span", { class: "xp-key" }, sym, h("span", { html: txt }));
const wrapDeg = (d) => ((d + 540) % 360) - 180;

// |DFT| of a real n x n image (any n), mean removed, zero frequency at floor(n/2): the same output as
// core's dft2mag, but about 3x faster (Hermitian symmetry, contiguous loops, no Math.hypot).
function dftMag(img, n) {
  const hn = Math.floor(n / 2), nk = hn + 1, cs = new Float64Array(n), sn = new Float64Array(n), m = mean(img);
  for (let a = 0; a < n; a++) { cs[a] = Math.cos(2 * Math.PI * a / n); sn[a] = -Math.sin(2 * Math.PI * a / n); }
  const Rr = new Float64Array(nk * n), Ri = new Float64Array(nk * n), row = new Float64Array(n);
  for (let i = 0; i < n; i++) {                                       // rows, kx = 0 … n/2, stored [kx][i]
    for (let j = 0; j < n; j++) row[j] = img[i * n + j] - m;
    for (let kx = 0; kx < nk; kx++) {
      let sr = 0, si = 0, q = 0;
      for (let j = 0; j < n; j++) { sr += row[j] * cs[q]; si += row[j] * sn[q]; q += kx; if (q >= n) q -= n; }
      Rr[kx * n + i] = sr; Ri[kx * n + i] = si;
    }
  }
  const out = new Float64Array(n * n), sh = (k) => ((k % n) + n + hn) % n;
  for (let kx = 0; kx < nk; kx++) for (let ky = 0; ky < n; ky++) {
    let sr = 0, si = 0, q = 0;
    for (let i = 0, o = kx * n; i < n; i++) {
      const a = Rr[o + i], b = Ri[o + i], c = cs[q], d = sn[q];
      sr += a * c - b * d; si += a * d + b * c; q += ky; if (q >= n) q -= n;
    }
    const v = Math.sqrt(sr * sr + si * si);
    out[sh(ky) * n + sh(kx)] = v; out[sh(-ky) * n + sh(-kx)] = v;      // F(−k) = F(k)* for real input
  }
  return out;
}

// Radix-2 FFT with precomputed tables for repeated transforms of length n (a power of 2).
function makeFFT(n) {
  const rev = new Uint32Array(n), bits = Math.log2(n), wr = new Float64Array(n / 2), wi = new Float64Array(n / 2);
  for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
  for (let k = 0; k < n / 2; k++) { wr[k] = Math.cos(-2 * Math.PI * k / n); wi[k] = Math.sin(-2 * Math.PI * k / n); }
  return (re, im) => {
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1, step = n / len;
      for (let i = 0; i < n; i += len) for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half, c = wr[k * step], d = wi[k * step];
        const tr = re[b] * c - im[b] * d, ti = re[b] * d + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  };
}

// ---------------------------------------------------------------------------------------------
// nyquist: a 2 Å lattice profile cos(2πx/a) sampled at pixel size Δx (samples at x = kΔx).
// The samples cannot tell f = 1/a from |f − n/Δx|; the recorded frequency is the one in
// [0, 1/(2Δx)]. Above Nyquist (Δx > a/2) a false period 1/|f − n fₛ| appears.
// ---------------------------------------------------------------------------------------------
register("nyquist", (el) => {
  const A = 2, F = 1 / A, X1 = 20, FMAX = 1.2, W = 1720;
  const top = chart({ width: W - 4, height: 318, x: { domain: [0, X1], label: "position x (Å)" },
    y: { domain: [-1.3, 1.3], axis: false, grid: false }, margin: { l: 24, r: 24, t: 10, b: 58 } });
  const bot = chart({ width: W - 4, height: 236, x: { domain: [0, FMAX], label: "spatial frequency f (Å⁻¹)", tickCount: 12 },
    y: { domain: [0, 1.5], axis: false, grid: false }, margin: { l: 24, r: 24, t: 6, b: 58 } });
  const head = arrowHead(bot, T.pois);
  const xs = linspace(0, X1, 1600);
  const out = readout("", "big");
  const sl = slider({ label: "pixel size Δx", min: 0.25, max: 2.5, value: 1.5, step: 0.05, digits: 2, unit: "Å", onInput: draw });
  el.append(card(
    controls(sl, legend([[T.truth, "true 2 Å lattice"], [T.data, "recorded pixels"], [T.pois, "alias", "dash"], [T.warn, "recordable band 0 … 1/(2Δx)"]])),
    top.el, bot.el, out));

  function draw(dx) {
    const fs = 1 / dx, fN = fs / 2, n = Math.round(F / fs), fa = Math.abs(F - n * fs), ppp = A / dx;
    const under = F > fN + 1e-9;
    // Top: real space
    top.clear();
    for (let k = 0; (k - 0.5) * dx < X1; k += 2) top.rect((k - 0.5) * dx, -1.3, (k + 0.5) * dx, 1.3, { fill: "#fff", opacity: 0.045 });
    top.line(xs.map((x) => [x, Math.cos(2 * Math.PI * F * x)]), { stroke: T.truth, width: 12, opacity: 0.35 });
    if (under) top.line(xs.map((x) => [x, Math.cos(2 * Math.PI * fa * x)]), { stroke: T.pois, width: 3.5, dash: "10 7" });
    const smp = [];
    for (let x = 0; x <= X1 + 1e-9; x += dx) smp.push([x, Math.cos(2 * Math.PI * F * x)]);
    top.line(smp, { stroke: T.data, width: 2.5 });
    top.dots(smp, { r: 7, fill: T.data, stroke: T.bg });
    // Bottom: frequency axis
    bot.clear();
    bot.rect(0, 0, Math.min(fN, FMAX), 1.5, { fill: T.warn, opacity: 0.12 });
    if (fN <= FMAX) {
      bot.ruleX(fN, { stroke: T.warn, width: 3, dash: null });
      bot.text(fN, 1.36, `Nyquist ${fmt(fN, 2)} Å⁻¹`, { fill: T.warn, anchor: fN > 0.95 ? "end" : "start", dx: fN > 0.95 ? -10 : 10 });
    } else bot.note(`recordable band reaches ${fmt(fN, 2)} Å⁻¹ →`, { corner: "tr", fill: T.warn });
    if (fs <= FMAX) {
      bot.ruleX(fs, { stroke: T.muted, width: 2 });
      bot.text(fs, 1.36, `fₛ = 1/Δx = ${fmt(fs, 2)} Å⁻¹`, { fill: T.muted, anchor: fs > 0.95 ? "end" : "start", dx: fs > 0.95 ? -10 : 10 });
    }
    const spike = (f, color, w) => { bot.line([[f, 0], [f, 0.8]], { stroke: color, width: w }); bot.dots([[f, 0.8]], { r: 7, fill: color }); };
    spike(F, T.truth, 6);
    bot.text(F, 0.42, "lattice 1/(2 Å) = 0.5 Å⁻¹", { fill: T.truth, dx: 12 });
    if (under) {
      spike(fa, T.pois, 6);
      const X0 = bot.x(F), Xa = bot.x(fa), Y = bot.y(0.9), Yc = bot.y(1.35);
      bot.gTop.append(s("path", { d: `M${X0},${Y} Q${(X0 + Xa) / 2},${Yc} ${Xa + (Xa < X0 ? 6 : -6)},${Y}`, fill: "none", stroke: T.pois,
        "stroke-width": 3, "marker-end": head }));
      bot.text(fa, 0.42, fa > 1e-6 ? `alias |f − ${n > 1 ? n : ""}fₛ| = ${fmt(fa, 2)} Å⁻¹` : "alias at f = 0", { fill: T.pois, dx: fa < 0.25 ? 12 : -12, anchor: fa < 0.25 ? "start" : "end" });
    }
    const cls = ppp >= 3 ? "good" : !under ? "warn" : "bad";
    const verdict = ppp >= 3 ? "fine: the period is clear" : !under ? "at or near the Nyquist limit: only just"
      : fa > 1e-6 ? `below Nyquist: a false ${fmt(1 / fa, 1)} Å period appears (dashed)`
      : "below Nyquist: all samples see the same phase, the lattice vanishes";
    out.set(`Pixel ${fmt(dx, 2)} Å → <b>${fmt(ppp, 2)} pixels per 2 Å period</b> → <span class="${cls}">${verdict}</span>`);
  }
  draw(sl.value);
});

// ---------------------------------------------------------------------------------------------
// alias2d: a 2 Å square lattice, I = (1 + cos k·u)(1 + cos k·v)/4, imaged over a 54 Å field with
// pixel size Δx, so n = round(54/Δx) pixels. The |DFT| covers ±1/(2Δx) per axis (the amber box).
// Main spots are at 1/a = 5 nm⁻¹, diagonal spots at √2/a = 7.07 nm⁻¹. A spot outside the box
// folds back by multiples of 1/Δx in each component and shows a false lattice.
// ---------------------------------------------------------------------------------------------
register("alias2d", (el, api) => {
  injectStyle();
  const A = 2, FIELD = 54, L = 10.5, S = 540, K0 = 10 / A;
  const cache = new Map();
  function compute(dx, deg) {
    const key = `${dx.toFixed(3)}|${deg}`;
    if (cache.has(key)) return cache.get(key);
    const n = Math.round(FIELD / dx), th = deg * Math.PI / 180, k = 2 * Math.PI / A, c = Math.cos(th), sn = Math.sin(th);
    const im = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x = j * dx, y = i * dx;
      im[i * n + j] = (Math.cos(k * (x * c + y * sn)) + 1) * (Math.cos(k * (-x * sn + y * c)) + 1) / 4;
    }
    const F = dftMag(im, n), hh = Math.floor(n / 2);
    let fmax = 0;
    for (const v of F) fmax = Math.max(fmax, v);
    const kOf = (idx) => (idx - hh) / (n * dx) * 10;                  // nm⁻¹
    const spots = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = F[i * n + j];
      if (v < 0.2 * fmax) continue;
      let peak = true;
      for (let a = -1; a <= 1 && peak; a++) for (let b = -1; b <= 1; b++)
        if ((a || b) && F[((i + a + n) % n) * n + ((j + b + n) % n)] > v) { peak = false; break; }
      if (peak) spots.push([kOf(j), kOf(i)]);
    }
    const r = { n, im, F, fmax, spots, k0: kOf(-0.5), k1: kOf(n - 0.5) };
    if (cache.size > 300) cache.clear();
    cache.set(key, r);
    return r;
  }

  const imgHold = h("div");
  const fft = chart({ width: S + 90, height: S + 76, bg: "#000", margin: { l: 80, r: 10, t: 6, b: 70 },
    x: { domain: [-L, L], label: "kx (nm⁻¹)" }, y: { domain: [L, -L], label: "ky (nm⁻¹)" } });
  const out = readout();
  let dirty = false;
  const touch = () => { dirty = true; };
  const sDx = slider({ label: "pixel size Δx", min: 0.25, max: 2.0, value: 1.5, step: 0.05, digits: 2, unit: "Å", onInput: touch });
  const sTh = slider({ label: "specimen rotation", min: 0, max: 45, value: 0, step: 1, digits: 0, unit: "°", onInput: touch });
  const key = h("div", { class: "xp-legend" },
    keyItem(keyRing(T.truth), "true spot position"), keyItem(keyRing("#fff", true), "measured spot"),
    keyItem(s("svg", { width: 34, height: 14 }, s("line", { x1: 2, x2: 32, y1: 7, y2: 7, stroke: T.warn, "stroke-width": 4, "stroke-dasharray": "7 5" })),
      "recordable range ±1/(2Δx)"));
  el.append(card(controls(sDx, sTh), row(imgHold, fft.el, h("div", { class: "xp-col xp-alias2d-side" }, key, out))));
  api.animate(() => { if (dirty) { dirty = false; draw(); } });

  const fold = (k, kn) => k - Math.round(k / (2 * kn)) * 2 * kn;
  function draw() {
    const dx = sDx.value, deg = sTh.value, th = deg * Math.PI / 180, r = compute(dx, deg), n = r.n, kn = 10 / (2 * dx);
    const ic = imageCanvas({ n, size: S, label: `${n} × ${n} pixels of ${fmt(dx, 2)} Å (54 Å field)` });
    ic.draw(r.im, 0, 1, "gray");
    imgHold.replaceChildren(ic);
    fft.clear();
    chartImage(fft, rasterURL(r.F.map((v) => Math.sqrt(v / r.fmax)), n, n, 0, 1, "magma"), r.k0, r.k1, r.k0, r.k1);
    fft.rect(-kn, -kn, kn, kn, { fill: "none", stroke: T.warn, "stroke-width": 3, "stroke-dasharray": "9 6" });
    for (let q = 0; q < 4; q++) {
      const a = th + q * Math.PI / 2, ad = a + Math.PI / 4;
      ring(fft, K0 * Math.cos(a), K0 * Math.sin(a), 15, T.truth);
      ring(fft, K0 * Math.SQRT2 * Math.cos(ad), K0 * Math.SQRT2 * Math.sin(ad), 11, T.truth, 2);
    }
    fft.dots(r.spots, { r: 5, fill: "#fff", top: true });
    // Fold the true main and diagonal spots into the recordable box.
    const gx = K0 * Math.cos(th), gy = K0 * Math.sin(th);
    const inside = (x, y) => Math.abs(x) < kn - 1e-9 && Math.abs(y) < kn - 1e-9;
    const mainOk = inside(gx, gy) && inside(-gy, gx), diagOk = inside(gx - gy, gx + gy) && inside(gx + gy, gy - gx);
    const fx = fold(gx, kn), fy = fold(gy, kn), kp = Math.hypot(fx, fy);
    const ang = ((Math.atan2(fy, fx) * 180 / Math.PI) % 90 + 90) % 90;
    let msg = `Pixel ${fmt(dx, 2)} Å = <b>${fmt(A / dx, 2)} pixels per 2 Å period</b>. Recordable range ±${fmt(kn, 2)} nm⁻¹ (amber box).<br>`;
    if (mainOk) msg += `<span class="good">The 5 nm⁻¹ spots lie inside: the 2 Å lattice is recorded correctly.</span>` +
      (diagOk ? "" : ` <span class="warn">Only the 1.41 Å diagonal spots (7.07 nm⁻¹) fold back.</span>`);
    else if (kp < 1e-6) msg += `<span class="bad">The 5 nm⁻¹ spots fold onto the centre: the lattice vanishes.</span>`;
    else msg += `<span class="bad">The 5 nm⁻¹ spots lie outside and fold to ${fmt(kp, 2)} nm⁻¹: a false ${fmt(10 / kp, 1)} Å lattice` +
      (Math.abs(wrapDeg(4 * (ang - deg)) / 4) > 0.5 ? `, turned to ${fmt(ang, 0)}° (true ${fmt(deg, 0)}°).` : ".") + `</span>`;
    out.set(msg);
  }
  draw();
});

// ---------------------------------------------------------------------------------------------
// moire: two triangular lattices, I_m(r) = (1.5 + Σⱼ cos gⱼ·r)/4.5, multiplied (transmission
// through two layers). Layer 1: a₁ = 2.46 Å, fixed. Layer 2 ("specimen"): a₂ = a₁(1 + Δa/a),
// turned by θ. The product contains cos((g₁ − g₂)·r): the Moiré wave vector is Δg = g₁ − g₂, and the
// Moiré lattice constant is D = 2/(√3 |Δg|) (|Δg| in Å⁻¹) = a₁ / |1 − e^{iθ}/(1 + Δa/a)|.
// The field is 160 Å on 512 pixels (0.31 Å/pixel, 7.9 pixels per lattice row), so the beat is real.
// ---------------------------------------------------------------------------------------------
register("moire", (el, api) => {
  injectStyle();
  const A = 2.46, N = 512, FIELD = 160, PX = FIELD / N, S = N, KR = 0.6, R = Math.round(KR * FIELD), ZS = 200, FP = 480;
  const G = 2 / (Math.sqrt(3) * A);                                       // |g| of layer 1 (Å⁻¹)
  const waves = (a, rot) => [0, 1, 2].map((j) => {
    const ang = (30 + 120 * j) * Math.PI / 180 + rot, g = 2 / (Math.sqrt(3) * a);
    return [g * Math.cos(ang), g * Math.sin(ang)];                       // Å⁻¹
  });
  const dgOf = (dl, thDeg) => {                                          // Δg₀ = g₁ − g₂ (Å⁻¹)
    const g1 = waves(A, 0)[0], g2 = waves(A * (1 + dl), thDeg * Math.PI / 180)[0];
    return [g1[0] - g2[0], g1[1] - g2[1]];
  };
  const coord = Float64Array.from({ length: N }, (_, j) => (j - N / 2) * PX);
  const hann = Float64Array.from({ length: N }, (_, j) => 0.5 - 0.5 * Math.cos(2 * Math.PI * j / N));
  const img = new Float64Array(N * N);

  function makeImage(dl, thDeg) {
    const W = [...waves(A, 0), ...waves(A * (1 + dl), thDeg * Math.PI / 180)];
    const tab = W.map(([kx, ky]) => {                                    // cos(kx x + ky y) = cx·cy − sx·sy
      const t = { cx: new Float64Array(N), sx: new Float64Array(N), cy: new Float64Array(N), sy: new Float64Array(N) };
      for (let j = 0; j < N; j++) {
        const px = 2 * Math.PI * kx * coord[j], py = 2 * Math.PI * ky * -coord[j];   // row i has y = −coord[i]
        t.cx[j] = Math.cos(px); t.sx[j] = Math.sin(px); t.cy[j] = Math.cos(py); t.sy[j] = Math.sin(py);
      }
      return t;
    });
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      let l1 = 1.5, l2 = 1.5;
      for (let w = 0; w < 3; w++) { const t = tab[w]; l1 += t.cx[j] * t.cy[i] - t.sx[j] * t.sy[i]; }
      for (let w = 3; w < 6; w++) { const t = tab[w]; l2 += t.cx[j] * t.cy[i] - t.sx[j] * t.sy[i]; }
      img[i * N + j] = l1 * l2 / 20.25;
    }
  }
  // |FFT| of the Hann-windowed image, only for |kx|, |ky| ≤ RC frequency pixels: all rows, then
  // only the columns that are needed. spec[(ky + RC)·M + kx + RC], ky upwards.
  const RC = 128, M = 2 * RC + 1, fft1n = makeFFT(N), spec = new Float64Array(M * M);
  const re = new Float64Array(N * N), im = new Float64Array(N * N), cr = new Float64Array(N), ci = new Float64Array(N);
  let smax = 1;
  function makeSpectrum() {
    const m = mean(img);
    im.fill(0);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) re[i * N + j] = (img[i * N + j] - m) * hann[i] * hann[j];
      fft1n(re.subarray(i * N, (i + 1) * N), im.subarray(i * N, (i + 1) * N));
    }
    smax = 0;
    for (let kx = -RC; kx <= RC; kx++) {
      const c = (kx + N) % N;
      for (let i = 0; i < N; i++) { cr[i] = re[i * N + c]; ci[i] = im[i * N + c]; }
      fft1n(cr, ci);
      for (let ky = -RC; ky <= RC; ky++) {
        const r = (N - ky) % N, v = Math.sqrt(cr[r] * cr[r] + ci[r] * ci[r]);   // image rows run downwards
        spec[(ky + RC) * M + kx + RC] = v;
        if (Math.abs(kx) + Math.abs(ky) > 2 && v > smax) smax = v;
      }
    }
  }
  // Log-scaled crop of ±r pixels around (cx, cy); row 0 is the lowest ky.
  const crop = (cx, cy, r) => {
    const m = 2 * r + 1, o = new Float64Array(m * m), lo = Math.log(smax * 2e-3);
    for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) {
      const ky = cy - r + a, kx = cx - r + b;
      const v = Math.abs(ky) <= RC && Math.abs(kx) <= RC ? spec[(ky + RC) * M + kx + RC] : 0;
      o[a * m + b] = Math.log(Math.max(v, smax * 2e-3)) - lo;
    }
    return { o, m, vmax: Math.log(smax) - lo };
  };

  // DOM
  const ic = imageCanvas({ n: N, size: S });
  const overlay = s("svg", { width: S, height: S, viewBox: `0 0 ${S} ${S}` });
  const imLabel = h("div", { class: "xp-imlabel" });
  const stage = h("div", { class: "xp-moire-stage", style: { width: `${S}px`, height: `${S}px` } }, ic.canvas, overlay);
  const fft = chart({ width: FP + 92, height: FP + 74, bg: "#000", margin: { l: 82, r: 10, t: 6, b: 68 },
    x: { domain: [-KR, KR], label: "kx (Å⁻¹)" }, y: { domain: [-KR, KR], label: "ky (Å⁻¹)" } });
  const zoomChart = (title) => chart({ width: ZS, height: ZS + 6, bg: "#000", title, grid: false, margin: { l: 4, r: 4, t: 40, b: 6 },
    x: { domain: [-1, 1], ticks: [] }, y: { domain: [-1, 1], axis: false, ticks: [] } });
  const z0 = zoomChart("zoom at 0: Moiré"), z1 = zoomChart("zoom at g₁: two lattices");
  const headT = [fft, z0, z1].map((c) => arrowHead(c, T.thread));
  const out = readout();
  let dl = 0.04, th = 1, dirty = 0, anim = null, lastTurn = null;   // dirty: 1 image, 2 image + FFT
  const sD = slider({ label: "lattice mismatch Δa/a", min: 0, max: 12, value: 4, step: 0.1, digits: 1, unit: "%", onInput: (v) => { dl = v / 100; dirty = 2; } });
  const sT = slider({ label: "twist θ", min: 0, max: 10, value: 1, step: 0.1, digits: 1, unit: "°", onInput: (v) => { th = v; dirty = 2; } });
  const bTurn = button("rotate specimen by 1°", () => {
    if (anim) return;
    const to = th + 1 <= 10 + 1e-9 ? th + 1 : th - 1;
    anim = { from: th, to, t: 0 };
  }, "primary");
  el.append(card(
    controls(sD, sT, bTurn),
    row(h("div", { class: "xp-imagebox" }, imLabel, stage), fft.el,
      col(z0.el, z1.el, h("div", { class: "xp-moire-key" }, legend([[T.truth, "layer 1: g₁ (fixed)"], [T.pink, "layer 2: g₂ (specimen)"], [T.thread, "Moiré: Δg = g₁ − g₂"]])))),
    out));

  api.animate((dt) => {
    if (anim) {
      anim.t = Math.min(1, anim.t + dt / 0.9);
      const e = anim.t < 0.5 ? 2 * anim.t * anim.t : 1 - 2 * (1 - anim.t) ** 2;
      th = anim.from + (anim.to - anim.from) * e;
      sT.set(th, false);
      if (anim.t >= 1) {
        th = anim.to; sT.set(th, false);
        const p0 = dgOf(dl, anim.from), p1 = dgOf(dl, anim.to);
        lastTurn = { dl, th, turn: anim.to - anim.from, dphi: wrapDeg((Math.atan2(p1[1], p1[0]) - Math.atan2(p0[1], p0[0])) * 180 / Math.PI) };
        anim = null; dirty = 2;
      } else dirty = Math.max(dirty, 1);
    }
    if (dirty) { const full = dirty === 2; dirty = 0; draw(full); }
  });

  const ang = (v) => Math.atan2(v[1], v[0]) * 180 / Math.PI;
  function draw(full = true) {
    makeImage(dl, th);
    ic.draw(img, 0, 0.42, "gray");
    const dg = dgOf(dl, th), adg = Math.hypot(dg[0], dg[1]), D = adg > 1e-12 ? 2 / (Math.sqrt(3) * adg) : Infinity;
    const phi = ang(dg) - 30;                                           // direction of the Moiré lattice vector
    imLabel.innerHTML = `layer 1 × layer 2, 160 Å field · Moiré lattice at ${Number.isFinite(D) ? fmt(phi, 1) + "°" : "–"}`;
    // Scale bar of length D from the AA point at the centre to the next AA point.
    overlay.replaceChildren();
    if (Number.isFinite(D)) {
      const sc = S / FIELD, c0 = S / 2, ex = c0 + D * sc * Math.cos(phi * Math.PI / 180), ey = c0 - D * sc * Math.sin(phi * Math.PI / 180);
      overlay.append(s("line", { x1: c0, y1: c0, x2: ex, y2: ey, stroke: T.bg, "stroke-width": 9, "stroke-linecap": "round", opacity: 0.6 }),
        s("line", { x1: c0, y1: c0, x2: ex, y2: ey, stroke: T.thread, "stroke-width": 5, "stroke-linecap": "round" }),
        s("circle", { cx: c0, cy: c0, r: 7, fill: T.thread }), s("circle", { cx: ex, cy: ey, r: 7, fill: T.thread }));
      const lx = clamp((c0 + ex) / 2 + 14, 10, S - 150), ly = clamp((c0 + ey) / 2 - 14, 28, S - 10);
      overlay.append(s("text", { x: lx, y: ly, fill: "#fff", "font-size": 24, "font-weight": 700, stroke: T.bg, "stroke-width": 5,
        "paint-order": "stroke" }, D > 999 ? "D > 1000 Å" : `D = ${fmt(D, 1)} Å`));
    }
    const g1 = waves(A, 0), g2 = waves(A * (1 + dl), th * Math.PI / 180);
    if (full) {
      makeSpectrum();
      const step = 1 / FIELD;                                            // Å⁻¹ per frequency pixel
      fft.clear();
      const cm = crop(0, 0, R);
      chartImage(fft, rasterURL(cm.o, cm.m, cm.m, 0, cm.vmax, "magma"), -(R + 0.5) * step, (R + 0.5) * step, -(R + 0.5) * step, (R + 0.5) * step);
      for (const [gx, gy] of g1) for (const sg of [1, -1]) ring(fft, sg * gx, sg * gy, 13, T.truth);
      for (const [gx, gy] of g2) for (const sg of [1, -1]) ring(fft, sg * gx, sg * gy, 8, T.pink);
      // Zooms: at the origin (Moiré spots ±Δg) and at g₁ (spots g₁ and g₂).
      const zr = clamp(Math.ceil(1.6 * adg * FIELD) + 3, 8, 40), half = (zr + 0.5) * step;
      const zoom = (c, cxk, cyk, title) => {
        const cx = Math.round(cxk / step), cy = Math.round(cyk / step), z = crop(cx, cy, zr);
        c.setDomain([cx * step - half, cx * step + half], [cy * step - half, cy * step + half]);
        c.clear();
        chartImage(c, rasterURL(z.o, z.m, z.m, 0, z.vmax, "magma"), cx * step - half, cx * step + half, cy * step - half, cy * step + half);
        c.note(title, { corner: "bl", size: 18, fill: T.muted });
      };
      zoom(z0, 0, 0, `×${fmt(R / zr, 1)} · ±${fmt(half, 3)} Å⁻¹`);
      for (let j = 0; j < 3; j++) {
        const d = [g1[j][0] - g2[j][0], g1[j][1] - g2[j][1]];
        for (const sg of [1, -1]) ring(z0, sg * d[0], sg * d[1], 9, T.thread, 2.5);
      }
      if (adg > 1e-9) z0.gTop.append(s("line", { x1: z0.x(0), y1: z0.y(0), x2: z0.x(dg[0]), y2: z0.y(dg[1]), stroke: T.thread, "stroke-width": 3, "marker-end": headT[1] }));
      zoom(z1, g1[0][0], g1[0][1], `×${fmt(R / zr, 1)} around g₁`);
      ring(z1, g1[0][0], g1[0][1], 11, T.truth); ring(z1, g2[0][0], g2[0][1], 11, T.pink);
      if (adg > 1e-9) z1.gTop.append(s("line", { x1: z1.x(g2[0][0]), y1: z1.y(g2[0][1]), x2: z1.x(g1[0][0]), y2: z1.y(g1[0][1]), stroke: T.thread, "stroke-width": 3, "marker-end": headT[2] }));
      if (adg > 1e-9) fft.gTop.append(s("line", { x1: fft.x(0), y1: fft.y(0), x2: fft.x(dg[0]), y2: fft.y(dg[1]), stroke: T.thread, "stroke-width": 3, "marker-end": headT[0] }));
    }
    // Readout
    const a2 = A * (1 + dl), Dmis = dl > 0 ? A * a2 / (a2 - A) : Infinity, Dtw = th > 0 ? A / (2 * Math.sin(th * Math.PI / 360)) : Infinity;
    const dS = (d) => (Number.isFinite(d) ? `${fmt(d, d < 100 ? 1 : 0)} Å` : "∞ (no Moiré)");
    let l1 = Number.isFinite(D)
      ? `Moiré period <b>D = ${fmt(D, 1)} Å</b> (= ${fmt(D / A, 1)} × a), from |Δg| = |g₁ − g₂| = ${fmt(adg, 4)} Å⁻¹ and D = 2/(√3 |Δg|).` +
        (D > FIELD / 2 ? ` <span class="warn">D is larger than half the field.</span>` : "")
      : `<span class="warn">Identical lattices: Δg = 0, no Moiré.</span>`;
    const l2 = `Mismatch only: D = a₁a₂/|a₂ − a₁| = ${dS(Dmis)}. Twist only: D = a/(2 sin(θ/2)) = ${dS(Dtw)}.`;
    let l3;
    if (lastTurn && lastTurn.dl === dl && Math.abs(lastTurn.th - th) < 1e-9) {
      l3 = `Specimen turned ${fmt(lastTurn.turn, 0)}°: its lattice turned ${fmt(lastTurn.turn, 0)}°, the <span class="bad">Moiré turned ${fmt(lastTurn.dphi, 1)}°</span> (× ${fmt(Math.abs(lastTurn.dphi / lastTurn.turn), 1)}).`;
    } else if (Number.isFinite(D)) {
      const dphi = wrapDeg(ang(dgOf(dl, th + 1 <= 10 + 1e-9 ? th + 1 : th - 1)) - ang(dg));
      l3 = `A 1° turn of the specimen turns its lattice 1°, but the <span class="bad">Moiré ${fmt(Math.abs(dphi), 1)}°</span>: real structure turns with the specimen, Moiré turns more.`;
    } else l3 = "Add mismatch or twist to make a Moiré.";
    out.set(`${l1}<br>${l2}<br>${l3}`);
  }
  draw(true);
});

// ---------------------------------------------------------------------------------------------
// psf: measured image = object ⊗ PSF + Poisson noise. Object (96 x 96 pixels): dumbbells of two
// Gaussian columns (σ = 0.6 px) d = 4 px apart (left), and a square lattice of 3 px period (right).
// PSF = (1 − w)·Gauss(σ) + w·Gauss(6 px). Scintillator + CMOS: σ = 2.5 px with a 15 % halo (the
// light spread in the scintillator); direct electron detector: σ = 0.5 px, no halo.
// MTF(f) = |FT(PSF)| from the discrete kernels. Close pair "resolved" when the dip between the two
// columns (mean over all dumbbells, 3 rows each) is below 0.81 of the peaks (Rayleigh-type limit).
// ---------------------------------------------------------------------------------------------
register("psf", (el) => {
  injectStyle();
  const N = 96, SC = 0.6, D = 4, P = 3, FL = 1 / P, BG = 0.05, HALO = 6, SZ = 384, PR = 12;
  const DET = { scint: { sigma: 2.5, w: 0.15 }, direct: { sigma: 0.5, w: 0 } };
  // Object
  const obj = new Float64Array(N * N), cols = [], pairs = [];
  for (let cy = 6; cy < N; cy += 12) for (let cx = 8; cx < 48; cx += 12) {
    cols.push([cx - D / 2, cy], [cx + D / 2, cy]); pairs.push([cx, cy]);
  }
  for (let cy = 3; cy < N - 1; cy += P) for (let cx = 57; cx < N - 1; cx += P) cols.push([cx, cy]);
  for (const [cx, cy] of cols)
    for (let i = Math.max(0, Math.floor(cy - 4)); i <= Math.min(N - 1, Math.ceil(cy + 4)); i++)
      for (let j = Math.max(0, Math.floor(cx - 4)); j <= Math.min(N - 1, Math.ceil(cx + 4)); j++)
        obj[i * N + j] += Math.exp(-((j - cx) ** 2 + (i - cy) ** 2) / (2 * SC * SC));

  const kernel = (sg) => {
    if (sg < 0.05) return [1];
    const r = Math.ceil(3.5 * sg), k = [];
    for (let m = -r; m <= r; m++) k.push(Math.exp(-m * m / (2 * sg * sg)));
    const t = k.reduce((a, b) => a + b, 0);
    return k.map((v) => v / t);
  };
  const mtfK = (k, f) => { const r = (k.length - 1) / 2; let v = 0; k.forEach((c, i) => { v += c * Math.cos(2 * Math.PI * f * (i - r)); }); return v; };
  const mtf = (sg, w, f) => Math.abs((1 - w) * mtfK(kernel(sg), f) + w * mtfK(kernel(HALO), f));
  function conv(src, k) {                                               // separable, edge pixels repeat
    const r = (k.length - 1) / 2, tmp = new Float64Array(N * N), out = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      let v = 0;
      for (let m = -r; m <= r; m++) v += k[m + r] * src[i * N + clamp(j + m, 0, N - 1)];
      tmp[i * N + j] = v;
    }
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      let v = 0;
      for (let m = -r; m <= r; m++) v += k[m + r] * tmp[clamp(i + m, 0, N - 1) * N + j];
      out[i * N + j] = v;
    }
    return out;
  }
  const haloImg = conv(obj, kernel(HALO));

  // DOM
  const imTrue = imageCanvas({ n: N, size: SZ, label: "true object (96 × 96 px)" });
  const imPsf = imageCanvas({ n: 2 * PR + 1, size: 200, label: "PSF (√ scale)" });
  const imMeas = imageCanvas({ n: N, size: SZ, label: "measured image" });
  const psfCap = caption("");
  const mc = chart({ width: 590, height: SZ + 24, x: { domain: [0, 0.5], label: "spatial frequency (cycles / pixel)" },
    y: { domain: [0, 1.05], label: "MTF = |FT(PSF)|" }, margin: { l: 76, r: 24, t: 14, b: 62 } });
  const out = readout();
  let det = "direct";
  const sSig = slider({ label: "PSF σ", min: 0, max: 4, value: DET[det].sigma, step: 0.05, digits: 2, unit: "px", onInput: draw });
  const sDose = slider({ label: "dose", min: 10, max: 10000, value: 300, log: true, digits: 0, unit: "counts / column", onInput: draw });
  const tDet = toggle({ label: "detector", value: det, options: [{ value: "scint", label: "scintillator + CMOS" }, { value: "direct", label: "direct electron detector" }],
    onChange: (v) => { det = v; sSig.set(DET[v].sigma, false); draw(); } });
  imTrue.draw(obj, 0, 1.05, "gray");
  el.append(card(controls(tDet, sSig, sDose),
    row(imTrue, h("div", { class: "xp-col xp-psf-mid" }, imPsf, psfCap), imMeas,
      h("div", {}, legend([[T.data, "scintillator + CMOS"], [T.data, "direct detector", "dash"], [T.thread, "your setting"]]), mc.el)), out));

  function draw() {
    const sg = sSig.value, w = DET[det].w, dose = sDose.value, k = kernel(sg);
    const core = conv(obj, k), blur = core.map((v, i) => (1 - w) * v + w * haloImg[i] + BG);
    const r = rng(17), meas = blur.map((v) => r.poisson(dose * v));
    let mx = 0;
    for (const v of blur) mx = Math.max(mx, v * dose);
    imMeas.draw(meas, 0, mx + 2 * Math.sqrt(mx), "gray");
    imMeas.setLabel(`measured = object ⊗ PSF + Poisson noise`);
    // PSF picture
    const m = 2 * PR + 1, pk = new Float64Array(m * m), kh = kernel(HALO), at = (kk, d) => { const rr = (kk.length - 1) / 2; return Math.abs(d) <= rr ? kk[d + rr] : 0; };
    for (let i = 0; i < m; i++) for (let j = 0; j < m; j++)
      pk[i * m + j] = Math.sqrt((1 - w) * at(k, i - PR) * at(k, j - PR) + w * at(kh, i - PR) * at(kh, j - PR));
    imPsf.draw(pk, 0, Math.max(...pk), "magma");
    psfCap.innerHTML = `25 × 25 px. Core σ = ${fmt(sg, 2)} px${w ? `, plus a ${fmt(100 * w, 0)} % halo (σ = ${HALO} px)` : ", no halo"}.`;
    // Close pair: dip between the columns relative to the peaks
    let pkSum = 0, mid = 0;
    for (const [cx, cy] of pairs) for (let dy = -1; dy <= 1; dy++) {
      const row0 = (cy + dy) * N;
      pkSum += meas[row0 + cx - D / 2] + meas[row0 + cx + D / 2]; mid += meas[row0 + cx];
    }
    const dip = mid / (pkSum / 2), res = dip < 0.81;
    // MTF chart
    const fs = linspace(0, 0.5, 120);
    mc.clear();
    mc.ruleX(FL, { stroke: T.truth, width: 2.5 });
    mc.text(FL, 1.0, `3 px lattice (${fmt(FL, 2)} px⁻¹)`, { fill: T.truth, dx: -10, anchor: "end", size: 19 });
    mc.line(fs.map((f) => [f, mtf(DET.scint.sigma, DET.scint.w, f)]), { stroke: T.data, width: 3 });
    mc.line(fs.map((f) => [f, mtf(DET.direct.sigma, DET.direct.w, f)]), { stroke: T.data, width: 3, dash: "9 7" });
    mc.line(fs.map((f) => [f, mtf(sg, w, f)]), { stroke: T.thread, width: 5 });
    const mL = mtf(sg, w, FL);
    mc.dots([[FL, mL]], { r: 8, fill: T.thread, stroke: "#fff", top: true });
    const mcls = mL >= 0.3 ? "good" : mL >= 0.1 ? "warn" : "bad";
    out.set(`Close pair (d = ${D} px): centre / peak = ${fmt(dip, 2)} → <span class="${res ? "good" : "bad"}">${res ? "resolved" : "not resolved"}</span> (limit 0.81).<br>` +
      `MTF at the lattice frequency ${fmt(FL, 2)} px⁻¹ = <span class="${mcls}">${fmt(mL, 2)}</span>: the 3 px lattice keeps ${fmt(100 * mL, 0)} % of its contrast.`);
  }
  draw();
});
