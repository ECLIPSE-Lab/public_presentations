// Interactive figures for DSEM Week 2. The OJS cells in 01_intro.qmd call these functions.
// The models are the same as in img/make_figures.py and img/make_visuals.py.
// All data are synthetic. A fixed seed gives the same default view on every load (and in the PDF).

export const C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"];
const INK = "#222222", INK2 = "#555555";

// ---------------------------------------------------------------------------------------------
// Random numbers
// ---------------------------------------------------------------------------------------------
export function rng(seed) {
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
    if (lam <= 0) return 0;
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

function logFactorial(k) {
  if (k < 20) { let s = 0; for (let i = 2; i <= k; i++) s += Math.log(i); return s; }
  const n = k + 1;                                    // Stirling series for log Gamma(n)
  return (n - 0.5) * Math.log(n) - n + 0.5 * Math.log(2 * Math.PI) + 1 / (12 * n) - 1 / (360 * n ** 3);
}

// ---------------------------------------------------------------------------------------------
// Fitting tools
// ---------------------------------------------------------------------------------------------
// Least squares A x = y with Householder QR (A is an array of rows).
function lstsq(A, y) {
  const m = A.length, n = A[0].length, R = A.map((r) => r.slice()), b = y.slice();
  for (let j = 0; j < n; j++) {
    let norm = 0;
    for (let i = j; i < m; i++) norm += R[i][j] ** 2;
    norm = Math.sqrt(norm);
    const v = new Array(m).fill(0);
    for (let i = j; i < m; i++) v[i] = R[i][j];
    v[j] -= R[j][j] > 0 ? -norm : norm;
    let vv = 0;
    for (let i = j; i < m; i++) vv += v[i] ** 2;
    if (vv === 0) continue;
    for (let k = j; k < n; k++) {
      let s = 0;
      for (let i = j; i < m; i++) s += v[i] * R[i][k];
      s *= 2 / vv;
      for (let i = j; i < m; i++) R[i][k] -= s * v[i];
    }
    let s = 0;
    for (let i = j; i < m; i++) s += v[i] * b[i];
    s *= 2 / vv;
    for (let i = j; i < m; i++) b[i] -= s * v[i];
  }
  const x = new Array(n).fill(0);
  for (let j = n - 1; j >= 0; j--) {
    let s = b[j];
    for (let k = j + 1; k < n; k++) s -= R[j][k] * x[k];
    x[j] = s / R[j][j];
  }
  return x;
}

// Legendre polynomials P_0 ... P_deg at z (same function space as np.polyfit, better conditioned).
function legendre(z, deg) {
  const P = [1, z];
  for (let k = 1; k < deg; k++) P.push(((2 * k + 1) * z * P[k] - k * P[k - 1]) / (k + 1));
  return P.slice(0, deg + 1);
}

function polyfit(z, y, deg, zEval) {
  const c = lstsq(z.map((t) => legendre(t, deg)), y);
  return zEval.map((t) => legendre(t, deg).reduce((s, p, i) => s + p * c[i], 0));
}

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
      else {
        for (let i = 1; i <= n; i++) { pts[i] = pts[i].map((v, j) => pts[0][j] + 0.5 * (v - pts[0][j])); vals[i] = f(pts[i]); }
      }
    }
  }
  const best = vals.indexOf(Math.min(...vals));
  return pts[best];
}

// ---------------------------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------------------------
const FONT = "font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;";

export function card(...children) {
  const d = document.createElement("div");
  d.style.cssText = `background:white; color:${INK}; border-radius:12px; padding:10px 14px; ${FONT}`;
  children.forEach((ch) => d.append(ch));
  return d;
}

export function row(...children) {
  const d = document.createElement("div");
  d.style.cssText = "display:flex; gap:18px; align-items:flex-start; justify-content:center;";
  children.forEach((ch) => d.append(ch));
  return d;
}

export function note(htmlText, size = 22) {
  const p = document.createElement("div");
  p.style.cssText = `font-size:${size}px; line-height:1.35; color:${INK}; margin:4px 2px; ${FONT}`;
  p.innerHTML = htmlText;
  return p;
}

function plotStyle(fontSize = 17) {
  return { fontSize: `${fontSize}px`, background: "white", color: INK, overflow: "visible" };
}

// Grey-scale image of an n x n array on a canvas of size x size display pixels.
function imageCanvas(data, n, size, vmin, vmax, title) {
  const cv = document.createElement("canvas");
  cv.width = n; cv.height = n;
  cv.style.cssText = `width:${size}px; height:${size}px; image-rendering:pixelated; display:block;`;
  const ctx = cv.getContext("2d"), im = ctx.createImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    const g = Math.max(0, Math.min(255, Math.round(255 * (data[i] - vmin) / (vmax - vmin))));
    im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = g; im.data[4 * i + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  const wrap = document.createElement("div");
  if (title) wrap.append(note(title, 20));
  wrap.append(cv);
  return wrap;
}

const fmt = (x, d = 1) => x.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });

// ---------------------------------------------------------------------------------------------
// 1. Bias-variance: polynomial fits of a synthetic carbon K-edge
// ---------------------------------------------------------------------------------------------
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const BV_E = linspace(250, 330, 60), BV_Et = linspace(250, 330, 300);
const toZ = (E) => (E - 290) / 40;

function edgeModel(E, [A, r, B], e0 = 284.0, w = 1.5) {
  const step = 1 / (1 + Math.exp(-(E - e0) / w));
  return A * (E / 250) ** -r + B * step * (E / e0) ** -3;
}

function fitEdge(y) {
  const loss = (p) => BV_E.reduce((s, E, i) => s + (edgeModel(E, p) - y[i]) ** 2, 0);
  return nelderMead(loss, [90, 2.5, 30], [10, 0.3, 5], 2000);
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

// Train and test error against polynomial degree, averaged over `reps` pairs of data sets.
export function biasVarianceCurves(reps = 200, seed = 7) {
  const r = rng(seed), truth = BV_E.map((E) => edgeModel(E, [100, 3, 40])), z = BV_E.map(toZ);
  const degs = Array.from({ length: 16 }, (_, i) => i);
  const tr = new Array(16).fill(0), te = new Array(16).fill(0);
  let tePhys = 0;
  for (let k = 0; k < reps; k++) {
    const y1 = truth.map(r.poisson), y2 = truth.map(r.poisson);
    for (const d of degs) {
      const f = polyfit(z, y1, d, z);
      tr[d] += mean(f.map((v, i) => (v - y1[i]) ** 2)) / reps;
      te[d] += mean(f.map((v, i) => (v - y2[i]) ** 2)) / reps;
    }
    const p = fitEdge(y1);
    tePhys += mean(BV_E.map((E, i) => (edgeModel(E, p) - y2[i]) ** 2)) / reps;
  }
  return { degs, tr, te, tePhys };
}

export function biasVariance(Plot, degree, curves, seed = 3) {
  const r = rng(seed), y = BV_E.map((E) => r.poisson(edgeModel(E, [100, 3, 40])));
  const poly = polyfit(BV_E.map(toZ), y, degree, BV_Et.map(toZ));
  const phys = fitEdge(y);
  const best = curves.te.indexOf(Math.min(...curves.te));
  const left = Plot.plot({
    width: 960, height: 470, marginLeft: 70, marginTop: 34, marginBottom: 55, style: plotStyle(),
    x: { label: "energy loss (eV) →", domain: [250, 330] }, y: { label: "↑ counts", domain: [0, 150], grid: true },
    marks: [
      Plot.line(BV_Et, { x: (d) => d, y: (d) => edgeModel(d, [100, 3, 40]), stroke: "#dddddd", strokeWidth: 8 }),
      Plot.dot(BV_E, { x: (d) => d, y: (d, i) => y[i], fill: INK2, r: 3.5 }),
      Plot.line(BV_Et, { x: (d) => d, y: (d) => edgeModel(d, phys), stroke: C[2], strokeWidth: 3, strokeDasharray: "8,5" }),
      Plot.line(BV_Et, { x: (d) => d, y: (d, i) => Math.max(0, Math.min(150, poly[i])), stroke: C[1], strokeWidth: 3.5 }),
      Plot.text([`polynomial, degree ${degree}`], { frameAnchor: "top-right", dx: -10, dy: 8, fill: C[1], fontSize: 21, fontWeight: "bold" }),
      Plot.text(["physical edge model (dashed)"], { frameAnchor: "top-right", dx: -10, dy: 36, fill: C[2], fontSize: 19 }),
    ],
  });
  const pts = curves.degs.flatMap((d) => [{ d, v: curves.tr[d], k: "training data" }, { d, v: curves.te[d], k: "new data" }]);
  const right = Plot.plot({
    width: 720, height: 470, marginLeft: 70, marginTop: 34, marginBottom: 55, style: plotStyle(),
    x: { label: "model complexity (polynomial degree) →", domain: [-0.5, 15.5] },
    y: { label: "↑ mean squared error", type: "log", grid: true },
    color: { domain: ["training data", "new data"], range: [C[0], C[1]], legend: false },
    marks: [
      Plot.ruleY([curves.tePhys], { stroke: C[2], strokeDasharray: "8,5", strokeWidth: 2.5 }),
      Plot.line(pts, { x: "d", y: "v", stroke: "k", strokeWidth: 2.5 }),
      Plot.dot(pts, { x: "d", y: "v", fill: "k", r: 4 }),
      Plot.ruleX([degree], { stroke: INK, strokeWidth: 2 }),
      Plot.dot(pts.filter((p) => p.d === degree), { x: "d", y: "v", fill: "k", r: 9, stroke: INK, strokeWidth: 2 }),
      Plot.text(["error on new data"], { frameAnchor: "top-right", dx: -10, dy: 8, fill: C[1], fontSize: 19 }),
      Plot.text(["error on training data"], { frameAnchor: "top-right", dx: -10, dy: 32, fill: C[0], fontSize: 19 }),
      Plot.text(["physical model, new data"], { frameAnchor: "top-right", dx: -10, dy: 56, fill: C[2], fontSize: 19 }),
    ],
  });
  const verdict = degree < best ? "too simple: high bias" : degree > best + 2 ? "too flexible: high variance" : "near the best polynomial";
  return card(row(left, right), note(
    `Degree ${degree} → <b>${verdict}</b>. Error on new data: ${fmt(curves.te[degree], 0)} ` +
    `(best polynomial: degree ${best}, ${fmt(curves.te[best], 0)}; physical model: ${fmt(curves.tePhys, 0)}). ` +
    `Averages of 200 synthetic data sets.`));
}

// ---------------------------------------------------------------------------------------------
// 2. Nyquist sampling of a 2 Å lattice profile (1-D)
// ---------------------------------------------------------------------------------------------
// Frequency (1/Å) that a cosine of period a appears at, after sampling with pixel dx.
function aliasFreq(f, dx) {
  const fs = 1 / dx;
  return Math.abs(f - Math.round(f / fs) * fs);
}

export function nyquist1D(Plot, dx, a = 2.0) {
  const xs = linspace(0, 12, 1200), samples = [];
  for (let x = 0; x <= 12 + 1e-9; x += dx) samples.push(x);
  const fa = aliasFreq(1 / a, dx), ppp = a / dx, under = dx > a / 2 + 1e-9;
  const col = ppp >= 3 ? C[2] : under ? C[1] : C[3];
  const marks = [
    Plot.line(xs, { x: (d) => d, y: (d) => Math.cos(2 * Math.PI * d / a), stroke: "#dddddd", strokeWidth: 7 }),
  ];
  if (under) marks.push(Plot.line(xs, { x: (d) => d, y: (d) => Math.cos(2 * Math.PI * fa * d), stroke: C[1], strokeWidth: 2.5, strokeDasharray: "8,6" }));
  marks.push(
    Plot.line(samples, { x: (d) => d, y: (d) => Math.cos(2 * Math.PI * d / a), stroke: col, strokeWidth: 2 }),
    Plot.dot(samples, { x: (d) => d, y: (d) => Math.cos(2 * Math.PI * d / a), fill: col, r: 7 }),
  );
  const p = Plot.plot({
    width: 900, height: 420, marginLeft: 30, marginTop: 34, marginBottom: 55, style: plotStyle(18),
    x: { label: "position (Å) →", domain: [0, 12] }, y: { domain: [-1.3, 1.3], axis: null }, marks,
  });
  const verdict = ppp >= 3 ? "fine: the period is clear" : !under ? "at or near the Nyquist limit: only just" :
    `below Nyquist: a false ${fmt(1 / fa, 1)} Å period appears (dashed)`;
  return card(p, note(
    `Pixel ${fmt(dx, 2)} Å = <b>${fmt(ppp, 2)} pixels per 2 Å period</b> → <span style="color:${col}"><b>${verdict}</b></span>`));
}

// ---------------------------------------------------------------------------------------------
// 3. Undersampled 2-D lattice and its FFT
// ---------------------------------------------------------------------------------------------
function lattice(n, dx, a, angle) {
  const k = 2 * Math.PI / a, c = Math.cos(angle), s = Math.sin(angle), im = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const x = j * dx, y = i * dx, u = x * c + y * s, v = -x * s + y * c;
    im[i * n + j] = (Math.cos(k * u) + 1) * (Math.cos(k * v) + 1) / 4;
  }
  return im;
}

// |FFT|, fftshifted, by two passes of a direct DFT (n is at most a few hundred).
function dft2abs(im, n) {
  const cs = new Float64Array(n * n), sn = new Float64Array(n * n);
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { const t = -2 * Math.PI * ((a * b) % n) / n; cs[a * n + b] = Math.cos(t); sn[a * n + b] = Math.sin(t); }
  const m = mean(im), re1 = new Float64Array(n * n), im1 = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let kx = 0; kx < n; kx++) {
    let sr = 0, si = 0;
    for (let j = 0; j < n; j++) { const v = im[i * n + j] - m; sr += v * cs[kx * n + j]; si += v * sn[kx * n + j]; }
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

export function undersampled2D(Plot, dx, angleDeg, a = 2.0, field = 54) {
  const n = Math.round(field / dx), th = angleDeg * Math.PI / 180;
  const im = lattice(n, dx, a, th), F = dft2abs(im, n), h = Math.floor(n / 2);
  const kOf = (idx) => (idx - h) / (n * dx) * 10;                     // nm^-1
  const fmax = Math.max(...F), spots = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (F[i * n + j] > 0.2 * fmax) spots.push({ kx: kOf(j), ky: kOf(i) });
  const kn = 10 / (2 * dx), k0 = 10 / a;
  const truth = [0, 90, 180, 270].map((d) => ({ kx: k0 * Math.cos(th + d * Math.PI / 180), ky: k0 * Math.sin(th + d * Math.PI / 180) }));
  const fold = (k) => k - Math.round(k / (2 * kn)) * 2 * kn;          // fold into [-kn, kn]
  const kpeak = Math.hypot(fold(k0 * Math.cos(th)), fold(k0 * Math.sin(th)));
  const L = 10.5;
  const fft = Plot.plot({
    width: 470, height: 470, margin: 40, style: { ...plotStyle(16), background: "black", color: "white" },
    x: { domain: [-L, L], label: "kx (nm⁻¹)" }, y: { domain: [L, -L], label: "ky (nm⁻¹)" },
    marks: [
      Plot.rect([0], { x1: -kn, x2: kn, y1: -kn, y2: kn, fill: "none", stroke: C[3], strokeWidth: 2.5, strokeDasharray: "8,5" }),
      Plot.dot(truth, { x: "kx", y: "ky", r: 12, stroke: C[1], strokeWidth: 2.5 }),
      Plot.dot(spots, { x: "kx", y: "ky", r: 7, fill: "white" }),
    ],
  });
  const img = imageCanvas(im, n, 470, 0, 1, `${n} × ${n} pixels of ${fmt(dx, 2)} Å (54 Å field)`);
  const ok = kn > k0 + 1e-9;
  const msg = ok ? `The 5 nm⁻¹ spots lie inside the recordable range ±${fmt(kn, 2)} nm⁻¹: the 2 Å lattice is recorded correctly.`
    : `Recordable range ±${fmt(kn, 2)} nm⁻¹ (orange box) is smaller than 5 nm⁻¹: the main spot folds to ${fmt(kpeak, 2)} nm⁻¹, a false ${fmt(10 / kpeak, 1)} Å lattice.`;
  const legend = note(`<span style="color:${C[1]}">○ true spot position</span> &nbsp; ● measured spot`, 19);
  const fftWrap = document.createElement("div"); fftWrap.append(legend, fft);
  return card(row(img, fftWrap), note(msg));
}

// ---------------------------------------------------------------------------------------------
// 4. Dose-dependent image quality
// ---------------------------------------------------------------------------------------------
const PH_N = 128;
const PHANTOM = (() => {                               // same phantom as the notebook, 128 x 128 crop
  const im = new Float64Array(PH_N * PH_N), sp = 16, w = 2;
  for (let cy = sp / 2; cy < PH_N; cy += sp) for (let cx = sp / 2; cx < PH_N; cx += sp)
    for (let i = 0; i < PH_N; i++) for (let j = 0; j < PH_N; j++) im[i * PH_N + j] += Math.exp(-((j - cx) ** 2 + (i - cy) ** 2) / (2 * w * w));
  let mx = 0;
  for (let i = 0; i < im.length; i++) { im[i] += 0.08; mx = Math.max(mx, im[i]); }
  return im.map((v) => v / mx);
})();
const PH_MEAN = mean(PHANTOM);

export function doseImage(Plot, dose, seed = 11) {
  const r = rng(seed + Math.round(dose * 100)), noisy = PHANTOM.map((v) => r.poisson(dose * v) / dose);
  const err = Math.sqrt(mean(noisy.map((v, i) => (v - PHANTOM[i]) ** 2)));
  const snrMeas = PH_MEAN / err, snrAvg = Math.sqrt(dose * PH_MEAN), snrPeak = Math.sqrt(dose);
  const img = imageCanvas(noisy, PH_N, 420, 0, 1, `dose ${fmt(dose, dose < 10 ? 1 : 0)} counts at a column peak`);
  const ds = Array.from({ length: 81 }, (_, i) => 10 ** (i / 20));
  const curve = (f, k) => ds.map((d) => ({ d, v: f(d), k }));
  const lines = [...curve((d) => Math.sqrt(d), "peak column: √dose"), ...curve((d) => Math.sqrt(d * PH_MEAN), "image average: √(dose × mean)")];
  const p = Plot.plot({
    width: 470, height: 420, marginLeft: 60, marginTop: 34, marginBottom: 55, style: plotStyle(16),
    x: { type: "log", label: "dose →", domain: [1, 10000] }, y: { type: "log", label: "↑ SNR", domain: [0.3, 120], grid: true },
    color: { domain: ["peak column: √dose", "image average: √(dose × mean)"], range: [C[0], C[1]] },
    marks: [
      Plot.line(lines, { x: "d", y: "v", stroke: "k", strokeWidth: 2.5 }),
      Plot.ruleY([1], { stroke: INK2, strokeDasharray: "4,4" }),
      Plot.dot([{ d: dose, v: snrPeak }, { d: dose, v: snrMeas }], { x: "d", y: "v", r: 8, fill: (d, i) => C[i], stroke: INK }),
      Plot.text(["peak column"], { frameAnchor: "top-left", dx: 12, dy: 8, fill: C[0], fontSize: 18 }),
      Plot.text(["image average"], { frameAnchor: "top-left", dx: 12, dy: 30, fill: C[1], fontSize: 18 }),
    ],
  });
  return card(row(img, p), note(
    `Image-average SNR: measured <b>${fmt(snrMeas, 1)}</b>, theory √(dose × ${fmt(PH_MEAN, 3)}) = ${fmt(snrAvg, 1)}. ` +
    `Peak column: √dose = ${fmt(snrPeak, 1)}.`, 20));
}

// ---------------------------------------------------------------------------------------------
// 5. Per-pixel loss: MSE vs Poisson NLL for one observed count
// ---------------------------------------------------------------------------------------------
export function lossCurves(Plot, x) {
  const lam = linspace(0.02, 16, 800);
  const pmin = x > 0 ? x - x * Math.log(x) : 0;
  const pois = (l) => l - x * Math.log(l) - pmin;
  const panel = (f, title, col) => Plot.plot({
    width: 820, height: 430, marginLeft: 60, marginTop: 34, marginBottom: 55, style: plotStyle(),
    x: { label: "predicted rate λ̂ (counts) →", domain: [0, 16] }, y: { label: "↑ loss", domain: [0, 30], grid: true },
    marks: [
      Plot.ruleX([x], { stroke: INK2, strokeDasharray: "4,4" }),
      Plot.line(lam.filter((l) => f(l) <= 30), { x: (d) => d, y: (d) => f(d), stroke: col, strokeWidth: 4 }),
      Plot.text([title], { frameAnchor: "top-right", dx: -10, dy: 6, fontSize: 21, fontWeight: "bold", fill: col }),
    ],
  });
  const curv = x > 0 ? `Curvature at the minimum: MSE 2 for every x; Poisson 1/x = <b>${fmt(1 / x, 2)}</b>. ` +
    `The Poisson loss gives brighter pixels less weight (1/variance).`
    : `The Poisson loss λ̂ rises linearly from λ̂ = 0; MSE gives only λ̂², which is tiny near zero.`;
  return card(row(panel((l) => (x - l) ** 2, "MSE (x − λ̂)²", C[0]), panel(pois, "Poisson NLL λ̂ − x log λ̂ (shifted)", C[1])),
    note(`Observed count x = <b>${x}</b>. ${curv}`));
}

// ---------------------------------------------------------------------------------------------
// 6. Variance-mean (photon transfer) plot
// ---------------------------------------------------------------------------------------------
export function ptc(Plot, g, sr, seed = 5) {
  const r = rng(seed), lams = Array.from({ length: 25 }, (_, i) => 10 ** (-0.5 + 3.5 * i / 24));
  const pts = lams.map((l) => {
    let s = 0, s2 = 0;
    const n = 4000;
    for (let k = 0; k < n; k++) { const v = g * r.poisson(l) + sr * r.normal(); s += v; s2 += v * v; }
    const m = s / n;
    return { m, v: s2 / n - m * m };
  });
  const mx = mean(pts.map((p) => p.m)), vy = mean(pts.map((p) => p.v));
  const slope = pts.reduce((s, p) => s + (p.m - mx) * (p.v - vy), 0) / pts.reduce((s, p) => s + (p.m - mx) ** 2, 0);
  const icpt = vy - slope * mx, mmax = Math.max(...pts.map((p) => p.m)), cross = sr * sr / g;
  const mm = linspace(0, mmax, 100);
  const lin = Plot.plot({
    width: 780, height: 430, marginLeft: 80, marginTop: 34, marginBottom: 55, style: plotStyle(),
    x: { label: "mean signal (DN) →" }, y: { label: "↑ variance (DN²)", grid: true },
    marks: [
      Plot.line(mm, { x: (d) => d, y: (d) => slope * d + icpt, stroke: C[1], strokeWidth: 3 }),
      Plot.dot(pts, { x: "m", y: "v", fill: C[0], r: 6 }),
      Plot.text([`fit: slope = ${fmt(slope, 2)}, intercept = ${fmt(icpt, 0)}`], { frameAnchor: "top-left", dx: 10, dy: 6, fill: C[1], fontSize: 20 }),
    ],
  });
  const ml = linspace(Math.log10(pts[0].m > 0 ? pts[0].m : 0.3), Math.log10(mmax), 100).map((t) => 10 ** t);
  const log = Plot.plot({
    width: 780, height: 430, marginLeft: 80, marginTop: 34, marginBottom: 55, style: plotStyle(),
    x: { type: "log", label: "mean signal (DN) →" }, y: { type: "log", label: "↑ variance (DN²)", grid: true },
    marks: [
      Plot.line(ml, { x: (d) => d, y: () => sr * sr, stroke: C[2], strokeWidth: 3, strokeDasharray: "8,5" }),
      Plot.line(ml, { x: (d) => d, y: (d) => g * d, stroke: C[1], strokeWidth: 3, strokeDasharray: "3,4" }),
      Plot.ruleX([cross], { stroke: INK2, strokeDasharray: "4,4" }),
      Plot.dot(pts.filter((p) => p.m > 0 && p.v > 0), { x: "m", y: "v", fill: C[0], r: 6 }),
      Plot.text(["read noise σr²"], { frameAnchor: "top-left", dx: 10, dy: 6, fill: C[2], fontSize: 20 }),
      Plot.text(["shot noise g·x̄"], { frameAnchor: "top-left", dx: 10, dy: 30, fill: C[1], fontSize: 20 }),
    ],
  });
  return card(row(lin, log), note(
    `True g = ${fmt(g, 1)}, σr² = ${fmt(sr * sr, 0)}; fitted g = <b>${fmt(slope, 2)}</b>, σr² = <b>${fmt(icpt, 0)}</b>. ` +
    `Crossover at x̄ = σr²/g = <b>${fmt(cross, 1)} DN</b>: below it read noise dominates, above it shot noise.`));
}

// ---------------------------------------------------------------------------------------------
// 7. Fitting a low-count peak with four losses
// ---------------------------------------------------------------------------------------------
const PK_E = linspace(0, 1, 64), PK_TRUE = [1.0, 0.05, 0.05];
export const KINDS = ["MSE", "weighted χ²", "Anscombe + MSE", "Poisson NLL"];

function peakModel([a, b, s], dose) {
  return PK_E.map((E) => dose * (a * Math.exp(-((E - 0.5) ** 2) / (2 * s * s)) + b));
}

// Same losses and bounds as fit_peak in img/make_figures.py; the bounds become a penalty.
export function fitPeak(x, dose, kind) {
  const loss = (q) => {
    const a = Math.max(q[0], 0), b = Math.max(q[1], 1e-6), s = Math.min(Math.max(q[2], 0.01), 0.3);
    const pen = 1e6 * ((a - q[0]) ** 2 + (b - q[1]) ** 2 + (s - q[2]) ** 2);
    const lam = peakModel([a, b, s], dose);
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

export function peakStats(dose, reps = 150, seed = 13) {
  const r = rng(seed + Math.round(dose * 10)), truth = peakModel(PK_TRUE, dose);
  const totTrue = peakModel(PK_TRUE, 1).reduce((s, v) => s + v, 0);
  const est = Object.fromEntries(KINDS.map((k) => [k, []]));
  for (let i = 0; i < reps; i++) {
    const x = truth.map(r.poisson);
    for (const k of KINDS) est[k].push(fitPeak(x, dose, k));
  }
  return KINDS.map((k) => {
    const tot = est[k].map((q) => peakModel(q, 1).reduce((s, v) => s + v, 0)), w = est[k].map((q) => q[2]);
    const mw = mean(w);
    return { kind: k, bias: 100 * (mean(tot) / totTrue - 1), sd: 100 * Math.sqrt(mean(w.map((v) => (v - mw) ** 2))) / PK_TRUE[2] };
  });
}

export function lowCountPeak(Plot, dose, draw = 0) {
  const r = rng(101 + draw), x = peakModel(PK_TRUE, dose).map(r.poisson);
  const fits = KINDS.map((k) => fitPeak(x, dose, k)), stats = peakStats(dose);
  const dash = ["", "8,5", "3,4", ""];
  const spec = Plot.plot({
    width: 760, height: 450, marginLeft: 60, marginTop: 34, marginBottom: 55, style: plotStyle(),
    x: { label: "energy / position (a.u.) →" }, y: { label: "↑ counts", grid: true },
    marks: [
      Plot.rectY(x, { x1: (d, i) => PK_E[i] - 0.5 / 63, x2: (d, i) => PK_E[i] + 0.5 / 63, y: (d) => d, fill: "#dddddd" }),
      ...fits.map((q, j) => Plot.line(peakModel(q, dose), { x: (d, i) => PK_E[i], y: (d) => d, stroke: C[j], strokeWidth: 3, strokeDasharray: dash[j] })),
      Plot.line(peakModel(PK_TRUE, dose), { x: (d, i) => PK_E[i], y: (d) => d, stroke: INK, strokeWidth: 2.5, strokeDasharray: "2,4" }),
      Plot.text([`one spectrum, peak dose ${fmt(dose, dose < 10 ? 1 : 0)}`], { frameAnchor: "top-left", dx: 8, dy: 4, fontSize: 19 }),
    ],
  });
  const bars = (key, label) => Plot.plot({
    width: 430, height: 450, marginLeft: 175, marginTop: 34, marginBottom: 55, style: plotStyle(),
    x: { label: `${label} →`, grid: true, ...(key === "bias" ? { domain: [Math.min(-25, ...stats.map((s) => s.bias)), 12] } : {}) },
    y: { domain: KINDS, label: null },
    marks: [
      Plot.barX(stats, { x: key, y: "kind", fill: (d, i) => C[i] }),
      Plot.ruleX([0], { stroke: INK }),
      Plot.text(stats, { x: (d) => Math.max(0, d[key]), y: "kind", text: (d) => fmt(Math.abs(d[key]) < 0.5 ? 0 : d[key], 0) + " %", dx: 8, textAnchor: "start", fontSize: 18 }),
    ],
  });
  return card(row(spec, bars("bias", "bias of integrated counts (%)"), bars("sd", "rel. std of width (%)")),
    note(`Bars: 150 synthetic spectra at this dose, each fitted with the four losses. Dotted black: true rate.`, 19));
}
