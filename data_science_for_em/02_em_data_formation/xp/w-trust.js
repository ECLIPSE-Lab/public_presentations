// Trust: data quality, metadata, pre-flight checklist and the two kinds of uncertainty.
//   artefacts   — eight failure modes on a synthetic HAADF lattice, each with the plot that catches it
//   metadata    — delete header fields and watch which downstream analyses die
//   preflight   — the six-item check before any training run
//   uncertainty — aleatoric (shot noise stays √λ) vs epistemic (gain error falls as 1/√N)
import {
  register, T, rng, poissonPmf, mean, fmt, fft1, fft2mag, h, s, row, col, card, controls,
  readout, slider, toggle, button, legend, chart, imageCanvas, logspace, sup,
} from "./core.js";

// ---------------------------------------------------------------------------------------------
// Styles (one <style> element for the whole group; every selector carries its widget name)
// ---------------------------------------------------------------------------------------------
if (!document.getElementById("xp-trust-style")) document.head.append(h("style", { id: "xp-trust-style", html: `
.xp-artefacts-chips { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.xp-artefacts-chips .xp-chip { font-size: 21px; }
.xp-artefacts-img { position: relative; width: 420px; height: 420px; }
.xp-artefacts-img svg { position: absolute; left: 0; top: 0; pointer-events: none; }
.xp-artefacts-diag { width: 830px; height: 470px; display: flex; justify-content: center; }
.xp-artefacts-verdict { width: 380px; display: flex; flex-direction: column; gap: 14px; padding-top: 34px; }
.xp-artefacts-status { font-size: 30px; font-weight: 700; letter-spacing: 0.03em; padding: 10px 16px; border-radius: 12px; border: 2px solid currentColor; }
.xp-artefacts-num { font-size: 40px; font-weight: 700; color: #fff; line-height: 1.1; font-variant-numeric: tabular-nums; }
.xp-artefacts-lbl { font-size: 19px; color: var(--muted); }
.xp-artefacts-enter { font-size: 20px; color: var(--muted); }
.xp-artefacts-enter b { color: var(--warn); }
.xp-artefacts-msg { font-size: 21px; color: var(--ink); border-left: 5px solid var(--warn); padding: 4px 12px; }
.xp-artefacts-ro .xp-readout { margin-top: 2px; }

.xp-metadata-file { width: 800px; font-family: "JetBrains Mono", "Fira Code", Consolas, monospace; font-size: 20px;
  background: #060912; border: 1px solid rgba(255,255,255,0.12); border-radius: 14px; padding: 14px 18px; }
.xp-metadata-head { color: var(--muted); border-bottom: 1px dashed rgba(255,255,255,0.15); padding-bottom: 8px; margin-bottom: 6px; }
.xp-metadata-head b { color: var(--pois); }
.xp-metadata-field { display: flex; gap: 12px; padding: 3px 8px; border-radius: 8px; cursor: pointer; user-select: none; transition: opacity 0.2s, background 0.15s; }
.xp-metadata-field:hover { background: rgba(244,63,94,0.12); }
.xp-metadata-field .k { color: #93c5fd; width: 210px; flex: none; }
.xp-metadata-field .v { color: var(--ink); }
.xp-metadata-field.off { opacity: 0.33; text-decoration: line-through; text-decoration-color: var(--thread); text-decoration-thickness: 3px; }
.xp-metadata-field.hl { background: rgba(251,191,36,0.16); }
.xp-metadata-tasks { width: 880px; display: flex; flex-direction: column; gap: 10px; }
.xp-metadata-task { display: flex; gap: 16px; align-items: center; padding: 8px 14px; border-radius: 12px;
  background: rgba(30,41,59,0.55); border: 2px solid transparent; transition: border-color 0.2s; }
.xp-metadata-task.hl { border-color: var(--warn); }
.xp-metadata-task .icon { font-size: 36px; font-weight: 800; width: 40px; text-align: center; }
.xp-metadata-task .name { font-size: 23px; color: #fff; }
.xp-metadata-task .sub { font-size: 19px; color: var(--muted); }
.xp-metadata-task .sub .miss { color: var(--thread); font-weight: 700; text-decoration: line-through; }

.xp-preflight-list { width: 1240px; font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
  background: #05080f; border: 2px solid rgba(251,191,36,0.55); border-radius: 14px; overflow: hidden; }
.xp-preflight-bar { background: var(--warn); color: #0b0f19; font-weight: 800; font-size: 22px; letter-spacing: 0.12em;
  padding: 8px 18px; display: flex; justify-content: space-between; }
.xp-preflight-item { padding: 9px 18px; border-top: 1px solid rgba(255,255,255,0.08); cursor: pointer; user-select: none; }
.xp-preflight-item:hover { background: rgba(255,255,255,0.04); }
.xp-preflight-line { display: flex; align-items: baseline; gap: 12px; font-size: 23px; }
.xp-preflight-line .n { color: var(--muted); width: 30px; }
.xp-preflight-line .t { color: #fff; font-weight: 700; letter-spacing: 0.04em; white-space: nowrap; }
.xp-preflight-line .what { color: var(--muted); font-size: 19px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.xp-preflight-line .dots { flex: 1; border-bottom: 3px dotted rgba(148,163,184,0.35); transform: translateY(-6px); min-width: 30px; }
.xp-preflight-line .resp { width: 170px; text-align: right; font-weight: 800; letter-spacing: 0.08em; color: var(--warn); }
.xp-preflight-item.done .resp { color: var(--truth); }
.xp-preflight-how { font-family: "Inter", sans-serif; font-size: 20px; color: var(--ink); margin: 4px 0 2px 42px;
  border-left: 4px solid var(--truth); padding-left: 12px; display: none; }
.xp-preflight-item.done .xp-preflight-how { display: block; }
.xp-preflight-side { width: 420px; display: flex; flex-direction: column; align-items: center; gap: 14px; }
.xp-uncertainty-claim { font-size: 27px; color: #fff; text-align: center; margin-top: 10px; border-top: 1px solid var(--line); padding-top: 10px; }
.xp-preflight-status { font-size: 25px; font-weight: 800; letter-spacing: 0.06em; padding: 10px 18px; border-radius: 12px;
  border: 3px solid currentColor; text-align: center; width: 100%; }
` }));

// ---------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------
const median = (a) => { const b = Float64Array.from(a).sort(), m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; };
const stdev = (a) => { const m = mean(a); let q = 0; for (const v of a) q += (v - m) ** 2; return Math.sqrt(q / (a.length - 1)); };
const corr = (a, b) => {
  const ma = mean(a), mb = mean(b); let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return sab / Math.sqrt(saa * sbb);
};

// card() fills the slide width; these widgets size to their content (≤ 1760 px).
const tcard = (...c) => { const e = card(...c); Object.assign(e.style, { width: "fit-content", maxWidth: "1760px", margin: "0 auto" }); return e; };

// Draw an n x m array into a chart's plot area as a pixelated SVG <image>.
// The extent [x0, x1] x [y0, y1] is in data units; (x0, y0) is the centre-edge of pixel (0, 0).
function chartImage(c, data, nx, ny, vmin, vmax, cmap, x0, x1, y0, y1) {
  const ic = imageCanvas({ n: nx, ny });
  ic.draw(data, vmin, vmax, cmap);
  const X0 = c.x(x0), X1 = c.x(x1), Y0 = c.y(y0), Y1 = c.y(y1);
  c.gPlot.prepend(s("image", { href: ic.canvas.toDataURL(), x: Math.min(X0, X1), y: Math.min(Y0, Y1),
    width: Math.abs(X1 - X0), height: Math.abs(Y1 - Y0), preserveAspectRatio: "none", style: "image-rendering: pixelated" }));
}

// In-place 2D FFT of complex n x n arrays (row-major).
function fft2c(re, im, n, inverse = false) {
  const r = new Float64Array(n), c = new Float64Array(n);
  for (let y = 0; y < n; y++) {
    const o = y * n, rr = re.subarray(o, o + n), ii = im.subarray(o, o + n);
    fft1(rr, ii, inverse);
  }
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { r[y] = re[y * n + x]; c[y] = im[y * n + x]; }
    fft1(r, c, inverse);
    for (let y = 0; y < n; y++) { re[y * n + x] = r[y]; im[y * n + x] = c[y]; }
  }
}

// ---------------------------------------------------------------------------------------------
// 1. artefacts — failure-mode explorer
// Specimen: square lattice of atom columns, a = 2.0 Å, rotated 4°, Gaussian columns σ = 0.35 Å,
// point-sampled (STEM) on 128 x 128 probe positions, pixel 0.25 Å (aliasing case: 1.7 Å).
// Counts: λ = 30 + 220·column e⁻ per pixel, Poisson. Detector: ADU = 100 + 8·counts, 12-bit ADC (max 4095).
// Each mode adds one artefact and computes the diagnostic that reveals it from the simulated data.
// ---------------------------------------------------------------------------------------------
const NA = 128, PIX = 0.25, A_LAT = 2.0, SIG = 0.35, THETA = 4 * Math.PI / 180;
const OFFSET = 100, GAIN = 8, ADC = 4095, BG = 30, AMP = 220;

// Expected counts λ(i, j). opt: p (pixel Å), dx/dy (content shift, px), rows (per-row x shift, px),
// dopant (amplitude factor of one column), dmg (radius of a damaged disc, px), scale (dose factor).
function lattice({ p = PIX, dx = 0, dy = 0, rows, dopant = 1, dmg = 0, scale = 1 } = {}) {
  const out = new Float64Array(NA * NA), cs = Math.cos(THETA), sn = Math.sin(THETA), k2 = 1 / (2 * SIG * SIG);
  for (let j = 0; j < NA; j++) for (let i = 0; i < NA; i++) {
    const x = (i - dx - (rows ? rows[j] : 0)) * p, y = (j - dy) * p;
    const u = x * cs + y * sn, v = -x * sn + y * cs;
    const mu = Math.round(u / A_LAT), mv = Math.round(v / A_LAT), du = u - mu * A_LAT, dv = v - mv * A_LAT;
    let col = Math.exp(-(du * du + dv * dv) * k2);
    if (dopant !== 1 && mu === 8 && mv === 7) col *= dopant;       // one heavy-atom column near the centre
    let bg = BG;
    if (dmg > 0) {                                                    // amorphised disc: mass loss and lost order
      const D = 1 / (1 + Math.exp((Math.hypot(i - 64, j - 64) - dmg) / 4));
      col *= 1 - 0.8 * D; bg *= 1 - 0.3 * D;
    }
    out[j * NA + i] = scale * (bg + AMP * col);
  }
  return out;
}
// Poisson draw: inversion by sequential search for λ < 30 (one uniform per draw, faster than Knuth), else core PTRS.
function poi(r, l) {
  if (l >= 30) return r.poisson(l);
  const u = r.u();
  let k = 0, p = Math.exp(-l), F = p;
  while (u > F && k < 200) { k++; p *= l / k; F += p; }
  return k;
}
const detect = (lam, r) => lam.map((l) => Math.min(ADC, OFFSET + GAIN * poi(r, l)));

// 3 x 3 maximum filter: makes single-bin FFT spots visible on screen (display only).
function dilate(a) {
  const o = new Float64Array(NA * NA);
  for (let y = 0; y < NA; y++) for (let x = 0; x < NA; x++) {
    let m = -Infinity;
    for (let b = Math.max(0, y - 1); b <= Math.min(NA - 1, y + 1); b++) for (let c = Math.max(0, x - 1); c <= Math.min(NA - 1, x + 1); c++) m = Math.max(m, a[b * NA + c]);
    o[y * NA + x] = m;
  }
  return o;
}

// log|FFT| with a Hann window (stops edge leakage of the rotated lattice).
function fftLog(img) {
  const m = mean(img), w = (i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / NA), a = new Float64Array(NA * NA);
  for (let j = 0; j < NA; j++) for (let i = 0; i < NA; i++) a[j * NA + i] = (img[j * NA + i] - m) * w(i) * w(j);
  const F = fft2mag(a, NA);
  return { F, L: F.map((v) => Math.log1p(v)) };
}

// Spectrum of a frame (mean removed), for cross-correlation.
function spectrum(f) {
  const m = mean(f), re = f.map((v) => v - m), im = new Float64Array(NA * NA);
  fft2c(re, im, NA);
  return { re, im };
}
// Sub-pixel shift of frame B relative to frame A (spectra): peak of the cross-correlation, parabolic refinement.
function xcorrShift(A, B) {
  const n = NA, ar = new Float64Array(n * n), ai = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) {                                   // conj(A)·B
    ar[i] = A.re[i] * B.re[i] + A.im[i] * B.im[i]; ai[i] = A.re[i] * B.im[i] - A.im[i] * B.re[i];
  }
  fft2c(ar, ai, n, true);
  const at = (x, y) => ar[((y + n) % n) * n + ((x + n) % n)];
  let bx = 0, by = 0, best = -Infinity;
  for (let y = -3; y <= 3; y++) for (let x = -3; x <= 3; x++) if (at(x, y) > best) { best = at(x, y); bx = x; by = y; }
  const par = (m, c0, p) => { const d = m - 2 * c0 + p; return d < 0 ? 0.5 * (m - p) / d : 0; };
  return [bx + par(at(bx - 1, by), best, at(bx + 1, by)), by + par(at(bx, by - 1), best, at(bx, by + 1))];
}

// z-map: (pixel − lattice average) / Poisson σ. The lattice average is a Bragg filter: keep the
// Fourier components well above the noise floor. Then group pixels with z > 6 into 8-connected clusters.
function zmap(img) {
  const n = NA, re = Float64Array.from(img), im = new Float64Array(n * n);
  fft2c(re, im, n);
  const mag = re.map((v, i) => Math.hypot(v, im[i])), thr = 8 * median(mag);
  for (let i = 1; i < n * n; i++) if (mag[i] < thr) { re[i] = 0; im[i] = 0; }
  fft2c(re, im, n, true);
  const z = img.map((v, i) => (v - re[i]) / Math.sqrt(GAIN * Math.max(re[i] - OFFSET, GAIN)));
  const seen = new Uint8Array(n * n), clusters = [];
  for (let s0 = 0; s0 < n * n; s0++) {
    if (seen[s0] || z[s0] <= 6) continue;
    const stack = [s0], cl = { area: 0, zmax: 0, at: s0 };
    seen[s0] = 1;
    while (stack.length) {
      const q = stack.pop(), x = q % n, y = (q - x) / n;
      cl.area++;
      if (z[q] > cl.zmax) { cl.zmax = z[q]; cl.at = q; }
      for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
        const xx = x + a, yy = y + b, k = yy * n + xx;
        if (xx >= 0 && yy >= 0 && xx < n && yy < n && !seen[k] && z[k] > 6) { seen[k] = 1; stack.push(k); }
      }
    }
    clusters.push(cl);
  }
  return { z, clusters: clusters.sort((p, q) => q.zmax - p.zmax) };
}

// Diagnostic chart sizes: wide plots fill the box; image plots are square.
const wideChart = (o) => chart({ width: 830, height: 470, ...o });
const squareChart = (o) => chart({ width: 76 + 390 + 22, height: 470, ...o });

function histChart(img, title) {
  const nb = 64, w = 4096 / nb, cnt = new Array(nb).fill(0);
  for (const v of img) cnt[Math.min(nb - 1, Math.floor(v / w))]++;
  const c = wideChart({ x: { domain: [0, 4096], label: "pixel value (ADU)", ticks: [0, 1000, 2000, 3000, 4095] },
    y: { domain: [0.7, 3e4], type: "log", label: "pixels (log)" }, title });
  c.bars(cnt.map((v, i) => [i * w, (i + 1) * w, v || NaN]), { fill: T.data, opacity: 0.9 });
  c.ruleX(ADC, { stroke: T.warn, dash: "8 6" });
  c.text(ADC, 3e4, "ADC max", { fill: T.warn, anchor: "end", dx: -8, dy: 26, size: 19 });
  return { c, cnt };
}

const MODES = [
  { id: "dead", label: "dead & hot pixels", enters: "detector" },
  { id: "sat", label: "saturation", enters: "ADC (digitisation)" },
  { id: "drift", label: "drift", enters: "stage / specimen" },
  { id: "scan", label: "scan noise", enters: "scan coils" },
  { id: "alias", label: "aliasing", enters: "sampling grid" },
  { id: "xray", label: "X-ray hit", enters: "detector" },
  { id: "dup", label: "duplicate frames", enters: "data handling" },
  { id: "dmg", label: "beam damage", enters: "specimen" },
];

// Each builder returns { img, vmin, vmax, imLabel, marks, diag(): chart element, found, nums, check, fix, msg }.
const BUILD = {
  dead(on) {
    const r = rng(11), img = detect(lattice(), r), marks = [];
    if (on) for (let k = 0; k < 23; k++) {
      const i = Math.floor(r.u() * NA * NA), hot = k >= 14;
      img[i] = hot ? ADC : 0;
      marks.push({ x: i % NA, y: Math.floor(i / NA), color: hot ? T.warn : T.gauss });
    }
    const nDead = img.filter((v) => v === 0).length, nHot = img.filter((v) => v === ADC).length;
    return { img, vmin: OFFSET, vmax: 2300, imLabel: "one frame (ADU)", marks,
      diag() {
        const { c } = histChart(img, "histogram of pixel values");
        if (nDead) c.text(0, 3e4, `${nDead} at 0`, { fill: T.gauss, dx: 8, dy: 26, size: 20, weight: 700 });
        if (nHot) c.text(ADC, 3e4, `${nHot} at ${ADC}`, { fill: T.warn, anchor: "end", dx: -8, dy: 52, size: 20, weight: 700 });
        return c.el;
      },
      found: nDead + nHot > 0, nums: [[`${nDead} · ${nHot}`, "dead (0) · hot (4095) pixels"]],
      check: `histogram (log y) → ${nDead + nHot ? `spikes: <span class="bad">${nDead} pixels at 0, ${nHot} at ${ADC}</span>` : `<span class="good">no spikes at 0 or ${ADC}</span>`}`,
      fix: "mask them with a defect map (from a dark and a flat frame); leave masked pixels out of the loss." };
  },
  sat(on) {
    const img = detect(lattice({ scale: on ? 2.6 : 1 }), rng(12));
    const nSat = img.filter((v) => v >= ADC).length, mx = Math.max(...img), frac = 100 * nSat / img.length;
    return { img, vmin: OFFSET, vmax: ADC, imLabel: `one frame (ADU), dose ×${on ? 2.6 : 1}`, marks: [],
      diag() {
        const { c } = histChart(img, "histogram of pixel values");
        if (nSat) c.text(ADC, 3e4, `pile-up: ${nSat} px`, { fill: T.thread, anchor: "end", dx: -8, dy: 52, size: 20, weight: 700 });
        else c.text(mx, 3e4, `max ${mx} ADU`, { fill: T.truth, dx: 10, dy: 52, size: 20, weight: 700 });
        return c.el;
      },
      found: nSat > 0, nums: nSat ? [[`${fmt(frac, 1)} %`, `pixels clipped at ${ADC} (12-bit max)`]] : [[`${fmt(100 * (1 - mx / ADC), 0)} %`, "headroom below the ADC max"]],
      check: `histogram → ${nSat ? `<span class="bad">${fmt(frac, 1)} % of pixels pile up at ${ADC}</span>: the column peaks are cut off` : `<span class="good">max ${mx} ADU, ${fmt(100 * (1 - mx / ADC), 0)} % headroom</span>`}`,
      fix: "lower the beam current or the gain; clipped pixels carry no information — mask them." };
  },
  drift(on) {
    const nf = 12, vx = on ? 0.3 : 0, vy = on ? 0.17 : 0, r = rng(13), frames = [];
    for (let f = 0; f < nf; f++) frames.push(detect(lattice({ dx: vx * f, dy: vy * f, scale: 0.2 }), r));
    const sum = new Float64Array(NA * NA);
    for (const fr of frames) for (let i = 0; i < sum.length; i++) sum[i] += fr[i];
    const pos = [[0, 0]];
    const sp = frames.map(spectrum);
    for (let f = 1; f < nf; f++) { const [sx, sy] = xcorrShift(sp[f - 1], sp[f]); pos.push([pos[f - 1][0] + sx, pos[f - 1][1] + sy]); }
    const rate = Math.hypot(pos[nf - 1][0], pos[nf - 1][1]) * PIX / (nf - 1), tot = rate * (nf - 1);
    const lo = Math.min(...sum), hi = Math.max(...sum);
    return { img: sum, vmin: lo, vmax: hi, imLabel: `sum of ${nf} frames, not aligned`, marks: [],
      diag() {
        const c = wideChart({ x: { domain: [0, nf - 1], label: "frame index" }, y: { domain: [-0.3, 1.2], label: "position (Å)" },
          title: "frame-to-frame cross-correlation → position" });
        c.ruleY(0, { stroke: T.faint });
        c.line(pos.map((p, f) => [f, p[0] * PIX]), { stroke: T.thread });
        c.dots(pos.map((p, f) => [f, p[0] * PIX]), { fill: T.thread, r: 5 });
        c.line(pos.map((p, f) => [f, p[1] * PIX]), { stroke: T.pink });
        c.dots(pos.map((p, f) => [f, p[1] * PIX]), { fill: T.pink, r: 5 });
        c.note("x shift", { fill: T.thread, line: 0, weight: 700 });
        c.note("y shift", { fill: T.pink, line: 1, weight: 700 });
        return c.el;
      },
      found: rate > 0.01, nums: [[`${rate < 0.001 ? "< 0.001" : fmt(rate, 3)} Å`, "drift per frame"], [`${fmt(tot, 2)} Å`, `over ${nf} frames (lattice a = ${A_LAT} Å)`]],
      check: `cross-correlate frame k with k+1 → ${rate > 0.01 ? `<span class="bad">drift ${fmt(rate, 3)} Å/frame, ${fmt(tot, 2)} Å in total</span>: the sum smears` : `<span class="good">net drift ${rate < 0.001 ? "< 0.001" : fmt(rate, 3)} Å/frame: no drift</span>`}`,
      fix: "align the frames by cross-correlation before you sum them; let the stage settle before you record." };
  },
  scan(on) {
    const r = rng(14), rows = on ? Array.from({ length: NA }, () => 0.8 * r.normal()) : null;
    const img = detect(lattice({ rows }), r), { F, L } = fftLog(img);
    // Streak: FFT power in the column through the (1,0) spots, 4–11 bins above and below the spot (the
    // (1,±1) spots lie 16 bins away), relative to the same band 6 bins to the side (background).
    const g = NA * PIX / A_LAT, kx = Math.round(g * Math.cos(THETA)), ky = Math.round(g * Math.sin(THETA));
    const band = (off) => {
      let P = 0;
      for (const sgn of [1, -1]) for (let ddx = -1; ddx <= 1; ddx++) for (let d = 4; d <= 11; d++) for (const sd of [1, -1])
        P += F[(NA / 2 + sgn * ky + sd * d) * NA + NA / 2 + sgn * (kx + off) + ddx] ** 2;
      return P;
    };
    const ratio = band(0) / band(6), Lsort = Float64Array.from(L).sort();
    const jitter = rows ? stdev(rows) * PIX : 0;
    return { img, vmin: OFFSET, vmax: 2300, imLabel: "one frame (ADU)", marks: [],
      diag() {
        const kmax = 1 / (2 * PIX), dk = 1 / (NA * PIX), K = 1.05;
        const c = squareChart({ x: { domain: [-K, K], label: "kx (Å⁻¹)", ticks: [-1, -0.5, 0, 0.5, 1] },
          y: { domain: [K, -K], label: "ky (Å⁻¹)", ticks: [-1, -0.5, 0, 0.5, 1] }, title: "FFT (log |F|)", grid: false });
        chartImage(c, L, NA, NA, Lsort[NA * NA * 0.5 | 0], Lsort[NA * NA - 1], "magma", -kmax - dk / 2, kmax - dk / 2, -kmax - dk / 2, kmax - dk / 2);
        if (on) c.text(kx * dk, -0.75, "streak", { fill: T.warn, anchor: "middle", size: 20, weight: 700 });
        return c.el;
      },
      found: ratio > 3, nums: [[`${fmt(ratio, ratio < 10 ? 1 : 0)} ×`, "streak power / background"], [`${fmt(jitter, 2)} Å`, "row jitter put into the simulation (rms)"]],
      check: `FFT → ${ratio > 3 ? `<span class="bad">the spots smear into vertical streaks, ${fmt(ratio, ratio < 10 ? 1 : 0)} × the background</span>` : `<span class="good">sharp spots, no streaks (${fmt(ratio, 1)} × background)</span>`}`,
      fix: "add a flyback delay, sync the lines to the mains; correct row shifts by row-to-row cross-correlation." };
  },
  alias(on) {
    const p = on ? 1.7 : PIX, img = detect(lattice({ p }), rng(15)), { F, L } = fftLog(img);
    let best = -1, bi = 0;
    for (let y = 0; y < NA; y++) for (let x = 0; x < NA; x++) {
      if (Math.hypot(x - NA / 2, y - NA / 2) < 3) continue;
      if (F[y * NA + x] > best) { best = F[y * NA + x]; bi = y * NA + x; }
    }
    const dk = 1 / (NA * p), kpk = Math.hypot((bi % NA - NA / 2) * dk, (Math.floor(bi / NA) - NA / 2) * dk);
    const per = 1 / kpk, nyq = 1 / (2 * p), Lsort = Float64Array.from(L).sort(), bad = Math.abs(per - A_LAT) > 0.1;
    return { img, vmin: OFFSET, vmax: 2300, imLabel: `one frame (ADU), pixel ${fmt(p, 2)} Å`, marks: [],
      diag() {
        const K = 0.65, kmax = nyq;
        const c = squareChart({ x: { domain: [-K, K], label: "kx (Å⁻¹)", ticks: [-0.5, 0, 0.5] },
          y: { domain: [K, -K], label: "ky (Å⁻¹)", ticks: [-0.5, 0, 0.5] }, title: "FFT (log |F|)", grid: false });
        chartImage(c, dilate(L), NA, NA, Lsort[NA * NA * 0.8 | 0], Lsort[NA * NA - 1], "magma", -kmax - dk / 2, kmax - dk / 2, -kmax - dk / 2, kmax - dk / 2);
        if (kmax < K) {
          c.rect(-kmax, -kmax, kmax, kmax, { fill: "none", stroke: T.warn, "stroke-width": 3, "stroke-dasharray": "8 6" });
          c.text(-kmax, -kmax, "Nyquist", { fill: T.warn, dy: -8, size: 19, weight: 700 });
        }
        const cs = Math.cos(THETA) / A_LAT, sn = Math.sin(THETA) / A_LAT;
        c.dots([[cs, sn], [-cs, -sn], [-sn, cs], [sn, -cs]], { r: 13, fill: "none", stroke: T.truth, strokeWidth: 3, top: true });
        c.text(cs, sn, "expected", { fill: T.truth, anchor: "middle", dy: 44, size: 19, weight: 700 });
        const kp = [(bi % NA - NA / 2) * dk, (Math.floor(bi / NA) - NA / 2) * dk];
        c.dots([kp, [-kp[0], -kp[1]]], { r: 11, fill: "none", stroke: T.thread, strokeWidth: 3, top: true });
        if (bad) c.text(kp[0], kp[1], "seen", { fill: T.thread, anchor: "middle", dy: -18, size: 19, weight: 700 });
        return c.el;
      },
      found: bad, nums: [[`${fmt(per, 2)} Å`, "period of the strongest FFT spot"], [`${fmt(A_LAT, 1)} Å`, `real lattice (needs pixel ≤ ${fmt(A_LAT / 2, 1)} Å)`]],
      check: `FFT → strongest spot at ${fmt(kpk, 2)} Å⁻¹ = ${bad ? `<span class="bad">${fmt(per, 2)} Å period — but the lattice is ${A_LAT} Å</span>: the green rings lie outside the Nyquist box` : `<span class="good">${fmt(per, 2)} Å: it matches the lattice</span>`}`,
      fix: `use a pixel ≤ a/2 = ${fmt(A_LAT / 2, 1)} Å (better a/3); always compare the FFT spots with the expected k.` };
  },
  xray(on) {
    // A hit on the detector lasts one dwell time: a spike of 1–2 pixels along the fast-scan direction.
    // A probe-sized bright blob instead is a heavy (dopant) column: probe FWHM 2.355σ → area ≈ π(FWHM/2)² px.
    const img = detect(lattice({ dopant: 1.8 }), rng(16));
    if (on) for (const i of [41, 42]) img[92 * NA + i] = Math.min(ADC, img[92 * NA + i] + 2600);
    const { z, clusters } = zmap(img), probeArea = Math.PI * (1.1775 * SIG / PIX) ** 2;
    const spikes = clusters.filter((c) => c.area < probeArea / 2), blobs = clusters.filter((c) => c.area >= probeArea / 2);
    const xy = (c) => [c.at % NA, Math.floor(c.at / NA)];
    const marks = [...spikes.map((c) => ({ x: xy(c)[0], y: xy(c)[1], color: T.thread, r: 5 })),
      ...blobs.map((c) => ({ x: xy(c)[0], y: xy(c)[1], color: T.truth, r: 6 }))];
    const sp = spikes[0], bl = blobs[0];
    const desc = (c) => `z = ${fmt(c.zmax, 0)}, ${c.area} px`;
    return { img, vmin: OFFSET, vmax: 2300, imLabel: "one frame (ADU)", marks,
      diag() {
        const c = squareChart({ x: { domain: [-0.5, NA - 0.5], label: "x (px)", ticks: [0, 32, 64, 96] },
          y: { domain: [NA - 0.5, -0.5], label: "y (px)", ticks: [0, 32, 64, 96] }, title: "z = (pixel − lattice average) / σ", grid: false });
        chartImage(c, z, NA, NA, -12, 12, "div", -0.5, NA - 0.5, -0.5, NA - 0.5);
        for (const [list, col, name] of [[spikes, T.warn, "spike"], [blobs, T.truth, "blob"]]) for (const cl of list) {
          c.dots([xy(cl)], { r: 15, fill: "none", stroke: col, strokeWidth: 3, top: true });
          c.text(...xy(cl), `${name}: z = ${fmt(cl.zmax, 0)}`, { fill: col, dx: 20, dy: 8, size: 20, weight: 700 });
        }
        return c.el;
      },
      found: !!sp,
      nums: [[sp ? desc(sp) : "none", "spike sharper than the probe"], [bl ? desc(bl) : "none", `blob as wide as the probe (≈ ${fmt(probeArea, 0)} px)`]],
      msg: "X-ray hit (remove) or heavy dopant atom (keep)? Decide with physics: a spike sharper than the probe cannot come from the specimen.",
      check: `z-map → ${sp ? `<span class="bad">a ${sp.area}-px spike (z = ${fmt(sp.zmax, 0)})</span> and ` : `<span class="good">no spike</span>, only `}a probe-sized blob${bl ? ` (z = ${fmt(bl.zmax, 0)})` : ""}`,
      fix: "replace the spike by the local median (X-ray hit); keep the blob: it is a heavy dopant column." };
  },
  dup(on) {
    const nf = 12, r = rng(17), lam = lattice({ scale: 0.15 }), frames = [];
    for (let f = 0; f < nf; f++) frames.push(detect(lam, r));
    if (on) frames[9] = frames[3];
    const M = new Float64Array(nf * nf);
    for (let a = 0; a < nf; a++) for (let b = a; b < nf; b++) M[a * nf + b] = M[b * nf + a] = a === b ? 1 : corr(frames[a], frames[b]);
    const off = []; let pair = null;
    for (let a = 0; a < nf; a++) for (let b = a + 1; b < nf; b++) { off.push(M[a * nf + b]); if (M[a * nf + b] > 0.999) pair = [a, b]; }
    const others = off.filter((v) => v < 0.999), lo = Math.min(...others), typ = mean(others);
    return { img: frames[9], vmin: OFFSET, vmax: 700, imLabel: `frame 9 of ${nf} (ADU)`, marks: [],
      diag() {
        const c = squareChart({ x: { domain: [-0.5, nf - 0.5], label: "frame", ticks: [0, 3, 6, 9] },
          y: { domain: [nf - 0.5, -0.5], label: "frame", ticks: [0, 3, 6, 9] }, title: "frame-similarity matrix ρ", grid: false });
        chartImage(c, M, nf, nf, lo - 0.01, 1, "magma", -0.5, nf - 0.5, -0.5, nf - 0.5);
        if (pair) {
          c.dots([[pair[0], pair[1]], [pair[1], pair[0]]], { r: 23, fill: "none", stroke: T.thread, strokeWidth: 4, top: true });
          c.text(pair[1], pair[0], "ρ = 1.000", { fill: "#fff", anchor: "middle", dy: 52, size: 20, weight: 700 });
        }
        return c.el;
      },
      found: !!pair, nums: [[pair ? `${pair[0]} = ${pair[1]}` : "none", "identical frame pair"], [`ρ ≈ ${fmt(typ, 2)}`, "between independent frames"]],
      check: `similarity matrix → ${pair ? `<span class="bad">frames ${pair[0]} and ${pair[1]} have ρ = 1.000</span> (others ${fmt(typ, 2)}): one frame saved twice` : `<span class="good">max off-diagonal ρ = ${fmt(Math.max(...off), 3)}: no duplicates</span>`}`,
      fix: "drop exact duplicates; split train and test by session, so near-copies cannot leak into the test set." };
  },
  dmg(on) {
    const nf = 20, r = rng(20), tot = [];
    let last;
    for (let f = 0; f < nf; f++) {
      const fr = detect(lattice({ dmg: on ? 55 * f / (nf - 1) : 0, scale: 0.25 }), r);
      tot.push(fr.reduce((a, v) => a + (v - OFFSET) / GAIN, 0));
      last = fr;
    }
    // Straight-line fit; the slope error uses the Poisson variance (= counts) of each total.
    const fbar = (nf - 1) / 2, ybar = mean(tot);
    let sxy = 0, sxx = 0;
    tot.forEach((v, f) => { sxy += (f - fbar) * (v - ybar); sxx += (f - fbar) ** 2; });
    const slope = sxy / sxx, a0 = ybar - slope * fbar, first = tot[0];
    const drop = Math.round(-1000 * slope * (nf - 1) / a0) / 10 || 0, sDrop = 100 * Math.sqrt(ybar / sxx) * (nf - 1) / a0, sig = drop > 3 * sDrop;
    return { img: last, vmin: OFFSET, vmax: 700, imLabel: `frame ${nf} of ${nf} (ADU)`, marks: [],
      diag() {
        const c = wideChart({ x: { domain: [0, nf - 1], label: "frame index" }, y: { domain: [0, 1.1 * first / 1e3], label: "total counts per frame (×10³ e⁻)" },
          title: "total counts per frame" });
        c.line(tot.map((v, f) => [f, v / 1e3]), { stroke: T.pois });
        c.dots(tot.map((v, f) => [f, v / 1e3]), { fill: T.pois, r: 5 });
        c.line([[0, a0 / 1e3], [nf - 1, (a0 + slope * (nf - 1)) / 1e3]], { stroke: T.muted, width: 2, dash: "6 6" });
        c.note(`fit: ${drop > 0 ? "−" : drop < 0 ? "+" : ""}${fmt(Math.abs(drop), 1)} ± ${fmt(sDrop, 2)} % over ${nf} frames`, { corner: "br", fill: sig ? T.thread : T.truth, weight: 700 });
        return c.el;
      },
      found: sig, nums: [[`${fmt(drop, 1)} ± ${fmt(sDrop, 2)} %`, `drop of the counts (fit, Poisson error)`]],
      check: `total counts per frame → ${sig ? `<span class="bad">falls by ${fmt(drop, 1)} % over ${nf} frames</span>: the beam changes the specimen` : `<span class="good">flat within Poisson noise (${fmt(drop, 1)} ± ${fmt(sDrop, 2)} %)</span>`}`,
      fix: "lower the dose; use only the early frames, or weight the frames by dose (dose-fractionation)." };
  },
};

register("artefacts", (el) => {
  let mode = "dead", on = true;
  const cache = new Map();
  const imc = imageCanvas({ n: NA, size: 420 });
  const overlay = s("svg", { width: 420, height: 420, viewBox: `0 0 ${NA} ${NA}` });
  const imWrap = h("div", { class: "xp-artefacts-img" }, imc.canvas, overlay);
  const imLabel = h("div", { class: "xp-imlabel" });
  const diagBox = h("div", { class: "xp-artefacts-diag" });
  const verdict = h("div", { class: "xp-artefacts-verdict" });
  const roCheck = readout(), roFix = readout();
  const chips = MODES.map((m) => {
    const c = h("span", { class: "xp-chip", onclick: () => { mode = m.id; update(); } }, m.label);
    c.dataset.id = m.id;
    return c;
  });
  const tg = toggle({ label: "artefact", options: [{ value: "on", label: "on", color: T.thread }, { value: "off", label: "off", color: T.truth }],
    value: "on", onChange: (v) => { on = v === "on"; update(); } });

  function update() {
    const t0 = performance.now(), key = `${mode}:${on}`;
    if (!cache.has(key)) cache.set(key, BUILD[mode](on));
    const R = cache.get(key), M = MODES.find((m) => m.id === mode);
    chips.forEach((c) => c.classList.toggle("on", c.dataset.id === mode));
    imc.draw(R.img, R.vmin, R.vmax, "gray");
    imLabel.textContent = R.imLabel;
    overlay.replaceChildren(...R.marks.map((m) => s("circle", { cx: m.x + 0.5, cy: m.y + 0.5, r: m.r || 3.5, fill: "none", stroke: m.color, "stroke-width": 0.9 })));
    diagBox.replaceChildren(R.diag());
    const st = R.found ? ["artefact found", T.thread] : ["looks normal", T.truth];
    verdict.replaceChildren(
      h("div", { class: "xp-artefacts-status", style: { color: st[1] } }, (R.found ? "⚠ " : "✓ ") + st[0]),
      ...R.nums.map(([n, l]) => h("div", {}, h("div", { class: "xp-artefacts-num" }, n), h("div", { class: "xp-artefacts-lbl" }, l))),
      h("div", { class: "xp-artefacts-enter", html: `enters at: <b>${M.enters}</b>` }),
      ...(R.msg ? [h("div", { class: "xp-artefacts-msg" }, R.msg)] : []));
    roCheck.set(`<b>Check:</b> ${R.check}`);
    roFix.set(`<b>Fix:</b> ${R.fix}`);
    el.dataset.ms = (performance.now() - t0).toFixed(0);
  }

  el.append(tcard(
    controls(h("div", { class: "xp-artefacts-chips" }, chips), tg),
    row(col(imLabel, imWrap), diagBox, verdict),
    h("div", { class: "xp-artefacts-ro" }, roCheck, roFix)));
  update();
  // Fill the cache in the background (one state per tick, ≤ ~150 ms each), so that clicks are instant.
  const todo = MODES.flatMap((m) => [`${m.id}:true`, `${m.id}:false`]);
  const prefetch = () => {
    const key = todo.find((k) => !cache.has(k));
    if (!key) return;
    const [id, st] = key.split(":");
    cache.set(key, BUILD[id](st === "true"));
    setTimeout(prefetch, 60);
  };
  setTimeout(prefetch, 300);
});

// ---------------------------------------------------------------------------------------------
// 2. metadata — strip the metadata and see which analyses survive
// Numbers: dose = I·t / (e·p²) = 25 pA · 10 µs / (1.602·10⁻¹⁹ C · (0.12 Å)²) ≈ 1.08·10⁵ e⁻/Å²;
// Nyquist: smallest trusted period = 2p; electron wavelength at 300 kV (relativistic) = 1.97 pm.
// ---------------------------------------------------------------------------------------------
const MD = { ht: 300e3, cur: 25e-12, dwell: 10e-6, px: 0.12, gain: 1.0 };
const FIELDS = [
  { id: "ht", k: "instrument", v: "STEM-300, Cs-corrected · HT 300 kV" },
  { id: "mode", k: "mode", v: "STEM HAADF · counting" },
  { id: "cur", k: "probe_current", v: "25 pA" },
  { id: "dwell", k: "dwell_time", v: "10 µs" },
  { id: "px", k: "pixel_size", v: "0.12 Å" },
  { id: "ang", k: "camera_length", v: "115 mm · HAADF 60–200 mrad" },
  { id: "gain", k: "detector", v: "counting · gain 1.0 count/e⁻" },
  { id: "ref", k: "dark/gain_ref", v: "dark_0412.mrc · gain_0412.mrc" },
  { id: "spec", k: "specimen_id", v: "STO-LAO-07" },
  { id: "sess", k: "session_id", v: "2026-10-28_s3" },
  { id: "op", k: "operator", v: "a.student" },
  { id: "time", k: "timestamp", v: "2026-10-28T14:32:07Z" },
];
const lambdaPm = (V) => { const h0 = 6.62607e-34, m = 9.10938e-31, e = 1.602177e-19, c = 2.99792e8;
  return 1e12 * h0 / Math.sqrt(2 * m * e * V * (1 + e * V / (2 * m * c * c))); };
const TASKS = [
  { name: "convert ADU → electron counts (Poisson NLL)", needs: ["gain", "ref"], res: () => `counts = (ADU − dark) ÷ ${fmt(MD.gain, 1)}` },
  { name: "dose in e⁻/Å²", needs: ["cur", "dwell", "px"],
    res: () => { const d = MD.cur * MD.dwell / 1.602177e-19 / MD.px ** 2, e = Math.floor(Math.log10(d)); return `dose = ${fmt(d / 10 ** e, 2)} × 10${sup(e)} e⁻/Å²`; } },
  { name: "Nyquist check: lattice or alias?", needs: ["px"], res: () => `trust periods ≥ 2p = ${fmt(2 * MD.px, 2)} Å` },
  { name: "compare with simulation", needs: ["ht", "mode", "ang"], res: () => `λ = ${fmt(lambdaPm(MD.ht), 2)} pm, HAADF 60–200 mrad` },
  { name: "honest train/test split", needs: ["spec", "sess"], res: () => "split by specimen and session" },
  { name: "reproduce / FAIR deposit", needs: FIELDS.map((f) => f.id), res: () => "complete record" },
];

register("metadata", (el) => {
  const gone = new Set(["px"]);
  let tiff = false;
  const head = h("div", { class: "xp-metadata-head" });
  const rows = FIELDS.map((f) => {
    const r = h("div", { class: "xp-metadata-field", onclick: () => { gone.has(f.id) ? gone.delete(f.id) : gone.add(f.id); tiff = false; update(); },
      onmouseenter: () => hover(f.id, true), onmouseleave: () => hover(f.id, false) },
    h("span", { class: "k" }, f.k + ":"), h("span", { class: "v" }, f.v));
    r.dataset.id = f.id;
    return r;
  });
  const taskEls = TASKS.map(() => h("div", { class: "xp-metadata-task" }));
  const ro = readout("", "big");
  const hover = (id, onOff) => TASKS.forEach((t, i) => taskEls[i].classList.toggle("hl", onOff && t.needs.includes(id) && t.needs.length < FIELDS.length));

  function update() {
    head.innerHTML = tiff
      ? `▣ <b>scan_0042.tif</b> · uint16 [1024 × 1024]<br>metadata: <span style="color:${T.thread}">none (TIFF export)</span>`
      : `▣ <b>scan_0042.emd</b> · HDF5 · uint16 [1024 × 1024]<br>/metadata/acquisition:`;
    rows.forEach((r) => r.classList.toggle("off", gone.has(r.dataset.id)));
    let ok = 0;
    TASKS.forEach((t, i) => {
      const miss = t.needs.filter((id) => gone.has(id)), pass = miss.length === 0;
      ok += pass;
      const all = t.needs.length === FIELDS.length;
      const needTxt = all ? (pass ? "needs: every field" : `needs: every field — <span style="color:${T.thread};font-weight:700">${miss.length} missing</span>`)
        : `needs: ${t.needs.map((id) => { const k = FIELDS.find((f) => f.id === id).k; return gone.has(id) ? `<span class="miss">${k}</span>` : k; }).join(", ")}`;
      taskEls[i].replaceChildren(
        h("div", { class: "icon", style: { color: pass ? T.truth : T.thread } }, pass ? "✓" : "✗"),
        h("div", {}, h("div", { class: "name" }, t.name),
          h("div", { class: "sub", html: pass ? `<span style="color:${T.truth}">${t.res()}</span> · ${needTxt}` : needTxt })));
    });
    const cls = ok === TASKS.length ? "good" : ok >= 3 ? "warn" : "bad";
    const lost = [...gone].map((id) => FIELDS.find((f) => f.id === id).k);
    ro.set(`<span class="${cls}">${ok} of ${TASKS.length}</span> analyses still possible` +
      (lost.length ? ` <span style="color:${T.muted};font-weight:400;font-size:22px">· deleted: ${tiff ? "all metadata" : lost.join(", ")}</span>`
        : ` <span style="color:${T.muted};font-weight:400;font-size:22px">· click a field to delete it</span>`));
  }

  el.append(tcard(
    row(
      col(h("div", { class: "xp-metadata-file" }, head, rows),
        controls(button("export as TIFF", () => { FIELDS.forEach((f) => gone.add(f.id)); tiff = true; update(); }, "primary"),
          button("restore all", () => { gone.clear(); tiff = false; update(); }),
          h("span", { class: "xp-caption" }, "click a field: delete / restore"))),
      h("div", { class: "xp-metadata-tasks" }, h("div", { class: "xp-slider-label" }, "downstream analyses"), taskEls)),
    ro));
  update();
});

// ---------------------------------------------------------------------------------------------
// 3. preflight — the six checks before training, as an aviation checklist
// ---------------------------------------------------------------------------------------------
const CHECKS = [
  { t: "UNITS & COUNTS", what: "offset subtracted, ÷ gain", how: "Average a dark frame → offset. Subtract it, divide by the gain (ADU per e⁻). Pixels are now electrons." },
  { t: "NOISE MODEL", what: "variance–mean plot → loss chosen", how: "Plot variance vs mean of a flat series. Slope 1 (in e⁻) → Poisson NLL; flat variance → MSE." },
  { t: "SAMPLING", what: "pixel vs smallest feature, FFT checked", how: "Pixel ≤ half the smallest period. FFT spots at the expected k, inside the Nyquist box." },
  { t: "DEFECTS", what: "dead/hot/saturated masked, drift checked", how: "Histogram (log y): spikes at 0 or the ADC max → mask. Cross-correlate frames → drift." },
  { t: "DUPLICATES & GROUPS", what: "specimen/session IDs for honest splits", how: "Frame-similarity matrix: off-diagonal ρ = 1 is a duplicate. Split by specimen and session." },
  { t: "METADATA & PROVENANCE", what: "native file, script, FAIR deposit", how: "Keep the native .emd/.dm4. Process with a script, not clicks. Plan the deposit (Zenodo, EMPIAR)." },
];

register("preflight", (el) => {
  const done = new Set([0, 1]);
  const items = CHECKS.map((c, i) => {
    const it = h("div", { class: "xp-preflight-item", onclick: () => { done.has(i) ? done.delete(i) : done.add(i); update(); } },
      h("div", { class: "xp-preflight-line" }, h("span", { class: "n" }, `${i + 1}.`), h("span", { class: "t" }, c.t),
        h("span", { class: "what" }, c.what), h("span", { class: "dots" }), h("span", { class: "resp" })),
      h("div", { class: "xp-preflight-how" }, h("b", {}, "how (5 min): "), c.how));
    return it;
  });
  const R = 150, C = 2 * Math.PI * R;
  const arc = s("circle", { cx: 180, cy: 180, r: R, fill: "none", "stroke-width": 26, "stroke-linecap": "round",
    transform: "rotate(-90 180 180)", "stroke-dasharray": `0 ${C}`, style: "transition: stroke-dasharray 0.4s, stroke 0.4s" });
  const big = s("text", { x: 180, y: 192, "text-anchor": "middle", fill: "#fff", "font-size": 92, "font-weight": 800 });
  const ring = s("svg", { width: 360, height: 360, viewBox: "0 0 360 360" },
    s("circle", { cx: 180, cy: 180, r: R, fill: "none", stroke: "rgba(148,163,184,0.18)", "stroke-width": 26 }), arc, big,
    s("text", { x: 180, y: 240, "text-anchor": "middle", fill: T.muted, "font-size": 24 }, "checks done"));
  const status = h("div", { class: "xp-preflight-status" });
  const ro = readout();

  function update() {
    items.forEach((it, i) => {
      it.classList.toggle("done", done.has(i));
      it.querySelector(".resp").textContent = done.has(i) ? "✓ CHECKED" : "— OPEN";
    });
    const k = done.size, all = k === CHECKS.length, col0 = all ? T.truth : T.thread;
    arc.setAttribute("stroke", col0);
    arc.setAttribute("stroke-dasharray", `${C * k / CHECKS.length} ${C}`);
    big.textContent = `${k}/${CHECKS.length}`;
    status.style.color = all ? T.truth : T.warn;
    status.textContent = all ? "✓ CLEARED FOR TRAINING" : "READY TO TRAIN? NOT YET";
    const open = CHECKS.filter((c, i) => !done.has(i)).map((c) => c.t.toLowerCase());
    ro.set(all ? `<span class="good">All ${CHECKS.length} checks done</span>: the data can go into the model.`
      : `<span class="warn">${open.length} open</span>: ${open.join(" · ")}`);
  }

  el.append(tcard(row(
    h("div", { class: "xp-preflight-list" },
      h("div", { class: "xp-preflight-bar" }, h("span", {}, "PRE-FLIGHT · BEFORE TRAINING"), h("span", {}, "CHALLENGE — RESPONSE")), items),
    h("div", { class: "xp-preflight-side" }, ring, status, button("reset", () => { done.clear(); update(); })))
  , ro));
  update();
});

// ---------------------------------------------------------------------------------------------
// 4. uncertainty — aleatoric vs epistemic
// Left: counts in one pixel, Poisson λ = 25, from N frames. The spread stays √λ = 5 for any N.
// Right: gain g = 4 ADU/e⁻ estimated as ĝ = variance / mean of N flat calibration frames (8 x 8 patch,
// λc = 100 e⁻, offset subtracted). Theory: std(ĝ)/g ≈ √(2 / (64N − 1)). Dots: 300 simulated
// calibrations per N, computed in chunks in the background.
// ---------------------------------------------------------------------------------------------
const LAM = 25, G_TRUE = 4, LAM_C = 100, PATCH = 64, REPS = 300;
const NGRID = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000];
const gainErrTheory = (N) => 100 * Math.sqrt(2 / (PATCH * N - 1));

register("uncertainty", (el, api) => {
  let seed = 101, frames = [];
  const draw = () => { const r = rng(seed); frames = Array.from({ length: 1000 }, () => r.poisson(LAM)); };
  draw();
  const sim = new Map();                                              // N → relative gain error (%)
  let simBusy = false;

  const cL = chart({ width: 830, height: 470, x: { domain: [0, 52], label: "counts in one pixel" },
    y: { domain: [0, 0.2], label: "fraction of frames" }, title: "Aleatoric: shot noise in one pixel" });
  const cR = chart({ width: 830, height: 470, x: { domain: [1, 1000], type: "log", label: "number of frames N" },
    y: { domain: [0.3, 40], type: "log", label: "relative error (%)", ticks: [0.5, 1, 2, 5, 10, 20], tickFormat: (v) => `${v}%` }, 
    title: "Epistemic: gain error vs N", margin: { l: 96 } });
  const roL = readout(), roR = readout();
  const sl = slider({ label: "frames N", min: 1, max: 1000, value: 10, log: true, step: 1, digits: 0, onInput: () => update() });

  function runSim() {                                                 // background: one N per chunk slice
    if (simBusy) return;
    simBusy = true;
    const r = rng(7), todo = NGRID.slice();
    let N = todo.shift(), rep = 0, est = [];
    const step = () => {
      const t0 = performance.now();
      while (N !== undefined && performance.now() - t0 < 25) {
        let s1 = 0, s2 = 0; const M = PATCH * N;
        for (let i = 0; i < M; i++) { const x = G_TRUE * r.poisson(LAM_C); s1 += x; s2 += x * x; }
        const m = s1 / M, v = (s2 - M * m * m) / (M - 1);
        est.push(v / m);
        if (++rep === REPS) { sim.set(N, 100 * stdev(est) / G_TRUE); N = todo.shift(); rep = 0; est = []; drawR(); }
      }
      if (N !== undefined) setTimeout(step, 0); else { simBusy = false; drawR(); }
    };
    setTimeout(step, 0);
  }

  function drawR() {
    const N = sl.value;
    cR.clear();
    const xs = logspace(1, 1000, 80);
    cR.line(xs.map((n) => [n, 100 / Math.sqrt(LAM)]), { stroke: T.pois, dash: "10 7" });
    cR.text(1.2, 100 / Math.sqrt(LAM), "shot noise σ/λ = 1/√λ: stays", { fill: T.pois, dy: -12, size: 20, weight: 700 });
    cR.line(xs.map((n) => [n, gainErrTheory(n)]), { stroke: T.gauss, width: 2.5, dash: "4 6" });
    cR.dots([...sim].map(([n, e]) => [n, e]), { fill: T.gauss, r: 7 });
    cR.text(30, gainErrTheory(30), "gain error ∝ 1/√N: falls", { fill: T.gauss, dx: 14, dy: -18, size: 20, weight: 700 });
    cR.ruleX(N, { stroke: T.thread, dash: "5 5" });
    cR.dots([[N, gainErrTheory(N)]], { fill: T.thread, r: 10, stroke: "#fff", top: true });
    if (sim.size < NGRID.length) cR.note(`simulating… ${sim.size}/${NGRID.length}`, { corner: "bl", fill: T.muted, size: 18 });
    else cR.note(`dots: ${REPS} simulated calibrations per N; dashed: √(2/${PATCH}N)`, { corner: "bl", fill: T.muted, size: 18 });
  }

  function update() {
    const t0 = performance.now(), N = sl.value, d = frames.slice(0, N);
    const cnt = new Map(); for (const k of d) cnt.set(k, (cnt.get(k) || 0) + 1);
    const maxF = Math.max(...cnt.values()) / N, ymax = Math.max(0.1, Math.ceil(maxF * 11) / 10);
    cL.setDomain(null, [0, ymax]);
    cL.clear();
    cL.bars([...cnt].map(([k, c]) => [k - 0.42, k + 0.42, c / N]), { fill: T.data, opacity: 0.85 });
    const ks = Array.from({ length: 53 }, (_, k) => k);
    cL.line(ks.map((k) => [k, poissonPmf(k, LAM)]), { stroke: T.pois, width: 3 });
    const m = mean(d), sp = N > 1 ? stdev(d) : NaN;
    if (N > 1) {
      const yb = ymax * 0.86;
      cL.line([[m - sp, yb], [m + sp, yb]], { stroke: T.thread, width: 4, top: true });
      cL.line([[m - sp, yb - ymax * 0.03], [m - sp, yb + ymax * 0.03]], { stroke: T.thread, width: 4, top: true });
      cL.line([[m + sp, yb - ymax * 0.03], [m + sp, yb + ymax * 0.03]], { stroke: T.thread, width: 4, top: true });
      cL.text(m - sp, yb, `spread ±${fmt(sp, 1)}`, { fill: T.thread, anchor: "end", dx: -12, dy: 8, size: 22, weight: 700 });
    }
    cL.note(`N = ${N} frame${N > 1 ? "s" : ""}`, { corner: "tr", fill: "#fff", weight: 700 });
    cL.note("Poisson pmf, λ = 25", { corner: "tr", line: 1, fill: T.pois });
    roL.set(N > 1 ? `spread = <span class="bad">${fmt(sp, 1)} counts</span> → stays √λ = ${fmt(Math.sqrt(LAM), 1)} for any N <span class="warn">(aleatoric)</span>`
      : `one frame: no spread yet. Add frames → spread → √λ = ${fmt(Math.sqrt(LAM), 1)}`);
    roR.set(`gain error <span class="bad">${fmt(gainErrTheory(N), N >= 100 ? 2 : 1)} %</span> → falls as 1/√N <span class="good">(epistemic)</span>`);
    drawR();
    el.dataset.ms = (performance.now() - t0).toFixed(0);
  }

  el.append(tcard(
    controls(sl, button("+10 frames", () => sl.set(Math.min(1000, sl.value + 10))),
      button("new frames", () => { seed++; draw(); update(); }),
      legend([[T.data, "measured counts"], [T.pois, "Poisson / shot noise"], [T.gauss, "gain estimate"], [T.thread, "current N"]])),
    row(col(cL.el, roL), col(cR.el, roR)),
    h("div", { class: "xp-uncertainty-claim" }, "More data shrinks what you don't know, not what nature randomises.")));
  update();
  api.onShow(runSim);
  runSim();
});
