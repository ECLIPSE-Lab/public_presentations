// Counting group: electrons arrive one at a time (electron-rain), Poisson noise in an image
// (dose-image), the mixed Poisson-Gaussian detector (mixed-detector) and the variance-mean plot (ptc).
import {
  register, T, rng, poissonPmf, logspace, mean, clamp, fmt, h, s, row, col, card, controls,
  readout, slider, toggle, button, legend, chart, imageCanvas,
} from "./core.js";

// Widget-specific styles, inserted once.
if (!document.getElementById("xp-noise-style")) {
  document.head.append(h("style", { id: "xp-noise-style" }, `
.xp-noise-card { max-width: 1760px; margin: 0 auto; }
.xp-noise-rain { display: block; border-radius: 12px; border: 1px solid rgba(255,255,255,0.09); background: rgba(5,8,15,0.55); }
.xp-noise-stack { position: relative; }
.xp-noise-stack svg { position: absolute; left: 0; top: 0; pointer-events: none; overflow: visible; }
.xp-noise-narrow .xp-slider input[type=range] { width: 230px; }
.xp-noise-side { width: 500px; display: flex; flex-direction: column; gap: 14px; padding-top: 10px; }
.xp-noise-side .xp-readout { margin-top: 0; line-height: 1.45; }
`));
}

// Standard normal CDF (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7).
function Phi(z) {
  if (z === Infinity) return 1;
  if (z === -Infinity) return 0;
  const x = Math.abs(z) / Math.SQRT2, t = 1 / (1 + 0.3275911 * x);
  const e = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-x * x);
  return z >= 0 ? 0.5 * (1 + e) : 0.5 * (1 - e);
}
const stats = (a, idx) => {                         // mean and sample variance of a[idx]
  let s1 = 0, s2 = 0;
  for (const i of idx) { s1 += a[i]; s2 += a[i] * a[i]; }
  const m = s1 / idx.length;
  return { m, v: (s2 - idx.length * m * m) / (idx.length - 1) };
};
const big = (x) => fmt(x, x < 1 ? 2 : x < 10 ? 1 : 0);
const pmErr = (x, e) => { const d = clamp(1 - Math.floor(Math.log10(e || 1)), 0, 3); return `${fmt(x, d)} ± ${fmt(e, d)}`; };
const ncard = (...c) => { const e = card(...c); e.classList.add("xp-noise-card"); return e; };

// ---------------------------------------------------------------------------------------------
// electron-rain. Each detector pixel sees a Poisson process: electrons arrive independently, at a
// constant rate λ/T_f, so the waiting times are exponential with mean T_f/λ. The number of hits in
// one frame (length T_f = 1.2 s at normal speed) is then Poisson(λ). Every frame adds 10 counts
// (10 pixels) to the histogram, which converges to the Poisson pmf: variance = mean = λ, SNR = √λ.
// ---------------------------------------------------------------------------------------------
register("electron-rain", (el, api) => {
  const NP = 10, W = 860, H = 560, RES = 2, TF = 1.2, FALL = 0.5;
  const X0 = 40, X1 = 816, PW = (X1 - X0) / NP, CX = (X0 + X1) / 2, SRC = 58, DY = 300, DH = 52, BASE = 540, COLH = 120;
  const r = rng(7);
  let lam = 3, speed = 1, playing = true, simT = 0, frameStart = 0, frameNo = 1, fly = [];
  const cnt = new Int32Array(NP), last = new Int32Array(NP), flash = new Float64Array(NP), next = new Float64Array(NP);
  const hist = new Float64Array(512);
  let nRec = 0, s1 = 0, s2 = 0;

  const cv = h("canvas", { class: "xp-noise-rain", width: W * RES, height: H * RES });
  cv.style.width = `${W}px`; cv.style.height = `${H}px`;
  const ctx = cv.getContext("2d");
  const sprite = h("canvas", { width: 48, height: 48 });           // glowing electron
  { const g = sprite.getContext("2d"), rg = g.createRadialGradient(24, 24, 0, 24, 24, 24);
    rg.addColorStop(0, "rgba(255,247,237,1)"); rg.addColorStop(0.18, "rgba(253,186,116,1)");
    rg.addColorStop(0.45, "rgba(251,146,60,0.45)"); rg.addColorStop(1, "rgba(251,146,60,0)");
    g.fillStyle = rg; g.fillRect(0, 0, 48, 48); }

  const expo = () => -Math.log(1 - r.u()) * TF / lam;              // waiting time to the next electron
  function restart() {
    simT = 0; frameStart = FALL; frameNo = 1; fly = [];
    cnt.fill(0); last.fill(-1); flash.fill(0);
    for (let p = 0; p < NP; p++) next[p] = FALL + expo();
  }
  function clearStats() { hist.fill(0); nRec = s1 = s2 = 0; }
  function record(vals) { for (const k of vals) { hist[Math.min(k, 511)]++; nRec++; s1 += k; s2 += k * k; } }
  function land(tMax) {                                             // electrons that reach the detector by tMax
    let j = 0;
    for (const e of fly) { if (e.t <= tMax) { cnt[e.p]++; flash[e.p] = 1; } else fly[j++] = e; }
    fly.length = j;
  }
  function endFrame() {
    record(cnt); last.set(cnt); cnt.fill(0); frameNo++; frameStart += TF;
    drawHist(); say();
  }
  function step(dt) {
    simT += dt;
    for (let p = 0; p < NP; p++) while (next[p] - FALL <= simT) {
      fly.push({ p, t: next[p], x: X0 + (p + 0.12 + 0.76 * r.u()) * PW, x0: CX + (r.u() - 0.5) * 36 });
      next[p] += expo();
    }
    while (simT >= frameStart + TF) { land(frameStart + TF); endFrame(); }
    land(simT);
  }

  function draw() {
    ctx.setTransform(RES, 0, 0, RES, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const g = ctx.createLinearGradient(0, SRC, 0, DY);               // beam cone
    g.addColorStop(0, "rgba(203,213,225,0.20)"); g.addColorStop(1, "rgba(203,213,225,0.035)");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.moveTo(CX - 20, SRC); ctx.lineTo(CX + 20, SRC); ctx.lineTo(X1, DY); ctx.lineTo(X0, DY); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#334155"; ctx.fillRect(CX - 44, SRC - 22, 88, 22);
    ctx.font = "20px Inter, sans-serif"; ctx.textBaseline = "alphabetic";
    ctx.fillStyle = T.muted; ctx.textAlign = "left"; ctx.fillText("electron beam", X0, 30);
    ctx.textAlign = "right"; ctx.fillStyle = T.ink;
    const tf = clamp(simT - frameStart, 0, TF);
    ctx.fillText(`frame ${frameNo}   ${fmt(tf, 1)} s / ${fmt(TF, 1)} s`, X1, 30);
    ctx.fillStyle = "rgba(148,163,184,0.25)"; ctx.fillRect(X1 - 230, 40, 230, 7);
    ctx.fillStyle = T.thread; ctx.fillRect(X1 - 230, 40, 230 * tf / TF, 7);
    for (const e of fly) {                                           // electrons in flight
      const f = 1 - (e.t - simT) / FALL;
      if (f < 0) continue;
      ctx.drawImage(sprite, e.x0 + (e.x - e.x0) * f - 13, SRC + f * (DY - SRC) - 13, 26, 26);
    }
    ctx.textAlign = "center";
    for (let p = 0; p < NP; p++) {                                   // detector pixels with counters
      const x = X0 + p * PW + 3;
      ctx.fillStyle = "rgba(30,41,59,0.95)"; ctx.fillRect(x, DY, PW - 6, DH);
      ctx.fillStyle = `rgba(251,146,60,${0.75 * flash[p]})`; ctx.fillRect(x, DY, PW - 6, DH);
      ctx.strokeStyle = "rgba(148,163,184,0.55)"; ctx.lineWidth = 1.5; ctx.strokeRect(x, DY, PW - 6, DH);
      ctx.fillStyle = "#fff"; ctx.font = "600 30px Inter, sans-serif"; ctx.fillText(cnt[p], x + (PW - 6) / 2, DY + 37);
    }
    ctx.font = "20px Inter, sans-serif"; ctx.fillStyle = T.muted; ctx.textAlign = "left";
    ctx.fillText("detector: 10 pixels, each counts its hits", X0, DY + DH + 26);
    ctx.fillText("counts in this frame (dashed: last frame)", X0, BASE - COLH - 16);
    const cmax = Math.max(4, lam + 3.5 * Math.sqrt(lam)), yv = (v) => BASE - Math.min(v / cmax, 1.05) * COLH;
    for (let p = 0; p < NP; p++) {                                   // columns: counts in this frame
      const x = X0 + p * PW + 16, w = PW - 32;
      ctx.fillStyle = T.pois; ctx.fillRect(x, yv(cnt[p]), w, BASE - yv(cnt[p]));
      if (last[p] >= 0) {
        ctx.setLineDash([5, 4]); ctx.strokeStyle = T.data; ctx.lineWidth = 2;
        ctx.strokeRect(x, yv(last[p]), w, BASE - yv(last[p])); ctx.setLineDash([]);
      }
    }
    ctx.strokeStyle = T.axis; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X0, BASE); ctx.lineTo(X1, BASE); ctx.stroke();
    ctx.setLineDash([8, 6]); ctx.strokeStyle = T.truth; ctx.beginPath(); ctx.moveTo(X0, yv(lam)); ctx.lineTo(X1, yv(lam)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = T.truth; ctx.textAlign = "left"; ctx.fillText("λ", X1 + 8, yv(lam) + 7);
  }

  const c = chart({ width: 820, height: 486, x: { domain: [-0.7, 10], label: "electrons per pixel per frame, k" },
    y: { domain: [0, 0.3], label: "fraction of pixel-frames" }, title: "histogram of counts" });
  function drawHist() {
    const kmax = Math.ceil(lam + 4.5 * Math.sqrt(lam) + 2);
    const pm = Array.from({ length: kmax + 1 }, (_, k) => poissonPmf(k, lam));
    const fr = pm.map((_, k) => (nRec ? hist[k] / nRec : 0));
    c.setDomain([-0.7, kmax + 0.7], [0, 1.18 * Math.max(...pm, ...fr)]);
    c.clear();
    if (nRec) c.bars(fr.map((v, k) => [k - 0.42, k + 0.42, v]), { fill: T.data, opacity: 0.5 });
    const px = c.x(1) - c.x(0);
    pm.forEach((v, k) => c.line([[k, 0], [k, v]], { stroke: T.pois, width: clamp(px * 0.2, 2, 4) }));
    c.dots(pm.map((v, k) => [k, v]), { r: clamp(px * 0.25, 3, 7), fill: T.pois, strokeWidth: 0 });
    if (last[0] >= 0) {                                              // the 10 counts of the last frame
      const seen = {};
      for (const k of last) {
        if (k > kmax) continue;
        const j = (seen[k] = (seen[k] || 0) + 1) - 1;
        c.gTop.append(s("circle", { cx: c.x(k), cy: c.y(0) - 9 - 15 * j, r: 6, fill: T.thread, stroke: T.bg, "stroke-width": 1.5 }));
      }
    }
    c.note(`Poisson pmf, λ = ${fmt(lam, 1)}`, { corner: "tr", fill: T.pois });
    c.note(`measured: ${nRec.toLocaleString("en-US")} pixel-frames`, { corner: "tr", line: 1, fill: T.data });
    c.note("● last frame (10 pixels)", { corner: "tr", line: 2, fill: T.thread });
  }

  const ro = readout();
  function say() {
    if (nRec < 2) {
      ro.set(`Same beam, same pixels, but <b>different counts</b> in every frame. ` +
        `Each frame adds 10 counts to the histogram.<br>Press <b>1000 frames</b> to collect many frames at once.`);
      return;
    }
    const m = s1 / nRec, v = (s2 - nRec * m * m) / (nRec - 1), q = v / m;
    const cls = Math.abs(q - 1) < 0.15 ? "good" : "warn";
    ro.set(`${nRec.toLocaleString("en-US")} pixel-frames: mean <b>${fmt(m, 2)}</b>, variance <b>${fmt(v, 2)}</b>, ` +
      `variance/mean = <span class="${cls}">${fmt(q, 2)}</span> (Poisson: 1)` +
      (nRec < 300 ? ` <span class="warn">few frames: noisy estimate</span>` : "") +
      `<br>SNR = mean/σ = <b>${fmt(m / Math.sqrt(v), 2)}</b>, √λ = <b>${fmt(Math.sqrt(lam), 2)}</b>. ` +
      `4× the dose gives only 2× the SNR.`);
  }

  const playBtn = button("❚❚ pause", () => { playing = !playing; playBtn.innerHTML = playing ? "❚❚ pause" : "▶ play"; });
  const reset = () => { restart(); clearStats(); drawHist(); say(); draw(); };
  const ctl = controls(
    slider({ label: "dose λ", min: 0.5, max: 50, value: lam, log: true, digits: 1, unit: "e⁻/pixel/frame",
      onInput: (v) => { lam = v; reset(); } }),
    playBtn,
    button("1000 frames", () => {
      const v = new Int32Array(NP);
      for (let f = 0; f < 1000; f++) { for (let p = 0; p < NP; p++) v[p] = r.poisson(lam); record(v); }
      last.set(v); drawHist(); say();
    }, "primary"),
    button("reset", reset),
    toggle({ label: "speed", value: 1, onChange: (v) => { speed = v; },
      options: [{ value: 0.25, label: "slow" }, { value: 1, label: "normal" }, { value: 4, label: "fast" }] }),
  );
  el.append(ncard(ctl, row(cv, c.el), ro));
  reset();
  api.animate((dt) => {
    if (playing) step(dt * speed);
    for (let p = 0; p < NP; p++) flash[p] *= Math.exp(-dt * 7);
    draw();
  });
});

// ---------------------------------------------------------------------------------------------
// dose-image. Phantom: 8 x 8 heavy atomic columns (Gaussian, σ = 2 px, spacing 16 px) on a
// background of 8 %, scaled to 1 at a column peak, plus one weak dopant column at 30 % between
// four heavy columns. "Dose" = expected counts at a heavy-column peak, so pixel counts are
// Poisson(dose · I). Peak SNR = √dose; image-average SNR = √(dose · mean I). Dopant contrast-to-noise
// CNR = (I_dopant − I_bg)/σ_bg, with I_dopant the mean of the 3 x 3 core and σ_bg the pixel noise of
// the background; Rose criterion: visible if CNR > 3. The Gaussian option adds noise with the same
// variance everywhere (the image-average Poisson variance): it hides that dark pixels are quiet.
// ---------------------------------------------------------------------------------------------
const PH = (() => {
  const n = 128, sp = 16, w = 2, bg = 0.08, dop = [64, 64], dopA = 0.3;
  const sig = new Float64Array(n * n), add = (cx, cy, a) => {
    for (let i = Math.max(0, cy - 12); i <= Math.min(n - 1, cy + 12); i++)
      for (let j = Math.max(0, cx - 12); j <= Math.min(n - 1, cx + 12); j++)
        sig[i * n + j] += a * Math.exp(-((j - cx) ** 2 + (i - cy) ** 2) / (2 * w * w));
  };
  const peaks = [];
  for (let cy = sp / 2; cy < n; cy += sp) for (let cx = sp / 2; cx < n; cx += sp) { add(cx, cy, 1); peaks.push(cy * n + cx); }
  add(dop[0], dop[1], dopA);
  let mx = 0;
  for (const v of sig) mx = Math.max(mx, v + bg);
  const im = sig.map((v) => (v + bg) / mx);
  const core = (cx, cy) => { const o = []; for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) o.push((cy + i) * n + cx + j); return o; };
  const bgIdx = [];
  sig.forEach((v, i) => { if (v < 1e-3) bgIdx.push(i); });
  const b = bg / mx, dCore = core(dop[0], dop[1]), hCore = core(24, 24);
  return { n, im, mean: mean(im), peaks, bgIdx, dCore, b, dop,
    dDelta: mean(dCore.map((i) => im[i])) - b, hDelta: mean(hCore.map((i) => im[i])) - b };
})();

register("dose-image", (el) => {
  const S = 540, n = PH.n;
  let dose = 50, model = "pois", seed = 11;
  const noisy = new Float64Array(n * n);
  const im = imageCanvas({ n, size: S });
  const imLabel = h("div", { class: "xp-imlabel" });
  const mk = s("svg", { width: S, height: S, viewBox: `0 0 ${S} ${S}` },
    s("circle", { cx: (PH.dop[0] + 0.5) * S / n, cy: (PH.dop[1] + 0.5) * S / n, r: 6.5 * S / n, fill: "none", stroke: T.thread, "stroke-width": 3 }),
    s("text", { x: (PH.dop[0] + 7.5) * S / n, y: (PH.dop[1] + 0.5) * S / n + 7, fill: T.thread, "font-size": 20,
      "font-weight": 600, "paint-order": "stroke", stroke: "rgba(11,15,25,0.8)", "stroke-width": 4 }, "dopant"));
  // Noise variance of a pixel with normalised intensity v (units of the peak, squared).
  const varAt = (v, d) => (model === "pois" ? v : PH.mean) / d;
  const peakSNR = (d) => 1 / Math.sqrt(varAt(1, d)), avgSNR = (d) => Math.sqrt(d * PH.mean);
  const dopCNR = (d) => PH.dDelta / Math.sqrt(varAt(PH.b, d));
  const need = (delta) => 9 * varAt(PH.b, 1) / delta ** 2;             // dose for CNR = 3

  const c = chart({ width: 1110, height: 572, x: { domain: [1, 1e4], type: "log", label: "dose (e⁻ at a heavy-column peak)" },
    y: { domain: [0.2, 400], type: "log", label: "SNR or CNR" }, margin: { r: 30 } });
  const ds = logspace(1, 1e4, 120);
  const ro = readout();

  function update() {
    const r = rng(seed * 7919 + Math.round(Math.log(dose) * 1000));
    if (model === "pois") for (let i = 0; i < n * n; i++) noisy[i] = r.poisson(dose * PH.im[i]) / dose;
    else { const sd = Math.sqrt(PH.mean / dose); for (let i = 0; i < n * n; i++) noisy[i] = PH.im[i] + sd * r.normal(); }
    im.draw(noisy, 0, 1, "gray");
    imLabel.innerHTML = `dose <b>${big(dose)}</b> e⁻ at a heavy-column peak · ${model === "pois" ? "Poisson" : "Gaussian"} noise`;
    let e2 = 0;
    for (let i = 0; i < n * n; i++) e2 += (noisy[i] - PH.im[i]) ** 2;
    const avg = PH.mean / Math.sqrt(e2 / (n * n)), pk = stats(noisy, PH.peaks), bg = stats(noisy, PH.bgIdx);
    const peak = pk.m / Math.sqrt(pk.v), cnr = (mean(PH.dCore.map((i) => noisy[i])) - bg.m) / Math.sqrt(bg.v);

    c.clear();
    c.ruleY(3, { stroke: T.muted, width: 2, dash: "4 6" });
    c.text(8500, 3, "visible: CNR > 3", { fill: T.muted, size: 19, anchor: "end", dy: -8 });
    c.ruleX(dose, { stroke: T.thread, width: 1.5, dash: "2 6" });
    c.line(ds.map((d) => [d, peakSNR(d)]), { stroke: T.pois, width: 3.5 });
    c.line(ds.map((d) => [d, avgSNR(d)]), { stroke: T.pink, width: 3.5 });
    c.line(ds.map((d) => [d, dopCNR(d)]), { stroke: T.thread, width: 3.5, dash: "10 6" });
    const d3 = need(PH.dDelta);
    if (d3 >= 1 && d3 <= 1e4) {
      c.line([[d3, 0.2], [d3, 3]], { stroke: T.thread, width: 2, dash: "3 5" });
      c.dots([[d3, 3]], { r: 6, fill: T.bg, stroke: T.thread, strokeWidth: 3 });
      c.text(d3, 0.2, `dopant visible from dose ${big(d3)}`, { fill: T.thread, size: 19, dx: 10, dy: -14 });
    }
    c.dots([[dose, peak]], { r: 9, fill: T.pois, stroke: "#fff", top: true });
    c.dots([[dose, avg]], { r: 9, fill: T.pink, stroke: "#fff", top: true });
    c.dots([[dose, cnr]], { r: 9, fill: T.thread, stroke: "#fff", top: true });
    c.note(`peak column SNR`, { corner: "tl", fill: T.pois });
    c.note(`image-average SNR`, { corner: "tl", line: 1, fill: T.pink });
    c.note(`dopant CNR`, { corner: "tl", line: 2, fill: T.thread });
    c.note(`lines: theory, dots: measured`, { corner: "tl", line: 3, fill: T.muted, size: 19 });

    const vis = cnr > 3 ? `<span class="good">visible (CNR > 3)</span>` : `<span class="bad">lost in the noise</span>`;
    ro.set(`Dopant (30 % of a heavy column): CNR = <b>${fmt(cnr, 1)}</b> (theory ${fmt(dopCNR(dose), 1)}) → ${vis}. ` +
      `CNR > 3 needs a dose of <b>${big(d3)}</b> for the dopant, <b>${big(need(PH.hDelta))}</b> for a heavy column.<br>` +
      `Peak SNR <b>${fmt(peak, 1)}</b> (theory ${fmt(peakSNR(dose), 1)}), image-average SNR <b>${fmt(avg, 1)}</b> ` +
      `(theory √(dose · ${fmt(PH.mean, 3)}) = ${fmt(avgSNR(dose), 1)}).`);
  }
  const ctl = controls(
    slider({ label: "dose", min: 1, max: 1e4, value: dose, log: true, format: big, unit: "e⁻",
      onInput: (v) => { dose = v; update(); } }),
    toggle({ label: "noise", value: model, onChange: (v) => { model = v; update(); },
      options: [{ value: "pois", label: "Poisson", color: T.pois }, { value: "gauss", label: "Gaussian, same mean variance", color: T.gauss }] }),
    toggle({ label: "marker", value: true, onChange: (v) => { mk.style.display = v ? "" : "none"; },
      options: [{ value: true, label: "show" }, { value: false, label: "hide" }] }),
    button("new sample", () => { seed++; update(); }),
  );
  el.append(ncard(ctl, row(col(imLabel, h("div", { class: "xp-noise-stack" }, im, mk)), c.el), ro));
  update();
});

// ---------------------------------------------------------------------------------------------
// mixed-detector. A pixel value in ADU is x = g · k + ε with k ~ Poisson(λ) electrons and read noise
// ε ~ N(0, σr²). 100 000 pixels. The density is a Poisson-weighted sum of Gaussians at k · g; the
// expected histogram is computed per bin from the normal CDF (exact, also for σr = 0).
// Mean = gλ, Var = g²λ + σr² = g·x̄ + σr². Rounding x/g to the nearest integer gives the wrong
// electron count with probability ≈ 2(1 − Φ(g / 2σr)): counting works only for σr ≪ g.
// ---------------------------------------------------------------------------------------------
register("mixed-detector", (el) => {
  const N = 100000, NB = 320;
  const ks = new Int32Array(N), eps = new Float64Array(N), hist = new Float64Array(NB);
  let lam = 2, g = 20, sr = 2;
  const PRESETS = {
    dd: { g: 20, sr: 2, label: "direct detector, counting" },
    cmos: { g: 5, sr: 12, label: "CMOS, integrating" },
    ccd: { g: 2, sr: 25, label: "CCD, high read noise" },
  };
  const sampleK = () => { const r = rng(21); for (let i = 0; i < N; i++) ks[i] = r.poisson(lam); };
  { const r = rng(22); for (let i = 0; i < N; i++) eps[i] = r.normal(); }   // σr = 1; scaled later

  const c = chart({ width: 1170, height: 560, x: { domain: [0, 100], label: "pixel value x (ADU)" },
    y: { domain: [0, 1], label: "probability density (1/ADU)", tickFormat: (v) => String(+v.toPrecision(2)) },
    margin: { l: 96, t: 48 } });
  const ro = readout(), roRegime = readout("", "big");

  function update() {
    const t0 = performance.now();
    const kHi = Math.ceil(lam + 4.5 * Math.sqrt(lam) + 2), pad = Math.max(4 * sr, 0.6 * g, 2);
    const lo = -pad, hi = kHi * g + pad, w = (hi - lo) / NB;
    hist.fill(0);
    let s1 = 0, s2 = 0, wrong = 0;
    for (let i = 0; i < N; i++) {
      const x = g * ks[i] + sr * eps[i];
      s1 += x; s2 += x * x;
      if (Math.max(0, Math.round(x / g)) !== ks[i]) wrong++;
      const b = Math.floor((x - lo) / w);
      if (b >= 0 && b < NB) hist[b]++;
    }
    const m = s1 / N, v = (s2 - N * m * m) / (N - 1), pred = g * m + sr * sr;
    const pm = Array.from({ length: kHi + 3 }, (_, k) => poissonPmf(k, lam));
    const cdf = (z) => (sr > 0 ? Phi(z / sr) : z > 0 ? 1 : z < 0 ? 0 : 0.5);
    const meas = [], theo = [];
    let ymax = 0;
    for (let b = 0; b < NB; b++) {
      const b0 = lo + b * w, b1 = b0 + w;
      let p = 0;
      pm.forEach((q, k) => { p += q * (cdf(b1 - k * g) - cdf(b0 - k * g)); });
      const dm = hist[b] / (N * w), dt = p / w;
      meas.push([b0, dm], [b1, dm]); theo.push([b0 + w / 2, dt]);
      ymax = Math.max(ymax, dm, dt);
    }
    c.setDomain([lo, hi], [0, 1.14 * ymax]);
    c.clear();
    const px = c.x(g) - c.x(0), every = Math.ceil(64 / px);
    for (let k = 0; k * g <= hi; k++) {                               // guides at k electrons
      if (px < 10 && k % every) continue;
      c.ruleX(k * g, { stroke: T.pois, width: 1.5, dash: "3 5" });
      if (k % every === 0) c.text(k * g, 1.14 * ymax, `${k} e⁻`, { fill: T.pois, size: 19, anchor: "middle", dy: -10 });
    }
    c.area(meas, meas.map(([x]) => [x, 0]), { fill: T.data, opacity: 0.35 });
    c.line(meas, { stroke: T.data, width: 1.5 });
    c.line(theo, { stroke: T.truth, width: 3 });

    const q = sr / g;
    roRegime.set(q < 0.3 ? `<span class="good">σr ≪ g → you can count single electrons</span>`
      : q < 0.6 ? `<span class="warn">σr ≈ g/2 → peaks start to merge</span>`
      : `<span class="bad">σr ≳ g → peaks merge into one Gaussian</span>`);
    ro.set(`mean x̄ = <b>${fmt(m, 1)}</b> ADU (g·λ = ${fmt(g * lam, 1)})<br>` +
      `variance = <b>${fmt(v, 1)}</b> ADU²<br>` +
      `g·x̄ + σr² = ${fmt(g * m, 1)} + ${fmt(sr * sr, 1)} = <b>${fmt(pred, 1)}</b> ` +
      `<span class="${Math.abs(v / pred - 1) < 0.03 ? "good" : "warn"}">(${v >= pred ? "+" : "−"}${fmt(Math.abs(v / pred - 1) * 100, 1)} %)</span><br>` +
      `σr / g = <b>${fmt(q, 2)}</b>: rounding x/g gives the wrong electron count for <b>${fmt(100 * wrong / N, wrong / N < 0.01 ? 2 : 1)} %</b> of pixels.`);
    ro.dataset.ms = (performance.now() - t0).toFixed(1);
  }

  let preset;
  const sl = {
    lam: slider({ label: "dose λ", min: 0.2, max: 20, value: lam, log: true, digits: 1, unit: "e⁻/pixel",
      onInput: (v) => { lam = v; sampleK(); update(); } }),
    g: slider({ label: "gain g", min: 1, max: 50, value: g, digits: 1, unit: "ADU/e⁻",
      onInput: (v) => { g = v; preset.set(null, false); update(); } }),
    sr: slider({ label: "read noise σr", min: 0, max: 60, value: sr, digits: 1, unit: "ADU",
      onInput: (v) => { sr = v; preset.set(null, false); update(); } }),
  };
  preset = toggle({ label: "detector", value: "dd",
    options: Object.entries(PRESETS).map(([value, p]) => ({ value, label: p.label })),
    onChange: (v) => { const p = PRESETS[v]; g = p.g; sr = p.sr; sl.g.set(g, false); sl.sr.set(sr, false); update(); } });
  const side = h("div", { class: "xp-noise-side" },
    legend([[T.data, "measured histogram"], [T.truth, "theory: Σ Poisson(k) · N(k·g, σr²)"], [T.pois, "k·g: k electrons", "dash"]]),
    roRegime, ro);
  el.append(ncard(h("div", { class: "xp-noise-narrow" }, controls(sl.lam, sl.g, sl.sr), controls(preset)), row(c.el, side)));
  sampleK();
  update();
});

// ---------------------------------------------------------------------------------------------
// ptc (variance-mean plot, photon transfer curve). 31 flat-field levels from 0.3 to 10⁵ e⁻ per
// pixel; each level is one frame of 4000 pixels: x = g · f_i · Poisson(λ) + N(0, σr²), with a fixed
// per-pixel gain factor f_i = 1 + p · N(0, 1) (the same map at every level). Across the frame,
// Var ≈ σr² + g·x̄ + (p·x̄)². Read noise dominates below x̄ = σr²/g, fixed-pattern noise above g/p².
// The linear fit (weights 1/Var², because the spread of a sample variance is ∝ Var) uses only the
// levels where (p·x̄)² < 2 % of g·x̄; its slope estimates g and its intercept σr².
// ---------------------------------------------------------------------------------------------
register("ptc", (el) => {
  const NPX = 4000, lams = logspace(10 ** -0.5, 1e5, 31), z = new Float64Array(NPX);
  { const r = rng(5); for (let i = 0; i < NPX; i++) z[i] = r.normal(); }
  let g = 2.5, sr = 6, p = 0.01;

  const lin = chart({ width: 830, height: 470, x: { domain: [0, 1], label: "mean signal x̄ (DN)", tickCount: 4 },
    y: { domain: [0, 1], label: "variance (DN²)", tickCount: 5 }, margin: { l: 96 }, title: "linear: the fitted points" });
  const lg = chart({ width: 860, height: 470, x: { domain: [0.1, 1e6], type: "log", label: "mean signal x̄ (DN)" },
    y: { domain: [0.1, 1e10], type: "log", label: "variance (DN²)" }, margin: { l: 96 }, title: "log-log: three regimes" });
  const ro = readout();
  const TERMS = [["read noise", T.gauss], ["shot noise", T.pois], ["fixed pattern", T.warn]];

  function update() {
    const t0 = performance.now();
    const pts = lams.map((l, j) => {
      const r = rng(100 + j);
      let s1 = 0, s2 = 0;
      for (let i = 0; i < NPX; i++) { const v = g * (1 + p * z[i]) * r.poisson(l) + sr * r.normal(); s1 += v; s2 += v * v; }
      const m = s1 / NPX;
      return { m, v: (s2 - NPX * m * m) / (NPX - 1) };
    });
    pts.forEach((q) => { q.fit = (p * q.m) ** 2 < 0.02 * g * q.m; });
    const F = pts.filter((q) => q.fit);
    let Sw = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;                    // weighted least squares v = a + b·m
    for (const q of F) { const w = 1 / q.v ** 2; Sw += w; Sx += w * q.m; Sy += w * q.v; Sxx += w * q.m * q.m; Sxy += w * q.m * q.v; }
    const D = Sw * Sxx - Sx * Sx, b = (Sw * Sxy - Sx * Sy) / D, a = (Sy - b * Sx) / Sw;
    const u = 2 / (NPX - 1), da = Math.sqrt(u * Sxx / D), db = Math.sqrt(u * Sw / D);  // Var(v̂) = 2v²/(n − 1)

    const xF = Math.max(...F.map((q) => q.m)) * 1.08, yF = Math.max(...F.map((q) => q.v)) * 1.1;
    lin.setDomain([0, xF], [0, yF]);
    lin.clear();
    lin.line([[0, a], [xF, a + b * xF]], { stroke: T.thread, width: 3 });
    lin.dots(F.map((q) => [q.m, q.v]), { r: 7, fill: T.data, strokeWidth: 0 });
    lin.note(`fit: slope = ${fmt(b, 2)} DN/e⁻`, { corner: "tl", fill: T.thread });
    lin.note(`intercept = ${fmt(a, 1)} DN²`, { corner: "tl", line: 1, fill: T.thread });
    lin.note(`${F.length} of ${pts.length} levels: (p·x̄)² < 2 % of g·x̄`, { corner: "br", fill: T.muted, size: 19 });

    const P = pts.filter((q) => q.m > 0 && q.v > 0);
    const x0 = 10 ** Math.floor(Math.log10(Math.min(...P.map((q) => q.m)))), x1 = 10 ** Math.ceil(Math.log10(Math.max(...P.map((q) => q.m))));
    const y0 = 10 ** Math.floor(Math.log10(Math.min(sr * sr, ...P.map((q) => q.v)))), y1 = 10 ** Math.ceil(Math.log10(Math.max(...P.map((q) => q.v))) + 0.3);
    lg.setDomain([x0, x1], [y0, y1]);
    lg.clear();
    const xs = logspace(x0, x1, 400), term = (x) => [sr * sr, g * x, (p * x) ** 2];
    const segs = [];                                                   // dominant term along x̄
    for (const x of xs) {
      const t = term(x), k = t.indexOf(Math.max(...t));
      if (!segs.length || segs[segs.length - 1].k !== k) segs.push({ k, a: x, b: x }); else segs[segs.length - 1].b = x;
    }
    for (let i = 0; i + 1 < segs.length; i++) {                        // exact crossover between neighbours
      const A = segs[i].k, B = segs[i + 1].k;
      const xc = A + B === 1 ? sr * sr / g : A + B === 3 ? g / (p * p) : sr / p;
      segs[i].b = segs[i + 1].a = xc;
    }
    for (const sg of segs) {
      lg.rect(sg.a, y0, sg.b, y1, { fill: TERMS[sg.k][1], opacity: 0.09 });
      if (Math.log10(sg.b / sg.a) > 0.75) lg.text(Math.sqrt(sg.a * sg.b), y1, TERMS[sg.k][0],
        { fill: TERMS[sg.k][1], size: 20, anchor: "middle", dy: 26, weight: 600 });
    }
    lg.line(xs.map((x) => [x, sr * sr]), { stroke: T.gauss, width: 3, dash: "10 6" });
    lg.line(xs.map((x) => [x, g * x]), { stroke: T.pois, width: 3, dash: "10 6" });
    if (p > 0) lg.line(xs.map((x) => [x, (p * x) ** 2]), { stroke: T.warn, width: 3, dash: "10 6" });
    lg.line(xs.map((x) => [x, term(x).reduce((u, v) => u + v)]), { stroke: T.truth, width: 2 });
    lg.dots(P.filter((q) => q.fit).map((q) => [q.m, q.v]), { r: 7, fill: T.data, strokeWidth: 0 });
    lg.dots(P.filter((q) => !q.fit).map((q) => [q.m, q.v]), { r: 6, fill: "none", stroke: T.data, strokeWidth: 2.5 });

    const where = segs.map((sg, i) => `<b style="color:${TERMS[sg.k][1]}">${TERMS[sg.k][0]}</b> ` +
      (i === 0 ? `below ${big(sg.b)} DN` : i === segs.length - 1 ? `above ${big(sg.a)} DN` : `from ${big(sg.a)} to ${big(sg.b)} DN`)).join(", ");
    const biased = p > 0 && Math.abs(b - g) > 3 * db, good = F.length >= 3 && Math.abs(b / g - 1) < 0.05 && !biased;
    ro.set(`Weighted fit on <b>${F.length}</b> of ${pts.length} levels (left out: (p·x̄)² ≥ 2 % of g·x̄): ` +
      `g = <span class="${good ? "good" : "warn"}">${pmErr(b, db)}</span> DN/e⁻ (true ${fmt(g, 2)}), ` +
      `σr² = <b>${pmErr(a, da)}</b> DN² (true ${fmt(sr * sr, 1)}).` +
      (biased ? ` <span class="warn">Leftover fixed-pattern noise lifts the slope.</span>` : "") + `<br>` +
      `Crossovers σr²/g = <b>${big(sr * sr / g)}</b> DN, g/p² = <b>${p > 0 ? big(g / (p * p)) : "∞"}</b> DN. ` +
      `Dominant: ${where}.`);
    ro.dataset.ms = (performance.now() - t0).toFixed(1);
  }
  const ctl = controls(
    slider({ label: "gain g", min: 0.5, max: 10, value: g, digits: 1, unit: "DN/e⁻", onInput: (v) => { g = v; update(); } }),
    slider({ label: "read noise σr", min: 0.5, max: 20, value: sr, digits: 1, unit: "DN", onInput: (v) => { sr = v; update(); } }),
    slider({ label: "fixed-pattern noise p", min: 0, max: 3, value: p * 100, digits: 2, unit: "%", onInput: (v) => { p = v / 100; update(); } }),
  );
  const key = legend([[T.data, "measured (hollow: left out of the fit)", "dot"], [T.thread, "linear fit"], [T.gauss, "σr²", "dash"],
    [T.pois, "g·x̄", "dash"], [T.warn, "(p·x̄)²", "dash"], [T.truth, "sum"]]);
  el.append(ncard(h("div", { class: "xp-noise-narrow" }, ctl), key, row(lin.el, lg.el), ro));
  update();
});
