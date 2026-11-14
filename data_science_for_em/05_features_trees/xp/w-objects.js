// Objects and descriptors (Acts 1 and 2), on real HAADF-STEM images of one oxide interface (STEM-Bank-26):
//   find-columns : ten aligned frames of one image stack. Sum n frames, smooth, subtract the background,
//                  take the local maxima above a threshold. The amorphous layer has no columns: each
//                  maximum there is a false detection.
//   descriptor   : click one candidate of image B. The table shows its 11 descriptors. A gain and an
//                  offset of the image, or a rotation, change some descriptors and not others.
import { register, T, fmt, h, s, row, col, card, controls, readout, slider, toggle, legend, chart } from "./core.js";
import { loadData } from "./data.js";
import { CLS, SRC, src, waiting, loadCols, colImage } from "./w5-common.js";

if (!document.getElementById("xp-obj-style")) {
  document.head.append(h("style", { id: "xp-obj-style" }, `
.xp-fc-img { position: relative; line-height: 0; }
.xp-fc-img svg { position: absolute; left: 0; top: 0; }
.xp-desc-wrap { transition: transform 0.6s; }
.xp-tbl.desc td { padding: 5px 12px; font-size: 22px; white-space: nowrap; }
.xp-tbl.desc td.about { white-space: normal; }
.xp-tbl.desc td.about { color: #94a3b8; font-size: 18px; text-align: left; max-width: 330px; line-height: 1.2; }
.xp-tbl.desc td.name { font-weight: 700; color: #fff; }
.xp-tbl.desc td.chg { color: #f43f5e; font-weight: 700; }
.xp-tbl.desc td.same { color: #34d399; font-weight: 700; }
`));
}
const GREY = "#94a3b8";

// ---------------------------------------------------------------------------------------------
// Filters on a W x H image (Float32Array)
// ---------------------------------------------------------------------------------------------
function gaussBlur(a, W, H, sig) {
  if (sig < 0.3) return Float32Array.from(a);
  const r = Math.ceil(3 * sig), k = new Float32Array(2 * r + 1); let sum = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-0.5 * (i / sig) ** 2); sum += k[i + r]; }
  for (let i = 0; i <= 2 * r; i++) k[i] /= sum;
  const t = new Float32Array(W * H), o = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = 0; for (let i = -r; i <= r; i++) { const xx = Math.min(W - 1, Math.max(0, x + i)); v += k[i + r] * a[y * W + xx]; }
    t[y * W + x] = v;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = 0; for (let i = -r; i <= r; i++) { const yy = Math.min(H - 1, Math.max(0, y + i)); v += k[i + r] * t[yy * W + x]; }
    o[y * W + x] = v;
  }
  return o;
}
// Box blur of half-width r with edge clamping (running sums); three passes approximate a Gaussian.
function boxBlur(a, W, H, r) {
  const t = new Float32Array(W * H), o = new Float32Array(W * H), n = 2 * r + 1;
  for (let y = 0; y < H; y++) {
    let acc = 0; const at = (x) => a[y * W + Math.min(W - 1, Math.max(0, x))];
    for (let i = -r; i <= r; i++) acc += at(i);
    for (let x = 0; x < W; x++) { t[y * W + x] = acc / n; acc += at(x + r + 1) - at(x - r); }
  }
  for (let x = 0; x < W; x++) {
    let acc = 0; const at = (y) => t[Math.min(H - 1, Math.max(0, y)) * W + x];
    for (let i = -r; i <= r; i++) acc += at(i);
    for (let y = 0; y < H; y++) { o[y * W + x] = acc / n; acc += at(y + r + 1) - at(y - r); }
  }
  return o;
}
const wideBlur = (a, W, H, sig) => { const r = Math.round(Math.sqrt(sig * sig + 1 / 3 - 1 / 3) ); let o = a; for (let i = 0; i < 3; i++) o = boxBlur(o, W, H, r); return o; };
// Local maxima of d in a (2w+1)^2 window. Returns [{x, y, v}] sorted by v, largest first.
function localMaxima(d, W, H, w) {
  const t = new Float32Array(W * H), m = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = -Infinity; for (let i = Math.max(0, x - w); i <= Math.min(W - 1, x + w); i++) v = Math.max(v, d[y * W + i]);
    t[y * W + x] = v;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = -Infinity; for (let i = Math.max(0, y - w); i <= Math.min(H - 1, y + w); i++) v = Math.max(v, t[i * W + x]);
    m[y * W + x] = v;
  }
  const out = [], b = w + 2;
  for (let y = b + 1; y < H - b; y++) for (let x = b + 1; x < W - b; x++) { const v = d[y * W + x]; if (v === m[y * W + x] && v > 0) out.push({ x, y, v }); }
  return out.sort((p, q) => q.v - p.v);
}

register("find-columns", (el, api) => {
  const w = waiting(el);
  loadData("frames").then((d) => {
    w.remove();
    const [NF, H, W] = d.frames.shape, sc = d.scale_nm, Z = d.zones, ZOOM = 2.4, PW = Math.round(W * ZOOM), PH = Math.round(H * ZOOM);
    const zoneOf = (x) => { const xn = d.x0_nm + x * sc; return xn < Z[0] ? 0 : xn > Z[1] && xn < Z[2] ? 1 : xn > Z[3] ? 2 : -1; };
    // cumulative sums of the frames
    const sums = [new Float32Array(W * H)];
    for (let f = 0; f < NF; f++) { const a = Float32Array.from(sums[f]); for (let i = 0; i < W * H; i++) a[i] += d.frames[f * W * H + i]; sums.push(a); }
    const bgCache = {}, pkCache = {};
    const peaks = (n, sig) => {
      const key = `${n}|${sig.toFixed(3)}`;
      if (pkCache[key]) return pkCache[key];
      const a = sums[n];
      if (!bgCache[n]) bgCache[n] = wideBlur(a, W, H, 0.8 / sc);
      const sm = gaussBlur(a, W, H, sig / sc), bg = bgCache[n], dd = new Float32Array(W * H);
      for (let i = 0; i < W * H; i++) dd[i] = sm[i] - bg[i];
      const sorted = Float32Array.from(dd).sort(), p995 = sorted[Math.floor(0.995 * (W * H - 1))];
      const pk = localMaxima(dd, W, H, Math.round(0.12 / sc));
      pk.forEach((p) => { p.z = zoneOf(p.x); p.rel = p.v / p995; });
      return (pkCache[key] = pk);
    };

    const cv = h("canvas", { width: W, height: H, class: "xp-image" });
    cv.style.width = `${PW}px`; cv.style.height = `${PH}px`; cv.style.imageRendering = "auto";
    const ctx = cv.getContext("2d"), im = ctx.createImageData(W, H);
    const svg = s("svg", { width: PW, height: PH, viewBox: `0 0 ${PW} ${PH}` }), gz = s("g"), gm = s("g");
    svg.append(gz, gm);
    const px = (xn) => (xn - d.x0_nm) / sc * ZOOM;
    for (const xn of [(Z[0] + Z[1]) / 2, (Z[2] + Z[3]) / 2])
      gz.append(s("line", { x1: px(xn), x2: px(xn), y1: 0, y2: PH, stroke: "#fff", "stroke-width": 2, "stroke-dasharray": "8 8", opacity: 0.6 }));
    const gl = s("g"); svg.append(gl);
    const lab = (xn, txt, c) => { const wd = txt.length * 12.5 + 18;
      gl.append(s("rect", { x: px(xn) - wd / 2, y: PH - 38, width: wd, height: 30, rx: 8, fill: "rgba(11,15,25,0.85)" }),
        s("text", { x: px(xn), y: PH - 16, fill: c, "font-size": 21, "font-weight": 700, "text-anchor": "middle" }, txt)); };
    lab((d.x0_nm + Z[0]) / 2, "substrate", CLS[0].color); lab((Z[1] + Z[2]) / 2, "film", CLS[1].color); lab((Z[3] + d.x0_nm + W * sc) / 2, "amorphous", CLS[2].color);
    const imgBox = h("div", { class: "xp-fc-img", style: { width: `${PW}px`, height: `${PH}px` } }, cv, svg);

    const ch = chart({ width: 820, height: 400, x: { domain: [0, 30], label: "threshold (% of the 99.5th percentile)" },
      y: { domain: [0, 100], label: "detected maxima" }, title: "Detected maxima per zone against the threshold" });
    const out = readout("");
    const draw = () => {
      const n = Math.round(sN.value), sig = sS.value, thr = sT.value / 100, pk = peaks(n, sig), a = sums[n];
      // image: 0.5 to 99.8 percentile
      const so = Float32Array.from(a).sort(), lo = so[Math.floor(0.005 * so.length)], hi = so[Math.floor(0.998 * so.length)];
      for (let i = 0; i < W * H; i++) { const v = Math.max(0, Math.min(255, 255 * (a[i] - lo) / (hi - lo))); im.data[4 * i] = v; im.data[4 * i + 1] = v; im.data[4 * i + 2] = v; im.data[4 * i + 3] = 255; }
      ctx.putImageData(im, 0, 0);
      gm.replaceChildren();
      const cnt = [0, 0, 0, 0];
      for (const p of pk) {
        if (p.rel <= thr) break;
        cnt[p.z < 0 ? 3 : p.z]++;
        gm.append(s("circle", { cx: (p.x + 0.5) * ZOOM, cy: (p.y + 0.5) * ZOOM, r: 7, fill: "none", stroke: p.z < 0 ? GREY : CLS[p.z].color, "stroke-width": 2.5 }));
      }
      // counts against the threshold
      const ts = Array.from({ length: 121 }, (_, i) => i * 0.25), curves = [[], [], []];
      for (const t of ts) { const c = [0, 0, 0]; for (const p of pk) { if (p.rel <= t / 100) break; if (p.z >= 0) c[p.z]++; } c.forEach((v, k) => curves[k].push([t, v])); }
      const ymax = Math.max(60, ...curves.map((c) => c[0][1])) * 1.08;
      ch.setDomain(null, [0, ymax]); ch.clear();
      curves.forEach((c, k) => ch.line(c, { stroke: CLS[k].color, width: 3.5 }));
      ch.ruleX(sT.value, { stroke: T.thread, width: 3 });
      out.set(`<b>${n}</b> frame${n > 1 ? "s" : ""} summed, threshold ${fmt(sT.value, 1)} %:<br>` +
        `<span style="color:${CLS[0].color}"><b>${cnt[0]}</b> in the substrate</span> · <span style="color:${CLS[1].color}"><b>${cnt[1]}</b> in the film</span><br>` +
        `<span style="color:${CLS[2].color}"><b>${cnt[2]}</b> in the amorphous layer</span> = <span class="${cnt[2] ? "bad" : "good"}">${cnt[2]} false detection${cnt[2] === 1 ? "" : "s"}</span><br>` +
        `<span style="color:${GREY}">${cnt[3]} near a boundary (no label)</span>`);
    };
    const sN = slider({ label: "frames summed (dose)", min: 1, max: NF, value: +(api.data.frames || NF), step: 1, digits: 0, onInput: draw });
    const sS = slider({ label: "smoothing σ", min: 0, max: 0.08, value: +(api.data.sigma || 0.04), step: 0.005, digits: 3, unit: "nm", onInput: draw });
    const sT = slider({ label: "threshold", min: 0, max: 30, value: +(api.data.thr || 4), step: 0.25, digits: 2, unit: "%", onInput: draw });
    el.append(row(
      col(imgBox, h("div", { class: "xp-caption", html: `Circles: local maxima above the threshold. ${fmt(W * sc, 1)} nm × ${fmt(H * sc, 1)} nm, pixel 20 pm.` })),
      h("div", { class: "xp-side", style: { width: "860px" } }, card(col(sN, sS, sT)), card(ch.el, legend(CLS.map((c, k) => [c.color, ["substrate", "film", "amorphous layer (no columns)"][k]]))), out)),
      src("Data: one HAADF-STEM image stack (10 frames) of an oxide film on LaAlO₃; STEM-Bank-26 (Hollyer &amp; Stach 2026), CC BY 4.0."));
    draw();
  });
});

// ---------------------------------------------------------------------------------------------
// descriptor
// ---------------------------------------------------------------------------------------------
register("descriptor", (el, api) => {
  const w = waiting(el);
  loadCols().then((d) => {
    w.remove();
    const K = 1, rows = []; d.g.forEach((g, i) => { if (g === K) rows.push(i); });
    const I = d.images[K], Z = I.zones, SIZE = 600;
    const im = colImage(d, K, SIZE); im.zones(true);
    const wrap = h("div", { class: "xp-desc-wrap" }, im);
    // default: a film column near the interface, at mid height
    let sel = +api.data.pick || rows.filter((i) => d.y[i] === 1).reduce((b, i) =>
      Math.hypot(d.pos[i][1] - (Z[1] + 0.45), d.pos[i][0] - I.size_nm[0] / 2) < Math.hypot(d.pos[b][1] - (Z[1] + 0.45), d.pos[b][0] - I.size_nm[0] / 2) ? i : b);
    const GAIN = 3, medBg = rows.map((i) => d.X[i][d.f("bg")]).sort((a, b) => a - b)[rows.length >> 1], OFF = 0.5 * medBg;
    // The image I' = GAIN * I + OFF: which descriptors change? (definitions: xp/prep/make_columns.py)
    const transform = (x) => {
      const o = x.slice(), a = d.f("amp"), b = d.f("bg"), c = d.f("contrast");
      o[a] = GAIN * x[a]; o[b] = GAIN * x[b] + OFF; o[c] = o[a] / o[b];
      return o;
    };
    const val = (j, v) => (d.units[j] === "counts" ? `${fmt(v / 1e6, 2)} × 10⁶` : d.names[j] === "n_near" ? fmt(v, 0) : fmt(v, 3)) + (d.units[j] === "nm" ? " nm" : "");
    const tbl = h("table", { class: "xp-tbl desc" }), out = readout("");
    // zoom: +-1 nm around the selected candidate
    const ZS = 300, ZR = 1.0, zc = h("canvas", { width: ZS, height: ZS, class: "xp-image" }), zsvg = s("svg", { width: ZS, height: ZS, viewBox: `0 0 ${ZS} ${ZS}` });
    zc.style.imageRendering = "auto";
    const zoomBox = h("div", { class: "xp-colimg", style: { width: `${ZS}px`, height: `${ZS}px` } }, zc, zsvg);
    const cvBig = im.querySelector("canvas"), n480 = I.img.shape[0];
    const drawZoom = () => {
      const [yn, xn] = d.pos[sel], k = n480 / I.size_nm[0], ctx = zc.getContext("2d");
      ctx.fillStyle = "#000"; ctx.fillRect(0, 0, ZS, ZS); ctx.imageSmoothingQuality = "high";
      ctx.drawImage(cvBig, (xn - ZR) * k, (yn - ZR) * k, 2 * ZR * k, 2 * ZR * k, 0, 0, ZS, ZS);
      const q = (a, a0) => (a - a0 + ZR) / (2 * ZR) * ZS;
      zsvg.replaceChildren(s("circle", { cx: ZS / 2, cy: ZS / 2, r: 0.9 / (2 * ZR) * ZS, fill: "none", stroke: "#fff", "stroke-width": 1.5, "stroke-dasharray": "6 6", opacity: 0.8 }));
      for (const j of d.nb[sel]) zsvg.append(s("line", { x1: ZS / 2, y1: ZS / 2, x2: q(d.pos[j][1], xn), y2: q(d.pos[j][0], yn), stroke: T.warn, "stroke-width": 3 }));
      for (const i of rows) { const X = q(d.pos[i][1], xn), Y = q(d.pos[i][0], yn);
        if (X > 4 && X < ZS - 4 && Y > 4 && Y < ZS - 4) zsvg.append(s("circle", { cx: X, cy: Y, r: i === sel ? 9 : 5, fill: i === sel ? "none" : d.y[i] < 0 ? GREY : CLS[d.y[i]].color,
          stroke: i === sel ? T.thread : "none", "stroke-width": 4 })); }
    };
    const state = { gain: api.data.gain === "1", rot: api.data.rotate === "1" };
    const draw = () => {
      im.marks(rows, (i) => ({ r: 3.6, fill: d.y[i] < 0 ? GREY : CLS[d.y[i]].color, opacity: i === sel ? 1 : 0.8 }), (i) => { sel = i; draw(); });
      const g = im.top; g.replaceChildren();
      const cx = im.sx(d.pos[sel][1]), cy = im.sy(d.pos[sel][0]);
      g.append(s("circle", { cx, cy, r: im.sx(0.9), fill: "rgba(255,255,255,0.05)", stroke: "#fff", "stroke-width": 1.5, "stroke-dasharray": "6 6", opacity: 0.8, "pointer-events": "none" }));
      for (const j of d.nb[sel]) g.append(s("line", { x1: cx, y1: cy, x2: im.sx(d.pos[j][1]), y2: im.sy(d.pos[j][0]), stroke: T.warn, "stroke-width": 3, "pointer-events": "none" }));
      g.append(s("circle", { cx, cy, r: 11, fill: "none", stroke: T.thread, "stroke-width": 4, "pointer-events": "none" }));
      wrap.style.transform = state.rot ? "rotate(90deg)" : ""; zoomBox.style.transform = wrap.style.transform; drawZoom();
      const x = d.X[sel], xt = state.gain ? transform(x) : x, on = state.gain || state.rot;
      tbl.replaceChildren(h("tr", {}, h("th", {}, "descriptor"), h("th", {}, "value"), on ? h("th", {}, "after the change") : null, h("th", { style: { textAlign: "left" } }, "meaning")));
      d.names.forEach((nm, j) => {
        const chg = Math.abs(xt[j] - x[j]) > 1e-9 * Math.abs(x[j]);
        tbl.append(h("tr", {}, h("td", { class: "name" }, nm), h("td", {}, val(j, x[j])),
          on ? h("td", { class: chg ? "chg" : "same" }, chg ? val(j, xt[j]) : "same") : null, h("td", { class: "about" }, d.about[j])));
      });
      const y = d.y[sel];
      out.set(`Selected: <b style="color:${y < 0 ? GREY : CLS[y].color}">${y < 0 ? "no label (near a boundary)" : d.classes[y]}</b>. ` +
        (state.gain ? `Image → ${GAIN} × image + offset: <span class="bad">amp, bg and contrast change</span>; <span class="good">the ratio I_rel, the widths and all distances stay</span>.`
          : state.rot ? `A rotation moves the column, but <span class="good">no descriptor changes</span>: distances and |ψ₄| do not depend on the direction.`
            : `Yellow: the 4 nearest candidates. Dashed circle: 0.9 nm, the cutoff of the radial fingerprint.`));
    };
    const tg = toggle({ label: "detector", options: [{ value: "0", label: "as recorded" }, { value: "1", label: `gain ×${GAIN} + offset` }], value: state.gain ? "1" : "0", onChange: (v) => { state.gain = v === "1"; draw(); } });
    const tr = toggle({ label: "scan", options: [{ value: "0", label: "as recorded" }, { value: "1", label: "rotate 90°" }], value: state.rot ? "1" : "0", onChange: (v) => { state.rot = v === "1"; draw(); } });
    el.append(row(
      col(wrap, legend([[CLS[0].color, "substrate"], [CLS[1].color, "film"], [CLS[2].color, "no column"], [GREY, "no label"]])),
      h("div", { class: "xp-side", style: { width: "1140px" } }, controls(tg, tr), card(row(tbl, col(zoomBox, h("div", { class: "xp-caption", html: "zoom: 2 nm × 2 nm" })))), out)),
      src(SRC + " Image B; click a candidate."));
    draw();
  });
});
