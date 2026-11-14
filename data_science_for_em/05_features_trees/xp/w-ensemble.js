// Ensemble group (Act 4), on real data (four HAADF-STEM images of one oxide interface, STEM-Bank-26):
//   bagging : the votes of 200 trees for the 580 labelled candidates of image B. The trees were trained
//             on the images A, C and D (six invariant descriptors). Slider: number of trees B.
//             Bagging against random forest; accuracy against B; the variance floor ρσ².
//   boost   : gradient boosting with small trees (depth 2, data-depth="1" gives stumps) for the squared
//             loss, computed here. Target: the peak height
//             across the interface (relative to the substrate columns). Training: image B.
//             Held-out: image D. Learning rate η, stage m, residuals and the next stump.
import { register, T, fmt, clamp, rng, h, s, row, col, card, controls, readout, slider, toggle, button, legend, chart } from "./core.js";
import { loadData } from "./data.js";
import { CLS, SRC, src, waiting, loadCols, colImage } from "./w5-common.js";

if (!document.getElementById("xp-ens-style")) {
  document.head.append(h("style", { id: "xp-ens-style" }, `
.xp-ens-side { width: 1010px; display: flex; flex-direction: column; gap: 10px; }
.xp-ens-side .xp-readout { margin-top: 0; line-height: 1.5; }
.xp-ens-side .xp-controls { margin: 0; }
.xp-ens-votes { min-height: 74px; font-size: 21px; color: #94a3b8; }
`));
}
const GOOD = "#34d399", BAD = "#f43f5e";

// ---------------------------------------------------------------------------------------------
// bagging
// ---------------------------------------------------------------------------------------------
register("bagging", (el, api) => {
  const w = waiting(el);
  Promise.all([loadCols(), loadData("models")]).then(([d, M]) => {
    w.remove();
    const NT = 200, n = M.votes_rows.length;
    // votes_rows counts in the table of labelled candidates; idx maps that table to the rows of columns.json
    const rows = Array.from(M.votes_rows, (r) => M.idx[r]), y = rows.map((r) => d.y[r]);
    const KINDS = ["bagging", "random forest"], COL = { bagging: T.gauss, "random forest": T.warn };
    let kind = api.data.kind === "bagging" ? "bagging" : "random forest";
    let B = clamp(+(api.data.b || 1), 1, NT), perm = Array.from({ length: NT }, (_, i) => i), seed = 1, sel = -1;

    // counts[kind][b] is not stored: accumulate on demand. acc[kind][b-1] for the current tree order.
    const accCurve = (k) => {
      const v = M.votes[k], cnt = new Uint16Array(3 * n), out = [];
      for (let b = 0; b < NT; b++) {
        const t = perm[b] * n; let ok = 0;
        for (let i = 0; i < n; i++) {
          cnt[3 * i + v[t + i]]++;
          const a = cnt[3 * i], c = cnt[3 * i + 1], e = cnt[3 * i + 2], p = a >= c && a >= e ? 0 : c >= e ? 1 : 2;
          if (p === y[i]) ok++;
        }
        out.push(ok / n);
      }
      return out;
    };
    const counts = (k, b) => {
      const v = M.votes[k], cnt = new Uint16Array(3 * n);
      for (let q = 0; q < b; q++) { const t = perm[q] * n; for (let i = 0; i < n; i++) cnt[3 * i + v[t + i]]++; }
      return cnt;
    };
    const single = Object.fromEntries(KINDS.map((k) => { let ok = 0; const v = M.votes[k]; for (let t = 0; t < NT; t++) for (let i = 0; i < n; i++) if (v[t * n + i] === y[i]) ok++; return [k, ok / (NT * n)]; }));
    let curves = Object.fromEntries(KINDS.map((k) => [k, accCurve(k)]));

    const im = colImage(d, 1, 610);
    const ch = chart({ width: 1000, height: 370, x: { domain: [1, 200], type: "log", label: "number of trees B", ticks: [1, 2, 5, 10, 20, 50, 100, 200], tickFormat: (v) => String(v) },
      y: { domain: [0.74, 0.9], label: "accuracy on image B", tickFormat: (v) => `${Math.round(v * 100)} %` }, margin: { l: 90, t: 14 } });
    const out = readout(), votes = h("div", { class: "xp-ens-votes" });
    const ring = s("circle", { r: 13, fill: "none", stroke: "#fff", "stroke-width": 3, visibility: "hidden" });
    im.top.append(ring);

    const voteBar = (cnt) => {
      if (sel < 0) { votes.innerHTML = "Click a candidate in the image to see the votes of its trees."; return; }
      const c = [cnt[3 * sel], cnt[3 * sel + 1], cnt[3 * sel + 2]], W = 620, svg = s("svg", { width: W, height: 34 });
      let x = 0;
      c.forEach((v, k) => { const wd = v / B * W; if (wd > 0) svg.append(s("rect", { x, y: 2, width: wd, height: 30, fill: CLS[k].color, opacity: 0.9 })); x += wd; });
      votes.replaceChildren(h("div", { html: `Votes for this candidate (truth: <b style="color:${CLS[y[sel]].color}">${CLS[y[sel]].label.replace(/ \(.*/, "")}</b>): ` +
        c.map((v, k) => `<b style="color:${CLS[k].color}">${v}</b> ${CLS[k].key}`).join(" · ") }), svg);
    };

    function draw() {
      const cnt = counts(kind, B), pred = new Uint8Array(n), share = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const a = cnt[3 * i], c = cnt[3 * i + 1], e = cnt[3 * i + 2];
        pred[i] = a >= c && a >= e ? 0 : c >= e ? 1 : 2; share[i] = Math.max(a, c, e) / B;
      }
      const at = new Map(rows.map((r, i) => [r, i]));
      im.marks(rows, (r) => { const i = at.get(r); return { fill: pred[i] === y[i] ? GOOD : BAD, r: 3 + 3.5 * (share[i] - 1 / 3) * 1.5, opacity: 0.9 }; },
        (r) => { sel = at.get(r); draw(); });
      if (sel >= 0) { ring.setAttribute("cx", im.sx(d.pos[rows[sel]][1])); ring.setAttribute("cy", im.sy(d.pos[rows[sel]][0])); ring.setAttribute("visibility", "visible"); }
      voteBar(cnt);

      ch.clear();
      for (const k of KINDS) {
        const on = k === kind;
        ch.line(curves[k].map((a, b) => [b + 1, a]), { stroke: COL[k], width: on ? 4 : 2.5, opacity: on ? 1 : 0.4 });
      }
      ch.ruleY(single[kind], { stroke: T.muted });
      ch.text(200, single[kind], `mean of single trees ${fmt(100 * single[kind], 1)} %`, { anchor: "end", dy: 24, dx: -8, fill: T.muted, size: 19 });
      ch.ruleX(B, { stroke: T.thread });
      ch.dots([[B, curves[kind][B - 1]]], { r: 9, fill: T.thread, stroke: "#fff", top: true });
      const rho = M.ens[kind].rho, acc = curves[kind][B - 1], wrong = Math.round((1 - acc) * n);
      out.set(`<b>${B}</b> ${B === 1 ? "tree" : "trees"} (${kind}): accuracy <b class="${acc > single[kind] + 0.03 ? "good" : "warn"}">${fmt(100 * acc, 1)} %</b>, ${wrong} of ${n} wrong.<br>` +
        `Variance of the average: ρσ² + (1 − ρ)σ²/B = (<b>${fmt(rho, 2)}</b> + ${fmt((1 - rho) / B, 3)}) σ² = <b>${fmt(rho + (1 - rho) / B, 2)} σ²</b>. ` +
        `The floor is ρσ² = ${fmt(rho, 2)} σ².`);
    }

    const sB = slider({ label: "trees B", min: 1, max: NT, value: B, log: true, digits: 0, format: (v) => String(Math.round(v)), onInput: (v) => { B = Math.round(v); draw(); } });
    const tk = toggle({ options: KINDS.map((k) => ({ value: k, label: k, color: COL[k] })), value: kind, onChange: (v) => { kind = v; draw(); } });
    const shuffle = button("shuffle tree order", () => {
      const R = rng(++seed * 7919);
      for (let i = NT - 1; i > 0; i--) { const j = Math.floor(R.u() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
      curves = Object.fromEntries(KINDS.map((k) => [k, accCurve(k)]));
      draw();
    });
    el.append(row(
      card(im, legend([[GOOD, "majority vote right"], [BAD, "majority vote wrong"]]), h("div", { class: "xp-caption", html: "Image B, not in the training data. Large dot = clear vote." })),
      h("div", { class: "xp-ens-side" }, controls(sB, tk, shuffle), card(ch.el), out, votes)), src(SRC));
    el.__summary = () => ({ kind, B, acc: curves[kind][B - 1], single, rho: M.ens[kind].rho, at: [1, 5, 10, 30, 100, 200].map((b) => [b, curves.bagging[b - 1], curves["random forest"][b - 1]]) });
    draw();
  }).catch((e) => { w.textContent = `data not loaded: ${e.message}`; });
});

// ---------------------------------------------------------------------------------------------
// boost
// ---------------------------------------------------------------------------------------------
// One regression tree of depth `depth` on a sorted 1-D feature: returns the cut points and leaf values.
function fitTree1d(xs, r, lo, hi, depth, minLeaf = 5) {
  let sum = 0; for (let i = lo; i < hi; i++) sum += r[i];
  const n = hi - lo, leaf = { leaf: true, v: sum / n };
  if (depth === 0 || n < 2 * minLeaf) return leaf;
  let best = -1, bestGain = 1e-12, sl = 0;
  for (let i = lo; i < hi - 1; i++) {
    sl += r[i];
    const nl = i - lo + 1, nr = n - nl;
    if (nl < minLeaf || nr < minLeaf || xs[i] === xs[i + 1]) continue;
    const gain = sl * sl / nl + (sum - sl) * (sum - sl) / nr - sum * sum / n;    // decrease of the summed squared error
    if (gain > bestGain) { bestGain = gain; best = i; }
  }
  if (best < 0) return leaf;
  return { thr: (xs[best] + xs[best + 1]) / 2, left: fitTree1d(xs, r, lo, best + 1, depth - 1, minLeaf), right: fitTree1d(xs, r, best + 1, hi, depth - 1, minLeaf) };
}
const evalTree = (t, x) => { while (!t.leaf) t = x <= t.thr ? t.left : t.right; return t.v; };

// Gradient boosting for the squared loss. xs must be sorted. Returns the trees and both error curves.
export function boostFit(xs, ys, xt, yt, eta, stages, depth = 1) {
  const n = xs.length, f0 = ys.reduce((a, b) => a + b, 0) / n;
  const F = new Float64Array(n).fill(f0), Ft = new Float64Array(xt.length).fill(f0), r = new Float64Array(n);
  const mse = (y, f) => { let q = 0; for (let i = 0; i < y.length; i++) q += (y[i] - f[i]) ** 2; return q / y.length; };
  const trees = [], train = [mse(ys, F)], test = [mse(yt, Ft)];
  for (let m = 0; m < stages; m++) {
    for (let i = 0; i < n; i++) r[i] = ys[i] - F[i];                             // pseudo-residuals of the squared loss
    const t = fitTree1d(xs, r, 0, n, depth);
    trees.push(t);
    for (let i = 0; i < n; i++) F[i] += eta * evalTree(t, xs[i]);
    for (let i = 0; i < xt.length; i++) Ft[i] += eta * evalTree(t, xt[i]);
    train.push(mse(ys, F)); test.push(mse(yt, Ft));
  }
  return { f0, trees, train, test, best: test.indexOf(Math.min(...test)) };
}

register("boost", (el, api) => {
  const w = waiting(el);
  loadCols().then((d) => {
    w.remove();
    const ETAS = [0.03, 0.1, 0.3, 1], STAGES = 200, jA = d.f("amp");
    // x: distance from the interface (nm); y: peak height / median peak height of the substrate columns
    const profile = (k) => {
      const z = d.images[k].zones, x0 = (z[0] + z[1]) / 2, sub = [], pts = [];
      for (let i = 0; i < d.N; i++) if (d.g[i] === k && d.y[i] === 0) sub.push(d.X[i][jA]);
      sub.sort((a, b) => a - b); const med = sub[sub.length >> 1];
      for (let i = 0; i < d.N; i++) if (d.g[i] === k) { const x = d.pos[i][1] - x0; if (x > -4.2 && x < 4.2) pts.push([x, d.X[i][jA] / med]); }
      return pts.sort((a, b) => a[0] - b[0]);
    };
    const tr = profile(1), te = profile(3);
    const xs = tr.map((p) => p[0]), ys = tr.map((p) => p[1]), xt = te.map((p) => p[0]), yt = te.map((p) => p[1]);
    const depth = +(api.data.depth || 2);                                  // depth 2: four leaves per tree
    const fits = Object.fromEntries(ETAS.map((e) => [e, boostFit(xs, ys, xt, yt, e, STAGES, depth)]));
    let eta = ETAS.includes(+api.data.eta) ? +api.data.eta : 0.3, m = clamp(+(api.data.m || 0), 0, STAGES), playing = false, acc = 0;

    const cTop = chart({ width: 900, height: 330, x: { domain: [-4.2, 4.2], label: "" }, y: { domain: [0, 1.5], label: "peak height / substrate" }, margin: { l: 80, b: 34, t: 14 } });
    const cRes = chart({ width: 900, height: 300, x: { domain: [-4.2, 4.2], label: "distance from the interface (nm)" }, y: { domain: [-0.7, 0.7], label: "residual y − F" }, margin: { l: 80, t: 14 } });
    const cErr = chart({ width: 800, height: 400, x: { domain: [0, STAGES], label: "stage m (number of small trees)" }, y: { domain: [0.003, 0.2], type: "log", label: "mean squared error", ticks: [0.003, 0.01, 0.03, 0.1], tickFormat: (v) => String(v) }, margin: { l: 90, t: 14 } });
    const out = readout();
    const GX = Array.from({ length: 421 }, (_, i) => -4.2 + 0.02 * i);

    function draw() {
      const f = fits[eta];
      const Fm = (x) => { let v = f.f0; for (let q = 0; q < m; q++) v += eta * evalTree(f.trees[q], x); return v; };
      cTop.clear();
      cTop.dots(te, { r: 3, fill: T.warn, opacity: 0.35 });
      cTop.dots(tr, { r: 3.5, fill: T.data, opacity: 0.6 });
      cTop.line(GX.map((x) => [x, Fm(x)]), { stroke: T.thread, width: 4 });
      cTop.note(`F${m === 0 ? "₀ = mean" : ` after ${m} ${m === 1 ? "tree" : "trees"}`}`, { corner: "bl", fill: T.thread, weight: 700 });
      cTop.ruleX(0, { stroke: T.faint });
      cTop.text(-0.15, 1.38, "substrate", { fill: CLS[0].color, size: 19, anchor: "end" });
      cTop.text(0.15, 1.38, "film, then amorphous layer", { fill: CLS[1].color, size: 19 });
      cRes.clear();
      cRes.ruleY(0, { stroke: T.faint, dash: null });
      cRes.dots(tr.map(([x, y]) => [x, y - Fm(x)]), { r: 3.5, fill: T.data, opacity: 0.6 });
      if (m < STAGES) {
        const t = f.trees[m];
        cRes.line(GX.map((x) => [x, evalTree(t, x)]), { stroke: T.pink, width: 4 });
        cRes.note(`next tree h${m + 1}: fitted to these residuals`, { corner: "tr", fill: T.pink, weight: 700 });
      }
      cErr.clear();
      for (const e of ETAS) if (e !== eta) cErr.line(fits[e].test.map((v, i) => [i, v]), { stroke: T.warn, width: 2, opacity: 0.25 });
      cErr.line(f.train.map((v, i) => [i, v]), { stroke: T.data, width: 3 });
      cErr.line(f.test.map((v, i) => [i, v]), { stroke: T.warn, width: 4 });
      cErr.ruleX(f.best, { stroke: T.truth });
      cErr.dots([[f.best, f.test[f.best]]], { r: 8, fill: T.truth, stroke: "#fff", top: true });
      cErr.note(`lowest held-out error: stage ${f.best}`, { corner: "tr", fill: T.truth, weight: 700 });
      cErr.ruleX(m, { stroke: T.thread });
      cErr.dots([[m, f.train[m]], [m, f.test[m]]], { r: 7, fill: T.thread, stroke: "#fff", top: true });
      const late = m > f.best && f.test[m] > 1.03 * f.test[f.best];
      out.set(`Update: <b>F ← F + η·h</b> with η = <b>${eta}</b>; stage <b>${m}</b>.<br>` +
        `Training MSE <b>${f.train[m].toFixed(4)}</b> · held-out MSE <b class="${late ? "bad" : m === f.best ? "good" : ""}">${f.test[m].toFixed(4)}</b>` +
        ` (lowest: ${f.test[f.best].toFixed(4)} at stage ${f.best}).` + (late ? `<br><span class="warn">Past the best stage: the trees now fit the noise of image B.</span>` : ""));
    }
    const sM = slider({ label: "stage m", min: 0, max: STAGES, value: m, step: 1, digits: 0, format: (v) => String(Math.round(v)), onInput: (v) => { m = Math.round(v); draw(); } });
    const tE = toggle({ label: "learning rate η", options: ETAS.map((e) => ({ value: e, label: String(e) })), value: eta, onChange: (v) => { eta = v; draw(); } });
    const play = button("▶ play", () => { playing = !playing; play.innerHTML = playing ? "❚❚ pause" : "▶ play"; if (playing && m >= STAGES) sM.set(0); }, "primary");
    api.animate((dt) => {
      if (!playing) return;
      acc += dt * (m < 10 ? 3 : m < 40 ? 12 : 40);
      if (acc >= 1) { const k = Math.floor(acc); acc -= k; sM.set(Math.min(STAGES, m + k)); if (m >= STAGES) { playing = false; play.innerHTML = "▶ play"; } }
    });
    el.append(row(
      card(legend([[T.data, "candidates of image B (training)"], [T.warn, "image D (held out)"], [T.thread, "model F"]]), cTop.el, cRes.el),
      h("div", { class: "xp-ens-side", style: { width: "820px" } }, controls(tE), controls(sM, play),
        card(legend([[T.data, "training MSE (image B)"], [T.warn, "held-out MSE (image D)"]]), cErr.el), out)), src(SRC));
    el.__summary = () => Object.fromEntries(ETAS.map((e) => [e, { best: fits[e].best, testBest: fits[e].test[fits[e].best], test200: fits[e].test[STAGES], train200: fits[e].train[STAGES], test0: fits[e].test[0] }]));
    draw();
  }).catch((e) => { w.textContent = `data not loaded: ${e.message}`; });
});
