// Shared parts of the Week 5 explainers: the column table of four real HAADF-STEM images
// (xp/data/columns.json), class colours, an image with clickable column markers, and a small CART.
import { T, h, s } from "./core.js";
import { loadData } from "./data.js";

export const CLS = [
  { key: "substrate", label: "substrate column", color: "#60a5fa" },
  { key: "film", label: "film column", color: "#fb923c" },
  { key: "none", label: "no column (noise maximum)", color: "#c084fc" },
];
export const SRC = "Data: HAADF-STEM of an oxide film on LaAlO₃, four images of one specimen; STEM-Bank-26 (Hollyer &amp; Stach 2026), CC BY 4.0.";
export const IMG_COL = ["#22d3ee", "#a3e635", "#fbbf24", "#f472b6"];
export const src = (html) => h("div", { class: "xp-src", html });
export const waiting = (el) => { const w = h("div", { class: "xp-caption" }, "loading data …"); el.append(w); return w; };

if (!document.getElementById("xp-w5-style")) {
  document.head.append(h("style", { id: "xp-w5-style" }, `
.xp-src { font-size: 17px; color: #64748b; margin-top: 4px; }
.xp-side { display: flex; flex-direction: column; gap: 14px; }
.xp-side .xp-readout { margin-top: 0; line-height: 1.5; }
.xp-big2 { font-family: Outfit, sans-serif; font-size: 64px; font-weight: 800; line-height: 1; }
.xp-colimg { position: relative; line-height: 0; }
.xp-colimg svg { position: absolute; left: 0; top: 0; }
.xp-colimg svg circle.pick { cursor: pointer; }
.xp-tbl { border-collapse: collapse; font-size: 21px; font-variant-numeric: tabular-nums; }
.xp-tbl td, .xp-tbl th { padding: 3px 12px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.07); }
.xp-tbl th { color: #94a3b8; font-weight: 500; }
.xp-tbl td:first-child, .xp-tbl th:first-child { text-align: left; }
`));
}

// The column table. Returns {names, units, about, X: [N][F], y, g, pos: [N][2] (y, x in nm), images, nb, lab (row indices
// with a label), F}.
let colsP = null;
export function loadCols() {
  if (!colsP) colsP = loadData("columns").then((d) => {
    const N = d.y.length, F = d.names.length;
    const X = Array.from({ length: N }, (_, i) => Array.from(d.X.subarray(i * F, (i + 1) * F)));
    const pos = Array.from({ length: N }, (_, i) => [d.pos[2 * i], d.pos[2 * i + 1]]);
    const nb = Array.from({ length: N }, (_, i) => Array.from(d.nb.subarray(4 * i, 4 * i + 4)));
    const y = Array.from(d.y), g = Array.from(d.g);
    const lab = []; y.forEach((v, i) => { if (v >= 0) lab.push(i); });
    return { ...d, N, F, X, pos, nb, y, g, lab, f: (name) => d.names.indexOf(name) };
  });
  return colsP;
}

// One of the four images with an SVG overlay in nm coordinates.
//   const im = colImage(d, 1, 620); im.marks(rows, (i) => ({fill, r, stroke, pick: true}), onPick)
export function colImage(d, k, size = 620) {
  const I = d.images[k], n = I.img.shape[0], [Hnm, Wnm] = I.size_nm;
  const cv = h("canvas", { width: n, height: n, class: "xp-image" });
  cv.style.width = `${size}px`; cv.style.height = `${size}px`; cv.style.imageRendering = "auto";
  const ctx = cv.getContext("2d"), im = ctx.createImageData(n, n);
  for (let i = 0; i < n * n; i++) { const v = I.img[i]; im.data[4 * i] = v; im.data[4 * i + 1] = v; im.data[4 * i + 2] = v; im.data[4 * i + 3] = 255; }
  ctx.putImageData(im, 0, 0);
  const svg = s("svg", { width: size, height: size, viewBox: `0 0 ${size} ${size}` });
  const gz = s("g"), gm = s("g"), gt = s("g");
  svg.append(gz, gm, gt);
  const el = h("div", { class: "xp-colimg", style: { width: `${size}px`, height: `${size}px` } }, cv, svg);
  el.sx = (xnm) => xnm / Wnm * size; el.sy = (ynm) => ynm / Hnm * size; el.size = size; el.top = gt; el.k = k;
  el.zones = (on = true) => {
    gz.replaceChildren();
    if (!on) return;
    const z = I.zones;
    for (const x of [(z[0] + z[1]) / 2, (z[2] + z[3]) / 2])
      gz.append(s("line", { x1: el.sx(x), x2: el.sx(x), y1: 0, y2: size, stroke: "#fff", "stroke-width": 2, "stroke-dasharray": "8 8", opacity: 0.55 }));
  };
  el.marks = (rows, style, onPick) => {
    gm.replaceChildren();
    for (const i of rows) {
      const st = style(i); if (!st) continue;
      const c = s("circle", { cx: el.sx(d.pos[i][1]), cy: el.sy(d.pos[i][0]), r: st.r || 5, fill: st.fill || "none", stroke: st.stroke || "none",
        "stroke-width": st.sw || 2, opacity: st.opacity ?? 0.95, class: onPick ? "pick" : null });
      if (onPick) c.addEventListener("click", () => onPick(i));
      gm.append(c);
    }
  };
  return el;
}

// ---------------------------------------------------------------------------------------------
// CART for classification (Gini), on rows of numbers. Small and direct: O(n log n) per feature and node.
// ---------------------------------------------------------------------------------------------
export const gini = (c) => { const n = c.reduce((a, b) => a + b, 0); return n ? 1 - c.reduce((a, b) => a + (b / n) ** 2, 0) : 0; };
export const entropy = (c) => { const n = c.reduce((a, b) => a + b, 0); return n ? -c.reduce((a, b) => a + (b ? b / n * Math.log2(b / n) : 0), 0) : 0; };

// Best threshold of feature j for the rows `idx`. Returns {thr, gain, curve: [[thr, gain]]}.
export function bestSplit(X, y, idx, j, K = 3, imp = gini, wantCurve = false) {
  const o = idx.slice().sort((a, b) => X[a][j] - X[b][j]), n = o.length;
  const tot = new Array(K).fill(0); o.forEach((i) => tot[y[i]]++);
  const L = new Array(K).fill(0), R = tot.slice(), I0 = imp(tot);
  let best = { thr: null, gain: 0 }; const curve = [];
  for (let q = 0; q < n - 1; q++) {
    const c = y[o[q]]; L[c]++; R[c]--;
    const a = X[o[q]][j], b = X[o[q + 1]][j];
    if (a === b) continue;
    const gain = I0 - (q + 1) / n * imp(L) - (n - q - 1) / n * imp(R), thr = (a + b) / 2;
    if (wantCurve) curve.push([thr, gain]);
    if (gain > best.gain) best = { thr, gain };
  }
  return { ...best, curve, parent: I0 };
}

export function cart(X, y, idx, { depth = 3, minLeaf = 1, K = 3, feats } = {}) {
  const F = feats || X[0].map((_, j) => j);
  const grow = (rows, d) => {
    const counts = new Array(K).fill(0); rows.forEach((i) => counts[y[i]]++);
    const node = { counts, n: rows.length, cls: counts.indexOf(Math.max(...counts)), depth: d };
    if (d >= depth || gini(counts) === 0 || rows.length < 2 * minLeaf) return node;
    let best = { gain: 1e-12 };
    for (const j of F) { const b = bestSplit(X, y, rows, j, K); if (b.thr !== null && b.gain > best.gain) best = { ...b, j }; }
    if (best.j === undefined) return node;
    const l = rows.filter((i) => X[i][best.j] <= best.thr), r = rows.filter((i) => X[i][best.j] > best.thr);
    if (l.length < minLeaf || r.length < minLeaf) return node;
    node.j = best.j; node.thr = best.thr; node.gain = best.gain;
    node.left = grow(l, d + 1); node.right = grow(r, d + 1);
    return node;
  };
  return grow(idx, 0);
}
export const leafOf = (t, x) => { while (t.left) t = x[t.j] <= t.thr ? t.left : t.right; return t; };
export const predictTree = (t, x) => leafOf(t, x).cls;
export const countLeaves = (t) => (t.left ? countLeaves(t.left) + countLeaves(t.right) : 1);
// Axis-aligned boxes of the leaves of a tree that splits on two features only: fx is the feature on the x axis,
// every other split is on the y axis. Nodes deeper than maxDepth count as leaves (a pruned view of a deep tree).
export function leafBoxes(t, box, fx, maxDepth = 99) {
  if (!t.left || t.depth >= maxDepth) return [{ box, node: t }];
  const [x0, x1, y0, y1] = box, onX = t.j === fx;
  const lb = onX ? [x0, Math.min(x1, t.thr), y0, y1] : [x0, x1, y0, Math.min(y1, t.thr)];
  const rb = onX ? [Math.max(x0, t.thr), x1, y0, y1] : [x0, x1, Math.max(y0, t.thr), y1];
  return [...leafBoxes(t.left, lb, fx, maxDepth), ...leafBoxes(t.right, rb, fx, maxDepth)];
}
// Prediction and leaf count of a tree cut at maxDepth.
export const predictAt = (t, x, maxDepth) => { while (t.left && t.depth < maxDepth) t = x[t.j] <= t.thr ? t.left : t.right; return t.cls; };
export const leavesAt = (t, maxDepth) => (t.left && t.depth < maxDepth ? leavesAt(t.left, maxDepth) + leavesAt(t.right, maxDepth) : 1);
export { T };
