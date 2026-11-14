// Tree group (Act 3), on the column table of four real HAADF-STEM images (STEM-Bank-26):
//   split     : one feature, one threshold. Class histograms, the impurity decrease Δ for every
//               threshold, and the counts of parent, left and right node.
//   grow-tree : CART on two descriptors. Training on images A, C, D; test on image B. The depth sets
//               the partition; a bootstrap sample of the training rows changes it.
import { register, T, fmt, rng, h, s, row, col, card, controls, readout, slider, toggle, button, legend, chart } from "./core.js";
import { CLS, SRC, src, waiting, loadCols, gini, entropy, bestSplit, cart, leafBoxes, predictAt, leavesAt } from "./w5-common.js";

if (!document.getElementById("xp-tree-style")) {
  document.head.append(h("style", { id: "xp-tree-style" }, `
.xp-node { display: grid; grid-template-columns: 150px 1fr 150px; align-items: center; gap: 12px; font-size: 22px; margin: 4px 0; }
.xp-node .bar { display: flex; height: 34px; border-radius: 6px; overflow: hidden; background: rgba(255,255,255,0.05); }
.xp-node .bar span { display: block; height: 100%; }
.xp-node .imp { text-align: right; font-variant-numeric: tabular-nums; color: #fff; font-weight: 600; }
.xp-eq { font-size: 23px; line-height: 1.6; font-variant-numeric: tabular-nums; }
.xp-eq .d { color: #f43f5e; font-weight: 800; font-size: 34px; }
`));
}
const SHORT = ["substrate", "film", "no column"];

// ---------------------------------------------------------------------------------------------
// split
// ---------------------------------------------------------------------------------------------
const FEATS = [
  { name: "I_rel", label: "I_rel", axis: "I_rel = peak height / median of the 4 nearest", lo: 0.3, hi: 2.2, k: 1 },
  { name: "d1", label: "d1", axis: "d1: distance to the nearest candidate (nm)", lo: 0.15, hi: 0.45, k: 1 },
  { name: "d4", label: "d4", axis: "d4: mean distance to the 4 nearest (nm)", lo: 0.22, hi: 0.5, k: 1 },
  { name: "d_spread", label: "d_spread", axis: "d_spread: std / mean of the 4 distances", lo: 0, hi: 0.32, k: 1 },
  { name: "psi4", label: "ψ₄", axis: "|ψ₄|: four-fold order of the 4 nearest", lo: 0, hi: 1, k: 1 },
  { name: "amp", label: "amp (raw)", axis: "amp: peak height (10⁶ counts)", lo: 0, hi: 2.6, k: 1e-6 },
];

register("split", (el, api) => {
  const w = waiting(el);
  loadCols().then((d) => {
    w.remove();
    const rows = d.lab, y = d.y, N = rows.length, NB = 56;
    let F = FEATS.find((f) => f.name === api.data.feature) || FEATS[2], impName = api.data.imp || "gini", X = null, res = null, thr = 0;
    const hx = { domain: [0, 1], label: "" };
    const hist = chart({ width: 1040, height: 340, x: hx, y: { domain: [0, 1], label: "candidates" }, margin: { b: 62, t: 14 } });
    const dch = chart({ width: 1040, height: 290, x: { domain: [0, 1], label: "threshold" }, y: { domain: [0, 0.4], label: "impurity decrease Δ" }, margin: { b: 58, t: 14 } });
    const nodes = h("div", {}), eq = h("div", { class: "xp-eq" }), out = readout("");
    const imp = () => (impName === "gini" ? gini : entropy);
    const node = (name, c) => {
      const n = c.reduce((a, b) => a + b, 0);
      return h("div", { class: "xp-node" }, h("div", { html: `${name}<br><span style="color:#94a3b8;font-size:19px">N = ${n}</span>` }),
        h("div", { class: "bar", style: { width: `${Math.max(2, 100 * n / N)}%` } }, c.map((v, k) => h("span", { style: { width: `${100 * v / Math.max(1, n)}%`, background: CLS[k].color } }))),
        h("div", { class: "imp" }, `I = ${fmt(imp()(c), 3)}`));
    };
    const setFeature = () => {
      const j = d.f(F.name);
      X = d.X.map((r) => [r[j] * F.k]);
      res = bestSplit(X, y, rows, 0, 3, imp(), true);
      // histograms
      const H = [0, 1, 2].map(() => new Array(NB).fill(0)), bw = (F.hi - F.lo) / NB;
      rows.forEach((i) => { const b = Math.floor((X[i][0] - F.lo) / bw); if (b >= 0 && b < NB) H[y[i]][b]++; });
      hist.axisLabel = F.axis; hist.H = H; hist.bw = bw;
      dch.ymax = Math.max(0.05, res.gain * 1.25);
    };
    const draw = () => {
      const H = hist.H, bw = hist.bw, ymax = Math.max(...H.flat()) * 1.08;
      hx.label = F.axis; hist.setDomain([F.lo, F.hi], [0, ymax]); hist.clear();
      H.forEach((hh, k) => hist.bars(hh.map((v, b) => [F.lo + b * bw, F.lo + (b + 1) * bw, v]), { fill: CLS[k].color, opacity: 0.55 }));
      hist.ruleX(thr, { stroke: T.thread, width: 4, dash: null });
      hist.text(thr, ymax, "≤ left", { anchor: "end", dx: -10, dy: 26, fill: "#fff", size: 20 }); hist.text(thr, ymax, "right >", { dx: 10, dy: 26, fill: "#fff", size: 20 });
      dch.setDomain([F.lo, F.hi], [0, dch.ymax]); dch.clear();
      dch.line(res.curve.filter((p) => p[0] >= F.lo && p[0] <= F.hi), { stroke: T.data, width: 3 });
      dch.ruleX(res.thr, { stroke: T.truth, width: 2 });
      dch.ruleX(thr, { stroke: T.thread, width: 4, dash: null });
      const L = [0, 0, 0], R = [0, 0, 0], P = [0, 0, 0];
      rows.forEach((i) => { P[y[i]]++; (X[i][0] <= thr ? L : R)[y[i]]++; });
      const nl = L[0] + L[1] + L[2], nr = N - nl, I = imp(), delta = I(P) - nl / N * I(L) - nr / N * I(R);
      dch.dots([[thr, delta]], { r: 8, fill: T.thread, stroke: "#fff" });
      dch.note(`best: ${F.label} ≤ ${fmt(res.thr, 3)}, Δ = ${fmt(res.gain, 3)}`, { corner: "tr", fill: T.truth, size: 21 });
      nodes.replaceChildren(node("parent", P), node(`left: ≤ ${fmt(thr, 3)}`, L), node(`right: > ${fmt(thr, 3)}`, R));
      eq.innerHTML = `Δ = I(parent) − (N<sub>L</sub>/N) · I(L) − (N<sub>R</sub>/N) · I(R)<br>` +
        `&nbsp;&nbsp; = ${fmt(I(P), 3)} − (${nl}/${N}) · ${fmt(I(L), 3)} − (${nr}/${N}) · ${fmt(I(R), 3)}<br>` +
        `&nbsp;&nbsp; = <span class="d">${fmt(delta, 3)}</span>`;
      const maj = (c) => SHORT[c.indexOf(Math.max(...c))];
      out.set(`Left says “<b>${maj(L)}</b>”, right says “<b>${maj(R)}</b>”. ` + (Math.abs(delta - res.gain) < 1e-6 ? `<span class="good">This is the best threshold for ${F.label}.</span>` : `The best threshold gives Δ = ${fmt(res.gain, 3)}.`));
    };
    const sl = slider({ label: "threshold", min: 0, max: 1, value: 0.5, digits: 3, format: (p) => fmt(F.lo + p * (F.hi - F.lo), 3), onInput: (p) => { thr = F.lo + p * (F.hi - F.lo); draw(); } });
    const setThr = (t) => { thr = t; sl.set((t - F.lo) / (F.hi - F.lo), false); draw(); };
    const tf = toggle({ label: "feature", options: FEATS.map((f) => ({ value: f.name, label: f.label })), value: F.name,
      onChange: (v) => { F = FEATS.find((f) => f.name === v); setFeature(); setThr((F.lo + F.hi) / 2); } });
    const ti = toggle({ options: [{ value: "gini", label: "Gini" }, { value: "entropy", label: "entropy" }], value: impName,
      onChange: (v) => { impName = v; setFeature(); draw(); } });
    el.append(row(
      col(controls(tf), card(hist.el, dch.el)),
      h("div", { class: "xp-side", style: { width: "690px" } }, controls(ti, button("best split", () => setThr(res.thr), "primary")), sl,
        card(legend(CLS.map((c, k) => [c.color, SHORT[k]])), nodes), card(eq), out)),
      src(SRC + ` ${N} labelled candidates.`));
    setFeature(); setThr(api.data.best === "1" ? res.thr : (F.lo + F.hi) / 2);
  });
});

// ---------------------------------------------------------------------------------------------
// grow-tree
// ---------------------------------------------------------------------------------------------
register("grow-tree", (el, api) => {
  const w = waiting(el);
  loadCols().then((d) => {
    w.remove();
    const fx = d.f("d4"), fy = d.f("I_rel"), DMAX = 12, BOX = [0.24, 0.5, 0.3, 2.2];
    const tr0 = d.lab.filter((i) => d.g[i] !== 1), te = d.lab.filter((i) => d.g[i] === 1);
    const R = rng(7);
    let tr = tr0, tree = null, acc = null, boot = 0;
    const sc = chart({ width: 800, height: 690, x: { domain: [BOX[0], BOX[1]], label: "d4: mean distance to the 4 nearest (nm)" },
      y: { domain: [BOX[2], BOX[3]], label: "I_rel: peak height / median of the 4 nearest" }, margin: { t: 14 } });
    const ac = chart({ width: 800, height: 380, x: { domain: [1, DMAX], label: "depth of the tree", ticks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] },
      y: { domain: [0.5, 1], label: "accuracy" }, title: "Accuracy against depth" });
    const out = readout("");
    const fit = () => {
      tree = cart(d.X, d.y, tr, { depth: DMAX, feats: [fx, fy] });
      acc = [];
      for (let k = 1; k <= DMAX; k++) {
        const a = (rows) => rows.reduce((n, i) => n + (predictAt(tree, d.X[i], k) === d.y[i]), 0) / rows.length;
        acc.push([k, a(tr), a(te), leavesAt(tree, k)]);
      }
    };
    const draw = () => {
      const k = Math.round(sd.value);
      sc.clear();
      for (const { box, node } of leafBoxes(tree, BOX, fx, k))
        sc.rect(box[0], box[2], box[1], box[3], { fill: CLS[node.cls].color, opacity: 0.2, stroke: "rgba(255,255,255,0.55)", "stroke-width": 1 });
      [0, 1, 2].forEach((c) => sc.dots(tr.filter((i) => d.y[i] === c).map((i) => [d.X[i][fx], d.X[i][fy]]), { r: 3.6, fill: CLS[c].color, opacity: 0.8 }));
      ac.clear();
      ac.line(acc.map((a) => [a[0], a[1]]), { stroke: T.data, width: 3.5 });
      ac.line(acc.map((a) => [a[0], a[2]]), { stroke: T.thread, width: 3.5 });
      ac.ruleX(k, { stroke: T.muted });
      ac.dots([[k, acc[k - 1][1]]], { r: 8, fill: T.data, stroke: "#fff" }); ac.dots([[k, acc[k - 1][2]]], { r: 8, fill: T.thread, stroke: "#fff" });
      const best = acc.reduce((b, a) => (a[2] > b[2] ? a : b));
      out.set(`Depth <b>${k}</b>: <b>${acc[k - 1][3]}</b> leaves. Training accuracy <b>${fmt(100 * acc[k - 1][1], 1)} %</b>, ` +
        `new image <span class="${acc[k - 1][2] >= best[2] - 0.005 ? "good" : "bad"}">${fmt(100 * acc[k - 1][2], 1)} %</span>.<br>` +
        `Best depth for the new image: ${best[0]} (${fmt(100 * best[2], 1)} %). ` +
        (boot ? `<span class="warn">Bootstrap sample ${boot}</span>: the same data source, another partition.` : `Training rows: all ${tr.length} candidates of images A, C, D.`));
    };
    const sd = slider({ label: "depth", min: 1, max: DMAX, value: +(api.data.depth || 3), step: 1, digits: 0, onInput: draw });
    const resample = () => { boot++; tr = tr0.map(() => tr0[Math.floor(R.u() * tr0.length)]); fit(); draw(); };
    const reset = () => { boot = 0; tr = tr0; fit(); draw(); };
    el.append(row(
      card(sc.el, legend(CLS.map((c, k) => [c.color, SHORT[k]]))),
      h("div", { class: "xp-side", style: { width: "840px" } }, controls(sd), controls(button("new bootstrap sample", resample, "primary"), button("all data", reset)),
        card(ac.el, legend([[T.data, "training images A, C, D"], [T.thread, "new image B"]])), out)),
      src(SRC + " Shaded boxes: the leaves of the tree, in the colour of their majority class."));
    fit();
    for (let b = 0; b < +(api.data.boot || 0); b++) { boot++; tr = tr0.map(() => tr0[Math.floor(R.u() * tr0.length)]); }
    if (boot) fit();
    draw();
  });
});
