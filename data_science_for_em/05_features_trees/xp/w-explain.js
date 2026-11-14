// Validate and Explain groups (Acts 5 and 6), on real data (four HAADF-STEM images of one oxide
// interface, STEM-Bank-26). All model results come from xp/prep/make_models.py (scikit-learn, shap).
//   validate-map : descriptor set × model × split. Where is the model right, and where wrong?
//                  Random 5-fold over the candidates against "leave one image out".
//                  data-set = raw | all | invariant | acsf, data-model = a model name, data-split = random | image
//   shap-click   : TreeSHAP values of a random forest (six invariant descriptors, trained on the images
//                  A, C, D) for the candidates of image B. Click a candidate: φ₀ + Σ φⱼ = f(x).
//                  data-cls = 0 | 1 | 2, data-pick = index of the candidate
import { register, T, fmt, clamp, h, s, row, col, card, controls, readout, toggle, legend, chart } from "./core.js";
import { loadData } from "./data.js";
import { CLS, SRC, src, waiting, loadCols, colImage, IMG_COL } from "./w5-common.js";

if (!document.getElementById("xp-exp-style")) {
  document.head.append(h("style", { id: "xp-exp-style" }, `
.xp-exp-side { width: 1000px; display: flex; flex-direction: column; gap: 12px; }
.xp-exp-side .xp-readout { margin-top: 0; line-height: 1.5; }
.xp-exp-side .xp-controls { margin: 0; }
.xp-exp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 14px; }
.xp-exp-grid .xp-imlabel { font-size: 19px; font-weight: 600; margin: 0 0 2px 2px; }
.xp-exp-two { display: flex; gap: 26px; align-items: flex-start; }
.xp-exp-acc { display: flex; gap: 34px; align-items: flex-end; }
.xp-exp-acc .lbl { font-size: 20px; color: #94a3b8; }
.xp-exp-acc .dim { opacity: 0.4; }
.xp-cm { border-collapse: collapse; font-size: 21px; font-variant-numeric: tabular-nums; }
.xp-cm td, .xp-cm th { padding: 7px 12px; text-align: center; }
.xp-cm th { color: #94a3b8; font-weight: 500; font-size: 19px; white-space: nowrap; }
.xp-cm td.c { min-width: 92px; border: 2px solid #0b0f19; border-radius: 8px; color: #fff; font-weight: 600; }
`));
}
const GOOD = "#34d399", BAD = "#f43f5e";
const SHORT = ["substrate", "film", "no column"];

// ---------------------------------------------------------------------------------------------
// validate-map
// ---------------------------------------------------------------------------------------------
register("validate-map", (el, api) => {
  const w = waiting(el);
  Promise.all([loadCols(), loadData("models")]).then(([d, M]) => {
    w.remove();
    const rows = Array.from(M.idx), y = rows.map((r) => d.y[r]), g = rows.map((r) => d.g[r]), n = rows.length;
    const SETS = [["raw", "raw (3)"], ["all", "all (11)"], ["invariant", "invariant (6)"], ["acsf", "ACSF (8)"]];
    const MODELS = M.models.filter((m) => m !== "majority");
    const MLAB = { logistic: "logistic", "tree (depth 3)": "tree, depth 3", "tree (full)": "full tree", "random forest": "random forest", boosting: "boosting" };
    let set = SETS.some(([k]) => k === api.data.set) ? api.data.set : "raw";
    let model = MODELS.includes(api.data.model) ? api.data.model : "random forest";
    let split = api.data.split === "image" ? "image" : "random";
    const byImg = [0, 1, 2, 3].map((k) => rows.filter((_, i) => g[i] === k));
    const at = new Map(rows.map((r, i) => [r, i]));

    const ims = [0, 1, 2, 3].map((k) => colImage(d, k, 335));
    const grid = h("div", { class: "xp-exp-grid" }, ims.map((im, k) => h("div", {},
      h("div", { class: "xp-imlabel", style: { color: IMG_COL[k] } }, `image ${d.images[k].name.replace(/\s+\(.*/, "")}`), im)));
    const accBox = h("div", { class: "xp-exp-acc" });
    const bars = chart({ width: 480, height: 300, x: { domain: [-0.5, 3.5], ticks: [0, 1, 2, 3], tickFormat: (v) => "ABCD"[v], label: "image", grid: false },
      y: { domain: [0, 1], label: "accuracy", tickFormat: (v) => `${Math.round(v * 100)} %` }, margin: { l: 86, t: 14 } });
    const cm = h("div"), out = readout();

    function draw() {
      const L = M.ladder[set][model], pred = split === "image" ? L.pred_image : L.pred_random;
      ims.forEach((im, k) => im.marks(byImg[k], (r) => { const i = at.get(r); return { fill: pred[i] === y[i] ? GOOD : BAD, r: pred[i] === y[i] ? 3 : 4.2, opacity: pred[i] === y[i] ? 0.75 : 1 }; }));
      const per = [0, 1, 2, 3].map((k) => { let ok = 0, tot = 0; for (let i = 0; i < n; i++) if (g[i] === k) { tot++; if (pred[i] === y[i]) ok++; } return ok / tot; });
      const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; for (let i = 0; i < n; i++) C[y[i]][pred[i]]++;
      const big = (v, lbl, on) => h("div", { class: on ? "" : "dim" },
        h("div", { class: "xp-big2", style: { color: on ? (lbl.startsWith("new") ? T.thread : T.truth) : T.muted, fontSize: on ? "76px" : "50px" } }, `${fmt(100 * v, 1)} %`),
        h("div", { class: "lbl" }, lbl));
      accBox.replaceChildren(big(L.random, "random 5-fold over the candidates", split === "random"), big(L.image, "new image (leave one image out)", split === "image"));
      bars.clear();
      bars.ruleY(M.ladder[set].majority.image, { stroke: T.muted });
      per.forEach((v, k) => { bars.bars([[k - 0.34, k + 0.34, v]], { fill: IMG_COL[k], opacity: 0.85 }); bars.text(k, v, `${Math.round(100 * v)}`, { anchor: "middle", dy: -8, size: 20, weight: 700 }); });
      const tot = C.map((r) => r.reduce((a, b) => a + b, 0));
      const t = h("table", { class: "xp-cm" },
        h("tr", {}, h("th", {}, "truth ↓ pred. →"), SHORT.map((c, k) => h("th", { style: { color: CLS[k].color } }, c))),
        C.map((r, a) => h("tr", {}, h("th", { style: { color: CLS[a].color, textAlign: "right" } }, SHORT[a]),
          r.map((v, b) => h("td", { class: "c", style: { background: a === b ? `rgba(52,211,153,${0.12 + 0.6 * v / tot[a]})` : `rgba(244,63,94,${v ? 0.12 + 0.75 * v / tot[a] : 0.04})` } }, String(v))))));
      cm.replaceChildren(t);
      const gap = L.random - L.image, wrong = n - C[0][0] - C[1][1] - C[2][2];
      out.set(`${wrong} of ${n} candidates wrong. Gap between the two splits: <b class="${gap > 0.1 ? "bad" : gap > 0.05 ? "warn" : "good"}">${fmt(100 * gap, 1)} points</b>. ` +
        `Balanced accuracy on a new image: ${fmt(100 * L.image_bal, 1)} %.`);
    }
    const tS = toggle({ label: "descriptors", options: SETS.map(([value, label]) => ({ value, label })), value: set, onChange: (v) => { set = v; draw(); } });
    const tM = toggle({ label: "model", options: MODELS.map((m) => ({ value: m, label: MLAB[m] || m })), value: model, onChange: (v) => { model = v; draw(); } });
    const tP = toggle({ label: "split", options: [{ value: "random", label: "random 5-fold", color: T.truth }, { value: "image", label: "new image", color: T.thread }], value: split, onChange: (v) => { split = v; draw(); } });
    el.append(row(
      card(grid, legend([[GOOD, "prediction right"], [BAD, "prediction wrong"]])),
      h("div", { class: "xp-exp-side" }, controls(tS), controls(tM), controls(tP), accBox,
        h("div", { class: "xp-exp-two" }, card(bars.el, legend([[T.muted, "always the majority class", "dash"]])), h("div", {}, cm)), out)), src(SRC));
    el.__summary = () => { const L = M.ladder[set][model]; return { set, model, split, random: L.random, image: L.image, per_image: L.per_image }; };
    draw();
  }).catch((e) => { w.textContent = `data not loaded: ${e.message}`; });
});

// ---------------------------------------------------------------------------------------------
// shap-click
// ---------------------------------------------------------------------------------------------
register("shap-click", (el, api) => {
  const w = waiting(el);
  Promise.all([loadCols(), loadData("models")]).then(([d, M]) => {
    w.remove();
    const S = M.shap, n = S.rows.length, F = S.names.length;
    // rows counts in the table of labelled candidates; idx maps that table to the rows of columns.json
    const rows = Array.from(S.rows, (r) => M.idx[r]), y = rows.map((r) => d.y[r]);
    const pred = rows.map((_, i) => { const p = [S.proba[3 * i], S.proba[3 * i + 1], S.proba[3 * i + 2]]; return p.indexOf(Math.max(...p)); });
    const units = S.names.map((nm) => d.units[d.f(nm)]), digits = S.names.map((nm) => (nm === "n_near" ? 0 : nm === "I_rel" || nm === "psi4" ? 2 : 3));
    let cls = clamp(+(api.data.cls ?? 1), 0, 2), sel = clamp(+(api.data.pick ?? 218), 0, n - 1);
    const at = new Map(rows.map((r, i) => [r, i]));

    const im = colImage(d, 1, 640);
    const ring = s("circle", { r: 14, fill: "none", stroke: "#fff", "stroke-width": 3.5 });
    im.top.append(ring);
    const W = 1000, H = 470, m = { l: 300, r: 30, t: 18, b: 56 }, NR = F + 2, rh = (H - m.t - m.b) / NR;
    const svg = s("svg", { class: "xp-chart", width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    const X = (p) => m.l + p * (W - m.l - m.r);
    const head = readout(), out = readout();

    function draw() {
      im.marks(rows, (r) => { const i = at.get(r), bad = pred[i] !== y[i]; return { fill: CLS[pred[i]].color, r: bad ? 5 : 4.2, stroke: bad ? "#fff" : "none", sw: 2, opacity: 0.9 }; },
        (r) => { sel = at.get(r); draw(); });
      const r = rows[sel];
      ring.setAttribute("cx", im.sx(d.pos[r][1])); ring.setAttribute("cy", im.sy(d.pos[r][0]));
      const phi = Array.from({ length: F }, (_, j) => S.values[(sel * F + j) * 3 + cls]), base = S.base[cls], fx = S.proba[3 * sel + cls];
      const order = phi.map((_, j) => j).sort((a, b) => Math.abs(phi[b]) - Math.abs(phi[a]));
      svg.replaceChildren();
      for (const p of [0, 0.25, 0.5, 0.75, 1]) {
        svg.append(s("line", { x1: X(p), x2: X(p), y1: m.t, y2: H - m.b, stroke: T.grid }),
          s("text", { x: X(p), y: H - m.b + 26, class: "xp-tick", "text-anchor": "middle" }, String(p)));
      }
      svg.append(s("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: T.axis, "stroke-width": 1.5 }),
        s("text", { x: (m.l + W - m.r) / 2, y: H - 6, class: "xp-axlabel", "text-anchor": "middle" }, `predicted probability of "${CLS[cls].label.replace(/ \(.*/, "")}"`));
      const label = (k, txt, bold) => svg.append(s("text", { x: m.l - 14, y: m.t + (k + 0.5) * rh + 7, "text-anchor": "end", fill: bold ? "#fff" : T.ink, "font-size": 21, "font-weight": bold ? 700 : 400 }, txt));
      const bar = (k, a, b, fill, txt) => {
        const x0 = X(Math.min(a, b)), x1 = X(Math.max(a, b)), yy = m.t + k * rh + 7;
        svg.append(s("rect", { x: x0, y: yy, width: Math.max(2, x1 - x0), height: rh - 14, rx: 4, fill }));
        const right = x1 + 84 < W - m.r;
        svg.append(s("text", { x: right ? x1 + 8 : x0 - 8, y: yy + (rh - 14) / 2 + 7, "text-anchor": right ? "start" : "end", fill: "#fff", "font-size": 20, "font-weight": 600, class: "xp-ctext" }, txt));
      };
      const link = (k, p) => svg.append(s("line", { x1: X(p), x2: X(p), y1: m.t + k * rh + 7, y2: m.t + (k + 1) * rh + 7, stroke: T.muted, "stroke-dasharray": "4 4" }));
      label(0, "φ₀ = mean prediction", true); bar(0, 0, base, T.faint, base.toFixed(3));
      let cur = base;
      order.forEach((j, q) => {
        link(q, cur);
        label(q + 1, `${S.names[j]} = ${S.x[sel * F + j].toFixed(digits[j])}${units[j] && units[j] !== "–" ? ` ${units[j]}` : ""}`);
        bar(q + 1, cur, cur + phi[j], phi[j] >= 0 ? T.thread : T.gauss, `${phi[j] >= 0 ? "+" : "−"}${Math.abs(phi[j]).toFixed(3)}`);
        cur += phi[j];
      });
      link(F, cur);
      label(F + 1, "f(x) = prediction", true); bar(F + 1, 0, fx, CLS[cls].color, fx.toFixed(3));
      const ok = pred[sel] === y[sel];
      head.set(`Candidate at x = ${d.pos[r][1].toFixed(2)} nm, y = ${d.pos[r][0].toFixed(2)} nm. Truth: <b style="color:${CLS[y[sel]].color}">${SHORT[y[sel]]}</b>. ` +
        `Model: <b style="color:${CLS[pred[sel]].color}">${SHORT[pred[sel]]}</b> <span class="${ok ? "good" : "bad"}">(${ok ? "right" : "wrong"})</span>.`);
      const sum = phi.reduce((a, b) => a + b, 0);
      out.set(`Check: φ₀ + Σφⱼ = ${base.toFixed(3)} ${sum >= 0 ? "+" : "−"} ${Math.abs(sum).toFixed(3)} = <b>${(base + sum).toFixed(3)}</b> = f(x) = <b>${fx.toFixed(3)}</b>. ` +
        `<span style="color:${T.thread}">Red</span> pushes the probability up, <span style="color:${T.gauss}">blue</span> pushes it down.`);
    }
    const tC = toggle({ label: "explain the class", options: CLS.map((c, k) => ({ value: k, label: SHORT[k], color: c.color })), value: cls, onChange: (v) => { cls = v; draw(); } });
    el.append(row(
      card(im, legend(CLS.map((c, k) => [c.color, SHORT[k]])), h("div", { class: "xp-caption", html: "Image B, not in the training data. Colour = predicted class. White ring = wrong. Click a candidate." })),
      h("div", { class: "xp-exp-side", style: { width: "1040px" } }, controls(tC), head, card(svg), out)), src(SRC));
    el.__summary = () => ({ sel, cls, base: S.base[cls], fx: S.proba[3 * sel + cls], truth: y[sel], pred: pred[sel], pos: d.pos[rows[sel]],
      phi: Object.fromEntries(S.names.map((nm, j) => [nm, [S.x[sel * F + j], S.values[(sel * F + j) * 3 + cls]]])), acc: pred.filter((p, i) => p === y[i]).length / n });
    draw();
  }).catch((e) => { w.textContent = `data not loaded: ${e.message}`; });
});
