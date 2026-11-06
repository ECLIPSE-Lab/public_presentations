// The red line of Week 3: six stations on the way from a spectrum-image datacube to a trusted
// phase map.
//   rail      : small progress rail, added to every <section data-station="k">
//   redline   : large SVG of the six stations (outline, act dividers, summary)
//               data-focus="k" highlights station k; data-mode="summary" shows the results.
import { register, h, s, T } from "./core.js";

export const STATIONS = [
  { key: "loss", label: "Loss", q: "What does \"best fit\" mean?",
    sum: ["loss = noise model", "outliers: Huber, not MSE"] },
  { key: "descend", label: "Descend", q: "How do we find the minimum?",
    sum: ["w ← w − η∇L", "scale features first"] },
  { key: "step", label: "Step", q: "Which step to take?",
    sum: ["SGD: cheap, noisy", "momentum, Adam: fewer zigzags"] },
  { key: "general", label: "Generalise", q: "Why does the best fit predict badly?",
    sum: ["error = bias² + variance", "train error always falls"] },
  { key: "regular", label: "Regularise", q: "How do we tame the variance?",
    sum: ["ridge: shrink", "lasso: select"] },
  { key: "validate", label: "Validate", q: "Is the score honest?",
    sum: ["split like you deploy", "fit everything inside the fold"] },
];

// ---------------------------------------------------------------------------------------------
// Icons, drawn in a 100 x 100 box centred on (0, 0)
// ---------------------------------------------------------------------------------------------
function icon(key, col) {
  const st = { stroke: col, "stroke-width": 5, fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round" };
  const g = s("g");
  if (key === "loss") {
    g.append(s("path", { d: "M-38 -30 Q0 70 38 -30", ...st }), s("circle", { cx: 0, cy: 20, r: 7, fill: col }),
      s("path", { d: "M-40 36 H40", ...st, "stroke-width": 3, opacity: 0.5 }));
  } else if (key === "descend") {
    for (const [rx, ry] of [[40, 22], [26, 14], [12, 6]]) g.append(s("ellipse", { rx, ry, ...st, "stroke-width": 3, opacity: 0.5 }));
    g.append(s("path", { d: "M-34 -30 L-14 -8 L-24 4 L-4 2 L-8 0", ...st }), s("circle", { cx: 0, cy: 0, r: 5, fill: col }));
  } else if (key === "step") {
    g.append(s("path", { d: "M-38 26 Q-10 -40 30 -6", ...st }), s("path", { d: "M22 -16 L32 -5 L18 0", ...st }));
    [[-30, 14], [-20, -6], [-6, -18]].forEach(([x, y], i) => g.append(s("circle", { cx: x, cy: y, r: 4 + i, fill: col, opacity: 0.4 + 0.2 * i })));
  } else if (key === "general") {
    g.append(s("path", { d: "M-38 -30 Q-20 30 38 30", ...st, "stroke-width": 4, opacity: 0.55 }),
      s("path", { d: "M-38 -26 Q0 50 38 -26", ...st }), s("circle", { cx: 0, cy: 12, r: 6, fill: col }));
  } else if (key === "regular") {
    g.append(s("circle", { cx: -14, cy: 0, r: 22, ...st, "stroke-width": 4 }), s("path", { d: "M18 -24 L40 0 L18 24 L-4 0 Z", ...st, "stroke-width": 4 }));
  } else if (key === "validate") {
    for (let i = 0; i < 5; i++) g.append(s("rect", { x: -40 + 16.5 * i, y: -14, width: 13, height: 28, rx: 2, fill: col, opacity: i === 3 ? 1 : 0.32 }));
    g.append(s("path", { d: "M10 22 L16 30 L30 14", ...st, "stroke-width": 4 }));
  }
  return g;
}

// ---------------------------------------------------------------------------------------------
// Rail: the six stations as a thin red line at the top right of every slide
// ---------------------------------------------------------------------------------------------
function rail(k) {
  const W = 760, x0 = 40, dx = (W - 2 * x0) / 5, y = 15;
  const svg = s("svg", { class: "xp-rail", viewBox: `0 0 ${W} 46`, width: W, height: 46 });
  svg.append(s("line", { x1: x0, x2: x0 + 5 * dx, y1: y, y2: y, stroke: T.faint, "stroke-width": 3 }));
  if (k > 1) svg.append(s("line", { x1: x0, x2: x0 + (k - 1) * dx, y1: y, y2: y, stroke: T.thread, "stroke-width": 4 }));
  STATIONS.forEach((st, i) => {
    const x = x0 + i * dx, n = i + 1, cur = n === k, done = n < k;
    svg.append(s("circle", { cx: x, cy: y, r: cur ? 10 : 6.5, fill: cur || done ? T.thread : "#1e293b",
      stroke: cur ? "#fff" : done ? T.thread : T.faint, "stroke-width": cur ? 3 : 2 }));
    svg.append(s("text", { x, y: 42, "text-anchor": "middle", "font-size": 17, "font-family": "Inter, sans-serif",
      fill: cur ? "#fff" : done ? "#fda4af" : T.muted, "font-weight": cur ? 700 : 400 }, st.label));
  });
  return svg;
}

function addRails() {
  for (const sec of document.querySelectorAll("section[data-station]")) {
    if (sec.querySelector(":scope > .xp-rail")) continue;
    const k = +sec.dataset.station;
    if (k >= 1 && k <= 6) sec.prepend(rail(k));
  }
}
addRails();

// ---------------------------------------------------------------------------------------------
// The large red line
// ---------------------------------------------------------------------------------------------
register("redline", (el, api) => {
  const focus = +(api.data.focus || 0), summary = api.data.mode === "summary";
  const W = 1760, H = summary ? 660 : 600, x0 = 150, dx = (W - 2 * x0) / 5;
  const pos = STATIONS.map((_, i) => [x0 + i * dx, 310 + (i % 2 ? 40 : -40)]);
  const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "xp-chart" });
  svg.append(s("defs", {},
    s("filter", { id: "xpglow", x: "-50%", y: "-50%", width: "200%", height: "200%" },
      s("feGaussianBlur", { stdDeviation: 6, result: "b" }),
      s("feMerge", {}, s("feMergeNode", { in: "b" }), s("feMergeNode", { in: "SourceGraphic" }))),
    s("marker", { id: "xparrow", viewBox: "0 0 10 10", refX: 6, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" },
      s("path", { d: "M0 0 L10 5 L0 10 Z", fill: T.thread }))));

  // smooth path through the station centres
  let d = `M${pos[0][0] - 110},${pos[0][1] + 40} C${pos[0][0] - 60},${pos[0][1] + 10} ${pos[0][0] - 50},${pos[0][1]} ${pos[0][0]},${pos[0][1]}`;
  for (let i = 1; i < pos.length; i++) {
    const [ax, ay] = pos[i - 1], [bx, by] = pos[i];
    d += ` C${ax + dx * 0.45},${ay} ${bx - dx * 0.45},${by} ${bx},${by}`;
  }
  const [lx, ly] = pos[5];
  d += ` C${lx + 50},${ly} ${lx + 60},${ly - 10} ${lx + 110},${ly - 40}`;
  const base = s("path", { d, fill: "none", stroke: T.faint, "stroke-width": 5, opacity: 0.6 });
  const thread = s("path", { d, fill: "none", stroke: T.thread, "stroke-width": 7, "stroke-linecap": "round", filter: "url(#xpglow)" });
  // the loop: the validation score sends you back to the choice of model and of lambda
  const [p4x, p4y] = pos[4], [p6x, p6y] = pos[5];
  const loop = s("path", { d: `M${p6x - 10},${p6y - 72} C${p6x - 60},${p6y - 300} ${p4x + 60},${p4y - 300} ${p4x + 10},${p4y - 72}`,
    fill: "none", stroke: T.thread, "stroke-width": 3, "stroke-dasharray": "10 9", "marker-end": "url(#xparrow)", opacity: 0.8 });
  const loopLabel = s("text", { x: W - 20, y: 52, "text-anchor": "end", fill: "#fda4af", "font-size": 24, "font-style": "italic" },
    "the validation score picks λ and the model; the test set is touched once");
  svg.append(base, thread, loop, loopLabel);

  const nodes = STATIONS.map((st, i) => {
    const [x, y] = pos[i], n = i + 1, on = !focus || focus === n, cur = focus === n;
    const g = s("g", { transform: `translate(${x} ${y})`, opacity: on ? 1 : 0.38 });
    g.append(s("circle", { r: cur ? 74 : 62, fill: "#0f172a", stroke: on ? T.thread : T.faint, "stroke-width": cur ? 6 : 4 }));
    g.append(icon(st.key, on ? "#f8fafc" : T.muted));
    g.append(s("text", { y: (cur ? 74 : 62) + 44, "text-anchor": "middle", fill: on ? "#fff" : T.muted,
      "font-size": cur ? 38 : 32, "font-weight": 700, "font-family": "Outfit, sans-serif" }, `${n} · ${st.label}`));
    const lines = summary ? st.sum : focus ? [] : wrap(st.q, 22);
    lines.forEach((t, j) => g.append(s("text", { class: "xp-rl-sub", y: (cur ? 74 : 62) + 84 + j * 32, "text-anchor": "middle",
      fill: summary ? "#fda4af" : cur ? "#fde68a" : T.muted, "font-size": summary ? 23 : cur ? 28 : 22, "font-weight": summary ? 600 : 400 }, t)));
    svg.append(g);
    return g;
  });

  const electron = s("circle", { r: 11, fill: "#fff", stroke: T.thread, "stroke-width": 4, filter: "url(#xpglow)" });
  svg.append(electron);
  el.append(svg);

  // Draw the thread up to the focus station, then let a marker run along it.
  let L = 0, stops = [], t0 = 0;
  const measure = () => {
    L = thread.getTotalLength();
    stops = pos.map(([px]) => { let lo = 0, hi = L; for (let k = 0; k < 30; k++) { const m = (lo + hi) / 2; if (thread.getPointAtLength(m).x < px) lo = m; else hi = m; } return lo; });
  };
  const target = () => (focus ? stops[focus - 1] : L);
  api.onShow(() => {
    if (!L) measure();
    t0 = performance.now() / 1000;
    thread.setAttribute("stroke-dasharray", `${L} ${L}`);
    nodes.forEach((g) => g.querySelectorAll(".xp-rl-sub").forEach((t) => { t.style.transition = "none"; t.style.opacity = summary ? 0 : 1; }));
  });
  api.animate((dt, t) => {
    if (!L) return;
    const age = t - t0, end = target(), draw = Math.min(end, end * Math.min(1, age / 2.2));
    thread.setAttribute("stroke-dashoffset", L - draw);
    // the electron follows the drawing front, then keeps cycling along the drawn part
    const run = age < 2.2 ? draw : (((age - 2.2) * 260) % (end + 1));
    const p = thread.getPointAtLength(Math.max(0, run));
    electron.setAttribute("cx", p.x); electron.setAttribute("cy", p.y);
    if (summary) nodes.forEach((g, i) => { if (age > 0.5 + 0.45 * i) g.querySelectorAll(".xp-rl-sub").forEach((x) => { x.style.transition = "opacity .5s"; x.style.opacity = 1; }); });
  });
});

function wrap(text, n) {
  const out = [];
  let line = "";
  for (const w of text.split(" ")) {
    if ((line + " " + w).trim().length > n && line) { out.push(line); line = w; } else line = (line + " " + w).trim();
  }
  if (line) out.push(line);
  return out;
}

