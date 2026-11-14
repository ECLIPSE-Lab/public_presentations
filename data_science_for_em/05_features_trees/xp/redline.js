// The red line of Week 5: six stations on the way from an atomic-resolution image to a trusted,
// explained classifier of its atom columns.
//   rail      : small progress rail, added to every <section data-station="k">
//   redline   : large SVG of the six stations (outline, act dividers, summary)
//               data-focus="k" highlights station k; data-mode="summary" shows the results.
import { register, h, s, T } from "./core.js";

export const STATIONS = [
  { key: "objects", label: "Objects", q: "Where are the objects in the image?",
    sum: ["smooth, find maxima, fit", "precision ∝ 1/√dose"] },
  { key: "descr", label: "Descriptors", q: "Which numbers describe one object?",
    sum: ["one row per object", "build in the invariances"] },
  { key: "tree", label: "Tree", q: "How does a tree ask questions?",
    sum: ["greedy splits on impurity", "deep tree = high variance"] },
  { key: "ensemble", label: "Ensemble", q: "Why are many trees better than one?",
    sum: ["bagging: variance down", "boosting: bias down"] },
  { key: "validate", label: "Validate", q: "Does it work on a new image?",
    sum: ["split by image, not by column", "start with a baseline"] },
  { key: "explain", label: "Explain", q: "Which descriptors does the model use?",
    sum: ["permutation on held-out data", "SHAP explains the model"] },
];

// ---------------------------------------------------------------------------------------------
// Icons, drawn in a 100 x 100 box centred on (0, 0)
// ---------------------------------------------------------------------------------------------
function icon(key, col) {
  const st = { stroke: col, "stroke-width": 5, fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round" };
  const g = s("g");
  if (key === "objects") {
    for (const [x, y] of [[-26, -26], [0, -26], [26, -26], [-26, 0], [0, 0], [26, 0], [-26, 26], [0, 26], [26, 26]])
      g.append(s("circle", { cx: x, cy: y, r: x === 0 && y === 0 ? 9 : 6, fill: col, opacity: x === 0 && y === 0 ? 1 : 0.45 }));
    g.append(s("circle", { cx: 0, cy: 0, r: 17, ...st, "stroke-width": 3 }));
  } else if (key === "descr") {
    g.append(s("rect", { x: -38, y: -30, width: 76, height: 60, rx: 5, ...st, "stroke-width": 4 }));
    for (const y of [-10, 10]) g.append(s("path", { d: `M-38 ${y} H38`, ...st, "stroke-width": 3, opacity: 0.6 }));
    for (const x of [-13, 13]) g.append(s("path", { d: `M${x} -30 V30`, ...st, "stroke-width": 3, opacity: 0.6 }));
    g.append(s("rect", { x: -38, y: -10, width: 76, height: 20, fill: col, opacity: 0.3 }));
  } else if (key === "tree") {
    g.append(s("path", { d: "M0 -30 L-24 2 M0 -30 L24 2 M-24 2 L-36 30 M-24 2 L-12 30 M24 2 L12 30 M24 2 L36 30", ...st, "stroke-width": 4 }));
    for (const [x, y] of [[0, -30], [-24, 2], [24, 2]]) g.append(s("circle", { cx: x, cy: y, r: 7, fill: col }));
    for (const x of [-36, -12, 12, 36]) g.append(s("rect", { x: x - 5, y: 25, width: 10, height: 10, fill: col, opacity: 0.7 }));
  } else if (key === "ensemble") {
    for (const [x, o] of [[-28, 0.5], [0, 1], [28, 0.5]]) {
      g.append(s("path", { d: `M${x} -28 L${x - 10} -6 M${x} -28 L${x + 10} -6`, ...st, "stroke-width": 4, opacity: o }));
      g.append(s("circle", { cx: x, cy: -28, r: 5, fill: col, opacity: o }));
    }
    g.append(s("path", { d: "M-34 8 H34", ...st, "stroke-width": 3, opacity: 0.5 }), s("path", { d: "M0 12 V30 M-9 22 L0 31 L9 22", ...st, "stroke-width": 4 }));
  } else if (key === "validate") {
    for (let i = 0; i < 4; i++) g.append(s("rect", { x: -40 + 21 * i, y: -22, width: 17, height: 30, rx: 2, fill: col, opacity: i === 2 ? 1 : 0.32 }));
    g.append(s("path", { d: "M-8 22 L-1 31 L16 13", ...st, "stroke-width": 4 }));
  } else if (key === "explain") {
    for (const [y, w, o] of [[-24, 62, 1], [-8, 40, 0.75], [8, 24, 0.55], [24, 10, 0.4]])
      g.append(s("rect", { x: -34, y: y - 5, width: w, height: 10, rx: 2, fill: col, opacity: o }));
    g.append(s("path", { d: "M-38 -34 V34", ...st, "stroke-width": 3, opacity: 0.6 }));
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
  // the loop: the score on a new image sends you back to the choice of descriptors
  const [p4x, p4y] = pos[1], [p6x, p6y] = pos[4];
  const loop = s("path", { d: `M${p6x - 10},${p6y - 72} C${p6x - 160},${p6y - 250} ${p4x + 160},${p4y - 250} ${p4x + 10},${p4y - 72}`,
    fill: "none", stroke: T.thread, "stroke-width": 3, "stroke-dasharray": "10 9", "marker-end": "url(#xparrow)", opacity: 0.8 });
  const loopLabel = s("text", { x: W - 20, y: 52, "text-anchor": "end", fill: "#fda4af", "font-size": 24, "font-style": "italic" },
    "the score on a new image sends you back to the descriptors, not to a bigger model");
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

