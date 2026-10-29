// The red line of the lecture: six stations on the path of one electron, from the model we want to
// learn to the loss that trains it.
//   rail      : small progress rail, added to every <section data-station="k">
//   redline   : large SVG of the six stations (outline, act dividers, summary)
//               data-focus="k" highlights station k; data-mode="summary" shows the results.
//   chain     : the data-formation chain in physical detail; click a link to see what enters there.
import { register, h, s, T } from "./core.js";

export const STATIONS = [
  { key: "learn", label: "Learn", q: "What does it mean to learn from data?",
    sum: ["data + loss → θ", "priors cut variance"] },
  { key: "signal", label: "Signal", q: "What does an EM pixel measure?",
    sum: ["each pixel counts e⁻", "shape follows modality"] },
  { key: "sampling", label: "Sampling", q: "Is the pixel grid fine enough?",
    sum: ["Δx ≤ d/2, better d/4", "aliases never average out"] },
  { key: "counting", label: "Counting", q: "Why is every pixel noisy?",
    sum: ["Var = mean, SNR = √λ", "Var = g·x̄ + σr²"] },
  { key: "loss", label: "Loss", q: "Which loss does the physics dictate?",
    sum: ["loss = −log p(y | θ)", "Poisson: λ − y log λ"] },
  { key: "trust", label: "Trust", q: "Can I trust the data and the model?",
    sum: ["check before training", "metadata is data"] },
];

// ---------------------------------------------------------------------------------------------
// Icons, drawn in a 100 x 100 box centred on (0, 0)
// ---------------------------------------------------------------------------------------------
function icon(key, col) {
  const st = { stroke: col, "stroke-width": 5, fill: "none", "stroke-linecap": "round", "stroke-linejoin": "round" };
  const g = s("g");
  if (key === "learn") {
    g.append(s("rect", { x: -26, y: -20, width: 52, height: 40, rx: 8, ...st }),
      s("text", { x: 0, y: 9, "text-anchor": "middle", fill: col, "font-size": 24, "font-style": "italic", "font-family": "serif" }, "fθ"),
      s("path", { d: "M-44 0 H-30 M30 0 H44 M38 -6 L44 0 L38 6", ...st }));
  } else if (key === "signal") {
    g.append(s("path", { d: "M-14 -42 L-4 -6 M14 -42 L4 -6", ...st, "stroke-width": 4 }));
    for (const x of [-30, -10, 10, 30]) g.append(s("circle", { cx: x, cy: 14, r: 8, fill: col, opacity: x === 10 || x === -10 ? 1 : 0.55 }));
    for (const x of [-20, 0, 20]) g.append(s("circle", { cx: x, cy: 32, r: 6, fill: col, opacity: 0.45 }));
  } else if (key === "sampling") {
    for (let i = 0; i < 4; i++) g.append(s("line", { x1: -36 + 24 * i, x2: -36 + 24 * i, y1: -34, y2: 34, stroke: col, "stroke-width": 1.5, opacity: 0.5 }));
    g.append(s("path", { d: "M-40 0 C-30 -30 -20 -30 -10 0 S10 30 20 0 S36 -30 40 -10", ...st, "stroke-width": 4 }));
    for (const x of [-36, -12, 12, 36]) {
      const y = 24 * Math.sin((x + 40) / 60 * 2 * Math.PI) * -1;
      g.append(s("circle", { cx: x, cy: y, r: 5.5, fill: col }));
    }
  } else if (key === "counting") {
    g.append(s("rect", { x: -40, y: 22, width: 80, height: 12, rx: 3, fill: col, opacity: 0.85 }));
    [[-28, -30], [-6, -14], [18, -38], [30, -6], [4, 6], [-20, 2]].forEach(([x, y]) => g.append(s("circle", { cx: x, cy: y, r: 4.5, fill: col })));
  } else if (key === "loss") {
    g.append(s("path", { d: "M-38 -34 Q0 64 38 -34", ...st }), s("circle", { cx: 0, cy: 15, r: 7, fill: col }),
      s("path", { d: "M-44 34 H44", ...st, "stroke-width": 2, opacity: 0.5 }));
  } else if (key === "trust") {
    g.append(s("path", { d: "M0 -40 L32 -28 V2 C32 22 16 34 0 42 C-16 34 -32 22 -32 2 V-28 Z", ...st }),
      s("path", { d: "M-14 2 L-3 13 L16 -10", ...st }));
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
  // the loop: the loss trains the model
  const [p1x, p1y] = pos[0], [p5x, p5y] = pos[4];
  const loop = s("path", { d: `M${p5x},${p5y - 70} C${p5x - 200},${p5y - 250} ${p1x + 200},${p1y - 210} ${p1x + 20},${p1y - 72}`,
    fill: "none", stroke: T.thread, "stroke-width": 3, "stroke-dasharray": "10 9", "marker-end": "url(#xparrow)", opacity: 0.8 });
  const loopLabel = s("text", { x: (p1x + p5x) / 2, y: 74, "text-anchor": "middle", fill: "#fda4af", "font-size": 24, "font-style": "italic" },
    "the loss closes the loop: it trains the model");
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

  // Draw the thread up to the focus station, then let an electron run along it.
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

// ---------------------------------------------------------------------------------------------
// The data-formation chain in physical detail
// ---------------------------------------------------------------------------------------------
const LINKS = [
  { name: "Specimen", act: 2, sub: "what we want",
    enters: ["beam damage, knock-on", "contamination build-up", "thickness, tilt, drift"],
    ex: "A zeolite loses its lattice after ~100 e⁻/Å². The specimen changes while you measure it." },
  { name: "Interaction", act: 2, sub: "beam meets atoms",
    enters: ["dynamical (multiple) scattering", "channelling: intensity not linear in thickness", "inelastic background"],
    ex: "HAADF intensity grows as Z^1.7, but not linearly with thickness: the forward model is non-linear." },
  { name: "Scan grid", act: 3, sub: "where we look",
    enters: ["pixel size vs lattice period", "aliasing, Moiré", "scan distortion, flyback"],
    ex: "A 2 Å lattice with 1.5 Å pixels appears as a false 6 Å lattice." },
  { name: "Detector", act: 4, sub: "electrons → charge",
    enters: ["shot noise (Poisson)", "point-spread function, DQE", "dead / hot pixels, saturation"],
    ex: "A scintillator spreads one electron over several pixels; a direct detector counts it once." },
  { name: "Digitisation", act: 4, sub: "charge → number",
    enters: ["gain g, offset", "read noise σr", "quantisation, ADC clipping"],
    ex: "The file stores g·counts + offset + read noise, not electron counts." },
  { name: "File", act: 6, sub: "number + context",
    enters: ["lost pixel size, dose, gain", "TIFF export strips headers", "no specimen / session IDs"],
    ex: "Without the pixel size you cannot tell a lattice from an alias." },
];

register("chain", (el, api) => {
  const W = 1740, H = 370, n = LINKS.length, x0 = 140, dx = (W - 2 * x0) / (n - 1), y = 160;
  const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "xp-chart" });
  svg.append(s("text", { x: W / 2, y: 34, "text-anchor": "middle", fill: "#f8fafc", "font-size": 30, "font-family": "serif", "font-style": "italic" },
    "y  ~  p( y | specimen ; beam, scan, detector, ADC )"));
  svg.append(s("line", { x1: x0 - 80, x2: x0 + (n - 1) * dx + 80, y1: y, y2: y, stroke: T.thread, "stroke-width": 6, "stroke-linecap": "round" }));
  const info = h("div", { class: "xp-card", style: { minHeight: "210px" } });
  const nodes = LINKS.map((L, i) => {
    const x = x0 + i * dx;
    const g = s("g", { transform: `translate(${x} ${y})`, style: "cursor:pointer" });
    const ring = s("circle", { r: 46, fill: "#0f172a", stroke: T.thread, "stroke-width": 4 });
    g.append(ring,
      s("text", { y: 12, "text-anchor": "middle", fill: "#fff", "font-size": 34, "font-weight": 700, "font-family": "Outfit, sans-serif" }, String(i + 1)),
      s("text", { y: 84, "text-anchor": "middle", fill: "#fff", "font-size": 28, "font-weight": 600, "font-family": "Outfit, sans-serif" }, L.name),
      s("text", { y: 116, "text-anchor": "middle", fill: T.muted, "font-size": 21 }, L.sub));
    // red "error enters" flashes
    const bolt = s("path", { d: "M-8 -92 L4 -70 L-4 -70 L8 -52", fill: "none", stroke: T.warn, "stroke-width": 4, "stroke-linecap": "round", opacity: 0 });
    g.append(bolt);
    g.addEventListener("click", () => select(i));
    svg.append(g);
    return { g, ring, bolt };
  });
  const electron = s("circle", { r: 12, cy: y, fill: "#fff", stroke: T.thread, "stroke-width": 4 });
  svg.append(electron);
  const hint = s("text", { x: W / 2, y: H - 10, "text-anchor": "middle", fill: T.muted, "font-size": 21 }, "click a link: what enters the data there?");
  svg.append(hint);
  el.append(svg, info);

  let sel = -1, ex = x0 - 80;
  function select(i) {
    sel = i;
    nodes.forEach((nd, j) => { nd.ring.setAttribute("fill", j === i ? T.thread : "#0f172a"); nd.bolt.setAttribute("opacity", j === i ? 1 : 0.0); });
    const L = LINKS[i];
    info.replaceChildren(
      h("div", { class: "xp-row", style: { justifyContent: "flex-start", gap: "60px" } },
        h("div", {}, h("div", { class: "xp-slider-label", html: `link ${i + 1} of 6 · treated in act ${L.act} (${STATIONS[L.act - 1].label})` }),
          h("div", { style: { fontSize: "40px", fontWeight: 700, fontFamily: "Outfit, sans-serif", color: "#fff" } }, L.name)),
        h("div", {}, h("div", { class: "xp-slider-label" }, "what enters the data here"),
          h("ul", { style: { margin: "6px 0 0 0", fontSize: "26px", color: "#fde68a" } }, L.enters.map((t) => h("li", {}, t)))),
        h("div", { style: { maxWidth: "620px" } }, h("div", { class: "xp-slider-label" }, "EM example"),
          h("div", { style: { fontSize: "26px", marginTop: "6px" } }, L.ex))));
  }
  select(3);
  // an electron travels along the chain
  api.animate((dt) => {
    ex += dt * 240;
    if (ex > x0 + (n - 1) * dx + 80) ex = x0 - 80;
    electron.setAttribute("cx", ex);
  });
});
