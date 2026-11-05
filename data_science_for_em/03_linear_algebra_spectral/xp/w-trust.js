// Trust group: ill-conditioning of overlapping EDS peaks (conditioning).
// The other Trust widgets (ambiguity, rare-phase) are in w-unmix.js and w-pca.js because they
// share the spectrum-image data with those groups.
import { register, T, rng, fmt, clamp, h, s, row, col, card, controls, readout, slider, toggle, legend, chart } from "./core.js";
import { eigSym, lstsq } from "./la.js";

if (!document.getElementById("xp-trust-style")) {
  document.head.append(h("style", { id: "xp-trust-style" }, `
.xp-cond-side { width: 560px; display: flex; flex-direction: column; gap: 12px; }
.xp-cond-side .xp-readout { margin-top: 0; line-height: 1.5; }
`));
}

// ---------------------------------------------------------------------------------------------
// conditioning. Two elements whose X-ray lines overlap. The fit model is y = a·f_A + b·f_B + c:
// a least-squares problem whose two peak columns are almost parallel. Then κ is large, and the
// Poisson noise of the counts is amplified into large, anti-correlated errors of a and b.
// Line energies (keV) are tabulated values; relative line weights are approximate.
// The detector resolution follows FWHM(E)² = FWHM(Mn Kα)² + 2.5 eV·(E − 5895 eV) (Fiori & Newbury).
// ---------------------------------------------------------------------------------------------
const CASES = {
  mos2: { label: "MoS₂: Mo L / S K", win: [2.0, 2.75],
    A: { name: "S", lines: [[2.3078, 1], [2.4640, 0.06]] }, B: { name: "Mo", lines: [[2.2932, 1], [2.3948, 0.53], [2.5184, 0.06]] },
    note: "S Kα and Mo Lα are 15 eV apart, a sixth of the detector FWHM. With the α lines alone, the two columns are almost parallel." },
  batio3: { label: "BaTiO₃: Ba L / Ti K", win: [4.15, 5.75],
    A: { name: "Ti", lines: [[4.5108, 1], [4.9318, 0.13]] }, B: { name: "Ba", lines: [[4.4663, 1], [4.8275, 0.6], [5.1565, 0.17], [5.5311, 0.08]] },
    note: "Ba Lα and Ti Kα are 45 eV apart, a third of the detector FWHM. A textbook overlap in ferroelectric oxides." },
  cevo: { label: "CeVO₃: Ce L / V K", win: [4.55, 6.25],
    A: { name: "V", lines: [[4.9522, 1], [5.4273, 0.13]] }, B: { name: "Ce", lines: [[4.8402, 1], [5.2622, 0.6], [5.6134, 0.2], [6.0526, 0.08]] },
    note: "Today's EDS data: V Kα and Ce Lα are 112 eV apart. That is why the eds-pca maps use Ce Lβ, not Ce Lα." },
  pbs: { label: "PbS: Pb M / S K", win: [2.05, 2.75],
    A: { name: "S", lines: [[2.3078, 1], [2.4640, 0.06]] }, B: { name: "Pb", lines: [[2.3455, 1], [2.4427, 0.55]] },
    note: "Galena: S Kα and Pb Mα are 38 eV apart. Without the Pb Mβ line the two columns are almost identical." },
  feni: { label: "FeNi: Fe K / Ni K", win: [6.0, 8.6],
    A: { name: "Fe", lines: [[6.4039, 1], [7.0580, 0.13]] }, B: { name: "Ni", lines: [[7.4781, 1], [8.2647, 0.13]] },
    note: "Fe Kα and Ni Kα are 1.07 keV apart: the columns are almost orthogonal, κ ≈ 1, and each error is the Poisson limit." },
};
const DE = 0.01;                                         // channel width 10 eV

register("conditioning", (el, api) => {
  let key = "mos2", fwhm0 = 130, dose = 2e4, family = false;
  const r = rng(11);
  const spec = chart({ width: 1000, height: 460, x: { domain: [4.15, 5.75], label: "X-ray energy (keV)" }, y: { domain: [0, 1], label: "counts / channel" } });
  const sc = chart({ width: 560, height: 460, x: { domain: [-0.3, 0.3], label: "relative error of A" }, y: { domain: [-0.3, 0.3], label: "relative error of B" },
    margin: { l: 86 } });
  const out = readout(), note = h("div", { class: "xp-caption", style: { fontSize: "22px", color: "#e2e8f0" } });
  const leg = h("div");

  function columns() {
    const C = CASES[key], E = [];
    for (let e = C.win[0]; e <= C.win[1] + 1e-9; e += DE) E.push(e);
    const shape = (lines) => {
      const use = family ? lines : lines.slice(0, 1);
      const f = E.map((e) => use.reduce((acc, [E0, w]) => {
        const fw = Math.sqrt(fwhm0 ** 2 + 2.5 * (E0 * 1000 - 5895)) / 1000, sd = fw / 2.3548;
        return acc + w * Math.exp(-0.5 * ((e - E0) / sd) ** 2);
      }, 0));
      const sum = f.reduce((a, b) => a + b, 0);
      return f.map((v) => v / sum);
    };
    return { E, fA: shape(C.A.lines), fB: shape(C.B.lines) };
  }

  function update() {
    const C = CASES[key], { E, fA, fB } = columns(), m = E.length, n = 3;
    const X = new Float64Array(m * n);
    for (let i = 0; i < m; i++) { X[i * 3] = fA[i]; X[i * 3 + 1] = fB[i]; X[i * 3 + 2] = 1; }
    const nA = dose / 2, nB = dose / 2, bg = 0.15 * dose / m;
    const mu = E.map((_, i) => nA * fA[i] + nB * fB[i] + bg);
    // condition number of the design matrix with unit-norm columns
    const nrm = [0, 1, 2].map((j) => Math.sqrt(E.reduce((acc, _, i) => acc + X[i * 3 + j] ** 2, 0)));
    const G = new Float64Array(9);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { let sum = 0; for (let i = 0; i < m; i++) sum += X[i * 3 + a] * X[i * 3 + b]; G[a * 3 + b] = sum / (nrm[a] * nrm[b]); }
    const ev = eigSym(G, 3).values, kappa = Math.sqrt(ev[0] / Math.max(ev[2], 1e-30));
    const cosAB = G[1];
    // repeated measurements: fit A, B, background; and A alone (B known) for comparison
    const est = [], estAlone = [];
    let y0 = null, fit0 = null;
    for (let t = 0; t < 260; t++) {
      const y = mu.map((v) => r.poisson(v));
      const w = lstsq(X, y, m, 3);
      est.push([w[0] / nA - 1, w[1] / nB - 1]);
      const y2 = y.map((v, i) => v - nB * fB[i]);
      const X2 = new Float64Array(m * 2);
      for (let i = 0; i < m; i++) { X2[i * 2] = fA[i]; X2[i * 2 + 1] = 1; }
      estAlone.push(lstsq(X2, y2, m, 2)[0] / nA - 1);
      if (!t) { y0 = y; fit0 = w; }
    }
    const sd = (a) => { const mu_ = a.reduce((p, q) => p + q, 0) / a.length; return Math.sqrt(a.reduce((p, q) => p + (q - mu_) ** 2, 0) / (a.length - 1)); };
    const sA = sd(est.map((e) => e[0])), sB = sd(est.map((e) => e[1])), sA1 = sd(estAlone);
    const mA = est.reduce((p, e) => p + e[0], 0) / est.length, mB = est.reduce((p, e) => p + e[1], 0) / est.length;
    const rho = est.reduce((p, e) => p + (e[0] - mA) * (e[1] - mB), 0) / ((est.length - 1) * sA * sB);

    // spectrum: one measurement and the fitted components
    const ymax = Math.max(...y0) * 1.15;
    spec.setDomain(C.win, [0, ymax]);
    spec.clear();
    spec.bars(E.map((e, i) => [e - DE / 2, e + DE / 2, y0[i]]), { fill: T.data, opacity: 0.35 });
    const comp = (f, a) => E.map((e, i) => [e, a * f[i] + fit0[2]]);
    spec.line(comp(fA, fit0[0]), { stroke: T.thread, width: 4 });
    spec.line(comp(fB, fit0[1]), { stroke: T.gauss, width: 4 });
    spec.line(E.map((e, i) => [e, fit0[0] * fA[i] + fit0[1] * fB[i] + fit0[2]]), { stroke: "#fff", width: 2, dash: "7 6" });
    const lab = (lines, c, nm) => (family ? lines : lines.slice(0, 1)).forEach(([E0], j) => { spec.ruleX(E0, { stroke: c, width: 1.5, dash: "3 6" }); if (!j) spec.text(E0, ymax * (c === T.thread ? 0.95 : 0.86), nm, { fill: c, size: 22, anchor: "middle", weight: 700 }); });
    lab(C.A.lines, T.thread, C.A.name); lab(C.B.lines, T.gauss, C.B.name);
    leg.replaceChildren(legend([[T.thread, `${C.A.name} fit`], [T.gauss, `${C.B.name} fit`], ["#fff", "sum + background", "dash"]]));

    // scatter of the estimates
    const lim = clamp(4 * Math.max(sA, sB, 0.01), 0.04, 1.5);
    sc.setDomain([-lim, lim], [-lim, lim]);
    sc.clear();
    sc.ruleX(0, { stroke: T.faint, dash: "4 6" }); sc.ruleY(0, { stroke: T.faint, dash: "4 6" });
    sc.dots(est, { r: 4.5, fill: T.warn, opacity: 0.7 });
    sc.dots(estAlone.map((e) => [e, 0]), { r: 3.5, fill: T.truth, opacity: 0.6 });
    sc.note(`${C.A.name} (A) vs ${C.B.name} (B), 260 repeats`, { corner: "tl" });

    out.set(`columns: cos θ = <b>${fmt(cosAB, 3)}</b> &nbsp; κ = <span class="${kappa > 5 ? "bad" : "good"}">${fmt(kappa, kappa < 10 ? 1 : 0)}</span><br>` +
      `error of ${C.A.name}: <b>±${fmt(100 * sA, 1)} %</b> (${C.B.name} known: <span class="good">±${fmt(100 * sA1, 1)} %</span>)<br>` +
      `error of ${C.B.name}: <b>±${fmt(100 * sB, 1)} %</b> &nbsp; correlation ρ = <span class="${rho < -0.8 ? "bad" : ""}">${fmt(rho, 2)}</span><br>` +
      `noise amplification: <b>${fmt(sA / sA1, 1)}×</b>`);
    note.textContent = C.note;
  }

  const which = toggle({ options: Object.entries(CASES).map(([k, c]) => ({ value: k, label: c.label })), value: key, onChange: (v) => { key = v; update(); } });
  const fam = toggle({ label: "lines", options: [{ value: "a", label: "α only" }, { value: "f", label: "full family" }], value: "a", onChange: (v) => { family = v === "f"; update(); } });
  const res = slider({ label: "FWHM at Mn Kα", min: 60, max: 180, value: fwhm0, digits: 0, unit: "eV", onInput: (v) => { fwhm0 = v; update(); } });
  const dz = slider({ label: "counts in both peaks", min: 1e3, max: 1e6, value: dose, log: true, digits: 0, onInput: (v) => { dose = v; update(); } });
  el.append(controls(which, fam), controls(res, dz),
    row(col(spec.el, leg), h("div", { class: "xp-cond-side" }, sc.el,
      legend([[T.warn, "fit both (A, B, background)", "dot"], [T.truth, "fit A, B known", "dot"]]))),
    row(h("div", { class: "xp-card", style: { minWidth: "700px" } }, out), card(note)));
  update();
});
