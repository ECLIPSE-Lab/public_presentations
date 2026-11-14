"""Render the static real-data figures of the Week 5 deck (img/fig_*.png) and print their numbers.

Usage: python xp/prep/make_figs.py [name ...]   (reads xp/data/*.json; needs numpy, scipy, matplotlib, scikit-learn)
Without arguments the script makes all figures. The speaker notes quote the printed numbers.

Figures
  images     : the four HAADF-STEM images of one interface, with the zones of the labels (opening)
  pipeline   : image -> filtered image -> local maxima -> fitted and labelled candidates (Act 1)
  precision  : position repeatability against dose for three estimators; one column cut with its fit (Act 1)
  shift      : peak height per image against relative intensity per image (Act 2)
  psi4       : four-fold bond order and distance spread of every candidate of image B (Act 2)
  acsf       : radial symmetry functions: the rings around one column and the fingerprint per class (Act 2)
  impurity   : Gini, entropy and misclassification error for two classes (Act 3)
  tree       : the depth-3 tree on the six invariant descriptors (Act 3)
  depth      : accuracy against depth: training, random folds, new image (Act 3)
  unstable   : two bootstrap samples, two different partitions (Act 3)
  ntrees     : accuracy against the number of trees; the variance of an average (Act 4)
  boost      : log-loss against the boosting stage for four learning rates (Act 4)
  ladder     : the baseline ladder, random folds against a new image (Act 5)
  confusion  : confusion matrices of the random forest on a new image (Act 5)
  importance : MDI, permutation on training data, permutation on a new image (Act 6)
  corr       : correlated descriptors share and hide importance (Act 6)
  shap       : SHAP summary of the class "film column" for image B (Act 6)
"""
import base64, json, sys, warnings
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
XP = Path(__file__).resolve().parents[1]
OUT = XP.parent / "img"
INK, MUTED, FAINT, PANEL = "#e2e8f0", "#94a3b8", "#475569", "#141928"
RED, BLUE, ORANGE, GREEN, YELLOW, PINK, VIOLET = "#f43f5e", "#60a5fa", "#fb923c", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"
CLS_COL = ["#60a5fa", "#fb923c", "#c084fc"]                       # substrate, film, no column
CLS = ["substrate", "film", "no column"]
IMG_COL = ["#22d3ee", "#a3e635", "#fbbf24", "#f472b6"]            # images A-D
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 21, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": FAINT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": PANEL, "axes.grid": True, "grid.color": "#94a3b8",
    "grid.alpha": 0.16, "axes.titlesize": 23, "axes.titlecolor": INK, "legend.frameon": False, "legend.fontsize": 19,
    "axes.titlepad": 12, "lines.linewidth": 3, "savefig.transparent": True, "axes.axisbelow": True,
})
LOWHIGH = LinearSegmentedColormap.from_list("lowhigh", ["#38bdf8", "#a78bfa", "#f43f5e"])


def load(name):
    def dec(v):
        if isinstance(v, dict) and "b64" in v:
            dt = {"f4": "<f4", "u2": "<u2", "u1": "u1", "i2": "<i2"}[v["dtype"]]
            a = np.frombuffer(base64.b64decode(v["b64"]), dt).reshape(v["shape"])
            return a.astype(float if v["dtype"] == "f4" else int)
        if isinstance(v, dict):
            return {k: dec(w) for k, w in v.items()}
        if isinstance(v, list):
            return [dec(w) for w in v]
        return v
    return dec(json.loads((XP / "data" / f"{name}.json").read_text()))


def new(w=17.4, h=7.2):
    f = plt.figure(figsize=(w, h), dpi=110)
    f.patch.set_alpha(0)
    return f


def save(f, name):
    f.savefig(OUT / f"fig_{name}.png")
    plt.close(f)
    print(f"-> img/fig_{name}.png")


D = load("columns")
M = load("models")
NAMES = D["names"]
X, Y, G, POS = D["X"], D["y"], D["g"], D["pos"]
LAB = np.nonzero(Y >= 0)[0]
GEOM = M["sets"]["invariant"]
UNIT = dict(zip(NAMES, D["units"]))
LETTER = "ABCD"
col = lambda name: X[:, NAMES.index(name)]
FI = X[LAB][:, [NAMES.index(n) for n in GEOM]]                     # invariant descriptors of the labelled rows
yl, gl = Y[LAB], G[LAB]


def show(ax, k, **kw):
    I = D["images"][k]; H, W = I["size_nm"]
    ax.imshow(I["img"], cmap="gray", extent=[0, W, H, 0], vmin=0, vmax=255, interpolation="bilinear", **kw)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color(FAINT)
    return H, W


def scalebar(ax, H, W, L=1.0, label="1 nm", fs=19):
    x0, y0 = 0.05 * W, 0.94 * H
    ax.plot([x0, x0 + L], [y0, y0], color="white", lw=6, solid_capstyle="butt")
    ax.text(x0 + L / 2, y0 - 0.025 * H, label, color="white", ha="center", va="bottom", fontsize=fs, fontweight="bold")


# ------------------------------------------------------------------------------------------------
def fig_images():
    f = new(17.4, 5.6)
    for k in range(4):
        ax = f.add_axes([0.005 + 0.25 * k, 0.02, 0.24, 0.82])
        H, W = show(ax, k); I = D["images"][k]; z = I["zones"]
        for x in ((z[0] + z[1]) / 2, (z[2] + z[3]) / 2):
            ax.axvline(x, color="white", ls=(0, (5, 4)), lw=2, alpha=0.75)
        scalebar(ax, H, W)
        n = int(((G == k) & (Y >= 0)).sum())
        fr = "10 frames" if "10 frames" in I["name"] else "1 frame"
        ax.set_title(f"{LETTER[k]}", color=IMG_COL[k], fontsize=30, fontweight="bold", loc="left", pad=8)
        ax.text(0.12, 1.03, f"{I['scale_nm'] * 1000:.1f} pm/px · {fr}", transform=ax.transAxes, fontsize=19, color=INK, va="bottom")
        if k == 0:
            for x, t, c in ((z[0] / 2, "substrate", CLS_COL[0]), ((z[1] + z[2]) / 2, "film", CLS_COL[1]), ((z[3] + W) / 2, "amorphous", CLS_COL[2])):
                ax.text(x, 0.065 * H, t, color=c, fontsize=15, fontweight="bold", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#0b0f19", ec="none", alpha=0.85))
        print(f"  image {LETTER[k]}: {W:.2f} x {H:.2f} nm, {I['scale_nm'] * 1000:.1f} pm/px, {int((G == k).sum())} candidates, {n} with a label; "
              f"boundaries at x = {(z[0] + z[1]) / 2:.2f} and {(z[2] + z[3]) / 2:.2f} nm")
    save(f, "images")


# ------------------------------------------------------------------------------------------------
def fig_pipeline():
    k = 1; I = D["images"][k]; H, W = I["size_nm"]; img = I["img"].astype(float); sc = W / img.shape[1]
    x0, x1, y0, y1 = 4.7, 9.1, 2.8, 7.2
    filt = ndi.gaussian_filter(img, 0.04 / sc) - ndi.gaussian_filter(img, 0.8 / sc)
    m = (G == k) & (POS[:, 1] > x0) & (POS[:, 1] < x1) & (POS[:, 0] > y0) & (POS[:, 0] < y1)
    f = new(17.4, 5.3)
    titles = ["1 · image", "2 · smooth − background", "3 · local maxima", "4 · Gaussian fit + label"]
    for q in range(4):
        ax = f.add_axes([0.005 + 0.25 * q, 0.02, 0.24, 0.84])
        if q == 1:
            ax.imshow(filt, cmap="magma", extent=[0, W, H, 0], vmin=np.percentile(filt, 1), vmax=np.percentile(filt, 99.7))
        else:
            ax.imshow(img, cmap="gray", extent=[0, W, H, 0], vmin=0, vmax=255, alpha=1 if q == 0 else 0.75)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(titles[q], fontsize=22, loc="left", pad=8)
        if q == 0:
            ax.plot([x0 + 0.2, x0 + 1.2], [y1 - 0.25] * 2, color="white", lw=6, solid_capstyle="butt")
            ax.text(x0 + 0.7, y1 - 0.36, "1 nm", color="white", ha="center", va="bottom", fontsize=19, fontweight="bold")
        if q == 2:
            ax.scatter(POS[m, 1], POS[m, 0], s=130, facecolors="none", edgecolors=YELLOW, linewidths=2)
        if q == 3:
            c = [CLS_COL[v] if v >= 0 else "#e2e8f0" for v in Y[m]]
            ax.scatter(POS[m, 1], POS[m, 0], s=60, c=c, edgecolors="#0b0f19", linewidths=0.8)
            for v, t in enumerate(CLS):
                ax.scatter([], [], s=90, c=CLS_COL[v], label=t)
            ax.scatter([], [], s=90, c="#e2e8f0", label="no label")
            ax.legend(loc="lower center", ncol=2, fontsize=15, frameon=True, facecolor="#0b0f19", edgecolor="none", framealpha=0.85,
                      handletextpad=0.1, columnspacing=0.6, borderpad=0.3)
    print(f"  crop of image B: x {x0}-{x1} nm, y {y0}-{y1} nm; {int(m.sum())} candidates in the crop "
          f"(substrate {int((Y[m] == 0).sum())}, film {int((Y[m] == 1).sum())}, no column {int((Y[m] == 2).sum())}, no label {int((Y[m] < 0).sum())})")
    save(f, "pipeline")


# ------------------------------------------------------------------------------------------------
def fig_precision():
    P = D["precision"]; ns = np.array([1, 2, 5])
    f = new(17.4, 7.0)
    ax = f.add_axes([0.075, 0.15, 0.40, 0.75])
    for name, c, mk in (("pixel maximum", MUTED, "s"), ("centroid", YELLOW, "D"), ("Gaussian fit", RED, "o")):
        v = [P[str(n)][name] for n in ns]
        ax.plot(ns, v, mk + "-", color=c, ms=13, label=name)
        print(f"  {name}: " + ", ".join(f"{n} frame(s) {q:.1f} pm" for n, q in zip(ns, v)))
    g1 = P["1"]["Gaussian fit"]
    ax.plot([1, 5], [g1 * 0.9, g1 * 0.9 / np.sqrt(5)], "--", color=INK, lw=2)
    ax.text(1.5, g1 * 0.9 / np.sqrt(1.5) * 0.9, "∝ 1/√dose", color=INK, fontsize=20, ha="center", va="top")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(ns); ax.set_xticklabels(["1", "2", "5"]); ax.minorticks_off()
    ax.set_yticks([5, 7, 10, 14]); ax.set_yticklabels(["5", "7", "10", "14"]); ax.set_ylim(4.6, 15.5); ax.set_xlim(0.85, 5.9)
    ax.set_xlabel("frames summed (dose)"); ax.set_ylabel("position repeatability (pm)")
    ax.set_title(f"{D['precision_n']} substrate columns of image D", loc="left"); ax.legend(loc="upper right")
    print(f"  ratio 1 frame / 5 frames (Gaussian fit): {g1 / P['5']['Gaussian fit']:.2f}; sqrt(5) = {np.sqrt(5):.2f}")

    # right: a cut through one substrate column, one frame and the sum of ten frames, with Gaussian fits
    F = load("frames"); fr = F["frames"].astype(float); sc = F["scale_nm"]
    xz = int((F["zones"][0] - F["x0_nm"]) / sc)
    S = fr.sum(0); sm = ndi.gaussian_filter(S, 2)
    sub = sm[20:-20, 12:xz - 12]; iy, ix = np.unravel_index(np.argmax(sub), sub.shape); iy += 20; ix += 12
    hw = 9; xs = (np.arange(-hw, hw + 1)) * sc * 1000                                         # pm
    gauss = lambda x, a, m, s, b: a * np.exp(-(x - m) ** 2 / (2 * s * s)) + b
    ax = f.add_axes([0.575, 0.15, 0.41, 0.75])
    xf = np.linspace(xs[0], xs[-1], 300)
    for data, c, lab, off in ((fr[0, iy, ix - hw:ix + hw + 1], BLUE, "1 frame", 0), (S[iy, ix - hw:ix + hw + 1] / 10, GREEN, "mean of 10 frames", 0)):
        p, cov = curve_fit(gauss, xs, data, p0=[data.max() - data.min(), 0, 60, data.min()])
        ax.plot(xs, data, "o", color=c, ms=10, label=lab)
        ax.plot(xf, gauss(xf, *p), "-", color=c, lw=2.5)
        ax.axvline(p[1], color=c, lw=1.5, ls=":")
        print(f"  column cut ({lab}): centre {p[1]:+.1f} +- {np.sqrt(cov[1, 1]):.1f} pm (fit error), width sigma {abs(p[2]):.0f} pm; pixel {sc * 1000:.0f} pm")
    ax.set_xlabel("x − x₀ (pm)"); ax.set_ylabel("counts per frame (8-bit units)")
    ax.set_title(f"one column, pixels of {sc * 1000:.0f} pm", loc="left")
    ax.legend(loc="upper right", fontsize=18)
    yl_ = ax.get_ylim(); ax.set_ylim(yl_[0], yl_[1] + 0.22 * (yl_[1] - yl_[0]))
    save(f, "precision")


# ------------------------------------------------------------------------------------------------
def fig_shift():
    f = new(17.4, 7.4); sub = Y == 0
    for q, (name, xlabel, logx) in enumerate((("amp", "peak height  amp (counts)", True), ("I_rel", "relative intensity  I_rel = amp / median amp of the 4 nearest", False))):
        ax = f.add_axes([0.05, 0.615 - 0.50 * q, 0.94, 0.30]); v = col(name); med = []
        lo, hi = (5.3, 6.6) if logx else (0.4, 1.58)
        for k in range(4):
            m = (G == k) & sub; x = np.log10(v[m]) if logx else v[m]
            h, e = np.histogram(x, bins=np.linspace(lo, hi, 90), density=True)
            h = ndi.gaussian_filter1d(h, 1.3); c = (e[:-1] + e[1:]) / 2; xx = 10 ** c if logx else c
            ax.plot(xx, h / h.max(), "-", color=IMG_COL[k]); ax.fill_between(xx, 0, h / h.max(), color=IMG_COL[k], alpha=0.25, lw=0)
            med.append(np.median(v[m]))
        ax.set_xlim(10 ** lo if logx else lo, 10 ** hi if logx else hi)
        if logx:
            ax.set_xscale("log"); ax.set_xlim(10 ** lo, 10 ** hi); ax.minorticks_off()
            ax.set_xticks([3e5, 5e5, 1e6, 2e6, 3e6]); ax.set_xticklabels(["3·10⁵", "5·10⁵", "10⁶", "2·10⁶", "3·10⁶"])
            for k in range(4):
                ax.text(med[k], 1.06, f"{LETTER[k]}", color=IMG_COL[k], fontsize=26, fontweight="bold", ha="center")
            r = max(med) / min(med)
            ax.set_title(f"raw peak height of the substrate columns: a factor {r:.1f} between the images", loc="left")
            print("  median amp per image, substrate: " + ", ".join(f"{LETTER[k]} {med[k]:.3g}" for k in range(4)) + f"; max/min = {r:.2f}")
        else:
            sd = [np.std(v[(G == k) & sub]) for k in range(4)]
            ax.set_title("relative intensity of the same columns: the four images agree", loc="left")
            print("  median I_rel per image, substrate: " + ", ".join(f"{LETTER[k]} {med[k]:.3f}" for k in range(4)))
            print("  std of I_rel per image, substrate: " + ", ".join(f"{LETTER[k]} {sd[k]:.3f}" for k in range(4)))
        ax.set_ylim(0, 1.32 if logx else 1.1); ax.set_yticks([]); ax.set_xlabel(xlabel, fontsize=19, labelpad=2); ax.tick_params(axis="x", labelsize=18)
    save(f, "shift")


# ------------------------------------------------------------------------------------------------
def fig_psi4():
    k = 1; m = G == k
    f = new(17.4, 7.4)
    for q, (name, title, cm, lim) in enumerate((("psi4", "ψ₄: four-fold order of the 4 nearest", "viridis", (0, 1)),
                                               ("d1", "d1: distance to the nearest candidate (nm)", "plasma", (0.2, 0.42)))):
        ax = f.add_axes([0.02 + 0.5 * q, 0.02, 0.40, 0.88])
        H, W = show(ax, k, alpha=0.55)
        s_ = ax.scatter(POS[m, 1], POS[m, 0], c=col(name)[m], cmap=cm, vmin=lim[0], vmax=lim[1], s=42, edgecolors="none")
        z = D["images"][k]["zones"]
        for x in ((z[0] + z[1]) / 2, (z[2] + z[3]) / 2):
            ax.axvline(x, color="white", ls=(0, (5, 4)), lw=2, alpha=0.7)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_title(title, loc="left"); scalebar(ax, H, W)
        cax = f.add_axes([0.425 + 0.5 * q, 0.06, 0.014, 0.8]); cb = f.colorbar(s_, cax=cax); cb.outline.set_edgecolor(FAINT)
        print(f"  {name} medians (all four images): " + ", ".join(f"{CLS[c]} {np.median(col(name)[Y == c]):.3f}" for c in range(3)))
    save(f, "psi4")


# ------------------------------------------------------------------------------------------------
def fig_acsf():
    rs = np.array(D["acsf_rs"]); A = D["acsf"]; k = 1
    f = new(17.4, 7.2)
    ax = f.add_axes([0.01, 0.03, 0.39, 0.86]); H, W = show(ax, k)
    # one film column in the middle of the film of image B
    z = D["images"][k]["zones"]; c0 = np.array([H / 2, (z[1] + z[2]) / 2])
    cand = np.nonzero((G == k) & (Y == 1))[0]; i0 = cand[np.argmin(np.hypot(*(POS[cand] - c0).T))]
    py, px = POS[i0]; R = 1.05
    ax.set_xlim(px - R, px + R); ax.set_ylim(py + R, py - R)
    I = D["images"][k]["img"]; s_ = I.shape[0] / H
    cr = I[int((py - R) * s_):int((py + R) * s_), int((px - R) * s_):int((px + R) * s_)]
    ax.images[0].set_clim(np.percentile(cr, 1), np.percentile(cr, 99.8))
    cm = plt.get_cmap("plasma")
    for q, r in enumerate(rs):
        ax.add_patch(Circle((px, py), r, fill=False, ec=cm(0.25 + 0.7 * q / 6), lw=2.5))
    ax.add_patch(Circle((px, py), 0.9, fill=False, ec="white", lw=2, ls="--"))
    nb = (G == k) & (np.hypot(POS[:, 0] - py, POS[:, 1] - px) < 0.9)
    ax.scatter(POS[nb, 1], POS[nb, 0], s=50, c="white", edgecolors="#0b0f19", linewidths=0.8, zorder=5)
    ax.scatter([px], [py], s=220, c=RED, edgecolors="white", linewidths=2, zorder=6)
    ax.set_title("rings at rₛ = 0.2 … 0.8 nm, cutoff 0.9 nm", loc="left", fontsize=21)
    ax = f.add_axes([0.485, 0.15, 0.505, 0.74])
    for c in range(3):
        a = A[Y == c]; q1, q2, q3 = np.percentile(a, [25, 50, 75], 0)
        ax.fill_between(rs, q1, q3, color=CLS_COL[c], alpha=0.22, lw=0)
        ax.plot(rs, q2, "o-", color=CLS_COL[c], ms=10, label=CLS[c] + (" column" if c < 2 else ""))
        print(f"  ACSF median {CLS[c]}: " + ", ".join(f"{r:.1f}: {v:.2f}" for r, v in zip(rs, q2)))
    ax.set_xlabel("ring radius rₛ (nm)"); ax.set_ylabel("G²(rₛ) = Σⱼ exp(−(rᵢⱼ − rₛ)²/2w²) · f_c(rᵢⱼ)", fontsize=19)
    ax.set_title("fingerprint per class: median and quartiles", loc="left"); ax.legend(loc="upper right")
    ax.set_xticks(rs)
    save(f, "acsf")


# ------------------------------------------------------------------------------------------------
def fig_impurity():
    p = np.linspace(1e-6, 1 - 1e-6, 400)
    gini = 2 * p * (1 - p); ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p)); mis = np.minimum(p, 1 - p)
    f = new(11.5, 6.8); ax = f.add_axes([0.11, 0.16, 0.86, 0.80])
    ax.plot(p, ent / 2, color=YELLOW, label="entropy / 2   (−Σ pₖ log₂ pₖ) / 2")
    ax.plot(p, gini, color=RED, label="Gini   1 − Σ pₖ²")
    ax.plot(p, mis, color=MUTED, ls="--", label="misclassification   1 − max pₖ")
    ax.set_xlabel("fraction p of class 1 in the node"); ax.set_ylabel("impurity"); ax.set_xlim(0, 1); ax.set_ylim(0, 0.72)
    ax.legend(loc="upper center", ncol=1, fontsize=18, bbox_to_anchor=(0.5, 1.0))
    ax.text(0.02, 0.03, "pure", color=GREEN, fontsize=20); ax.text(0.98, 0.03, "pure", color=GREEN, fontsize=20, ha="right")
    print(f"  at p = 0.5: Gini {gini[200]:.2f}, entropy {ent[200]:.2f} bit, misclassification {mis[200]:.2f}; at p = 0.1: Gini 0.18, entropy {-(0.1 * np.log2(0.1) + 0.9 * np.log2(0.9)):.2f} bit")
    save(f, "impurity")


# ------------------------------------------------------------------------------------------------
def nice(name, thr):
    u = UNIT[name]
    return f"{name} ≤ {thr:.2f}" + (f" {u}" if u == "nm" else "")


def fig_tree():
    m = DecisionTreeClassifier(max_depth=3, random_state=0).fit(FI, yl); t = m.tree_
    acc = (m.predict(FI) == yl).mean()
    leaves = [i for i in range(t.node_count) if t.children_left[i] < 0]
    order = []
    def walk(i):
        if t.children_left[i] < 0: order.append(i)
        else: walk(t.children_left[i]); walk(t.children_right[i])
    walk(0)
    xpos, ypos = {}, {}
    def place(i, d):
        ypos[i] = d
        if t.children_left[i] < 0: xpos[i] = order.index(i)
        else:
            place(t.children_left[i], d + 1); place(t.children_right[i], d + 1)
            xpos[i] = (xpos[t.children_left[i]] + xpos[t.children_right[i]]) / 2
    place(0, 0)
    nl = len(order)
    f = new(17.4, 7.6); ax = f.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(-0.62, nl - 0.38); ax.set_ylim(3.62, -0.5)
    bw, bh = 0.44, 0.30
    counts = lambda i: (t.value[i][0] * t.n_node_samples[i]).round().astype(int) if t.value[i][0].sum() <= 1.0001 else t.value[i][0].astype(int)
    for i in range(t.node_count):
        x, y_ = xpos[i], ypos[i]; leaf = t.children_left[i] < 0; c = counts(i); maj = int(np.argmax(c))
        if not leaf:
            for ch, lab in ((t.children_left[i], "yes"), (t.children_right[i], "no")):
                ax.plot([x, xpos[ch]], [y_ + bh, ypos[ch] - bh], color=FAINT, lw=2.5, zorder=1)
                ax.text((x + xpos[ch]) / 2 + (-0.06 if lab == "yes" else 0.06), (y_ + ypos[ch]) / 2, lab, color=MUTED, fontsize=17,
                        ha="right" if lab == "yes" else "left", va="center")
        ec = CLS_COL[maj] if leaf else FAINT
        ax.add_patch(FancyBboxPatch((x - bw, y_ - bh), 2 * bw, 2 * bh, boxstyle="round,pad=0,rounding_size=0.06", fc=PANEL, ec=ec, lw=3 if leaf else 2, zorder=2))
        top = f"→ {CLS[maj]}" if leaf else nice(GEOM[t.feature[i]], t.threshold[i])
        ax.text(x, y_ - 0.13, top, ha="center", va="center", fontsize=17 if leaf else (20 if len(top) <= 11 else 17 if len(top) <= 13 else 14), fontweight="bold", color=CLS_COL[maj] if leaf else "white", zorder=3)
        x0 = x - bw + 0.05; wtot = 2 * bw - 0.10
        for k in range(3):
            w = wtot * c[k] / c.sum()
            ax.add_patch(Rectangle((x0, y_ + 0.02), w, 0.10, fc=CLS_COL[k], ec="none", zorder=3)); x0 += w
        ax.text(x, y_ + 0.21, f"n = {c.sum()}" + (f" · {c[maj] / c.sum():.0%}" if leaf else ""), ha="center", va="center", fontsize=16, color=MUTED, zorder=3)
        if not leaf:
            print(f"  node depth {y_}: {nice(GEOM[t.feature[i]], t.threshold[i])}; n = {c.sum()}, counts {c.tolist()}, Gini {t.impurity[i]:.3f}")
        else:
            print(f"  leaf: {CLS[maj]}; n = {c.sum()}, counts {c.tolist()}, purity {c[maj] / c.sum():.2f}")
    for k in range(3):
        ax.add_patch(Rectangle((nl - 2.05, -0.42 + 0.16 * k), 0.16, 0.10, fc=CLS_COL[k], ec="none"))
        ax.text(nl - 1.83, -0.37 + 0.16 * k, CLS[k] + (" column" if k < 2 else ""), fontsize=18, va="center")
    ax.text(-0.55, -0.37, f"depth 3 · {nl} leaves · training accuracy {acc:.1%}", fontsize=20, va="center", color=INK)
    r = next(q for q in M["depth"]["invariant"] if q[0] == 3)
    print(f"  depth-3 tree, six invariant descriptors, {len(yl)} rows: training accuracy {acc:.3f}; models.json: random folds {r[2]:.3f}, new image {r[3]:.3f}; {nl} leaves")
    save(f, "tree")


# ------------------------------------------------------------------------------------------------
def fig_depth():
    f = new(17.4, 7.0)
    for q, (sn, title) in enumerate((("raw", "raw features: amp, bg, sigma"), ("invariant", "six invariant descriptors"))):
        r = np.array(M["depth"][sn]); ax = f.add_axes([0.07 + 0.5 * q, 0.16, 0.42, 0.72])
        ax.plot(r[:, 0], 100 * r[:, 1], "o-", color=MUTED, ms=8, label="training")
        ax.plot(r[:, 0], 100 * r[:, 2], "o-", color=GREEN, ms=8, label="random 5-fold")
        ax.plot(r[:, 0], 100 * r[:, 3], "o-", color=RED, ms=8, label="new image")
        b = int(np.argmax(r[:, 3]))
        ax.plot(r[b, 0], 100 * r[b, 3], "o", ms=20, mfc="none", mec="white", mew=2.5)
        ax.set_ylim(40, 102); ax.set_xlim(0.5, 14.5); ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14])
        ax.set_xlabel("maximum depth of the tree"); ax.set_title(title, loc="left")
        if q == 0:
            ax.set_ylabel("accuracy (%)")
        else:
            ax.set_yticklabels([]); ax.legend(loc="lower right")
        ax.text(r[-1, 0], 100 * r[-1, 1] - 5.5, f"{int(r[-1, 4])} leaves", color=MUTED, fontsize=17, ha="right")
        print(f"  {sn}: best depth on a new image {int(r[b, 0])} ({r[b, 3]:.3f}); depth 14: training {r[-1, 1]:.3f}, random {r[-1, 2]:.3f}, new image {r[-1, 3]:.3f}, {int(r[-1, 4])} leaves; "
              f"depth 3: {r[2, 1]:.3f}/{r[2, 2]:.3f}/{r[2, 3]:.3f}")
    save(f, "depth")


# ------------------------------------------------------------------------------------------------
def fig_unstable():
    fa, fb = "d4", "d_spread"
    F2 = np.c_[col(fa)[LAB], col(fb)[LAB]]; n = len(yl); depth = 3
    lim = [0.17, 0.42, 0.0, 0.42]
    gx, gy = np.meshgrid(np.linspace(lim[0], lim[1], 400), np.linspace(lim[2], lim[3], 400)); grid = np.c_[gx.ravel(), gy.ravel()]
    trees = []
    for seed in range(20):
        b = np.random.default_rng(seed).integers(0, n, n)
        trees.append((seed, b, DecisionTreeClassifier(max_depth=depth, random_state=0).fit(F2[b], yl[b])))
    dis = np.array([[np.mean(a[2].predict(F2) != b[2].predict(F2)) for b in trees] for a in trees])
    iu = np.triu_indices(20, 1); i, j = [q[np.argmax(dis[iu])] for q in iu]
    print(f"  depth-{depth} trees on ({fa}, {fb}), 20 bootstrap samples: the predictions of two trees differ for {np.median(dis[iu]):.1%} of the candidates (median of 190 pairs), "
          f"at most {dis[iu].max():.1%} (the pair shown: samples {i} and {j})")
    f = new(17.4, 7.2)
    cmap = matplotlib.colors.ListedColormap(CLS_COL)
    for q, (seed, b, m) in enumerate((trees[i], trees[j])):
        ax = f.add_axes([0.07 + 0.5 * q, 0.15, 0.42, 0.74])
        ax.pcolormesh(gx, gy, m.predict(grid).reshape(gx.shape), cmap=cmap, vmin=0, vmax=2, alpha=0.30, shading="auto", rasterized=True)
        u = np.unique(b)
        ax.scatter(F2[u, 0], F2[u, 1], c=[CLS_COL[v] for v in yl[u]], s=9, alpha=0.75, lw=0)
        t = m.tree_
        thr = [(("d4", "d_spread")[t.feature[k]], t.threshold[k]) for k in range(t.node_count) if t.children_left[k] >= 0]
        print(f"    sample {seed}: root split {thr[0][0]} ≤ {thr[0][1]:.3f}; all splits: " + ", ".join(f"{a} {v:.3f}" for a, v in thr)
              + f"; accuracy on all rows {np.mean(m.predict(F2) == yl):.3f}")
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.grid(False)
        ax.set_xlabel("d4: mean distance to the 4 nearest (nm)"); ax.set_title(f"bootstrap sample {q + 1}: tree of depth {depth}", loc="left")
        if q == 0:
            ax.set_ylabel("d_spread")
            for v in range(3):
                ax.scatter([], [], c=CLS_COL[v], s=90, label=CLS[v])
            ax.legend(loc="upper right", fontsize=17, frameon=True, facecolor="#0b0f19", edgecolor="none", framealpha=0.8)
        else:
            ax.set_yticklabels([])
    save(f, "unstable")


# ------------------------------------------------------------------------------------------------
def fig_ntrees():
    E = M["ens"]; B = np.arange(1, 201)
    f = new(17.4, 7.0); ax = f.add_axes([0.07, 0.16, 0.50, 0.72])
    for kind, c in (("bagging", BLUE), ("random forest", GREEN)):
        a = 100 * np.array(E[kind]["acc"])
        ax.plot(B, a, color=c, label=f"{kind}  (ρ = {E[kind]['rho']:.2f})")
        print(f"  {kind}: mean single tree {E[kind]['single']:.3f}; B = 1 {a[0]:.1f}, 10 {a[9]:.1f}, 30 {a[29]:.1f}, 100 {a[99]:.1f}, 200 {a[-1]:.1f} %; rho {E[kind]['rho']:.2f}")
    s1 = 100 * E["bagging"]["single"]
    ax.axhline(s1, color=MUTED, ls="--", lw=2); ax.text(1.5, s1 - 0.2, "one tree (mean)", color=MUTED, fontsize=18, ha="left", va="top")
    ax.set_xscale("log"); ax.set_xticks([1, 3, 10, 30, 100, 200]); ax.set_xticklabels(["1", "3", "10", "30", "100", "200"]); ax.minorticks_off()
    ax.set_xlabel("number of trees B"); ax.set_ylabel("accuracy on a new image (%)"); ax.legend(loc="lower right")
    ax.set_title("measured: six invariant descriptors", loc="left"); ax.set_xlim(0.9, 220)
    ax = f.add_axes([0.655, 0.16, 0.335, 0.72])
    for rho, c, lab in ((E["bagging"]["rho"], BLUE, "bagging"), (E["random forest"]["rho"], GREEN, "random forest"), (0.0, INK, "independent trees")):
        ax.plot(B, rho + (1 - rho) / B, color=c, ls="-" if rho else ":", lw=3 if rho else 2.5)
        up = c != GREEN
        ax.text(215, rho + (1 - rho) / 200 + (0.025 if up else -0.03), f"ρ = {rho:.2f}" if rho else "ρ = 0", color=c, fontsize=18, ha="right", va="bottom" if up else "top")
    ax.set_xscale("log"); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"]); ax.minorticks_off()
    ax.set_ylim(0, 1.05); ax.set_xlim(0.9, 220); ax.set_xlabel("number of trees B"); ax.set_ylabel("variance / σ²")
    ax.set_title("theory: ρ + (1 − ρ)/B", loc="left")
    save(f, "ntrees")


# ------------------------------------------------------------------------------------------------
def fig_boost():
    f = new(17.4, 7.0); ax = f.add_axes([0.075, 0.16, 0.915, 0.76]); st = np.arange(1, 301)
    for eta, c in (("0.03", BLUE), ("0.1", GREEN), ("0.3", YELLOW), ("1.0", RED)):
        te, tr = np.array(M["boost"][eta]["test"]), np.array(M["boost"][eta]["train"]); b = int(np.argmin(te))
        ax.plot(st, te, color=c, label=f"η = {eta}")
        ax.plot(st, tr, color=c, ls="--", lw=1.8, alpha=0.8)
        ax.plot(b + 1, te[b], "o", ms=15, mfc=c, mec="white", mew=2.5, zorder=5)
        print(f"  eta {eta}: best stage {b + 1}, new-image log-loss {te[b]:.3f} (accuracy {M['boost'][eta]['acc'][b]:.3f}); stage 300: {te[-1]:.3f}, training {tr[-1]:.4f}")
    ax.plot([], [], "-", color=INK, label="new image"); ax.plot([], [], "--", color=INK, lw=1.8, label="training")
    ax.plot([], [], "o", ms=13, mfc=INK, mec="white", mew=2, ls="none", label="best stage")
    ax.set_xscale("log"); ax.set_xticks([1, 3, 10, 30, 100, 300]); ax.set_xticklabels(["1", "3", "10", "30", "100", "300"]); ax.minorticks_off()
    ax.set_xlim(0.9, 320); ax.set_ylim(0, 1.5); ax.set_xlabel("boosting stage (number of small trees)"); ax.set_ylabel("log-loss")
    ax.legend(loc="upper center", ncol=3, fontsize=18, columnspacing=1.5)
    save(f, "boost")


# ------------------------------------------------------------------------------------------------
def fig_ladder():
    models = M["models"]; short = ["majority", "logistic", "tree\ndepth 3", "tree\nfull", "random\nforest", "boosting"]
    f = new(17.4, 7.2)
    for q, (sn, title) in enumerate((("raw", "raw features: amp, bg, sigma"), ("invariant", "six invariant descriptors"))):
        ax = f.add_axes([0.065 + 0.5 * q, 0.19, 0.43, 0.70])
        r = [100 * M["ladder"][sn][m]["random"] for m in models]; im = [100 * M["ladder"][sn][m]["image"] for m in models]
        x = np.arange(len(models))
        ax.bar(x - 0.2, r, 0.38, color=GREEN, alpha=0.38, label="random 5-fold")
        ax.bar(x + 0.2, im, 0.38, color=RED, label="new image")
        for xi, a, b in zip(x, r, im):
            ax.text(xi - 0.2, a + 1, f"{a:.0f}", ha="center", fontsize=17, color="#a7f3d0")
            ax.text(xi + 0.2, b + 1, f"{b:.0f}", ha="center", fontsize=17, color="white", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=17); ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left")
        if q == 0:
            ax.set_ylabel("accuracy (%)"); ax.legend(loc="upper left", ncol=2, fontsize=17, bbox_to_anchor=(0.0, 1.0), borderpad=0.1)
            ax.set_ylim(0, 118)
        else:
            ax.set_yticklabels([]); ax.set_ylim(0, 118)
        print(f"  {sn}: " + "; ".join(f"{m} {a:.1f} -> {b:.1f}" for m, a, b in zip(models, r, im)))
    save(f, "ladder")


# ------------------------------------------------------------------------------------------------
def fig_confusion():
    f = new(17.4, 7.2)
    cm_ = LinearSegmentedColormap.from_list("c", [PANEL, "#1d4ed8", "#60a5fa"])
    for q, (sn, title) in enumerate((("raw", "raw features"), ("invariant", "six invariant descriptors"))):
        p = M["ladder"][sn]["random forest"]["pred_image"]; C = confusion_matrix(yl, p); R = C / C.sum(1, keepdims=True)
        ax = f.add_axes([0.13 + 0.47 * q, 0.04, 0.36, 0.72]); ax.imshow(R, cmap=cm_, vmin=0, vmax=1); ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i - 0.08, f"{R[i, j]:.0%}", ha="center", va="center", fontsize=30, fontweight="bold", color="white" if i == j else (RED if R[i, j] > 0.1 else MUTED))
                ax.text(j, i + 0.26, f"{C[i, j]}", ha="center", va="center", fontsize=17, color=INK if i == j else MUTED)
        ax.set_xticks(range(3)); ax.set_xticklabels(CLS, fontsize=19); ax.xaxis.tick_top()
        ax.set_yticks(range(3)); ax.set_yticklabels(CLS if q == 0 else [], fontsize=19)
        for t, c in zip(ax.get_xticklabels(), CLS_COL): t.set_color(c)
        for t, c in zip(ax.get_yticklabels(), CLS_COL): t.set_color(c)
        ax.tick_params(length=0)
        acc = np.trace(C) / C.sum()
        ax.set_title(f"{title}: {acc:.1%}", loc="left", pad=74)
        if q == 0:
            ax.set_ylabel("truth", fontsize=21)
        ax.text(1, -0.86, "prediction", ha="center", fontsize=18, color=MUTED)
        print(f"  {sn}: accuracy {acc:.3f}; recall " + ", ".join(f"{CLS[i]} {R[i, i]:.2f}" for i in range(3)) + f"; counts {C.tolist()}")
    save(f, "confusion")


# ------------------------------------------------------------------------------------------------
def fig_importance():
    I = M["imp"]; names = I["names"]; o = np.argsort(I["perm_test"])                 # ascending: the best is on top
    f = new(17.4, 7.6)
    panels = (("mdi", "impurity (MDI), from training", None, 1), ("perm_train", "permutation, training images", None, 100), ("perm_test", "permutation, new image", "perm_test_sd", 100))
    for q, (key, title, sd, mult) in enumerate(panels):
        ax = f.add_axes([0.105 + 0.30 * q, 0.14, 0.265, 0.76]); v = mult * np.array(I[key])[o]
        colr = [RED if names[k] == "bg" else (YELLOW if names[k] == "noise" else BLUE) for k in o]
        ax.barh(range(len(o)), v, color=colr, xerr=None if sd is None else mult * np.array(I[sd])[o], error_kw=dict(ecolor=INK, lw=2, capsize=4))
        ax.set_yticks(range(len(o))); ax.set_yticklabels([names[k] for k in o] if q == 0 else [], fontsize=19)
        if q == 0:
            for t, k in zip(ax.get_yticklabels(), o):
                t.set_color(RED if names[k] == "bg" else (YELLOW if names[k] == "noise" else INK))
        ax.axvline(0, color=INK, lw=1.5); ax.grid(axis="y", visible=False); ax.set_title(title, loc="left", fontsize=20)
        ax.set_xlabel("share of the impurity decrease" if q == 0 else "accuracy drop (percentage points)", fontsize=18)
        ax.tick_params(axis="x", labelsize=17)
        if q:
            ax.set_xlim(-1.2, 4.6)
    rank = lambda key: 1 + int(np.argsort(-np.array(I[key])).tolist().index(names.index("bg")))
    print(f"  bg: rank {rank('mdi')} of 12 by MDI ({I['mdi'][names.index('bg')]:.3f}), rank {rank('perm_train')} by permutation on training images "
          f"({I['perm_train'][names.index('bg')]:+.3f}), rank {rank('perm_test')} on a new image ({I['perm_test'][names.index('bg')]:+.3f})")
    print(f"  noise: MDI {I['mdi'][names.index('noise')]:.3f}, permutation on a new image {I['perm_test'][names.index('noise')]:+.3f}")
    print("  order on a new image: " + ", ".join(f"{names[k]} {I['perm_test'][k]:+.3f}" for k in o[::-1]))
    save(f, "importance")


# ------------------------------------------------------------------------------------------------
def fig_corr():
    C = M["corr"]; names = C["names"]; x = np.arange(len(names))
    f = new(17.4, 7.0); ax = f.add_axes([0.07, 0.16, 0.55, 0.74])
    for q, (key, c, lab) in enumerate((("p0", BLUE, "six descriptors"), ("p1", YELLOW, "+ 1 exact copy of d4"), ("p2", RED, "+ 2 exact copies of d4"))):
        ax.bar(x + 0.27 * (q - 1), 100 * np.array(C[key][:6]), 0.26, color=c, label=lab)
    k4 = names.index("d4")
    for q, key in enumerate(("p0", "p1", "p2")):
        v = 100 * C[key][k4]; ax.text(k4 + 0.27 * (q - 1), max(v, 0) + 0.25, f"{v:.1f}".replace("-", "−"), ha="center", fontsize=16, color="white", fontweight="bold")
    ax.axvspan(k4 - 0.46, k4 + 0.46, color="white", alpha=0.06, lw=0)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=19); ax.axhline(0, color=INK, lw=1.5); ax.grid(axis="x", visible=False)
    ax.set_ylabel("accuracy drop on a new image\n(percentage points)", fontsize=19); ax.legend(loc="upper right", fontsize=17)
    ax.set_title("permutation importance", loc="left"); ax.set_ylim(-1, 14.5)
    print("  d4 with 0, 1, 2 exact copies: " + ", ".join(f"{C[k][k4]:+.3f}" for k in ("p0", "p1", "p2")) + f"; r(d1, d4) = {C['r_d1_d4']:.2f}")
    ax = f.add_axes([0.71, 0.16, 0.28, 0.74]); dr = C["drop"]; keys = list(dr)
    v = [100 * dr[k] for k in keys]
    ax.bar(range(len(keys)), v, 0.62, color=[MUTED, BLUE, BLUE, RED])
    for i, a in enumerate(v):
        ax.text(i, a + 0.25, f"{a:.1f}", ha="center", fontsize=18, color="white", fontweight="bold")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(["all 6", "− d1", "− d4", "− d1\n− d4"], fontsize=18); ax.set_ylim(74, 86.5)
    ax.set_ylabel("accuracy on a new image (%)", fontsize=19); ax.set_title("drop and retrain", loc="left"); ax.grid(axis="x", visible=False)
    print("  drop-column accuracy: " + ", ".join(f"{k} {a:.1f} %" for k, a in zip(keys, v)))
    save(f, "corr")


# ------------------------------------------------------------------------------------------------
def fig_shap():
    S = M["shap"]; sv = S["values"][:, :, 1]; xv = S["x"]; names = S["names"]
    o = np.argsort(np.abs(sv).mean(0)); rng = np.random.default_rng(0)
    f = new(17.4, 7.2); ax = f.add_axes([0.11, 0.15, 0.775, 0.76])
    for r, j in enumerate(o):
        v = sv[:, j]; rk = np.argsort(np.argsort(xv[:, j])) / (len(v) - 1)
        h, e = np.histogram(v, bins=60); dens = h[np.clip(np.digitize(v, e) - 1, 0, 59)] / h.max()
        ax.scatter(v, r + rng.uniform(-1, 1, len(v)) * 0.36 * np.sqrt(dens), c=rk, cmap=LOWHIGH, s=16, lw=0, alpha=0.9, rasterized=True)
    ax.axvline(0, color=INK, lw=1.5); ax.set_yticks(range(len(o))); ax.set_yticklabels([names[j] for j in o], fontsize=21)
    ax.set_xlabel("SHAP value φⱼ for the class “film column” (change of the predicted probability)"); ax.grid(axis="y", visible=False)
    ax.set_title(f"{sv.shape[0]} candidates of image B · φ₀ = {S['base'][1]:.2f}", loc="left")
    sm = plt.cm.ScalarMappable(cmap=LOWHIGH); cax = f.add_axes([0.905, 0.15, 0.014, 0.76]); cb = f.colorbar(sm, cax=cax, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"], fontsize=18); cb.set_label("descriptor value (rank)", fontsize=18, labelpad=-18); cb.outline.set_edgecolor(FAINT)
    print("  mean |SHAP| (film): " + ", ".join(f"{names[j]} {np.abs(sv[:, j]).mean():.3f}" for j in o[::-1]) + f"; base {np.round(S['base'], 3).tolist()}")
    for j in o[::-1][:3]:
        r_ = np.corrcoef(np.argsort(np.argsort(xv[:, j])), sv[:, j])[0, 1]
        print(f"    {names[j]}: rank correlation of value and SHAP {r_:+.2f}; SHAP range {sv[:, j].min():+.2f} … {sv[:, j].max():+.2f}")
    save(f, "shap")


ALL = ["images", "pipeline", "precision", "shift", "psi4", "acsf", "impurity", "tree", "depth", "unstable", "ntrees", "boost",
       "ladder", "confusion", "importance", "corr", "shap"]
if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name in (sys.argv[1:] or ALL):
        print(f"== {name}")
        globals()[f"fig_{name}"]()
