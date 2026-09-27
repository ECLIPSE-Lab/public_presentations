"""Reproducible figures for DSEM Week 5 — From images to features: descriptors & tree ensembles.

Run from the repo root (single-threaded BLAS/OpenMP keeps it fast on shared machines):
    OMP_NUM_THREADS=1 .venv/bin/python data_science_for_em/05_features_trees/img/make_figures.py

The HAADF simulation / column finding / descriptor code is identical to
notebooks/week05_features_trees.ipynb so that the numbers on the slides match the notebook.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from itertools import product
from math import factorial

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from skimage import filters, measure, segmentation
from skimage.feature import peak_local_max
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
# validated 3-slot categorical palette (host / vacancy / substitution)
CCOL = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED = "#1f2937", "#6b7280"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ============================================================================
# Simulation + column finding + descriptors (same as the notebook)
# ============================================================================
A = 12.0          # lattice spacing (px)
N_PIX = 240       # image size (px)
SIG = 2.0         # column width (px)
CLASSES = ["host", "vacancy", "substitution"]
REL_I = np.array([1.0, 0.75, 1.35])   # column intensity relative to host (~Z^1.7 / partial occupancy)


def simulate_haadf(rng, n_pix=N_PIX, a=A):
    """Simulate one HAADF-STEM image of a square lattice with vacancy/substitution columns."""
    n = int(n_pix // a)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    pos = np.stack([ii.ravel() * a + a / 2, jj.ravel() * a + a / 2], 1).astype(float)
    p_vac, p_sub = rng.uniform(0.03, 0.10, 2)            # defect content varies per image
    cls = rng.choice(3, size=len(pos), p=[1 - p_vac - p_sub, p_vac, p_sub])
    # local relaxation: substitution pushes neighbours out, vacancy pulls them in
    disp = np.zeros_like(pos)
    tree = cKDTree(pos)
    for k in np.where(cls > 0)[0]:
        for j in tree.query_ball_point(pos[k], r=1.5 * a):
            if j == k:
                continue
            v = pos[j] - pos[k]
            r = np.linalg.norm(v)
            amp = 0.8 if cls[k] == 2 else -0.6
            disp[j] += amp * (a / r) ** 2 * v / r
    pos_true = pos + disp + rng.normal(0, 0.12, pos.shape)
    # image-level nuisance: probe current / detector gain, background, dose
    scale = rng.uniform(0.6, 1.4)
    inten = scale * REL_I[cls] * (1 + 0.10 * rng.normal(size=len(pos)))
    sig = SIG * (1 + 0.04 * rng.normal(size=len(pos)))
    img = np.zeros((n_pix, n_pix))
    w = 8
    for (r0, c0), I0, s in zip(pos_true, inten, sig):
        ri, ci = int(round(r0)), int(round(c0))
        rs, cs = slice(max(ri - w, 0), min(ri + w + 1, n_pix)), slice(max(ci - w, 0), min(ci + w + 1, n_pix))
        R, C = np.mgrid[rs, cs]
        img[rs, cs] += I0 * np.exp(-((R - r0) ** 2 + (C - c0) ** 2) / (2 * s ** 2))
    yy, xx = np.mgrid[0:n_pix, 0:n_pix] / n_pix
    bg = scale * (rng.uniform(0.05, 0.25) + rng.uniform(-0.08, 0.08) * xx + rng.uniform(-0.08, 0.08) * yy)
    dose = rng.uniform(25, 60)                           # counts at a host-column peak
    img = rng.poisson(np.clip(img + bg, 0, None) * dose) / dose
    return img, pos_true, cls


def _gauss2d(rc, amp, r0, c0, s, b):
    R, C = rc
    return (amp * np.exp(-((R - r0) ** 2 + (C - c0) ** 2) / (2 * s ** 2)) + b).ravel()


def find_columns(img, a=A, w=4):
    """Peak finding on a smoothed image, then 2-D Gaussian refinement per column."""
    sm = ndimage.gaussian_filter(img, 1.5)
    base = np.median(sm)
    thr = base + 0.3 * (np.percentile(sm, 99.5) - base)
    peaks = peak_local_max(sm, min_distance=int(0.4 * a), threshold_abs=thr, exclude_border=w)
    out = []
    for r, c in peaks:
        win = img[r - w:r + w + 1, c - w:c + w + 1]
        R, C = np.mgrid[r - w:r + w + 1, c - w:c + w + 1]
        p0 = [win.max() - win.min(), r, c, SIG, win.min()]
        try:
            p, _ = curve_fit(_gauss2d, (R, C), win.ravel(), p0=p0, maxfev=2000)
            if abs(p[1] - r) > 2 or abs(p[2] - c) > 2 or not (0.5 < abs(p[3]) < 5):
                raise RuntimeError
        except Exception:
            p = p0
        amp, r0, c0, s, b = p
        integ = (win - b).sum()
        out.append([r0, c0, amp, abs(s), b, integ])
    return np.array(out)   # columns: row, col, amp, sigma, bg, integrated intensity


FEATURES = ["amp", "I_int", "sigma", "bg_local", "x_pos",
            "I_rel", "amp_rel", "n_nb", "d_mean", "d_std", "ang_std", "G_0.9a", "G_1.0a", "G_1.1a"]
RAW_FEATURES = FEATURES[:5]


def local_descriptors(cols, a=A, n_pix=N_PIX):
    """Local-environment descriptors for every detected column (2-D ACSF analogue)."""
    xy = cols[:, :2]
    tree = cKDTree(xy)
    feats = []
    for k in range(len(cols)):
        nb = [j for j in tree.query_ball_point(xy[k], r=1.25 * a) if j != k]
        v = xy[nb] - xy[k]
        d = np.linalg.norm(v, axis=1)
        if len(nb) >= 2:
            ang = np.sort(np.arctan2(v[:, 0], v[:, 1]))
            gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
            ang_std = np.degrees(gaps.std())
        else:
            ang_std = 90.0
        I_rel = cols[k, 5] / np.median(cols[nb, 5]) if len(nb) else 1.0     # integrated, relative
        amp_rel = cols[k, 2] / np.median(cols[nb, 2]) if len(nb) else 1.0   # peak height, relative
        # 2-D radial symmetry functions G(R_s) = sum_j exp(-eta (r_ij - R_s)^2) over a wider shell
        nb2 = [j for j in tree.query_ball_point(xy[k], r=1.6 * a) if j != k]
        d2 = np.linalg.norm(xy[nb2] - xy[k], axis=1)
        eta = 1.0 / (0.05 * a) ** 2
        G = [np.exp(-eta * (d2 - Rs * a) ** 2).sum() for Rs in (0.9, 1.0, 1.1)]
        feats.append([cols[k, 2], cols[k, 5], cols[k, 3], cols[k, 4], cols[k, 1],
                      I_rel, amp_rel, len(nb), d.mean() if len(nb) else a, d.std() if len(nb) else 0.0,
                      ang_std, *G])
    return np.array(feats)


def build_dataset(n_images=12, seed=0):
    rng = np.random.default_rng(seed)
    X, y, g, stats = [], [], [], []
    for im in range(n_images):
        img, pos_true, cls = simulate_haadf(rng)
        cols = find_columns(img)
        F = local_descriptors(cols)
        d, idx = cKDTree(pos_true).query(cols[:, :2])
        ok = d < A / 3
        inner = np.all((cols[:, :2] > A) & (cols[:, :2] < N_PIX - A), axis=1)
        keep = ok & inner
        X.append(F[keep]); y.append(cls[idx[keep]]); g.append(np.full(keep.sum(), im))
        stats.append(dict(n_true=len(pos_true), n_found=len(cols), n_matched=ok.sum(),
                          pos_err=d[ok].mean()))
    return np.vstack(X), np.concatenate(y), np.concatenate(g), stats


def exact_shapley(predict, x, background):
    """Exact interventional Shapley values for one sample (brute force over all 2^M coalitions)."""
    M = len(x)
    masks = np.array(list(product([0, 1], repeat=M)), dtype=bool)          # (2^M, M)
    B = len(background)
    Z = np.where(masks[:, None, :], x[None, None, :], background[None, :, :])   # (2^M, B, M)
    v = predict(Z.reshape(-1, M)).reshape(len(masks), B).mean(1)
    idx = {tuple(m): i for i, m in enumerate(masks)}
    w = np.array([factorial(s) * factorial(M - s - 1) / factorial(M) for s in range(M)])
    phi = np.zeros(M)
    for i, m in enumerate(masks):
        for j in np.where(~m)[0]:
            m2 = m.copy(); m2[j] = True
            phi[j] += w[m.sum()] * (v[idx[tuple(m2)]] - v[i])
    return phi, v[0]


# ============================================================================
# Figures
# ============================================================================
def fig_segmentation():
    rng = np.random.default_rng(3)
    n = 256
    img = np.zeros((n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(45):
        r0, c0 = rng.uniform(15, n - 15, 2)
        a, b = rng.uniform(5, 13), rng.uniform(4, 10)
        th = rng.uniform(0, np.pi)
        u = ((xx - c0) * np.cos(th) + (yy - r0) * np.sin(th)) / a
        v = (-(xx - c0) * np.sin(th) + (yy - r0) * np.cos(th)) / b
        img[u ** 2 + v ** 2 < 1] = 1.0
    img = ndimage.gaussian_filter(img, 1.2) + 0.15
    img = rng.poisson(img * 30) / 30
    sm = ndimage.gaussian_filter(img, 1.5)
    t = filters.threshold_otsu(sm)
    mask = sm > t
    dist = ndimage.distance_transform_edt(mask)
    pk = peak_local_max(dist, min_distance=5, labels=measure.label(mask))
    markers = np.zeros_like(mask, int)
    markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = segmentation.watershed(-dist, markers, mask=mask)
    props = measure.regionprops_table(lab, properties=("area", "eccentricity", "equivalent_diameter_area"))
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.6))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("1. noisy BF/ADF image")
    ax[1].hist(sm.ravel(), 80, color="#9ca3af"); ax[1].axvline(t, color=CCOL[1], lw=2)
    ax[1].set_title(f"2. histogram + Otsu threshold"); ax[1].set_xlabel("smoothed intensity")
    ax[2].imshow(mask, cmap="gray"); ax[2].set_title("3. binary mask (touching particles!)")
    ax[3].imshow(lab, cmap="nipy_spectral", interpolation="nearest")
    ax[3].set_title(f"4. distance-transform watershed\n{lab.max()} labelled particles")
    ax[4].scatter(props["equivalent_diameter_area"], props["eccentricity"], s=30, color=CCOL[0])
    ax[4].set_xlabel("equivalent diameter (px)"); ax[4].set_ylabel("eccentricity")
    ax[4].set_title("5. regionprops → feature table")
    for a in ax[[0, 2, 3]]:
        a.axis("off")
    save(fig, "seg_pipeline.png")


def fig_lattice_and_finding():
    rng = np.random.default_rng(0)
    img, pos_true, cls = simulate_haadf(rng)
    cols = find_columns(img)
    d, idx = cKDTree(pos_true).query(cols[:, :2])
    lab = cls[idx]
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.2))
    ax[0].imshow(img, cmap="magma"); ax[0].set_title("simulated HAADF-STEM (Poisson noise, 400 columns)")
    ax[1].imshow(img, cmap="gray")
    for c in range(3):
        m = lab == c
        ax[1].scatter(cols[m, 1], cols[m, 0], s=28 if c else 10, facecolor="none" if c == 0 else CCOL[c],
                      edgecolor=CCOL[c], lw=1.2, label=f"{CLASSES[c]} ({m.sum()})")
    ax[1].set_title("detected + Gaussian-refined columns, coloured by ground truth")
    ax[1].legend(loc="upper right", fontsize=10, framealpha=0.9)
    for a in ax:
        a.axis("off")
    save(fig, "haadf_lattice.png")

    # atom-finding steps on a zoomed crop
    sl = (slice(60, 132), slice(60, 132))
    sm = ndimage.gaussian_filter(img, 1.5)
    inside = (cols[:, 0] > 60) & (cols[:, 0] < 132) & (cols[:, 1] > 60) & (cols[:, 1] < 132)
    base = np.median(sm)
    thr = base + 0.3 * (np.percentile(sm, 99.5) - base)
    pk = peak_local_max(sm, min_distance=int(0.4 * A), threshold_abs=thr, exclude_border=4)
    pin = (pk[:, 0] > 60) & (pk[:, 0] < 132) & (pk[:, 1] > 60) & (pk[:, 1] < 132)
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.2))
    ax[0].imshow(img[sl], cmap="gray"); ax[0].set_title("1. raw crop (counts)")
    ax[1].imshow(sm[sl], cmap="gray"); ax[1].set_title("2. Gaussian-smoothed (σ = 1.5 px)")
    ax[2].imshow(sm[sl], cmap="gray")
    ax[2].scatter(pk[pin, 1] - 60, pk[pin, 0] - 60, marker="+", s=90, color=CCOL[1], lw=2)
    ax[2].set_title("3. local maxima (integer pixels)")
    ax[3].imshow(img[sl], cmap="gray")
    ax[3].scatter(cols[inside, 1] - 60, cols[inside, 0] - 60, s=40, facecolor="none", edgecolor=CCOL[2], lw=1.6,
                  label="2-D Gaussian fit")
    tr = (pos_true[:, 0] > 60) & (pos_true[:, 0] < 132) & (pos_true[:, 1] > 60) & (pos_true[:, 1] < 132)
    ax[3].scatter(pos_true[tr, 1] - 60, pos_true[tr, 0] - 60, s=8, color=CCOL[0], label="ground truth")
    ax[3].set_title(f"4. sub-pixel refinement: mean error {d.mean():.2f} px")
    ax[3].legend(fontsize=9, loc="lower right")
    for a in ax:
        a.axis("off")
    save(fig, "atom_finding_steps.png")


def fig_local_env(X, y):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    # schematic environment with relaxation
    for k, (c, amp, title) in enumerate([(0, 0.0, "host"), (1, -0.6, "vacancy column"), (2, 0.8, "substitution")]):
        pass
    a = ax[0]
    nb = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], float)
    for (dx, dy) in nb:
        r = np.hypot(dx, dy)
        s = 0.8 * 0.08 / r ** 2
        a.add_patch(Circle((dx * (1 + s), dy * (1 + s)), 0.13, color=CCOL[0], alpha=0.85))
        if r == 1:
            a.plot([0, dx * (1 + s)], [0, dy * (1 + s)], color=MUTED, lw=1.5, ls="--")
    a.add_patch(Circle((0, 0), 0.17, color=CCOL[2]))
    th = np.linspace(0, np.pi / 2, 30)
    a.plot(0.35 * np.cos(th), 0.35 * np.sin(th), color=INK, lw=1.2)
    a.text(0.3, 0.3, r"$\theta_{jik}$", fontsize=14)
    a.text(0.45, -0.14, r"$r_{ij}$", fontsize=14)
    a.add_patch(Circle((0, 0), 1.25, fill=False, ls=":", color=MUTED))
    a.text(-0.95, -1.35, r"cutoff $r_c = 1.25a$", color=MUTED)
    a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); a.set_aspect("equal"); a.axis("off")
    a.set_title("environment of column $i$")
    # G(R_s) curves
    a = ax[1]
    Rs = np.linspace(0.8, 1.55, 200)
    eta = 1.0 / (0.05) ** 2
    rng = np.random.default_rng(1)
    for c, amp in [(0, 0.0), (1, -0.6 / A), (2, 0.8 / A)]:
        d1 = 1 + amp; d2 = np.sqrt(2) * (1 + amp / 2)
        dists = np.r_[np.full(4, d1), np.full(4, d2)]
        G = np.exp(-eta * (dists[None] - Rs[:, None]) ** 2).sum(1)
        a.plot(Rs, G, color=CCOL[c], lw=2.2, label=CLASSES[c])
    for r in (0.9, 1.0, 1.1):
        a.axvline(r, color=MUTED, ls=":", lw=1)
    a.set_xlabel(r"shell radius $R_s$ / $a$"); a.set_ylabel(r"$G_i(R_s)=\sum_j e^{-\eta(r_{ij}-R_s)^2}$")
    a.set_title("2-D radial symmetry function (ACSF analogue)")
    a.legend()
    # measured descriptor scatter
    a = ax[2]
    iI, iG = FEATURES.index("I_rel"), FEATURES.index("G_1.1a")
    for c in range(3):
        m = y == c
        a.scatter(X[m, iI], X[m, FEATURES.index("d_mean")], s=6 if c == 0 else 14, alpha=0.5 if c == 0 else 0.8,
                  color=CCOL[c], label=CLASSES[c])
    a.set_xlabel(r"$I_\mathrm{rel}$ = column intensity / median neighbour intensity")
    a.set_ylabel(r"mean neighbour distance $\bar d$ (px)")
    a.set_title("measured descriptors (12 images, 3 888 columns)")
    a.legend(markerscale=2)
    save(fig, "local_env_descriptors.png")


def fig_tree(X, y):
    f2 = [FEATURES.index("I_rel"), FEATURES.index("d_mean")]
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X[:, f2], y)
    fig, ax = plt.subplots(1, 2, figsize=(21, 7), gridspec_kw={"width_ratios": [1, 1.4]})
    xx, yy = np.meshgrid(np.linspace(0.4, 1.9, 400), np.linspace(10.5, 13.5, 400))
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    from matplotlib.colors import ListedColormap
    ax[0].contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], cmap=ListedColormap(CCOL), alpha=0.18)
    for c in range(3):
        m = y == c
        ax[0].scatter(X[m, f2[0]], X[m, f2[1]], s=6 if c == 0 else 14, color=CCOL[c], alpha=0.6, label=CLASSES[c])
    ax[0].set_xlim(0.4, 1.9); ax[0].set_ylim(10.5, 13.5)
    ax[0].set_xlabel(r"$I_\mathrm{rel}$"); ax[0].set_ylabel(r"$\bar d$ (px)")
    ax[0].set_title("depth-2 tree: axis-aligned boxes"); ax[0].legend(markerscale=2)
    plot_tree(tree, feature_names=["I_rel", "d_mean"], class_names=CLASSES, filled=False, impurity=True,
              proportion=False, rounded=True, fontsize=12, ax=ax[1])
    ax[1].set_title("the same tree as a flowchart (Gini impurity per node)")
    save(fig, "tree_partition.png")


def fig_ladder(X, y, g):
    gkf = GroupKFold(4)
    kf = KFold(4, shuffle=True, random_state=0)
    raw = [FEATURES.index(f) for f in RAW_FEATURES]
    allf = list(range(X.shape[1]))
    models = {
        "majority": DummyClassifier(),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random forest": RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced_subsample",
                                                random_state=0),
        "grad. boosting": HistGradientBoostingClassifier(max_iter=150, class_weight="balanced", random_state=0),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier((64, 64), max_iter=500, early_stopping=True,
                                                             random_state=0)),
    }
    res = {}
    for name, m in models.items():
        for fs, cols in [("raw", raw), ("all", allf)]:
            sg = cross_val_score(m, X[:, cols], y, cv=gkf, groups=g, scoring="balanced_accuracy").mean()
            sr = cross_val_score(m, X[:, cols], y, cv=kf, scoring="balanced_accuracy").mean()
            res[(name, fs)] = (sg, sr)
            print(f"  {name:15s} {fs:4s}  GroupKFold {sg:.3f}  random KFold {sr:.3f}")
    names = list(models)
    xpos = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(19, 6.2))
    w = 0.38
    for k, (fs, lab, col) in enumerate([("raw", "raw intensity features only", "#9ca3af"),
                                        ("all", "+ local-environment descriptors", CCOL[0])]):
        vals = [res[(n, fs)][0] for n in names]
        b = ax[0].bar(xpos + (k - 0.5) * w, vals, w * 0.95, color=col, label=lab)
        for xi, v in zip(xpos + (k - 0.5) * w, vals):
            ax[0].text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, color=INK)
    ax[0].axhline(1 / 3, color=MUTED, ls=":", lw=1)
    ax[0].set_xticks(xpos); ax[0].set_xticklabels(names)
    ax[0].set_ylabel("balanced accuracy (GroupKFold by image)"); ax[0].set_ylim(0, 1.25)
    ax[0].set_title("baseline ladder: descriptors matter more than the model")
    ax[0].legend(loc="upper left")
    for k, (idx, lab, col) in enumerate([(1, "random KFold (leaky)", CCOL[1]), (0, "GroupKFold by image (honest)", CCOL[0])]):
        vals = [res[(n, "raw")][idx] for n in names[1:4]]
        ax[1].bar(np.arange(3) + (k - 0.5) * w, vals, w * 0.95, color=col, label=lab)
        for xi, v in zip(np.arange(3) + (k - 0.5) * w, vals):
            ax[1].text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, color=INK)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names[1:4]); ax[1].set_ylim(0, 0.75)
    ax[1].axhline(1 / 3, color=MUTED, ls=":", lw=1)
    ax[1].set_ylabel("balanced accuracy (raw features)")
    ax[1].set_title("raw features: the background level identifies the image → leakage")
    ax[1].legend(loc="upper left")
    save(fig, "baseline_ladder.png")
    return res


def fig_importance(X, y, g):
    tr, te = next(GroupKFold(4).split(X, y, g))
    rf = RandomForestClassifier(200, min_samples_leaf=1, class_weight="balanced_subsample",
                                random_state=0).fit(X[tr], y[tr])
    mdi = rf.feature_importances_
    pi = permutation_importance(rf, X[te], y[te], scoring="balanced_accuracy", n_repeats=10, random_state=0)
    order = np.argsort(pi.importances_mean)
    fig, ax = plt.subplots(1, 2, figsize=(18, 6.5), sharey=True)
    cols = [CCOL[1] if FEATURES[i] == "x_pos" else CCOL[0] for i in order]
    ax[0].barh(np.arange(len(order)), mdi[order], color=cols)
    ax[0].set_yticks(np.arange(len(order))); ax[0].set_yticklabels([FEATURES[i] for i in order])
    ax[0].set_title("impurity (MDI) importance — training data")
    ax[0].set_xlabel("mean decrease in Gini impurity")
    ax[1].barh(np.arange(len(order)), pi.importances_mean[order], xerr=pi.importances_std[order], color=cols)
    ax[1].set_title("permutation importance — held-out images")
    ax[1].set_xlabel("drop in balanced accuracy")
    ax[1].axvline(0, color=INK, lw=0.8)
    save(fig, "mdi_vs_permutation.png")
    print("  MDI x_pos = %.3f, perm x_pos = %.3f" % (mdi[FEATURES.index("x_pos")],
                                                   pi.importances_mean[FEATURES.index("x_pos")]))

    # Pitfall 1 -- a near-duplicate descriptor splits the importance of the original.
    #   I_rel_r3 = the same relative intensity integrated over a slightly smaller window (simulated
    #   as I_rel with 2 % measurement noise). Adding it halves the apparent importance of I_rel.
    rng = np.random.default_rng(0)
    iI = FEATURES.index("I_rel")
    Xd = np.c_[X, X[:, iI] * (1 + 0.02 * rng.normal(size=len(X)))]
    rf_d = clone(rf).fit(Xd[tr], y[tr])
    pi_d = permutation_importance(rf_d, Xd[te], y[te], scoring="balanced_accuracy", n_repeats=10, random_state=0)
    base_d = balanced_accuracy_score(y[te], rf_d.predict(Xd[te]))

    def group_drop(model, Xte, cols, base, n=10):
        out = []
        for _ in range(n):
            Xp = Xte.copy(); perm = rng.permutation(len(Xte))
            Xp[:, cols] = Xp[perm][:, cols]
            out.append(base - balanced_accuracy_score(y[te], model.predict(Xp)))
        return np.mean(out)

    dup_single = pi.importances_mean[iI]
    dup_a, dup_b = pi_d.importances_mean[iI], pi_d.importances_mean[-1]
    dup_group = group_drop(rf_d, Xd[te], [iI, Xd.shape[1] - 1], base_d)
    # Pitfall 2 -- permuting one of several *dependent* geometry descriptors creates impossible
    #   combinations (d_mean inconsistent with G), so single-feature importances over-count.
    geo = [FEATURES.index(f) for f in ["d_mean", "G_0.9a", "G_1.0a", "G_1.1a"]]
    geo_sum = pi.importances_mean[geo].sum()
    base = balanced_accuracy_score(y[te], rf.predict(X[te]))
    geo_group = group_drop(rf, X[te], geo, base)
    print(f"  I_rel alone {dup_single:.3f} | with duplicate: I_rel {dup_a:.3f}, I_rel_r3 {dup_b:.3f}, "
          f"grouped {dup_group:.3f}")
    print(f"  geometry: sum of single {geo_sum:.3f}, grouped {geo_group:.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(18, 5.8))
    labs = ["I_rel\n(no duplicate)", "I_rel\n(+ duplicate)", "I_rel_r3\n(the duplicate)", "both permuted\ntogether"]
    vals = [dup_single, dup_a, dup_b, dup_group]
    cols_ = [CCOL[0], "#9ca3af", "#9ca3af", CCOL[0]]
    ax[0].bar(range(4), vals, color=cols_, width=0.6)
    for k, v in enumerate(vals):
        ax[0].text(k, v + 0.004, f"{v:.3f}", ha="center")
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labs)
    ax[0].set_ylabel("drop in balanced accuracy")
    ax[0].set_title("pitfall 1: a correlated copy splits (hides) the importance")
    vals2 = [geo_sum, geo_group]
    ax[1].bar([0, 1], vals2, color=["#9ca3af", CCOL[0]], width=0.5)
    for k, v in enumerate(vals2):
        ax[1].text(k, v + 0.008, f"{v:.2f}", ha="center")
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["sum of single-feature\npermutations",
                                                     "geometry group\npermuted together"])
    ax[1].set_title("pitfall 2: dependent features → unphysical permutations")
    ax[1].set_ylabel("drop in balanced accuracy")
    save(fig, "correlated_permutation.png")
    return rf, tr, te


def fig_shapley(X, y, g):
    tr, te = next(GroupKFold(4).split(X, y, g))
    feats = ["amp", "I_rel", "d_mean", "d_std", "ang_std", "G_0.9a", "G_1.1a", "bg_local", "x_pos"]
    fi = [FEATURES.index(f) for f in feats]
    gbm = HistGradientBoostingClassifier(max_iter=150, class_weight="balanced", random_state=0).fit(X[tr][:, fi], y[tr])
    rng = np.random.default_rng(0)
    bg = X[tr][rng.choice(len(tr), 32, replace=False)][:, fi]
    # explain P(vacancy) as a log-odds-free probability for readability
    predict = lambda Z: gbm.predict_proba(Z)[:, 1]
    te_idx = np.r_[rng.choice(np.where(y[te] == 1)[0], 20, replace=False),
                   rng.choice(np.where(y[te] == 0)[0], 20, replace=False),
                   rng.choice(np.where(y[te] == 2)[0], 10, replace=False)]
    PHI = []
    for i in te_idx:
        phi, base = exact_shapley(predict, X[te][i, fi], bg)
        PHI.append(phi)
    PHI = np.array(PHI)
    vac_rows = te_idx[:20]
    p_vac = predict(X[te][vac_rows][:, fi])
    x0 = X[te][vac_rows[np.argmin(np.abs(p_vac - 0.7))], fi]   # an informative, not-saturated example
    phi0, base0 = exact_shapley(predict, x0, bg)
    fig, ax = plt.subplots(1, 2, figsize=(19, 6.2))
    o = np.argsort(np.abs(PHI).mean(0))
    ax[0].barh(np.arange(len(feats)), np.abs(PHI).mean(0)[o], color=CCOL[0])
    ax[0].set_yticks(range(len(feats))); ax[0].set_yticklabels([feats[i] for i in o])
    ax[0].set_xlabel(r"mean $|\phi_j|$ for $P(\mathrm{vacancy})$")
    ax[0].set_title("global view: mean |Shapley value| (50 held-out columns)")
    # waterfall for one vacancy column
    o2 = np.argsort(np.abs(phi0))[::-1]
    cum = base0
    for k, j in enumerate(o2):
        col = CCOL[2] if phi0[j] > 0 else CCOL[1]
        ax[1].barh(k, phi0[j], left=cum, color=col)
        ax[1].text(max(cum, cum + phi0[j]) + 0.01, k, f"{phi0[j]:+.2f}", va="center", fontsize=10)
        cum += phi0[j]
    ax[1].set_yticks(range(len(feats))); ax[1].set_yticklabels([f"{feats[j]} = {x0[j]:.2f}" for j in o2])
    ax[1].invert_yaxis()
    ax[1].axvline(base0, color=MUTED, ls=":"); ax[1].axvline(cum, color=INK, ls="--")
    ax[1].set_xlabel("P(vacancy)"); ax[1].set_xlim(-0.05, 1.2)
    ax[1].set_title(f"local view: one vacancy column, baseline {base0:.2f} → prediction {cum:.2f}")
    save(fig, "shapley_vacancy.png")
    print("  shapley completeness check:", round(base0 + phi0.sum(), 4), round(predict(x0[None])[0], 4))


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["seg", "lattice", "env", "tree", "importance", "shapley", "ladder"]
    if "seg" in todo:
        fig_segmentation()
    if "lattice" in todo:
        fig_lattice_and_finding()
    X, y, g, st = build_dataset()
    print("dataset", X.shape, np.bincount(y), "mean pos err %.3f px" % np.mean([s["pos_err"] for s in st]))
    for key, fn in [("env", fig_local_env), ("tree", fig_tree)]:
        if key in todo:
            fn(X, y)
    for key, fn in [("importance", fig_importance), ("shapley", fig_shapley), ("ladder", fig_ladder)]:
        if key in todo:
            fn(X, y, g)
