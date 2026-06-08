"""Generate intuition figures for the decision-tree opening slides.

Recreates the pedagogy of the MLU-explain "Decision Trees" article
(https://mlu-explain.github.io/decision-tree/) for absolute beginners:
a farmer classifies Apple / Cherry / Oak trees from trunk Diameter & Height.

All figures use transparent backgrounds and light strokes so they sit on the
dark Eclipse-Lab RevealJS theme. Run from the unit folder:

    ../../.venv/bin/python make_intuition_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.tree import DecisionTreeClassifier

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)

# --- Eclipse-Lab dark-theme palette -----------------------------------------
LIGHT = "#dbe4f0"
MUTED = "#94a3b8"
ACCENT = "#60a5fa"
GRID = (1, 1, 1, 0.10)
CLASSES = ["Apple", "Cherry", "Oak"]
COLORS = {"Apple": "#f87171", "Cherry": "#c084fc", "Oak": "#4ade80"}

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "text.color": LIGHT,
    "axes.labelcolor": LIGHT,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 15,
    "font.family": "sans-serif",
    "axes.titlesize": 17,
})


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(True, color="white", alpha=0.10, linewidth=0.8)
    ax.tick_params(length=0)


# --- synthetic "orchard" data, echoing the article's splits ------------------
def make_data(seed=3):
    rng = np.random.default_rng(seed)
    # Oak: large diameter (>= ~0.55)
    oak = np.column_stack([rng.uniform(0.55, 0.95, 26),
                           rng.uniform(1.5, 9.0, 26)])
    # Cherry: small diameter, short trunk (height <= ~4.8)
    cherry = np.column_stack([rng.uniform(0.07, 0.5, 24),
                              rng.uniform(0.6, 4.4, 24)])
    # Apple: small diameter, tall trunk
    apple = np.column_stack([rng.uniform(0.1, 0.5, 24),
                             rng.uniform(5.0, 9.2, 24)])
    X = np.vstack([apple, cherry, oak])
    y = np.array([0] * len(apple) + [1] * len(cherry) + [2] * len(oak))
    return X, y


def scatter(ax, X, y, legend=True):
    for k, name in enumerate(CLASSES):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=70, c=COLORS[name], edgecolors="white",
                   linewidths=0.7, label=name, zorder=3)
    ax.set_xlabel("Trunk diameter (m)")
    ax.set_ylabel("Trunk height (m)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 10)
    if legend:
        leg = ax.legend(loc="upper center", ncol=3, frameon=False,
                        bbox_to_anchor=(0.5, 1.12), handletextpad=0.2,
                        columnspacing=1.1)
        for t in leg.get_texts():
            t.set_color(LIGHT)


# 1) Raw data: the farmer's problem ------------------------------------------
def fig_data():
    X, y = make_data()
    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=160)
    scatter(ax, X, y)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "intuition_data.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


# 2) Recursive axis-aligned partition ----------------------------------------
def fig_partition():
    X, y = make_data()
    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=160)
    # shaded regions matching a depth-2 tree
    d_split = 0.52
    h_split = 4.7
    ax.axvspan(d_split, 1.0, color=COLORS["Oak"], alpha=0.12, zorder=0)
    ax.add_patch(plt.Rectangle((0, 0), d_split, h_split,
                 color=COLORS["Cherry"], alpha=0.12, zorder=0))
    ax.add_patch(plt.Rectangle((0, h_split), d_split, 10 - h_split,
                 color=COLORS["Apple"], alpha=0.12, zorder=0))
    # boundaries
    ax.axvline(d_split, color=ACCENT, lw=2.4, zorder=2)
    ax.plot([0, d_split], [h_split, h_split], color=ACCENT, lw=2.4, zorder=2)
    ax.text(d_split + 0.012, 9.4, "diameter ≥ 0.52  → Oak", color=LIGHT,
            fontsize=12.5, fontweight="bold")
    ax.text(0.02, h_split + 0.18, "height ≥ 4.7  → Apple", color=LIGHT,
            fontsize=12.5, fontweight="bold")
    ax.text(0.02, 0.4, "else → Cherry", color=LIGHT, fontsize=12.5,
            fontweight="bold")
    scatter(ax, X, y)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "intuition_partition.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


# 3) The corresponding tree diagram (hand-drawn for the dark theme) -----------
def _node(ax, xy, w, h, text, edge, fill="#111827"):
    box = FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         linewidth=2.2, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(box)
    ax.text(xy[0], xy[1], text, ha="center", va="center", color=LIGHT,
            fontsize=13, zorder=4)


def _arrow(ax, a, b, label, dx=0.0):
    ar = FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16,
                         color=MUTED, lw=1.8, zorder=2,
                         shrinkA=18, shrinkB=18)
    ax.add_patch(ar)
    mx, my = (a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2
    ax.text(mx, my, label, color=MUTED, fontsize=11, ha="center",
            va="center", backgroundcolor="none")


def fig_tree():
    fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    _node(ax, (5, 8.6), 4.2, 1.2, "diameter ≥ 0.52 ?", ACCENT)
    _node(ax, (8.0, 5.2), 2.6, 1.1, "Oak", COLORS["Oak"], "#14241a")
    _node(ax, (3.2, 5.2), 3.6, 1.2, "height ≥ 4.7 ?", ACCENT)
    _node(ax, (1.4, 1.8), 2.6, 1.1, "Cherry", COLORS["Cherry"], "#241a30")
    _node(ax, (5.0, 1.8), 2.6, 1.1, "Apple", COLORS["Apple"], "#2a1414")
    _arrow(ax, (5, 8.0), (7.6, 5.8), "yes", dx=0.5)
    _arrow(ax, (5, 8.0), (3.6, 5.8), "no", dx=-0.4)
    _arrow(ax, (3.0, 4.6), (1.6, 2.4), "yes", dx=-0.45)
    _arrow(ax, (3.4, 4.6), (4.8, 2.4), "no", dx=0.45)
    fig.tight_layout()
    fig.savefig(OUT / "intuition_tree.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


# 4) Purity / impurity bubbles with entropy values ---------------------------
def _entropy(counts):
    p = np.array(counts, float)
    p = p / p.sum()
    p = p[p > 0]
    return abs(float(-(p * np.log2(p)).sum()))


def fig_entropy():
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.4), dpi=160)
    panels = [
        ("Pure node", [10, 0]),
        ("Mostly one class", [8, 2]),
        ("Maximally mixed", [5, 5]),
    ]
    rng = np.random.default_rng(0)
    cols = [COLORS["Apple"], COLORS["Cherry"]]
    for ax, (title, counts) in zip(axes, panels):
        pts = []
        labs = []
        for ci, n in enumerate(counts):
            pts += [ci] * n
        rng.shuffle(pts)
        # arrange dots in a circle
        n = len(pts)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = 0.7 + 0.06 * (np.arange(n) % 3)
        xs, ys = r * np.cos(ang), r * np.sin(ang)
        for x, y, c in zip(xs, ys, pts):
            ax.scatter(x, y, s=150, c=cols[c], edgecolors="white", linewidths=0.8)
        ax.add_patch(plt.Circle((0, 0), 1.05, fill=False, color=MUTED, lw=1.4, ls="--"))
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.7, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color=LIGHT, pad=6)
        ax.text(0, -1.45, f"entropy = {_entropy(counts):.2f}", ha="center",
                color=ACCENT, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "intuition_entropy.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


# 5) Information gain vs split threshold -------------------------------------
def fig_infogain():
    X, y = make_data()
    d = X[:, 0]
    # treat as Oak-vs-rest along diameter to mirror the article's first split
    y_bin = (y == 2).astype(int)
    parent = _entropy(np.bincount(y_bin))
    thr = np.linspace(d.min() + 0.02, d.max() - 0.02, 200)
    gain = []
    for t in thr:
        left, right = y_bin[d <= t], y_bin[d > t]
        if len(left) == 0 or len(right) == 0:
            gain.append(0)
            continue
        e = (len(left) * _entropy(np.bincount(left, minlength=2)) +
             len(right) * _entropy(np.bincount(right, minlength=2))) / len(y_bin)
        gain.append(parent - e)
    gain = np.array(gain)
    best = thr[int(gain.argmax())]
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=160)
    ax.plot(thr, gain, color=ACCENT, lw=3, zorder=3)
    ax.axvline(best, color="#f8fafc", ls="--", lw=1.6, zorder=2)
    ax.scatter([best], [gain.max()], s=120, c="#f8fafc", zorder=4)
    ax.annotate(f"best split\ndiameter = {best:.2f}",
                xy=(best, gain.max()), xytext=(best - 0.34, gain.max() * 0.62),
                color=LIGHT, fontsize=13,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("Candidate split threshold (diameter)")
    ax.set_ylabel("Information gain")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, gain.max() * 1.25)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "intuition_infogain.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


# 6) Instability: tiny data change -> very different tree --------------------
def make_overlap_data(seed=1):
    """Heavily overlapping classes + few points, so a full-depth tree memorises
    each point and the box patchwork is very sensitive to small data changes."""
    rng = np.random.default_rng(seed)
    n = 22
    a = rng.normal([0.46, 4.8], [0.20, 2.3], size=(n, 2))
    b = rng.normal([0.54, 5.4], [0.20, 2.3], size=(n, 2))
    X = np.vstack([a, b])
    X[:, 0] = np.clip(X[:, 0], 0.03, 0.97)
    X[:, 1] = np.clip(X[:, 1], 0.3, 9.7)
    y = np.array([0] * n + [1] * n)
    return X, y


def _boundary2(ax, X, y, title):
    clf = DecisionTreeClassifier(random_state=0).fit(X, y)  # full depth
    xx, yy = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 10, 400))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cmap = matplotlib.colors.ListedColormap([COLORS["Apple"], COLORS["Oak"]])
    ax.contourf(xx, yy, Z, alpha=0.18, cmap=cmap, levels=[-0.5, 0.5, 1.5])
    for k, c in enumerate([COLORS["Apple"], COLORS["Oak"]]):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=46, c=c, edgecolors="white",
                   linewidths=0.6, zorder=3)
    ax.set_xlabel("Trunk diameter (m)")
    ax.set_ylabel("Trunk height (m)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 10)
    ax.set_title(title, color=LIGHT, pad=6)
    style_ax(ax)


def fig_instability():
    X, y = make_overlap_data()
    rng = np.random.default_rng(4)
    # perturb ~10% of points -> a fully-grown tree reshuffles its boxes a lot
    Xp = X.copy()
    idx = rng.choice(len(X), size=max(4, len(X) // 10), replace=False)
    Xp[idx, 0] += rng.normal(0, 0.10, len(idx))
    Xp[idx, 1] += rng.normal(0, 1.4, len(idx))
    Xp = np.clip(Xp, [0.03, 0.3], [0.97, 9.7])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=160)
    _boundary2(axes[0], X, y, "Original data")
    _boundary2(axes[1], Xp, y, "10% of points nudged")
    # mark the moved points so the cause of the change is visible
    axes[1].scatter(Xp[idx, 0], Xp[idx, 1], s=150, facecolors="none",
                    edgecolors="#fbbf24", linewidths=2.0, zorder=5)
    fig.tight_layout()
    fig.savefig(OUT / "intuition_instability.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


# 7) Impurity measures: Gini vs entropy vs misclassification ------------------
def fig_impurity_curves():
    p = np.linspace(1e-6, 1 - 1e-6, 400)
    gini = 2 * p * (1 - p)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p)) / 2  # scaled to max 0.5
    misclass = np.minimum(p, 1 - p)
    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=160)
    ax.plot(p, entropy, color="#c084fc", lw=3, label="Entropy (scaled)")
    ax.plot(p, gini, color=ACCENT, lw=3, label="Gini")
    ax.plot(p, misclass, color="#f87171", lw=3, label="Misclassification")
    ax.axvline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.set_xlabel(r"Class proportion $p$")
    ax.set_ylabel("Impurity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.55)
    leg = ax.legend(frameon=False, loc="lower center", ncol=3,
                    bbox_to_anchor=(0.5, 1.01), columnspacing=1.3)
    for t in leg.get_texts():
        t.set_color(LIGHT)
    ax.annotate("misclassification is\npiecewise-linear → 'blind'\nto sub-majority gains",
                xy=(0.18, misclass[80]), xytext=(0.02, 0.40), color=LIGHT,
                fontsize=11.5, arrowprops=dict(arrowstyle="->", color=MUTED))
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "impurity_curves.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


# 8) Bagging/RF smooths the decision boundary --------------------------------
def fig_rf_smoothing():
    from sklearn.datasets import make_moons
    from sklearn.ensemble import RandomForestClassifier
    X, y = make_moons(n_samples=240, noise=0.30, random_state=3)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 400),
                         np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    cmap = matplotlib.colors.ListedColormap([COLORS["Apple"], COLORS["Oak"]])
    models = [("Single full-depth tree", DecisionTreeClassifier(random_state=0)),
              ("Random forest (200 trees)",
               RandomForestClassifier(n_estimators=200, random_state=0))]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), dpi=160)
    for ax, (title, clf) in zip(axes, models):
        clf.fit(X, y)
        Z = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.18, cmap=cmap, levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, levels=[0.5], colors=[ACCENT], linewidths=1.8)
        for k, c in enumerate([COLORS["Apple"], COLORS["Oak"]]):
            m = y == k
            ax.scatter(X[m, 0], X[m, 1], s=26, c=c, edgecolors="white",
                       linewidths=0.5, zorder=3)
        ax.set_title(title, color=LIGHT, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "rf_smoothing.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


# 9) Gradient boosting: stagewise fit ----------------------------------------
def fig_boosting_stages():
    from sklearn.ensemble import GradientBoostingRegressor
    rng = np.random.default_rng(0)
    x = np.linspace(0, 6, 80)
    y_true = np.sin(2 * x) + 0.3 * x
    y = y_true + rng.normal(0, 0.3, x.size)
    X = x.reshape(-1, 1)
    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.3,
                                    max_depth=2, random_state=0).fit(X, y)
    stages = list(gbr.staged_predict(X))
    picks = [1, 3, 10, 50]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), dpi=160)
    for ax, m in zip(axes.ravel(), picks):
        pred = stages[m - 1]
        ax.scatter(x, y, s=18, c=ACCENT, alpha=0.45, zorder=2, edgecolors="none")
        ax.plot(x, y_true, color=MUTED, ls="--", lw=1.8, zorder=3)
        ax.plot(x, pred, color="#4ade80", lw=2.8, zorder=4)
        rmse = np.sqrt(np.mean((y - pred) ** 2))
        ax.set_title(f"{m} tree" + ("s" if m > 1 else "") +
                     f"   ·   train RMSE = {rmse:.2f}", color=LIGHT,
                     fontsize=13, pad=4)
        ax.set_xlim(0, 6); ax.set_ylim(-2, 3)
        ax.set_xticks([]); ax.set_yticks([])
        style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "boosting_stages.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


# 10) Why trees beat DL: axis-aligned vs rotated -----------------------------
def fig_trees_vs_dl():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(2)
    n = 130
    # axis-aligned structure: class depends on a vertical-ish step
    X = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)])
    y = (X[:, 0] > 0.0).astype(int)
    flip = rng.random(n) < 0.06
    y[flip] ^= 1  # a little label noise near the boundary

    def panel(ax, Xd, title):
        clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xd, y)
        pad = 0.6
        xx, yy = np.meshgrid(
            np.linspace(Xd[:, 0].min() - pad, Xd[:, 0].max() + pad, 400),
            np.linspace(Xd[:, 1].min() - pad, Xd[:, 1].max() + pad, 400))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        cmap = matplotlib.colors.ListedColormap([COLORS["Apple"], COLORS["Oak"]])
        ax.contourf(xx, yy, Z, alpha=0.18, cmap=cmap, levels=[-0.5, 0.5, 1.5])
        ax.contour(xx, yy, Z, levels=[0.5], colors=[ACCENT], linewidths=1.6)
        for k, c in enumerate([COLORS["Apple"], COLORS["Oak"]]):
            m = y == k
            ax.scatter(Xd[m, 0], Xd[m, 1], s=26, c=c, edgecolors="white",
                       linewidths=0.5, zorder=3)
        ax.set_title(title, color=LIGHT, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_color(MUTED)

    theta = np.deg2rad(45)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    Xr = X @ R.T
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4), dpi=160)
    panel(axes[0], X, "Original axes — boundary is axis-aligned")
    panel(axes[1], Xr, "Same data rotated 45° — forced staircase")
    fig.tight_layout()
    fig.savefig(OUT / "trees_vs_dl.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


# 11) Materials example: R^2 + SHAP bars -------------------------------------
def fig_materials_bars():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=160)
    models = ["Linear\n(quad. feats)", "Random\nForest", "XGBoost"]
    r2 = [0.62, 0.84, 0.88]
    bar_cols = [MUTED, ACCENT, "#4ade80"]
    bars = ax1.bar(models, r2, color=bar_cols, edgecolor="white", linewidth=0.6,
                   width=0.62)
    for b, v in zip(bars, r2):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                 ha="center", color=LIGHT, fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel(r"Held-out $R^2$")
    ax1.set_title("Yield-strength prediction", color=LIGHT, pad=6)
    style_ax(ax1)

    feats = ["Mo fraction", "C fraction", "Anneal temp.", "Cr fraction"]
    shap = [0.09, 0.14, 0.22, 0.31]
    ax2.barh(feats, shap, color=ACCENT, edgecolor="white", linewidth=0.6,
             height=0.6)
    ax2.set_xlabel("mean |SHAP| (impact on prediction)")
    ax2.set_title("Top features (SHAP, not Gini)", color=LIGHT, pad=6)
    ax2.set_xlim(0, 0.36)
    style_ax(ax2)
    fig.tight_layout()
    fig.savefig(OUT / "materials_bars.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


# 12) Permutation vs impurity importance: the noise-feature trap ------------
def fig_perm_importance():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(1)
    n = 400                              # small -> deep trees overfit the noise
    phase = rng.integers(0, 2, n)       # strong but LOW-cardinality (binary) driver
    anneal = rng.uniform(600, 1100, n)  # moderate driver
    c_frac = rng.uniform(0, 0.02, n)    # weak driver
    mo = rng.uniform(0, 0.05, n)        # very weak driver
    noise = rng.uniform(0, 1, n)        # pure noise, HIGH-cardinality continuous
    y = (120 * phase + 0.18 * anneal + 3000 * c_frac
         + rng.normal(0, 45, n))        # noise feature does NOT enter y
    X = np.column_stack([phase, anneal, c_frac, mo, noise])
    names = ["Phase (binary)", "Anneal temp.", "C fraction",
             "Mo fraction", "Random descriptor\n(pure noise)"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
    rf.fit(Xtr, ytr)
    mdi = rf.feature_importances_
    perm = permutation_importance(rf, Xte, yte, n_repeats=20,
                                  random_state=0, n_jobs=-1).importances_mean
    perm = np.clip(perm, 0, None)

    order = np.argsort(mdi)
    yps = np.arange(len(names))
    hi = names.index("Random descriptor\n(pure noise)")
    colA = [("#f87171" if order[i] == hi else ACCENT) for i in range(len(names))]
    colB = [("#f87171" if order[i] == hi else "#4ade80") for i in range(len(names))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0), dpi=160)
    ax1.barh(yps, mdi[order], color=colA, edgecolor="white", linewidth=0.6, height=0.62)
    ax1.set_yticks(yps); ax1.set_yticklabels([names[order[i]] for i in range(len(names))])
    ax1.set_title("Impurity (MDI) importance — biased", color=LIGHT, pad=6)
    ax1.set_xlabel("relative impurity reduction")
    style_ax(ax1)

    ax2.barh(yps, perm[order], color=colB, edgecolor="white", linewidth=0.6, height=0.62)
    ax2.set_yticks(yps); ax2.set_yticklabels([names[order[i]] for i in range(len(names))])
    ax2.set_title("Permutation importance — honest", color=LIGHT, pad=6)
    ax2.set_xlabel(r"drop in held-out $R^2$ when shuffled")
    style_ax(ax2)

    ax1.annotate("noise looks\nimportant!", xy=(mdi[hi], np.where(order == hi)[0][0]),
                 xytext=(mdi[hi] * 0.45, len(names) - 1.6), color="#fca5a5",
                 fontsize=12, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#fca5a5"))
    fig.tight_layout()
    fig.savefig(OUT / "perm_importance.png", transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_perm_importance()
    fig_data()
    fig_partition()
    fig_tree()
    fig_entropy()
    fig_infogain()
    fig_instability()
    fig_impurity_curves()
    fig_rf_smoothing()
    fig_boosting_stages()
    fig_trees_vs_dl()
    fig_materials_bars()
    print("wrote figures to", OUT)
