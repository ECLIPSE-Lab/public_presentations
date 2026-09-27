"""Reproducible figures for DSEM WS26/27 Week 4 (regression, optimisation & honest validation).

Generates the figures added in the WS26/27 update (the older figures in this folder
-- loss_landscape.png, lr_effects.png, optimizer_comparison.png, ... -- predate this script).

Run from repo root:
    .venv/bin/python data_science_for_em/04_regression_validation/img/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, lasso_path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
BLUE, RED, GREEN, ORANGE, GREY = "#2471a3", "#c0392b", "#27ae60", "#e67e22", "#7f8c8d"


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# 1. Convexity: chord definition (convex vs non-convex 1D loss)
# ---------------------------------------------------------------------------
def fig_convexity():
    x = np.linspace(-3, 3, 400)
    f_cvx = lambda t: 0.5 * t ** 2
    f_ncv = lambda t: 0.5 * t ** 2 + 1.5 * np.cos(2.2 * t) + 1.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, f, col, title in [(axes[0], f_cvx, BLUE, "Convex: chord always above the curve"),
                              (axes[1], f_ncv, RED, "Non-convex: chord dips below the curve")]:
        ax.plot(x, f(x), color=col, lw=3)
        a, b = (-2.4, 1.8) if f is f_cvx else (-1.9, 1.9)
        ax.plot([a, b], [f(a), f(b)], "k--", lw=2, label=r"chord $\lambda f(a)+(1-\lambda)f(b)$")
        ax.plot([a, b], [f(a), f(b)], "ko", ms=8)
        if f is f_ncv:
            m = x[(x > a) & (x < b)]
            chord = f(a) + (f(b) - f(a)) * (m - a) / (b - a)
            ax.fill_between(m, chord, f(m), where=f(m) > chord, color=RED, alpha=0.25,
                            label="curve above chord")
            for xm in [-1.35, 1.35]:
                ax.plot(xm, f(xm), "v", color=ORANGE, ms=12)
            ax.plot(0, f(0), "^", color=GREY, ms=11)
            ax.annotate("two local minima", (1.35, f(1.35)), (1.2, 5.0), fontsize=12,
                        arrowprops=dict(arrowstyle="->", color=ORANGE), color=ORANGE)
        else:
            ax.plot(0, 0, "*", color="gold", ms=20, mec="k", label="unique global minimum")
        ax.set_title(title, color=col, fontweight="bold")
        ax.set_xlabel(r"weight $w$"); ax.set_ylabel(r"$\hat R(w)$")
        ax.set_ylim(-0.5, 6.5); ax.grid(alpha=0.3); ax.legend(fontsize=11, loc="upper center")
    save(fig, "convexity_chord.png")


# ---------------------------------------------------------------------------
# 2. Non-convex landscape: saddle point + empirical vs true risk
# ---------------------------------------------------------------------------
def fig_nonconvex():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # (a) saddle f = w1^2 - w2^2 (+ small quartic so it is bounded) with GD path
    ax = axes[0]
    w1, w2 = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-2, 2, 300))
    f = lambda a, b: a ** 2 - b ** 2 + 0.25 * b ** 4
    grad = lambda a, b: np.array([2 * a, -2 * b + b ** 3])
    cs = ax.contourf(w1, w2, f(w1, w2), levels=30, cmap="RdBu_r", alpha=0.85)
    fig.colorbar(cs, ax=ax, shrink=0.85, label=r"$\hat R$")
    for w0, col, lab in [((1.8, 1e-4), "k", "start exactly on ridge: stalls at saddle"),
                         ((1.8, 0.08), GREEN, "tiny perturbation (SGD noise): escapes")]:
        w = np.array(w0, float); path = [w.copy()]
        for _ in range(60):
            w = w - 0.1 * grad(*w); path.append(w.copy())
        p = np.array(path)
        ax.plot(p[:, 0], p[:, 1], "o-", color=col, ms=3, lw=2, label=lab)
    ax.plot(0, 0, "X", color="yellow", mec="k", ms=15, label=r"saddle: $\nabla\hat R=0$")
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_title("Saddle point: zero gradient, not a minimum", fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    # (b) empirical risk vs risk (after d2l optimization-intro)
    ax = axes[1]
    x = np.linspace(0.5, 1.5, 400)
    risk = x * np.cos(np.pi * x)
    emp = risk + 0.2 * np.cos(5 * np.pi * x)
    ax.plot(x, risk, color=BLUE, lw=3, label=r"true risk $R(w)$ (new specimens)")
    ax.plot(x, emp, color=RED, lw=2, ls="--", label=r"empirical risk $\hat R(w)$ (training set)")
    i_e, i_r = np.argmin(emp), np.argmin(risk)
    ax.plot(x[i_e], emp[i_e], "v", color=RED, ms=14)
    ax.plot(x[i_r], risk[i_r], "v", color=BLUE, ms=14)
    ax.annotate(r"min of $\hat R$", (x[i_e], emp[i_e]), (0.62, -1.15), color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.annotate(r"min of $R$", (x[i_r], risk[i_r]), (1.2, -0.45), color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.set_xlabel("weight $w$"); ax.set_ylabel("loss")
    ax.set_title("Optimisation goal ≠ generalisation goal", fontweight="bold")
    ax.legend(fontsize=11, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "nonconvex_saddle_risk.png")


# ---------------------------------------------------------------------------
# 3. Ridge vs Lasso geometry
# ---------------------------------------------------------------------------
def fig_ridge_lasso_geometry():
    w_ols = np.array([1.6, 0.55])
    A = np.array([[1.0, 0.55], [0.55, 0.8]])  # correlated features -> tilted ellipses
    w1, w2 = np.meshgrid(np.linspace(-1.5, 2.5, 400), np.linspace(-1.5, 2.0, 400))
    d = np.stack([w1 - w_ols[0], w2 - w_ols[1]], -1)
    loss = np.einsum("...i,ij,...j->...", d, A, d)
    t = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, kind in zip(axes, ["ridge", "lasso"]):
        ax.contour(w1, w2, loss, levels=np.linspace(0.05, 4, 14), cmap="Greys_r", linewidths=1.2)
        # constrained minimum by brute force on the boundary of the ball
        th = np.linspace(0, 2 * np.pi, 4000)
        if kind == "ridge":
            bx, by = t * np.cos(th), t * np.sin(th)
            ax.fill(bx, by, color=BLUE, alpha=0.3)
            ttl, col = r"Ridge: $\|\mathbf{w}\|_2^2\leq t$ (disc) — shrinks, keeps all", BLUE
        else:
            bx = t * np.sign(np.cos(th)) * np.cos(th) ** 2
            by = t * np.sign(np.sin(th)) * np.sin(th) ** 2
            ax.fill(bx, by, color=RED, alpha=0.3)
            ttl, col = r"Lasso: $\|\mathbf{w}\|_1\leq t$ (diamond) — corner ⇒ $w_2=0$", RED
        db = np.stack([bx - w_ols[0], by - w_ols[1]], -1)
        lb = np.einsum("...i,ij,...j->...", db, A, db)
        k = np.argmin(lb)
        ax.contour(w1, w2, loss, levels=[lb[k]], colors=[col], linewidths=2.5)
        ax.plot(*w_ols, "k*", ms=18, label=r"$\hat{\mathbf{w}}_{\mathrm{OLS}}$")
        ax.plot(bx[k], by[k], "o", color=col, mec="k", ms=13,
                label=rf"constrained solution ({bx[k]:.2f}, {by[k]:.2f})")
        ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
        ax.set_aspect("equal"); ax.set_xlim(-1.5, 2.5); ax.set_ylim(-1.5, 2.0)
        ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
        ax.set_title(ttl, color=col, fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    save(fig, "ridge_lasso_geometry.png")


# ---------------------------------------------------------------------------
# Synthetic EELS-window dataset shared by figs 4 and 5
# ---------------------------------------------------------------------------
FEATS = ["Fe-L3", "Fe-L2", "O-K", "Mn-L3", "thickness", "bkg slope", "noise 1", "noise 2"]


def eels_windows(n=80, seed=0):
    """Integrated EELS window intensities (features) -> Fe content (target).
    Fe-L3 and Fe-L2 are strongly correlated (same element, same edge family)."""
    rng = np.random.default_rng(seed)
    fe = rng.uniform(0.1, 0.6, n)
    thick = rng.uniform(0.3, 1.2, n)
    L3 = fe * thick + rng.normal(0, 0.02, n)
    L2 = 0.5 * fe * thick + rng.normal(0, 0.012, n)   # ~ L3/2  -> collinear with L3
    O = (1 - fe) * thick + rng.normal(0, 0.03, n)
    Mn = rng.uniform(0, 0.2, n) * thick + rng.normal(0, 0.02, n)
    slope = -thick + rng.normal(0, 0.1, n)
    X = np.column_stack([L3, L2, O, Mn, thick, slope, rng.normal(size=n), rng.normal(size=n)])
    y = 100 * fe + rng.normal(0, 2.0, n)                 # at% Fe
    return X, y


def fig_reg_paths():
    X, y = eels_windows()
    Xs = StandardScaler().fit_transform(X); yc = y - y.mean()
    lams = np.logspace(-3, 3.5, 80)
    ridge_coefs = np.array([Ridge(alpha=a).fit(Xs, yc).coef_ for a in lams])
    la, lasso_coefs, _ = lasso_path(Xs, yc, alphas=np.logspace(-3, 1.5, 80))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cols = plt.cm.tab10(np.arange(len(FEATS)))
    for j, name in enumerate(FEATS):
        axes[0].semilogx(lams, ridge_coefs[:, j], color=cols[j], lw=2.2, label=name)
        axes[1].semilogx(la, lasso_coefs[j], color=cols[j], lw=2.2, label=name)
    axes[0].set_title("Ridge path: smooth shrinkage, nothing exactly 0", color=BLUE, fontweight="bold")
    axes[1].set_title("Lasso path: features drop out one by one", color=RED, fontweight="bold")
    for ax, lab in zip(axes, [r"$\lambda$ (Ridge alpha)", r"$\lambda$ (Lasso alpha)"]):
        ax.axhline(0, color="k", lw=0.8); ax.grid(alpha=0.3)
        ax.set_xlabel(lab + r"  →  stronger regularisation")
        ax.set_ylabel("standardised coefficient")
    axes[1].legend(fontsize=10, ncol=2, loc="upper right")
    fig.suptitle("Synthetic EELS-window regression (target: Fe at%), standardised features",
                 fontsize=13)
    fig.tight_layout()
    save(fig, "regularisation_paths.png")


def fig_coef_instability():
    """Bootstrap the coefficients of the two collinear Fe windows: OLS vs Ridge vs Lasso."""
    X, y = eels_windows(n=40, seed=3)
    rng = np.random.default_rng(1)
    models = {"OLS": LinearRegression(), "Ridge (λ=1)": Ridge(alpha=1.0),
              "Lasso (λ=0.5)": Lasso(alpha=0.5, max_iter=20000)}
    res = {k: [] for k in models}
    for _ in range(300):
        idx = rng.integers(0, len(y), len(y))
        sc = StandardScaler().fit(X[idx])
        for k, m in models.items():
            c = m.fit(sc.transform(X[idx]), y[idx]).coef_
            res[k].append(c[:2])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, (k, c), col in zip(axes, res.items(), [GREY, BLUE, RED]):
        c = np.array(c)
        ax.scatter(c[:, 0], c[:, 1], s=12, alpha=0.5, color=col)
        ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
        flip = np.mean(c[:, 0] < 0) * 100
        ax.set_title(f"{k}: coef(Fe-L3) < 0 in {flip:.0f}% of refits", fontsize=12,
                     color=col, fontweight="bold")
        ax.set_xlabel("coef Fe-L3 (standardised)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("coef Fe-L2 (standardised)")
    fig.suptitle("300 bootstrap refits, N = 40 spectra: collinear features trade weight "
                 "along a ridge", fontsize=13)
    fig.tight_layout()
    save(fig, "coef_instability.png")


# ---------------------------------------------------------------------------
# 6. Preprocessing leakage on pure noise
# ---------------------------------------------------------------------------
def fig_preproc_leakage():
    n, D, k, reps = 60, 2000, 20, 20
    rows = {"scaler+PCA outside CV": [], "scaler+PCA inside Pipeline": [],
            "SelectKBest outside CV": [], "SelectKBest inside Pipeline": []}
    for r in range(reps):
        rng = np.random.default_rng(100 + r)
        X = rng.normal(size=(n, D)); y = rng.normal(size=n)   # NO signal at all
        cv = KFold(5, shuffle=True, random_state=r)
        Xp = PCA(10).fit_transform(StandardScaler().fit_transform(X))
        rows["scaler+PCA outside CV"].append(cross_val_score(Ridge(1.0), Xp, y, cv=cv).mean())
        pipe = Pipeline([("s", StandardScaler()), ("p", PCA(10)), ("r", Ridge(1.0))])
        rows["scaler+PCA inside Pipeline"].append(cross_val_score(pipe, X, y, cv=cv).mean())
        Xk = SelectKBest(f_regression, k=k).fit_transform(X, y)
        rows["SelectKBest outside CV"].append(cross_val_score(Ridge(1.0), Xk, y, cv=cv).mean())
        pipe = Pipeline([("k", SelectKBest(f_regression, k=k)), ("r", Ridge(1.0))])
        rows["SelectKBest inside Pipeline"].append(cross_val_score(pipe, X, y, cv=cv).mean())
    fig, ax = plt.subplots(figsize=(10, 4.8))
    names = list(rows)
    means = [np.mean(rows[k]) for k in names]; stds = [np.std(rows[k]) for k in names]
    cols = [ORANGE, GREEN, RED, GREEN]
    ax.barh(names, means, xerr=stds, color=cols, capsize=5, edgecolor="k")
    for i, m in enumerate(means):
        ax.text(0.62, i, f"$R^2$ = {m:+.2f}", va="center", fontsize=13, fontweight="bold")
    ax.axvline(0, color="k", lw=1)
    ax.invert_yaxis(); ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(r"5-fold CV $R^2$ (mean ± std over 20 datasets)")
    ax.set_title(f"Pure noise: N={n} spectra, D={D} channels, target independent of X\n"
                 "Honest answer is R² ≤ 0 — anything above is leakage", fontsize=13)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    save(fig, "preproc_leakage.png")


if __name__ == "__main__":
    fig_convexity()
    fig_nonconvex()
    fig_ridge_lasso_geometry()
    fig_reg_paths()
    fig_coef_instability()
    fig_preproc_leakage()
