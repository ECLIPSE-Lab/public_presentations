"""Render the static real-data figures of the Week 4 deck (img/fig_*.png) and print their numbers.

Usage: python xp/prep/make_figs.py [name ...]   (reads xp/data/*.json; needs numpy, matplotlib, scikit-learn)
Without arguments the script makes all figures. The speaker notes quote the printed numbers.

Figures
  sgd       : SGD noise floor on the real background fit: step size, schedule and batch size (Act 3)
  basis     : a polynomial background is linear in its weights: basis functions and fits (Act 4)
  pooling   : validation error against the number of pooled pixels, degrees 0-3 (Act 4)
  lasso     : lasso on a degree-8 background: the validation curve chooses lambda (Act 5)
  logo      : error for each held-out specimen, random split against leave-one-specimen-out (Act 6)
  spread    : the spread of the score: 50 random hold-outs against 45 grouped hold-outs (Act 6)
  learning  : learning curves in pixels and in specimens (Act 6)
  baselines : mean baseline, grouped models, leaky model, calibrated model (Act 6)
  nested    : nested cross-validation: selection in the inner loop, score in the outer loop (Act 6)
  poisson   : no figure; prints the Poisson fit against the log least-squares fit at lower dose (Act 1 notes)
"""
import base64, itertools, json, sys, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error as mae, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
XP = Path(__file__).resolve().parents[1]
OUT = XP.parent / "img"
INK, MUTED, FAINT, PANEL = "#e2e8f0", "#94a3b8", "#475569", "#141928"
RED, BLUE, ORANGE, GREEN, YELLOW, PINK, VIOLET = "#f43f5e", "#60a5fa", "#fb923c", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"
SPEC_COL = ["#60a5fa", "#fb923c", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#22d3ee", "#f87171", "#a3e635", "#e2e8f0"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 21, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": FAINT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": PANEL, "axes.grid": True, "grid.color": "#94a3b8",
    "grid.alpha": 0.16, "axes.titlesize": 23, "axes.titlecolor": INK, "legend.frameon": False, "legend.fontsize": 19,
    "axes.titlepad": 12, "lines.linewidth": 3, "savefig.transparent": True, "axes.axisbelow": True,
})


def load(name):
    def dec(v):
        if isinstance(v, dict) and "b64" in v:
            dt = {"f4": "<f4", "u2": "<u2", "u1": "u1", "i2": "<i2"}[v["dtype"]]
            return np.frombuffer(base64.b64decode(v["b64"]), dt).reshape(v["shape"]).astype(float)
        if isinstance(v, dict):
            return {k: dec(w) for k, w in v.items()}
        if isinstance(v, list):
            return [dec(w) for w in v]
        return v
    return dec(json.loads((XP / "data" / f"{name}.json").read_text()))


def new(w=17.4, h=7.4):
    f = plt.figure(figsize=(w, h), dpi=110)
    f.patch.set_alpha(0)
    return f


def save(f, name):
    f.savefig(OUT / f"fig_{name}.png")
    plt.close(f)
    print(f"-> img/fig_{name}.png")


# ------------------------------------------------------------------------------------------------
# The pre-edge background data: the same pixels, windows and feature as the explainers (w-general.js)
# ------------------------------------------------------------------------------------------------
def preedge():
    d = load("preedge")
    E = d["E"]
    px = np.maximum(d["cube"][26:32, 33:38].reshape(-1, len(E)) * d["scale"], 1e-3)   # 30 pixels, CeVOx region
    tr, va = (E >= 470) & (E <= 496), (E >= 496.5) & (E <= 508)
    lx = np.log(E)
    u = (lx - lx[tr].mean()) / lx[tr].std()                                           # standardised log E
    return E, px, tr, va, u


def design(u, deg):
    return np.stack([u ** j for j in range(deg + 1)], 1)


# ------------------------------------------------------------------------------------------------
# sgd: the noise floor of SGD and three ways to lower it
# ------------------------------------------------------------------------------------------------
def fig_sgd():
    E, px, tr, va, _ = preedge()
    sel = (E >= 470) & (E <= 508)
    x, y = np.log(E[sel]), np.log(px.mean(0)[sel])
    X = np.c_[np.ones(sel.sum()), (x - x.mean()) / x.std()]                           # standardised: kappa = 1
    n = len(y)
    w_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    loss = lambda w: ((y - X @ w) ** 2).mean() / 2
    Lmin = loss(w_hat)
    w0 = np.array([y.mean() + 0.15, 0.0])                                              # the start of the explainer
    runs = [(f"full batch ({n} points), η = 0.5", n, lambda t: 0.5, BLUE),
            ("batch 4, η = 0.5", 4, lambda t: 0.5, PINK),
            ("batch 4, η = 0.05", 4, lambda t: 0.05, VIOLET),
            ("batch 16, η = 0.5", 16, lambda t: 0.5, YELLOW),
            ("batch 4, η = 0.5 / (1 + t/20)", 4, lambda t: 0.5 / (1 + t / 20), GREEN)]
    budget = 40000                                                                     # gradient evaluations of one point
    f = new()
    aL = f.add_axes([0.075, 0.14, 0.50, 0.76]); aW = f.add_axes([0.655, 0.14, 0.33, 0.76])
    print("sgd: n =", n, "Lmin =", f"{Lmin:.2e}", "start error =", f"{loss(w0) - Lmin:.2e}")
    for label, b, eta, colr in runs:
        curves, tail, path = [], [], None
        for seed in range(20):
            R = np.random.default_rng(seed)
            w, hist, cost, pts = w0.copy(), [], [], [w0.copy()]
            for t in range(budget // b):
                I = np.arange(n) if b == n else R.integers(0, n, b)
                w = w + eta(t) * X[I].T @ (y[I] - X[I] @ w) / b
                hist.append(loss(w) - Lmin); cost.append((t + 1) * b); pts.append(w.copy())
            curves.append(hist); tail.append(np.mean(hist[-len(hist) // 5:]))
            if seed == 0:
                path = np.array(pts)
        med = np.exp(np.mean(np.log(np.maximum(curves, 1e-16)), 0))                     # geometric mean of 20 runs
        aL.plot(cost, med, color=colr, label=label, lw=3)
        print(f"  {label:32s} floor (last 20 %) = {np.mean(tail):.1e}   first cost below 1e-4 x start: "
              f"{next((c for c, v in zip(cost, med) if v < 1e-4 * (loss(w0) - Lmin)), None)}")
        if b in (4, 16) or b == n:
            k = 400 if b != n else 30
            aW.plot(path[:k, 1], path[:k, 0], color=colr, lw=1.3, alpha=0.9, marker="o", ms=2.5)
    aL.set_xscale("log"); aL.set_yscale("log"); aL.set_ylim(1e-12, 1e2); aL.set_xlim(4, budget)
    aL.set_xlabel("gradient evaluations (data points processed)"); aL.set_ylabel("L(w) − L_min")
    aL.legend(loc="upper right", fontsize=17); aL.set_title("Constant step: a noise floor. Decaying step: no floor.")
    aW.plot(w_hat[1], w_hat[0], "x", color="white", ms=16, mew=3, zorder=9)
    aW.set_xlabel("w₁ (slope)"); aW.set_ylabel("w₀ (level)"); aW.set_title("Paths: the first 400 steps")
    aW.set_xlim(w_hat[1] - 0.05, w_hat[1] + 0.05); aW.set_ylim(w_hat[0] - 0.05, w_hat[0] + 0.18)
    save(f, "sgd")


# ------------------------------------------------------------------------------------------------
# basis: linear in the weights
# ------------------------------------------------------------------------------------------------
def fig_basis():
    E, px, tr, va, u = preedge()
    vis = (E >= 465) & (E <= 512)
    sp = np.log(px[7])
    f = new()
    aB = f.add_axes([0.065, 0.14, 0.40, 0.76]); aF = f.add_axes([0.555, 0.14, 0.43, 0.76])
    cols = [MUTED, BLUE, YELLOW, PINK]
    for j in range(4):
        aB.plot(E[vis], u[vis] ** j, color=cols[j], label=["φ₀ = 1", "φ₁ = u", "φ₂ = u²", "φ₃ = u³"][j])
    aB.axvspan(470, 496, color=GREEN, alpha=0.10); aB.axvspan(496.5, 508, color=YELLOW, alpha=0.12)
    aB.set_ylim(-6, 12); aB.set_xlabel("energy loss E (eV)"); aB.set_ylabel("basis function φⱼ(u)")
    aB.set_title("Features: powers of u = standardised log E"); aB.legend(loc="upper left", ncol=2)
    aB.text(483, -5.2, "train", color=GREEN, ha="center"); aB.text(502, -5.2, "validation", color=YELLOW, ha="center")
    aF.plot(E[vis], sp[vis], color="white", lw=2, label="one pixel, log I")
    for deg, colr, ls in [(1, GREEN, "-"), (3, PINK, "--")]:
        Phi = design(u, deg)
        w = np.linalg.lstsq(Phi[tr], sp[tr], rcond=None)[0]
        aF.plot(E[vis], (Phi @ w)[vis], color=colr, ls=ls, lw=3.5,
                label=f"degree {deg}: w = [" + ", ".join(f"{v:+.3f}" for v in w) + "]")
        print(f"basis: degree {deg} weights", np.round(w, 4), "val MSE", f"{((Phi[va] @ w - sp[va]) ** 2).mean():.1e}",
              "cond(Phi)", f"{np.linalg.cond(Phi[tr]):.1f}")
    aF.axvspan(470, 496, color=GREEN, alpha=0.10); aF.axvspan(496.5, 508, color=YELLOW, alpha=0.12)
    aF.set_xlabel("energy loss E (eV)"); aF.set_ylabel("log I"); aF.legend(loc="lower left", fontsize=18)
    aF.set_title("Prediction = Φ w: a weighted sum of the features")
    save(f, "basis")


# ------------------------------------------------------------------------------------------------
# pooling: more data lowers the variance, not the bias
# ------------------------------------------------------------------------------------------------
def fig_pooling():
    E, px, tr, va, u = preedge()
    R = np.random.default_rng(1)
    ns = [1, 2, 3, 5, 8, 12, 15]
    f = new(10.6, 7.4)
    ax = f.add_axes([0.14, 0.14, 0.83, 0.76])
    res = {}
    for deg, colr in zip(range(4), [MUTED, GREEN, YELLOW, PINK]):
        Phi = design(u, deg)
        err = []
        for n in ns:
            e = []
            for _ in range(400):
                idx = R.permutation(30)
                fit_px, ref_px = idx[:n], idx[n:]                  # the reference never contains the fitted pixels
                sp, ref = np.log(px[fit_px].mean(0)), np.log(px[ref_px].mean(0))
                w = np.linalg.lstsq(Phi[tr], sp[tr], rcond=None)[0]
                e.append(((Phi[va] @ w - ref[va]) ** 2).mean())
            err.append(np.mean(e))
        res[deg] = err
        ax.plot(ns, err, "o-", color=colr, ms=9, label=f"degree {deg}" + (" (power law)" if deg == 1 else ""))
        print(f"pooling: degree {deg}", " ".join(f"n={n}: {v:.1e}" for n, v in zip(ns, err)))
    ax.plot(ns, res[1][0] / np.array(ns), ":", color=INK, lw=2, label="∝ 1/N")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns); ax.minorticks_off()
    ax.set_xlabel("pixels pooled into the fitted spectrum, N"); ax.set_ylabel("validation MSE (log I)")
    ax.set_title("Degree 0: a bias floor. Degrees 1–3: variance ∝ 1/N.")
    ax.set_ylim(4e-6, 4e-2); ax.legend(loc="lower left", ncol=2)
    save(f, "pooling")


# ------------------------------------------------------------------------------------------------
# lasso: the validation curve chooses lambda
# ------------------------------------------------------------------------------------------------
def fig_lasso():
    E, px, tr, va, u = preedge()
    X = design(u, 8)[:, 1:]
    Z = (X - X[tr].mean(0)) / X[tr].std(0)                          # every column standardised on the training window
    lams = np.logspace(-5, -0.75, 35)
    T_, V_, NZ = [], [], []
    for lam in lams:
        t, v, nz = [], [], []
        for p in px:
            y = np.log(p)
            m = Lasso(alpha=lam, max_iter=200000, tol=1e-10).fit(Z[tr], y[tr])
            t.append(((m.predict(Z[tr]) - y[tr]) ** 2).mean()); v.append(((m.predict(Z[va]) - y[va]) ** 2).mean())
            nz.append((np.abs(m.coef_) > 1e-12).sum())
        T_.append(np.mean(t)); V_.append(np.mean(v)); NZ.append(np.mean(nz))
    ib = int(np.argmin(V_))
    ols1 = np.mean([((design(u, 1)[va] @ np.linalg.lstsq(design(u, 1)[tr], np.log(p)[tr], rcond=None)[0] - np.log(p)[va]) ** 2).mean() for p in px])
    ols8 = np.mean([((design(u, 8)[va] @ np.linalg.lstsq(design(u, 8)[tr], np.log(p)[tr], rcond=None)[0] - np.log(p)[va]) ** 2).mean() for p in px])
    print(f"lasso: best lambda = {lams[ib]:.1e}: train {T_[ib]:.1e}, val {V_[ib]:.1e}, non-zero terms {NZ[ib]:.1f}")
    print(f"       smallest lambda: val {V_[0]:.1e}, terms {NZ[0]:.1f}; largest lambda: val {V_[-1]:.1e}, terms {NZ[-1]:.1f}")
    print(f"       OLS degree 1: val {ols1:.1e}; OLS degree 8: val {ols8:.1e}")
    f = new()
    aE = f.add_axes([0.075, 0.38, 0.40, 0.52]); aN = f.add_axes([0.075, 0.14, 0.40, 0.20]); aF = f.add_axes([0.565, 0.14, 0.42, 0.76])
    aE.plot(lams, T_, color=GREEN, label="train"); aE.plot(lams, V_, color=YELLOW, label="validation")
    aE.axvline(lams[ib], color=RED, lw=2); aE.set_xscale("log"); aE.set_yscale("log"); aE.set_xticklabels([])
    aE.set_ylabel("MSE (log I)"); aE.legend(loc="center left"); aE.set_title("Degree-8 background + lasso, mean of 30 pixels")
    aE.text(lams[ib] * 1.25, 30, "best λ", color=RED)
    aN.step(lams, NZ, color=BLUE, where="mid"); aN.axvline(lams[ib], color=RED, lw=2); aN.set_xscale("log")
    aN.set_ylim(-0.4, 8.4); aN.set_yticks([0, 4, 8]); aN.set_xlabel("penalty λ"); aN.set_ylabel("terms ≠ 0")
    vis = (E >= 465) & (E <= 512)
    y = np.log(px[7])
    aF.plot(E[vis], y[vis], color="white", lw=2, label="one pixel, log I")
    for lam, colr, ls, name in [(lams[0], PINK, "--", "λ too small"), (lams[ib], RED, "-", "best λ"), (lams[-1], MUTED, ":", "λ too large")]:
        m = Lasso(alpha=lam, max_iter=200000, tol=1e-10).fit(Z[tr], y[tr])
        nz = np.flatnonzero(np.abs(m.coef_) > 1e-12) + 1
        aF.plot(E[vis], m.predict(Z)[vis], color=colr, ls=ls, lw=3.5, label=f"{name}: powers {list(nz) if len(nz) else 'none'}")
        print(f"       pixel 7, {name}: powers {list(nz)}")
    aF.axvspan(470, 496, color=GREEN, alpha=0.10); aF.axvspan(496.5, 508, color=YELLOW, alpha=0.12)
    lo, hi = y[vis].min(), y[vis].max()
    aF.set_ylim(lo - 0.35, hi + 0.2); aF.set_xlabel("energy loss E (eV)"); aF.set_ylabel("log I"); aF.legend(loc="lower left", fontsize=18)
    aF.set_title("The fits at three values of λ")
    save(f, "lasso")


# ------------------------------------------------------------------------------------------------
# The thickness data: ten specimens, 400 pixels each (the same as the split-leak explainer)
# ------------------------------------------------------------------------------------------------
def thick():
    t = load("thick")
    return t["g"].astype(int), t["t"], np.c_[t["adf"] / 1e6, t["ep"]], [s["name"] for s in t["specs"]]


knn = lambda k=5: make_pipeline(StandardScaler(), KNeighborsRegressor(k))
lin = lambda: make_pipeline(StandardScaler(), LinearRegression())


def calibrated(g, y, X, m, reps=20, seed=1):
    """A line t/λ = a + b·ADF for each specimen, fitted on m labelled pixels of that specimen."""
    R = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        P, test = np.zeros_like(y), np.ones_like(y, bool)
        for s in range(10):
            idx = np.flatnonzero(g == s)
            c = R.choice(idx, m, replace=False)
            w = np.linalg.lstsq(np.c_[np.ones(m), X[c, 0]], y[c], rcond=None)[0]
            P[idx] = w[0] + w[1] * X[idx, 0]; test[c] = False
        out.append((mae(y[test], P[test]), r2_score(y[test], P[test])))
    return np.mean(out, 0)


def wrap(n):
    n = n.replace(" / ", "/").replace(" (", "\n(").replace(", ", "\n")
    return "\n".join(p.replace(" ", "\n", 1) if len(p) > 11 else p for p in n.split("\n"))


def fig_logo():
    g, y, X, names = thick()
    p_rand = cross_val_predict(knn(), X, y, cv=KFold(5, shuffle=True, random_state=1))
    p_logo = cross_val_predict(knn(), X, y, groups=g, cv=LeaveOneGroupOut())
    base = np.zeros_like(y)
    for s in range(10):
        base[g == s] = y[g != s].mean()                             # the mean of the nine training specimens
    m_r = [mae(y[g == s], p_rand[g == s]) for s in range(10)]
    m_l = [mae(y[g == s], p_logo[g == s]) for s in range(10)]
    m_b = [mae(y[g == s], base[g == s]) for s in range(10)]
    r_l = [r2_score(y[g == s], p_logo[g == s]) for s in range(10)]
    print("logo: pooled random  R2 %.2f MAE %.3f" % (r2_score(y, p_rand), mae(y, p_rand)))
    print("      pooled grouped R2 %.2f MAE %.3f   mean baseline MAE %.3f" % (r2_score(y, p_logo), mae(y, p_logo), mae(y, base)))
    rmse = lambda a, b: np.sqrt(((a - b) ** 2).mean())
    print("      pooled RMSE random %.3f, grouped %.3f, baseline %.3f; sd of t/λ %.3f" % (rmse(y, p_rand), rmse(y, p_logo), rmse(y, base), y.std()))
    print("      per-specimen MAE random ", np.round(m_r, 2)); print("      per-specimen MAE grouped", np.round(m_l, 2))
    print("      per-specimen MAE baseline", np.round(m_b, 2)); print("      per-specimen R2 grouped ", np.round(r_l, 1))
    print("      specimens where the grouped model beats the baseline:", int(np.sum(np.array(m_l) < np.array(m_b))), "of 10")
    f = new()
    ax = f.add_axes([0.07, 0.27, 0.92, 0.63])
    xx = np.arange(10)
    ax.bar(xx - 0.2, m_r, 0.38, color=GREEN, label="random pixel split (5-fold)")
    ax.bar(xx + 0.2, m_l, 0.38, color=RED, label="this specimen held out")
    ax.hlines(m_b, xx - 0.42, xx + 0.42, color="white", lw=3.5, label="baseline: mean t/λ of the other nine", zorder=5)
    ax.set_xticks(xx); ax.set_xticklabels([wrap(n) for n in names], fontsize=16)
    for tick, c in zip(ax.get_xticklabels(), SPEC_COL):
        tick.set_color(c)
    ax.set_ylabel("MAE of t/λ on this specimen"); ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
    ax.set_title("5-NN thickness model: the error for each specimen")
    save(f, "logo")


def fig_spread():
    g, y, X, _ = thick()
    R = np.random.default_rng(0)
    r_rand = []
    for _ in range(50):
        idx = R.permutation(len(y)); te, tr = idx[:800], idx[800:]
        r_rand.append(r2_score(y[te], knn().fit(X[tr], y[tr]).predict(X[te])))
    r_grp = []
    for a, b in itertools.combinations(range(10), 2):
        te = (g == a) | (g == b)
        r_grp.append(r2_score(y[te], knn().fit(X[~te], y[~te]).predict(X[te])))
    r_rand, r_grp = np.array(r_rand), np.array(r_grp)
    print("spread: random 80/20 hold-out R2 = %.3f ± %.3f (50 splits)" % (r_rand.mean(), r_rand.std()))
    print("        grouped hold-out of 2 specimens (45 choices): median %.2f, quartiles %.2f / %.2f, min %.1f, max %.2f, above 0: %d of 45, below -3: %d"
          % (np.median(r_grp), *np.percentile(r_grp, [25, 75]), r_grp.min(), r_grp.max(), (r_grp > 0).sum(), (r_grp < -3).sum()))
    f = new()
    ax = f.add_axes([0.07, 0.14, 0.92, 0.76])
    bins = np.linspace(-3, 1, 41)
    ax.hist(np.clip(r_grp, -2.999, None), bins, color=RED, alpha=0.9, label="2 of 10 specimens held out (all 45 choices)")
    ax.hist(r_rand, bins, color=GREEN, alpha=0.9, label="random 20 % of the pixels held out (50 repeats)")
    ax.axvline(0, color="white", lw=2, ls="--"); ax.text(0.03, ax.get_ylim()[1] * 0.60, "R² = 0:\nthe mean\npredictor", color=INK, fontsize=18)
    ax.annotate(f"{(r_grp < -3).sum()} choices\nbelow −3\n(lowest {r_grp.min():.0f})", (-2.95, (r_grp < -3).sum()), (-2.75, ax.get_ylim()[1] * 0.55),
                color=RED, fontsize=18, arrowprops=dict(color=RED, arrowstyle="->"))
    ax.set_xlabel("R² of the held-out set"); ax.set_ylabel("number of splits"); ax.legend(loc="upper left"); ax.set_xlim(-3, 1.05)
    ax.set_title("The same 5-NN model, the same data: the score of many splits")
    save(f, "spread")


def fig_learning():
    g, y, X, _ = thick()
    R = np.random.default_rng(1)
    n_px = [20, 50, 100, 300, 1000, 3200]
    e_px = []
    for n in n_px:
        e = []
        for _ in range(30):
            idx = R.permutation(len(y)); te, tr = idx[:800], idx[800:800 + n]
            e.append(mae(y[te], knn().fit(X[tr], y[tr]).predict(X[te])))
        e_px.append(np.mean(e))
    n_sp = list(range(1, 10))
    e_sp = {"5-NN": [], "linear": []}
    for name, mk in [("5-NN", knn), ("linear", lin)]:
        for n in n_sp:
            e = []
            for te in range(10):
                others = [s for s in range(10) if s != te]
                for _ in range(12):
                    trs = R.choice(others, n, replace=False); m = np.isin(g, trs)
                    e.append(mae(y[g == te], mk().fit(X[m], y[m]).predict(X[g == te])))
            e_sp[name].append(np.mean(e))
    base = np.mean([mae(y[g == s], np.full((g == s).sum(), y[g != s].mean())) for s in range(10)])
    cal = calibrated(g, y, X, 5)
    print("learning: MAE against training pixels (random split):", dict(zip(n_px, np.round(e_px, 3))))
    for k, v in e_sp.items():
        print(f"          MAE against training specimens, {k}:", dict(zip(n_sp, np.round(v, 3))))
    print("          baseline (mean of the other specimens) MAE %.3f; calibrated with 5 pixels MAE %.3f R2 %.2f" % (base, *cal))
    f = new()
    aP = f.add_axes([0.07, 0.14, 0.43, 0.76]); aS = f.add_axes([0.555, 0.14, 0.43, 0.76])
    aP.plot(n_px, e_px, "o-", color=GREEN, ms=10); aP.set_xscale("log")
    aP.set_xlabel("training pixels (random split)"); aP.set_ylabel("MAE of t/λ on held-out data")
    aP.set_title("More pixels: the random-split error falls")
    aS.plot(n_sp, e_sp["5-NN"], "o-", color=RED, ms=10, label="5-NN"); aS.plot(n_sp, e_sp["linear"], "s--", color=ORANGE, ms=9, label="linear")
    aS.set_xlabel("training specimens (one new specimen held out)"); aS.set_title("More specimens: the honest error hardly moves")
    aS.set_xticks(n_sp); aS.legend(loc="lower left")
    for ax in (aP, aS):
        ax.axhline(base, color="white", ls="--", lw=2); ax.set_ylim(0, 0.5)
    aP.text(22, base + 0.012, "baseline: predict the mean", color=INK, fontsize=18)
    save(f, "learning")


def fig_baselines():
    g, y, X, _ = thick()
    base = np.zeros_like(y)
    for s in range(10):
        base[g == s] = y[g != s].mean()
    rows = [("baseline: mean t/λ of the training specimens", mae(y, base), MUTED, "new specimen"),
            ("linear model, specimens held out", mae(y, cross_val_predict(lin(), X, y, groups=g, cv=LeaveOneGroupOut())), ORANGE, "new specimen"),
            ("5-NN, specimens held out", mae(y, cross_val_predict(knn(), X, y, groups=g, cv=LeaveOneGroupOut())), RED, "new specimen"),
            ("5-NN, random pixel split", mae(y, cross_val_predict(knn(), X, y, cv=KFold(5, shuffle=True, random_state=1))), GREEN, "new pixel of a known specimen"),
            ("line per specimen, calibrated on 5 labelled pixels", calibrated(g, y, X, 5)[0], BLUE, "new pixel of a calibrated specimen")]
    for m in (2, 3, 5, 10, 20):
        print("baselines: calibrated with %d pixels: MAE %.3f, R2 %.2f" % (m, *calibrated(g, y, X, m)))
    f = new(17.4, 6.6)
    ax = f.add_axes([0.40, 0.17, 0.585, 0.71])
    for i, (name, v, colr, claim) in enumerate(rows):
        ax.barh(i, v, color=colr, height=0.62)
        ax.text(v + 0.006, i, f"{v:.3f}   claim: {claim}", va="center", color=INK, fontsize=19)
        print(f"baselines: {name:52s} MAE {v:.3f}  ({claim})")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=19, color=INK); ax.invert_yaxis()
    ax.set_xlim(0, 0.62); ax.set_xlabel("MAE of t/λ   (t/λ has a spread of 0.40 over all pixels)"); ax.grid(axis="y", visible=False)
    ax.set_title("One data set, five numbers: each belongs to a claim")
    save(f, "baselines")


def fig_nested():
    g, y, X, _ = thick()
    cfgs = [(f"{k}-NN, {fn}", make_pipeline(StandardScaler(), KNeighborsRegressor(k)), cols) for k in (1, 3, 5, 10, 20, 40)
            for fn, cols in (("ADF", [0]), ("ADF + Eₚ", [0, 1]))]
    cfgs += [("linear, ADF", lin(), [0]), ("linear, ADF + Eₚ", lin(), [0, 1])]

    def cv_mae(model, cols, groups_in):
        """Leave-one-specimen-out MAE over the specimens in groups_in (mean of the per-specimen MAE)."""
        e = []
        for te in groups_in:
            trm = np.isin(g, [s for s in groups_in if s != te])
            e.append(mae(y[g == te], model.fit(X[trm][:, cols], y[trm]).predict(X[g == te][:, cols])))
        return np.mean(e)

    allg = list(range(10))
    flat = [cv_mae(m, c, allg) for _, m, c in cfgs]
    ib = int(np.argmin(flat))
    print("nested: flat CV (select and report on the same folds): best = %s, MAE %.3f; worst MAE %.3f" % (cfgs[ib][0], flat[ib], max(flat)))
    outer, chosen = [], []
    for te in allg:
        inner = [s for s in allg if s != te]
        scores = [cv_mae(m, c, inner) for _, m, c in cfgs]
        j = int(np.argmin(scores))
        trm = g != te
        outer.append(mae(y[g == te], cfgs[j][1].fit(X[trm][:, cfgs[j][2]], y[trm]).predict(X[g == te][:, cfgs[j][2]])))
        chosen.append(cfgs[j][0])
    base = np.mean([mae(y[g == s], np.full((g == s).sum(), y[g != s].mean())) for s in allg])
    print("        nested CV (select inside, report outside): MAE %.3f; mean baseline %.3f" % (np.mean(outer), base))
    print("        chosen per outer fold:", chosen)
    # schematic: 10 specimens; outer fold holds out one; inner folds hold out one of the remaining nine
    f = new(17.4, 6.4)
    ax = f.add_axes([0.01, 0.02, 0.98, 0.9]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    def rowp(y0, test, val=None, hgt=3.4, x0=24, wd=4.3, dim=()):
        for s in range(10):
            role = RED if s == test else YELLOW if s == val else GREEN
            ax.add_patch(Rectangle((x0 + s * (wd + 0.5), y0), wd, hgt, color=role, alpha=0.15 if s in dim else 0.95, lw=0))
            ax.text(x0 + s * (wd + 0.5) + wd / 2, y0 + hgt / 2, str(s + 1), ha="center", va="center", color="#0b0f19", fontsize=17, weight="bold")
    ax.text(1, 36.6, "OUTER LOOP: the score", color=RED, fontsize=20, weight="bold")
    for i, te in enumerate([0, 1]):
        rowp(31 - i * 4.4, te); ax.text(1, 32.2 - i * 4.4, f"outer fold {te + 1}", fontsize=19)
    ax.text(24, 23.2, "⋮   ten outer folds: each specimen is the test set once", fontsize=19, color=MUTED)
    ax.text(1, 18.4, "INNER LOOP (inside outer fold 1): the choice of λ, k, features, model", color=YELLOW, fontsize=20, weight="bold")
    for i, v in enumerate([1, 2, 3]):
        rowp(12.8 - i * 4.4, None, v, dim=(0,)); ax.text(1, 14.0 - i * 4.4, f"inner fold {i + 1}", fontsize=19)
    ax.text(24, 0.6, "⋮   the test specimen (1) takes no part in the choice", fontsize=19, color=MUTED)
    for i, (colr, lab) in enumerate([(GREEN, "train: fit the weights"), (YELLOW, "validation: choose"), (RED, "test: report once")]):
        ax.add_patch(Rectangle((75, 31 - i * 4.4), 2.4, 2.4, color=colr, lw=0)); ax.text(78.6, 31.5 - i * 4.4, lab, fontsize=19)
    ax.text(75, 12.5, f"14 candidate models, real data\nflat CV, best model:  MAE {flat[ib]:.3f}\nnested CV:"
            f"  MAE {np.mean(outer):.3f}\nmean baseline:  MAE {base:.3f}", fontsize=18.5, color=INK, va="top", linespacing=1.5)
    save(f, "nested")


def check_poisson():
    """Print only: Poisson likelihood against least squares in log I for the power-law background, at lower dose."""
    from scipy.optimize import minimize
    E, px, tr, va, u = preedge()
    R = np.random.default_rng(0)

    def fit_log(c):
        ok = tr & (c > 0)
        w = np.linalg.lstsq(np.c_[np.ones(ok.sum()), u[ok]], np.log(c[ok]), rcond=None)[0]
        return np.exp(w[0] + w[1] * u)

    def fit_pois(c):
        f = lambda w: (np.exp(w[0] + w[1] * u[tr]) - c[tr] * (w[0] + w[1] * u[tr])).sum()
        g = lambda w: np.array([(np.exp(w[0] + w[1] * u[tr]) - c[tr]).sum(), ((np.exp(w[0] + w[1] * u[tr]) - c[tr]) * u[tr]).sum()])
        r = minimize(f, [np.log(max(c[tr].mean(), 1e-3)), 0.0], jac=g, method="BFGS")
        return np.exp(r.x[0] + r.x[1] * u)

    ref = fit_pois(px.sum(0)) / 30                                   # reference: the fit of the 30-pixel sum
    for frac in (1, 0.1, 0.01, 0.001, 0.0001):
        a, b = [], []
        for _ in range(1 if frac == 1 else 20):
            for p in px:
                c = p if frac == 1 else R.binomial(np.round(p).astype(int), frac).astype(float)   # thin the real counts
                a.append(fit_log(c)[va].sum() / (ref[va].sum() * frac)); b.append(fit_pois(c)[va].sum() / (ref[va].sum() * frac))
        print(f"poisson: {px[:, tr].mean() * frac:9.1f} counts/channel: background under the validation window, fit / reference: "
              f"log least squares {np.mean(a):.3f} ± {np.std(a):.3f}, Poisson {np.mean(b):.3f} ± {np.std(b):.3f}")


FIGS = {"poisson": check_poisson, "sgd": fig_sgd, "basis": fig_basis, "pooling": fig_pooling, "lasso": fig_lasso, "logo": fig_logo,
        "spread": fig_spread, "learning": fig_learning, "baselines": fig_baselines, "nested": fig_nested}
if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
