"""Figures for DSEM WS26/27 Week 11 (Uncertainty, GPs & autonomous EM) that are new in this update.

Run from repo root:  .venv/bin/python data_science_for_em/11_uncertainty_autonomous_em/img/make_figures.py

Generates
  cqr_vs_split.png     split conformal (constant width) vs CQR (adaptive width) — same setup as
                       notebooks/week11_gp_bo.ipynb Part B, so the numbers match the notebook.
  ood_trust_gate.png   Mahalanobis OOD score in a 2-D feature space + score histograms with a
                       95th-percentile threshold (the "trust gate").

The remaining figures in this folder come from the May-2026 Week 9/10 decks (GP, BO, RL figures),
ML-PC unit 12 (MetalDAM MC-dropout maps / operating curve, make_mcdropout_figures.py there),
the May-2026 trust deck (data_manifold.png) and Roccapriore et al., ACS Nano 16, 7605 (2022)
(roccapriore2022_*.png, reproduced with attribution).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

OUT = Path(__file__).resolve().parent
SEED = 42
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})


# --------------------------------------------------------------------------- CQR vs split
def cqr_figure():
    rng = np.random.default_rng(SEED)

    def grain_true(x):
        return 1.0 + 0.8 * np.sin(2.5 * np.pi * x) * np.exp(-x)

    def sample(n, lo=0.0, hi=1.0, rng=rng):
        x = rng.uniform(lo, hi, n)
        return x, grain_true(x) + rng.normal(0, 1, n) * (0.05 + 0.30 * x)

    x_tr, y_tr = sample(300)
    x_cal, y_cal = sample(500)
    x_te, y_te = sample(2000)
    alpha = 0.10

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=SEED)
    rf.fit(x_tr.reshape(-1, 1), y_tr)
    f = lambda x: rf.predict(np.asarray(x).reshape(-1, 1))

    def cq(s):
        n = len(s)
        return np.sort(s)[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1]

    q_naive = np.quantile(np.abs(y_tr - f(x_tr)), 1 - alpha)
    q = cq(np.abs(y_cal - f(x_cal)))
    cov = lambda lo, hi, y: np.mean((y >= lo) & (y <= hi))
    kw = dict(n_estimators=300, max_depth=2, learning_rate=0.05, min_samples_leaf=20, random_state=SEED)
    lo_m = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, **kw).fit(x_tr.reshape(-1, 1), y_tr)
    hi_m = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, **kw).fit(x_tr.reshape(-1, 1), y_tr)
    lc, hc = lo_m.predict(x_cal.reshape(-1, 1)), hi_m.predict(x_cal.reshape(-1, 1))
    qc = cq(np.maximum(lc - y_cal, y_cal - hc))
    lt, ht = lo_m.predict(x_te.reshape(-1, 1)) - qc, hi_m.predict(x_te.reshape(-1, 1)) + qc
    c_split = cov(f(x_te) - q, f(x_te) + q, y_te)
    c_naive = cov(f(x_te) - q_naive, f(x_te) + q_naive, y_te)
    c_cqr = cov(lt, ht, y_te)
    hi_mask = x_te > 0.8
    c_split_hi = cov(f(x_te[hi_mask]) - q, f(x_te[hi_mask]) + q, y_te[hi_mask])
    c_cqr_hi = cov(lt[hi_mask], ht[hi_mask], y_te[hi_mask])

    xs = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax in axes:
        ax.scatter(x_te[:500], y_te[:500], s=9, color="0.55", alpha=0.5, label="test data")
        ax.plot(xs, grain_true(xs), "--", color="#2E7D32", lw=1.5, label="true mean")
        ax.set_xlabel("normalised electron dose $x$")
    ax = axes[0]
    ax.fill_between(xs, f(xs) - q, f(xs) + q, color="#43A047", alpha=0.25,
                    label=f"split conformal: cov {c_split:.2f} (high dose {c_split_hi:.2f})")
    ax.plot(xs, f(xs), color="#0D47A1", lw=1.8, label="random-forest prediction")
    ax.plot(xs, f(xs) - q_naive, ":", color="#C62828", lw=1.6)
    ax.plot(xs, f(xs) + q_naive, ":", color="#C62828", lw=1.6, label=f"naive (training residuals): cov {c_naive:.2f}")
    ax.set_title("Split conformal — constant width $\\pm\\hat q$")
    ax.set_ylabel("grain-size estimate (arb. u.)")
    ax = axes[1]
    ax.fill_between(xs, lo_m.predict(xs.reshape(-1, 1)) - qc, hi_m.predict(xs.reshape(-1, 1)) + qc,
                    color="#8E24AA", alpha=0.25, label=f"CQR: cov {c_cqr:.2f} (high dose {c_cqr_hi:.2f})")
    ax.set_title("CQR — width adapts to heteroscedastic noise")
    for ax in axes:
        ax.legend(fontsize=11, loc="upper right")
        ax.set_ylim(-0.3, 2.6)
    fig.suptitle(f"Target coverage $1-\\alpha$ = 0.90  ($n_\\mathrm{{train}}$=300, $n_\\mathrm{{cal}}$=500, $n_\\mathrm{{test}}$=2000)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "cqr_vs_split.png", dpi=130)
    plt.close(fig)
    print(f"cqr: naive {c_naive:.3f} split {c_split:.3f} (hi {c_split_hi:.3f}) cqr {c_cqr:.3f} (hi {c_cqr_hi:.3f})")


# --------------------------------------------------------------------------- OOD gate
def ood_figure():
    rng = np.random.default_rng(SEED)
    cov_tr = np.array([[1.0, 0.7], [0.7, 1.0]])
    X_tr = rng.multivariate_normal([0, 0], cov_tr, 800)
    X_val = rng.multivariate_normal([0, 0], cov_tr, 400)
    X_new_mic = rng.multivariate_normal([2.2, -1.4], 0.35 * np.eye(2), 150)   # new detector / microscope
    X_contam = rng.multivariate_normal([-3.0, 2.5], 0.25 * np.eye(2), 60)     # contamination / damage

    mu = X_tr.mean(0)
    P = np.linalg.inv(np.cov(X_tr.T))
    maha = lambda X: np.sqrt(np.einsum("ij,jk,ik->i", X - mu, P, X - mu))
    tau = np.quantile(maha(X_val), 0.95)
    d_val, d_mic, d_con = maha(X_val), maha(X_new_mic), maha(X_contam)
    tpr = np.mean(np.concatenate([d_mic, d_con]) > tau)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    ax = axes[0]
    g = np.linspace(-5, 5, 300)
    GX, GY = np.meshgrid(g, g)
    D = maha(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
    ax.contourf(GX, GY, D <= tau, levels=[0.5, 1.5], colors=["#E3F2FD"])
    ax.contour(GX, GY, D, levels=[tau], colors="#1565C0", linewidths=2)
    ax.scatter(*X_tr.T, s=6, color="#1565C0", alpha=0.35, label="training features")
    ax.scatter(*X_new_mic.T, s=22, marker="x", color="#E65100", label="new detector (shifted contrast)")
    ax.scatter(*X_contam.T, s=26, marker="^", color="#C62828", label="contamination / beam damage")
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.set_xlabel("feature 1 (penultimate-layer embedding)"); ax.set_ylabel("feature 2")
    ax.set_title("Accept region: Mahalanobis distance $\\leq \\tau$")
    ax.legend(fontsize=10, loc="lower left")

    ax = axes[1]
    bins = np.linspace(0, 9, 46)
    ax.hist(d_val, bins, color="#1565C0", alpha=0.6, label="in-distribution (validation)")
    ax.hist(np.concatenate([d_mic, d_con]), bins, color="#E65100", alpha=0.6, label="OOD inputs")
    ax.axvline(tau, color="k", ls="--", lw=2)
    ax.text(tau + 0.15, ax.get_ylim()[1] * 0.85, f"$\\tau$ = 95th pct of ID\n→ 5% false refusals\n→ {100*tpr:.0f}% of OOD refused",
            fontsize=12)
    ax.set_xlabel("Mahalanobis distance $d_M(\\mathbf{z})$"); ax.set_ylabel("count")
    ax.set_title("Trust gate: refuse and route to a human if $d_M > \\tau$")
    ax.legend(fontsize=11, loc="upper right", bbox_to_anchor=(1.0, 0.6))
    fig.tight_layout()
    fig.savefig(OUT / "ood_trust_gate.png", dpi=130)
    plt.close(fig)
    print(f"ood: tau {tau:.2f}, OOD refused {tpr:.3f}")


if __name__ == "__main__":
    cqr_figure()
    ood_figure()
