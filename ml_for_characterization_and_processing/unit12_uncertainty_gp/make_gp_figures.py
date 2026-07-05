"""Generate the Case-study-A GP figures from the real Mantzoukas 2021 data
(data_mantzoukas2021_table3.csv) and the Case-study-C active-learning panels
(simulated L-PBF density surface, labeled as simulation on the slide).

Run from the unit folder with the repo venv:
    ../.venv/bin/python make_gp_figures.py
Writes PNGs into images/ and prints the fitted numbers quoted on the slides.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

rng = np.random.default_rng(0)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

BLUE, ORANGE, RED, GRAY = "#1f77b4", "#ff7f0e", "#d62728", "#666666"

# ---------------------------------------------------------------- data
rows = [l.split(",") for l in open("data_mantzoukas2021_table3.csv")
        if l[0].isdigit()]
T = np.array([float(r[0]) for r in rows])
y = np.array([float(r[2]) for r in rows])
temps = np.unique(T)
group_means = np.array([y[T == t].mean() for t in temps])
group_sds = np.array([y[T == t].std(ddof=1) for t in temps])
n_rep = np.array([(T == t).sum() for t in temps])

# replicate-based noise estimate (pooled within-group SD), fixed during MLE
sigma_n = np.sqrt(np.mean(group_sds**2))
print(f"group means (HV30): {np.round(group_means, 1)}")
print(f"group SDs   (HV30): {np.round(group_sds, 1)}")
print(f"pooled sigma_n = {sigma_n:.1f} HV30")

y_mean = y.mean()


def fit_gp_fixed_ell(T_fit, y_fit, ell):
    kernel = C(15.0**2, (2.0**2, 60.0**2)) * Matern(
        length_scale=ell, length_scale_bounds="fixed", nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n**2,
                                  n_restarts_optimizer=3,
                                  normalize_y=False, random_state=0)
    gp.fit(T_fit.reshape(-1, 1), y_fit - y_mean)
    return gp


# Free MLE runs ell to a degenerate 0 (the LML is nearly flat below the
# 25-degree sampling spacing), so we use MAP with a lognormal prior on ell
# centered at the spacing: ell ~ LogNormal(ln 25, 0.5). sigma_f stays MLE
# (profiled at each ell). This is the "informative-prior MAP, not free MLE"
# discipline the deck itself teaches.
def log_prior_ell(ell, mu=25.0, s=0.5):
    return (-np.log(ell * s * np.sqrt(2 * np.pi))
            - (np.log(ell / mu))**2 / (2 * s**2))


def fit_gp(T_fit, y_fit):
    ell_grid = np.exp(np.linspace(np.log(5), np.log(120), 25))
    best, best_obj = None, -np.inf
    for ell in ell_grid:
        g = fit_gp_fixed_ell(T_fit, y_fit, ell)
        obj = g.log_marginal_likelihood_value_ + log_prior_ell(ell)
        if obj > best_obj:
            best, best_obj, = g, obj
    return best


def predict(gp, Tq):
    mu, sd = gp.predict(Tq.reshape(-1, 1), return_std=True)
    return mu + y_mean, sd


gp = fit_gp(T, y)
k_opt = gp.kernel_
sigma_f = np.sqrt(k_opt.k1.constant_value)
ell = k_opt.k2.length_scale
lml = gp.log_marginal_likelihood_value_
print(f"fitted: sigma_f = {sigma_f:.1f} HV30, ell = {ell:.1f} C, "
      f"log marginal likelihood = {lml:.1f}")

Tq = np.linspace(460, 590, 400)
mu, sd = predict(gp, Tq)

# leave-one-specimen-out 95% coverage (real diagnostic quoted on slide 10)
hits = 0
for i in range(len(y)):
    m = np.ones(len(y), bool)
    m[i] = False
    gp_i = fit_gp_fixed_ell(T[m], y[m], ell)  # hyperparams from full fit
    mu_i, sd_i = predict(gp_i, np.array([T[i]]))
    tot = np.sqrt(sd_i[0]**2 + sigma_n**2)  # predictive: latent + noise
    hits += abs(y[i] - mu_i[0]) <= 1.96 * tot
print(f"LOO 95% predictive coverage: {hits}/{len(y)} = {hits/len(y):.0%}")


def plot_data(ax, Td=T, yd=y, means=True, color=BLUE):
    ax.plot(Td, yd, "o", ms=5, mfc="none", mec=color, alpha=0.55,
            label="specimens (HV30)")
    if means:
        sel = np.isin(temps, np.unique(Td))
        ax.errorbar(temps[sel], group_means[sel],
                    yerr=1.96 * group_sds[sel] / np.sqrt(n_rep[sel]),
                    fmt="D", ms=7, color=RED, capsize=4, zorder=5,
                    label="group mean ± 95% CI")


# --------------------------------------- Fig 1 (slide 08): the fit
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.fill_between(Tq, mu - 1.96 * np.sqrt(sd**2 + sigma_n**2),
                mu + 1.96 * np.sqrt(sd**2 + sigma_n**2),
                color=BLUE, alpha=0.12, label="95% predictive (incl. noise)")
ax.fill_between(Tq, mu - 1.96 * sd, mu + 1.96 * sd, color=BLUE, alpha=0.30,
                label="95% CI on mean response")
ax.plot(Tq, mu, color=BLUE, lw=2, label="GP posterior mean")
plot_data(ax)
ax.set_xlabel("Tempering temperature [°C]")
ax.set_ylabel("Hardness HV30")
ax.set_xlim(460, 590)
ax.set_title("GP fit — 1.7709 tempering data (Mantzoukas et al. 2021, n = 50)")
ax.legend(loc="upper right", fontsize=10)
fig.savefig("images/gp_17709_fit.png")
plt.close(fig)

# ------------------------- Fig 2 (slide 09): prior samples, two length scales
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
for ax, ell_i, tag in [(axes[0], ell, f"fitted  $\\ell$ = {ell:.0f} °C"),
                       (axes[1], 200.0, "oversmoothed  $\\ell$ = 200 °C")]:
    k = C(sigma_f**2) * Matern(length_scale=ell_i, nu=2.5)
    Kp = k(Tq.reshape(-1, 1)) + 1e-8 * np.eye(len(Tq))
    L = np.linalg.cholesky(Kp)
    for j in range(5):
        ax.plot(Tq, y_mean + L @ rng.standard_normal(len(Tq)), lw=1.4,
                alpha=0.85)
    ax.errorbar(temps, group_means, yerr=1.96 * group_sds / np.sqrt(n_rep),
                fmt="D", ms=6, color=RED, capsize=3, zorder=5)
    ax.set_title(tag, fontsize=13)
    ax.set_xlabel("Tempering temperature [°C]")
    ax.set_xlim(460, 590)
axes[0].set_ylabel("Hardness HV30")
fig.suptitle("Samples from the GP prior, $\\sigma_f$ = %.0f HV30 "
             "(red: measured group means)" % sigma_f, fontsize=13)
fig.tight_layout()
fig.savefig("images/gp_17709_prior_samples.png")
plt.close(fig)

# ----------------- Fig 3 (slide 10): held-out 525 °C group + posterior draws
hold = T == 525
gp_h = fit_gp(T[~hold], y[~hold])
mu_h, sd_h = predict(gp_h, Tq)
mu525, sd525 = predict(gp_h, np.array([525.0]))
print(f"held-out 525 C: GP predicts {mu525[0]:.0f} ± {1.96*sd525[0]:.0f} "
      f"(95%, mean response); measured group mean = "
      f"{y[hold].mean():.0f} ± {1.96*y[hold].std(ddof=1)/np.sqrt(10):.0f}")

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.fill_between(Tq, mu_h - 1.96 * sd_h, mu_h + 1.96 * sd_h, color=BLUE,
                alpha=0.25, label="95% CI on mean response")
Kpost = None
mu_v, cov_v = gp_h.predict(Tq.reshape(-1, 1), return_cov=True)
Lp = np.linalg.cholesky(cov_v + 1e-8 * np.eye(len(Tq)))
for j in range(4):
    ax.plot(Tq, y_mean + mu_v + Lp @ rng.standard_normal(len(Tq)),
            lw=1.0, color=GRAY, alpha=0.7,
            label="posterior samples" if j == 0 else None)
ax.plot(Tq, mu_h, color=BLUE, lw=2, label="GP posterior mean")
ax.plot(T[~hold], y[~hold], "o", ms=5, mfc="none", mec=BLUE, alpha=0.55,
        label="training specimens")
ax.errorbar([525], [y[hold].mean()],
            yerr=1.96 * y[hold].std(ddof=1) / np.sqrt(10), fmt="D", ms=8,
            color=RED, capsize=4, zorder=6, label="held-out 525 °C group")
ax.plot(T[hold], y[hold], "o", ms=5, mfc="none", mec=RED, alpha=0.6)
ax.set_xlabel("Tempering temperature [°C]")
ax.set_ylabel("Hardness HV30")
ax.set_xlim(460, 590)
ax.set_title("Held-out check: 525 °C group never seen by the GP")
ax.legend(loc="upper right", fontsize=10)
fig.savefig("images/gp_17709_heldout.png")
plt.close(fig)

# ------------------------------- Fig 4 (slide 11): honest extrapolation
Tx = np.linspace(390, 660, 500)
mu_x, sd_x = predict(gp, Tx)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.fill_between(Tx, mu_x - 1.96 * sd_x, mu_x + 1.96 * sd_x, color=BLUE,
                alpha=0.25, label="95% CI on mean response")
ax.plot(Tx, mu_x, color=BLUE, lw=2, label="GP posterior mean")
plot_data(ax, means=False)
ax.axvspan(390, 475, color=GRAY, alpha=0.12)
ax.axvspan(575, 660, color=GRAY, alpha=0.12)
ax.axhline(y_mean, color=GRAY, lw=1, ls="--", label="prior mean")
ax.annotate("no data:\nCI reverts to prior", xy=(620, y_mean + 38),
            ha="center", fontsize=11, color=GRAY)
ax.annotate("no data:\nCI reverts to prior", xy=(428, y_mean + 38),
            ha="center", fontsize=11, color=GRAY)
ax.set_xlabel("Tempering temperature [°C]")
ax.set_ylabel("Hardness HV30")
ax.set_xlim(390, 660)
ax.set_title("Outside 475–575 °C the GP honestly returns to the prior")
ax.legend(loc="lower left", fontsize=10)
fig.savefig("images/gp_17709_extrapolation.png")
plt.close(fig)

# =================================================================
# Case study C (slide 22): active learning on an L-PBF density
# surface grounded in REAL data — Diaz Vallejo et al. 2021, Metals
# 11(5):832 (doi:10.3390/met11050832, CC-BY): 35 cubes of 316L,
# P in {125..350} W, v in 100-3400 mm/s, layer 0.03 mm, hatch
# 0.12 mm. The ground truth is a GP fit to those 35 measurements
# (heteroscedastic noise from the reported replicate std); the AL
# campaign then queries that surface with sigma = 0.002 noise,
# which matches the dataset's median replicate std.
# Acquisition: straddle heuristic for the rho = 0.995 level set.
# =================================================================

dv_rows = [l.split(",") for l in open("data_diazvallejo2021_table1.csv")
           if l[0].isdigit()]
X_dv = np.array([[float(r[0]), float(r[1])] for r in dv_rows])
y_dv = np.array([float(r[2]) for r in dv_rows])
sd_dv = np.array([float(r[3]) for r in dv_rows])
print(f"\nDiaz Vallejo 2021: {len(y_dv)} cubes, "
      f"median replicate std = {np.median(sd_dv):.4f}")

gt_kernel = C(0.03**2, (1e-4, 1.0)) * Matern(
    length_scale=[100.0, 500.0],
    length_scale_bounds=[(40, 500), (150, 3000)], nu=2.5)
gt_gp = GaussianProcessRegressor(gt_kernel, alpha=np.maximum(sd_dv, 1e-4)**2,
                                 n_restarts_optimizer=8, random_state=0)
gt_gp.fit(X_dv, y_dv - 0.98)
print(f"ground-truth GP kernel: {gt_gp.kernel_}")

P_lim, v_lim = (125, 350), (100, 3400)
gP, gv = np.meshgrid(np.linspace(*P_lim, 90), np.linspace(*v_lim, 90))
Xgrid = np.column_stack([gP.ravel(), gv.ravel()])


def rho_rel(P, v):
    """Ground-truth relative density: GP mean fit to the 35 real cubes."""
    Pq = np.column_stack([np.ravel(P), np.ravel(v)])
    out = gt_gp.predict(Pq) + 0.98
    return out.reshape(np.shape(P)) if np.ndim(P) else float(out[0])


truth = rho_rel(gP, gv)
noise_al = 0.002  # measurement noise on rho_rel (= dataset's replicate std)


def fit_gp2d(X, z):
    kernel = C(0.03**2, (1e-4, 1.0)) * Matern(
        length_scale=[80.0, 400.0],
        length_scale_bounds=[(20, 500), (100, 3000)], nu=2.5)
    g = GaussianProcessRegressor(kernel=kernel, alpha=noise_al**2,
                                 n_restarts_optimizer=8, random_state=0)
    g.fit(X, z - 0.98)
    return g


# seed: 6 space-filling points (the anti-pattern slide says: seed first)
Xs = np.array([[150, 500], [150, 2900], [237, 1700],
               [325, 500], [325, 2900], [237, 250]], float)
zs = rho_rel(Xs[:, 0], Xs[:, 1]) + rng.normal(0, noise_al, len(Xs))

snapshots = {}
X_al, z_al = Xs.copy(), zs.copy()
for it in range(26):
    g2 = fit_gp2d(X_al, z_al)
    m2, s2 = g2.predict(Xgrid, return_std=True)
    m2 += 0.98
    if it in (0, 10, 25):
        snapshots[it] = (m2.copy(), s2.copy(), X_al.copy())
    if it == 25:
        break
    # straddle: high where the 0.995 level set is uncertain
    acq = 1.96 * s2 - np.abs(m2 - 0.995)
    Xn = Xgrid[np.argmax(acq)]
    X_al = np.vstack([X_al, Xn])
    z_al = np.append(z_al, rho_rel(Xn[0], Xn[1]) + rng.normal(0, noise_al))

for it, (m2, s2, Xa) in snapshots.items():
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    pc = ax.contourf(gP, gv, s2.reshape(gP.shape), levels=12, cmap="Greys")
    ax.contour(gP, gv, truth, levels=[0.995], colors=ORANGE,
               linewidths=2.5, linestyles="--")
    cs = ax.contour(gP, gv, m2.reshape(gP.shape), levels=[0.995],
                    colors=BLUE, linewidths=2.5)
    ax.plot(Xa[:6, 0], Xa[:6, 1], "s", color=RED, ms=7, mec="white",
            label="seed (6)")
    if len(Xa) > 6:
        ax.plot(Xa[6:, 0], Xa[6:, 1], "o", color=RED, ms=7, mec="white",
                label=f"AL picks ({len(Xa)-6})")
    ax.set_xlabel("Laser power P [W]")
    ax.set_ylabel("Scan speed v [mm/s]")
    n_exp = len(Xa)
    ax.set_title(f"{n_exp} experiments", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    cb = fig.colorbar(pc, ax=ax, shrink=0.85)
    cb.set_label("GP posterior σ", fontsize=10)
    fig.savefig(f"images/al_lpbf_iter{it:02d}.png")
    plt.close(fig)
    err = np.mean((m2 > 0.995) != (truth.ravel() > 0.995))
    print(f"AL iter {it:2d}: {len(Xa)} experiments, "
          f"window misclassification = {err:.1%}")

# ---------------------------------------------------------------
# Effort curve (slide 23): fraction of the true process window
# confidently identified vs number of experiments, AL vs grid,
# mean +- 1 std over 8 random campaigns (seed jitter + noise).
# ---------------------------------------------------------------
from scipy.special import ndtr

true_in = truth.ravel() > 0.995
print(f"true window area: {true_in.mean():.1%} of the (P, v) plane")


def fit_gp2d_fast(X, z):
    kernel = C(0.03**2, (1e-4, 1.0)) * Matern(
        length_scale=[80.0, 400.0],
        length_scale_bounds=[(20, 500), (100, 3000)], nu=2.5)
    g = GaussianProcessRegressor(kernel=kernel, alpha=noise_al**2,
                                 n_restarts_optimizer=2, random_state=0)
    g.fit(X, z - 0.98)
    return g


def window_found(g2):
    """Fraction of the true window area confidently identified:
    P[rho > 0.995] > 0.9 AND truly inside the window."""
    m2, s2 = g2.predict(Xgrid, return_std=True)
    m2 += 0.98
    p_in = ndtr((m2 - 0.995) / np.maximum(s2, 1e-9))
    return ((p_in > 0.9) & true_in).sum() / true_in.sum()


N_SEEDS, N_TOTAL = 8, 50


def run_al_campaign(seed):
    r = np.random.default_rng(seed)
    jit = np.column_stack([
        r.normal(0, 0.04 * (P_lim[1] - P_lim[0]), len(Xs)),
        r.normal(0, 0.04 * (v_lim[1] - v_lim[0]), len(Xs))])
    Xc = np.clip(Xs + jit, [P_lim[0], v_lim[0]], [P_lim[1], v_lim[1]])
    zc = rho_rel(Xc[:, 0], Xc[:, 1]) + r.normal(0, noise_al, len(Xc))
    ns, fr = [], []
    while len(Xc) <= N_TOTAL:
        g2 = fit_gp2d_fast(Xc, zc)
        ns.append(len(Xc))
        fr.append(window_found(g2))
        m2, s2 = g2.predict(Xgrid, return_std=True)
        acq = 1.96 * s2 - np.abs((m2 + 0.98) - 0.995)
        Xn = Xgrid[np.argmax(acq)]
        Xc = np.vstack([Xc, Xn])
        zc = np.append(zc, rho_rel(Xn[0], Xn[1]) + r.normal(0, noise_al))
    return np.array(ns), np.array(fr)


al_curves = np.array([run_al_campaign(s)[1] for s in range(N_SEEDS)])
al_ns = run_al_campaign(0)[0]

grid_ks = np.arange(3, 15)
grid_curves = np.zeros((N_SEEDS, len(grid_ks)))
for si in range(N_SEEDS):
    r = np.random.default_rng(100 + si)
    for ki, k in enumerate(grid_ks):
        gp_, gv_ = np.meshgrid(np.linspace(*P_lim, k), np.linspace(*v_lim, k))
        Xg = np.column_stack([gp_.ravel(), gv_.ravel()])
        zg = rho_rel(Xg[:, 0], Xg[:, 1]) + r.normal(0, noise_al, len(Xg))
        grid_curves[si, ki] = window_found(fit_gp2d_fast(Xg, zg))

al_m, al_s = al_curves.mean(0), al_curves.std(0)
gr_m, gr_s = grid_curves.mean(0), grid_curves.std(0)

fig, ax = plt.subplots(figsize=(8.4, 4.6))
ax.fill_between(al_ns, al_m - al_s, al_m + al_s, color=BLUE, alpha=0.2)
ax.plot(al_ns, al_m, color=BLUE, lw=2.5,
        label=f"active learning (straddle), mean ± 1σ over {N_SEEDS} seeds")
ax.fill_between(grid_ks**2, gr_m - gr_s, gr_m + gr_s, color=ORANGE, alpha=0.2)
ax.plot(grid_ks**2, gr_m, "o-", color=ORANGE, lw=2,
        label="grid sweep k×k + same GP")
ax.axhline(1.0, color=GRAY, lw=1, ls="--")
ax.axvline(31, color=GRAY, lw=1, ls=":")
ax.annotate("AL budget\n(31 cubes)", xy=(31, 0.18), ha="left", fontsize=10,
            color=GRAY, xytext=(34, 0.12))
ax.set_xlabel("Number of experiments (cubes printed)")
ax.set_ylabel("Fraction of true window\nconfidently identified")
ax.set_xlim(5, 200)
ax.set_ylim(0, 1.1)
ax.legend(loc="lower right", fontsize=10)
ax.set_title("Window discovery vs experimental effort "
             "(ground truth: Diaz Vallejo 2021 fit)")
fig.savefig("images/al_lpbf_effort_curve.png")
plt.close(fig)

i31 = np.argmin(np.abs(al_ns - 31))
print(f"AL @ {al_ns[i31]} cubes: {al_m[i31]:.1%} ± {al_s[i31]:.1%} of window")
for ki, k in enumerate(grid_ks):
    if gr_m[ki] >= al_m[i31]:
        print(f"grid needs {k}x{k} = {k*k} cubes to match ({gr_m[ki]:.1%})")
        break
else:
    print(f"grid does not match AL@31 even at 14x14=196 ({gr_m[-1]:.1%})")

print("done — figures written to images/")
