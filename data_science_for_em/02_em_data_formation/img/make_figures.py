"""Reproducible figures for DSEM Week 2 (signal formation, noise & loss).

Run from repo root:
    .venv/bin/python data_science_for_em/02_em_data_formation/img/make_figures.py

Produces (in this folder):
    data_formation_chain.png   specimen -> interaction -> detector -> digitisation -> metadata
    poisson_vs_mse_loss.png    per-pixel loss as a function of predicted rate, for small counts
    variance_mean_ptc.png      photon-transfer (variance-mean) plot of a Poisson-Gaussian detector
    anscombe_vst.png           std after Anscombe / generalised Anscombe transform vs mean count
    lowcount_fit_bias.png      fitting a low-count peak with MSE, weighted chi2, Anscombe+MSE, Poisson NLL
The existing figures (dose_snr_simulation.png, moire_simulation.png, fig1a_imaging.png) are not
regenerated here.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(2026)

# categorical palette (fixed order) + text inks
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#222222", "#555555", "#dddddd"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 150,
})


# --------------------------------------------------------------------------------------
# 1. Data-formation chain
# --------------------------------------------------------------------------------------
def fig_chain():
    stages = [
        ("Specimen", "structure, Z,\nthickness, strain", "beam damage,\ncontamination, drift"),
        ("Interaction", "elastic / inelastic\nscattering, X-rays", "multiple scattering,\ndynamical effects"),
        ("Detector", "HAADF, pixel camera,\nspectrometer", "shot noise, PSF,\nDQE, dead pixels"),
        ("Digitisation", "gain, ADC,\nsampling grid", "read noise, saturation,\nquantisation, aliasing"),
        ("Metadata", "kV, current, dwell,\npixel size, detector", "lost / wrong units\n= unusable data"),
    ]
    fig, ax = plt.subplots(figsize=(18, 5.2))
    ax.set_xlim(0, 18); ax.set_ylim(0, 5.2); ax.axis("off")
    w, h, gap = 3.0, 1.5, 0.6
    for i, (name, what, err) in enumerate(stages):
        x0 = 0.3 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x0, 2.6), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc="#e8f1fb" if i < 4 else "#fdf1e0",
                                    ec=C[0] if i < 4 else C[3], lw=2))
        ax.text(x0 + w / 2, 3.72, name, ha="center", va="center", fontsize=19, weight="bold", color=INK)
        ax.text(x0 + w / 2, 3.0, what, ha="center", va="center", fontsize=13, color=INK2)
        ax.text(x0 + w / 2, 1.55, err, ha="center", va="center", fontsize=13, color="#b3441c")
        ax.add_patch(FancyArrowPatch((x0 + w / 2, 2.5), (x0 + w / 2, 2.05), arrowstyle="-|>",
                                     mutation_scale=16, color="#b3441c", lw=1.5))
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w + 0.05, 3.35), (x0 + w + gap - 0.05, 3.35),
                                         arrowstyle="-|>", mutation_scale=22, color=INK2, lw=2))
    ax.text(0.3, 4.75, r"forward model:  $\mathbf{y} \sim p(\mathbf{y}\,|\,\mathcal{D}(\mathcal{Q}(h * f_\theta(\mathrm{specimen}))))$",
            fontsize=17, color=INK)
    ax.text(0.3, 0.55, "red: where errors / noise enter the chain  (each one must be modelled, calibrated or recorded)",
            fontsize=13, color="#b3441c")
    fig.savefig(OUT / "data_formation_chain.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 2. Poisson NLL vs MSE as a function of the prediction
# --------------------------------------------------------------------------------------
def fig_loss_curves():
    lam = np.linspace(0.02, 12, 600)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    for j, x in enumerate([0, 2, 8]):
        mse = (x - lam) ** 2
        pois = lam - x * np.log(lam)
        pois = pois - pois.min()
        axes[0].plot(lam, mse, color=C[j], ls="-", label=f"observed x = {x}")
        axes[1].plot(lam, pois, color=C[j], ls="-", label=f"observed x = {x}")
        axes[0].axvline(x, color=C[j], lw=1, ls=":")
        axes[1].axvline(max(x, 0.02), color=C[j], lw=1, ls=":")
    axes[0].set_title(r"MSE $(x-\hat\lambda)^2$: same penalty everywhere")
    axes[1].set_title(r"Poisson NLL $\hat\lambda - x\log\hat\lambda$ (shifted)")
    for ax in axes:
        ax.set_xlabel(r"predicted rate $\hat\lambda$ (counts)")
        ax.set_ylim(0, 30); ax.legend(loc="upper right")
    axes[0].set_ylabel("loss")
    axes[1].annotate(r"$\hat\lambda\to 0$ with $x>0$: loss $\to\infty$", xy=(0.15, 26), xytext=(2.2, 25),
                     fontsize=13, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2))
    fig.tight_layout()
    fig.savefig(OUT / "poisson_vs_mse_loss.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 3. Variance-mean (photon transfer) plot
# --------------------------------------------------------------------------------------
def fig_ptc():
    g, sr = 2.5, 6.0   # gain (DN per electron), read noise (DN)
    lam = np.logspace(-0.5, 3, 25)
    means, varis = [], []
    for l in lam:
        frames = g * rng.poisson(l, size=4000) + rng.normal(0, sr, size=4000)
        means.append(frames.mean()); varis.append(frames.var())
    means, varis = np.array(means), np.array(varis)
    slope, icpt = np.polyfit(means, varis, 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    mm = np.linspace(0, means.max(), 100)
    axes[0].plot(means, varis, "o", color=C[0], ms=8, label="simulated pixels")
    axes[0].plot(mm, slope * mm + icpt, color=C[1], label=f"fit: slope g = {slope:.2f}, intercept = {icpt:.0f}")
    axes[0].set_xlabel("mean signal (DN)"); axes[0].set_ylabel("variance (DN$^2$)")
    axes[0].set_title(r"Var$(x) = g\,\bar{x} + \sigma_r^2$  (true $g$=2.5, $\sigma_r^2$=36)")
    axes[0].legend(loc="upper left")
    axes[1].loglog(means, varis, "o", color=C[0], ms=8, label="total variance")
    axes[1].loglog(mm[1:], np.full(99, sr ** 2), color=C[2], ls="--", label=r"read noise $\sigma_r^2$")
    axes[1].loglog(mm[1:], g * mm[1:], color=C[1], ls="-.", label=r"shot noise $g\bar{x}$")
    axes[1].axvline(sr ** 2 / g, color=INK2, lw=1, ls=":")
    axes[1].text(sr ** 2 / g * 1.15, 3e3, "crossover\n" + r"$\bar x = \sigma_r^2/g$", fontsize=13, color=INK2)
    axes[1].set_xlabel("mean signal (DN)"); axes[1].set_ylabel("variance (DN$^2$)")
    axes[1].set_title("log-log: read-noise floor → shot-noise regime")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "variance_mean_ptc.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 4. Anscombe / generalised Anscombe variance stabilisation
# --------------------------------------------------------------------------------------
def fig_anscombe():
    lam = np.logspace(-1, 2, 40)
    n = 200_000
    raw, ans, gat = [], [], []
    g, sr = 1.0, 2.0
    for l in lam:
        k = rng.poisson(l, n)
        raw.append(k.std())
        ans.append((2 * np.sqrt(k + 3 / 8)).std())
        z = g * k + rng.normal(0, sr, n)
        gat.append((2 / g * np.sqrt(np.maximum(g * z + 3 / 8 * g ** 2 + sr ** 2, 0))).std())
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    axes[0].semilogx(lam, raw, "o-", color=C[0], ms=6, label=r"raw counts: std $=\sqrt{\lambda}$")
    axes[0].semilogx(lam, ans, "s-", color=C[1], ms=6, label=r"Anscombe $2\sqrt{x+3/8}$")
    axes[0].semilogx(lam, gat, "^-", color=C[2], ms=6, label=r"generalised Anscombe ($\sigma_r$=2)")
    axes[0].axhline(1, color=INK2, lw=1, ls=":")
    axes[0].set_ylim(0, 4); axes[0].set_xlabel(r"mean count $\lambda$"); axes[0].set_ylabel("standard deviation")
    axes[0].set_title("Variance stabilisation: std → 1 for λ ≳ 4")
    axes[0].legend(loc="upper left")
    # bias of the naive algebraic inverse
    naive_bias = []
    for l in lam:
        k = rng.poisson(l, n)
        t = 2 * np.sqrt(k + 3 / 8)
        naive_bias.append(((t.mean() / 2) ** 2 - 3 / 8) / l - 1)
    axes[1].semilogx(lam, 100 * np.array(naive_bias), "o-", color=C[3], ms=6,
                     label="naive inverse of the mean")
    axes[1].axhline(0, color=INK2, lw=1)
    axes[1].set_xlabel(r"mean count $\lambda$"); axes[1].set_ylabel("relative bias of recovered mean (%)")
    axes[1].set_title("…but the back-transform is biased at low counts")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "anscombe_vst.png", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 5. Low-count peak fit: MSE vs weighted chi2 vs Anscombe+MSE vs Poisson NLL
#    (same experiment as Part 9 of notebooks/week02_poisson_noise.ipynb)
# --------------------------------------------------------------------------------------
E = np.linspace(0, 1, 64)
TRUE = dict(a=1.0, b=0.05, s=0.05)


def peak_model(q, dose):
    a, b, s = q
    return dose * (a * np.exp(-(E - 0.5) ** 2 / (2 * s ** 2)) + b)


def fit_peak(x, dose, kind):
    def loss(q):
        lam = peak_model(q, dose)
        if kind == "MSE":
            return np.sum((x - lam) ** 2) / dose ** 2
        if kind == "weighted χ²":
            return np.sum((x - lam) ** 2 / np.maximum(x, 1)) / dose
        if kind == "Anscombe + MSE":
            return np.sum((2 * np.sqrt(x + 3 / 8) - 2 * np.sqrt(lam + 3 / 8)) ** 2)
        lam = np.maximum(lam, 1e-12)
        return np.sum(lam - x * np.log(lam)) / dose
    r = minimize(loss, [1.2, 0.1, 0.06], method="L-BFGS-B",
                 bounds=[(0, None), (1e-6, None), (0.01, 0.3)])
    return r.x


KINDS = ["MSE", "weighted χ²", "Anscombe + MSE", "Poisson NLL"]
STY = [("o", "-"), ("s", "--"), ("^", "-."), ("D", "-")]


def fig_lowcount():
    doses = np.array([2, 5, 10, 20, 50, 200])
    reps = 300
    q_true = np.array([TRUE["a"], TRUE["b"], TRUE["s"]])
    tot_true = peak_model(q_true, 1).sum()
    bias_tot = {k: [] for k in KINDS}
    std_s = {k: [] for k in KINDS}
    for d in doses:
        est = {k: [] for k in KINDS}
        for _ in range(reps):
            x = rng.poisson(peak_model(q_true, d))
            for k in KINDS:
                est[k].append(fit_peak(x, d, k))
        for k in KINDS:
            e = np.array(est[k])
            tot = np.array([peak_model(q, 1).sum() for q in e])
            bias_tot[k].append(100 * (tot.mean() / tot_true - 1))
            std_s[k].append(100 * e[:, 2].std() / TRUE["s"])
    # example spectrum at dose 5
    d = 5
    x = rng.poisson(peak_model(q_true, d))
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))
    axes[0].step(E, x, where="mid", color=INK2, lw=1.5, label="counts (dose 5)")
    axes[0].plot(E, peak_model(q_true, d), color=INK, lw=2.5, ls=":", label="true rate")
    for k, c, (m, ls) in zip(KINDS, C, STY):
        axes[0].plot(E, peak_model(fit_peak(x, d, k), d), color=c, ls=ls, label=k)
    axes[0].set_xlabel("energy / position (a.u.)"); axes[0].set_ylabel("counts")
    axes[0].set_title("One low-count spectrum, four losses"); axes[0].legend(fontsize=12, loc="upper left")
    for k, c, (m, ls) in zip(KINDS, C, STY):
        axes[1].semilogx(doses, bias_tot[k], marker=m, ls=ls, color=c, ms=8, label=k)
        axes[2].semilogx(doses, std_s[k], marker=m, ls=ls, color=c, ms=8, label=k)
    axes[1].axhline(0, color=INK2, lw=1)
    axes[1].set_xlabel("peak dose (counts at maximum)"); axes[1].set_ylabel("bias of integrated counts (%)")
    axes[1].set_title(f"Bias ({reps} repeats/dose; MSE and Poisson overlap)"); axes[1].legend(fontsize=12, loc="lower right")
    axes[2].set_xlabel("peak dose (counts at maximum)"); axes[2].set_ylabel("rel. std of fitted width (%)")
    axes[2].set_title("Precision of the width"); axes[2].legend(fontsize=12, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "lowcount_fit_bias.png", facecolor="white")
    plt.close(fig)
    for k in KINDS:
        print(f"{k:16s} bias_tot% {np.round(bias_tot[k], 1)}  std_s% {np.round(std_s[k], 1)}")


if __name__ == "__main__":
    fig_chain()
    fig_loss_curves()
    fig_ptc()
    fig_anscombe()
    fig_lowcount()
    print("figures written to", OUT)
