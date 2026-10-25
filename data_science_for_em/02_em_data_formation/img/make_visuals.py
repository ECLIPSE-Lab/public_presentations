"""Explanatory figures for the text-heavy slides of DSEM Week 2.

All data are synthetic. Every number in a figure is computed here, from these data.
Same style as make_figures.py (white background, same palette and font sizes).

Run:
    ~/miniforge3/bin/python3 data_science_for_em/02_em_data_formation/img/make_visuals.py

Produces (in this folder):
    learning_types.png        supervised / unsupervised / self-supervised, one EM example each
    bias_variance.png         EELS edge fitted by a line, a free polynomial and the physical model
    nyquist_sampling.png      a lattice profile sampled at 4, 2 and 1.5 pixels per period
    undersampled_lattice.png  2 A lattice at 0.5 A and 1.5 A pixels: the 6 A alias fringe and its FFT
    noise_sources.png         Gaussian vs Poisson noise on the same intensity ramp
    mixed_detector.png        x = g*Poisson(lambda) + N(0, sigma_r^2): electron peaks merge with dose
    noise_budget.png          variance of each noise contribution vs mean signal (example detector)
    failure_modes.png         six common artefacts and the quick check that finds each one
    aleatoric_epistemic.png   count scatter does not shrink with more frames; gain error does
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import curve_fit

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(2026)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#222222", "#555555", "#dddddd"
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 150,
})


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def img_ax(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=14)


def lattice(n, pitch_A, a_A=2.0, angle=0.0):
    """Cosine lattice with period a_A, sampled with pixel pitch pitch_A."""
    y, x = np.mgrid[:n, :n] * pitch_A
    k = 2 * np.pi / a_A
    u = x * np.cos(angle) + y * np.sin(angle)
    v = -x * np.sin(angle) + y * np.cos(angle)
    return (np.cos(k * u) + 1) * (np.cos(k * v) + 1) / 4


# --------------------------------------------------------------------------------------
# 1. Three types of learning
# --------------------------------------------------------------------------------------
def fig_learning_types():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    # Supervised: two diffraction features, labelled phases, a linear boundary fitted to them.
    ax = axes[0]
    n = 60
    a = rng.normal([2.0, 1.0], 0.35, (n, 2)); b = rng.normal([3.0, 2.0], 0.35, (n, 2))
    X = np.vstack([a, b]); y = np.r_[np.zeros(n), np.ones(n)]
    Xa = np.c_[X, np.ones(len(X))]
    w = np.linalg.lstsq(Xa, 2 * y - 1, rcond=None)[0]          # least-squares linear classifier
    acc = np.mean((Xa @ w > 0) == (y == 1))
    ax.scatter(*a.T, s=28, color=C[0], label="phase α (labelled)")
    ax.scatter(*b.T, s=28, color=C[1], label="phase β (labelled)")
    xs = np.linspace(0.8, 5.4, 10)
    ax.plot(xs, -(w[0] * xs + w[2]) / w[1], color=INK, ls="--", label=f"learned boundary,\n{acc:.0%} correct")
    ax.set_xlim(0.8, 5.4); ax.set_ylim(-0.2, 3.2)
    ax.set_xlabel("Bragg spacing $d_1$ (Å)"); ax.set_ylabel("Bragg spacing $d_2$ (Å)")
    ax.set_title("Supervised: labels given", loc="left", weight="bold")
    ax.legend(fontsize=11.5, loc="upper right")
    # Unsupervised: EELS spectra of two hidden phases, grouped by k-means without labels.
    ax = axes[1]
    E = np.linspace(500, 600, 200)
    def edge(e0, slope):
        return 1 / (1 + np.exp(-(E - e0) / 2)) * np.exp(-(E - e0).clip(0) / slope)
    spectra, truth = [], []
    for i in range(40):
        ph = i % 2
        spectra.append((edge(530, 60) if ph == 0 else edge(545, 30)) * rng.uniform(0.8, 1.2)
                       + rng.normal(0, 0.04, E.size)); truth.append(ph)
    S = np.array(spectra); truth = np.array(truth)
    cent = S[[0, 1]].copy()
    lab = np.zeros(len(S), int)
    for _ in range(20):                                         # k-means, k = 2
        lab = np.argmin(((S[:, None] - cent[None]) ** 2).sum(-1), 1)
        cent = np.array([S[lab == j].mean(0) for j in range(2)])
    agree = max(np.mean(lab == truth), np.mean(lab != truth))
    for s, l in zip(S, lab):
        ax.plot(E, s, color=C[2] if l == 0 else C[3], lw=0.8, alpha=0.5)
    ax.set_xlabel("energy loss (eV)"); ax.set_ylabel("counts (a.u.)"); ax.set_yticks([])
    ax.set_title("Unsupervised: find the groups", loc="left", weight="bold")
    ax.text(0.03, 0.97, f"k-means: 2 clusters,\n{agree:.0%} match the hidden phases", transform=ax.transAxes,
            ha="left", va="top", fontsize=12, color=INK2)
    # Self-supervised: two noisy copies of one image; one is the target of the other.
    ax = axes[2]; ax.axis("off")
    clean = lattice(48, 0.5, a_A=3.0) * 20 + 2
    n1, n2 = rng.poisson(clean), rng.poisson(clean)
    ax.set_title("Self-supervised: labels from the data", loc="left", weight="bold")
    i1 = ax.inset_axes([0.0, 0.12, 0.42, 0.7]); i2 = ax.inset_axes([0.58, 0.12, 0.42, 0.7])
    i1.imshow(n1, cmap="gray"); img_ax(i1, "noisy frame 1 = input")
    i2.imshow(n2, cmap="gray"); img_ax(i2, "noisy frame 2 = target")
    ax.annotate("", xy=(0.57, 0.47), xytext=(0.43, 0.47), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    ax.text(0.5, 0.0, "Noise2Noise: no clean image needed", transform=ax.transAxes, ha="center", fontsize=12, color=INK2)
    fig.tight_layout()
    save(fig, "learning_types.png")


# --------------------------------------------------------------------------------------
# 2. Bias-variance: an EELS edge
# --------------------------------------------------------------------------------------
def edge_model(E, A, r, B, e0=284.0, w=1.5):
    """Power-law background plus a smoothed edge step that decays as a power law."""
    step = 1 / (1 + np.exp(-(E - e0) / w))
    return A * (E / 250) ** (-r) + B * step * (E / e0) ** (-3)


def fig_bias_variance():
    E = np.linspace(250, 330, 60)
    truth = edge_model(E, 100, 3.0, 40)
    Et = np.linspace(250, 330, 400)

    def fits(y, Eeval):
        z = (E - 290) / 40; ze = (Eeval - 290) / 40
        line = np.polyval(np.polyfit(z, y, 1), ze)
        poly = np.polyval(np.polyfit(z, y, 15), ze)
        p, _ = curve_fit(edge_model, E, y, p0=[90, 2.5, 30], maxfev=20000)
        return line, poly, edge_model(Eeval, *p)

    y = rng.poisson(truth).astype(float)
    line, poly, phys = fits(y, Et)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax.plot(E, y, "o", ms=4, color=INK2, label="measured counts")
    ax.plot(Et, edge_model(Et, 100, 3.0, 40), color=GRID, lw=6, label="true signal")
    ax.plot(Et, line, color=C[0], label="straight line: underfits (high bias)")
    ax.plot(Et, np.clip(poly, 0, 150), color=C[1], label="degree-15 polynomial: overfits (high variance)")
    ax.plot(Et, phys, color=C[2], label="physical edge model: sweet spot")
    ax.set_xlabel("energy loss (eV)"); ax.set_ylabel("counts"); ax.set_ylim(0, 150)
    ax.set_title("Carbon K-edge (synthetic): three models, one data set", loc="left", weight="bold")
    ax.legend(fontsize=11.5, loc="lower left")
    # Error on new data vs model complexity: polynomial degree, averaged over 200 data sets.
    degs = np.arange(0, 16)
    tr, te = np.zeros(degs.size), np.zeros(degs.size)
    te_phys = 0.0
    z = (E - 290) / 40
    for _ in range(200):
        y1 = rng.poisson(truth); y2 = rng.poisson(truth)       # train set and an independent test set
        for i, d in enumerate(degs):
            f = np.polyval(np.polyfit(z, y1, d), z)
            tr[i] += np.mean((f - y1) ** 2) / 200; te[i] += np.mean((f - y2) ** 2) / 200
        p, _ = curve_fit(edge_model, E, y1, p0=[90, 2.5, 30], maxfev=20000)
        te_phys += np.mean((edge_model(E, *p) - y2) ** 2) / 200
    best = degs[np.argmin(te)]
    ax2.plot(degs, tr, "o-", color=C[0], label="error on training data")
    ax2.plot(degs, te, "o-", color=C[1], label="error on new data")
    ax2.axhline(te_phys, color=C[2], ls="--", label=f"physical model on new data ({te_phys:.0f})")
    ax2.axvline(best, color=INK2, ls=":", lw=1.5)
    ax2.text(best + 0.3, te.max() * 0.6, f"best polynomial:\ndegree {best} ({te.min():.0f})", color=INK2, fontsize=12)
    ax2.set_yscale("log"); ax2.set_xlabel("model complexity (polynomial degree)"); ax2.set_ylabel("mean squared error")
    ax2.set_title("Training error always falls; error on new data does not", loc="left", weight="bold", fontsize=14)
    ax2.legend(fontsize=12)
    fig.tight_layout()
    save(fig, "bias_variance.png")
    return best, te.min(), te_phys


# --------------------------------------------------------------------------------------
# 3. Nyquist sampling of a lattice profile
# --------------------------------------------------------------------------------------
def fig_nyquist():
    a = 2.0                                    # lattice period (A)
    x = np.linspace(0, 12, 2000)
    f = lambda t: np.cos(2 * np.pi * t / a)
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.4), sharex=True)
    for ax, ppp, col in zip(axes, [4, 2, 1.333], [C[2], C[3], C[1]]):
        dx = a / ppp
        xs = np.arange(0, 12 + 1e-9, dx)
        ax.plot(x, f(x), color=GRID, lw=4)
        ax.plot(xs, f(xs), "o-", color=col, ms=7, lw=1.5)
        ax.set_yticks([]); ax.set_ylim(-1.4, 1.6)
        verdict = {4: "fine: the period is clear", 2: "at the Nyquist limit: only just",
                   1.333: "below Nyquist: a false 6 Å period appears"}[ppp]
        ax.text(0.01, 0.97, f"pixel {dx:.2f} Å  =  {ppp:.1f} pixels per 2 Å period  →  {verdict}",
                transform=ax.transAxes, va="top", fontsize=13, color=col if ppp != 4 else INK)
    axes[-1].set_xlabel("position (Å)")
    fig.tight_layout()
    save(fig, "nyquist_sampling.png")


# --------------------------------------------------------------------------------------
# 4. Undersampled 2-D lattice and its FFT
# --------------------------------------------------------------------------------------
def fig_undersampled():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.9))
    out = {}
    for j, (pitch, n) in enumerate([(0.5, 108), (1.5, 36)]):       # both 54 A wide
        im = lattice(n, pitch, a_A=2.0, angle=0.0)
        F = np.abs(np.fft.fftshift(np.fft.fft2(im - im.mean())))
        k = np.fft.fftshift(np.fft.fftfreq(n, d=pitch / 10))     # nm^-1
        iy, ix = np.unravel_index(np.argmax(F), F.shape)
        kpeak = np.hypot(k[iy], k[ix]); out[pitch] = kpeak
        L = n * pitch
        axes[2 * j].imshow(im, cmap="gray", extent=[0, L, L, 0], interpolation="nearest")
        img_ax(axes[2 * j], f"pixel {pitch} Å: " + ("the true 2 Å lattice" if pitch == 0.5 else "looks like a 6 Å lattice!"))
        axes[2 * j].plot([3, 13], [L - 3] * 2, color=C[3], lw=5)
        axes[2 * j].text(8, L - 5, "10 Å", color=C[3], ha="center", fontsize=13, weight="bold")
        kn = 10 / (2 * pitch)
        ax = axes[2 * j + 1]
        ax.set_facecolor("black")
        pk = np.argwhere(F > 0.2 * F.max())
        ax.scatter(k[pk[:, 1]], k[pk[:, 0]], s=90, color="white")
        ax.set_xlim(-10.5, 10.5); ax.set_ylim(10.5, -10.5); img_ax(ax, f"FFT: spots at {kpeak:.2f} nm⁻¹")
        ax.add_patch(Rectangle((-kn, -kn), 2 * kn, 2 * kn, fill=False, ec=C[3], lw=2, ls="--"))
        ax.text(0, min(kn, 9.6) - 0.3, f"recordable: ±{kn:.2f} nm⁻¹", color=C[3], ha="center", va="bottom", fontsize=11)
        if pitch == 1.5:
            ax.scatter([5, -5, 0, 0], [0, 0, 5, -5], s=110, facecolor="none", edgecolor=C[1], lw=2)
            ax.text(7.2, 1.3, "true 5 nm⁻¹\n(not recorded)", color=C[1], ha="center", va="top", fontsize=11)
    fig.tight_layout()
    save(fig, "undersampled_lattice.png")
    return out


# --------------------------------------------------------------------------------------
# 5. Gaussian vs Poisson noise on one ramp
# --------------------------------------------------------------------------------------
def fig_noise_sources():
    ramp = np.tile(np.linspace(1, 100, 300), (60, 1))
    sig_r = 5.0
    gauss = ramp + rng.normal(0, sig_r, ramp.shape)
    pois = rng.poisson(ramp).astype(float)
    fig = plt.figure(figsize=(17, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.45, wspace=0.18)
    for r, (im, name, col) in enumerate([(gauss, f"Gaussian read noise (σ = {sig_r:.0f}, same everywhere)", C[0]),
                                         (pois, "Poisson shot noise (σ = √λ, grows with the signal)", C[1])]):
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(im, cmap="gray", aspect="auto", vmin=-10, vmax=115)
        img_ax(ax); ax.set_title(name, loc="left", fontsize=14, color=col, weight="bold")
    ax = fig.add_subplot(gs[:, 1])
    cols = ramp[0]; bins = np.arange(0, 300, 15)
    lam = [cols[b:b + 15].mean() for b in bins]
    ax.plot(lam, [gauss[:, b:b + 15].std() for b in bins], "o", color=C[0], label="Gaussian: flat")
    ax.plot(lam, [pois[:, b:b + 15].std() for b in bins], "o", color=C[1], label="Poisson: √λ")
    l = np.linspace(1, 100, 100); ax.plot(l, np.sqrt(l), color=C[1], lw=1, ls="--"); ax.axhline(sig_r, color=C[0], lw=1, ls="--")
    ax.set_xlabel("mean signal λ (counts)"); ax.set_ylabel("noise σ (counts)"); ax.legend(fontsize=13)
    ax.set_title("measured from the two images", loc="left", fontsize=14)
    save(fig, "noise_sources.png")


# --------------------------------------------------------------------------------------
# 6. Mixed detector: single-electron peaks merge as the dose grows
# --------------------------------------------------------------------------------------
def fig_mixed_detector():
    g = 20.0                                   # ADU per electron
    cases = [(3, 4.0, "counting detector: σ_r = 4 ADU ≪ g"), (3, 25.0, "integrating detector: σ_r = 25 ADU ≈ g"),
             (30, 25.0, "integrating, higher dose")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (lam, sig_r, name) in zip(axes, cases):
        x = g * rng.poisson(lam, 200_000) + rng.normal(0, sig_r, 200_000)
        ax.hist(x, bins=np.arange(-80, x.max() + 2, 2), color=C[0], alpha=0.85)
        ax.set_yticks([]); ax.set_xlabel("pixel value (ADU)")
        m, v = x.mean(), x.var()
        ax.set_title(f"{name}\nλ = {lam} e⁻/pixel", loc="left", fontsize=14)
        ax.text(0.97, 0.95, f"variance {v:.0f}\n= g·mean + σ_r²\n({g * m + sig_r**2:.0f})",
                transform=ax.transAxes, ha="right", va="top", fontsize=12, color=INK2)
        if sig_r < 10:
            for ne in range(0, 6):
                ax.text(ne * g, ax.get_ylim()[1] * 0.02, f"{ne}", color=C[1], ha="center", va="bottom",
                        fontsize=11, weight="bold")
            ax.text(0.02, 0.97, "peaks = 0, 1, 2 … electrons", transform=ax.transAxes, color=C[1], va="top", fontsize=12)
    fig.suptitle(f"x = g·Poisson(λ) + N(0, σ_r²),  g = {g:.0f} ADU/e⁻: single electrons are countable only if σ_r ≪ g",
                 fontsize=14, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "mixed_detector.png")


# --------------------------------------------------------------------------------------
# 7. Noise budget vs signal (example detector)
# --------------------------------------------------------------------------------------
def fig_noise_budget():
    lam = np.logspace(-1, 4, 200)             # mean signal in electrons per pixel
    parts = {
        "shot noise (Poisson): λ": lam,
        "read noise: σ_r² = 9": np.full_like(lam, 9.0),
        "dark current: 0.5 e⁻": np.full_like(lam, 0.5),
        "fixed pattern (2 % gain spread): (0.02 λ)²": (0.02 * lam) ** 2,
        "quantisation (14 bit, 1 ADU = 1 e⁻): 1/12": np.full_like(lam, 1 / 12),
    }
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for (name, v), col in zip(parts.items(), C):
        ax.loglog(lam, v, color=col, label=name)
    tot = sum(parts.values()); ax.loglog(lam, tot, color=INK, lw=3, ls="--", label="total variance")
    cross_rn = 9.0; cross_fp = 1 / 0.02**2
    ax.axvspan(lam[0], cross_rn, color=C[1], alpha=0.07); ax.axvspan(cross_fp, lam[-1], color=C[3], alpha=0.07)
    ax.text(0.13, 4e3, "read noise\ndominates", color=C[1], fontsize=13)
    ax.text(15, 4e3, "shot noise dominates", color=C[0], fontsize=13)
    ax.text(2.7e3, 3e-2, "flat-field\nneeded", color=C[3], fontsize=13)
    ax.set_xlabel("mean signal λ (e⁻ per pixel)"); ax.set_ylabel("variance (e⁻²)")
    ax.set_ylim(1e-2, 1e5); ax.legend(fontsize=11.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Which noise matters depends on the dose (example detector values)", loc="left", weight="bold", fontsize=15)
    save(fig, "noise_budget.png")


# --------------------------------------------------------------------------------------
# 8. Failure modes gallery
# --------------------------------------------------------------------------------------
def fig_failure_modes():
    n = 96
    base = lattice(n, 0.5, a_A=3.0) * 60 + 20
    fig, axes = plt.subplots(2, 6, figsize=(19, 6.6), gridspec_kw={"height_ratios": [1, 0.75]})
    # a) hot/dead pixels -> histogram
    im = rng.poisson(base).astype(float); hot = rng.integers(0, n, (12, 2))
    im[hot[:6, 0], hot[:6, 1]] = 1500; im[hot[6:, 0], hot[6:, 1]] = 0
    axes[0, 0].imshow(im, cmap="gray", vmax=120); img_ax(axes[0, 0], "hot / dead pixels")
    axes[1, 0].hist(im.ravel(), bins=60, color=C[0], log=True); axes[1, 0].set_title("check: histogram", fontsize=12)
    # b) saturation -> clipped histogram
    im = np.minimum(rng.poisson(base * 3), 160)
    axes[0, 1].imshow(im, cmap="gray"); img_ax(axes[0, 1], "saturation")
    axes[1, 1].hist(im.ravel(), bins=60, color=C[0], log=True); axes[1, 1].set_title("check: pile-up at max", fontsize=12)
    # c) scan drift -> lattice sheared, FFT spots smeared
    y, x = np.mgrid[:n, :n] * 0.5
    sh = x + 0.25 * y
    im = rng.poisson(((np.cos(2 * np.pi * sh / 3) + 1) * (np.cos(2 * np.pi * y / 3) + 1) / 4) * 60 + 20)
    axes[0, 2].imshow(im, cmap="gray"); img_ax(axes[0, 2], "scan drift (shear)")
    F = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(im - im.mean()))))
    axes[1, 2].imshow(F[24:72, 24:72], cmap="gray", vmin=np.percentile(F, 90)); img_ax(axes[1, 2]); axes[1, 2].set_title("check: FFT spots off-grid", fontsize=12)
    # d) line jitter / flyback
    im = rng.poisson(base).astype(float)
    shifts = rng.normal(0, 2.0, n).round().astype(int)
    im = np.array([np.roll(r, s) for r, s in zip(im, shifts)])
    axes[0, 3].imshow(im, cmap="gray"); img_ax(axes[0, 3], "scan-line jitter")
    axes[1, 3].plot(shifts, color=C[1]); axes[1, 3].set_title("check: row-to-row shift", fontsize=12)
    axes[1, 3].set_xlabel("scan line")
    # e) contamination build-up over a series -> per-frame sum
    frames = np.arange(40); total = 1 - 0.012 * frames + rng.normal(0, 0.004, 40)
    yy, xx = np.mgrid[:n, :n]
    haze = 1 - 0.7 * np.exp(-((yy - 48) ** 2 + (xx - 48) ** 2) / (2 * 22 ** 2))   # carbon build-up in the scan area
    axes[0, 4].imshow(rng.poisson(base * haze), cmap="gray"); img_ax(axes[0, 4], "contamination build-up")
    axes[1, 4].plot(frames, total, "o-", ms=3, color=C[1]); axes[1, 4].set_title("check: per-frame sum", fontsize=12)
    axes[1, 4].set_xlabel("frame")
    # f) aliasing -> FFT spot inside Nyquist at wrong k
    im = rng.poisson(lattice(n, 1.5, a_A=2.0) * 60 + 20)
    axes[0, 5].imshow(im, cmap="gray"); img_ax(axes[0, 5], "aliasing (pixel too large)")
    F = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(im - im.mean()))))
    axes[1, 5].imshow(F[8:88, 8:88], cmap="gray", vmin=np.percentile(F, 90)); img_ax(axes[1, 5]); axes[1, 5].set_title("check: FFT vs expected k", fontsize=12)
    for ax in axes[1, [0, 1]]:
        ax.set_yticks([]); ax.set_xlabel("pixel value")
    for ax in axes[1, [3, 4]]:
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, "failure_modes.png")


# --------------------------------------------------------------------------------------
# 9. Aleatoric vs epistemic
# --------------------------------------------------------------------------------------
def fig_aleatoric_epistemic():
    lam, g_true, sig_r = 25.0, 2.0, 3.0
    Ns = np.array([2, 5, 10, 20, 50, 100, 200, 500])
    spread, gerr = [], []
    for N in Ns:
        reps = []
        sp = []
        for _ in range(300):
            counts = rng.poisson(lam, N)
            sp.append(counts.var(ddof=1))
            # calibrate the gain from N flat frames at two signal levels (variance-mean slope)
            lv = np.array([20, 200])
            xs = [g_true * rng.poisson(L, (N, 64)) + rng.normal(0, sig_r, (N, 64)) for L in lv]
            m = np.array([x.mean() for x in xs]); v = np.array([x.var(ddof=1) for x in xs])
            reps.append(np.polyfit(m, v, 1)[0])
        spread.append(np.sqrt(np.mean(sp))); gerr.append(np.std(reps))   # sqrt of mean unbiased variance
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax.semilogx(Ns, spread, "o-", color=C[1], label="scatter of the counts (std)")
    ax.axhline(np.sqrt(lam), color=C[1], ls="--", lw=1); ax.text(Ns[0], np.sqrt(lam) * 1.08, "√λ = 5", color=C[1])
    ax.set_ylim(0, 8); ax.set_xlabel("number of frames N"); ax.set_ylabel("counts")
    ax.set_title("Aleatoric: shot noise stays, however many frames", loc="left", weight="bold", fontsize=14)
    ax2.loglog(Ns, gerr, "o-", color=C[0], label="error of the fitted gain g")
    ax2.loglog(Ns, gerr[0] * np.sqrt(Ns[0] / Ns), color=C[0], ls="--", lw=1, label="∝ 1/√N")
    ax2.set_xlabel("number of calibration frames N"); ax2.set_ylabel("std of fitted gain (ADU/e⁻)")
    ax2.set_title("Epistemic: calibration error shrinks with data", loc="left", weight="bold", fontsize=14)
    ax2.legend(fontsize=12)
    fig.tight_layout()
    save(fig, "aleatoric_epistemic.png")
    return spread, gerr


if __name__ == "__main__":
    fig_learning_types()
    print("bias-variance best degree:", fig_bias_variance())
    fig_nyquist()
    print("FFT peak |k| (nm^-1) per pixel size:", {k: round(v, 2) for k, v in fig_undersampled().items()})
    fig_noise_sources()
    fig_mixed_detector()
    fig_noise_budget()
    fig_failure_modes()
    sp, ge = fig_aleatoric_epistemic()
    print("count std (first, last):", round(sp[0], 2), round(sp[-1], 2), "| gain error (first, last):", round(ge[0], 3), round(ge[-1], 4))
