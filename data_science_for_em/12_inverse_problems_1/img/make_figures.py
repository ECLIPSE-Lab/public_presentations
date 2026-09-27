"""Reproducible figures for DSEM Week 12 (inverse problems I: regularisation, tomography, sensor fusion).

Generates the figures added in the WS 26/27 update (same code/seed as the Week 12 notebook,
so the numbers in the captions match `notebooks/week12_inverse_deblurring.ipynb`):
  - tikhonov_vs_tv.png          1-D step signal: gradient-Tikhonov vs TV (ADMM), lambda sweeps
  - radon_sinogram.png          phantom -> sinogram (Radon transform) -> Fourier coverage
  - fourier_slice_numeric.png   numerical check of the Fourier-slice theorem
  - backprojection_angles.png   unfiltered back-projection with 1/3/12/60 angles vs FBP
  - tomo_fbp_vs_tv.png          full-range FBP vs limited-wedge FBP / SART / SART+TV

Run from repo root:
    .venv/bin/python data_science_for_em/12_inverse_problems_1/img/make_figures.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
from skimage.data import shepp_logan_phantom
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import iradon, iradon_sart, radon, resize

OUT = Path(__file__).resolve().parent
SEED = 42
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13})
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#e67e22", "#2ca02c", "#d62728", "#555555"


# ---------------------------------------------------------------------------
# 1. Tikhonov (gradient penalty) vs TV on a piecewise-constant signal
# ---------------------------------------------------------------------------
def step_problem():
    rng = np.random.default_rng(SEED)
    n = 96
    x = np.full(n, 0.2)
    x[30:55] = 0.8
    x[55:] = 0.3
    s = 4.0
    psf = np.exp(-0.5 * (np.arange(n) - n // 2) ** 2 / s**2)
    psf /= psf.sum()
    c = np.roll(psf, -n // 2)
    H = toeplitz(c, c[::-1])
    H /= H.sum(1, keepdims=True)
    y = H @ x + rng.standard_normal(n) * 0.06
    D = np.diff(np.eye(n), axis=0)
    return x, H, y, D


def tikhonov_grad(H, y, D, lam):
    return np.linalg.solve(H.T @ H + lam * D.T @ D, H.T @ y)


def tv_admm(H, y, D, lam, rho=1.0, n_iter=500):
    """min 0.5||Hx-y||^2 + lam ||Dx||_1 via ADMM (split z = Dx)."""
    A_inv = np.linalg.inv(H.T @ H + rho * D.T @ D)
    Hty = H.T @ y
    z = np.zeros(D.shape[0])
    u = np.zeros(D.shape[0])
    for _ in range(n_iter):
        x = A_inv @ (Hty + rho * D.T @ (z - u))
        v = D @ x + u
        z = np.sign(v) * np.maximum(np.abs(v) - lam / rho, 0.0)
        u = v - z
    return x


def fig_tikhonov_vs_tv():
    x, H, y, D = step_problem()
    rmse = lambda a: np.sqrt(np.mean((a - x) ** 2))
    lams = np.logspace(-3, 1, 25)
    r_tik = np.array([rmse(tikhonov_grad(H, y, D, l)) for l in lams])
    r_tv = np.array([rmse(tv_admm(H, y, D, l)) for l in lams])
    lt, lv = lams[np.argmin(r_tik)], lams[np.argmin(r_tv)]
    xt, xv = tikhonov_grad(H, y, D, lt), tv_admm(H, y, D, lv)
    print(f"[TV] Tikhonov best RMSE={r_tik.min():.4f} @ {lt:.3f}; TV best RMSE={r_tv.min():.4f} @ {lv:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    t = np.arange(len(x))
    ax[0].plot(t, y, color=GREY, lw=1, alpha=0.7, label="measurement $y$ (blur + 6% noise)")
    ax[0].plot(t, x, "k-", lw=2.5, label="truth $x$")
    ax[0].plot(t, xt, color=BLUE, lw=2.2, label=f"Tikhonov $\\|\\nabla x\\|_2^2$ (RMSE {r_tik.min():.3f})")
    ax[0].plot(t, xv, color=ORANGE, lw=2.2, label=f"TV $\\|\\nabla x\\|_1$ (RMSE {r_tv.min():.3f})")
    ax[0].set_xlabel("pixel")
    ax[0].set_ylabel("intensity")
    ax[0].set_title("Grain-boundary steps: best $\\lambda$ for each regulariser")
    ax[0].legend(fontsize=11, loc="upper right")
    ax[1].semilogx(lams, r_tik, "o-", color=BLUE, ms=4, label="Tikhonov (gradient)")
    ax[1].semilogx(lams, r_tv, "s-", color=ORANGE, ms=4, label="TV (ADMM)")
    ax[1].set_xlabel("$\\lambda$")
    ax[1].set_ylabel("RMSE vs truth")
    ax[1].set_title("$\\lambda$ sweep: both U-shaped, TV lower")
    ax[1].legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "tikhonov_vs_tv.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tomography helpers (identical to the notebook)
# ---------------------------------------------------------------------------
def phantom():
    return resize(shepp_logan_phantom(), (128, 128), anti_aliasing=True)


def disk_mask(n=128):
    yy, xx = np.meshgrid(np.arange(n) - (n - 1) / 2, np.arange(n) - (n - 1) / 2)
    return np.hypot(xx, yy) < n / 2


def fig_radon_sinogram():
    img = phantom()
    th = np.linspace(-90, 90, 180, endpoint=False)
    sino = radon(img, theta=th, circle=True)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1, 1.25, 1]})
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("object $f(x,z)$ (Shepp–Logan)")
    ax[0].axis("off")
    ax[1].imshow(sino, cmap="gray", aspect="auto", extent=[th[0], th[-1] + 1, -1, 1])
    ax[1].set_xlabel("tilt angle $\\theta$ (deg)")
    ax[1].set_ylabel("detector coordinate $s$")
    ax[1].set_title("sinogram $p_\\theta(s) = \\mathcal{R}f$")
    # Fourier coverage for a +-60 deg, 5 deg-step tilt series
    a = ax[2]
    for t in np.arange(-60, 61, 5):
        r = np.deg2rad(t)
        a.plot([-np.cos(r), np.cos(r)], [-np.sin(r), np.sin(r)], color=BLUE, lw=1.3)
    for sgn in (1, -1):
        ang = np.deg2rad(np.linspace(60, 120, 50))
        a.fill(np.r_[0, np.cos(ang)] * sgn, np.r_[0, np.sin(ang)] * sgn, color=RED, alpha=0.25)
    a.text(0, 0.8, "missing\nwedge", ha="center", color=RED, fontsize=13)
    a.set_xlim(-1.05, 1.05)
    a.set_ylim(-1.05, 1.05)
    a.set_aspect("equal")
    a.set_xlabel("$k_x$")
    a.set_ylabel("$k_z$")
    a.set_title("Fourier coverage, $\\pm 60°$ tilt, 5° step")
    fig.tight_layout()
    fig.savefig(OUT / "radon_sinogram.png", dpi=120)
    plt.close(fig)


def fig_fourier_slice_numeric():
    img = phantom()
    n = img.shape[0]
    # projection along z (axis 0) at theta=0 -> p(s) = sum_z f(z, s)
    p = img.sum(axis=0)
    P1 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(p)))
    F2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    central = F2[n // 2, :]
    k = np.arange(n) - n // 2
    err = np.max(np.abs(P1 - central)) / np.max(np.abs(central))
    print(f"[FST] max relative deviation = {err:.2e}")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(p, color="k", lw=2)
    ax[0].set_title("projection $p_0(s) = \\int f(x,z)\\,dz$")
    ax[0].set_xlabel("$s$ (pixel)")
    ax[1].imshow(np.log1p(np.abs(F2)), cmap="magma")
    ax[1].axhline(n // 2, color="cyan", lw=2)
    ax[1].set_title("$\\log|\\mathcal{F}_{2D} f|$ with central slice $k_z=0$")
    ax[1].axis("off")
    ax[2].semilogy(k, np.abs(central), color=BLUE, lw=3, label="central slice of $\\mathcal{F}_{2D}f$")
    ax[2].semilogy(k, np.abs(P1), "--", color=ORANGE, lw=2, label="$\\mathcal{F}_{1D}\\,p_0$")
    ax[2].set_xlabel("$k_x$")
    ax[2].set_title(f"identical (max rel. dev. {err:.0e})")
    ax[2].legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fourier_slice_numeric.png", dpi=120)
    plt.close(fig)


def fig_backprojection_angles():
    img = phantom()
    fig, ax = plt.subplots(1, 5, figsize=(22, 5))
    for a, n_ang in zip(ax[:4], [1, 3, 12, 60]):
        th = np.linspace(0, 180, n_ang, endpoint=False)
        bp = iradon(radon(img, theta=th, circle=True), theta=th, filter_name=None, circle=True)
        a.imshow(bp, cmap="gray")
        a.set_title(f"back-projection, {n_ang} angle{'s' if n_ang > 1 else ''}")
        a.axis("off")
    th = np.linspace(0, 180, 60, endpoint=False)
    fbp = iradon(radon(img, theta=th, circle=True), theta=th, filter_name="ramp", circle=True)
    ax[4].imshow(fbp, cmap="gray", vmin=0, vmax=1)
    ax[4].set_title("filtered BP (ramp), 60 angles")
    ax[4].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "backprojection_angles.png", dpi=110)
    plt.close(fig)


def sart_tv(sino, th, n_iter=10, tv_weight=0.0):
    rec = None
    for _ in range(n_iter):
        rec = iradon_sart(sino, theta=th, image=rec, relaxation=0.25, clip=(0, 2))
        if tv_weight > 0:
            rec = denoise_tv_chambolle(rec, weight=tv_weight)
    return rec


def fig_tomo_fbp_vs_tv():
    img = phantom()
    rng = np.random.default_rng(SEED)
    mask = disk_mask()
    rmse = lambda a: np.sqrt(np.mean(((a - img) * mask) ** 2))
    th_full = np.linspace(-90, 90, 60, endpoint=False)
    th_lim = np.linspace(-60, 60, 25)

    def noisy(th):
        s = radon(img, theta=th, circle=True)
        return s + rng.standard_normal(s.shape) * 0.05 * s.max()

    s_full, s_lim = noisy(th_full), noisy(th_lim)
    recs = {
        "FBP, 60 tilts, 180°": iradon(s_full, theta=th_full, filter_name="ramp", circle=True),
        "FBP, 25 tilts, ±60°": iradon(s_lim, theta=th_lim, filter_name="ramp", circle=True),
        "SART, 25 tilts, ±60°": sart_tv(s_lim, th_lim),
        "SART + TV, 25 tilts, ±60°": sart_tv(s_lim, th_lim, tv_weight=0.04),
    }
    fig, ax = plt.subplots(1, 5, figsize=(24, 5.6))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("ground truth")
    ax[0].axis("off")
    for a, (name, r) in zip(ax[1:], recs.items()):
        a.imshow(r * mask, cmap="gray", vmin=0, vmax=1)
        a.set_title(f"{name}\nRMSE {rmse(r):.3f}")
        a.axis("off")
        print(f"[TOMO] {name}: RMSE={rmse(r):.4f}")
    fig.tight_layout()
    fig.savefig(OUT / "tomo_fbp_vs_tv.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    fig_tikhonov_vs_tv()
    fig_radon_sinogram()
    fig_fourier_slice_numeric()
    fig_backprojection_angles()
    fig_tomo_fbp_vs_tv()
    print("done")
