"""Reproducible figures for DSEM Week 3 (spectral unmixing additions, WS 26/27).

Run from repo root:
    .venv/bin/python data_science_for_em/03_linear_algebra_spectral/img/make_figures.py

Generates (into this img/ folder):
    unmixing_synthetic_truth.png   ground-truth endmembers + abundance maps
    pca_vs_nmf_endmembers.png      PCA vs NMF vs truth (spectra + maps)
    nmf_rotational_ambiguity.png   simplex picture: two equally good factorisations
    eels_preprocessing.png         power-law background, energy alignment, derivative-shaped PC
    poisson_weighting_scree.png    unweighted vs Poisson-weighted PCA scree
    pca_truncation_bias.png        rare phase erased by truncation; residual map reveals it
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

OUT = Path(__file__).resolve().parent
BG = "#1a1a2e"
FG = "#dbe4f0"
C = ["#4fc3f7", "#ff8a65", "#aed581", "#ce93d8", "#ffd54f"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": FG, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG, "axes.titlesize": 13,
    "axes.labelsize": 12, "font.size": 11, "legend.facecolor": "#262640",
    "legend.edgecolor": "#555577", "axes.grid": True, "grid.color": "#2c2c48",
})


# ----------------------------------------------------------------------------
# Synthetic 3-phase core-loss spectrum image (background already removed)
# ----------------------------------------------------------------------------
def edge(E, onset, jump, wl=0.0, wl_w=4.0, width=3.0):
    """Idealised ionisation edge: smoothed step with E^-2 tail plus white line."""
    step = 1.0 / (1.0 + np.exp(-(E - onset) / width))
    tail = np.clip(E / onset, 1e-6, None) ** -2.0
    white = wl * np.exp(-0.5 * ((E - onset - 4) / wl_w) ** 2)
    return jump * step * tail + white


def make_endmembers(E):
    tio2 = edge(E, 456, 0.6, wl=1.2) + edge(E, 532, 0.5, wl=0.5)   # Ti-L + O-K
    feo = edge(E, 708, 0.5, wl=1.4) + edge(E, 532, 0.35, wl=0.4)   # Fe-L + O-K
    ni = edge(E, 855, 0.7, wl=1.0)                                  # Ni-L (metal)
    M = np.stack([tio2, feo, ni])
    return M / M.max(axis=1, keepdims=True)


def make_abundances(ny=48, nx=48, seed=0):
    """Smooth abundance maps on the simplex, with pure regions."""
    yy, xx = np.mgrid[0:ny, 0:nx] / max(ny, nx)
    a1 = np.exp(-(((xx - 0.28) ** 2 + (yy - 0.35) ** 2) / 0.02))      # particle 1
    a2 = np.exp(-(((xx - 0.72) ** 2 + (yy - 0.62) ** 2) / 0.03))      # particle 2
    a1, a2 = np.clip(1.6 * a1, 0, 1), np.clip(1.6 * a2, 0, 1)
    a3 = np.clip(1.0 - a1 - a2, 0, None)                               # matrix
    A = np.stack([a1, a2, a3], axis=-1)
    A = A / A.sum(axis=-1, keepdims=True)
    return A.reshape(-1, 3)


def synthetic_si(dose=60.0, seed=0, ny=48, nx=48, ne=300):
    rng = np.random.default_rng(seed)
    E = np.linspace(400, 900, ne)
    M = make_endmembers(E)
    A = make_abundances(ny, nx)
    clean = A @ M + 0.05
    noisy = rng.poisson(clean * dose) / dose
    return E, M, A, clean, noisy, (ny, nx)


def match_components(M_true, M_est):
    """Hungarian matching on correlation; returns permutation and scales."""
    Ct = (M_true - M_true.mean(1, keepdims=True))
    Ce = (M_est - M_est.mean(1, keepdims=True))
    corr = (Ct / np.linalg.norm(Ct, axis=1, keepdims=True)) @ \
           (Ce / np.linalg.norm(Ce, axis=1, keepdims=True)).T
    r, c = linear_sum_assignment(-corr)
    return c, corr[r, c]


def fig_truth_and_pca_vs_nmf():
    E, M, A, clean, noisy, (ny, nx) = synthetic_si()
    names = ["TiO$_2$-like (Ti-L + O-K)", "FeO-like (Fe-L + O-K)", "Ni metal (Ni-L)"]

    # --- ground truth figure ---
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2), gridspec_kw={"width_ratios": [2, 1, 1, 1]})
    for k in range(3):
        ax[0].plot(E, M[k], color=C[k], lw=2, label=names[k])
        ax[k + 1].imshow(A[:, k].reshape(ny, nx), cmap="magma", vmin=0, vmax=1)
        ax[k + 1].set_title(f"abundance {k+1}", color=C[k]); ax[k + 1].axis("off")
    ax[0].set_xlabel("Energy loss (eV)"); ax[0].set_ylabel("Intensity (norm.)")
    ax[0].set_title("Ground-truth endmember spectra"); ax[0].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "unmixing_synthetic_truth.png", dpi=150); plt.close(fig)

    # --- PCA ---
    mu = noisy.mean(0)
    U, s, Vt = np.linalg.svd(noisy - mu, full_matrices=False)
    pca_S, pca_T = Vt[:3], U[:, :3] * s[:3]
    # --- NMF ---
    nmf = NMF(n_components=3, init="nndsvda", max_iter=800, random_state=0)
    W = nmf.fit_transform(noisy)
    H = nmf.components_
    perm, _ = match_components(M, H)
    H, W = H[perm], W[:, perm]
    sc = H.max(1); H = H / sc[:, None]; W = W * sc[None, :]

    fig, ax = plt.subplots(3, 4, figsize=(18, 11), gridspec_kw={"width_ratios": [2, 1, 1, 1]})
    rows = [("Ground truth", M, A), ("PCA (K=3)", pca_S, pca_T), ("NMF (K=3)", H, W)]
    for r, (lab, S, T) in enumerate(rows):
        for k in range(3):
            ax[r, 0].plot(E, S[k], color=C[k], lw=2)
            im = T[:, k].reshape(ny, nx)
            cmap = "RdBu_r" if lab.startswith("PCA") else "magma"
            vmax = np.abs(im).max()
            ax[r, k + 1].imshow(im, cmap=cmap, vmin=-vmax if cmap == "RdBu_r" else 0, vmax=vmax)
            ax[r, k + 1].axis("off")
        ax[r, 0].axhline(0, color="#888", lw=0.8)
        ax[r, 0].set_title(f"{lab}: spectra"); ax[r, 0].set_ylabel("a.u.")
        ax[r, 1].set_title(f"{lab}: maps")
    ax[2, 0].set_xlabel("Energy loss (eV)")
    fig.suptitle("Same noisy spectrum image, two factorisations: PCA components are signed mixtures; "
                 "NMF components look like phases", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "pca_vs_nmf_endmembers.png", dpi=130); plt.close(fig)


def fig_rotational_ambiguity():
    E, M, A, clean, noisy, _ = synthetic_si(dose=200)
    mu = clean.mean(0)
    _, _, Vt = np.linalg.svd(clean - mu, full_matrices=False)
    P = Vt[:2]
    pts = (noisy - mu) @ P.T
    M = M + 0.05                              # constant offset belongs to every endmember
    V = (M - mu) @ P.T                        # true vertices in PC plane
    cen = V.mean(0)
    # enlarge simplex while keeping vertex spectra non-negative
    for f in np.linspace(2.0, 1.0, 101):
        Vb = cen + f * (V - cen)
        # spectra of enlarged vertices = barycentric combos of true endmembers
        B = np.linalg.solve(np.c_[V, np.ones(3)].T, np.c_[Vb, np.ones(3)].T).T  # weights
        Mb = B @ M
        if Mb.min() >= -1e-9:
            break
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(pts[:, 0], pts[:, 1], s=4, color="#9aa7c7", alpha=0.5, label="pixels")
    tri = np.vstack([V, V[:1]]); trib = np.vstack([Vb, Vb[:1]])
    ax[0].plot(tri[:, 0], tri[:, 1], "-o", color=C[2], lw=2.5, label="true endmembers")
    ax[0].plot(trib[:, 0], trib[:, 1], "--s", color=C[1], lw=2.5,
               label=f"alternative (scaled {f:.2f}x), equally good fit")
    ax[0].set_xlabel("PC1 score"); ax[0].set_ylabel("PC2 score")
    ax[0].set_title("Linear mixing: pixels fill a simplex")
    ax[0].legend(fontsize=10, loc="best")
    for k in range(3):
        ax[1].plot(E, M[k], color=C[k], lw=2)
        ax[1].plot(E, Mb[k] / Mb[k].max(), color=C[k], lw=1.5, ls="--")
    ax[1].set_title("Endmember spectra: true (solid) vs alternative (dashed)\n"
                    "both non-negative, both reproduce every pixel exactly")
    ax[1].set_xlabel("Energy loss (eV)")
    fig.tight_layout(); fig.savefig(OUT / "nmf_rotational_ambiguity.png", dpi=140); plt.close(fig)


def fig_eels_preprocessing():
    rng = np.random.default_rng(1)
    E = np.linspace(350, 700, 700)
    A_bg, r = 4e9, 3.0
    bg = A_bg * E ** -r
    signal = 0.9 * edge(E, 456, 60, wl=90) + 0.9 * edge(E, 532, 40, wl=30)
    spec = rng.poisson(bg + signal).astype(float)
    win = (E > 400) & (E < 445)
    coef = np.polyfit(np.log(E[win]), np.log(spec[win]), 1)
    bg_fit = np.exp(np.polyval(coef, np.log(E)))

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    ax[0].plot(E, spec, color=C[0], lw=1, label="raw counts")
    ax[0].plot(E, bg_fit, color=C[1], lw=2, label=f"power law $AE^{{-r}}$, r={-coef[0]:.2f}")
    ax[0].axvspan(400, 445, color=C[1], alpha=0.15, label="fit window")
    ax[0].set_yscale("log"); ax[0].set_xlabel("Energy loss (eV)"); ax[0].set_ylabel("Counts")
    ax[0].set_title("1. Fit pre-edge background"); ax[0].legend(fontsize=9)
    ax[1].plot(E, spec - bg_fit, color=C[0], lw=1, label="raw - fit")
    ax[1].plot(E, signal, color=C[2], lw=2, ls="--", label="true edges")
    ax[1].axhline(0, color="#888", lw=0.8)
    ax[1].set_xlabel("Energy loss (eV)"); ax[1].set_title("2. Subtract: Ti-L$_{2,3}$ and O-K edges")
    ax[1].legend(fontsize=9)

    # energy drift -> derivative-shaped PC
    E2 = np.linspace(440, 480, 200)
    base = edge(E2, 456, 1, wl=2)
    shifts = rng.normal(0, 0.8, 300)
    X = np.array([edge(E2, 456 + d, 1, wl=2) for d in shifts])
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    deriv = np.gradient(base, E2)
    pc1 = Vt[0] * np.sign(Vt[0] @ deriv)
    ax[2].plot(E2, base / base.max(), color=C[0], lw=2, label="edge (norm.)")
    ax[2].plot(E2, pc1 / np.abs(pc1).max(), color=C[1], lw=2, label="PC1 of drifting stack")
    ax[2].plot(E2, deriv / np.abs(deriv).max(), color=C[2], lw=1.5, ls="--", label="dI/dE")
    ax[2].set_xlabel("Energy loss (eV)")
    ax[2].set_title("3. Align first: drift makes PC1 = derivative")
    ax[2].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "eels_preprocessing.png", dpi=140); plt.close(fig)


def fig_poisson_weighting():
    rng = np.random.default_rng(2)
    ne, N = 300, 2500
    E = np.linspace(400, 900, ne)
    bgshape = (E / 400) ** -3
    thick = rng.uniform(0.5, 2.0, N)                     # thickness varies -> strong bg
    frac = np.clip(rng.normal(0.5, 0.25, N), 0, 1)       # weak edge component
    weak = edge(E, 708, 0.04, wl=0.08)
    clean = 100 * (np.outer(thick, bgshape) + np.outer(frac * thick, weak))
    X = rng.poisson(clean).astype(float)

    def scree(Y):
        s = np.linalg.svd(Y - Y.mean(0), compute_uv=False)
        return s ** 2 / (s ** 2).sum()

    g = X.mean(1, keepdims=True); h = X.mean(0, keepdims=True)
    W = 1.0 / np.sqrt(g @ h / X.mean())                  # Keenan-Kotula style weighting
    s_raw, s_w = scree(X), scree(X * W)
    k = np.arange(1, 21)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for a, s, t in [(ax[0], s_raw, "Unweighted PCA"), (ax[1], s_w, "Poisson-weighted PCA")]:
        a.semilogy(k, s[:20], "o-", color=C[0], lw=2)
        a.set_title(t); a.set_xlabel("Component k"); a.set_ylabel("Variance fraction (log)")
        a.axvline(2.5, color=C[1], ls="--", lw=1.5)
    ax[0].text(3, s_raw[1] * 1.5, f"PC2 / PC3 = {s_raw[1]/s_raw[2]:.1f}\nweak edge barely above a sloped floor", fontsize=11)
    ax[1].text(3, s_w[1] * 1.5, f"PC2 / PC3 = {s_w[1]/s_w[2]:.1f}\nflat noise floor, clear elbow", fontsize=11)
    for a in ax:
        a.set_xticks(range(0, 21, 2))
    fig.suptitle("Thickness-varying background + weak Fe-L edge, Poisson counts", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "poisson_weighting_scree.png", dpi=140); plt.close(fig)


def fig_truncation_bias():
    rng = np.random.default_rng(3)
    ny = nx = 64
    E = np.linspace(400, 900, 300)
    M = make_endmembers(E)
    mn = edge(E, 640, 0.6, wl=1.3); mn /= mn.max()          # rare Mn-oxide-like phase
    A = make_abundances(ny, nx)
    yy, xx = np.mgrid[0:ny, 0:nx]
    rare = ((xx - 50) ** 2 + (yy - 14) ** 2 < 5).ravel() * 0.7  # 13 px = 0.3 % of pixels
    clean = (A * (1 - rare[:, None])) @ M + np.outer(rare, mn) + 0.05
    dose = 60
    noisy = rng.poisson(clean * dose) / dose
    mu = noisy.mean(0)
    U, s, Vt = np.linalg.svd(noisy - mu, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()

    def chi2_map(K):
        rec = mu + (U[:, :K] * s[:K]) @ Vt[:K]
        # Poisson-normalised residual: ~1 per channel if only noise is left
        r = (noisy - rec) ** 2 / np.clip(rec, 0.05, None) * dose
        return r.mean(1).reshape(ny, nx)

    r2, r3 = chi2_map(2), chi2_map(8)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    ax[0].semilogy(np.arange(1, 16), var[:15], "o-", color=C[0], lw=2)
    ax[0].axvline(2.5, color=C[1], ls="--")
    ax[0].set_title("Scree: clear elbow at k=2"); ax[0].set_xlabel("Component k")
    ax[1].imshow(rare.reshape(ny, nx), cmap="magma"); ax[1].set_title("Rare phase: 13 px (0.3 %)")
    vmax = np.percentile(r2, 99.9)
    ax[2].imshow(r2, cmap="magma", vmin=0.7, vmax=vmax); ax[2].set_title("Normalised residual at the elbow, K=2")
    ax[3].imshow(r3, cmap="magma", vmin=0.7, vmax=vmax); ax[3].set_title("K=8: absorbed, but 6 noise PCs added back")
    for a in ax[1:]:
        a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "pca_truncation_bias.png", dpi=140); plt.close(fig)
    print(f"truncation fig: var[:5]={var[:5].round(4)}")


if __name__ == "__main__":
    fig_truth_and_pca_vs_nmf()
    fig_rotational_ambiguity()
    fig_eels_preprocessing()
    fig_poisson_weighting()
    fig_truncation_bias()
    print("done")
