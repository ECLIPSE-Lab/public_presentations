"""Reproducible figures for DSEM Week 9 (Unsupervised learning, autoencoders & latent spaces).

Run from the repo root:
    .venv/bin/python data_science_for_em/09_unsupervised_latent/img/make_figures.py

The script executes the code cells of notebooks/week09_autoencoder_vae.ipynb (same seeds),
so every number on the new slides matches the executed notebook. It then generates
(into this img/ folder):
    vae_latent_prior.png       - negative-ELBO terms, 2-D VAE latent vs prior, spectra sampled from the prior
    gmm_interface_trap.png     - BIC curve, GMM ellipses + A->B interface path, responsibilities along the path
    tsne_perplexity_sweep.png  - same AE latent codes embedded with t-SNE at 4 perplexities
    beta_active_dims.png       - mean posterior sigma per latent dim for beta = 0.1, 1, 4
    latent_map_4dstem.png      - synthetic 4D-STEM scan -> per-pixel latent coordinates -> latent/cluster maps

Copied (not generated) figures in this folder:
    vae_reparam_kingma.png, vae_aggregate_posterior.png   (from MFML Unit 11)
    em_gmm_iterations.png                                 (from MFML Unit 5; Bishop 2006 Fig. 9.8)
    kalinin2021_*.jpeg                                    (Kalinin et al., Sci. Adv. 2021, via SS25 DSEM deck)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

OUT = Path(__file__).resolve().parent
NB = OUT.parents[1] / "notebooks" / "week09_autoencoder_vae.ipynb"
plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14})


def run_notebook():
    """Execute all code cells of the Week 9 notebook in one namespace."""
    ns = {"__name__": "__nb__"}
    cells = json.loads(NB.read_text())["cells"]
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        exec(compile(src, "<nb>", "exec"), ns)
        plt.close("all")
    return ns


def fig_vae(ns):
    E, mu_np, labels, hist = ns["E"], ns["mu_np"], ns["labels_true"], ns["vae_hist"]
    cols, names = ns["phase_cols"], ns["phase_names_str"]
    torch = ns["torch"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
    ax = axes[0]
    ax.plot(hist[:, 0], color="#3498db", lw=2, label="reconstruction  $-\\log p_\\theta(x|z)$")
    ax.plot(hist[:, 1], color="#e74c3c", lw=2, label="KL$(q_\\phi(z|x)\\,\\|\\,p(z))$")
    ax.axvline(50, color="grey", ls=":", label="end of KL warm-up")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("nats per spectrum")
    ax.set_title("(a) the two ELBO terms during training"); ax.legend(fontsize=12)
    ax = axes[1]
    for k in range(4):
        m = labels == k
        ax.scatter(mu_np[m, 0], mu_np[m, 1], c=cols[k], s=18, alpha=0.8, label=names[k])
    th = np.linspace(0, 2 * np.pi, 200)
    for r in (1, 2):
        ax.plot(r * np.cos(th), r * np.sin(th), "k--", lw=1, alpha=0.5)
    ax.set_aspect("equal"); ax.set_xlabel("$\\mu_1$"); ax.set_ylabel("$\\mu_2$")
    ax.set_title("(b) VAE latent means vs prior (1σ, 2σ)"); ax.legend(fontsize=11, loc="upper right")
    ax = axes[2]
    torch.manual_seed(ns["SEED"])
    with torch.no_grad():
        xg = ns["vae"].decoder(torch.randn(6, 2)).numpy()
    for j in range(6):
        ax.plot(E, xg[j] + 0.3 * j, lw=2)
    ax.set_xlabel("Energy (eV)"); ax.set_ylabel("intensity (offset)")
    ax.set_title("(c) decoded from $z\\sim\\mathcal{N}(0,I)$")
    plt.tight_layout(); fig.savefig(OUT / "vae_latent_prior.png", dpi=150); plt.close(fig)


def fig_gmm(ns):
    mu_np, labels, gmm = ns["mu_np"], ns["labels_true"], ns["gmm"]
    cols, names, c2p = ns["phase_cols"], ns["phase_names_str"], ns["comp2phase"]
    fr, resp, mu_mix, bics, K_BIC = ns["fractions"], ns["resp_mix"], ns["mu_mix"], ns["bics"], ns["K_BIC"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
    axes[0].plot(list(ns["K_RANGE"]), bics, "o-", color="#8e44ad", lw=2)
    axes[0].axvline(K_BIC, color="grey", ls=":")
    axes[0].set_xlabel("number of components K"); axes[0].set_ylabel("BIC (lower = better)")
    axes[0].set_title(f"(a) GMM model selection: K* = {K_BIC}")
    ax = axes[1]
    for k in range(4):
        m = labels == k
        ax.scatter(mu_np[m, 0], mu_np[m, 1], c=cols[k], s=12, alpha=0.45)
    for c in range(K_BIC):
        vals, vecs = np.linalg.eigh(gmm.covariances_[c])
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(gmm.means_[c], 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=ang,
                             fill=False, color=cols[c2p[c]], lw=2.5))
    ax.plot(mu_mix[:, 0], mu_mix[:, 1], "k.-", lw=1.2, ms=8, label="A→B linear mixtures")
    ax.set_aspect("equal"); ax.set_xlabel("$\\mu_1$"); ax.set_ylabel("$\\mu_2$"); ax.legend(fontsize=12)
    ax.set_title("(b) GMM 2σ ellipses on the VAE latent")
    ax = axes[2]
    for c in range(K_BIC):
        ax.plot(fr, resp[:, c], "o-", ms=5, lw=2, color=cols[c2p[c]], label=names[c2p[c]])
    ax.axvspan(0.275, 0.575, color="#f39c12", alpha=0.12)
    ax.set_xlabel("fraction of phase B in the mixture"); ax.set_ylabel("responsibility $\\gamma_{ik}$")
    ax.set_title("(c) interface spectra → confidently 'Fe₃O₄'"); ax.legend(fontsize=11)
    plt.tight_layout(); fig.savefig(OUT / "gmm_interface_trap.png", dpi=150); plt.close(fig)


def fig_tsne(ns):
    rows, embeds, labels, cols = ns["rows"], ns["embeds"], ns["labels_true"], ns["phase_cols"]
    fig, axes = plt.subplots(1, len(rows), figsize=(22, 5.8))
    for ax, (p, rho, rr, sil2) in zip(axes, rows):
        Y = embeds[p]
        for k in range(4):
            m = labels == k
            ax.scatter(Y[m, 0], Y[m, 1], c=cols[k], s=10, alpha=0.85)
        ax.set_title(f"perplexity {p}\nSpearman ρ = {rho:.2f} · size ratio {rr:.2f}", fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); fig.savefig(OUT / "tsne_perplexity_sweep.png", dpi=150); plt.close(fig)


def fig_beta(ns):
    torch, rows_b = ns["torch"], ns["beta_rows"]
    # recompute per-dimension sigma for the plot (same seeds as the notebook's Exercise B)
    sig = []
    for beta in ns["BETAS"]:
        torch.manual_seed(ns["SEED"])
        v4 = ns["SpectralVAE"](latent_dim=4)
        ns["train_vae"](v4, ns["Xn_t"], ns["Xc_t"], beta=beta)
        v4.eval()
        with torch.no_grad():
            _, lv = v4.encode(ns["Xn_t"])
        sig.append(torch.exp(0.5 * lv).mean(0).numpy())
    sig = np.array(sig)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    w = 0.26
    for i, (beta, mse, kl, nact, ari) in enumerate(rows_b):
        ax.bar(np.arange(4) + (i - 1) * w, sig[i], w,
               label=f"β = {beta:g}: {nact}/4 active, KL = {kl:.1f}, MSE = {mse * 1e4:.1f}e-4")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.text(3.45, 0.52, "active threshold", ha="right", fontsize=11)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"$z_{j + 1}$" for j in range(4)])
    ax.set_ylabel("mean posterior σ (prior σ = 1)"); ax.set_ylim(0, 1.5)
    ax.set_title("4-D VAE: the KL term switches latent dimensions off")
    ax.legend(fontsize=11, loc="upper left")
    plt.tight_layout(); fig.savefig(OUT / "beta_active_dims.png", dpi=150); plt.close(fig)


def fig_4dstem():
    """Synthetic 4D-STEM: 2 grains (rotated spot patterns) + thickness gradient -> latent maps."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(0)
    S, Q = 40, 32                        # scan S x S, pattern Q x Q
    yy, xx = np.mgrid[0:S, 0:S]
    boundary = 0.45 * S + 0.25 * (xx - S / 2) + 2.5 * np.sin(xx / 5)
    grain = (yy > boundary).astype(int)          # 0 / 1
    angle = np.where(grain == 1, 28.0, 0.0) + 3.0 * (xx / S)   # small continuous rotation (bending)
    thick = 0.6 + 0.8 * (xx / S)                  # thickness gradient -> intensity
    ky, kx = np.mgrid[0:Q, 0:Q] - Q / 2
    g = 7.0
    base = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)], float)
    X = np.zeros((S * S, Q * Q))
    for n, (a, t) in enumerate(zip(angle.ravel(), thick.ravel())):
        r = np.deg2rad(a); R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        spots = (base * g) @ R.T
        pat = 4.0 * np.exp(-(kx ** 2 + ky ** 2) / 3.0)
        for sy, sx in spots:
            pat += t * np.exp(-((ky - sy) ** 2 + (kx - sx) ** 2) / 2.0)
        X[n] = rng.poisson(pat.ravel() * 20) / 20.0
    Xs = np.sqrt(X)                               # Anscombe-like variance stabilisation (Week 2)
    Z = PCA(n_components=3, random_state=0).fit_transform(Xs - Xs.mean(0))
    lab = GaussianMixture(2, random_state=0, n_init=3).fit_predict(Z)
    maps = Z.reshape(S, S, 3)
    rgb = (maps - maps.min((0, 1))) / (np.ptp(maps, axis=(0, 1)) + 1e-9)
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
    for ax, idx, ttl in [(axes[0], (int(0.2 * S), 5), "pixel in grain 1"), (axes[1], (int(0.85 * S), 30), "pixel in grain 2")]:
        ax.imshow(X.reshape(S, S, Q, Q)[idx], cmap="magma"); ax.set_title(f"diffraction pattern\n{ttl}")
        ax.axis("off")
    for ax, j in [(axes[2], 0), (axes[3], 1)]:
        im = ax.imshow(maps[..., j], cmap="RdBu_r"); ax.set_title(f"latent map $z_{j + 1}(x, y)$")
        ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    axes[4].imshow(rgb); axes[4].contour(lab.reshape(S, S), levels=[0.5], colors="w", linewidths=2)
    axes[4].set_title("RGB = $(z_1, z_2, z_3)$\n+ GMM boundary (white)"); axes[4].axis("off")
    plt.tight_layout(); fig.savefig(OUT / "latent_map_4dstem.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_4dstem()
    ns = run_notebook()
    fig_vae(ns)
    fig_gmm(ns)
    fig_tsne(ns)
    fig_beta(ns)
    print("figures written to", OUT)
