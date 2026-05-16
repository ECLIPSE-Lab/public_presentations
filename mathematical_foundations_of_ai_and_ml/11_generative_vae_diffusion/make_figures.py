"""
Locally generated figures for Unit 11 (VAE & Diffusion).

Replaces two borrowed Murphy plate-notation diagrams that were conceptually
wrong for a VAE (they depict *discrete-mixture* latent-variable models — GMM
and mixture-of-factor-analysers — whereas a VAE has a *continuous* z ~ N(0,I)
and a neural decoder).

  images/vae_aggregate_posterior.png  -> slide "VAE — the prior on z"
  images/vae_generation_pipeline.png  -> slide "Generating new data from a trained VAE"

Run:  python3 make_figures.py
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUT = "images"
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def fig_aggregate_posterior():
    """Per-x Gaussian posteriors q(z|x) collectively tiling the N(0,I) prior."""
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(6.6, 6.0))

    # Prior p(z) = N(0, I): 1σ / 2σ reference rings.
    for r, lbl in [(1.0, None), (2.0, None)]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls="--",
                                 ec="0.35", lw=1.8, zorder=1))
    ax.plot([], [], ls="--", color="0.35", lw=1.8,
            label=r"prior $p(z)=\mathcal{N}(0,I)$  (1σ, 2σ)")

    # ~14 per-x posteriors q(z|x): small Gaussians whose union covers the prior.
    n = 14
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = np.sqrt(rng.uniform(0.0, 1.0, n)) * 1.55
    means = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    cmap = plt.cm.viridis(np.linspace(0, 1, n))
    for (mx, my), c in zip(means, cmap):
        sx, sy = rng.uniform(0.30, 0.55, 2)
        th = rng.uniform(0, np.pi)
        for k, a in [(2.2, 0.12), (1.4, 0.22)]:
            e = Ellipse((mx, my), width=k * sx, height=k * sy,
                        angle=np.degrees(th), fc=c, ec="none",
                        alpha=a, zorder=2)
            ax.add_patch(e)
        ax.plot(mx, my, ".", color=c, ms=6, zorder=3)
    ax.plot([], [], "o", color=plt.cm.viridis(0.5),
            label=r"per-$x$ posteriors $q_\phi(z\mid x)$", alpha=0.6)

    ax.set_title("Aggregate posterior tiles the prior")
    ax.text(0.5, -0.13,
            r"KL term $\Rightarrow$ the union of $q_\phi(z\mid x)$ covers $\mathcal{N}(0,I)$"
            "\nso a fresh $z\\sim\\mathcal{N}(0,I)$ decodes to a valid sample",
            transform=ax.transAxes, ha="center", va="top", fontsize=13)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.savefig(f"{OUT}/vae_aggregate_posterior.png")
    plt.close(fig)


def fig_generation_pipeline():
    """VAE generative direction: sample z ~ N(0,I), push through decoder g_theta."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=(700, 2))

    # A fixed smooth nonlinear "decoder" g: R^2 -> R^2 (S-curve-like manifold),
    # purely illustrative of "Gaussian latent -> structured data".
    u = z[:, 0]
    v = z[:, 1]
    x1 = 1.6 * np.tanh(0.9 * u) + 0.18 * v
    x2 = 0.9 * np.sin(1.3 * u) + 0.55 * v + 0.25 * u
    color = u  # carry latent coordinate through so the mapping is visible

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 5.2))

    axL.add_patch(plt.Circle((0, 0), 1, fill=False, ls="--", ec="0.35", lw=1.6))
    axL.add_patch(plt.Circle((0, 0), 2, fill=False, ls="--", ec="0.35", lw=1.6))
    axL.scatter(z[:, 0], z[:, 1], c=color, cmap="plasma", s=10, alpha=0.7)
    axL.set_title(r"1.  sample  $z\sim\mathcal{N}(0,I)$")
    axL.set_xlabel(r"$z_1$")
    axL.set_ylabel(r"$z_2$")
    axL.set_xlim(-3.2, 3.2)
    axL.set_ylim(-3.2, 3.2)
    axL.set_aspect("equal")
    axL.grid(alpha=0.15)

    axR.scatter(x1, x2, c=color, cmap="plasma", s=10, alpha=0.7)
    axR.set_title(r"2.  $x_{\mathrm{new}} = g_\theta(z)$")
    axR.set_xlabel(r"$x_1$")
    axR.set_ylabel(r"$x_2$")
    axR.set_aspect("equal")
    axR.grid(alpha=0.15)

    # Arrow between panels: the decoder.
    fig.text(0.502, 0.52, r"$g_\theta$", ha="center", va="center",
             fontsize=20, fontweight="bold")
    fig.text(0.502, 0.43, "decoder", ha="center", va="center", fontsize=12)
    ann = fig.add_axes([0.455, 0.46, 0.09, 0.06])
    ann.axis("off")
    ann.annotate("", xy=(1, 0.5), xytext=(0, 0.5),
                 arrowprops=dict(arrowstyle="-|>", lw=2.4, color="0.2"))

    fig.suptitle("Generation = sample the prior, then decode",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.savefig(f"{OUT}/vae_generation_pipeline.png")
    plt.close(fig)


def fig_quality_vs_steps():
    """The real diffusion tradeoff: sample quality vs number of sampling steps
    (NFEs). Replaces a borrowed MH-proposal-scale figure whose accept/reject /
    mode-trapping failure modes do not transfer to diffusion sampling."""
    steps = np.array([1, 2, 4, 8, 16, 25, 50, 100, 250, 1000], dtype=float)
    q = 1.0 - np.exp(-steps / 16.0)            # saturating quality curve
    q = 0.05 + 0.93 * q / q.max()

    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    ax.axvspan(10, 25, color="tab:green", alpha=0.10, zorder=0)
    ax.plot(steps, q, "-o", lw=2.6, ms=6, color="tab:blue", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 25, 50, 100, 250, 1000])
    ax.set_xticklabels(["1", "2", "5", "10", "25", "50", "100", "250", "1000"])
    ax.set_xlim(0.9, 1200)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([])
    ax.set_xlabel("number of sampling steps  =  network calls per sample")
    ax.set_ylabel(r"sample quality  $\longrightarrow$")
    ax.set_title("Quality saturates — extra steps buy compute, not quality")

    ax.annotate("1 step:\nconsistency models\n(fast, slightly lower quality)",
                xy=(1, q[0]), xytext=(1.15, 0.44), fontsize=11.5,
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.6))
    ax.annotate("10–25 steps:\nDDIM / DPM-Solver\n(modern default)",
                xy=(20, np.interp(20, steps, q)), xytext=(2.0, 0.80),
                fontsize=11.5,
                arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=1.8))
    ax.annotate("1000 steps:\nvanilla DDPM\n(slow, no quality gain)",
                xy=(1000, q[-1]), xytext=(70, 0.30), fontsize=11.5,
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.6))
    ax.grid(alpha=0.18, which="both")
    fig.savefig(f"{OUT}/diffusion_quality_vs_steps.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_aggregate_posterior()
    fig_generation_pipeline()
    fig_quality_vs_steps()
    print("wrote images/vae_aggregate_posterior.png, "
          "images/vae_generation_pipeline.png, "
          "images/diffusion_quality_vs_steps.png")
