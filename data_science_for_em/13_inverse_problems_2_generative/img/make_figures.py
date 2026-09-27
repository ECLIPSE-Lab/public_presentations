"""Reproducible figures for DSEM Week 13 (inverse problems II: ptychography, generative priors & synthesis).

Generates the figures added in the WS 26/27 update:
  - course_arc.png            the NEW 13-week arc (WS 26/27) as one pipeline
  - xai_ladder.png            E1-E6 explainability ladder mapped to the weeks where each level was taught
  - trust_failures_recap.png  the four recurring trust failures (W4 leakage, W7 shortcut, W11 miscalibration/OOD, W13 hallucination)
  - dps_toy.png               diffusion posterior sampling on a 2-D toy prior (exact mixture score + DPS guidance)

gd_vs_epie.png and epie_convergence.png are written by notebooks/week13_ptychography.ipynb.
spa_vs_multislice.png, ms_ptycho_prsco3.png, resolution_history.png are copied from the SS25 deck
(09_imaging_inverse_problems2/ptycho_images, slide credit D. Muller / Chen et al. 2021).

Run from repo root:
    .venv/bin/python data_science_for_em/13_inverse_problems_2_generative/img/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13})
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1f77b4", "#e67e22", "#2ca02c", "#d62728", "#8e44ad", "#555555"


def box(ax, x, y, w, h, text, fc, ec=GREY, fs=11, weight="normal", color="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, color=color,
            wrap=True)


# ---------------------------------------------------------------------------
# 1. Course arc (new WS 26/27 13-week list)
# ---------------------------------------------------------------------------
def course_arc():
    weeks = [
        ("W1", "Python &\nEM data deluge"),
        ("W2", "Signal formation,\nnoise & loss"),
        ("W3", "Linear algebra,\nPCA & unmixing"),
        ("W4", "Regression &\nhonest validation"),
        ("W5", "Features &\ntree ensembles"),
        ("W6", "Neural nets &\nbackprop"),
        ("W7", "CNNs & U-Nets"),
        ("W8", "Small data &\nself-supervision"),
        ("W9", "Autoencoders &\nlatent spaces"),
        ("W10", "Transformers &\ngraphs"),
        ("W11", "Uncertainty, GPs &\nautonomous EM"),
        ("W12", "Inverse problems I:\nregularisation"),
        ("W13", "Inverse problems II:\nptycho & generative"),
    ]
    phases = [  # (first week idx, last idx, label, colour)
        (0, 3, "Data, noise & honest learning", "#aec7e8"),
        (4, 9, "Representations & model families", "#98df8a"),
        (10, 10, "Knowing what we don't know", "#ffbb78"),
        (11, 12, "Physics-based reconstruction", "#ff9896"),
    ]
    colour_of = {i: col for a_, b_, _, col in phases for i in range(a_, b_ + 1)}
    fig, ax = plt.subplots(figsize=(19, 8))
    ax.set_xlim(-0.3, 17.8); ax.set_ylim(-2.2, 5.4); ax.axis("off")
    w, h, dx = 2.15, 1.6, 2.55
    for i, (wk, title) in enumerate(weeks):
        row, col = (0, i) if i < 7 else (1, i - 7 + 0.5)
        x, y = col * dx, 3.2 - row * 2.6
        box(ax, x, y, w, h, f"{wk}\n{title}", colour_of[i], fs=12.5, ec=RED if i == 12 else GREY)
        if i < 12 and i != 6:
            ax.annotate("", xy=(x + dx, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5))
    ax.annotate("", xy=(0.5 * dx + w / 2, 0.6 + h), xytext=(6 * dx + w / 2, 3.2),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5, connectionstyle="arc3,rad=-0.05"))
    for k, (_, _, lab, col) in enumerate(phases):
        ax.add_patch(FancyBboxPatch((0.2 + k * 4.4, -0.55), 0.5, 0.35, boxstyle="round,pad=0.02", fc=col, ec=GREY))
        ax.text(0.85 + k * 4.4, -0.37, lab, va="center", fontsize=12)
    threads = [
        "Cross-cutting threads:  physics → loss / prior (W2, W8, W12–13)   ·   honest validation (W4 → every model)",
        "architecture = inductive bias: MLP (W6) → CNN (W7) → GNN / transformer (W10)   ·   explainability with every model family (E1–E6, W4–W13)",
    ]
    for k, t in enumerate(threads):
        ax.text(8.75, -1.2 - 0.5 * k, t, ha="center", va="center", fontsize=12.5, color="#333333")
    ax.set_title("The 13-week arc (WS 26/27): from raw EM arrays to trustworthy, physics-based reconstruction",
                 fontsize=16, weight="bold", pad=4)
    fig.savefig(OUT / "course_arc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. E1-E6 explainability ladder mapped to the course
# ---------------------------------------------------------------------------
def xai_ladder():
    levels = [
        ("E1  Data", "What data was used? provenance, dose, splits",
         "W2 data-formation chain & FAIR metadata · W4 leakage taxonomy"),
        ("E2  Process", "Which physical process does the model describe?",
         "W2 noise model → loss · W8 physics-respecting augmentation · W12–13 forward models"),
        ("E3  Feature", "Which inputs matter (globally)?",
         "W4 coefficients (E1/E2 intro) · W5 permutation importance & SHAP · W9 latent-space pitfalls"),
        ("E4  Model", "How does the model work? architecture, training",
         "W6 training diagnostics · W7 ResNet/U-Net inductive bias · W10 attention ≠ explanation"),
        ("E5  Prediction", "Why this output for this sample — and how sure?",
         "W7 saliency, Grad-CAM, occlusion · W11 calibrated uncertainty, conformal, OOD gate"),
        ("E6  Decision", "What action follows, and who signs it off?",
         "W11 BO / autonomous acquisition · W13 trusting a reconstruction (residuals, ensembles, dose check)"),
    ]
    cols = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c"]
    fig, ax = plt.subplots(figsize=(17, 7.5))
    ax.set_xlim(0, 17); ax.set_ylim(-0.3, 6.6); ax.axis("off")
    for k, (lev, q, where) in enumerate(levels):
        y = k * 1.0
        indent = k * 0.35
        box(ax, indent, y, 2.6, 0.8, lev, cols[k], fs=13, weight="bold", color="white" if k >= 3 else "black")
        ax.text(indent + 2.85, y + 0.55, q, fontsize=12.5, va="center", weight="bold")
        ax.text(indent + 2.85, y + 0.22, where, fontsize=11.5, va="center", color="#333333")
    ax.annotate("", xy=(0.15, 6.2), xytext=(0.15, -0.1), arrowprops=dict(arrowstyle="-|>", lw=2, color=GREY))
    ax.text(16.9, 6.35, "data → process → features → model → prediction → decision  (the course pipeline)",
            ha="right", fontsize=12, style="italic", color=GREY)
    ax.set_title("The E1–E6 explainability ladder [Neuer 2024] — and where each rung was taught in this course",
                 fontsize=15, weight="bold")
    fig.savefig(OUT / "xai_ladder.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Recurring trust failures
# ---------------------------------------------------------------------------
def trust_failures():
    items = [
        ("Week 4\nLeakage", "Same specimen in train\nand test → inflated score", "#f8d0d0", RED),
        ("Week 7\nShortcut learning", "Classifier keys on a corner\nartefact, saliency reveals it", "#fde2c4", ORANGE),
        ("Week 11\nMiscalibration / OOD", "95 % intervals cover 60 %;\nconfident on unseen data", "#d8ecd0", GREEN),
        ("Week 13\nHallucination", "Generative prior invents\nstructure not in the data", "#e8daef", PURPLE),
    ]
    fig, ax = plt.subplots(figsize=(17, 5.2))
    ax.set_xlim(0, 17); ax.set_ylim(0, 5.2); ax.axis("off")
    for k, (t, d, fc, ec) in enumerate(items):
        x = 0.3 + k * 4.2
        ax.add_patch(FancyBboxPatch((x, 1.9), 3.5, 2.5, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=2))
        ax.text(x + 1.75, 3.75, t, ha="center", va="center", fontsize=14, weight="bold", color=ec)
        ax.text(x + 1.75, 2.6, d, ha="center", va="center", fontsize=12, style="italic")
        if k < 3:
            ax.annotate("", xy=(x + 4.15, 3.15), xytext=(x + 3.55, 3.15),
                        arrowprops=dict(arrowstyle="-|>", lw=2, color=GREY))
    ax.add_patch(FancyBboxPatch((2.0, 0.3), 13, 1.05, boxstyle="round,pad=0.03", fc="#fffbe6", ec=GREY))
    ax.text(8.5, 0.82, "Shared root cause: the model exploited a statistical association (or a prior) instead of the\n"
                       "physical signal — and a single accuracy number could not reveal it.",
            ha="center", va="center", fontsize=12.5)
    ax.set_title("Recurring trust failures across the course", fontsize=15, weight="bold")
    fig.savefig(OUT / "trust_failures_recap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Diffusion posterior sampling on a 2-D toy
# ---------------------------------------------------------------------------
def dps_toy():
    torch.manual_seed(0)
    # prior: mixture of 4 anisotropic Gaussians ("four structural variants")
    mus = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
    sig = 0.45
    wts = torch.full((4,), 0.25)
    T = 400
    betas = torch.linspace(1e-4, 0.03, T)
    alphas = 1 - betas
    abar = torch.cumprod(alphas, 0)

    def log_pt(x, t):  # log density of the noised prior at step t (exact, closed form)
        m = abar[t].sqrt() * mus
        var = abar[t] * sig ** 2 + (1 - abar[t])
        d2 = ((x[:, None, :] - m[None]) ** 2).sum(-1)
        return torch.logsumexp(torch.log(wts)[None] - 0.5 * d2 / var - torch.log(2 * np.pi * var), dim=1)

    def score(x, t):
        x = x.detach().requires_grad_(True)
        g, = torch.autograd.grad(log_pt(x, t).sum(), x)
        return g

    # measurement: y = x_1 + noise (we only observe the first coordinate, like a projection)
    H = torch.tensor([1.0, 0.0])
    y_obs, sig_y = 1.6, 0.35

    def sample(n, guided, zeta=0.02):
        x = torch.randn(n, 2)
        for t in range(T - 1, -1, -1):
            x = x.detach().requires_grad_(True)
            s = score(x, t)
            if guided:
                # Tweedie estimate of the clean sample, then gradient of the data misfit w.r.t. x_t
                x0 = (x + (1 - abar[t]) * s) / abar[t].sqrt()
                norm = ((y_obs - x0 @ H) ** 2 + 1e-8).sqrt().sum()   # DPS: gradient of ||y - H x0_hat||
                g, = torch.autograd.grad(norm, x)
            with torch.no_grad():
                mean = (x + betas[t] * s) / alphas[t].sqrt()
                if guided:
                    mean = mean - zeta * g
                x = mean + (betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
        return x.detach().numpy()

    prior_s = sample(600, guided=False)
    post_s = sample(600, guided=True)
    # exact posterior by importance-resampling prior samples
    rng = np.random.default_rng(0)
    comp = rng.choice(4, 20000)
    ex = mus.numpy()[comp] + sig * rng.standard_normal((20000, 2))
    lw = -0.5 * (y_obs - ex[:, 0]) ** 2 / sig_y ** 2
    p = np.exp(lw - lw.max()); p /= p.sum()
    exact = ex[rng.choice(20000, 600, p=p)]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharex=True, sharey=True)
    for ax in axes:
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
        ax.set_xlabel("$x_1$ (measured direction)")
    axes[0].set_ylabel("$x_2$ (null space of $H$)")
    axes[0].scatter(prior_s[:, 0], prior_s[:, 1], s=6, c=BLUE, alpha=0.6)
    axes[0].set_title("Learned prior $p(x)$:\nreverse diffusion samples")
    for ax in axes[1:]:
        ax.axvspan(y_obs - sig_y, y_obs + sig_y, color=ORANGE, alpha=0.18, label="likelihood $p(y|x)$, $y=x_1+n$")
    axes[1].scatter(exact[:, 0], exact[:, 1], s=6, c=GREY, alpha=0.6)
    axes[1].set_title("Exact posterior $p(x|y)$\n(reference)")
    axes[2].scatter(post_s[:, 0], post_s[:, 1], s=6, c=RED, alpha=0.6)
    axes[2].set_title("Diffusion posterior sampling:\nprior score + data-consistency gradient")
    axes[1].legend(loc="lower left", fontsize=10)
    frac_top = (post_s[:, 1] > 0).mean()
    axes[2].text(-3.8, 3.7, f"upper / lower mode: {100 * frac_top:.0f} % / {100 * (1 - frac_top):.0f} %\n"
                 f"$x_1$ spread: {post_s[:, 0].std():.2f} (exact {exact[:, 0].std():.2f})\n"
                 "→ DPS is an approximation", fontsize=10.5, va="top")
    axes[1].text(-3.8, 3.7, f"upper / lower mode: {100 * (exact[:, 1] > 0).mean():.0f} % / "
                 f"{100 * (exact[:, 1] <= 0).mean():.0f} %\n$x_1$ spread: {exact[:, 0].std():.2f}",
                 fontsize=10.5, va="top")
    fig.suptitle("The data fix $x_1$; the prior alone decides $x_2$ — two equally plausible answers", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "dps_toy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"dps_toy: upper-mode fraction {frac_top:.2f}; x1 mean/std = {post_s[:, 0].mean():.2f}/{post_s[:, 0].std():.2f} "
          f"(exact {exact[:, 0].mean():.2f}/{exact[:, 0].std():.2f})")


if __name__ == "__main__":
    course_arc()
    xai_ladder()
    trust_failures()
    dps_toy()
    print("figures written to", OUT)
