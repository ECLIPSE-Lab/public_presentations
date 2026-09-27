"""Reproducible figures for DSEM Week 8 (small data: augmentation, transfer & self-supervision).

Run from the repo root:
    .venv/bin/python data_science_for_em/08_small_data_self_supervised/img/make_figures.py

Figures written next to this script:
    ssl_views.png         - SimCLR positive/negative views of synthetic grain micrographs (+ an illegal zoom view)
    mae_masking.png       - masked-image-modelling input: 75 % of 4x4 patches of a HAADF-like lattice masked
    ssl_families.png      - schematic of the three SSL families (contrastive, masked, self-distillation)
    label_efficiency.png  - protocol comparison vs number of labels (numbers from the executed
                            notebook notebooks/week08_embeddings_transfer.ipynb, SEED=42)

Figures copied from other course decks (not generated here, see captions for attribution):
    simclr_framework_fig2.png (Chen et al. 2020), usam_fig1_overview.png (Archit et al.),
    emssl_fig1_pipeline.png (Kazimi et al. 2024), construction_zone_segmentation.png (Rakowski et al. 2024)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter, zoom

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 13, "savefig.dpi": 150, "savefig.bbox": "tight"})
S = 96


def voronoi(n_seeds, rng, size=S):
    seeds = rng.uniform(0, size, (n_seeds, 2))
    yy, xx = np.mgrid[:size, :size]
    gid = ((yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2).argmin(-1)
    img = rng.uniform(0.2, 0.9, n_seeds)[gid]
    b = np.zeros((size, size))
    b[:-1] += gid[:-1] != gid[1:]
    b[:, :-1] += gid[:, :-1] != gid[:, 1:]
    img = gaussian_filter(img - 0.45 * np.clip(gaussian_filter(b, 0.6) * 2, 0, 1), 0.8)
    return np.clip(rng.poisson(120 * np.clip(img, 0.01, 1)) / 120, 0, 1.2)


def fig_ssl_views():
    rng = np.random.default_rng(3)
    x = voronoi(14, rng)
    other = voronoi(40, np.random.default_rng(11))
    v1 = np.rot90(x, 1) * 1.15
    v2 = np.fliplr(x) * 0.8 + 0.05 + 0.06 * rng.standard_normal(x.shape)
    c = x[24:72, 24:72]
    v_zoom = zoom(c, S / c.shape[0], order=1)
    panels = [(x, "original micrograph $\\mathbf{x}$", "k"),
              (v1, "view 1: rot 90° + gain", "tab:green"),
              (v2, "view 2: flip + offset + noise", "tab:green"),
              (other, "another image (negative)", "tab:red"),
              (v_zoom, "zoom ×2: ILLEGAL here", "tab:red")]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    for ax, (im, t, col) in zip(axes, panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1.1)
        ax.set_title(t, fontsize=13, color=col)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(3)
    axes[4].plot([0, S - 1], [0, S - 1], color="tab:red", lw=3)
    axes[4].plot([0, S - 1], [S - 1, 0], color="tab:red", lw=3)
    fig.text(0.39, 0.06, "positive pair: pull embeddings together", ha="center", color="tab:green", fontsize=14)
    fig.text(0.70, 0.06, "push apart", ha="center", color="tab:red", fontsize=14)
    fig.text(0.90, 0.06, "zoom changes grain size = the label", ha="center", color="tab:red", fontsize=12)
    fig.savefig(OUT / "ssl_views.png"); plt.close(fig)


def haadf_lattice(size=64, a=6.0, rng=None):
    """Two grains: hexagonal columns (left) and a rotated square lattice (right), few vacancies."""
    yy, xx = np.mgrid[:size, :size].astype(float)
    pos = []
    for i in range(-2, int(size / a) + 3):
        for j in range(-2, int(size / (0.87 * a)) + 3):
            cx, cy = i * a + (j % 2) * a / 2, j * a * 0.87
            if cx < size / 2:
                pos.append((cx, cy))
    th = np.deg2rad(25)
    for i in range(-15, 16):
        for j in range(-15, 16):
            u, v = i * a * 1.1, j * a * 1.1
            cx, cy = size / 2 + 3 + u * np.cos(th) - v * np.sin(th), v * np.cos(th) + u * np.sin(th)
            if size / 2 + 1.5 <= cx < size + 3 and -3 < cy < size + 3:
                pos.append((cx, cy))
    img = np.zeros((size, size))
    for cx, cy in pos:
        amp = 1.0 if rng.random() > 0.05 else 0.3
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.1 ** 2))
    img = img / img.max()
    return np.clip(rng.poisson(60 * (img + 0.08)) / 60, 0, 1.3)


def fig_mae():
    rng = np.random.default_rng(0)
    img = haadf_lattice(rng=rng)
    p = 8
    n = img.shape[0] // p
    mask = rng.random((n, n)) < 0.75
    masked = img.copy()
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                masked[i * p:(i + 1) * p, j * p:(j + 1) * p] = np.nan
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("HAADF-like image (two grains)", fontsize=12)
    cm = plt.get_cmap("gray").copy(); cm.set_bad("#3b6ea8")
    axes[1].imshow(masked, cmap=cm); axes[1].set_title(f"encoder input: {100 * (1 - mask.mean()):.0f} % of patches visible", fontsize=12)
    loss_map = np.where(np.kron(mask, np.ones((p, p))) > 0, img, np.nan)
    cm2 = plt.get_cmap("magma").copy(); cm2.set_bad("white")
    axes[2].imshow(loss_map, cmap=cm2); axes[2].set_title("loss only on masked patches", fontsize=12)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Masked autoencoder (MAE): predict what is hidden — lattice periodicity makes it learnable", y=1.02)
    fig.savefig(OUT / "mae_masking.png"); plt.close(fig)


def box(ax, xy, w, h, text, fc):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec="k", lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=10)


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.5, color=kw.get("color", "k")))


def fig_families():
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.2))
    titles = ["Contrastive (SimCLR, MoCo)", "Masked modelling (MAE)", "Self-distillation (BYOL, DINO)"]
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off"); ax.set_title(t, fontsize=15, weight="bold")
    # contrastive
    ax = axes[0]
    box(ax, (0.2, 4.4), 2.0, 1.2, "view 1", "#dff0d8"); box(ax, (0.2, 1.4), 2.0, 1.2, "view 2", "#dff0d8")
    box(ax, (3.2, 4.4), 2.2, 1.2, "encoder f\n+ proj. g", "#e8eef8"); box(ax, (3.2, 1.4), 2.2, 1.2, "encoder f\n+ proj. g", "#e8eef8")
    arrow(ax, (2.2, 5.0), (3.2, 5.0)); arrow(ax, (2.2, 2.0), (3.2, 2.0))
    box(ax, (6.4, 2.7), 3.3, 1.6, "InfoNCE: twins close,\nother images far", "#fcf3cf")
    arrow(ax, (5.4, 5.0), (6.6, 4.3)); arrow(ax, (5.4, 2.0), (6.6, 2.7))
    ax.text(5, 0.4, "needs many negatives (large batch)", ha="center", fontsize=11, style="italic")
    # MAE
    ax = axes[1]
    box(ax, (0.2, 2.9), 2.2, 1.2, "image,\n75 % patches\nmasked", "#dff0d8")
    box(ax, (3.0, 2.9), 2.0, 1.2, "ViT encoder\n(visible only)", "#e8eef8")
    box(ax, (5.6, 2.9), 1.8, 1.2, "light\ndecoder", "#e8eef8")
    box(ax, (7.9, 2.9), 1.9, 1.2, "MSE on\nmasked pixels", "#fcf3cf")
    arrow(ax, (2.4, 3.5), (3.0, 3.5)); arrow(ax, (5.0, 3.5), (5.6, 3.5)); arrow(ax, (7.4, 3.5), (7.9, 3.5))
    ax.text(5, 0.4, "minimal augmentation, no negatives; cheap", ha="center", fontsize=11, style="italic")
    # DINO
    ax = axes[2]
    box(ax, (0.2, 4.4), 2.0, 1.2, "global crop", "#dff0d8"); box(ax, (0.2, 1.4), 2.0, 1.2, "local crop", "#dff0d8")
    box(ax, (3.0, 4.4), 2.8, 1.2, "teacher\n(EMA of student)", "#f2dede"); box(ax, (3.2, 1.4), 2.4, 1.2, "student", "#e8eef8")
    arrow(ax, (2.2, 5.0), (3.2, 5.0)); arrow(ax, (2.2, 2.0), (3.2, 2.0))
    box(ax, (6.5, 2.6), 3.4, 1.8, "cross-entropy:\nstudent matches\nteacher\n(centred, sharpened)", "#fcf3cf")
    arrow(ax, (5.6, 5.0), (6.8, 4.3)); arrow(ax, (5.6, 2.0), (6.8, 2.7))
    ax.annotate("", xy=(4.4, 4.4), xytext=(4.4, 2.6), arrowprops=dict(arrowstyle="-|>", ls="--", color="gray"))
    ax.text(4.55, 3.5, "EMA", color="gray", fontsize=11)
    ax.text(5, 0.4, "no negatives; collapse avoided by EMA + centring", ha="center", fontsize=11, style="italic")
    fig.savefig(OUT / "ssl_families.png"); plt.close(fig)


# Mean test accuracies from the executed notebook (SEED=42, 2 draws per N, 630 test images).
# Update these if the notebook is changed and re-run.
N_VALUES = [9, 30, 90, 300]
NB_RESULTS = {
    "LP: random init":              [0.424, 0.531, 0.576, 0.658],
    "LP: sim-supervised":           [0.685, 0.664, 0.769, 0.798],
    "LP: SimCLR (real, no labels)": [0.571, 0.662, 0.763, 0.821],
    "from scratch":                 [0.559, 0.613, 0.667, 0.624],
    "LP→FT: SimCLR":                [0.575, 0.650, 0.729, 0.852],
}


def fig_label_efficiency():
    styles = {"LP: random init": ("0.55", ":", "o"), "LP: sim-supervised": ("tab:blue", "-", "s"),
              "LP: SimCLR (real, no labels)": ("tab:green", "-", "^"), "from scratch": ("tab:red", "--", "x"),
              "LP→FT: SimCLR": ("tab:purple", "-", "D")}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k, v in NB_RESULTS.items():
        c, ls, m = styles[k]
        ax.plot(N_VALUES, v, color=c, ls=ls, marker=m, lw=2.5, ms=8, label=k)
    ax.axhline(1 / 3, color="k", lw=0.8, ls=":", label="chance (3 classes)")
    ax.set_xscale("log"); ax.set_xticks(N_VALUES); ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.set_xlabel("number of labelled real images N"); ax.set_ylabel("test accuracy (real instrument)")
    ax.set_ylim(0.3, 0.9); ax.grid(alpha=0.3); ax.legend(fontsize=11, loc="lower right")
    ax.set_title("Week 8 notebook: grain-size class, 32×32 synthetic 'real-instrument' images")
    fig.savefig(OUT / "label_efficiency.png"); plt.close(fig)


if __name__ == "__main__":
    fig_ssl_views()
    fig_mae()
    fig_families()
    fig_label_efficiency()
    print("figures written to", OUT)
