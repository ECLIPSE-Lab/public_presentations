"""CRISP-DM walked through one EM example: nanoparticle sizes from HAADF images.

All images are synthetic: Gaussian-profile particles with log-normal diameters,
imaged with Poisson noise. The pipeline (smooth -> Otsu threshold -> label) is a
simple baseline, and every number in the figure is computed from these data.

Run:  python make_crispdm_nanoparticles.py   ->  crispdm_nanoparticles.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

BG, FG, MUTED = "#0b0f19", "#dbe4f0", "#94a3b8"
BLUE, VIOLET, AMBER = "#60a5fa", "#a78bfa", "#fbbf24"
PIX_NM = 0.2          # pixel size
N = 256               # image size in pixels
rng = np.random.default_rng(42)


def specimen(n_particles, dose, mean_d_nm=3.5):
    """Synthetic HAADF image of supported nanoparticles, and the true diameters."""
    yy, xx = np.mgrid[:N, :N]
    clean = np.full((N, N), 0.15)
    diam = []
    for _ in range(n_particles):
        d = rng.lognormal(np.log(mean_d_nm), 0.25)
        r = d / 2 / PIX_NM
        cy, cx = rng.uniform(r + 4, N - r - 4, size=2)
        rr2 = (yy - cy) ** 2 + (xx - cx) ** 2
        if clean[rr2 < (r + 3) ** 2].max() > 0.2:   # skip overlaps
            continue
        clean += np.exp(-(rr2 / r**2) ** 4)           # flat-top particle
        diam.append(d)
    return rng.poisson(clean * dose) / dose, np.array(diam)


def measure(img):
    """Baseline: Gaussian smoothing, Otsu threshold, equivalent-disc diameters."""
    smooth = ndi.gaussian_filter(img, 1.5)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    mask = smooth > threshold_otsu(smooth)
    lab, n = ndi.label(mask)
    areas = ndi.sum(mask, lab, range(1, n + 1)) * PIX_NM**2
    diam = 2 * np.sqrt(areas[areas > 1.0] / np.pi)
    return smooth, mask, diam


# Phase 2-5 data: specimens A and B are used for tuning, C is held out.
train = [specimen(70, dose=40) for _ in range(2)]
test_img, test_true = specimen(70, dose=40)
smooth, mask, test_pred = measure(test_img)

# Phase 6: 30 live frames; from frame 20 on, a thicker, uneven support film
# adds background with a gradient, and the global threshold breaks down.
ramp = 0.8 * (0.5 + np.mgrid[:N, :N][1] / N)
live = np.array([measure(specimen(70, dose=40)[0] + (ramp if f >= 20 else 0))[2].mean()
                 for f in range(30)])
ref = live[:20]
band = ref.mean() + 3 * ref.std() * np.array([-1, 1])   # control limits, mean ± 3σ

plt.rcParams.update({"font.family": "Inter", "font.size": 10, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": MUTED, "axes.facecolor": BG})
fig = plt.figure(figsize=(12.8, 5.9), dpi=150, facecolor=BG)
gs = fig.add_gridspec(2, 3, left=0.04, right=0.985, top=0.9, bottom=0.12,
                      wspace=0.32, hspace=0.62)
axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
titles = ["1  Question", "2  Data understanding", "3  Data preparation",
          "4  Modelling", "5  Evaluation", "6  Deployment"]
subs = ["Does catalyst activity depend on particle size?",
        f"raw HAADF · {PIX_NM} nm/px · Poisson noise",
        "smooth + normalise · split by specimen",
        "baseline: threshold + label (later: U-Net)",
        "held-out specimen C, never seen in tuning",
        "live size readout at the microscope"]
for ax, t, s in zip(axes, titles, subs):
    ax.set_title(f"{t}\n", loc="left", fontsize=13, fontweight="bold", color="#f8fafc")
    ax.text(0, 1.04, s, transform=ax.transAxes, fontsize=9.5, color=MUTED)

# 1 Question: the hypothesis, drawn as a sketch (no data).
ax = axes[0]
d = np.linspace(1, 8, 200)
ax.plot(d, np.exp(-((d - 3.2) / 1.6) ** 2), color=AMBER, lw=2.5, ls="--")
ax.text(5.6, 0.45, "?", fontsize=34, color=AMBER, fontweight="bold")
ax.set_xlabel("particle diameter (nm)"); ax.set_ylabel("activity (a.u.)")
ax.set_yticks([]); ax.text(0.98, 0.92, "hypothesis, sketched", transform=ax.transAxes,
                           ha="right", fontsize=8.5, color=MUTED)

# 2 Data: raw image.
crop = (slice(0, 160), slice(0, 160))
axes[1].imshow(test_img[crop], cmap="gray")
# 3 Preparation: smoothed image; three specimens, C held out.
axes[2].imshow(smooth[crop], cmap="gray")
for i, (lbl, col) in enumerate([("A train", BLUE), ("B train", BLUE), ("C test", AMBER)]):
    axes[2].text(4 + 52 * i, 12, lbl, color=BG, fontsize=8.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none"))
# 4 Model: contours of the segmentation on the raw image.
axes[3].imshow(test_img[crop], cmap="gray")
axes[3].contour(mask[crop], levels=[0.5], colors=[VIOLET], linewidths=1.4)
for ax in axes[1:4]:
    ax.set_xticks([]); ax.set_yticks([])
    ax.plot([115, 115 + 5 / PIX_NM], [148, 148], color="white", lw=3)
    ax.text(115 + 2.5 / PIX_NM, 141, "5 nm", color="white", ha="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.8))

# 5 Evaluation: true vs measured size distribution on specimen C.
ax = axes[4]
bins = np.linspace(1.5, 6.5, 16)
ax.hist(test_true, bins, color=MUTED, alpha=0.6, label=f"true (n={len(test_true)})")
ax.hist(test_pred, bins, histtype="step", lw=2.2, color=VIOLET,
        label=f"measured (n={len(test_pred)})")
err = test_pred.mean() - test_true.mean()
ax.set_xlabel("diameter (nm)"); ax.set_ylabel("count")
ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.text(0.02, 0.9, f"mean error {err:+.2f} nm", transform=ax.transAxes, fontsize=9.5)
ax.text(0.02, 0.78, "fails? → back to 3", transform=ax.transAxes, fontsize=9.5, color=AMBER)

# 6 Deployment: per-frame mean diameter; drift after the dose drop.
ax = axes[5]
f = np.arange(30)
ax.axhspan(*band, color=BLUE, alpha=0.15, lw=0)
ok = (live > band[0]) & (live < band[1])
ax.plot(f, live, color=MUTED, lw=1)
ax.scatter(f[ok], live[ok], s=16, color=BLUE, zorder=3)
ax.scatter(f[~ok], live[~ok], s=16, color=AMBER, zorder=3)
ax.axvline(19.5, color=AMBER, lw=1, ls=":")
ax.text(19.2, ax.get_ylim()[1] - 0.05 * np.ptp(ax.get_ylim()), "thicker support\n→ drift alarm,\nretrain",
        color=AMBER, fontsize=8.5, va="top", ha="right")
ax.text(0.5, band[1] + 0.08, "mean ± 3σ of frames 0–19", color=BLUE, fontsize=8)
ax.set_xlabel("frame"); ax.set_ylabel("mean diameter (nm)")

# Arrows between panels in each row; the panel numbers give the order.
def arrow(a, b, rad=0.0, color=MUTED):
    fig.patches.append(FancyArrowPatch(a, b, transform=fig.transFigure, color=color,
                                       arrowstyle="-|>", mutation_scale=16, lw=1.6,
                                       connectionstyle=f"arc3,rad={rad}"))
for x in (0.325, 0.655):
    arrow((x, 0.73), (x + 0.025, 0.73))
for x in (0.30, 0.655):
    arrow((x, 0.25), (x + 0.025, 0.25))
fig.text(0.04, 0.01, "Synthetic data: log-normal diameters, Poisson noise; panel 6 adds a background gradient from frame 20. "
         "Numbers computed from these images by the baseline in panel 4.",
         fontsize=8, color=MUTED)

out = Path(__file__).with_name("crispdm_nanoparticles.png")
fig.savefig(out, facecolor=BG)
print(f"saved {out}; test n_true={len(test_true)} n_pred={len(test_pred)} err={err:+.3f} nm; "
      f"live out of band: {np.flatnonzero(~ok).tolist()}")
