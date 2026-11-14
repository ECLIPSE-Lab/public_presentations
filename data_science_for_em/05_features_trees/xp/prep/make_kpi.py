"""Render the three small real-data pictures under the numbers of the opening KPI slide.

Usage: python xp/prep/make_kpi.py   (reads xp/data/columns.json and models.json, writes img/kpi_*.png)
Each picture uses the same data and numbers as the figures of make_figs.py. No number is typed by hand.

  kpi_table      a crop of image B with one marked candidate per class, and the three table rows that they become
  kpi_gap        random forest on the raw features: random 5-fold accuracy against the accuracy on a new image
  kpi_invariant  accuracy on a new image: raw features against the six invariant descriptors
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figs as F                                                   # data, colours and the image helper

OUT = F.OUT if hasattr(F, "OUT") else Path(__file__).resolve().parents[2] / "img"
TXT, MUTED = "#e2e8f0", "#94a3b8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "text.color": TXT, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 15})


def fig(w=4.6, h=3.0):
    f = plt.figure(figsize=(w, h), dpi=200)
    f.patch.set_alpha(0)
    return f


def done(f, name):
    p = OUT / f"kpi_{name}.png"
    f.savefig(p, transparent=True); plt.close(f); print(f"-> img/{p.name}")


# 1. from the image to the table: one candidate of each class in image B, and its row
k = 1; x0, y0, R = 6.9, 5.0, 2.2                                        # crop centre and half width (nm), across the interface
inb = (F.G == k) & (np.abs(F.POS[:, 1] - x0) < R - 0.45) & (np.abs(F.POS[:, 0] - y0) < R - 0.45)
cols = ("I_rel", "d1", "psi4"); V = np.stack([F.col(n) for n in cols], 1)
pick = []
for c in range(3):                                                      # the candidate nearest to the class median of the three columns
    i = np.nonzero(inb & (F.Y == c))[0]
    if len(i) == 0:
        raise SystemExit(f"no candidate of class {c} in the crop")
    a = (F.G == k) & (F.Y == c); med = np.median(V[a], 0)
    pick.append(i[np.argmin((((V[i] - med) / V[F.LAB].std(0)) ** 2).sum(1))])
f = fig(4.0, 3.0)
ax = f.add_axes([0.0, 0.5 - 0.47 * 2 / 3, 0.47, 0.47 * 4 / 3]); F.show(ax, k)
ax.set_xlim(x0 - R, x0 + R); ax.set_ylim(y0 + R, y0 - R); ax.axis("off")
I = F.D["images"][k]["img"]; s_ = I.shape[0] / F.D["images"][k]["size_nm"][0]
cr = I[int((y0 - R) * s_):int((y0 + R) * s_), int((x0 - R) * s_):int((x0 + R) * s_)]
ax.images[0].set_clim(np.percentile(cr, 1), np.percentile(cr, 99.8))
for i in pick:
    ax.plot(F.POS[i, 1], F.POS[i, 0], "o", ms=10, mfc="none", mec=F.CLS_COL[F.Y[i]], mew=2.4)
ax = f.add_axes([0.0, 0.0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.annotate("", (0.57, 0.5), (0.49, 0.5), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2.2, mutation_scale=18))
xs = (0.705, 0.825, 0.94)
for name, x in zip(cols, xs):
    ax.text(x, 0.70, name, ha="center", va="center", fontsize=11.5, color=MUTED)
ax.plot([0.60, 0.99], [0.65, 0.65], color=MUTED, lw=1)
print("kpi_table: image B, crop centre (%.1f, %.1f) nm, width %.1f nm" % (x0, y0, 2 * R))
for r, i in enumerate(pick):
    y = 0.575 - 0.115 * r
    ax.plot(0.625, y, "o", ms=8, color=F.CLS_COL[F.Y[i]])
    v = [F.col(n)[i] for n in cols]
    for val, x in zip(v, xs):
        ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=11.5)
    ax.plot([0.60, 0.99], [y - 0.057, y - 0.057], color=MUTED, lw=0.5, alpha=0.5)
    print(f"  {F.CLS[F.Y[i]]}: I_rel {v[0]:.2f}, d1 {v[1]:.2f} nm, psi4 {v[2]:.2f}  (x {F.POS[i, 1]:.2f}, y {F.POS[i, 0]:.2f} nm)")
ax.text(0.795, 0.22, "one row per candidate", ha="center", va="center", fontsize=10, color=MUTED)
done(f, "table")


def bars(name, vals, labels, colors, ghost=None):
    f = fig(); ax = f.add_axes([0.04, 0.2, 0.92, 0.76])
    ax.grid(False); ax.bar(range(len(vals)), vals, 0.62, color=colors)
    if ghost is not None:                                               # dashed level: the random 5-fold accuracy of the same model
        for i, g in enumerate(ghost):
            ax.plot([i - 0.31, i + 0.31], [g, g], color=TXT, lw=1.8, ls=(0, (4, 3)))
            ax.text(i, g + 1.5, f"random folds {g:.1f} %", ha="center", va="bottom", fontsize=10.5, color=MUTED)
    for i, v in enumerate(vals):
        ax.text(i, v - 6, f"{v:.0f} %", ha="center", va="top", fontsize=24, fontweight="bold", color="#0b0f19")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=14, color=TXT); ax.set_yticks([])
    ax.set_ylim(0, 106 if ghost is not None else 100); ax.set_xlim(-0.6, len(vals) - 0.4); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(0, color=MUTED, lw=1.5)
    done(f, name)


# 2. the gap: random forest on the raw features
L = F.M["ladder"]
r, n = 100 * L["raw"]["random forest"]["random"], 100 * L["raw"]["random forest"]["image"]
print(f"kpi_gap: raw features, random forest: random 5-fold {r:.1f} %, new image {n:.1f} %")
bars("gap", [r, n], ["random folds", "new image"], [F.GREEN, F.RED])

# 3. the cure: the same forest on the six invariant descriptors (dashed outline = random 5-fold)
ri, ni = 100 * L["invariant"]["random forest"]["random"], 100 * L["invariant"]["random forest"]["image"]
print(f"kpi_invariant: new image raw {n:.1f} %, invariant {ni:.1f} % (random 5-fold: raw {r:.1f} %, invariant {ri:.1f} %)")
bars("invariant", [n, ni], ["raw features", "invariant"], [F.RED, F.BLUE], ghost=[r, ri])
