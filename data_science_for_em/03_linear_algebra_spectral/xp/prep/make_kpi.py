"""Render the three small real-data pictures under the numbers of the opening KPI slide.

Usage: python xp/prep/make_kpi.py   (reads xp/data/*.json, writes img/kpi_*.png)
Each picture uses the same data and the same method as the explainer the card points to.
"""
import base64, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XP = Path(__file__).resolve().parents[1]
OUT = XP.parent / "img"
TXT, MUTED = "#e2e8f0", "#94a3b8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "text.color": TXT})


def load(name):
    def dec(v):
        if isinstance(v, dict) and "b64" in v:
            dt = {"f4": "<f4", "u2": "<u2", "u1": "u1", "i2": "<i2"}[v["dtype"]]
            return np.frombuffer(base64.b64decode(v["b64"]), dt).reshape(v["shape"]).astype(float)
        return {k: dec(w) for k, w in v.items()} if isinstance(v, dict) else v
    return dec(json.loads((XP / "data" / f"{name}.json").read_text()))


def fig(w=4.6, h=2.9):
    f = plt.figure(figsize=(w, h), dpi=200)
    f.patch.set_alpha(0)
    return f


def two_maps(a, b, ta, tb, cmap, out, vmin=None, vmax=None):
    f = fig()
    for k, (m, t) in enumerate([(a, ta), (b, tb)]):
        ax = f.add_axes([0.02 + 0.5 * k, 0.02, 0.46, 0.84])
        ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(t, fontsize=15, color=TXT, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#475569")
    f.savefig(out, transparent=True); plt.close(f)
    print(out.name)


# 1. 58 counts per pixel: raw Ce-Lβ window (0.4 counts per pixel) vs weighted PCA (K = 4) of the same window
d = load("cevo3_eds")
C, E, w = d["cube"], d["E"], d["wtd"]
ny, nx, D = C.shape
print("counts per pixel:", C.reshape(-1, D).sum(1).mean().round(1))
sel = (E >= 5.22) & (E < 5.30)                             # Ce-Lβ window of the eds-pca explainer
raw = C[..., sel].sum(-1)
K = 4
Xh = (w["mean"] + w["T"][:, :K] @ w["V"][:K]) * d["G"][:, None] * d["H"][None]
den = Xh[:, sel].sum(-1).reshape(ny, nx)
lo, hi = np.percentile(den, [0.5, 99.5])
two_maps(raw, den, "raw Ce-Lβ counts", "PCA, K = 4", "magma", OUT / "kpi_counts.png", lo, hi)

# 2. drift: measured ZLP shift map vs the PC1 score map of the unaligned ZLP stack
z = load("zlp")
sh = z["shift"]
t1 = z["raw"]["T1"].reshape(sh.shape)
r = np.corrcoef(sh.ravel(), t1.ravel())[0, 1]
print("corr(shift, PC1) =", r.round(3))
nz = lambda m: (m - m.mean()) / m.std()
two_maps(nz(sh), nz(t1), "ZLP energy shift", "PCA component 1", "RdBu_r", OUT / "kpi_drift.png", -2.5, 2.5)

# 3. infinitely many unmixings: mixed pixels (all a_k > 0.2) on the simplex, two feasible triangles
Cb = C.reshape(ny // 2, 2, nx // 2, 2, D).sum((1, 3)).reshape(-1, D)
tot = Cb.sum(1)
Y = Cb / tot[:, None]
m = d["simplex"]["M"]
B = (m[:2] - m[2]).T
a12 = np.linalg.lstsq(B, (Y - m[2]).T, rcond=None)[0].T
A = np.c_[a12, 1 - a12.sum(1)]
keep = (tot > 60) & (A > 0.2).all(1)
print("mixed pixels:", keep.sum())
V = np.array([[0, 1], [-np.sin(np.pi / 3), -0.5], [np.sin(np.pi / 3), -0.5]])
P = A[keep] @ V
f = fig(4.6, 3.1)
ax = f.add_axes([0, 0, 1, 1]); ax.set_aspect("equal"); ax.axis("off")
ax.scatter(P[:, 0], P[:, 1], s=5, c="#cbd5e1", alpha=0.5, lw=0)
c0 = V.mean(0)
for s_, col, lab in [(1.0, "#f43f5e", "answer A"), (0.42, "#fbbf24", "")]:
    T = c0 + s_ * (V - c0)
    ax.add_patch(plt.Polygon(T, fill=False, ec=col, lw=2.5))
    ax.text(T[0, 0] + 0.16, T[0, 1] - 0.06, lab, color=col, fontsize=15, va="top")
for k, n in enumerate(["phase 1", "phase 2", "phase 3"]):
    ax.text(*(V[k] * 1.0 + [0, 0.07 if k == 0 else -0.13]), n, color=MUTED, fontsize=13, ha="center")
ax.text(0.62, 0.05, "answer B", color="#fbbf24", fontsize=15, va="center")
ax.annotate("", xy=(0.36, 0.05), xytext=(0.6, 0.05), arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.8))
ax.set_xlim(-1.3, 1.62); ax.set_ylim(-0.72, 1.22)
f.savefig(OUT / "kpi_unmix.png", transparent=True); plt.close(f)
print("kpi_unmix.png")
