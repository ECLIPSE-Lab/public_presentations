"""Render the three small real-data pictures under the numbers of the opening KPI slide.

Usage: python xp/prep/make_kpi.py   (reads xp/data/*.json, writes img/kpi_*.png)
Each picture uses the same data and method as the explainer the card points to.
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
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "text.color": TXT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
SPEC_COL = ["#60a5fa", "#fb923c", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#22d3ee", "#f87171", "#a3e635", "#e2e8f0"]


def load(name):
    def dec(v):
        if isinstance(v, dict) and "b64" in v:
            dt = {"f4": "<f4", "u2": "<u2", "u1": "u1", "i2": "<i2"}[v["dtype"]]
            return np.frombuffer(base64.b64decode(v["b64"]), dt).reshape(v["shape"]).astype(float)
        if isinstance(v, dict):
            return {k: dec(w) for k, w in v.items()}
        if isinstance(v, list):
            return [dec(w) for w in v]
        return v
    return dec(json.loads((XP / "data" / f"{name}.json").read_text()))


def fig(w=4.6, h=3.0):
    f = plt.figure(figsize=(w, h), dpi=200)
    f.patch.set_alpha(0)
    return f


def lattice_fit(p, hk, huber=False, delta=1.0):
    n = len(p)
    X = np.zeros((2 * n, 6)); X[0::2, 0] = 1; X[1::2, 1] = 1; X[0::2, 2:4] = hk; X[1::2, 4:6] = hk
    y = p.ravel(); w = np.ones(2 * n)
    for _ in range(60 if huber else 1):
        sol, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
        r = np.repeat(np.hypot(*(y - X @ sol).reshape(-1, 2).T), 2)
        w = np.where(r <= delta, 1, delta / np.maximum(r, 1e-9))
    return sol


# 1. one frame: detected disks; red = misdetections (residual > 2 px from the robust lattice)
d = load("disks")
e = d["examples"][0]
p, hk = np.array(e["p"]), np.array(e["hk"])
ox, oy = d["pat_origin"]; b = d["pat_bin"]
sol = lattice_fit(p, hk, True)
res = np.hypot(p[:, 0] - sol[0] - sol[2] * hk[:, 0] - sol[3] * hk[:, 1], p[:, 1] - sol[1] - sol[4] * hk[:, 0] - sol[5] * hk[:, 1])
print("frame", e["frame"], "disks", len(p), "misdetected", (res > 2).sum())
f = fig(3.2, 3.2); ax = f.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.imshow(d["pat"][0], cmap="gray", extent=[ox, ox + 96 * b, oy + 96 * b, oy])
ok = res <= 2
ax.scatter(p[ok, 0], p[ok, 1], s=45, c="#fbbf24", edgecolors="white", linewidths=1, zorder=5)
ax.scatter(p[~ok, 0], p[~ok, 1], s=90, c="#f43f5e", edgecolors="white", linewidths=1.2, zorder=6)
ax.set_aspect("equal")
f.savefig(OUT / "kpi_disks.png", transparent=True); plt.close(f); print("kpi_disks.png")

# 2. the background bowl in raw log E (a thin valley) and standardised (round)
pe = load("preedge")
E = pe["E"]; C = pe["cube"] * pe["scale"]
sp = C[26:32, 33:38].reshape(-1, C.shape[-1]).mean(0)
sel = (E >= 470) & (E <= 508); x, y = np.log(E[sel]), np.log(sp[sel])
f = fig()
for k_, (nm, xx) in enumerate([("raw log E", x), ("standardised", (x - x.mean()) / x.std())]):
    ax = f.add_axes([0.03 + 0.5 * k_, 0.03, 0.44, 0.78])
    A = np.c_[np.ones_like(xx), xx]; w = np.linalg.lstsq(A, y, rcond=None)[0]
    ev = np.linalg.eigvalsh(A.T @ A / len(y)); kappa = ev.max() / ev.min()
    d1 = 4 if k_ == 0 else 0.25; d0 = 4 * x.mean() if k_ == 0 else 0.25
    W1, W0 = np.meshgrid(np.linspace(w[1] - d1, w[1] + d1, 160), np.linspace(w[0] - d0, w[0] + d0, 160))
    L = ((y[None, None] - W0[..., None] - W1[..., None] * xx[None, None]) ** 2).mean(-1)
    ax.contourf(W1, W0, np.log(L - L.min() + 1e-6), 14, cmap="magma")
    ax.plot(w[1], w[0], "wx", ms=9, mew=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{nm}\nκ = {kappa:.0e}" if kappa > 100 else f"{nm}\nκ = {kappa:.0f}", fontsize=14, color=TXT)
f.savefig(OUT / "kpi_bowl.png", transparent=True); plt.close(f); print("kpi_bowl.png")

# 3. ADF vs t/lambda for ten specimens: ten different lines
t = load("thick")
f = fig(); ax = f.add_axes([0.13, 0.18, 0.84, 0.78])
for g in range(10):
    m = t["g"] == g
    ax.scatter(t["adf"][m] / 1e6, t["t"][m], s=2.5, color=SPEC_COL[g], alpha=0.6, lw=0)
ax.set_xlabel("ADF intensity (a.u.)", fontsize=13); ax.set_ylabel("t/λ", fontsize=13)
for s_ in ax.spines.values(): s_.set_color("#475569")
ax.tick_params(labelsize=11)
f.savefig(OUT / "kpi_specimens.png", transparent=True); plt.close(f); print("kpi_specimens.png")
