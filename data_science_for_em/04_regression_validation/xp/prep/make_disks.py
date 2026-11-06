"""Build xp/data/disks.json: Bragg-disk positions of a real 4D-STEM scan, for the loss-function slides.

Usage (needs hyperspy, scipy):  python xp/prep/make_disks.py RAW_DIR
RAW_DIR/strain_mills/Al_irradiated.dm4: Zenodo 10.5281/zenodo.7041997 (CC BY 4.0), Mills et al.,
Acta Mater. 246, 118721 (2023). Ion-irradiated Al, [001], 40 x 40 positions, 512 x 512 patterns.

Same disk finder as Week 3 (xp/prep/make_strain.py there), but nothing is thrown away:
- every candidate disk (position, index h k, correlation weight), so that the slides can fit the
  lattice with MSE, MAE or Huber and show what one wrong disk does;
- the residuals of the plain weighted least-squares fit of every frame (no outlier removal):
  their histogram shows the real noise model (Gaussian core, heavy tails).
"""
import base64, json, sys
from pathlib import Path

import numpy as np
import hyperspy.api as hs
from numpy.fft import rfft2, irfft2
from scipy.ndimage import gaussian_filter

RAW = Path(sys.argv[1]) / "strain_mills"
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05           # disk detection threshold (fraction of the brightest disk)
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)


def arr(a, dtype="f4"):
    a = np.ascontiguousarray(np.asarray(a).astype({"f4": "<f4", "u1": "u1", "i2": "<i2"}[dtype]))
    return {"shape": list(a.shape), "dtype": dtype, "b64": base64.b64encode(a.tobytes()).decode()}


D = hs.load(RAW / "Al_irradiated.dm4").data.astype(np.float32)
mx = D.max(0)
hp = lambda im: im - gaussian_filter(im, 12)
yy, xx = np.mgrid[:512, :512]
cy, cx, R = 264, 265, 38
win = (np.hypot(yy - cy, xx - cx) < R).astype(np.float32)
t = hp(mx) * win
t[win > 0] -= t[win > 0].mean()
t *= win
FT = np.conj(rfft2(np.roll(t, (-cy, -cx), (0, 1))))
corr = lambda img: (lambda F: irfft2(F / (np.abs(F) ** 0.5 + 1e-6), s=(512, 512)))(rfft2(hp(np.asarray(img, np.float32))) * FT)


def subpx(c, y, x):
    a, b, d = c[y - 1, x], c[y, x], c[y + 1, x]
    dy = 0.5 * (a - d) / (a - 2 * b + d)
    a, d = c[y, x - 1], c[y, x + 1]
    return y + dy, x + 0.5 * (a - d) / (a - 2 * b + d)


def candidates(img, c0, G0, HK, w=14):
    c = corr(img)
    pts, hk, wts = [], [], []
    for h, k in [(0, 0)] + [tuple(v) for v in HK]:
        p = c0 + G0 @ [h, k]
        y0, x0 = int(round(p[0])), int(round(p[1]))
        if not (w + 2 <= y0 < 510 - w and w + 2 <= x0 < 510 - w):
            continue
        blk = c[y0 - w:y0 + w + 1, x0 - w:x0 + w + 1]
        j = np.unravel_index(np.argmax(blk), blk.shape)
        if 1 <= j[0] < 2 * w and 1 <= j[1] < 2 * w:
            pts.append(subpx(c, y0 - w + j[0], x0 - w + j[1])); hk.append((h, k)); wts.append(blk[j])
    pts, hk, wts = np.array(pts), np.array(hk), np.array(wts)
    keep = wts > THR * wts.max()                                     # detection threshold, relative to the brightest disk
    return pts[keep][:, ::-1], hk[keep], wts[keep]                   # positions as (x, y)


def design(hk):                                                     # p = c + G [h, k]: rows (x, y) per disk
    n = len(hk)
    X = np.zeros((2 * n, 6)); X[0::2, 0] = 1; X[1::2, 1] = 1; X[0::2, 2:4] = hk; X[1::2, 4:6] = hk
    return X


def lsq(pts, hk, wts):                                               # unweighted least squares (MSE)
    X = design(hk)
    sol, *_ = np.linalg.lstsq(X, pts.ravel(), rcond=None)
    return sol, (pts.ravel() - X @ sol).reshape(-1, 2)


def huber(pts, hk, delta=1.0, it=50):                                # Huber loss by iteratively reweighted least squares
    X, y = design(hk), pts.ravel()
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(it):
        r = np.repeat(np.hypot(*(y - X @ sol).reshape(-1, 2).T), 2)
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-9))
        sol, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
    return sol, (y - X @ sol).reshape(-1, 2)


c0, G0 = np.array([cy, cx], float), np.c_[[71.0, 46.0], [-49.0, 72.0]]
mk = lambda G: np.array([(h, k) for h in range(-4, 5) for k in range(-4, 5) if 0 < np.hypot(*(G @ [h, k])) < 215])
for _ in range(2):                                                    # reference lattice from the max image
    p, hk, w = candidates(mx, c0, G0, mk(G0))
    sol, _ = lsq(p, hk, w)
    c0, G0 = sol[1::-1].copy(), np.array([[sol[4], sol[5]], [sol[2], sol[3]]])
HK = mk(G0)

frames, res, resH, worst = [], [], [], []
for i in range(1600):
    p, hk, w = candidates(D[i], c0, G0, HK)
    if len(p) < 5:
        frames.append(None); worst.append(0); continue
    sol, r = lsq(p, hk, w)
    frames.append((p, hk, w)); res.append(r); worst.append(np.hypot(*r.T).max()); resH.append(huber(p, hk)[1])
R, RH = np.concatenate(res), np.concatenate(resH)
mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
print("Huber residuals: MAD-sigma %.3f px, frac |r| > 3 px %.4f, frac > 5 sigma %.4f" % (mad(RH), (np.abs(RH) > 3).mean(), (np.abs(RH) > 5 * mad(RH)).mean()))
worst = np.array(worst)
print("frames", sum(f is not None for f in frames), "residual components", R.size,
      "std %.3f px, MAD-sigma %.3f px, frac |r| > 3 px: %.4f" % (R.std(), 1.4826 * np.median(np.abs(R - np.median(R))), (np.abs(R) > 3).mean()))
# example frames: the strongest real outliers, and one clean frame for comparison
order = np.argsort(-worst)
ex = [int(i) for i in order[:80] if frames[i] is not None and len(frames[i][0]) >= 7][:4]
clean = int(np.argmin(np.where([f is not None and len(f[0]) >= 8 for f in frames], worst, 99)))
ex.append(clean)
print("example frames", ex, "worst residual px", worst[ex].round(2))
y0, x0 = int(round(c0[0])) - 192, int(round(c0[1])) - 192
pats, exd = [], []
for i in ex:
    pp = D[i, y0:y0 + 384, x0:x0 + 384].reshape(96, 4, 96, 4).sum((1, 3))
    lp = np.log1p(pp); lo, hi = np.percentile(lp, [1, 99.7])
    pats.append(np.clip(255 * (lp - lo) / (hi - lo), 0, 255).round())
    p, hk, w = frames[i]
    exd.append({"frame": i, "p": p.round(3).tolist(), "hk": hk.tolist(), "w": (w / w.max()).round(4).tolist()})
# reference lattice in (x, y): columns g1, g2
Gref = G0[::-1].copy()
obj = {"resid": arr(R), "residH": arr(RH), "thr": THR, "examples": exd, "pat": arr(np.array(pats), "u1"), "pat_origin": [x0, y0], "pat_bin": 4,
       "Gref": Gref.round(4).tolist(), "n_frames": int(sum(f is not None for f in frames))}
(OUT / "disks.json").write_text(json.dumps(obj, separators=(",", ":")))
print("disks.json %.2f MB" % ((OUT / "disks.json").stat().st_size / 1e6))
