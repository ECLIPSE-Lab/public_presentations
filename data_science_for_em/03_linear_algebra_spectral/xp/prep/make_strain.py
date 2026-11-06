"""Build xp/data/strain4d.json: a real 4D-STEM strain map, computed with the methods of this lecture.

Usage (needs hyperspy, scipy):
    python xp/prep/make_strain.py RAW_DIR

RAW_DIR/strain_mills/Al_irradiated.dm4 is from Zenodo 10.5281/zenodo.7041997 (CC BY 4.0):
S. Mills, S. Zeltmann, P. Ercius, A. Kohnert, B. Uberuaga, A. Minor, "Nanoscale mapping of point
defect concentrations with 4D-STEM", Acta Mater. 246, 118721 (2023). Ion-irradiated Al, [001] zone
axis, 40 x 40 scan positions, 512 x 512 convergent-beam patterns, "bullseye" probe aperture.

Pipeline (each step is a lecture idea):
1. Disk positions: correlate each pattern with a template disk (taken from the max projection),
   search near the positions predicted by a reference lattice, sub-pixel peak by a parabola.
2. Lattice fit by linear least squares (slide "Least squares"): every disk p_i with index (h_i, k_i)
   gives p_i = c + h_i g1 + k_i g2, i.e. two linear equations in the six unknowns (c, G = [g1 g2]).
   Weighted by the correlation peak height; outliers (residual > 2 px) removed, then refitted.
3. Deformation: A = G G_ref^-1 (reciprocal space), F = A^-T (real space). The reference G_ref is the
   median lattice of the scan, so the strain is relative to the mean crystal.
4. SVD / polar decomposition of F (slide "A = U Sigma V^T"): F = U S V^T, rotation R = U V^T,
   principal strains = S - 1. The widget does this step live in JS.

Checks printed below: (a) the calibrated 3 % elliptical distortion of the patterns (calibration
file) changes the relative strains by < 2e-4, because it is common to G and G_ref; (b) the polar
rotation agrees with the small-angle rotation (F21 - F12)/2.
"""
import base64, json, sys
from pathlib import Path

import numpy as np
import hyperspy.api as hs
from numpy.fft import rfft2, irfft2
from scipy.ndimage import gaussian_filter

RAW = Path(sys.argv[1]) / "strain_mills"
OUT = Path(__file__).resolve().parents[1] / "data"
ELLIPSE = (1.0149178660711329, 0.9850821339288671, -0.5868451325189119)   # a, b, theta from Al_irradiated_calibration.h5


def arr(a, dtype="f4"):
    a = np.ascontiguousarray(np.asarray(a).astype({"f4": "<f4", "u1": "u1"}[dtype]))
    return {"shape": list(a.shape), "dtype": dtype, "b64": base64.b64encode(a.tobytes()).decode()}


D = hs.load(RAW / "Al_irradiated.dm4").data.astype(np.float32)            # (1600, 512, 512)
NS, NY, NX = 40, 512, 512
mx = D.max(0)
hp = lambda im: im - gaussian_filter(im, 12)
yy, xx = np.mgrid[:NY, :NX]
cy, cx, R = 264, 265, 38                                                    # central disk in the max projection
win = (np.hypot(yy - cy, xx - cx) < R).astype(np.float32)
t = hp(mx) * win
t[win > 0] -= t[win > 0].mean()
t *= win
FT = np.conj(rfft2(np.roll(t, (-cy, -cx), (0, 1))))


def corr(img):                                                              # hybrid correlation: sharp peaks
    F = rfft2(hp(np.asarray(img, np.float32))) * FT
    return irfft2(F / (np.abs(F) ** 0.5 + 1e-6), s=(NY, NX))


def subpx(c, y, x):
    a, b, d = c[y - 1, x], c[y, x], c[y + 1, x]
    dy = 0.5 * (a - d) / (a - 2 * b + d)
    a, d = c[y, x - 1], c[y, x + 1]
    return y + dy, x + 0.5 * (a - d) / (a - 2 * b + d)


def mk_hk(G):
    return np.array([(h, k) for h in range(-4, 5) for k in range(-4, 5) if 0 < np.hypot(*(G @ [h, k])) < 215])


def fit(img, c_guess, G0, HK, w=14):
    c = corr(img)
    pts, hk, wts = [], [], []
    for h, k in [(0, 0)] + [tuple(v) for v in HK]:
        p = np.asarray(c_guess) + G0 @ [h, k]
        y0, x0 = int(round(p[0])), int(round(p[1]))
        if not (w + 2 <= y0 < NY - w - 2 and w + 2 <= x0 < NX - w - 2):
            continue
        blk = c[y0 - w:y0 + w + 1, x0 - w:x0 + w + 1]
        j = np.unravel_index(np.argmax(blk), blk.shape)
        if 1 <= j[0] < 2 * w and 1 <= j[1] < 2 * w:
            pts.append(subpx(c, y0 - w + j[0], x0 - w + j[1])); hk.append((h, k)); wts.append(blk[j])
    pts, hk, wts = np.array(pts), np.array(hk), np.array(wts)
    keep = wts > 0.4 * np.median(wts)

    def lsq(sel):                                                           # p = c + G [h, k]: 6 unknowns
        n = sel.sum()
        X = np.zeros((2 * n, 6))
        X[0::2, 0] = 1; X[1::2, 1] = 1; X[0::2, 2:4] = hk[sel]; X[1::2, 4:6] = hk[sel]
        sw = np.repeat(np.sqrt(wts[sel]), 2)
        sol, *_ = np.linalg.lstsq(X * sw[:, None], pts[sel].ravel() * sw, rcond=None)
        return sol, pts[sel].ravel() - X @ sol

    sol, res = lsq(keep)
    rr = np.hypot(res[0::2], res[1::2])
    if (rr > 2).any():
        keep[np.flatnonzero(keep)[rr > 2]] = False
        sol, res = lsq(keep)
    return sol, int(keep.sum()), float(np.sqrt((res ** 2).mean()))


# reference lattice: square Al {200} net; start from the autocorrelation peaks, refine on the max image
c0, G0 = np.array([cy, cx], float), np.c_[[71.0, 46.0], [-49.0, 72.0]]     # columns g1, g2 as (y, x) px
for _ in range(2):
    sol, n, e = fit(mx, c0, G0, mk_hk(G0))
    c0, G0 = sol[:2], sol[2:].reshape(2, 2)
HK = mk_hk(G0)
print("reference lattice (y, x) px:", G0.round(2).tolist(), "spots", n, "rms", round(e, 3))

SOL, NSP, RMS = np.zeros((1600, 6)), np.zeros(1600, int), np.zeros(1600)
for i in range(1600):
    SOL[i], NSP[i], RMS[i] = fit(D[i], c0, G0, HK)
ok = (NSP >= 5) & (RMS < 1.0)
print("frames fitted:", ok.sum(), "/ 1600; median spots", np.median(NSP), "median rms px", np.median(RMS).round(3))

# G in pattern pixels, columns g1, g2 as (x, y) with y downwards (drawing); centre c as (x, y)
Gpix = np.stack([s_[2:].reshape(2, 2)[::-1] for s_ in SOL])
Cpix = SOL[:, 1::-1].copy()
Gref = np.median(Gpix[ok], 0)
Gpix[~ok] = Gref


def strain(G, Gref, E=None):
    if E is not None:
        G = np.einsum("ij,njk->nik", np.linalg.inv(E), G); Gref = np.linalg.inv(E) @ Gref
    A = G @ np.linalg.inv(Gref)
    F = np.transpose(np.linalg.inv(A), (0, 2, 1))
    e = 0.5 * (F + np.transpose(F, (0, 2, 1))) - np.eye(2)
    return F, e, 0.5 * (F[:, 1, 0] - F[:, 0, 1])


F, e, w = strain(Gpix, Gref)
a, b, th = ELLIPSE
Rt = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
_, e2, _ = strain(Gpix, Gref, Rt @ np.diag([a, b]) @ Rt.T)
U, S, Vt = np.linalg.svd(F)
Rm = U @ Vt
print("strain exx / eyy / exy std (1e-3):", [round(1e3 * v[ok].std(), 2) for v in (e[:, 0, 0], e[:, 1, 1], e[:, 0, 1])])
print("max change by ellipse correction (1e-3):", round(1e3 * np.abs(e - e2)[ok].max(), 3))
print("polar vs small-angle rotation, max diff (rad):", np.abs(np.arctan2(Rm[:, 1, 0], Rm[:, 0, 0]) - w)[ok].max())
okm = ok.reshape(40, 40)
for nm, v in [("exx", e[:, 0, 0]), ("rotation", w)]:
    m = v.reshape(40, 40); d = np.diff(m, axis=1)[okm[:, 1:] & okm[:, :-1]]
    print(f"noise estimate {nm} (std of neighbour differences / sqrt 2, 1e-3):", round(1e3 * d.std() / np.sqrt(2), 2))

# bright-field image (central disk) and patterns on every 2nd scan position (crop 384, bin 4 -> 96 x 96, log scale)
bf = np.stack([f[int(c0[0]) - 30:int(c0[0]) + 30, int(c0[1]) - 30:int(c0[1]) + 30].sum() for f in D]).reshape(40, 40)
y0, x0 = int(round(c0[0])) - 192, int(round(c0[1])) - 192
sel = [(iy, ix) for iy in range(0, 40, 2) for ix in range(0, 40, 2)]
pats = []
for iy, ix in sel:
    p = D[iy * 40 + ix, y0:y0 + 384, x0:x0 + 384].reshape(96, 4, 96, 4).sum((1, 3))
    lp = np.log1p(p)
    lo, hi = np.percentile(lp, [1, 99.7])
    pats.append(np.clip(255 * (lp - lo) / (hi - lo), 0, 255).round())
obj = {"bf": arr(bf), "G": arr(Gpix), "c": arr(Cpix), "Gref": arr(Gref), "ok": arr(ok, "u1"), "n": arr(NSP, "u1"), "rms": arr(RMS),
       "pat": arr(np.array(pats), "u1"), "pat_origin": [x0, y0], "pat_bin": 4, "scan": [40, 40], "q_px_per_nm": 1 / 0.05823560030271533,
       "hk": HK.tolist()}
p = OUT / "strain4d.json"
p.write_text(json.dumps(obj, separators=(",", ":")))
print(f"{p.name}: {p.stat().st_size / 1e6:.2f} MB")
