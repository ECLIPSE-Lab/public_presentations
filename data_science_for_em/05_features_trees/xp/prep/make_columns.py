"""Build xp/data/columns.json and xp/data/frames.json: atom columns of four real HAADF-STEM images of
one oxide interface, with a feature table, labels, and a dose series.

Usage (needs hyperspy, rosettasciio, scipy):
    uv run --no-project --with hyperspy --with rosettasciio --with scipy python xp/prep/make_columns.py RAW_DIR
RAW_DIR/haadf_SFO_LAO/ holds the four files "SFO LAO_JEOL ADF1_*.dm4" from STEM-Bank-26
(Hollyer & Stach, arXiv:2610.36086, CC BY 4.0; `hf buckets ls ghollyer/STEM-Bank-26`).

The specimen is a cross-section: a bright square lattice on the left (the substrate), a darker
crystalline film in the middle, and an amorphous layer on the right. The file names say
"SFO LAO". By Z-contrast, the bright lattice with one visible sublattice is LaAlO3 (La, Z = 57) and
the darker lattice is the Sr-Fe oxide film. The metadata do not state this; it is an inference.

Pipeline for each image (all lengths in nm, so that images with different pixel sizes agree):
  1. stacks of 10 frames: align by cross-correlation (integer shifts), sum
  2. smooth with a Gaussian (sigma 0.04 nm), subtract a wide Gaussian (0.8 nm) as the background
  3. candidates = local maxima (window +-0.12 nm) above 4 % of the 99.5th percentile
  4. refine each candidate with a 2-D Gaussian + constant (least squares, window +-0.11 nm)
  5. descriptors from the fit and from the four nearest candidates (see NAMES)
Labels: by position. The two zone boundaries of each image were set by hand (ZONES, in nm). Candidates
closer than about 0.1 nm to a boundary, and candidates within 0.45 nm of the image border, have no label.
  0 = substrate column, 1 = film column, 2 = no column (a maximum of the noise in the amorphous layer)
"""
import base64, json, sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import hyperspy.api as hs

RAW = Path(sys.argv[1]) / "haadf_SFO_LAO"
OUT = Path(__file__).resolve().parents[1] / "data"
FILES = ["SFO LAO_JEOL ADF1_200kV_20MX_0006.dm4", "SFO LAO_JEOL ADF1_200kV_20MX_0019.dm4",
         "SFO LAO_JEOL ADF1_200kV_25MX_0002.dm4", "SFO LAO_JEOL ADF1_20M_01.dm4"]
SHORT = ["A  (20 MX, 4.7 pm/px)", "B  (20 MX, 9.8 pm/px)", "C  (25 MX, 15.5 pm/px, 10 frames)", "D  (20 MX, 10 pm/px, 10 frames)"]
# zone boundaries in nm (x): substrate < z0 | z1 < film < z2 | z3 < amorphous
ZONES = [(3.7, 3.85, 6.45, 6.65), (5.4, 5.55, 8.2, 8.35), (4.3, 4.45, 6.9, 7.1), (4.6, 4.75, 7.5, 7.7)]
NAMES = ["amp", "bg", "sigma", "resid", "contrast", "I_rel", "d1", "d4", "d_spread", "n_near", "psi4"]
UNITS = ["counts", "counts", "nm", "–", "–", "–", "nm", "nm", "–", "–", "–"]
ABOUT = ["peak height of the Gaussian fit", "constant under the peak", "width of the Gaussian fit",
         "fit residual (RMS) / peak height", "peak height / background", "peak height / median peak height of the 4 nearest",
         "distance to the nearest candidate", "mean distance to the 4 nearest", "std / mean of the 4 distances",
         "candidates within 0.45 nm", "four-fold bond order |mean exp(4iθ)| of the 4 nearest"]
NDISP = 480                                                          # display size of each image (pixels)


def arr(a, dtype="f4"):
    a = np.ascontiguousarray(np.asarray(a).astype({"f4": "<f4", "u1": "u1", "u2": "<u2", "i2": "<i2"}[dtype]))
    return {"shape": list(a.shape), "dtype": dtype, "b64": base64.b64encode(a.tobytes()).decode()}


def shift_of(a, b, m=24):
    """Integer shift (dy, dx) that moves b onto a; searched within +-m pixels (the lattice is periodic)."""
    c = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real)
    cy, cx = c.shape[0] // 2, c.shape[1] // 2
    w = c[cy - m:cy + m + 1, cx - m:cx + m + 1]
    i = np.unravel_index(np.argmax(w), w.shape)
    return int(i[0] - m), int(i[1] - m)


def align(st):
    """Align the frames of a stack to frame 0 (frame-to-frame shifts, accumulated). Returns the cropped frames."""
    sh, prev = [(0, 0)], st[0].astype(float)
    for f in st[1:]:
        s0 = shift_of(ndi.gaussian_filter(prev, 2), ndi.gaussian_filter(f.astype(float), 2))
        sh.append((sh[-1][0] + s0[0], sh[-1][1] + s0[1])); prev = f.astype(float)
    m = max(max(abs(a), abs(b)) for a, b in sh) + 1
    fr = np.stack([np.roll(f.astype(float), s, (0, 1))[m:-m, m:-m] for f, s in zip(st, sh)])
    return fr, sh


def find(img, scale, sig_nm=0.04, dmin_nm=0.12, thr=0.04):
    sm = ndi.gaussian_filter(img, sig_nm / scale)
    d = sm - ndi.gaussian_filter(img, 0.8 / scale)
    w = int(round(dmin_nm / scale))
    pk = (d == ndi.maximum_filter(d, size=2 * w + 1)) & (d > thr * np.percentile(d, 99.5))
    y, x = np.nonzero(pk); b = w + 2
    k = (y > b) & (y < img.shape[0] - b) & (x > b) & (x < img.shape[1] - b)
    return y[k], x[k]


def refine(img, y, x, scale, hw_nm=0.11):
    """2-D Gaussian + constant at every candidate. Returns rows (y, x [px], amp, sigma [nm], bg, rms residual)."""
    hw = int(round(hw_nm / scale)); yy, xx = np.mgrid[-hw:hw + 1, -hw:hw + 1]
    out = []
    for y0, x0 in zip(y, x):
        p = img[y0 - hw:y0 + hw + 1, x0 - hw:x0 + hw + 1].astype(float)
        f = lambda q: (q[0] * np.exp(-((yy - q[1]) ** 2 + (xx - q[2]) ** 2) / (2 * q[3] ** 2)) + q[4] - p).ravel()
        r = least_squares(f, [p.max() - p.min(), 0, 0, hw / 2, p.min()],
                          bounds=([0, -hw, -hw, 0.5, -np.inf], [np.inf, hw, hw, 3 * hw, np.inf]))
        q = r.x
        out.append([y0 + q[1], x0 + q[2], q[0], q[3] * scale, q[4], np.sqrt(np.mean(r.fun ** 2))])
    return np.array(out)


def table(a, sc):
    y, x = find(a, sc); R = refine(a, y, x, sc)
    P = R[:, :2] * sc; T = cKDTree(P)                                  # positions (y, x) in nm
    d, j = T.query(P, k=5); d, j = d[:, 1:], j[:, 1:]
    amp = R[:, 2]; F = np.zeros((len(R), len(NAMES)))
    F[:, 0] = amp; F[:, 1] = R[:, 4]; F[:, 2] = R[:, 3]; F[:, 3] = R[:, 5] / np.maximum(amp, 1e-9)
    F[:, 4] = amp / np.maximum(R[:, 4], 1e-9); F[:, 5] = amp / np.median(amp[j], 1)
    F[:, 6] = d[:, 0]; F[:, 7] = d.mean(1); F[:, 8] = d.std(1) / d.mean(1)
    F[:, 9] = [len(q) - 1 for q in T.query_ball_point(P, 0.45)]
    v = P[j] - P[:, None, :]
    F[:, 10] = np.abs(np.exp(4j * np.arctan2(v[..., 0], v[..., 1])).mean(1))
    return P, F, j


def acsf(P, rs=np.linspace(0.2, 0.8, 7), rc=0.9, w=0.04):
    """Radial symmetry functions G2 (Behler-Parrinello form): sum_j exp(-(r_ij - r_s)^2 / 2w^2) f_c(r_ij)."""
    T = cKDTree(P); A = np.zeros((len(P), len(rs)))
    for a, nb in enumerate(T.query_ball_point(P, rc)):
        nb = [b for b in nb if b != a]
        if nb:
            r = np.hypot(*(P[nb] - P[a]).T); fc = 0.5 * (np.cos(np.pi * r / rc) + 1)
            A[a] = (np.exp(-(r[:, None] - rs[None]) ** 2 / (2 * w * w)) * fc[:, None]).sum(0)
    return A


def precision(fr, sc, ref, hw_nm=0.11):
    """Repeatability of column positions: two disjoint sums of n frames, same columns, position difference / sqrt 2."""
    out = {}
    hw = int(round(hw_nm / sc))
    for n in (1, 2, 5):
        a, b = fr[:n].sum(0), fr[n:2 * n].sum(0)
        res = {}
        for name in ("pixel maximum", "centroid", "Gaussian fit"):
            pos = []
            for im in (a, b):
                sm = ndi.gaussian_filter(im, 0.04 / sc); pp = []
                for y0, x0 in ref:
                    win = sm[y0 - hw:y0 + hw + 1, x0 - hw:x0 + hw + 1]
                    iy, ix = np.unravel_index(np.argmax(win), win.shape)
                    if name == "pixel maximum":
                        pp.append((y0 - hw + iy, x0 - hw + ix))
                    elif name == "centroid":
                        cy, cx = y0 - hw + iy, x0 - hw + ix               # window centred on this image's own maximum
                        raw = im[cy - hw:cy + hw + 1, cx - hw:cx + hw + 1]; wgt = np.clip(raw - raw.min(), 0, None)
                        yy, xx = np.mgrid[-hw:hw + 1, -hw:hw + 1]
                        pp.append((cy + (wgt * yy).sum() / wgt.sum(), cx + (wgt * xx).sum() / wgt.sum()))
                    else:
                        pp.append(refine(im, [y0 - hw + iy], [x0 - hw + ix], sc)[0, :2])
                pos.append(np.array(pp, float))
            d = (pos[0] - pos[1]) * sc * 1000                          # pm
            d = d - np.median(d, 0)                                    # remove the residual rigid shift
            ok = np.abs(d).max(1) < 60                                 # a jump to a neighbour is not "precision"
            res[name] = float(np.sqrt((d[ok] ** 2).mean()) / np.sqrt(2))   # RMS of one coordinate of one measurement
        out[n] = res
    return out


imgs, scales, stacks = [], [], {}
for i, fn in enumerate(FILES):
    s = hs.load(str(RAW / fn)); a = np.asarray(s.data, dtype=np.float32); sc = float(s.axes_manager.signal_axes[0].scale)
    if a.ndim == 3:
        fr, sh = align(a); stacks[i] = fr; a = fr.sum(0)
        print(SHORT[i], "frame shifts (px):", sh)
    imgs.append(a.astype(float)); scales.append(sc)

X, Y, G, PP, AC, NB, disp = [], [], [], [], [], [], []
for i, (a, sc) in enumerate(zip(imgs, scales)):
    P, F, j = table(a, sc)
    x = P[:, 1]; z = ZONES[i]; H, W = a.shape[0] * sc, a.shape[1] * sc
    lab = np.full(len(x), -1); lab[x < z[0]] = 0; lab[(x > z[1]) & (x < z[2])] = 1; lab[x > z[3]] = 2
    lab[(P[:, 0] < 0.45) | (P[:, 0] > H - 0.45) | (P[:, 1] < 0.45) | (P[:, 1] > W - 0.45)] = -1
    print(f"{SHORT[i]}: {a.shape}, {len(x)} candidates; substrate {np.sum(lab == 0)}, film {np.sum(lab == 1)}, "
          f"no column {np.sum(lab == 2)}, no label {np.sum(lab < 0)}; median amp {np.median(F[lab == 0, 0]):.3g}")
    off = sum(len(q) for q in Y)
    X.append(F); Y.append(lab); G.append(np.full(len(x), i)); PP.append(P); AC.append(acsf(P)); NB.append(j + off)
    g = ndi.gaussian_filter(a, max(0.5, 0.5 * a.shape[0] / NDISP - 0.5))
    g = ndi.zoom(g, NDISP / a.shape[0], order=1)[:NDISP, :NDISP]
    lo, hi = np.percentile(g, 0.5), np.percentile(g, 99.8)
    disp.append({"name": SHORT[i], "file": FILES[i], "size_nm": [H, W], "scale_nm": sc, "zones": list(ZONES[i]),
                 "img": arr(np.clip((g - lo) / (hi - lo), 0, 1) * 255, "u1")})

# repeatability of the positions against dose (stack D: 10 frames of 1024 x 1024)
fr, sc = stacks[3], scales[3]
y0, x0 = find(imgs[3], sc)
keep = (x0 * sc < ZONES[3][0] - 0.3) & (x0 * sc > 0.5) & (y0 * sc > 0.5) & (y0 * sc < imgs[3].shape[0] * sc - 0.5)
amp_ref = refine(imgs[3], y0[keep], x0[keep], sc)[:, 2]
strong = amp_ref > 0.5 * np.median(amp_ref)                          # the bright sublattice of the substrate
ref = np.c_[y0[keep][strong], x0[keep][strong]]
prec = precision(fr, sc, ref)
print(f"position repeatability (pm, one coordinate), {len(ref)} substrate columns of image D:")
for n, r in prec.items():
    print(f"  {n} frame(s):", ", ".join(f"{k} {v:.1f}" for k, v in r.items()))
counts = [float(np.median(f[:, : int(ZONES[3][0] / sc)])) for f in fr]
print(f"  median counts per pixel and frame in the substrate: {np.mean(counts):.3g} (detector units)")

out = {"names": NAMES, "units": UNITS, "about": ABOUT, "classes": ["substrate column", "film column", "no column"],
       "images": disp, "X": arr(np.vstack(X)), "y": arr(np.concatenate(Y), "i2"), "g": arr(np.concatenate(G), "u1"),
       "pos": arr(np.vstack(PP)), "acsf": arr(np.vstack(AC)), "acsf_rs": np.linspace(0.2, 0.8, 7).tolist(),
       "nb": arr(np.vstack(NB), "u2"),
       "precision": {str(n): r for n, r in prec.items()}, "precision_n": int(len(ref)),
       "source": "STEM-Bank-26 (Hollyer & Stach 2026), folder haadf_SFO_LAO, CC BY 4.0"}
(OUT / "columns.json").write_text(json.dumps(out))
print("columns.json", (OUT / "columns.json").stat().st_size // 1024, "kB;", len(np.concatenate(Y)), "candidates,",
      int((np.concatenate(Y) >= 0).sum()), "with a label")

# dose series for the column-finder explainer: a crop of stack D, ten aligned frames, binned 2 x 2, 8 bit
fr, sc = stacks[3], scales[3]
y0c, x0c, hc, wc = 200, 360, 512, 576
crop = fr[:, y0c:y0c + hc, x0c:x0c + wc]
crop = crop.reshape(10, hc // 2, 2, wc // 2, 2).sum((2, 4))
hi = np.percentile(crop, 99.95)
fj = {"frames": arr(np.clip(crop / hi, 0, 1) * 255, "u1"), "scale_nm": 2 * sc, "x0_nm": x0c * sc, "y0_nm": y0c * sc,
      "zones": list(ZONES[3]), "name": SHORT[3], "counts_full": float(hi),
      "source": "STEM-Bank-26 (Hollyer & Stach 2026), SFO LAO_JEOL ADF1_20M_01, CC BY 4.0"}
(OUT / "frames.json").write_text(json.dumps(fj))
print("frames.json", (OUT / "frames.json").stat().st_size // 1024, "kB; crop", crop.shape, "from x =", round(x0c * sc, 2), "nm,",
      "pixel", round(2 * sc * 1000, 1), "pm")
