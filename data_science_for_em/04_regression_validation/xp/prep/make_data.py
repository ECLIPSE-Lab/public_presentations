"""Build the spectral data files of the Week 4 explainers (xp/data/*.json) from the raw open data.

Usage (needs hyperspy, exspy):  python xp/prep/make_data.py RAW_DIR
The other two data files have their own scripts: make_disks.py (4D-STEM disks) and make_thick.py
(thickness of ten specimens).

Data sets
- CeVO3 on Al2O3, STEM-Bank-26 session "CeVO3-Al2O3-Ox-03082023", SI 94: EELS (high loss) and EDS.
  Hollyer & Stach, arXiv:2610.36086, CC BY 4.0.
- Fe-L EELS spectrum image of a lunar pyroxene and the Fe valence standards. Cymes, Burgess &
  Stroud, Zenodo 10.5281/zenodo.7439174, CC BY 4.0.
- X-ray line energies and relative weights: exspy element database.
Arrays are stored as {"shape", "dtype", "b64"} (little-endian) and decoded by xp/data.js.
"""
import base64, json, sys
from pathlib import Path

import numpy as np
import hyperspy.api as hs

RAW = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
CEVO3 = RAW / "stembank26/CeVO3_Al2O3_SI94"
CYMES = RAW / "eels_eds_cymes_pyroxene"


def arr(a, dtype="f4"):
    a = np.ascontiguousarray(np.asarray(a).astype({"f4": "<f4", "u2": "<u2", "u1": "u1", "i2": "<i2"}[dtype]))
    return {"shape": list(a.shape), "dtype": dtype, "b64": base64.b64encode(a.tobytes()).decode()}


def save(name, obj):
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, separators=(",", ":")))
    print(f"{p.name}: {p.stat().st_size / 1e6:.2f} MB")


# --------------------------------------------------------------------------------------------
# 1. Pre-edge EELS spectra of every pixel (Acts 2 and 4: gradient descent, polynomial degree,
#    bias and variance over many real pixels). 2 x 2 binned pixels, 1 eV channels, 455-560 eV.
# --------------------------------------------------------------------------------------------
def preedge():
    s = hs.load(CEVO3 / "EELS Spectrum Image (high-loss).dm4")
    ax = s.axes_manager[-1]
    X = s.data.astype(float).reshape(40, 2, 40, 2, 512, 4).sum((1, 3, 5))
    E = ax.offset + ax.scale * (np.arange(512) * 4 + 1.5)
    sel = (E >= 455) & (E <= 560)
    X, E = X[..., sel], E[sel]
    tot = X.sum(-1)
    vac = tot < 0.5 * np.median(tot)
    scale = X.max() / 65000
    print("pre-edge cube", X.shape, "E", E[0], E[-1], "vacuum pixels", vac.sum())
    save("preedge", {"E": arr(E), "cube": arr(np.round(X / scale), "u2"), "scale": scale, "vac": arr(vac, "u1"),
                     "onset": 513.0, "note": "V-L2,3 onset 513 eV, O-K at 532 eV"})


# --------------------------------------------------------------------------------------------
# 2. Fe-L: standards and the per-pixel spectra of the pyroxene rim (Act 5: ridge, collinear refs)
# --------------------------------------------------------------------------------------------
def fe():
    grid = np.arange(704.0, 724.0 + 1e-9, 0.2)                     # L3 and L2, 101 channels

    def prep(e, y, pre=(690, 704)):
        m = (e > pre[0]) & (e < pre[1])
        b, a = np.polyfit(np.log(e[m]), np.log(np.maximum(y[m], 1e-9)), 1)
        return np.interp(grid, e, y - np.exp(a) * e ** b)

    refs = {}
    for key, fn in [("Fe0", "FeNi_Fe0"), ("Fe2", "FeO_Fe2+"), ("Fe3", "Air Glass_Fe3+"), ("Fe23", "Magnetite_Fe2+Fe3+")]:
        s = hs.load(CYMES / f"{fn}EELS Standard.dm4")                 # standards: background already removed
        es, ys = s.axes_manager[-1].axis, s.data.astype(float)
        y = np.interp(grid, es, ys - ys[(es > 700) & (es < 704)].mean())
        refs[key] = y / y.sum()                                     # unit sum: weights = fractions of the Fe signal
    si = hs.load(CYMES / "Fig. 10_EELS-SI_FeL_650-752eV_EELS Spectrum Image.dm4")
    e = si.axes_manager[-1].axis
    X = si.data.astype(float)[:162].reshape(54, 3, 40, 3, -1).sum((1, 3))   # 3 x 3 binned pixels
    # energy alignment of the image to the standards: shift that best matches the mean spectrum to
    # the least-squares mix of the standards (grid search, 0.05 eV steps)
    mean = X.mean((0, 1))
    best = None
    for sh in np.arange(-2.0, 2.0001, 0.05):
        y = prep(e + sh, mean)
        R = np.stack([refs[k] for k in ("Fe0", "Fe2", "Fe3")], 1)
        w = np.linalg.lstsq(R, y, rcond=None)[0]
        err = np.linalg.norm(y - R @ w) / np.linalg.norm(y)
        if best is None or err < best[0]:
            best = (err, sh)
    print("Fe-L image: energy shift to the standards %.2f eV (rel. residual %.3f)" % (best[1], best[0]))
    Y = np.stack([prep(e + best[1], x) for x in X.reshape(-1, X.shape[-1])]).reshape(54, 40, -1)
    sig = Y.sum(-1)
    print("Fe-L per binned pixel: signal median %.0f counts" % np.median(sig))
    R = np.stack([refs[k] for k in ("Fe0", "Fe2", "Fe3")], 1)
    G = R.T @ R
    print("Gram eigenvalues", np.linalg.eigvalsh(G).round(6), "kappa(R) = %.1f" % np.sqrt(np.linalg.cond(G)))
    scale = np.abs(Y).max() / 32000
    save("fel", {"E": arr(grid), "refs": {k: arr(v) for k, v in refs.items()}, "cube": arr(np.round(Y / scale), "i2"), "scale": scale,
                 "shift": float(best[1]), "shape": [54, 40]})


# --------------------------------------------------------------------------------------------
# 3. EDS spectra and an X-ray line library (Act 5: lasso selects the elements)
# --------------------------------------------------------------------------------------------
def eds():
    from exspy.misc.elements import elements_db as db
    s = hs.load(CEVO3 / "EDS Spectrum Image.dm4")
    E = s.axes_manager[-1].axis
    sel = (E >= 0.2) & (E < 10.2)
    C = s.data[:, :, sel].astype(float).reshape(80, 80, -1, 4).sum(-1)
    E = E[sel].reshape(-1, 4).mean(-1)
    total = C.sum((0, 1))
    small = C[52:57, 20:25].sum((0, 1))                             # 5 x 5 pixels on the CeVOx particle
    print("EDS: total counts %.0f, small region %.0f" % (total.sum(), small.sum()))
    fw = lambda E0: np.sqrt(0.130 ** 2 + 0.0025 * (E0 - 5.895)) / 2.3548   # detector resolution (Mn-Ka 130 eV)
    lib, names = [], []
    for el, v in db.as_dictionary().items():
        if not isinstance(v, dict):
            continue
        try:
            Z = v["General_properties"]["Z"]
            lines = v["Atomic_properties"]["Xray_lines"]
        except KeyError:
            continue
        if Z < 5 or Z > 83:
            continue
        prof = np.zeros_like(E)
        for ln, lv in lines.items():
            E0, w = lv["energy (keV)"], lv["weight"]
            if 0.25 < E0 < 10.1:
                prof += w * np.exp(-0.5 * ((E - E0) / max(fw(max(E0, 0.3)), 0.03)) ** 2)
        if prof.max() > 0:
            lib.append(prof / prof.sum()); names.append(el)
    dd = db.as_dictionary()
    order = np.argsort([dd[n]["General_properties"]["Z"] for n in names])
    lib, names = np.array(lib)[order], [names[i] for i in order]
    print("line library:", len(names), "elements")
    save("eds", {"E": arr(E), "total": arr(total), "small": arr(small), "lib": arr(lib), "names": names,
                 "small_box": [52, 57, 20, 25], "truth": ["O", "Al", "V", "Ce"], "expected_extra": ["C", "Cu", "Si"]})


# --------------------------------------------------------------------------------------------
# 4. Spectrum alignment (Act 2: a non-convex loss). The FeO standard and the mean Fe-L spectrum of
#    the pyroxene image on their own (uncorrected) energy axes, 690-745 eV, background removed.
# --------------------------------------------------------------------------------------------
def align():
    grid = np.arange(690.0, 745.0 + 1e-9, 0.1)
    s = hs.load(CYMES / "FeO_Fe2+EELS Standard.dm4")
    es, ys = s.axes_manager[-1].axis, s.data.astype(float)
    ref = np.interp(grid, es, ys - ys[(es > 695) & (es < 704)].mean())
    si = hs.load(CYMES / "Fig. 10_EELS-SI_FeL_650-752eV_EELS Spectrum Image.dm4")
    e, X = si.axes_manager[-1].axis, si.data.astype(float)
    y = X[20:140, 10:110].mean((0, 1))
    m = (e > 680) & (e < 704)
    b, a = np.polyfit(np.log(e[m]), np.log(y[m]), 1)
    y = np.interp(grid, e, y - np.exp(a) * e ** b)
    save("align", {"E": arr(grid), "ref": arr(ref / ref.max()), "y": arr(y / y.max())})


for f in sys.argv[2:] or ["preedge", "fe", "eds", "align"]:
    globals()[f]()
