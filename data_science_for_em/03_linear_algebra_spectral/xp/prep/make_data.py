"""Build the real-data files of the Week 3 explainers (xp/data/*.json) from the raw open data sets.

Usage (needs hyperspy, exspy, rosettasciio, scikit-learn):
    python xp/prep/make_data.py RAW_DIR

RAW_DIR must contain the downloaded files at the paths given in SOURCES below. All data sets are
experimental and openly licensed; the slides cite them (see ref.bib). Arrays are stored as
{"shape", "dtype", "b64"} (little-endian) and decoded by xp/data.js.

Data sets
- CeVO3 on Al2O3, STEM-Bank-26 session "CeVO3-Al2O3-Ox-03082023", SI 94: simultaneous EELS
  (high loss) + EDS + ADF. Hollyer & Stach, arXiv:2610.36086, CC BY 4.0.
- Pyroxene from Apollo 17 soil: Fe-L EELS SI, ZLP SI and Fe valence standards. Cymes, Burgess &
  Stroud, Zenodo 10.5281/zenodo.7439174, CC BY 4.0.
- SrFeO_x / LaAlO3 interface, HAADF (JEOL, 200 kV), STEM-Bank-26, CC BY 4.0.
- Twinned GaAs nanowire, SPED, pyxem demo data, Zenodo 10.5281/zenodo.15490547, CC BY 4.0
  (Martineau et al., Adv. Struct. Chem. Imaging 5, 3 (2019)).
"""
import base64, json, sys
from pathlib import Path

import numpy as np
import hyperspy.api as hs
from scipy.ndimage import shift as ndshift
from sklearn.decomposition import NMF

RAW = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
SOURCES = {
    "cevo3": RAW / "stembank26/CeVO3_Al2O3_SI94",
    "cymes": RAW / "eels_eds_cymes_pyroxene",
    "haadf": RAW / "stembank26/haadf_SFO_LAO/SFO LAO_JEOL ADF1_200kV_20MX_0019.dm4",
    "sped": RAW / "sped_pyxem/twinned_nanowire.hdf5",
}


def arr(a, dtype="f4"):
    a = np.ascontiguousarray(np.asarray(a).astype({"f4": "<f4", "u2": "<u2", "u1": "u1", "i2": "<i2"}[dtype]))
    return {"shape": list(a.shape), "dtype": dtype, "b64": base64.b64encode(a.tobytes()).decode()}


def save(name, obj):
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, separators=(",", ":")))
    print(f"{p.name}: {p.stat().st_size / 1e6:.2f} MB")


def u8(img, lo=None, hi=None):
    lo = np.percentile(img, 0.5) if lo is None else lo
    hi = np.percentile(img, 99.5) if hi is None else hi
    return np.clip(255 * (img - lo) / (hi - lo), 0, 255).round()


# --------------------------------------------------------------------------------------------
# 1. Fe-L valence standards and a pyroxene spectrum (Act 1: dot product and projection)
# --------------------------------------------------------------------------------------------
def fe_refs():
    cy = SOURCES["cymes"]
    grid = np.arange(700.0, 730.0 + 1e-9, 0.1)

    def prep(e, y):
        y = np.interp(grid, e, y)
        pre = (grid >= 700) & (grid < 704)
        y = y - y[pre].mean()                                   # flat pre-edge after background removal
        return y / np.trapezoid(y[(grid >= 704)], grid[grid >= 704])  # unit area: compare shapes, not thickness

    out = {"E": arr(grid), "refs": {}}
    for key, fn in [("Fe0", "FeNi_Fe0"), ("Fe2", "FeO_Fe2+"), ("Fe23", "Magnetite_Fe2+Fe3+"), ("Fe3", "Air Glass_Fe3+")]:
        s = hs.load(cy / f"{fn}EELS Standard.dm4")
        out["refs"][key] = arr(prep(s.axes_manager[-1].axis, s.data.astype(float)))
    si = hs.load(cy / "Fig. 10_EELS-SI_FeL_650-752eV_EELS Spectrum Image.dm4")
    e = si.axes_manager[-1].axis
    X = si.data.astype(float)
    # background: power law fitted in 680-704 eV for the region sum
    pre = (e > 680) & (e < 704)
    spec = X[20:140, 10:110].mean((0, 1))
    b, a = np.polyfit(np.log(e[pre]), np.log(spec[pre]), 1)
    out["pyroxene"] = arr(prep(e, spec - np.exp(a) * e ** b))
    # least-squares check: magnetite as a mixture of the Fe2+ and Fe3+ standards
    R = np.stack([np.frombuffer(base64.b64decode(out["refs"][k]["b64"]), "<f4") for k in ("Fe2", "Fe3")], 1)
    y = np.frombuffer(base64.b64decode(out["refs"]["Fe23"]["b64"]), "<f4")
    w = np.linalg.lstsq(R, y, rcond=None)[0]
    print("magnetite = %.2f Fe2+ + %.2f Fe3+  -> Fe3+/sum = %.2f (stoichiometry 0.67)" % (w[0], w[1], w[1] / w.sum()))
    save("fe_refs", out)


# --------------------------------------------------------------------------------------------
# 2. CeVO3 / Al2O3 EELS cube (Act 2: datacube, least squares = background fit)
# --------------------------------------------------------------------------------------------
def cevo3_eels():
    d = SOURCES["cevo3"]
    s = hs.load(d / "EELS Spectrum Image (high-loss).dm4")
    ax = s.axes_manager
    X = s.data.astype(float)                                     # (80, 80, 2048), 463-975 eV, 0.25 eV
    X = X.reshape(40, 2, 40, 2, 512, 4).sum((1, 3, 5))          # 2x2 pixels, 1 eV channels
    E = ax[-1].offset + ax[-1].scale * (np.arange(512) * 4 + 1.5)
    scale = X.max() / 65000
    adf = hs.load(d / "ADF Image.dm4").data.astype(float)
    print("EELS cube", X.shape, "max", X.max(), "mean per pixel", X.sum(-1).mean())
    save("cevo3_eels", {"E": arr(E), "cube": arr(np.round(X / scale), "u2"), "scale": scale,
                        "px_nm": 2 * ax[0].scale, "adf": arr(u8(adf), "u1"), "adf_px_nm": ax[0].scale})


# --------------------------------------------------------------------------------------------
# 3. CeVO3 / Al2O3 EDS cube (Acts 4-6: PCA, Poisson weighting, NMF, simplex, residual)
# --------------------------------------------------------------------------------------------
def cevo3_eds():
    d = SOURCES["cevo3"]
    s = hs.load(d / "EDS Spectrum Image.dm4")
    E = s.axes_manager[-1].axis
    sel = (E >= 0.2) & (E < 10.2)
    C = s.data[:, :, sel].astype(float).reshape(80, 80, -1, 4).sum(-1)   # 40 eV channels
    E = E[sel].reshape(-1, 4).mean(-1)
    X = C.reshape(-1, C.shape[-1])
    N, D = X.shape
    print("EDS cube", C.shape, "max", C.max(), "counts per pixel", X.sum(1).mean())
    K = 12

    def pca(Y):
        mu = Y.mean(0)
        U, sv, Vt = np.linalg.svd(Y - mu, full_matrices=False)
        T = U[:, :K] * sv[:K]
        res0 = ((Y - mu) ** 2).sum(1)                            # ||y - mu||² per pixel
        return {"mean": arr(mu), "sv": arr(sv), "V": arr(Vt[:K]), "T": arr(T), "norm2": arr(res0)}

    # Poisson weighting (Keenan & Kotula 2004): divide by sqrt(pixel total) and sqrt(mean spectrum)
    g = X.sum(1, keepdims=True)
    G = np.sqrt(g / g.mean())
    Hw = np.sqrt(X.mean(0, keepdims=True)) + 1e-6
    raw, wtd = pca(X), pca(X / G / Hw)

    # simplex picture on 2x2-binned pixels: weighted PCA plane + NMF(K=3, KL) vertices
    C2 = C.reshape(40, 2, 40, 2, -1).sum((1, 3))
    X2 = C2.reshape(-1, D)
    nmf = NMF(3, init="nndsvda", beta_loss="kullback-leibler", solver="mu", max_iter=2000, tol=1e-6, random_state=0)
    A = nmf.fit_transform(X2)
    M = nmf.components_
    tot = M.sum(1, keepdims=True)
    A, M = A * tot.T, M / tot                                    # spectra of unit area, abundances in counts
    frac = A / A.sum(1, keepdims=True)                           # sum-to-one abundances
    # the data on the sum-to-one simplex: normalised spectra y = x / sum(x)
    Y = X2 / X2.sum(1, keepdims=True)
    mu2 = Y.mean(0)
    U2, s2, Vt2 = np.linalg.svd(Y - mu2, full_matrices=False)
    P = (Y - mu2) @ Vt2[:2].T
    Vert = (M - mu2) @ Vt2[:2].T
    print("NMF vertices in PC plane", Vert.round(4))
    simplex = {"mean": arr(mu2), "V": arr(Vt2[:2]), "P": arr(P), "vert": arr(Vert), "M": arr(M), "frac": arr(frac)}
    # planted truth for the rare-phase test: three 2 x 2 "Pt particles" (12 pixels) with on average
    # PT_COUNTS Poisson counts of a Pt spectrum (Pt M at 2.05 keV, Pt L at 9.44 keV) added to the real cube
    PT_COUNTS = 10
    rng = np.random.default_rng(3)
    fw = lambda E0: np.sqrt(130 ** 2 + 2.5 * (E0 * 1000 - 5895)) / 1000 / 2.3548
    pt = sum(w * np.exp(-0.5 * ((E - E0) / fw(E0)) ** 2) for E0, w in [(2.048, 1.0), (9.442, 0.7), (8.268, 0.08)])
    pt /= pt.sum()
    mask = np.zeros((80, 80), bool)
    for y, x in [(30, 25), (55, 45), (40, 62)]:
        mask[y:y + 2, x:x + 2] = True
    add = rng.poisson(PT_COUNTS * pt, (mask.sum(), D))
    Xp = X.copy()
    Xp[mask.ravel()] += add
    gp = Xp.sum(1, keepdims=True)
    Gp = np.sqrt(gp / gp.mean())
    Hp = np.sqrt(Xp.mean(0, keepdims=True)) + 1e-6
    planted = pca(Xp / Gp / Hp)
    planted.update({"idx": arr(np.flatnonzero(mask.ravel()), "u2"), "add": arr(add, "u1"), "G": arr(Gp[:, 0]), "H": arr(Hp[0]),
                    "counts": PT_COUNTS})
    print("planted Pt: sv² head", (np.frombuffer(base64.b64decode(planted["sv"]["b64"]), "<f4")[:7] ** 2).round(0))
    save("cevo3_eds", {"E": arr(E), "cube": arr(C, "u1"), "raw": raw, "wtd": wtd,
                       "G": arr(G[:, 0]), "H": arr(Hw[0]), "simplex": simplex, "planted": planted, "px_nm": 0.6272})


# --------------------------------------------------------------------------------------------
# 4. ZLP drift (Act 4: misalignment makes a derivative-shaped component)
# --------------------------------------------------------------------------------------------
def zlp():
    s = hs.load(SOURCES["cymes"] / "Fig. 10_EELS-SI_ZLP_-10-92eV_EELS Spectrum Image.dm4")
    e = s.axes_manager[-1].axis
    Z = s.data.astype(float)[:162].reshape(81, 2, 60, 2, -1).sum((1, 3))
    sel = (e >= -2.5) & (e <= 3.0)
    e, Z = e[sel], Z[:, :, sel]
    Z = Z / Z.sum(-1, keepdims=True)
    de = e[1] - e[0]
    # sub-channel peak position: centroid of the top of the ZLP
    pk = np.argmax(Z, -1)
    idx = np.arange(Z.shape[-1])
    w = np.where(np.abs(idx[None, None] - pk[..., None]) <= 6, Z, 0)
    pos = (w * e).sum(-1) / w.sum(-1)
    shift = pos - np.median(pos)
    Za = np.stack([ndshift(z, -sh / de, order=3, mode="nearest") for z, sh in zip(Z.reshape(-1, Z.shape[-1]), shift.ravel())])

    def pca(Y):
        mu = Y.mean(0)
        U, sv, Vt = np.linalg.svd(Y - mu, full_matrices=False)
        return mu, U * sv, sv, Vt

    mu, T, sv, Vt = pca(Z.reshape(-1, Z.shape[-1]))
    mua, Ta, sva, Vta = pca(Za)
    dmu = np.gradient(mu, de)
    r = np.corrcoef(T[:, 0], shift.ravel())[0, 1]
    if r < 0:                                                    # sign convention: PC1 score grows with the shift
        T[:, 0], Vt[0], r = -T[:, 0], -Vt[0], -r
    c = np.dot(Vt[0], dmu) / np.linalg.norm(dmu)
    print("ZLP drift: std %.3f eV, corr(PC1 score, shift) = %.3f, cos(PC1, dI/dE) = %.3f" % (shift.std(), r, c))
    zs = Z.max() / 65000                                         # the raw (normalised) stack, for the notebook
    save("zlp", {"E": arr(e), "mean": arr(mu), "dmean": arr(dmu), "shift": arr(shift),
                 "cube": arr(np.round(Z / zs), "u2"), "cube_scale": zs,
                 "raw": {"sv": arr(sv[:20]), "V": arr(Vt[:3]), "T1": arr(T[:, 0])},
                 "aligned": {"sv": arr(sva[:20]), "V": arr(Vta[:3]), "T1": arr(Ta[:, 0])},
                 "shape": list(shift.shape)})


# --------------------------------------------------------------------------------------------
# 5. HAADF image of a SrFeO_x / LaAlO3 interface (Act 3: best rank-k image)
# --------------------------------------------------------------------------------------------
def haadf():
    s = hs.load(SOURCES["haadf"])
    img = s.data.astype(float)                                   # 1024 x 1024
    c = img[256:768, 288:800].reshape(256, 2, 256, 2).mean((1, 3))  # 512 crop across the interface, 2x binned
    c = (c - c.mean()) / c.std()
    U, sv, Vt = np.linalg.svd(c)
    k = 96
    print("HAADF", c.shape, "energy in rank 10: %.3f" % ((sv[:10] ** 2).sum() / (sv ** 2).sum()))
    save("haadf", {"img": arr(c), "U": arr(U[:, :k]), "sv": arr(sv), "Vt": arr(Vt[:k]),
                   "px_nm": 2 * s.axes_manager[0].scale})


# --------------------------------------------------------------------------------------------
# 6. SPED of a twinned GaAs nanowire (Act 5: the same factorisation on diffraction patterns)
# --------------------------------------------------------------------------------------------
def sped():  # noqa: D103
    s = hs.load(SOURCES["sped"], reader="hspy")
    P = s.data.astype(float)                                     # (100, 30, 144, 144)
    P = P[:, :, 8:136, 8:136].reshape(100, 30, 64, 2, 64, 2).sum((3, 5))
    X = P.reshape(3000, -1)
    X = np.sqrt(X)                                               # compress the direct beam
    sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
    nmf = NMF(4, init="nndsvda", max_iter=1500, random_state=0)
    A = nmf.fit_transform(X)
    M = nmf.components_
    sc = M.max(1, keepdims=True)
    A, M = A * sc.T, M / sc
    save("sped", {"maps": arr(A.T.reshape(4, 100, 30)), "patterns": arr(np.round(255 * M).reshape(4, 64, 64), "u1"),
                  "sv": arr(sv[:30]), "vdf": arr(u8(P.sum((2, 3))), "u1"),
                  "mean": arr(u8(X.mean(0).reshape(64, 64)), "u1")})


if __name__ == "__main__":
    which = sys.argv[2:] or ["fe_refs", "cevo3_eels", "cevo3_eds", "zlp", "haadf", "sped"]
    for w in which:
        globals()[w]()
