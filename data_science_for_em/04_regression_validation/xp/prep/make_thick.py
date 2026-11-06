"""Build xp/data/thick.json: thickness (t/lambda) and ADF intensity of ten real specimens, for the
validation and leakage slides.

Usage (needs hyperspy, scikit-learn):  python xp/prep/make_thick.py RAW_DIR
RAW_DIR/thick/NN/ and RAW_DIR/stembank26/<name>/ hold the files "EELS Spectrum Image (low-loss).dm4"
and "ADF Image.dm4" (the ADF signal recorded at the same time, on the same pixel grid) of ten
spectrum images from STEM-Bank-26 (Hollyer & Stach, arXiv:2610.36086, CC BY 4.0), one per session.

Label: t/lambda by the log-ratio method, t/lambda = ln(I_total / I_ZLP) (Egerton). ZLP integral from
the start of the spectrum to the first minimum after the ZLP (mean spectrum of the image).
Feature: the ADF intensity in detector units (arbitrary, set by gain and offset of each session) and
the plasmon-peak energy E_p (maximum of the low-loss spectrum between ZLP end + 3 eV and 45 eV).

Excluded specimens (checked by hand): spectra without a ZLP in range, images with negative t/lambda
(detector artefacts), and an image of vacuum only.
"""
import base64, json, sys
from pathlib import Path

import numpy as np
import hyperspy.api as hs

RAW = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / "data"
SPEC = [  # folder, short name, material (from the session name / element list)
    ("thick/01", "Co catalyst (Wachs)", "Co on C support"),
    ("thick/03", "LiTFSI, cryo", "Li salt, frozen"),
    ("thick/05", "D10 reduced", "oxide powder"),
    ("thick/07", "SI 35, 2023-01-19", "oxide powder"),
    ("thick/08", "calixarene drop", "organic film"),
    ("thick/09", "CeVO3 / Al2O3", "CeVO3 on Al2O3"),
    ("thick/12", "MgO cubes", "MgO"),
    ("thick/13", "LiFSI, cryo", "Li salt, frozen"),
    ("stembank26/CeVO3_Al2O3_SI94", "CeVO3 / Al2O3 (SI 94)", "CeVO3 on Al2O3"),
    ("stembank26/CuMn2O4_SI52", "CuMn2O4", "CuMn2O4 spinel"),
]
NPIX = 400                                                           # pixels kept per specimen (random, fixed seed)


def arr(a, dtype="f4"):
    a = np.ascontiguousarray(np.asarray(a).astype({"f4": "<f4", "u1": "u1", "u2": "<u2"}[dtype]))
    return {"shape": list(a.shape), "dtype": dtype, "b64": base64.b64encode(a.tobytes()).decode()}


rng = np.random.default_rng(0)
rows, specs = [], []
for g, (folder, name, mat) in enumerate(SPEC):
    d = RAW / folder
    ll = hs.load(d / "EELS Spectrum Image (low-loss).dm4"); A = hs.load(d / "ADF Image.dm4").data.astype(float)
    L = ll.data.astype(float); ax = ll.axes_manager[-1]; E = ax.axis
    m = L.reshape(-1, L.shape[-1]).mean(0); p = np.argmax(m)
    mn = p + np.argmin(m[p:p + int(8 / ax.scale)])
    t = np.log(np.maximum(L.sum(-1), 1) / np.maximum(L[..., :mn + 1].sum(-1), 1))
    sel = (E > E[mn] + 3) & (E < 45)
    Ep = E[sel][np.argmax(L[..., sel], -1)]
    ny, nx = t.shape
    idx = rng.choice(ny * nx, size=min(NPIX, ny * nx), replace=False)
    yy, xx = np.unravel_index(idx, (ny, nx))
    for i, y, x in zip(idx, yy, xx):
        rows.append((g, A.ravel()[i], Ep.ravel()[i], t.ravel()[i], y / ny, x / nx))
    meta = json.load(open(d / "meta.json")) if (d / "meta.json").exists() else {"folder": folder, "date": ""}
    specs.append({"name": name, "material": mat, "date": meta.get("date", "")[:10], "folder": meta["folder"],
                  "t_med": float(np.median(t)), "adf_med": float(np.median(A)), "Ep_med": float(np.median(Ep))})
    print(f"{g} {name:24s} t/l {np.percentile(t, 5):.2f}-{np.percentile(t, 95):.2f}  ADF {np.median(A):9.1f}  Ep {np.median(Ep):5.1f} eV  r(ADF,t) {np.corrcoef(A.ravel(), t.ravel())[0, 1]:.2f}")
R = np.array(rows)
G, X, y = R[:, 0].astype(int), R[:, 1:3], R[:, 3]

# reference numbers (the widget recomputes them in JS)
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
for nm, model in [("linear", make_pipeline(StandardScaler(), LinearRegression())), ("5-NN", make_pipeline(StandardScaler(), KNeighborsRegressor(5)))]:
    for cols, fn in [([0], "ADF"), ([0, 1], "ADF+Ep")]:
        for cv_name, cv in [("random 5-fold", KFold(5, shuffle=True, random_state=0)), ("GroupKFold by specimen", GroupKFold(5))]:
            pr = cross_val_predict(model, X[:, cols], y, cv=cv, groups=G)
            print(f"{nm:7s} {fn:7s} {cv_name:24s} R2 {r2_score(y, pr):6.3f}  MAE {mean_absolute_error(y, pr):.3f}")
obj = {"g": arr(G, "u1"), "adf": arr(X[:, 0]), "ep": arr(X[:, 1]), "t": arr(y), "yx": arr(R[:, 4:6]), "specs": specs, "npix": NPIX}
(OUT / "thick.json").write_text(json.dumps(obj, separators=(",", ":")))
print("thick.json %.2f MB" % ((OUT / "thick.json").stat().st_size / 1e6))
