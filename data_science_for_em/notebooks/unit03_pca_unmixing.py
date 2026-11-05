# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "numpy", "matplotlib"]
# ///
import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # Unit 3 · Hands-on: PCA and spectral unmixing on real EM data

    **Data Science for Electron Microscopy** · Prof. Dr. Philipp Pelz · FAU Erlangen-Nürnberg · WS 26/27

    This notebook goes with the Week 3 lecture *Linear algebra, PCA & spectral unmixing*.
    Every data set is a real, openly licensed measurement — the same data as in the lecture explainers.
    You project spectra onto references, fit EELS backgrounds, run the SVD and PCA yourself,
    weight Poisson counts, find a drift artefact, write NMF from scratch, and test when a phase map lies.

    | part | content | data |
    |---|---|---|
    | **1 · Vectors** | dot product, cos θ, projection onto references | Fe-L standards (Cymes et al.) |
    | **2 · Matrix** | datacube → data matrix, power-law background by least squares | CeVO₃/Al₂O₃ EELS (STEM-Bank-26) |
    | **3 · SVD** | strain from a 2 × 2 matrix, the best rank-k image | SrFeOₓ/LaAlO₃ HAADF (STEM-Bank-26) |
    | **4 · PCA** | PCA, scree, Poisson weighting, denoising | CeVO₃/Al₂O₃ EDS (STEM-Bank-26) |
    | **5 · Drift** | a derivative-shaped component, alignment | ZLP spectrum image (Cymes et al.) |
    | **6 · NMF** | multiplicative updates with the KL (Poisson) loss | CeVO₃/Al₂O₃ EDS |
    | **7 · Trust** | rotational ambiguity, a hidden rare phase, overlapping peaks | EDS + planted truth |

    **After this notebook you can:**

    1. Compute cos θ and projections of spectra, and fit fractions of reference spectra by least squares.
    2. Reshape a spectrum image into a data matrix and back.
    3. Write a power-law background fit as a linear least-squares problem and solve it for every pixel at once.
    4. Read rotation and principal strains from the SVD of a 2 × 2 matrix, and verify the Eckart–Young error formula.
    5. Implement PCA (centre, SVD, scores, reconstruction) and the Poisson weighting of counts.
    6. Recognise and remove a drift component.
    7. Implement KL-NMF with multiplicative updates.
    8. Test the uniqueness of an unmixing, and find a hidden phase with a residual map.

    **Time:** about 2 hours; the ⭐ exercises are optional. **Prerequisites:** the Unit 1 and Unit 2 notebooks.
    The notebook uses only NumPy and Matplotlib, and it runs on a CPU. It downloads about 9 MB of data once.

    **To edit and run this notebook:** a shared molab link opens as a read-only preview.
    Sign in to molab and click **Fork** to get your own copy.

    Cells marked ✏️ are exercises. A check under each exercise tells you if your answer is correct.
    Solutions are in closed boxes: open them only after you have tried.
    """)
    return


@app.cell(hide_code=True)
def marimo_intro(mo):
    mo.callout(mo.md(r"""
    ### How this notebook works (marimo is not Jupyter)

    - **Run a cell:** `Ctrl`/`Cmd` + `Enter`. When you change a cell or move a slider, every cell that
      *uses* its variables runs again by itself.
    - **One name, one cell.** Each variable is defined in exactly one cell. Names that start with an
      underscore (`_x`, `_fig`) are private to their cell.
    - **Exercises** define functions or variables with `None`. Replace each `None` and the check below runs.
    - **Data** are loaded from the course repository on GitHub (or from the local repository, if you
      run the notebook from a clone).
    """), kind="info")
    return


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
    return mo, np, plt


@app.cell(hide_code=True)
def helpers(mo, np):
    import base64 as _b64
    import json as _json
    import urllib.request as _ur
    from pathlib import Path as _Path

    _URL = ("https://raw.githubusercontent.com/ECLIPSE-Lab/public_presentations/main/"
            "data_science_for_em/03_linear_algebra_spectral/xp/data/{}.json")
    _DT = {"f4": "<f4", "u2": "<u2", "u1": "u1", "i2": "<i2"}

    def _decode(v):
        if isinstance(v, dict) and "b64" in v and "dtype" in v:
            return np.frombuffer(_b64.b64decode(v["b64"]), _DT[v["dtype"]]).reshape(v["shape"]).astype(float)
        if isinstance(v, dict):
            return {k: _decode(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_decode(x) for x in v]
        return v

    def load(name):
        """Load one data set of the Week 3 lecture: the local repository first, else GitHub."""
        _here = _Path(mo.notebook_dir() or ".")
        _local = _here.parent / "03_linear_algebra_spectral" / "xp" / "data" / f"{name}.json"
        if _local.exists():
            _txt = _local.read_text()
        else:
            with _ur.urlopen(_URL.format(name), timeout=120) as _r:
                _txt = _r.read().decode()
        return _decode(_json.loads(_txt))

    def checklist(items, empty_msg=None):
        """items: list of (passed, text). Shows a green or orange box with one line per check."""
        if empty_msg is not None:
            return mo.callout(mo.md(empty_msg), kind="neutral")
        ok = all(p for p, _ in items)
        lines = [("✅ " if p else "❌ ") + t for p, t in items]
        head = "**All checks pass. Well done!**\n\n" if ok else "**Not yet.** Look at the red lines.\n\n"
        return mo.callout(mo.md(head + "\n\n".join(lines)), kind="success" if ok else "warn")

    def safe(f, *a):
        """Call a student function; return (result, error message)."""
        try:
            return f(*a), None
        except Exception as e:  # noqa: BLE001
            return None, f"`{type(e).__name__}: {e}`"

    return checklist, load, safe


@app.cell(hide_code=True)
def reference(np):
    # Reference implementations. The checks compare your answers with these functions,
    # and later parts use them, so that every part runs even if an exercise is open.
    def ref_pca(X):
        mu = X.mean(0)
        U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
        return mu, U * s, Vt, s

    def ref_weight(X):
        g = X.sum(1, keepdims=True)
        G = np.sqrt(g / g.mean())
        H = np.sqrt(X.mean(0, keepdims=True)) + 1e-6
        return X / G / H, G, H

    def ref_powerlaw(E, X, lo, hi):
        m = (E >= lo) & (E < hi)
        A = np.stack([np.ones(m.sum()), np.log(E[m])], 1)
        Y = np.log(np.clip(X[:, m], 1e-9, None)).T
        w = np.linalg.solve(A.T @ A, A.T @ Y)
        return w[0], -w[1]

    def ref_kl_step(X, W, H, eps=1e-12):
        H = H * (W.T @ (X / (W @ H + eps))) / (W.sum(0)[:, None] + eps)
        W = W * ((X / (W @ H + eps)) @ H.T) / (H.sum(1)[None, :] + eps)
        return W, H

    def kl_loss(X, W, H, eps=1e-12):
        M = W @ H + eps
        return float(np.sum(np.where(X > 0, X * np.log(np.where(X > 0, X, 1) / M), 0) - X + M))

    return kl_loss, ref_kl_step, ref_pca, ref_powerlaw, ref_weight


@app.cell
def load_data(load):
    fe = load("fe_refs")        # Fe-L EELS standards (Cymes et al.)
    eels = load("cevo3_eels")   # CeVO3 / Al2O3 EELS spectrum image (STEM-Bank-26)
    eds = load("cevo3_eds")     # CeVO3 / Al2O3 EDS spectrum image, same area
    zlp = load("zlp")           # ZLP spectrum image with energy drift (Cymes et al.)
    haadf = load("haadf")       # SrFeOx / LaAlO3 HAADF image (STEM-Bank-26)
    return eds, eels, fe, haadf, zlp


@app.cell(hide_code=True)
def s1(mo):
    mo.md(r"""
    ---
    # Part 1 · Vectors: when are two spectra alike?

    ## 1.1 · Four Fe-L₂,₃ standards

    An EELS spectrum with $D$ channels is a vector $\mathbf x\in\mathbb R^D$. Here $D = 301$ channels
    (700–730 eV, 0.1 eV), background removed, normalised to unit area. The four standards are
    Fe metal (Fe⁰), FeO (Fe²⁺), magnetite (Fe²⁺ + 2 Fe³⁺) and a Fe³⁺ glass, from Cymes, Burgess & Stroud,
    *Meteoritics & Planetary Science* **58**, 259 (2023). They used such standards to detect Fe³⁺ in a lunar pyroxene.

    The L₃ maximum moves to higher energy with the oxidation state: about 708.8 eV for Fe²⁺, 710.2 eV for Fe³⁺.
    """)
    return


@app.cell
def fe_plot(fe, plt):
    E_fe = fe["E"]
    REF = {k: fe["refs"][k] for k in ("Fe0", "Fe2", "Fe23", "Fe3")}
    NAMES = {"Fe0": "Fe⁰ (FeNi)", "Fe2": "Fe²⁺ (FeO)", "Fe23": "magnetite", "Fe3": "Fe³⁺ (glass)"}
    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    for _k, _y in REF.items():
        _ax.plot(E_fe, _y, lw=2, label=NAMES[_k])
    _ax.set_xlabel("energy loss (eV)")
    _ax.set_ylabel("intensity (unit area)")
    _ax.legend()
    _ax.set_title("Fe-L₂,₃ standards")
    _fig.tight_layout()
    _fig
    return E_fe, NAMES, REF


@app.cell(hide_code=True)
def s1_2(mo):
    mo.md(r"""
    ## 1.2 · ✏️ Exercise 1: cos θ and the Fe³⁺ fraction

    1. Write `cosine(a, b)` $= \mathbf a^T\mathbf b / (\lVert\mathbf a\rVert\,\lVert\mathbf b\rVert)$.
    2. Write `fit_fractions(y, R)`: the least-squares weights $\hat{\mathbf w}$ of $\mathbf y \approx \mathbf R\mathbf w$,
       where the columns of `R` (shape `(D, k)`) are reference spectra. Return `(w, r)` with the residual
       $\mathbf r = \mathbf y - \mathbf R\hat{\mathbf w}$. *Hint:* `np.linalg.lstsq(R, y, rcond=None)[0]`.

    The check tests a planted mixture (you know the answer), the orthogonality $\mathbf R^T\mathbf r = 0$,
    and then applies your fit to real magnetite.
    """)
    return


@app.cell
def ex1(np):
    # ✏️ YOUR TURN: replace each `None`.
    def cosine(a, b):
        return None

    def fit_fractions(y, R):
        w = None
        r = None
        return w, r

    return cosine, fit_fractions


@app.cell(hide_code=True)
def ex1_check(E_fe, NAMES, REF, checklist, cosine, fit_fractions, mo, np, plt, safe):
    _R = np.stack([REF["Fe2"], REF["Fe3"]], 1)
    _c, _e1 = safe(cosine, REF["Fe2"], REF["Fe23"])
    if _c is None and _e1 is None:
        _out = checklist([], "Write `cosine` and `fit_fractions` in the cell above. The check runs automatically.")
    else:
        _ref_c = float(REF["Fe2"] @ REF["Fe23"] / (np.linalg.norm(REF["Fe2"]) * np.linalg.norm(REF["Fe23"])))
        _ok_c = _c is not None and np.isclose(float(_c), _ref_c, atol=1e-6)
        _y = 0.7 * REF["Fe2"] + 0.3 * REF["Fe3"]
        _res, _e2 = safe(fit_fractions, _y, _R)
        _ok_w = _ok_r = False
        if _res is not None and _res[0] is not None:
            _w, _r = np.asarray(_res[0], float), np.asarray(_res[1], float)
            _ok_w = _w.shape == (2,) and np.allclose(_w, [0.7, 0.3], atol=1e-6)
            _ok_r = _r.shape == _y.shape and np.allclose(_R.T @ _r, 0, atol=1e-8)
        _items = [
            (_ok_c, f"`cosine(Fe²⁺, magnetite)` = {_c if _e1 is None else _e1} (expected {_ref_c:.4f})"),
            (_ok_w, "planted mixture 0.7·Fe²⁺ + 0.3·Fe³⁺: `fit_fractions` recovers w = (0.7, 0.3)" + (f" — error {_e2}" if _e2 else "")),
            (_ok_r, "the residual is orthogonal to every reference: Rᵀr = 0"),
        ]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _w, _r = fit_fractions(REF["Fe23"], _R)
            _f3 = _w[1] / _w.sum()
            _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.6))
            _ax[0].plot(E_fe, REF["Fe23"], "k", lw=2, label="magnetite (real)")
            _ax[0].plot(E_fe, _R @ _w, "r", lw=2, label="projection Rŵ")
            _ax[0].legend()
            _ax[0].set_xlabel("energy loss (eV)")
            _ax[1].plot(E_fe, _r, color="#f59e0b")
            _ax[1].axhline(0, color="gray")
            _ax[1].set_title("residual r")
            _ax[1].set_xlabel("energy loss (eV)")
            _fig.tight_layout()
            _cos = ", ".join(f"{NAMES[k]}: {cosine(REF[k], REF['Fe23']):.3f}" for k in ("Fe0", "Fe2", "Fe3"))
            _parts += [_fig, mo.md(
                f"**Real magnetite:** Fe³⁺/ΣFe = **{_f3:.2f}**, but the stoichiometry says 0.67. "
                f"cos θ with each standard: {_cos}.\n\n"
                "**Think:** the residual has structure near the L₃ peak. What does that tell you about the references? "
                "(Half of the Fe³⁺ in magnetite sits on tetrahedral sites; the glass standard has another Fe³⁺ environment.) "
                "Why can a single cos θ value not tell Fe²⁺ from Fe³⁺ here?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex1_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def cosine(a, b):
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    def fit_fractions(y, R):
        w = np.linalg.lstsq(R, y, rcond=None)[0]
        r = y - R @ w
        return w, r
    ```
    Least squares projects $\mathbf y$ onto the span of the references. The residual is perpendicular to
    that span, so $\mathbf R^T\mathbf r = 0$ (the normal equations).
    """)})
    return


@app.cell(hide_code=True)
def s2(mo):
    mo.md(r"""
    ---
    # Part 2 · Matrix: a spectrum image is a data matrix

    ## 2.1 · The CeVO₃ / Al₂O₃ EELS spectrum image

    CeVO₃ particles on an Al₂O₃ support, from the open data set STEM-Bank-26 (Hollyer & Stach, arXiv:2610.36086,
    CC BY 4.0). Here 40 × 40 pixels (2 × 2 binned, 1.25 nm) × 512 channels (1 eV, 463–975 eV).
    Edges: V-L₂,₃ (513 eV), O-K (532 eV), Ce-M₄,₅ (883 eV).

    ## 2.2 · ✏️ Exercise 2: reshape

    1. Set `X_eels`: the cube `eels_cube` (shape `(40, 40, 512)`) as a data matrix with one **row per pixel**.
    2. Set `ce_raw_map`: for each pixel, the sum over the channels with 878 ≤ E < 910 eV, as a `(40, 40)` image.
       *Hint:* a boolean mask `m = (E_eels >= 878) & (E_eels < 910)` selects columns: `X_eels[:, m]`.
    """)
    return


@app.cell
def eels_data(eels, np):
    eels_cube = eels["cube"] * eels["scale"]
    E_eels = np.asarray(eels["E"])
    return E_eels, eels_cube


@app.cell
def ex2(E_eels, eels_cube, np):
    # ✏️ YOUR TURN
    X_eels = None       # shape (1600, 512)
    ce_raw_map = None   # shape (40, 40)
    return ce_raw_map, X_eels


@app.cell(hide_code=True)
def ex2_check(E_eels, X_eels, ce_raw_map, checklist, eels_cube, mo, np, plt):
    if X_eels is None and ce_raw_map is None:
        _out = checklist([], "Fill in `X_eels` and `ce_raw_map`.")
    else:
        _X = None if X_eels is None else np.asarray(X_eels)
        _ok1 = _X is not None and _X.shape == (1600, 512)
        _ok2 = _ok1 and np.allclose(_X[0], eels_cube[0, 0]) and np.allclose(_X[41], eels_cube[1, 1])
        _m = (E_eels >= 878) & (E_eels < 910)
        _ref = eels_cube[:, :, _m].sum(-1)
        _ok3 = ce_raw_map is not None and np.shape(ce_raw_map) == (40, 40) and np.allclose(ce_raw_map, _ref)
        _items = [(_ok1, "`X_eels` has shape (1600, 512)"),
                  (_ok2, "row 0 is pixel (0, 0) and row 41 is pixel (1, 1)"),
                  (_ok3, "`ce_raw_map` is the Ce-M window sum, shape (40, 40)")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _fig, _ax = plt.subplots(1, 2, figsize=(11, 4))
            _ax[0].imshow(ce_raw_map, cmap="magma")
            _ax[0].set_title("Ce-M window, raw sum")
            _ax[1].semilogy(E_eels, X_eels[7 * 40 + 35])
            _ax[1].axvspan(878, 910, color="orange", alpha=0.3)
            _ax[1].set_title("spectrum of pixel (7, 35) = row 315")
            _ax[1].set_xlabel("energy loss (eV)")
            _fig.tight_layout()
            _parts += [_fig, mo.md("The raw window sum mostly shows **thickness**: the window sits on a large background. "
                                   "The bright pixels at the top left are the first scan lines (a scan-start artefact). "
                                   "Part 2.3 removes the background.")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex2_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    X_eels = eels_cube.reshape(-1, eels_cube.shape[-1])
    _m = (E_eels >= 878) & (E_eels < 910)
    ce_raw_map = X_eels[:, _m].sum(1).reshape(40, 40)
    ```
    """)})
    return


@app.cell(hide_code=True)
def s2_3(mo):
    mo.md(r"""
    ## 2.3 · ✏️ Exercise 3: the power-law background for every pixel at once

    Below an edge, $I(E) = A E^{-r}$, so $\log I = \log A - r \log E$: **linear** in $(\log A, r)$.
    With the design matrix $\mathbf B = [\mathbf 1, \log\mathbf E]$ (one row per channel in the fit window)
    and $\mathbf Y = \log \mathbf I$ (one *column* per pixel), the normal equations
    $\mathbf B^T\mathbf B\,\hat{\mathbf W} = \mathbf B^T\mathbf Y$ solve all 1 600 fits in one call.

    Write `fit_powerlaw(E, X, lo, hi)` that returns two arrays of length N: `logA` and `r`.
    *Hint:* `np.linalg.solve(B.T @ B, B.T @ Y)` gives a `(2, N)` array.
    """)
    return


@app.cell
def ex3(np):
    # ✏️ YOUR TURN
    def fit_powerlaw(E, X, lo, hi):
        m = (E >= lo) & (E < hi)
        logA = None
        r = None
        return logA, r

    return (fit_powerlaw,)


@app.cell(hide_code=True)
def ex3_check(E_eels, checklist, eels_cube, fit_powerlaw, mo, np, ref_powerlaw, safe):
    _X = eels_cube.reshape(-1, eels_cube.shape[-1])
    _res, _err = safe(fit_powerlaw, E_eels, _X, 840, 875)
    if _err is None and (_res is None or _res[0] is None):
        _out = checklist([], "Complete `fit_powerlaw`.")
    else:
        _a0, _r0 = ref_powerlaw(E_eels, _X, 840, 875)
        _ok = _res is not None and np.allclose(_res[0], _a0, rtol=1e-6, atol=1e-6) and np.allclose(_res[1], _r0, rtol=1e-6, atol=1e-6)
        _out = checklist([(_ok, "`fit_powerlaw` agrees with the reference for all 1 600 pixels" + (f" — {_err}" if _err else "")),
                          (_ok and 1 < np.median(_res[1]) < 6, "the median exponent r is between 1 and 6 (typical for EELS)")])
    _out
    return


@app.cell(hide_code=True)
def ex3_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def fit_powerlaw(E, X, lo, hi):
        m = (E >= lo) & (E < hi)
        B = np.stack([np.ones(m.sum()), np.log(E[m])], 1)        # (n_channels, 2)
        Y = np.log(np.clip(X[:, m], 1e-9, None)).T              # (n_channels, N)
        W = np.linalg.solve(B.T @ B, B.T @ Y)                    # (2, N)
        return W[0], -W[1]
    ```
    """)})
    return


@app.cell(hide_code=True)
def s2_4(mo):
    mo.md(r"""
    ## 2.4 · Try it: edge maps after background removal

    Move the fit window. Watch the extracted signal of the selected edge.
    """)
    return


@app.cell
def bg_controls(mo):
    edge_pick = mo.ui.dropdown({"V-L₂,₃": "V", "Ce-M₄,₅": "Ce"}, value="Ce-M₄,₅", label="edge")
    win_start = mo.ui.slider(470, 870, value=840, step=2, label="window start (eV)")
    win_width = mo.ui.slider(8, 60, value=35, step=1, label="window width (eV)")
    mo.hstack([edge_pick, win_start, win_width])
    return edge_pick, win_start, win_width


@app.cell
def bg_maps(E_eels, edge_pick, eels_cube, np, plt, ref_powerlaw, win_start, win_width):
    _X = eels_cube.reshape(-1, eels_cube.shape[-1])
    _sig = {"V": (512, 530), "Ce": (878, 910)}[edge_pick.value]
    _lo, _hi = win_start.value, win_start.value + win_width.value
    _a, _r = ref_powerlaw(E_eels, _X, _lo, _hi)
    _bg = np.exp(_a[:, None]) * E_eels[None, :] ** (-_r[:, None])
    _m = (E_eels >= _sig[0]) & (E_eels < _sig[1])
    _edge = (_X - _bg)[:, _m].sum(1)
    _thin = _X.sum(1) < 0.5 * np.median(_X.sum(1))
    _edge[_thin | (_r < 0.5) | (_r > 8)] = 0
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4))
    _ax[0].imshow(_edge.reshape(40, 40), cmap="magma", vmin=np.percentile(_edge, 1), vmax=np.percentile(_edge, 99))
    _ax[0].set_title(f"{edge_pick.selected_key} signal, window {_lo}–{_hi} eV")
    _s = _X.mean(0)
    _a1, _r1 = ref_powerlaw(E_eels, _s[None], _lo, _hi)
    _v = (E_eels > _lo - 20) & (E_eels < _sig[1] + 60)
    _ax[1].plot(E_eels[_v], _s[_v], "k", label="mean spectrum")
    _ax[1].plot(E_eels[_v], np.exp(_a1[0]) * E_eels[_v] ** -_r1[0], "g--", label=f"A·E^-r, r = {_r1[0]:.2f}")
    _ax[1].axvspan(_lo, _hi, color="g", alpha=0.15)
    _ax[1].axvspan(*_sig, color="orange", alpha=0.2)
    _ax[1].legend()
    _ax[1].set_xlabel("energy loss (eV)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def s3(mo):
    mo.md(r"""
    ---
    # Part 3 · SVD: what does a matrix do?

    ## 3.1 · ✏️ Exercise 4: strain from the SVD of a 2 × 2 matrix

    In 4D-STEM strain mapping, a 2 × 2 matrix $\mathbf A$ maps the reference lattice vectors onto the measured ones.
    $\mathbf A = \mathbf U\boldsymbol\Sigma\mathbf V^T$: the **principal strains** are $\sigma_k - 1$, and the lattice
    **rotation** is the rotation matrix $\mathbf U\mathbf V^T$ (polar decomposition).

    For the matrix `A_strain` below, set `strains` (array of the two principal strains, largest first)
    and `rot_deg` (the rotation angle of $\mathbf U\mathbf V^T$ in degrees; use `np.arctan2(R[1, 0], R[0, 0])`).
    """)
    return


@app.cell
def strain_matrix(np):
    _t, _p = np.deg2rad(4.0), np.deg2rad(25.0)
    _rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    A_strain = _rot(_t) @ _rot(_p) @ np.diag([1.025, 0.99]) @ _rot(-_p)   # a measured lattice distortion
    A_strain
    return (A_strain,)


@app.cell
def ex4(A_strain, np):
    # ✏️ YOUR TURN
    strains = None   # e.g. np.array([σ1 - 1, σ2 - 1])
    rot_deg = None
    return rot_deg, strains


@app.cell(hide_code=True)
def ex4_check(checklist, mo, np, rot_deg, strains):
    if strains is None and rot_deg is None:
        _out = checklist([], "Fill in `strains` and `rot_deg`.")
    else:
        _ok1 = strains is not None and np.allclose(np.asarray(strains, float), [0.025, -0.01], atol=1e-6)
        _ok2 = rot_deg is not None and np.isclose(float(rot_deg), 4.0, atol=1e-6)
        _out = mo.vstack([checklist([(_ok1, f"principal strains = {strains} (expected +2.5 % and −1.0 %)"),
                                     (_ok2, f"rotation = {rot_deg}° (expected 4°)")]),
                          mo.md("The 25° in the construction is the direction of the principal strain axes, not a rotation of the lattice. "
                                "SVD separates the two.") if _ok1 and _ok2 else mo.md("")])
    _out
    return


@app.cell(hide_code=True)
def ex4_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    U, s, Vt = np.linalg.svd(A_strain)
    strains = s - 1
    R = U @ Vt
    rot_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    ```
    """)})
    return


@app.cell(hide_code=True)
def s3_2(mo):
    mo.md(r"""
    ## 3.2 · The best rank-k image (Eckart–Young)

    A HAADF image of a SrFeOₓ / LaAlO₃ interface (STEM-Bank-26), 256 × 256, standardised.
    The truncated SVD $\mathbf X_k = \sum_{j\le k}\sigma_j\mathbf u_j\mathbf v_j^T$ is the best rank-k approximation, and
    $\lVert\mathbf X-\mathbf X_k\rVert_F = \sqrt{\sum_{j>k}\sigma_j^2}$.
    """)
    return


@app.cell
def haadf_svd(haadf, np):
    img = haadf["img"]
    U_img, s_img, Vt_img = np.linalg.svd(img)
    return U_img, Vt_img, img, s_img


@app.cell
def rank_slider(mo):
    k_rank = mo.ui.slider(1, 80, value=8, label="rank k")
    k_rank
    return (k_rank,)


@app.cell
def rank_plot(U_img, Vt_img, img, k_rank, np, plt, s_img):
    _k = k_rank.value
    _Xk = (U_img[:, :_k] * s_img[:_k]) @ Vt_img[:_k]
    _fig, _ax = plt.subplots(1, 4, figsize=(15, 3.8))
    _ax[0].imshow(img, cmap="gray", vmin=-2.5, vmax=3)
    _ax[0].set_title("X")
    _ax[1].imshow(_Xk, cmap="gray", vmin=-2.5, vmax=3)
    _ax[1].set_title(f"X_k, k = {_k}")
    _ax[2].imshow(img - _Xk, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    _ax[2].set_title("X − X_k")
    _ax[3].loglog(np.arange(1, 257), s_img, "k")
    _ax[3].axvline(_k, color="r", ls="--")
    _ax[3].set_title("singular values")
    for _a in _ax[:3]:
        _a.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def s3_3(mo):
    mo.md(r"""
    ## 3.3 · ✏️ Exercise 5: verify Eckart–Young

    For `k = 10`, set `err_direct` $=\lVert\mathbf X-\mathbf X_{10}\rVert_F$ (compute $\mathbf X_{10}$ from `U_img`, `s_img`, `Vt_img`
    and use `np.linalg.norm`) and `err_formula` $=\sqrt{\sum_{j>10}\sigma_j^2}$. Set `kept_fraction`: the fraction of
    $\sum_j\sigma_j^2$ in the first 10 terms.
    """)
    return


@app.cell
def ex5(U_img, Vt_img, img, np, s_img):
    # ✏️ YOUR TURN
    err_direct = None
    err_formula = None
    kept_fraction = None
    return err_direct, err_formula, kept_fraction


@app.cell(hide_code=True)
def ex5_check(U_img, Vt_img, checklist, err_direct, err_formula, img, kept_fraction, np, s_img):
    if err_direct is None and err_formula is None:
        _out = checklist([], "Fill in `err_direct`, `err_formula` and `kept_fraction`.")
    else:
        _ref = np.linalg.norm(img - (U_img[:, :10] * s_img[:10]) @ Vt_img[:10])
        _ok1 = err_direct is not None and np.isclose(float(err_direct), _ref, rtol=1e-5)
        _ok2 = err_formula is not None and np.isclose(float(err_formula), _ref, rtol=1e-5)
        _kf = (s_img[:10] ** 2).sum() / (s_img ** 2).sum()
        _ok3 = kept_fraction is not None and np.isclose(float(kept_fraction), _kf, rtol=1e-5)
        _out = checklist([(_ok1, f"‖X − X₁₀‖_F = {err_direct}"), (_ok2, f"√Σ_(j>10) σⱼ² = {err_formula}: the same number"),
                          (_ok3, f"kept fraction = {kept_fraction} (expected {_kf:.3f})")])
    _out
    return


@app.cell(hide_code=True)
def ex5_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    X10 = (U_img[:, :10] * s_img[:10]) @ Vt_img[:10]
    err_direct = np.linalg.norm(img - X10)
    err_formula = np.sqrt(np.sum(s_img[10:] ** 2))
    kept_fraction = np.sum(s_img[:10] ** 2) / np.sum(s_img ** 2)
    ```
    The crystal is periodic and aligned with the scan, so few terms describe it. The interface and the noise need the rest.
    """)})
    return


@app.cell(hide_code=True)
def s4(mo):
    mo.md(r"""
    ---
    # Part 4 · PCA on a real EDS spectrum image

    ## 4.1 · The data: about 58 counts per pixel

    The EDS spectrum image of the same CeVO₃ / Al₂O₃ area: 80 × 80 pixels × 250 channels (40 eV, 0.2–10.2 keV).
    The values are **counts**: most channels of one pixel are 0 or 1.
    """)
    return


@app.cell
def eds_data(eds, np, plt):
    C_eds = eds["cube"]                      # (80, 80, 250) counts
    E_eds = np.asarray(eds["E"])
    X_eds = C_eds.reshape(-1, C_eds.shape[-1])
    LINES = {"C-K": 0.28, "O-K": 0.52, "Al-K": 1.49, "V-Kα": 4.95, "Ce-Lβ": 5.26, "Cu-Kα": 8.05}
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 2, 2]})
    _ax[0].imshow(X_eds.sum(1).reshape(80, 80), cmap="gray")
    _ax[0].set_title(f"total counts (mean {X_eds.sum(1).mean():.0f} / pixel)")
    _ax[0].axis("off")
    _ax[1].semilogy(E_eds, X_eds.sum(0), "k")
    for _n, _e in LINES.items():
        _ax[1].axvline(_e, color="gray", ls=":")
        _ax[1].text(_e, X_eds.sum(0).max(), _n, rotation=90, va="top", fontsize=8)
    _ax[1].set_title("sum spectrum of all pixels")
    _ax[1].set_xlabel("keV")
    _ax[2].bar(E_eds, X_eds[20 * 80 + 40], width=0.04, color="gray")
    _ax[2].set_title("one pixel (20, 40)")
    _ax[2].set_xlabel("keV")
    _fig.tight_layout()
    _fig
    return C_eds, E_eds, LINES, X_eds


@app.cell(hide_code=True)
def s4_2(mo):
    mo.md(r"""
    ## 4.2 · ✏️ Exercise 6: PCA and the Poisson weighting

    1. Write `my_pca(X)` returning `(mu, T, Vt, s)`: the mean spectrum, the scores $\mathbf T = \mathbf U\boldsymbol\Sigma$,
       the loadings (rows of `Vt`) and the singular values of the **centred** data. Use `full_matrices=False`.
    2. Write `poisson_weight(X)` returning `(Xw, G, H)` with
       $G_i = \sqrt{g_i/\bar g}$ ($g_i$ = total counts of pixel $i$, as a column `(N, 1)`),
       $H_j = \sqrt{\bar x_j} + 10^{-6}$ (mean spectrum, as a row `(1, D)`), and $X^w_{ij} = X_{ij}/(G_i H_j)$
       (Keenan & Kotula 2004).
    """)
    return


@app.cell
def ex6(np):
    # ✏️ YOUR TURN
    def my_pca(X):
        mu = None
        T = None
        Vt = None
        s = None
        return mu, T, Vt, s

    def poisson_weight(X):
        G = None
        H = None
        Xw = None
        return Xw, G, H

    return my_pca, poisson_weight


@app.cell(hide_code=True)
def ex6_check(X_eds, checklist, my_pca, np, poisson_weight, ref_pca, ref_weight, safe):
    _r1, _e1 = safe(my_pca, X_eds)
    if _e1 is None and (_r1 is None or _r1[0] is None):
        _out = checklist([], "Complete `my_pca` and `poisson_weight`.")
    else:
        _mu, _T, _Vt, _s = ref_pca(X_eds)
        _ok1 = _r1 is not None and np.allclose(_r1[0], _mu)
        _ok2 = _r1 is not None and np.allclose(_r1[3][:10], _s[:10], rtol=1e-6)
        _ok3 = _r1 is not None and np.allclose(np.abs(np.sum(_r1[2][:3] * _Vt[:3], 1)), 1, atol=1e-6)
        _ok4 = _r1 is not None and np.allclose(np.abs(_r1[1][:, :3]), np.abs(_T[:, :3]), atol=1e-5)
        _r2, _e2 = safe(poisson_weight, X_eds)
        _Xw, _G, _H = ref_weight(X_eds)
        _ok5 = _r2 is not None and _r2[0] is not None and np.allclose(_r2[0], _Xw, rtol=1e-6)
        _out = checklist([(_ok1, "mean spectrum" + (f" — {_e1}" if _e1 else "")), (_ok2, "singular values"),
                          (_ok3, "first three loadings (up to sign)"), (_ok4, "first three score columns (up to sign)"),
                          (_ok5, "`poisson_weight`" + (f" — {_e2}" if _e2 else ""))])
    _out
    return


@app.cell(hide_code=True)
def ex6_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def my_pca(X):
        mu = X.mean(0)
        U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
        return mu, U * s, Vt, s

    def poisson_weight(X):
        g = X.sum(1, keepdims=True)
        G = np.sqrt(g / g.mean())
        H = np.sqrt(X.mean(0, keepdims=True)) + 1e-6
        return X / G / H, G, H
    ```
    """)})
    return


@app.cell(hide_code=True)
def s4_3(mo):
    mo.md(r"""
    ## 4.3 · Raw vs weighted: the scree plot and the components

    PCA minimises a squared error, so it assumes equal-variance Gaussian noise (Week 2). Poisson counts have
    Var = mean: bright channels dominate. Compare the two scree plots: where is the noise floor?
    """)
    return


@app.cell
def pca_both(X_eds, ref_pca, ref_weight):
    pca_raw = ref_pca(X_eds)
    Xw_eds, G_eds, H_eds = ref_weight(X_eds)
    pca_w = ref_pca(Xw_eds)
    return G_eds, H_eds, pca_raw, pca_w


@app.cell
def comp_controls(mo):
    which_pca = mo.ui.radio({"raw counts": "raw", "Poisson-weighted": "w"}, value="Poisson-weighted", label="PCA of", inline=True)
    comp_k = mo.ui.slider(1, 8, value=1, label="component")
    mo.hstack([which_pca, comp_k])
    return comp_k, which_pca


@app.cell
def comp_plot(E_eds, H_eds, LINES, comp_k, np, pca_raw, pca_w, plt, which_pca):
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1.2, 2, 1]})
    for _p, _c, _n in ((pca_raw, "#3b82f6", "raw"), (pca_w, "#f97316", "weighted")):
        _f = _p[3] ** 2 / np.sum(_p[3] ** 2)
        _ax[0].loglog(np.arange(1, 61), _f[:60], "o-", ms=3, color=_c, label=_n)
    _ax[0].legend()
    _ax[0].set_title("scree: variance fraction")
    _ax[0].set_xlabel("component k")
    _p = pca_w if which_pca.value == "w" else pca_raw
    _k = comp_k.value - 1
    _ld = _p[2][_k] * (H_eds[0] if which_pca.value == "w" else 1)
    _ax[1].plot(E_eds, _ld, color="crimson")
    _ax[1].axhline(0, color="gray")
    for _n, _e in LINES.items():
        _ax[1].axvline(_e, color="gray", ls=":", lw=0.8)
    _ax[1].set_title(f"loading of PC{_k + 1} ({which_pca.value})")
    _ax[1].set_xlabel("keV")
    _sc = _p[1][:, _k].reshape(80, 80)
    _m = np.percentile(np.abs(_sc), 99.5)
    _ax[2].imshow(_sc, cmap="RdBu_r", vmin=-_m, vmax=_m)
    _ax[2].set_title(f"score map PC{_k + 1}")
    _ax[2].axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def s4_4(mo):
    mo.md(r"""
    **Questions.** (a) How many components stand above the flat noise floor after weighting? Before?
    (b) Why are loadings and score maps signed? Is PC2 a phase? (c) In the raw PCA, which channels dominate PC1?

    ## 4.4 · Try it: denoising a 58-count map

    Reconstruct with K weighted components, $\hat{\mathbf X} = \mathbf G\,(\bar{\mathbf x} + \mathbf T_K\mathbf V_K^T)\,\mathbf H$,
    and compare the Ce-Lβ map with the raw counts.
    """)
    return


@app.cell
def k_slider(mo):
    K_den = mo.ui.slider(1, 12, value=4, label="keep K components")
    K_den
    return (K_den,)


@app.cell
def denoise_plot(E_eds, G_eds, H_eds, K_den, X_eds, np, pca_w, plt):
    _mu, _T, _Vt, _s = pca_w
    _K = K_den.value
    _Xh = (_mu + _T[:, :_K] @ _Vt[:_K]) * G_eds * H_eds
    _m = (E_eds >= 5.22) & (E_eds < 5.30)
    _raw = X_eds[:, _m].sum(1).reshape(80, 80)
    _den = _Xh[:, _m].sum(1).reshape(80, 80)
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 1, 2]})
    _hi = np.percentile(_den, 99.5)
    _ax[0].imshow(_raw, cmap="magma", vmin=0, vmax=_hi)
    _ax[0].set_title("Ce-Lβ: raw counts")
    _ax[1].imshow(_den, cmap="magma", vmin=0, vmax=_hi)
    _ax[1].set_title(f"Ce-Lβ: weighted PCA, K = {_K}")
    for _a in _ax[:2]:
        _a.axis("off")
    _i = 12 * 80 + 70
    _ax[2].bar(E_eds, X_eds[_i], width=0.04, color="gray", label="raw counts")
    _ax[2].plot(E_eds, _Xh[_i], "crimson", label=f"reconstruction, K = {_K}")
    _ax[2].set_title("pixel (12, 70), in the Ce-rich particle")
    _ax[2].legend()
    _ax[2].set_xlabel("keV")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def s5(mo):
    mo.md(r"""
    ---
    # Part 5 · A component that is not chemistry: drift

    A zero-loss-peak (ZLP) spectrum image, as recorded (81 × 60 pixels, 0.05 eV channels, each spectrum
    normalised to unit sum). The spectrometer drifts during the scan. Taylor: $I(E-\delta)\approx I(E) - \delta\,I'(E)$.
    So a shift adds a multiple of the derivative, with a weight $\delta$ that changes from pixel to pixel: a rank-1 term.
    """)
    return


@app.cell
def zlp_pca(np, plt, ref_pca, zlp):
    Z = (zlp["cube"] * zlp["cube_scale"]).reshape(-1, zlp["cube"].shape[-1])
    E_z = np.asarray(zlp["E"])
    zshape = zlp["cube"].shape[:2]
    pca_z = ref_pca(Z)
    _mu, _T, _Vt, _s = pca_z
    _d = np.gradient(_mu, E_z)
    _c = _Vt[0] @ _d / np.linalg.norm(_d)
    # measured shift: centroid of the top of each ZLP
    _pk = np.argmax(Z, 1)
    _idx = np.arange(Z.shape[1])
    _w = np.where(np.abs(_idx[None] - _pk[:, None]) <= 6, Z, 0)
    zshift = (_w * E_z).sum(1) / _w.sum(1)
    zshift = zshift - np.median(zshift)
    _r = np.corrcoef(_T[:, 0], zshift)[0, 1]
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [2, 1, 1]})
    _ax[0].plot(E_z, _Vt[0], "crimson", label="PC1 loading")
    _ax[0].plot(E_z, (_Vt[0] @ _d) / (_d @ _d) * _d, "g--", label="best multiple of dI/dE")
    _ax[0].legend()
    _ax[0].set_xlabel("energy loss (eV)")
    _ax[0].set_title(f"cos(PC1, dI/dE) = {abs(_c):.2f}; PC1 = {100 * _s[0] ** 2 / np.sum(_s ** 2):.0f} % of the variance")
    _ax[1].imshow(zshift.reshape(zshape), cmap="RdBu_r")
    _ax[1].set_title("measured shift δ")
    _ax[2].imshow(_T[:, 0].reshape(zshape), cmap="RdBu_r")
    _ax[2].set_title(f"PC1 score map (r = {_r:.2f})")
    for _a in _ax[1:]:
        _a.axis("off")
    _fig.tight_layout()
    _fig
    return E_z, Z, pca_z, zshape, zshift


@app.cell(hide_code=True)
def s5_2(mo):
    mo.md(r"""
    ## 5.1 · ⭐ Exercise 7: align and decompose again

    Set `Z_aligned`: each spectrum (row of `Z`) shifted so that its ZLP sits at 0 eV.
    A spectrum with shift $\delta_i$ is $I(E - \delta_i)$, so evaluate it at $E + \delta_i$:
    `np.interp(E_z + zshift[i], E_z, Z[i])`. Then the check runs PCA on your aligned stack.
    """)
    return


@app.cell
def ex7(E_z, Z, np, zshift):
    # ✏️ YOUR TURN
    Z_aligned = None
    return (Z_aligned,)


@app.cell(hide_code=True)
def ex7_check(Z, Z_aligned, checklist, mo, np, ref_pca, zshift):
    if Z_aligned is None:
        _out = checklist([], "Fill in `Z_aligned`.")
    else:
        _Za = np.asarray(Z_aligned, float)
        _ok0 = _Za.shape == Z.shape
        _out_parts = []
        if _ok0:
            _, _Ta, _, _sa = ref_pca(_Za)
            _, _, _, _s0 = ref_pca(Z)
            _drop = np.sum(_s0 ** 2) / np.sum(_sa ** 2)
            _r = np.corrcoef(_Ta[:, 0], zshift)[0, 1]
            _items = [(True, "shape"), (_drop > 50, f"total variance falls {_drop:.0f}-fold (expected > 50)"),
                      (abs(_r) < 0.3, f"the new PC1 score map no longer follows the drift: r = {_r:.2f}")]
        else:
            _items = [(False, f"`Z_aligned` must have shape {Z.shape}")]
        _out = mo.vstack([checklist(_items), mo.md(
            "Most of the variance of the recorded stack was the drift. Align first, then decompose. "
            "A loading with a positive and a negative lobe around a peak is the fingerprint of a shift.")])
    _out
    return


@app.cell(hide_code=True)
def ex7_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    Z_aligned = np.stack([np.interp(E_z + zshift[i], E_z, Z[i]) for i in range(len(Z))])
    ```
    """)})
    return


@app.cell(hide_code=True)
def s6(mo):
    mo.md(r"""
    ---
    # Part 6 · NMF: phases with only non-negative parts

    $\mathbf X\approx\mathbf W\mathbf H$ with $\mathbf W\ge 0$ (maps, N × K) and $\mathbf H\ge 0$ (spectra, K × D).
    For counts, the loss is the generalised KL divergence — the Poisson NLL up to a constant (Week 2).
    The Lee–Seung multiplicative updates:

    $$\mathbf H\leftarrow\mathbf H\odot\frac{\mathbf W^T(\mathbf X/\mathbf{WH})}{\mathbf W^T\mathbf 1},\qquad
      \mathbf W\leftarrow\mathbf W\odot\frac{(\mathbf X/\mathbf{WH})\,\mathbf H^T}{\mathbf 1\,\mathbf H^T}$$

    (element-wise products and divisions; update **H first**, then W with the new H).

    ## 6.1 · ✏️ Exercise 8: one KL update

    Write `kl_step(X, W, H)` that returns the updated `(W, H)`. Add `eps = 1e-12` to every denominator.
    *Hint:* $\mathbf W^T\mathbf 1$ is `W.sum(0)[:, None]` (shape K × 1); $\mathbf 1\mathbf H^T$ is `H.sum(1)[None, :]` (shape 1 × K).
    """)
    return


@app.cell
def ex8(np):
    # ✏️ YOUR TURN
    def kl_step(X, W, H, eps=1e-12):
        H = None
        W = None
        return W, H

    return (kl_step,)


@app.cell(hide_code=True)
def ex8_check(checklist, kl_loss, kl_step, np, ref_kl_step, safe):
    _rng = np.random.default_rng(0)
    _X = _rng.poisson(5 * _rng.random((30, 3)) @ _rng.random((3, 20))).astype(float)
    _W0, _H0 = _rng.random((30, 3)) + 0.1, _rng.random((3, 20)) + 0.1
    _r, _e = safe(kl_step, _X, _W0, _H0)
    if _e is None and (_r is None or _r[0] is None):
        _out = checklist([], "Complete `kl_step`.")
    else:
        _ok1 = _r is not None and np.allclose(_r[0], ref_kl_step(_X, _W0, _H0)[0]) and np.allclose(_r[1], ref_kl_step(_X, _W0, _H0)[1])
        _ok2 = _ok3 = False
        if _ok1:
            _W, _H, _L = _W0.copy(), _H0.copy(), [kl_loss(_X, _W0, _H0)]
            for _ in range(50):
                _W, _H = kl_step(_X, _W, _H)
                _L.append(kl_loss(_X, _W, _H))
            _ok2 = all(b <= a + 1e-9 for a, b in zip(_L, _L[1:]))
            _ok3 = (_W >= 0).all() and (_H >= 0).all()
        _out = checklist([(_ok1, "one step equals the reference update" + (f" — {_e}" if _e else "")),
                          (_ok2, "the KL loss never increases over 50 steps"), (_ok3, "W and H stay non-negative")])
    _out
    return


@app.cell(hide_code=True)
def ex8_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def kl_step(X, W, H, eps=1e-12):
        H = H * (W.T @ (X / (W @ H + eps))) / (W.sum(0)[:, None] + eps)
        W = W * ((X / (W @ H + eps)) @ H.T) / (H.sum(1)[None, :] + eps)
        return W, H
    ```
    A gradient step with a per-element step size $\eta = \mathbf H / \mathbf W^T\mathbf 1$. All factors are ≥ 0, so the update keeps W, H ≥ 0.
    """)})
    return


@app.cell(hide_code=True)
def s6_2(mo):
    mo.md(r"""
    ## 6.2 · Run NMF on the EDS counts

    2 × 2 binned (40 × 40 pixels, ≈ 230 counts per pixel). Choose K and the random start, then compare starts.
    """)
    return


@app.cell
def nmf_controls(mo):
    K_nmf = mo.ui.slider(2, 5, value=3, label="K")
    seed_nmf = mo.ui.number(0, 99, value=0, label="random start")
    iters_nmf = mo.ui.slider(50, 600, value=300, step=50, label="iterations")
    mo.hstack([K_nmf, seed_nmf, iters_nmf])
    return K_nmf, iters_nmf, seed_nmf


@app.cell
def nmf_run(C_eds, E_eds, K_nmf, iters_nmf, kl_loss, np, plt, ref_kl_step, seed_nmf):
    X_bin = C_eds.reshape(40, 2, 40, 2, -1).sum((1, 3)).reshape(1600, -1)

    def run_nmf(X, K, seed, iters):
        _rng = np.random.default_rng(seed)
        _sc = np.sqrt(X.mean() / K)
        W = _sc * (0.2 + _rng.random((X.shape[0], K)))
        H = _sc * (0.2 + _rng.random((K, X.shape[1])))
        for _ in range(iters):
            W, H = ref_kl_step(X, W, H)
        _n = H.sum(1, keepdims=True)
        return W * _n.T, H / _n                                  # spectra of unit area

    W_nmf, H_nmf = run_nmf(X_bin, K_nmf.value, seed_nmf.value, iters_nmf.value)
    _fig, _ax = plt.subplots(1, K_nmf.value + 1, figsize=(4 + 3 * K_nmf.value, 3.6),
                             gridspec_kw={"width_ratios": [2.5] + [1] * K_nmf.value})
    for _k in range(K_nmf.value):
        _ax[0].semilogy(E_eds, H_nmf[_k] + 1e-5, label=f"component {_k + 1}")
        _ax[_k + 1].imshow(W_nmf[:, _k].reshape(40, 40), cmap="magma")
        _ax[_k + 1].set_title(f"map {_k + 1}")
        _ax[_k + 1].axis("off")
    _ax[0].legend(fontsize=8)
    _ax[0].set_ylim(1e-4, 0.5)
    _ax[0].set_xlabel("keV")
    _ax[0].set_title(f"KL loss = {kl_loss(X_bin, W_nmf, H_nmf):.4g}")
    _fig.tight_layout()
    _fig
    return H_nmf, W_nmf, X_bin, run_nmf


@app.cell(hide_code=True)
def s6_3(mo):
    mo.md(r"""
    **Questions.** (a) Which component is the C film / Cu grid, which Al₂O₃, which CeVOₓ? Use the lines C-K, Al-K, V-K, Ce-L, Cu-K.
    (b) Change the random start. Do the spectra change? (c) With K = 4, what does the extra component describe?

    ---
    # Part 7 · Trust

    ## 7.1 · ⭐ Exercise 9: the same fit, other factors

    For any invertible $\mathbf T$, $\mathbf{WH} = (\mathbf W\mathbf T^{-1})(\mathbf T\mathbf H)$. Use
    $\mathbf T_s = s\,\mathbf I + \frac{1-s}{K}\mathbf 1\mathbf 1^T$: it moves every endmember spectrum away from
    (s > 1) or towards (s < 1) their mean.

    Write `transform(W, H, s)` that returns `(W2, H2)` with `H2 = T_s @ H` and `W2 = W @ inv(T_s)`. The check scans
    `s` and reports for which values both factors stay non-negative (within the noise tolerance) —
    first with all pixels, then without the pure pixels.
    """)
    return


@app.cell
def ex9(np):
    # ✏️ YOUR TURN
    def transform(W, H, s):
        K = H.shape[0]
        T = None
        W2 = None
        H2 = None
        return W2, H2

    return (transform,)


@app.cell(hide_code=True)
def ex9_check(H_nmf, W_nmf, checklist, mo, np, safe, transform):
    _r, _e = safe(transform, W_nmf, H_nmf, 1.2)
    if _e is None and (_r is None or _r[0] is None):
        _out = checklist([], "Complete `transform`.")
    else:
        _ok1 = _r is not None and np.allclose(_r[0] @ _r[1], W_nmf @ H_nmf, rtol=1e-6, atol=1e-8)
        _K = H_nmf.shape[0]
        _T = 1.2 * np.eye(_K) + (1 - 1.2) / _K * np.ones((_K, _K))
        _ok2 = _r is not None and np.allclose(_r[1], _T @ H_nmf)
        _parts = [checklist([(_ok1, "W2 @ H2 equals W @ H (the fit is unchanged)" + (f" — {_e}" if _e else "")), (_ok2, "H2 = T_s H")])]
        if _ok1 and _ok2:
            _frac = W_nmf / W_nmf.sum(1, keepdims=True)
            _thick = W_nmf.sum(1) > np.percentile(W_nmf.sum(1), 5)
            _rows = []
            for _name, _keep in (("all pixels", _thick), ("no pure pixels (all fractions > 0.2)", _thick & (_frac > 0.2).all(1))):
                _ok = []
                for _s in np.round(np.arange(0.3, 1.61, 0.01), 2):
                    _W2, _H2 = transform(W_nmf[_keep], H_nmf, _s)
                    _a = _W2 / _W2.sum(1, keepdims=True)
                    if (_H2.min() >= -0.003 * H_nmf.max()) and ((_a < -0.1).any(1).mean() <= 0.05):
                        _ok.append(_s)
                _rows.append(f"| {_name} | {_keep.sum()} | " + (f"[{min(_ok):.2f}, {max(_ok):.2f}]" if _ok else "–") + " |")
            _parts.append(mo.md("| pixels | N | feasible s |\n|---|---|---|\n" + "\n".join(_rows) +
                                "\n\nEvery s in the feasible range fits the data **exactly equally well**. With pure pixels the range is narrow; "
                                "without them it is wide. Constraints (closure, known spectra, pure regions) are what make an unmixing unique."))
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex9_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def transform(W, H, s):
        K = H.shape[0]
        T = s * np.eye(K) + (1 - s) / K * np.ones((K, K))
        return W @ np.linalg.inv(T), T @ H
    ```
    """)})
    return


@app.cell(hide_code=True)
def s7_2(mo):
    mo.md(r"""
    ## 7.2 · ✏️ Exercise 10: find the hidden phase

    As in the lecture, a rare phase is planted into the real data: three tiny Pt particles (12 pixels, 0.2 % of the map), each pixel
    with on average 10 extra Pt counts (Pt-M 2.05 keV, Pt-L 9.44 keV). You do not know where.
    The cell below runs the weighted PCA on the planted cube. Its scree shows nothing unusual after K = 4.

    Compute the **Pearson residual map** at K = 4:
    $\rho_i = \frac1D\sum_j (x_{ij}-\hat x_{ij})^2/\hat x_{ij}$ with $\hat{\mathbf X} = \mathbf G(\bar{\mathbf x} + \mathbf T_4\mathbf V_4^T)\mathbf H$
    and $\hat x_{ij}$ clipped below at 0.05. Set `rho` (length 6400) and `suspects`: the indices of the 12 largest values of
    `rho` among pixels with at least 40 counts (thick enough for a fair test). *Hint:* `np.argsort`.
    """)
    return


@app.cell
def planted(X_eds, eds, np, plt, ref_pca, ref_weight):
    # The planted truth of the lecture: 12 pixels with ≈ 10 extra Pt counts each (Poisson), stored with the data.
    hidden = eds["planted"]["idx"].astype(int)
    X_pt = X_eds.copy()
    X_pt[hidden] += eds["planted"]["add"]
    Xw_pt, G_pt, H_pt = ref_weight(X_pt)
    pca_pt = ref_pca(Xw_pt)
    _s = pca_pt[3]
    _fig, _ax = plt.subplots(figsize=(5, 3))
    _ax.loglog(np.arange(1, 41), _s[:40] ** 2 / np.sum(_s ** 2), "o-", ms=3)
    _ax.axvline(4.5, color="r", ls="--")
    _ax.set_title("scree of the planted data (weighted)")
    _fig.tight_layout()
    _fig
    return G_pt, H_pt, X_pt, hidden, pca_pt


@app.cell
def ex10(G_pt, H_pt, X_pt, np, pca_pt):
    # ✏️ YOUR TURN
    rho = None
    suspects = None
    return rho, suspects


@app.cell(hide_code=True)
def ex10_check(G_pt, H_pt, X_pt, checklist, hidden, mo, np, pca_pt, plt, rho, suspects):
    if rho is None and suspects is None:
        _out = checklist([], "Fill in `rho` and `suspects`.")
    else:
        _mu, _T, _Vt, _ = pca_pt
        _Xh = np.clip((_mu + _T[:, :4] @ _Vt[:4]) * G_pt * H_pt, 0.05, None)
        _ref = ((X_pt - _Xh) ** 2 / _Xh).mean(1)
        _ok1 = rho is not None and np.allclose(np.asarray(rho, float), _ref, rtol=1e-5)
        _hits = len(set(np.asarray(suspects if suspects is not None else [], int).tolist()) & set(hidden.tolist()))
        _ok2 = suspects is not None and len(suspects) == 12
        _parts = [checklist([(_ok1, "Pearson residual map"), (_ok2, "12 suspects"),
                             (_hits >= 4, f"{_hits} of your 12 suspects are planted Pt pixels (≥ 4 needed; by chance you would expect 0)")])]
        if _ok1:
            _fig, _ax = plt.subplots(1, 2, figsize=(10, 4.2))
            _ax[0].imshow(_ref.reshape(80, 80), cmap="magma", vmin=0.3, vmax=2.2)
            _ax[0].set_title("Pearson residual, K = 4")
            _ax[1].imshow(X_pt.sum(1).reshape(80, 80), cmap="gray")
            _yy, _xx = np.divmod(hidden, 80)
            _ax[1].scatter(_xx, _yy, s=40, facecolors="none", edgecolors="lime")
            _ax[1].set_title("total counts + the truth (green)")
            for _a in _ax:
                _a.axis("off")
            _fig.tight_layout()
            _parts += [_fig, mo.md("The scree plot and the K = 4 maps hide the Pt; the residual map shows it. "
                                   "Also note the real structure in the residual: the thick Ce-rich particle at the top right. "
                                   "A structured residual means *the model misses something*.")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex10_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    _mu, _T, _Vt, _ = pca_pt
    _Xh = np.clip((_mu + _T[:, :4] @ _Vt[:4]) * G_pt * H_pt, 0.05, None)
    rho = ((X_pt - _Xh) ** 2 / _Xh).mean(1)
    _cand = np.flatnonzero(X_pt.sum(1) >= 40)
    suspects = _cand[np.argsort(rho[_cand])[-12:]]
    ```
    """)})
    return


@app.cell(hide_code=True)
def s7_3(mo):
    mo.md(r"""
    ## 7.3 · Overlapping peaks: the condition number

    In MoS₂, S-Kα (2.308 keV) and Mo-Lα (2.293 keV) are 15 eV apart; the detector FWHM there is about 80 eV.
    The least-squares design matrix has two almost parallel columns. Compare κ with the α lines only and with the full
    line families, and the error amplification $1/\sqrt{1-\cos^2\theta}$ of a two-column fit.
    """)
    return


@app.cell
def cond_demo(mo, np):
    _E = np.arange(2.0, 2.75, 0.01)
    _fw = lambda E0: np.sqrt(130 ** 2 + 2.5 * (E0 * 1000 - 5895)) / 1000 / 2.3548
    _g = lambda lines: sum(w * np.exp(-0.5 * ((_E - E0) / _fw(E0)) ** 2) for E0, w in lines)
    _S = {"α only": ([(2.3078, 1)], [(2.2932, 1)]),
          "full family": ([(2.3078, 1), (2.4640, 0.06)], [(2.2932, 1), (2.3948, 0.53), (2.5184, 0.06)])}
    _rows = []
    for _n, (_a, _b) in _S.items():
        _A = np.stack([_g(_a), _g(_b), np.ones_like(_E)], 1)
        _A = _A / np.linalg.norm(_A, axis=0)
        _c = _A[:, 0] @ _A[:, 1]
        _rows.append(f"| {_n} | {_c:.3f} | {np.linalg.cond(_A):.1f} | {1 / np.sqrt(1 - _c ** 2):.1f}× |")
    mo.md("| lines | cos θ (S, Mo) | κ | error amplification |\n|---|---|---|---|\n" + "\n".join(_rows) +
           "\n\nThe extra Mo-Lβ line adds independent information: κ falls. Cures for ill-conditioning: more independent "
           "information, constraints, or ridge regularisation (Week 4).")
    return


@app.cell(hide_code=True)
def summary(mo):
    mo.md(r"""
    ---
    ## Summary

    | concept | formula / code |
    |---|---|
    | similarity, projection | $\cos\theta = \mathbf a^T\mathbf b/(\lVert\mathbf a\rVert\lVert\mathbf b\rVert)$; $c = \hat{\mathbf a}^T\mathbf b$ |
    | least squares = projection | `np.linalg.lstsq(R, y)`; $\mathbf R^T\mathbf r = 0$; structured residual → missing reference |
    | data matrix | `X = cube.reshape(-1, D)`; maps: `X[:, m].sum(1).reshape(ny, nx)` |
    | power-law background | $\log I = \log A - r\log E$; all pixels: `solve(B.T @ B, B.T @ Y)` |
    | SVD | $\mathbf A = \mathbf U\boldsymbol\Sigma\mathbf V^T$; strains $\sigma-1$, rotation $\mathbf U\mathbf V^T$; $\lVert\mathbf X-\mathbf X_k\rVert_F^2 = \sum_{j>k}\sigma_j^2$ |
    | PCA | centre, `svd`, scores `U*s`, loadings `Vt`; signed, variance-ordered |
    | Poisson weighting | $\mathbf X/(\sqrt{g_i}\sqrt{\bar x_j})$ before PCA; un-weight after |
    | drift | derivative-shaped loading, score map = shift map; align first |
    | KL-NMF | multiplicative updates; W, H ≥ 0; Poisson maximum likelihood |
    | ambiguity | $\mathbf{WH} = (\mathbf W\mathbf T^{-1})(\mathbf T\mathbf H)$; pure pixels and constraints narrow the band |
    | rare phases | truncation keeps variance, not importance: read the residual map |
    | conditioning | κ = σ_max/σ_min; overlapping peaks amplify noise |
    """)
    return


@app.cell(hide_code=True)
def refs(mo):
    mo.md(r"""
    ---
    ## Next week and references

    **Next: Unit 4, *Regression, optimisation & honest validation*.** Least squares was a projection today; next week it
    becomes learning, with gradient descent, ridge for large κ, and validation that does not leak (fit the PCA inside the training fold).

    **Data**

    - G. Hollyer, E. Stach, *STEM-Bank-26: an open dataset of analytical STEM images and spectra*, arXiv:2610.36086 (2026),
      CC BY 4.0. CeVO₃/Al₂O₃ EELS + EDS (session CeVO3-Al2O3-Ox-03082023, SI 94); SrFeOₓ/LaAlO₃ HAADF.
    - B. A. Cymes, K. D. Burgess, R. M. Stroud, *Detection of ferric iron in an exsolved lunar pyroxene using EELS*,
      Meteoritics & Planetary Science **58**, 259 (2023), <https://doi.org/10.1111/maps.13941>; data: Zenodo 10.5281/zenodo.7439174, CC BY 4.0.
    - Processing of all data: `03_linear_algebra_spectral/xp/prep/make_data.py` in the course repository.

    **Methods**

    - C. Eckart, G. Young, Psychometrika **1**, 211 (1936). — M. R. Keenan, P. G. Kotula, Surf. Interface Anal. **36**, 203 (2004).
    - D. D. Lee, H. S. Seung, Nature **401**, 788 (1999); Adv. Neural Inf. Process. Syst. **13** (2001). — A. de Juan, R. Tauler, MCR-ALS review (2014).
    - P. Potapov, A. Lubk, *Optimal principal component analysis of STEM XEDS spectrum images*, Adv. Struct. Chem. Imaging (2019).
    - L. Lichtert, J. Verbeeck, *Statistical consequences of applying a PCA noise filter on EELS spectrum images*, Ultramicroscopy (2013).
    """)
    return


if __name__ == "__main__":
    app.run()
