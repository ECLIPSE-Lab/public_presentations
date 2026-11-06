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
    # Unit 4 · Hands-on: regression, optimisation and honest validation on real EM data

    **Data Science for Electron Microscopy** · Prof. Dr. Philipp Pelz · FAU Erlangen-Nürnberg · WS 26/27

    This notebook goes with the Week 4 lecture *Regression, optimisation & honest validation*.
    Every data set is a real, openly licensed measurement, the same data as in the lecture explainers.
    You write a robust lattice fit, gradient descent, momentum and Adam, measure bias and variance on real pixels,
    write ridge and lasso, and find out how much a random split overstates a score.

    | part | content | data |
    |---|---|---|
    | **1 · Loss** | MSE versus Huber (IRLS), the noise model of real residuals | 4D-STEM disks, ion-irradiated Al (Mills et al.) |
    | **2 · Descend** | gradient descent, condition number, feature scaling, a non-convex loss | CeVO₃ pre-edge EELS (STEM-Bank-26); Fe-L (Cymes et al.) |
    | **3 · Step** | momentum and Adam | the same background fit |
    | **4 · Generalise** | polynomial degree, train versus validation, bias² + variance | 30 real pixels |
    | **5 · Regularise** | ridge on collinear Fe standards, lasso selects elements | Fe-L image (Cymes et al.); EDS (STEM-Bank-26) |
    | **6 · Validate** | random versus grouped CV, preprocessing leakage | thickness of ten specimens (STEM-Bank-26) |

    **After this notebook you can:**

    1. Derive MSE, MAE and Huber from a noise model, and fit with iteratively reweighted least squares.
    2. Write gradient descent, and predict its speed from the condition number of the Hessian.
    3. Write momentum and Adam, and explain what each one fixes.
    4. Choose a model complexity with held-out data, and measure bias² and variance.
    5. Write ridge in closed form and lasso by coordinate descent, and read their results critically.
    6. Choose a split that matches the deployment claim, and keep every learned step inside the fold.

    **Time:** about 2 hours; the ⭐ exercises are optional. **Prerequisites:** the Unit 1–3 notebooks.
    The notebook uses only NumPy and Matplotlib, and it runs on a CPU. It downloads about 1.5 MB of data once.

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
            "data_science_for_em/04_regression_validation/xp/data/{}.json")
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
        """Load one data set of the Week 4 lecture: the local repository first, else GitHub."""
        _here = _Path(mo.notebook_dir() or ".")
        _local = _here.parent / "04_regression_validation" / "xp" / "data" / f"{name}.json"
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

    def safe(f, *a, **kw):
        """Call a student function; return (result, error message)."""
        try:
            return f(*a, **kw), None
        except Exception as e:  # noqa: BLE001
            return None, f"`{type(e).__name__}: {e}`"

    return checklist, load, safe


@app.cell(hide_code=True)
def reference(np):
    # Reference implementations. The checks compare your answers with these functions,
    # and later parts use them, so that every part runs even if an exercise is open.
    def design(hk):
        n = len(hk)
        X = np.zeros((2 * n, 6))
        X[0::2, 0] = 1
        X[1::2, 1] = 1
        X[0::2, 2:4] = hk
        X[1::2, 4:6] = hk
        return X

    def ref_lattice(p, hk, delta=None, iters=50):
        X, y = design(hk), p.ravel()
        sol = np.linalg.lstsq(X, y, rcond=None)[0]
        if delta is not None:
            for _ in range(iters):
                r = np.repeat(np.hypot(*(y - X @ sol).reshape(-1, 2).T), 2)
                w = np.minimum(1.0, delta / np.maximum(r, 1e-12))
                sol = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        return sol, (y - X @ sol).reshape(-1, 2)

    def strain_of(sol, Gref):
        """Strain of the fitted lattice against the reference (x right, y up). Returns exx, eyy, exy, rotation (rad)."""
        G = np.array([[sol[2], sol[3]], [sol[4], sol[5]]])
        P = np.diag([1.0, -1.0])
        A = (P @ G @ P) @ np.linalg.inv(P @ Gref @ P)
        F = np.linalg.inv(A).T
        return F[0, 0] - 1, F[1, 1] - 1, (F[0, 1] + F[1, 0]) / 2, (F[1, 0] - F[0, 1]) / 2

    def ref_gd(F, y, eta, steps, w0):
        w, n = np.array(w0, float), len(y)
        W, L = [w.copy()], [np.sum((y - F @ w) ** 2) / (2 * n)]
        for _ in range(steps):
            w = w + eta * F.T @ (y - F @ w) / n
            W.append(w.copy())
            L.append(np.sum((y - F @ w) ** 2) / (2 * n))
        return np.array(W), np.array(L)

    def ref_ridge(R, Y, lam):
        k = R.shape[1]
        return np.linalg.solve(R.T @ R + lam * np.eye(k), R.T @ Y)

    def ref_lasso(X, y, lam, iters=500, b0=None):
        b = np.zeros(X.shape[1]) if b0 is None else b0.copy()
        r = y - X @ b
        for _ in range(iters):
            dmax = 0.0
            for k in range(X.shape[1]):
                g = X[:, k] @ r + b[k]
                nb = max(0.0, g - lam)
                if nb != b[k]:
                    r -= (nb - b[k]) * X[:, k]
                    dmax = max(dmax, abs(nb - b[k]))
                    b[k] = nb
            if dmax < 1e-9:
                break
        return b

    def ref_knn(Xtr, ytr, Xte, k):
        d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        idx = np.argsort(d, 1)[:, :k]
        return ytr[idx].mean(1)

    def r2(y, p):
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

    return design, r2, ref_gd, ref_knn, ref_lasso, ref_lattice, ref_ridge, strain_of


@app.cell
def load_data(load):
    disks = load("disks")       # Bragg-disk positions, 4D-STEM of ion-irradiated Al (Mills et al.)
    pre = load("preedge")       # CeVO3 / Al2O3 EELS, 455-560 eV, 40 x 40 pixels (STEM-Bank-26)
    align = load("align")       # FeO standard and pyroxene Fe-L spectrum (Cymes et al.)
    fel = load("fel")           # Fe-L spectrum image and three standards (Cymes et al.)
    eds = load("eds")           # CeVO3 / Al2O3 EDS spectra and an X-ray line library (STEM-Bank-26, exspy)
    thick = load("thick")       # t/lambda, ADF and plasmon energy of ten specimens (STEM-Bank-26)
    return align, disks, eds, fel, pre, thick


@app.cell(hide_code=True)
def s1(mo):
    mo.md(r"""
    ---
    # Part 1 · Loss: what does "best fit" mean?

    ## 1.1 · Disks of one 4D-STEM pattern

    A 4D-STEM scan records one convergent-beam diffraction pattern per probe position. Here: ion-irradiated Al,
    [001] zone axis (Mills et al., *Acta Mater.* **246**, 118721 (2023); data CC BY 4.0). A disk finder searched every
    predicted disk position and kept the maxima above 5 % of the brightest disk. Weak disks then give **misdetections**.

    The lattice model: every disk with index $(h, k)$ sits at $\mathbf p = \mathbf c + h\,\mathbf g_1 + k\,\mathbf g_2$.
    That is linear in the six unknowns $(c_x, c_y, g_{1x}, g_{2x}, g_{1y}, g_{2y})$: two equations per disk.
    The cell below shows frame 60 with its detected disks.
    """)
    return


@app.cell
def disk_plot(disks, np, plt):
    EX = disks["examples"]
    Gref = np.array(disks["Gref"])
    _e = EX[0]
    _ox, _oy = disks["pat_origin"]
    _b = disks["pat_bin"]
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.imshow(disks["pat"][0], cmap="gray", extent=[_ox, _ox + 96 * _b, _oy + 96 * _b, _oy])
    _p = np.array(_e["p"])
    _ax.scatter(_p[:, 0], _p[:, 1], s=30, c="orange", edgecolors="w")
    for (_x, _y), (_h, _k) in zip(_p, _e["hk"]):
        _ax.text(_x + 8, _y - 8, f"{int(_h)},{int(_k)}", color="w", fontsize=7)
    _ax.set_title(f"frame {int(_e['frame'])}: {len(_p)} detected disks (pixels of the pattern)")
    _fig.tight_layout()
    _fig
    return EX, Gref


@app.cell(hide_code=True)
def s1_2(mo):
    mo.md(r"""
    ## 1.2 · ✏️ Exercise 1: least squares and Huber by IRLS

    Write `fit_lattice(p, hk, delta=None)`. `p` has shape `(n, 2)` (disk positions $x, y$), `hk` has shape `(n, 2)`.
    The design matrix is given: `design(hk)` returns `X` with shape `(2n, 6)`; `y = p.ravel()`.

    1. `delta is None`: plain least squares (MSE), `np.linalg.lstsq(X, y, rcond=None)[0]`.
    2. Otherwise **Huber by iteratively reweighted least squares (IRLS)**, 50 iterations: start from the MSE solution;
       compute for each disk the length of its 2-D residual $|\mathbf r_i|$; weight $w_i=\min(1, \delta/|\mathbf r_i|)$
       (each disk weight appears twice, once for x and once for y); solve the weighted problem
       `lstsq(X * sqrt(w)[:, None], y * sqrt(w))`; repeat.

    Return `(sol, res)`: the six parameters and the residuals with shape `(n, 2)`.
    """)
    return


@app.cell
def ex1(design, np):
    # ✏️ YOUR TURN: replace each `None`.
    def fit_lattice(p, hk, delta=None):
        X, y = design(hk), p.ravel()
        sol = None
        res = None
        return sol, res

    return (fit_lattice,)


@app.cell(hide_code=True)
def ex1_check(EX, Gref, checklist, fit_lattice, mo, np, ref_lattice, safe, strain_of):
    _e, _c = EX[0], EX[-1]
    _p, _hk = np.array(_e["p"]), np.array(_e["hk"])
    _r0, _err0 = safe(fit_lattice, _p, _hk)
    if _err0 is None and (_r0 is None or _r0[0] is None):
        _out = checklist([], "Write `fit_lattice` in the cell above. The check runs automatically.")
    else:
        _ok = []
        _sm, _ = ref_lattice(_p, _hk)
        _sh, _ = ref_lattice(_p, _hk, 1.0)
        _a, _e1 = safe(fit_lattice, _p, _hk)
        _b, _e2 = safe(fit_lattice, _p, _hk, 1.0)
        _okm = _a is not None and _a[0] is not None and np.allclose(_a[0], _sm, atol=1e-6)
        _okh = _b is not None and _b[0] is not None and np.allclose(_b[0], _sh, atol=1e-3)
        _okr = _b is not None and _b[1] is not None and np.shape(_b[1]) == (len(_p), 2)
        # planted truth: a perfect lattice with one disk displaced by 15 px
        _G = np.array([[60.0, -40.0], [40.0, 60.0]])
        _hk2 = np.array([(h, k) for h in range(-2, 3) for k in range(-2, 3)], float)
        _p2 = np.array([200, 200]) + _hk2 @ _G.T
        _p2[3] += [15, 0]
        _c2, _e3 = safe(fit_lattice, _p2, _hk2, 1.0)
        _okp = _c2 is not None and _c2[0] is not None and np.allclose(_c2[0][2:], [60, -40, 40, 60], atol=0.05)
        _items = [(_okm, "MSE solution equals the least-squares reference" + (f" — {_e1}" if _e1 else "")),
                  (_okh, "Huber (δ = 1 px) solution equals the IRLS reference" + (f" — {_e2}" if _e2 else "")),
                  (_okr, "residuals have shape (n, 2)"),
                  (_okp, "planted lattice with one disk 15 px off: Huber recovers g₁, g₂ to 0.05 px" + (f" — {_e3}" if _e3 else ""))]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _rows = []
            for _ex in EX:
                _pp, _hh = np.array(_ex["p"]), np.array(_ex["hk"])
                _s1 = strain_of(fit_lattice(_pp, _hh)[0], Gref)
                _s2 = strain_of(fit_lattice(_pp, _hh, 1.0)[0], Gref)
                _nbad = int((np.hypot(*fit_lattice(_pp, _hh, 1.0)[1].T) > 2).sum())
                _rows.append(f"| {int(_ex['frame'])} | {len(_pp)} | {_nbad} | {100 * _s1[0]:.2f} / {100 * _s1[1]:.2f} | "
                             f"{100 * _s2[0]:.2f} / {100 * _s2[1]:.2f} | {100 * max(abs(_s1[0] - _s2[0]), abs(_s1[1] - _s2[1])):.2f} |")
            _parts.append(mo.md("| frame | disks | misdetected (> 2 px) | MSE εxx / εyy (%) | Huber εxx / εyy (%) | fake strain (%) |\n"
                                "|---|---|---|---|---|---|\n" + "\n".join(_rows) +
                                "\n\nThe noise of the strain map is about 0.17 %. **Think:** why does the last frame (clean) show no difference?"))
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex1_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def fit_lattice(p, hk, delta=None):
        X, y = design(hk), p.ravel()
        sol = np.linalg.lstsq(X, y, rcond=None)[0]
        if delta is not None:
            for _ in range(50):
                r = np.repeat(np.hypot(*(y - X @ sol).reshape(-1, 2).T), 2)
                w = np.minimum(1.0, delta / np.maximum(r, 1e-12))
                sol = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        res = (y - X @ sol).reshape(-1, 2)
        return sol, res
    ```
    With MSE every disk pulls with a force proportional to its residual. With Huber the force is constant
    beyond δ: a misdetection cannot drag the lattice. IRLS turns the robust fit into a sequence of weighted projections.
    """)})
    return


@app.cell(hide_code=True)
def s1_3(mo):
    mo.md(r"""
    ## 1.3 · ✏️ Exercise 2: the noise model of 21 000 real residuals

    `disks["residH"]` holds the residuals (both coordinates) of the Huber lattice fits of all 1 513 usable frames.
    A loss is the negative log-likelihood of a noise model; so look at the noise first.

    1. Write `robust_sigma(r)`: $1.4826\cdot\mathrm{median}(|r-\mathrm{median}(r)|)$. For Gaussian data it equals σ,
       but outliers do not change it.
    2. Write `tail_fraction(r, k)`: the fraction of residuals with $|r - \mathrm{median}(r)| > k\,\sigma$ (σ from `robust_sigma`).
    """)
    return


@app.cell
def ex2(np):
    # ✏️ YOUR TURN: replace each `None`.
    def robust_sigma(r):
        return None

    def tail_fraction(r, k):
        return None

    return robust_sigma, tail_fraction


@app.cell(hide_code=True)
def ex2_check(checklist, disks, mo, np, plt, robust_sigma, safe, tail_fraction):
    import math as _math
    _R = disks["residH"].ravel()
    _s, _e1 = safe(robust_sigma, _R)
    if _s is None and _e1 is None:
        _out = checklist([], "Write `robust_sigma` and `tail_fraction` in the cell above.")
    else:
        _med = np.median(_R)
        _ref_s = 1.4826 * np.median(np.abs(_R - _med))
        _t3, _e2 = safe(tail_fraction, _R, 3)
        _ref_t3 = np.mean(np.abs(_R - _med) > 3 * _ref_s)
        _rng = np.random.default_rng(0)
        _g = _rng.normal(0, 2, 200000)
        _g[:2000] = 50
        _sg, _ = safe(robust_sigma, _g)
        _items = [(_s is not None and np.isclose(_s, _ref_s, rtol=1e-6), f"robust σ of the disk residuals = {_s if _e1 is None else _e1} (expected {_ref_s:.3f} px)"),
                  (_t3 is not None and np.isclose(_t3, _ref_t3, rtol=1e-6), f"fraction beyond 3σ = {_t3 if _e2 is None else _e2} (expected {_ref_t3:.4f})"),
                  (_sg is not None and abs(_sg - 2) < 0.03, "planted: Gaussian σ = 2 with 1 % outliers at 50: robust σ ≈ 2")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.8))
            _x = np.linspace(-5, 5, 401)
            for _a, _log in zip(_ax, (False, True)):
                _a.hist(_R, bins=np.arange(-5, 5.05, 0.1), density=True, color="0.6")
                _a.plot(_x, np.exp(-0.5 * (_x / _s) ** 2) / (_s * np.sqrt(2 * np.pi)), "b", lw=2, label="Gaussian (robust σ)")
                if _log:
                    _a.set_yscale("log")
                    _a.set_ylim(1e-5, 3)
                _a.set_xlabel("residual (px)")
                _a.legend()
            _ax[0].set_title("linear axis: looks Gaussian")
            _ax[1].set_title("log axis: heavy tails")
            _fig.tight_layout()
            _rows = "\n".join(f"| {k} | {100 * tail_fraction(_R, k):.2f} % | {100 * _math.erfc(k / _math.sqrt(2)):.2g} % |" for k in (2, 3, 5))
            _parts += [_fig, mo.md("| beyond | data | Gaussian |\n|---|---|---|\n" + _rows +
                                   "\n\n**Think:** which loss is the negative log-likelihood of 'Gaussian core + rare outliers'? "
                                   "Why does a plain MSE fit make the residuals wider (try `disks['resid']`)?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex2_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def robust_sigma(r):
        return 1.4826 * np.median(np.abs(r - np.median(r)))

    def tail_fraction(r, k):
        return np.mean(np.abs(r - np.median(r)) > k * robust_sigma(r))
    ```
    The core is Gaussian with σ ≈ 0.28 px, but 5 % of the residuals lie beyond 5σ (Gaussian: 6·10⁻⁵ %).
    A Gaussian core with heavy tails is the case for the Huber loss.
    """)})
    return


@app.cell(hide_code=True)
def s2(mo):
    mo.md(r"""
    ---
    # Part 2 · Descend: how do we find the minimum?

    ## 2.1 · The background fit as an optimisation problem

    Below an EELS edge the background is a power law, $\log I = w_0 + w_1\,f(\log E)$ (Week 3).
    Here: the mean spectrum of 30 pixels of the CeVOₓ region, 470–508 eV. The loss is
    $L(\mathbf w) = \frac{1}{2n}\lVert\mathbf y - \mathbf F\mathbf w\rVert^2$, with $\mathbf F = [\mathbf 1, \mathbf f]$.
    The feature $f$ can be the raw $\log E$, the centred $\log E - \overline{\log E}$, or the standardised value.
    """)
    return


@app.cell
def preedge_points(np, pre):
    E_pre = pre["E"]
    _C = pre["cube"] * pre["scale"]
    PIX30 = _C[26:32, 33:38].reshape(-1, _C.shape[-1])        # 30 pixels of the CeVOx region
    _m = (E_pre >= 470) & (E_pre <= 508)
    x_bg = np.log(E_pre[_m])
    y_bg = np.log(PIX30.mean(0)[_m])
    FEAT = {"raw": x_bg, "centred": x_bg - x_bg.mean(), "standardised": (x_bg - x_bg.mean()) / x_bg.std()}
    DESIGNS = {k: np.stack([np.ones_like(v), v], 1) for k, v in FEAT.items()}
    return DESIGNS, E_pre, PIX30, y_bg


@app.cell(hide_code=True)
def s2_2(mo):
    mo.md(r"""
    ## 2.2 · ✏️ Exercise 3: gradient descent and its speed limit

    1. Write `gd(F, y, eta, steps, w0)`: gradient descent on $L(\mathbf w)$. The gradient is
       $\nabla L = -\frac1n\mathbf F^T(\mathbf y - \mathbf F\mathbf w)$. Return `(W, L)`: all iterates (shape `(steps+1, 2)`,
       the start included) and the loss of each iterate.
    2. Write `hessian_info(F)`: the Hessian $\mathbf H = \mathbf F^T\mathbf F/n$; return `(kappa, eta_max)` with
       $\kappa = \lambda_{\max}/\lambda_{\min}$ and $\eta_{\max} = 2/\lambda_{\max}$ (use `np.linalg.eigvalsh`).
    """)
    return


@app.cell
def ex3(np):
    # ✏️ YOUR TURN: replace each `None`.
    def gd(F, y, eta, steps, w0):
        w, n = np.array(w0, float), len(y)
        W, L = None, None
        return W, L

    def hessian_info(F):
        kappa = None
        eta_max = None
        return kappa, eta_max

    return gd, hessian_info


@app.cell(hide_code=True)
def ex3_check(DESIGNS, checklist, gd, hessian_info, mo, np, plt, ref_gd, safe, y_bg):
    _F = DESIGNS["standardised"]
    _res, _e1 = safe(gd, _F, y_bg, 0.5, 20, [0.0, 0.0])
    if _e1 is None and (_res is None or _res[0] is None):
        _out = checklist([], "Write `gd` and `hessian_info` in the cell above.")
    else:
        _Wr, _Lr = ref_gd(_F, y_bg, 0.5, 20, [0.0, 0.0])
        _okg = _res is not None and _res[0] is not None and np.shape(_res[0]) == (21, 2) and np.allclose(_res[0], _Wr, atol=1e-9) and np.allclose(_res[1], _Lr, atol=1e-12)
        _hi, _e2 = safe(hessian_info, DESIGNS["raw"])
        _ev = np.linalg.eigvalsh(DESIGNS["raw"].T @ DESIGNS["raw"] / len(y_bg))
        _okh = _hi is not None and _hi[0] is not None and np.isclose(_hi[0], _ev[-1] / _ev[0], rtol=1e-3) and np.isclose(_hi[1], 2 / _ev[-1], rtol=1e-6)
        _items = [(_okg, "gd: 20 steps with η = 0.5 equal the reference iterates and losses" + (f" — {_e1}" if _e1 else "")),
                  (_okh, "hessian_info(raw): κ and η_max are correct" + (f" — {_e2}" if _e2 else ""))]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _fig, _ax = plt.subplots(figsize=(8, 3.8))
            _rows = []
            for _k, _Fk in DESIGNS.items():
                _kap, _em = hessian_info(_Fk)
                _wh = np.linalg.lstsq(_Fk, y_bg, rcond=None)[0]
                _Lmin = np.sum((y_bg - _Fk @ _wh) ** 2) / (2 * len(y_bg))
                _w0 = np.linalg.lstsq(_Fk, np.full_like(y_bg, y_bg.mean() + 0.15), rcond=None)[0]    # a flat line 0.15 above the mean
                _W, _L = gd(_Fk, y_bg, 0.9 * _em, 3000, _w0)
                _d = _L - _Lmin
                _n = int(np.argmax(_d < 1e-4 * _d[0])) if np.any(_d < 1e-4 * _d[0]) else None
                _ax.loglog(np.arange(1, len(_d) + 1), np.maximum(_d, 1e-12), label=_k)
                _rows.append(f"| {_k} | {_kap:.3g} | {_n if _n is not None else '> 3000'} |")
            _ax.set_xlabel("iteration")
            _ax.set_ylabel("L − L_min")
            _ax.legend()
            _ax.set_title("gradient descent, η = 0.9 · 2/λ_max, same start line")
            _fig.tight_layout()
            _parts += [_fig, mo.md("| features | κ | steps to 10⁻⁴ of the start error |\n|---|---|---|\n" + "\n".join(_rows) +
                                   "\n\n**Think:** the same data, the same model, the same minimum. Why does only the parametrisation change the run time? "
                                   "What does `StandardScaler` do to κ?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex3_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def gd(F, y, eta, steps, w0):
        w, n = np.array(w0, float), len(y)
        W, L = [w.copy()], [np.sum((y - F @ w) ** 2) / (2 * n)]
        for _ in range(steps):
            w = w + eta * F.T @ (y - F @ w) / n
            W.append(w.copy()); L.append(np.sum((y - F @ w) ** 2) / (2 * n))
        return np.array(W), np.array(L)

    def hessian_info(F):
        ev = np.linalg.eigvalsh(F.T @ F / len(F))
        return ev[-1] / ev[0], 2 / ev[-1]
    ```
    Per step, the error along each eigenvector of H shrinks by |1 − ηλ|. η is limited by λ_max, so the flattest
    direction shrinks only by about 2/κ per step: about κ steps. Raw log E: κ ≈ 3·10⁶; standardised: κ = 1.
    """)})
    return


@app.cell(hide_code=True)
def s2_3(mo):
    mo.md(r"""
    ## 2.3 · A non-convex loss: aligning two spectra

    The energy shift $s$ that aligns the pyroxene Fe-L spectrum with the FeO standard minimises
    $L(s) = \sum_E (y(E) - a\,\mathrm{ref}(E - s))^2$, with the best scale $a$ for every $s$.
    A shift is a non-linear parameter: the loss has side minima. Move the start and watch gradient descent.
    """)
    return


@app.cell
def align_controls(mo):
    s_start = mo.ui.slider(-19, 19, step=0.5, value=-9, label="start s₀ (eV)", show_value=True)
    s_start
    return (s_start,)


@app.cell
def align_plot(align, np, plt, s_start):
    _E, _ref, _y = align["E"], align["ref"], align["y"]

    def _loss(s):
        _r = np.interp(_E - s, _E, _ref, left=0, right=0)
        _a = (_r @ _y) / (_r @ _r + 1e-12)
        return np.sum((_y - _a * _r) ** 2)

    _S = np.arange(-20, 20.01, 0.1)
    _LS = np.array([_loss(v) for v in _S])
    _s = float(s_start.value)
    _path = [_s]
    for _ in range(400):
        _g = (_loss(_s + 0.02) - _loss(_s - 0.02)) / 0.04
        _s = float(np.clip(_s - 0.25 * _g / (_LS.max() / 20), -20, 20))
        _path.append(_s)
    _fig, _ax = plt.subplots(figsize=(9, 3.4))
    _ax.plot(_S, _LS, "k", lw=2)
    _ax.plot(_path, [_loss(v) for v in _path], "r.-", ms=3)
    _ax.plot(_path[-1], _loss(_path[-1]), "ro", ms=9)
    _ax.set_xlabel("energy shift s (eV)")
    _ax.set_ylabel("L(s)")
    _ax.set_title(f"start {s_start.value} eV → GD ends at {_path[-1]:.2f} eV; global minimum {_S[np.argmin(_LS)]:.1f} eV")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def s3(mo):
    mo.md(r"""
    ---
    # Part 3 · Step: momentum and Adam

    Use the **centred** feature: κ ≈ 2 000. Plain gradient descent crawls along the flat direction.

    ## 3.1 · ✏️ Exercise 4: two better steps

    Write both functions; each returns the array of losses $L(\mathbf w_t)$ for $t = 0 \dots$ `steps`.
    `grad(w)` returns the full gradient (given in the check; write your own for testing).

    - `momentum(grad, loss, w0, eta, beta, steps)`: $\mathbf v \leftarrow \beta\mathbf v + \nabla L$, $\;\mathbf w \leftarrow \mathbf w - \eta\mathbf v$, with $\mathbf v_0 = 0$.
    - `adam(grad, loss, w0, alpha, steps)`: $\mathbf m \leftarrow 0.9\mathbf m + 0.1\mathbf g$, $\;\mathbf s \leftarrow 0.999\mathbf s + 0.001\mathbf g^2$,
      bias corrections $\hat{\mathbf m} = \mathbf m/(1-0.9^t)$, $\hat{\mathbf s} = \mathbf s/(1-0.999^t)$ (t = 1, 2, …),
      $\mathbf w \leftarrow \mathbf w - \alpha\,\hat{\mathbf m}/(\sqrt{\hat{\mathbf s}} + 10^{-8})$.
    """)
    return


@app.cell
def ex4(np):
    # ✏️ YOUR TURN: replace each `None`.
    def momentum(grad, loss, w0, eta, beta, steps):
        w = np.array(w0, float)
        v = np.zeros_like(w)
        L = None
        return L

    def adam(grad, loss, w0, alpha, steps):
        w = np.array(w0, float)
        m, s = np.zeros_like(w), np.zeros_like(w)
        L = None
        return L

    return adam, momentum


@app.cell(hide_code=True)
def ex4_check(DESIGNS, adam, checklist, mo, momentum, np, plt, ref_gd, safe, y_bg):
    _F = DESIGNS["centred"]
    _n = len(y_bg)
    _grad = lambda w: -_F.T @ (y_bg - _F @ w) / _n
    _loss = lambda w: np.sum((y_bg - _F @ w) ** 2) / (2 * _n)
    _ev = np.linalg.eigvalsh(_F.T @ _F / _n)
    _eta = 0.9 * 2 / _ev[-1]
    _w0 = np.array([y_bg.mean() + 0.15, 0.0])
    _Lm, _e1 = safe(momentum, _grad, _loss, _w0, _eta, 0.9, 3000)
    _La, _e2 = safe(adam, _grad, _loss, _w0, 0.02, 3000)
    if _e1 is None and _e2 is None and _Lm is None and _La is None:
        _out = checklist([], "Write `momentum` and `adam` in the cell above.")
    else:
        _wh = np.linalg.lstsq(_F, y_bg, rcond=None)[0]
        _Lmin = _loss(_wh)
        _steps = lambda L: int(np.argmax(np.asarray(L) - _Lmin < 1e-4 * (L[0] - _Lmin))) if np.any(np.asarray(L) - _Lmin < 1e-4 * (L[0] - _Lmin)) else None
        _okm = _Lm is not None and len(_Lm) == 3001 and _steps(_Lm) is not None and _steps(_Lm) < 600
        _oka = _La is not None and len(_La) == 3001 and _steps(_La) is not None and _steps(_La) < 600
        _items = [(_okm, f"momentum converges (to 10⁻⁴ of the start error) in fewer than 600 steps: {_steps(_Lm) if _Lm is not None else _e1}"),
                  (_oka, f"Adam converges in fewer than 600 steps: {_steps(_La) if _La is not None else _e2}")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _, _Lg = ref_gd(_F, y_bg, _eta, 3000, _w0)
            _fig, _ax = plt.subplots(figsize=(8, 3.6))
            for _L, _lab in [(_Lg, "GD"), (_Lm, "momentum"), (_La, "Adam")]:
                _ax.loglog(np.arange(1, 3002), np.maximum(np.asarray(_L) - _Lmin, 1e-14), label=_lab)
            _ax.set_xlabel("iteration")
            _ax.set_ylabel("L − L_min")
            _ax.legend()
            _fig.tight_layout()
            _parts += [_fig, mo.md(f"GD: {_steps(_Lg) or '> 3000'} steps. **Think:** Adam rescales each parameter separately. "
                                   "Why does that help here (centred features: the valley is aligned with the axes), and why would it help much less "
                                   "with the raw feature (a tilted valley)? Try it.")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex4_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def momentum(grad, loss, w0, eta, beta, steps):
        w = np.array(w0, float); v = np.zeros_like(w); L = [loss(w)]
        for _ in range(steps):
            v = beta * v + grad(w); w = w - eta * v; L.append(loss(w))
        return np.array(L)

    def adam(grad, loss, w0, alpha, steps):
        w = np.array(w0, float); m, s = np.zeros_like(w), np.zeros_like(w); L = [loss(w)]
        for t in range(1, steps + 1):
            g = grad(w)
            m = 0.9 * m + 0.1 * g; s = 0.999 * s + 0.001 * g ** 2
            mh, sh = m / (1 - 0.9 ** t), s / (1 - 0.999 ** t)
            w = w - alpha * mh / (np.sqrt(sh) + 1e-8); L.append(loss(w))
        return np.array(L)
    ```
    Momentum adds up consistent steps (up to 1/(1−β) = 10×) and cancels oscillating ones. Adam divides each
    gradient component by its own running RMS, so the flat and the steep parameter take similar steps.
    """)})
    return


@app.cell(hide_code=True)
def s4(mo):
    mo.md(r"""
    ---
    # Part 4 · Generalise: why does the best fit predict badly?

    ## 4.1 · ✏️ Exercise 5: polynomial degree, train versus validation, bias² + variance

    Fit $\log I$ as a polynomial of degree $d$ in the standardised $\log E$, on the train channels 470–496 eV.
    Validate on 497–508 eV (next to the edge, where the background is extrapolated).

    1. Write `poly_fit(u, y, deg)`: the least-squares coefficients $c_0 \dots c_d$ of $\sum_j c_j u^j$
       (use `np.vander(u, deg + 1, increasing=True)`).
    2. Write `bias_variance(P, truth)`: `P` has shape `(n_pixels, n_channels)`, one prediction per pixel;
       return `(bias2, var)` averaged over the channels: $\mathrm{bias}^2 = \overline{(\bar P - \mathrm{truth})^2}$,
       $\mathrm{var} = \overline{\mathrm{Var}_{\text{pixels}}(P)}$.
    """)
    return


@app.cell
def ex5(np):
    # ✏️ YOUR TURN: replace each `None`.
    def poly_fit(u, y, deg):
        return None

    def bias_variance(P, truth):
        bias2 = None
        var = None
        return bias2, var

    return bias_variance, poly_fit


@app.cell(hide_code=True)
def ex5_check(E_pre, PIX30, bias_variance, checklist, mo, np, plt, poly_fit, safe):
    _tr = (E_pre >= 470) & (E_pre <= 496)
    _va = (E_pre > 496) & (E_pre <= 508)
    _x = np.log(E_pre)
    _u = (_x - _x[_tr].mean()) / _x[_tr].std()
    _c, _e1 = safe(poly_fit, _u[_tr], np.log(PIX30[7][_tr]), 2)
    if _c is None and _e1 is None:
        _out = checklist([], "Write `poly_fit` and `bias_variance` in the cell above.")
    else:
        _V = np.vander(_u[_tr], 3, increasing=True)
        _ref = np.linalg.lstsq(_V, np.log(PIX30[7][_tr]), rcond=None)[0]
        _okp = _c is not None and np.allclose(_c, _ref, atol=1e-8)
        _P = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        _bv, _e2 = safe(bias_variance, _P, np.array([2.0, 5.0]))
        _okb = _bv is not None and _bv[0] is not None and np.allclose(_bv, [(1 + 1) / 2, 8 / 3], atol=1e-9)
        _items = [(_okp, "poly_fit (degree 2, one pixel) equals the least-squares reference" + (f" — {_e1}" if _e1 else "")),
                  (_okb, "bias_variance on a small planted example: bias² = 1, var = 8/3" + (f" — {_e2}" if _e2 else ""))]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _ev = lambda c, uu: np.vander(uu, len(c), increasing=True) @ c
            _truth = np.log(PIX30.mean(0))[_va]
            _tre, _vae, _b2, _vr = [], [], [], []
            for _d in range(9):
                _cc = poly_fit(_u[_tr], np.log(PIX30[7][_tr]), _d)
                _tre.append(np.mean((np.log(PIX30[7][_tr]) - _ev(_cc, _u[_tr])) ** 2))
                _vae.append(np.mean((np.log(PIX30[7][_va]) - _ev(_cc, _u[_va])) ** 2))
                _Pd = np.array([_ev(poly_fit(_u[_tr], np.log(np.maximum(_p[_tr], 1e-3)), _d), _u[_va]) for _p in PIX30])
                _b, _v = bias_variance(_Pd, _truth)
                _b2.append(_b)
                _vr.append(_v)
            _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.8))
            _ax[0].semilogy(range(9), _tre, "o-", label="train")
            _ax[0].semilogy(range(9), _vae, "o-", label="validation")
            _ax[0].set_xlabel("degree d")
            _ax[0].set_title("one pixel")
            _ax[0].legend()
            _ax[1].semilogy(range(9), _b2, "o-", label="bias²")
            _ax[1].semilogy(range(9), _vr, "o-", label="variance")
            _ax[1].semilogy(range(9), np.array(_b2) + np.array(_vr), "k--", label="sum")
            _ax[1].set_xlabel("degree d")
            _ax[1].set_title("30 real pixels")
            _ax[1].legend()
            _fig.tight_layout()
            _parts += [_fig, mo.md(f"Best validation degree: **{int(np.argmin(_vae))}**; best bias² + variance: **{int(np.argmin(np.array(_b2) + np.array(_vr)))}**. "
                                   "**Think:** why can the training error never choose the degree? Which physical model is degree 1?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex5_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def poly_fit(u, y, deg):
        return np.linalg.lstsq(np.vander(u, deg + 1, increasing=True), y, rcond=None)[0]

    def bias_variance(P, truth):
        mean = P.mean(0)
        return np.mean((mean - truth) ** 2), np.mean(P.var(0))
    ```
    Degree 1 is the power law A·E⁻ʳ. Higher degrees lower the training error, but the variance at the
    validation channels explodes: overfitting is variance.
    """)})
    return


@app.cell(hide_code=True)
def s5(mo):
    mo.md(r"""
    ---
    # Part 5 · Regularise: ridge and lasso

    ## 5.1 · ✏️ Exercise 6: ridge on collinear Fe standards

    `fel["cube"]` is the Fe-L₂,₃ spectrum image of a lunar pyroxene rim (54 × 40 pixels, 3 × 3 binned,
    background removed, 704–724 eV). The three standards Fe⁰, Fe²⁺ and Fe³⁺ are the columns of $\mathbf R$.
    Fe⁰ and Fe²⁺ have nearly the same shape.

    Write `ridge(R, Y, lam)`: $\hat{\mathbf W} = (\mathbf R^T\mathbf R + \lambda\mathbf I)^{-1}\mathbf R^T\mathbf Y$ for all pixels at once
    (`Y` has shape `(D, N)`, the result `(3, N)`). Use `np.linalg.solve`, not `inv`.
    """)
    return


@app.cell
def fel_data(fel, np):
    R_fe = np.stack([fel["refs"][k] for k in ("Fe0", "Fe2", "Fe3")], 1)          # (D, 3)
    Y_fe = (fel["cube"] * fel["scale"]).reshape(-1, R_fe.shape[0]).T             # (D, N)
    good_fe = Y_fe.sum(0) > np.median(Y_fe.sum(0))                              # the brightest half of the pixels
    return R_fe, Y_fe, good_fe


@app.cell
def ex6(np):
    # ✏️ YOUR TURN: replace `None`.
    def ridge(R, Y, lam):
        return None

    return (ridge,)


@app.cell(hide_code=True)
def ex6_check(R_fe, Y_fe, checklist, good_fe, mo, np, plt, ref_ridge, ridge, safe):
    _W, _e1 = safe(ridge, R_fe, Y_fe[:, :50], 1e-4)
    if _W is None and _e1 is None:
        _out = checklist([], "Write `ridge` in the cell above.")
    else:
        _ok = _W is not None and np.shape(_W) == (3, 50) and np.allclose(_W, ref_ridge(R_fe, Y_fe[:, :50], 1e-4), rtol=1e-8)
        _W0, _ = safe(ridge, R_fe, Y_fe[:, :50], 0.0)
        _ok0 = _W0 is not None and np.allclose(_W0, np.linalg.lstsq(R_fe, Y_fe[:, :50], rcond=None)[0], rtol=1e-6)
        _items = [(_ok, "ridge equals the closed-form reference" + (f" — {_e1}" if _e1 else "")),
                  (_ok0, "λ = 0 gives ordinary least squares")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _tr = np.trace(R_fe.T @ R_fe) / 3
            _rows, _fig, _ax = [], *plt.subplots(1, 2, figsize=(11, 4.2))
            for _i, _rel in enumerate([0, 1e-3, 1e-2, 3e-2, 1e-1]):
                _Wl = ridge(R_fe, Y_fe, _rel * _tr)[:, good_fe]
                _res = np.linalg.norm(Y_fe[:, good_fe] - R_fe @ _Wl) / np.linalg.norm(Y_fe[:, good_fe])
                _rows.append(f"| {_rel:g} | {np.corrcoef(_Wl[0], _Wl[1])[0, 1]:.2f} | {_Wl[0].std():.0f} | {_Wl[0].mean():.0f} | {_res:.4f} |")
                if _rel in (0, 3e-2):
                    _ax[0 if _rel == 0 else 1].scatter(_Wl[0], _Wl[1], s=3, alpha=0.4)
                    _ax[0 if _rel == 0 else 1].set_title(f"λ = {_rel:g} · tr(RᵀR)/3")
            for _a in _ax:
                _a.set_xlabel("weight of Fe⁰")
                _a.set_ylabel("weight of Fe²⁺")
                _a.set_xlim(-30000, 50000)
                _a.set_ylim(-20000, 50000)
            _fig.tight_layout()
            _parts += [_fig, mo.md("| λ (× tr/3) | corr(Fe⁰, Fe²⁺) | spread of Fe⁰ | mean Fe⁰ | fit residual |\n|---|---|---|---|---|\n" + "\n".join(_rows) +
                                   f"\n\nκ(R) = {np.linalg.cond(R_fe):.1f}. **Think:** the fit residual barely changes, but the mean Fe⁰ weight moves. "
                                   "Where does that change come from: the data or the prior? Which quantity would you report?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex6_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def ridge(R, Y, lam):
        return np.linalg.solve(R.T @ R + lam * np.eye(R.shape[1]), R.T @ Y)
    ```
    λI lifts the small eigenvalue of RᵀR (the Fe⁰ − Fe²⁺ direction): the spread falls, the fit hardly changes.
    The shift of the mean weights is bias from the prior "small weights". The data fix only the sum Fe⁰ + Fe²⁺.
    """)})
    return


@app.cell(hide_code=True)
def s5_2(mo):
    mo.md(r"""
    ## 5.2 · ✏️ Exercise 7 ⭐: lasso selects the elements of an EDS spectrum

    The model: spectrum = Σ wₖ · (line pattern of element k) + smooth background, wₖ ≥ 0, for 78 elements.
    The cell below prepares `X_l` and `y_l` for you: Poisson weights, the background projected out, every column
    standardised to unit length.

    Write `lasso_nn(X, y, lam, iters=500)`: **non-negative lasso by coordinate descent**. Start with $\mathbf b = 0$
    and the residual $\mathbf r = \mathbf y$. For each column $k$: $g = \mathbf x_k^T\mathbf r + b_k$; new $b_k = \max(0, g - \lambda)$;
    update $\mathbf r$ by the change. Repeat until no weight changes by more than 10⁻⁹.
    """)
    return


@app.cell
def eds_prep(eds, np):
    names_l = list(eds["names"])

    def prep_lasso(y):
        E = eds["E"]
        w = 1 / np.sqrt(np.maximum(y, 1))
        B = np.stack([np.ones_like(E), E, E ** 2, np.exp(-E)], 1) * w[:, None]
        Q, _ = np.linalg.qr(B)
        P = lambda v: v - Q @ (Q.T @ v)
        X = P(eds["lib"].T * w[:, None])
        X = X / np.linalg.norm(X, axis=0)
        return X, P(y * w)

    X_l, y_l = prep_lasso(eds["total"])
    return X_l, names_l, prep_lasso, y_l


@app.cell
def ex7(np):
    # ✏️ YOUR TURN (⭐): replace `None`.
    def lasso_nn(X, y, lam, iters=500):
        b = np.zeros(X.shape[1])
        r = y.copy()
        return None

    return (lasso_nn,)


@app.cell(hide_code=True)
def ex7_check(X_l, checklist, eds, lasso_nn, mo, names_l, np, prep_lasso, ref_lasso, safe, y_l):
    _lmax = float(np.max(X_l.T @ y_l))
    _b, _e1 = safe(lasso_nn, X_l, y_l, 0.3 * _lmax)
    if _b is None and _e1 is None:
        _out = checklist([], "Write `lasso_nn` in the cell above (optional).")
    else:
        _ref = ref_lasso(X_l, y_l, 0.3 * _lmax)
        _ok = _b is not None and np.shape(_b) == (len(names_l),) and np.allclose(_b, _ref, atol=1e-4 * _lmax)
        _okn = _b is not None and np.all(np.asarray(_b) >= 0)
        _items = [(_ok, "λ = 0.3 λ_max: weights equal the coordinate-descent reference" + (f" — {_e1}" if _e1 else "")),
                  (_okn, "all weights ≥ 0")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _lines = []
            for _key in ("total", "small"):
                _X, _y = prep_lasso(eds[_key])
                _lm = float(np.max(_X.T @ _y))
                _order = []
                for _f in np.logspace(0, -2.5, 30):
                    for _k in np.flatnonzero(lasso_nn(_X, _y, _f * _lm) > 0):
                        if names_l[_k] not in _order:
                            _order.append(names_l[_k])
                _lines.append(f"- **{_key}** spectrum, entry order: " + " → ".join(_order[:9]))
            _parts.append(mo.md("\n".join(_lines) + "\n\nThe sample is CeVO₃ on Al₂O₃ (O, Al, V, Ce); C film and Cu grid are expected extras. "
                                "**Think:** Br Lα (1.480 keV) lies 6 eV from Al Kα (1.487 keV). What does lasso do with two almost collinear columns? "
                                "The small spectrum comes from 5 × 5 pixels on a CeVOₓ particle: why does Ce enter before Al there, but not in the total spectrum?"))
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex7_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def lasso_nn(X, y, lam, iters=500):
        b = np.zeros(X.shape[1]); r = y.copy()
        for _ in range(iters):
            dmax = 0.0
            for k in range(X.shape[1]):
                g = X[:, k] @ r + b[k]          # columns have unit length
                nb = max(0.0, g - lam)
                if nb != b[k]:
                    r -= (nb - b[k]) * X[:, k]; dmax = max(dmax, abs(nb - b[k])); b[k] = nb
            if dmax < 1e-9:
                break
        return b
    ```
    The soft threshold max(0, g − λ) sets weights exactly to zero: lasso selects. For a collinear pair it keeps one
    or both arbitrarily; physics must decide.
    """)})
    return


@app.cell(hide_code=True)
def s6(mo):
    mo.md(r"""
    ---
    # Part 6 · Validate: is the score honest?

    ## 6.1 · Ten specimens, one question

    Ten spectrum images from ten sessions of STEM-Bank-26, 400 pixels each. Target: the thickness $t/\lambda$
    (log-ratio method on the low-loss spectrum). Features: the ADF intensity (detector units) and the plasmon energy $E_p$.
    Inside one specimen, ADF predicts $t/\lambda$ well. Across specimens?
    """)
    return


@app.cell
def thick_data(np, plt, thick):
    g_t = thick["g"].astype(int)
    X_t = np.stack([thick["adf"] / 1e6, thick["ep"]], 1)
    y_t = thick["t"]
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    for _g in range(10):
        _m = g_t == _g
        _ax.scatter(X_t[_m, 0], y_t[_m], s=3, label=thick["specs"][_g]["name"])
    _ax.set_xlabel("ADF intensity (×10⁶ detector units)")
    _ax.set_ylabel("t/λ")
    _ax.legend(fontsize=6, markerscale=3)
    _fig.tight_layout()
    _fig
    return X_t, g_t, y_t


@app.cell(hide_code=True)
def s6_2(mo):
    mo.md(r"""
    ## 6.2 · ✏️ Exercise 8: k-nearest neighbours and a cross-validation that keeps the scaler inside

    1. Write `knn_predict(Xtr, ytr, Xte, k)`: for every test row, the mean target of the $k$ nearest training rows
       (Euclidean distance).
    2. Write `cv_predict(X, y, folds, k)`: for each fold $f$, standardise with the mean and std of the **training rows only**,
       predict the test rows with `knn_predict`, and return all out-of-fold predictions.

    The check then compares a random split of the pixels with `GroupKFold` by specimen.
    """)
    return


@app.cell
def ex8(np):
    # ✏️ YOUR TURN: replace each `None`.
    def knn_predict(Xtr, ytr, Xte, k):
        return None

    def cv_predict(X, y, folds, k):
        pred = np.zeros_like(y)
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            pred[te] = None
        return pred

    return cv_predict, knn_predict


@app.cell(hide_code=True)
def ex8_check(X_t, checklist, cv_predict, g_t, knn_predict, mo, np, r2, ref_knn, safe, y_t):
    _rng = np.random.default_rng(1)
    _Xa, _ya, _Xb = _rng.normal(size=(50, 2)), _rng.normal(size=50), _rng.normal(size=(7, 2))
    _p, _e1 = safe(knn_predict, _Xa, _ya, _Xb, 3)
    if _p is None and _e1 is None:
        _out = checklist([], "Write `knn_predict` and `cv_predict` in the cell above.")
    else:
        _okk = _p is not None and np.allclose(_p, ref_knn(_Xa, _ya, _Xb, 3))
        _f_rand = _rng.permutation(len(y_t)) % 5
        _f_grp = g_t % 5
        _pr, _e2 = safe(cv_predict, X_t, y_t, _f_rand, 5)
        _okc = False
        if _pr is not None and _pr is not None and np.all(np.isfinite(_pr)):
            _ref = np.zeros_like(y_t)
            for _f in range(5):
                _tr, _te = _f_rand != _f, _f_rand == _f
                _mu, _sd = X_t[_tr].mean(0), X_t[_tr].std(0)
                _ref[_te] = ref_knn((X_t[_tr] - _mu) / _sd, y_t[_tr], (X_t[_te] - _mu) / _sd, 5)
            _okc = np.allclose(_pr, _ref, atol=1e-9)
        _items = [(_okk, "knn_predict equals the reference" + (f" — {_e1}" if _e1 else "")),
                  (_okc, "cv_predict (scaler fitted on the training rows) equals the reference" + (f" — {_e2}" if _e2 else ""))]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _rows = []
            for _k in (1, 5, 25):
                _rr = r2(y_t, cv_predict(X_t, y_t, _f_rand, _k))
                _rg = r2(y_t, cv_predict(X_t, y_t, _f_grp, _k))
                _rows.append(f"| {_k} | {_rr:.2f} | {_rg:.2f} |")
            _parts.append(mo.md("| k | R², random pixels | R², GroupKFold by specimen |\n|---|---|---|\n" + "\n".join(_rows) +
                                "\n\n**Think:** what does the random split measure? Write the deployment claim that each column supports. "
                                "What would you need to make the model work on a new specimen?"))
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex8_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def knn_predict(Xtr, ytr, Xte, k):
        d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        return ytr[np.argsort(d, 1)[:, :k]].mean(1)

    def cv_predict(X, y, folds, k):
        pred = np.zeros_like(y)
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            mu, sd = X[tr].mean(0), X[tr].std(0)
            pred[te] = knn_predict((X[tr] - mu) / sd, y[tr], (X[te] - mu) / sd, k)
        return pred
    ```
    Random pixels: the test pixel has near-copies from its own specimen in the training set. GroupKFold holds out
    whole specimens; the ADF units differ between sessions, so nothing transfers (R² < 0).
    """)})
    return


@app.cell(hide_code=True)
def s6_3(mo):
    mo.md(r"""
    ## 6.3 · ✏️ Exercise 9: preprocessing leakage on pure noise

    80 pixels with their real $t/\lambda$, and 1 000 features of pure Gaussian noise. No feature can predict the thickness.

    Write `select_top(Xn, y, k)`: the indices of the $k$ columns with the largest $|\mathrm{corr}(x_j, y)|$.
    The check then runs a 5-fold CV of a linear regression twice: with the selection on **all** 80 pixels, and inside each training fold.
    """)
    return


@app.cell
def ex9(np):
    # ✏️ YOUR TURN: replace `None`.
    def select_top(Xn, y, k):
        return None

    return (select_top,)


@app.cell(hide_code=True)
def ex9_check(checklist, mo, np, r2, safe, select_top, y_t):
    _rng = np.random.default_rng(11)
    _y = y_t[_rng.choice(len(y_t), 80, replace=False)]
    _s, _e1 = safe(select_top, _rng.normal(size=(80, 1000)), _y, 10)
    if _s is None and _e1 is None:
        _out = checklist([], "Write `select_top` in the cell above.")
    else:
        _Xc = _rng.normal(size=(80, 50))
        _Xc[:, 7] += 3 * (_y - _y.mean())
        _Xc[:, 21] -= 2 * (_y - _y.mean())
        _sc, _e2 = safe(select_top, _Xc, _y, 2)
        _ok = _sc is not None and set(np.asarray(_sc).tolist()) == {7, 21}
        _items = [(_ok, "planted: the two informative columns (7 and 21) are selected, also the negative one" + (f" — {_e2 or _e1}" if (_e2 or _e1) else ""))]
        _parts = [checklist(_items)]
        if _ok:
            def _cv(Xn, inside):
                _f = np.arange(80) % 5
                _p = np.zeros(80)
                _S = None if inside else select_top(Xn, _y, 10)
                for _k in range(5):
                    _tr, _te = _f != _k, _f == _k
                    _sel = select_top(Xn[_tr], _y[_tr], 10) if inside else _S
                    _A = np.c_[np.ones(_tr.sum()), Xn[_tr][:, _sel]]
                    _w = np.linalg.lstsq(_A, _y[_tr], rcond=None)[0]
                    _p[_te] = np.c_[np.ones(_te.sum()), Xn[_te][:, _sel]] @ _w
                return r2(_y, _p)
            _rows = []
            for _seed in range(5):
                _Xn = np.random.default_rng(100 + _seed).normal(size=(80, 1000))
                _rows.append(f"| {_seed} | {_cv(_Xn, False):.2f} | {_cv(_Xn, True):.2f} |")
            _parts.append(mo.md("| noise draw | R², selection on all data | R², selection inside each fold |\n|---|---|---|\n" + "\n".join(_rows) +
                                "\n\n**Think:** the features are pure noise. Where does the positive R² come from? "
                                "Name two EM preprocessing steps that leak in the same way."))
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex9_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def select_top(Xn, y, k):
        Xc, yc = Xn - Xn.mean(0), y - y.mean()
        corr = (Xc.T @ yc) / (np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc))
        return np.argsort(-np.abs(corr))[:k]
    ```
    With 1 000 noise columns, some correlate with t/λ by chance, also on the test pixels. Selecting on all data
    lets the test pixels choose the features. Every step that learns from data belongs inside the fold (a `Pipeline`).
    Examples: PCA fitted on all spectra; energy windows chosen where the signal "correlates best"; intensity normalisation from the whole data set.
    """)})
    return


@app.cell(hide_code=True)
def s6_4(mo):
    mo.md(r"""
    ## 6.4 · ✏️ Exercise 10: name the split

    For each scenario, choose the split that matches the claim. Write `"random"`, `"group"` or `"chronological"`.

    1. 512 crops from 8 micrographs of 8 specimens; claim: "works on new specimens".
    2. An in-situ heating series of 2 000 frames; claim: "predicts the next minutes of the experiment".
    3. 10 000 simulated diffraction patterns, each from an independent random structure; claim: "works on new simulated structures".
    4. EELS spectra from 3 sessions on 2 microscopes; claim: "works on our new microscope".
    """)
    return


@app.cell
def ex10():
    # ✏️ YOUR TURN: replace each `None` with "random", "group" or "chronological".
    split_answers = {1: None, 2: None, 3: None, 4: None}
    return (split_answers,)


@app.cell(hide_code=True)
def ex10_check(checklist, split_answers):
    _key = {1: "group", 2: "chronological", 3: "random", 4: "group"}
    if all(v is None for v in split_answers.values()):
        _out = checklist([], "Fill in `split_answers` in the cell above.")
    else:
        _why = {1: "crops of one specimen are correlated: GroupKFold by specimen",
                2: "neighbouring frames are near-copies, and the claim is about the future: train on the past, test on the later part",
                3: "the patterns are independent by construction: a random K-fold is honest",
                4: "the claim is a new instrument: hold out a whole microscope (group = microscope)"}
        _out = checklist([(split_answers.get(i) == _key[i], f"scenario {i}: " + (_why[i] if split_answers.get(i) == _key[i] else "not yet")) for i in range(1, 5)])
    _out
    return


@app.cell(hide_code=True)
def summary(mo):
    mo.md(r"""
    ---
    ## Summary

    | concept | formula / code |
    |---|---|
    | loss = noise model | $\rho(r) = -\log p(r)$: MSE ↔ Gaussian, MAE ↔ Laplace, Huber = Gaussian core + outliers |
    | IRLS | weights $\min(1, \delta/|r_i|)$, weighted least squares, repeat |
    | gradient descent | $\mathbf w \leftarrow \mathbf w - \eta\nabla L$; stable for $\eta < 2/\lambda_{\max}$; about κ steps; standardise |
    | non-convex losses | side minima: good start, grid, restarts, coarse-to-fine |
    | momentum, Adam | $\mathbf v \leftarrow \beta\mathbf v + \mathbf g$; per-parameter scale $\hat{\mathbf m}/\sqrt{\hat{\mathbf s}}$ |
    | bias–variance | expected error = bias² + variance + noise; the training error always falls |
    | ridge | $(\mathbf R^T\mathbf R + \lambda\mathbf I)^{-1}\mathbf R^T\mathbf y$: shrinks the unstable direction; λ is a prior |
    | lasso | soft threshold $\max(0, g - \lambda)$: exact zeros; standardise; collinear pairs are arbitrary |
    | honest validation | the split matches the claim (random / group / chronological); every learned step inside the fold |
    """)
    return


@app.cell(hide_code=True)
def refs(mo):
    mo.md(r"""
    ---
    ## Next week and references

    **Next: Unit 5, *From images to features: descriptors & tree ensembles*.** Today's linear model becomes the baseline;
    trees must beat it under the same grouped validation.

    **Data**

    - S. H. Mills, S. E. Zeltmann, P. Ercius, A. A. Kohnert, B. P. Uberuaga, A. M. Minor, *Nanoscale mapping of point defect
      concentrations with 4D-STEM*, Acta Mater. **246**, 118721 (2023); data: Zenodo 10.5281/zenodo.7041997, CC BY 4.0.
    - G. Hollyer, E. Stach, *STEM-Bank-26*, arXiv:2610.36086 (2026), CC BY 4.0: CeVO₃/Al₂O₃ EELS and EDS; ten low-loss + ADF spectrum images.
    - B. A. Cymes, K. D. Burgess, R. M. Stroud, Meteoritics & Planetary Science **58**, 259 (2023),
      <https://doi.org/10.1111/maps.13941>; data: Zenodo 10.5281/zenodo.7439174, CC BY 4.0.
    - X-ray line energies and weights: the exspy element database.
    - Processing of all data: `04_regression_validation/xp/prep/*.py` in the course repository.

    **Methods**

    - P. J. Huber, *Robust estimation of a location parameter*, Ann. Math. Stat. **35**, 73 (1964).
    - D. P. Kingma, J. Ba, *Adam: a method for stochastic optimization*, arXiv:1412.6980 (2014).
    - A. E. Hoerl, R. W. Kennard, *Ridge regression*, Technometrics **12**, 55 (1970). — R. Tibshirani, *The lasso*, J. R. Stat. Soc. B **58**, 267 (1996).
    - S. Kapoor, A. Narayanan, *Leakage and the reproducibility crisis in ML-based science*, Patterns **4**, 100804 (2023).
    - R. F. Egerton, *Electron Energy-Loss Spectroscopy in the Electron Microscope*, 3rd ed., Springer (2011).
    """)
    return


if __name__ == "__main__":
    app.run()
