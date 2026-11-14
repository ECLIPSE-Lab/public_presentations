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
    # Unit 5 · Hands-on: from an image to a classifier you can explain, on real HAADF-STEM data

    **Data Science for Electron Microscopy** · Prof. Dr. Philipp Pelz · FAU Erlangen-Nürnberg · WS 26/27

    This notebook goes with the Week 5 lecture *From images to features: descriptors & tree ensembles*.
    The data are four real HAADF-STEM images of one oxide film on LaAlO₃, the same data as in the lecture explainers.
    You find the atom columns, compute descriptors, write a decision tree, bagging and boosting in NumPy,
    and then measure how honest the score is and which descriptors the model uses.

    | part | content | data |
    |---|---|---|
    | **1 · Objects** | smooth, local maxima, threshold; the effect of the dose | ten frames of image D |
    | **2 · Descriptors** | `d1`, `d4`, `psi4`, `I_rel`; invariance tests | candidates of image B |
    | **3 · Tree** | Gini, the best split, a recursive tree | 1 934 labelled candidates, 4 images |
    | **4 · Ensemble** | bagging and the ρσ² floor; boosting with stumps and early stopping (⭐) | the same table |
    | **5 · Validate** | random folds versus image folds; the baseline ladder | the same table |
    | **6 · Explain** | impurity importance versus permutation importance; a noise column and a copied column | the same table |

    **After this notebook you can:**

    1. Turn an image into a list of objects, and say which errors the detection adds to the table.
    2. Compute local descriptors, and test if they are invariant to rotation, gain and offset.
    3. Write the split search of a decision tree, and explain why a deep tree has a high variance.
    4. Explain what bagging and boosting each reduce, and stop boosting at the correct stage.
    5. Choose folds that match the claim "this works on a new image", and read a baseline ladder.
    6. Measure permutation importance on held-out data, and name its two traps.

    **Time:** about 2 hours; the ⭐ exercise is optional. **Prerequisites:** the Unit 1–4 notebooks.
    The notebook uses only NumPy and Matplotlib, and it runs on a CPU. It downloads about 2.5 MB of data once.

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
    - **Exercises** define functions with `None`. Replace each `None` and the check below runs.
    - **Reference functions** (`ref_…`) are correct versions of the exercise functions. Later parts use them,
      so every part runs even if an earlier exercise is open. Do not read them before you have tried.
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
            "data_science_for_em/05_features_trees/xp/data/{}.json")
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
        """Load one data set of the Week 5 lecture: the local repository first, else GitHub."""
        _here = _Path(mo.notebook_dir() or ".")
        _local = _here.parent / "05_features_trees" / "xp" / "data" / f"{name}.json"
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

    def same(a, b, tol=1e-6):
        """True if a is an array of the shape of b with the same numbers."""
        try:
            a = np.asarray(a, float)
            return a.shape == np.shape(b) and bool(np.allclose(a, b, rtol=tol, atol=tol))
        except Exception:  # noqa: BLE001
            return False

    def todo(f):
        """True if the exercise function still has a `None` placeholder."""
        import inspect as _inspect
        import re as _re
        try:
            return bool(_re.search(r"(=\s*None\s*(#.*)?$)|(return None\s*(#.*)?$)|(if None:)", _inspect.getsource(f), _re.M))
        except Exception:  # noqa: BLE001
            return False

    def max_filter(a, w):
        """Maximum of `a` in a window of (2w+1) x (2w+1) pixels around every pixel."""
        for _ax in (0, 1):
            _p = np.pad(a, [(w, w) if _i == _ax else (0, 0) for _i in (0, 1)], constant_values=-np.inf)
            a = np.max([np.take(_p, np.arange(_k, _k + a.shape[_ax]), axis=_ax) for _k in range(2 * w + 1)], 0)
        return a

    return checklist, load, max_filter, safe, same, todo


@app.cell(hide_code=True)
def reference(max_filter, np):
    # Reference implementations. The checks compare your answers with these functions,
    # and later parts use them, so that every part runs even if an exercise is open.
    def ref_smooth(img, sigma):
        r = int(np.ceil(3 * sigma))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k /= k.sum()
        p = np.pad(img, r, mode="reflect")
        p = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, p)
        return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 1, p)

    def ref_find(img, sigma=2.0, thr=0.04, w=6, bg=40.0):
        d = ref_smooth(img, sigma) - ref_smooth(img, bg)
        pk = (d == max_filter(d, w)) & (d > thr * np.percentile(d, 99.5))
        pk[:w + 2] = pk[-w - 2:] = False
        pk[:, :w + 2] = pk[:, -w - 2:] = False
        return np.argwhere(pk)

    def ref_neighbours(P, k=4):
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        np.fill_diagonal(D, np.inf)
        j = np.argsort(D, 1)[:, :k]
        return np.take_along_axis(D, j, 1), j

    def ref_descriptors(P):
        d, j = ref_neighbours(P)
        v = P[j] - P[:, None, :]
        psi4 = np.abs(np.exp(4j * np.arctan2(v[..., 0], v[..., 1])).mean(1))
        return d[:, 0], d.mean(1), psi4

    def ref_gini(y, K=3):
        if len(y) == 0:
            return 0.0
        p = np.bincount(y, minlength=K) / len(y)
        return float(1 - np.sum(p ** 2))

    def ref_best_split(x, y, K=3):
        n = len(y)
        if n < 2:
            return None, 0.0
        o = np.argsort(x, kind="stable")
        xs, ys = x[o], y[o]
        L = np.cumsum(np.eye(K)[ys], 0)
        tot, L = L[-1], L[:-1]
        R = tot - L
        nL = np.arange(1, n)
        nR = n - nL
        gain = (1 - np.sum((tot / n) ** 2)) - (nL * (1 - np.sum((L / nL[:, None]) ** 2, 1))
                                              + nR * (1 - np.sum((R / nR[:, None]) ** 2, 1))) / n
        gain[xs[1:] == xs[:-1]] = -1
        q = int(np.argmax(gain))
        if gain[q] <= 1e-12:
            return None, 0.0
        return float(0.5 * (xs[q] + xs[q + 1])), float(gain[q])

    def ref_grow(X, y, depth, max_features=None, rng=None, K=3):
        c = np.bincount(y, minlength=K)
        node = {"n": len(y), "counts": c, "cls": int(np.argmax(c))}
        if depth == 0 or c.max() == len(y):
            return node
        feats = np.arange(X.shape[1])
        if max_features:
            feats = rng.choice(feats, max_features, replace=False)
        best = (None, 0.0, None)
        for j in feats:
            t, g = ref_best_split(X[:, j], y, K)
            if t is not None and g > best[1]:
                best = (t, g, int(j))
        if best[0] is None:
            return node
        t, g, j = best
        m = X[:, j] <= t
        node.update(j=j, thr=t, gain=g, left=ref_grow(X[m], y[m], depth - 1, max_features, rng, K),
                    right=ref_grow(X[~m], y[~m], depth - 1, max_features, rng, K))
        return node

    def tree_predict(t, X):
        """Class of every row of X from the tree `t` (a nested dict, see Exercise 6)."""
        if "j" not in t:
            return np.full(len(X), t["cls"])
        out = np.empty(len(X), int)
        m = X[:, t["j"]] <= t["thr"]
        out[m] = tree_predict(t["left"], X[m])
        out[~m] = tree_predict(t["right"], X[~m])
        return out

    def n_leaves(t):
        return 1 if "j" not in t else n_leaves(t["left"]) + n_leaves(t["right"])

    def ref_majority(V, K=3):
        return np.argmax(np.array([(V == k).sum(0) for k in range(K)]), 0)

    def ref_forest(X, y, B, depth, max_features, rng):
        T = []
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            T.append(ref_grow(X[i], y[i], depth, max_features, rng))
        return T

    def forest_predict(T, X):
        return ref_majority(np.array([tree_predict(t, X) for t in T]))

    def ref_stump(X, r):
        best = (-np.inf,)
        for j in range(X.shape[1]):
            x = X[:, j]
            o = np.argsort(x, kind="stable")
            xs, rs = x[o], r[o]
            n = len(r)
            cs = np.cumsum(rs)[:-1]
            nL = np.arange(1, n)
            tot = rs.sum()
            sc = cs ** 2 / nL + (tot - cs) ** 2 / (n - nL)
            sc[xs[1:] == xs[:-1]] = -np.inf
            q = int(np.argmax(sc))
            if sc[q] > best[0]:
                best = (sc[q], j, 0.5 * (xs[q] + xs[q + 1]), cs[q] / nL[q], (tot - cs[q]) / (n - nL[q]))
        return best[1:]

    def ref_boost(X, y, Xv, yv, eta, M):
        F = np.full(len(y), y.mean())
        Fv = np.full(len(yv), y.mean())
        tr, va = [], []
        for _ in range(M):
            j, t, a, b = ref_stump(X, y - F)
            F = F + eta * np.where(X[:, j] <= t, a, b)
            Fv = Fv + eta * np.where(Xv[:, j] <= t, a, b)
            tr.append(np.mean((y - F) ** 2))
            va.append(np.mean((yv - Fv) ** 2))
        return np.array(tr), np.array(va)

    def ref_cv_predict(fit_predict, X, y, folds):
        pred = np.empty(len(y), int)
        for te in folds:
            tr = np.setdiff1d(np.arange(len(y)), te)
            pred[te] = fit_predict(X[tr], y[tr], X[te])
        return pred

    def ref_perm(predict, X, y, rng, n_rep=5):
        base = np.mean(predict(X) == y)
        out = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            for _ in range(n_rep):
                Xp = X.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                out[j] += (base - np.mean(predict(Xp) == y)) / n_rep
        return out

    return (forest_predict, n_leaves, ref_best_split, ref_boost, ref_cv_predict, ref_descriptors, ref_find,
            ref_forest, ref_gini, ref_grow, ref_majority, ref_neighbours, ref_perm, ref_smooth, ref_stump,
            tree_predict)


@app.cell
def load_data(load, np):
    cols = load("columns")     # the feature table of four images
    fr = load("frames")        # ten aligned frames of a crop of image D

    NAMES = cols["names"]                       # 11 descriptor names
    CLASSES = cols["classes"]                   # substrate column, film column, no column
    _lab = cols["y"] >= 0                       # candidates near a boundary or the border have no label
    X = cols["X"][_lab]                         # [1934, 11] descriptors
    y = cols["y"][_lab].astype(int)             # class 0, 1, 2
    g = cols["g"][_lab].astype(int)             # image index 0..3 (images A..D)
    pos = cols["pos"][_lab]                     # (y, x) in nm
    col = NAMES.index                           # col("d4") -> column index
    INV = [col(_n) for _n in ("I_rel", "d1", "d4", "d_spread", "n_near", "psi4")]
    RAW = [col(_n) for _n in ("amp", "bg", "sigma")]
    CCOL = ["tab:blue", "tab:orange", "tab:purple"]
    print(f"{len(y)} labelled candidates; per class {np.bincount(y)}; per image {np.bincount(g)}")
    return CCOL, CLASSES, INV, NAMES, RAW, X, col, cols, fr, g, pos, y


@app.cell(hide_code=True)
def s1(mo):
    mo.md(r"""
    ---
    # Part 1 · Objects: from an image to a list of candidates

    ## 1.1 · The data

    `fr["frames"]` holds ten aligned frames of one region (5.8 nm × 5.1 nm, 20 pm per pixel).
    From left to right you see the LaAlO₃ substrate, the oxide film, and amorphous material without columns.
    The sum of ten frames has ten times the dose of one frame.
    """)
    return


@app.cell
def frames_plot(fr, plt):
    FR = fr["frames"]                # [10, 256, 288]
    SC = fr["scale_nm"]              # nm per pixel
    # zone boundaries in pixels of the crop: substrate | film | amorphous
    ZPX = [(_z - fr["x0_nm"]) / SC for _z in fr["zones"]]
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for _a, _im, _t in zip(_ax, (FR[0], FR.mean(0)), ("1 frame", "mean of 10 frames")):
        _a.imshow(_im, cmap="gray")
        for _x in (0.5 * (ZPX[0] + ZPX[1]), 0.5 * (ZPX[2] + ZPX[3])):
            _a.axvline(_x, color="w", ls="--", lw=1)
        _a.set_title(_t)
        _a.grid(False)
    _fig.tight_layout()
    _fig
    return FR, SC, ZPX


@app.cell(hide_code=True)
def s1_2(mo):
    mo.md(r"""
    ## 1.2 · ✏️ Exercise 1: Gaussian smoothing

    A local maximum of a noisy image is mostly noise. Smoothing with a Gaussian of about the column width
    is the matched filter for a Gaussian peak in white noise.

    Write `gauss_smooth(img, sigma)`:

    1. Make the 1-D kernel $k_i = e^{-i^2/2\sigma^2}$ for $i = -r \dots r$ with $r = \lceil 3\sigma\rceil$. Divide it by its sum.
    2. Pad the image by $r$ pixels on every side with `np.pad(img, r, mode="reflect")`.
    3. Convolve every column with the kernel, then every row (`np.convolve(v, k, "valid")`).
       The Gaussian is separable, so two 1-D convolutions are equal to one 2-D convolution.
    """)
    return


@app.cell
def ex1(np):
    # ✏️ YOUR TURN: replace each `None`.
    def gauss_smooth(img, sigma):
        r = int(np.ceil(3 * sigma))
        k = None                      # normalised 1-D kernel
        p = None                      # padded image
        p = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, p)
        return None                   # the same along axis 1

    return (gauss_smooth,)


@app.cell(hide_code=True)
def ex1_check(FR, checklist, gauss_smooth, mo, np, plt, ref_smooth, safe, same, todo):
    _img = FR[0]
    _s, _e = safe(gauss_smooth, _img, 2.0)
    if todo(gauss_smooth):
        _out = checklist([], "Write `gauss_smooth` in the cell above.")
    else:
        _c, _ = safe(gauss_smooth, np.full((20, 30), 7.0), 1.5)
        _items = [(_s is not None and np.shape(_s) == _img.shape, f"the output has the shape of the image {_e or ''}"),
                  (same(_c, np.full((20, 30), 7.0)), "a constant image stays constant (the kernel sum is 1)"),
                  (same(_s, ref_smooth(_img, 2.0)), "frame 0 with σ = 2 px is equal to the reference")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _fig, _ax = plt.subplots(1, 3, figsize=(12, 3.6))
            for _a, _sg in zip(_ax, (0.7, 2.0, 5.0)):
                _a.imshow(gauss_smooth(_img, _sg)[60:160, 20:140], cmap="gray")
                _a.set_title(f"σ = {_sg} px = {20 * _sg:.0f} pm")
                _a.grid(False)
            _fig.tight_layout()
            _parts += [_fig, mo.md("**Think:** the columns have a width σ of about 60 pm. Why is σ = 5 px too much "
                                   "for the film, where two sublattices are 0.19 nm apart?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex1_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def gauss_smooth(img, sigma):
        r = int(np.ceil(3 * sigma))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k = k / k.sum()
        p = np.pad(img, r, mode="reflect")
        p = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, p)
        return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 1, p)
    ```
    A small σ keeps the noise. A large σ merges neighbour columns and moves the maxima. The detection step
    has a parameter, and the feature table depends on it.
    """)})
    return


@app.cell(hide_code=True)
def s1_3(mo):
    mo.md(r"""
    ## 1.3 · ✏️ Exercise 2: local maxima and a threshold

    Write `find_columns(img, sigma, thr)`. It returns an integer array `[N, 2]` of (row, column) positions.

    1. Band-pass: `d = ref_smooth(img, sigma) - ref_smooth(img, 40.0)`. The second term removes the slow background.
    2. A pixel is a candidate if it is the maximum of its neighbourhood, `d == max_filter(d, 6)`
       (window ±6 px = ±0.12 nm), **and** if `d > thr * np.percentile(d, 99.5)`.
    3. Remove candidates closer than 8 px to the border. Return `np.argwhere(mask)`.
    """)
    return


@app.cell
def ex2(max_filter, np, ref_smooth):
    # ✏️ YOUR TURN: replace each `None`.
    def find_columns(img, sigma=2.0, thr=0.04):
        d = None                     # band-pass image
        mask = None                  # local maximum AND above the threshold
        mask[:8] = mask[-8:] = False
        mask[:, :8] = mask[:, -8:] = False
        return np.argwhere(mask)

    return (find_columns,)


@app.cell
def thr_controls(mo):
    thr_s = mo.ui.slider(0.0, 0.5, 0.01, value=0.04, label="threshold (fraction of the 99.5 % level)", show_value=True)
    nfr_s = mo.ui.slider(1, 10, 1, value=10, label="number of frames (dose)", show_value=True)
    mo.hstack([thr_s, nfr_s])
    return nfr_s, thr_s


@app.cell(hide_code=True)
def ex2_check(FR, SC, ZPX, checklist, find_columns, mo, nfr_s, np, plt, ref_find, safe, thr_s, todo):
    def _zones(P):
        x = P[:, 1]
        return int(np.sum(x < ZPX[0])), int(np.sum((x > ZPX[1]) & (x < ZPX[2]))), int(np.sum(x > ZPX[3]))

    _m10 = FR.mean(0)
    _P, _e = safe(find_columns, _m10, 2.0, 0.04)
    if todo(find_columns):
        _out = checklist([], "Write `find_columns` in the cell above.")
    else:
        _R = ref_find(_m10, 2.0, 0.04)
        _P2, _ = safe(find_columns, _m10, 2.0, 0.2)
        _R2 = ref_find(_m10, 2.0, 0.2)
        _ok = lambda a, b: a is not None and np.shape(a) == b.shape and bool(np.all(np.asarray(a) == b))
        _items = [(_ok(_P, _R), f"10 frames, threshold 0.04: {len(_R)} candidates {_e or ''}"),
                  (_ok(_P2, _R2), f"10 frames, threshold 0.20: {len(_R2)} candidates")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _img = FR[:nfr_s.value].mean(0)
            _Q = find_columns(_img, 2.0, thr_s.value)
            _fig, _ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
            _ax[0].imshow(_img, cmap="gray")
            _ax[0].plot(_Q[:, 1], _Q[:, 0], "o", mfc="none", mec="tab:red", ms=7)
            _ax[0].grid(False)
            _ax[0].set_title(f"{nfr_s.value} frame(s), threshold {thr_s.value:.2f}: substrate / film / amorphous = "
                             + " / ".join(str(v) for v in _zones(_Q)))
            _t = np.linspace(0, 0.5, 26)
            for _n, _ls in ((10, "-"), (1, "--")):
                _c = np.array([_zones(find_columns(FR[:_n].mean(0), 2.0, _v)) for _v in _t])
                for _k, (_lb, _cl) in enumerate((("substrate", "tab:blue"), ("film", "tab:orange"), ("amorphous (false)", "tab:purple"))):
                    _ax[1].plot(_t, _c[:, _k], _ls, color=_cl, label=f"{_lb}, {_n} frame(s)")
            _ax[1].axvline(thr_s.value, color="k", lw=0.8)
            _ax[1].set_xlabel("threshold")
            _ax[1].set_ylabel("number of candidates")
            _ax[1].legend(fontsize=8)
            _fig.tight_layout()
            _parts += [_fig, mo.md(f"The crop is {FR.shape[1] * SC:.1f} nm × {FR.shape[2] * SC:.1f} nm. "
                                   "**Think:** is there one threshold that keeps all film columns and no false detection? "
                                   "How does the answer change with one frame? What happens to the rows of the "
                                   "feature table in each case?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex2_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def find_columns(img, sigma=2.0, thr=0.04):
        d = ref_smooth(img, sigma) - ref_smooth(img, 40.0)
        mask = (d == max_filter(d, 6)) & (d > thr * np.percentile(d, 99.5))
        mask[:8] = mask[-8:] = False
        mask[:, :8] = mask[:, -8:] = False
        return np.argwhere(mask)
    ```
    No threshold separates the weak film columns from the noise maxima of the amorphous region.
    The lecture keeps a low threshold and adds a third class, *no column*. Then the classifier removes the false detections.
    A missed column is lost; no later step can bring it back.
    """)})
    return


@app.cell(hide_code=True)
def s2(mo):
    mo.md(r"""
    ---
    # Part 2 · Descriptors: from a candidate to a row of numbers

    ## 2.1 · The feature table

    The lecture pipeline refined every candidate with a 2-D Gaussian fit and made the table `X`: one row per candidate,
    11 descriptors. `y` is the class (0 substrate column, 1 film column, 2 no column), `g` is the image (0…3 = A…D),
    `pos` is the position (y, x) in nm.
    """)
    return


@app.cell
def table_plot(CCOL, CLASSES, NAMES, X, cols, g, mo, np, plt, pos, y):
    _k = 1                                        # image B
    _I = cols["images"][_k]
    _m = g == _k
    _fig, _ax = plt.subplots(figsize=(6.2, 6.2))
    _ax.imshow(_I["img"], cmap="gray", extent=[0, _I["size_nm"][1], _I["size_nm"][0], 0])
    for _c in range(3):
        _q = _m & (y == _c)
        _ax.plot(pos[_q, 1], pos[_q, 0], "o", mfc="none", mec=CCOL[_c], ms=5, label=CLASSES[_c])
    _ax.legend(loc="lower left", fontsize=8)
    _ax.set_xlabel("x (nm)")
    _ax.grid(False)
    _rows = "\n".join(f"| `{n}` | {u} | {a} | {np.median(X[:, i]):.3g} |"
                      for i, (n, u, a) in enumerate(zip(NAMES, cols["units"], cols["about"])))
    mo.hstack([_fig, mo.md("| descriptor | unit | meaning | median |\n|---|---|---|---|\n" + _rows)], widths=[1, 1.2])
    return


@app.cell(hide_code=True)
def s2_2(mo):
    mo.md(r"""
    ## 2.2 · ✏️ Exercise 3: three geometric descriptors

    `P` is an array `[N, 2]` of positions (y, x) in nm of **all** candidates of one image.

    1. Write `neighbours(P, k)`: the distances to the `k` nearest other candidates (sorted, `[N, k]`) and their indices (`[N, k]`).
       A full distance matrix is fine for 700 candidates. Put `np.inf` on the diagonal.
    2. Write `descriptors(P)`: it returns `d1` (distance to the nearest), `d4` (mean distance to the 4 nearest) and
       `psi4` $= \left|\frac14\sum_{j=1}^{4} e^{4i\theta_j}\right|$, where $\theta_j$ is the angle of the bond to neighbour $j$,
       `np.arctan2(dy, dx)`.
    """)
    return


@app.cell
def ex3(np):
    # ✏️ YOUR TURN: replace each `None`.
    def neighbours(P, k=4):
        D = None                     # [N, N] distances, np.inf on the diagonal
        j = np.argsort(D, 1)[:, :k]
        return np.take_along_axis(D, j, 1), j

    def descriptors(P):
        d, j = neighbours(P, 4)
        v = P[j] - P[:, None, :]     # [N, 4, 2] bond vectors (dy, dx)
        d1 = None
        d4 = None
        psi4 = None
        return d1, d4, psi4

    return descriptors, neighbours


@app.cell
def image_b(cols):
    # all candidates of image B (with and without a label), as the lecture pipeline used them
    _m = cols["g"] == 1
    PB = cols["pos"][_m]
    XB = cols["X"][_m]
    YB = cols["y"][_m].astype(int)
    return PB, XB, YB


@app.cell(hide_code=True)
def ex3_check(CCOL, PB, XB, YB, checklist, col, descriptors, mo, np, plt, neighbours, safe, todo):
    _r, _e = safe(descriptors, PB)
    if todo(neighbours) or todo(descriptors):
        _out = checklist([], "Write `neighbours` and `descriptors` in the cell above.")
    else:
        _ok = lambda a, n: a is not None and np.shape(a) == (len(PB),) and bool(np.allclose(a, XB[:, col(n)], atol=2e-5))
        _r = _r or (None, None, None)
        _items = [(_ok(_r[0], "d1"), f"`d1` is equal to the column `d1` of the table {_e or ''}"),
                  (_ok(_r[1], "d4"), "`d4` is equal to the column `d4` of the table"),
                  (_ok(_r[2], "psi4"), "`psi4` is equal to the column `psi4` of the table")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _fig, _ax = plt.subplots(1, 3, figsize=(12, 3.4))
            for _a, _v, _n in zip(_ax, _r, ("d1 (nm)", "d4 (nm)", "psi4")):
                for _c in range(3):
                    _a.hist(_v[YB == _c], bins=40, range=(np.percentile(_v, 0.5), np.percentile(_v, 99.5)),
                            color=CCOL[_c], alpha=0.6)
                _a.set_xlabel(_n)
            _fig.tight_layout()
            _parts += [_fig, mo.md("Blue: substrate column, orange: film column, purple: no column. "
                                   "**Think:** which descriptor separates which pair of classes? Why is `psi4` near 1 "
                                   "in the substrate and lower in the film?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex3_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def neighbours(P, k=4):
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        np.fill_diagonal(D, np.inf)
        j = np.argsort(D, 1)[:, :k]
        return np.take_along_axis(D, j, 1), j

    def descriptors(P):
        d, j = neighbours(P, 4)
        v = P[j] - P[:, None, :]
        d1 = d[:, 0]
        d4 = d.mean(1)
        psi4 = np.abs(np.exp(4j * np.arctan2(v[..., 0], v[..., 1])).mean(1))
        return d1, d4, psi4
    ```
    The substrate shows one sublattice: a square lattice with 0.38 nm, so `d4` ≈ 0.38 nm and `psi4` ≈ 1.
    The film shows two sublattices, so the nearest neighbours are closer. Noise maxima have irregular distances.
    """)})
    return


@app.cell(hide_code=True)
def s2_3(mo):
    mo.md(r"""
    ## 2.3 · ✏️ Exercise 4: test the invariance

    A descriptor must not change when the microscope settings change but the specimen does not.
    Do not trust an argument; apply the transformation and measure.

    1. Write `i_rel(amp, j)`: the peak height of every candidate divided by the **median** peak height of its 4 nearest
       neighbours (`j` has the indices, `[N, 4]`).
    2. Write `rotate(P, angle_deg)`: rotate all positions around the origin by the angle.

    The check then applies a rotation by 30°, and a detector gain of 3 with an offset, to image B.
    """)
    return


@app.cell
def ex4(np):
    # ✏️ YOUR TURN: replace each `None`.
    def i_rel(amp, j):
        return None

    def rotate(P, angle_deg):
        a = np.deg2rad(angle_deg)
        return None                  # [N, 2], columns (y, x)

    return i_rel, rotate


@app.cell(hide_code=True)
def ex4_check(PB, XB, checklist, col, i_rel, mo, np, ref_descriptors, ref_neighbours, rotate, safe):
    _amp, _bg = XB[:, col("amp")], XB[:, col("bg")]
    _d, _j = ref_neighbours(PB)
    _ir, _e1 = safe(i_rel, _amp, _j)
    _Pr, _e2 = safe(rotate, PB, 30.0)
    if _ir is None and _Pr is None and _e1 is None and _e2 is None:
        _out = checklist([], "Write `i_rel` and `rotate` in the cell above.")
    else:
        _okr = (_Pr is not None and np.shape(_Pr) == PB.shape
                and np.allclose(np.hypot(*np.asarray(_Pr).T), np.hypot(*PB.T))
                and np.allclose(np.abs(np.cross(np.asarray(_Pr), PB)) / np.maximum(np.hypot(*PB.T) ** 2, 1e-12), 0.5))
        _items = [(_ir is not None and np.shape(_ir) == _amp.shape and np.allclose(_ir, XB[:, col("I_rel")], atol=2e-5),
                   f"`i_rel` is equal to the column `I_rel` of the table {_e1 or ''}"),
                  (bool(_okr), f"`rotate` keeps all lengths and turns every vector by 30° {_e2 or ''}")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _P2 = np.asarray(_Pr)
            _d2, _j2 = ref_neighbours(_P2)
            _v, _v2 = PB[_j] - PB[:, None], _P2[_j2] - _P2[:, None]
            _th = lambda v: np.arctan2(v[:, 0, 0], v[:, 0, 1])           # angle of the first bond: not invariant
            _g = lambda a: 3 * a                                          # gain 3: the fitted peak height scales
            _gb = lambda b: 3 * b + 5e5                                   # gain 3 + offset: the background also shifts
            _rows = [("angle of the nearest bond", "rotation 30°", np.abs(np.angle(np.exp(1j * (_th(_v2) - _th(_v))))).max()),
                     ("`d4`", "rotation 30°", np.abs(_d2.mean(1) - _d.mean(1)).max() / _d.mean()),
                     ("`psi4`", "rotation 30°", np.abs(ref_descriptors(_P2)[2] - ref_descriptors(PB)[2]).max()),
                     ("`amp`", "gain 3 + offset", np.abs(_g(_amp) - _amp).max() / _amp.mean()),
                     ("`contrast` = amp / bg", "gain 3 + offset", np.abs(_g(_amp) / _gb(_bg) - _amp / _bg).max() / (_amp / _bg).mean()),
                     ("`I_rel`", "gain 3 + offset", np.abs(i_rel(_g(_amp), _j) - i_rel(_amp, _j)).max())]
            _tbl = "\n".join(f"| {a} | {b} | {c:.2g} | {'✅ invariant' if c < 1e-6 else '❌ changes'} |" for a, b, c in _rows)
            _parts += [mo.md("| descriptor | transformation | largest relative change | |\n|---|---|---|---|\n" + _tbl +
                             "\n\n**Think:** a model that uses `amp` learns the detector gain of the training images. "
                             "Which descriptors in the table can it use on an image with another magnification? "
                             "(`d1` and `d4` are in nm: the calibration is part of the descriptor.)")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex4_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def i_rel(amp, j):
        return amp / np.median(amp[j], axis=1)

    def rotate(P, angle_deg):
        a = np.deg2rad(angle_deg)
        y, x = P[:, 0], P[:, 1]
        return np.c_[np.cos(a) * y + np.sin(a) * x, -np.sin(a) * y + np.cos(a) * x]
    ```
    A ratio of two peak heights removes the gain. The Gaussian fit has a constant term, so the peak height does not
    contain the offset. `contrast` divides by the background, and the background contains the offset: it is not invariant.
    Distances and `psi4` ignore a rotation; a raw angle does not.
    """)})
    return


@app.cell(hide_code=True)
def s3(mo):
    mo.md(r"""
    ---
    # Part 3 · Tree: a model made of yes/no questions

    ## 3.1 · ✏️ Exercise 5: Gini impurity and the best split

    A node holds rows with class fractions $p_k$. Its Gini impurity is $I = 1 - \sum_k p_k^2$.
    A split at threshold $t$ sends $N_L$ rows left ($x \le t$) and $N_R$ rows right. The impurity decrease is

    $$\Delta = I(\text{parent}) - \frac{N_L}{N} I(L) - \frac{N_R}{N} I(R).$$

    1. Write `gini(y)` for an integer class vector (3 classes). An empty node has impurity 0.
    2. Write `best_split(x, y)`: test the midpoint between each pair of adjacent **different** sorted values of `x`.
       Return `(t, Δ)` of the best threshold, or `(None, 0.0)` if no split has Δ > 0. A loop over thresholds is fine.
    """)
    return


@app.cell
def ex5(np):
    # ✏️ YOUR TURN: replace each `None`.
    def gini(y):
        return None

    def best_split(x, y):
        best_t, best_gain = None, 0.0
        xs = np.unique(x)                         # sorted, different values
        for t in 0.5 * (xs[1:] + xs[:-1]):
            left = x <= t
            gain = None                           # impurity decrease of this threshold
            if gain > best_gain:
                best_t, best_gain = float(t), float(gain)
        return best_t, best_gain

    return best_split, gini


@app.cell(hide_code=True)
def ex5_check(CCOL, NAMES, X, best_split, checklist, col, gini, mo, np, plt, ref_best_split, ref_gini, safe, y):
    _g0, _e1 = safe(gini, y)
    if _g0 is None and _e1 is None:
        _out = checklist([], "Write `gini` and `best_split` in the cell above.")
    else:
        _gp, _ = safe(gini, np.array([1, 1, 1]))
        _ge, _ = safe(gini, np.array([0, 1, 2]))
        _x = X[:, col("d4")]
        _bs, _e2 = safe(best_split, _x, y)
        _rt, _rg = ref_best_split(_x, y)
        _cs, _ = safe(best_split, np.ones(5), np.array([0, 1, 0, 1, 2]))
        _close = lambda a, b: a is not None and b is not None and abs(a - b) < 1e-6
        _items = [(_close(_g0, ref_gini(y)), f"Gini of all {len(y)} rows = {ref_gini(y):.3f} {_e1 or ''}"),
                  (_close(_gp, 0.0) and _close(_ge, 2 / 3), "a pure node gives 0; three equal classes give 2/3"),
                  (_bs is not None and _close(_bs[0], _rt) and _close(_bs[1], _rg),
                   f"best split on `d4`: t = {_rt:.3f} nm, Δ = {_rg:.3f} {_e2 or ''}"),
                  (_cs is not None and _cs[0] is None, "a constant feature has no split: `(None, 0.0)`")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.6))
            for _c in range(3):
                _ax[0].hist(_x[y == _c], bins=60, range=(0.15, 0.6), color=CCOL[_c], alpha=0.6)
            _ax[0].axvline(_rt, color="k")
            _ax[0].set_xlabel("d4 (nm)")
            _ax[0].set_title(f"root split: d4 ≤ {_rt:.3f} nm")
            _gains = [ref_best_split(X[:, _j], y)[1] for _j in range(X.shape[1])]
            _ax[1].barh(NAMES[::-1], _gains[::-1], color="0.5")
            _ax[1].set_xlabel("largest impurity decrease Δ at the root")
            _fig.tight_layout()
            _parts += [_fig, mo.md("**Think:** the greedy tree takes the descriptor with the longest bar. "
                                   "Multiply `d4` by 1000 (nm → pm). Does Δ change? What follows for feature scaling?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex5_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def gini(y):
        if len(y) == 0:
            return 0.0
        p = np.bincount(y, minlength=3) / len(y)
        return 1 - np.sum(p ** 2)

    def best_split(x, y):
        best_t, best_gain = None, 0.0
        xs = np.unique(x)
        for t in 0.5 * (xs[1:] + xs[:-1]):
            left = x <= t
            gain = gini(y) - left.mean() * gini(y[left]) - (1 - left.mean()) * gini(y[~left])
            if gain > best_gain:
                best_t, best_gain = float(t), float(gain)
        return best_t, best_gain
    ```
    The split depends only on the **order** of the values. A monotonic change of a feature (unit, log, gain)
    moves the threshold but gives the same children. So trees need no feature scaling.
    The reference `ref_best_split` gives the same result with one sort and cumulative sums; it is much faster.
    """)})
    return


@app.cell(hide_code=True)
def s3_2(mo):
    mo.md(r"""
    ## 3.2 · ✏️ Exercise 6: grow a tree

    A tree is a nested `dict`. Every node has `"n"` (rows), `"counts"` (rows per class) and `"cls"` (majority class).
    An inner node has in addition `"j"` (feature), `"thr"` (threshold), `"gain"` (Δ), `"left"` and `"right"`.

    Complete `grow_tree(X, y, depth)`:

    1. Stop if `depth == 0` or if the node is pure.
    2. Find the feature and threshold with the largest Δ (use the fast `ref_best_split(x, y)`).
    3. Stop if no feature gives a split. Else split the rows and call `grow_tree` for both children with `depth - 1`.

    `tree_predict(tree, X)` and `n_leaves(tree)` are given.
    """)
    return


@app.cell
def ex6(np, ref_best_split):
    # ✏️ YOUR TURN: replace each `None`.
    def grow_tree(X, y, depth):
        counts = np.bincount(y, minlength=3)
        node = {"n": len(y), "counts": counts, "cls": int(np.argmax(counts))}
        if None:                                  # stop condition
            return node
        best = (None, 0.0, None)                  # threshold, gain, feature
        for j in range(X.shape[1]):
            t, gain = ref_best_split(X[:, j], y)
            if None:                              # is this split better than `best`?
                best = (t, gain, j)
        if best[0] is None:
            return node
        t, gain, j = best
        left = X[:, j] <= t
        node.update(j=j, thr=t, gain=gain,
                    left=None,                    # the tree of the left rows
                    right=None)                   # the tree of the right rows
        return node

    return (grow_tree,)


@app.cell
def depth_control(mo):
    depth_s = mo.ui.slider(1, 14, 1, value=3, label="depth of the tree", show_value=True)
    depth_s
    return (depth_s,)


@app.cell(hide_code=True)
def ex6_check(CCOL, X, checklist, col, depth_s, g, grow_tree, mo, n_leaves, np, plt, ref_grow, safe, tree_predict, y):
    _J = [col("d4"), col("I_rel")]
    _tr, _te = g != 1, g == 1                    # train on images A, C, D; test on image B
    _Xa, _ya, _Xb, _yb = X[_tr][:, _J], y[_tr], X[_te][:, _J], y[_te]
    _t1, _e = safe(grow_tree, _Xa, _ya, 1)
    if _e is None and (_t1 is None or "j" not in _t1 or _t1.get("left") is None):
        _out = checklist([], "Complete `grow_tree` in the cell above. (A tree of depth 1 must have one split.)")
    else:
        def _same_tree(d):
            t, e = safe(grow_tree, _Xa, _ya, d)
            if t is None:
                return False
            p, e = safe(tree_predict, t, _Xb)
            r = ref_grow(_Xa, _ya, d)
            return p is not None and bool(np.all(p == tree_predict(r, _Xb))) and n_leaves(t) == n_leaves(r)

        _r1 = ref_grow(_Xa, _ya, 1)
        _items = [(_e is None and _t1 is not None and _t1.get("j") == _r1["j"] and abs(_t1["thr"] - _r1["thr"]) < 1e-9,
                   f"depth 1: the root asks `d4 ≤ {_r1['thr']:.3f}` {_e or ''}"),
                  (_same_tree(3), "depth 3: the same leaves and predictions as the reference"),
                  (_same_tree(30), "depth 30: the tree grows until every leaf is pure")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _ds = list(range(1, 15))
            _trees = [grow_tree(_Xa, _ya, _d) for _d in _ds]
            _atr = [np.mean(tree_predict(_t, _Xa) == _ya) for _t in _trees]
            _ate = [np.mean(tree_predict(_t, _Xb) == _yb) for _t in _trees]
            _T = _trees[depth_s.value - 1]
            _fig, _ax = plt.subplots(1, 2, figsize=(12, 4.2))
            _xx, _yy = np.meshgrid(np.linspace(0.15, 0.6, 300), np.linspace(0.2, 2.2, 300))
            _Z = tree_predict(_T, np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
            _ax[0].contourf(_xx, _yy, _Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=CCOL, alpha=0.25)
            for _c in range(3):
                _ax[0].plot(_Xb[_yb == _c, 0], _Xb[_yb == _c, 1], ".", color=CCOL[_c], ms=4)
            _ax[0].set_xlim(0.15, 0.6)
            _ax[0].set_ylim(0.2, 2.2)
            _ax[0].set_xlabel("d4 (nm)")
            _ax[0].set_ylabel("I_rel")
            _ax[0].set_title(f"depth {depth_s.value}: {n_leaves(_T)} leaves; points = image B (not in the training)")
            _ax[1].plot(_ds, _atr, "o-", label="training images A, C, D")
            _ax[1].plot(_ds, _ate, "o-", label="new image B")
            _ax[1].axvline(depth_s.value, color="k", lw=0.8)
            _ax[1].set_xlabel("depth")
            _ax[1].set_ylabel("accuracy")
            _ax[1].legend()
            _fig.tight_layout()
            _parts += [_fig, mo.md("**Think:** every leaf is a rectangle with one constant answer. "
                                   "What does the tree predict for `d4` = 0.9 nm, a value outside the training range? "
                                   "At which depth does the tree start to learn the noise of the training images?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex6_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def grow_tree(X, y, depth):
        counts = np.bincount(y, minlength=3)
        node = {"n": len(y), "counts": counts, "cls": int(np.argmax(counts))}
        if depth == 0 or counts.max() == len(y):
            return node
        best = (None, 0.0, None)
        for j in range(X.shape[1]):
            t, gain = ref_best_split(X[:, j], y)
            if t is not None and gain > best[1]:
                best = (t, gain, j)
        if best[0] is None:
            return node
        t, gain, j = best
        left = X[:, j] <= t
        node.update(j=j, thr=t, gain=gain,
                    left=grow_tree(X[left], y[left], depth - 1),
                    right=grow_tree(X[~left], y[~left], depth - 1))
        return node
    ```
    The training accuracy rises to 100 %. The accuracy on the new image is best for a shallow tree and then falls:
    a deep tree has a low bias and a high variance. A tree cannot extrapolate; outside the training range it repeats
    the answer of the outermost leaf.
    """)})
    return


@app.cell(hide_code=True)
def s4(mo):
    mo.md(r"""
    ---
    # Part 4 · Ensemble: many trees are better than one

    ## 4.1 · ✏️ Exercise 7: bagging

    A deep tree changes a lot when the training rows change. **Bagging** uses this: grow $B$ deep trees, each on a
    bootstrap sample (draw $N$ rows **with replacement**), and let the trees vote.
    For $B$ predictions with variance $\sigma^2$ and pairwise correlation $\rho$, the variance of the average is

    $$\rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 .$$

    1. Write `bootstrap(n, rng)`: `n` row indices, drawn with replacement with `rng.integers`.
    2. Write `majority(V)`: `V` is `[B, N]`, the class that each of `B` trees predicts for `N` rows. Return the class with the most votes per row.
    3. Write `ensemble_variance(rho, sigma2, B)`: the formula above.
    """)
    return


@app.cell
def ex7(np):
    # ✏️ YOUR TURN: replace each `None`.
    def bootstrap(n, rng):
        return None

    def majority(V):
        votes = np.array([(V == k).sum(0) for k in range(3)])     # [3, N] votes per class
        return None

    def ensemble_variance(rho, sigma2, B):
        return None

    return bootstrap, ensemble_variance, majority


@app.cell
def grow_forests(INV, X, forest_predict, g, np, ref_forest, tree_predict, y):
    # 60 deep trees with bagging (all 6 descriptors at every split) and 60 trees of a random forest
    # (2 random descriptors at every split). Training: images A, C, D. Test: image B.
    _tr, _te = g != 1, g == 1
    XA, YA, XBt, YBt = X[_tr][:, INV], y[_tr], X[_te][:, INV], y[_te]
    _rng = np.random.default_rng(0)
    T_BAG = ref_forest(XA, YA, 60, 12, None, _rng)
    T_RF = ref_forest(XA, YA, 60, 12, 2, _rng)
    V_BAG = np.array([tree_predict(_t, XBt) for _t in T_BAG])     # [60, 580] votes
    V_RF = np.array([tree_predict(_t, XBt) for _t in T_RF])
    print(f"bagging: one tree {np.mean(V_BAG == YBt):.3f}, 60 trees {np.mean(forest_predict(T_BAG, XBt) == YBt):.3f}")
    print(f"random forest: one tree {np.mean(V_RF == YBt):.3f}, 60 trees {np.mean(forest_predict(T_RF, XBt) == YBt):.3f}")
    return T_RF, V_BAG, V_RF, XA, XBt, YA, YBt


@app.cell(hide_code=True)
def ex7_check(V_BAG, V_RF, YBt, bootstrap, checklist, ensemble_variance, majority, mo, np, plt, ref_majority, safe):
    _i, _e1 = safe(bootstrap, 1000, np.random.default_rng(0))
    _m, _e2 = safe(majority, V_BAG)
    _v, _e3 = safe(ensemble_variance, 0.5, 2.0, 10)
    if all(v is None for v in (_i, _m, _v, _e1, _e2, _e3)):
        _out = checklist([], "Write `bootstrap`, `majority` and `ensemble_variance` in the cell above.")
    else:
        _oki = (_i is not None and np.shape(_i) == (1000,) and np.min(_i) >= 0 and np.max(_i) < 1000
                and 0.60 < len(np.unique(_i)) / 1000 < 0.67)
        _items = [(bool(_oki), f"`bootstrap`: 1000 indices; about 63 % of the rows are in the sample {_e1 or ''}"),
                  (_m is not None and np.shape(_m) == YBt.shape and bool(np.all(_m == ref_majority(V_BAG))),
                   f"`majority`: equal to the reference for 60 trees {_e2 or ''}"),
                  (_v is not None and abs(_v - 1.1) < 1e-9 and abs(ensemble_variance(0.0, 2.0, 10) - 0.2) < 1e-9,
                   f"`ensemble_variance(0.5, 2, 10)` = 1.1 and `(0, 2, 10)` = 0.2 {_e3 or ''}")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _Bs = [1, 2, 3, 5, 10, 20, 30, 60]
            _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.8))
            _txt = []
            for _V, _n, _c in ((V_BAG, "bagging", "tab:blue"), (V_RF, "random forest", "tab:green")):
                _rng = np.random.default_rng(1)
                _acc = [np.mean([np.mean(majority(_V[_rng.permutation(60)[:_B]]) == YBt) for _ in range(20)]) for _B in _Bs]
                _ax[0].plot(_Bs, _acc, "o-", color=_c, label=_n)
                _Z = (_V == 1).astype(float)                        # vote "film column" of every tree
                _C = np.corrcoef(_Z)
                _rho = (_C.sum() - 60) / (60 * 59)                  # mean correlation of two trees
                _b = np.arange(1, 201)
                _ax[1].plot(_b, ensemble_variance(_rho, 1.0, _b), color=_c, label=f"{_n}: ρ = {_rho:.2f}")
                _txt.append(f"{_n}: one tree {np.mean(_V == YBt):.1%}, 60 trees {np.mean(majority(_V) == YBt):.1%}, ρ = {_rho:.2f}")
            _ax[1].plot(_b, 1 / _b, "k:", label="independent trees: ρ = 0")
            _ax[0].set_xscale("log")
            _ax[1].set_xscale("log")
            _ax[0].set_xlabel("number of trees B")
            _ax[0].set_ylabel("accuracy on the new image B")
            _ax[1].set_xlabel("number of trees B")
            _ax[1].set_ylabel("variance / σ²")
            _ax[0].legend()
            _ax[1].legend()
            _fig.tight_layout()
            _parts += [_fig, mo.md("; ".join(_txt) + ".\n\n**Think:** why does the accuracy stop rising after about 30 trees? "
                                   "The random forest tests only 2 random descriptors at each split. Each of its trees "
                                   "is not better than a bagged tree. Why is the forest better?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex7_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def bootstrap(n, rng):
        return rng.integers(0, n, n)

    def majority(V):
        votes = np.array([(V == k).sum(0) for k in range(3)])
        return np.argmax(votes, 0)

    def ensemble_variance(rho, sigma2, B):
        return rho * sigma2 + (1 - rho) * sigma2 / B
    ```
    More trees remove only the second term. The floor ρσ² stays, because all trees see almost the same rows and
    choose the same strong descriptors. The random forest lowers ρ: a strong descriptor cannot be the root of every tree.
    A bootstrap sample contains about 63 % of the rows; the other 37 % are a free test set for that tree (out-of-bag score).
    """)})
    return


@app.cell(hide_code=True)
def s4_2(mo):
    mo.md(r"""
    ## 4.2 · ✏️ Exercise 8 (⭐): boosting with stumps and early stopping

    Bagging averages deep trees (low bias, high variance). **Boosting** adds shallow trees (high bias, low variance)
    one after the other. For the squared loss the negative gradient is the residual, so each new tree fits the residuals:

    $$F_0 = \bar y,\qquad r = y - F_{m-1},\qquad F_m = F_{m-1} + \eta\, h_m(\mathbf x),\quad h_m = \text{stump fitted to } r .$$

    The task here: predict the indicator "this candidate is a film column" (1 or 0) from the six invariant descriptors.
    Training image: C. Validation image: D.

    Write `boost(X, y, Xv, yv, eta, M)`. It returns the training MSE and the validation MSE after each of `M` stages.
    `ref_stump(X, r)` returns `(j, t, a, b)`: the stump predicts `a` where `X[:, j] <= t`, else `b`.
    """)
    return


@app.cell
def ex8(np, ref_stump):
    # ✏️ YOUR TURN: replace each `None`.
    def boost(X, y, Xv, yv, eta, M):
        F = np.full(len(y), y.mean())             # prediction for the training rows
        Fv = np.full(len(yv), y.mean())           # prediction for the validation rows
        mse_train, mse_val = [], []
        for m in range(M):
            j, t, a, b = None                     # stump for the residuals
            F = None                              # update the training prediction
            Fv = None                             # update the validation prediction
            mse_train.append(np.mean((y - F) ** 2))
            mse_val.append(np.mean((yv - Fv) ** 2))
        return np.array(mse_train), np.array(mse_val)

    return (boost,)


@app.cell(hide_code=True)
def ex8_check(INV, X, boost, checklist, g, mo, np, plt, ref_boost, safe, same, y):
    _Xc, _yc = X[g == 2][:, INV], (y[g == 2] == 1).astype(float)
    _Xd, _yd = X[g == 3][:, INV], (y[g == 3] == 1).astype(float)
    _r, _e = safe(boost, _Xc, _yc, _Xd, _yd, 0.3, 20)
    if _r is None and (_e is None or "NoneType" in _e):
        _out = checklist([], "Optional ⭐: write `boost` in the cell above.")
    else:
        _ref = ref_boost(_Xc, _yc, _Xd, _yd, 0.3, 20)
        _items = [(_r is not None and same(_r[0], _ref[0]), f"training MSE of 20 stages with η = 0.3 {_e or ''}"),
                  (_r is not None and same(_r[1], _ref[1]), "validation MSE of 20 stages with η = 0.3")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _fig, _ax = plt.subplots(figsize=(8, 3.8))
            _txt = []
            for _eta, _c in ((0.03, "tab:green"), (0.1, "tab:blue"), (0.3, "tab:orange"), (1.0, "tab:red")):
                _tr, _va = boost(_Xc, _yc, _Xd, _yd, _eta, 300)
                _b = int(np.argmin(_va))
                _ax.plot(np.arange(1, 301), _va, color=_c, label=f"η = {_eta}: validation")
                _ax.plot(np.arange(1, 301), _tr, ":", color=_c)
                _ax.plot(_b + 1, _va[_b], "o", color=_c)
                _txt.append(f"| {_eta} | {_b + 1} | {_va[_b]:.4f} | {_va[-1]:.4f} | {_tr[-1]:.4f} |")
            _ax.set_xscale("log")
            _ax.set_xlabel("boosting stage m")
            _ax.set_ylabel("MSE (dotted: training image C)")
            _ax.legend(fontsize=8)
            _fig.tight_layout()
            _parts += [_fig, mo.md("| η | best stage | validation MSE there | validation MSE at stage 300 | training MSE at stage 300 |\n"
                                   "|---|---|---|---|---|\n" + "\n".join(_txt) +
                                   "\n\n**Think:** the training MSE always falls. Which curve tells you when to stop? "
                                   "May you report the validation MSE at the best stage as the final score of the model?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex8_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def boost(X, y, Xv, yv, eta, M):
        F = np.full(len(y), y.mean())
        Fv = np.full(len(yv), y.mean())
        mse_train, mse_val = [], []
        for m in range(M):
            j, t, a, b = ref_stump(X, y - F)
            F = F + eta * np.where(X[:, j] <= t, a, b)
            Fv = Fv + eta * np.where(Xv[:, j] <= t, a, b)
            mse_train.append(np.mean((y - F) ** 2))
            mse_val.append(np.mean((yv - Fv) ** 2))
        return np.array(mse_train), np.array(mse_val)
    ```
    Boosting is gradient descent in function space: the step is a small tree, η is the learning rate.
    The number of stages is the complexity. Stop at the minimum of the validation curve (early stopping).
    The stage was chosen with image D, so the score on image D is optimistic: the final score needs a third image.
    """)})
    return


@app.cell(hide_code=True)
def s5(mo):
    mo.md(r"""
    ---
    # Part 5 · Validate: which score do you report?

    ## 5.1 · ✏️ Exercise 9: random folds against image folds

    The claim of this model is: *it labels the columns of a **new image** of this specimen.*
    A random split puts candidates of the same image in the training and in the test set. Then the test measures another claim.

    1. Write `group_folds(g)`: a list with one array of row indices per image (the test rows of each fold).
    2. Write `cv_predict(fit_predict, X, y, folds)`: for each fold, train on all other rows and predict the fold.
       `fit_predict(X_train, y_train, X_test)` returns the predicted classes. Return one prediction per row.
    """)
    return


@app.cell
def ex9(np):
    # ✏️ YOUR TURN: replace each `None`.
    def group_folds(g):
        return None

    def cv_predict(fit_predict, X, y, folds):
        pred = np.empty(len(y), int)
        for test in folds:
            train = np.setdiff1d(np.arange(len(y)), test)
            pred[test] = None
        return pred

    return cv_predict, group_folds


@app.cell
def models(forest_predict, np, ref_forest, ref_grow, tree_predict):
    # The baseline ladder: every model is a function fit_predict(X_train, y_train, X_test).
    def m_majority(Xa, ya, Xb):
        return np.full(len(Xb), np.bincount(ya).argmax())

    def m_linear(Xa, ya, Xb):
        # least squares on one-hot targets, with standardised features (learned on the training rows only)
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
        W = np.linalg.lstsq(np.c_[np.ones(len(Xa)), (Xa - mu) / sd], np.eye(3)[ya], rcond=None)[0]
        return np.argmax(np.c_[np.ones(len(Xb)), (Xb - mu) / sd] @ W, 1)

    def m_tree(Xa, ya, Xb):
        return tree_predict(ref_grow(Xa, ya, 30), Xb)

    def m_forest(Xa, ya, Xb):
        return forest_predict(ref_forest(Xa, ya, 30, 10, 2, np.random.default_rng(0)), Xb)

    MODELS = {"majority class": m_majority, "linear": m_linear, "one deep tree": m_tree, "random forest": m_forest}
    return MODELS, m_forest, m_majority


@app.cell(hide_code=True)
def ex9_check(INV, MODELS, RAW, X, checklist, cv_predict, g, group_folds, m_majority, mo, np, plt, ref_cv_predict, safe, y):
    _f, _e1 = safe(group_folds, g)
    _gf = [np.nonzero(g == _k)[0] for _k in range(4)]
    _p, _e2 = safe(cv_predict, m_majority, X, y, _gf)
    if _f is None and _e1 is None:
        _out = checklist([], "Write `group_folds` and `cv_predict` in the cell above.")
    else:
        _okf = (_f is not None and len(_f) == 4 and all(len(np.unique(g[np.asarray(a)])) == 1 for a in _f)
                and sorted(np.concatenate([np.asarray(a) for a in _f]).tolist()) == list(range(len(g))))
        _items = [(bool(_okf), f"`group_folds`: 4 folds, one image per fold, every row once {_e1 or ''}"),
                  (_e2 is None and _p is not None and bool(np.all(_p == ref_cv_predict(m_majority, X, y, _gf))),
                   f"`cv_predict` with the majority model is equal to the reference {_e2 or ''}")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _perm = np.random.default_rng(0).permutation(len(y))
            _rf = [_perm[_i::5] for _i in range(5)]                 # 5 random folds
            _res = {}
            for _sn, _J in (("raw (amp, bg, sigma)", RAW), ("invariant (6 descriptors)", INV)):
                for _mn, _fp in MODELS.items():
                    _res[_sn, _mn] = (np.mean(cv_predict(_fp, X[:, _J], y, _rf) == y),
                                      np.mean(cv_predict(_fp, X[:, _J], y, group_folds(g)) == y))
            _fig, _ax = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
            for _a, _sn in zip(_ax, ("raw (amp, bg, sigma)", "invariant (6 descriptors)")):
                _v = np.array([_res[_sn, _mn] for _mn in MODELS])
                _a.bar(np.arange(4) - 0.2, _v[:, 0], 0.4, label="5 random folds", color="tab:green")
                _a.bar(np.arange(4) + 0.2, _v[:, 1], 0.4, label="4 image folds", color="tab:red")
                _a.set_xticks(np.arange(4), list(MODELS), fontsize=8)
                _a.set_title(_sn)
                _a.set_ylim(0.3, 1)
            _ax[0].set_ylabel("accuracy")
            _ax[0].legend()
            _fig.tight_layout()
            _rows = "\n".join(f"| {sn} | {mn} | {100 * a:.1f} % | {100 * b:.1f} % |" for (sn, mn), (a, b) in _res.items())
            _parts += [_fig, mo.md("| features | model | random folds | image folds |\n|---|---|---|---|\n" + _rows +
                                   "\n\n**Think:** the random forest on the raw features loses about 20 points on a new image. "
                                   "What did it learn? Which row of the table do you report in a paper? "
                                   "Why does the linear model gain more from good descriptors than from a bigger model?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex9_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def group_folds(g):
        return [np.nonzero(g == k)[0] for k in np.unique(g)]

    def cv_predict(fit_predict, X, y, folds):
        pred = np.empty(len(y), int)
        for test in folds:
            train = np.setdiff1d(np.arange(len(y)), test)
            pred[test] = fit_predict(X[train], y[train], X[test])
        return pred
    ```
    Peak height, background and peak width depend on the gain, the dose and the magnification of each image.
    With random folds the forest finds the image again from these numbers, and each image has its own intensity levels.
    On a new image this knowledge is useless. The invariant descriptors lose only a few points: the score you report
    is the score on image folds, and the first fix is a better descriptor, not a bigger model.
    """)})
    return


@app.cell(hide_code=True)
def s6(mo):
    mo.md(r"""
    ---
    # Part 6 · Explain: which descriptors does the forest use?

    ## 6.1 · ✏️ Exercise 10: permutation importance

    Two ways to rank descriptors:

    - **Impurity importance (MDI):** add $\frac{N_\text{node}}{N}\Delta$ of every split that uses the descriptor.
      It is free, but it is computed on the training rows.
    - **Permutation importance:** shuffle one column of **held-out** data, and measure how much the accuracy falls.

    Write `permutation_importance(predict, X, y, rng, n_rep)`: for each column `j`, repeat `n_rep` times:
    copy `X`, replace column `j` by `rng.permutation(...)` of itself, and compute the accuracy.
    Return the mean decrease of the accuracy (accuracy of the intact data minus accuracy with the shuffled column) per column.

    The test adds two columns to the six invariant descriptors: pure **noise**, and an exact **copy of `d4`**.
    """)
    return


@app.cell
def ex10(np):
    # ✏️ YOUR TURN: replace each `None`.
    def permutation_importance(predict, X, y, rng, n_rep=5):
        base = np.mean(predict(X) == y)
        out = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            for _rep in range(n_rep):
                Xp = X.copy()
                Xp[:, j] = None
                out[j] += None
        return out

    return (permutation_importance,)


@app.cell
def importance_models(INV, X, col, forest_predict, g, np, ref_forest, y):
    # Training: images A, C, D. Held out: image B.
    _rng = np.random.default_rng(1)
    X8 = np.c_[X[:, INV], _rng.normal(size=len(y)), X[:, col("d4")]]
    N8 = ["I_rel", "d1", "d4", "d_spread", "n_near", "psi4", "noise", "d4 copy"]
    TR8, TE8 = g != 1, g == 1
    F8 = ref_forest(X8[TR8], y[TR8], 60, 10, 3, np.random.default_rng(0))          # with noise and copy
    F7 = ref_forest(X8[TR8][:, :7], y[TR8], 60, 10, 3, np.random.default_rng(0))   # without the copy

    def mdi(T, F):
        out = np.zeros(F)

        def walk(t, n0):
            if "j" in t:
                out[t["j"]] += t["n"] / n0 * t["gain"]
                walk(t["left"], n0)
                walk(t["right"], n0)

        for t in T:
            walk(t, t["n"])
        return out / out.sum()

    print(f"accuracy on image B: {np.mean(forest_predict(F8, X8[TE8]) == y[TE8]):.3f}")
    return F7, F8, N8, TE8, TR8, X8, mdi


@app.cell(hide_code=True)
def ex10_check(F7, F8, N8, TE8, TR8, X8, checklist, forest_predict, mdi, mo, np, permutation_importance, plt, ref_perm, safe, y):
    _pf = lambda Xq: forest_predict(F8, Xq)
    _Xt, _yt = X8[TE8], y[TE8]
    _r, _e = safe(permutation_importance, _pf, _Xt, _yt, np.random.default_rng(0), 3)
    if _r is None and (_e is None or "NoneType" in _e or "None" in _e):
        _out = checklist([], "Write `permutation_importance` in the cell above.")
    else:
        _ref = ref_perm(_pf, _Xt, _yt, np.random.default_rng(0), 3)
        _toy = lambda Xq: (Xq[:, 0] > 0).astype(int)
        _Xtoy = np.c_[np.linspace(-1, 1, 400), np.linspace(-1, 1, 400)[::-1]]
        _rt, _ = safe(permutation_importance, _toy, _Xtoy, _toy(_Xtoy), np.random.default_rng(0), 5)
        _items = [(_rt is not None and np.shape(_rt) == (2,) and 0.35 < _rt[0] < 0.65 and abs(_rt[1]) < 1e-12,
                   "toy model that reads only column 0: importance about 0.5 for column 0, exactly 0 for column 1"),
                  (_r is not None and np.shape(_r) == (8,) and bool(np.allclose(_r, _ref, atol=1e-9)),
                   f"forest on image B: equal to the reference (same random numbers) {_e or ''}")]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _rng = np.random.default_rng(2)
            _pte = permutation_importance(_pf, _Xt, _yt, _rng, 5)
            _ptr = permutation_importance(_pf, X8[TR8], y[TR8], _rng, 2)
            _p7 = permutation_importance(lambda Xq: forest_predict(F7, Xq), _Xt[:, :7], _yt, _rng, 5)
            _m = mdi(F8, 8)
            _fig, _ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
            _yy = np.arange(8)[::-1]
            _ax[0].barh(_yy, _m, color="0.5")
            _ax[0].set_title("impurity importance (MDI)")
            _ax[1].barh(_yy + 0.2, _ptr, 0.4, color="tab:green", label="training images")
            _ax[1].barh(_yy - 0.2, _pte, 0.4, color="tab:red", label="new image B")
            _ax[1].set_title("permutation importance")
            _ax[1].legend(fontsize=8)
            _ax[2].barh(_yy[:7], _p7, color="tab:red")
            _ax[2].set_title("permutation, forest without the copy")
            _ax[0].set_yticks(_yy, N8)
            for _a in _ax[1:]:
                _a.set_xlabel("decrease of the accuracy")
            _fig.tight_layout()
            _parts += [_fig, mo.md(f"MDI gives the noise column {100 * _m[6]:.0f} % of the credit; its permutation importance "
                                   f"on the new image is {_pte[6]:+.3f}. `d4`: {_pte[2]:.3f} with the copy in the model, "
                                   f"{_p7[2]:.3f} without it.\n\n"
                                   "**Think:** why does MDI give credit to noise? Why does `d4` look less important when a copy "
                                   "of it is in the table? Does a large importance say that the descriptor **causes** the class?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex10_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def permutation_importance(predict, X, y, rng, n_rep=5):
        base = np.mean(predict(X) == y)
        out = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            for _rep in range(n_rep):
                Xp = X.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                out[j] += (base - np.mean(predict(Xp) == y)) / n_rep
        return out
    ```
    - **MDI is biased.** Deep trees split on noise to make the training leaves pure; each such split counts.
      On held-out rows the noise column has an importance of zero.
    - **Correlated descriptors share the credit.** When you shuffle `d4`, the forest still reads the copy.
      Both look unimportant, but together they carry the information. Permute correlated descriptors together, or drop one.
    - Importance describes **the model**, not the physics. It tells you which numbers the forest reads, not what makes a column a film column.
    """)})
    return


@app.cell(hide_code=True)
def summary(mo):
    mo.md(r"""
    ---
    ## Summary

    | step | what you must remember |
    |---|---|
    | objects | smooth → local maxima → threshold; every parameter changes the rows of the table; a missed object is lost |
    | descriptors | ratios, distances and bond order are invariant to gain, offset and rotation; **test** the invariance |
    | tree | greedy split with the largest $\Delta = I(\text{parent}) - \frac{N_L}{N}I(L) - \frac{N_R}{N}I(R)$; no scaling, no extrapolation, high variance |
    | bagging, random forest | variance $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$; random feature subsets lower ρ |
    | boosting | fit a small tree to the negative gradient, add $\eta\,h$; stop at the minimum of the validation curve |
    | validation | folds by image, session or specimen; raw intensities identify the image and inflate random-fold scores |
    | baseline ladder | majority → linear → tree → forest on the same folds; good descriptors help the linear model most |
    | importance | MDI gives credit to noise; permutation on held-out groups; correlated descriptors share the credit |
    """)
    return


@app.cell(hide_code=True)
def refs(mo):
    mo.md(r"""
    ---
    ## Next week and references

    **Next: Unit 6, *Neural networks & backpropagation*.** The descriptors of today are made by hand. Next the model learns its own representation.

    **Data**

    - G. Hollyer, E. Stach, *STEM-Bank-26*, arXiv:2610.36086 (2026), CC BY 4.0: folder `haadf_SFO_LAO`,
      four HAADF-STEM images of one oxide film on LaAlO₃ (JEOL, 200 kV).
    - Processing of all data (alignment, column finding, Gaussian fits, labels): `05_features_trees/xp/prep/make_columns.py`
      in the course repository. The labels come from the position of a candidate (substrate, film, amorphous region).

    **Methods**

    - L. Breiman, *Bagging predictors*, Mach. Learn. **24**, 123 (1996). — L. Breiman, *Random forests*, Mach. Learn. **45**, 5 (2001).
    - L. Breiman, J. Friedman, R. Olshen, C. Stone, *Classification and Regression Trees*, Wadsworth (1984).
    - J. H. Friedman, *Greedy function approximation: a gradient boosting machine*, Ann. Stat. **29**, 1189 (2001).
    - L. Grinsztajn, E. Oyallon, G. Varoquaux, *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS (2022).
    - S. M. Lundberg et al., *From local explanations to global understanding with explainable AI for trees*, Nat. Mach. Intell. **2**, 56 (2020).
    - C. Strobl, A.-L. Boulesteix, A. Zeileis, T. Hothorn, *Bias in random forest variable importance measures*, BMC Bioinformatics **8**, 25 (2007).
    """)
    return


if __name__ == "__main__":
    app.run()
