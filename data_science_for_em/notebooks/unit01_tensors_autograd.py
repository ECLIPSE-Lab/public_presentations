# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "numpy", "matplotlib", "torch"]
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # Unit 1 · Hands-on: tensors and automatic differentiation

    **Data Science for Electron Microscopy** · Prof. Dr. Philipp Pelz · FAU Erlangen-Nürnberg · WS 26/27

    This notebook follows **slides 26–55** of the Unit 1 deck, one section per slide.
    Each section has the code from the slide, an explanation, and an EM example where one helps.

    | part | slides | content |
    |---|---|---|
    | Course outline | 26 | the 13 weeks |
    | **Data manipulation** | 27–42 | PyTorch tensors: create, reshape, index, compute, broadcast, save memory, convert |
    | **Automatic differentiation** | 43–53 | gradients, accumulation, non-scalar outputs, `detach`, control flow, exercises |
    | Wrap-up | 54–55 | next week, references |

    Cells marked ✏️ are exercises. A check under each exercise tells you if your answer is correct.
    Solutions are in closed boxes: open them only after you have tried.
    """)
    return


@app.cell(hide_code=True)
def marimo_intro(mo):
    mo.callout(mo.md(r"""
    ### How this notebook works (marimo is not Jupyter)

    - **Run a cell:** `Ctrl`/`Cmd` + `Enter`. `Shift` + `Enter` runs it and moves to the next cell.
    - **It is reactive.** When you change a cell or move a slider, every cell that *uses* its
      variables runs again by itself.
    - **One name, one cell.** Each variable is defined in exactly one cell. Names that start with an
      underscore (`_x`, `_fig`) stay private to their cell. This notebook uses them for throw-away values,
      so you can run each cell again and again without side effects.
    - **Your own copy:** to keep your changes, save your own copy of this notebook in molab.
    """), kind="info")
    return


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import time

    return mo, np, plt, time, torch


@app.cell(hide_code=True)
def helpers(mo, np, plt):
    def show_tensor(t, title="", ax=None, fmt="{:g}", cmap="viridis"):
        """Draw a small 1-D or 2-D tensor as a coloured grid with its values written in."""
        a = t.detach().cpu().float().numpy()
        if a.ndim == 1:
            a = a[None, :]
        if ax is None:
            _, ax = plt.subplots(figsize=(0.6 * a.shape[1] + 1, 0.6 * a.shape[0] + 0.8))
        ax.imshow(a, cmap=cmap)
        for (i, j), v in np.ndenumerate(a):
            ax.text(j, i, fmt.format(v), ha="center", va="center", color="white", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.25, lw=0))
        ax.set_xticks(range(a.shape[1]))
        ax.set_yticks(range(a.shape[0]))
        ax.set_title(f"{title}  shape {tuple(t.shape)}", fontsize=10)
        return ax


    def checklist(items, empty_msg=None):
        """items: list of (passed, text). Shows a green or orange box with one line per check."""
        if empty_msg is not None:
            return mo.callout(mo.md(empty_msg), kind="neutral")
        ok = all(p for p, _ in items)
        lines = [("✅ " if p else "❌ ") + t for p, t in items]
        head = "**All checks pass. Well done!**\n\n" if ok else ""
        return mo.callout(mo.md(head + "\n\n".join(lines)), kind="success" if ok else "warn")

    return checklist, show_tensor


@app.cell
def device_info(mo, torch):
    _cuda = torch.cuda.is_available()
    mo.md(f"""
    **This machine:** PyTorch {torch.__version__} ·
    GPU: {"**" + torch.cuda.get_device_name(0) + "**" if _cuda else "none (CPU only)"}
    """)
    return


@app.cell(hide_code=True)
def s26(mo):
    mo.md(r"""
    ---
    ## Slide 26 · Course outline

    | Wk | Date | Topic | Wk | Date | Topic |
    |---|---|---|---|---|---|
    | **1** | **23.10.2026** | **Python crash course & the EM data deluge** | 8 | 11.12.2026 | Small data: augmentation, transfer & self-supervision |
    | 2 | 30.10.2026 | Learning from EM data: signal formation, noise & loss | 9 | 18.12.2026 | Unsupervised learning, autoencoders & latent spaces |
    | 3 | 06.11.2026 | Linear algebra, PCA & spectral unmixing | 10 | 08.01.2027 | Attention, transformers & graph networks for EM |
    | 4 | 13.11.2026 | Regression, optimisation & honest validation | 11 | 15.01.2027 | Uncertainty, Gaussian processes & autonomous EM |
    | 5 | 20.11.2026 | From images to features: descriptors & tree ensembles | 12 | 22.01.2027 | Inverse problems I: regularisation, tomography & sensor fusion |
    | 6 | 27.11.2026 | Neural networks & backpropagation | 13 | 29.01.2027 | Inverse problems II: ptychography, generative priors & synthesis |
    | 7 | 04.12.2026 | CNNs & U-Nets for microscopy | — | 05.02.2027 | Exam revision |

    Today's two tools appear in every later week: **tensors** hold the data, and
    **automatic differentiation** computes the gradients that train every model (Week 4 onwards)
    and that solve every reconstruction (Weeks 12–13).
    """)
    return


@app.cell(hide_code=True)
def s27(mo):
    mo.md(r"""
    ---
    # Part 1 · Data manipulation (slides 27–42)

    ## Slide 27 · Why a tensor class?

    Every data-science task has two steps: **get** the data and **process** it. The basic container for
    both is the *n*-dimensional array, which deep-learning frameworks call a **tensor**
    (`torch.Tensor` in PyTorch). It works like NumPy's `ndarray`, with two additions:

    1. **Automatic differentiation:** the tensor can record how it was computed, and then give you
       derivatives (Part 2).
    2. **GPU acceleration:** the same code runs on a graphics card. NumPy runs only on the CPU.

    The cell below multiplies two 4096 × 4096 matrices (69 billion floating-point operations)
    on the CPU and, if there is one, on the GPU.
    """)
    return


@app.cell
def s27_gpu(mo, time, torch):
    _n = 4096
    _A = torch.randn(_n, _n)
    _t0 = time.perf_counter()
    _ = _A @ _A
    _t_cpu = time.perf_counter() - _t0
    _rows = [f"| CPU | {_t_cpu * 1e3:.0f} ms | 1× |"]
    if torch.cuda.is_available():
        _Ag = _A.to("cuda")                # copy the tensor to GPU memory
        _ = _Ag @ _Ag                       # warm-up: the first call loads the GPU kernels
        torch.cuda.synchronize()            # GPU work is asynchronous: wait before timing
        _t0 = time.perf_counter()
        _ = _Ag @ _Ag
        torch.cuda.synchronize()
        _t_gpu = time.perf_counter() - _t0
        _rows.append(f"| GPU | {_t_gpu * 1e3:.1f} ms | {_t_cpu / _t_gpu:.0f}× faster |")
        del _Ag, _
        torch.cuda.empty_cache()            # give the GPU memory back
    mo.md("| device | time for `A @ A` | speed-up |\n|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def s28(mo):
    mo.md(r"""
    ---
    ## Slide 28 · Getting started 1: axes and orders

    `import torch` gives you the library. A tensor with **one axis** is a *vector*, with **two axes**
    a *matrix*, and with **k > 2 axes** a *k-th order tensor*. EM data has all of them:
    """)
    return


@app.cell
def s28_orders(mo, torch):
    spectrum = torch.zeros(1024)                    # EELS spectrum: energy
    image = torch.zeros(512, 512)                   # HAADF image: y, x
    spectrum_image = torch.zeros(64, 64, 1024)      # EELS map: y, x, energy
    four_d_stem = torch.zeros(16, 16, 64, 64)       # 4D-STEM (small): scan y, scan x, detector ky, kx

    _rows = []
    for _name, _t in [("spectrum", spectrum), ("image", image),
                      ("spectrum_image", spectrum_image), ("four_d_stem", four_d_stem)]:
        _rows.append(f"| `{_name}` | {_t.ndim} | `{tuple(_t.shape)}` | {_t.numel():,} | "
                     f"{_t.numel() * _t.element_size() / 1e3:,.0f} kB |")
    mo.md("| tensor | axes (`ndim`) | `shape` | elements (`numel()`) | memory |\n|---|---|---|---|---|\n"
          + "\n".join(_rows)
          + "\n\nA real 4D-STEM scan of 256 × 256 positions with a 128 × 128 detector has "
          + f"{256**2 * 128**2:,} elements: **{256**2 * 128**2 * 4 / 1e9:.1f} GB** in float32.")
    return


@app.cell(hide_code=True)
def s29(mo):
    mo.md(r"""
    ---
    ## Slide 29 · Getting started 2: `arange`, `numel`, `shape`, `reshape`

    `torch.arange(12)` makes the values 0, 1, …, 11. By default a new tensor lives in main memory
    and is processed by the CPU. `numel()` gives the number of elements, and `shape` the length of each axis.

    `reshape` changes the shape and keeps the values and their order: it fills the new shape
    **row by row** ("row-major").
    """)
    return


@app.cell
def s29_code(torch):
    x = torch.arange(12, dtype=torch.float32)
    X = x.reshape(3, 4)

    print("x        =", x)
    print("x.numel()=", x.numel())
    print("x.shape  =", x.shape)
    print("X =\n", X)
    return X, x


@app.cell(hide_code=True)
def s29_view(mo):
    mo.md(r"""
    **Good to know:** `reshape` does not copy the data when it does not have to. `X` is a new *view*
    of the same memory as `x`. Change one, and the other changes too:
    """)
    return


@app.cell
def s29_view_code(X, mo, x):
    mo.md(f"""
    `x.data_ptr() == X.data_ptr()` → **{x.data_ptr() == X.data_ptr()}**: both names point to the same memory.
    """)
    return


@app.cell(hide_code=True)
def s30(mo):
    mo.md(r"""
    ---
    ## Slide 30 · Getting started 3: shape inference and initialisation

    Write `-1` for one axis, and PyTorch computes its length: for *n* elements and *h* rows,
    the number of columns is *w = n / h*. **Try it:** choose a number of rows.
    """)
    return


@app.cell
def s30_rows(mo):
    n_rows = mo.ui.dropdown(options=["1", "2", "3", "4", "6", "12"], value="3",
                             label="x.reshape(rows, -1) with rows =")
    n_rows
    return (n_rows,)


@app.cell
def s30_reshape_plot(mo, n_rows, show_tensor, x):
    _R = x.reshape(int(n_rows.value), -1)
    _ax = show_tensor(_R, f"x.reshape({n_rows.value}, -1)")
    _ax.figure.tight_layout()
    mo.vstack([mo.md(f"`-1` became **{_R.shape[1]}**, because 12 / {n_rows.value} = {_R.shape[1]}. "
                     "The numbers still run row by row."), _ax.figure])
    return


@app.cell(hide_code=True)
def s30_init(mo):
    mo.md(r"""
    Four common ways to fill a new tensor. `torch.randn` draws from a standard normal distribution;
    `torch.manual_seed` makes the random numbers repeatable.
    """)
    return


@app.cell
def s30_init_code(plt, show_tensor, torch):
    torch.manual_seed(0)
    zeros_234 = torch.zeros((2, 3, 4))        # all 0, three axes
    ones_234 = torch.ones((2, 3, 4))          # all 1
    noise_34 = torch.randn(3, 4)              # standard normal: mean 0, standard deviation 1
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])   # your own values

    _fig, _axes = plt.subplots(1, 2, figsize=(9, 2.6))
    show_tensor(noise_34, "torch.randn(3, 4)", _axes[0], fmt="{:.2f}", cmap="coolwarm")
    show_tensor(Y, "torch.tensor([[2, 1, 4, 3], ...])", _axes[1])
    _fig.tight_layout()
    _fig
    return (Y,)


@app.cell(hide_code=True)
def s31(mo):
    mo.md(r"""
    ---
    ## Slides 31–32 · Indexing and slicing

    Indexing works as for Python lists: the first element is **0**, `-1` is the last one, and
    `start:stop` is a slice that **excludes** `stop`. One index or slice acts on **axis 0** (the rows).

    You can also **assign** through an index: `X[1, 2] = 17` changes one element, and `X[:2, :] = 12`
    changes the first two rows (`:` means "all elements along this axis").

    The cell below works on a copy, `X.clone()`. Assignment changes a tensor *in place*: without the
    copy, it would also change `X` and `x` (they share memory, see slide 29).
    """)
    return


@app.cell
def s31_code(X, plt, show_tensor):
    print("X[-1]  =", X[-1])
    print("X[1:3] =\n", X[1:3])

    X_edit = X.clone()
    _steps = [("X", X.clone())]
    X_edit[1, 2] = 17
    _steps.append(("after X[1, 2] = 17", X_edit.clone()))
    X_edit[:2, :] = 12
    _steps.append(("after X[:2, :] = 12", X_edit.clone()))

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 2.6))
    for _ax, (_title, _t) in zip(_axes, _steps):
        show_tensor(_t, _title, _ax)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def stem_md(mo):
    mo.md(r"""
    ### EM example: an image is a 2-D tensor

    The cell below builds a synthetic HAADF-STEM image: bright 5 × 5-pixel atomic columns on a
    20-pixel lattice, a background of 50 counts, and **Poisson noise** (an electron detector counts
    discrete electrons; Week 2 explains why this gives Poisson statistics).
    It uses only today's tools: `arange`, broadcasting and logical operations.
    """)
    return


@app.cell
def stem_code(plt, torch):
    def make_stem_image(n=256, spacing=20, half_width=2, column_counts=800.0,
                        background=50.0, seed=42):
        """Return (clean, noisy) synthetic HAADF-STEM images as float32 tensors."""
        gen = torch.Generator().manual_seed(seed)
        centres = torch.arange(spacing // 2, n, spacing)                        # column centres
        near = (torch.arange(n)[:, None] - centres[None, :]).abs() <= half_width
        on_lattice = near.any(dim=1)                                            # (n,) bool
        columns = on_lattice[:, None] & on_lattice[None, :]                     # (n, n) bool
        clean = background + column_counts * columns.float()
        noisy = torch.poisson(clean, generator=gen)                             # count electrons
        return clean, noisy


    N = 256
    clean_image, stem_image = make_stem_image(N)

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for _ax, _img, _title in zip(_axes, [clean_image, stem_image], ["noise-free", "measured (Poisson counts)"]):
        _im = _ax.imshow(_img, cmap="gray")
        _ax.set_title(_title)
        _fig.colorbar(_im, ax=_ax, label="counts")
    _fig.tight_layout()
    _fig
    return N, clean_image, make_stem_image, stem_image


@app.cell(hide_code=True)
def ex_roi_md(mo):
    mo.md(r"""
    ### ✏️ Exercise: crop a region of interest

    1. Crop the 64 × 64 pixels in the centre of `stem_image` into `roi` (use `N // 2`: a slice index must be an integer).
    2. Store the mean of `roi` as a Python number in `roi_mean` (hint: `.mean()` gives a 0-D tensor; `.item()` turns it into a number).
    """)
    return


@app.cell
def ex_roi():
    # ✏️ YOUR TURN: replace each `None`, then run the cell.
    roi = None
    roi_mean = None
    return roi, roi_mean


@app.cell(hide_code=True)
def ex_roi_check(N, checklist, roi, roi_mean, stem_image, torch):
    if roi is None:
        _out = checklist([], "Fill in `roi` in the cell above. The check runs automatically.")
    else:
        _c = N // 2
        _ref = stem_image[_c - 32:_c + 32, _c - 32:_c + 32]
        _shape_ok = tuple(roi.shape) == (64, 64)
        _items = [
            (_shape_ok, f"`roi.shape` is `{tuple(roi.shape)}` (want `(64, 64)`)"
             + ("  · the stop index is excluded: no `+ 1`" if 65 in tuple(roi.shape) else "")),
            (_shape_ok and torch.equal(roi, _ref), "`roi` is the centre crop of `stem_image`"),
            (isinstance(roi_mean, float) and abs(roi_mean - _ref.mean().item()) < 1e-3,
             f"`roi_mean` is a Python float equal to the ROI mean ({_ref.mean().item():.1f})"),
        ]
        _out = checklist(_items)
    _out
    return


@app.cell(hide_code=True)
def ex_roi_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    _c = N // 2                                   # 128
    roi = stem_image[_c - 32:_c + 32, _c - 32:_c + 32]
    roi_mean = roi.mean().item()
    ```
    `roi` is a view of `stem_image`: no pixels are copied.
    """)})
    return


@app.cell(hide_code=True)
def s33(mo):
    mo.md(r"""
    ---
    ## Slide 33 · Operations 1: elementwise functions

    A *unary* elementwise operation applies a function $f:\mathbb{R}\rightarrow\mathbb{R}$ to every
    element. The slide's example is $e^x$:
    """)
    return


@app.cell
def s33_code(torch, x):
    print("torch.exp(x) =", torch.exp(x))
    return


@app.cell(hide_code=True)
def s33_em(mo):
    mo.md(r"""
    **EM example: the log scale for diffraction.** The squared modulus of the Fourier transform of the
    image, $|\mathcal{F}(I)|^2$, looks like a diffraction pattern. Its central spot is millions of times
    brighter than the Bragg spots. A linear display shows only the centre. The elementwise
    `torch.log1p` (that is, $\log(1 + I)$) makes the weak spots visible.
    """)
    return


@app.cell
def s33_fft(mo, plt, stem_image, torch):
    _pattern = torch.fft.fftshift(torch.fft.fft2(stem_image)).abs() ** 2   # elementwise |.|²
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.3))
    _axes[0].imshow(_pattern, cmap="magma")
    _axes[0].set_title("linear: |F(image)|²")
    _axes[1].imshow(torch.log1p(_pattern), cmap="magma")
    _axes[1].set_title("elementwise log1p(|F(image)|²)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Dynamic range: max / median = **{(_pattern.max() / _pattern.median()).item():.1e}**.")])
    return


@app.cell(hide_code=True)
def s34(mo):
    mo.md(r"""
    ---
    ## Slide 34 · Operations 2: elementwise binary operators

    A *binary* elementwise operation, $f:\mathbb{R},\mathbb{R}\rightarrow\mathbb{R}$, takes the
    elements at the same position in two tensors of the same shape. The slide calls the tensors `x` and `y`;
    here they are `p` and `q`, because `x` is already defined.
    """)
    return


@app.cell
def s34_code(mo, torch):
    p = torch.tensor([1.0, 2, 4, 8])
    q = torch.tensor([2, 2, 2, 2])

    mo.md(f"""
    | expression | result | dtype |
    |---|---|---|
    | `p` | `{p.tolist()}` | {p.dtype} |
    | `q` | `{q.tolist()}` | {q.dtype} |
    | `p + q` | `{(p + q).tolist()}` | {(p + q).dtype} |
    | `p - q` | `{(p - q).tolist()}` | {(p - q).dtype} |
    | `p * q` | `{(p * q).tolist()}` | {(p * q).dtype} |
    | `p / q` | `{(p / q).tolist()}` | {(p / q).dtype} |
    | `p ** q` | `{(p ** q).tolist()}` | {(p ** q).dtype} |

    `q` holds integers (`int64`) because all its values are written without a decimal point. When you
    combine it with the float tensor `p`, PyTorch converts the result to float (*type promotion*).
    """)
    return


@app.cell(hide_code=True)
def s35(mo):
    mo.md(r"""
    ---
    ## Slide 35 · Operations 3: concatenation

    `torch.cat` joins tensors along an existing axis. Along `dim=0` the row counts add up
    (3 + 3 = 6 rows); along `dim=1` the column counts add up (4 + 4 = 8 columns).
    """)
    return


@app.cell
def s35_code(X, Y, plt, show_tensor, torch):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    show_tensor(torch.cat((X, Y), dim=0), "torch.cat((X, Y), dim=0)", _axes[0])
    show_tensor(torch.cat((X, Y), dim=1), "torch.cat((X, Y), dim=1)", _axes[1])
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def s35_em(mo):
    mo.md(r"""
    **EM example: a frame series.** Fast detectors record many short frames of the same area.
    `torch.stack` puts the frames on a **new** axis (shape: frames × y × x), and the mean over
    that axis adds up the dose. For Poisson data the relative noise falls as $1/\sqrt{\text{frames}}$.
    **Try it:** change the number of frames.
    """)
    return


@app.cell
def s35_frames_ui(mo):
    n_frames = mo.ui.slider(start=1, stop=64, value=1, show_value=True, label="number of frames")
    n_frames
    return (n_frames,)


@app.cell
def s35_frames(N, make_stem_image, mo, n_frames, plt, torch):
    _clean, _ = make_stem_image(N, column_counts=8.0, background=0.5)       # one low-dose frame
    _gen = torch.Generator().manual_seed(1)
    _frames = torch.stack([torch.poisson(_clean, generator=_gen) for _ in range(n_frames.value)])
    _mean = _frames.mean(dim=0)
    _bg = _clean == _clean.min()
    _rel = (_mean[_bg].std() / _mean[_bg].mean()).item()
    _fig, _ax = plt.subplots(figsize=(4.6, 4.6))
    _ax.imshow(_mean[:96, :96], cmap="gray")
    _ax.set_title(f"mean of {n_frames.value} frame(s), 96×96 crop")
    _ax.axis("off")
    mo.hstack([_fig, mo.md(f"""
    `torch.stack(...)` shape: `{tuple(_frames.shape)}`

    relative noise in the background:
    **{_rel:.2f}** (Poisson prediction {1 / (0.5 * n_frames.value) ** 0.5:.2f})
    """)], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def s36(mo):
    mo.md(r"""
    ---
    ## Slide 36 · Operations 4: logical operations and sums

    A comparison such as `X == Y` gives a tensor of `True`/`False` values, one per element.
    `X.sum()` adds all elements and gives a tensor with a single element.
    """)
    return


@app.cell
def s36_code(X, Y):
    print("X == Y =\n", X == Y)
    print("X.sum() =", X.sum())
    return


@app.cell(hide_code=True)
def s36_em(mo):
    mo.md(r"""
    **EM example: a threshold mask.** `stem_image > threshold` marks the pixels that belong to atomic
    columns. Because `True` counts as 1, `.sum()` of the mask counts those pixels.
    **Try it:** move the threshold close to the background (50 counts) or to the columns (850 counts), and watch the wrong pixels appear.
    """)
    return


@app.cell
def s36_thr_ui(mo):
    threshold = mo.ui.slider(start=0, stop=900, step=10, value=400, show_value=True, label="threshold (counts)")
    threshold
    return (threshold,)


@app.cell
def s36_thr(clean_image, mo, plt, stem_image, threshold):
    _mask = stem_image > threshold.value
    _truth = clean_image > clean_image.min()
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4.3))
    _axes[0].imshow(_mask[:96, :96], cmap="gray")
    _axes[0].set_title(f"stem_image > {threshold.value}")
    _axes[1].imshow((_mask != _truth)[:96, :96], cmap="Reds")
    _axes[1].set_title("wrong pixels (mask != truth)")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"`mask.sum()` = **{int(_mask.sum())}** pixels marked · true column pixels: {int(_truth.sum())} · "
        f"wrong: **{int((_mask != _truth).sum())}** · `mask.dtype` = {_mask.dtype}")])
    return


@app.cell(hide_code=True)
def s37(mo):
    mo.md(r"""
    ---
    ## Slide 37 · Broadcasting

    Broadcasting lets an elementwise operation combine tensors of **different shapes**, in two steps:

    1. Line up the shapes from the right. A missing axis counts as length 1.
    2. Copy (virtually) every **length-1** axis until it matches the other tensor. Then operate elementwise.

    Two lengths are compatible if they are equal or if one of them is 1. Otherwise you get an error.
    """)
    return


@app.cell
def s37_code(plt, show_tensor, torch):
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))

    _fig, _axes = plt.subplots(1, 3, figsize=(9, 2.8), gridspec_kw={"width_ratios": [1, 2, 2]})
    show_tensor(a, "a", _axes[0])
    show_tensor(b, "b", _axes[1])
    show_tensor(a + b, "a + b", _axes[2])
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def s37_try(mo):
    mo.md(r"""
    **Try it:** pick two shapes and see the result shape, or the error PyTorch gives.
    """)
    return


@app.cell
def s37_ui(mo):
    _shapes = ["(3, 1)", "(1, 2)", "(3, 2)", "(2,)", "(3,)", "(4, 1, 2)", "(4, 3, 1)", "(1,)"]
    shape_a = mo.ui.dropdown(options=_shapes, value="(3, 1)", label="shape of a")
    shape_b = mo.ui.dropdown(options=_shapes, value="(1, 2)", label="shape of b")
    mo.hstack([shape_a, shape_b], justify="start", gap=2)
    return shape_a, shape_b


@app.cell
def s37_result(mo, shape_a, shape_b, torch):
    import ast as _ast

    _sa = _ast.literal_eval(shape_a.value)
    _sb = _ast.literal_eval(shape_b.value)
    try:
        _res = torch.ones(_sa) + torch.ones(_sb)
        _out = mo.callout(mo.md(f"`torch.ones{_sa} + torch.ones{_sb}` → shape **`{tuple(_res.shape)}`**"), kind="success")
    except RuntimeError as _err:
        _out = mo.callout(mo.md(f"`torch.ones{_sa} + torch.ones{_sb}` → **error**:\n\n`{_err}`"), kind="danger")
    _out
    return


@app.cell(hide_code=True)
def s37_em(mo):
    mo.md(r"""
    **EM example: scan stripes.** In a scanned image, each scan line can get its own offset. The row
    means, `img.mean(dim=1, keepdim=True)`, have shape (64, 1). Subtracting them from the (64, 64) image
    broadcasts one value across each row and removes the stripes.
    """)
    return


@app.cell
def s37_stripes(plt, torch):
    _gen = torch.Generator().manual_seed(7)
    _clean = torch.poisson(torch.full((64, 64), 200.0), generator=_gen)
    _stripes = torch.randn(64, 1, generator=_gen) * 50                 # one offset per row: (64, 1)
    _stripy = _clean + _stripes                                          # (64, 64) + (64, 1)
    _corrected = _stripy - _stripy.mean(dim=1, keepdim=True) + _clean.mean()

    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for _ax, _img, _t in zip(_axes, [_clean, _stripy, _corrected], ["clean", "with stripes", "row means removed"]):
        _ax.imshow(_img, cmap="gray", vmin=100, vmax=300)
        _ax.set_title(_t)
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def s38(mo):
    mo.md(r"""
    ---
    ## Slides 38–39 · Saving memory

    `Y = X + Y` computes a **new** tensor, puts it into **new memory**, and then points the name `Y`
    at it. Python's `id()` shows this: it changes. This is a problem when

    - you update the parameters of a model many times per second (each update allocates memory), and
    - other variables still point to the old tensor and do not see the update.

    **In-place** operations write into the existing memory: `Z[:] = X + Y` or `X += Y`.
    """)
    return


@app.cell
def s38_code(mo, torch):
    _X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    _Y = torch.ones(3, 4)
    _rows = []

    _before = id(_Y)
    _Y = _Y + _X
    _rows.append(("`Y = Y + X`", _before == id(_Y)))

    _Z = torch.zeros_like(_Y)
    _before = id(_Z)
    _Z[:] = _X + _Y
    _rows.append(("`Z[:] = X + Y`", _before == id(_Z)))

    _before = id(_X)
    _X += _Y
    _rows.append(("`X += Y`", _before == id(_X)))

    mo.md("| operation | same memory afterwards? |\n|---|---|\n"
          + "\n".join(f"| {_op} | {'✅ yes (in place)' if _same else '❌ no (new tensor)'} |" for _op, _same in _rows))
    return


@app.cell(hide_code=True)
def s38_note(mo):
    mo.callout(mo.md(r"""
    **Two warnings about in-place operations.**
    `Z[:] = X + Y` still computes `X + Y` in a temporary tensor first; only the result goes into `Z`.
    And autograd (Part 2) refuses some in-place changes to tensors it needs for the gradient.
    Use in-place operations where memory matters, for example on large 4D-STEM arrays on a GPU.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def s40(mo):
    mo.md(r"""
    ---
    ## Slide 40 · Conversion to other Python objects

    - `X.numpy()` turns a (CPU) tensor into a NumPy array, and `torch.from_numpy(A)` does the reverse.
      Both **share memory**: no data is copied.
    - `.item()`, `float(...)` or `int(...)` turn a one-element tensor into a Python number.

    In EM, this is the bridge to file readers: `tifffile.imread(...)` or `hyperspy` give you a NumPy
    array, and `torch.from_numpy` makes it a tensor without a copy.
    """)
    return


@app.cell
def s40_code(X, np, torch):
    _arr = np.zeros(3)
    _t = torch.from_numpy(_arr)
    _arr[0] = 5.0                          # change the NumPy array ...
    print("tensor after changing the array:", _t)    # ... and the tensor changes too
    print("type(X.numpy()):", type(X.numpy()))

    _s = torch.tensor([3.5])
    print("_s.item() =", _s.item(), " float(_s) =", float(_s), " int(_s) =", int(_s))
    return


@app.cell(hide_code=True)
def s40_note(mo):
    mo.md(r"""
    Two cases need one extra step: a tensor on the GPU needs `.cpu()` first, and a tensor that records
    gradients needs `.detach()` first: `t.detach().cpu().numpy()`.
    """)
    return


@app.cell(hide_code=True)
def s41(mo):
    mo.md(r"""
    ---
    ## Slide 41 · Summary of Part 1

    | task | code |
    |---|---|
    | create | `torch.arange(n)`, `torch.zeros(shape)`, `torch.ones(shape)`, `torch.randn(shape)`, `torch.tensor([...])` |
    | inspect | `t.shape`, `t.numel()`, `t.ndim`, `t.dtype` |
    | reshape | `t.reshape(3, -1)` (a view where possible) |
    | index and slice | `t[-1]`, `t[1:3]`, `t[1, 2] = 17`, `t[:2, :] = 12` |
    | compute | `torch.exp(t)`, `p + q`, `p ** q`, `torch.cat(..., dim=0)`, `t == u`, `t.sum()` |
    | broadcast | `(3, 1) + (1, 2) → (3, 2)` |
    | save memory | `Z[:] = X + Y`, `X += Y` |
    | convert | `t.numpy()`, `torch.from_numpy(a)`, `t.item()` |

    ---
    ## Slide 42 · Exercises

    **Exercise 1: other comparisons.** Pick an operator. Look at the values and at the dtype of the result.
    """)
    return


@app.cell
def ex1_ui(mo):
    compare_op = mo.ui.dropdown(options=["==", "<", ">", "<=", ">=", "!="], value="<", label="X ? Y")
    compare_op
    return (compare_op,)


@app.cell
def ex1_result(X, Y, compare_op, mo, plt, show_tensor, torch):
    _ops = {"==": torch.eq, "<": torch.lt, ">": torch.gt, "<=": torch.le, ">=": torch.ge, "!=": torch.ne}
    _res = _ops[compare_op.value](X, Y)
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 2.6))
    show_tensor(X, "X", _axes[0])
    show_tensor(Y, "Y", _axes[1])
    show_tensor(_res.int(), f"X {compare_op.value} Y  (1 = True)", _axes[2], cmap="Greens")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"Result dtype: **{_res.dtype}**: a comparison always gives a boolean tensor. "
                           f"`(X {compare_op.value} Y).sum()` = {int(_res.sum())} elements are `True`.")])
    return


@app.cell(hide_code=True)
def ex2_md(mo):
    mo.md(r"""
    **✏️ Exercise 2: broadcasting with 3-D tensors.** Choose two shapes with up to three axes in the cell
    below. *Before* you run it, write the shape you expect for `a3 + b3` into `predicted_shape`
    (or `"error"` if you expect them to be incompatible).
    """)
    return


@app.cell
def ex2(torch):
    # ✏️ YOUR TURN: change the shapes, predict the result, run the cell.
    a3 = torch.ones(2, 1, 4)
    b3 = torch.ones(3, 1)
    predicted_shape = None            # for example (2, 3, 4), or "error"
    return a3, b3, predicted_shape


@app.cell(hide_code=True)
def ex2_check(a3, b3, checklist, mo, predicted_shape):
    def _broadcast_steps(sa, sb):
        _n = max(len(sa), len(sb))
        _pa = (1,) * (_n - len(sa)) + tuple(sa)
        _pb = (1,) * (_n - len(sb)) + tuple(sb)
        return _pa, _pb


    if predicted_shape is None:
        _out = checklist([], "Write your prediction into `predicted_shape` in the cell above.")
    else:
        _pa, _pb = _broadcast_steps(a3.shape, b3.shape)
        try:
            _actual = tuple((a3 + b3).shape)
        except RuntimeError:
            _actual = "error"
        _pred = tuple(predicted_shape) if not isinstance(predicted_shape, str) else predicted_shape
        _out = mo.vstack([
            checklist([(_pred == _actual, f"you predicted `{_pred}`, PyTorch gives `{_actual}`")]),
            mo.md(f"Step 1, pad on the left with 1s: `{_pa}` and `{_pb}`. "
                  "Step 2, for each axis the lengths must be equal or 1; the result takes the larger length."),
        ])
    _out
    return


@app.cell(hide_code=True)
def s43(mo):
    mo.md(r"""
    ---
    # Part 2 · Automatic differentiation (slides 43–53)

    ## Slide 43 · Why automatic differentiation?

    Training a model means changing its parameters so that a loss gets smaller. For that you need
    the **derivative of the loss with respect to every parameter**. By hand this is tedious and error-prone,
    and for a network with millions of parameters it is impossible.

    Deep-learning frameworks therefore compute derivatives **automatically** (*autograd*):

    1. While your code runs, PyTorch records every operation in a **computational graph**.
    2. `backward()` walks through this graph from the output back to the inputs and applies the
       **chain rule** at each node (*backpropagation*).

    The graph for the first example, $y = 2\,\mathbf{x}^\top\mathbf{x}$:
    """)
    return


@app.cell(hide_code=True)
def s43_graph(mo):
    mo.mermaid("""
    graph LR
        x["x (requires_grad)"] --> dot["dot(x, x) = xᵀx"]
        x --> dot
        dot --> mul["× 2"]
        mul --> y["y (a scalar)"]
    """)
    return


@app.cell(hide_code=True)
def s44(mo):
    mo.md(r"""
    ---
    ## Slides 44–45 · A simple function

    Goal: the gradient of $y = 2\,\mathbf{x}^\top\mathbf{x}$ with respect to $\mathbf{x}$.
    By hand: $\partial y / \partial \mathbf{x} = 4\mathbf{x}$.

    `x.requires_grad_(True)` tells PyTorch to record operations on `x` and to keep space for its
    gradient in `x.grad`. The gradient has the same shape as `x`. PyTorch reuses this space for every
    new gradient, which matters because training computes gradients millions of times.
    """)
    return


@app.cell
def s44_code(torch):
    _x = torch.arange(4.0)
    _x.requires_grad_(True)
    print("x      =", _x)
    print("x.grad =", _x.grad, "  ← nothing computed yet")

    _y = 2 * torch.dot(_x, _x)
    print("y      =", _y, "  ← grad_fn records the last operation")

    _y.backward()
    print("x.grad =", _x.grad)
    print("x.grad == 4 * x:", _x.grad == 4 * _x)
    return


@app.cell(hide_code=True)
def s45(mo):
    mo.md(r"""
    **Gradients add up.** A second `backward()` *adds* its gradient to the value already in `x.grad`.
    This is useful when a loss is a sum of several parts (or of several mini-batches). It is also a classic
    bug: before each new gradient, reset with `x.grad.zero_()`.
    """)
    return


@app.cell
def s45_code(torch):
    _x = torch.arange(4.0, requires_grad=True)
    (2 * torch.dot(_x, _x)).backward()
    print("after y = 2 xᵀx:              x.grad =", _x.grad, " (= 4x)")

    _x.sum().backward()
    print("then y = sum(x), no reset:    x.grad =", _x.grad, " (= 4x + 1, added up!)")

    _x.grad.zero_()
    _x.sum().backward()
    print("reset with zero_(), again:    x.grad =", _x.grad, " (= 1, correct)")
    return


@app.cell(hide_code=True)
def s46(mo):
    mo.md(r"""
    ---
    ## Slide 46 · Backward for non-scalar variables

    If `y` is a **vector**, its derivative with respect to the vector `x` is a matrix, the **Jacobian**
    $J_{ij} = \partial y_i / \partial x_j$. In deep learning you usually want the gradient of a *sum*
    (for example a loss summed over a batch), so PyTorch asks for a scalar.

    `y.backward(gradient=v)` computes $\mathbf{v}^\top J$. With $\mathbf{v}$ = all ones, this is the
    gradient of `y.sum()`. For $y = x \odot x$ the Jacobian is diagonal with entries $2x_i$:
    """)
    return


@app.cell
def s46_code(show_tensor, torch):
    _x = torch.arange(4.0, requires_grad=True)
    _y = _x * _x
    _y.backward(gradient=torch.ones(len(_y)))       # the same as _y.sum().backward()
    print("x.grad (= vᵀJ with v = ones) =", _x.grad)

    _J = torch.autograd.functional.jacobian(lambda _t: _t * _t, torch.arange(4.0))
    _ax = show_tensor(_J, "Jacobian of y = x * x")
    _ax.figure.tight_layout()
    _ax.figure
    return


@app.cell(hide_code=True)
def s47(mo):
    mo.md(r"""
    ---
    ## Slide 47 · Detaching computation

    Sometimes a value must be used **without** passing gradients through it. Example: `y = x * x` and
    `z = y * x`. If you want only the *direct* influence of `x` on `z`, treat `y` as a constant:
    `u = y.detach()` has the same values as `y`, but no history. Then $\partial z/\partial x = u$,
    not $3x^2$.

    The original graph of `y` is unchanged: you can still backpropagate through `y` itself.
    In EM work, `detach()` is common for a quantity that a step only *uses*, for example a
    fixed probe while you update an object.
    """)
    return


@app.cell
def s47_code(torch):
    _x = torch.arange(4.0, requires_grad=True)
    _y = _x * _x
    _u = _y.detach()
    _z = _u * _x
    _z.sum().backward()
    print("dz/dx == u  :", _x.grad == _u, "  (u acts as a constant)")

    _x.grad.zero_()
    _y.sum().backward()
    print("dy/dx == 2x :", _x.grad == 2 * _x, "  (the graph of y still exists)")
    return


@app.cell(hide_code=True)
def s48(mo):
    mo.md(r"""
    ---
    ## Slides 48–49 · Gradients and Python control flow

    The graph is built **while the code runs**. So autograd also works through `while` loops, `if`
    branches and function calls, even when the path depends on the data. Each input gets its own graph.
    This is important for models with variable structure, for example variable-length sequences.
    """)
    return


@app.cell
def s48_f(torch):
    def f(a):
        b = a * 2
        while torch.linalg.vector_norm(b) < 1000:    # the slide writes b.norm()
            b = b * 2
        if b.sum() > 0:
            c = b
        else:
            c = 100 * b
        return c

    return (f,)


@app.cell(hide_code=True)
def s49(mo):
    mo.md(r"""
    For a scalar `a`, `f(a)` only multiplies `a` by a constant *k* (a power of 2, and maybe 100).
    So $d = f(a) = k\,a$, and the derivative is $k = d / a$.
    **Try it:** draw new random values of `a`.
    """)
    return


@app.cell
def s49_ui(mo):
    new_a = mo.ui.button(value=0, on_click=lambda v: v + 1, label="🎲 draw a new a")
    new_a
    return (new_a,)


@app.cell
def s49_code(f, mo, new_a, torch):
    torch.manual_seed(new_a.value)
    _a = torch.randn(size=(), requires_grad=True)
    _d = f(_a)
    _d.backward()
    _k = (_d / _a).item()
    _branch = "b.sum() > 0 → c = b" if _a.item() > 0 else "b.sum() ≤ 0 → c = 100 * b"
    mo.md(f"""
    | quantity | value |
    |---|---|
    | `a` | {_a.item():.4f} |
    | `d = f(a)` | {_d.item():.1f} |
    | branch taken | {_branch} |
    | factor *k* = d / a | {_k:,.0f} |
    | `a.grad` | {_a.grad.item():,.0f} |
    | `a.grad == d / a` | **{bool(torch.isclose(_a.grad, _d / _a))}** |
    """)
    return


@app.cell(hide_code=True)
def s50(mo):
    mo.md(r"""
    ---
    ## Slide 50 · Discussion

    Automatic differentiation is a large productivity gain: you write the *forward* computation, and
    the framework gives you the derivatives. You can then design complex models and spend your time on
    the science. Behind it are optimised libraries, compilers and graph tools that save memory and compute.

    **The basic workflow, four steps:**

    1. Attach gradients to the variables you want to optimise (`requires_grad=True`).
    2. Compute the target value (the graph is recorded).
    3. Run backpropagation (`backward()`).
    4. Read the gradient (`.grad`).

    In this course you will use exactly this loop to train networks (Week 6) and to reconstruct an
    object from ptychography data by gradient descent (Week 13).
    """)
    return


@app.cell(hide_code=True)
def s51(mo):
    mo.md(r"""
    ---
    ## Slides 51–53 · Exercises

    ### Exercise 1: run the backward pass twice
    """)
    return


@app.cell
def ex_bw_twice(torch):
    _x = torch.arange(4.0, requires_grad=True)
    _y = 2 * torch.dot(_x, _x)
    _y.backward()
    print("first backward : x.grad =", _x.grad)
    try:
        _y.backward()
    except RuntimeError as _err:
        print("second backward: RuntimeError:", str(_err).split(".")[0] + ".")

    _x2 = torch.arange(4.0, requires_grad=True)
    _y2 = 2 * torch.dot(_x2, _x2)
    _y2.backward(retain_graph=True)
    _y2.backward()
    print("with retain_graph=True: x.grad =", _x2.grad, " (= 8x: the two gradients add up)")
    return


@app.cell(hide_code=True)
def ex_bw_why(mo):
    mo.accordion({"💡 Explanation": mo.md(r"""
    To save memory, PyTorch **frees the graph** after `backward()`. A second `backward()` on the same
    `y` finds no graph and raises an error. With `retain_graph=True` the graph stays, and the second
    call **adds** its gradient to `x.grad` (slide 45), so you get $4x + 4x = 8x$.
    """)})
    return


@app.cell(hide_code=True)
def ex_vec_md(mo):
    mo.md(r"""
    ### Exercise 2: control flow with a vector `a`
    Change `a` in the control-flow example to a vector. Now `d` is a vector too.
    """)
    return


@app.cell
def ex_vec(f, torch):
    torch.manual_seed(0)
    _a = torch.randn(4, requires_grad=True)
    _d = f(_a)
    try:
        _d.backward()
    except RuntimeError as _err:
        print("d.backward():", str(_err).split(".")[0] + ".")
    _d.sum().backward()
    print("a       =", _a.detach())
    print("a.grad  =", _a.grad)
    print("d / a   =", (_d / _a).detach())
    return


@app.cell(hide_code=True)
def ex_vec_why(mo):
    mo.accordion({"💡 Explanation": mo.md(r"""
    `d` is a vector, so `d.backward()` needs a `gradient` argument (slide 46); `d.sum().backward()` works.
    The loop now checks the norm of the **whole vector**, and the `if` checks the **sum** of all elements.
    So all elements share one factor *k*: $\mathbf{d} = k\,\mathbf{a}$, and every entry of `a.grad` is *k*.
    `d / a` gives *k* for each element again, so the two agree.
    """)})
    return


@app.cell(hide_code=True)
def ex_sin_md(mo):
    mo.md(r"""
    ### ✏️ Exercise 3: plot $\sin x$ and its derivative, using autograd only
    Do not type the known derivative. Let autograd compute it.
    """)
    return


@app.cell
def ex_sin(torch):
    # ✏️ YOUR TURN
    xs = torch.linspace(-2 * torch.pi, 2 * torch.pi, 200, requires_grad=True)
    ys = torch.sin(xs)
    # 1. call backward on a scalar made from ys (which one gives one derivative per point?)
    # 2. read the gradient of xs
    dys = None
    return dys, xs, ys


@app.cell(hide_code=True)
def ex_sin_check(checklist, dys, mo, plt, torch, xs, ys):
    if dys is None:
        _out = checklist([], "Compute `dys` in the cell above. The check and the plot appear here.")
    else:
        _ok = tuple(dys.shape) == (200,) and torch.allclose(dys.detach(), torch.cos(xs.detach()), atol=1e-5)
        _fig, _ax = plt.subplots(figsize=(8, 3.2))
        _ax.plot(xs.detach(), ys.detach(), label="sin x")
        _ax.plot(xs.detach(), dys.detach(), label="autograd derivative")
        _ax.legend()
        _ax.grid(alpha=0.3)
        _ax.set_xlabel("x")
        _out = mo.vstack([checklist([(_ok, "`dys` equals cos(x) at all 200 points")]), _fig])
    _out
    return


@app.cell(hide_code=True)
def ex_sin_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    ys.sum().backward()       # d(sum_i sin x_i)/dx_j = cos x_j: one derivative per point
    dys = xs.grad
    ```
    """)})
    return


@app.cell(hide_code=True)
def ex_chain_md(mo):
    mo.md(r"""
    ### ✏️ Exercises 4–6: the chain rule by hand
    Let $f(x) = \big(\log x^2\big)\cdot \sin x + x^{-1}$.

    4. Draw the **dependency graph** from $x$ to $f(x)$ on paper: one node per elementary operation.
    5. Use the **chain rule** to compute $df/dx$, and write each term next to its edge in the graph.
    6. Type your result into `my_dfdx` below. The check compares it with autograd at seven points.
    """)
    return


@app.function
# ✏️ YOUR TURN: your hand-derived derivative (use torch.sin, torch.cos, torch.log)
def my_dfdx(x):
    return None


@app.cell(hide_code=True)
def ex_chain_check(checklist, torch):
    def f_chain(x):
        return torch.log(x ** 2) * torch.sin(x) + 1 / x


    _pts = torch.tensor([-3.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0], requires_grad=True)
    f_chain(_pts).sum().backward()
    _auto = _pts.grad.detach()
    _mine = my_dfdx(_pts.detach())
    if _mine is None:
        _out = checklist([], "Write your derivative into `my_dfdx`. The check runs automatically.")
    else:
        _mine = torch.as_tensor(_mine, dtype=torch.float32)
        _items = [(bool(torch.isclose(_m, _a, rtol=1e-4, atol=1e-5)),
                   f"x = {_x:+.1f}: yours {_m.item():+.5f}, autograd {_a.item():+.5f}")
                  for _x, _m, _a in zip(_pts.detach().tolist(), _mine, _auto)]
        _out = checklist(_items)
    _out
    return


@app.cell(hide_code=True)
def ex_chain_sol(mo):
    mo.accordion({"💡 Solution: graph and derivative": mo.vstack([
        mo.mermaid("""
    graph LR
        x((x)) -->|"2x"| u1["u₁ = x²"]
        u1 -->|"1/u₁"| u2["u₂ = log u₁"]
        x -->|"cos x"| u3["u₃ = sin x"]
        u2 -->|"u₃"| u4["u₄ = u₂ · u₃"]
        u3 -->|"u₂"| u4
        x -->|"−1/x²"| u5["u₅ = 1/x"]
        u4 -->|"1"| f["f = u₄ + u₅"]
        u5 -->|"1"| f
    """),
        mo.md(r"""
    Multiply the edge derivatives along each path from $x$ to $f$, and add the paths:

    $$
    \frac{df}{dx}
    = \underbrace{u_3 \cdot \frac{1}{u_1}\cdot 2x}_{x\to u_1\to u_2\to u_4}
    + \underbrace{u_2 \cdot \cos x}_{x\to u_3\to u_4}
    + \underbrace{\left(-\frac{1}{x^2}\right)}_{x\to u_5}
    = \frac{2\sin x}{x} + \log\!\left(x^2\right)\cos x - \frac{1}{x^2}.
    $$

    ```python
    def my_dfdx(x):
        return 2 * torch.sin(x) / x + torch.log(x ** 2) * torch.cos(x) - 1 / x ** 2
    ```
    This is exactly what `backward()` does: it multiplies the local derivatives along the graph.
    """),
    ])})
    return


@app.cell(hide_code=True)
def s54(mo):
    mo.md(r"""
    ---
    ## Slides 54–55 · Next week and references

    **Next: Unit 2, *Learning from EM data: signal formation, noise & loss*.** You will see why the
    Poisson noise in today's images decides which loss function is correct.

    Before Unit 2: run every cell of this notebook once more, change at least one value in each part,
    and finish the exercises you did not complete in class. Questions go to the StudOn forum.

    **References**

    - A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, *Dive into Deep Learning*, Cambridge University Press (2023),
      sections 2.1 (Data manipulation) and 2.5 (Automatic differentiation), <https://d2l.ai>. Slides 27–53 follow these sections.
    """)
    return


if __name__ == "__main__":
    app.run()


