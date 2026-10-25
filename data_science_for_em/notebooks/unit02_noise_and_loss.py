# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "numpy", "matplotlib", "scipy"]
# ///
import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # Unit 2 · Hands-on: Poisson noise, SNR and the right loss

    **Data Science for Electron Microscopy** · Prof. Dr. Philipp Pelz · FAU Erlangen-Nürnberg · WS 26/27

    This notebook goes with the Week 2 lecture *Learning from EM data: signal formation, noise & loss*.
    You add Poisson noise to a synthetic STEM image and measure how the SNR grows with the dose.
    You find the dose at which a defect column disappears in the noise.
    Then you measure the gain and the read noise of a detector, and you derive the loss from the noise model.
    At the end, you fit a low-count peak with four losses. Only the loss changes, but the answer changes too.

    | part | sections | content |
    |---|---|---|
    | **1 · Why Poisson?** | 1.1 | electron counts, Var = mean = λ, SNR = √λ |
    | **2 · A clean phantom** | 2.1 | a noiseless STEM-like lattice of atomic columns |
    | **3 · Poisson noise and dose** | 3.1 | noisy images at five doses |
    | **4 · SNR vs dose** | 4.1 | measure the SNR and confirm the √λ law |
    | **5 · A defect column** | 5.1–5.2 | one column with 10 % less intensity |
    | **6 · Contrast-to-noise ratio** | 6.1–6.3 | CNR vs dose, dose slider, ✏️ threshold dose |
    | **7 · Real detectors** | 7.1–7.2 | variance–mean plot: gain *g* and read noise σ_r |
    | **8 · From noise model to loss** | 8.1–8.3 | MSE and Poisson NLL, loss explorer, ✏️ Poisson NLL |
    | **9 · Does the loss matter?** | 9.1–9.3 | low-count peak fit with four losses, ✏️ count conservation |

    **After this notebook you can:**

    1. Explain why electron counts follow Poisson statistics.
    2. Simulate Poisson noise on an image with `rng.poisson`.
    3. Measure the signal-to-noise ratio (SNR) at different doses.
    4. Confirm the law $\mathrm{SNR} \propto \sqrt{\lambda}$ with a simulation.
    5. Find the dose below which a feature is no longer detectable.
    6. Get the detector gain and read noise from a variance–mean plot.
    7. Derive and implement the Poisson negative log-likelihood (NLL) as a loss.
    8. Show that four losses give different answers on low-count data, and find the one without bias.

    **Time:** 90–120 minutes. **Prerequisites:** the Unit 1 notebook (arrays, slicing, plots).
    The notebook uses only NumPy, SciPy and Matplotlib, and it runs on a CPU.

    **To edit and run this notebook:** a shared molab link opens as a read-only preview.
    Sign in to molab (free, with Google or an e-mail address) and click **Fork**: you get your own copy,
    saved in your workspace. **Run it now** also works, but it does not save your changes.

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
    - **Random numbers:** each cell makes its own generator, `np.random.default_rng(42)`.
      So each cell gives the same result every time it runs.
    - **Your own copy:** to keep your changes, save your own copy of this notebook in molab.
    """), kind="info")
    return


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.optimize import minimize
    from scipy.special import gammaln
    from scipy.stats import poisson

    return gammaln, minimize, mo, mpatches, np, plt, poisson


@app.cell(hide_code=True)
def helpers(mo):
    def checklist(items, empty_msg=None):
        """items: list of (passed, text). Shows a green or orange box with one line per check."""
        if empty_msg is not None:
            return mo.callout(mo.md(empty_msg), kind="neutral")
        ok = all(p for p, _ in items)
        lines = [("✅ " if p else "❌ ") + t for p, t in items]
        head = "**All checks pass. Well done!**\n\n" if ok else "**Not yet.** Look at the red lines.\n\n"
        return mo.callout(mo.md(head + "\n\n".join(lines)), kind="success" if ok else "warn")

    return (checklist,)


@app.cell(hide_code=True)
def s1(mo):
    mo.md(r"""
    ---
    # Part 1 · Why Poisson?

    ## 1.1 · Electrons are counted

    In STEM, a focused probe scans the sample in a raster. At each probe position (one pixel), the
    annular HAADF detector collects the electrons that scatter to high angles. The detector **counts**
    the electrons that arrive.

    **The physics:**

    - Each electron arrives independently of the others.
    - In a fixed dwell time, the *average* number of electrons is $\lambda$. It is proportional to the
      local scattering cross-section and to the beam current.
    - The actual count $N$ in one pixel fluctuates around $\lambda$. It follows the **Poisson distribution**:

    $$P(N = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

    **Properties of the Poisson distribution:**

    - mean $\mathbb{E}[N] = \lambda$ and variance $\mathrm{Var}(N) = \lambda$: **Var = mean = λ**
    - standard deviation $\sigma = \sqrt{\lambda}$
    - $\mathrm{SNR} = \lambda / \sigma = \sqrt{\lambda}$

    So **two times the dose gives only $\sqrt{2} \approx 1.41$ times the SNR.**
    To get two times the SNR, you need four times the dose.
    """)
    return


@app.cell
def s1_pmf(np, plt, poisson):
    _k = np.arange(0, 101)
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4))
    for _ax, _lam in zip(_axes, [2, 10, 50]):        # mean counts per pixel
        _pmf = poisson.pmf(_k, _lam)                   # λ^k e^(−λ) / k!
        _snr = np.sqrt(_lam)
        _ax.bar(_k, _pmf, color="#3b82f6", alpha=0.7, width=0.8)
        _ax.axvline(_lam, color="#f59e0b", lw=2, label=f"mean λ = {_lam}")
        _ax.axvline(_lam - _snr, color="#ef4444", lw=1.5, ls="--", label=f"±σ = ±{_snr:.1f}")
        _ax.axvline(_lam + _snr, color="#ef4444", lw=1.5, ls="--")
        _ax.set_title(f"Poisson(λ = {_lam})\nSNR = √{_lam} ≈ {_snr:.1f}", fontsize=10)
        _ax.set_xlabel("count k")
        _ax.set_ylabel("P(N = k)")
        _ax.set_xlim(0, min(100, _lam * 3))
        _ax.legend(fontsize=8)
    _fig.suptitle("Poisson distribution at three mean values (λ = 2, 10, 50)", y=1.02, fontsize=11)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def s1_note(mo):
    mo.md(r"""
    When λ increases, the distribution becomes more symmetric and looks like a Gaussian.
    But the variance stays equal to the mean. For Gaussian noise, the variance does not depend on the mean.
    """)
    return


@app.cell(hide_code=True)
def s2(mo):
    mo.md(r"""
    ---
    # Part 2 · A clean phantom

    ## 2.1 · Build a noiseless STEM image

    We make a synthetic STEM-like **phantom**: a clean image without noise. It is the image that an
    experiment with infinite dose would give. The phantom has a regular lattice of atomic columns
    (bright spots) on a dark background, as a perfect crystal along a zone axis.
    """)
    return


@app.cell
def s2_phantom(mo, np, plt):
    N = 256          # image size (pixels)
    spacing = 16     # distance between atomic columns (pixels)
    col_radius = 2   # Gaussian width of one column (pixels)

    _x = np.arange(N)
    X, Y = np.meshgrid(_x, _x)


    def column(rx, ry):
        """One atomic column: a Gaussian blob with height 1 at (rx, ry)."""
        return np.exp(-((X - rx) ** 2 + (Y - ry) ** 2) / (2 * col_radius ** 2))


    phantom = np.zeros((N, N), dtype=float)
    _n_cols = 0
    for _ry in range(spacing // 2, N, spacing):
        for _rx in range(spacing // 2, N, spacing):
            phantom += column(_rx, _ry)
            _n_cols += 1
    phantom = phantom + 0.08              # weak background (HAADF Z-contrast background)
    phantom = phantom / phantom.max()     # normalise to a maximum of 1

    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _im = _ax.imshow(phantom, cmap="gray", origin="upper", vmin=0, vmax=1)
    _ax.set_title("Clean STEM phantom (noiseless)", fontsize=11)
    _ax.set_xlabel("x (pixels)")
    _ax.set_ylabel("y (pixels)")
    _fig.colorbar(_im, ax=_ax, label="normalised intensity")
    _fig.tight_layout()
    mo.hstack([_fig, mo.md(f"""
    | property | value |
    |---|---|
    | shape | `{phantom.shape}` |
    | minimum | {phantom.min():.3f} |
    | maximum | {phantom.max():.3f} |
    | atomic columns | {_n_cols} |
    """)], justify="start", gap=2)
    return N, column, phantom, spacing


@app.cell(hide_code=True)
def s3(mo):
    mo.md(r"""
    ---
    # Part 3 · Poisson noise and dose

    ## 3.1 · Noisy images at five doses

    The dose sets how many electrons, on average, form each pixel. Here the **dose** is the expected
    count at the peak of a bright column (the phantom has a maximum of 1).
    We multiply the phantom by the dose and get the expected count $\lambda$ for each pixel.
    Then `rng.poisson(lam_map)` draws the actual counts.
    """)
    return


@app.cell
def s3_noisy(np, phantom, plt):
    _rng = np.random.default_rng(seed=42)
    dose_levels = [5000, 500, 50, 10, 5]     # expected counts at the peak of a bright column

    _fig, _axes = plt.subplots(1, len(dose_levels), figsize=(17, 3.9))
    for _ax, _dose in zip(_axes, dose_levels):
        _lam_map = _dose * phantom                          # λ(x, y) = dose · phantom(x, y)
        _noisy = _rng.poisson(_lam_map).astype(float)       # draw the counts
        _ax.imshow(_noisy, cmap="gray", origin="upper", vmin=0, vmax=_dose * 1.1)
        _ax.set_title(f"dose = {_dose} counts/column peak\nSNR ≈ √{_dose} ≈ {np.sqrt(_dose):.1f}", fontsize=9)
        _ax.set_xlabel("x (px)", fontsize=8)
        _ax.set_ylabel("y (px)", fontsize=8)
        _ax.tick_params(labelsize=7)
    _fig.suptitle("Simulated noisy HAADF-STEM images at five doses (noise from rng.poisson)", y=1.02, fontsize=10)
    _fig.tight_layout()
    _fig
    return (dose_levels,)


@app.cell(hide_code=True)
def s3_note(dose_levels, mo, np):
    mo.md(f"""
    The titles give the SNR of one pixel at a column peak: √dose. It goes from
    **{np.sqrt(dose_levels[0]):.1f}** at dose {dose_levels[0]} to **{np.sqrt(dose_levels[-1]):.1f}** at dose {dose_levels[-1]}.
    The lecture figure gives the *image-average* SNR, √(dose × mean intensity). That value is lower,
    because most pixels are dark background.
    """)
    return


@app.cell(hide_code=True)
def s4(mo):
    mo.md(r"""
    ---
    # Part 4 · SNR vs dose

    ## 4.1 · Measure the SNR and confirm the √λ law

    We measure the SNR in three steps:

    1. Take the mean count in a 6 × 6 pixel ROI on a bright column (the signal).
    2. Take the standard deviation of the counts in the flat background between the columns (the noise).
    3. Compute SNR = signal / noise.

    The column nearest to the image centre is at (136, 136). The centre itself, `N // 2` = 128, lies
    *between* two columns. The background is every pixel where the clean phantom is flat
    (less than 1 % above its minimum). There, all variation in the noisy image is noise.

    Then we compare the measured SNR with the prediction: signal $\lambda_\text{sig}$ over noise
    $\sqrt{\lambda_\text{bg}}$. Both $\lambda$ values are proportional to the dose, so $\mathrm{SNR} \propto \sqrt{\text{dose}}$.
    """)
    return


@app.cell
def s4_snr(N, mo, np, phantom, plt, spacing):
    doses_for_plot = np.logspace(0.3, 4, 40)    # 2 to 10 000 counts/column peak

    _cx, _cy = N // 2 + spacing // 2, N // 2 + spacing // 2    # column nearest the centre
    _half_s = 3                                                # 6 × 6 pixel ROI
    _sig_roi = (slice(_cy - _half_s, _cy + _half_s), slice(_cx - _half_s, _cx + _half_s))
    bg_mask = phantom < 1.01 * phantom.min()                   # flat background pixels

    _sig_phantom = phantom[_sig_roi].mean()     # signal per unit dose
    _bg_phantom = phantom[bg_mask].mean()       # background per unit dose

    _rng = np.random.default_rng(42)
    measured_snr = []
    for _dose in doses_for_plot:
        _noisy = _rng.poisson(_dose * phantom).astype(float)
        _sig_mean = _noisy[_sig_roi].mean()
        _noise_std = _noisy[bg_mask].std()
        measured_snr.append(_sig_mean / _noise_std if _noise_std > 0 else np.nan)
    measured_snr = np.array(measured_snr)
    theoretical_snr = doses_for_plot * _sig_phantom / np.sqrt(doses_for_plot * _bg_phantom)

    _fig, _ax = plt.subplots(figsize=(8, 4.8))
    _ax.loglog(doses_for_plot, measured_snr, "o", color="#3b82f6", ms=4, label="measured SNR (simulation)")
    _ax.loglog(doses_for_plot, theoretical_snr, "-", color="#f59e0b", lw=2,
               label=r"theory: SNR $\propto \sqrt{\lambda}$")
    _ax.set_xlabel("dose (expected counts at column peak)", fontsize=11)
    _ax.set_ylabel("SNR", fontsize=11)
    _ax.set_title("SNR vs dose: the Poisson √λ law", fontsize=12)
    _ax.legend(fontsize=10)
    _ax.grid(True, which="both", alpha=0.3)
    _fig.tight_layout()
    _slope = np.polyfit(np.log(doses_for_plot), np.log(measured_snr), 1)[0]
    mo.vstack([_fig, mo.md(
        f"Signal ROI: {_sig_phantom:.3f} × dose · background: {_bg_phantom:.4f} × dose "
        f"({int(bg_mask.sum()):,} pixels) · slope of log SNR vs log dose: **{_slope:.3f}** (√λ law: 0.5). "
        "The small scatter at low dose comes from the few counts in the 36-pixel signal ROI.")])
    return (bg_mask,)


@app.cell(hide_code=True)
def s5(mo):
    mo.md(r"""
    ---
    # Part 5 · A defect column

    ## 5.1 · A column that is 10 % dimmer

    Experiments often look for one *specific* feature. Examples are a vacancy (a missing atom in a column)
    or a lighter element in place of a heavier one (less HAADF intensity).

    We make a "defect": one atomic column gets 90 % of its normal intensity (10 % dimmer).
    Then we ask: at which dose does this contrast become impossible to detect?
    """)
    return


@app.cell
def s5_defect(column, mpatches, phantom, plt, spacing):
    defect_contrast = 0.90                    # 90 % of the normal brightness (10 % dimmer)
    cx_def = spacing // 2 + 2 * spacing       # third column from the left (x = 40)
    cy_def = spacing // 2 + 2 * spacing       # third row from the top (y = 40)

    # remove the normal column and put it back at 90 %
    phantom_defect = phantom - column(cx_def, cy_def) + defect_contrast * column(cx_def, cy_def)
    phantom_defect = phantom_defect / phantom_defect.max()


    def mark_defect(ax, lw=1.5):
        """Draw a red box around the defect column."""
        ax.add_patch(mpatches.Rectangle((cx_def - spacing // 2, cy_def - spacing // 2), spacing, spacing,
                                        lw=lw, edgecolor="red", facecolor="none"))


    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.6))
    _titles = ["normal phantom",
               f"defect phantom\n(column at ({cx_def}, {cy_def}) → {100 * defect_contrast:.0f} % intensity)"]
    for _ax, _img, _title in zip(_axes, [phantom, phantom_defect], _titles):
        _im = _ax.imshow(_img, cmap="gray", origin="upper", vmin=0, vmax=1)
        _ax.set_title(_title, fontsize=10)
        _ax.set_xlabel("x (px)")
        _ax.set_ylabel("y (px)")
        _fig.colorbar(_im, ax=_ax, label="normalised intensity")
    mark_defect(_axes[1])
    _axes[1].annotate("defect\ncolumn", xy=(cx_def, cy_def), xytext=(cx_def + spacing, cy_def - spacing),
                      color="red", fontsize=8, arrowprops=dict(arrowstyle="->", color="red", lw=1))
    _fig.suptitle("Clean vs defect phantom (noiseless)", y=1.02, fontsize=11)
    _fig.tight_layout()
    _fig
    return cx_def, cy_def, defect_contrast, mark_defect, phantom_defect


@app.cell(hide_code=True)
def s5_doses_md(mo):
    mo.md(r"""
    ## 5.2 · The defect at three doses

    The red box marks the defect column. Compare it with its neighbours.
    """)
    return


@app.cell
def s5_doses(mark_defect, np, phantom_defect, plt):
    _rng = np.random.default_rng(42)
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for _ax, _dose in zip(_axes, [2000, 200, 20]):
        _noisy = _rng.poisson(_dose * phantom_defect).astype(float)
        _ax.imshow(_noisy, cmap="gray", origin="upper", vmin=0, vmax=_dose * 1.1)
        _ax.set_title(f"dose = {_dose} counts/column peak\nSNR ≈ {np.sqrt(_dose):.0f}", fontsize=9)
        _ax.set_xlabel("x (px)", fontsize=8)
        _ax.set_ylabel("y (px)", fontsize=8)
        _ax.tick_params(labelsize=7)
        mark_defect(_ax)
    _fig.suptitle("Defect visibility at three doses (red box = defect column)", y=1.02, fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def s5_doses_note(mo):
    mo.md(r"""
    - At dose 2000 the defect is clearly a dimmer column.
    - At dose 200 you can see it, but only if you look carefully.
    - At dose 20 the defect is probably lost in the noise.

    Part 6 makes "visible" a number.
    """)
    return


@app.cell(hide_code=True)
def s6(mo):
    mo.md(r"""
    ---
    # Part 6 · Contrast-to-noise ratio

    ## 6.1 · CNR vs dose

    The **contrast-to-noise ratio (CNR)** tells how well you can tell the defect column from its neighbour:

    $$\mathrm{CNR} = \frac{|I_\text{defect} - I_\text{normal}|}{\sigma_\text{noise}}$$

    $I_\text{defect}$ and $I_\text{normal}$ are the mean counts in a 4 × 4 pixel ROI on the defect column
    and on its right neighbour. $\sigma_\text{noise}$ is the standard deviation in the flat background (Part 4).
    Reliable detection by eye needs a CNR of about 3–5. Below CNR ≈ 1 the feature is invisible.

    The theory curve uses the expected contrast of the two ROIs, $\Delta \cdot \text{dose}$, and the expected
    background noise, $\sqrt{\lambda_\text{bg}}$. So $\mathrm{CNR} \propto \sqrt{\text{dose}}$, like the SNR.
    The ROI contrast $\Delta$ is less than the 10 % at the column peak, because the ROI also covers the
    sides of the column.
    """)
    return


@app.cell
def s6_cnr(bg_mask, cx_def, cy_def, defect_contrast, mo, np, phantom_defect, plt, spacing):
    doses_cnr = np.logspace(0.5, 4, 50)       # 3 to 10 000 counts/column peak
    cx_nbr, cy_nbr = cx_def + spacing, cy_def   # the normal neighbour, one spacing to the right
    roi_half = 2                                # 4 × 4 pixel ROI


    def roi_mean(img, x0, y0):
        return img[y0 - roi_half:y0 + roi_half, x0 - roi_half:x0 + roi_half].mean()


    def measure_cnr(img):
        """CNR of the defect column in one image."""
        sigma = img[bg_mask].std()
        if sigma == 0:
            return np.nan
        return abs(roi_mean(img, cx_def, cy_def) - roi_mean(img, cx_nbr, cy_nbr)) / sigma


    contrast_roi = roi_mean(phantom_defect, cx_nbr, cy_nbr) - roi_mean(phantom_defect, cx_def, cy_def)
    bg_level = phantom_defect[bg_mask].mean()


    def cnr_expected(dose):
        """Expected CNR: contrast Δ·dose over the background noise √(λ_bg)."""
        return contrast_roi * dose / np.sqrt(bg_level * dose)


    cnr_theory = cnr_expected(doses_cnr)
    _rng = np.random.default_rng(42)
    cnr_values = np.array([measure_cnr(_rng.poisson(_d * phantom_defect).astype(float)) for _d in doses_cnr])

    _fig, _ax = plt.subplots(figsize=(8, 4.8))
    _ax.loglog(doses_cnr, cnr_values, "o", ms=4, color="#3b82f6", label="measured CNR")
    _ax.loglog(doses_cnr, cnr_theory, "-", lw=2, color="#f59e0b", label=r"theory: CNR $\propto \sqrt{\lambda}$")
    _ax.axhline(5, color="#22c55e", lw=1.5, ls="--", label="CNR = 5 (reliable detection)")
    _ax.axhline(1, color="#ef4444", lw=1.5, ls="--", label="CNR = 1 (detection limit)")
    _ax.set_xlabel("dose (expected counts at column peak)", fontsize=11)
    _ax.set_ylabel("CNR (contrast-to-noise ratio)", fontsize=11)
    _ax.set_title(f"CNR vs dose for a defect with {100 * (1 - defect_contrast):.0f} % contrast", fontsize=11)
    _ax.legend(fontsize=9)
    _ax.grid(True, which="both", alpha=0.3)
    _fig.tight_layout()

    _idx5 = int(np.argmax(cnr_theory >= 5))
    _msg = (f"Reliable detection (CNR ≥ 5) needs a dose of at least **{doses_cnr[_idx5]:.0f} counts/column peak**."
            if cnr_theory[_idx5] >= 5 else "The CNR = 5 line is not reached in this dose range.")
    mo.vstack([_fig, mo.md(
        f"{_msg} ROI contrast Δ = {contrast_roi:.4f} × dose; background λ_bg = {bg_level:.4f} × dose. "
        "At low dose the measured points scatter a lot: each ROI mean comes from only 16 pixels.")])
    return bg_level, cnr_expected, cnr_theory, cnr_values, contrast_roi, doses_cnr, measure_cnr


@app.cell(hide_code=True)
def s6_try_md(mo):
    mo.md(r"""
    ## 6.2 · Try it: watch the defect disappear

    Move the dose slider (logarithmic steps). The red box marks the defect column. The right panel
    shows the counts along the row through the defect, divided by the dose. The dashed line is the clean
    profile. At high dose the dip at the defect is clear. At low dose the noise is larger than the dip.
    """)
    return


@app.cell
def s6_try_ui(mo, np):
    _steps = np.unique(np.round(np.logspace(0, np.log10(5000), 41))).astype(int).tolist()
    dose_slider = mo.ui.slider(steps=_steps, value=min(_steps, key=lambda _s: abs(_s - 500)), show_value=True,
                               label="dose (expected counts at column peak)", full_width=True)
    dose_slider
    return (dose_slider,)


@app.cell
def s6_try(cnr_expected, cx_def, cy_def, dose_slider, mark_defect, measure_cnr, mo, np, phantom_defect, plt, spacing):
    _dose = dose_slider.value
    _rng = np.random.default_rng(42)
    _noisy = _rng.poisson(_dose * phantom_defect).astype(float)
    _cnr_meas = measure_cnr(_noisy)
    _cnr_th = float(cnr_expected(_dose))

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    _axes[0].imshow(_noisy, cmap="gray", origin="upper", vmin=0, vmax=_dose * 1.1)
    _axes[0].set_title(f"full image, dose = {_dose}", fontsize=10)
    mark_defect(_axes[0])
    _crop = 5 * spacing                                  # 5 × 5 columns, defect in the centre
    _axes[1].imshow(_noisy[:_crop, :_crop], cmap="gray", origin="upper", vmin=0, vmax=_dose * 1.1)
    _axes[1].set_title("zoom: 5 × 5 columns", fontsize=10)
    mark_defect(_axes[1], lw=2)
    _cols = np.arange(_crop)
    _axes[2].plot(_cols, _noisy[cy_def, :_crop] / _dose, "-", color="#3b82f6", lw=1.2, label="noisy counts / dose")
    _axes[2].plot(_cols, phantom_defect[cy_def, :_crop], "k--", lw=1.5, label="clean profile")
    _axes[2].axvspan(cx_def - spacing // 2, cx_def + spacing // 2, color="red", alpha=0.08, label="defect column")
    _axes[2].set_ylim(-0.2, 1.6)
    _axes[2].set_xlabel("x (px)")
    _axes[2].set_ylabel("counts / dose")
    _axes[2].set_title(f"profile along y = {cy_def}", fontsize=10)
    _axes[2].legend(fontsize=8, loc="upper right")
    _axes[2].grid(alpha=0.3)
    _fig.tight_layout()

    if _cnr_th >= 5:
        _kind, _verdict = "success", "**detectable**: the expected CNR is 5 or more (reliable detection)."
    elif _cnr_th >= 1:
        _kind, _verdict = "warn", "**not reliable**: the expected CNR is between 1 and 5. You may see it, or you may not."
    else:
        _kind, _verdict = "danger", "**invisible**: the expected CNR is below 1 (detection limit)."
    mo.vstack([_fig, mo.callout(mo.md(
        f"Dose **{_dose}** counts/column peak · SNR at a column peak ≈ {np.sqrt(_dose):.1f} · "
        f"expected CNR **{_cnr_th:.2f}** · CNR measured on this image **{_cnr_meas:.2f}**\n\n"
        f"The defect is {_verdict}"), kind=_kind)])
    return


@app.cell(hide_code=True)
def ex_thr_md(mo):
    mo.md(r"""
    ## 6.3 · ✏️ Exercise: the threshold dose

    Find the *dose threshold* above which the defect column is detectable, for a CNR threshold that you choose.

    1. Keep `cnr_threshold = 3.0` or choose your own value.
    2. Find the index `threshold_idx` of the first dose in `doses_cnr` where `cnr_theory >= cnr_threshold`.
       *Hint:* `np.argmax(condition)` gives the index of the first `True` in `condition`.
    3. Set `threshold_dose` to the dose at that index.
    4. Simulate the noisy image `noisy_thresh` at exactly that dose: use `rng_thresh.poisson` on `phantom_defect`.

    When your answer is correct, the check shows the image with the defect box (step 5), and the
    CNR plot with your threshold (step 6). Then answer: can you see the defect by eye?
    """)
    return


@app.cell
def ex_thr(np):
    # ✏️ YOUR TURN: replace each `None`, then run the cell.
    cnr_threshold = 3.0
    rng_thresh = np.random.default_rng(42)
    threshold_idx = None       # first index where cnr_theory >= cnr_threshold
    threshold_dose = None      # the dose in doses_cnr at that index
    noisy_thresh = None        # Poisson image of phantom_defect at threshold_dose (use rng_thresh)
    return cnr_threshold, noisy_thresh, threshold_dose, threshold_idx


@app.cell(hide_code=True)
def ex_thr_check(checklist, cnr_theory, cnr_threshold, cnr_values, cx_def, cy_def, defect_contrast, doses_cnr, mark_defect, mo, noisy_thresh, np, phantom_defect, plt, spacing, threshold_dose, threshold_idx):
    def _as_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None


    if threshold_idx is None and threshold_dose is None and noisy_thresh is None:
        _out = checklist([], "Fill in `threshold_idx`, `threshold_dose` and `noisy_thresh` in the cell above. "
                             "The check runs automatically.")
    else:
        _ref_idx = int(np.argmax(cnr_theory >= cnr_threshold))
        _reached = bool(cnr_theory[_ref_idx] >= cnr_threshold) and _ref_idx > 0
        _idx = _as_float(threshold_idx)
        _dose = _as_float(threshold_dose)
        _idx_ok = _idx is not None and _idx == _ref_idx
        _dose_ok = _dose is not None and np.isclose(_dose, doses_cnr[_ref_idx])
        _img_ok = False
        if _dose_ok and noisy_thresh is not None:
            _arr = np.asarray(noisy_thresh, dtype=float)
            _expect = _dose * phantom_defect.mean()
            _img_ok = (_arr.shape == phantom_defect.shape and np.allclose(_arr, np.round(_arr))
                       and abs(_arr.mean() - _expect) < 0.05 * _expect)
        _items = [
            (_reached, f"the CNR threshold {cnr_threshold} is reached inside the dose range (and not at the first dose)"),
            (_idx_ok, f"`threshold_idx` = {threshold_idx} is the first index with `cnr_theory >= {cnr_threshold}`"),
            (_dose_ok, f"`threshold_dose` = {threshold_dose} is `doses_cnr[threshold_idx]`"),
            (_img_ok, "`noisy_thresh` has integer counts with mean ≈ `threshold_dose * phantom_defect.mean()`"),
        ]
        _parts = [checklist(_items)]
        if all(p for p, _ in _items):
            _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.8))
            _im = _axes[0].imshow(noisy_thresh, cmap="gray", origin="upper", vmin=0, vmax=_dose * 1.1)
            _axes[0].set_title(f"noisy image at threshold dose = {_dose:.0f}\n"
                               f"CNR ≈ {cnr_threshold:.1f}  (SNR ≈ {np.sqrt(_dose):.1f})", fontsize=9)
            _axes[0].set_xlabel("x (px)")
            _axes[0].set_ylabel("y (px)")
            _fig.colorbar(_im, ax=_axes[0])
            mark_defect(_axes[0], lw=1.8)
            _axes[0].annotate("defect", xy=(cx_def, cy_def), xytext=(cx_def + spacing * 1.5, cy_def),
                              color="red", fontsize=8, arrowprops=dict(arrowstyle="->", color="red", lw=1))
            _axes[1].loglog(doses_cnr, cnr_theory, "-", lw=2, color="#f59e0b", label=r"theory: CNR $\propto \sqrt{\lambda}$")
            _axes[1].loglog(doses_cnr, cnr_values, "o", ms=3, color="#3b82f6", label="measured CNR")
            _axes[1].axhline(cnr_threshold, color="#ef4444", lw=1.8, ls="--", label=f"CNR = {cnr_threshold} (your threshold)")
            _axes[1].axvline(_dose, color="#22c55e", lw=1.5, ls=":", label=f"threshold dose = {_dose:.0f}")
            _axes[1].set_xlabel("dose (counts/column peak)")
            _axes[1].set_ylabel("CNR")
            _axes[1].set_title("CNR vs dose with your threshold", fontsize=10)
            _axes[1].legend(fontsize=8)
            _axes[1].grid(True, which="both", alpha=0.3)
            _fig.tight_layout()
            _parts += [_fig, mo.md(
                f"**Summary:** the defect is {100 * (1 - defect_contrast):.0f} % dimmer than a normal column. "
                f"CNR ≥ {cnr_threshold} needs a dose of at least **{_dose:.0f} counts/column peak**. "
                f"The SNR at a column peak is then {np.sqrt(_dose):.1f}. Can you see the defect by eye?")]
        _out = mo.vstack(_parts)
    _out
    return


@app.cell(hide_code=True)
def ex_thr_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    cnr_threshold = 3.0
    rng_thresh = np.random.default_rng(42)
    threshold_idx = np.argmax(cnr_theory >= cnr_threshold)     # first True
    threshold_dose = doses_cnr[threshold_idx]
    noisy_thresh = rng_thresh.poisson(threshold_dose * phantom_defect).astype(float)
    ```
    `np.argmax` on a boolean array returns the first index of the maximum, `True`.
    Be careful: if no element is `True`, it returns 0. So always check that the threshold is reached.
    """)})
    return


@app.cell(hide_code=True)
def s7(mo):
    mo.md(r"""
    ---
    # Part 7 · Real detectors

    ## 7.1 · The variance–mean plot

    A real camera does not give electron counts directly. A common model for one pixel is

    $$x = g\cdot \mathrm{Poisson}(\lambda) + \mathcal{N}(0, \sigma_r^2),$$

    with the gain $g$ (digital numbers, DN, per electron) and the read-noise standard deviation $\sigma_r$ (DN).
    The mean and the variance are

    $$\mathbb{E}[x] = g\lambda, \qquad \mathrm{Var}(x) = g^2\lambda + \sigma_r^2 = g\,\mathbb{E}[x] + \sigma_r^2 .$$

    So a plot of **variance vs mean** for several illumination levels is a straight line:
    **slope = gain $g$**, **intercept = read-noise variance $\sigma_r^2$**.
    This is the *photon-transfer curve*, which calibrates every scientific camera.
    When you know $g$ and $\sigma_r$, you know the noise regime of your data, and so the correct loss (Part 8).

    The cell simulates 20 flat-field frames of 64 × 64 pixels, from 0.3 to 1000 electrons per pixel.
    **Try it:** change the gain and the read noise, and watch the fit and the crossover move.
    """)
    return


@app.cell
def s7_ui(mo):
    gain_slider = mo.ui.slider(start=0.5, stop=5.0, step=0.1, value=2.5, show_value=True, label="gain g (DN per electron)")
    read_noise_slider = mo.ui.slider(start=0.0, stop=20.0, step=0.5, value=6.0, show_value=True, label="read noise σ_r (DN)")
    mo.hstack([gain_slider, read_noise_slider], justify="start", gap=2)
    return gain_slider, read_noise_slider


@app.cell
def s7_ptc(checklist, gain_slider, mo, np, plt, read_noise_slider):
    g_true, sigma_r_true = gain_slider.value, read_noise_slider.value
    _illum = np.logspace(-0.5, 3, 20)       # mean electrons per pixel for each flat field
    _n_pix = 64 * 64                        # pixels per flat-field frame

    _rng = np.random.default_rng(42)
    _means, _variances = [], []
    for _lam in _illum:
        _frame = g_true * _rng.poisson(_lam, size=_n_pix) + _rng.normal(0, sigma_r_true, size=_n_pix)
        _means.append(_frame.mean())
        _variances.append(_frame.var())
    _means, _variances = np.array(_means), np.array(_variances)

    _slope, _intercept = np.polyfit(_means, _variances, 1)
    _crossover = sigma_r_true ** 2 / g_true

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5))
    _mm = np.linspace(0, _means.max(), 100)
    _axes[0].plot(_means, _variances, "o", color="#2a78d6", label="flat-field frames")
    _axes[0].plot(_mm, _slope * _mm + _intercept, "-", color="#eb6834",
                  label=f"fit: slope = {_slope:.2f}, intercept = {_intercept:.0f}")
    _axes[0].set_xlabel("mean (DN)")
    _axes[0].set_ylabel("variance (DN²)")
    _axes[0].set_title("Variance–mean plot (linear axes)")
    _axes[0].legend()
    _axes[1].loglog(_means, _variances, "o", color="#2a78d6", label="total variance")
    if sigma_r_true > 0:
        _axes[1].axhline(sigma_r_true ** 2, color="#1baf7a", ls="--", label="read-noise floor σ_r²")
        _axes[1].axvline(_crossover, color="#555555", lw=1, ls=":", label="crossover σ_r²/g")
    _axes[1].loglog(_mm[1:], g_true * _mm[1:], "-.", color="#eb6834", label="shot noise g·mean")
    _axes[1].set_xlabel("mean (DN)")
    _axes[1].set_ylabel("variance (DN²)")
    _axes[1].set_title("log–log: two regimes")
    _axes[1].legend()
    for _ax in _axes:
        _ax.grid(True, which="both", alpha=0.3)
    _fig.tight_layout()

    mo.vstack([_fig, mo.md(f"""
    | quantity | fitted | true |
    |---|---|---|
    | gain *g* = slope | **{_slope:.2f}** DN/e⁻ | {g_true:.2f} |
    | read noise σ_r = √intercept | **{np.sqrt(max(_intercept, 0)):.2f}** DN | {sigma_r_true:.2f} |
    | crossover (shot noise = read noise) | | mean ≈ {_crossover:.1f} DN |
    """), checklist([(abs(_slope - g_true) / g_true < 0.1, "the fitted gain is within 10 % of the true gain")]),
              mo.md("The slope is precise. The intercept is less precise: the bright frames have the largest "
                    "variances, and they dominate the unweighted `np.polyfit`.")])
    return


@app.cell(hide_code=True)
def s7_questions(mo):
    mo.vstack([mo.md(r"""
    ## 7.2 · Questions to think about

    - Left of the crossover, the variance is almost constant: the noise is approximately Gaussian with a
      fixed $\sigma$. Right of it, the variance grows with the signal. Which loss do you choose in each regime?
    - Before you apply a Poisson model to camera data, you must convert DN to electrons: subtract the offset,
      then divide by $g$. What happens to a Poisson fit if you forget this step?
    """), mo.accordion({"💡 Answers": mo.md(r"""
    - Read-noise regime (left): constant variance, so the Gaussian NLL, which is the MSE.
      Shot-noise regime (right): variance = mean, so the Poisson NLL. Between them: a mixed model or a
      variance-stabilising transform (Part 8).
    - Without the conversion, the Poisson NLL sees numbers that are not counts. Their variance is
      $g$ times their mean, not equal to it. So the loss gives each pixel the wrong weight, and quantities
      such as "counts" are off by the factor $g$.
    """)})])
    return


@app.cell(hide_code=True)
def s8(mo):
    mo.md(r"""
    ---
    # Part 8 · From noise model to loss

    ## 8.1 · The loss is the negative log-likelihood

    Write each measurement as a random draw $y_i \sim p(y \mid \mu_i)$. Here $\mu_i = f_\theta(\mathbf{x}_i)$
    is the model prediction. **Maximum likelihood** picks the parameters that make the observed data
    most probable. The negative logarithm turns the product over independent pixels into a sum: the **loss**.

    | noise model | $-\log p(y_i\mid\mu_i)$ | loss without the constants |
    |---|---|---|
    | Gaussian, constant $\sigma$ | $\frac{(y_i-\mu_i)^2}{2\sigma^2} + \frac12\log 2\pi\sigma^2$ | **MSE** $\;(y_i-\mu_i)^2$ |
    | Poisson, rate $\lambda_i$ | $\lambda_i - y_i\log\lambda_i + \log(y_i!)$ | **Poisson NLL** $\;\ell = \lambda_i - y_i\log\lambda_i$ |

    The Poisson gradient $\partial\ell/\partial\lambda = (\lambda - y)/\lambda$ is the MSE gradient divided by
    the variance $\lambda$. So the Poisson loss weights each pixel by $1/\text{variance}$ automatically.

    Two "shortcuts" are also common:

    - **Weighted χ² with the observed counts** (Neyman's χ²): $\sum_i (y_i - \lambda_i)^2 / \max(y_i, 1)$.
      It uses $\sigma_i^2 = y_i$.
    - **Anscombe + MSE:** transform $z = 2\sqrt{y + 3/8}$ (variance ≈ 1), then fit with MSE on $z$.

    ## 8.2 · Try it: the loss explorer

    Choose the observed count $y$ of one pixel. The plots show both losses as functions of the
    predicted rate $\hat\lambda$. The Poisson NLL is shifted so that its minimum is 0, as in the lecture.
    The grey curves are the lecture examples $y$ = 0, 2 and 8.
    """)
    return


@app.cell
def s8_loss_ui(mo):
    y_obs = mo.ui.slider(start=0, stop=20, step=1, value=2, show_value=True, label="observed count y")
    y_obs
    return (y_obs,)


@app.cell
def s8_loss_plot(mo, np, plt, y_obs):
    _y = y_obs.value
    _lam = np.linspace(0.02, 25, 1200)


    def _curves(y):
        _mse = (y - _lam) ** 2
        _pois = _lam - y * np.log(_lam)
        return _mse, _pois - _pois.min()


    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for _ref in (0, 2, 8):
        _m, _p = _curves(_ref)
        _axes[0].plot(_lam, _m, color="#bbbbbb", lw=1)
        _axes[1].plot(_lam, _p, color="#bbbbbb", lw=1)
    _m, _p = _curves(_y)
    _axes[0].plot(_lam, _m, color="#2a78d6", lw=2.5, label=f"observed y = {_y}")
    _axes[1].plot(_lam, _p, color="#eb6834", lw=2.5, label=f"observed y = {_y}")
    for _ax in _axes:
        _ax.axvline(max(_y, 0.02), color="#555555", lw=1, ls=":")
        _ax.set_xlabel(r"predicted rate $\hat\lambda$ (counts)")
        _ax.set_ylim(0, 30)
        _ax.legend(loc="upper right")
        _ax.grid(alpha=0.3)
    _axes[0].set_ylabel("loss")
    _axes[0].set_title(r"MSE $(y-\hat\lambda)^2$: same penalty everywhere")
    _axes[1].set_title(r"Poisson NLL $\hat\lambda - y\log\hat\lambda$ (shifted)")
    _fig.tight_layout()

    if _y == 0:
        _txt = ("**y = 0:** the MSE is $\\hat\\lambda^2$, very flat near 0. The Poisson NLL is $\\hat\\lambda$: "
                "it grows linearly, and its minimum is at $\\hat\\lambda = 0$.")
    else:
        _txt = (f"**y = {_y}:** both minima are at $\\hat\\lambda = y = {_y}$. The curvature at the minimum is "
                f"2 for the MSE (for every y) and $y/\\hat\\lambda^2 = 1/y$ = **{1 / _y:.3f}** for the Poisson NLL: "
                "one over the variance. A bright pixel pulls on the fit less than a dark pixel. "
                "If $\\hat\\lambda \\to 0$ while counts arrived, the Poisson NLL goes to infinity.")
    mo.vstack([_fig, mo.md(_txt)])
    return


@app.cell(hide_code=True)
def ex_nll_md(mo):
    mo.md(r"""
    ## 8.3 · ✏️ Exercise: implement the Poisson NLL

    1. Implement `my_poisson_nll(y, lam)`. It returns the **full** negative log-likelihood, summed over
       all pixels, with the $\log(y!)$ term. *Hint:* `gammaln(y + 1)` equals $\log(y!)$.
    2. Implement `my_poisson_loss(y, lam)` without the constant term.
    3. Protect the logarithm: use `np.maximum(lam, 1e-12)` in place of `lam`.

    The check compares your NLL with `scipy.stats.poisson.logpmf`. It also confirms that, for one pixel
    with observation $y$, your loss has its minimum at $\lambda = y$.
    """)
    return


@app.cell
def ex_nll():
    # ✏️ YOUR TURN: replace each `return None` (np and gammaln are already imported).
    def my_poisson_nll(y, lam):
        # full NLL summed over pixels, with log(y!)
        return None


    def my_poisson_loss(y, lam):
        # the same, without the constant log(y!) term
        return None

    return my_poisson_loss, my_poisson_nll


@app.cell(hide_code=True)
def ex_nll_check(checklist, gammaln, mo, my_poisson_loss, my_poisson_nll, np, poisson):
    _rng = np.random.default_rng(42)
    _y_test = _rng.poisson(3.0, size=1000)
    _lam_test = _rng.uniform(0.5, 6.0, size=1000)
    _ref = -poisson.logpmf(_y_test, _lam_test).sum()


    def _call(f, *args):
        """None if f returns None; NaN if f fails or returns no number."""
        try:
            _v = f(*args)
            return None if _v is None else float(_v)
        except Exception:
            return float("nan")


    _nll = _call(my_poisson_nll, _y_test, _lam_test)
    _loss = _call(my_poisson_loss, _y_test, _lam_test)
    if _nll is None and _loss is None:
        _out = checklist([], "Write `my_poisson_nll` and `my_poisson_loss` in the cell above. The check runs automatically.")
    else:
        _lam_grid = np.linspace(0.05, 15, 3000)
        _argmins = {}
        for _y_obs in [1, 4, 9]:
            _vals = [_call(my_poisson_loss, _y_obs, _l) for _l in _lam_grid]
            _argmins[_y_obs] = None if None in _vals else _lam_grid[int(np.argmin(_vals))]
        _items = [
            (_nll is not None and np.isclose(_nll, _ref),
             f"`my_poisson_nll` = {_nll if _nll is None else round(_nll, 4)}, scipy = {_ref:.4f}"),
            (_nll is not None and _loss is not None and np.isclose(_nll - _loss, gammaln(_y_test + 1).sum()),
             "the two losses differ only by Σ log(y!)"),
            (_argmins[4] is not None and abs(_argmins[4] - 4) < 0.02,
             "for one pixel the loss is smallest at λ = y: "
             + ", ".join(f"y = {_k} → λ = {'—' if _v is None else f'{_v:.2f}'}" for _k, _v in _argmins.items())),
        ]
        _out = checklist(_items)
    _out
    return


@app.cell(hide_code=True)
def ex_nll_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    def my_poisson_nll(y, lam):
        lam = np.maximum(lam, 1e-12)          # guard against log(0)
        return np.sum(lam - y * np.log(lam) + gammaln(y + 1))

    def my_poisson_loss(y, lam):
        lam = np.maximum(lam, 1e-12)
        return np.sum(lam - y * np.log(lam))
    ```
    Minimum at $\lambda = y$: $\;\frac{d}{d\lambda}(\lambda - y\log\lambda) = 1 - y/\lambda = 0 \Rightarrow \lambda = y$.

    In PyTorch, the same loss is `torch.nn.PoissonNLLLoss(log_input=False, full=False)`.
    With `log_input=True` the network predicts $\log\lambda$, which is numerically more stable.
    """)})
    return


@app.cell(hide_code=True)
def s9(mo):
    mo.md(r"""
    ---
    # Part 9 · Does the loss matter?

    ## 9.1 · Fit a low-count peak with four losses

    We simulate a typical quantification task: a **Gaussian peak on a flat background** in a spectrum with
    64 channels. Think of an EELS edge, an EDS line, or a line profile across an atomic column.
    The rate in channel $i$ is

    $$\lambda_i = D\,\big(a\, e^{-(E_i - 0.5)^2 / 2s^2} + b\big),$$

    with the dose scale $D$ (counts at the peak maximum), the amplitude $a$, the background $b$ and the width $s$.
    We fit $(a, b, s)$ with four losses and `scipy.optimize.minimize`. The fits use the same code as the
    lecture figure (`02_em_data_formation/img/make_figures.py`).

    The most important quantity is the **integrated number of counts** in the model, $\sum_i \lambda_i$.
    Elemental quantification uses it.
    """)
    return


@app.cell
def s9_model(minimize, np):
    E_axis = np.linspace(0, 1, 64)
    true_params = np.array([1.0, 0.05, 0.05])     # a, b, s


    def peak_rate(params, dose):
        a, b, s = params
        return dose * (a * np.exp(-(E_axis - 0.5) ** 2 / (2 * s ** 2)) + b)


    def mse_loss(y, mu):
        return np.sum((y - mu) ** 2)


    def poisson_loss(y, lam):
        lam = np.maximum(lam, 1e-12)
        return np.sum(lam - y * np.log(lam))


    def make_objective(y, dose, kind):
        """The loss as a function of the parameters (scaled by the dose for good conditioning)."""
        def objective(params):
            lam = peak_rate(params, dose)
            if kind == "MSE":
                return mse_loss(y, lam) / dose ** 2
            if kind == "weighted χ²":
                return np.sum((y - lam) ** 2 / np.maximum(y, 1)) / dose
            if kind == "Anscombe + MSE":
                return mse_loss(2 * np.sqrt(y + 3 / 8), 2 * np.sqrt(lam + 3 / 8))
            if kind == "Poisson NLL":
                return poisson_loss(y, lam) / dose
            raise ValueError(kind)
        return objective


    def fit_peak(y, dose, kind):
        res = minimize(make_objective(y, dose, kind), x0=[1.2, 0.1, 0.06], method="L-BFGS-B",
                       bounds=[(0, None), (1e-6, None), (0.01, 0.3)])
        return res.x


    LOSSES = ["MSE", "weighted χ²", "Anscombe + MSE", "Poisson NLL"]
    COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    STYLES = [("o", "-"), ("s", "--"), ("^", "-."), ("D", "-")]
    return COLORS, E_axis, LOSSES, STYLES, fit_peak, peak_rate, true_params


@app.cell
def s9_demo_ui(mo):
    demo_dose = mo.ui.slider(steps=[2, 3, 5, 10, 20, 50, 100], value=5, show_value=True,
                             label="peak dose D (counts at maximum)")
    demo_dose
    return (demo_dose,)


@app.cell
def s9_demo(COLORS, E_axis, LOSSES, STYLES, demo_dose, fit_peak, mo, np, peak_rate, plt, true_params):
    _dose = demo_dose.value
    _rng = np.random.default_rng(42)
    _y = _rng.poisson(peak_rate(true_params, _dose))

    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _ax.step(E_axis, _y, where="mid", color="#555555", lw=1.3, label=f"counts (dose {_dose})")
    _ax.plot(E_axis, peak_rate(true_params, _dose), "k:", lw=2.5, label="true rate")
    _rows = []
    for _kind, _c, (_m, _ls) in zip(LOSSES, COLORS, STYLES):
        _p = fit_peak(_y, _dose, _kind)
        _ax.plot(E_axis, peak_rate(_p, _dose), ls=_ls, color=_c, lw=2, label=_kind)
        _rows.append(f"| {_kind} | {_p[0]:.3f} | {_p[1]:.3f} | {_p[2]:.3f} | {peak_rate(_p, _dose).sum():.1f} |")
    _ax.set_xlabel("energy / position (a.u.)")
    _ax.set_ylabel("counts")
    _ax.set_title("One low-count spectrum fitted with four losses")
    _ax.legend(fontsize=8)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    mo.hstack([_fig, mo.md(
        "| loss | a | b | s | integrated counts |\n|---|---|---|---|---|\n" + "\n".join(_rows)
        + f"\n| **data** | | | | **{_y.sum()}** |\n\ntrue: a = 1.0, b = 0.05, s = 0.05")], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def s9_demo_note(mo):
    mo.md(r"""
    Look at the table: the **Poisson NLL** fit gives *exactly* the total number of detected counts.
    The Poisson maximum-likelihood estimate always does this for a model with a free overall scale:
    it **conserves counts**. Plain MSE also conserves them here, because the free constant background
    forces the least-squares residuals to sum to zero. The weighted χ² and the Anscombe fits lose counts.

    One spectrum can be luck. So we repeat the experiment many times.

    ## 9.2 · Bias and scatter over many repeats

    The next cell fits 200 spectra at each of four doses with all four losses (3200 fits, about 15 s).
    """)
    return


@app.cell
def s9_bias(COLORS, LOSSES, STYLES, fit_peak, mo, np, peak_rate, plt, true_params):
    doses_fit = np.array([2, 5, 20, 100])
    n_rep = 200
    _tot_true = peak_rate(true_params, 1).sum()
    _rng = np.random.default_rng(42)

    bias_tot = {_k: [] for _k in LOSSES}      # bias of the integrated counts (%)
    std_width = {_k: [] for _k in LOSSES}     # relative scatter of the fitted width (%)
    for _dose in doses_fit:
        _estimates = {_k: [] for _k in LOSSES}
        for _ in range(n_rep):
            _y = _rng.poisson(peak_rate(true_params, _dose))
            for _k in LOSSES:
                _estimates[_k].append(fit_peak(_y, _dose, _k))
        for _k in LOSSES:
            _est = np.array(_estimates[_k])
            _tot = np.array([peak_rate(_q, 1).sum() for _q in _est])
            bias_tot[_k].append(100 * (_tot.mean() / _tot_true - 1))
            std_width[_k].append(100 * _est[:, 2].std() / true_params[2])

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for _k, _c, (_m, _ls) in zip(LOSSES, COLORS, STYLES):
        _axes[0].semilogx(doses_fit, bias_tot[_k], marker=_m, ls=_ls, color=_c, ms=7, label=_k)
        _axes[1].semilogx(doses_fit, std_width[_k], marker=_m, ls=_ls, color=_c, ms=7, label=_k)
    _axes[0].axhline(0, color="k", lw=1)
    _axes[0].set_title("Bias of integrated counts (MSE and Poisson overlap near 0)")
    _axes[0].set_ylabel("bias (%)")
    _axes[1].set_title("Relative scatter of the fitted width")
    _axes[1].set_ylabel("std / true (%)")
    for _ax in _axes:
        _ax.set_xlabel("peak dose (counts at maximum)")
        _ax.legend(fontsize=8)
        _ax.grid(True, which="both", alpha=0.3)
    _fig.tight_layout()

    _head = "| loss | " + " | ".join(f"dose {_d}" for _d in doses_fit) + " |\n|---|" + "---|" * len(doses_fit) + "\n"
    _bias_rows = "\n".join(f"| {_k} | " + " | ".join(f"{_b:+.1f} %" for _b in bias_tot[_k]) + " |" for _k in LOSSES)
    _std_rows = "\n".join(f"| {_k} | " + " | ".join(f"{_s:.1f} %" for _s in std_width[_k]) + " |" for _k in LOSSES)
    mo.vstack([_fig, mo.hstack([
        mo.md("**Bias of the integrated counts**\n\n" + _head + _bias_rows),
        mo.md("**Relative scatter of the width**\n\n" + _head + _std_rows)], justify="start", gap=2)])
    return


@app.cell(hide_code=True)
def s9_bias_note(mo):
    mo.md(r"""
    **What you see**

    - **Weighted χ²** (weights from the *observed* counts) is biased **low** by 10–20 % at a few counts per
      channel. Channels that fluctuated low get a large weight and pull the fit down.
    - **Anscombe + MSE** is also biased low at low dose. The square root is not linear, so
      $\mathbb{E}[\sqrt{y}] \ne \sqrt{\mathbb{E}[y]}$. It becomes accurate above about 10–20 counts.
    - **Plain MSE** is (almost) unbiased for this model, but its estimates **scatter more** than the Poisson fit.
      It ignores that bright channels are noisier. In a dose-limited experiment, more scatter means
      that you need more dose for the same precision.
    - **Poisson NLL** is unbiased *and* has the smallest scatter. It is the maximum-likelihood estimator for counts.

    The data are the same for all four fits. **Only the loss changes**, and at low dose the answer moves by 10–20 %.
    """)
    return


@app.cell(hide_code=True)
def ex_cons_md(mo):
    mo.md(r"""
    ## 9.3 · ✏️ Exercise: count conservation

    Confirm that the Poisson fit conserves counts. For 20 random spectra at dose 3:

    1. Fit each spectrum with `fit_peak(y, 3, kind)` for `kind` = `'Poisson NLL'` and `'weighted χ²'`.
    2. Compute the fitted rates `lam_hat = peak_rate(params, 3)`.
    3. Append the relative difference $(\sum_i \hat\lambda_i - \sum_i y_i) / \sum_i y_i$ to `rel_diff[kind]`.
    """)
    return


@app.cell
def ex_cons(np, peak_rate, true_params):
    # ✏️ YOUR TURN: replace the two `None` lines inside the loop, then run the cell.
    rel_diff = {"Poisson NLL": [], "weighted χ²": []}
    _rng = np.random.default_rng(42)
    for _ in range(20):
        _y = _rng.poisson(peak_rate(true_params, 3))
        for _kind in rel_diff:
            _lam_hat = None      # fitted rates: fit with fit_peak, then evaluate peak_rate
            _rd = None           # relative difference of the sums
            rel_diff[_kind].append(_rd)
    return (rel_diff,)


@app.cell(hide_code=True)
def ex_cons_check(checklist, mo, np, rel_diff):
    try:
        _vals = {_k: np.asarray(_v, dtype=float) for _k, _v in rel_diff.items()}
        _ready = all(len(_v) == 20 and np.all(np.isfinite(_v)) for _v in _vals.values())
    except (TypeError, ValueError):
        _ready = False
    if not _ready:
        _out = checklist([], "Fill in the two `None` lines in the loop above. The check runs automatically.")
    else:
        _p, _w = _vals["Poisson NLL"], _vals["weighted χ²"]
        _items = [
            (np.max(np.abs(_p)) < 0.01,
             f"Poisson NLL conserves the total counts: mean {100 * _p.mean():+.2f} %, max |diff| {100 * np.max(np.abs(_p)):.2f} % (want < 1 %)"),
            (np.mean(_w) < -0.05,
             f"weighted χ² loses counts: mean {100 * _w.mean():+.2f} %, max |diff| {100 * np.max(np.abs(_w)):.2f} % (want mean < −5 %)"),
        ]
        _out = checklist(_items)
    _out
    return


@app.cell(hide_code=True)
def ex_cons_sol(mo):
    mo.accordion({"💡 Solution": mo.md(r"""
    ```python
    _lam_hat = peak_rate(fit_peak(_y, 3, _kind), 3)
    _rd = (_lam_hat.sum() - _y.sum()) / _y.sum()
    ```
    Add `fit_peak` to the names the cell uses; marimo does this for you.

    *Why does the Poisson fit conserve counts?* Set the derivative of $\sum_i(\lambda_i - y_i\log\lambda_i)$
    with respect to an overall scale factor to zero. This gives $\sum_i \lambda_i = \sum_i y_i$.
    Here the model is homogeneous in $(a, b)$: scaling both by $c$ scales every $\lambda_i$ by $c$.
    So the combination $a\,\partial/\partial a + b\,\partial/\partial b$ of the two stationarity conditions
    gives $\sum_i(\lambda_i - y_i) = 0$.
    """)})
    return


@app.cell(hide_code=True)
def summary(mo):
    mo.md(r"""
    ---
    ## Summary

    | concept | formula / code |
    |---|---|
    | Poisson noise model | `noisy = rng.poisson(dose * phantom)` |
    | Var = mean | $\mathrm{Var}(N) = \lambda$: the basic Poisson property |
    | SNR law | $\mathrm{SNR} = \sqrt{\lambda}$: it grows only as √dose |
    | two times the SNR | needs four times the dose, not two times |
    | CNR | $\mathrm{CNR} = \lvert I_\text{defect} - I_\text{normal}\rvert / \sigma_\text{noise}$ |
    | detection threshold | CNR ≈ 3–5 for reliable detection by eye |
    | mixed model (real detector) | $x = g \cdot \mathrm{Poisson}(\lambda) + \mathcal{N}(0, \sigma_r^2)$ |
    | variance–mean plot | $\mathrm{Var}(x) = g\,\bar x + \sigma_r^2$: slope = gain, intercept = read-noise variance |
    | loss = NLL | Gaussian → MSE; Poisson → $\ell = \lambda - y\log\lambda$ |
    | low-count fits | weighted χ² and Anscombe + MSE are biased low; MSE is unbiased but scatters more; Poisson NLL is unbiased, most precise and conserves counts |
    """)
    return


@app.cell(hide_code=True)
def next_week(mo):
    mo.md(r"""
    ---
    ## Next week and references

    **Next: Unit 3, *Linear algebra, PCA & spectral unmixing*.** PCA minimises a least-squares objective.
    So it assumes Gaussian noise with equal variance: the MSE world of Part 8. On low-count EELS and EDS
    spectrum images this assumption fails. Week 3 therefore applies a Poisson weighting before PCA, and it
    uses NMF. The KL-divergence version of NMF is exactly the Poisson NLL that you implemented today.

    Before Unit 3: run every cell of this notebook once more, move each slider, and finish the exercises
    that you did not complete in class. You need the √λ law and the recipe "noise model → loss" to
    understand why PCA components of low-dose spectrum images can contain noise. Questions go to the StudOn forum.

    **References**

    - Lecture slides, Week 2: *Learning from EM data: signal formation, noise & loss*
      (`data_science_for_em/02_em_data_formation/01_intro.qmd`). The low-count fit uses the code of
      `02_em_data_formation/img/make_figures.py`.
    - F. J. Anscombe, *The transformation of Poisson, binomial and negative-binomial data*,
      Biometrika **35**, 246–254 (1948), <https://doi.org/10.1093/biomet/35.3-4.246>.
    """)
    return


if __name__ == "__main__":
    app.run()
