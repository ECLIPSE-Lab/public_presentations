"""IMC21 slide galleries for the MEMS vortex probes (image panels).

Unlike `vortex_panels.py`, these need the reconstructed probe/object stacks themselves,
so they import the paper's `common.py` and read DATA_DIR. If the raw reconstructions are
not mounted, every gallery is skipped with a message instead of crashing, so the deck can
still be rebuilt from a machine without them.

The paper's own galleries are dense (10 rows x 16 columns) because a caption can carry
the reading instructions. A slide cannot, so each gallery here shows one thing: the
dominant mode across the whole charge series, or the mode stack of a single charge.

Run:  python vortex_galleries.py          (env: DATA_DIR, DARK)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from talkstyle import FG, cbar, savepanel, style

PAPER_SCRIPTS = os.environ.get(
    "PAPER_SCRIPTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "mems_ptychography", "scripts"),
)
sys.path.insert(0, os.path.abspath(PAPER_SCRIPTS))

# --- geometry / calibration (matches fig_mode_gallery.py) --------------------
N_PROBE = 192
ALPHA_MRAD = 1.5           # calibrated convergence semi-angle = aperture edge
LAMBDA_A = 0.025079        # 200 kV electron wavelength (Angstrom)
R_BF_PX = 38.0
DQ_INV_A = (ALPHA_MRAD * 1e-3 / LAMBDA_A) / R_BF_PX
DX_A = 1.0 / (N_PROBE * DQ_INV_A)          # real-space px -> Angstrom (~3.31)
MRAD_PER_PX = ALPHA_MRAD / R_BF_PX

HR, HF = 72, 46            # real-space / aperture crop half-widths (px)
GAMMA_R, GAMMA_F = 0.8, 0.95
NCOL = 7                   # 21 charges -> 3 x 7 grid

REAL_BAR_NM = 10.0
FOUR_BAR_MRAD = 1.0

# Representative charge for the single-charge mode stack: the largest programmed
# charge, where the mixedness is most pronounced.
STACK_STEM = "0036"

# Round-beam controls. These live in their own reconstruction folder, and it is a
# 4-physical-mode recon while the charge series is 8-mode. Since p_0 falls mechanically
# as more modes become available, quoting the series' l=0 point (p_0 ~ 0.64, 8 modes)
# beside a 4-mode control would flatter the claim that the device adds mixedness. So the
# gallery stays inside the control set, where the retracted-vs-inserted contrast is
# measured under identical settings.
CONTROL_DATA_DIR = os.environ.get(
    "CONTROL_DATA_DIR",
    "/mnt/data/raw/2025-05-24_vortex_data/descan_corrected/roundbeam_nomask",
)
CONTROL_ROWS = [
    ("0013", "device retracted"),
    ("0037", r"device inserted, $\ell = 0$"),
]


def _imports():
    """Import the paper helpers lazily so a missing repo gives one clear message."""
    import common
    import fig_objects
    return common, fig_objects


def have_data(common) -> bool:
    if os.path.isdir(common.DATA_DIR):
        return True
    print(f"SKIP galleries: DATA_DIR not found ({common.DATA_DIR}).\n"
          f"  Mount the reconstructions or set DATA_DIR to their location.")
    return False


def series(common):
    """One stem per charge from scans 0015-0036, descending delivered charge."""
    rows, seen = [], set()
    for i in range(15, 37):
        s = f"{i:04d}"
        if s == "0026":                       # duplicate l=-1 acquisition; 0027 kept
            continue
        l = common.L_MAP.get(s)
        if l is None or l in seen:
            continue
        seen.add(l)
        rows.append((l, s))
    rows.sort(key=lambda t: -t[0])
    return rows


def crop(a, half):
    H, W = a.shape[-2:]
    return a[..., H // 2 - half:H // 2 + half, W // 2 - half:W // 2 + half]


def com_center_each(common, P):
    """Centre every mode at its own intensity COM, as fig_mode_gallery does.

    Joint centring leaves a residual phase ramp on the weaker modes, which shows up in
    the aperture plane as an off-centre spiral.
    """
    return np.stack([common.com_center(P[m:m + 1])[0] for m in range(P.shape[0])])


def scalebar(ax, length_px, label, n):
    """Light scale bar, lower left, with the label above it."""
    from matplotlib.patches import Rectangle
    mgn, bar_h = 0.07 * n, 0.035 * n
    x0, y1 = mgn, n - mgn
    ax.add_patch(Rectangle((x0, y1 - bar_h), length_px, bar_h,
                           facecolor="white", edgecolor="none", zorder=5))
    ax.text(x0, y1 - bar_h - 0.03 * n, label, ha="left", va="bottom",
            color="white", fontsize=12, fontweight="bold", zorder=5)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ------------------------------------------------------------------ mode galleries

def _mode0_gallery(common, domain: str, name: str):
    """Dominant probe mode across the whole charge series, domain-coloured."""
    import matplotlib.pyplot as plt
    rows = series(common)
    slope = float(np.load(os.path.join(common.FIG_DIR, "fig_oam_sorter.npz"))["fit_mean"][0])
    half = HR if domain == "real" else HF
    gamma = GAMMA_R if domain == "real" else GAMMA_F

    nrow = int(np.ceil(len(rows) / NCOL))
    fig, axes = plt.subplots(nrow, NCOL, figsize=(2.35 * NCOL, 2.55 * nrow))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        bare(ax)
        ax.set_visible(False)

    for i, (lprog, s) in enumerate(rows):
        ax = axes[i // NCOL, i % NCOL]
        ax.set_visible(True)
        P, pw = common.load_probe(s)
        P = com_center_each(common, P)
        if domain != "real":
            P = common.to_fourier(P)
        m0 = crop(P[0], half)
        ax.imshow(common.complex_to_rgb(m0, gamma=gamma), interpolation="nearest")
        ax.set_title(rf"$\ell = {int(round(slope * lprog)):+d}$", fontsize=17,
                     color=FG, pad=4)
        if i == 0:
            n = 2 * half
            if domain == "real":
                scalebar(ax, REAL_BAR_NM * 10.0 / DX_A, f"{REAL_BAR_NM:.0f} nm", n)
            else:
                scalebar(ax, FOUR_BAR_MRAD / MRAD_PER_PX, f"{FOUR_BAR_MRAD:.0f} mrad", n)
    fig.subplots_adjust(wspace=0.06, hspace=0.16)
    savepanel(fig, name)


def mode_gallery_real():
    common, _ = _imports()
    if have_data(common):
        _mode0_gallery(common, "real", "mode_gallery_real")


def mode_gallery_fourier():
    common, _ = _imports()
    if have_data(common):
        _mode0_gallery(common, "fourier", "mode_gallery_fourier")


def mode_stack():
    """The four physical modes of one high-charge dataset: the mixed state, made visible."""
    import matplotlib.pyplot as plt
    common, _ = _imports()
    if not have_data(common):
        return
    slope = float(np.load(os.path.join(common.FIG_DIR, "fig_oam_sorter.npz"))["fit_mean"][0])
    P, pw = common.load_probe(STACK_STEM)
    P = com_center_each(common, P)
    F = common.to_fourier(P)
    nm = min(4, P.shape[0])
    ell = int(round(slope * common.L_MAP[STACK_STEM]))

    fig, axes = plt.subplots(2, nm, figsize=(2.7 * nm, 5.9))
    for m in range(nm):
        for r, (stack, half, g) in enumerate(
                ((P, HR, GAMMA_R), (F, HF, GAMMA_F))):
            ax = axes[r, m]
            bare(ax)
            ax.imshow(common.complex_to_rgb(crop(stack[m], half), gamma=g),
                      interpolation="nearest")
        axes[0, m].set_title(f"mode {m}\n{100 * pw[m]:.0f}% of the power",
                             fontsize=16, color=FG, pad=5)
    axes[0, 0].set_ylabel("sample plane", fontsize=17, color=FG, labelpad=10)
    axes[1, 0].set_ylabel("aperture plane", fontsize=17, color=FG, labelpad=10)
    # Clear of the two-line column titles.
    fig.suptitle(rf"delivered $\ell = {ell:+d}$", fontsize=21, color=FG, y=1.04)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    savepanel(fig, "mode_stack")


def roundbeam_controls():
    """Round-beam controls: mixedness is a property of the device, not of the algorithm."""
    import matplotlib.pyplot as plt
    common, _ = _imports()
    if not os.path.isdir(CONTROL_DATA_DIR):
        print(f"SKIP roundbeam_controls: CONTROL_DATA_DIR not found ({CONTROL_DATA_DIR})")
        return
    nm = 4
    fig, axes = plt.subplots(len(CONTROL_ROWS), nm,
                             figsize=(2.55 * nm, 2.8 * len(CONTROL_ROWS)))
    axes = np.atleast_2d(axes)
    orig = common.DATA_DIR
    common.DATA_DIR = CONTROL_DATA_DIR          # controls are a separate reconstruction
    try:
        rows = [(s, label, common.load_probe(s)) for s, label in CONTROL_ROWS]
    finally:
        common.DATA_DIR = orig
    for r, (s, label, (P, pw)) in enumerate(rows):
        F = common.to_fourier(common.com_center(P))
        for m in range(nm):
            ax = axes[r, m]
            bare(ax)
            ax.imshow(common.complex_to_rgb(crop(F[m], HF), gamma=GAMMA_F),
                      interpolation="nearest")
            ax.text(0.03, 0.97, f"{100 * pw[m]:.0f}%", transform=ax.transAxes,
                    fontsize=14, color="white", ha="left", va="top", fontweight="bold")
            if r == 0:
                ax.set_title(f"mode {m}", fontsize=16, color=FG, pad=5)
        axes[r, 0].set_ylabel(label, fontsize=15, color=FG, labelpad=12)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    savepanel(fig, "roundbeam_controls", pad_inches=0.25)


def objects():
    """Reconstructed object phase for representative charges: same specimen, every probe."""
    import matplotlib.pyplot as plt
    common, fig_objects = _imports()
    if not have_data(common):
        return
    slope = float(np.load(os.path.join(common.FIG_DIR, "fig_oam_sorter.npz"))["fit_mean"][0])
    want = [10, 5, 1, 0, -1, -5, -10]        # programmed charges, matches the coherence maps
    picks = [(l, s) for l, s in series(common) if l in want]

    fig, axes = plt.subplots(1, len(picks), figsize=(2.6 * len(picks), 3.2))
    for ax, (lprog, s) in zip(np.atleast_1d(axes), picks):
        bare(ax)
        ph = fig_objects.process(common.load_object(s)[0])
        lo, hi = np.percentile(ph, (2, 98))
        ax.imshow(ph, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(rf"$\ell = {int(round(slope * lprog)):+d}$", fontsize=17,
                     color=FG, pad=4)
    scalebar(np.atleast_1d(axes)[0], 20.0 * 10.0 / DX_A, "20 nm",
             fig_objects.process(common.load_object(picks[0][1])[0]).shape[0])
    fig.subplots_adjust(wspace=0.05)
    savepanel(fig, "objects")


GALLERIES = [mode_gallery_real, mode_gallery_fourier, mode_stack,
             roundbeam_controls, objects]


def main():
    style()
    for fn in GALLERIES:
        fn()


if __name__ == "__main__":
    main()
