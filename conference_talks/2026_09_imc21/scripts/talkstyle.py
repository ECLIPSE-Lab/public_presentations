"""Shared matplotlib style for the IMC21 slide figures.

The deck (`custom.scss`) is a black-background reveal.js theme, so figures are drawn
with a transparent canvas and light foreground: they sit on the slide instead of
punching a white rectangle into it. Fonts are ~1.8x the paper sizes because a panel
that fills half a slide is viewed from the back of a lecture hall.

Set DARK=0 in the environment to get the paper's white-background look instead (useful
when a panel has to be printed or dropped into a manuscript).

Panel letters are deliberately absent: every panel is saved as its own figure, and the
slide title carries what the caption letter used to.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DARK = os.environ.get("DARK", "1") == "1"

# Foreground / background for text, spines and ticks.
FG = "#eeeeee" if DARK else "#222222"
BG = "none" if DARK else "white"
# Neutral reference-curve colour: replaces the paper's "k--" ideal-aperture lines,
# which are invisible on a black slide.
REF = "#ffffff" if DARK else "black"
# Semi-opaque box behind inset annotations, matched to the slide background.
BOX = dict(boxstyle="round,pad=0.35", fc="#111111" if DARK else "white",
           ec="#666666", alpha=0.85)

# Brighter tab10 in the same order, so C0/C1/C2 keep the paper's meaning while staying
# legible against black.
CYCLE_DARK = ["#4da3ff", "#ffa64d", "#5ddd7a", "#ff6b6b", "#c79bf0",
              "#d0a087", "#ff9ad5", "#b0b0b0", "#e8e34a", "#4fd8e8"]

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "figs_imc21", "vortex")


def style():
    """Install the slide rcParams. Call once at the top of a figure script."""
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.transparent": DARK,

        "text.color": FG,
        "axes.labelcolor": FG,
        "axes.edgecolor": FG,
        "xtick.color": FG,
        "ytick.color": FG,

        "axes.labelsize": 21,
        "axes.titlesize": 21,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 15,
        "font.size": 17,

        "axes.linewidth": 1.4,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "lines.linewidth": 2.2,
        "lines.markersize": 8,

        "grid.color": FG,
        "grid.alpha": 0.25,
        "legend.framealpha": 0.0,
        "legend.labelcolor": FG,

        "axes.prop_cycle": plt.cycler(color=CYCLE_DARK if DARK
                                      else plt.rcParams["axes.prop_cycle"].by_key()["color"]),
    })


def newfig(w=7.4, h=5.6):
    """A single-axes figure sized for roughly half a 1920x1080 slide."""
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def savepanel(fig, name: str, dpi: int = 200, pdf: bool = True, pad_inches: float = 0.12):
    """Write `<name>.png` (+ .pdf) into figs_imc21/vortex/ and close the figure."""
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches,
                transparent=DARK)
    print(f"wrote {os.path.relpath(out)}")
    if pdf:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight",
                    pad_inches=pad_inches, transparent=DARK)
    plt.close(fig)
    return out


def cbar(fig, ax, mappable, label, fraction=0.046, pad=0.04):
    """Colourbar with light frame/ticks so it reads on the dark slide."""
    cb = fig.colorbar(mappable, ax=ax, fraction=fraction, pad=pad)
    cb.set_label(label, color=FG, fontsize=19)
    cb.ax.tick_params(colors=FG, labelsize=15)
    cb.outline.set_edgecolor(FG)
    return cb
