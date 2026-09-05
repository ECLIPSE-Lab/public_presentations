"""Placeholder tile art for the roadmap.

Only part 3 (liquid cell) needs one: the other four tiles point straight at real figures
and are cropped by the CSS (object-fit: cover). Replace this file once the liquid-cell
figures exist.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "figs_imc21", "tiles")


def liquidcell_placeholder():
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#3a4a5a")
    ax.text(0.5, 0.58, "liquid cell", ha="center", va="center", fontsize=30,
            color="#87cefa", transform=ax.transAxes)
    ax.text(0.5, 0.38, "figures TODO", ha="center", va="center", fontsize=18,
            color="#6a7a8a", transform=ax.transAxes, style="italic")
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "liquidcell_placeholder.png")
    fig.savefig(p, dpi=150, facecolor="#0a0a0a", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"wrote {os.path.relpath(p)}")


if __name__ == "__main__":
    liquidcell_placeholder()
