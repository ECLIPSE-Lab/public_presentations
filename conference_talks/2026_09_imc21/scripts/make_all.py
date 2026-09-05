"""Regenerate every IMC21 vortex figure.

    python make_all.py

Environment:
  DARK=0              white-background variant (default is the dark deck style)
  PAPER_FIGS=...      where the paper's companion npz live
  DATA_DIR=...        reconstructed probe/object stacks (galleries only)
  CONTROL_DATA_DIR=...  round-beam control reconstructions

The quantitative panels only need PAPER_FIGS. The galleries need the reconstructions and
are skipped with a message if they are not mounted, so the deck stays buildable anywhere.
"""
import vortex_galleries
import vortex_panels
from talkstyle import DARK, style


def main():
    print(f"style: {'dark (transparent, light text)' if DARK else 'light (white background)'}")
    style()
    print("\n--- quantitative panels (npz-driven) ---")
    for fn in vortex_panels.PANELS:
        fn()
    print("\n--- galleries (reconstruction-driven) ---")
    for fn in vortex_galleries.GALLERIES:
        fn()


if __name__ == "__main__":
    main()
