"""Extract Figure 2 and Figure 4 from Hofer et al. (J. Microscopy 2025)."""
import fitz
from pathlib import Path

PDF = Path(__file__).parent / (
    "Journal of Microscopy - 2025 - Hofer - "
    "Detecting charge transfer at defects in 2D materials with electron ptychography.pdf"
)
OUT = Path(__file__).parent / "hofer2025"
OUT.mkdir(exist_ok=True)

FIG_PAGES = {2: 4, 4: 7}  # figure number -> 1-based page index
DPI = 300


def crop_figure_region(page: fitz.Page) -> fitz.Rect:
    """Crop from top margin to just above the FIGURE caption block."""
    page_rect = page.rect
    caption_y = page_rect.height
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        text = "".join(
            span["text"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )
        if "F I G U R E" in text:
            caption_y = min(caption_y, block["bbox"][1] - 4)
    top = 50  # skip header
    return fitz.Rect(30, top, page_rect.width - 30, caption_y)


def main() -> None:
    doc = fitz.open(PDF)
    mat = fitz.Matrix(DPI / 72, DPI / 72)

    for fig_num, page_num in FIG_PAGES.items():
        page = doc[page_num - 1]
        crop = crop_figure_region(page)
        pix = page.get_pixmap(matrix=mat, clip=crop, alpha=False)
        out_png = OUT / f"fig{fig_num}.png"
        pix.save(out_png)
        print(f"Saved {out_png} ({pix.width}x{pix.height})")

        # Also save embedded raster images on that page
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            ext = base["ext"]
            out_embed = OUT / f"page_{page_num:03d}_img_{img_index + 1:03d}.{ext}"
            out_embed.write_bytes(base["image"])
            print(f"  embedded -> {out_embed.name}")

    doc.close()


if __name__ == "__main__":
    main()
