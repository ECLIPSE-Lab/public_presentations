"""Download PDFs for the remaining 7 case-study papers in unit 04 and extract figure images.

Uses verified DOIs (no title search). Tries Unpaywall first; falls back to direct
publisher URLs for known open-access venues (Nature OA, Frontiers, IOP, etc.).
"""
import os
import sys
import time
import requests
import fitz  # PyMuPDF
import urllib.parse

EMAIL = "braunphil@gmail.com"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verified keys + DOIs from ref.bib. Known arXiv preprints are listed as
# fallbacks for papers whose accepted publisher version is paywalled.
papers = [
    {"key": "ostormujof2022deep",     "doi": "10.1016/j.matchar.2021.111638",  "arxiv": "2112.03072"},
    {"key": "durmaz2021deep",         "doi": "10.1038/s41467-021-26565-5"},
    {"key": "yang2018cnn",            "doi": "10.1016/j.commatsci.2018.05.014"},
    {"key": "heidenreich2023yield",   "doi": "10.1016/j.ijplas.2022.103506"},
    {"key": "fischer2022xception",    "doi": "10.1016/j.matdes.2022.111029"},
    {"key": "oster2024thermographic", "doi": "10.1007/s10845-023-02117-0"},
    {"key": "loganathan2026fsw",      "doi": "10.1016/j.jmsy.2026.01.007"},
    {"key": "parmar2026weld",         "doi": "10.1038/s41598-026-44874-x"},
]


def unpaywall_pdf(doi):
    url = f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}"
    try:
        res = requests.get(url, timeout=15)
        data = res.json()
        best_oa = data.get("best_oa_location")
        if best_oa and best_oa.get("url_for_pdf"):
            return best_oa["url_for_pdf"]
        # Fallback: any OA location
        for loc in data.get("oa_locations", []) or []:
            if loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
    except Exception as e:
        print(f"  Unpaywall error: {e}")
    return None


def download_pdf(url, dest_path):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            "Accept": "application/pdf,*/*;q=0.8",
        }
        res = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        res.raise_for_status()
        ctype = res.headers.get("Content-Type", "")
        if ctype.startswith("application/pdf") or url.endswith(".pdf") or res.content[:4] == b"%PDF":
            with open(dest_path, "wb") as f:
                f.write(res.content)
            return True
        print(f"  Not a PDF (Content-Type={ctype}, first bytes={res.content[:8]!r})")
    except Exception as e:
        print(f"  Download error: {e}")
    return False


def extract_images(pdf_path, key, min_bytes=20000):
    try:
        doc = fitz.open(pdf_path)
        count = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base = doc.extract_image(xref)
                if len(base["image"]) < min_bytes:
                    continue
                filename = f"{key}_p{page_num+1}_{img_index+1}.{base['ext']}"
                with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
                    f.write(base["image"])
                count += 1
        return count
    except Exception as e:
        print(f"  Extract error: {e}")
        return 0


for paper in papers:
    key, doi = paper["key"], paper["doi"]
    print(f"\n=== {key}  ({doi}) ===")

    # Skip if we already have images for this key
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(key + "_p")]
    if existing:
        print(f"  Already have {len(existing)} images; skipping.")
        continue

    candidate_urls = []
    pdf_url = unpaywall_pdf(doi)
    if pdf_url:
        candidate_urls.append(("unpaywall", pdf_url))
    # Direct Nature OA URL fallback
    if doi.startswith("10.1038/"):
        nat_id = doi.split("/", 1)[1]
        candidate_urls.append(("nature-direct", f"https://www.nature.com/articles/{nat_id}.pdf"))
    # arXiv preprint fallback
    if "arxiv" in paper:
        candidate_urls.append(("arxiv", f"https://arxiv.org/pdf/{paper['arxiv']}.pdf"))

    if not candidate_urls:
        print("  No PDF URL found.")
        continue

    pdf_path = f"/tmp/{key}.pdf"
    downloaded = False
    for label, url in candidate_urls:
        print(f"  Trying [{label}] {url}")
        if download_pdf(url, pdf_path):
            print(f"  Downloaded {os.path.getsize(pdf_path)} bytes from {label}.")
            downloaded = True
            break

    if not downloaded:
        print("  All download attempts failed.")
        continue

    n = extract_images(pdf_path, key)
    print(f"  Extracted {n} images.")

    time.sleep(1)

print("\nDone.")
