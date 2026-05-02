import os
import requests
import fitz

OUTPUT_DIR = "/home/philipp/projects/_public_presentations/ml_for_characterization_and_processing/unit04_microstructure_representations/images"

papers = [
    {"key": "roberts2019stemdefects", "url": "https://www.nature.com/articles/s41598-019-49105-0.pdf"},
    {"key": "park2017xrd", "url": "https://www.nature.com/articles/s41467-019-13749-3.pdf"},
    {"key": "muller2021battery", "url": "https://www.nature.com/articles/s41467-021-26480-9.pdf"},
    {"key": "weld2026cnnvit", "url": "https://www.nature.com/articles/s41598-026-44874-x.pdf"},
    {"key": "mianroodi2021cp", "url": "https://www.nature.com/articles/s41524-021-00571-z.pdf"},
    {"key": "generale2022robust", "url": "https://www.frontiersin.org/articles/10.3389/fmats.2022.851085/pdf"}
]

def extract_images(pdf_path, key):
    try:
        doc = fitz.open(pdf_path)
        count = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                if len(image_bytes) < 20000:
                    continue
                filename = f"{key}_p{page_num+1}_{img_index+1}.{image_ext}"
                with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
                    f.write(image_bytes)
                count += 1
        return count
    except Exception as e:
        print(f"Extract error: {e}")
        return 0

for p in papers:
    key = p["key"]
    url = p["url"]
    print(f"Downloading {key} from {url}")
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}, timeout=15)
        if res.status_code == 200 and res.content.startswith(b'%PDF'):
            pdf_path = f"/tmp/{key}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(res.content)
            img_count = extract_images(pdf_path, key)
            print(f"Extracted {img_count} images.")
        else:
            print(f"Failed. Status {res.status_code}, content prefix: {res.content[:20]}")
    except Exception as e:
        print(f"Error: {e}")
