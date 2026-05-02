import os
import requests
import fitz  # PyMuPDF
import json
import urllib.parse
import xml.etree.ElementTree as ET
import time

EMAIL = "philipp@example.com"
OUTPUT_DIR = "/home/philipp/projects/_public_presentations/ml_for_characterization_and_processing/unit04_microstructure_representations/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

papers = [
    {"key": "azimi2018advanced", "title": "Advanced Steel Microstructural Classification by Deep Learning Methods"},
    {"key": "decost2017uhcs", "title": "Exploring the microstructure manifold: Image texture representations applied to ultrahigh carbon steel microstructures"},
    {"key": "decost2019uhcs", "title": "High Throughput Quantitative Metallography for Complex Microstructures Using Deep Learning"},
    {"key": "holm2020overview", "title": "Overview: Computer Vision and Machine Learning for Microstructural Characterization and Analysis"},
    {"key": "bachmann2022leveraging", "title": "Leveraging EBSD data by deep learning for bainite, ferrite and martensite segmentation"},
    {"key": "stoll2021deep", "title": "A deep learning approach for complex microstructure inference"},
    {"key": "trampert2023deep", "title": "Deep Learning of Crystalline Defects from TEM images: A Solution for the Problem of Never Enough Training Data", "arxiv": "2307.06322"},
    {"key": "roberts2019stemdefects", "title": "Deep Learning for Semantic Segmentation of Defects in Advanced STEM Images of Steels"},
    {"key": "yang2018cnn", "title": "Deep learning approaches for mining structure-property linkages in high contrast composites from simulation datasets"},
    {"key": "frankel2022yield", "title": "Modeling structure-property relationships with convolutional neural networks: Yield surface prediction based on microstructure images"},
    {"key": "park2017xrd", "title": "A deep-learning technique for phase identification in multiphase inorganic compounds using synthetic XRD powder patterns"},
    {"key": "muller2021battery", "title": "Deep learning-based segmentation of lithium-ion battery microstructures enhanced by artificially generated electrodes"},
    {"key": "generale2022robust", "title": "Development of a Robust CNN Model for Capturing Microstructure-Property Linkages and Building Property Closures Supporting Material Design"},
    {"key": "grasso2024lpbf", "title": "Deep learning-based automated defect classification for powder bed fusion -- Laser beam"},
    {"key": "gobert2024pore", "title": "A deep learning framework for defect prediction based on thermographic in-situ monitoring in laser powder bed fusion"},
    {"key": "fsw2026realtime", "title": "Machine vision defect segmentation and geometric measurement for real time quality monitoring in friction stir welding"},
    {"key": "weld2026cnnvit", "title": "A misclassification-aware explainable hybrid CNN-vision transformer framework for radiographic weld inspection"},
    {"key": "mianroodi2021cp", "title": "Teaching solid mechanics to artificial intelligence -- a fast solver for heterogeneous materials"}
]

def search_crossref(title):
    url = f"https://api.crossref.org/works?query.title={urllib.parse.quote(title)}&rows=1"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        items = data['message']['items']
        if items:
            return items[0]['DOI']
    except Exception as e:
        print(f"  Crossref error: {e}")
    return None

def search_unpaywall(doi):
    url = f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        best_oa = data.get('best_oa_location')
        if best_oa and best_oa.get('url_for_pdf'):
            return best_oa['url_for_pdf']
    except Exception as e:
        print(f"  Unpaywall error: {e}")
    return None

def search_arxiv(title):
    clean_title = title.replace('"', '').replace("'", "")
    query = f"ti:\"{clean_title}\""
    url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&max_results=1"
    try:
        res = requests.get(url, timeout=10)
        root = ET.fromstring(res.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        if entry is not None:
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    return link.get('href') + ".pdf"
    except Exception as e:
        print(f"  ArXiv error: {e}")
    return None

def download_pdf(url, dest_path):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        if res.headers.get('Content-Type', '').startswith('application/pdf') or url.endswith('.pdf') or res.content.startswith(b'%PDF'):
            with open(dest_path, 'wb') as f:
                f.write(res.content)
            return True
    except Exception as e:
        print(f"  Download error: {e}")
    return False

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
        print(f"  Extract error: {e}")
        return 0

for paper in papers:
    key = paper['key']
    title = paper['title']
    print(f"\\nProcessing {key}: {title}")
    
    pdf_url = None
    if 'arxiv' in paper:
        pdf_url = f"https://arxiv.org/pdf/{paper['arxiv']}.pdf"
    
    if not pdf_url:
        print("  Searching arXiv...")
        pdf_url = search_arxiv(title)
        
    if not pdf_url:
        print("  Searching Crossref...")
        doi = search_crossref(title)
        if doi:
            print(f"  Found DOI: {doi}")
            pdf_url = search_unpaywall(doi)
    
    if pdf_url:
        print(f"  Found PDF URL: {pdf_url}")
        pdf_path = f"/tmp/{key}.pdf"
        if download_pdf(pdf_url, pdf_path):
            print("  Downloaded PDF.")
            img_count = extract_images(pdf_path, key)
            print(f"  Extracted {img_count} images.")
        else:
            print("  Failed to download PDF.")
    else:
        print("  Could not find open access PDF.")
    
    time.sleep(1)

print("\\nDone.")
