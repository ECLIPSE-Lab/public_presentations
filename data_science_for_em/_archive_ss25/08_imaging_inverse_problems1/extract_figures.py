import fitz  # PyMuPDF
import os

pdf_path = "data_science_for_em/08_imaging_inverse_problems1/deep_inverse_reader.pdf"
output_dir = "data_science_for_em/08_imaging_inverse_problems1/extracted_figures"

os.makedirs(output_dir, exist_ok=True)

doc = fitz.open(pdf_path)
img_count = 0

for page_num in range(len(doc)):
    page = doc[page_num]
    images = page.get_images(full=True)
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        img_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
        img_path = os.path.join(output_dir, img_filename)
        with open(img_path, "wb") as img_file:
            img_file.write(image_bytes)
        img_count += 1

print(f"Extracted {img_count} images to '{output_dir}'")