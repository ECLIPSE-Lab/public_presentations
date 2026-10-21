#!/usr/bin/env python3
"""
PDF Image Extractor
===================

This script extracts all images from a PDF file and saves them as PNG files.
Uses pdf2image for high-quality extraction.

Requirements:
- pdf2image: pip install pdf2image
- poppler-utils (system dependency)
"""

import os
import subprocess
import sys
from pathlib import Path
import fitz  # PyMuPDF

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fitz
        print("✓ PyMuPDF is installed")
    except ImportError:
        print("✗ PyMuPDF is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF"])
        print("✓ PyMuPDF installed successfully")
    
    return True

def extract_images_with_pymupdf(pdf_path, output_dir, dpi=300):
    """
    Extract images from PDF using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
        dpi: Resolution for extraction
    """
    try:
        import fitz
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        print(f"PDF has {len(doc)} pages")
        
        extracted_count = 0
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Get image list for this page
            image_list = page.get_images()
            
            if image_list:
                print(f"Page {page_num + 1}: Found {len(image_list)} images")
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert to RGB if necessary
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        else:  # CMYK
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        
                        # Save image
                        output_filename = f"page_{page_num + 1:03d}_img_{img_index + 1:03d}.png"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        pix1.save(output_path)
                        print(f"  Saved: {output_filename}")
                        extracted_count += 1
                        
                        pix1 = None  # Free memory
                        
                    except Exception as e:
                        print(f"  Error extracting image {img_index + 1}: {e}")
                        continue
            else:
                print(f"Page {page_num + 1}: No images found")
        
        doc.close()
        return extracted_count
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return 0

def extract_images_with_pdfimages(pdf_path, output_dir):
    """
    Extract images using pdfimages command (poppler-utils).
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Change to output directory for pdfimages
        original_dir = os.getcwd()
        os.chdir(output_dir)
        
        # Run pdfimages command
        cmd = ['pdfimages', '-all', '-p', str(pdf_path), 'extracted_image']
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Change back to original directory
        os.chdir(original_dir)
        
        # Count extracted files
        extracted_files = list(Path(output_dir).glob("extracted_image*"))
        print(f"Extracted {len(extracted_files)} images")
        
        return len(extracted_files)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running pdfimages: {e}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0

def main():
    """Main function to extract images from PDF."""
    print("PDF Image Extractor")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("Dependencies not met. Exiting.")
        return
    
    # Define paths
    pdf_file = "SummerSchool_MLinEM_Lec20_01.pdf"
    output_dir = "extracted_images"
    
    if not os.path.exists(pdf_file):
        print(f"Error: {pdf_file} not found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting images from: {pdf_file}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Try PyMuPDF first (more reliable)
    print("Method 1: Using PyMuPDF...")
    count1 = extract_images_with_pymupdf(pdf_file, output_dir)
    
    # Try pdfimages as backup
    print("\nMethod 2: Using pdfimages...")
    count2 = extract_images_with_pdfimages(pdf_file, output_dir)
    
    print("-" * 50)
    print(f"Extraction complete!")
    print(f"PyMuPDF extracted: {count1} images")
    print(f"pdfimages extracted: {count2} images")
    
    if count1 > 0 or count2 > 0:
        print(f"\nImages have been saved in the '{output_dir}' directory.")
        
        # List extracted files
        extracted_files = list(Path(output_dir).glob("*.png")) + list(Path(output_dir).glob("*.ppm")) + list(Path(output_dir).glob("*.pbm"))
        if extracted_files:
            print(f"\nExtracted files:")
            for file in sorted(extracted_files):
                print(f"  - {file.name}")
    else:
        print("No images were extracted. The PDF might not contain extractable images.")

if __name__ == "__main__":
    main() 