#!/usr/bin/env python3
"""
PDF to PNG Converter
====================

This script converts all PDF files in the img folder to PNG format.
Uses pdf2image library for high-quality conversion.

Requirements:
- pdf2image: pip install pdf2image
- poppler-utils (system dependency)
  - Ubuntu/Debian: sudo apt-get install poppler-utils
  - macOS: brew install poppler
  - Windows: Download from http://blog.alivate.com.au/poppler-windows/
"""

import os
import glob
from pathlib import Path
import subprocess
import sys

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import pdf2image
        print("✓ pdf2image is installed")
    except ImportError:
        print("✗ pdf2image is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdf2image"])
        print("✓ pdf2image installed successfully")
    
    # Check if poppler is available
    try:
        result = subprocess.run(['pdftoppm', '-h'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ poppler-utils is available")
        else:
            print("✗ poppler-utils not found")
            print("Please install poppler-utils:")
            print("  Ubuntu/Debian: sudo apt-get install poppler-utils")
            print("  macOS: brew install poppler")
            print("  Windows: Download from http://blog.alivate.com.au/poppler-windows/")
            return False
    except FileNotFoundError:
        print("✗ poppler-utils not found")
        print("Please install poppler-utils:")
        print("  Ubuntu/Debian: sudo apt-get install poppler-utils")
        print("  macOS: brew install poppler")
        print("  Windows: Download from http://blog.alivate.com.au/poppler-windows/")
        return False
    
    return True

def convert_pdf_to_png(pdf_path, output_dir, dpi=300):
    """
    Convert a single PDF file to PNG.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save PNG files
        dpi (int): Resolution for the output PNG (default: 300)
    """
    try:
        from pdf2image import convert_from_path
        
        # Get the base filename without extension
        base_name = Path(pdf_path).stem
        
        # Convert PDF to images
        print(f"Converting {pdf_path}...")
        images = convert_from_path(pdf_path, dpi=dpi)
        
        # Save each page as PNG
        for i, image in enumerate(images):
            if len(images) == 1:
                # Single page PDF
                output_path = os.path.join(output_dir, f"{base_name}.png")
            else:
                # Multi-page PDF
                output_path = os.path.join(output_dir, f"{base_name}_page_{i+1}.png")
            
            image.save(output_path, 'PNG')
            print(f"  Saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")
        return False

def main():
    """Main function to convert all PDFs in the img folder."""
    print("PDF to PNG Converter")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("Dependencies not met. Exiting.")
        return
    
    # Define paths
    img_dir = Path("img")
    if not img_dir.exists():
        print(f"Error: {img_dir} directory not found!")
        return
    
    # Find all PDF files
    pdf_files = list(img_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the img directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s):")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    print(f"\nConverting PDFs to PNG (DPI: 300)...")
    print("-" * 50)
    
    # Convert each PDF
    successful_conversions = 0
    failed_conversions = 0
    
    for pdf_file in pdf_files:
        if convert_pdf_to_png(str(pdf_file), str(img_dir)):
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    print("-" * 50)
    print(f"Conversion complete!")
    print(f"Successful: {successful_conversions}")
    print(f"Failed: {failed_conversions}")
    
    if successful_conversions > 0:
        print(f"\nPNG files have been saved in the {img_dir} directory.")
        print("You can now delete the original PDF files if desired.")

if __name__ == "__main__":
    main() 