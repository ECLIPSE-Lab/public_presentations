#!/usr/bin/env python3
"""
PDF to PNG Converter with Auto-Cropping
=======================================

This script converts all PDF files in the img folder to PNG format
and automatically crops them to remove white space around the content.

Requirements:
- poppler-utils (pdftoppm)
- ImageMagick (convert) for cropping
"""

import os
import subprocess
import sys
from pathlib import Path

def check_system_tools():
    """Check which system tools are available for PDF conversion."""
    tools = {}
    
    # Check for pdftoppm (poppler-utils)
    try:
        result = subprocess.run(['pdftoppm', '-h'], capture_output=True, text=True)
        if result.returncode == 0:
            tools['pdftoppm'] = True
            print("✓ pdftoppm (poppler-utils) is available")
        else:
            tools['pdftoppm'] = False
    except FileNotFoundError:
        tools['pdftoppm'] = False
    
    # Check for convert (ImageMagick)
    try:
        result = subprocess.run(['convert', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            tools['convert'] = True
            print("✓ convert (ImageMagick) is available")
        else:
            tools['convert'] = False
    except FileNotFoundError:
        tools['convert'] = False
    
    if not tools.get('pdftoppm'):
        print("✗ pdftoppm (poppler-utils) not found!")
        print("Please install: sudo apt-get install poppler-utils")
        return None
    
    if not tools.get('convert'):
        print("✗ convert (ImageMagick) not found!")
        print("Please install: sudo apt-get install imagemagick")
        return None
    
    return tools

def get_image_dimensions(image_path):
    """Get image dimensions using ImageMagick identify."""
    cmd = ['identify', '-format', '%wx%h', str(image_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split('x'))
        return width, height
    except subprocess.CalledProcessError:
        return None, None

def crop_image_to_content(image_path, output_path, fuzz_percent=5):
    """
    Crop image to remove white/transparent borders around content.
    
    Args:
        image_path: Path to input image
        output_path: Path to output cropped image
        fuzz_percent: Percentage of color variation to consider as background
    """
    cmd = [
        'convert',
        str(image_path),
        '-fuzz', f'{fuzz_percent}%',
        '-trim',  # Remove surrounding white/transparent areas
        '+repage',  # Reset canvas to cropped size
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error cropping {image_path}: {e}")
        return False

def convert_pdf_with_cropping(pdf_path, output_dir, dpi=300, fuzz_percent=5):
    """
    Convert PDF to PNG and crop to content area.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save PNG files
        dpi: Resolution for conversion
        fuzz_percent: Percentage of color variation for cropping
    """
    base_name = Path(pdf_path).stem
    
    # Step 1: Convert PDF to PNG (full page)
    temp_png = os.path.join(output_dir, f"{base_name}_temp.png")
    output_png = os.path.join(output_dir, f"{base_name}.png")
    
    cmd = [
        'pdftoppm', 
        '-png', 
        '-r', str(dpi),
        '-singlefile',
        str(pdf_path),
        temp_png.replace('.png', '')  # pdftoppm adds .png extension
    ]
    
    try:
        print(f"Converting {pdf_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Step 2: Crop to content area
        print(f"  Cropping to content...")
        if crop_image_to_content(temp_png, output_png, fuzz_percent):
            # Get dimensions for comparison
            orig_width, orig_height = get_image_dimensions(temp_png)
            crop_width, crop_height = get_image_dimensions(output_png)
            
            if orig_width and orig_height and crop_width and crop_height:
                reduction = ((orig_width * orig_height) - (crop_width * crop_height)) / (orig_width * orig_height) * 100
                print(f"  Cropped: {orig_width}x{orig_height} → {crop_width}x{crop_height} ({reduction:.1f}% size reduction)")
            
            # Clean up temporary file
            os.remove(temp_png)
            return True
        else:
            # If cropping failed, use the original
            os.rename(temp_png, output_png)
            print(f"  Used original (cropping failed)")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"  Error converting {pdf_path}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_png):
            os.remove(temp_png)
        return False

def main():
    """Main function to convert all PDFs in the img folder."""
    print("PDF to PNG Converter with Auto-Cropping")
    print("=" * 50)
    
    # Check available tools
    tools = check_system_tools()
    if not tools:
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
    
    print(f"\nConverting PDFs to PNG (DPI: 300) with auto-cropping...")
    print("-" * 50)
    
    # Convert each PDF
    successful_conversions = 0
    failed_conversions = 0
    
    for pdf_file in pdf_files:
        if convert_pdf_with_cropping(pdf_file, img_dir):
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    print("-" * 50)
    print(f"Conversion complete!")
    print(f"Successful: {successful_conversions}")
    print(f"Failed: {failed_conversions}")
    
    if successful_conversions > 0:
        print(f"\nCropped PNG files have been saved in the {img_dir} directory.")
        print("The images have been automatically cropped to remove white space.")
        print("You can now delete the original PDF files if desired.")

if __name__ == "__main__":
    main() 