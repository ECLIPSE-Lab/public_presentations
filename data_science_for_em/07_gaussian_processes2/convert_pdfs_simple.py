#!/usr/bin/env python3
"""
Simple PDF to PNG Converter
===========================

This script converts all PDF files in the img folder to PNG format
using system tools (pdftoppm or convert).

Requirements:
- poppler-utils (pdftoppm) OR ImageMagick (convert)
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
    
    if not any(tools.values()):
        print("✗ No PDF conversion tools found!")
        print("Please install one of the following:")
        print("  poppler-utils: sudo apt-get install poppler-utils")
        print("  ImageMagick: sudo apt-get install imagemagick")
        return None
    
    return tools

def convert_with_pdftoppm(pdf_path, output_dir, dpi=300):
    """Convert PDF to PNG using pdftoppm."""
    base_name = Path(pdf_path).stem
    output_path = os.path.join(output_dir, f"{base_name}.png")
    
    cmd = [
        'pdftoppm', 
        '-png', 
        '-r', str(dpi),
        '-singlefile',
        str(pdf_path),
        output_path.replace('.png', '')  # pdftoppm adds .png extension
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  Converted: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e}")
        return False

def convert_with_imagemagick(pdf_path, output_dir, dpi=300):
    """Convert PDF to PNG using ImageMagick convert."""
    base_name = Path(pdf_path).stem
    output_path = os.path.join(output_dir, f"{base_name}.png")
    
    cmd = [
        'convert',
        '-density', str(dpi),
        '-quality', '100',
        str(pdf_path),
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  Converted: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e}")
        return False

def main():
    """Main function to convert all PDFs in the img folder."""
    print("Simple PDF to PNG Converter")
    print("=" * 50)
    
    # Check available tools
    tools = check_system_tools()
    if not tools:
        return
    
    # Choose the best available tool
    if tools.get('pdftoppm'):
        convert_func = convert_with_pdftoppm
        tool_name = "pdftoppm"
    else:
        convert_func = convert_with_imagemagick
        tool_name = "convert"
    
    print(f"Using: {tool_name}")
    
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
        print(f"Converting {pdf_file.name}...")
        if convert_func(pdf_file, img_dir):
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