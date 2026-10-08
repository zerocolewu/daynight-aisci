import os
import requests
import time
import re
import json
import shutil
import zipfile
from pathlib import Path
from dotenv import load_dotenv

def sanitize_filename(filename):
    """Clean filename, remove illegal characters"""
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
    filename = filename.strip()
    filename = re.sub(r'\s+', "_", filename)
    if len(filename) > 100:
        filename = filename[:100]
    return filename


def convert_pdfs_to_markdown(folder_path, api_token=None):
    """
    Convert all PDFs in a folder to markdown format using Mineru API.
    
    Args:
        folder_path: Path to folder containing PDF files
        api_token: API token for Mineru service (optional, uses default if not provided)
    
    Returns:
        List of successfully converted markdown file paths
    """
    # Use default token if not provided
    if api_token is None:
        load_dotenv()
        api_token = os.getenv("MINERU_API_TOKEN")
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder_path}")
        return []
    
    # Setup directories
    temp_folder = folder / "_mineru_temp"
    temp_folder.mkdir(exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(folder.glob('*.pdf'))
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Prepare API request
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_token}'
    }
    
    file_data = [{
        "name": f.name,
        "is_ocr": True,
        "data_id": f"file_{i}"
    } for i, f in enumerate(pdf_files)]
    
    # Request upload URLs
    print("Requesting upload URLs...")
    response = requests.post(
        'https://mineru.net/api/v4/file-urls/batch',
        headers=headers,
        json={
            "enable_formula": True,
            "language": "auto",
            "layout_model": "doclayout_yolo",
            "enable_table": True,
            "files": file_data
        }
    )
    
    if response.status_code != 200 or response.json()["code"] != 0:
        print(f"Failed to request upload URLs: {response.text}")
        return []
    
    batch_id = response.json()["data"]["batch_id"]
    urls = response.json()["data"]["file_urls"]
    
    # Upload files
    print("Uploading files...")
    for pdf_file, url in zip(pdf_files, urls):
        with open(pdf_file, 'rb') as f:
            upload_response = requests.put(url, data=f)
            if upload_response.status_code == 200:
                print(f"  ✓ {pdf_file.name}")
            else:
                print(f"  ✗ {pdf_file.name} failed")
    
    # Wait for processing
    print("\nWaiting for conversion...")
    status_url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'
    
    while True:
        status_response = requests.get(status_url, headers=headers)
        if status_response.status_code != 200:
            print(f"Failed to check status: {status_response.text}")
            return []
        
        batch_data = status_response.json()["data"]
        all_done = True
        
        for result in batch_data["extract_result"]:
            if result["state"] not in ["done", "failed"]:
                all_done = False
                if result["state"] == "running":
                    progress = result.get("extract_progress", {})
                    pages = f"{progress.get('extracted_pages', '?')}/{progress.get('total_pages', '?')}"
                    print(f"  Processing: {result['file_name']} ({pages} pages)")
        
        if all_done:
            break
        
        time.sleep(10)

    # Download and extract results
    print("\nDownloading results...")
    converted_files = []
    papers_info = []
    
    for result, pdf_file in zip(batch_data["extract_result"], pdf_files):
        if result["state"] != "done" or "full_zip_url" not in result:
            print(f"  ✗ {pdf_file.name} - conversion failed")
            continue
        
        # Download zip
        zip_url = result["full_zip_url"]
        zip_path = temp_folder / f"{pdf_file.stem}.zip"
        extract_folder = temp_folder / pdf_file.stem
        extract_folder.mkdir(exist_ok=True)
        
        zip_response = requests.get(zip_url, stream=True)
        if zip_response.status_code != 200:
            print(f"  ✗ {pdf_file.name} - download failed")
            continue
        
        with open(zip_path, 'wb') as f:
            for chunk in zip_response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        
        # Find markdown file
        markdown_files = list(extract_folder.glob('**/*.md'))
        if not markdown_files:
            print(f"  ✗ {pdf_file.name} - no markdown found")
            zip_path.unlink()
            continue
        
        # Create output folder with original PDF name
        safe_name = sanitize_filename(pdf_file.stem)
        output_folder = folder / safe_name
        
        # Handle duplicates
        counter = 1
        while output_folder.exists():
            output_folder = folder / f"{safe_name}_{counter}"
            counter += 1
        
        output_folder.mkdir(exist_ok=True)
        
        # Copy markdown and resources
        md_source = markdown_files[0]
        md_target = output_folder / f"{output_folder.name}.md"
        shutil.copy2(md_source, md_target)
        
        for item in md_source.parent.glob('*'):
            if item.is_file() and item != md_source:
                shutil.copy2(item, output_folder / item.name)
        
        converted_files.append(str(md_target))
        papers_info.append({
            "title": pdf_file.stem,
            "markdown_path": str(md_target),
            "original_pdf_path": str(pdf_file)
        })
        
        print(f"  ✓ {pdf_file.name} → {md_target.name}")
        
        # Cleanup
        zip_path.unlink()
    
    # Save papers info
    if papers_info:
        json_path = folder / f"{folder.name}_papers.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(papers_info, f, ensure_ascii=False, indent=2)
        print(f"\nPaper info saved to: {json_path}")
    
    # Cleanup temp folder
    shutil.rmtree(temp_folder, ignore_errors=True)
    
    print(f"\n✓ Conversion complete! {len(converted_files)}/{len(pdf_files)} files converted")
    return converted_files


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PDFs to Markdown using Mineru API')
    parser.add_argument('folder_path', nargs='?', help='Path to folder containing PDF files')
    args = parser.parse_args()
    
    if args.folder_path:
        folder_path = args.folder_path
    else:
        # Default path or interactive input
        folder_path = input("Please enter the folder path containing PDFs: ").strip()
        if not folder_path:
            print("No folder path provided. Exiting.")
            sys.exit(1)
            
    convert_pdfs_to_markdown(folder_path)