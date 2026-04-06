#!/usr/bin/env python
"""
Download all files from a Dataverse dataset.

This script queries dataset metadata by DOI, downloads each file through the
Dataverse access API, and stores results in organized local subdirectories.

Defaults are configured for one dataset DOI, but DATASET_DOI and DOWNLOAD_DIR
can be updated for other Dataverse sources.
"""

import requests
import os
import json
from pathlib import Path
from tqdm import tqdm

# Configuration
DATASET_DOI = "doi:10.7910/DVN/VZD5S6"
DATAVERSE_BASE_URL = "https://dataverse.harvard.edu"
DOWNLOAD_DIR = "raw"  # Base directory for downloads

# Optional: Add your API token here if you need to access restricted files
# You can get an API token by logging into Dataverse and going to your account settings
API_TOKEN = None  # Set to your token string if needed: "YOUR-API-TOKEN-HERE"


def get_dataset_metadata(doi):
    """Get dataset metadata from Dataverse API"""
    url = f"{DATAVERSE_BASE_URL}/api/datasets/:persistentId/?persistentId={doi}"

    headers = {}
    if API_TOKEN:
        headers['X-Dataverse-key'] = API_TOKEN

    print(f"Fetching dataset metadata for {doi}...")
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to get dataset metadata: {response.status_code} - {response.text}")

    return response.json()


def download_file(file_id, filename, download_path, api_token=None):
    """Download a single file from Dataverse"""
    url = f"{DATAVERSE_BASE_URL}/api/access/datafile/{file_id}"

    headers = {}
    if api_token:
        headers['X-Dataverse-key'] = api_token

    print(f"Downloading: {filename}")

    # Stream the download to handle large files
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code != 200:
        print(f"  Failed to download {filename}: {response.status_code}")
        return False

    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(download_path), exist_ok=True)

    # Download with progress bar
    with open(download_path, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)

    print(f"  ✓ Saved to: {download_path}")
    return True


def organize_by_type(filename):
    """Determine the subdirectory based on file type"""
    filename_lower = filename.lower()

    if filename_lower.endswith('.xtf'):
        return 'sss'  # Side-scan sonar
    elif filename_lower.endswith('.bag'):
        return 'bags'  # ROS bags
    elif filename_lower.endswith(('.json', '.csv', '.txt')):
        return 'metadata'
    elif filename_lower.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
        return 'images'
    else:
        return 'other'


def main():
    print("=" * 80)
    print("Harvard Dataverse Dataset Downloader")
    print("=" * 80)
    print(f"Dataset DOI: {DATASET_DOI}")
    print(f"Download directory: {DOWNLOAD_DIR}")
    print()

    try:
        # Get dataset metadata
        metadata = get_dataset_metadata(DATASET_DOI)

        # Extract file list
        files = metadata['data']['latestVersion']['files']

        print(f"Found {len(files)} files in dataset")
        print()

        # Save metadata
        metadata_path = os.path.join(DOWNLOAD_DIR, 'dataset_metadata.json')
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved dataset metadata to: {metadata_path}")
        print()

        # Display file information
        print("Files to download:")
        print("-" * 80)
        total_size = 0
        for i, file_info in enumerate(files, 1):
            datafile = file_info.get('dataFile', {})
            filename = datafile.get('filename', 'unknown')
            filesize = datafile.get('filesize', 0)
            file_type = datafile.get('contentType', 'unknown')

            # Convert bytes to human readable
            size_mb = filesize / (1024 * 1024)
            total_size += filesize

            print(f"{i:3d}. {filename:50s} ({size_mb:8.2f} MB) - {file_type}")

        print("-" * 80)
        print(f"Total size: {total_size / (1024 * 1024 * 1024):.2f} GB")
        print()

        # Ask for confirmation
        response = input("Do you want to proceed with the download? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return

        print()
        print("Starting downloads...")
        print()

        # Download each file
        success_count = 0
        failed_files = []

        for i, file_info in enumerate(files, 1):
            datafile = file_info.get('dataFile', {})
            file_id = datafile.get('id')
            filename = datafile.get('filename', f'file_{file_id}')

            # Organize files by type
            subdir = organize_by_type(filename)
            download_path = os.path.join(DOWNLOAD_DIR, subdir, filename)

            print(f"\n[{i}/{len(files)}]")

            # Skip if file already exists
            if os.path.exists(download_path):
                file_size = os.path.getsize(download_path)
                expected_size = datafile.get('filesize', 0)
                if file_size == expected_size:
                    print(f"  ⏭ Skipping (already exists): {filename}")
                    success_count += 1
                    continue

            # Download the file
            if download_file(file_id, filename, download_path, API_TOKEN):
                success_count += 1
            else:
                failed_files.append(filename)

        # Summary
        print()
        print("=" * 80)
        print("Download Summary")
        print("=" * 80)
        print(f"Successfully downloaded: {success_count}/{len(files)} files")

        if failed_files:
            print(f"Failed downloads: {len(failed_files)}")
            for filename in failed_files:
                print(f"  - {filename}")
        else:
            print("All files downloaded successfully! ✓")

        print()
        print(f"Files organized in: {os.path.abspath(DOWNLOAD_DIR)}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
