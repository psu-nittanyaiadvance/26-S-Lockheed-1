#!/usr/bin/env python3
"""
Incremental download, ranking, and pruning pipeline.

This script processes remote dataset files in streaming fashion to keep local
disk usage bounded while preserving only the highest-value images.

Workflow:
    1. Fetch dataset metadata from a Dataverse source.
    2. For each downloadable file/archive:
         a. Download it
         b. Extract images when needed
         c. Score images with CLIP + lightweight CV heuristics
         d. Update a global top-K heap
         e. Prune files that are not in the retained top-K
    3. Finalize ranked outputs with neighbor-context metadata.

The default DOI is configurable via constants/CLI options.
"""

import argparse
import csv
import heapq
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
import tarfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set, Generator
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import requests
from PIL import Image, ImageFile
import torch
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from transformers import CLIPProcessor, CLIPModel
from scipy.ndimage import convolve

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATASET_DOI = "doi:10.7910/DVN/VZD5S6"
DATAVERSE_BASE_URL = "https://dataverse.harvard.edu"

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_K = 200
DEFAULT_BATCH_SIZE = 16
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".7z"}

# For split 7z archives (camera.7z.001, camera.7z.002, etc.)
SPLIT_7Z_PATTERN = r"\.7z\.\d{3}$"

# Positive prompts - looking for OBJECTS on seafloor (not empty sand)
POSITIVE_PROMPTS = [
    # Shipwrecks and vessels
    "sonar image of a shipwreck on the ocean floor",
    "underwater photo of a sunken ship or boat wreckage",
    "side-scan sonar showing a vessel or ship debris on seabed",

    # Man-made objects and structures
    "sonar image showing man-made objects on the seafloor",
    "underwater image of debris, wreckage, or artifacts on ocean bottom",
    "acoustic image of underwater infrastructure, pipes, or cables",
    "sonar scan of an anchor, chain, or maritime equipment",

    # Vehicles and aircraft
    "sonar image of a submerged vehicle or aircraft wreckage",
    "underwater photo of a car, plane, or machinery on seabed",

    # General objects of interest
    "side-scan sonar with distinct object casting shadow on seafloor",
    "underwater image showing anomaly or unusual object on ocean floor",
    "sonar image with clear man-made structure or debris field",
]

# Negative prompts - penalize empty/boring images
NEGATIVE_PROMPTS = [
    # Empty seafloor
    "empty sandy seafloor with no objects or features",
    "flat featureless ocean bottom with just sand and sediment",
    "uniform seabed texture with nothing interesting",
    "boring empty underwater scene with only sand",

    # Technical issues
    "completely black empty image with no content",
    "pure white overexposed blank image",
    "random noise static corruption artifacts",
    "extremely blurry out of focus underwater image",
    "corrupted unreadable sonar data",
]

# Weights tuned for OBJECT DETECTION (not just image quality)
# CLIP scores matter most - they detect if there's something interesting
DEFAULT_WEIGHTS = {
    "clip_positive": 0.50,    # HIGH - "does it have objects?"
    "clip_negative": 0.20,    # "is it NOT empty sand?"
    "entropy": 0.05,          # LOW - technical quality less important
    "laplacian_var": 0.05,    # LOW - slightly blurry shipwreck still valuable
    "saturation_penalty": 0.10,
    "edge_density": 0.10,     # edges might indicate object boundaries
}


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class ImageRecord:
    """Record for a scored image."""
    path: str                          # Current absolute path
    original_archive: str              # Source archive name
    relative_path: str                 # Path within archive
    score: float
    components: Dict[str, float]
    global_index: int                  # Global ordering index


@dataclass
class TopKEntry:
    """Entry in the top-K heap (min-heap by score)."""
    score: float
    global_index: int
    path: str
    original_archive: str
    relative_path: str
    components: Dict[str, float]

    def __lt__(self, other):
        return self.score < other.score


@dataclass
class FinalRecord:
    """Final output record with neighbor context."""
    rank: int
    path: str
    relative_path: str
    original_archive: str
    score: float
    score_components: Dict[str, float]
    prev_5_paths: List[str]
    next_5_paths: List[str]


# -----------------------------------------------------------------------------
# Dataverse API Functions
# -----------------------------------------------------------------------------

def get_dataset_metadata(doi: str, api_token: Optional[str] = None) -> Dict:
    """Fetch dataset metadata from Dataverse API."""
    url = f"{DATAVERSE_BASE_URL}/api/datasets/:persistentId/?persistentId={doi}"
    headers = {"X-Dataverse-key": api_token} if api_token else {}

    logging.info(f"Fetching dataset metadata for {doi}...")
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to get metadata: {response.status_code} - {response.text}")

    return response.json()


def download_file(
    file_id: int,
    filename: str,
    download_path: Path,
    api_token: Optional[str] = None
) -> bool:
    """Download a single file from Dataverse with progress bar."""
    url = f"{DATAVERSE_BASE_URL}/api/access/datafile/{file_id}"
    headers = {"X-Dataverse-key": api_token} if api_token else {}

    logging.info(f"Downloading: {filename}")

    response = requests.get(url, headers=headers, stream=True)
    if response.status_code != 200:
        logging.error(f"Failed to download {filename}: {response.status_code}")
        return False

    total_size = int(response.headers.get('content-length', 0))
    download_path.parent.mkdir(parents=True, exist_ok=True)

    with open(download_path, 'wb') as f, tqdm(
        desc=f"  {filename}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)

    return True


# -----------------------------------------------------------------------------
# File Organization (from download_dataset.py)
# -----------------------------------------------------------------------------

def organize_by_type(filename: str) -> str:
    """Determine the subdirectory based on file type."""
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


def list_dataset_files(doi: str, api_token: Optional[str] = None) -> List[Dict]:
    """
    List all files in the dataset with their metadata.
    Returns list of dicts with id, filename, size, content_type, subdir.
    """
    metadata = get_dataset_metadata(doi, api_token)
    files = metadata['data']['latestVersion']['files']

    result = []
    for file_info in files:
        datafile = file_info.get('dataFile', {})
        filename = datafile.get('filename', '')
        result.append({
            'id': datafile.get('id'),
            'filename': filename,
            'size': datafile.get('filesize', 0),
            'content_type': datafile.get('contentType', 'unknown'),
            'subdir': organize_by_type(filename),
        })

    return result


def print_dataset_summary(files: List[Dict]) -> None:
    """Print summary of files in dataset."""
    print("\n" + "=" * 80)
    print("DATASET FILES SUMMARY")
    print("=" * 80)

    by_type = {}
    total_size = 0
    for f in files:
        subdir = f['subdir']
        by_type.setdefault(subdir, []).append(f)
        total_size += f['size']

    for subdir, subfiles in sorted(by_type.items()):
        subdir_size = sum(f['size'] for f in subfiles)
        print(f"\n{subdir.upper()} ({len(subfiles)} files, {subdir_size / (1024**2):.1f} MB):")
        for f in subfiles[:5]:  # Show first 5
            print(f"  - {f['filename']} ({f['size'] / (1024**2):.1f} MB)")
        if len(subfiles) > 5:
            print(f"  ... and {len(subfiles) - 5} more")

    print(f"\nTotal: {len(files)} files, {total_size / (1024**3):.2f} GB")
    print("=" * 80 + "\n")


# -----------------------------------------------------------------------------
# Archive Extraction
# -----------------------------------------------------------------------------

def extract_archive(archive_path: Path, extract_dir: Path) -> List[Path]:
    """
    Extract archive and return list of extracted image paths.
    Supports zip, tar, tar.gz, tar.bz2, 7z.
    """
    extracted_images = []
    archive_str = str(archive_path).lower()

    try:
        if archive_str.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for member in zf.namelist():
                    ext = os.path.splitext(member)[1].lower()
                    if ext in IMAGE_EXTENSIONS and not member.startswith('__MACOSX'):
                        zf.extract(member, extract_dir)
                        extracted_images.append(extract_dir / member)

        elif archive_str.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2')):
            mode = 'r:gz' if '.gz' in archive_str else 'r:bz2' if '.bz2' in archive_str else 'r'
            with tarfile.open(archive_path, mode) as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        ext = os.path.splitext(member.name)[1].lower()
                        if ext in IMAGE_EXTENSIONS:
                            tf.extract(member, extract_dir)
                            extracted_images.append(extract_dir / member.name)

        elif archive_str.endswith('.7z') or re.search(SPLIT_7Z_PATTERN, archive_str):
            # Use 7z command-line tool
            extracted_images = extract_7z_archive(archive_path, extract_dir)

        else:
            logging.warning(f"Unknown archive format: {archive_path}")

    except Exception as e:
        logging.error(f"Failed to extract {archive_path}: {e}")

    return extracted_images


def extract_7z_archive(archive_path: Path, extract_dir: Path) -> List[Path]:
    """
    Extract 7z archive using py7zr.
    Works with split archives (.7z.001) - just point to the first part.
    """
    extracted_images = []

    try:
        import py7zr
        print(f"    Using py7zr to extract {archive_path.name}...")

        # For split archives, py7zr needs the .001 file
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            print(f"    Archive contains {len(z.getnames())} files")
            z.extractall(path=extract_dir)
            print(f"    Extraction complete")

    except Exception as e:
        logging.error(f"7z extraction failed: {e}")

        # Try command-line 7z as fallback (if installed)
        try:
            print(f"    Trying 7z command-line as fallback...")
            cmd = ['7z', 'x', str(archive_path), f'-o{extract_dir}', '-y']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"7z command failed: {result.stderr}")
                return []
        except FileNotFoundError:
            print(f"    ERROR: Neither py7zr nor 7z command-line tool could extract the archive")
            print(f"    Install 7-Zip from https://www.7-zip.org/ and add to PATH")
            return []

    # Find all extracted images
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                extracted_images.append(Path(root) / f)

    print(f"    Found {len(extracted_images)} images")
    return extracted_images


def is_image_file(path: Path) -> bool:
    """Check if path is a supported image file."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_archive_file(path: Path) -> bool:
    """Check if path is a supported archive file."""
    name = path.name.lower()
    # Check standard extensions
    if any(name.endswith(ext) for ext in ARCHIVE_EXTENSIONS):
        return True
    # Check split 7z archives (e.g., .7z.001)
    if re.search(SPLIT_7Z_PATTERN, name):
        return True
    return False


def is_split_7z_part(filename: str) -> bool:
    """Check if file is part of a split 7z archive."""
    return bool(re.search(SPLIT_7Z_PATTERN, filename.lower()))


def get_split_7z_base(filename: str) -> str:
    """Get base name of split 7z archive (e.g., camera.7z from camera.7z.001)."""
    return re.sub(r"\.\d{3}$", "", filename)


# -----------------------------------------------------------------------------
# CV Heuristics
# -----------------------------------------------------------------------------

def compute_grayscale_entropy(img_array: np.ndarray) -> float:
    """Compute Shannon entropy of grayscale histogram, normalized to [0, 1]."""
    if img_array.ndim == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = img_array

    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist))
    return min(entropy / 8.0, 1.0)


def compute_laplacian_variance(img_array: np.ndarray) -> float:
    """Compute Laplacian variance (sharpness), normalized to [0, 1]."""
    if img_array.ndim == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    else:
        gray = img_array.astype(np.float32)

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    laplacian = convolve(gray, kernel, mode='reflect')
    variance = laplacian.var()
    normalized = 2.0 / (1.0 + np.exp(-variance / 500.0)) - 1.0
    return float(np.clip(normalized, 0, 1))


def compute_saturation_penalty(img_array: np.ndarray) -> float:
    """Compute penalty for saturated pixels. Returns 1.0 if no saturation."""
    if img_array.ndim == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array.astype(np.float64)

    total_pixels = gray.size
    near_black = np.sum(gray < 5)
    near_white = np.sum(gray > 250)
    saturated_ratio = (near_black + near_white) / total_pixels
    return float(np.exp(-5.0 * saturated_ratio))


def compute_edge_density(img_array: np.ndarray) -> float:
    """Compute edge density, normalized to [0, 1]."""
    if img_array.ndim == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = img_array.astype(np.uint8)

    if HAS_OPENCV:
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
    else:
        gx = np.diff(gray.astype(np.float32), axis=1)
        gy = np.diff(gray.astype(np.float32), axis=0)
        gx, gy = gx[:-1, :], gy[:, :-1]
        magnitude = np.sqrt(gx**2 + gy**2)
        edge_ratio = np.sum(magnitude > 30) / magnitude.size

    return float(min(edge_ratio * 10, 1.0))


def compute_cv_heuristics(img_array: np.ndarray) -> Dict[str, float]:
    """Compute all CV heuristic scores."""
    try:
        return {
            "entropy": compute_grayscale_entropy(img_array),
            "laplacian_var": compute_laplacian_variance(img_array),
            "saturation_penalty": compute_saturation_penalty(img_array),
            "edge_density": compute_edge_density(img_array),
        }
    except Exception as e:
        logging.debug(f"CV heuristics failed: {e}")
        return {"entropy": 0.0, "laplacian_var": 0.0, "saturation_penalty": 0.0, "edge_density": 0.0}


# -----------------------------------------------------------------------------
# CLIP Scorer
# -----------------------------------------------------------------------------

class CLIPScorer:
    """CLIP-based image scoring with precomputed text embeddings."""

    def __init__(self, model_name: str, device: str):
        self.device = device
        logging.info(f"Loading CLIP model: {model_name} on {device}")

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            # Positive embeddings
            pos_inputs = self.processor(text=POSITIVE_PROMPTS, return_tensors="pt", padding=True, truncation=True)
            pos_inputs = {k: v.to(device) for k, v in pos_inputs.items() if k != "pixel_values"}
            pos_output = self.model.get_text_features(**pos_inputs)
            # Handle both tensor and BaseModelOutputWithPooling returns (transformers version compatibility)
            self.pos_embeds = pos_output if isinstance(pos_output, torch.Tensor) else pos_output.pooler_output
            self.pos_embeds = self.pos_embeds / self.pos_embeds.norm(dim=-1, keepdim=True)

            # Negative embeddings
            neg_inputs = self.processor(text=NEGATIVE_PROMPTS, return_tensors="pt", padding=True, truncation=True)
            neg_inputs = {k: v.to(device) for k, v in neg_inputs.items() if k != "pixel_values"}
            neg_output = self.model.get_text_features(**neg_inputs)
            self.neg_embeds = neg_output if isinstance(neg_output, torch.Tensor) else neg_output.pooler_output
            self.neg_embeds = self.neg_embeds / self.neg_embeds.norm(dim=-1, keepdim=True)

    def score_batch(self, images: List[Image.Image]) -> List[Dict[str, float]]:
        """Score a batch of PIL images."""
        if not images:
            return []

        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            img_output = self.model.get_image_features(**inputs)
            img_embeds = img_output if isinstance(img_output, torch.Tensor) else img_output.pooler_output
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

            pos_sim = torch.matmul(img_embeds, self.pos_embeds.T).mean(dim=-1)
            neg_sim = torch.matmul(img_embeds, self.neg_embeds.T).mean(dim=-1)

        pos_scores = ((pos_sim.cpu().numpy() + 1) / 2).clip(0, 1)
        neg_scores = 1.0 - ((neg_sim.cpu().numpy() + 1) / 2).clip(0, 1)

        return [{"clip_positive": float(p), "clip_negative": float(n)} for p, n in zip(pos_scores, neg_scores)]


# -----------------------------------------------------------------------------
# Image Loading
# -----------------------------------------------------------------------------

def load_image_safe(path: Path, max_size: int = 1024) -> Optional[Tuple[Image.Image, np.ndarray]]:
    """Safely load and resize an image."""
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        arr = np.array(img)
        if arr.size == 0 or arr.ndim < 2:
            return None

        return img, arr
    except Exception as e:
        logging.debug(f"Failed to load {path}: {e}")
        return None


# -----------------------------------------------------------------------------
# Scoring Functions
# -----------------------------------------------------------------------------

def compute_final_score(
    clip_scores: Dict[str, float],
    cv_scores: Dict[str, float],
    weights: Dict[str, float]
) -> Tuple[float, Dict[str, float]]:
    """Combine CLIP and CV scores into final score in [0, 1]."""
    components = {**clip_scores, **cv_scores}
    total_weight = sum(weights.values())
    final_score = sum(weights.get(k, 0) * v for k, v in components.items())
    if total_weight > 0:
        final_score /= total_weight
    return float(np.clip(final_score, 0, 1)), components


# -----------------------------------------------------------------------------
# Incremental Pipeline
# -----------------------------------------------------------------------------

class IncrementalRanker:
    """
    Downloads, scores, and prunes images incrementally.

    Maintains a global top-K heap across all downloaded archives.
    After processing each archive, deletes images not in top-K.
    """

    def __init__(
        self,
        output_dir: Path,
        k: int = DEFAULT_K,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        weights: Dict[str, float] = None,
        api_token: Optional[str] = None,
        keep_archives: bool = False,
    ):
        self.output_dir = output_dir
        self.k = k
        self.batch_size = batch_size
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.api_token = api_token
        self.keep_archives = keep_archives

        # Directories
        self.images_dir = output_dir / "images"
        self.temp_dir = output_dir / "temp"
        self.results_dir = output_dir / "results"

        for d in [self.images_dir, self.temp_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize CLIP scorer
        self.clip_scorer = CLIPScorer(model_name, device)

        # Global state
        self.top_k_heap: List[TopKEntry] = []  # min-heap
        self.global_index = 0  # Global ordering counter
        self.scan_order: List[Tuple[int, str]] = []  # (global_index, relative_path) for neighbor lookup

        # Stats
        self.total_images = 0
        self.processed_images = 0
        self.failed_images = 0
        self.deleted_images = 0

    def process_dataset(self, doi: str = DATASET_DOI) -> List[FinalRecord]:
        """Main entry point: process entire dataset incrementally."""

        # Get all files in dataset
        all_files = list_dataset_files(doi, self.api_token)
        print_dataset_summary(all_files)

        # Group split 7z archives together
        split_archives = {}  # base_name -> list of parts
        regular_files = []

        for f in all_files:
            filename = f['filename']

            if is_split_7z_part(filename):
                base = get_split_7z_base(filename)
                if base not in split_archives:
                    split_archives[base] = []
                split_archives[base].append({
                    'id': f['id'],
                    'filename': filename,
                    'size': f['size'],
                })
            else:
                ext = os.path.splitext(filename)[1].lower()
                if ext in IMAGE_EXTENSIONS or is_archive_file(Path(filename)):
                    regular_files.append({
                        'id': f['id'],
                        'filename': filename,
                        'size': f['size'],
                    })

        # Build list of items to process (split archives as groups, regular files individually)
        relevant_files = []

        # Add split archive groups
        for base_name, parts in sorted(split_archives.items()):
            # Sort parts by filename to ensure correct order
            parts = sorted(parts, key=lambda x: x['filename'])
            total_size = sum(p['size'] for p in parts)
            relevant_files.append({
                'type': 'split_7z',
                'base_name': base_name,
                'parts': parts,
                'size': total_size,
            })

        # Add regular files
        for f in regular_files:
            relevant_files.append({
                'type': 'single',
                'id': f['id'],
                'filename': f['filename'],
                'size': f['size'],
            })

        logging.info(f"Found {len(relevant_files)} relevant items to process ({len(split_archives)} split archives, {len(regular_files)} regular files)")

        # Process each item ONE AT A TIME
        total_items = len(relevant_files)
        print(f"\n{'#'*60}")
        print(f"# STARTING INCREMENTAL PROCESSING: {total_items} items")
        print(f"# Each item will be: Downloaded -> Extracted -> Scored -> Pruned")
        print(f"# Only top-{self.k} images will be kept on disk at any time")
        print(f"{'#'*60}\n")

        for i, item in enumerate(relevant_files, 1):
            item_size_mb = item['size'] / (1024 * 1024)

            if item['type'] == 'split_7z':
                # Split 7z archive - need to download all parts first
                print(f"\n{'='*60}")
                print(f"BATCH {i}/{total_items}: {item['base_name']} (split archive)")
                print(f"Parts: {len(item['parts'])}, Total size: {item_size_mb:.1f} MB")
                print(f"{'='*60}")

                self._process_split_7z(item)

            else:
                # Single file
                print(f"\n{'='*60}")
                print(f"BATCH {i}/{total_items}: {item['filename']}")
                print(f"Size: {item_size_mb:.1f} MB")
                print(f"{'='*60}")

                self._process_single_file(item)

            # Status update after each batch
            print(f"\n--- Batch {i} Complete ---")
            print(f"  Images in top-K heap: {len(self.top_k_heap)}/{self.k}")
            if self.top_k_heap:
                min_score = self.top_k_heap[0].score
                max_score = max(e.score for e in self.top_k_heap)
                print(f"  Score range: {min_score:.4f} - {max_score:.4f}")
            print(f"  Total processed so far: {self.processed_images}")
            print(f"  Total deleted so far: {self.deleted_images}")

        # Finalize results with neighbor context
        logging.info("\nFinalizing results with neighbor context...")
        results = self._finalize_results()

        # Write outputs
        self._write_outputs(results)

        return results

    def _process_split_7z(self, item: Dict) -> None:
        """Download all parts of a split 7z archive, then extract and process."""

        base_name = item['base_name']
        parts = item['parts']

        print(f"\n  Downloading {len(parts)} parts from Harvard Dataverse...")

        # Download all parts
        downloaded_parts = []
        for j, part in enumerate(parts, 1):
            part_path = self.temp_dir / part['filename']
            print(f"    Part {j}/{len(parts)}: {part['filename']}")

            if not download_file(part['id'], part['filename'], part_path, self.api_token):
                print(f"  ERROR: Failed to download {part['filename']}")
                # Cleanup already downloaded parts
                for p in downloaded_parts:
                    if p.exists():
                        p.unlink()
                return

            downloaded_parts.append(part_path)

        # Extract from the first part (7z will automatically find the rest)
        first_part = downloaded_parts[0]
        self._process_archive(first_part, base_name)

        # Cleanup all downloaded parts
        if not self.keep_archives:
            for p in downloaded_parts:
                if p.exists():
                    p.unlink()
            print(f"  Cleaned up {len(downloaded_parts)} temporary files")

    def _process_single_file(self, file_info: Dict) -> None:
        """Download and process a single file (archive or image)."""

        file_id = file_info['id']
        filename = file_info['filename']
        download_path = self.temp_dir / filename

        # Download from Harvard Dataverse
        print(f"\n  Downloading from Harvard Dataverse...")
        if not download_file(file_id, filename, download_path, self.api_token):
            print(f"  ERROR: Download failed!")
            return

        # Determine if archive or direct image and process accordingly
        if is_archive_file(download_path):
            self._process_archive(download_path, filename)
        elif is_image_file(download_path):
            self._process_direct_image(download_path, filename)

        # Cleanup downloaded file to save disk space
        if download_path.exists() and not self.keep_archives:
            download_path.unlink()
            print(f"  Cleaned up temporary download")

    def _process_archive(self, archive_path: Path, archive_name: str) -> None:
        """Extract, score, and prune images from an archive."""

        extract_dir = self.temp_dir / f"extract_{archive_path.stem}"
        extract_dir.mkdir(exist_ok=True)

        # Step 1: Extract images
        print(f"\n  [Step 1/4] Extracting images from archive...")
        image_paths = extract_archive(archive_path, extract_dir)

        if not image_paths:
            print(f"  WARNING: No images found in {archive_name}")
            shutil.rmtree(extract_dir, ignore_errors=True)
            return

        print(f"  Found {len(image_paths)} images in archive")
        self.total_images += len(image_paths)

        # Sort for deterministic ordering
        image_paths = sorted(image_paths, key=lambda p: str(p))

        # Step 2: Score all images with CLIP + CV heuristics
        print(f"  [Step 2/4] Scoring {len(image_paths)} images...")
        scored_images = self._score_images(image_paths, archive_name)

        # Step 3: Update global top-K heap
        print(f"  [Step 3/4] Updating global top-{self.k} heap...")
        images_to_keep = self._update_heap_and_select(scored_images)

        # Step 4: Move kept images to permanent storage, delete rest
        print(f"  [Step 4/4] Pruning - keeping {len(images_to_keep)} images, deleting rest...")
        kept_count = 0
        deleted_count = 0

        for img_path in image_paths:
            rel_path = img_path.relative_to(extract_dir)

            if str(img_path) in images_to_keep:
                # Move to permanent storage
                dest = self.images_dir / archive_name.replace('.', '_') / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), str(dest))

                # Update path in heap entry
                for entry in self.top_k_heap:
                    if entry.path == str(img_path):
                        entry.path = str(dest)

                kept_count += 1
            else:
                # Delete
                if img_path.exists():
                    img_path.unlink()
                deleted_count += 1

        self.deleted_images += deleted_count

        # Cleanup extract directory
        shutil.rmtree(extract_dir, ignore_errors=True)

        print(f"  Result: Kept {kept_count}, Deleted {deleted_count}")

    def _process_direct_image(self, image_path: Path, filename: str) -> None:
        """Process a directly downloaded image (not in archive)."""

        self.total_images += 1

        result = load_image_safe(image_path)
        if result is None:
            self.failed_images += 1
            return

        pil_img, np_arr = result

        # Score
        try:
            clip_scores = self.clip_scorer.score_batch([pil_img])[0]
        except Exception:
            clip_scores = {"clip_positive": 0.0, "clip_negative": 0.5}

        cv_scores = compute_cv_heuristics(np_arr)
        final_score, components = compute_final_score(clip_scores, cv_scores, self.weights)

        self.processed_images += 1

        # Record in scan order
        rel_path = filename
        self.scan_order.append((self.global_index, rel_path))

        # Update heap
        entry = TopKEntry(
            score=final_score,
            global_index=self.global_index,
            path=str(image_path),
            original_archive="direct",
            relative_path=rel_path,
            components=components,
        )

        self.global_index += 1

        if len(self.top_k_heap) < self.k:
            heapq.heappush(self.top_k_heap, entry)
            # Move to permanent storage
            dest = self.images_dir / "direct" / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(dest))
            entry.path = str(dest)
        elif final_score > self.top_k_heap[0].score:
            # Remove old minimum
            old_entry = heapq.heapreplace(self.top_k_heap, entry)
            # Delete old image
            if Path(old_entry.path).exists():
                Path(old_entry.path).unlink()
                self.deleted_images += 1
            # Move new image to permanent storage
            dest = self.images_dir / "direct" / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(dest))
            entry.path = str(dest)
        else:
            # Not good enough, delete
            if image_path.exists():
                image_path.unlink()
            self.deleted_images += 1

    def _score_images(
        self,
        image_paths: List[Path],
        archive_name: str
    ) -> List[Tuple[Path, float, Dict[str, float], int, str]]:
        """Score all images in a list, return (path, score, components, global_idx, rel_path)."""

        results = []
        batch_data = []

        pbar = tqdm(image_paths, desc="    Scoring", unit="img", leave=False)

        for img_path in pbar:
            result = load_image_safe(img_path)

            if result is None:
                self.failed_images += 1
                continue

            pil_img, np_arr = result
            rel_path = f"{archive_name}/{img_path.name}"

            batch_data.append((img_path, pil_img, np_arr, rel_path))

            if len(batch_data) >= self.batch_size:
                batch_results = self._process_scoring_batch(batch_data)
                results.extend(batch_results)
                batch_data = []

        # Process remaining
        if batch_data:
            batch_results = self._process_scoring_batch(batch_data)
            results.extend(batch_results)

        return results

    def _process_scoring_batch(
        self,
        batch_data: List[Tuple[Path, Image.Image, np.ndarray, str]]
    ) -> List[Tuple[Path, float, Dict[str, float], int, str]]:
        """Process a batch for scoring."""

        results = []
        pil_images = [item[1] for item in batch_data]

        try:
            clip_scores_list = self.clip_scorer.score_batch(pil_images)
        except Exception as e:
            logging.warning(f"CLIP batch failed: {e}")
            clip_scores_list = [{"clip_positive": 0.0, "clip_negative": 0.5}] * len(batch_data)

        for i, (img_path, pil_img, np_arr, rel_path) in enumerate(batch_data):
            cv_scores = compute_cv_heuristics(np_arr)
            clip_scores = clip_scores_list[i] if i < len(clip_scores_list) else {"clip_positive": 0.0, "clip_negative": 0.5}

            final_score, components = compute_final_score(clip_scores, cv_scores, self.weights)

            # Record in scan order
            self.scan_order.append((self.global_index, rel_path))

            results.append((img_path, final_score, components, self.global_index, rel_path))
            self.global_index += 1
            self.processed_images += 1

        return results

    def _update_heap_and_select(
        self,
        scored_images: List[Tuple[Path, float, Dict[str, float], int, str]]
    ) -> Set[str]:
        """Update global heap with new scores, return set of paths to keep."""

        images_to_keep = set()

        for img_path, score, components, global_idx, rel_path in scored_images:
            entry = TopKEntry(
                score=score,
                global_index=global_idx,
                path=str(img_path),
                original_archive=rel_path.split('/')[0] if '/' in rel_path else "unknown",
                relative_path=rel_path,
                components=components,
            )

            if len(self.top_k_heap) < self.k:
                heapq.heappush(self.top_k_heap, entry)
                images_to_keep.add(str(img_path))
            elif score > self.top_k_heap[0].score:
                old_entry = heapq.heapreplace(self.top_k_heap, entry)
                images_to_keep.add(str(img_path))
                images_to_keep.discard(old_entry.path)

                # Delete old image if it's in permanent storage
                old_path = Path(old_entry.path)
                if old_path.exists() and str(self.images_dir) in str(old_path):
                    old_path.unlink()
                    self.deleted_images += 1

        return images_to_keep

    def _finalize_results(self) -> List[FinalRecord]:
        """Build final results with neighbor context using scan order."""

        # Sort heap entries by score descending for ranking
        sorted_entries = sorted(self.top_k_heap, key=lambda e: e.score, reverse=True)

        # Build index -> position map for scan order
        idx_to_scan_pos = {idx: pos for pos, (idx, _) in enumerate(self.scan_order)}

        results = []

        for rank, entry in enumerate(sorted_entries, start=1):
            scan_pos = idx_to_scan_pos.get(entry.global_index, -1)

            # Get prev 5 and next 5 neighbors in scan order
            prev_5 = []
            next_5 = []

            if scan_pos >= 0:
                for i in range(max(0, scan_pos - 5), scan_pos):
                    prev_5.append(self.scan_order[i][1])
                for i in range(scan_pos + 1, min(len(self.scan_order), scan_pos + 6)):
                    next_5.append(self.scan_order[i][1])

            results.append(FinalRecord(
                rank=rank,
                path=entry.path,
                relative_path=entry.relative_path,
                original_archive=entry.original_archive,
                score=entry.score,
                score_components=entry.components,
                prev_5_paths=prev_5,
                next_5_paths=next_5,
            ))

        return results

    def _write_outputs(self, results: List[FinalRecord]) -> None:
        """Write all output files."""

        # JSONL
        jsonl_path = self.results_dir / "top_images.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                obj = {
                    "rank": r.rank,
                    "path": r.path,
                    "relative_path": r.relative_path,
                    "original_archive": r.original_archive,
                    "score": r.score,
                    "score_components": r.score_components,
                    "prev_5_paths": r.prev_5_paths,
                    "next_5_paths": r.next_5_paths,
                }
                f.write(json.dumps(obj) + "\n")
        logging.info(f"Wrote: {jsonl_path}")

        # Summary JSON
        summary_path = self.results_dir / "summary.json"
        summary = {
            "metadata": {
                "k": self.k,
                "total_images": self.total_images,
                "processed_images": self.processed_images,
                "failed_images": self.failed_images,
                "deleted_images": self.deleted_images,
                "final_top_k_count": len(results),
            },
            "ranked_images": [
                {
                    "rank": r.rank,
                    "path": r.path,
                    "relative_path": r.relative_path,
                    "original_archive": r.original_archive,
                    "score": r.score,
                    "score_components": r.score_components,
                    "prev_5_paths": r.prev_5_paths,
                    "next_5_paths": r.next_5_paths,
                }
                for r in results
            ]
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Wrote: {summary_path}")

        # CSV
        csv_path = self.results_dir / "top_images.csv"
        fieldnames = [
            "rank", "relative_path", "original_archive", "score",
            "clip_positive", "clip_negative", "entropy",
            "laplacian_var", "saturation_penalty", "edge_density",
            "prev_5_paths", "next_5_paths"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "rank": r.rank,
                    "relative_path": r.relative_path,
                    "original_archive": r.original_archive,
                    "score": f"{r.score:.6f}",
                    "clip_positive": f"{r.score_components.get('clip_positive', 0):.6f}",
                    "clip_negative": f"{r.score_components.get('clip_negative', 0):.6f}",
                    "entropy": f"{r.score_components.get('entropy', 0):.6f}",
                    "laplacian_var": f"{r.score_components.get('laplacian_var', 0):.6f}",
                    "saturation_penalty": f"{r.score_components.get('saturation_penalty', 0):.6f}",
                    "edge_density": f"{r.score_components.get('edge_density', 0):.6f}",
                    "prev_5_paths": ";".join(r.prev_5_paths),
                    "next_5_paths": ";".join(r.next_5_paths),
                })
        logging.info(f"Wrote: {csv_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Harvard Dataverse dataset, rank images, keep only top-K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--outdir", "-o",
        type=str,
        default="ranked_dataset",
        help="Output directory for results and kept images.",
    )
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=DEFAULT_K,
        help="Number of top images to keep.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face CLIP model identifier.",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device for inference.",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for CLIP inference.",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help="Dataverse API token (if needed for restricted files).",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archives after processing.",
    )
    parser.add_argument(
        "--doi",
        type=str,
        default=DATASET_DOI,
        help="Dataset DOI to download.",
    )

    # Scoring weights
    parser.add_argument("--weight-clip-positive", type=float, default=DEFAULT_WEIGHTS["clip_positive"])
    parser.add_argument("--weight-clip-negative", type=float, default=DEFAULT_WEIGHTS["clip_negative"])
    parser.add_argument("--weight-entropy", type=float, default=DEFAULT_WEIGHTS["entropy"])
    parser.add_argument("--weight-laplacian", type=float, default=DEFAULT_WEIGHTS["laplacian_var"])
    parser.add_argument("--weight-saturation", type=float, default=DEFAULT_WEIGHTS["saturation_penalty"])
    parser.add_argument("--weight-edge", type=float, default=DEFAULT_WEIGHTS["edge_density"])

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU")
        device = "cpu"

    logging.info(f"Using device: {device}")

    # Build weights
    weights = {
        "clip_positive": args.weight_clip_positive,
        "clip_negative": args.weight_clip_negative,
        "entropy": args.weight_entropy,
        "laplacian_var": args.weight_laplacian,
        "saturation_penalty": args.weight_saturation,
        "edge_density": args.weight_edge,
    }

    # Create ranker and run
    ranker = IncrementalRanker(
        output_dir=Path(args.outdir),
        k=args.k,
        batch_size=args.batch_size,
        model_name=args.model,
        device=device,
        weights=weights,
        api_token=args.api_token,
        keep_archives=args.keep_archives,
    )

    results = ranker.process_dataset(args.doi)

    # Summary
    logging.info("\n" + "="*60)
    logging.info("PIPELINE COMPLETE")
    logging.info("="*60)
    logging.info(f"Total images encountered: {ranker.total_images}")
    logging.info(f"Successfully processed: {ranker.processed_images}")
    logging.info(f"Failed/skipped: {ranker.failed_images}")
    logging.info(f"Deleted (not in top-K): {ranker.deleted_images}")
    logging.info(f"Final top-K count: {len(results)}")
    logging.info(f"Output directory: {args.outdir}")

    if results:
        logging.info("\nTop 5 images:")
        for r in results[:5]:
            logging.info(f"  #{r.rank}: {r.relative_path} (score: {r.score:.4f})")


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# Basic usage (auto-detects CUDA, keeps top 200):
#   python download_and_rank.py --outdir ranked_dataset --k 200
#
# With CUDA on RTX 3080, larger batch:
#   python download_and_rank.py --outdir ranked_dataset --device cuda --batch-size 32
#
# Keep more images, different model:
#   python download_and_rank.py --k 500 --model openai/clip-vit-large-patch14
#
# Keep downloaded archives for inspection:
#   python download_and_rank.py --outdir ranked_dataset --keep-archives
#
# Custom scoring weights:
#   python download_and_rank.py --weight-clip-positive 0.5 --weight-entropy 0.1
#
# Full verbose run:
#   python download_and_rank.py \
#     --outdir ranked_dataset \
#     --k 200 \
#     --device cuda \
#     --batch-size 32 \
#     --verbose
#
# =============================================================================
