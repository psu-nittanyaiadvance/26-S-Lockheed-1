#!/usr/bin/env python3
"""
preprocess_datasets.py - Underwater dataset preprocessing for Gaussian splatting.

This script provides a two-phase pipeline for preparing underwater image datasets
for use with 3D Gaussian splatting reconstruction. The goal is to filter out
low-quality or irrelevant images and remove camera housing artifacts so that
only clean, useful frames are fed into the reconstruction pipeline.

Phase 1 ("analyze"):
  - Uses OpenAI's CLIP model for zero-shot image classification to categorize
    each image (e.g., fish, coral, shipwreck, open water, blurry, etc.)
  - Detects static border artifacts (e.g., camera housing, ROV frame) by
    comparing border regions across many images to find unchanging pixels
  - Outputs per-image classification CSVs and a JSON summary

Phase 2 ("clean"):
  - Reads the analysis results from Phase 1
  - Removes images classified as "empty" (open water, murky, dark, blurry)
  - Crops detected border artifacts from the remaining useful images
  - Converts TIFF images to PNG for smaller file sizes
  - Exports the cleaned dataset preserving the original directory structure

Usage:
  python preprocess_datasets.py analyze [--outdir DIR] [--device auto] [--batch-size 16]
  python preprocess_datasets.py clean [--analysis-dir DIR] [--outdir cleaned_datasets]
"""

# --- Standard library imports ---
import argparse
import csv
import json
import logging
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

# Suppress noisy warnings from transformers/torch that clutter output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Third-party imports ---
import numpy as np
from PIL import Image, ImageFile
import torch
from tqdm import tqdm

# Allow loading of truncated images without raising errors.
# Some underwater datasets have partially written files from interrupted transfers.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# OpenCV is optional — only needed if future extensions add OpenCV-based processing.
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Hugging Face CLIP model for zero-shot image classification.
# CLIP compares images against text descriptions to classify without training.
from transformers import CLIPProcessor, CLIPModel

# =============================================================================
# Configuration
# =============================================================================

# CLIP model to use for zero-shot classification. The ViT-B/32 variant is a
# good balance of speed and accuracy — it uses a Vision Transformer with 32x32
# patch size, running reasonably fast even on CPU/MPS.
DEFAULT_MODEL = "openai/clip-vit-base-patch32"

# Number of images to classify at once. Larger batches are faster on GPU but
# use more memory. 16 works well on a MacBook with MPS or 8GB+ GPU.
DEFAULT_BATCH_SIZE = 16

# Output directories for each phase of the pipeline
DEFAULT_OUTDIR = "classification_output"       # Phase 1 (analyze) results
DEFAULT_CLEAN_OUTDIR = "cleaned_datasets"       # Phase 2 (clean) results

# --- Artifact detection parameters ---
# Number of images to randomly sample for detecting static border artifacts.
# 50 is enough to reliably distinguish static elements from varying scene content.
ARTIFACT_SAMPLE_SIZE = 50

# Maximum per-pixel standard deviation across sampled images for a border
# region to be considered "static" (i.e., an artifact like camera housing).
# Lower values = more aggressive detection; 25.0 is a conservative default.
ARTIFACT_VARIANCE_THRESHOLD = 25.0

# Fraction of image dimensions to scan from each edge when looking for artifacts.
# 0.20 means scan the outer 20% of each border (top, bottom, left, right).
ARTIFACT_SCAN_FRACTION = 0.20

# =============================================================================
# CLIP Classification Categories
# =============================================================================
# Each category has multiple text prompt templates. CLIP compares images against
# these prompts to determine which category best describes the image.
# Using multiple prompts per category ("prompt ensembling") improves accuracy
# because CLIP's text understanding varies with phrasing.

UNDERWATER_CATEGORIES = {
    # ----- Biological categories -----
    # Images containing living organisms — useful for 3D reconstruction
    "fish": [
        "a photo of fish swimming underwater",
        "a photo of a school of fish in the ocean",
    ],
    "coral": [
        "a photo of a coral reef underwater",
        "a photo of coral formations on the seafloor",
    ],
    "marine_animal": [
        "a photo of a marine animal underwater such as a turtle or ray or shark",
        "a photo of a sea creature like a jellyfish or octopus or crab",
    ],
    "vegetation": [
        "a photo of underwater vegetation or seaweed or algae",
        "a photo of marine plants growing on the seafloor",
    ],

    # ----- Terrain categories -----
    # Static seafloor environments — good for 3D reconstruction of terrain
    "sand_seafloor": [
        "a photo of a sandy seafloor with no objects",
        "a photo of flat sandy ocean bottom",
    ],
    "rocks": [
        "a photo of underwater rocks and boulders on the seafloor",
        "a photo of a rocky underwater terrain",
    ],

    # ----- Man-made object categories -----
    # Structures and debris — key targets for inspection/survey tasks
    "shipwreck_debris": [
        "a photo of an underwater shipwreck or debris",
        "a photo of sunken wreckage and man-made debris on the ocean floor",
    ],
    "man_made_structure": [
        "a photo of an underwater man-made structure like a pier or pipe or cable",
        "a photo of underwater infrastructure or equipment",
    ],

    # ----- Human presence -----
    "diver": [
        "a photo of a scuba diver underwater",
        "a photo of a human diver swimming underwater",
    ],

    # ----- "Empty" scene categories -----
    # These images are NOT useful for Gaussian splatting reconstruction because
    # they lack discernible structure, texture, or features for point matching.
    "open_water": [
        "a photo of open blue water with nothing visible",
        "a photo of empty underwater scene with only water",
    ],
    "murky_turbid": [
        "a photo of murky turbid underwater conditions with poor visibility",
        "a photo of cloudy green or brown underwater water with suspended particles",
    ],
    "dark_overexposed": [
        "a completely dark or black underwater image with nothing visible",
        "a completely white overexposed underwater image",
    ],
    "blurry_corrupt": [
        "an extremely blurry out of focus underwater image",
        "a corrupted or garbled underwater image with artifacts",
    ],
}

# Categories whose images should be REMOVED during the clean phase.
# These have no usable structure for 3D reconstruction.
EMPTY_CATEGORIES = {"open_water", "murky_turbid", "dark_overexposed", "blurry_corrupt"}

# Ordered list of all category names — used as CSV column headers and for indexing
CATEGORY_NAMES = list(UNDERWATER_CATEGORIES.keys())

# =============================================================================
# Dataset Configurations
# =============================================================================
# Each dataset has a unique directory structure. These configs tell the pipeline:
#   - name:         Short identifier used for output file naming
#   - root:         Absolute path to the dataset's top-level directory
#   - extensions:   Set of valid image file extensions to look for
#   - image_subdir: Name of the subdirectory containing images within each
#                   location folder (e.g., "imgs" or "camera"). The pipeline
#                   only processes images found inside directories with this name.
#   - description:  Human-readable label for logging and summary output

DATASET_CONFIGS = [
    {
        "name": "flsea_vi",
        "root": "/Users/ethanknox/Desktop/Advance Spring 26 (Lockheed Martin)/flsea-vi",
        "extensions": {".tiff", ".tif"},
        "image_subdir": "imgs",
        "description": "FLSEA-VI underwater scenes (canyons + Red Sea)",
    },
    {
        "name": "shipwreck",
        "root": "/Users/ethanknox/Desktop/Advance Spring 26 (Lockheed Martin)/Shipwreck",
        "extensions": {".jpg", ".jpeg"},
        "image_subdir": "camera",
        "description": "Shipwreck survey recordings",
    },
    {
        "name": "sunboat",
        "root": "/Users/ethanknox/Desktop/Advance Spring 26 (Lockheed Martin)/Sunboat_03-09-2023",
        "extensions": {".png"},
        "image_subdir": "camera",
        "description": "Sunboat mission recordings",
    },
]


# =============================================================================
# Chunk 1: Image Discovery & Loading
# =============================================================================
# Functions for finding images within dataset directory trees and loading them
# safely. Datasets are organized hierarchically:
#
#   dataset_root/
#     location_A/
#       imgs/          <-- image_subdir (e.g., "imgs" or "camera")
#         frame001.tiff
#         frame002.tiff
#     location_B/
#       imgs/
#         frame001.tiff
#
# The "location" is derived from the path above the image_subdir, which lets
# us track per-location statistics (e.g., how many empty images in each dive site).


def iter_dataset_images(
    root: str,
    extensions: Set[str],
    image_subdir: str,
) -> Generator[Tuple[str, str, str], None, None]:
    """
    Recursively walk the dataset root directory, yielding images that are
    inside subdirectories named `image_subdir` and match the given file extensions.

    This filtering ensures we only pick up actual image data (not thumbnails,
    metadata, or other files that might live elsewhere in the tree).

    Yields tuples of:
        - absolute_path:  Full filesystem path to the image
        - relative_path:  Path relative to dataset root (used to preserve structure)
        - location_name:  The parent path above the image_subdir, identifying which
                          dive site or recording session the image belongs to.
                          e.g. "canyons/flatiron" for "canyons/flatiron/imgs/123.tiff"

    Images are sorted by relative path for deterministic, reproducible ordering.
    """
    root = os.path.abspath(root)
    all_images = []

    # os.walk recursively traverses the directory tree
    for dirpath, dirnames, filenames in os.walk(root):
        # Only process directories that match the expected image subdirectory name
        if os.path.basename(dirpath) != image_subdir:
            continue
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, root)

                # Extract the "location" by stripping the image_subdir and filename
                # from the relative path. For example:
                #   rel_path = "canyons/flatiron/imgs/file.tiff"
                #   parts    = ["canyons", "flatiron", "imgs", "file.tiff"]
                #   location = "canyons/flatiron"  (everything before "imgs")
                parts = rel_path.split(os.sep)
                if len(parts) > 2:
                    location = os.sep.join(parts[:-2])
                else:
                    # Image is directly in root/image_subdir/ with no nesting
                    location = "root"
                all_images.append((abs_path, rel_path, location))

    # Sort alphabetically by relative path for deterministic ordering
    all_images.sort(key=lambda x: x[1])
    for item in all_images:
        yield item


def count_dataset_images(root: str, extensions: Set[str], image_subdir: str) -> int:
    """
    Count total images in a dataset without loading them.
    Used to initialize progress bars with accurate totals before processing.
    Uses the same directory-walking logic as iter_dataset_images.
    """
    count = 0
    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath) != image_subdir:
            continue
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in extensions:
                count += 1
    return count


def load_image_safe(
    path: str, max_size: int = 512
) -> Optional[Tuple[Image.Image, np.ndarray]]:
    """
    Load an image safely for CLIP classification, with error handling.

    Performs three operations:
      1. Converts to RGB (handles grayscale, RGBA, palette-mode images)
      2. Downscales if larger than max_size (512px default) to save memory.
         CLIP internally resizes to 224x224 anyway, so full resolution is wasteful.
      3. Converts to numpy array for any pixel-level checks

    Returns:
        Tuple of (PIL Image, numpy array) on success, or None on failure.
        Failures include corrupted files, unsupported formats, or empty images.
    """
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Downscale large images to save memory — CLIP only needs 224x224
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        arr = np.array(img)
        # Reject degenerate images (zero-size or single-channel after conversion)
        if arr.size == 0 or arr.ndim < 2:
            return None
        return img, arr
    except Exception as e:
        logging.debug(f"Failed to load {path}: {e}")
        return None


def load_image_full_resolution(path: str) -> Optional[Image.Image]:
    """
    Load an image at its original full resolution for the clean/export phase.
    Unlike load_image_safe(), no downscaling is applied because we need the
    full-quality image for the final cropped output.

    Returns:
        PIL Image on success, or None on failure.
    """
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logging.debug(f"Failed to load {path}: {e}")
        return None


# =============================================================================
# Chunk 2: CLIP Zero-Shot Classifier
# =============================================================================
# CLIP (Contrastive Language-Image Pretraining) maps both images and text into
# a shared embedding space. By comparing an image's embedding against text
# descriptions of each category, we can classify images without any task-specific
# training data — this is "zero-shot" classification.
#
# The key idea: if an image is closer in embedding space to "a photo of coral"
# than to "a photo of open water", it's likely a coral image.


@dataclass
class ClassificationResult:
    """
    Data container for a single image's classification output.
    Stores the predicted category, confidence score, and the full probability
    distribution across all categories for later analysis.
    """
    image_path: str                                     # Absolute path to the image file
    relative_path: str                                  # Path relative to dataset root
    dataset: str                                        # Which dataset this image belongs to
    location: str                                       # Dive site / recording session
    top_category: str                                   # Highest-scoring category name
    confidence: float                                   # Probability of the top category (0-1)
    is_empty: bool                                      # True if top_category is in EMPTY_CATEGORIES
    category_scores: Dict[str, float] = field(default_factory=dict)  # Full score distribution


class CLIPZeroShotClassifier:
    """
    CLIP-based zero-shot classifier for underwater image categories.

    How it works:
      1. At initialization, all text prompts for each category are encoded into
         CLIP's text embedding space. Multiple prompts per category are averaged
         ("prompt ensembling") to get a more robust category representation.
      2. At inference time, each image is encoded into CLIP's image embedding space.
      3. Cosine similarity between image and category embeddings determines scores.
      4. Softmax converts similarities to probabilities.

    Prompt ensembling (averaging embeddings from multiple phrasings like "a photo
    of fish swimming" and "a photo of a school of fish") helps because CLIP's
    text encoder can be sensitive to exact wording.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        categories: Dict[str, List[str]] = None,
    ):
        self.device = device
        self.model_name = model_name
        self.categories = categories or UNDERWATER_CATEGORIES
        self.category_names = list(self.categories.keys())

        # Load the pretrained CLIP model and its input processor from Hugging Face
        logging.info(f"Loading CLIP model: {model_name} on {device}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode (disables dropout, etc.)

        # CLIP has a learned temperature parameter (logit_scale) that controls
        # how "peaked" the softmax distribution is. We extract it here so the
        # output probabilities are properly calibrated.
        self.logit_scale = self.model.logit_scale.exp().detach()

        # Precompute text embeddings once at init rather than per-batch.
        # This is a major optimization — text encoding only happens once.
        self.category_embeddings = self._precompute_category_embeddings()
        logging.info(
            f"Precomputed embeddings for {len(self.category_names)} categories"
        )

    def _precompute_category_embeddings(self) -> torch.Tensor:
        """
        Encode all text prompts for every category and produce a single averaged
        embedding per category. This implements "prompt ensembling":

          1. For category "fish" with prompts ["a photo of fish swimming...", "a photo of a school..."]:
             - Encode each prompt → get 2 embedding vectors
             - L2-normalize each embedding (unit sphere)
             - Average the 2 vectors → single "fish" embedding
             - Re-normalize the average (back to unit sphere)

          2. Stack all category embeddings into a matrix [num_categories, embed_dim]

        Returns:
            Tensor of shape [num_categories, embed_dim] with one row per category.
        """
        all_embeddings = []

        with torch.no_grad():  # No gradient computation needed for inference
            for cat_name in self.category_names:
                prompts = self.categories[cat_name]
                # Tokenize all prompts for this category at once
                inputs = self.processor(
                    text=prompts, return_tensors="pt", padding=True, truncation=True
                )
                # Move to device, but exclude pixel_values (text-only encoding)
                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items()
                    if k != "pixel_values"
                }
                # Get text embeddings from CLIP's text encoder
                text_embeds = self.model.get_text_features(**inputs)
                # L2-normalize each prompt's embedding to the unit hypersphere
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                # Average across all prompts for this category, then re-normalize
                avg_embed = text_embeds.mean(dim=0)
                avg_embed = avg_embed / avg_embed.norm()
                all_embeddings.append(avg_embed)

        # Stack into a single matrix: [num_categories, embed_dim]
        return torch.stack(all_embeddings, dim=0)

    def classify_batch(
        self, images: List[Image.Image]
    ) -> List[Dict[str, float]]:
        """
        Classify a batch of PIL images against all categories simultaneously.

        Steps:
          1. Preprocess images (resize to 224x224, normalize) via CLIP's processor
          2. Encode images through CLIP's vision transformer → image embeddings
          3. Compute cosine similarity between each image and each category
          4. Apply learned logit scale and softmax → calibrated probabilities

        Args:
            images: List of PIL Image objects (any size, will be resized internally)

        Returns:
            List of dicts, one per image, mapping category_name → probability (0-1).
            Probabilities sum to 1.0 across categories for each image.
        """
        if not images:
            return []

        # Preprocess images for CLIP (resize, center-crop, normalize pixel values)
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Encode images through CLIP's vision transformer
            image_embeds = self.model.get_image_features(**inputs)
            # L2-normalize to unit sphere (required for cosine similarity)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # Cosine similarity between each image and each category embedding.
        # Result shape: [batch_size, num_categories]
        # Higher values = image is more similar to that category's description
        similarities = torch.matmul(
            image_embeds, self.category_embeddings.T
        )

        # Scale by CLIP's learned temperature and convert to probabilities.
        # The logit_scale amplifies small differences in cosine similarity,
        # making the softmax output more decisive (less uniform).
        logits = self.logit_scale * similarities
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Convert numpy array to list of dicts for easier downstream use
        results = []
        for i in range(len(images)):
            scores = {
                name: float(probs[i, j])
                for j, name in enumerate(self.category_names)
            }
            results.append(scores)

        return results


# =============================================================================
# Chunk 3: Artifact Detection (Static Border Detection)
# =============================================================================
# Underwater cameras mounted on ROVs or AUVs often have visible housing edges,
# instrument overlays, or timestamp bars that appear in every frame at the same
# position. These "static artifacts" need to be cropped out before 3D
# reconstruction because they confuse feature matching (they match across ALL
# frames, creating false correspondences).
#
# Detection strategy:
#   1. Sample ~50 images spread across the dataset
#   2. Resize all to the same dimensions for pixel-wise comparison
#   3. Compute the pixel-wise median across all samples — static artifacts
#      show up clearly because they don't change, while scene content averages out
#   4. Compute cross-image correlation at each pixel — high correlation means
#      the pixel looks the same in every image (= static artifact)
#   5. Scan inward from each edge to find where the artifact zone ends


@dataclass
class CropParams:
    """
    Stores how many pixels to remove from each edge of an image.
    All values default to 0 (no cropping). These are in original-resolution pixels.
    """
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    def is_empty(self) -> bool:
        """Returns True if no cropping is needed on any edge."""
        return self.top == 0 and self.bottom == 0 and self.left == 0 and self.right == 0

    def to_dict(self) -> Dict[str, int]:
        """Convert to a plain dict for JSON serialization."""
        return {"top": self.top, "bottom": self.bottom, "left": self.left, "right": self.right}


def detect_static_artifacts(
    root: str,
    extensions: Set[str],
    image_subdir: str,
    sample_size: int = ARTIFACT_SAMPLE_SIZE,
    scan_fraction: float = ARTIFACT_SCAN_FRACTION,
    variance_threshold: float = ARTIFACT_VARIANCE_THRESHOLD,
) -> CropParams:
    """
    Detect static artifacts (vehicle housing, camera borders, timestamp overlays)
    by analyzing border regions across a sample of images from the dataset.

    The core insight: if a border pixel looks nearly identical across 50 different
    images showing different scenes, it must be a static overlay — not scene content.

    Uses two complementary signals:
      1. Cross-image correlation: if border pixels correlate highly across
         different images, something static is there (even if semi-transparent).
      2. Cross-image variance: artifact pixels have LOW variance across images
         (they don't change), while scene pixels have HIGH variance.

    Args:
        root:               Dataset root directory path
        extensions:         Valid image file extensions (e.g., {".tiff", ".tif"})
        image_subdir:       Name of image subdirectory (e.g., "imgs")
        sample_size:        How many images to sample for analysis (default: 50)
        scan_fraction:      How far inward from each edge to scan (default: 20%)
        variance_threshold: Max std dev for a region to be "static" (default: 25.0)

    Returns:
        CropParams with number of pixels to remove from each edge.
    """
    logging.info(f"Detecting artifacts: sampling {sample_size} images...")

    # Get all image paths in the dataset
    all_paths = list(iter_dataset_images(root, extensions, image_subdir))
    if not all_paths:
        logging.warning("No images found for artifact detection")
        return CropParams()

    # Sample images evenly across the dataset (not just the first N).
    # This ensures we get diverse scenes for a reliable comparison.
    # If we have 1000 images and want 50, we take every 20th image.
    step = max(1, len(all_paths) // sample_size)
    sampled_paths = all_paths[::step][:sample_size]

    # Load all sampled images, resized to a common resolution for pixel-wise comparison.
    # We standardize to 640x480 because:
    #   - It's large enough to detect border artifacts with reasonable precision
    #   - It's small enough to stack ~50 images in memory comfortably
    target_size = (640, 480)  # (width, height)
    arrays = []
    original_sizes = []  # Track original sizes to scale crop params back later

    for abs_path, _, _ in sampled_paths:
        try:
            img = Image.open(abs_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            original_sizes.append(img.size)  # (width, height)
            img_resized = img.resize(target_size, Image.LANCZOS)
            arrays.append(np.array(img_resized))
        except Exception:
            continue

    # Need at least 5 images for meaningful statistical comparison
    if len(arrays) < 5:
        logging.warning(f"Only {len(arrays)} images loaded, skipping artifact detection")
        return CropParams()

    # Stack all images into a 4D array: [num_images, height, width, 3_channels]
    stack = np.stack(arrays, axis=0).astype(np.float32)
    h, w = stack.shape[1], stack.shape[2]

    # Calculate how many rows/columns to scan from each edge
    scan_rows = max(1, int(h * scan_fraction))  # For top/bottom edges
    scan_cols = max(1, int(w * scan_fraction))   # For left/right edges

    # Convert RGB to grayscale using standard luminance weights.
    # Grayscale simplifies the per-pixel statistics (1 value instead of 3).
    # Weights: 0.2989*R + 0.5870*G + 0.1140*B (ITU-R BT.601 standard)
    gray_stack = np.dot(stack[..., :3], [0.2989, 0.5870, 0.1140])  # [N, H, W]

    # Compute the pixel-wise median image across all samples.
    # The median is robust to outliers — scene content varies wildly across images
    # and gets "washed out", while static artifacts remain sharp and clear.
    median_img = np.median(gray_stack, axis=0)  # [H, W]

    # Compute per-pixel Pearson correlation between each image and the median.
    # Static artifacts will have correlation near 1.0 (they always look like the median).
    # Varying scene content will have lower correlation.
    mean_gray = gray_stack.mean(axis=0, keepdims=True)   # Mean across images at each pixel
    mean_median = median_img.mean()                       # Overall mean of the median image
    corr_num = ((gray_stack - mean_gray) * (median_img - mean_median)).mean(axis=0)
    corr_den = gray_stack.std(axis=0) * median_img.std() + 1e-8  # +epsilon to avoid division by zero
    correlation_map = corr_num / corr_den  # [H, W], values near 1.0 = static artifact

    crop = CropParams()

    # Determine the typical original image size (using median of all sampled sizes)
    # so we can scale the crop parameters back to original resolution
    orig_h = int(np.median([s[1] for s in original_sizes])) if original_sizes else h
    orig_w = int(np.median([s[0] for s in original_sizes])) if original_sizes else w

    # Scan each of the four edges independently for artifacts
    for edge_name in ("top", "bottom", "left", "right"):
        crop_px = _find_artifact_edge(
            gray_stack, median_img, correlation_map,
            edge_name, scan_rows if edge_name in ("top", "bottom") else scan_cols,
            h, w, variance_threshold,
        )
        if crop_px > 0:
            # Scale the detected crop (in 640x480 space) back to the original resolution.
            # e.g., if we found 20px to crop at 480px height, and original is 1920px,
            # the actual crop is 20 * (1920/480) = 80px.
            if edge_name in ("top", "bottom"):
                crop_px_orig = int(crop_px * orig_h / h)
            else:
                crop_px_orig = int(crop_px * orig_w / w)

            if edge_name == "top":
                crop.top = crop_px_orig
            elif edge_name == "bottom":
                crop.bottom = crop_px_orig
            elif edge_name == "left":
                crop.left = crop_px_orig
            else:
                crop.right = crop_px_orig

            logging.info(f"  Artifact on {edge_name}: crop {crop_px_orig}px (original res)")

    return crop


def _find_artifact_edge(
    gray_stack: np.ndarray,
    median_img: np.ndarray,
    corr_map: np.ndarray,
    edge_name: str,
    scan_size: int,
    h: int,
    w: int,
    var_threshold: float,
) -> int:
    """
    Scan inward from one edge of the image to find where a static artifact ends.

    The algorithm walks row-by-row (for top/bottom) or column-by-column (for
    left/right) from the edge toward the center. Each row/column is tested for
    two artifact indicators:

      1. Low cross-image standard deviation (< 80% of center std):
         Artifact pixels look similar across all images, so their variance is low.
         Center pixels show different scenes, so their variance is high.

      2. High cross-image correlation with the median (> 0.7):
         Artifact pixels closely match the median image (they're always the same).

    A row/column is flagged as "artifact" if EITHER condition is true.

    The scan allows small gaps (up to 3 clean rows) to handle gradual transitions
    at the artifact boundary. It stops if 3+ consecutive rows are clean.

    Args:
        gray_stack:    Grayscale image stack [num_images, height, width]
        median_img:    Pixel-wise median image [height, width]
        corr_map:      Cross-image correlation map [height, width]
        edge_name:     Which edge to scan from ("top", "bottom", "left", "right")
        scan_size:     How many rows/columns to scan inward
        h, w:          Image dimensions
        var_threshold: Variance threshold (currently unused; see center_std comparison)

    Returns:
        Number of rows/columns to crop from this edge (0 if no artifact found).
    """
    # Compute the average cross-image std of the CENTER region of the image.
    # This represents "normal" scene content variance — artifact regions should
    # have notably lower variance than this baseline.
    center_std = gray_stack[:, h // 3: 2 * h // 3, w // 3: 2 * w // 3].std(axis=0).mean()

    if edge_name in ("top", "bottom"):
        # ---- Scan rows for top/bottom edges ----
        artifact_rows = 0  # Tracks the deepest confirmed artifact row
        for i in range(scan_size):
            # Start from the edge and move inward
            if edge_name == "bottom":
                row_idx = h - 1 - i  # Bottom edge: start from last row, go up
            else:
                row_idx = i          # Top edge: start from first row, go down

            # Cross-image std for this row: take std across images, average across columns.
            # Low value = this row looks the same in every image = likely artifact.
            row_std = gray_stack[:, row_idx, :].std(axis=0).mean()

            # Average correlation for this row (from the precomputed correlation map)
            row_corr = corr_map[row_idx, :].mean()

            # A row is classified as "artifact" if:
            #   - Its variance is < 80% of the center region's variance, OR
            #   - Its correlation with the median exceeds 0.7
            is_artifact = (row_std < center_std * 0.8) or (row_corr > 0.7)

            if is_artifact:
                # Extend the artifact zone to include this row
                artifact_rows = i + 1
            else:
                # Allow up to 3 consecutive non-artifact rows (handles gradual
                # transitions like semi-transparent housing edges). But if we
                # see 3+ clean rows in a row, the artifact has ended.
                if i - artifact_rows >= 3:
                    break

        return artifact_rows

    else:
        # ---- Scan columns for left/right edges ----
        # Same logic as row scanning, but operates column-by-column
        artifact_cols = 0
        for i in range(scan_size):
            if edge_name == "right":
                col_idx = w - 1 - i  # Right edge: start from last column, go left
            else:
                col_idx = i          # Left edge: start from first column, go right

            col_std = gray_stack[:, :, col_idx].std(axis=0).mean()
            col_corr = corr_map[:, col_idx].mean()

            is_artifact = (col_std < center_std * 0.8) or (col_corr > 0.7)

            if is_artifact:
                artifact_cols = i + 1
            else:
                if i - artifact_cols >= 3:
                    break

        return artifact_cols


# =============================================================================
# Chunk 4: Analysis Processing Loop & Output
# =============================================================================
# This section ties together the CLIP classifier and artifact detector into a
# complete analysis pipeline. For each dataset it:
#   1. Runs artifact detection (sampling border regions)
#   2. Iterates over ALL images in batches, classifying each with CLIP
#   3. Writes per-image results to a CSV file
#   4. Collects aggregate statistics (category counts, per-location breakdowns)
#   5. Writes a JSON summary and prints a human-readable table


def analyze_dataset(
    config: Dict,
    classifier: CLIPZeroShotClassifier,
    outdir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict:
    """
    Run the complete analysis pipeline on a single dataset.

    Pipeline steps:
      1. Detect static border artifacts (camera housing, etc.)
      2. Classify every image using CLIP zero-shot classification
      3. Write per-image results to a CSV file (one row per image)
      4. Collect and return summary statistics

    The CSV contains: image paths, predicted category, confidence score,
    empty/useful flag, and the full probability distribution across all categories.

    Args:
        config:     Dataset configuration dict (name, root, extensions, etc.)
        classifier: Pre-initialized CLIP classifier instance
        outdir:     Directory to write output CSV and summary files
        batch_size: Number of images to classify per CLIP inference call

    Returns:
        Dict containing summary statistics for this dataset, including
        category counts, empty image count, per-location breakdowns, etc.
    """
    name = config["name"]
    root = config["root"]
    extensions = config["extensions"]
    image_subdir = config["image_subdir"]

    logging.info(f"\n{'='*60}")
    logging.info(f"Analyzing: {name} ({config['description']})")
    logging.info(f"Root: {root}")
    logging.info(f"{'='*60}")

    # Step 1: Detect static artifacts (camera housing, borders, overlays)
    crop_params = detect_static_artifacts(root, extensions, image_subdir)
    if not crop_params.is_empty():
        logging.info(f"Crop parameters: {crop_params.to_dict()}")
    else:
        logging.info("No static artifacts detected")

    # Step 2: Count total images for the progress bar
    total_count = count_dataset_images(root, extensions, image_subdir)
    logging.info(f"Total images: {total_count}")

    if total_count == 0:
        logging.warning(f"No images found in {root}")
        return {"name": name, "total": 0, "error": "no images found"}

    # Step 3: Classify all images in batches and write results to CSV

    # CSV columns: metadata fields + one column per category (probability scores)
    csv_path = outdir / f"{name}_classifications.csv"
    fieldnames = [
        "image_path", "relative_path", "dataset", "location",
        "top_category", "confidence", "is_empty",
    ] + CATEGORY_NAMES

    # Initialize running statistics that get updated as each batch is processed
    stats = {
        "name": name,
        "description": config["description"],
        "root": root,
        "total_images": total_count,
        "failed_images": 0,                    # Images that couldn't be loaded
        "category_counts": defaultdict(int),    # How many images per category
        "empty_count": 0,                       # Total empty/useless images
        "crop_params": crop_params.to_dict(),   # Artifact crop parameters
        # Per-location breakdown (e.g., per dive site)
        "location_stats": defaultdict(lambda: {"total": 0, "empty": 0, "categories": defaultdict(int)}),
    }

    # Accumulate images into batches before sending to CLIP
    batch_imgs: List[Image.Image] = []
    batch_meta: List[Tuple[str, str, str]] = []  # (abs_path, rel_path, location)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        pbar = tqdm(
            total=total_count,
            desc=f"Classifying {name}",
            unit="img",
        )

        # Iterate over every image in the dataset
        for abs_path, rel_path, location in iter_dataset_images(root, extensions, image_subdir):
            # Load and preprocess the image (resize for CLIP, convert to RGB)
            result = load_image_safe(abs_path)
            if result is None:
                stats["failed_images"] += 1
                pbar.update(1)
                continue

            pil_img, _ = result
            batch_imgs.append(pil_img)
            batch_meta.append((abs_path, rel_path, location))

            # When the batch is full, classify all images at once and write results
            if len(batch_imgs) >= batch_size:
                _process_and_write_batch(
                    batch_imgs, batch_meta, name, classifier, writer, stats
                )
                batch_imgs = []
                batch_meta = []

            pbar.update(1)

        # Process the final partial batch (fewer than batch_size images remaining)
        if batch_imgs:
            _process_and_write_batch(
                batch_imgs, batch_meta, name, classifier, writer, stats
            )

        pbar.close()

    logging.info(f"Results saved to: {csv_path}")

    # Convert defaultdicts to regular dicts so they can be JSON-serialized
    stats["category_counts"] = dict(stats["category_counts"])
    location_stats = {}
    for loc, loc_data in stats["location_stats"].items():
        location_stats[loc] = {
            "total": loc_data["total"],
            "empty": loc_data["empty"],
            "categories": dict(loc_data["categories"]),
        }
    stats["location_stats"] = location_stats

    return stats


def _process_and_write_batch(
    images: List[Image.Image],
    meta: List[Tuple[str, str, str]],
    dataset_name: str,
    classifier: CLIPZeroShotClassifier,
    writer: csv.DictWriter,
    stats: Dict,
) -> None:
    """
    Classify a batch of images with CLIP and write each result to the CSV.

    For each image in the batch:
      1. Get probability scores for all categories from CLIP
      2. Determine the top (most likely) category and its confidence
      3. Flag the image as "empty" if the top category is in EMPTY_CATEGORIES
      4. Write a CSV row with all metadata and scores
      5. Update the running statistics (category counts, location stats)

    If batch classification fails (e.g., GPU memory error), all images in the
    batch get zero scores as a fallback rather than crashing the pipeline.
    """
    try:
        scores_list = classifier.classify_batch(images)
    except Exception as e:
        logging.warning(f"Batch classification failed: {e}")
        # Fallback: assign zero probability to all categories for failed batch
        scores_list = [{cat: 0.0 for cat in CATEGORY_NAMES} for _ in images]

    for i, (abs_path, rel_path, location) in enumerate(meta):
        # Safely get scores (handles edge case where batch returned fewer results)
        scores = scores_list[i] if i < len(scores_list) else {cat: 0.0 for cat in CATEGORY_NAMES}

        # Find the category with the highest probability
        top_category = max(scores, key=scores.get)
        confidence = scores[top_category]
        # Mark as "empty" if the best match is an empty/useless category
        is_empty = top_category in EMPTY_CATEGORIES

        # Build and write the CSV row
        row = {
            "image_path": abs_path,
            "relative_path": rel_path,
            "dataset": dataset_name,
            "location": location,
            "top_category": top_category,
            "confidence": f"{confidence:.4f}",
            "is_empty": is_empty,
        }
        # Add individual category probability scores as additional columns
        for cat in CATEGORY_NAMES:
            row[cat] = f"{scores.get(cat, 0.0):.4f}"
        writer.writerow(row)

        # Update running aggregate statistics
        stats["category_counts"][top_category] += 1
        if is_empty:
            stats["empty_count"] += 1
        stats["location_stats"][location]["total"] += 1
        if is_empty:
            stats["location_stats"][location]["empty"] += 1
        stats["location_stats"][location]["categories"][top_category] += 1


def write_summary(all_stats: List[Dict], outdir: Path, model_name: str, device: str) -> None:
    """
    Write an aggregate summary JSON file combining results from all datasets.

    The summary includes:
      - Run metadata (model used, device, category definitions)
      - Per-dataset statistics (from each analyze_dataset call)
      - Aggregate totals across all datasets (total images, empty %, distribution)

    This JSON file is also read by the "clean" phase to retrieve crop parameters.
    """
    # Sum up totals across all datasets
    total_images = sum(s.get("total_images", 0) for s in all_stats)
    total_empty = sum(s.get("empty_count", 0) for s in all_stats)
    total_failed = sum(s.get("failed_images", 0) for s in all_stats)

    # Merge category counts across datasets
    aggregate_categories = defaultdict(int)
    for s in all_stats:
        for cat, count in s.get("category_counts", {}).items():
            aggregate_categories[cat] += count

    summary = {
        "run_metadata": {
            "model": model_name,
            "categories": CATEGORY_NAMES,
            "empty_categories": list(EMPTY_CATEGORIES),
            "device": device,
        },
        # Per-dataset results keyed by dataset name
        "datasets": {s["name"]: s for s in all_stats},
        # Cross-dataset aggregate statistics
        "aggregate": {
            "total_images": total_images,
            "total_failed": total_failed,
            "total_processed": total_images - total_failed,
            "total_empty": total_empty,
            "empty_percentage": (
                round(100 * total_empty / (total_images - total_failed), 1)
                if (total_images - total_failed) > 0 else 0
            ),
            "category_distribution": dict(aggregate_categories),
        },
    }

    summary_path = outdir / "classification_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logging.info(f"Summary saved to: {summary_path}")


def print_summary_table(all_stats: List[Dict]) -> None:
    """
    Print a human-readable summary table to the console after analysis.

    Shows per-dataset breakdowns including:
      - Total/failed/empty/useful image counts
      - Detected artifact crop parameters
      - Category distribution sorted by frequency
      - Aggregate totals across all datasets

    Categories marked with * are "empty" categories that will be removed during cleaning.
    """
    print(f"\n{'='*70}")
    print("DATASET ANALYSIS SUMMARY")
    print(f"{'='*70}")

    for stats in all_stats:
        name = stats.get("name", "?")
        total = stats.get("total_images", 0)
        failed = stats.get("failed_images", 0)
        processed = total - failed
        empty = stats.get("empty_count", 0)
        empty_pct = round(100 * empty / processed, 1) if processed > 0 else 0
        crop = stats.get("crop_params", {})

        print(f"\n  {name}")
        print(f"  {'─'*40}")
        print(f"  Total images:    {total:,}")
        print(f"  Failed to load:  {failed:,}")
        print(f"  Empty/useless:   {empty:,} ({empty_pct}%)")
        print(f"  Useful:          {processed - empty:,} ({100 - empty_pct}%)")

        if any(v > 0 for v in crop.values()):
            print(f"  Artifact crop:   top={crop.get('top',0)} bottom={crop.get('bottom',0)} "
                  f"left={crop.get('left',0)} right={crop.get('right',0)}")
        else:
            print(f"  Artifact crop:   none detected")

        # Show category breakdown sorted by count (most frequent first)
        cats = stats.get("category_counts", {})
        if cats:
            print(f"  Categories:")
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                pct = round(100 * count / processed, 1) if processed > 0 else 0
                marker = " *" if cat in EMPTY_CATEGORIES else ""
                print(f"    {cat:25s} {count:>6,} ({pct:>5.1f}%){marker}")

    # Print aggregate totals across all datasets
    total_all = sum(s.get("total_images", 0) for s in all_stats)
    failed_all = sum(s.get("failed_images", 0) for s in all_stats)
    empty_all = sum(s.get("empty_count", 0) for s in all_stats)
    processed_all = total_all - failed_all

    print(f"\n{'='*70}")
    print(f"TOTAL across all datasets:")
    print(f"  Images: {total_all:,} | Empty: {empty_all:,} | "
          f"Useful: {processed_all - empty_all:,}")
    print(f"  (* = empty category)")
    print(f"{'='*70}\n")


# =============================================================================
# Chunk 5: Clean & Export
# =============================================================================
# Phase 2 of the pipeline. This reads the analysis results from Phase 1 and
# produces a cleaned dataset by:
#   - Skipping images that were classified as "empty" (open water, blurry, etc.)
#   - Cropping detected border artifacts from the remaining useful images
#   - Saving the results in a clean directory structure
#
# The cleaned output is ready to be fed into Gaussian splatting pipelines like
# 3DGS, which need clean, artifact-free images for accurate reconstruction.


def clean_dataset(
    config: Dict,
    analysis_dir: Path,
    outdir: Path,
) -> Dict:
    """
    Clean a single dataset using the analysis results from Phase 1.

    For each image in the dataset:
      1. Check the CSV classification — if it was flagged as "empty", skip it
      2. Load the original full-resolution image
      3. Apply the artifact crop (remove static borders) if any was detected
      4. Save the cropped image to the output directory

    The output directory mirrors the original dataset structure, making it
    a drop-in replacement for the original dataset in downstream pipelines.

    TIFF files are converted to PNG during export to save disk space
    (PNG is lossless but significantly smaller than uncompressed TIFF).

    Args:
        config:       Dataset configuration dict (name, root, extensions, etc.)
        analysis_dir: Directory containing Phase 1 output (CSVs + summary JSON)
        outdir:       Root directory for cleaned dataset output

    Returns:
        Dict with cleaning statistics (kept/removed/failed counts).
    """
    name = config["name"]
    csv_path = analysis_dir / f"{name}_classifications.csv"
    summary_path = analysis_dir / "classification_summary.json"

    # The CSV from Phase 1 must exist — it tells us which images to keep/remove
    if not csv_path.exists():
        logging.error(f"Analysis CSV not found: {csv_path}")
        logging.error(f"Run 'analyze' first for dataset '{name}'")
        return {"name": name, "error": "no analysis CSV"}

    # Load crop parameters from the Phase 1 summary JSON.
    # These tell us how many pixels to remove from each edge to eliminate
    # static artifacts (camera housing, overlays, etc.)
    crop = CropParams()
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        ds_summary = summary.get("datasets", {}).get(name, {})
        cp = ds_summary.get("crop_params", {})
        crop = CropParams(
            top=cp.get("top", 0),
            bottom=cp.get("bottom", 0),
            left=cp.get("left", 0),
            right=cp.get("right", 0),
        )

    logging.info(f"\n{'='*60}")
    logging.info(f"Cleaning: {name}")
    if not crop.is_empty():
        logging.info(f"Applying crop: {crop.to_dict()}")
    logging.info(f"Output: {outdir / name}")
    logging.info(f"{'='*60}")

    # Read all classification results from the Phase 1 CSV
    kept = 0      # Images successfully cleaned and saved
    removed = 0   # Images skipped because they were classified as "empty"
    failed = 0    # Images that couldn't be loaded from disk

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    pbar = tqdm(rows, desc=f"Cleaning {name}", unit="img")

    for row in pbar:
        # Check if this image was classified as empty/useless in Phase 1
        is_empty = row["is_empty"].strip().lower() == "true"

        if is_empty:
            # Skip empty images — they have no useful structure for 3D reconstruction
            removed += 1
            continue

        # Load the original image at full resolution (no downscaling)
        abs_path = row["image_path"]
        rel_path = row["relative_path"]
        img = load_image_full_resolution(abs_path)

        if img is None:
            failed += 1
            continue

        # Apply artifact crop if artifacts were detected for this dataset.
        # The crop removes a fixed number of pixels from each edge.
        if not crop.is_empty():
            w, h = img.size
            # Calculate the crop box: (left, upper, right, lower)
            left = crop.left
            top = crop.top
            right = w - crop.right
            bottom = h - crop.bottom
            # Safety check: only crop if the resulting image would be non-degenerate
            if right > left and bottom > top:
                img = img.crop((left, top, right, bottom))

        # Save to output directory, preserving the original relative path structure.
        # e.g., input:  flsea-vi/canyons/flatiron/imgs/001.tiff
        #       output: cleaned_datasets/flsea_vi/canyons/flatiron/imgs/001.png
        out_path = outdir / name / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert TIFF → PNG for smaller file sizes (both are lossless,
        # but PNG uses compression while raw TIFF does not)
        if out_path.suffix.lower() in (".tiff", ".tif"):
            out_path = out_path.with_suffix(".png")

        img.save(out_path)
        kept += 1

    pbar.close()

    stats = {
        "name": name,
        "total": total,
        "kept": kept,
        "removed": removed,
        "failed": failed,
        "crop_params": crop.to_dict(),
    }

    logging.info(f"  Kept: {kept:,} | Removed: {removed:,} | Failed: {failed:,}")
    return stats


# =============================================================================
# CLI: Command-Line Interface & Entry Point
# =============================================================================
# The script uses argparse with subcommands:
#   python preprocess_datasets.py analyze [options]   → Phase 1
#   python preprocess_datasets.py clean [options]     → Phase 2
#
# Typical workflow:
#   1. Run "analyze" first to classify all images and detect artifacts
#   2. Review the output summary to verify results look reasonable
#   3. Run "clean" to produce the filtered, cropped dataset


def get_device(device_str: str) -> str:
    """
    Resolve the device string to an actual PyTorch device for CLIP inference.

    "auto" will select the best available device:
      - CUDA (NVIDIA GPU) if available — fastest
      - MPS (Apple Silicon GPU) if available — good on M1/M2/M3 Macs
      - CPU as fallback — slowest but always works
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments. Defines two subcommands:

    analyze:
      --outdir       Where to save classification CSVs and summary JSON
      --model        Which CLIP model to use (default: ViT-B/32)
      --device       Compute device: auto, cpu, cuda, mps
      --batch-size   Images per CLIP inference call
      --datasets     Process only specific datasets (by name)
      --verbose      Enable debug-level logging

    clean:
      --analysis-dir Where to find Phase 1 results (CSVs + summary)
      --outdir       Where to save the cleaned dataset
      --datasets     Clean only specific datasets (by name)
      --verbose      Enable debug-level logging
    """
    parser = argparse.ArgumentParser(
        description="Underwater dataset preprocessing for Gaussian splatting.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- analyze subcommand (Phase 1) ---
    analyze_parser = subparsers.add_parser(
        "analyze", help="Classify images and detect artifacts"
    )
    analyze_parser.add_argument(
        "--outdir", default=DEFAULT_OUTDIR,
        help="Output directory for analysis results",
    )
    analyze_parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="CLIP model identifier",
    )
    analyze_parser.add_argument(
        "--device", default="auto", choices=["cpu", "cuda", "mps", "auto"],
        help="Device for inference",
    )
    analyze_parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Batch size for CLIP inference",
    )
    analyze_parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Which datasets to process (default: all). Options: flsea_vi, shipwreck, sunboat",
    )
    analyze_parser.add_argument("--verbose", action="store_true")

    # --- clean subcommand (Phase 2) ---
    clean_parser = subparsers.add_parser(
        "clean", help="Crop artifacts and remove empty images"
    )
    clean_parser.add_argument(
        "--analysis-dir", default=DEFAULT_OUTDIR,
        help="Directory containing analysis results from 'analyze'",
    )
    clean_parser.add_argument(
        "--outdir", default=DEFAULT_CLEAN_OUTDIR,
        help="Output directory for cleaned datasets",
    )
    clean_parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Which datasets to clean (default: all)",
    )
    clean_parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    """
    Main entry point. Dispatches to either the "analyze" or "clean" pipeline
    based on the subcommand provided on the command line.
    """
    args = parse_args()

    if not args.command:
        print("Usage: python preprocess_datasets.py {analyze,clean} [options]")
        print("Run with --help for details.")
        sys.exit(1)

    # Configure logging — verbose mode shows DEBUG messages for troubleshooting
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # If --datasets was specified, only process those datasets.
    # Otherwise, process all datasets defined in DATASET_CONFIGS.
    selected_names = set(args.datasets) if args.datasets else None
    configs = [
        c for c in DATASET_CONFIGS
        if selected_names is None or c["name"] in selected_names
    ]

    if not configs:
        logging.error(f"No matching datasets. Available: {[c['name'] for c in DATASET_CONFIGS]}")
        sys.exit(1)

    if args.command == "analyze":
        # ---- Phase 1: Analyze ----
        # Classify all images with CLIP and detect border artifacts
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Initialize the compute device (GPU if available, else CPU)
        device = get_device(args.device)
        logging.info(f"Using device: {device}")

        # Load the CLIP model once and reuse it across all datasets.
        # Model loading is expensive (~2-5 seconds), so we do it once upfront.
        classifier = CLIPZeroShotClassifier(
            model_name=args.model, device=device
        )

        # Process each dataset sequentially
        all_stats = []
        for config in configs:
            stats = analyze_dataset(
                config, classifier, outdir, batch_size=args.batch_size
            )
            all_stats.append(stats)

        # Write the combined summary JSON and print results to console
        write_summary(all_stats, outdir, args.model, device)
        print_summary_table(all_stats)

    elif args.command == "clean":
        # ---- Phase 2: Clean ----
        # Read Phase 1 results, filter out empty images, crop artifacts, export
        analysis_dir = Path(args.analysis_dir)
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Clean each dataset sequentially
        all_clean_stats = []
        for config in configs:
            stats = clean_dataset(config, analysis_dir, outdir)
            all_clean_stats.append(stats)

        # Print a summary table of cleaning results
        print(f"\n{'='*60}")
        print("CLEANING SUMMARY")
        print(f"{'='*60}")
        for s in all_clean_stats:
            if "error" in s:
                print(f"  {s['name']}: ERROR - {s['error']}")
                continue
            print(f"  {s['name']:15s} kept={s['kept']:>6,}  removed={s['removed']:>6,}  "
                  f"failed={s['failed']:>4,}  (of {s['total']:,})")
        print(f"{'='*60}\n")


# Standard Python entry point — only runs when the script is executed directly,
# not when imported as a module.
if __name__ == "__main__":
    main()
