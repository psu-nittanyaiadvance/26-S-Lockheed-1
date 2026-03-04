#!/usr/bin/env python3
"""
=============================================================================
Underwater Dataset - Comprehensive Exploratory Data Analysis (EDA) Tool
=============================================================================
OPTIMIZED VERSION — reads each image only ONCE instead of 3 times.

This script performs a full EDA on underwater image datasets structured for
3D reconstruction pipelines (NeRF, Gaussian Splatting, etc.). It analyzes
image properties, color profiles, quality metrics, scene content, and
COLMAP sparse reconstruction data, then generates an interactive HTML
report with Chart.js visualizations.

Performance: ~100-200ms per image (vs ~400-600ms in the non-optimized version).
For 8,000 images, expect roughly 15-25 minutes instead of 45-90 minutes.

The script is designed to work with ANY underwater image dataset that follows
a common structure:
    dataset_root/
        scene_1/
            images_wb/ (or Images_wb/ or images/)
                *.png / *.jpg
            sparse/0/
                cameras.bin
                images.bin
                points3D.bin
            poses_bounds.npy (optional)
        scene_2/
            ...

Requirements:
    pip install Pillow numpy opencv-python-headless

Usage:
    python underwater_eda_optimized.py                        # auto-detect dataset in same directory
    python underwater_eda_optimized.py /path/to/dataset       # specify dataset path explicitly

Output:
    underwater_eda_report.html  (saved in the dataset directory)
    eda_output/eda_data.json    (raw analysis data for further use)

Author: Generated for Lockheed Martin NAISS Gaussian Splatting Project
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os          # File system operations (path joining, directory listing, etc.)
import sys         # Command-line argument parsing
import json        # Serialize analysis results to JSON for the report
import struct      # Parse COLMAP binary files (cameras.bin, images.bin, points3D.bin)
import time        # Timing and ETA calculations
import numpy as np # Numerical computations (mean, std, percentiles, etc.)
from PIL import Image           # Read image metadata (dimensions, color mode)
import cv2                      # Computer vision: color analysis, blur detection, edge detection
from collections import defaultdict  # Convenient counting dictionaries
from pathlib import Path        # Modern path handling
import warnings
warnings.filterwarnings('ignore')  # Suppress PIL/OpenCV deprecation warnings


# =============================================================================
# PROGRESS BAR UTILITY
# =============================================================================
# A simple terminal progress bar that requires no external packages (no tqdm).
# Shows: [=====>    ] 55% (4400/8000) | 12.3 img/s | ETA: 4m 52s
# =============================================================================
def print_progress(current, total, start_time, prefix="Progress"):
    """
    Print a dynamic progress bar to the terminal.

    Args:
        current: number of items processed so far
        total: total number of items to process
        start_time: time.time() when processing started (for ETA calculation)
        prefix: label to show before the progress bar
    """
    bar_length = 40
    fraction = current / total if total > 0 else 1
    filled = int(bar_length * fraction)
    bar = '=' * filled + '>' * (1 if filled < bar_length else 0) + '.' * (bar_length - filled - 1)

    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0 else 0

    if rate > 0 and current < total:
        remaining = (total - current) / rate
        if remaining > 60:
            eta_str = f"{remaining/60:.0f}m {remaining%60:.0f}s"
        else:
            eta_str = f"{remaining:.0f}s"
    elif current >= total:
        eta_str = "Done!"
    else:
        eta_str = "calculating..."

    print(f"\r  {prefix}: [{bar}] {fraction*100:.0f}% ({current}/{total}) | {rate:.1f} img/s | ETA: {eta_str}   ", end='', flush=True)

    if current >= total:
        print()  # Newline when complete


# =============================================================================
# CONFIGURATION
# =============================================================================
# Determine where the dataset lives. If the user passes a path as a command-line
# argument, use that. Otherwise, assume the dataset is in the same directory as
# this script (useful if you drop the script into your dataset folder).
if len(sys.argv) > 1:
    BASE_DIR = sys.argv[1]
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output paths
OUT_DIR = os.path.join(BASE_DIR, "eda_output")       # Directory for raw JSON data
REPORT_PATH = os.path.join(BASE_DIR, "underwater_eda_report.html")  # Final HTML report
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Dataset directory: {BASE_DIR}")
print(f"Report will be saved to: {REPORT_PATH}")


# =============================================================================
# PHASE 1: DATASET STRUCTURE ANALYSIS
# =============================================================================
# Walk through the dataset directory tree and catalog every file.
# We separate files into three categories:
#   - Images (png, jpg, jpeg, tiff, tif, bmp)
#   - COLMAP sparse reconstruction files (anything under a 'sparse' directory)
#   - Other files (poses_bounds.npy, README, etc.)
#
# SCENE DETECTION (auto-detects two common dataset layouts):
#   Layout A (flat):   dataset/scene1/images/  dataset/scene2/images/  ...
#   Layout B (nested): dataset/category1/scene1/imgs/  dataset/category1/scene2/imgs/  ...
#
# If a top-level directory contains image files (directly or in a standard image
# subdirectory), it is treated as a scene. Otherwise, we check one level deeper
# to see if it's a category folder containing scene subfolders (like FLSea).
#
# FOLDER EXCLUSIONS: directories named 'depth', 'seaErra', 'semantics', or
# 'segmentation' are skipped when searching for images, since these contain
# derived data (depth maps, error maps, etc.), not camera images.
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1: Dataset Structure Analysis")
print("=" * 60)

# Recognized image file extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')

# Directories to skip when looking for camera images (contain derived data, not photos)
SKIP_DIRS = {'depth', 'semantics', 'segmentation', 'navigation', 'concat'}

# Standard image subdirectory names (where camera images typically live)
IMAGE_DIR_NAMES = {'imgs', 'images', 'images_wb', 'image', 'camera'}


def dir_has_images(path):
    """Check if a directory (or its standard image subdirs) contains image files."""
    # Check the directory itself
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                return True
    # Check standard image subdirectories (imgs/, images/, images_wb/)
    for sub in IMAGE_DIR_NAMES:
        sub_path = os.path.join(path, sub)
        if os.path.isdir(sub_path):
            for f in os.listdir(sub_path):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    return True
        # Also check case-insensitive variants (e.g., Images_wb)
        for actual_dir in os.listdir(path) if os.path.isdir(path) else []:
            if actual_dir.lower() == sub and os.path.isdir(os.path.join(path, actual_dir)):
                for f in os.listdir(os.path.join(path, actual_dir)):
                    if f.lower().endswith(IMAGE_EXTENSIONS):
                        return True
    return False


# --- AUTO-DETECT SCENE STRUCTURE ---
# Recursively searches up to 3 levels deep for scene directories that contain
# images. This handles all common underwater dataset layouts:
#   Layout A (flat):       dataset/scene1/images/
#   Layout B (2-level):    dataset/category/scene1/imgs/
#   Layout C (3-level):    dataset/site/date/recording/camera/
#
# A "scene" is the deepest directory that directly contains (or has a standard
# image subdirectory like camera/, imgs/, images/ that contains) image files.
# Everything above that is treated as organizational grouping.

def is_skippable_dir(name):
    """Check if a directory name should be skipped during scene search."""
    return (name.startswith('.')
            or name == 'eda_output'
            or name.lower() in SKIP_DIRS
            or 'sonar' in name.lower()
            or name == 'calibration')

def find_scenes(base, prefix="", max_depth=3):
    """
    Recursively find scene directories up to max_depth levels deep.

    A directory is a scene if it contains image files (directly or in a
    standard image subdirectory like camera/, imgs/, images/, etc.).
    If it doesn't contain images, we recurse into its subdirectories
    to look for scenes deeper in the tree.

    Args:
        base: absolute path to search in
        prefix: relative path prefix for scene naming (e.g., "Shipwreck/07-03-2022")
        max_depth: how many more levels to recurse (0 = stop)

    Returns: list of (scene_name, scene_abs_path) tuples
    """
    if max_depth < 0:
        return []

    found = []

    if dir_has_images(base):
        # This directory is a scene
        scene_name = prefix if prefix else os.path.basename(base)
        found.append((scene_name, base))
    else:
        # Not a scene — check subdirectories
        try:
            sub_dirs = [sd for sd in os.listdir(base)
                        if os.path.isdir(os.path.join(base, sd))
                        and not is_skippable_dir(sd)]
        except PermissionError:
            return []

        for sd in sorted(sub_dirs):
            sd_path = os.path.join(base, sd)
            child_prefix = f"{prefix}/{sd}" if prefix else sd
            found.extend(find_scenes(sd_path, child_prefix, max_depth - 1))

    return found

# Run the scene finder starting from BASE_DIR
top_dirs = [d for d in os.listdir(BASE_DIR)
            if os.path.isdir(os.path.join(BASE_DIR, d))
            and not is_skippable_dir(d)]
top_dirs.sort()

scenes = []          # Scene names (may include paths like "Shipwreck/07-03-2022/rec0")
scene_paths = {}     # Map scene name → absolute path on disk

for d in sorted(top_dirs):
    d_path = os.path.join(BASE_DIR, d)
    for scene_name, scene_abs_path in find_scenes(d_path, prefix=d, max_depth=3):
        scenes.append(scene_name)
        scene_paths[scene_name] = scene_abs_path

scenes.sort()
print(f"Scenes found ({len(scenes)}): {scenes}")

# Data structures to accumulate results
structure_data = {}      # Per-scene file inventory
total_images = 0         # Global image count
total_size_bytes = 0     # Global dataset size in bytes
all_image_paths = []     # Flat list of every image path (used by later phases)

for scene in scenes:
    scene_path = scene_paths[scene]
    scene_info = {"images": [], "sparse_files": [], "other_files": []}
    scene_size = 0

    # os.walk recursively visits every subdirectory
    for root, dirs, files in os.walk(scene_path):
        # Skip directories that contain derived data (depth maps, error maps, etc.)
        # Modifying dirs in-place tells os.walk to skip those subdirectories
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]

        for f in files:
            # Skip macOS metadata files
            if f == '.DS_Store':
                continue

            fpath = os.path.join(root, f)
            fsize = os.path.getsize(fpath)
            scene_size += fsize
            total_size_bytes += fsize

            # Relative path within the scene folder (used to detect 'sparse' subdirectory)
            rel_path = os.path.relpath(fpath, scene_path)

            # Categorize the file by extension or directory location
            if f.lower().endswith(IMAGE_EXTENSIONS):
                scene_info["images"].append(fpath)
                all_image_paths.append(fpath)
                total_images += 1
            elif 'sparse' in rel_path:
                scene_info["sparse_files"].append(fpath)
            else:
                scene_info["other_files"].append(fpath)

    scene_info["total_size_bytes"] = scene_size
    scene_info["num_images"] = len(scene_info["images"])
    structure_data[scene] = scene_info
    print(f"  {scene}: {scene_info['num_images']} images, {scene_size/1024/1024:.1f} MB")

print(f"\nTotal images: {total_images}")
print(f"Total dataset size: {total_size_bytes/1024/1024:.1f} MB")

# Count how many images of each file type exist (e.g., {'.png': 88})
file_types = defaultdict(int)
for img_path in all_image_paths:
    ext = os.path.splitext(img_path)[1].lower()
    file_types[ext] += 1
print(f"File types: {dict(file_types)}")

# Safety check: abort if no images were found
if total_images == 0:
    print("\nERROR: No images found in the dataset directory.")
    print("Make sure the dataset has scene subdirectories containing image files.")
    print(f"Searched in: {BASE_DIR}")
    sys.exit(1)


# =============================================================================
# PHASE 2: SINGLE-PASS IMAGE ANALYSIS (Metadata + Color + Quality + Content)
# =============================================================================
# OPTIMIZATION: The original script read each image 3 times from disk (once
# for color, once for quality, once for content). This version reads each
# image ONCE with cv2.imread and performs ALL analyses on that single load.
# This cuts runtime by roughly 60-70% for large datasets.
#
# For each image we compute:
#   A) METADATA — dimensions, file size, color mode, megapixels (via PIL)
#   B) COLOR PROFILE — mean RGB/HSV/LAB, blue/red ratio, color classification
#   C) QUALITY — Laplacian variance (blur), contrast, edge density, quality score
#   D) CONTENT — scene content classification (coral, rock, open water, etc.)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 2: Single-Pass Image Analysis (Color + Quality + Content)")
print(f"         Processing {total_images} images...")
print("=" * 60)

image_metadata = []
color_data = []
quality_data = []
content_data = []

start_time = time.time()

for idx, img_path in enumerate(all_image_paths):
    try:
        # --- Determine which scene this image belongs to ---
        # Match against the scene's actual filesystem path for reliable detection,
        # especially with nested layouts like "canyons/horse_canyon"
        scene_name = ""
        for s in scenes:
            if scene_paths[s] in img_path:
                scene_name = s
                break

        # =====================================================================
        # SINGLE cv2.imread — this is the expensive disk read we do ONCE
        # =====================================================================
        img = cv2.imread(img_path)
        if img is None:
            continue

        # =====================================================================
        # A) METADATA — try PIL first; fall back to cv2 if PIL rejects the file
        # (Some PNGs have non-standard headers that macOS can read but PIL cannot)
        # =====================================================================
        fsize = os.path.getsize(img_path)  # File size on disk
        try:
            pil_img = Image.open(img_path)
            w, h = pil_img.size        # Width and height in pixels
            mode = pil_img.mode        # Color mode: 'RGB', 'L' (grayscale), 'RGBA', etc.
        except Exception:
            # cv2 already loaded the image successfully, so derive metadata from it
            h, w = img.shape[:2]
            # Infer color mode from cv2 channel count
            if len(img.shape) == 2:
                mode = "L"             # Grayscale
            elif img.shape[2] == 4:
                mode = "RGBA"
            else:
                mode = "RGB"

        image_metadata.append({
            "path": img_path,
            "filename": os.path.basename(img_path),
            "scene": scene_name,
            "width": w,
            "height": h,
            "mode": mode,
            "file_size_bytes": fsize,
            "aspect_ratio": round(w / h, 3),          # e.g., 1.778 for 16:9
            "megapixels": round(w * h / 1e6, 2)       # e.g., 2.1 MP
        })

        # Pre-compute shared conversions (used by color, quality, and content)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_img, w_img = gray.shape

        # Pre-compute shared features
        edges = cv2.Canny(gray, 50, 150)               # Binary edge map
        edge_density = edges.sum() / (h_img * w_img * 255)  # Fraction of edge pixels

        # =====================================================================
        # B) COLOR PROFILE ANALYSIS
        # =====================================================================
        # Underwater images suffer from wavelength-dependent light absorption:
        #   - Red light absorbed first (~5m), green further (~15-20m), blue most (~60-75m)
        # We classify each image by brightness tier + blue/green ratio.

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Mean pixel values across entire image for each color space
        mean_bgr = img.mean(axis=(0, 1))   # [Blue, Green, Red] means
        mean_hsv = hsv.mean(axis=(0, 1))   # [Hue, Saturation, Value] means
        mean_lab = lab.mean(axis=(0, 1))   # [L, A, B] means

        # Standard deviation per BGR channel
        std_bgr = img.std(axis=(0, 1))

        # Extract individual metrics
        brightness = mean_hsv[2]     # V channel: 0=black, 255=white
        saturation = mean_hsv[1]     # S channel: 0=grey, 255=vivid color
        hue = mean_hsv[0]            # H channel: 0-180 in OpenCV

        # Color ratios (adding 1e-6 prevents division by zero)
        b, g, r = mean_bgr
        blue_ratio = b / (r + 1e-6)    # Blue / Red
        green_ratio = g / (r + 1e-6)   # Green / Red

        # --- COLOR CLASSIFICATION ---
        if brightness < 60:
            color_class = "Very Dark / Deep Water"
        elif brightness < 100:
            if blue_ratio > 1.3:
                color_class = "Dark Blue"
            elif green_ratio > 1.2:
                color_class = "Dark Green/Teal"
            else:
                color_class = "Dark Grey/Murky"
        elif brightness < 150:
            if blue_ratio > 1.3:
                color_class = "Medium Blue"
            elif green_ratio > 1.2 and blue_ratio > 1.0:
                color_class = "Blue-Green/Teal"
            elif green_ratio > 1.2:
                color_class = "Green/Algae"
            else:
                color_class = "Neutral/Grey"
        else:
            if blue_ratio > 1.3:
                color_class = "Bright Blue"
            elif green_ratio > 1.2:
                color_class = "Bright Green"
            else:
                color_class = "Bright/Well-lit"

        # Dominant channel
        channel_order = np.argsort(mean_bgr)[::-1]
        dominant_channel = ["Blue", "Green", "Red"][channel_order[0]]

        color_data.append({
            "path": img_path,
            "filename": os.path.basename(img_path),
            "scene": scene_name,
            "mean_r": float(r), "mean_g": float(g), "mean_b": float(b),
            "brightness": float(brightness),
            "saturation": float(saturation),
            "hue": float(hue),
            "blue_ratio": float(blue_ratio),
            "green_ratio": float(green_ratio),
            "std_r": float(std_bgr[2]), "std_g": float(std_bgr[1]), "std_b": float(std_bgr[0]),
            "color_class": color_class,
            "dominant_channel": dominant_channel,
            "mean_lab_l": float(mean_lab[0]),
            "mean_lab_a": float(mean_lab[1]),
            "mean_lab_b": float(mean_lab[2]),
        })

        # =====================================================================
        # C) IMAGE QUALITY ANALYSIS
        # =====================================================================
        # Laplacian variance (blur detection):
        #   < 50 = Very Blurry, 50-200 = Somewhat Blurry,
        #   200-500 = Acceptable, > 500 = Sharp
        # Contrast: grayscale std dev
        # Edge density: fraction of Canny edge pixels (already computed above)
        # Dynamic range: max pixel - min pixel
        # Quality score: weighted combination (sharpness 40%, contrast 30%, dynamic range 30%)

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()

        # Brightness uniformity across 4 quadrants (high = uneven lighting)
        quadrants = [
            gray[:h_img//2, :w_img//2].mean(),
            gray[:h_img//2, w_img//2:].mean(),
            gray[h_img//2:, :w_img//2].mean(),
            gray[h_img//2:, w_img//2:].mean()
        ]
        brightness_uniformity = np.std(quadrants)

        snr = gray.mean() / (gray.std() + 1e-6)
        dynamic_range = int(gray.max()) - int(gray.min())

        # Blur classification
        if laplacian_var < 50:
            blur_class = "Very Blurry"
        elif laplacian_var < 200:
            blur_class = "Somewhat Blurry"
        elif laplacian_var < 500:
            blur_class = "Acceptable"
        else:
            blur_class = "Sharp"

        # Composite quality score (0-100)
        sharp_score = min(laplacian_var / 500, 1.0)
        contrast_score = min(contrast / 60, 1.0)
        dr_score = min(dynamic_range / 200, 1.0)
        quality_score = (sharp_score * 0.4 + contrast_score * 0.3 + dr_score * 0.3) * 100

        quality_data.append({
            "path": img_path,
            "filename": os.path.basename(img_path),
            "scene": scene_name,
            "laplacian_var": float(laplacian_var),
            "contrast": float(contrast),
            "brightness_uniformity": float(brightness_uniformity),
            "edge_density": float(edge_density),
            "snr": float(snr),
            "dynamic_range": int(dynamic_range),
            "blur_class": blur_class,
            "quality_score": float(quality_score),
        })

        # =====================================================================
        # D) SCENE CONTENT CLASSIFICATION
        # =====================================================================
        # Heuristic classifier based on edge density + color variance.
        # CAVEAT: "Coral Reef" vs "Rocky Substrate" is really measuring
        # "colorful texture" vs "monochrome texture", not actual biology.

        mean_brightness = float(hsv[:, :, 2].mean())
        mean_saturation = float(hsv[:, :, 1].mean())
        color_variance = float(img.std())

        # Compare top vs bottom half (seafloor = more edges at bottom)
        bottom_edges = edges[h_img//2:, :].sum() / (h_img//2 * w_img * 255)
        top_edges = edges[:h_img//2, :].sum() / (h_img//2 * w_img * 255)

        # Feature flags
        has_structure = edge_density > 0.05
        has_color_variety = color_variance > 35
        is_uniform = color_variance < 25
        is_bright = mean_brightness > 120
        is_dark = mean_brightness < 80
        bottom_heavier = bottom_edges > top_edges * 1.3

        # Build content tags
        content_tags = []
        if edge_density > 0.08 and has_color_variety:
            content_tags.append("Coral Reef")
        if edge_density > 0.05 and not has_color_variety:
            content_tags.append("Rocky Substrate")
        if bottom_heavier and bottom_edges > 0.04:
            content_tags.append("Seafloor Visible")
        if is_uniform and edge_density < 0.03:
            content_tags.append("Open Water / Featureless")
        if mean_saturation > 80 and has_color_variety:
            content_tags.append("Marine Life / Color")
        if is_dark and edge_density < 0.04:
            content_tags.append("Low Visibility / Murky")
        if edge_density > 0.1:
            content_tags.append("High Detail / Complex Scene")
        if not content_tags:
            if has_structure:
                content_tags.append("Underwater Scene (Mixed)")
            else:
                content_tags.append("Water Column / Ambient")

        # Pick primary classification (priority order)
        if "Coral Reef" in content_tags:
            primary_class = "Coral Reef"
        elif "Open Water / Featureless" in content_tags:
            primary_class = "Open Water"
        elif "Rocky Substrate" in content_tags:
            primary_class = "Rocky Substrate"
        elif "Seafloor Visible" in content_tags:
            primary_class = "Seafloor"
        elif "Low Visibility / Murky" in content_tags:
            primary_class = "Low Visibility"
        else:
            primary_class = "Underwater Scene (Mixed)"

        # "Just water" detection — no visible features
        is_just_water = ((edge_density < 0.03 and is_uniform) or
                         (edge_density < 0.02 and mean_saturation < 50))

        content_data.append({
            "path": img_path,
            "filename": os.path.basename(img_path),
            "scene": scene_name,
            "primary_class": primary_class,
            "content_tags": content_tags,
            "edge_density": float(edge_density),
            "color_variance": float(color_variance),
            "mean_brightness": float(mean_brightness),
            "mean_saturation": float(mean_saturation),
            "is_just_water": is_just_water,
        })

    except Exception as e:
        print(f"\n  Error processing {img_path}: {e}")

    # Update progress bar every image (or every 10 for very large datasets)
    if total_images < 500 or (idx + 1) % 10 == 0 or (idx + 1) == total_images:
        print_progress(idx + 1, total_images, start_time)

elapsed_total = time.time() - start_time
print(f"\n  Finished in {elapsed_total:.1f}s ({total_images / elapsed_total:.1f} images/sec)")

# --- Print summaries ---

# Metadata summary
widths = [m["width"] for m in image_metadata]
heights = [m["height"] for m in image_metadata]
filesizes = [m["file_size_bytes"] for m in image_metadata]
print(f"\nImage dimensions: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
print(f"Unique dimensions: {set((m['width'], m['height']) for m in image_metadata)}")
print(f"File sizes: {min(filesizes)/1024:.0f} KB to {max(filesizes)/1024:.0f} KB (avg: {np.mean(filesizes)/1024:.0f} KB)")

# Color summary
color_classes = defaultdict(int)
for cd in color_data:
    color_classes[cd["color_class"]] += 1
print("\nColor Classification Distribution:")
for cls, count in sorted(color_classes.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {count} images ({100 * count / len(color_data):.1f}%)")

# Quality summary
blur_classes_count = defaultdict(int)
for qd in quality_data:
    blur_classes_count[qd["blur_class"]] += 1
print("\nBlur Classification:")
for cls in ["Sharp", "Acceptable", "Somewhat Blurry", "Very Blurry"]:
    count = blur_classes_count.get(cls, 0)
    print(f"  {cls}: {count} images ({100 * count / len(quality_data):.1f}%)")

# Content summary
primary_classes_count = defaultdict(int)
for cd in content_data:
    primary_classes_count[cd["primary_class"]] += 1
print("\nScene Content Classification:")
for cls, count in sorted(primary_classes_count.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {count} images ({100 * count / len(content_data):.1f}%)")

just_water_count = sum(1 for cd in content_data if cd["is_just_water"])
print(f"\nImages with 'just water' (no visible features): {just_water_count} "
      f"({100 * just_water_count / len(content_data):.1f}%)")


# =============================================================================
# PHASE 3: COLMAP SPARSE RECONSTRUCTION ANALYSIS
# =============================================================================
# COLMAP is the standard Structure-from-Motion (SfM) pipeline used to
# estimate camera poses and create a sparse 3D point cloud from images.
#
# COLMAP binary file formats:
#   cameras.bin  — Camera intrinsics (focal length, distortion, etc.)
#   images.bin   — Per-image poses (rotation quaternion + translation) + 2D-3D matches
#   points3D.bin — Sparse 3D point cloud with colors and reprojection errors
#
# Key Gaussian Splatting readiness metrics:
#   - Registration rate: % of images with estimated poses (100% ideal)
#   - Reprojection error: <0.5px good, >1px concerning
#   - Point count: more = better initialization
#   - Track length: avg # images per 3D point (longer = more reliable)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 3: COLMAP & Pose Data Analysis")
print("=" * 60)


def read_colmap_cameras_binary(path):
    """
    Parse COLMAP cameras.bin file.
    Format per camera: camera_id(u32), model_id(i32), width(u64), height(u64), params(f64[])
    """
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            model_names = {
                0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
                6: "FULL_OPENCV", 7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE",
                9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE"
            }
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
            n = num_params.get(model_id, 4)
            params = struct.unpack(f"<{n}d", f.read(8 * n))
            cameras[camera_id] = {
                "model": model_names.get(model_id, f"UNKNOWN_{model_id}"),
                "width": width, "height": height, "params": params
            }
    return cameras


def read_colmap_images_binary(path):
    """
    Parse COLMAP images.bin file.
    Format per image: image_id(u32), qvec(4xf64), tvec(3xf64), camera_id(u32),
                      name(null-terminated), num_points2D(u64), points2D(x,y,point3D_id)
    """
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            points2D = []
            for _ in range(num_points2D):
                x, y = struct.unpack("<2d", f.read(16))
                point3D_id = struct.unpack("<q", f.read(8))[0]
                points2D.append((x, y, point3D_id))
            images[image_id] = {
                "name": name, "qvec": (qw, qx, qy, qz), "tvec": (tx, ty, tz),
                "camera_id": camera_id, "num_points2D": num_points2D,
                "num_matched": sum(1 for p in points2D if p[2] != -1)
            }
    return images


def read_colmap_points3D_binary(path):
    """
    Parse COLMAP points3D.bin file.
    Format per point: point_id(u64), xyz(3xf64), rgb(3xu8), error(f64),
                      track_length(u64), track(image_id(u32), point2D_idx(u32))[]
    """
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            for _ in range(track_length):
                struct.unpack("<II", f.read(8))  # Skip track entries (save memory)
            points[point_id] = {
                "xyz": (x, y, z), "rgb": (r, g, b),
                "error": error, "track_length": track_length
            }
    return points


# Parse COLMAP data for each scene
colmap_data = {}
for scene in scenes:
    sp = scene_paths[scene]  # Use resolved path (handles nested layouts)
    sparse_dir = os.path.join(sp, "sparse", "0")
    cameras_path = os.path.join(sparse_dir, "cameras.bin")
    images_path = os.path.join(sparse_dir, "images.bin")
    points_path = os.path.join(sparse_dir, "points3D.bin")
    poses_path = os.path.join(sp, "poses_bounds.npy")

    scene_colmap = {}

    if os.path.exists(cameras_path):
        cameras = read_colmap_cameras_binary(cameras_path)
        scene_colmap["cameras"] = cameras
        print(f"\n{scene} - Cameras:")
        for cid, cam in cameras.items():
            print(f"  Camera {cid}: {cam['model']} {cam['width']}x{cam['height']}")

    if os.path.exists(images_path):
        images = read_colmap_images_binary(images_path)
        scene_colmap["images"] = images
        matched_counts = [img["num_matched"] for img in images.values()]
        total_2d = [img["num_points2D"] for img in images.values()]
        print(f"  Registered images: {len(images)}")
        print(f"  2D points per image: min={min(total_2d)}, max={max(total_2d)}, mean={np.mean(total_2d):.0f}")
        print(f"  Matched 3D points per image: min={min(matched_counts)}, max={max(matched_counts)}, mean={np.mean(matched_counts):.0f}")

    if os.path.exists(points_path):
        points = read_colmap_points3D_binary(points_path)
        scene_colmap["points3D"] = points
        errors = [p["error"] for p in points.values()]
        tracks = [p["track_length"] for p in points.values()]
        xyz = np.array([p["xyz"] for p in points.values()])
        print(f"  3D points: {len(points)}")
        print(f"  Reprojection error: min={min(errors):.3f}, max={max(errors):.3f}, mean={np.mean(errors):.3f}")
        print(f"  Track length: min={min(tracks)}, max={max(tracks)}, mean={np.mean(tracks):.1f}")

    if os.path.exists(poses_path):
        poses_bounds = np.load(poses_path)
        scene_colmap["poses_bounds"] = poses_bounds
        print(f"  Poses bounds shape: {poses_bounds.shape}")
        if poses_bounds.shape[1] == 17:
            near_bounds = poses_bounds[:, -2]
            far_bounds = poses_bounds[:, -1]
            print(f"  Near bounds: min={near_bounds.min():.3f}, max={near_bounds.max():.3f}")
            print(f"  Far bounds: min={far_bounds.min():.3f}, max={far_bounds.max():.3f}")

    colmap_data[scene] = scene_colmap


# =============================================================================
# PHASE 4: DATASET COMPLETENESS & GAUSSIAN SPLATTING READINESS
# =============================================================================
# Check whether each scene has everything needed for Gaussian Splatting:
#   - Enough images (20+ ideal, 15 minimum)
#   - 100% COLMAP registration
#   - Sufficient 3D points (5000+)
#   - Camera calibration + pose bounds
#   - Continuous image numbering
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 4: Dataset Completeness & Gaussian Splatting Readiness")
print("=" * 60)

completeness = {}
for scene in scenes:
    sc = colmap_data.get(scene, {})
    n_images = structure_data[scene]["num_images"]
    n_registered = len(sc.get("images", {}))
    n_points = len(sc.get("points3D", {}))
    has_poses = "poses_bounds" in sc
    has_cameras = "cameras" in sc

    registration_rate = n_registered / n_images * 100 if n_images > 0 else 0

    # Check for gaps in image numbering
    img_numbers = []
    for f in structure_data[scene]["images"]:
        basename = os.path.basename(f)
        num = ''.join(filter(str.isdigit, basename))
        if num:
            img_numbers.append(int(num))
    img_numbers.sort()

    gaps = []
    for i in range(len(img_numbers) - 1):
        if img_numbers[i + 1] - img_numbers[i] > 1:
            gaps.append((img_numbers[i], img_numbers[i + 1]))

    completeness[scene] = {
        "n_images": n_images,
        "n_registered": n_registered,
        "registration_rate": registration_rate,
        "n_3d_points": n_points,
        "has_poses_bounds": has_poses,
        "has_cameras": has_cameras,
        "image_number_range": (min(img_numbers), max(img_numbers)) if img_numbers else (0, 0),
        "numbering_gaps": gaps,
        "points_per_image": n_points / n_registered if n_registered > 0 else 0,
    }

    print(f"\n{scene}:")
    print(f"  Images: {n_images}, Registered: {n_registered} ({registration_rate:.0f}%)")
    if n_registered > 0:
        print(f"  3D points: {n_points}, Points/image: {n_points // n_registered}")
    print(f"  Poses bounds: {'Yes' if has_poses else 'MISSING'}")
    print(f"  Camera model: {'Yes' if has_cameras else 'MISSING'}")
    if gaps:
        print(f"  WARNING: Numbering gaps: {gaps}")
    else:
        print(f"  OK: No gaps in numbering")

# Readiness assessment
print("\n--- Gaussian Splatting Readiness ---")
for scene in scenes:
    sc = completeness[scene]
    issues = []
    if sc["registration_rate"] < 90:
        issues.append(f"Low registration ({sc['registration_rate']:.0f}%)")
    if sc["n_3d_points"] < 5000:
        issues.append(f"Few 3D points ({sc['n_3d_points']})")
    if not sc["has_poses_bounds"]:
        issues.append("Missing poses_bounds.npy")
    if not sc["has_cameras"]:
        issues.append("Missing camera calibration")
    if sc["n_images"] < 20:
        issues.append(f"Small image count ({sc['n_images']})")
    if sc["numbering_gaps"]:
        issues.append(f"Gaps in sequence: {sc['numbering_gaps']}")

    if not issues:
        print(f"  {scene}: READY")
    else:
        print(f"  {scene}: Issues — " + "; ".join(issues))


# =============================================================================
# PHASE 5: SERIALIZE DATA & PREPARE FOR REPORT
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 5: Preparing data for HTML report")
print("=" * 60)

colmap_summary = {}
for scene in scenes:
    sc = colmap_data.get(scene, {})
    summary = {}
    if "cameras" in sc:
        summary["cameras"] = {str(k): {
            "model": v["model"], "width": v["width"], "height": v["height"],
            "params": [float(p) for p in v["params"]]
        } for k, v in sc["cameras"].items()}
    if "images" in sc:
        matched = [img["num_matched"] for img in sc["images"].values()]
        total_2d = [img["num_points2D"] for img in sc["images"].values()]
        summary["n_registered"] = len(sc["images"])
        summary["matched_per_image"] = {
            "min": int(min(matched)), "max": int(max(matched)),
            "mean": float(np.mean(matched))
        }
        summary["points2d_per_image"] = {
            "min": int(min(total_2d)), "max": int(max(total_2d)),
            "mean": float(np.mean(total_2d))
        }
    if "points3D" in sc:
        errors = [p["error"] for p in sc["points3D"].values()]
        tracks = [p["track_length"] for p in sc["points3D"].values()]
        summary["n_points3D"] = len(sc["points3D"])
        summary["reproj_error"] = {
            "min": float(min(errors)), "max": float(max(errors)),
            "mean": float(np.mean(errors)), "median": float(np.median(errors))
        }
        summary["track_length"] = {
            "min": int(min(tracks)), "max": int(max(tracks)),
            "mean": float(np.mean(tracks))
        }
    if "poses_bounds" in sc:
        pb = sc["poses_bounds"]
        if pb.shape[1] == 17:
            summary["near_bound"] = {"min": float(pb[:, -2].min()), "max": float(pb[:, -2].max())}
            summary["far_bound"] = {"min": float(pb[:, -1].min()), "max": float(pb[:, -1].max())}
    colmap_summary[scene] = summary

# Create aliases for the report template
colmap = colmap_summary
structure = {scene: {
    "num_images": sd["num_images"],
    "total_size_bytes": sd["total_size_bytes"],
} for scene, sd in structure_data.items()}

# IMPORTANT: alias so the HTML template can use 'total_size'
total_size = total_size_bytes

# Save raw JSON data
all_data = {
    "scenes": scenes,
    "total_images": total_images,
    "total_size_bytes": total_size_bytes,
    "file_types": dict(file_types),
    "image_metadata": image_metadata,
    "color_data": color_data,
    "quality_data": quality_data,
    "content_data": [{k: v for k, v in cd.items()} for cd in content_data],
    "completeness": completeness,
    "structure_data": dict(structure),
    "colmap_summary": colmap_summary,
}
json_path = os.path.join(OUT_DIR, "eda_data.json")
with open(json_path, "w") as f:
    json.dump(all_data, f, indent=2, default=str)
print(f"Raw data saved to: {json_path}")


# =============================================================================
# PHASE 6: GENERATE HTML REPORT
# =============================================================================
# Self-contained HTML with Chart.js CDN, dark theme, 18 interactive charts.
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 6: Generating HTML Report")
print("=" * 60)

# --- Precompute chart data ---
color_class_counts = defaultdict(int)
for cd in color_data:
    color_class_counts[cd["color_class"]] += 1

blur_counts = defaultdict(int)
for qd in quality_data:
    blur_counts[qd["blur_class"]] += 1

content_counts = defaultdict(int)
for cd in content_data:
    content_counts[cd["primary_class"]] += 1

scene_content = {}
for scene in scenes:
    sc = defaultdict(int)
    for cd in content_data:
        if cd["scene"] == scene:
            sc[cd["primary_class"]] += 1
    scene_content[scene] = dict(sc)

scene_quality = {}
for scene in scenes:
    scene_quality[scene] = [qd["quality_score"] for qd in quality_data if qd["scene"] == scene]

scene_brightness = {}
for scene in scenes:
    scene_brightness[scene] = [cd["brightness"] for cd in color_data if cd["scene"] == scene]

scene_laplacian = {}
for scene in scenes:
    scene_laplacian[scene] = [qd["laplacian_var"] for qd in quality_data if qd["scene"] == scene]

scene_filesizes = {}
for scene in scenes:
    scene_filesizes[scene] = [m["file_size_bytes"] for m in image_metadata if m["scene"] == scene]

unique_dims = set((m["width"], m["height"]) for m in image_metadata)
scene_dims = {}
for scene in scenes:
    scene_dims[scene] = set((m["width"], m["height"]) for m in image_metadata if m["scene"] == scene)

all_filesizes = [m["file_size_bytes"] for m in image_metadata]
all_mp = [m["megapixels"] for m in image_metadata]

# =============================================================================
# BUILD THE HTML STRING
# =============================================================================
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Underwater Dataset EDA Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0a0e1a; --bg-card: #1a2332;
            --text-primary: #e2e8f0; --text-secondary: #94a3b8; --text-muted: #64748b;
            --accent-cyan: #06b6d4; --accent-green: #10b981; --accent-amber: #f59e0b;
            --accent-red: #ef4444; --accent-purple: #8b5cf6; --border: #1e293b;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Inter', -apple-system, sans-serif; background: var(--bg-primary); color: var(--text-primary); line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #0c1929, #1a2744, #0f2027); padding: 48px 40px; border-bottom: 1px solid var(--border); position: relative; overflow: hidden; }}
        .header::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(ellipse at 30% 50%, rgba(6,182,212,0.08), transparent 60%), radial-gradient(ellipse at 70% 50%, rgba(59,130,246,0.06), transparent 60%); }}
        .header-content {{ position: relative; z-index: 1; max-width: 1400px; margin: 0 auto; }}
        .header h1 {{ font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #e2e8f0, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }}
        .header .subtitle {{ color: var(--text-secondary); font-size: 1.05rem; margin-bottom: 24px; }}
        .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-top: 24px; }}
        .kpi-card {{ background: rgba(255,255,255,0.04); border: 1px solid var(--border); border-radius: 12px; padding: 20px; text-align: center; }}
        .kpi-value {{ font-size: 2rem; font-weight: 700; color: var(--accent-cyan); }}
        .kpi-label {{ font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 32px 40px; }}
        .section {{ margin-bottom: 40px; }}
        .section-title {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 6px; }}
        .section-desc {{ color: var(--text-muted); font-size: 0.9rem; margin-bottom: 20px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        .card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; }}
        .card-title {{ font-size: 1rem; font-weight: 600; margin-bottom: 16px; }}
        .chart-container {{ position: relative; width: 100%; height: 320px; }}
        .chart-container.tall {{ height: 400px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        th {{ text-align: left; padding: 12px 16px; background: rgba(255,255,255,0.03); border-bottom: 1px solid var(--border); color: var(--text-muted); font-weight: 500; text-transform: uppercase; font-size: 0.8rem; }}
        td {{ padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.03); color: var(--text-secondary); }}
        .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 500; }}
        .badge-green {{ background: rgba(16,185,129,0.15); color: #34d399; }}
        .badge-yellow {{ background: rgba(245,158,11,0.15); color: #fbbf24; }}
        .badge-red {{ background: rgba(239,68,68,0.15); color: #f87171; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
        .stat-item {{ background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }}
        .stat-item .label {{ font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; }}
        .stat-item .value {{ font-size: 1.3rem; font-weight: 600; margin-top: 2px; }}
        .alert {{ border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; font-size: 0.9rem; }}
        .alert-warning {{ background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2); color: #fbbf24; }}
        .alert-success {{ background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); color: #34d399; }}
        .footer {{ text-align: center; padding: 32px; color: var(--text-muted); font-size: 0.85rem; border-top: 1px solid var(--border); }}
        @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} .kpi-row {{ grid-template-columns: repeat(2, 1fr); }} .container {{ padding: 20px; }} }}
    </style>
</head>
<body>

<div class="header"><div class="header-content">
    <h1>Underwater Dataset &mdash; Exploratory Data Analysis</h1>
    <div class="subtitle">Comprehensive analysis for 3D reconstruction (NeRF / Gaussian Splatting) &bull; {total_images} images across {len(scenes)} scenes</div>
    <div class="kpi-row">
        <div class="kpi-card"><div class="kpi-value">{total_images}</div><div class="kpi-label">Total Images</div></div>
        <div class="kpi-card"><div class="kpi-value">{len(scenes)}</div><div class="kpi-label">Scenes</div></div>
        <div class="kpi-card"><div class="kpi-value">{total_size_bytes/1024/1024:.0f} MB</div><div class="kpi-label">Dataset Size</div></div>
        <div class="kpi-card"><div class="kpi-value">{sum(colmap[s].get('n_points3D',0) for s in scenes):,}</div><div class="kpi-label">3D Points</div></div>
        <div class="kpi-card"><div class="kpi-value">{just_water_count}</div><div class="kpi-label">Empty/Water-Only</div></div>
    </div>
</div></div>

<div class="container">

<div class="section">
    <div class="section-title">Dataset Structure</div>
    <div class="section-desc">{len(scenes)} scenes detected. Each scene analyzed for images, COLMAP data, and pose information.</div>
    <div class="card" style="margin-bottom:24px;overflow-x:auto;"><table><thead><tr><th>Scene</th><th>Images</th><th>Resolution</th><th>Size</th><th>3D Points</th><th>Camera</th><th>Status</th></tr></thead><tbody>"""

for scene in scenes:
    n_img = structure[scene]["num_images"]
    sz = structure[scene]["total_size_bytes"]
    dims = list(scene_dims[scene])
    dim_str = f"{dims[0][0]}&times;{dims[0][1]}" if dims else "N/A"
    n_pts = colmap[scene].get("n_points3D", 0)
    cam = list(colmap[scene].get("cameras", {}).values())
    cam_model = cam[0]["model"] if cam else "N/A"
    gaps = completeness[scene]["numbering_gaps"]
    status = '<span class="badge badge-green">Ready</span>' if not gaps else '<span class="badge badge-yellow">Gaps</span>'
    html += f"<tr><td><strong>{scene}</strong></td><td>{n_img}</td><td>{dim_str}</td><td>{sz/1024/1024:.1f} MB</td><td>{n_pts:,}</td><td>{cam_model}</td><td>{status}</td></tr>"

html += """</tbody></table></div>
    <div class="grid-2">
        <div class="card"><div class="card-title">Images per Scene</div><div class="chart-container"><canvas id="c1"></canvas></div></div>
        <div class="card"><div class="card-title">Storage per Scene (MB)</div><div class="chart-container"><canvas id="c2"></canvas></div></div>
    </div>
</div>

<div class="section">
    <div class="section-title">Image Properties</div>"""

html += f"""
    <div class="stat-grid" style="margin-bottom:24px;">
        <div class="stat-item"><div class="label">Format</div><div class="value">{'PNG' if '.png' in file_types else 'Mixed'} ({max(file_types.values())}/{total_images})</div></div>
        <div class="stat-item"><div class="label">Resolutions</div><div class="value">{len(unique_dims)} variant{'s' if len(unique_dims)>1 else ''}</div></div>
        <div class="stat-item"><div class="label">Megapixels</div><div class="value">{min(all_mp)}&ndash;{max(all_mp)} MP</div></div>
        <div class="stat-item"><div class="label">Avg File Size</div><div class="value">{np.mean(all_filesizes)/1024:.0f} KB</div></div>
    </div>
    <div class="grid-2">
        <div class="card"><div class="card-title">File Size by Scene</div><div class="chart-container"><canvas id="c3"></canvas></div></div>
        <div class="card"><div class="card-title">Resolution Distribution</div><div class="chart-container"><canvas id="c4"></canvas></div></div>
    </div>
</div>

<div class="section">
    <div class="section-title">Color Profile Analysis</div>
    <div class="section-desc">Underwater color cast analysis. Blue/red ratio &gt;1 indicates typical underwater blue dominance.</div>
    <div class="grid-2">
        <div class="card"><div class="card-title">Color Classification</div><div class="chart-container tall"><canvas id="c5"></canvas></div></div>
        <div class="card"><div class="card-title">Brightness per Scene</div><div class="chart-container tall"><canvas id="c6"></canvas></div></div>
    </div>
    <div class="grid-2" style="margin-top:24px;">
        <div class="card"><div class="card-title">Mean RGB per Scene</div><div class="chart-container"><canvas id="c7"></canvas></div></div>
        <div class="card"><div class="card-title">Blue/Red Ratio</div><div class="chart-container"><canvas id="c8"></canvas></div></div>
    </div>
</div>

<div class="section">
    <div class="section-title">Image Quality</div>
    <div class="grid-2">
        <div class="card"><div class="card-title">Blur Classification</div><div class="chart-container"><canvas id="c9"></canvas></div></div>
        <div class="card"><div class="card-title">Quality Score per Scene</div><div class="chart-container"><canvas id="c10"></canvas></div></div>
    </div>
    <div class="grid-2" style="margin-top:24px;">
        <div class="card"><div class="card-title">Sharpness (Laplacian Variance)</div><div class="chart-container"><canvas id="c11"></canvas></div></div>
        <div class="card"><div class="card-title">Edge Density</div><div class="chart-container"><canvas id="c12"></canvas></div></div>
    </div>
</div>

<div class="section">
    <div class="section-title">Scene Content</div>
    <div class="section-desc">Heuristic classification based on edge density and color variance. Note: these labels are approximate.</div>
    <div class="grid-2">
        <div class="card"><div class="card-title">Content Distribution</div><div class="chart-container"><canvas id="c13"></canvas></div></div>
        <div class="card"><div class="card-title">Content per Scene</div><div class="chart-container"><canvas id="c14"></canvas></div></div>
    </div>
</div>

<div class="section">
    <div class="section-title">COLMAP Reconstruction</div>
    <div class="card" style="margin-bottom:24px;overflow-x:auto;"><table><thead><tr><th>Scene</th><th>Registered</th><th>3D Points</th><th>Reproj. Error</th><th>Track Length</th></tr></thead><tbody>"""

for scene in scenes:
    sc = colmap[scene]
    n_reg = sc.get("n_registered", 0)
    n_pts = sc.get("n_points3D", 0)
    re = sc.get("reproj_error", {})
    tl = sc.get("track_length", {})
    badge = "badge-green" if re.get("mean", 999) < 0.5 else "badge-yellow"
    html += f'<tr><td><strong>{scene}</strong></td><td>{n_reg}/{structure[scene]["num_images"]}</td><td>{n_pts:,}</td><td><span class="badge {badge}">{re.get("mean",0):.3f} px</span></td><td>{tl.get("mean",0):.1f}</td></tr>'

html += """</tbody></table></div>
    <div class="grid-2">
        <div class="card"><div class="card-title">3D Points per Scene</div><div class="chart-container"><canvas id="c15"></canvas></div></div>
        <div class="card"><div class="card-title">Reprojection Error</div><div class="chart-container"><canvas id="c16"></canvas></div></div>
    </div>
    <div class="grid-2" style="margin-top:24px;">
        <div class="card"><div class="card-title">Matched Points per Image</div><div class="chart-container"><canvas id="c17"></canvas></div></div>
        <div class="card"><div class="card-title">Depth Range</div><div class="chart-container"><canvas id="c18"></canvas></div></div>
    </div>
</div>

<div class="section">
    <div class="section-title">Readiness Assessment</div>"""

for scene in scenes:
    c = completeness[scene]
    sc_c = colmap.get(scene, {})
    issues = []
    if c["numbering_gaps"]:
        issues.append(f"Image numbering gaps: {c['numbering_gaps']}")
    if c["n_images"] < 20:
        issues.append(f"Small image count ({c['n_images']})")
    re_mean = sc_c.get("reproj_error", {}).get("mean", 0)
    if re_mean > 0.5:
        issues.append(f"Elevated reproj. error ({re_mean:.3f} px)")
    if not issues:
        html += f'<div class="alert alert-success"><strong>{scene}</strong> &mdash; Ready. {c["n_registered"]}/{c["n_images"]} registered, {c["n_3d_points"]:,} 3D points.</div>'
    else:
        html += f'<div class="alert alert-warning"><strong>{scene}</strong> &mdash; '
        html += "; ".join(issues)
        html += f'. {c["n_registered"]}/{c["n_images"]} registered, {c["n_3d_points"]:,} 3D points.</div>'

html += """
</div>
</div>

<div class="footer">Underwater Dataset EDA &bull; Generated automatically (Optimized single-pass analysis)</div>

<script>
Chart.defaults.color='#94a3b8';Chart.defaults.borderColor='rgba(255,255,255,0.06)';
const S=""" + json.dumps(scenes) + """;
const SC=['#3b82f6','#e91e63','#4caf50','#ff9800','#8b5cf6','#06b6d4','#ef4444','#f59e0b'];
const SB=['rgba(59,130,246,0.7)','rgba(233,30,99,0.7)','rgba(76,175,80,0.7)','rgba(255,152,0,0.7)','rgba(139,92,246,0.7)','rgba(6,182,212,0.7)','rgba(239,68,68,0.7)','rgba(245,158,11,0.7)'];
const bar=(id,l,d,o={})=>new Chart(document.getElementById(id),{type:'bar',data:{labels:l,datasets:d},options:{responsive:true,maintainAspectRatio:false,...o}});
const don=(id,l,d,c)=>new Chart(document.getElementById(id),{type:'doughnut',data:{labels:l,datasets:[{data:d,backgroundColor:c,borderWidth:2,borderColor:'#1a2332'}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom'}}}});

bar('c1',S,[{data:""" + json.dumps([structure[s]["num_images"] for s in scenes]) + """,backgroundColor:SB,borderColor:SC,borderWidth:1,borderRadius:6}],{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}});
bar('c2',S,[{data:""" + json.dumps([round(structure[s]["total_size_bytes"]/1024/1024,1) for s in scenes]) + """,backgroundColor:SB,borderColor:SC,borderWidth:1,borderRadius:6}],{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}});
bar('c3',S,[{label:'Min KB',data:""" + json.dumps([round(min(scene_filesizes[s])/1024) for s in scenes]) + """,backgroundColor:'rgba(59,130,246,0.5)',borderRadius:4},{label:'Avg KB',data:""" + json.dumps([round(np.mean(scene_filesizes[s])/1024) for s in scenes]) + """,backgroundColor:'rgba(6,182,212,0.5)',borderRadius:4},{label:'Max KB',data:""" + json.dumps([round(max(scene_filesizes[s])/1024) for s in scenes]) + """,backgroundColor:'rgba(139,92,246,0.5)',borderRadius:4}],{scales:{y:{beginAtZero:true}}});
"""

# Resolution doughnut
res_counts = defaultdict(int)
for m in image_metadata:
    res_counts[f"{m['width']}x{m['height']}"] += 1
html += "don('c4'," + json.dumps(list(res_counts.keys())) + "," + json.dumps(list(res_counts.values())) + ",['rgba(59,130,246,0.7)','rgba(6,182,212,0.7)','rgba(76,175,80,0.7)','rgba(255,152,0,0.7)']);\n"

# Color classification
color_labels = list(color_class_counts.keys())
color_values = list(color_class_counts.values())
color_map = {"Very Dark / Deep Water":"#1a237e","Dark Blue":"#1565c0","Medium Blue":"#42a5f5","Bright Blue":"#90caf9","Dark Grey/Murky":"#616161","Neutral/Grey":"#9e9e9e","Blue-Green/Teal":"#00897b","Dark Green/Teal":"#2e7d32","Green/Algae":"#66bb6a","Bright Green":"#a5d6a7","Bright/Well-lit":"#fff176"}
color_chart_colors = [color_map.get(c, "#888") for c in color_labels]
html += "new Chart(document.getElementById('c5'),{type:'bar',data:{labels:" + json.dumps(color_labels) + ",datasets:[{data:" + json.dumps(color_values) + ",backgroundColor:" + json.dumps(color_chart_colors) + ",borderRadius:6}]},options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{beginAtZero:true}}}});\n"

# Brightness per scene
brightness_stats = {}
for scene in scenes:
    bv = scene_brightness[scene]
    brightness_stats[scene] = {"min":float(np.min(bv)),"q1":float(np.percentile(bv,25)),"med":float(np.median(bv)),"q3":float(np.percentile(bv,75)),"max":float(np.max(bv))}
html += "bar('c6',S,[{label:'Min',data:" + json.dumps([brightness_stats[s]["min"] for s in scenes]) + ",backgroundColor:'rgba(30,41,59,0.8)',borderRadius:4},{label:'Q1',data:" + json.dumps([brightness_stats[s]["q1"]-brightness_stats[s]["min"] for s in scenes]) + ",backgroundColor:'rgba(59,130,246,0.3)',borderRadius:4},{label:'Median',data:" + json.dumps([brightness_stats[s]["med"]-brightness_stats[s]["q1"] for s in scenes]) + ",backgroundColor:'rgba(59,130,246,0.6)',borderRadius:4},{label:'Q3',data:" + json.dumps([brightness_stats[s]["q3"]-brightness_stats[s]["med"] for s in scenes]) + ",backgroundColor:'rgba(59,130,246,0.3)',borderRadius:4},{label:'Max',data:" + json.dumps([brightness_stats[s]["max"]-brightness_stats[s]["q3"] for s in scenes]) + ",backgroundColor:'rgba(30,41,59,0.8)',borderRadius:4}],{scales:{x:{stacked:true},y:{stacked:true,beginAtZero:true,max:180}}});\n"

# Mean RGB
rgb_stats = {}
for scene in scenes:
    sc = [cd for cd in color_data if cd["scene"] == scene]
    rgb_stats[scene] = {"r":float(np.mean([c["mean_r"] for c in sc])),"g":float(np.mean([c["mean_g"] for c in sc])),"b":float(np.mean([c["mean_b"] for c in sc]))}
html += "bar('c7',S,[{label:'Red',data:" + json.dumps([round(rgb_stats[s]["r"],1) for s in scenes]) + ",backgroundColor:'rgba(239,68,68,0.6)',borderRadius:4},{label:'Green',data:" + json.dumps([round(rgb_stats[s]["g"],1) for s in scenes]) + ",backgroundColor:'rgba(16,185,129,0.6)',borderRadius:4},{label:'Blue',data:" + json.dumps([round(rgb_stats[s]["b"],1) for s in scenes]) + ",backgroundColor:'rgba(59,130,246,0.6)',borderRadius:4}],{scales:{y:{beginAtZero:true,max:180}}});\n"

# Blue/Red ratio
br_stats = {}
for scene in scenes:
    sc = [cd for cd in color_data if cd["scene"] == scene]
    ratios = [c["blue_ratio"] for c in sc]
    br_stats[scene] = {"min":float(np.min(ratios)),"mean":float(np.mean(ratios)),"max":float(np.max(ratios))}
html += "bar('c8',S,[{label:'Min',data:" + json.dumps([round(br_stats[s]["min"],2) for s in scenes]) + ",backgroundColor:'rgba(59,130,246,0.3)',borderRadius:4},{label:'Mean',data:" + json.dumps([round(br_stats[s]["mean"],2) for s in scenes]) + ",backgroundColor:'rgba(59,130,246,0.6)',borderRadius:4},{label:'Max',data:" + json.dumps([round(br_stats[s]["max"],2) for s in scenes]) + ",backgroundColor:'rgba(59,130,246,0.9)',borderRadius:4}],{scales:{y:{beginAtZero:true}}});\n"

# Blur doughnut
blur_labels = ["Sharp","Acceptable","Somewhat Blurry","Very Blurry"]
blur_values = [blur_counts.get(l,0) for l in blur_labels]
html += "don('c9'," + json.dumps(blur_labels) + "," + json.dumps(blur_values) + ",['#10b981','#3b82f6','#f59e0b','#ef4444']);\n"

# Quality score per scene
quality_stats = {}
for scene in scenes:
    q = scene_quality[scene]
    quality_stats[scene] = {"min":float(np.min(q)),"mean":float(np.mean(q)),"max":float(np.max(q))}
html += "bar('c10',S,[{label:'Min',data:" + json.dumps([round(quality_stats[s]["min"],1) for s in scenes]) + ",backgroundColor:SB,borderRadius:4},{label:'Mean',data:" + json.dumps([round(quality_stats[s]["mean"],1) for s in scenes]) + ",backgroundColor:SB.map(c=>c.replace('0.7','0.4')),borderRadius:4},{label:'Max',data:" + json.dumps([round(quality_stats[s]["max"],1) for s in scenes]) + ",backgroundColor:SB.map(c=>c.replace('0.7','0.9')),borderRadius:4}],{scales:{y:{beginAtZero:true,max:110}}});\n"

# Laplacian per scene
html += "bar('c11',S,[{label:'Min',data:" + json.dumps([round(min(scene_laplacian[s]),1) for s in scenes]) + ",backgroundColor:'rgba(239,68,68,0.4)',borderRadius:4},{label:'Mean',data:" + json.dumps([round(float(np.mean(scene_laplacian[s])),1) for s in scenes]) + ",backgroundColor:'rgba(245,158,11,0.5)',borderRadius:4},{label:'Max',data:" + json.dumps([round(max(scene_laplacian[s]),1) for s in scenes]) + ",backgroundColor:'rgba(16,185,129,0.5)',borderRadius:4}],{scales:{y:{beginAtZero:true}}});\n"

# Edge density per scene
edge_stats = {}
for scene in scenes:
    ed = [qd["edge_density"] for qd in quality_data if qd["scene"] == scene]
    edge_stats[scene] = {"min":float(np.min(ed)),"mean":float(np.mean(ed)),"max":float(np.max(ed))}
html += "bar('c12',S,[{label:'Min',data:" + json.dumps([round(edge_stats[s]["min"]*100,2) for s in scenes]) + ",backgroundColor:'rgba(139,92,246,0.3)',borderRadius:4},{label:'Mean',data:" + json.dumps([round(edge_stats[s]["mean"]*100,2) for s in scenes]) + ",backgroundColor:'rgba(139,92,246,0.6)',borderRadius:4},{label:'Max',data:" + json.dumps([round(edge_stats[s]["max"]*100,2) for s in scenes]) + ",backgroundColor:'rgba(139,92,246,0.9)',borderRadius:4}],{scales:{y:{beginAtZero:true}}});\n"

# Content doughnut
content_labels = list(content_counts.keys())
content_values = list(content_counts.values())
content_colors = ["rgba(16,185,129,0.7)","rgba(245,158,11,0.7)","rgba(59,130,246,0.7)","rgba(239,68,68,0.7)","rgba(139,92,246,0.7)","rgba(6,182,212,0.7)"]
html += "don('c13'," + json.dumps(content_labels) + "," + json.dumps(content_values) + "," + json.dumps(content_colors[:len(content_labels)]) + ");\n"

# Content per scene stacked
all_ct = sorted(set(c["primary_class"] for c in content_data))
ct_ds = []
for i, ct in enumerate(all_ct):
    ct_ds.append({"label":ct,"data":[scene_content[s].get(ct,0) for s in scenes],"backgroundColor":content_colors[i%len(content_colors)]})
html += "bar('c14',S," + json.dumps(ct_ds) + ",{scales:{x:{stacked:true},y:{stacked:true,beginAtZero:true}}});\n"

# 3D points per scene
html += "bar('c15',S,[{data:" + json.dumps([colmap[s].get("n_points3D",0) for s in scenes]) + ",backgroundColor:SB,borderColor:SC,borderWidth:1,borderRadius:6}],{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}});\n"

# Reprojection error
html += "bar('c16',S,[{label:'Mean',data:" + json.dumps([round(colmap[s].get("reproj_error",{}).get("mean",0),3) for s in scenes]) + ",backgroundColor:'rgba(245,158,11,0.6)',borderRadius:4},{label:'Median',data:" + json.dumps([round(colmap[s].get("reproj_error",{}).get("median",0),3) for s in scenes]) + ",backgroundColor:'rgba(16,185,129,0.6)',borderRadius:4}],{scales:{y:{beginAtZero:true}}});\n"

# Matched points per image
matched_stats = {s: colmap[s].get("matched_per_image",{"min":0,"mean":0,"max":0}) for s in scenes}
html += "bar('c17',S,[{label:'Min',data:" + json.dumps([matched_stats[s]["min"] for s in scenes]) + ",backgroundColor:'rgba(239,68,68,0.4)',borderRadius:4},{label:'Mean',data:" + json.dumps([round(matched_stats[s]["mean"]) for s in scenes]) + ",backgroundColor:'rgba(6,182,212,0.6)',borderRadius:4},{label:'Max',data:" + json.dumps([matched_stats[s]["max"] for s in scenes]) + ",backgroundColor:'rgba(16,185,129,0.6)',borderRadius:4}],{scales:{y:{beginAtZero:true}}});\n"

# Depth range
depth_data = []
for scene in scenes:
    sc = colmap[scene]
    nb = sc.get("near_bound",{"min":0,"max":0})
    fb = sc.get("far_bound",{"min":0,"max":0})
    depth_data.append({"nm":nb["min"],"nx":nb["max"],"fm":fb["min"],"fx":fb["max"]})
html += "bar('c18',S,[{label:'Near min',data:" + json.dumps([round(d["nm"],2) for d in depth_data]) + ",backgroundColor:'rgba(59,130,246,0.5)',borderRadius:4},{label:'Near max',data:" + json.dumps([round(d["nx"],2) for d in depth_data]) + ",backgroundColor:'rgba(59,130,246,0.8)',borderRadius:4},{label:'Far min',data:" + json.dumps([round(d["fm"],2) for d in depth_data]) + ",backgroundColor:'rgba(139,92,246,0.5)',borderRadius:4},{label:'Far max',data:" + json.dumps([round(d["fx"],2) for d in depth_data]) + ",backgroundColor:'rgba(139,92,246,0.8)',borderRadius:4}],{scales:{y:{beginAtZero:true}}});\n"

html += "</script></body></html>"

# --- WRITE THE REPORT ---
with open(REPORT_PATH, "w") as f:
    f.write(html)

print(f"\nReport saved to: {REPORT_PATH}")
print(f"Report size: {os.path.getsize(REPORT_PATH)/1024:.0f} KB")
print("\nDone! Open the HTML file in any browser to view the interactive report.")
