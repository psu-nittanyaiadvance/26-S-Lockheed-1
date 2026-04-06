#!/usr/bin/env python3
"""
Detect and crop platform/camera artifacts from underwater image datasets.

This script targets persistent border structures that appear in most frames
(camera housing, vehicle body, mounts, or overlays). It detects these regions
from cross-image statistics and edge profiles, then supports review and batch
cropping through explicit subcommands.

Detection strategy:
    1. Build a median composite from sampled frames.
    2. Analyze edge-adjacent brightness profiles.
    3. Detect strong inward transitions with gradient cues.
    4. Validate candidates with cross-image consistency checks.

Subcommands:
    detect   - Analyze sampled images and write diagnostic outputs.
    preview  - Show before/after crop previews (auto or manual crop values).
    apply    - Apply crop settings to all images and export results.

Usage:
    python vehicle_artifact_crop.py detect --datasets dataset_name
    python vehicle_artifact_crop.py preview --datasets dataset_name --crop bottom=80
    python vehicle_artifact_crop.py apply --datasets dataset_name --output-dir cropped_datasets
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading of partially corrupted images

# =============================================================================
# Configuration
# =============================================================================

# Number of images sampled per dataset to build the median composite.
# 50 is enough to wash out scene content while preserving static artifacts.
ARTIFACT_SAMPLE_SIZE = 50

# Fraction of image dimensions to scan inward from each edge when looking
# for artifact boundaries. 0.25 = search the outer 25% of each side.
ARTIFACT_SCAN_FRACTION = 0.25

DEFAULT_OUTPUT_DIR = "cropped_datasets"   # where `apply` writes cropped images
ANALYSIS_DIR = "artifact_analysis"        # where `detect` writes diagnostic plots

# Analysis resolution — all sampled images are resized to this before processing.
# Large enough to detect fine artifact boundaries, small enough to stack ~50 in RAM.
ANALYSIS_SIZE = (640, 480)  # (width, height)

# =============================================================================
# Dataset Configurations
# =============================================================================

DATASET_CONFIGS = {
    "sunboat": {
        "name": "sunboat",
        "root": "/Users/ethanknox/Desktop/Advance Spring 26 (Lockheed Martin)/Sunboat_03-09-2023",
        "extensions": {".png"},
        "image_subdir": "camera",
        "description": "Sunboat mission recordings",
    },
    "shipwreck": {
        "name": "shipwreck",
        "root": "/Users/ethanknox/Desktop/Advance Spring 26 (Lockheed Martin)/Shipwreck",
        "extensions": {".jpg", ".jpeg"},
        "image_subdir": "camera",
        "description": "Shipwreck survey recordings",
    },
    "flsea_vi": {
        "name": "flsea_vi",
        "root": "/Users/ethanknox/Desktop/Advance Spring 26 (Lockheed Martin)/flsea-vi",
        "extensions": {".tiff", ".tif"},
        "image_subdir": "imgs",
        "description": "FLSEA-VI underwater scenes (canyons + Red Sea)",
    },
}


# =============================================================================
# Image Discovery
# =============================================================================

def iter_dataset_images(
    root: str,
    extensions: Set[str],
    image_subdir: str,
) -> Generator[Tuple[str, str, str], None, None]:
    """
    Recursively walk the dataset root, yielding images inside subdirectories
    named `image_subdir` that match the given file extensions.

    Yields (absolute_path, relative_path, location_name) tuples sorted by
    relative path for deterministic ordering.
    """
    root = os.path.abspath(root)
    all_images = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Only look inside directories named `image_subdir` (e.g. "camera", "imgs")
        if os.path.basename(dirpath) != image_subdir:
            continue
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, root)
                # Derive a location label from the parent directories above image_subdir.
                # e.g. "mission01/dive03" for deeply nested datasets, or "root" if flat.
                parts = rel_path.split(os.sep)
                if len(parts) > 2:
                    location = os.sep.join(parts[:-2])
                else:
                    location = "root"
                all_images.append((abs_path, rel_path, location))

    # Sort by relative path for reproducible ordering across runs
    all_images.sort(key=lambda x: x[1])
    for item in all_images:
        yield item


# =============================================================================
# Sampling & Median Image
# =============================================================================

def sample_and_load_images(
    root: str,
    extensions: Set[str],
    image_subdir: str,
    sample_size: int = ARTIFACT_SAMPLE_SIZE,
    target_size: Tuple[int, int] = ANALYSIS_SIZE,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[str]]:
    """
    Evenly sample N images from the dataset, resize to common dims, return as
    a stacked numpy array.

    Returns:
        stack: float32 array [N, H, W, 3] of RGB images
        original_sizes: list of (width, height) tuples for each loaded image
        paths: list of absolute paths for each loaded image
    """
    # Collect all image paths first so we can evenly sample across the dataset
    all_paths = list(iter_dataset_images(root, extensions, image_subdir))
    if not all_paths:
        return np.array([]), [], []

    # Evenly space samples across the full dataset (avoids clustering in one dive/location)
    step = max(1, len(all_paths) // sample_size)
    sampled = all_paths[::step][:sample_size]

    arrays = []
    original_sizes = []
    paths = []

    for abs_path, _, _ in sampled:
        try:
            img = Image.open(abs_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Track original size so we can scale crop coordinates back later
            original_sizes.append(img.size)
            # Resize to common analysis resolution for pixel-wise median computation
            img_resized = img.resize(target_size, Image.LANCZOS)
            arrays.append(np.array(img_resized))
            paths.append(abs_path)
        except Exception:
            # Skip corrupt or unreadable images silently
            continue

    # Require at least 5 images for a meaningful median (too few = noisy composite)
    if len(arrays) < 5:
        return np.array([]), original_sizes, paths

    # Stack into [N, H, W, 3] float32 array for numerical operations
    stack = np.stack(arrays, axis=0).astype(np.float32)
    return stack, original_sizes, paths


def compute_median_image(stack: np.ndarray) -> np.ndarray:
    """Compute pixel-wise median across sampled images. Returns [H, W, 3] uint8."""
    median = np.median(stack, axis=0)
    return median.astype(np.uint8)


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array to grayscale using ITU-R BT.601 weights."""
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# =============================================================================
# Artifact Detection — Multi-Signal Edge Scanning
# =============================================================================

def detect_artifact_edges(
    stack: np.ndarray,
    median_rgb: np.ndarray,
    scan_fraction: float = ARTIFACT_SCAN_FRACTION,
) -> Dict[str, int]:
    """
    Detect vehicle/camera housing artifacts from each edge using three signals:
      1. Brightness anomaly — rows/columns near edge much brighter/darker than scene
      2. Gradient boundary — sharpest brightness transition marks housing-scene boundary
      3. Cross-image consistency — low variance confirms static artifact

    Args:
        stack: float32 [N, H, W, 3] image stack
        median_rgb: uint8 [H, W, 3] median composite
        scan_fraction: fraction of image to scan from each edge

    Returns:
        Dict with keys "top", "bottom", "left", "right" — pixels to crop at
        analysis resolution (needs scaling to original resolution).
    """
    h, w = median_rgb.shape[:2]
    gray_median = to_grayscale(median_rgb.astype(np.float32))

    # Convert each sampled image to grayscale for cross-image variance analysis.
    # Artifacts (housing, instrument overlays) are static across frames, so their
    # pixel variance will be LOW. Scene content changes frame-to-frame → HIGH variance.
    gray_stack = to_grayscale(stack)  # [N, H, W]
    pixel_std = gray_stack.std(axis=0)  # [H, W] — per-pixel std across all samples

    # Use the center 50% of the image as a reference for "normal" scene statistics.
    # This avoids contamination from edge artifacts when computing baseline values.
    center_slice = gray_median[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    center_brightness = float(np.mean(center_slice))     # typical scene brightness
    center_std_val = float(np.std(center_slice))          # spatial brightness spread
    center_cross_std = float(pixel_std[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean())  # typical cross-image variation

    # Scan each of the four edges independently
    crop = {}
    for edge in ("top", "bottom", "left", "right"):
        # Vertical edges (top/bottom) scan rows; horizontal edges (left/right) scan columns
        is_vertical = edge in ("top", "bottom")
        scan_size = max(1, int((h if is_vertical else w) * scan_fraction))
        crop[edge] = _detect_single_edge(
            gray_median, pixel_std, edge, scan_size,
            center_brightness, center_std_val, center_cross_std,
        )

    return crop


def _detect_single_edge(
    gray_median: np.ndarray,
    pixel_std: np.ndarray,
    edge: str,
    scan_size: int,
    center_brightness: float,
    center_std_val: float,
    center_cross_std: float,
) -> int:
    """
    Scan inward from one edge to detect artifact boundary using a
    gradient-first approach.

    Key insight: vehicle housing artifacts have SHARP boundaries (step-function
    in brightness), while lens vignetting is a smooth ramp. By requiring a
    strong gradient as the primary gate, we avoid false positives from vignetting.

    Strategy:
      1. Build brightness + cross-image-std profiles from edge inward
      2. Find the strongest gradient peak in the near-edge region
      3. Verify the edge-side of the peak is anomalous (brightness or consistency)
      4. Walk past the gradient peak until brightness settles to scene level

    Returns number of rows/columns to crop from this edge (at analysis resolution).
    """
    h, w = gray_median.shape

    # Build 1D brightness profile and cross-image-std profile along the scan axis.
    # Index 0 = the outermost row/column (at the edge), increasing index = moving inward.
    # For top/bottom we average across columns (row means); for left/right we average
    # across rows (column means). This reduces 2D data to a 1D edge→center profile.
    if edge == "top":
        brightness = np.array([gray_median[i, :].mean() for i in range(scan_size)])
        cross_std = np.array([pixel_std[i, :].mean() for i in range(scan_size)])
    elif edge == "bottom":
        # Scan from the bottom edge inward (h-1, h-2, ...) so index 0 = bottom row
        brightness = np.array([gray_median[h - 1 - i, :].mean() for i in range(scan_size)])
        cross_std = np.array([pixel_std[h - 1 - i, :].mean() for i in range(scan_size)])
    elif edge == "left":
        brightness = np.array([gray_median[:, i].mean() for i in range(scan_size)])
        cross_std = np.array([pixel_std[:, i].mean() for i in range(scan_size)])
    else:  # right — scan from right edge inward
        brightness = np.array([gray_median[:, w - 1 - i].mean() for i in range(scan_size)])
        cross_std = np.array([pixel_std[:, w - 1 - i].mean() for i in range(scan_size)])

    if len(brightness) < 10:
        return 0

    # --- Signal 1: Gradient profile (primary gating signal) ---
    # Take the absolute first derivative of brightness along the edge→center axis.
    # Vehicle housing creates a sharp brightness step (large |gradient|);
    # lens vignetting produces a smooth ramp (small, uniform |gradient|).
    gradient = np.abs(np.diff(brightness))

    if len(gradient) == 0:
        return 0

    # Only search the first 15% of the image for the housing boundary.
    # Housing artifacts sit right at the frame edge, not deep inside the scene.
    dim = h if edge in ("top", "bottom") else w
    max_boundary_search = max(10, int(dim * 0.15))
    search_region = gradient[:max_boundary_search]

    # Median gradient serves as a baseline for "normal" brightness change rate.
    # We compare the peak against this to distinguish housing edges from vignetting.
    median_grad = float(np.median(gradient))
    if median_grad <= 0:
        median_grad = 0.1  # avoid division by zero for flat images

    # Find the single strongest gradient spike in the near-edge search window
    peak_idx = int(np.argmax(search_region))
    peak_val = float(search_region[peak_idx])

    # --- Gate: require a genuinely sharp transition ---
    # Both thresholds must be exceeded to trigger detection:
    #   - Relative: peak must be 5x the median gradient (rules out gentle vignetting)
    #   - Absolute: peak must be >= 2.0 intensity/pixel (rules out noise in dark images)
    # Typical values: vignetting gradient ~0.2-0.5/px, housing boundary ~5-30+/px.
    min_gradient_ratio = 5.0
    min_absolute_gradient = 2.0

    if peak_val < median_grad * min_gradient_ratio or peak_val < min_absolute_gradient:
        return 0  # No sharp transition found → no artifact on this edge

    # --- Signal 2: Verify edge-side content is anomalous ---
    # A sharp gradient alone isn't proof of an artifact (could be a scene feature).
    # We also require the edge-side region to be abnormal in at least one way:
    #   (a) Brightness anomaly: significantly brighter/darker than the scene center
    #   (b) Static content: very low cross-image variance (same pixels every frame)
    edge_region_end = max(1, peak_idx + 1)
    edge_brightness = float(np.mean(brightness[:edge_region_end]))  # avg brightness edge→peak
    edge_cross_std = float(np.mean(cross_std[:edge_region_end]))    # avg frame-to-frame variation

    # How different is edge brightness from the scene center?
    brightness_diff = abs(edge_brightness - center_brightness)
    # Threshold: at least 15 intensity levels, or 1.2x the center's spatial std
    brightness_threshold = max(15.0, center_std_val * 1.2)
    # Static threshold: edge variance must be < 40% of center variance to count as "static"
    consistency_threshold = center_cross_std * 0.4

    is_brightness_anomaly = brightness_diff > brightness_threshold
    is_static = edge_cross_std < consistency_threshold

    if not (is_brightness_anomaly or is_static):
        return 0  # Edge region looks like normal scene content → no artifact

    # --- Walk past gradient peak to find where brightness settles ---
    # The gradient peak marks where the housing-to-scene transition STARTS, but
    # there may be a transition zone (blurred edge, anti-aliasing) that extends
    # a few more pixels inward. We walk past the peak until brightness converges
    # to the "scene level" — estimated from the 20-40% depth range, which should
    # be well past any artifact and solidly inside the scene.
    scene_start = max(peak_idx + 5, int(scan_size * 0.20))
    scene_end = min(scan_size, int(scan_size * 0.40))
    if scene_end > scene_start:
        scene_level = float(np.mean(brightness[scene_start:scene_end]))
    else:
        scene_level = center_brightness  # fallback to overall center brightness

    # Settling tolerance: how close to scene_level brightness must be before we
    # consider the artifact transition "done". Scale with the artifact-to-scene
    # brightness jump to handle both subtle and strong artifacts.
    artifact_brightness = float(np.mean(brightness[:max(1, peak_idx + 1)]))
    jump_size = abs(artifact_brightness - scene_level)
    settle_tolerance = max(5.0, jump_size * 0.30)

    # Walk inward from the gradient peak (up to 30 additional pixels) until
    # the brightness settles within tolerance of the scene level
    settle_idx = peak_idx + 1
    for i in range(peak_idx + 1, min(peak_idx + 30, scan_size)):
        if abs(brightness[i] - scene_level) < settle_tolerance:
            settle_idx = i
            break
        settle_idx = i + 1

    # Add a 2-pixel safety margin past the settle point to ensure clean cropping
    artifact_extent = settle_idx + 2

    # --- Safety cap: never crop more than 15% of the image dimension ---
    # This prevents runaway cropping if the detection gets confused
    max_crop = int(dim * 0.15)
    artifact_extent = min(artifact_extent, max_crop)

    # --- Minimum artifact size: at least 5 rows/columns ---
    # Very small detections (1-2 px) are likely edge noise, not real artifacts
    if artifact_extent < 5:
        artifact_extent = 0

    return artifact_extent


# =============================================================================
# Verification Image Generation
# =============================================================================

def save_verification_images(
    dataset_name: str,
    median_rgb: np.ndarray,
    crop_analysis: Dict[str, int],
    crop_original: Dict[str, int],
    sample_paths: List[str],
    analysis_dir: Path,
    original_size: Tuple[int, int],
) -> None:
    """
    Generate and save verification images for the detected crop region.

    Outputs:
      1. median_image.png — raw median composite
      2. detection_overlay.png — median with red crop lines
      3. sample_before_after.png — sample images before/after crop
      4. crop_params.json — detected crop values at original resolution
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    out_dir = analysis_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save the raw median composite — useful for visual inspection of what
    #    the algorithm "sees" as the static background content
    Image.fromarray(median_rgb).save(out_dir / "median_image.png")
    logging.info(f"  Saved {out_dir / 'median_image.png'}")

    # 2. Detection overlay — draw the detected crop boundaries on top of the
    #    median image so users can verify correctness at a glance
    h, w = median_rgb.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(median_rgb)
    ax.set_title(f"{dataset_name} — Artifact Detection Overlay", fontsize=14)

    # Draw dashed red lines where the crop will occur (at analysis resolution)
    line_kwargs = dict(color="red", linewidth=2, linestyle="--")
    has_crop = False
    for edge, val in crop_analysis.items():
        if val <= 0:
            continue
        has_crop = True
        # Position crop line relative to the correct edge of the image
        if edge == "top":
            ax.axhline(y=val, **line_kwargs, label=f"top={val}px")
        elif edge == "bottom":
            ax.axhline(y=h - val, **line_kwargs, label=f"bottom={val}px")
        elif edge == "left":
            ax.axvline(x=val, **line_kwargs, label=f"left={val}px")
        elif edge == "right":
            ax.axvline(x=w - val, **line_kwargs, label=f"right={val}px")

    # Shade the regions that will be cropped away in translucent red
    if crop_analysis.get("top", 0) > 0:
        ax.add_patch(patches.Rectangle((0, 0), w, crop_analysis["top"],
                                        facecolor="red", alpha=0.2))
    if crop_analysis.get("bottom", 0) > 0:
        ax.add_patch(patches.Rectangle((0, h - crop_analysis["bottom"]), w,
                                        crop_analysis["bottom"],
                                        facecolor="red", alpha=0.2))
    if crop_analysis.get("left", 0) > 0:
        ax.add_patch(patches.Rectangle((0, 0), crop_analysis["left"], h,
                                        facecolor="red", alpha=0.2))
    if crop_analysis.get("right", 0) > 0:
        ax.add_patch(patches.Rectangle((w - crop_analysis["right"], 0),
                                        crop_analysis["right"], h,
                                        facecolor="red", alpha=0.2))

    if has_crop:
        ax.legend(loc="upper right", fontsize=10)
        # Annotate with original-resolution crop values
        info_text = "Original res crop: " + ", ".join(
            f"{k}={v}" for k, v in crop_original.items() if v > 0
        )
        ax.text(5, h - 5, info_text, fontsize=9, color="yellow",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
    else:
        ax.text(w // 2, h // 2, "No artifacts detected", fontsize=16,
                color="white", ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    plt.tight_layout()
    fig.savefig(out_dir / "detection_overlay.png", dpi=150)
    plt.close(fig)
    logging.info(f"  Saved {out_dir / 'detection_overlay.png'}")

    # 3. Before/after samples — side-by-side comparison of original vs. cropped
    #    images so the user can visually confirm the crop removes only artifacts
    num_samples = min(4, len(sample_paths))
    if num_samples > 0:
        fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]  # ensure axes is always a list of [before_ax, after_ax]

        # Pick evenly spaced samples to show diverse examples from the dataset
        indices = np.linspace(0, len(sample_paths) - 1, num_samples, dtype=int)

        for row_idx, path_idx in enumerate(indices):
            path = sample_paths[path_idx]
            try:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
            except Exception:
                continue

            orig_w, orig_h = img.size

            # Before
            axes[row_idx][0].imshow(img)
            axes[row_idx][0].set_title(f"Before ({orig_w}x{orig_h})", fontsize=10)
            axes[row_idx][0].axis("off")

            # Apply crop at original resolution
            left = crop_original.get("left", 0)
            top = crop_original.get("top", 0)
            right = orig_w - crop_original.get("right", 0)
            bottom = orig_h - crop_original.get("bottom", 0)

            if right > left and bottom > top:
                cropped = img.crop((left, top, right, bottom))
            else:
                cropped = img

            cw, ch = cropped.size
            axes[row_idx][1].imshow(cropped)
            axes[row_idx][1].set_title(f"After crop ({cw}x{ch})", fontsize=10)
            axes[row_idx][1].axis("off")

        fig.suptitle(f"{dataset_name} — Before / After Crop", fontsize=14, y=1.01)
        plt.tight_layout()
        fig.savefig(out_dir / "sample_before_after.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"  Saved {out_dir / 'sample_before_after.png'}")

    # 4. Save crop parameters as JSON for downstream use by `preview` and `apply`.
    #    Stores both analysis-resolution and original-resolution crop values so
    #    the user can verify and the apply command can use them directly.
    params = {
        "dataset": dataset_name,
        "original_image_size": {"width": original_size[0], "height": original_size[1]},
        "analysis_resolution": {"width": ANALYSIS_SIZE[0], "height": ANALYSIS_SIZE[1]},
        "crop_at_analysis_res": crop_analysis,
        "crop_at_original_res": crop_original,
    }
    json_path = out_dir / "crop_params.json"
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)
    logging.info(f"  Saved {json_path}")


def save_brightness_profile(
    dataset_name: str,
    median_rgb: np.ndarray,
    pixel_std: np.ndarray,
    crop_analysis: Dict[str, int],
    analysis_dir: Path,
) -> None:
    """
    Save brightness and cross-image-std profiles for each edge as a diagnostic plot.

    This is a 2x2 grid (top/bottom/left/right) showing how brightness (blue) and
    cross-image standard deviation (red) change as you move inward from each edge.
    A vertical green line marks the detected crop position. Useful for understanding
    why the algorithm did or did not detect an artifact on a given edge.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = analysis_dir / dataset_name
    gray = to_grayscale(median_rgb.astype(np.float32))
    h, w = gray.shape

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    edges = [("top", axes[0, 0]), ("bottom", axes[0, 1]),
             ("left", axes[1, 0]), ("right", axes[1, 1])]

    for edge, ax in edges:
        # Compute 1D profiles from the edge inward (same logic as in _detect_single_edge)
        scan_size = max(1, int((h if edge in ("top", "bottom") else w) * ARTIFACT_SCAN_FRACTION))

        if edge == "top":
            brightness = [gray[i, :].mean() for i in range(scan_size)]
            std_vals = [pixel_std[i, :].mean() for i in range(scan_size)]
        elif edge == "bottom":
            brightness = [gray[h - 1 - i, :].mean() for i in range(scan_size)]
            std_vals = [pixel_std[h - 1 - i, :].mean() for i in range(scan_size)]
        elif edge == "left":
            brightness = [gray[:, i].mean() for i in range(scan_size)]
            std_vals = [pixel_std[:, i].mean() for i in range(scan_size)]
        else:
            brightness = [gray[:, w - 1 - i].mean() for i in range(scan_size)]
            std_vals = [pixel_std[:, w - 1 - i].mean() for i in range(scan_size)]

        # Plot brightness (blue, left axis) and cross-image std (red, right axis)
        x = range(len(brightness))
        ax.plot(x, brightness, "b-", label="Brightness", linewidth=1.5)
        ax2 = ax.twinx()
        ax2.plot(x, std_vals, "r-", alpha=0.6, label="Cross-img std", linewidth=1)

        # Mark the detected crop position with a green dashed line
        crop_val = crop_analysis.get(edge, 0)
        if crop_val > 0:
            ax.axvline(x=crop_val, color="green", linestyle="--", linewidth=2,
                       label=f"Crop @ {crop_val}px")

        ax.set_title(f"{edge.capitalize()} edge (inward →)", fontsize=11)
        ax.set_xlabel("Pixels from edge")
        ax.set_ylabel("Brightness", color="blue")
        ax2.set_ylabel("Cross-img std", color="red")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"{dataset_name} — Edge Profiles (brightness + cross-image std)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "edge_profiles.png", dpi=150)
    plt.close(fig)
    logging.info(f"  Saved {out_dir / 'edge_profiles.png'}")


# =============================================================================
# Crop & Export
# =============================================================================

def crop_and_export(
    config: Dict,
    crop_params: Dict[str, int],
    output_dir: Path,
) -> Dict:
    """
    Apply crop to all images in the dataset and export to output directory.
    Preserves directory structure. Converts TIFF → PNG.

    Args:
        config: dataset configuration dict
        crop_params: {"top": N, "bottom": N, "left": N, "right": N} at original res
        output_dir: root output directory

    Returns:
        Stats dict with counts.
    """
    name = config["name"]
    root = config["root"]
    extensions = config["extensions"]
    image_subdir = config["image_subdir"]

    all_paths = list(iter_dataset_images(root, extensions, image_subdir))
    total = len(all_paths)

    if total == 0:
        logging.warning(f"No images found for {name}")
        return {"name": name, "total": 0, "exported": 0, "failed": 0}

    # Pre-extract crop values; skip cropping entirely if all edges are 0
    has_crop = any(v > 0 for v in crop_params.values())
    top = crop_params.get("top", 0)
    bottom = crop_params.get("bottom", 0)
    left = crop_params.get("left", 0)
    right = crop_params.get("right", 0)

    exported = 0
    failed = 0

    for abs_path, rel_path, _ in tqdm(all_paths, desc=f"Exporting {name}", unit="img"):
        try:
            img = Image.open(abs_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception:
            failed += 1
            continue

        # Apply the crop box: (left, top) is the upper-left corner,
        # (orig_w - right, orig_h - bottom) is the lower-right corner
        if has_crop:
            orig_w, orig_h = img.size
            x0 = left
            y0 = top
            x1 = orig_w - right
            y1 = orig_h - bottom
            # Sanity check: only crop if the result would be a valid (non-empty) image
            if x1 > x0 and y1 > y0:
                img = img.crop((x0, y0, x1, y1))

        # Mirror the source directory structure under the output directory
        out_path = output_dir / name / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert TIFF → PNG for broader compatibility and smaller file size
        if out_path.suffix.lower() in (".tiff", ".tif"):
            out_path = out_path.with_suffix(".png")

        img.save(out_path)
        exported += 1

    stats = {
        "name": name,
        "total": total,
        "exported": exported,
        "failed": failed,
        "crop_params": crop_params,
    }
    logging.info(f"  {name}: exported {exported:,} / {total:,} images (failed: {failed})")
    return stats


# =============================================================================
# Subcommand: detect
# =============================================================================

def cmd_detect(args: argparse.Namespace) -> None:
    """Run artifact detection on selected datasets, save verification images."""
    configs = _resolve_datasets(args.datasets)
    analysis_dir = Path(ANALYSIS_DIR)

    for config in configs:
        name = config["name"]
        print(f"\n{'='*60}")
        print(f"Detecting artifacts: {name} ({config['description']})")
        print(f"{'='*60}")

        # Step 1: Sample images from across the dataset and load at analysis resolution
        stack, original_sizes, paths = sample_and_load_images(
            config["root"], config["extensions"], config["image_subdir"],
        )

        if stack.size == 0:
            print(f"  Not enough images loaded for {name}, skipping.")
            continue

        n_loaded = stack.shape[0]
        print(f"  Loaded {n_loaded} sample images at {ANALYSIS_SIZE[0]}x{ANALYSIS_SIZE[1]}")

        # Step 2: Compute pixel-wise median — static artifacts persist, scene washes out
        median_rgb = compute_median_image(stack)

        # Step 3: Run multi-signal edge detection (at analysis resolution)
        crop_analysis = detect_artifact_edges(stack, median_rgb)

        # Also compute cross-image pixel std for the diagnostic profile plots
        gray_stack = to_grayscale(stack)
        pixel_std = gray_stack.std(axis=0)

        # Step 4: Scale crop coordinates from analysis resolution back to original.
        # Use the median original size across samples (handles minor size variations).
        orig_w = int(np.median([s[0] for s in original_sizes]))
        orig_h = int(np.median([s[1] for s in original_sizes]))
        ah, aw = ANALYSIS_SIZE[1], ANALYSIS_SIZE[0]  # analysis height, width

        crop_original = {}
        for edge, val in crop_analysis.items():
            # Scale proportionally: analysis pixels → original pixels
            if edge in ("top", "bottom"):
                crop_original[edge] = int(val * orig_h / ah)
            else:
                crop_original[edge] = int(val * orig_w / aw)

        # Print results
        has_artifacts = any(v > 0 for v in crop_original.values())
        if has_artifacts:
            print(f"  Artifacts detected!")
            print(f"    Analysis res crop:  {crop_analysis}")
            print(f"    Original res crop:  {crop_original}")
            print(f"    Original image size: {orig_w}x{orig_h}")
        else:
            print(f"  No significant artifacts detected.")

        # Save verification images
        print(f"  Saving verification images to {analysis_dir / name}/")
        save_verification_images(
            name, median_rgb, crop_analysis, crop_original,
            paths, analysis_dir, (orig_w, orig_h),
        )
        save_brightness_profile(
            name, median_rgb, pixel_std, crop_analysis, analysis_dir,
        )

    print(f"\nDone. Check {analysis_dir}/ for verification images.")


# =============================================================================
# Subcommand: preview
# =============================================================================

def cmd_preview(args: argparse.Namespace) -> None:
    """
    Show before/after crop on sample images. Supports auto-detected or manual crop.
    Reads crop params from the JSON saved by `detect`, or uses --crop overrides.
    Outputs a preview image to the analysis directory without modifying source images.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (no GUI window needed)
    import matplotlib.pyplot as plt

    configs = _resolve_datasets(args.datasets)

    # Parse manual crop overrides if provided (e.g. --crop bottom=80 left=20)
    manual_crop = {}
    if args.crop:
        for item in args.crop:
            key, val = item.split("=")
            manual_crop[key.strip()] = int(val.strip())

    for config in configs:
        name = config["name"]
        print(f"\nPreviewing crop for: {name}")

        # Load crop params — manual override or from detection JSON
        if manual_crop:
            crop_original = {
                "top": manual_crop.get("top", 0),
                "bottom": manual_crop.get("bottom", 0),
                "left": manual_crop.get("left", 0),
                "right": manual_crop.get("right", 0),
            }
            print(f"  Using manual crop: {crop_original}")
        else:
            json_path = Path(ANALYSIS_DIR) / name / "crop_params.json"
            if json_path.exists():
                with open(json_path) as f:
                    params = json.load(f)
                crop_original = params.get("crop_at_original_res", {})
                print(f"  Using auto-detected crop: {crop_original}")
            else:
                print(f"  No crop_params.json found. Run 'detect' first or provide --crop.")
                continue

        # Load a few sample images and show before/after
        all_paths = list(iter_dataset_images(
            config["root"], config["extensions"], config["image_subdir"],
        ))
        num_samples = min(args.num_samples, len(all_paths))
        if num_samples == 0:
            print(f"  No images found.")
            continue

        indices = np.linspace(0, len(all_paths) - 1, num_samples, dtype=int)

        fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for row_idx, path_idx in enumerate(indices):
            abs_path = all_paths[path_idx][0]
            try:
                img = Image.open(abs_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
            except Exception:
                continue

            orig_w, orig_h = img.size
            axes[row_idx][0].imshow(img)
            axes[row_idx][0].set_title(f"Before ({orig_w}x{orig_h})", fontsize=10)
            axes[row_idx][0].axis("off")

            left = crop_original.get("left", 0)
            top = crop_original.get("top", 0)
            right_edge = orig_w - crop_original.get("right", 0)
            bottom_edge = orig_h - crop_original.get("bottom", 0)

            if right_edge > left and bottom_edge > top:
                cropped = img.crop((left, top, right_edge, bottom_edge))
            else:
                cropped = img

            cw, ch = cropped.size
            axes[row_idx][1].imshow(cropped)
            axes[row_idx][1].set_title(f"After crop ({cw}x{ch})", fontsize=10)
            axes[row_idx][1].axis("off")

        fig.suptitle(f"{name} — Preview (crop: {crop_original})", fontsize=14, y=1.01)
        plt.tight_layout()

        out_path = Path(ANALYSIS_DIR) / name / "preview.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved preview to {out_path}")


# =============================================================================
# Subcommand: apply
# =============================================================================

def cmd_apply(args: argparse.Namespace) -> None:
    """
    Apply crop to all images in the selected datasets and export to the output directory.
    Uses either the auto-detected crop from `detect` (stored in crop_params.json) or
    manual --crop overrides. Source images are never modified; results go to --output-dir.
    """
    configs = _resolve_datasets(args.datasets)
    output_dir = Path(args.output_dir)

    # Parse manual crop overrides if provided (e.g. --crop bottom=80 left=20)
    manual_crop = {}
    if args.crop:
        for item in args.crop:
            key, val = item.split("=")
            manual_crop[key.strip()] = int(val.strip())

    all_stats = []
    for config in configs:
        name = config["name"]
        print(f"\n{'='*60}")
        print(f"Applying crop: {name}")
        print(f"{'='*60}")

        # Load crop params — manual override or from detection JSON
        if manual_crop:
            crop_params = {
                "top": manual_crop.get("top", 0),
                "bottom": manual_crop.get("bottom", 0),
                "left": manual_crop.get("left", 0),
                "right": manual_crop.get("right", 0),
            }
            print(f"  Using manual crop: {crop_params}")
        else:
            json_path = Path(ANALYSIS_DIR) / name / "crop_params.json"
            if json_path.exists():
                with open(json_path) as f:
                    params = json.load(f)
                crop_params = params.get("crop_at_original_res", {})
                print(f"  Using auto-detected crop: {crop_params}")
            else:
                print(f"  No crop_params.json found. Run 'detect' first or provide --crop.")
                print(f"  Skipping {name}.")
                continue

        stats = crop_and_export(config, crop_params, output_dir)
        all_stats.append(stats)

    # Summary
    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")
    for s in all_stats:
        print(f"  {s['name']:15s} exported={s['exported']:>6,}  "
              f"failed={s['failed']:>4,}  (of {s['total']:,})")
        crop = s.get("crop_params", {})
        if any(v > 0 for v in crop.values()):
            print(f"  {'':15s} crop: {crop}")
    print(f"{'='*60}\n")


# =============================================================================
# Helpers
# =============================================================================

def _resolve_datasets(dataset_names: Optional[List[str]]) -> List[Dict]:
    """
    Resolve user-provided dataset names to their config dicts from DATASET_CONFIGS.
    If no names are given (None), returns configs for all registered datasets.
    Exits with an error if an unrecognized dataset name is provided.
    """
    if dataset_names is None:
        return list(DATASET_CONFIGS.values())

    configs = []
    for name in dataset_names:
        if name in DATASET_CONFIGS:
            configs.append(DATASET_CONFIGS[name])
        else:
            available = ", ".join(DATASET_CONFIGS.keys())
            logging.error(f"Unknown dataset '{name}'. Available: {available}")
            sys.exit(1)
    return configs


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and crop vehicle/camera housing artifacts from underwater datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- detect ---
    detect_parser = subparsers.add_parser(
        "detect", help="Analyze images, detect crop regions, save verification images",
    )
    detect_parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Which datasets to process (default: all). Options: sunboat, shipwreck, flsea_vi",
    )
    detect_parser.add_argument("--verbose", action="store_true")

    # --- preview ---
    preview_parser = subparsers.add_parser(
        "preview", help="Show before/after crop on sample images",
    )
    preview_parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Which datasets to preview (default: all)",
    )
    preview_parser.add_argument(
        "--crop", nargs="*", default=None,
        help="Manual crop overrides, e.g. --crop bottom=80 left=20",
    )
    preview_parser.add_argument(
        "--num-samples", type=int, default=6,
        help="Number of sample images to show (default: 6)",
    )
    preview_parser.add_argument("--verbose", action="store_true")

    # --- apply ---
    apply_parser = subparsers.add_parser(
        "apply", help="Apply crop to all images and export",
    )
    apply_parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Which datasets to process (default: all)",
    )
    apply_parser.add_argument(
        "--crop", nargs="*", default=None,
        help="Manual crop overrides, e.g. --crop bottom=80 left=20",
    )
    apply_parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    apply_parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.command:
        print("Usage: python vehicle_artifact_crop.py {detect,preview,apply} [options]")
        print("Run with --help for details.")
        sys.exit(1)

    # Set up logging — verbose mode enables DEBUG-level output for extra diagnostics
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Dispatch to the appropriate subcommand handler
    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "preview":
        cmd_preview(args)
    elif args.command == "apply":
        cmd_apply(args)


if __name__ == "__main__":
    main()
