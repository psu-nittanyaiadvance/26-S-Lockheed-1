"""
Sunboat Persistent Object Detection (Layer Accumulation)

This script replaces the previous highest-point cropping strategy with a simple,
dataset-level persistence approach:

  1) Run Canny edge detection on every full image
  2) Convert edges to binary masks
  3) Accumulate masks across the dataset
     persistent_map = sum(all_binary_masks) / num_images
  4) Threshold persistence to keep only pixels that appear in >= X% of images
     final_mask = persistent_map >= X
  5) Save an overlay of final_mask on a sample image for verification

Important:
  - This script does NOT crop images.
  - No shape filtering, bottom-crop logic, or boundary heuristics are used.

Usage:
  python sunboat_robust.py
  python sunboat_robust.py --threshold 0.65
  python sunboat_robust.py --input-dir path/to/images --output-dir out_dir
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional, Set, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

matplotlib.use("Agg")
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Config ────────────────────────────────────────────────────────────────────

INPUT_DIR = "watersplatting_data/Sunboat_03-09-2023"
OUTPUT_DIR = "artifact_analysis/sunboat/persistence"

PERSISTENCE_THRESHOLD = 0.60
CANNY_LOW = 80
CANNY_HIGH = 180
GAUSSIAN_KERNEL = 5

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def iter_dataset_images(
    root: str,
    extensions: Set[str] = IMG_EXTENSIONS,
) -> Generator[str, None, None]:
    """Recursively yield image paths under `root`, sorted for reproducibility."""
    root = os.path.abspath(root)
    all_images = []

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if os.path.splitext(filename)[1] in extensions:
                all_images.append(os.path.join(dirpath, filename))

    all_images.sort()
    for image_path in all_images:
        yield image_path


def load_image_safe(image_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Load an image robustly.
    Returns (image_bgr, error_message). Only one of them is non-None.
    """
    try:
        # PIL path first for better handling of truncated/corrupted inputs
        try:
            pil_img = Image.open(image_path)
            pil_img.verify()
            pil_img = Image.open(image_path)

            # Normalize to 3-channel RGB if needed (e.g., grayscale/RGBA)
            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")
            elif pil_img.mode == "L":
                pil_img = pil_img.convert("RGB")

            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return None, "Could not load image"

        return img_bgr, None
    except Exception as exc:
        return None, str(exc)


# ── Core accumulation logic ───────────────────────────────────────────────────

def canny_binary_mask(
    img_bgr: np.ndarray,
    canny_low: int,
    canny_high: int,
    gaussian_kernel: int,
) -> np.ndarray:
    """Return full-image Canny edge mask as uint8 binary {0, 1}."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Ensure odd kernel size >= 1
    if gaussian_kernel < 1:
        gaussian_kernel = 1
    if gaussian_kernel % 2 == 0:
        gaussian_kernel += 1

    if gaussian_kernel > 1:
        gray = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)

    edges = cv2.Canny(gray, canny_low, canny_high)
    return (edges > 0).astype(np.uint8)


def choose_reference_shape(image_paths: list) -> Tuple[Tuple[int, int], np.ndarray, str]:
    """
    Find first loadable image and use its shape as dataset reference shape.
    Returns ((h, w), sample_bgr_resized, sample_path).
    """
    for image_path in image_paths:
        img_bgr, error = load_image_safe(image_path)
        if img_bgr is not None:
            h, w = img_bgr.shape[:2]
            return (h, w), img_bgr, image_path

        if error:
            continue

    raise RuntimeError("No loadable images found in dataset.")


def save_visualizations(
    sample_bgr: np.ndarray,
    persistent_map: np.ndarray,
    final_mask: np.ndarray,
    output_dir: str,
    timestamp: str,
) -> None:
    """Save persistence map, final mask, and overlay preview artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    # Save raw arrays for later downstream steps
    np.save(os.path.join(output_dir, f"persistent_map_{timestamp}.npy"), persistent_map)
    np.save(os.path.join(output_dir, f"final_mask_{timestamp}.npy"), final_mask.astype(np.uint8))

    # Convert sample to RGB for plotting
    sample_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)

    # Build color overlay (red where mask is persistent)
    red_overlay = np.zeros_like(sample_rgb)
    red_overlay[..., 0] = final_mask.astype(np.uint8) * 255
    overlay_rgb = cv2.addWeighted(sample_rgb, 0.75, red_overlay, 0.35, 0)

    # Save panel: sample, persistence heatmap, final overlay
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(sample_rgb)
    axes[0].set_title("Sample Image")
    axes[0].axis("off")

    heat = axes[1].imshow(persistent_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("Persistence Map")
    axes[1].axis("off")
    plt.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Final Mask Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    panel_path = os.path.join(output_dir, f"persistence_overlay_{timestamp}.png")
    plt.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also save standalone binary mask image for quick visual checks
    mask_png_path = os.path.join(output_dir, f"final_mask_{timestamp}.png")
    cv2.imwrite(mask_png_path, final_mask.astype(np.uint8) * 255)

    print("\nSaved outputs:")
    print(f"  - {panel_path}")
    print(f"  - {mask_png_path}")
    print(f"  - {os.path.join(output_dir, f'persistent_map_{timestamp}.npy')}")
    print(f"  - {os.path.join(output_dir, f'final_mask_{timestamp}.npy')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect the most persistent object via edge-mask accumulation",
    )
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Dataset root directory")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--threshold",
        type=float,
        default=PERSISTENCE_THRESHOLD,
        help="Persistence threshold X in [0, 1] for final_mask = persistent_map >= X",
    )
    parser.add_argument("--canny-low", type=int, default=CANNY_LOW, help="Canny low threshold")
    parser.add_argument("--canny-high", type=int, default=CANNY_HIGH, help="Canny high threshold")
    parser.add_argument(
        "--gaussian-kernel",
        type=int,
        default=GAUSSIAN_KERNEL,
        help="Gaussian blur kernel size before Canny (odd positive integer)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for debugging; process only first N images",
    )
    args = parser.parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        print("Error: --threshold must be between 0 and 1.")
        sys.exit(1)

    image_paths = list(iter_dataset_images(args.input_dir, IMG_EXTENSIONS))
    if args.max_images is not None:
        image_paths = image_paths[: max(0, args.max_images)]

    if not image_paths:
        print(f"No images found in: {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")
    print("Step 1/4: Running Canny on full images and collecting binary masks...")

    # Determine reference shape using first valid image
    try:
        (ref_h, ref_w), sample_bgr, sample_path = choose_reference_shape(image_paths)
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Reference shape: {ref_w}x{ref_h} (from {os.path.basename(sample_path)})")

    # Accumulate sum of binary masks (float32 accumulator)
    sum_map = np.zeros((ref_h, ref_w), dtype=np.float32)
    processed_count = 0
    failed_count = 0
    resized_count = 0

    for image_path in tqdm(image_paths, unit="img"):
        img_bgr, error = load_image_safe(image_path)
        if img_bgr is None:
            failed_count += 1
            continue

        h, w = img_bgr.shape[:2]
        if (h, w) != (ref_h, ref_w):
            img_bgr = cv2.resize(img_bgr, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
            resized_count += 1

        binary_mask = canny_binary_mask(
            img_bgr,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            gaussian_kernel=args.gaussian_kernel,
        )

        sum_map += binary_mask.astype(np.float32)
        processed_count += 1

    print("\nStep 2/4: Accumulating over dataset")
    if processed_count == 0:
        print("No valid images could be processed.")
        sys.exit(1)

    persistent_map = sum_map / float(processed_count)

    print("Step 3/4: Thresholding by persistence")
    final_mask = persistent_map >= args.threshold

    print("Step 4/4: Saving verification outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_visualizations(
        sample_bgr=sample_bgr if sample_bgr.shape[:2] == (ref_h, ref_w)
        else cv2.resize(sample_bgr, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR),
        persistent_map=persistent_map,
        final_mask=final_mask,
        output_dir=args.output_dir,
        timestamp=timestamp,
    )

    mask_ratio = float(np.mean(final_mask))
    print("\nSummary:")
    print(f"  Processed images: {processed_count}")
    print(f"  Failed images:    {failed_count}")
    print(f"  Resized images:   {resized_count}")
    print(f"  Threshold (X):    {args.threshold:.3f}")
    print(f"  Final mask fill:  {mask_ratio:.2%} of pixels")
    print("\nDone. No cropping performed.")


if __name__ == "__main__":
    main()
