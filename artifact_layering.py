"""
Persistent Artifact Region Analysis (Median + Variance).

Builds a statistical composite from a dataset to highlight stable image regions
that likely represent camera/platform artifacts or other persistent structures.

Method summary:
    1. Load and normalize image sizes.
    2. Stack images and compute median image.
    3. Compute per-pixel variance map across the stack.
    4. Threshold low-variance regions as persistent-mask candidates.
    5. Export median image, variance visualization, and overlays for review.

Important:
    - This script is analysis-only and does not crop files.

Usage:
    python artifact_layering.py
    python artifact_layering.py --variance-threshold 0.0025
    python artifact_layering.py --input-dir path/to/images --output-dir out_dir
"""

import argparse
import os
import sys
from datetime import datetime
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
OUTPUT_DIR = "artifact_analysis/sunboat/median_variance"

STACK_COLOR_SPACE = "LAB"
VARIANCE_THRESHOLD = 0.0025

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


# ── Core median/variance logic ───────────────────────────────────────────────

def convert_to_color_space(img_bgr: np.ndarray, color_space: str) -> np.ndarray:
    """Convert BGR image to the requested color space for stacking."""
    color_space = color_space.upper()
    if color_space == "BGR":
        return img_bgr
    if color_space == "RGB":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if color_space == "HSV":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if color_space == "LAB":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    raise ValueError(f"Unsupported color space: {color_space}")


def convert_from_color_space(img_space_u8: np.ndarray, color_space: str) -> np.ndarray:
    """Convert image from stack color space back to BGR for visualization."""
    color_space = color_space.upper()
    if color_space == "BGR":
        return img_space_u8
    if color_space == "RGB":
        return cv2.cvtColor(img_space_u8, cv2.COLOR_RGB2BGR)
    if color_space == "HSV":
        return cv2.cvtColor(img_space_u8, cv2.COLOR_HSV2BGR)
    if color_space == "LAB":
        return cv2.cvtColor(img_space_u8, cv2.COLOR_LAB2BGR)
    raise ValueError(f"Unsupported color space: {color_space}")


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
    median_image: np.ndarray,
    variance_map: np.ndarray,
    persistent_mask: np.ndarray,
    color_space: str,
    variance_threshold: float,
    output_dir: str,
    timestamp: str,
) -> None:
    """Save median image, variance map, and persistent-mask preview artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    median_u8 = np.clip(median_image * 255.0, 0, 255).astype(np.uint8)
    median_bgr = convert_from_color_space(median_u8, color_space)

    np.save(os.path.join(output_dir, f"median_image_{timestamp}.npy"), median_image.astype(np.float32))
    np.save(os.path.join(output_dir, f"variance_map_{timestamp}.npy"), variance_map.astype(np.float32))
    np.save(os.path.join(output_dir, f"persistent_mask_{timestamp}.npy"), persistent_mask.astype(np.uint8))

    median_png_path = os.path.join(output_dir, f"median_image_{timestamp}.png")
    cv2.imwrite(median_png_path, median_bgr)

    sample_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
    median_rgb = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2RGB)

    # Build persistent mask overlay on sample image
    red_overlay = np.zeros_like(sample_rgb)
    red_overlay[..., 0] = persistent_mask.astype(np.uint8) * 255
    overlay_rgb = cv2.addWeighted(sample_rgb, 0.75, red_overlay, 0.35, 0)

    variance_png_path = os.path.join(output_dir, f"variance_map_{timestamp}.png")
    var_norm = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(variance_png_path, var_norm)

    mask_png_path = os.path.join(output_dir, f"persistent_mask_{timestamp}.png")
    cv2.imwrite(mask_png_path, persistent_mask.astype(np.uint8) * 255)

    # Save panel: sample, median image, variance map, persistent overlay
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(sample_rgb)
    axes[0].set_title("Sample Image")
    axes[0].axis("off")

    axes[1].imshow(median_rgb)
    axes[1].set_title("Median Image")
    axes[1].axis("off")

    heat = axes[2].imshow(variance_map, cmap="magma")
    axes[2].set_title(f"Variance Map (threshold={variance_threshold:.6f})")
    axes[2].axis("off")
    plt.colorbar(heat, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(overlay_rgb)
    axes[3].set_title("Persistent Mask Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    panel_path = os.path.join(output_dir, f"median_variance_overlay_{timestamp}.png")
    plt.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved outputs:")
    print(f"  - {median_png_path}")
    print(f"  - {variance_png_path}")
    print(f"  - {panel_path}")
    print(f"  - {mask_png_path}")
    print(f"  - {os.path.join(output_dir, f'median_image_{timestamp}.npy')}")
    print(f"  - {os.path.join(output_dir, f'variance_map_{timestamp}.npy')}")
    print(f"  - {os.path.join(output_dir, f'persistent_mask_{timestamp}.npy')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect persistent dataset structure via median image and variance thresholding",
    )
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Dataset root directory")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--color-space",
        type=str,
        default=STACK_COLOR_SPACE,
        choices=["BGR", "RGB", "HSV", "LAB"],
        help="Color space used when stacking images",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=VARIANCE_THRESHOLD,
        help="Threshold for persistent_mask = variance_map < threshold",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for debugging; process only first N images",
    )
    args = parser.parse_args()

    if args.variance_threshold < 0.0:
        print("Error: --variance-threshold must be >= 0.")
        sys.exit(1)

    image_paths = list(iter_dataset_images(args.input_dir, IMG_EXTENSIONS))
    if args.max_images is not None:
        image_paths = image_paths[: max(0, args.max_images)]

    if not image_paths:
        print(f"No images found in: {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")
    print("Step 1/5: Loading images and resizing to consistent shape...")

    # Determine reference shape using first valid image
    try:
        (ref_h, ref_w), sample_bgr, sample_path = choose_reference_shape(image_paths)
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Reference shape: {ref_w}x{ref_h} (from {os.path.basename(sample_path)})")

    stacked_images = []
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

        img_space = convert_to_color_space(img_bgr, args.color_space)
        stacked_images.append(img_space.astype(np.float32) / 255.0)
        processed_count += 1

    print("\nStep 2/5: Stacking images and computing median image")
    if processed_count == 0:
        print("No valid images could be processed.")
        sys.exit(1)

    try:
        all_images = np.stack(stacked_images, axis=0).astype(np.float32)
    except MemoryError:
        print("MemoryError: Unable to stack all images in memory. Try --max-images for debugging.")
        sys.exit(1)

    median_image = np.median(all_images, axis=0).astype(np.float32)

    print("Step 3/5: Computing variance map")
    variance_volume = np.var(all_images, axis=0)
    variance_map = variance_volume.mean(axis=2).astype(np.float32)

    print("Step 4/5: Thresholding variance map to build persistent mask")
    persistent_mask = variance_map < float(args.variance_threshold)

    print("Step 5/5: Saving verification outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_visualizations(
        sample_bgr=sample_bgr if sample_bgr.shape[:2] == (ref_h, ref_w)
        else cv2.resize(sample_bgr, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR),
        median_image=median_image,
        variance_map=variance_map,
        persistent_mask=persistent_mask,
        color_space=args.color_space,
        variance_threshold=float(args.variance_threshold),
        output_dir=args.output_dir,
        timestamp=timestamp,
    )

    mask_ratio = float(np.mean(persistent_mask))
    print("\nSummary:")
    print(f"  Processed images: {processed_count}")
    print(f"  Failed images:    {failed_count}")
    print(f"  Resized images:   {resized_count}")
    print(f"  Color space:      {args.color_space}")
    print(f"  Var threshold:    {args.variance_threshold:.6f}")
    print(f"  Mask fill ratio:  {mask_ratio:.2%} of pixels")
    print("\nDone. No cropping performed.")


if __name__ == "__main__":
    main()
