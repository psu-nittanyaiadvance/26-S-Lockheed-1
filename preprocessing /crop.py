#!/usr/bin/env python3
"""
Manual image crop utility. Provide path to dataset and crop dataset to desired crop line.

This script crops a dataset using a user-provided crop line. It does not run
artifact detection or hue analysis.

You choose:
  - crop line (pixel index)
  - axis: horizontal or vertical
  - side to keep: top/bottom/left/right

Usage:
  python crop.py --line 420 --axis horizontal --keep top
  python crop.py --line 320 --axis vertical --keep left --apply
"""

import argparse
import os
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Generator, List, Optional, Set, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

INPUT_DIR = "watersplatting_data/Sunboat_03-09-2023"
PREVIEW_DIR = "artifact_analysis/sunboat/previews"
OUTPUT_DIR = "artifact_analysis/sunboat/cropped"

PREVIEW_SAMPLES = 9
RANDOM_SEED = 42

# Image validation thresholds
BLANK_STD_THRESHOLD = 2.0
BLANK_RANGE_THRESHOLD = 8

# Manual crop defaults. CLI values override these.
# Set CROP_LINE to an integer pixel index to run without passing --line.
CROP_LINE = None
CROP_AXIS = "horizontal"   # horizontal or vertical
CROP_KEEP = "top"          # horizontal -> top/bottom, vertical -> left/right

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def iter_dataset_images(root: str, extensions: Set[str] = IMG_EXTENSIONS) -> Generator[str, None, None]:
    """Yield absolute image paths recursively, sorted for deterministic runs."""
    root = os.path.abspath(root)
    all_images: List[str] = []

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if os.path.splitext(filename)[1] in extensions:
                all_images.append(os.path.join(dirpath, filename))

    all_images.sort()
    for image_path in all_images:
        yield image_path


def load_image_bgr(image_path: str) -> Optional[np.ndarray]:
    """Robustly load image with PIL-first fallback to OpenCV."""
    try:
        try:
            pil_img = Image.open(image_path)
            pil_img.verify()
            pil_img = Image.open(image_path)
            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")
            elif pil_img.mode == "L":
                pil_img = pil_img.convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return cv2.imread(image_path)
    except Exception:
        return None


def is_strict_blank_image(img_bgr: np.ndarray) -> bool:
    """
    Strict blank check with no thresholds.

    An image is considered blank only when every pixel has the same grayscale
    value (for example all black, all white, or a constant flat frame).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return int(gray.min()) == int(gray.max())


def report_dataset_health(all_files: List[str]) -> None:
    """Report unopenable files, strict blank files, and image dimension counts."""
    unopenable: List[str] = []
    blank_files: List[str] = []
    dim_counts = {}

    for image_path in tqdm(all_files, desc="Scanning dataset", unit="img"):
        img_bgr = load_image_bgr(image_path)
        if img_bgr is None:
            unopenable.append(image_path)
            continue

        h, w = img_bgr.shape[:2]
        key = (w, h)
        dim_counts[key] = dim_counts.get(key, 0) + 1

        if is_strict_blank_image(img_bgr):
            blank_files.append(image_path)

    print("\nPre-crop dataset report:")
    print(f"  Total discovered images: {len(all_files)}")
    print(f"  Unopenable files:        {len(unopenable)}")
    print(f"  Strict blank images:     {len(blank_files)}")
    print(f"  Unique dimensions:       {len(dim_counts)}")

    if dim_counts:
        print("\nDimension distribution (width x height -> count):")
        sorted_dims = sorted(dim_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
        for (w, h), count in sorted_dims:
            print(f"  {w}x{h} -> {count}")

    if blank_files:
        print("\nStrict blank image files (up to 50):")
        for path in blank_files[:50]:
            print(f"  {path}")
        if len(blank_files) > 50:
            print(f"  ... and {len(blank_files) - 50} more")

    if unopenable:
        print("\nUnopenable image files (up to 50):")
        for path in unopenable[:50]:
            print(f"  {path}")
        if len(unopenable) > 50:
            print(f"  ... and {len(unopenable) - 50} more")


def is_blank_image(img_bgr: np.ndarray) -> bool:
    """Detect near-blank images using grayscale spread metrics."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    std_val = float(np.std(gray))
    range_val = int(np.max(gray)) - int(np.min(gray))
    return std_val < BLANK_STD_THRESHOLD or range_val < BLANK_RANGE_THRESHOLD


def validate_keep_option(axis: str, keep: str) -> bool:
    """Ensure keep value is valid for the selected axis."""
    if axis == "horizontal":
        return keep in {"top", "bottom"}
    return keep in {"left", "right"}


def crop_image(img_bgr: np.ndarray, line: int, axis: str, keep: str) -> Optional[np.ndarray]:
    """Crop image by line/axis/keep selection."""
    h, w = img_bgr.shape[:2]

    if axis == "horizontal":
        if line <= 0 or line >= h:
            return None
        if keep == "top":
            return img_bgr[:line, :]
        return img_bgr[line:, :]

    if line <= 0 or line >= w:
        return None
    if keep == "left":
        return img_bgr[:, :line]
    return img_bgr[:, line:]


def draw_crop_guide(img_rgb: np.ndarray, line: int, axis: str) -> np.ndarray:
    """Overlay crop guide line on RGB image for preview."""
    vis_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = vis_bgr.shape[:2]
    if axis == "horizontal":
        y = max(0, min(h - 1, line))
        cv2.line(vis_bgr, (0, y), (w - 1, y), (0, 0, 255), 2)
    else:
        x = max(0, min(w - 1, line))
        cv2.line(vis_bgr, (x, 0), (x, h - 1), (0, 0, 255), 2)
    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


def save_preview(
    sample_paths: List[str],
    output_path: str,
    line: int,
    axis: str,
    keep: str,
) -> None:
    """Save 2-panel preview grid (original with guide, cropped output)."""
    valid_images: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for image_path in sample_paths:
        img_bgr = load_image_bgr(image_path)
        if img_bgr is None:
            continue
        cropped = crop_image(img_bgr, line, axis, keep)
        if cropped is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        valid_images.append((os.path.basename(image_path), img_rgb, crop_rgb))
        if len(valid_images) >= PREVIEW_SAMPLES:
            break

    if not valid_images:
        print("No valid images available for preview with current crop settings.")
        return

    n = len(valid_images)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = np.array([axes])

    for idx, (filename, img_rgb, crop_rgb) in enumerate(valid_images):
        guide = draw_crop_guide(img_rgb, line, axis)

        axes[idx, 0].imshow(guide)
        axes[idx, 0].set_title(f"Original + Guide\n{filename}", fontsize=10)
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(crop_rgb)
        axes[idx, 1].set_title(f"Cropped ({axis}, keep={keep})\n{crop_rgb.shape[1]}x{crop_rgb.shape[0]} px", fontsize=10)
        axes[idx, 1].axis("off")

    plt.suptitle(f"Manual Crop Preview (line={line}, axis={axis}, keep={keep})", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPreview saved to: {output_path}")


def crop_and_save(args: Tuple[str, str, str, int, str, str]) -> Tuple[bool, str]:
    """Worker: crop one image and save preserving relative path."""
    image_path, input_dir, output_root, line, axis, keep = args
    rel_path = os.path.relpath(image_path, input_dir)
    out_path = os.path.join(output_root, rel_path)

    try:
        img_bgr = load_image_bgr(image_path)
        if img_bgr is None:
            return False, f"Could not load image: {image_path}"

        cropped = crop_image(img_bgr, line, axis, keep)
        if cropped is None:
            return False, f"Invalid crop line for image shape: {image_path}"

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ok = cv2.imwrite(out_path, cropped)
        if not ok:
            return False, f"Failed to write image: {out_path}"
        return True, out_path
    except Exception as exc:
        return False, f"Error processing {image_path}: {exc}"


def validate_images_for_processing(all_files: List[str]) -> Tuple[List[str], List[Tuple[str, str]], Optional[Tuple[int, int]]]:
    """
    Validate files before preview/apply and return eligible images.

    Rules:
      - skip unopenable images
      - skip blank/near-blank images
      - skip images whose dimensions differ from the first valid reference image
    """
    valid_files: List[str] = []
    skipped: List[Tuple[str, str]] = []
    ref_shape: Optional[Tuple[int, int]] = None

    for image_path in tqdm(all_files, desc="Validating images", unit="img"):
        img_bgr = load_image_bgr(image_path)
        if img_bgr is None:
            skipped.append((image_path, "unopenable"))
            continue

        if is_blank_image(img_bgr):
            skipped.append((image_path, "blank_or_near_blank"))
            continue

        h, w = img_bgr.shape[:2]
        if ref_shape is None:
            ref_shape = (h, w)
        elif (h, w) != ref_shape:
            skipped.append((image_path, f"dimension_mismatch ({w}x{h} != {ref_shape[1]}x{ref_shape[0]})"))
            continue

        valid_files.append(image_path)

    return valid_files, skipped, ref_shape


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual crop tool using user-provided crop line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Use values hard-coded in config
  python crop.py

  python crop.py --line 420 --axis horizontal --keep top
  python crop.py --line 320 --axis vertical --keep left
  python crop.py --line 320 --axis vertical --keep left --apply
        """,
    )
    parser.add_argument(
        "--line",
        type=int,
        default=CROP_LINE,
        help="Crop line pixel index (defaults to CROP_LINE in config)",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default=CROP_AXIS,
        choices=["horizontal", "vertical"],
        help="Crop axis: horizontal uses row index, vertical uses column index (default from config)",
    )
    parser.add_argument(
        "--keep",
        type=str,
        default=CROP_KEEP,
        choices=["top", "bottom", "left", "right"],
        help="Side to keep after crop (default from config; valid pairs: horizontal->top/bottom, vertical->left/right)",
    )
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Dataset root directory")
    parser.add_argument("--preview-dir", default=PREVIEW_DIR, help="Directory to save preview image")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory to save cropped images")
    parser.add_argument("--apply", action="store_true", help="Apply crop to all images (default: preview only)")
    parser.add_argument("--preview-samples", type=int, default=PREVIEW_SAMPLES, help="Max images shown in preview")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Seed for deterministic preview sampling")
    args = parser.parse_args()

    if args.line is None:
        print("Error: crop line not set. Pass --line or set CROP_LINE in the config section.")
        sys.exit(1)

    if args.line < 0:
        print("Error: crop line must be >= 0")
        sys.exit(1)

    if not validate_keep_option(args.axis, args.keep):
        print(
            "Error: invalid --keep for selected --axis. "
            "Use top/bottom for horizontal, left/right for vertical."
        )
        sys.exit(1)

    all_files = list(iter_dataset_images(args.input_dir, IMG_EXTENSIONS))
    if not all_files:
        print(f"No images found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(all_files)} images")
    print(f"Crop settings: line={args.line}, axis={args.axis}, keep={args.keep}")

    # Report image quality/shape issues before preview/apply so training-time
    # resize/skip decisions can be made up front.
    report_dataset_health(all_files)

    valid_files, skipped_files, ref_shape = validate_images_for_processing(all_files)
    if ref_shape is not None:
        print(f"Reference dimensions: {ref_shape[1]}x{ref_shape[0]}")

    if skipped_files:
        reason_counts = {}
        for _, reason in skipped_files:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        print("Validation skipped files:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[0]):
            print(f"  {reason}: {count}")

        print("First skipped files (up to 20):")
        for path, reason in skipped_files[:20]:
            print(f"  {os.path.basename(path)} -> {reason}")
        if len(skipped_files) > 20:
            print(f"  ... and {len(skipped_files) - 20} more")

    if not valid_files:
        print("No valid images remain after validation. Exiting.")
        sys.exit(1)

    print(f"Images eligible for processing: {len(valid_files)}")

    global PREVIEW_SAMPLES, RANDOM_SEED
    PREVIEW_SAMPLES = max(1, args.preview_samples)
    RANDOM_SEED = args.seed

    rng = np.random.RandomState(RANDOM_SEED)
    shuffled = [valid_files[i] for i in rng.permutation(len(valid_files))]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.preview_dir, exist_ok=True)
    preview_path = os.path.join(args.preview_dir, f"preview_manual_crop_{timestamp}.png")
    save_preview(shuffled, preview_path, args.line, args.axis, args.keep)

    if not args.apply:
        print("\nPreview only complete.")
        print("To write cropped outputs, re-run with --apply")
        return

    output_dir_timestamped = os.path.join(args.output_dir, f"cropped_{timestamp}")
    os.makedirs(output_dir_timestamped, exist_ok=True)

    worker_args = [
        (image_path, os.path.abspath(args.input_dir), output_dir_timestamped, args.line, args.axis, args.keep)
        for image_path in valid_files
    ]

    print(f"\nApplying crop to all images -> {output_dir_timestamped}")
    workers = max(1, cpu_count() - 1)
    ok_count = 0
    fail_count = 0

    with Pool(workers) as pool:
        for ok, msg in tqdm(pool.imap(crop_and_save, worker_args), total=len(worker_args), unit="img"):
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                if fail_count <= 20:
                    print(f"Warning: {msg}")

    print("\nDone")
    print(f"  Preview: {preview_path}")
    print(f"  Output:  {output_dir_timestamped}")
    print(f"  Success: {ok_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Skipped during validation: {len(skipped_files)}")


if __name__ == "__main__":
    main()
