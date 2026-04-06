"""
Sunboat Persistent Structure Detection (Median Mask + Threshold)

Strategy:
    1) Load every image and resize it to a consistent shape.
    2) Stream images into a disk-backed uint8 memmap so the full stack does
       not have to live in RAM.
    3) Compute the exact per-pixel/channel median in row chunks.
    4) Compute a variance map from streaming sum and sum-of-squares moments.
    5) Threshold the variance map to build a persistent mask for visual
       verification before any cropping decisions.

Important:
    - This script does NOT crop images.
    - No shape filtering, boundary logic, or cropping heuristics are used.

Usage:
    python artifact_layering.py
    python artifact_layering.py --variance-threshold 0.0025
    python artifact_layering.py --input-dir path/to/images --output-dir out_dir
"""

import argparse
import concurrent.futures
import os
import shutil
import sys
from datetime import datetime
from typing import Generator, List, Optional, Set, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

matplotlib.use("Agg")
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Config ────────────────────────────────────────────────────────────────────

INPUT_DIR = "artifact_analysis/dc_train/median_variance_overlay_20260325_154032.png"
OUTPUT_DIR = "artifact_analysis/dc_train"

STACK_COLOR_SPACE = "LAB"
VARIANCE_THRESHOLD = 0.0075 

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


def compute_exact_median_from_memmap(
    images_memmap: np.memmap,
    num_images: int,
    ref_h: int,
    ref_w: int,
    chunk_rows: int,
) -> np.ndarray:
    """Compute exact per-pixel/channel median using row chunks from a disk-backed memmap."""
    if num_images <= 0:
        raise ValueError("num_images must be > 0")

    median_image = np.empty((ref_h, ref_w, 3), dtype=np.float32)
    chunk_rows = max(1, int(chunk_rows))

    for row_start in tqdm(range(0, ref_h, chunk_rows), desc="Median chunks", unit="chunk"):
        row_end = min(ref_h, row_start + chunk_rows)

        # Materialize only the current row band in RAM.
        block = np.array(images_memmap[:num_images, row_start:row_end, :, :], dtype=np.uint8, copy=True)

        if num_images % 2 == 1:
            mid = num_images // 2
            part = np.partition(block, mid, axis=0)
            median_block = part[mid].astype(np.float32) / 255.0
        else:
            mid_hi = num_images // 2
            mid_lo = mid_hi - 1
            part = np.partition(block, (mid_lo, mid_hi), axis=0)
            lo = part[mid_lo].astype(np.float32)
            hi = part[mid_hi].astype(np.float32)
            median_block = (lo + hi) * (0.5 / 255.0)

        median_image[row_start:row_end, :, :] = median_block

    return median_image


def parse_gpu_devices(devices_arg: str, available_count: int) -> List[int]:
    """Parse comma-separated GPU ids and keep only valid device indices."""
    if not devices_arg.strip():
        return []

    parsed = []
    seen = set()
    for raw in devices_arg.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            dev = int(token)
        except ValueError:
            continue
        if 0 <= dev < available_count and dev not in seen:
            parsed.append(dev)
            seen.add(dev)
    return parsed


def _torch_median_block(block_u8: np.ndarray, num_images: int, device_id: int) -> np.ndarray:
    """Compute exact per-pixel/channel median for one block on a specific CUDA device."""
    import torch

    with torch.cuda.device(device_id):
        t_block = torch.from_numpy(block_u8).to(device=f"cuda:{device_id}", dtype=torch.uint8)

        if num_images % 2 == 1:
            k = (num_images // 2) + 1
            values = torch.kthvalue(t_block, k=k, dim=0).values
            out = values.to(dtype=torch.float32) / 255.0
        else:
            k_lo = num_images // 2
            k_hi = k_lo + 1
            lo = torch.kthvalue(t_block, k=k_lo, dim=0).values.to(dtype=torch.float32)
            hi = torch.kthvalue(t_block, k=k_hi, dim=0).values.to(dtype=torch.float32)
            out = (lo + hi) * (0.5 / 255.0)

        out_np = out.detach().cpu().numpy().astype(np.float32, copy=False)
        del t_block, out
        return out_np


def compute_exact_median_from_memmap_torch(
    images_memmap: np.memmap,
    num_images: int,
    ref_h: int,
    ref_w: int,
    chunk_rows: int,
    device_ids: List[int],
) -> np.ndarray:
    """Compute exact median using torch CUDA, dispatching chunks across GPUs."""
    if num_images <= 0:
        raise ValueError("num_images must be > 0")
    if not device_ids:
        raise ValueError("device_ids must not be empty")

    median_image = np.empty((ref_h, ref_w, 3), dtype=np.float32)
    chunk_rows = max(1, int(chunk_rows))
    row_ranges = [(row_start, min(ref_h, row_start + chunk_rows)) for row_start in range(0, ref_h, chunk_rows)]

    def worker(row_start: int, row_end: int, device_id: int) -> Tuple[int, np.ndarray]:
        block = np.array(images_memmap[:num_images, row_start:row_end, :, :], dtype=np.uint8, copy=True)
        median_block = _torch_median_block(block, num_images, device_id)
        return row_start, median_block

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        for idx, (row_start, row_end) in enumerate(row_ranges):
            device_id = device_ids[idx % len(device_ids)]
            futures.append(executor.submit(worker, row_start, row_end, device_id))

        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Median chunks (GPU)", unit="chunk"):
            row_start, median_block = fut.result()
            row_end = row_start + median_block.shape[0]
            median_image[row_start:row_end, :, :] = median_block

    return median_image


def format_bytes(num_bytes: int) -> str:
    """Human-readable byte formatter."""
    size = float(max(0, num_bytes))
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


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
    parser.add_argument(
        "--median-chunk-rows",
        type=int,
        default=8,
        help="Row chunk size for exact median computation (smaller uses less RAM, slower)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory for memmap storage (default: output-dir to avoid small /tmp filesystems)",
    )
    parser.add_argument(
        "--keep-work-files",
        action="store_true",
        help="Keep temporary memmap files for inspection/debugging",
    )
    parser.add_argument(
        "--compute-backend",
        type=str,
        default="torch",
        choices=["cpu", "torch"],
        help="Compute backend for median chunks. 'torch' enables CUDA acceleration if available.",
    )
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default="0,1",
        help="Comma-separated CUDA device ids used when --compute-backend torch (example: 0,1)",
    )
    args = parser.parse_args()

    if args.variance_threshold < 0.0:
        print("Error: --variance-threshold must be >= 0.")
        sys.exit(1)
    if args.median_chunk_rows <= 0:
        print("Error: --median-chunk-rows must be >= 1.")
        sys.exit(1)

    use_torch_gpu = args.compute_backend == "torch"
    gpu_device_ids: List[int] = []

    if use_torch_gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                print("Warning: torch CUDA not available. Falling back to CPU backend.")
                use_torch_gpu = False
            else:
                gpu_device_ids = parse_gpu_devices(args.gpu_devices, torch.cuda.device_count())
                if not gpu_device_ids:
                    print("Warning: no valid CUDA devices selected. Falling back to CPU backend.")
                    use_torch_gpu = False
        except Exception as exc:
            print(f"Warning: torch backend unavailable ({exc}). Falling back to CPU backend.")
            use_torch_gpu = False

    image_paths = list(iter_dataset_images(args.input_dir, IMG_EXTENSIONS))
    if args.max_images is not None:
        image_paths = image_paths[: max(0, args.max_images)]

    if not image_paths:
        print(f"No images found in: {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")
    if use_torch_gpu:
        print(f"Compute backend: torch CUDA on devices {gpu_device_ids}")
    else:
        print("Compute backend: CPU numpy")
    print("Step 1/5: Loading images and resizing to consistent shape...")

    # Determine reference shape using first valid image
    try:
        (ref_h, ref_w), sample_bgr, sample_path = choose_reference_shape(image_paths)
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Reference shape: {ref_w}x{ref_h} (from {os.path.basename(sample_path)})")

    # Use a disk-backed array so we do not keep the full image stack in RAM.
    # Default to output_dir rather than /tmp to avoid small root/tmp filesystems.
    memmap_dir = args.work_dir if args.work_dir else args.output_dir
    os.makedirs(memmap_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memmap_path = os.path.join(memmap_dir, f"sunboat_stack_{timestamp}.dat")

    expected_memmap_bytes = len(image_paths) * ref_h * ref_w * 3
    free_bytes = shutil.disk_usage(memmap_dir).free
    safety_margin_bytes = 2 * 1024 * 1024 * 1024
    required_with_margin = expected_memmap_bytes + safety_margin_bytes

    print("Allocating disk-backed image stack...")
    print(f"  Memmap path: {memmap_path}")
    print(f"  Expected memmap size: {format_bytes(expected_memmap_bytes)}")
    print(f"  Free space in work-dir filesystem: {format_bytes(free_bytes)}")

    if free_bytes < required_with_margin:
        print("Error: insufficient free space for memmap-backed stack.")
        print(f"  Needed (with safety margin): {format_bytes(required_with_margin)}")
        print(f"  Available:                  {format_bytes(free_bytes)}")
        print("Tip: pass --work-dir to a larger filesystem (example: /data/Lockheed1-Spring26).")
        sys.exit(1)

    print("Step 1/5: Loading images, resizing, color conversion, and streaming stats...")

    # uint8 memmap keeps storage compact while preserving exact pixel values.
    try:
        images_memmap = np.memmap(
            memmap_path,
            dtype=np.uint8,
            mode="w+",
            shape=(len(image_paths), ref_h, ref_w, 3),
        )
    except OSError as exc:
        print(f"Error: failed to create memmap file at {memmap_path}: {exc}")
        print("Tip: choose a writable path with enough free space via --work-dir.")
        sys.exit(1)

    processed_count = 0
    failed_count = 0
    resized_count = 0

    # Exact variance from streaming moments on uint8 values.
    sum_channels = np.zeros((ref_h, ref_w, 3), dtype=np.uint64)
    sumsq_channels = np.zeros((ref_h, ref_w, 3), dtype=np.uint64)

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

        img_u8 = img_space.astype(np.uint8)
        images_memmap[processed_count] = img_u8

        img_u64 = img_u8.astype(np.uint64)
        sum_channels += img_u64
        sumsq_channels += img_u64 * img_u64

        processed_count += 1

    images_memmap.flush()

    print("\nStep 2/5: Computing exact median image in row chunks")
    if processed_count == 0:
        print("No valid images could be processed.")
        del images_memmap
        if not args.keep_work_files and os.path.exists(memmap_path):
            os.remove(memmap_path)
        sys.exit(1)

    try:
        if use_torch_gpu:
            median_image = compute_exact_median_from_memmap_torch(
                images_memmap=images_memmap,
                num_images=processed_count,
                ref_h=ref_h,
                ref_w=ref_w,
                chunk_rows=args.median_chunk_rows,
                device_ids=gpu_device_ids,
            )
        else:
            median_image = compute_exact_median_from_memmap(
                images_memmap=images_memmap,
                num_images=processed_count,
                ref_h=ref_h,
                ref_w=ref_w,
                chunk_rows=args.median_chunk_rows,
            )
    except MemoryError:
        print(
            "MemoryError during median chunks. Try reducing --median-chunk-rows "
            "(example: --median-chunk-rows 4)."
        )
        del images_memmap
        if not args.keep_work_files and os.path.exists(memmap_path):
            os.remove(memmap_path)
        sys.exit(1)

    print("Step 3/5: Computing variance map from streaming moments")
    n = float(processed_count)
    mean_norm = (sum_channels.astype(np.float64) / n) / 255.0
    mean_sq_norm = (sumsq_channels.astype(np.float64) / n) / (255.0 * 255.0)
    variance_volume = np.maximum(0.0, mean_sq_norm - (mean_norm * mean_norm))
    variance_map = variance_volume.mean(axis=2).astype(np.float32)

    del images_memmap
    if not args.keep_work_files and os.path.exists(memmap_path):
        os.remove(memmap_path)

    print("Step 4/5: Thresholding variance map to build persistent mask")
    persistent_mask = variance_map < float(args.variance_threshold)

    print("Step 5/5: Saving verification outputs")
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
