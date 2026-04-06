"""
Template-Matching Artifact Boundary Detection and Cropping.

This script locates a recurring border artifact by matching a reference template
inside a configurable image corner/edge search region, then derives a global
crop row from the highest confident match.

Method summary:
    1. Load a template image that represents the artifact pattern.
    2. Recursively discover dataset images.
    3. Run template matching in a restricted border region.
    4. Keep matches above MATCH_THRESHOLD.
    5. Apply edge-proximity checks to reduce false positives.
    6. Select a global crop row from valid detections.
    7. Save preview visualizations and optionally crop all images.

Usage:
    python artifact_templatematch.py
    python artifact_templatematch.py --crop

Configuration:
    Update INPUT_DIR, TEMPLATE_PATH, PREVIEW_DIR, OUTPUT_DIR, and threshold values
    in the config section below.

Requirements:
    pip install opencv-python numpy tqdm pillow matplotlib
"""

import cv2
import numpy as np
import os
import sys
import argparse
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Set, Generator
from PIL import Image, ImageFile
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Allow loading of partially corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Config ────────────────────────────────────────────────────────────────────

# Hard-coded dataset path - change this to your dataset location
INPUT_DIR = "watersplatting_data/Sunboat_03-09-2023"

# Path to the pre-cropped template image of the camera mount corner
TEMPLATE_PATH = "artifact_analysis/sunboat/template1.png"

# Output directories
PREVIEW_DIR = "artifact_analysis/sunboat/previews"
OUTPUT_DIR = "artifact_analysis/sunboat/cropped"

# Number of images to show in preview (first N images selected by seed)
# The first image will always be the highest detection, followed by additional samples
PREVIEW_SAMPLES = 9

# Random seed for deterministic sample selection
# Changing this will select different preview images
# Keep it constant to see the same images when changing detection configs
RANDOM_SEED = 42

# Restrict detection to bottom N% of image
BOTTOM_FRACTION = 0.25

# Restrict detection to rightmost N% of image
RIGHT_FRACTION = 0.20

# Minimum normalised cross-correlation score to accept a match (0.0 – 1.0).
# Raise to be more strict; lower to accept weaker matches.
MATCH_THRESHOLD = 0.4

# Safety padding subtracted from the global minimum detected row.
# The final crop row = min(detected_rows) - CROP_PADDING
CROP_PADDING = 0  # pixels

# Bottom-edge proximity gate: the bottom of the matched template bounding box
# (match_top_y + template_height) must be within this many pixels of the very
# bottom of the image.  Rejects matches that float too high in the frame and
# are therefore unlikely to be the mount bar touching the bottom edge.
BOTTOM_EDGE_PROXIMITY = 50  # pixels from image bottom

# Supported image extensions
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}

# ──────────────────────────────────────────────────────────────────────────────


def iter_dataset_images(
    root: str,
    extensions: Set[str] = IMG_EXTENSIONS,
) -> Generator[str, None, None]:
    """
    Recursively walk the dataset root, yielding all image files that match
    the given file extensions.

    Args:
        root: Root directory to search
        extensions: Set of file extensions to include (e.g., {'.png', '.jpg'})

    Yields:
        Absolute paths to image files, sorted for deterministic ordering
    """
    root = os.path.abspath(root)
    all_images = []

    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname)[1] in extensions:
                abs_path = os.path.join(dirpath, fname)
                all_images.append(abs_path)

    # Sort for reproducible ordering across runs
    all_images.sort()
    for path in all_images:
        yield path


# ── Worker-process global (template loaded once per worker via pool initializer) ──
_worker_template: np.ndarray = None


def _init_worker(template_path: str) -> None:
    """Pool initializer: load the template image into each worker process once."""
    global _worker_template
    _worker_template = cv2.imread(template_path, cv2.IMREAD_COLOR)


def detect_mount_row(image_path: str) -> dict:
    """
    Detect the camera mount top-left corner in a single image using template matching.
    The search is restricted to the bottom-right region defined by BOTTOM_FRACTION /
    RIGHT_FRACTION.  The top-left Y coordinate of the best match (in absolute image
    coordinates) is returned as detected_row.

    Returns a dict with:
      path, filename, detected_row, match_score, match_loc, error
    """
    result = {
        'path': image_path,
        'filename': os.path.basename(image_path),
        'detected_row': None,
        'match_score': None,
        'match_loc': None,
        'error': None,
    }

    try:
        template = _worker_template
        if template is None:
            result['error'] = 'Template not loaded in worker'
            return result

        th, tw = template.shape[:2]

        # Load image (PIL first for graceful handling of corrupted files)
        try:
            pil_img = Image.open(image_path)
            pil_img.verify()
            pil_img = Image.open(image_path)
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            img_bgr = cv2.imread(image_path)

        if img_bgr is None:
            result['error'] = 'Could not load image'
            return result

        h, w = img_bgr.shape[:2]

        # Define search region: bottom-right corner
        scan_top    = int(h * (1 - BOTTOM_FRACTION))
        right_start = int(w * (1 - RIGHT_FRACTION))
        region = img_bgr[scan_top:, right_start:]

        rh, rw = region.shape[:2]
        if rh < th or rw < tw:
            result['error'] = (
                f'Search region ({rw}x{rh}) smaller than template ({tw}x{th})'
            )
            return result

        # Run template matching
        match_map = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(match_map)

        if max_val < MATCH_THRESHOLD:
            result['error'] = (
                f'Match score too low ({max_val:.3f} < {MATCH_THRESHOLD})'
            )
            return result

        # max_loc is (col, row) of top-left in region coords
        match_col_region, match_row_region = max_loc
        abs_row = scan_top + match_row_region
        abs_col = right_start + match_col_region

        # Gate: bottom of the matched bounding box must be near the image bottom
        match_bottom = abs_row + th
        dist_from_bottom = h - match_bottom
        if dist_from_bottom > BOTTOM_EDGE_PROXIMITY:
            result['error'] = (
                f'Match bottom too far from image bottom '
                f'({dist_from_bottom}px > {BOTTOM_EDGE_PROXIMITY}px)'
            )
            return result

        result['detected_row'] = abs_row
        result['match_score']  = float(max_val)
        result['match_loc']    = (abs_col, abs_row)

    except Exception as e:
        result['error'] = str(e)

    return result


def create_match_analysis_image(
    image_path: str, crop_row: int, template: np.ndarray
) -> tuple:
    """
    Visualise the template-match result for a single image.

    Overlays on the original frame:
      - JET heatmap of the normalised match scores inside the search region
      - Green bounding box at the best match location (template size)
      - Red dot at the top-left corner of the match (the detected row point)
      - Cyan rectangle showing the search region boundary
      - Blue horizontal line at crop_row

    Returns:
      (vis_rgb, detection_point)  where detection_point is (col, row) or None.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, None

    h, w = img_bgr.shape[:2]
    th, tw = template.shape[:2]

    scan_top    = int(h * (1 - BOTTOM_FRACTION))
    right_start = int(w * (1 - RIGHT_FRACTION))
    region = img_bgr[scan_top:, right_start:]
    rh, rw = region.shape[:2]

    vis = img_bgr.copy()
    detection_point = None

    if rh >= th and rw >= tw:
        match_map = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(match_map)

        # ── Heatmap overlay on search region ──────────────────────────────
        norm_map = cv2.normalize(match_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap  = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        # Heatmap is (rh - th + 1) x (rw - tw + 1); pad to full region size
        pad_h = rh - heatmap.shape[0]
        pad_w = rw - heatmap.shape[1]
        heatmap_padded = cv2.copyMakeBorder(
            heatmap, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )
        vis[scan_top:, right_start:] = cv2.addWeighted(
            vis[scan_top:, right_start:], 0.5, heatmap_padded, 0.5, 0
        )

        # ── Best-match bounding box (green) ───────────────────────────────
        match_col_region, match_row_region = max_loc
        abs_col = right_start + match_col_region
        abs_row_match = scan_top + match_row_region
        cv2.rectangle(
            vis,
            (abs_col, abs_row_match),
            (abs_col + tw, abs_row_match + th),
            (0, 255, 0), 2
        )
        cv2.putText(
            vis, f'Score: {max_val:.3f}',
            (abs_col, max(abs_row_match - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )

        detection_point = (abs_col, abs_row_match)

        # ── Red dot at detection corner ────────────────────────────────────
        cv2.circle(vis, detection_point, 9, (0, 0, 255), -1)

    # ── Search region boundary (cyan) ─────────────────────────────────────
    cv2.rectangle(vis, (right_start, scan_top), (w, h), (0, 255, 255), 2)
    cv2.putText(vis, 'Search Region', (right_start + 5, scan_top + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ── Crop line (blue) ──────────────────────────────────────────────────
    cv2.line(vis, (0, crop_row), (w, crop_row), (255, 0, 0), 3)
    cv2.putText(vis, f'Crop Row: {crop_row}', (20, max(crop_row - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), detection_point


def save_sample_preview(
    results: list,
    output_path: str,
    crop_row: int,
    preview_paths: list,
    template: np.ndarray,
):
    """
    Save a preview showing pre-selected sample images.

    For each image, displays 3 panels:
    1. Original image
    2. Template-match heatmap + best-match bounding box
    3. Final cropped result (using the provided crop_row)

    The first image in the preview is always the highest detection (lowest row).
    The remaining images are from the pre-selected sample set (preview_paths).

    Args:
        results:       List of detection result dicts from detect_mount_row
        output_path:   Path to save the preview image
        crop_row:      The crop row to apply to all images
        preview_paths: List of pre-selected image paths to show in preview
        template:      The loaded template image (BGR numpy array)
    """
    # Filter successful detections
    successful = [r for r in results if r['detected_row'] is not None]
    if not successful:
        print("No successful detections to preview")
        return
    
    # Sort by detected_row (primary) and path (secondary) for deterministic ordering
    successful.sort(key=lambda x: (x['detected_row'], x['path']))
    
    # Start with the top detection (highest/lowest row number)
    top_detection = successful[0]
    sample_results = [top_detection]
    
    # Add the first X valid images from the shuffled list
    # Only include images that had a successful detection
    results_dict = {r['path']: r for r in results}
    for path in preview_paths:
        if len(sample_results) >= PREVIEW_SAMPLES:
            break
        if path == top_detection['path']:
            continue  # Already added as top detection
        r = results_dict.get(path)
        if r is None:
            continue
        if r['detected_row'] is None:
            continue  # Skip anything that didn't pass all detection rules
        sample_results.append(r)
    
    n_images = len(sample_results)
    fig, axes = plt.subplots(n_images, 3, figsize=(18, 5 * n_images))
    
    # Handle single image case
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(sample_results):
        image_path = result['path']
        detected_row = result['detected_row']
        filename = result['filename']
        
        try:
            # Load original image (already validated, but use safe loading)
            try:
                pil_img = Image.open(image_path)
                pil_img.verify()
                pil_img = Image.open(image_path)
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                img_bgr = cv2.imread(image_path)
            
            if img_bgr is None:
                # This shouldn't happen since we validated, but handle it anyway
                raise ValueError("Could not load image")
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Create template-match analysis visualization
            hue_vis, detection_point = create_match_analysis_image(image_path, crop_row, template)
            
            # Create cropped version
            cropped_rgb = img_rgb[:crop_row, :]
            
            # Panel 1: Original
            # Label the first image as "TOP DETECTION", others show detection status
            if idx == 0:
                title_prefix = "**TOP DETECTION**\n"
            elif detected_row is not None:
                title_prefix = f"Detected Row: {detected_row}\n"
            else:
                title_prefix = "NO DETECTION\n"
            axes[idx, 0].imshow(img_rgb)
            axes[idx, 0].set_title(f'{title_prefix}Original\n{filename}', fontsize=10)
            axes[idx, 0].axis('off')
            
            # Panel 2: Hue Analysis
            if hue_vis is not None:
                if detection_point is not None:
                    point_str = f'({detection_point[0]}, {detection_point[1]})'
                else:
                    point_str = 'N/A'
                axes[idx, 1].imshow(hue_vis)
                score_str = f'{result["match_score"]:.3f}' if result.get('match_score') is not None else 'N/A'
                if detected_row is not None:
                    axes[idx, 1].set_title(f'Match Heatmap  score={score_str}\nMatch loc: {point_str}, Crop: {crop_row}', fontsize=10)
                else:
                    axes[idx, 1].set_title(f'Match Heatmap  (no detection)\nCrop: {crop_row}', fontsize=10)
            else:
                axes[idx, 1].text(0.5, 0.5, 'Analysis N/A', ha='center', va='center')
                axes[idx, 1].set_title('Match Heatmap', fontsize=10)
            axes[idx, 1].axis('off')
            
            # Panel 3: Cropped Result
            axes[idx, 2].imshow(cropped_rgb)
            axes[idx, 2].set_title(f'Cropped Result\n{cropped_rgb.shape[0]}x{cropped_rgb.shape[1]} px', fontsize=10)
            axes[idx, 2].axis('off')
            
        except Exception as e:
            # This shouldn't happen since we pre-validated, but show error if it does
            print(f"Unexpected error visualizing {filename}: {e}")
            for col in range(3):
                axes[idx, col].text(0.5, 0.5, f'Error: {str(e)}', 
                                   ha='center', va='center', fontsize=8)
                axes[idx, col].axis('off')
    
    plt.suptitle(f'Template Match Preview: Top Detection + {n_images - 1} Seeded Samples  |  Crop Row: {crop_row}  |  Seed: {RANDOM_SEED}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nPreview saved to: {output_path}')


def crop_and_save(args):
    """Crop and save a single image, with error handling for corrupted files."""
    image_path, crop_row, output_dir = args
    try:
        # Use PIL first to handle corrupted images
        try:
            pil_img = Image.open(image_path)
            pil_img.verify()  # Verify it's a valid image
            pil_img = Image.open(image_path)  # Need to reopen after verify
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except:
            # Fallback to cv2
            img = cv2.imread(image_path)
        
        if img is None:
            print(f"Warning: Could not load {os.path.basename(image_path)}")
            return
        
        cropped = img[:crop_row, :]
        out_path = os.path.join(output_dir, os.path.basename(image_path))
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else output_dir, exist_ok=True)
        
        cv2.imwrite(out_path, cropped)
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")


def main():
    parser = argparse.ArgumentParser(
          description='Template-matching artifact boundary detection and cropping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
      python artifact_templatematch.py              # Preview only
      python artifact_templatematch.py --crop       # Preview + crop all images
        """)
    parser.add_argument('--crop', action='store_true', 
                       help='Crop all images based on highest detection (default: preview only)')
    args = parser.parse_args()

    # ── Load template ─────────────────────────────────────────────────────────
    if not os.path.isfile(TEMPLATE_PATH):
        print(f'ERROR: Template not found at {TEMPLATE_PATH}')
        print('Place a cropped reference image of the camera mount corner at that path.')
        sys.exit(1)
    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
    if template is None:
        print(f'ERROR: Could not read template image at {TEMPLATE_PATH}')
        sys.exit(1)
    th, tw = template.shape[:2]
    print(f'Template loaded: {TEMPLATE_PATH}  ({tw}x{th} px)')

    # ── Gather images recursively ────────────────────────────────────────────
    print(f'Searching for images in {INPUT_DIR} (recursive)...')
    all_files = list(iter_dataset_images(INPUT_DIR, IMG_EXTENSIONS))

    if not all_files:
        print(f'No images found in {INPUT_DIR}')
        sys.exit(1)

    print(f'Found {len(all_files)} images across all subdirectories\n')
    
    # ── Shuffle images using seed (before detection) ──────────────────────────
    # This ensures the same images are shown in previews when configs change
    print(f'Shuffling images for preview selection (seed={RANDOM_SEED})...')
    rng = np.random.RandomState(RANDOM_SEED)
    
    # Shuffle all images - preview function will pick first X valid ones
    shuffled_indices = rng.permutation(len(all_files))
    preview_paths = [all_files[i] for i in shuffled_indices]
    
    print(f'Will select first {PREVIEW_SAMPLES} valid images from shuffled list\\n')

    # ── Phase 1: Detect mount row in all images ───────────────────────────────
    print('Detecting mount boundary via template matching...')
    workers = max(1, cpu_count() - 1)
    with Pool(
        workers,
        initializer=_init_worker,
        initargs=(os.path.abspath(TEMPLATE_PATH),),
    ) as pool:
        results = list(tqdm(
            pool.imap(detect_mount_row, all_files),
            total=len(all_files),
            unit='img'
        ))

    # ── Phase 2: Analyse results ──────────────────────────────────────────────
    successful = [r for r in results if r['detected_row'] is not None]
    failed     = [r for r in results if r['detected_row'] is None]

    print(f'\nDetection complete:')
    print(f'  Successful: {len(successful)} / {len(results)}')
    print(f'  Failed:     {len(failed)} / {len(results)}')

    if failed:
        print(f'\nFailed images (check manually):')
        for r in failed[:20]:
            print(f'  {r["filename"]}: {r["error"]}')
        if len(failed) > 20:
            print(f'  ... and {len(failed) - 20} more')

    if not successful:
        print('No successful detections. Exiting.')
        sys.exit(1)

    # Get the single highest crop (lowest row number) minus safety padding
    successful.sort(key=lambda x: x['detected_row'])
    highest_crop_row = max(0, successful[0]['detected_row'] - CROP_PADDING)

    detected_rows = [r['detected_row'] for r in successful]
    median_row = float(np.median(detected_rows))
    std_row = float(np.std(detected_rows))

    print(f'\nDetection Statistics:')
    print(f'  Total successful: {len(successful)}')
    print(f'  Highest crop row: {highest_crop_row} (single highest detection)')
    print(f'  Median:           {median_row:.1f}')
    print(f'  Std:              {std_row:.1f}')

    # Generate timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Phase 2: Generate Preview ─────────────────────────────────────────────
    print(f'\nGenerating preview (top detection + {len(preview_paths)} seeded samples)...')
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    preview_path = os.path.join(PREVIEW_DIR, f'preview_{timestamp}.png')
    save_sample_preview(results, preview_path, highest_crop_row, preview_paths, template)

    if not args.crop:
        print(f'\nPreview complete. Check {PREVIEW_DIR}/ for results.')
        print(f'\nTo crop all images, re-run with: python artifact_templatematch.py --crop')
        return

    # ── Phase 3: Crop all images ──────────────────────────────────────────────
    print(f'\nCropping all images at row {highest_crop_row}...')
    output_dir_timestamped = os.path.join(OUTPUT_DIR, f'cropped_{timestamp}')
    os.makedirs(output_dir_timestamped, exist_ok=True)

    crop_args = [(r['path'], highest_crop_row, output_dir_timestamped) for r in results]
    workers = max(1, cpu_count() - 1)
    with Pool(workers) as pool:
        list(tqdm(
            pool.imap(crop_and_save, crop_args),
            total=len(crop_args),
            unit='img'
        ))

    print(f'\nDone!')
    print(f'  Preview: {preview_path}')
    print(f'  Cropped images: {output_dir_timestamped}/')
    print(f'  Crop row used: {highest_crop_row} (single highest detection)')


if __name__ == '__main__':
    main()