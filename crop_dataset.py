"""
Color-Band Artifact Boundary Detection and Cropping.

This script scans an image dataset for a persistent color-band artifact near
the image border (for example, camera housing or platform structure), then
derives a global crop row from the highest valid detection.

Method summary:
    1. Recursively discover images from INPUT_DIR.
    2. Restrict detection to a configurable border region.
    3. Build a hue-range mask and run connected-components.
    4. Keep the largest component and validate it with:
             - proximity to the expected border
             - cluster concentration (dominant component check)
    5. Aggregate detections and choose a single crop row.
    6. Save a preview panel, and optionally crop all images.

Usage:
    python crop_dataset.py
    python crop_dataset.py --crop

Configuration:
    Update INPUT_DIR, PREVIEW_DIR, OUTPUT_DIR, and detection thresholds in the
    config section below.

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

# Output directories
PREVIEW_DIR = "artifact_analysis/sunboat/previews"
OUTPUT_DIR = "artifact_analysis/sunboat/cropped"

# Number of top detections to use for crop row (use highest from this many)
TOP_N = 1

# Number of images to show in preview (first N images selected by seed)
# The first image will always be the highest detection, followed by additional samples
PREVIEW_SAMPLES = 9

# Random seed for deterministic sample selection
# Changing this will select different preview images
# Keep it constant to see the same images when changing detection configs
RANDOM_SEED = 42

# Hue range for yellow-green band (OpenCV hue: 0-179) 
HUE_LOW  = 25 ## dont change this is pretested 
HUE_HIGH = 75

# Restrict detection to bottom N% of image
BOTTOM_FRACTION = 0.25

# Restrict detection to rightmost N% of image
RIGHT_FRACTION = 0.20

# Minimum component area to be considered the mount (filters tiny noise)
MIN_COMPONENT_AREA = 500

# Right-edge proximity threshold: highest point must be within this many pixels
# of the right edge to be considered the mount (filters scene content false positives).
# The camera mount's top corner always touches near the right side, whereas false
# positives from objects (wreck, seafloor, algae) have highest points further left.
RIGHT_EDGE_PROXIMITY = 50  # pixels from right edge

# Cluster concentration threshold: the largest yellow-green component must account
# for at least this fraction of ALL yellow-green pixels in the scan zone.
# A real camera mount is one tight bar dominating the region (concentration ~0.8+).
# Contaminated images (algae, seafloor) spread yellow-green pixels across many
# small clusters evenly, giving a low concentration ratio.
# Both proximity and concentration checks must pass for a valid detection.
MIN_CLUSTER_CONCENTRATION = 0.60  # largest component must own >= 60% of all yellow-green pixels

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


def detect_mount_row(image_path: str) -> dict:
    """
    Detects the highest row of the camera mount in a single image.
    Returns a dict with detection results.
    """
    result = {
        'path': image_path,
        'filename': os.path.basename(image_path),
        'detected_row': None,
        'component_area': None,
        'n_components': None,
        'concentration': None,
        'error': None,
    }

    try:
        # Try PIL first for better corrupted image handling
        try:
            pil_img = Image.open(image_path)
            pil_img.verify()
            pil_img = Image.open(image_path)  # Reopen after verify
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except:
            # Fallback to cv2
            img_bgr = cv2.imread(image_path)
        
        if img_bgr is None:
            result['error'] = 'Could not load image'
            return result

        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Restrict to bottom-right region
        scan_top   = int(h * (1 - BOTTOM_FRACTION))
        right_start = int(w * (1 - RIGHT_FRACTION))
        region_hsv = hsv[scan_top:, right_start:]

        H = region_hsv[:, :, 0]

        # Yellow-green hue mask
        mask = ((H >= HUE_LOW) & (H <= HUE_HIGH)).astype(np.uint8)

        # Connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        result['n_components'] = num_labels - 1

        if num_labels < 2:
            result['error'] = 'No components found'
            return result

        # Filter components below minimum area
        valid = [(lid, stats[lid, cv2.CC_STAT_AREA])
                 for lid in range(1, num_labels)
                 if stats[lid, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA]

        if not valid:
            result['error'] = f'No components above min area ({MIN_COMPONENT_AREA}px)'
            return result

        # Largest valid component = the mount
        largest_label, largest_area = max(valid, key=lambda x: x[1])
        largest_mask = (labels == largest_label)

        # Highest row of this component
        rows_found = np.where(largest_mask.any(axis=1))[0]
        highest_in_region = int(rows_found.min())
        
        # Find the column position(s) of the highest point
        cols_at_highest = np.where(largest_mask[highest_in_region, :])[0]
        if len(cols_at_highest) == 0:
            result['error'] = 'No columns found at highest row'
            return result
        
        # Get the rightmost column of the highest point (closest to right edge)
        rightmost_col_in_region = int(cols_at_highest.max())
        # Convert to absolute coordinates
        rightmost_col_abs = right_start + rightmost_col_in_region
        
        #### Clustering Rule 1: Check proximity to right edge
        distance_from_right = w - rightmost_col_abs
        if distance_from_right > RIGHT_EDGE_PROXIMITY:
            result['error'] = f'Highest point too far from right edge ({distance_from_right}px > {RIGHT_EDGE_PROXIMITY}px)'
            return result

        # ### Clustering Rule 2: Check cluster concentration
        # The largest component must dominate the yellow-green pixels in the scan zone.
        # Evenly spread pixels = contamination; one tight cluster = camera mount.
        total_yellow_green = int(np.sum(mask))
        concentration = largest_area / total_yellow_green if total_yellow_green > 0 else 0.0
        if concentration < MIN_CLUSTER_CONCENTRATION:
            result['error'] = f'Cluster too diffuse ({concentration:.1%} < {MIN_CLUSTER_CONCENTRATION:.1%}) — yellow-green pixels spread across scan zone'
            return result
        
        abs_row = scan_top + highest_in_region

        result['detected_row']   = abs_row
        result['component_area'] = int(largest_area)
        result['concentration']  = float(concentration)




    except Exception as e:
        result['error'] = str(e)

    return result


def create_hue_analysis_image(image_path: str, crop_row: int) -> tuple:
    """
    Create a visualization of the hue analysis showing the yellow-green detection.
    Returns a tuple of (visualization as RGB numpy array, detection point coordinates).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, None

    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Restrict to bottom-right region (same as detection)
    scan_top = int(h * (1 - BOTTOM_FRACTION))
    right_start = int(w * (1 - RIGHT_FRACTION))
    region_hsv = hsv[scan_top:, right_start:]
    
    # Create visualization
    vis = img_bgr.copy()
    
    # Highlight the yellow-green hue region
    H = hsv[:, :, 0]
    mask = ((H >= HUE_LOW) & (H <= HUE_HIGH)).astype(np.uint8) * 255
    
    # Re-run detection to find the exact highest point location
    region_H = region_hsv[:, :, 0]
    region_mask = ((region_H >= HUE_LOW) & (region_H <= HUE_HIGH)).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region_mask, connectivity=8)
    
    detection_point = None
    if num_labels >= 2:
        # Find largest component above minimum area
        valid = [(lid, stats[lid, cv2.CC_STAT_AREA])
                 for lid in range(1, num_labels)
                 if stats[lid, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA]
        
        if valid:
            largest_label, _ = max(valid, key=lambda x: x[1])
            largest_mask = (labels == largest_label)
            
            # Find the highest row in this component
            rows_with_component = np.where(largest_mask.any(axis=1))[0]
            if len(rows_with_component) > 0:
                highest_row_in_region = int(rows_with_component.min())
                # Find columns in that row that have the component
                cols_in_highest_row = np.where(largest_mask[highest_row_in_region, :])[0]
                if len(cols_in_highest_row) > 0:
                    # Use the RIGHTMOST column (closest to right edge) - this is what proximity check uses
                    rightmost_col_in_region = int(cols_in_highest_row.max())
                    # Convert back to absolute coordinates
                    detection_point = (right_start + rightmost_col_in_region, scan_top + highest_row_in_region)
    
    # Color the detected region in green overlay
    green_overlay = np.zeros_like(vis)
    green_overlay[:, :, 1] = mask  # Green channel
    vis = cv2.addWeighted(vis, 0.7, green_overlay, 0.3, 0)
    
    # Draw region boundaries
    cv2.rectangle(vis, (right_start, scan_top), (w, h), (0, 255, 255), 2)
    
    # Draw the detected crop line
    cv2.line(vis, (0, crop_row), (w, crop_row), (255, 0, 0), 3)
    
    # Draw large red dot at detection point
    if detection_point is not None:
        cv2.circle(vis, detection_point, 9, (0, 0, 255), -1)  # Filled red circle    
        
    # Add labels
    cv2.putText(vis, 'Scan Region', (right_start + 5, scan_top + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, f'Crop Row: {crop_row}', (20, crop_row - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), detection_point


def save_sample_preview(results: list, output_path: str, crop_row: int, preview_paths: list):
    """
    Save a preview showing pre-selected sample images.
    
    For each image, displays 3 panels:
    1. Original image
    2. Hue analysis visualization (showing detection regions)
    3. Final cropped result (using the provided crop_row)
    
    The first image in the preview is always the highest detection (lowest row).
    The remaining images are from the pre-selected sample set (preview_paths).
    
    Args:
        results: List of detection result dicts from detect_mount_row
        output_path: Path to save the preview image
        crop_row: The crop row to apply to all images
        preview_paths: List of pre-selected image paths to show in preview
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
            
            # Create hue analysis visualization
            hue_vis, detection_point = create_hue_analysis_image(image_path, crop_row)
            
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
                # Format detection point for display
                if detection_point is not None:
                    point_str = f'({detection_point[0]}, {detection_point[1]})'
                else:
                    point_str = 'N/A'
                axes[idx, 1].imshow(hue_vis)
                if detected_row is not None:
                    axes[idx, 1].set_title(f'Hue Analysis\nHighest Pixel: {point_str}, Crop: {crop_row}', fontsize=10)
                else:
                    axes[idx, 1].set_title(f'Hue Analysis\nNo Detection - Crop: {crop_row}', fontsize=10)
            else:
                axes[idx, 1].text(0.5, 0.5, 'Analysis N/A', ha='center', va='center')
                axes[idx, 1].set_title('Hue Analysis', fontsize=10)
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
    
    plt.suptitle(f'Preview: Top Detection + {n_images - 1} Seeded Samples (Crop Row: {crop_row}, Seed: {RANDOM_SEED})', fontsize=14, y=0.995)
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
          description='Color-band artifact boundary detection and cropping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
      python crop_dataset.py              # Preview only
      python crop_dataset.py --crop       # Preview + crop all images
        """)
    parser.add_argument('--crop', action='store_true', 
                       help='Crop all images based on highest detection (default: preview only)')
    args = parser.parse_args()

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
    print('Detecting mount boundary...')
    workers = max(1, cpu_count() - 1)
    with Pool(workers) as pool:
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

    # Get the single highest crop (lowest row number)
    successful.sort(key=lambda x: x['detected_row'])
    highest_crop_row = successful[0]['detected_row']

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
    save_sample_preview(results, preview_path, highest_crop_row, preview_paths)

    if not args.crop:
        print(f'\nPreview complete. Check {PREVIEW_DIR}/ for results.')
        print(f'\nTo crop all images, re-run with: python crop_dataset.py --crop')
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