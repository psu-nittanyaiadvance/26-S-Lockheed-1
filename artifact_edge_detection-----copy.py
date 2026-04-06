"""
Edge-Based Artifact Boundary Detector.

Detects the inner boundary of a border-connected object (artifact or structure)
protruding from one side of an image. The detector returns boundary points and
supports overlay generation for visual verification.

Dependencies:
    opencv-python, numpy, scipy, matplotlib

Library usage:
    Import detect_edge_object(gray_image) and consume result['edge_points'].

CLI usage:
    python artifact_edge_detection-----copy.py --image path/to/image.png
"""

import argparse
import os
from datetime import datetime

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks


# Local testing overrides. Set TEST_IMAGE_PATH to run without --image.
TEST_IMAGE_PATH = "artifact_analysis/sunboat/sunboat_mean_image.png"
OUTPUT_DIR = os.path.join("artifact_analysis", "sunboat", "edge_detection")

# Tunable defaults for edge detection
DEFAULT_EDGE = "auto"
DEFAULT_SEARCH_FRAC = 0.25
DEFAULT_USE_SEARCH_BOUND = False
DEFAULT_CHUNK_SIZE = 10
DEFAULT_SOBEL_KSIZE = 5
DEFAULT_SMOOTH_SIGMA = 3.0
DEFAULT_PEAK_HEIGHT = 0.12
DEFAULT_PEAK_DISTANCE = 15
DEFAULT_REJECT_OUTLIERS = False
DEFAULT_OUTLIER_WINDOW = 11
DEFAULT_OUTLIER_THRESHOLD = 15

# Output formatting defaults
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
HEADER_HEIGHT = 74


def detect_edge_object(gray, edge=DEFAULT_EDGE, search_frac=DEFAULT_SEARCH_FRAC,
                       use_search_bound=DEFAULT_USE_SEARCH_BOUND, chunk_size=DEFAULT_CHUNK_SIZE,
                       sobel_ksize=DEFAULT_SOBEL_KSIZE, smooth_sigma=DEFAULT_SMOOTH_SIGMA,
                       peak_height=DEFAULT_PEAK_HEIGHT, peak_distance=DEFAULT_PEAK_DISTANCE,
                       reject_outliers=DEFAULT_REJECT_OUTLIERS,
                       outlier_window=DEFAULT_OUTLIER_WINDOW, outlier_threshold=DEFAULT_OUTLIER_THRESHOLD):
    """
    Detect the innermost boundary of an object protruding from an image edge.

    Parameters
    ----------
    gray : ndarray (H, W)
        Grayscale image.
    edge : str
        Which edge to check: 'top', 'bottom', 'left', 'right', or 'auto'.
    search_frac : float
        Fraction of image to search inward from the edge (default 0.25).
    use_search_bound : bool
        If True, only search the bottom search_frac of the rotated image.
        If False, search the full rotated image height.
    chunk_size : int
        Width of vertical slices for averaging the Sobel response (default 10).
    sobel_ksize : int
        Sobel kernel size (default 5).
    smooth_sigma : float
        Gaussian smoothing on gradient profiles (default 3).
    peak_height : float
        Minimum gradient magnitude to register as an edge (default 0.12).
    peak_distance : int
        Minimum spacing between detected peaks in pixels (default 15).
    reject_outliers : bool
        If True, apply local median filter to remove outlier detections
        (default False).
    outlier_window : int
        Median filter window size for outlier rejection (default 11).
        Only used when reject_outliers=True.
    outlier_threshold : int
        Max deviation from local median in pixels before a point is
        rejected (default 15). Only used when reject_outliers=True.

    Returns
    -------
    dict with:
        detected_edge : str
        edge_points : tuple (xs, ys) in original image coordinates
        n_points : int
        n_rejected : int (only present when reject_outliers=True)
    """
    H, W = gray.shape

    # ---- Step 1: Auto-detect which edge ----
    if edge == 'auto':
        edge = _auto_detect_edge(gray)

    # ---- Step 2: Rotate so object is at the bottom ----
    rot_k = {'bottom': 0, 'right': 1, 'top': 2, 'left': 3}[edge]
    work = np.rot90(gray, k=rot_k)
    wH, wW = work.shape
    search_start = int(wH * (1 - search_frac)) if use_search_bound else 0

    # ---- Step 3: Sobel gradient ----
    sobel = cv2.Sobel(work.astype(float), cv2.CV_64F, 0, 1, ksize=sobel_ksize)

    # ---- Step 4: Chunked peak detection ----
    strongest_e, strongest_c = [], []
    all_e, all_c = [], []

    for cs in range(0, wW, chunk_size):
        ce = min(cs + chunk_size, wW)
        cc = (cs + ce) // 2
        chunk = sobel[search_start:, cs:ce].mean(axis=1)
        smooth = gaussian_filter1d(chunk, sigma=smooth_sigma)
        peaks, props = find_peaks(smooth, height=peak_height,
                                  distance=peak_distance)
        if len(peaks) > 0:
            best = np.argmax(props['peak_heights'])
            strongest_e.append(peaks[best] + search_start)
            strongest_c.append(cc)
            for p in peaks:
                all_e.append(p + search_start)
                all_c.append(cc)

    strongest_e = np.array(strongest_e)
    strongest_c = np.array(strongest_c)
    all_e = np.array(all_e)
    all_c = np.array(all_c)

    if len(strongest_e) == 0:
        return {'detected_edge': edge,
                'edge_points': (np.array([]), np.array([])),
                'n_points': 0}

    # ---- Step 5: Adaptive band isolation ----
    mf = median_filter(strongest_e,
                       size=min(11, len(strongest_e)))
    bad_frac = (np.abs(strongest_e - mf) > 30).mean()

    if bad_frac > 0.3:
        # Bimodal — isolate the innermost band from all peaks
        det_e, det_c = _isolate_innermost_band(all_e, all_c)

        # Deduplicate: median per column slice
        unique_cols = sorted(set(det_c.tolist()))
        dedup_e, dedup_c = [], []
        for col in unique_cols:
            mask = det_c == col
            dedup_e.append(np.median(det_e[mask]))
            dedup_c.append(col)
        det_e = np.array(dedup_e)
        det_c = np.array(dedup_c)
    else:
        # Single band — use strongest peak per slice
        det_e = strongest_e
        det_c = strongest_c

    # ---- Step 5b: Optional outlier rejection ----
    n_before = len(det_e)
    n_rejected = 0

    if reject_outliers and len(det_e) > outlier_window:
        mf_clean = median_filter(det_e, size=outlier_window)
        clean_mask = np.abs(det_e - mf_clean) < outlier_threshold
        det_e = det_e[clean_mask]
        det_c = det_c[clean_mask]
        n_rejected = int(n_before - len(det_e))

    # ---- Step 6: Map back to original coordinates ----
    orig_x, orig_y = _map_coords(det_c, det_e, rot_k, wH, wW)

    out = {
        'detected_edge': edge,
        'edge_points': (orig_x, orig_y),
        'n_points': len(det_e),
    }
    if reject_outliers:
        out['n_rejected'] = n_rejected

    return out


def render_overlay(rgb, result):
    """Render an RGB overlay image with detection points drawn on top."""
    overlay_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    px, py = result['edge_points']
    for x_val, y_val in zip(px, py):
        x = int(round(float(x_val)))
        y = int(round(float(y_val)))
        if 0 <= x < overlay_bgr.shape[1] and 0 <= y < overlay_bgr.shape[0]:
            cv2.circle(overlay_bgr, (x, y), 4, (0, 255, 0), thickness=-1)
            cv2.circle(overlay_bgr, (x, y), 5, (0, 0, 0), thickness=1)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


def save_overlay(rgb, result, output_path):
    """
    Save an overlay image with detection points for human verification.

    Parameters
    ----------
    rgb : ndarray (H, W, 3)
        Original image in RGB.
    result : dict
        Output from detect_edge_object().
    output_path : str
        File path to save the overlay image.
    """
    overlay_rgb = render_overlay(rgb, result)
    cv2.imwrite(output_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))


def write_detection_report(output_path, result, reject_outliers, image_path=None):
    """Write metadata and detection points into a single CSV-style report file."""
    px, py = result['edge_points']
    with open(output_path, 'w', encoding='utf-8') as handle:
        if image_path is not None:
            handle.write(f"# image: {image_path}\n")
        handle.write(f"# detected_edge: {result['detected_edge']}\n")
        handle.write(f"# n_points: {result['n_points']}\n")
        handle.write(f"# reject_outliers: {str(bool(reject_outliers)).lower()}\n")
        if reject_outliers:
            handle.write(f"# n_rejected: {result.get('n_rejected', 0)}\n")
        handle.write("x,y\n")
        for x_val, y_val in zip(px, py):
            handle.write(f"{float(x_val):.3f},{float(y_val):.3f}\n")


# =====================================================================
# Internal helpers
# =====================================================================

def _auto_detect_edge(gray):
    """Score each edge by gradient strength + std deviation."""
    H, W = gray.shape
    ef = 0.15

    row_grad = np.abs(np.gradient(
        gaussian_filter1d(gray.mean(axis=1).astype(float), sigma=5)))
    col_grad = np.abs(np.gradient(
        gaussian_filter1d(gray.mean(axis=0).astype(float), sigma=5)))
    row_std = gray.astype(float).std(axis=1)
    col_std = gray.astype(float).std(axis=0)

    metrics = {
        'top':    (row_grad[:int(H*ef)].max(), row_std[:int(H*ef)].max()),
        'bottom': (row_grad[int(H*(1-ef)):].max(), row_std[int(H*(1-ef)):].max()),
        'left':   (col_grad[:int(W*ef)].max(), col_std[:int(W*ef)].max()),
        'right':  (col_grad[int(W*(1-ef)):].max(), col_std[int(W*(1-ef)):].max()),
    }

    g_vals = [v[0] for v in metrics.values()]
    s_vals = [v[1] for v in metrics.values()]
    g_range = max(g_vals) - min(g_vals) + 1e-10
    s_range = max(s_vals) - min(s_vals) + 1e-10

    scores = {e: (g - min(g_vals)) / g_range + (s - min(s_vals)) / s_range
              for e, (g, s) in metrics.items()}

    return max(scores, key=scores.get)


def _isolate_innermost_band(edges, cols):
    """When multiple gradient bands exist, keep only the innermost."""
    if len(edges) < 5:
        return edges, cols

    n_bins = max(20, len(edges) // 5)
    hist, bin_edges = np.histogram(edges, bins=n_bins)
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)
    hist_peaks, _ = find_peaks(hist_smooth, height=2, distance=2)

    if len(hist_peaks) == 0:
        threshold = np.percentile(edges, 25)
        mask = edges < threshold
        return edges[mask], cols[mask]

    # Innermost band = smallest row value
    band_center = (bin_edges[hist_peaks[0]] + bin_edges[hist_peaks[0] + 1]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    band_radius = max(bin_width * 3, 40)

    mask = np.abs(edges - band_center) < band_radius

    if mask.sum() < 10 and len(hist_peaks) > 1:
        next_center = (bin_edges[hist_peaks[1]] + bin_edges[hist_peaks[1] + 1]) / 2
        band_radius = abs(next_center - band_center) / 2
        mask = np.abs(edges - band_center) < band_radius

    return edges[mask], cols[mask]


def _map_coords(work_cols, work_rows, rot_k, wH, wW):
    """Map (col, row) from rot90(k) work space to original image."""
    if rot_k == 0: return work_cols.copy(), work_rows.copy()
    if rot_k == 1: return work_rows.copy(), (wW - 1 - work_cols)
    if rot_k == 2: return (wW - 1 - work_cols), (wH - 1 - work_rows)
    if rot_k == 3: return (wH - 1 - work_rows), work_cols.copy()


def run_side_by_side_comparison(
    image_path,
    edge=DEFAULT_EDGE,
    search_frac=DEFAULT_SEARCH_FRAC,
    use_search_bound=DEFAULT_USE_SEARCH_BOUND,
    chunk_size=DEFAULT_CHUNK_SIZE,
    sobel_ksize=DEFAULT_SOBEL_KSIZE,
    smooth_sigma=DEFAULT_SMOOTH_SIGMA,
    peak_height=DEFAULT_PEAK_HEIGHT,
    peak_distance=DEFAULT_PEAK_DISTANCE,
    outlier_window=DEFAULT_OUTLIER_WINDOW,
    outlier_threshold=DEFAULT_OUTLIER_THRESHOLD,
):
    """Run edge detection with outlier rejection off/on and save side-by-side outputs."""
    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    stem = os.path.splitext(os.path.basename(image_path))[0]
    output_root = os.path.abspath(OUTPUT_DIR)
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    output_dir = os.path.join(output_root, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        img_bgr = img
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    common_kwargs = {
        'edge': edge,
        'search_frac': search_frac,
        'use_search_bound': use_search_bound,
        'chunk_size': chunk_size,
        'sobel_ksize': sobel_ksize,
        'smooth_sigma': smooth_sigma,
        'peak_height': peak_height,
        'peak_distance': peak_distance,
        'outlier_window': outlier_window,
        'outlier_threshold': outlier_threshold,
    }

    result_off = detect_edge_object(gray, reject_outliers=False, **common_kwargs)
    result_on = detect_edge_object(gray, reject_outliers=True, **common_kwargs)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay_off = cv2.cvtColor(render_overlay(rgb, result_off), cv2.COLOR_RGB2BGR)
    overlay_on = cv2.cvtColor(render_overlay(rgb, result_on), cv2.COLOR_RGB2BGR)

    if overlay_off.shape[0] != overlay_on.shape[0]:
        target_h = min(overlay_off.shape[0], overlay_on.shape[0])
        overlay_off = cv2.resize(
            overlay_off,
            (int(overlay_off.shape[1] * target_h / overlay_off.shape[0]), target_h),
            interpolation=cv2.INTER_AREA,
        )
        overlay_on = cv2.resize(
            overlay_on,
            (int(overlay_on.shape[1] * target_h / overlay_on.shape[0]), target_h),
            interpolation=cv2.INTER_AREA,
        )

    panel = np.hstack([overlay_off, overlay_on])
    header = np.full((HEADER_HEIGHT, panel.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(
        header,
        f"Outlier OFF: points={result_off['n_points']}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"Outlier ON: points={result_on['n_points']}, rejected={result_on.get('n_rejected', 0)}",
        (panel.shape[1] // 2 + 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "Left: outlier rejection OFF | Right: outlier rejection ON",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    panel_path = os.path.join(output_dir, f"{stem}_overlay_side_by_side.png")
    cv2.imwrite(panel_path, np.vstack([header, panel]))

    report_off_path = os.path.join(output_dir, f"{stem}_report_outlier_off.csv")
    report_on_path = os.path.join(output_dir, f"{stem}_report_outlier_on.csv")
    write_detection_report(report_off_path, result_off, reject_outliers=False, image_path=image_path)
    write_detection_report(report_on_path, result_on, reject_outliers=True, image_path=image_path)

    return {
        'output_dir': output_dir,
        'panel': panel_path,
        'report_off': report_off_path,
        'report_on': report_on_path,
        'result_off': result_off,
        'result_on': result_on,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run edge detection on one mean image and save outlier OFF/ON side-by-side comparison",
    )
    parser.add_argument('--image', default=None, help='Path to input mean image')
    parser.add_argument('--edge', choices=['auto', 'top', 'bottom', 'left', 'right'], default=DEFAULT_EDGE)
    parser.add_argument('--search-frac', type=float, default=DEFAULT_SEARCH_FRAC)
    search_group = parser.add_mutually_exclusive_group()
    search_group.add_argument(
        '--use-search-bound',
        dest='use_search_bound',
        action='store_true',
        help='Restrict peak search to the bottom search-frac region of the rotated image',
    )
    search_group.add_argument(
        '--no-search-bound',
        dest='use_search_bound',
        action='store_false',
        help='Search the full rotated image height for peaks',
    )
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument('--sobel-ksize', type=int, default=DEFAULT_SOBEL_KSIZE)
    parser.add_argument('--smooth-sigma', type=float, default=DEFAULT_SMOOTH_SIGMA)
    parser.add_argument('--peak-height', type=float, default=DEFAULT_PEAK_HEIGHT)
    parser.add_argument('--peak-distance', type=int, default=DEFAULT_PEAK_DISTANCE)
    parser.add_argument('--outlier-window', type=int, default=DEFAULT_OUTLIER_WINDOW)
    parser.add_argument('--outlier-threshold', type=int, default=DEFAULT_OUTLIER_THRESHOLD)
    parser.set_defaults(use_search_bound=DEFAULT_USE_SEARCH_BOUND)
    args = parser.parse_args()

    image_path = args.image or TEST_IMAGE_PATH
    if image_path is None:
        parser.error('Provide --image or set TEST_IMAGE_PATH in edge_detection.py')

    outputs = run_side_by_side_comparison(
        image_path=image_path,
        edge=args.edge,
        search_frac=args.search_frac,
        use_search_bound=args.use_search_bound,
        chunk_size=args.chunk_size,
        sobel_ksize=args.sobel_ksize,
        smooth_sigma=args.smooth_sigma,
        peak_height=args.peak_height,
        peak_distance=args.peak_distance,
        outlier_window=args.outlier_window,
        outlier_threshold=args.outlier_threshold,
    )

    print("Saved outputs:")
    print(f"  output dir: {outputs['output_dir']}")
    print(f"  panel:      {outputs['panel']}")
    print(f"  report off: {outputs['report_off']}")
    print(f"  report on:  {outputs['report_on']}")
    print(
        f"Results: off={outputs['result_off']['n_points']} points, "
        f"on={outputs['result_on']['n_points']} points, "
        f"rejected={outputs['result_on'].get('n_rejected', 0)}"
    )


if __name__ == '__main__':
    main()