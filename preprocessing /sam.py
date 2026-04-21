"""
SAM-based persistent artifact segmentation on a median image.

Designed for underwater ROV/submersible footage where parts of the vehicle
(mounts, housings, brackets, light rigs) intrude into the frame and persist
across all scenes.

Pipeline:
  1) Load median image produced by frame averaging/layering.
  2) Validate and normalize to uint8 RGB.
  3) Run Segment Anything automatic mask generation.
  4) Post-filter masks: remove background/scene masks, keep only likely
     equipment intrusions based on spatial, shape, and intensity heuristics.
  5) Export binary masks for predicted persistent artifacts.
  6) Optionally cross-check with a variance threshold mask.

Filtering strategy:
  - Edge-proximity filter: persistent hardware intrusions almost always
    appear at frame edges (camera housing, mounts, light rigs), not floating
    in the center of the frame.
  - Solidity filter: real equipment has solid, non-wispy shapes. Removes
    fan-shaped light artifacts and gradient hallucinations from SAM.
  - Intensity-contrast filter: intrusions typically differ in intensity
    from the surrounding water/scene in the median image.
  - Size filter: reject masks that are too large (scene background) or
    too small (noise).

Example:
  python sam_improved.py \
    --median-image artifact_analysis/sunboat/sunboat_mean_image.png \
    --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
    --sam-model-type vit_h \
    --variance-map artifact_analysis/sunboat/variance_map.npy \
    --variance-threshold 0.0075
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


# ===================================================================
# CONFIGURATION — tune these to adjust detection behavior
# ===================================================================

# --- File paths ---
DEFAULT_MEDIAN_IMAGE = "artifact_analysis/sunboat/sunboat_mean_image.png"
DEFAULT_OUTPUT_DIR = "artifact_analysis/sam_outputs"

# --- Tune these per dataset ------------------------------------------------
# Artifact size and visibility vary a lot between cameras/deployments.

DEFAULT_MAX_AREA_FRACTION = 0.4               # max fraction of image a single mask
                                              # can cover; rejects background segments
                                              # raise if equipment covers a large part
                                              # of frame, lower to be more strict

DEFAULT_MIN_AREA_FRACTION = 0.0005            # min fraction of image; rejects noise
                                              # lower if you have small brackets/bolts
                                              # raise if getting too many tiny detections

DEFAULT_MIN_INTENSITY_CONTRAST = 2.0          # min abs intensity difference between
                                              # mask region and background
                                              # lower for dark-on-dark (deep water)
                                              # raise if getting false positives in
                                              # uniform water regions

# --- Stable defaults (rarely need changing) --------------------------------
DEFAULT_SAM_CHECKPOINT = "artifact_analysis/sam/sam_vit_h_4b8939.pth"
DEFAULT_SAM_MODEL_TYPE = "vit_h"
DEFAULT_DEVICE = "cpu"
DEFAULT_POINTS_PER_SIDE = 32
DEFAULT_PRED_IOU_THRESH = 0.86
DEFAULT_STABILITY_SCORE_THRESH = 0.92
DEFAULT_MIN_MASK_REGION_AREA = 100
DEFAULT_MAX_EXPORTED_MASKS = 200
DEFAULT_MIN_SOLIDITY = 0.4                    # equipment is solid; 0.4 is generous
DEFAULT_MIN_EDGE_TOUCH = 0.1                  # equipment intrudes from frame edges
DEFAULT_EDGE_MARGIN_FRACTION = 0.05           # 5% border zone width

# ===================================================================


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------
@dataclass
class PipelineState:
    median_rgb_u8: Optional[np.ndarray] = None
    sam_masks: Optional[List[Dict[str, Any]]] = None
    filtered_masks: Optional[List[Dict[str, Any]]] = None
    union_mask: Optional[np.ndarray] = None
    filter_report: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def log_step(step_num: int, total_steps: int, message: str) -> None:
    print(f"\nStep {step_num}/{total_steps}: {message}")
    print("-" * 60)


def ensure_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


# ---------------------------------------------------------------------------
# Image loading / normalization
# ---------------------------------------------------------------------------
def normalize_image_to_u8_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(
            f"Expected image shape [H, W, 3/4] or [H, W], got {image.shape}"
        )

    if image.shape[2] == 4:
        image = image[:, :, :3]

    img = image.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)

    max_val = float(np.max(img))
    min_val = float(np.min(img))

    if max_val <= 1.0 and min_val >= 0.0:
        img = img * 255.0
    elif max_val <= 255.0 and min_val >= 0.0:
        pass
    else:
        denom = max(max_val - min_val, 1e-8)
        img = (img - min_val) / denom * 255.0

    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def load_median_image(path: str) -> np.ndarray:
    ensure_exists(path, "Median image")
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        return normalize_image_to_u8_rgb(np.load(path))

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image file: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# SAM model
# ---------------------------------------------------------------------------
def build_sam_generator(
    checkpoint: str,
    model_type: str,
    device: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    min_mask_region_area: int,
):
    try:
        from segment_anything import (  # pyright: ignore[reportMissingImports]
            SamAutomaticMaskGenerator,
            sam_model_registry,
        )
    except ImportError as exc:
        raise ImportError(
            "segment_anything is not installed. "
            "pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc

    ensure_exists(checkpoint, "SAM checkpoint")
    if model_type not in ("vit_h", "vit_l", "vit_b"):
        raise ValueError("--sam-model-type must be one of: vit_h, vit_l, vit_b")

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)

    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )


# ---------------------------------------------------------------------------
# Mask shape analysis
# ---------------------------------------------------------------------------
def compute_mask_properties(
    mask_dict: Dict[str, Any],
    image_h: int,
    image_w: int,
    median_gray: np.ndarray,
    edge_margin_frac: float = DEFAULT_EDGE_MARGIN_FRACTION,
) -> Dict[str, Any]:
    """Compute geometric and photometric properties for a single SAM mask."""
    seg = mask_dict["segmentation"].astype(np.uint8)
    area = int(mask_dict.get("area", int(seg.sum())))

    # --- Geometric properties ---
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"valid": False}

    largest = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest)

    # Solidity = area / convex hull area
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / max(hull_area, 1)

    # Bounding box
    bbox = mask_dict.get("bbox", cv2.boundingRect(largest))
    bx, by, bw, bh = [int(v) for v in bbox]

    # Aspect ratio of bounding box
    aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)

    # --- Edge proximity ---
    margin_y = int(image_h * edge_margin_frac)
    margin_x = int(image_w * edge_margin_frac)

    edge_zone = np.zeros((image_h, image_w), dtype=bool)
    edge_zone[:margin_y, :] = True   # top
    edge_zone[-margin_y:, :] = True  # bottom
    edge_zone[:, :margin_x] = True   # left
    edge_zone[:, -margin_x:] = True  # right

    mask_bool = seg.astype(bool)
    pixels_in_edge = np.sum(mask_bool & edge_zone)
    edge_touch_ratio = pixels_in_edge / max(area, 1)

    # Does the mask touch the actual image border? (within 2px)
    touches_border = (
        np.any(mask_bool[:2, :])
        or np.any(mask_bool[-2:, :])
        or np.any(mask_bool[:, :2])
        or np.any(mask_bool[:, -2:])
    )

    # --- Centroid position (normalized 0-1) ---
    moments = cv2.moments(seg)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"] / image_w
        cy = moments["m01"] / moments["m00"] / image_h
    else:
        cx, cy = 0.5, 0.5

    # --- Size relative to image ---
    image_area = image_h * image_w
    area_fraction = area / image_area

    # --- Intensity contrast ---
    mask_intensity = float(median_gray[mask_bool].mean()) if np.any(mask_bool) else 0.0
    bg_mask = ~mask_bool
    bg_intensity = float(median_gray[bg_mask].mean()) if np.any(bg_mask) else 0.0
    intensity_contrast = abs(mask_intensity - bg_intensity)

    # --- Intensity std within the mask (low = uniform = likely equipment) ---
    mask_intensity_std = (
        float(median_gray[mask_bool].std()) if np.any(mask_bool) else 999.0
    )

    return {
        "valid": True,
        "area": area,
        "area_fraction": area_fraction,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "edge_touch_ratio": edge_touch_ratio,
        "touches_border": touches_border,
        "centroid_x": cx,
        "centroid_y": cy,
        "intensity_contrast": intensity_contrast,
        "mask_intensity_std": mask_intensity_std,
        "mask_intensity_mean": mask_intensity,
        "bbox": [bx, by, bw, bh],
        "predicted_iou": float(mask_dict.get("predicted_iou", 0.0)),
        "stability_score": float(mask_dict.get("stability_score", 0.0)),
    }


# ---------------------------------------------------------------------------
# Mask filtering
# ---------------------------------------------------------------------------
def classify_mask(
    props: Dict[str, Any],
    min_solidity: float,
    max_area_fraction: float,
    min_area_fraction: float,
    min_edge_touch: float,
    min_intensity_contrast: float,
) -> Tuple[bool, str]:
    """
    Decide whether a mask is a likely persistent equipment intrusion.

    Returns (keep, reason).
    - keep=True  -> this looks like real equipment
    - keep=False -> rejected, reason explains why
    """
    if not props.get("valid", False):
        return False, "invalid_contour"

    af = props["area_fraction"]

    # Too large = SAM segmented the entire background/scene
    if af > max_area_fraction:
        return False, f"too_large (area_frac={af:.3f} > {max_area_fraction})"

    # Too small = noise
    if af < min_area_fraction:
        return False, f"too_small (area_frac={af:.5f} < {min_area_fraction})"

    # Low solidity = wispy/fan-shaped light artifacts, not solid equipment
    sol = props["solidity"]
    if sol < min_solidity:
        return False, f"low_solidity (solidity={sol:.3f} < {min_solidity})"

    # Equipment intrudes from edges. Masks floating in the center of the
    # frame with no edge contact are likely scene content or SAM artifacts.
    et = props["edge_touch_ratio"]
    tb = props["touches_border"]
    if not tb and et < min_edge_touch:
        return (
            False,
            f"not_edge_attached (edge_touch={et:.3f}, touches_border={tb})",
        )

    # Very low intensity contrast with the background suggests SAM carved
    # up a uniform region arbitrarily.
    ic = props["intensity_contrast"]
    if ic < min_intensity_contrast:
        return (
            False,
            f"low_contrast (intensity_contrast={ic:.1f} < {min_intensity_contrast})",
        )

    return True, "passed_all_filters"


def filter_masks(
    sam_masks: List[Dict[str, Any]],
    image_h: int,
    image_w: int,
    median_gray: np.ndarray,
    min_solidity: float = DEFAULT_MIN_SOLIDITY,
    max_area_fraction: float = DEFAULT_MAX_AREA_FRACTION,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    min_edge_touch: float = DEFAULT_MIN_EDGE_TOUCH,
    min_intensity_contrast: float = DEFAULT_MIN_INTENSITY_CONTRAST,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter SAM masks to keep only likely persistent equipment intrusions.

    Returns:
        kept_masks:   list of mask dicts that passed all filters
        filter_report: per-mask classification report for debugging
    """
    kept = []
    report = []

    for idx, m in enumerate(sam_masks):
        props = compute_mask_properties(m, image_h, image_w, median_gray)
        keep, reason = classify_mask(
            props,
            min_solidity=min_solidity,
            max_area_fraction=max_area_fraction,
            min_area_fraction=min_area_fraction,
            min_edge_touch=min_edge_touch,
            min_intensity_contrast=min_intensity_contrast,
        )

        entry = {
            "mask_index": idx,
            "kept": keep,
            "reason": reason,
            **{k: v for k, v in props.items() if k != "valid"},
        }
        report.append(entry)

        if keep:
            kept.append(m)

    return kept, report


# ---------------------------------------------------------------------------
# Mask union
# ---------------------------------------------------------------------------
def masks_to_union(
    masks: List[Dict[str, Any]], shape_hw: Tuple[int, int]
) -> np.ndarray:
    union = np.zeros(shape_hw, dtype=bool)
    for m in masks:
        seg = m.get("segmentation")
        if seg is not None and seg.shape == shape_hw:
            union |= seg.astype(bool)
    return union


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------
def save_outputs(
    out_dir: str,
    median_rgb_u8: np.ndarray,
    all_sam_masks: List[Dict[str, Any]],
    filtered_masks: List[Dict[str, Any]],
    union_mask: np.ndarray,
    filter_report: List[Dict[str, Any]],
    max_exported_masks: int,
    timestamp: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.join(out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    masks_dir = os.path.join(run_dir, "sam_masks")
    os.makedirs(masks_dir, exist_ok=True)

    # Union mask
    np.save(
        os.path.join(run_dir, "sam_union_mask.npy"),
        union_mask.astype(np.uint8),
    )
    cv2.imwrite(
        os.path.join(run_dir, "sam_union_mask.png"),
        union_mask.astype(np.uint8) * 255,
    )

    # Individual filtered masks (sorted by area descending)
    sorted_masks = sorted(
        filtered_masks, key=lambda m: int(m.get("area", 0)), reverse=True
    )
    export_n = min(max_exported_masks, len(sorted_masks))
    for idx in range(export_n):
        seg = sorted_masks[idx]["segmentation"].astype(np.uint8)
        area = int(sorted_masks[idx].get("area", int(seg.sum())))
        cv2.imwrite(
            os.path.join(masks_dir, f"mask_{idx:04d}_area_{area}.png"), seg * 255
        )

    # Filter report as JSON (much more portable than object-dtype npy)
    report_path = os.path.join(run_dir, "filter_report.json")
    with open(report_path, "w") as f:
        json.dump(filter_report, f, indent=2, default=str)

    # --- Visualization panel ---
    fig_cols = 3
    fig, axes = plt.subplots(1, fig_cols, figsize=(6 * fig_cols, 6))

    # 1) Median image
    axes[0].imshow(median_rgb_u8)
    axes[0].set_title("Median Image")
    axes[0].axis("off")

    # 2) ALL SAM masks (raw, before filtering) for comparison
    all_union = masks_to_union(all_sam_masks, union_mask.shape)
    raw_overlay = median_rgb_u8.copy()
    raw_overlay[all_union] = [255, 100, 100]
    axes[1].imshow(raw_overlay)
    axes[1].set_title(f"Raw SAM ({len(all_sam_masks)} masks)")
    axes[1].axis("off")

    # 3) Filtered masks only
    filtered_overlay = median_rgb_u8.copy()
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 100, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 128, 0],
        [128, 0, 255],
    ]
    for i, m in enumerate(sorted_masks[:max_exported_masks]):
        seg = m["segmentation"].astype(bool)
        color = colors[i % len(colors)]
        filtered_overlay[seg] = color
    axes[2].imshow(filtered_overlay)
    axes[2].set_title(f"Filtered ({len(filtered_masks)} persistent artifacts)")
    axes[2].axis("off")

    plt.tight_layout()
    panel_path = os.path.join(run_dir, "sam_pipeline_panel.png")
    plt.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved outputs:")
    print(f"  Run directory:   {run_dir}")
    print(f"  Union mask:      {run_dir}/sam_union_mask.png")
    print(f"  Individual masks: {masks_dir}/ ({export_n} masks)")
    print(f"  Filter report:   {report_path}")
    print(f"  Visual panel:    {panel_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run SAM on a median image to detect persistent equipment "
            "intrusions (submersible mounts, housing, light rigs) and "
            "export filtered binary masks."
        )
    )
    # Input/output
    p.add_argument("--median-image", type=str, default=DEFAULT_MEDIAN_IMAGE)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    # SAM model
    p.add_argument("--sam-checkpoint", type=str, default=DEFAULT_SAM_CHECKPOINT)
    p.add_argument(
        "--sam-model-type",
        type=str,
        default=DEFAULT_SAM_MODEL_TYPE,
        choices=["vit_h", "vit_l", "vit_b"],
    )
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    # SAM generator params
    p.add_argument("--points-per-side", type=int, default=DEFAULT_POINTS_PER_SIDE)
    p.add_argument("--pred-iou-thresh", type=float, default=DEFAULT_PRED_IOU_THRESH)
    p.add_argument("--stability-score-thresh", type=float, default=DEFAULT_STABILITY_SCORE_THRESH)
    p.add_argument("--min-mask-region-area", type=int, default=DEFAULT_MIN_MASK_REGION_AREA)
    p.add_argument("--max-exported-masks", type=int, default=DEFAULT_MAX_EXPORTED_MASKS)

    # Post-processing filter thresholds
    p.add_argument(
        "--min-solidity",
        type=float,
        default=DEFAULT_MIN_SOLIDITY,
        help="Minimum solidity (area/convex_hull). Lower = allow more irregular shapes. "
        f"Rejects fan-shaped light artifacts. (default: {DEFAULT_MIN_SOLIDITY})",
    )
    p.add_argument(
        "--max-area-fraction",
        type=float,
        default=DEFAULT_MAX_AREA_FRACTION,
        help="Max fraction of image a single mask can cover. "
        f"Rejects background/scene segments. (default: {DEFAULT_MAX_AREA_FRACTION})",
    )
    p.add_argument(
        "--min-area-fraction",
        type=float,
        default=DEFAULT_MIN_AREA_FRACTION,
        help="Min fraction of image a mask must cover. "
        f"Rejects tiny noise segments. (default: {DEFAULT_MIN_AREA_FRACTION})",
    )
    p.add_argument(
        "--min-edge-touch",
        type=float,
        default=DEFAULT_MIN_EDGE_TOUCH,
        help="Min fraction of mask pixels in the edge zone for "
        "masks that don't touch the border. Equipment intrudes from edges. "
        f"(default: {DEFAULT_MIN_EDGE_TOUCH})",
    )
    p.add_argument(
        "--min-intensity-contrast",
        type=float,
        default=DEFAULT_MIN_INTENSITY_CONTRAST,
        help="Min absolute intensity difference between mask region and "
        f"background. Rejects uniform-region hallucinations. (default: {DEFAULT_MIN_INTENSITY_CONTRAST})",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    total_steps = 5
    state = PipelineState()

    if args.max_exported_masks <= 0:
        print("Error: --max-exported-masks must be >= 1")
        sys.exit(1)
    if args.points_per_side <= 0:
        print("Error: --points-per-side must be >= 1")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Step 1: Load median image
    # ------------------------------------------------------------------
    log_step(1, total_steps, "Load and validate median image")
    state.median_rgb_u8 = load_median_image(args.median_image)
    h, w = state.median_rgb_u8.shape[:2]
    print(f"  Shape: {state.median_rgb_u8.shape}  dtype: {state.median_rgb_u8.dtype}")

    # Precompute grayscale for intensity analysis during filtering
    median_gray = cv2.cvtColor(state.median_rgb_u8, cv2.COLOR_RGB2GRAY)

    # ------------------------------------------------------------------
    # Step 2: Load SAM
    # ------------------------------------------------------------------
    log_step(2, total_steps, "Load SAM model")
    mask_generator = build_sam_generator(
        checkpoint=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=args.device,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )
    print("  SAM model loaded")

    # ------------------------------------------------------------------
    # Step 3: Run SAM inference
    # ------------------------------------------------------------------
    log_step(3, total_steps, "Run SAM mask generation")
    state.sam_masks = mask_generator.generate(state.median_rgb_u8)
    if len(state.sam_masks) == 0:
        raise RuntimeError(
            "SAM produced zero masks. Try lowering thresholds or "
            "increasing points-per-side."
        )
    print(f"  Raw SAM masks: {len(state.sam_masks)}")

    # ------------------------------------------------------------------
    # Step 4: Post-filter masks for persistent equipment
    # ------------------------------------------------------------------
    log_step(4, total_steps, "Filter masks for persistent equipment intrusions")
    state.filtered_masks, state.filter_report = filter_masks(
        sam_masks=state.sam_masks,
        image_h=h,
        image_w=w,
        median_gray=median_gray,
        min_solidity=args.min_solidity,
        max_area_fraction=args.max_area_fraction,
        min_area_fraction=args.min_area_fraction,
        min_edge_touch=args.min_edge_touch,
        min_intensity_contrast=args.min_intensity_contrast,
    )

    # Print filtering summary
    kept_count = len(state.filtered_masks)
    rejected_count = len(state.sam_masks) - kept_count
    print(f"  Kept:     {kept_count}")
    print(f"  Rejected: {rejected_count}")

    # Summarize rejection reasons
    reasons: Dict[str, int] = {}
    for entry in state.filter_report:
        if not entry["kept"]:
            reason_key = entry["reason"].split(" (")[0]
            reasons[reason_key] = reasons.get(reason_key, 0) + 1
    if reasons:
        print("  Rejection breakdown:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    # Build union from FILTERED masks only
    state.union_mask = masks_to_union(state.filtered_masks, (h, w))
    union_ratio = float(np.mean(state.union_mask))
    print(f"  Filtered union fill ratio: {union_ratio:.2%}")

    # ------------------------------------------------------------------
    # Step 5: Save outputs
    # ------------------------------------------------------------------
    log_step(5, total_steps, "Save outputs")
    save_outputs(
        out_dir=args.output_dir,
        median_rgb_u8=state.median_rgb_u8,
        all_sam_masks=state.sam_masks,
        filtered_masks=state.filtered_masks,
        union_mask=state.union_mask,
        filter_report=state.filter_report,
        max_exported_masks=args.max_exported_masks,
        timestamp=timestamp,
    )
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()