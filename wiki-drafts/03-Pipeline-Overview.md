# Pipeline Overview

The pipeline takes raw underwater imagery and prepares it for 3D Gaussian splatting reconstruction. Each step reads from and writes to well-defined directories, so you can enter at any point. If you already have clean images, skip straight to reconstruction.

---

## Step 1: Artifact Layering

**Script:** `preprocessing/artifact_layering.py`

Loads all images from your dataset, resizes them to a consistent shape, and streams them into a disk-backed memmap array (so you don't need to hold the full stack in RAM). Computes the exact per-pixel, per-channel median image and a per-pixel variance map.

Static artifacts (camera housing, mounting bars, light rigs) appear sharp and unchanged in the median image because they are identical in every frame. Scene content washes out because it varies between frames. The variance map highlights which pixels change across frames (scene) vs. stay constant (artifacts).

| Input | Output |
|-------|--------|
| Directory of raw images | Median image, variance map, persistent artifact mask, overlay visualization |

This step does **not** crop anything. It gives you the visual evidence to confirm where artifacts are before proceeding.

---

## Step 2: Edge Detection

**Script:** `preprocessing/artifact_edge_detection.py`

Takes the median image from Step 1 and uses Sobel edge detection to find the inner boundary of border-connected artifacts. The script automatically detects which edge the artifact protrudes from (top, bottom, left, or right), or you can specify it manually.

The detector works in vertical slices across the image, averaging the Sobel gradient response in each slice, smoothing the profiles, and finding peaks that mark the transition from artifact to scene. The result is a set of boundary points tracing the artifact edge.

| Input | Output |
|-------|--------|
| Median image (from Step 1) | Boundary points, overlay visualization showing detected edge |

Use this to see exactly where the artifact boundary is. The boundary points can guide your crop line in Step 4.

---

## Step 3: SAM Segmentation

**Script:** `preprocessing/sam.py`

Runs the Segment Anything Model (SAM) on the median image to produce fine-grained artifact masks. SAM generates many candidate masks automatically, then the script filters them through several stages:

1. **Edge-proximity filter** — Keeps masks that touch or are near the frame edges (where camera housing and mounts actually are)
2. **Solidity filter** — Keeps solid, non-wispy shapes (real equipment is solid, not fan-shaped)
3. **Intensity-contrast filter** — Keeps masks that differ in brightness from the surrounding area
4. **Size filter** — Rejects masks that are too large (scene background) or too small (noise)
5. **Variance cross-check** — Optionally compares with the variance map from Step 1 to verify that candidate regions are truly static

| Input | Output |
|-------|--------|
| Median image (from Step 1), optionally variance map | Binary artifact masks, visualization overlays |

**SAM checkpoint required:** Download `sam_vit_h_4b8939.pth` (~2.5 GB) from the [SAM GitHub releases](https://github.com/facebookresearch/segment-anything#model-checkpoints) and place it where the script can find it (default: `artifact_analysis/sam/sam_vit_h_4b8939.pth`).

---

## Step 4: Crop

**Script:** `preprocessing/crop.py`

A manual crop utility. Using the information gathered from Steps 1-3 (the median image, edge boundary points, and SAM masks), you decide on a crop line and apply it to every image in the dataset.

You specify:
- **Crop line** — the pixel index where the crop should happen
- **Axis** — horizontal or vertical
- **Side to keep** — top/bottom (for horizontal) or left/right (for vertical)

The script also reports dataset health: unopenable files, blank images, and dimension distribution.

| Input | Output |
|-------|--------|
| Raw image dataset + your chosen crop parameters | Cropped dataset with artifacts removed |

```bash
# Preview the crop on sample images
python preprocessing/crop.py --line 420 --axis horizontal --keep top

# Apply the crop to all images
python preprocessing/crop.py --line 420 --axis horizontal --keep top --apply
```

---

## Reconstruction

Once you have cleaned, artifact-free images, choose a reconstruction framework based on your data type.

### Optical Path: WaterSplatting

For standard RGB camera footage. Built as a Nerfstudio plugin, WaterSplatting combines 3D Gaussian Splatting with a learned water medium model that accounts for underwater light scattering and absorption. Requires COLMAP camera pose estimation as a preprocessing step.

Best for: clear-to-moderate visibility underwater footage from standard cameras.

| Input | Output |
|-------|--------|
| Cleaned RGB images + COLMAP sparse reconstruction | Trained Gaussian model, interactive 3D viewer |

See [[WaterSplatting]] for full details.

### Sonar Path: SonarSplat

For imaging sonar data. SonarSplat uses a custom rasterizer that projects 3D Gaussians into range/azimuth sonar images. Uses sensor poses directly from `.pkl` data files (no COLMAP needed).

Best for: turbid or low-visibility conditions where optical cameras fail.

| Input | Output |
|-------|--------|
| Sonar `.pkl` files with sensor poses + sonar images | Trained Gaussian model, rendered sonar images, point cloud/mesh exports |

See [[SonarSplat]] for full details.

---

## Quick Reference: When to Use Which Tool

| Situation | What to do |
|-----------|------------|
| You have raw images with visible camera housing or artifacts | Run Steps 1-4 in order |
| You already have clean, artifact-free images | Skip to Reconstruction |
| You have RGB camera video | Use WaterSplatting |
| You have imaging sonar data | Use SonarSplat |
| You have both optical and sonar data | Run both frameworks on their respective data |
| You only need sonar reconstruction | SonarSplat works with its own `.pkl` data format — the preprocessing pipeline is not needed |
