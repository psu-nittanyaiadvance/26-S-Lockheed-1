# Preprocessing Pipeline

Underwater cameras mounted on ROVs and AUVs often have visible housing edges, instrument overlays, or mounting bars that appear in every frame at the same pixel locations. These static artifacts must be removed before 3D reconstruction because feature matching algorithms will match artifact pixels across all frames, creating false correspondences that distort the 3D model.

The preprocessing pipeline is four scripts run in order. Each script produces outputs that inform the next step. All scripts live in the `preprocessing/` directory.

```
preprocessing/
├── artifact_layering.py         Step 1: Median image + variance map
├── artifact_edge_detection.py   Step 2: Edge-based boundary detection
├── sam.py                       Step 3: SAM artifact segmentation
├── crop.py                      Step 4: Manual crop using info from Steps 1-3
└── Archived_Preprocessing/      Older versions of the preprocessing pipeline (useful for reference)
```

---

## Step 1: artifact_layering.py

Computes a median image and variance map from your entire dataset to reveal persistent structures.

### What it does

1. Loads all images from the input directory and resizes them to a consistent shape
2. Streams images into a disk-backed uint8 memmap (the full image stack never has to fit in RAM)
3. Computes the exact per-pixel, per-channel median across all images in row chunks
4. Computes a per-pixel variance map from streaming sum and sum-of-squares moments
5. Thresholds the variance map to create a persistent artifact mask
6. Saves the median image, variance map, persistent mask, and an overlay visualization

**Why median?** Static artifacts appear identically in every frame, so they remain sharp and clear in the median. Scene content varies wildly between images and averages out to a blur. This makes artifacts easy to spot.

**Why variance?** Pixels that stay constant across all frames (artifacts) have near-zero variance. Pixels that change (scene content) have high variance. Thresholding the variance map gives you a binary mask of where the artifacts are.

### Usage

```bash
# Basic usage with defaults
python preprocessing/artifact_layering.py

# Custom paths and threshold
python preprocessing/artifact_layering.py \
    --input-dir path/to/images \
    --output-dir artifact_analysis/my_dataset \
    --variance-threshold 0.0025
```

### Key configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-dir` | `artifact_analysis/dc_train/...` | Path to image directory |
| `--output-dir` | `artifact_analysis/dc_train` | Where to save outputs |
| `--variance-threshold` | `0.0075` | Threshold for the variance map. Lower = more aggressive detection |
| `STACK_COLOR_SPACE` | `LAB` | Color space for processing (set in source code) |

### Output files

| File | Description |
|------|-------------|
| `median_image.png` | The computed median image — artifacts are sharp, scene is blurred |
| `variance_map.png` | Heatmap showing per-pixel variance across all frames |
| `persistent_mask.png` | Binary mask: white = artifact (low variance), black = scene |
| `median_variance_overlay_*.png` | Side-by-side visualization for quick review |

### Important

This script does **not** crop images. It produces visualizations for you to review before making any cropping decisions. Look at the median image and overlay to confirm where artifacts are before running subsequent steps.

---

## Step 2: artifact_edge_detection.py

Detects the inner boundary of border-connected artifacts using Sobel edge detection.

### What it does

1. Takes a grayscale image (typically the median image from Step 1)
2. Automatically detects which edge the artifact protrudes from (top, bottom, left, right), or uses a manually specified edge
3. Rotates the image so the target edge is always at the bottom (simplifies processing)
4. Divides the image into vertical slices of configurable width
5. In each slice, computes the Sobel gradient magnitude along the vertical axis
6. Smooths the gradient profile with a Gaussian filter
7. Finds peaks in the gradient profile (these mark transitions between artifact and scene)
8. The outermost significant peak in each slice gives the boundary point
9. Optionally applies outlier rejection using a median filter to smooth the boundary
10. Returns boundary points and generates overlay visualizations

### Usage

```bash
# Auto-detect which edge the artifact is on
python preprocessing/artifact_edge_detection.py --image path/to/median_image.png

# Specify the edge manually
python preprocessing/artifact_edge_detection.py --image path/to/median_image.png --edge right
```

### Key configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image` | (test path in source) | Path to image (usually the median image from Step 1) |
| `--edge` | `auto` | Which edge to check: `top`, `bottom`, `left`, `right`, or `auto` |
| `--search-frac` | `0.25` | Fraction of image to search inward from the edge |
| `--chunk-size` | `10` | Width of vertical slices in pixels |
| `--sobel-ksize` | `5` | Sobel kernel size |
| `--smooth-sigma` | `3.0` | Gaussian smoothing sigma on gradient profiles |
| `--peak-height` | `0.12` | Minimum gradient magnitude to count as an edge |
| `--peak-distance` | `15` | Minimum pixel spacing between detected peaks |
| `--reject-outliers` | `False` | Apply median filter to reject outlier boundary points |

### Output

The script generates an overlay image showing the detected boundary line on top of the input image. The boundary points are also available programmatically if you import the `detect_edge_object()` function.

Use this overlay to visually confirm the boundary looks correct before choosing a crop line in Step 4.

---

## Step 3: sam.py

Uses the Segment Anything Model (SAM) to produce precise artifact masks.

### What it does

1. Loads the median image from Step 1
2. Runs SAM's automatic mask generator, which produces many candidate segmentation masks
3. Filters the masks through multiple stages to keep only likely equipment intrusions:
   - **Edge-proximity filter:** Keeps masks that touch or are near frame edges (where camera housing and mounts are)
   - **Solidity filter:** Keeps solid, compact shapes. Removes fan-shaped light artifacts and gradient hallucinations
   - **Intensity-contrast filter:** Keeps masks where the region's brightness differs from the background. Equipment usually looks different from water
   - **Size filter:** Rejects masks covering too much of the image (scene background) or too little (noise)
4. Optionally cross-checks with the variance map from Step 1 — verifies that candidate artifact regions have low variance (truly static)
5. Exports binary artifact masks and visualization overlays

### SAM model setup

Download the SAM checkpoint before running this script:

1. Go to [SAM Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)
2. Download `sam_vit_h_4b8939.pth` (~2.5 GB)
3. Place it at `artifact_analysis/sam/sam_vit_h_4b8939.pth` (or specify the path with `--sam-checkpoint`)

### Usage

```bash
# Basic usage
python preprocessing/sam.py \
    --median-image artifact_analysis/sunboat/sunboat_mean_image.png

# With variance cross-check
python preprocessing/sam.py \
    --median-image artifact_analysis/sunboat/sunboat_mean_image.png \
    --variance-map artifact_analysis/sunboat/variance_map.npy \
    --variance-threshold 0.0075

# Custom SAM checkpoint location
python preprocessing/sam.py \
    --median-image path/to/median.png \
    --sam-checkpoint path/to/sam_vit_h_4b8939.pth \
    --sam-model-type vit_h
```

### Key configuration

These parameters control how aggressively the script filters masks. You may need to adjust them per dataset.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--median-image` | (default in source) | Path to the median image from Step 1 |
| `--output-dir` | `artifact_analysis/sam_outputs` | Where to save masks and visualizations |
| `--sam-checkpoint` | `artifact_analysis/sam/sam_vit_h_4b8939.pth` | Path to SAM model weights |
| `--sam-model-type` | `vit_h` | SAM model type (`vit_h`, `vit_l`, or `vit_b`) |
| `--variance-map` | None | Optional path to variance map from Step 1 for cross-checking |
| `--variance-threshold` | `0.0075` | Variance threshold for cross-check |
| `MAX_AREA_FRACTION` | `0.4` | Max fraction of image a single mask can cover (rejects background). Raise if equipment covers a large part of frame |
| `MIN_AREA_FRACTION` | `0.0005` | Min fraction of image (rejects noise). Lower if you have small brackets/bolts |
| `MIN_INTENSITY_CONTRAST` | `2.0` | Min brightness difference between mask and background. Lower for dark-on-dark scenes |

### Output files

| File | Description |
|------|-------------|
| `artifact_mask.png` | Combined binary mask of all detected artifacts |
| `individual_masks/` | Individual binary mask for each detected artifact region |
| `overlay_*.png` | Visualization showing detected artifacts overlaid on the median image |
| `filter_report.json` | Details on which masks passed/failed each filter stage |

---

## Step 4: crop.py

A manual image crop utility. Using the visual evidence from Steps 1-3, you choose a crop line and apply it to every image in the dataset.

### What it does

1. Loads images from the input directory
2. Applies the crop you specify: a pixel index, an axis, and which side to keep
3. Reports dataset health: unopenable files, blank images, dimension distribution
4. In preview mode: shows the crop on a sample of images for visual confirmation
5. In apply mode: crops all images and saves them to the output directory

### Usage

```bash
# Preview only — see the crop on sample images before committing
python preprocessing/crop.py --line 420 --axis horizontal --keep top

# Preview + apply — crop all images
python preprocessing/crop.py --line 420 --axis horizontal --keep top --apply

# Vertical crop example
python preprocessing/crop.py --line 320 --axis vertical --keep left --apply

# Custom input/output directories
python preprocessing/crop.py \
    --input-dir path/to/images \
    --output-dir path/to/cropped \
    --line 420 --axis horizontal --keep top --apply
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--line` | None | Pixel index for the crop line (required unless set in source code) |
| `--axis` | `horizontal` | Crop axis: `horizontal` (cut across the width) or `vertical` (cut along the height) |
| `--keep` | `top` | Which side to keep. For horizontal: `top` or `bottom`. For vertical: `left` or `right` |
| `--apply` | False | If set, actually crop and save. Otherwise preview only |
| `--input-dir` | (default in source) | Path to image directory |
| `--output-dir` | (default in source) | Where to save cropped images |
| `PREVIEW_SAMPLES` | `9` | Number of images to show in preview (set in source code) |

### How to choose the crop line

1. Look at the **median image** from Step 1 — the artifacts are clearly visible
2. Look at the **edge detection overlay** from Step 2 — it shows exactly where the artifact boundary is
3. Look at the **SAM mask overlay** from Step 3 — it shows the precise artifact shape
4. Pick a crop line that removes the entire artifact region. It's better to be slightly aggressive (remove a few pixels of scene) than to leave artifact remnants

### Output

Cropped images are saved to the output directory, preserving the same filenames. The script also prints a dataset health report showing how many images were processed, how many were skipped (unopenable or blank), and the dimension distribution of the cropped output.

---

## Full Workflow Example

```bash
# Activate the data pipeline environment
source .venv/bin/activate

# Step 1: Generate median image and variance map
python preprocessing/artifact_layering.py \
    --input-dir path/to/raw/images \
    --output-dir artifact_analysis/my_dataset

# Review the outputs in artifact_analysis/my_dataset/
# Look at the median image and overlay to confirm artifact locations

# Step 2: Detect artifact boundary
python preprocessing/artifact_edge_detection.py \
    --image artifact_analysis/my_dataset/median_image.png

# Review the edge detection overlay

# Step 3: Generate SAM artifact masks
python preprocessing/sam.py \
    --median-image artifact_analysis/my_dataset/median_image.png \
    --variance-map artifact_analysis/my_dataset/variance_map.npy \
    --output-dir artifact_analysis/my_dataset/sam_outputs

# Review the SAM mask overlays

# Step 4: Preview the crop, then apply
python preprocessing/crop.py --line 420 --axis horizontal --keep top
# If the preview looks good:
python preprocessing/crop.py --line 420 --axis horizontal --keep top --apply
```

After Step 4, your cleaned images are ready for 3D reconstruction with WaterSplatting or SonarSplat.
