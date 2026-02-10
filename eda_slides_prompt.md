# Prompt: Create 5-Slide EDA Presentation

Create a clean, professional 5-slide presentation summarizing the Exploratory Data Analysis of underwater sonar and optical datasets for our project: **3D Gaussian Splatting for Synthetic Underwater 3D Model Generation**. Use the data below verbatim. Keep slides visual and data-dense — no filler text.

---

## SLIDE 1: Dataset Landscape Overview

**Title:** "Underwater Dataset Landscape: 30 Datasets Across Sonar & Optical Domains"

Content to include:
- 21 sonar datasets (REMARO OpenSonarDatasets) + 9 optical underwater datasets analyzed
- Year range: 2010–2025, with 17 of 21 sonar datasets published since 2020 (6 in 2024 alone)
- Total samples across all datasets: ~1.5M sonar + 655K optical video frames
- **Sonar type breakdown** (bar or pie): SSS dominates (11 datasets), FLS (5), MSIS (2), MBES (2), SAS (1)
- **Task breakdown**: Segmentation (7), Object Detection (5), Classification (4), SLAM (3), None (2)
- **Data format split**: Images (13), Raw (3), Rosbag (2), GeoTIFF (1), Mixed (2)
- Key callout: "85.7% have published papers; average completeness score 85.7/100"

Use a 2x2 grid layout: sonar type pie chart (top-left), task bar chart (top-right), timeline of publications (bottom-left), data format pie (bottom-right).

---

## SLIDE 2: Per-Dataset Statistical Profiles

**Title:** "Statistical Health Check: Class Imbalance, Entropy & Power Analysis"

Content to include:
- Table or heatmap of key metrics for top datasets:

| Dataset | Classes | Samples | Imbalance Ratio | Normalized Entropy | Gini | Viable? |
|---|---|---|---|---|---|---|
| BenthiCat | 12 | 950,000 | 4.75 | 0.964 | 0.217 | From scratch |
| Seafloor Sediments | 3 | 434,164 | 2.00 | 0.963 | 0.154 | From scratch |
| FishNet | 6 | 94,532 | 1.43 | 0.993 | 0.032 | From scratch |
| UXO | 3 | 74,437 | 1.89 | 0.974 | 0.116 | From scratch |
| SubPipe | 5 | 10,030 | — | — | — | Fine-tuning |
| UATD | 10 | 9,200 | 3.27 | 0.936 | 0.269 | Transfer only |
| SeabedObjects-KLSG | 11 | 1,190 | 11.48 | 0.741 | 0.485 | Transfer only |

- Imbalance grouping bar chart:
  - Severe (>5x): 1 dataset (SeabedObjects-KLSG)
  - Moderate (2–5x): 9 datasets
  - Balanced (<2x): 9 datasets
- Rules of thumb: ~500/class (transfer learning) | ~2K/class (fine-tuning) | ~10K/class (from scratch)
- Key callout: "Most datasets are moderate; only SeabedObjects-KLSG has severe class imbalance (11.48x)"

---

## SLIDE 3: Content Classification & Label Harmonization

**Title:** "What's Actually in the Data: Content Flags & Unified Label Pools"

Content to include:
- **Content flag heatmap** (datasets as rows, columns = has_boats, has_fish, has_manmade_objects):
  - 6 datasets contain boats/wrecks, 7 contain fish, 16 contain man-made objects
  - Important finding: **No surface boat contamination** — all "boat" content is sunken wrecks = ideal 3D reconstruction targets
- **Boat/wreck breakdown**:
  - SeabedObjects-KLSG: 385 wreck samples (SSS)
  - AI4Shipwreck: 286 shipwreck tiles (SSS)
  - NKSID: 872 propeller samples (FLS)
  - SonarSplat: monohansett_3D scene = full shipwreck with GT mesh
- **Unified label pools** (after harmonizing 28 raw labels across 30 datasets):
  - natural: 1,290,840 samples (11 datasets)
  - marine_life: 208,503 samples (9 datasets)
  - mine/UXO: 76,473 samples (4 datasets)
  - debris: 15,512 samples (5 datasets)
  - pipeline: 10,239 samples (2 datasets)
  - infrastructure: 10,185 samples (4 datasets)
  - wreck: 2,053 samples (4 datasets)
- Key callout: "Wreck pool is small (2,053) but these are the highest-value targets for 3D reconstruction"

---

## SLIDE 4: Gaussian Splatting Compatibility Scoring

**Title:** "3DGS Compatibility: Which Datasets Can Actually Build 3D Models?"

Content to include:
- Scoring rubric (show as small legend):
  - Multi-view/video: 30 pts
  - SE(3) poses: 25 pts
  - Sensor compatibility (FLS=15, Camera=8, SSS=5): 15 pts
  - 3D structure in scene: 15 pts
  - Calibration/depth maps: 15 pts
  - Max: 100 pts

- **Horizontal bar chart** of all 30 datasets ranked by GS compatibility score, color-coded:
  - Green (>=70): SonarSplat (85), Cave Sonar (80), Aurora (75), DRUVA (73), MBES-Slam (70)
  - Yellow (40–69): BenthiCat, Sea-thru, UVEB, SUVE, UVE-38K, SQUID
  - Red (<40): All single-image datasets (UATD, UXO, NKSID, etc.)
  - Grey (0–14): Flat seabed, signal data, fish-only datasets

- **Feature availability heatmap**: datasets as rows, columns = multi-view, poses, depth, video, calibration, sensor_compat
- Key callout: "Only 5 datasets score >=70 (ready for GS); 6 more need COLMAP pose extraction from video"

---

## SLIDE 5: Final Tier List & Recommended Pipeline

**Title:** "Dataset Tier List for Synthetic Underwater 3D Model Generation"

Content to include:
- **Tier S — Ready for Gaussian Splatting** (multi-view + poses + calibration):
  1. **SonarSplat** — 2,987 FLS frames, 9 scenes, SE(3) poses, 2 GT meshes. ALREADY IN WORKSPACE. Includes shipwreck (monohansett_3D).
  2. **Cave Sonar** — MSIS+Multi, ROSBAG with SLAM navigation. Rich 3D cave geometry.
  3. **Aurora** — MBES+SSS+Multi, full navigation. Large-scale seabed mapping.
  4. **MBES-Slam** — MBES, 4 sequences with full nav. Small but complete.
  5. **BenthiCat** — 950K tiles, SSS+Camera, AUV navigation. Opti-acoustic tools can extract camera sequences with poses.

- **Tier A — High Potential** (video data, needs COLMAP for poses):
  6. **DRUVA** — 20 videos, 360-deg views of artifacts. BEST for optical GS.
  7. **Sea-thru** — RAW images + depth maps + calibration.
  8. **UVEB** — 1,308 videos, 453K frames, 38% 4K. Largest UW video dataset.
  9. **SUVE** — 840 videos, 140K frames.
  10. **UVE-38K** — 50 videos, 38K frames.

- **Quantified data pool**:
  - Tier S sonar frames: ~953K
  - Tier A optical frames: ~655K (pipeline: Video -> COLMAP SfM -> 3DGS)
  - Total usable: **~1.6M frames** for 3D model generation

- **Recommended next steps** (as a simple numbered list):
  1. Start with SonarSplat (already available, proven pipeline)
  2. Acquire DRUVA for optical 3DGS (360-deg = best reconstruction quality)
  3. Scale with UVEB (453K frames, 4K resolution)

- Key callout: "1.6M frames across sonar + optical domains available for synthetic 3D underwater model generation"

---

## Style Notes
- Color scheme: dark navy/teal background with white text OR clean white background with navy accents
- Use consistent color coding: Green = Tier S, Blue = Tier A, Orange = Tier B, Grey = Tier C
- Every chart should have axis labels and a title
- Minimize bullet text — prefer charts, tables, and heatmaps
- Add project title in footer of each slide: "Nittany AI — Underwater 3D Gaussian Splatting"
- Font: clean sans-serif (Helvetica, Inter, or similar)
