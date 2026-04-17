# EDA and Dataset Downloads (Optional)

> **These scripts are optional.** They were written for the specific datasets we used during this project (primarily FLSea from Harvard Dataverse) and will not work on arbitrary datasets out of the box. They are included here as reference for how we performed our exploratory data analysis and dataset acquisition. If you are bringing your own images, you can skip this entire section and go straight to the [[Preprocessing Pipeline]].

All scripts in this section live in `eda_and_dataset_downloads/`.

```
eda_and_dataset_downloads/
├── download_and_rank.py                      Download + CLIP ranking (FLSea specific)
├── download_dataset.py                       Bulk download from Harvard Dataverse
├── rank_images.py                            Standalone image ranking (local files)
├── optical_imagery_eda.py                    HTML report generation for optical datasets
├── underwater_optical_datasets_analysis.py   Optical dataset catalog with suitability scores
├── sonar_datasets_eda.ipynb                  Jupyter notebook: sonar dataset survey
└── Download Datasets/
    ├── create_valid_scene.py                 COLMAP scene creator for WaterSplatting
    └── download_seaThruNerf.py               SeaThru-NeRF dataset downloader
```

---

## Dataset Download Scripts

### download_and_rank.py

> **Dataset-specific.** Defaults to the FLSea dataset on Harvard Dataverse (DOI: `10.7910/DVN/VZD5S6`). The scoring prompts are tuned for shipwrecks, debris, and underwater structures. Will not produce meaningful rankings on unrelated datasets.

Downloads archives from Harvard Dataverse one at a time, extracts images, scores them using CLIP semantic similarity combined with computer vision heuristics, and retains only the top-K most useful images. Disk usage stays bounded because images not in the current top-K are deleted after each archive is processed.

**Scoring formula:** Each image gets a composite score from 0 to 1 combining six weighted metrics:

| Metric | Default Weight | What it measures |
|--------|---------------|-----------------|
| `clip_positive` | 50% | Similarity to 12 positive prompts (shipwrecks, debris, structures) |
| `clip_negative` | 20% | Dissimilarity to 9 negative prompts (empty sand, blank images) |
| `entropy` | 5% | Information content (Shannon entropy of grayscale histogram) |
| `laplacian_var` | 5% | Sharpness (variance of Laplacian convolution) |
| `saturation_penalty` | 10% | Exposure quality (penalizes blown-out pixels) |
| `edge_density` | 10% | Structural content (Canny edge pixel ratio) |

```bash
# Basic usage — downloads FLSea, keeps top 200 images
python eda_and_dataset_downloads/download_and_rank.py --outdir ranked_dataset --k 200

# GPU acceleration
python eda_and_dataset_downloads/download_and_rank.py --outdir ranked_dataset --device cuda --batch-size 32

# Custom scoring weights
python eda_and_dataset_downloads/download_and_rank.py --weight-clip-positive 0.6 --weight-entropy 0.1
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--outdir` | `ranked_dataset` | Output directory |
| `--k` | `200` | Number of top images to keep |
| `--model` | `openai/clip-vit-base-patch32` | CLIP model identifier |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |
| `--batch-size` | `16` | Batch size for CLIP inference |
| `--doi` | `doi:10.7910/DVN/VZD5S6` | Dataset DOI to download |
| `--keep-archives` | False | Keep downloaded archives after processing |

---

### download_dataset.py

> **Dataset-specific.** Hardcoded for Harvard Dataverse. Organizes files by type based on file extensions specific to the FLSea dataset.

Simple bulk downloader with no scoring or ranking. Downloads every file from a Harvard Dataverse dataset and organizes them by type (sonar, images, metadata, ROS bags).

```bash
python eda_and_dataset_downloads/download_dataset.py
```

---

### rank_images.py

> **More general-purpose** than the other download scripts, but the CLIP prompts are still tuned for underwater shipwreck/debris imagery.

Ranks images that already exist in a local directory. Uses the same scoring engine as `download_and_rank.py` but with no download step. Two-pass approach: first scores all images, then attaches neighbor context for the top-K.

```bash
python eda_and_dataset_downloads/rank_images.py --root /path/to/images --k 200 --outdir results/
```

---

## EDA Scripts

### optical_imagery_eda.py

> **Dataset-specific.** Generates an HTML report with Chart.js visualizations for a specific dataset. The analysis categories and thresholds are tuned for underwater optical imagery.

Comprehensive per-image and per-dataset analysis including:
- Color profile analysis (blue/red ratio, brightness distribution)
- Quality metrics (blur detection, sharpness, edge density)
- Content classification via CLIP zero-shot
- COLMAP reconstruction analysis (if COLMAP data is present)
- Gaussian splatting readiness assessment

```bash
python eda_and_dataset_downloads/optical_imagery_eda.py --input /path/to/dataset
```

Output: `underwater_eda_report.html` with 18+ interactive charts.

---

### underwater_optical_datasets_analysis.py

> **Reference only.** This script prints a hardcoded catalog of underwater optical datasets with Gaussian splatting suitability scores. It does not download anything or process any data.

Catalogs 30+ underwater optical datasets (enhancement, detection, segmentation, tracking, classification) and scores each 0-100 for 3D reconstruction suitability based on: video sequences available, known camera poses, depth maps, frame overlap.

```bash
python eda_and_dataset_downloads/underwater_optical_datasets_analysis.py
```

---

### sonar_datasets_eda.ipynb

> **Reference only.** Interactive Jupyter notebook surveying 21+ open sonar datasets. Does not download anything.

Analyzes sonar datasets by type (SSS, FLS, MBES, MSIS, SAS), annotation task, sample count, and completeness score. A pre-exported HTML version is available at `outputs/sonar_datasets_eda.html`.

```bash
jupyter notebook eda_and_dataset_downloads/sonar_datasets_eda.ipynb
```

---

## WaterSplatting Data Preparation Scripts

These two scripts in `eda_and_dataset_downloads/Download Datasets/` are used to prepare data specifically for WaterSplatting. See the [[WaterSplatting]] page for full details.

### create_valid_scene.py

Runs the full COLMAP pipeline on a folder of images to estimate camera poses and create a WaterSplatting-ready scene directory. This one is general-purpose — it works on any folder of images.

### download_seaThruNerf.py

> **Dataset-specific.** Downloads the SeaThru-NeRF dataset from a specific Google Drive link. Only useful if you want that particular dataset.

Downloads a pre-processed underwater optical dataset with white-balanced images and COLMAP camera poses already computed.
