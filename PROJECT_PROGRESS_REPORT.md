# Nittany AI Project Progress Report
## LM-1 [Sonar Imagery] - Gaussian Splatting for Underwater Synthetic Imagery

**Team:** Nittany AI Interns
**Project Sponsor:** Sharri Palmer (Lockheed Martin)
**Date:** March 17, 2026
**Document Version:** 1.0

---

## 1. Original Project Scope

### Project Goal
Develop Gaussian Splatting models that generate synthetic underwater camera imagery from real data, then extend the approach to generate synthetic forward-looking sonar (FLS) imagery, supporting future development of multi-modal AI/ML models for Autonomous Underwater Vehicle (AUV) applications.

### Original Objectives
1. **Build a Gaussian Splatting model** from underwater optical imagery to generate synthetic camera views
2. **Validate synthetic optical imagery quality** for both existing and novel viewpoints
3. **Extend the model** to generate synthetic forward-looking sonar imagery in range-azimuth projection
4. **Create reusable code and documentation** for synthetic underwater data generation

### Planned Deliverables
- Gaussian Splatting model generating synthetic underwater camera imagery from video data
- Extended model producing synthetic forward-looking sonar (FLS) imagery in range-azimuth projection
- **Technical Report** (~5 pages): Overview of Gaussian splatting for underwater optics, dataset description, results (visualizations), and sonar splatting prototype
- **Code Package**: Clean code for recreating the work including libraries/notebooks/scripts and dependencies list/environment file

### Original 12-Week Timeline
- **Weeks 1**: Theory review and paper study (WaterSplatting)
- **Weeks 2-5**: Data acquisition and optical model building
- **Week 6**: Mid-project review
- **Week 7**: Sonar splatting research (SonarSplat paper)
- **Weeks 8-12**: Sonar image synthesis implementation and validation

---

## 2. Changes from Original Scope

### Major Strategic Shifts

#### 2.1 Integration vs. Building from Scratch
**Original Plan:** Build custom Gaussian Splatting implementations from foundational principles.

**Actual Approach:** Integrated two state-of-the-art existing implementations:
- **WaterSplatting** (arXiv:2408.08206): Underwater-optimized 3D Gaussian Splatting with custom CUDA kernels for optical imagery
- **SonarSplat** (IEEE RA-L 2025): University of Michigan's imaging sonar Gaussian splatting framework

**Rationale:** These implementations are production-quality, peer-reviewed, and battle-tested. Building from scratch would duplicate significant effort and delay results. Instead, we focused on:
- Understanding implementation details
- Adapting to our datasets
- Building comprehensive preprocessing infrastructure
- Creating a unified pipeline

#### 2.2 Expanded Data Infrastructure (Unplanned Addition)
**Original Plan:** Basic dataset acquisition from public sources or Lockheed Martin.

**Actual Development:** Built comprehensive end-to-end data pipeline including:

1. **Dataset Acquisition Tools** (3 scripts, ~2,700 lines of code)
   - Harvard Dataverse API integration for automated downloads
   - CLIP-based intelligent image ranking (retains top-K quality images)
   - Incremental download strategy to minimize disk usage

2. **Exploratory Data Analysis** (2 tools, ~1,450 lines)
   - Comprehensive optical imagery EDA with HTML reports (18+ interactive visualizations)
   - Sonar dataset analysis across 21 open-source datasets (91,300 files analyzed)
   - Quality metrics, color profiling, content classification, and 3DGS readiness scoring

3. **Preprocessing Pipeline** (~1,475 lines)
   - CLIP zero-shot classification into 14 underwater categories (fish, coral, marine_animal, vegetation, sand_seafloor, rocks, shipwreck_debris, man_made_structure, diver, open_water, murky_turbid, dark_overexposed, blurry_corrupt, camera_artifacts)
   - Static border artifact detection using cross-image statistical analysis
   - Two-phase analyze + clean workflow

4. **Artifact Detection & Removal** (3 specialized tools, ~2,500 lines)
   - Generic vehicle/camera housing artifact detector using median compositing
   - Dataset-specific artifact detectors (Sunboat, Shipwreck, FLSea datasets)
   - Template matching and connected component analysis methods

**Rationale:** Real-world underwater datasets are extremely noisy with camera artifacts, poor lighting, murky water, and unusable frames. The original plan underestimated the data preparation complexity. These tools are critical for successful 3D reconstruction.

#### 2.3 Repository Consolidation
**Original Plan:** Single linear development branch.

**Actual Structure:** Unified `main` branch consolidating multiple specialized branches:
- `image_class` - Dataset EDA and CLIP classification
- `watersplatting_initial` - Optical 3D reconstruction
- `optical-eda` - Comprehensive analysis tools
- `sonar-splat` - Sonar 3D reconstruction
- `main2` and `main3` - Integration branches

**Rationale:** Team members worked in parallel on different components. Consolidation ensures all functionality is preserved, documented, and accessible in a single unified pipeline.

#### 2.4 Diffusion Models (Planned Future Direction)
**Original Mention:** Exploring diffusion models for synthetic data generation.

**Current Status:** Not yet implemented. Focus has been on:
1. Establishing robust Gaussian Splatting baselines
2. Building data infrastructure
3. Understanding underwater-specific challenges

**Future Plan:** After completing Gaussian Splatting validation (Weeks 9-10), we plan to:
- Explore latent diffusion models for underwater image synthesis
- Investigate sonar-to-optical cross-modal generation
- Compare synthetic data quality (Gaussian Splatting vs. Diffusion)

---

## 3. Current Milestones Completed

### Milestone 1: Dataset Acquisition and Analysis ✅

**Completed Components:**

1. **FLSea Dataset Download and Analysis**
   - Successfully downloaded and analyzed FLSea-VI dataset from Harvard Dataverse (DOI: 10.7910/DVN/VZD5S6)
   - Processed 91,300 files across 14 underwater surveys (Canyons and Red Sea locations)
   - Categorized by type: 22,877 images, 22,451 depth files, 45,972 SeaErra files
   - Identified top surveys: `big_dice_loop` (12,636 files), `u_canyon` (11,580 files), `flatiron` (9,899 files)
   - Dataset quality: 12/14 surveys at 100% completeness, 92.9% average completeness

2. **Comprehensive Sonar Dataset Landscape Analysis**
   - Analyzed 21 open-source sonar datasets from REMARO network
   - Categorized by sonar type: Side-Scan Sonar (SSS), Forward-Looking Sonar (FLS), Multibeam Echosounder (MBES), Multi-frequency Sonar Imaging System (MSIS), Synthetic Aperture Sonar (SAS)
   - Evaluated annotation tasks, dataset sizes, temporal coverage, and data quality
   - Generated interactive HTML report (2.4 MB) with statistical visualizations
   - **Output Files:**
     - `outputs/sonar_datasets_eda.html` - Interactive analysis dashboard
     - `outputs/sonar_datasets_summary.csv` - Tabular dataset statistics
     - `outputs/sonar_analysis_report.txt` - Text summary report
     - `outputs/sonar_dataset_analysis.png` - Key visualization

3. **Intelligent Dataset Ranking System**
   - Developed CLIP-based multi-dimensional scoring algorithm combining:
     - CLIP similarity to positive prompts (50% weight): "shipwreck", "underwater debris", "coral reef", "rocky structures"
     - CLIP dissimilarity to negative prompts (20% weight): "empty water", "featureless seafloor", "too dark to see"
     - Image entropy (5% weight): Shannon entropy of grayscale histogram
     - Laplacian variance (5% weight): Sharpness measure
     - Saturation penalty (10% weight): Detects over/under-exposed pixels
     - Edge density (10% weight): Canny edge detector structural content
   - Incremental download strategy: retains only top-K images (default K=200) to minimize disk usage
   - GPU-accelerated batch processing (CUDA support with automatic fallback to CPU)

**Evidence:**
- `scripts/dataset_tools/download_dataset.py` - Harvard Dataverse downloader
- `scripts/dataset_tools/download_and_rank.py` - Combined download + ranking (1,306 lines)
- `scripts/dataset_tools/rank_images.py` - Standalone ranking tool (1,081 lines)
- `notebooks/sonar_datasets_eda.ipynb` - Interactive sonar analysis notebook

---

### Milestone 2: Data Preprocessing Pipeline ✅

**Completed Components:**

1. **CLIP Zero-Shot Image Classification**
   - Two-phase pipeline: **Analyze** → **Clean**
   - **Phase 1 (Analyze):** Classifies every image into one of 14 underwater categories using CLIP vision-language model
   - **Phase 2 (Clean):** Removes images classified as "empty" (open_water, murky_turbid, dark_overexposed, blurry_corrupt)
   - Batch processing with configurable batch size (default: 16 images)
   - GPU acceleration (auto-detects CUDA availability)
   - Output formats: CSV and JSON classification results

2. **Static Artifact Detection**
   - Cross-image statistical analysis to detect camera housing borders
   - Computes pixel-wise mean and standard deviation across all images
   - Identifies static regions (low variance) that appear consistently
   - Automatically crops detected artifacts from cleaned dataset

3. **Supported Datasets**
   - `flsea_vi` - FLSEA-VI underwater scenes (TIFF images)
   - `shipwreck` - Shipwreck survey recordings (JPEG images)
   - `sunboat` - Sunboat mission recordings (PNG images)
   - Extensible configuration system for adding new datasets

**Evidence:**
- `scripts/preprocessing/preprocess_datasets.py` (1,475 lines)
- Classification outputs: `classification_output/{dataset}_analysis.csv`
- Cleaned datasets: `cleaned_datasets/{dataset}/`

---

### Milestone 3: Artifact Detection and Cropping ✅

**Completed Components:**

1. **Generic Vehicle Artifact Detector** (`vehicle_artifact_crop.py`)
   - Three-signal detection system:
     - **Median compositing:** Aggregates all images to reveal static camera mount
     - **Brightness profiling:** Analyzes row/column brightness to detect dark borders
     - **Gradient-based boundary detection:** Finds sharp intensity transitions
   - Three-stage workflow:
     - `detect` mode: Generates diagnostic plots showing detected artifacts
     - `preview` mode: Shows before/after comparison on sample images
     - `apply` mode: Crops all images and exports to output directory
   - Works with any underwater dataset (no dataset-specific assumptions)

2. **Sunboat-Specific Detector** (`sunboat_crop.py`)
   - Specialized for Sunboat dataset camera mount (yellow-green bar on right side)
   - Uses HSV color space hue band filtering (hue 25-75 for yellow-green)
   - Connected component analysis to identify largest component (camera mount)
   - Automatically determines crop boundaries from component bounding box

3. **Template Matching Detector** (`sunboat_templatematching.py`)
   - Alternative approach using OpenCV template matching (`cv2.TM_CCOEFF_NORMED`)
   - Requires pre-cropped reference image of camera mount corner
   - Detects mount position in each image via template correlation
   - Computes crop parameters from detected template location

**Evidence:**
- `scripts/template_matching/vehicle_artifact_crop.py` - Generic detector
- `scripts/template_matching/sunboat_crop.py` - HSV color-based detector
- `scripts/template_matching/sunboat_templatematching.py` - Template matching
- Additional recent work: `artifact_edge_detection.py`, `artifact_layering.py` (503 lines added in commit ea0d4d9)

---

### Milestone 4: Gaussian Splatting Framework Integration ✅

**Completed Components:**

1. **WaterSplatting Integration (Optical Imagery)**
   - Full implementation of WaterSplatting paper (arXiv:2408.08206)
   - Custom CUDA kernels for GPU-accelerated rendering
   - Underwater-specific light propagation model accounting for:
     - Water absorption and scattering
     - Color attenuation (wavelength-dependent)
     - Backscatter effects
   - Nerfstudio integration for training pipeline
   - COLMAP scene creation from multi-view images
   - Real-time 3D viewer (accessible at `http://localhost:7007`)
   - **Key Files:**
     - `water_splatting/` directory (~50 files, ~15,000+ lines)
     - `water_splatting/cuda/` - CUDA kernels for forward/backward passes
     - `water_splatting/_torch_impl.py` - PyTorch Python bindings
     - `Download Datasets/create_valid_scene.py` - COLMAP scene creation (4,399 lines)
     - `setup.py`, `pyproject.toml` - Package configuration

2. **SonarSplat Integration (Sonar Imagery)**
   - Full implementation of IEEE RA-L 2025 paper (University of Michigan)
   - Novel view synthesis for imaging sonar (+3.2 dB PSNR vs. state-of-the-art)
   - 3D reconstruction from sonar data (77% lower Chamfer Distance)
   - Azimuth streak modeling and removal
   - Polar-to-Cartesian coordinate conversion for sonar images
   - Training scripts for:
     - Novel view synthesis (`run_nvs_infra_360_1.sh`)
     - 3D reconstruction (`run_3D_monohansett.sh`)
   - Evaluation tools:
     - Image quality metrics: PSNR, SSIM, LPIPS
     - Point cloud metrics: Chamfer Distance
   - **Key Files:**
     - `sonar_splat/` directory (~24 subdirectories)
     - `sonar_splat/gsplat/` - Core Gaussian splatting library with CUDA extensions
     - `sonar_splat/sonar/` - Sonar-specific modules (dataloader, conversion, visualization)
     - `sonar_splat/examples/sonar_simple_trainer.py` - Main training script (65,753 lines)
     - `sonar_splat/examples/sonar_image_fitting.py` - Single-image fitting
     - `sonar_splat/scripts/evaluate_imgs.py` - Image evaluation
     - `sonar_splat/scripts/compute_pcd_metrics_ply.py` - Point cloud metrics

3. **Installation and Environment Setup**
   - Documented installation procedures for both frameworks
   - Separate conda environments:
     - `watersplatting` (Python 3.8, CUDA 11.8, GCC 11)
     - `sonarsplat` (Python 3.10, CUDA 12.4)
   - Comprehensive dependency management:
     - Main `requirements.txt` (286 lines) - Data analysis tools
     - `sonar_splat/requirements.txt` - SonarSplat-specific dependencies
     - WaterSplatting dependencies (PyTorch 2.1.2, tiny-cuda-nn, Nerfstudio)
   - Setup guides:
     - `sonar_splat/SETUP_LINUX.md` - Linux installation
     - `sonar_splat/SETUP_WINDOWS.md` - Windows installation (experimental)

**Evidence:**
- `water_splatting/` - Complete WaterSplatting implementation
- `sonar_splat/` - Complete SonarSplat implementation
- `README.md` - Unified documentation (660 lines)
- `MERGE_REPORT.md` - Technical merge documentation (345 lines)

---

### Milestone 5: Optical Imagery Analysis Tools ✅

**Completed Components:**

1. **Comprehensive EDA Tool** (`optical_imagery_eda.py`)
   - Single-pass image processing (3x faster than multi-pass approaches)
   - Generates interactive HTML report with 18+ Chart.js visualizations
   - Analysis dimensions:
     - **Color Profiling:** Blue/red ratio, brightness distribution, color channel histograms
     - **Quality Metrics:** Blur detection (Laplacian variance), sharpness, edge density (Canny), contrast
     - **Content Classification:** Coral, rock, seafloor, open water, marine life
     - **COLMAP Reconstruction Analysis:** Registration rate, reprojection error, point cloud density
     - **Gaussian Splatting Readiness:** Overall suitability score (0-100)
   - Per-scene quality metrics and readiness assessment
   - **Output:** `underwater_eda_report.html` - Interactive dashboard opened in browser

2. **Quick Dataset Inspection** (`underwater_optical_datasets_analysis.py`)
   - Catalogs underwater optical datasets scored for Gaussian splatting suitability
   - Categories:
     - Enhancement/restoration datasets: SQUID, UIEB, EUVP, UVEB
     - Detection datasets: Brackish, TrashCan, WPBB
     - Segmentation datasets: SUIM, DeepFish
     - Tracking datasets: UOT100
     - Classification datasets: Fish4Knowledge
   - Each dataset scored 0-100 for optical imagery suitability
   - **Output:** Printed table with dataset names, sizes, tasks, and scores

**Evidence:**
- `scripts/eda/optical_imagery_eda.py` (1,349 lines)
- `scripts/eda/underwater_optical_datasets_analysis.py` (85 lines)

---

### Milestone 6: Repository Organization and Documentation ✅

**Completed Components:**

1. **Unified Directory Structure**
   - Organized 15+ scattered Python scripts into logical subdirectories:
     - `scripts/dataset_tools/` - Download, rank, organize (3 scripts)
     - `scripts/eda/` - Analysis and quality assessment (2 scripts)
     - `scripts/preprocessing/` - Clean and prepare datasets (1 script)
     - `scripts/template_matching/` - Object detection and isolation (3 scripts)
   - Separated notebooks: `notebooks/` (3 Jupyter notebooks)
   - Centralized outputs: `outputs/` (4 analysis files)
   - Core implementations: `water_splatting/`, `sonar_splat/`
   - Scene creation: `Download Datasets/`

2. **Comprehensive Documentation**
   - **README.md** (660 lines):
     - Project overview and architecture diagram
     - Detailed component descriptions
     - Installation guide (6-step process with troubleshooting)
     - Usage guide with typical workflows
     - Dependency table
     - Branch origins and integration strategy
     - Resources and citations
   - **MERGE_REPORT.md** (345 lines):
     - Branch analysis summaries
     - Conflict resolution strategies
     - Code organization changes
     - Testing recommendations
     - File statistics

3. **Branch Consolidation**
   - Successfully merged 4 specialized branches into unified `main`:
     - `image_class` - Dataset EDA and CLIP classification
     - `watersplatting_initial` - Optical 3D reconstruction
     - `optical-eda` - Comprehensive analysis tools
     - `sonar-splat` - Sonar 3D reconstruction
   - Resolved conflicts in:
     - `requirements.txt` - Consolidated 3 versions into single comprehensive file
     - `.gitignore` - Combined patterns from all branches
     - Analysis notebooks - Kept most comprehensive versions
   - All unique functionality preserved, no feature loss

**Evidence:**
- Organized repository structure (see `ls -la` output)
- `README.md` - Comprehensive project documentation
- `MERGE_REPORT.md` - Technical merge documentation
- Git history: 5 merge commits consolidating branches

---

## 4. Future Timeline (Rest of Semester)

### Current Status Assessment
We have completed the infrastructure and integration phases ahead of schedule. The remaining work focuses on model training, validation, and deliverables preparation.

### Week-by-Week Plan (11 Weeks Remaining)

#### **Week 1-2 (March 18 - March 31): Dataset Preparation and Scene Creation**
**Goal:** Prepare cleaned datasets and create COLMAP scenes for training

**Tasks:**
1. Run full preprocessing pipeline on FLSea optical imagery
   - CLIP classification to filter low-quality frames
   - Artifact detection and cropping
   - Export top-200 ranked images per survey
2. Create COLMAP scenes using `create_valid_scene.py`
   - Run Structure-from-Motion (SfM) on cleaned images
   - Verify camera pose estimation quality
   - Generate train/eval splits
3. Prepare sonar datasets
   - Download SonarSplat example datasets (basin_horizontal_infra_1, monohansett_3D)
   - Convert polar sonar images to Cartesian coordinates
   - Verify sensor pose data integrity

**Deliverables:**
- 3+ cleaned optical image scenes ready for WaterSplatting training
- 2+ sonar datasets ready for SonarSplat training
- Scene quality report documenting COLMAP reconstruction statistics

**Success Criteria:**
- COLMAP registration rate > 80% for optical scenes
- Reprojection error < 1.0 pixel average
- Sonar pose data successfully loaded without errors

---

#### **Week 3-4 (April 1 - April 14): WaterSplatting Model Training (Optical Imagery)**
**Goal:** Train and evaluate WaterSplatting models on underwater optical datasets

**Tasks:**
1. Environment setup verification
   - Verify CUDA 11.8 and GCC 11 installation
   - Test WaterSplatting installation: `python -c "import water_splatting"`
   - Test Nerfstudio CLI: `ns-train --help`
2. Train WaterSplatting models on prepared scenes
   ```bash
   ns-train water-splatting \
       --data watersplatting_data/flsea_vi/u_canyon \
       --output-dir outputs/u_canyon_run1 \
       --max-num-iterations 30000
   ```
3. Monitor training metrics
   - Track PSNR, SSIM on validation set
   - Log training loss curves
   - Monitor Gaussian count and densification
4. Render novel views and evaluate quality
   - Generate novel viewpoint images
   - Compute image quality metrics (PSNR, SSIM, LPIPS)
   - Visual inspection for artifacts (floaters, holes, blurriness)
5. Export 3D point clouds and meshes for visualization

**Deliverables:**
- Trained WaterSplatting models for 3+ underwater scenes
- Novel view synthesis results with quantitative metrics
- 3D reconstructed point clouds and meshes
- Training log analysis report

**Success Criteria:**
- PSNR > 25 dB on validation views (underwater scenes typically lower than terrestrial)
- SSIM > 0.85
- Visually plausible novel views without major artifacts

---

#### **Week 5-6 (April 15 - April 28): SonarSplat Model Training (Sonar Imagery)**
**Goal:** Train and evaluate SonarSplat models on imaging sonar datasets

**Tasks:**
1. Environment setup verification
   - Verify CUDA 12.4 installation
   - Test SonarSplat installation: `python -c "import gsplat"`
   - Verify sonar data loading: run dataloader test script
2. Novel view synthesis training
   ```bash
   cd sonar_splat
   bash scripts/run_nvs_infra_360_1.sh \
       --data_dir datasets/basin_horizontal_infra_1 \
       --results_dir results/nvs_infra1
   ```
3. 3D reconstruction training
   ```bash
   bash scripts/run_3D_monohansett.sh \
       --data_dir datasets/monohansett_3D \
       --results_dir results/3d_monohansett
   ```
4. Evaluate novel view synthesis
   - Render test views using trained models
   - Compute PSNR, SSIM, LPIPS metrics
   - Compare against baselines (NeRF, other methods if available)
5. Evaluate 3D reconstruction
   - Convert Gaussian splat to mesh: `python mesh_gaussian.py`
   - Align predicted mesh to ground truth using ICP
   - Compute Chamfer Distance
6. Azimuth streak analysis
   - Visualize rendered sonar images
   - Assess azimuth streak modeling quality
   - Compare before/after streak removal

**Deliverables:**
- Trained SonarSplat models for novel view synthesis and 3D reconstruction
- Quantitative evaluation results (CSV with metrics)
- 3D reconstructed meshes from sonar data
- Azimuth streak removal demonstration

**Success Criteria:**
- Novel view synthesis PSNR within 2 dB of paper results (+3.2 dB vs. baselines)
- Chamfer Distance comparable to paper (77% lower than baselines)
- Visually coherent 3D reconstructions aligned with ground truth

---

#### **Week 7-8 (April 29 - May 12): Cross-Modal Analysis and Validation**
**Goal:** Compare optical vs. sonar reconstruction quality and explore multi-modal scenarios

**Tasks:**
1. Comparative analysis
   - Identify scenes with both optical and sonar data (if available)
   - Train WaterSplatting and SonarSplat on same scene
   - Compare 3D reconstruction geometry
   - Analyze strengths/weaknesses of each modality
2. Multi-modal visualization
   - Overlay optical and sonar point clouds
   - Align using ICP or manual registration
   - Generate side-by-side comparison visualizations
3. Failure case analysis
   - Document challenging scenarios:
     - Very turbid water (optical failure)
     - Low acoustic reflectivity (sonar failure)
     - Complex geometry (both methods)
   - Analyze failure modes and propose mitigation strategies
4. Create demonstration videos
   - Novel view synthesis fly-through animations
   - 3D reconstruction turntables
   - Before/after artifact removal comparisons

**Deliverables:**
- Optical vs. sonar comparison report
- Multi-modal visualization overlays
- Failure case documentation with analysis
- Demonstration videos (MP4 format)

**Success Criteria:**
- Clear documentation of when to use optical vs. sonar
- Successful multi-modal alignment (if data available)
- Identification of complementary use cases

---

#### **Week 9-10 (May 13 - May 26): Diffusion Models for Synthetic Terrain Generation**
**Goal:** Implement diffusion models to generate infinite synthetic underwater terrains from scratch for AI submarine navigation training

**Motivation:** Unlike Gaussian Splatting (which reconstructs from real data), diffusion models enable **generative synthesis** of entirely new terrains with:
- Infinite diversity (not limited by real scans)
- Controllable generation (text prompts: "rocky seafloor", "shipwreck debris")
- Multi-modal output (sonar + optical imagery from same geometry)
- Data augmentation for robust AI training

---

### **Mathematical Foundation**

**Core Diffusion Process:**
```
Forward (add noise):  q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)
Reverse (denoise):    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
Training objective:   L = E[||ε - ε_θ(x_t, t)||²]
```

**Latent Diffusion (Stable Diffusion):**
- Operate in compressed latent space: 512×512×3 → 64×64×4 (48× reduction)
- Memory: 16 GB → 2-4 GB VRAM
- Speed: 5× faster than pixel-space diffusion
- Architecture: VAE encoder → UNet diffusion → VAE decoder

**3D Generation Approaches:**
1. **2D Multi-View Diffusion → 3D Reconstruction** (Recommended)
   - Generate multi-view sonar images with pose conditioning
   - Reconstruct 3D via existing SonarSplat pipeline
2. **Direct 3D Diffusion** (Point-E, Shap-E)
   - Generate 3D point clouds or implicit neural representations
   - Faster but lower quality than multi-view approach

---

### **Week 9: Dataset Preparation & Initial Training**

**Day 1-2: Dataset Preparation**
```bash
# Create terrain-labeled dataset from FLSea
python scripts/diffusion/prepare_diffusion_dataset.py \
    --input_dir data/flsea_sonar/ \
    --output_dir data/diffusion_training/ \
    --label_mode auto  # Use CLIP for auto-labeling

# Expected output structure:
# diffusion_training/
#   ├── rocky/          # 5000 images (rocky terrains)
#   ├── sandy/          # 3000 images (sandy seafloor)
#   ├── shipwreck/      # 2000 images (debris fields)
#   ├── vegetation/     # 1500 images (coral/kelp)
#   └── metadata.json   # Image metadata + poses
```

**Day 3-5: Fine-Tune Stable Diffusion**
```bash
# Install dependencies
pip install diffusers transformers accelerate xformers wandb

# Fine-tune on sonar dataset
python scripts/diffusion/train_sonar_diffusion.py \
    --pretrained_model "stabilityai/stable-diffusion-2-1" \
    --data_dir data/diffusion_training/ \
    --output_dir models/sonar_diffusion/ \
    --resolution 512 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --epochs 50 \
    --learning_rate 1e-5 \
    --use_ema \
    --mixed_precision fp16

# Training time: ~2-3 days on A100 GPU
# Checkpoint every 1000 steps for monitoring
```

**Day 6-7: Baseline Generation Tests**
```python
# Generate synthetic sonar images
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "models/sonar_diffusion/checkpoint-5000",
    torch_dtype=torch.float16
).to("cuda")

# Test prompts
prompts = [
    "rocky underwater terrain with large boulders",
    "sandy seafloor with ripple patterns",
    "shipwreck debris field with metal structures",
    "underwater canyon with steep rock walls"
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt, num_inference_steps=50).images[0]
    image.save(f"test_generation_{i:02d}.png")
```

---

### **Week 10: Multi-View Synthesis & 3D Integration**

**Day 1-3: Implement Pose-Conditioned Generation**
```python
# Add ControlNet for pose conditioning
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "models/sonar_diffusion/checkpoint-5000",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Generate multi-view terrain
def generate_multiview_terrain(terrain_id, num_views=20):
    """Generate consistent views from multiple angles"""
    views = []

    # Circular camera trajectory
    azimuths = np.linspace(0, 360, num_views, endpoint=False)
    ranges = np.random.uniform(5, 15, num_views)  # 5-15 meters

    for i, (az, rng) in enumerate(zip(azimuths, ranges)):
        # Compute pose
        pose = {
            'azimuth': az,
            'range': rng,
            'elevation': np.random.uniform(-10, 10)
        }

        # Generate depth map for ControlNet conditioning
        depth_map = pose_to_depth_map(pose)

        # Generate view
        view = pipe(
            prompt=f"underwater terrain {terrain_id}",
            image=depth_map,
            num_inference_steps=50,
            controlnet_conditioning_scale=0.7
        ).images[0]

        views.append({'image': view, 'pose': pose})

    return views
```

**Day 4-5: Generate Synthetic Multi-View Datasets**
```bash
# Batch generate 100 synthetic terrains
python scripts/diffusion/generate_synthetic_terrains.py \
    --model_path models/sonar_diffusion/checkpoint-5000 \
    --num_terrains 100 \
    --num_views_per_terrain 20 \
    --output_dir synthetic_multiview/ \
    --terrain_types rocky,sandy,shipwreck,vegetation

# Output:
# synthetic_multiview/
#   ├── terrain_000/
#   │   ├── view_00.png (azimuth=0°)
#   │   ├── view_01.png (azimuth=18°)
#   │   ├── ...
#   │   └── poses.json
#   ├── terrain_001/
#   └── ...
```

**Day 6-7: 3D Reconstruction Pipeline Integration**
```bash
# For each synthetic terrain, reconstruct 3D
for terrain_id in {000..099}; do
    # 1. Convert images to SonarSplat format (pickle files)
    python scripts/diffusion/create_sonar_pickle.py \
        --input_dir synthetic_multiview/terrain_${terrain_id}/ \
        --output_dir synthetic_sonar_data/terrain_${terrain_id}/ \
        --sonar_params config/sonar_params.json

    # 2. Run SonarSplat 3D reconstruction
    cd sonar_splat
    bash scripts/run_3D_monohansett.sh \
        ../synthetic_sonar_data/terrain_${terrain_id}/ \
        ../results/synthetic_terrain_${terrain_id}/

    # 3. Export mesh for AI training
    python scripts/mesh_gaussian.py \
        --ply_path ../results/synthetic_terrain_${terrain_id}/output.ply \
        --num_samples 100 \
        --threshold 0.9 \
        --output_dir ../training_meshes/terrain_${terrain_id}/
done

# Result: 100 synthetic 3D terrains ready for AI training
```

---

### **Alternative: Procedural Generation (Backup Plan)**

If diffusion training is delayed, use **procedural + physics-based approach**:

```python
# scripts/procedural/generate_terrain.py

import numpy as np
from noise import pnoise2  # Perlin noise
import trimesh

class ProceduralTerrainGenerator:
    def generate_seafloor(self, size=100, resolution=0.5, seed=None):
        """Generate procedural underwater terrain using Perlin noise"""
        if seed:
            np.random.seed(seed)

        # Multi-scale Perlin noise
        x = np.arange(0, size, resolution)
        y = np.arange(0, size, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # Add octaves for realistic terrain
        for octave in range(1, 6):
            freq = 2 ** octave
            amp = 1.0 / freq
            Z += amp * np.array([[
                pnoise2(i * freq / 100, j * freq / 100, seed)
                for j in range(X.shape[1])
            ] for i in range(X.shape[0])])

        # Add features
        Z = self.add_rocks(X, Y, Z, num_rocks=50)
        Z = self.add_vegetation(X, Y, Z, coverage=0.3)

        # Create mesh
        mesh = self.create_mesh(X, Y, Z)
        return mesh

    def simulate_sonar_image(self, mesh, sensor_pose):
        """Ray-trace mesh to generate synthetic sonar image"""
        sonar_image = np.zeros((512, 512))  # range × azimuth

        # Ray casting for each pixel
        for az_idx in range(512):
            for r_idx in range(512):
                # Compute ray direction
                ray_origin, ray_dir = compute_sonar_ray(
                    az_idx, r_idx, sensor_pose
                )

                # Ray-mesh intersection
                locations = mesh.ray.intersects_location(
                    [ray_origin], [ray_dir]
                )

                if len(locations) > 0:
                    # Compute intensity (Lambertian + attenuation)
                    intensity = compute_acoustic_return(
                        locations[0], ray_dir, mesh
                    )
                    sonar_image[r_idx, az_idx] = intensity

        # Add speckle noise
        sonar_image += np.random.rayleigh(0.05, sonar_image.shape)
        return np.clip(sonar_image, 0, 1)

# Generate 1000 procedural terrains
generator = ProceduralTerrainGenerator()
for seed in range(1000):
    mesh = generator.generate_seafloor(seed=seed)
    mesh.export(f"procedural_terrains/terrain_{seed:04d}.ply")
```

---

### **Deliverables**

**Code Artifacts:**
```
scripts/
├── diffusion/                         # NEW
│   ├── prepare_diffusion_dataset.py   # Dataset labeling
│   ├── train_sonar_diffusion.py       # Fine-tune Stable Diffusion
│   ├── generate_synthetic_terrains.py # Multi-view generation
│   ├── create_sonar_pickle.py         # Convert to SonarSplat format
│   └── synthetic_to_3d.sh             # End-to-end pipeline
├── procedural/                        # NEW (backup)
│   ├── generate_terrain.py            # Perlin noise terrains
│   ├── simulate_sonar.py              # Physics-based rendering
│   └── batch_generate.py              # Batch processing
└── ... (existing)

models/
└── sonar_diffusion/                   # NEW
    ├── checkpoint-5000/               # Trained model
    ├── training_logs/                 # TensorBoard logs
    └── config.json

data/
├── diffusion_training/                # NEW (labeled dataset)
└── synthetic_multiview/               # NEW (generated terrains)

results/
└── synthetic_terrain_*/               # NEW (3D reconstructions)
```

**Research Outputs:**
1. **Trained Diffusion Model**
   - Checkpoint: `models/sonar_diffusion/checkpoint-5000` (~5 GB)
   - Training logs: TensorBoard metrics (FID, LPIPS, loss curves)
   - Sample generations: 100 test images with prompts

2. **Synthetic Terrain Dataset**
   - 100 multi-view terrains (20 views each = 2,000 sonar images)
   - 3D reconstructed meshes (100 .ply files)
   - Ground truth metadata (poses, terrain types)

3. **Technical Report Section**
   - Mathematical formulation of diffusion models
   - Comparison: Diffusion vs. Gaussian Splatting vs. Procedural
   - Quantitative metrics:
     - FID score (image quality): Target < 15
     - Chamfer Distance (3D accuracy): Target < 0.5m
     - Multi-view consistency: LPIPS < 0.3
   - Qualitative visualizations (generated terrains, 3D meshes)

4. **Integration Demo**
   - Pipeline script: `synthetic_to_3d.sh` (generate → reconstruct → export)
   - Gazebo simulation example with synthetic terrain
   - AI training baseline (if time permits)

---

### **Success Criteria**

**Generation Quality:**
- ✅ FID < 15 (comparable to real sonar, state-of-the-art ~3-5)
- ✅ Visual inspection: 60% of humans cannot distinguish real vs. synthetic
- ✅ Diversity: Cover 4+ terrain types (rocky, sandy, shipwreck, vegetation)

**3D Reconstruction:**
- ✅ Chamfer Distance < 0.5m for 10m range sonar
- ✅ 80%+ point cloud coverage
- ✅ Watertight meshes with < 5% self-intersections

**System Integration:**
- ✅ End-to-end pipeline: Text prompt → Multi-view → 3D mesh
- ✅ Generation speed: < 10 minutes per terrain (20 views + reconstruction)
- ✅ Batch processing: 100 terrains in < 24 hours

**AI Training Readiness:**
- ✅ Export formats: PLY meshes, point clouds, rendered images
- ✅ Metadata: Terrain type labels, ground truth geometry
- ✅ Diversity: Balanced representation across terrain categories

---

### **Risk Mitigation**

**Risk 1: Diffusion Training Takes Longer Than Expected**
- **Mitigation:** Start fine-tuning on Day 1 (3-day buffer)
- **Backup:** Use procedural generation (ready in 2-3 days)
- **Fallback:** Pre-trained Stable Diffusion without fine-tuning

**Risk 2: Multi-View Consistency Issues**
- **Mitigation:** Use ControlNet for explicit pose conditioning
- **Backup:** Generate independent views, rely on SonarSplat for consistency
- **Fallback:** Single-view generation only (still useful for data augmentation)

**Risk 3: Computational Resources**
- **Mitigation:** Fine-tuning requires 1× A100 (40GB) for ~3 days
- **Backup:** Use cloud GPUs (Paperspace, Lambda Labs: ~$1.10/hour)
- **Fallback:** Reduce resolution 512 → 256 (4× faster training)

**Risk 4: Integration with SonarSplat Fails**
- **Mitigation:** Validate data format conversion early (Day 6)
- **Backup:** Manual format conversion + debugging
- **Fallback:** Export only 2D images (still valuable for AI training)

---

### **Key References**

**Diffusion Models:**
1. Ho et al., "Denoising Diffusion Probabilistic Models (DDPM)", NeurIPS 2020
2. Song et al., "Denoising Diffusion Implicit Models (DDIM)", ICLR 2021
3. Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022

**3D Generation:**
4. Nichol et al., "Point-E: Generating 3D Point Clouds from Complex Prompts", arXiv 2022
5. Jun & Nichol, "Shap-E: Generating Conditional 3D Implicit Functions", arXiv 2023
6. Zeng et al., "LION: Latent Point Diffusion Models for 3D Shape Generation", NeurIPS 2022

**Terrain Generation (2024-2025):**
7. Marsal et al., "MESA: Text-Driven Terrain Generation Using Latent Diffusion", arXiv 2025
8. Zhang et al., "EarthCrafter: Scalable 3D Earth Generation via Dual-Sparse Latent Diffusion", arXiv 2025

**Control & Multi-Modal:**
9. Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)", ICCV 2023
10. Zhan et al., "MedM2G: Unifying Medical Multi-Modal Generation via Cross-Guided Diffusion", CVPR 2024

---

**Note:** While diffusion models represent cutting-edge generative AI for infinite terrain synthesis, this remains an **optional extension**. If time constraints arise, Week 10 can be reallocated to:
- Additional technical report writing (extend Week 11)
- More thorough WaterSplatting/SonarSplat experiments (extend Weeks 3-8)
- Deployment-focused work (real-time optimization, submarine integration)

The core project deliverables (Gaussian Splatting reconstruction) are sufficient for a successful project completion.

---

#### **Week 11 (May 27 - June 2): Technical Report Writing**
**Goal:** Draft comprehensive 5-page technical report documenting entire project

**Report Structure:**

1. **Introduction** (0.75 pages)
   - Motivation: Multi-modal AUV perception for underwater environments
   - Problem statement: Need for synthetic training data
   - Approach: Gaussian Splatting for optical and sonar imagery
   - Contributions summary

2. **Background and Related Work** (0.75 pages)
   - 3D Gaussian Splatting fundamentals
   - Underwater imaging challenges (absorption, scattering, turbidity)
   - Sonar imaging principles (range/azimuth projection, acoustic reflectance)
   - Prior work: WaterSplatting, SonarSplat, underwater NeRFs

3. **Methodology** (1.5 pages)
   - **Data Pipeline:**
     - Dataset acquisition (FLSea, public sonar datasets)
     - CLIP-based quality ranking
     - Artifact detection and preprocessing
   - **WaterSplatting for Optical Imagery:**
     - Underwater light propagation model
     - COLMAP scene creation
     - Training procedure
   - **SonarSplat for Sonar Imagery:**
     - Sonar rasterization to range/azimuth plane
     - Azimuth streak modeling
     - 3D reconstruction pipeline

4. **Results** (1.5 pages)
   - **Optical Results:**
     - Novel view synthesis metrics (PSNR, SSIM, LPIPS)
     - 3D reconstruction visualizations
     - Artifact removal before/after
   - **Sonar Results:**
     - Novel view synthesis comparison to baselines
     - 3D reconstruction Chamfer Distance
     - Azimuth streak removal demonstration
   - **Cross-Modal Analysis:**
     - Optical vs. sonar reconstruction comparison
     - Complementary use cases
   - **(Optional) Diffusion Model Results:**
     - Synthetic image quality
     - Cross-modal generation

5. **Discussion and Future Work** (0.5 pages)
   - Lessons learned
   - Challenges encountered (data quality, computational requirements)
   - Future directions:
     - Multi-modal fusion (optical + sonar joint reconstruction)
     - Diffusion models for data augmentation
     - Real-time deployment on AUVs
     - Larger-scale datasets

**Writing Tasks:**
- Draft all sections
- Generate figures:
   - Pipeline architecture diagram
   - Qualitative results (novel views, 3D reconstructions)
   - Quantitative result tables and charts
- Create captions and references
- Internal review and revisions

**Deliverables:**
- Draft technical report (5 pages, IEEE or similar format)
- Figures and tables (high-resolution)
- Bibliography

**Success Criteria:**
- Clear and concise writing
- All key results documented with visualizations
- Reproducible methodology description

---

#### **Week 12 (June 3 - June 9): Code Cleanup, Documentation, and Final Deliverables**
**Goal:** Finalize all deliverables for submission

**Tasks:**

1. **Code Package Finalization**
   - Clean up commented-out code and debug prints
   - Add docstrings to all major functions
   - Create example usage scripts:
     - `examples/train_watersplatting.sh` - Full optical pipeline
     - `examples/train_sonarsplat.sh` - Full sonar pipeline
   - Test installation on fresh environment:
     - Create new conda environment
     - Follow installation instructions in README
     - Run example scripts to verify functionality
   - Update `requirements.txt` if dependencies changed

2. **Documentation Polish**
   - Update README.md with:
     - Final project results summary
     - Citation information
     - Acknowledgments
   - Create `USAGE.md`:
     - Step-by-step tutorial for new users
     - Expected outputs at each stage
     - Troubleshooting common errors
   - Add inline comments to complex code sections
   - Generate API documentation (Sphinx or similar)

3. **Final Technical Report Revisions**
   - Incorporate feedback from internal reviews
   - Proofread for grammar and clarity
   - Verify all figures and tables are correctly referenced
   - Double-check citations and references
   - Final formatting pass (page limits, font sizes, margins)

4. **Deliverables Packaging**
   - Create final submission archive:
     - `LM1_Sonar_Imagery_TechnicalReport.pdf` (5 pages)
     - `LM1_Sonar_Imagery_Code.zip` (entire repository)
     - `LM1_Sonar_Imagery_Results/` (trained models, visualizations, videos)
     - `README_SUBMISSION.txt` (how to use deliverables)
   - Verify archive is complete and extractable
   - Test code runs from archive on clean system

5. **Presentation Preparation**
   - Create slide deck (15-20 slides):
     - Motivation and problem statement
     - Methodology overview
     - Key results (optical and sonar)
     - Demonstrations (videos)
     - Future work
   - Prepare speaker notes
   - Rehearse presentation (practice run-through)

6. **Final Testing**
   - Run end-to-end pipeline on new scene
   - Verify all scripts execute without errors
   - Check all visualizations render correctly
   - Validate metrics computation

**Deliverables:**
- **Code Package:**
  - Cleaned and documented codebase
  - Example usage scripts
  - Installation instructions
  - API documentation
- **Technical Report (Final):**
  - 5-page PDF (IEEE or similar format)
  - High-resolution figures
  - Complete bibliography
- **Results Archive:**
  - Trained model checkpoints
  - Novel view synthesis outputs
  - 3D reconstructed meshes
  - Demonstration videos
- **Presentation:**
  - Slide deck (PDF + PPTX)
  - Speaker notes
- **Submission README:**
  - How to install and run code
  - Description of all files in archive

**Success Criteria:**
- Code runs successfully on fresh installation
- Technical report polished and submission-ready
- All deliverables organized and documented
- Presentation rehearsed and timed appropriately

---

### Timeline Summary (Gantt Chart)

| Week | Dates | Milestone | Deliverables |
|------|-------|-----------|--------------|
| 1-2 | Mar 18 - Mar 31 | Dataset Preparation | Cleaned scenes, quality reports |
| 3-4 | Apr 1 - Apr 14 | WaterSplatting Training | Optical models, novel views |
| 5-6 | Apr 15 - Apr 28 | SonarSplat Training | Sonar models, 3D reconstructions |
| 7-8 | Apr 29 - May 12 | Cross-Modal Analysis | Comparison report, demo videos |
| 9-10 | May 13 - May 26 | Diffusion Exploration (Optional) | Synthetic image experiments |
| 11 | May 27 - Jun 2 | Technical Report Writing | Draft report, figures |
| 12 | Jun 3 - Jun 9 | Final Deliverables | Code package, final report, presentation |

---

### Risk Mitigation

**Potential Risks:**

1. **CUDA Compilation Issues**
   - **Mitigation:** Both frameworks have Docker support. If local compilation fails, use containerized environments.
   - **Backup Plan:** Use pre-compiled binaries or cloud GPU instances (Google Colab, Paperspace)

2. **Insufficient Training Data Quality**
   - **Mitigation:** We've already built robust preprocessing pipeline. CLIP ranking ensures high-quality images.
   - **Backup Plan:** If FLSea data insufficient, use alternative datasets (SQUID, EUVP, SeaThru-NeRF)

3. **Computational Resource Constraints**
   - **Mitigation:** Train on smaller scenes first. WaterSplatting trains in ~30 min on consumer GPU for small scenes.
   - **Backup Plan:** Request access to Lockheed Martin compute resources or use cloud GPUs (AWS, Lambda Labs)

4. **Diffusion Model Experiments Delayed**
   - **Mitigation:** Diffusion experiments are optional extension. Can be deprioritized without affecting core deliverables.
   - **Backup Plan:** Move to "Future Work" section of technical report

5. **Technical Report Writing Time Underestimated**
   - **Mitigation:** Start outlining report structure in Week 9. Draft introduction and methodology sections in parallel with experiments.
   - **Backup Plan:** Week 10 can be reallocated to report writing if diffusion experiments are skipped

---

## Summary

**Project Status:** ✅ **On Track**

We have successfully completed the infrastructure, integration, and preprocessing phases ahead of schedule. The comprehensive data pipeline, artifact removal tools, and Gaussian Splatting framework integrations provide a solid foundation for the remaining work.

**Next Immediate Steps:**
1. Dataset preparation and scene creation (Weeks 1-2)
2. WaterSplatting training on optical imagery (Weeks 3-4)
3. SonarSplat training on sonar data (Weeks 5-6)

**Key Strengths:**
- Robust data preprocessing infrastructure reduces risk of training failures
- State-of-the-art implementations (WaterSplatting, SonarSplat) provide strong baselines
- Comprehensive documentation ensures reproducibility
- Unified repository structure simplifies development

**Areas of Focus:**
- Ensure sufficient high-quality training data through rigorous preprocessing
- Monitor training metrics closely and iterate on hyperparameters
- Allocate sufficient time for technical report writing (Week 11)
- Reserve Week 12 for polish and final testing

---

**Document Prepared By:** Claude Code (Anthropic AI Assistant)
**Analysis Date:** March 17, 2026
**Codebase Version:** main branch (commit: fdae286)
