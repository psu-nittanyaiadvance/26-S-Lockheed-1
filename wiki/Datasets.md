# Datasets

Data lives in a directory you specify at container launch time via `DATASET_PATH`.
Inside the container it is always mounted at `/data`.

**Before running the container, set `DATASET_PATH` to your local dataset root:**

```bash
export DATASET_PATH="/path/to/your/data"   # e.g. /media/priyanshu/2TB SSD
bash docker/run.sh
```

All training scripts reference `/data/...` paths inside the container, so as long as
your directory mirrors the structure below the scripts work unchanged.

---

## SonarSplat datasets

Root: `/media/priyanshu/2TB SSD/sonarsplat_dataset/`

Format: directories of PKL files. Each PKL contains one sonar sweep (range × azimuth intensity image + pose).

| Dataset | Frames | Has GT mesh | Scene type | Far plane |
|---------|--------|-------------|------------|-----------|
| `monohansett_3D` | 1228 | yes | Real shipwreck, open water | 10 m |
| `concrete_piling_3D` | 687 | yes | Piling, open water | 10 m |
| `infra_360_1` | 336 | no | Infrastructure, 360° orbit | 5 m |
| `basin_horizontal_infra_1` | 171 | no | Indoor tank, infrastructure | 5 m |
| `rock_semicircle1` | 161 | no | Indoor tank, rock | 5 m |
| `basin_horizontal_empty1` | 163 | no | Indoor tank, empty | 5 m |
| `basin_horizontal_piling_up_down_4` | 111 | no | Indoor tank, piling (up/down) | 5 m |
| `basin_horizontal_piling_1` | 101 | no | Indoor tank, piling | 5 m |
| `pole_qual1` | 29 | no | Indoor tank, pole | 5 m |

**Sonar hardware (all SonarSplat datasets): Oculus M750d**

```
n_array_elements  = 64
element_spacing   = 0.003 m      (~λ/2 at 1.1 MHz)
center_frequency  = 1 100 000 Hz
bandwidth         = 30 000 Hz
speed_of_sound    = 1500 m/s
sigma_r           = c / (4B) = 1.25 cm   (range resolution)
```

---

## AONeuS dataset

Root: `/media/priyanshu/2TB SSD/aoneus_dataset/`

Synthetic turtle scene — the only public AONeuS data release.

```
data/
├── reduced_baseline_0.6x_rgb/      # 60 RGB images + cameras_sphere.npz
└── reduced_baseline_0.6x_sonar/    # 60 sonar PKL files + Config.json
transformed_data/                   # COLMAP format (created by create_valid_z_splat_scene.py)
├── images/
├── sparse/0/
└── depth/                          # per-image depth.npy (sonar slant range)
```

**Sonar configuration (from Config.json):**

```
n_range_bins  = 256
n_az_bins     = 96
sonar_max_range = 5.0 m
depth_scale   ≈ 1.315  (auto-calibrated: median nonzero depth / avg cam dist)
```

**Train/test split:** 52 train / 8 test (every 8th frame is test).

**One-time setup:**

```bash
conda activate sonarsplat
cd "Download Datasets"
python create_valid_z_splat_scene.py
```

This is required before the first Z-Splat run. The `transformed_data/` output is already present.

---

## Converting SonarSplat PKL datasets for Z-Splat v2

Any PKL dataset can be converted to COLMAP format for `train_v2.py` using the universal converter.

```bash
conda activate sonarsplat
cd "Download Datasets"

# Output auto-named ~/datasets/<dataset_name>_zsplat
python convert_to_zsplat.py --data_dir "<data_dir>"

# Options:
#   --output_dir /path    override output location
#   --skip_frames N       subsample (N=1 = all frames, default)
#   --img_threshold 0.02  suppress low-intensity pixels (default 0.02)
#   --max_seed_pts 200    seed points per sampled frame for points3D.bin
```

The converter reads `Config.json` automatically to extract `RangeBins`, `AzimuthBins`, `RangeMax`, and `Azimuth`. It works with all datasets that follow the standard SonarSplat layout (`Data/*.pkl` + `Config.json`).

After conversion, train with:

```bash
cd ../gaussian-splatting-with-depth
bash scripts/run_sonar_v2.sh \
    ~/datasets/<name>_zsplat \
    "<original_pkl_dir>" \
    "<results_dir>" \
    30000
```

See [Training](Training.md) for full examples and hardware presets.

---

## Sunboat dataset

Root: `/media/priyanshu/2TB SSD/sunboat_dataset/processed_session1/`

Real-world dataset. A surface boat in 9–15 m of water imaged by a side-scan sonar.

```
~382 frames (every 10th of 3829, session 1)
Sonar: Oculus M1200d — 256 elements, 0.000625 m spacing, 1.2 MHz, max range ~26.8 m
Near-field clear: 0–5 m wake clutter → range_clear_end = 48 bins
```

**One-time conversion:**

```bash
cd "Download Datasets"
python convert_sunboat.py --subsample 10
```

**Run:**

```bash
cd sonar_splat
bash scripts/run_sunboat_v2.sh "/media/priyanshu/2TB SSD/results"
```

Note that `run_sunboat_v2.sh` uses different hardware parameters from the other SonarSplat scripts. Do not use `run_v2.sh` for sunboat without overriding `--n_array_elements 256 --element_spacing 0.000625 --center_frequency 1200000.0 --far_plane 30 --near_plane -30 --range_clear_end 48`.

---

## Depth scale calibration (AONeuS / Z-Splat)

`depth_scale` converts unitless COLMAP world-units to metres for the Z loss.

Auto-calibration (applied at the start of every Z-Splat v2 training run):

```python
depth_scale = median(nonzero sonar depth in metres) / mean(camera distance from scene centre in world units)
```

For AONeuS this yields ≈ 1.315. The calibrated value is printed at startup as `depth_scale = X.XXX`.

Pass `--depth_scale <value>` to override.
