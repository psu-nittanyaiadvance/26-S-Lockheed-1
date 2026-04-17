# Troubleshooting

---

## Installation Issues

### "CUDA not available" but I have an NVIDIA GPU

PyTorch was likely installed without CUDA support or with the wrong CUDA version.

```bash
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"
nvcc --version
```

The PyTorch CUDA version must match your system:
- SonarSplat needs CUDA 12.4
- WaterSplatting needs CUDA 11.8
- Preprocessing pipeline works without CUDA (CPU or MPS)

If `torch.version.cuda` shows `None`, reinstall:
```bash
# SonarSplat
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# WaterSplatting
pip install torch==2.1.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### gsplat build fails with "unsupported GPU architecture"

`TORCH_CUDA_ARCH_LIST` must match your GPU:

```bash
export TORCH_CUDA_ARCH_LIST="8.6"   # Change to your GPU
```

| GPU | Value |
|-----|-------|
| RTX 2070/2080 (Turing) | 7.5 |
| A100 (Ampere) | 8.0 |
| RTX 3070/3080/3090 (Ampere) | 8.6 |
| RTX 4070/4080/4090 (Ada) | 8.9 |
| H100 (Hopper) | 9.0 |

### gsplat build fails with compiler errors

```bash
echo $CUDA_HOME          # Should print conda prefix
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
which nvcc               # Should find nvcc
pip install ninja         # Speeds up compilation
```

### tiny-cuda-nn won't compile (WaterSplatting)

Most common WaterSplatting issue. Must have GCC 11:

```bash
gcc --version            # Must show 11.x.x
conda install -y -c conda-forge gcc=11 gxx=11
echo $LD_LIBRARY_PATH    # Must include $CONDA_PREFIX/lib
```

### "ModuleNotFoundError: No module named 'transformers'"

Wrong environment. Use the data pipeline venv:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### "ModuleNotFoundError: No module named 'nerfstudio'"

Wrong conda environment:
```bash
conda activate water_splatting
```

### pip install fails with "externally-managed-environment"

Use a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Preprocessing Pipeline Issues

### Artifact layering uses too much memory

The script uses disk-backed memmap arrays, but very large datasets with very high resolution images can still be demanding. If you run out of memory:
- Reduce the number of images (the script processes all images it finds)
- Reduce image resolution (the script resizes to a consistent shape)

### Edge detection finds no peaks

- Check that the median image actually has a visible artifact boundary
- Try lowering `--peak-height` (default 0.12) to detect weaker edges
- Try `--edge auto` to let the script detect which edge the artifact is on
- Increase `--search-frac` to search further into the image

### SAM produces no masks or too many masks

- Verify the SAM checkpoint is downloaded and the path is correct
- For no masks: lower `MIN_AREA_FRACTION` and `MIN_INTENSITY_CONTRAST` in the source code
- For too many masks: raise `MIN_INTENSITY_CONTRAST` or lower `MAX_AREA_FRACTION`
- Ensure you're passing the median image (not a raw frame) — SAM works best on the median because artifacts are sharp

### crop.py preview looks wrong

- Double-check axis and keep side: `--axis horizontal --keep top` removes the bottom portion
- Verify the crop line value makes sense by looking at the median image pixel coordinates
- Use the edge detection boundary points as a guide for choosing the crop line

---

## SonarSplat Issues

### Training produces black or empty renders

- Verify `.pkl` files contain `data['PoseSensor']` (4x4 matrix) and `data['ImagingSonar']` (numpy array)
- Check `bounds.txt` — values should be reasonable 3D coordinates
- Check `Config.json` — sonar parameters must match the actual sensor
- Open a few `.png` files from `sonar_images/` to verify they have content

### Training loss doesn't decrease

- Verify poses form a coherent trajectory (not all identical, not wildly scattered)
- Check sonar images have actual content (not all black)
- Ensure `--skip_frames 1` (not 0)

### Poor 3D reconstruction metrics

**ICP alignment is mandatory.** Predicted and ground truth point clouds must be in the same coordinate frame before computing Chamfer Distance.

### "No module named 'gsplat'"

Rebuild gsplat:
```bash
cd sonar_splat
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.6"
pip install --no-build-isolation -e .
```

---

## WaterSplatting Issues

### COLMAP produces 0 models

COLMAP couldn't estimate camera poses. Common causes:
- **Insufficient overlap:** Adjacent frames need 60%+ visual overlap
- **Featureless images:** Plain sand or murky water have too few features
- **Pure rotation:** Camera must translate, not just rotate
- **Too few images:** Use 50+ images per scene
- **Blurry images:** Motion blur destroys features

### Viewer won't start

```bash
nvidia-smi                                          # Check GPU memory
ls outputs/<run_name>/nerfstudio_models/             # Verify checkpoint exists
ns-viewer --load-config outputs/<run>/config.yml --viewer.port 8080  # Try different port
```

### Port forwarding not working (remote server)

```bash
# Verify tunnel is active
netstat -an | grep 7007

# Two-hop SSH
ssh -L 7007:localhost:7007 -J user@jump-host user@compute-node

# If port blocked, remap
ssh -L 8080:localhost:7007 user@remote-server
# Then open http://localhost:8080
```

### Training is very slow

- First evaluation step is slow (caching undistorted images). Subsequent ones are faster
- Check GPU utilization with `nvidia-smi` — low utilization may indicate a data loading bottleneck

### "ModuleNotFoundError: No module named 'tinycudann'"

Reinstall and watch for errors:
```bash
conda activate water_splatting
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Check GCC version (must be 11.x) if compilation fails.

---

## Hardware Requirements

| Component | Preprocessing | SonarSplat | WaterSplatting |
|-----------|--------------|------------|----------------|
| **GPU** | Optional (CUDA/MPS for SAM) | NVIDIA required | NVIDIA required |
| **VRAM** | ~2 GB for SAM | 8 GB+ (16+ recommended) | 8 GB+ (16+ recommended) |
| **RAM** | 4 GB | 16 GB | 16 GB |
| **Disk** | Minimal | ~10 GB | ~10 GB |

### Apple Silicon Notes

- The preprocessing pipeline works on Apple Silicon Macs using MPS acceleration
- SAM can run on MPS but is slower than CUDA
- SonarSplat and WaterSplatting **do not** work on Apple Silicon — they require NVIDIA CUDA
