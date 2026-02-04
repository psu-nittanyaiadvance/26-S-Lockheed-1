WaterSplatting Setup and Profiling Guide
Complete guide for installing, running, and profiling the WaterSplatting underwater 3D reconstruction system.

Part 1: Installation
Step 1: Create Clean Conda Environment
conda create --name water_splatting -y python=3.8
conda activate water_splatting
python -m pip install --upgrade pip

Step 2: Install PyTorch with CUDA 11.8 (CRITICAL - Must Be First!)
# Clean any existing installations
pip uninstall torch torchvision functorch tinycudann

# Install PyTorch with CUDA 11.8 support
python3 -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

Step 3: Install CUDA Toolkit via Conda
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

Step 4: Install GCC 11 (Required for Compilation)
conda install -y -c conda-forge gcc=11 gxx=11

Step 5: Set Environment Variables for CUDA
# Set library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LDFLAGS="-L$CONDA_PREFIX/lib -L/usr/lib/x86_64-linux-gnu"

# Verify libcuda.so symlink exists
ls -la $CONDA_PREFIX/lib/libcuda.so

Step 6: Verify Installation So Far
conda activate water_splatting

echo "=== GCC Version ==="
gcc --version | head -1
g++ --version | head -1

echo -e "\n=== CMake ==="
which cmake
cmake --version 2>/dev/null || echo "CMake not found"

echo -e "\n=== CUDA Variables ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc path: $(which nvcc)"
nvcc --version | tail -1

echo -e "\n=== PyTorch CUDA ==="
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"

Expected output:
GCC/G++ version 11.x.x
nvcc version 11.8
PyTorch CUDA: 11.8
CUDA available: True
Device: [Your GPU name]
Step 7: Install tiny-cuda-nn
python3 -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

This step compiles CUDA code and may take 5-10 minutes.
Step 8: Install Nerfstudio
pip install nerfstudio==1.1.4
ns-install-cli

Step 9: Install WaterSplatting
# Navigate to your WaterSplatting directory
cd /path/to/26-S-Lockheed-1

# Install in editable mode (use --no-use-pep517 to avoid build isolation issues)
python3 -m pip install --no-use-pep517 -e .



Part 2: Running the Viewer
Start the Interactive Viewer
# Navigate to your data directory
cd /data/Lockheed1-Spring26

# Start viewer with a trained model
ns-viewer --load-config outputs/curasao_run1/water-splatting/2026-01-30_101250/config.yml

The viewer will:
Load the checkpoint
Cache/undistort evaluation images (takes a few minutes first time)
Start server at http://0.0.0.0:7007
Access Viewer from Your Local Machine
Since you're on a remote server, you need SSH port forwarding:
Single command (recommended):
ssh -L 7007:localhost:7007 -J apd6062@ssh.ist.psu.edu apd6062@nittanyaicompute.ist.psu.edu

Or two-step approach:
# Step 1: Connect to IST SSH aggregator with port forwarding
ssh -L 7007:localhost:7007 apd6062@ssh.ist.psu.edu

# Step 2: From that session, connect to compute node
ssh -L 7007:localhost:7007 apd6062@nittanyaicompute.ist.psu.edu

Then open in your browser:
http://localhost:7007

You'll see the interactive 3D viewer with real-time FPS display in the bottom-left corner.
Troubleshooting
Common Issues
1. ModuleNotFoundError: No module named 'torch' during installation:
# Make sure PyTorch is installed BEFORE water_splatting
python3 -c "import torch; print(torch.__version__)"
# Should print: 2.1.2+cu118

2. CUDA compilation errors:
# Check GCC version (must be 11.x)
gcc --version

# Check CUDA paths
echo $LD_LIBRARY_PATH
ls -la $CONDA_PREFIX/lib/libcuda.so

3. Viewer won't start or crashes:
# Check GPU memory
nvidia-smi

# Verify checkpoint exists
ls -la /data/Lockheed1-Spring26/outputs/curasao_run1/water-splatting/2026-01-30_101250/nerfstudio_models/

4. Port forwarding not working:
# On local machine, verify tunnel is established
netstat -an | grep 7007

# Try different port if 7007 is blocked
ssh -L 8080:localhost:7007 -J apd6062@ssh.ist.psu.edu apd6062@nittanyaicompute.ist.psu.edu
# Then open http://localhost:8080


Quick Reference Commands
# Activate environment
conda activate water_splatting

# Start viewer
cd /data/Lockheed1-Spring26
ns-viewer --load-config outputs/curasao_run1/water-splatting/2026-01-30_101250/config.yml

# Quick GPU check
nvidia-smi

# Profile memory
python profile_memory.py

# Benchmark FPS
python benchmark_render.py


Additional Resources
WaterSplatting Paper: https://arxiv.org/pdf/2408.08206
WaterSplatting GitHub: https://github.com/water-splatting/water-splatting
Nerfstudio Docs: https://docs.nerf.studio/

