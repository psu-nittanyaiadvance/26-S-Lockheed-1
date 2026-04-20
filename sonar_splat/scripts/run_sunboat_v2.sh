#!/bin/bash
# Run sonar_simple_trainer_v2.py on the converted sunboat_dataset.
#
# Sonar geometry (empirically determined from 497×902 sector-scan PNGs):
#   - 130° horizontal azimuth, 20° elevation — same as monohansett
#   - Max range ≈ 26.8 m  (scale = 0.054 m/px, apex at row 496)
#   - Real-world dataset: surface boat in 9-15 m of water
#   - ~382 frames (session 1, every 10th of 3829)
#
# Sonar model: assumed Oculus M1200d
#   center_frequency = 1.2 MHz, element_spacing = λ/2 = 0.000625 m
#   n_array_elements = 256
#   bandwidth = 30 kHz (same as monohansett, sigma_r = 1.25 cm)
#
# Near-field clear: wake clutter at 0-5 m → clear first 48 range bins
#   range_clear_end = 48  (48/256 * 26.8 = 5.0 m)
#
# Pre-requisite: run the conversion script first:
#   cd "Download Datasets"
#   python convert_sunboat.py --subsample 10
#
# Usage:
#   cd gaussian-splatting-with-depth  (or sonar_splat)
#   cd sonar_splat
#   bash scripts/run_sunboat_v2.sh <results_dir>

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <results_dir>"
    exit 1
fi

DATA_DIR="/media/priyanshu/2TB SSD/sunboat_dataset/processed_session1"
RESULTS_DIR="$1/sunboat_v2_$(date +%Y%m%d_%H%M%S)"
CONDA_PYTHON="/home/priyanshu/miniconda3/envs/sonarsplat/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# nerfacc CUDA JIT build needs these headers from the conda CUDA toolkit
export CPLUS_INCLUDE_PATH="/home/priyanshu/miniconda3/envs/sonarsplat/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH}"
export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Results  → $RESULTS_DIR"
echo "Data dir → $DATA_DIR"

CUDA_VISIBLE_DEVICES=0 $CONDA_PYTHON -u "$SCRIPT_DIR/examples/sonar_simple_trainer_v2.py" \
    "prune_only" \
    "--batch_size" "1" \
    "--camera_model" "ortho" \
    "--data_dir" "$DATA_DIR" \
    "--result_dir" "$RESULTS_DIR" \
    "--data_factor" "1" \
    "--end_at_frame" "10000" \
    "--far_plane" "30" \
    "--near_plane" "-30" \
    "--global_scale" "1.0" \
    "--img_threshold" "0.02" \
    "--render_eval" \
    "--init_extent" "1.0" \
    "--init_num_pts" "100000" \
    "--init_opa" "0.9" \
    "--init_scale" "0.01" \
    "--init_threshold" "0.2" \
    "--init_type" "predefined" \
    "--max_steps" "30000" \
    "--normalize_world_space" \
    "--num_random_points" "2000" \
    "--range_clear_end" "48" \
    "--range_clear_start" "800" \
    "--randomize_elevation" \
    "--skip_frames" "1" \
    "--strategy.grow_grad2d" "0.0002" \
    "--strategy.grow_scale2d" "0.05" \
    "--strategy.grow_scale3d" "0.01" \
    "--strategy.key_for_gradient" "means2d" \
    "--strategy.pause_refine_after_reset" "0" \
    "--strategy.prune_opa" "0.005" \
    "--strategy.prune_scale2d" "0.15" \
    "--strategy.prune_scale3d" "0.1" \
    "--strategy.refine_every" "1000" \
    "--strategy.refine_scale2d_stop_iter" "0" \
    "--strategy.refine_start_iter" "0" \
    "--strategy.refine_stop_iter" "15000" \
    "--strategy.reset_every" "3000" \
    "--strategy.verbose" \
    "--test_every" "8" \
    "--disable_viewer" \
    "--train" \
    "--speed_of_sound" "1500.0" \
    "--bandwidth" "30000.0" \
    "--n_array_elements" "256" \
    "--element_spacing" "0.000625" \
    "--center_frequency" "1200000.0" \
    "--reflectivity_lr" "0.01" \
    "--w_e" "1.0" \
    "--w_e_final" "0.1" \
    "--w_e_anneal_steps" "10000" \
    "--reflectivity_reg_weight" "0.1" \
    "--lambda_reg" "0.01" \
    "--reflectivity_reg_every" "100"

echo "Done. Results at $RESULTS_DIR"
