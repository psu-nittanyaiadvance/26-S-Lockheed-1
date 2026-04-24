#!/bin/bash
# Run sonar_simple_trainer.py (upstream v1 baseline, L1+SSIM loss) on sunboat_dataset.
# Used to benchmark against sonar_simple_trainer_v2.py (Gamma NLL + r_tilde).
#
# Same dataset, same GS strategy, same number of steps as run_sunboat_v2.sh
# so the comparison is apples-to-apples.
#
# Usage:
#   cd sonar_splat
#   bash scripts/run_sunboat_v1.sh <results_dir>

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <results_dir>"
    exit 1
fi

DATA_DIR="/media/priyanshu/2TB SSD/sunboat_dataset/processed_session1"
RESULTS_DIR="$1/sunboat_v1_$(date +%Y%m%d_%H%M%S)"
CONDA_PYTHON="/home/priyanshu/miniconda3/envs/sonarsplat/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export CPLUS_INCLUDE_PATH="/home/priyanshu/miniconda3/envs/sonarsplat/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH}"
export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Results  → $RESULTS_DIR"
echo "Data dir → $DATA_DIR"

CUDA_VISIBLE_DEVICES=0 $CONDA_PYTHON -u "$SCRIPT_DIR/examples/sonar_simple_trainer.py" \
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
    "--render_eval" \
    "--skip_frames" "1" \
    "--ssim_lambda" "0.2" \
    "--sh_degree" "3" \
    "--sh_degree_interval" "1000" \
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
    "--train"

echo "Done. Results at $RESULTS_DIR"
