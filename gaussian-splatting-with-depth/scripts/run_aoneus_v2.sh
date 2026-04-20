#!/bin/bash
# Run train_v2.py (Z-Splat + Priyanshu's sonar physics math) on AONeuS RGB+sonar data.
#
# What's new vs train.py:
#   - GaussianModelV2 with per-Gaussian r_tilde reflectivity (sigmoid)
#   - ULA beam pattern weights via torch.sinc (no NaN in backward)
#   - Z loss: gamma NLL on a DIFFERENTIABLE depth histogram
#       * Python scatter_add replaces the missing CUDA z_density output
#       * gradient now flows: ZL → z_density → eff_opacity → r_tilde
#   - Elevation constraint loss (annealed)
#   - Reflectivity spatial regularizer (kNN, 4096 subsample)
#
# Baseline to beat: test PSNR=36.94  (train.py, L1-only, 30K steps)
# Previous best v2: test PSNR=37.77  (but r_tilde was stuck at 0.5, Z loss no-op)
#
# Usage:
#   bash scripts/run_aoneus_v2.sh <results_dir>
#
# Tuning notes:
#   --depth_scale is derived automatically: sonar_max_range / cameras_extent.
#     If the Z loss drives r_tilde in the wrong direction, try --z_loss_weight 0
#     to disable it and confirm RGB baseline is intact first.
#   --z_loss_weight 0.1 is conservative; increase to 0.5 once r_tilde is moving.

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <results_dir>"
    exit 1
fi

DATA_DIR="/media/priyanshu/2TB SSD/aoneus_dataset/transformed_data"
SONAR_DIR="/media/priyanshu/2TB SSD/aoneus_dataset/data/reduced_baseline_0.6x_sonar"
RESULTS_DIR="$1/aoneus_v2_$(date +%Y%m%d_%H%M%S)"

CONDA_PYTHON="/home/priyanshu/miniconda3/envs/sonarsplat/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Results → $RESULTS_DIR"
echo "Data    → $DATA_DIR"

CUDA_VISIBLE_DEVICES=0 $CONDA_PYTHON -u "$SCRIPT_DIR/train_v2.py" \
    -s "$DATA_DIR"                       \
    -m "$RESULTS_DIR"                    \
    --eval                               \
    -i images                            \
    --depth_loss                         \
    --iterations 30000                   \
    --test_iterations 5000 8000 11000 14000 17000 20000 23000 26000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --speed_of_sound 1500.0              \
    --bandwidth 30000.0                  \
    --n_array_elements 64                \
    --element_spacing 0.003              \
    --center_frequency 1100000.0         \
    --r_tilde_lr 0.01                    \
    --w_e 1.0                            \
    --w_e_final 0.1                      \
    --w_e_anneal_steps 10000             \
    --reflectivity_reg_weight 0.1        \
    --lambda_reg 0.01                    \
    --reflectivity_reg_every 100         \
    --sonar_max_range 5.0                \
    --z_loss_weight 0.12                 \
    --camera_loss_weight 1.0             \
    --sonar_data_dir "$SONAR_DIR"        \
    --use_rl_controller                  \
    --rl_target_ratio 1.0               \
    --rl_adapt_every 200

echo "Done. Results at $RESULTS_DIR"
