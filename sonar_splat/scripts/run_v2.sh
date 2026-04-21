#!/bin/bash
# General-purpose launcher for sonar_simple_trainer_v2.py
# Runs v2 (gamma NLL, r_tilde, ULA beam, elevation constraint) on any SonarSplat dataset.
#
# Usage:
#   bash scripts/run_v2.sh <dataset> <results_dir> [steps] [extra args...]
#
# <dataset> can be:
#   - A known name: monohansett_3D, concrete_piling_3D, infra_360_1,
#     basin_horizontal_infra_1, rock_semicircle1, basin_horizontal_empty1,
#     basin_horizontal_piling_up_down_4, basin_horizontal_piling_1, pole_qual1,
#     aoneus_sonar
#   - A full path to any dataset directory containing PKL files
#
# [steps]     : training steps, default 30000
# [extra args]: passed through verbatim (override any preset), e.g.:
#                 --z_loss_weight 0.0   (disable physics losses, L1 baseline)
#                 --z_loss_weight 1.0   (full v2 physics, default)
#                 --max_steps 5000      (quick smoke test)
#
# Examples:
#   bash scripts/run_v2.sh monohansett_3D "/media/priyanshu/2TB SSD/results" 40000
#   bash scripts/run_v2.sh infra_360_1   "/media/priyanshu/2TB SSD/results" 20000 --z_loss_weight 0.0
#   bash scripts/run_v2.sh /custom/path  "/media/priyanshu/2TB SSD/results"
#
# Quick smoke test (2K steps, any dataset):
#   bash scripts/run_v2.sh monohansett_3D /tmp/test 2000

set -e

DATASET_ROOT="/media/priyanshu/2TB SSD/sonarsplat_dataset"
CONDA_PYTHON="/home/priyanshu/miniconda3/envs/sonarsplat/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── argument parsing ──────────────────────────────────────────────────────────
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset_name_or_path> <results_dir> [steps] [extra args...]"
    echo ""
    echo "Known datasets: monohansett_3D, concrete_piling_3D, infra_360_1,"
    echo "  basin_horizontal_infra_1, rock_semicircle1, basin_horizontal_empty1,"
    echo "  basin_horizontal_piling_up_down_4, basin_horizontal_piling_1,"
    echo "  pole_qual1, aoneus_sonar"
    exit 1
fi

DATASET_ARG="$1"
RESULTS_BASE="$2"
STEPS="${3:-30000}"
shift 3 2>/dev/null || shift $#   # remaining args passed through

# ── dataset presets ───────────────────────────────────────────────────────────
# All real-world SonarSplat datasets use the Oculus M750d:
#   64 elements, 3 mm spacing, 1.1 MHz centre freq, 30 kHz bandwidth
#   → sigma_r = 1500 / (4 * 30000) = 1.25 cm
#
# far_plane / near_plane roughly match the max sonar range of each scene.
# range_clear_end: bins to suppress near-field clutter (0 = no clearing).

case "$DATASET_ARG" in
    monohansett_3D)
        DATA_DIR="$DATASET_ROOT/monohansett_3D"
        FAR=10; NEAR=-10; RANGE_CLEAR_END=0; ENERGY_WEIGHT=0.1
        ;;
    concrete_piling_3D)
        DATA_DIR="$DATASET_ROOT/concrete_piling_3D"
        FAR=10; NEAR=-10; RANGE_CLEAR_END=0; ENERGY_WEIGHT=0.1
        ;;
    infra_360_1)
        DATA_DIR="$DATASET_ROOT/infra_360_1"
        FAR=5; NEAR=-5; RANGE_CLEAR_END=0; ENERGY_WEIGHT=0.0
        ;;
    basin_horizontal_infra_1)
        DATA_DIR="$DATASET_ROOT/basin_horizontal_infra_1"
        FAR=5; NEAR=-5; RANGE_CLEAR_END=0; ENERGY_WEIGHT=0.0
        ;;
    rock_semicircle1)
        DATA_DIR="$DATASET_ROOT/rock_semicircle1"
        FAR=5; NEAR=-5; RANGE_CLEAR_END=0; ENERGY_WEIGHT=0.0
        ;;
    basin_horizontal_empty1)
        DATA_DIR="$DATASET_ROOT/basin_horizontal_empty1"
        FAR=5; NEAR=-5; RANGE_CLEAR_END=0; ENERGY_WEIGHT=0.0
        ;;
    basin_horizontal_piling_up_down_4)
        DATA_DIR="$DATASET_ROOT/basin_horizontal_piling_up_down_4"
        FAR=5; NEAR=-5; RANGE_CLEAR_END=0; ENERGY_WEIGHT=0.0
        ;;
    basin_horizontal_piling_1)
        DATA_DIR="$DATASET_ROOT/basin_horizontal_piling_1"
        FAR=5; NEAR=-5; RANGE_CLEAR_END=0; ENERGY_WEIGHT=0.0
        ;;
    pole_qual1)
        DATA_DIR="$DATASET_ROOT/pole_qual1"
        FAR=5; NEAR=-5; RANGE_CLEAR_END=0; ENERGY_WEIGHT=0.0
        ;;
    aoneus_sonar)
        DATA_DIR="/media/priyanshu/2TB SSD/aoneus_dataset/data/reduced_baseline_0.6x_sonar"
        FAR=5; NEAR=-5; RANGE_CLEAR_END=0; ENERGY_WEIGHT=0.0
        ;;
    /*)
        # Absolute path — use it directly with monohansett defaults
        DATA_DIR="$DATASET_ARG"
        FAR=10; NEAR=-10; RANGE_CLEAR_END=0; ENERGY_WEIGHT=0.0
        ;;
    *)
        echo "Unknown dataset '$DATASET_ARG'. Pass a full path or one of the known names."
        exit 1
        ;;
esac

# Derive a clean tag from the dataset name for the output directory
DATASET_TAG=$(basename "$DATA_DIR" | tr ' ' '_')
RESULTS_DIR="$RESULTS_BASE/${DATASET_TAG}_v2_$(date +%Y%m%d_%H%M%S)"

# ── environment ───────────────────────────────────────────────────────────────
export CPLUS_INCLUDE_PATH="/home/priyanshu/miniconda3/envs/sonarsplat/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH}"
export TORCH_CUDA_ARCH_LIST="8.6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Dataset  → $DATA_DIR"
echo "Results  → $RESULTS_DIR"
echo "Steps    → $STEPS"
[ $# -gt 0 ] && echo "Extra    → $*"

# ── launch ────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 $CONDA_PYTHON -u "$SCRIPT_DIR/examples/sonar_simple_trainer_v2.py" \
    "prune_only" \
    "--batch_size"                       "1" \
    "--camera_model"                     "ortho" \
    "--color_prior_asym"                 "0.08" \
    "--color_prior_weight"               "0.0" \
    "--data_dir"                         "$DATA_DIR" \
    "--result_dir"                       "$RESULTS_DIR" \
    "--data_factor"                      "1" \
    "--elevate_end_step"                 "1500" \
    "--elevate_loss_select"              "100" \
    "--elevate_num_samples"              "5" \
    "--elevate_sampling_duty_cycle"      "0.2" \
    "--elevate_start_step"              "0" \
    "--elevation_sampling_every"         "1000" \
    "--end_at_frame"                     "10000" \
    "--far_plane"                        "$FAR" \
    "--global_scale"                     "1.0" \
    "--img_threshold"                    "0.0" \
    "--init_extent"                      "1.0" \
    "--init_num_pts"                     "100000" \
    "--init_opa"                         "0.9" \
    "--init_scale"                       "0.01" \
    "--init_threshold"                   "0.4" \
    "--init_type"                        "predefined" \
    "--intermediate_azimuth_resolution"  "0.05" \
    "--lpips_net"                        "alex" \
    "--max_size_prior_weight"            "500.0" \
    "--max_steps"                        "$STEPS" \
    "--near_plane"                       "$NEAR" \
    "--normalize_world_space" \
    "--num_random_points"                "2000" \
    "--opacity_prior_weight"             "0.0" \
    "--opacity_reg"                      "0.0" \
    "--opacity_supervision_end_step"     "10000" \
    "--opacity_supervision_start_step"   "0" \
    "--opacity_supervision_thresh"       "0.1" \
    "--opacity_supervision_weight"       "0.0" \
    "--port"                             "8080" \
    "--pose_noise"                       "0.0" \
    "--pose_opt_lr"                      "1.0e-05" \
    "--pose_opt_reg"                     "1.0e-06" \
    "--random_bkgd" \
    "--randomize_elevation" \
    "--range_clear_end"                  "$RANGE_CLEAR_END" \
    "--range_clear_start"                "800" \
    "--render_eval" \
    "--render_traj_amplitude"            "0.25" \
    "--render_traj_freq"                 "10.0" \
    "--render_traj_interp_val"           "1" \
    "--render_traj_path"                 "unchanged" \
    "--sat_bg_prior_weight"              "0.0" \
    "--sat_region_prior_weight"          "0.0" \
    "--sat_sparsity_prior_weight"        "0.0" \
    "--sat_thresh"                       "0.03" \
    "--scale_reg"                        "0.0" \
    "--sh_degree"                        "3" \
    "--sh_degree_interval"               "1000" \
    "--skip_frames"                      "1" \
    "--ssim_lambda"                      "0.2" \
    "--start_from_frame"                 "0" \
    "--steps_scaler"                     "1.0" \
    "--strategy.grow_grad2d"             "0.0002" \
    "--strategy.grow_scale2d"            "0.05" \
    "--strategy.grow_scale3d"            "0.01" \
    "--strategy.key_for_gradient"        "means2d" \
    "--strategy.pause_refine_after_reset" "0" \
    "--strategy.prune_opa"               "0.005" \
    "--strategy.prune_scale2d"           "0.15" \
    "--strategy.prune_scale3d"           "0.1" \
    "--strategy.refine_every"            "1000" \
    "--strategy.refine_scale2d_stop_iter" "0" \
    "--strategy.refine_start_iter"       "0" \
    "--strategy.refine_stop_iter"        "15000" \
    "--strategy.reset_every"             "3000" \
    "--strategy.verbose" \
    "--streak_end_step"                  "3000" \
    "--streak_interval"                  "1000" \
    "--streak_interval_ratio"            "0.6" \
    "--streak_start_step"                "2000" \
    "--tb_every"                         "100" \
    "--tb_save_image" \
    "--test_every"                       "8" \
    "--disable_viewer" \
    "--train" \
    "--speed_of_sound"                   "1500.0" \
    "--bandwidth"                        "30000.0" \
    "--n_array_elements"                 "64" \
    "--element_spacing"                  "0.003" \
    "--center_frequency"                 "1100000.0" \
    "--reflectivity_lr"                  "0.01" \
    "--w_e"                              "1.0" \
    "--w_e_final"                        "0.1" \
    "--w_e_anneal_steps"                 "10000" \
    "--reflectivity_reg_weight"          "0.1" \
    "--lambda_reg"                       "0.01" \
    "--reflectivity_reg_every"           "100" \
    "--mask_threshold"                   "0.01" \
    "--energy_loss_weight"               "$ENERGY_WEIGHT" \
    "--reflectivity_warmup_steps"        "0" \
    "--reflectivity_floor_start"         "0.0" \
    "--reflectivity_floor_anneal_end_step" "0" \
    "--beam_anneal_end_step"             "0" \
    "--reflectivity_reg_start_step"      "2000" \
    "--reflectivity_reg_full_step"       "8000" \
    "--best_ckpt_min_energy_ratio"       "0.5" \
    "--best_ckpt_max_energy_ratio"       "1.5" \
    "$@"

echo "Done. Results at $RESULTS_DIR"
