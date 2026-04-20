#!/usr/bin/env python3
"""
Bayesian hyperparameter search for train_v2.py using Optuna (TPE sampler).

Each trial runs a full training job as a subprocess and extracts the best
test PSNR from the log.  Optuna fits a surrogate model over the objective
surface and picks the next trial intelligently — far more sample-efficient
than random or grid search for expensive experiments.

Usage:
    python scripts/hparam_search.py \
        --data_dir "/media/priyanshu/2TB SSD/aoneus_dataset/transformed_data" \
        --sonar_max_range 5.0 \
        --results_dir /path/to/results \
        --n_trials 15 \
        --iterations 20000

Tip: start with --iterations 20000 to keep each trial ~5 min, then
do a final 40K run with the best params.

Dataset-specific args to set:
    AONeuS:      --data_dir .../aoneus_dataset/transformed_data --sonar_max_range 5.0
    monohansett: --data_dir .../sonarsplat_dataset/monohansett_3D --sonar_max_range <range>
"""

import argparse
import os
import re
import subprocess
import sys
import time

import optuna

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONDA_PYTHON = "/home/priyanshu/miniconda3/envs/sonarsplat/bin/python"
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_v2.py")

# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------
# Adjust these bounds based on what you've already learned:
#   z_loss_weight: ablations showed 0.5 >> 0.1; try wider range
#   r_tilde_lr:    0.05 worked well; explore 0.01–0.1
#   reflectivity_reg_weight: currently fixed at 0.1; open it up
#   lambda_reg:    L1 reg on r_tilde, currently 0.01
SEARCH_SPACE = {
    "z_loss_weight":           ("float_log", 0.1,  2.0),
    "r_tilde_lr":              ("float_log", 0.005, 0.2),
    "reflectivity_reg_weight": ("float_log", 0.01, 1.0),
    "lambda_reg":              ("float_log", 0.001, 0.1),
}


def suggest(trial, name, spec):
    kind = spec[0]
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    elif kind == "float_log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    elif kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"Unknown kind: {kind}")


# ---------------------------------------------------------------------------
# Parse metrics from a training log
# ---------------------------------------------------------------------------
# Combined objective: RGB_PSNR + Z_WEIGHT * Z_loss
#
# Why add Z_loss directly (not negate it)?
#   gamma-NLL is "less negative = better aligned".  The stuck state is ~-4.065;
#   a well-aligned model reaches ~-2.3.  So maximising Z_loss (towards -2.3)
#   is exactly what we want.  Z_WEIGHT scales the sonar signal relative to PSNR.
#
# Rule of thumb for Z_WEIGHT:
#   |ΔPSNR_range| ≈ 2 dB across runs,  |ΔZ_loss_range| ≈ 1.8 nats.
#   Z_WEIGHT = 1.0 makes sonar alignment worth ~1 PSNR dB.  Start at 1.0;
#   increase if r_tilde collapses; decrease if RGB quality degrades.
Z_WEIGHT = 1.0


def parse_metrics(log_path: str):
    """Return (best_combined_score, best_psnr, best_z_loss) from a training log.

    Combined score at each checkpoint = RGB_PSNR + Z_WEIGHT * Z_loss.
    Returns the checkpoint with the highest combined score.
    Z_loss line must appear *before* the PSNR line for the same ITER
    (training_report prints Z-loss first, then eval lines).
    """
    if not os.path.exists(log_path):
        return float("-inf"), float("-inf"), float("inf")

    psnr_pat  = re.compile(r"\[ITER\s+(\d+)\] Evaluating test:.*PSNR\s+([\d.eE+\-]+)")
    zloss_pat = re.compile(r"\[ITER\s+(\d+)\] Z-loss:\s+([-\d.eE+]+)")

    z_by_iter   = {}   # iter → z_loss float
    best_combined = float("-inf")
    best_psnr   = float("-inf")
    best_z      = float("inf")

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            zm = zloss_pat.search(line)
            if zm:
                z_by_iter[int(zm.group(1))] = float(zm.group(2))
                continue
            pm = psnr_pat.search(line)
            if pm:
                it   = int(pm.group(1))
                psnr = float(pm.group(2))
                z    = z_by_iter.get(it, float("nan"))
                if z != z:          # nan → no Z line seen yet
                    combined = psnr
                else:
                    combined = psnr + Z_WEIGHT * z
                if combined > best_combined:
                    best_combined = combined
                    best_psnr = psnr
                    best_z    = z

    return best_combined, best_psnr, best_z


def parse_latest_combined(log_path: str) -> float:
    """Return the most recent combined score for Optuna intermediate reporting."""
    combined, _, _ = parse_metrics(log_path)
    return combined


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def make_objective(results_root: str, iterations: int, data_dir: str,
                   sonar_max_range: float, fixed_args: dict):
    os.makedirs(results_root, exist_ok=True)
    study_log = os.path.join(results_root, "study_results.tsv")
    # Write header once
    if not os.path.exists(study_log):
        with open(study_log, "w") as f:
            header_cols = ["trial", "combined", "psnr", "z_loss"] + list(SEARCH_SPACE.keys())
            f.write("\t".join(header_cols) + "\n")

    def objective(trial):
        params = {name: suggest(trial, name, spec)
                  for name, spec in SEARCH_SPACE.items()}

        trial_dir = os.path.join(results_root, f"trial_{trial.number:03d}")
        log_path = os.path.join(results_root, f"trial_{trial.number:03d}.log")

        # Build test_iterations: evaluate at ~5 checkpoints for early pruning
        step = iterations // 5
        test_iters = list(range(step, iterations + 1, step))

        cmd = [
            CONDA_PYTHON, "-u", TRAIN_SCRIPT,
            "-s", data_dir,
            "-m", trial_dir,
            "--eval", "-i", "images",
            "--depth_loss",
            "--iterations",    str(iterations),
            "--test_iterations", *[str(i) for i in test_iters],
            "--save_iterations", str(iterations),
            "--speed_of_sound", "1500.0",
            "--bandwidth",      "30000.0",
            "--n_array_elements", "64",
            "--element_spacing", "0.003",
            "--center_frequency", "1100000.0",
            "--sonar_max_range",  str(sonar_max_range),
            "--w_e",              "1.0",
            "--w_e_final",        "0.1",
            "--w_e_anneal_steps", "10000",
            "--reflectivity_reg_every", "100",
            # Tunable params
            "--z_loss_weight",           str(params["z_loss_weight"]),
            "--r_tilde_lr",              str(params["r_tilde_lr"]),
            "--reflectivity_reg_weight", str(params["reflectivity_reg_weight"]),
            "--lambda_reg",              str(params["lambda_reg"]),
        ]
        # Allow caller to inject extra fixed args (e.g. --depth_scale)
        for k, v in fixed_args.items():
            cmd += [f"--{k}", str(v)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: {params}")
        print(f"Log  → {log_path}")
        print(f"{'='*60}")

        t0 = time.time()
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd, stdout=logf, stderr=logf, env=env
            )

            # Poll for Optuna intermediate reporting + early pruning.
            # Report combined score (RGB_PSNR + Z_WEIGHT * Z_loss) so the
            # pruner kills trials that are bad on EITHER dimension.
            check_idx = 0
            while proc.poll() is None:
                time.sleep(30)
                current_score, _, _ = parse_metrics(log_path)
                if current_score > float("-inf"):
                    trial.report(current_score, step=check_idx)
                    check_idx += 1
                    if trial.should_prune():
                        proc.terminate()
                        proc.wait()
                        print(f"  [pruned] combined score {current_score:.3f} below median")
                        raise optuna.exceptions.TrialPruned()

            proc.wait()

        elapsed = time.time() - t0
        best_combined, best_psnr, best_z = parse_metrics(log_path)
        print(f"  combined={best_combined:.4f}  PSNR={best_psnr:.4f}  Z-loss={best_z:.4f}"
              f"  ({elapsed/60:.1f} min)")

        # Append to study log
        with open(study_log, "a") as f:
            row = ([str(trial.number), f"{best_combined:.4f}",
                    f"{best_psnr:.4f}", f"{best_z:.4f}"] +
                   [f"{params[k]:.6g}" for k in SEARCH_SPACE])
            f.write("\t".join(row) + "\n")

        return best_combined

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparam search for Z-Splat train_v2.py"
    )
    parser.add_argument("--data_dir", required=True,
                        help="Path to the COLMAP-format dataset directory "
                             "(same as -s in train_v2.py)")
    parser.add_argument("--sonar_max_range", type=float, required=True,
                        help="Sonar max range in metres for this dataset "
                             "(e.g. 5.0 for AONeuS, 8.0 for monohansett)")
    parser.add_argument("--results_dir", required=True,
                        help="Root directory to store all trial outputs")
    parser.add_argument("--n_trials", type=int, default=15,
                        help="Number of Optuna trials (default 15)")
    parser.add_argument("--iterations", type=int, default=20000,
                        help="Training iterations per trial (default 20K)")
    parser.add_argument("--study_name", type=str, default="zsplat_v2",
                        help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///study.db). "
                             "If None, auto-creates sqlite in results_dir.")
    parser.add_argument("--depth_scale", type=float, default=0.0,
                        help="Override depth_scale (0 = auto-calibrate from data)")
    args = parser.parse_args()

    fixed = {}
    if args.depth_scale != 0.0:
        fixed["depth_scale"] = args.depth_scale

    # Use MedianPruner: kill a trial if its intermediate PSNR is below
    # the median of completed trials at the same step.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    sampler = optuna.samplers.TPESampler(seed=42)

    storage = args.storage
    if storage is None:
        # Default: persist to SQLite in results dir so study is resumable
        os.makedirs(args.results_dir, exist_ok=True)
        storage = f"sqlite:///{args.results_dir}/optuna_study.db"
        print(f"Persisting study to {storage}")

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective(args.results_dir, args.iterations,
                               args.data_dir, args.sonar_max_range, fixed)
    study.optimize(objective, n_trials=args.n_trials)

    print("\n" + "="*60)
    print("SEARCH COMPLETE")
    print(f"  Objective: combined = RGB_PSNR + {Z_WEIGHT} * Z_loss")
    print(f"  Z-loss scale: ~-4.0 (stuck) → ~-2.3 (well aligned)")
    print("="*60)
    best = study.best_trial
    print(f"Best combined score : {best.value:.4f}")
    print(f"Best params:")
    for k, v in best.params.items():
        print(f"  {k:35s} = {v:.6g}")

    # Print top-5 with both PSNR and Z-loss from study log
    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True
    )
    # Read per-trial PSNR and Z-loss from the TSV log for display
    psnr_map = {}
    zloss_map = {}
    study_log = os.path.join(args.results_dir, "study_results.tsv")
    if os.path.exists(study_log):
        with open(study_log) as f:
            header = f.readline().split("\t")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    try:
                        psnr_map[int(parts[0])]  = float(parts[2])
                        zloss_map[int(parts[0])] = float(parts[3])
                    except ValueError:
                        pass

    print("\nTop 5 trials (combined / PSNR / Z-loss):")
    print(f"  {'#':>4}  {'combined':>10}  {'PSNR':>8}  {'Z-loss':>8}  " +
          "  ".join(k for k in SEARCH_SPACE))
    for t in trials_sorted[:5]:
        vals = "  ".join(f"{t.params.get(k, float('nan')):.4g}"
                         for k in SEARCH_SPACE)
        psnr  = psnr_map.get(t.number, float("nan"))
        zloss = zloss_map.get(t.number, float("nan"))
        print(f"  {t.number:>4}  {t.value:>10.4f}  {psnr:>8.4f}  {zloss:>8.4f}  {vals}")

    print(f"\nTo re-run with the best params at 40K iterations, "
          f"add to run_aoneus_v2.sh:")
    args_str = " ".join(
        f"--{k} {best.params[k]:.6g}" for k in SEARCH_SPACE
    )
    print(f"  {args_str} --iterations 40000")


if __name__ == "__main__":
    main()
