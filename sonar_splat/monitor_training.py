#!/usr/bin/env python3
"""
SonarSplat Training Monitor
============================
Lightweight GPU + training monitor. Run alongside your training job.

Usage:
    # Terminal 1: start training
    CUDA_VISIBLE_DEVICES=0 bash scripts/run_3D_monohansett.sh ...

    # Terminal 2: start monitor
    python monitor_training.py --results_dir results/singlegpu_baseline --poll_interval 5

    # Or for multi-GPU:
    python monitor_training.py --results_dir results/multigpu_official --gpus 0,1 --poll_interval 5

It writes a JSON-lines log to {results_dir}/monitor_log.jsonl that the dashboard can read.
Also prints a live summary to the terminal.
"""

import argparse
import json
import os
import subprocess
import time
import glob
import re
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path


def get_gpu_stats(gpu_ids):
    """Query nvidia-smi for GPU stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        stats = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                idx = int(parts[0])
                if idx in gpu_ids:
                    stats.append({
                        "gpu_id": idx,
                        "util_pct": float(parts[1]),
                        "mem_used_mb": float(parts[2]),
                        "mem_total_mb": float(parts[3]),
                        "temp_c": float(parts[4]),
                        "power_w": float(parts[5]) if parts[5] != "[N/A]" else 0,
                        "clock_mhz": float(parts[6]) if parts[6] != "[N/A]" else 0,
                    })
        return stats
    except Exception as e:
        return []


def get_latest_val(results_dir):
    """Find the most recent val_step*.json and parse it."""
    pattern = os.path.join(results_dir, "val_step*.json")
    files = glob.glob(pattern)
    if not files:
        return None, None
    latest = max(files, key=lambda f: int(re.search(r"step(\d+)", f).group(1)))
    step = int(re.search(r"step(\d+)", latest).group(1))
    try:
        with open(latest) as f:
            data = json.load(f)
        return step, data
    except:
        return step, None


def get_training_step_from_log(results_dir):
    """Try to infer current training step from the most recent files."""
    # Check for tensorboard events (modification time indicates progress)
    events = glob.glob(os.path.join(results_dir, "**", "events.out*"), recursive=True)
    ply_files = glob.glob(os.path.join(results_dir, "*.ply"))
    val_files = glob.glob(os.path.join(results_dir, "val_step*.json"))

    latest_val_step = 0
    if val_files:
        steps = [int(re.search(r"step(\d+)", f).group(1)) for f in val_files if re.search(r"step(\d+)", f)]
        latest_val_step = max(steps) if steps else 0

    return latest_val_step


def count_val_checkpoints(results_dir):
    """Count how many val checkpoints exist."""
    return len(glob.glob(os.path.join(results_dir, "val_step*.json")))


def format_duration(seconds):
    """Format seconds into human readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def format_mem(mb):
    """Format MB into GB string."""
    return f"{mb/1024:.1f}GB"


class TrainingMonitor:
    def __init__(self, results_dir, gpu_ids, poll_interval, max_steps):
        self.results_dir = results_dir
        self.gpu_ids = gpu_ids
        self.poll_interval = poll_interval
        self.max_steps = max_steps
        self.start_time = time.time()
        self.log_path = os.path.join(results_dir, "monitor_log.jsonl")
        self.peak_mem = {gid: 0 for gid in gpu_ids}
        self.peak_util = {gid: 0 for gid in gpu_ids}
        self.peak_temp = {gid: 0 for gid in gpu_ids}
        self.total_energy_wh = {gid: 0.0 for gid in gpu_ids}
        self.last_poll_time = time.time()
        self.last_val_step = 0
        self.step_times = []  # (step, timestamp) pairs for step rate estimation

        os.makedirs(results_dir, exist_ok=True)

    def poll(self):
        now = time.time()
        dt = now - self.last_poll_time
        self.last_poll_time = now

        gpu_stats = get_gpu_stats(self.gpu_ids)
        val_step, val_data = get_latest_val(self.results_dir)
        num_checkpoints = count_val_checkpoints(self.results_dir)

        # Track step progression
        if val_step and val_step != self.last_val_step:
            self.step_times.append((val_step, now))
            self.last_val_step = val_step

        # Update peaks and energy
        for gs in gpu_stats:
            gid = gs["gpu_id"]
            self.peak_mem[gid] = max(self.peak_mem[gid], gs["mem_used_mb"])
            self.peak_util[gid] = max(self.peak_util[gid], gs["util_pct"])
            self.peak_temp[gid] = max(self.peak_temp[gid], gs["temp_c"])
            self.total_energy_wh[gid] += gs["power_w"] * (dt / 3600)

        # Estimate step rate and ETA
        step_rate = None
        eta_str = "N/A"
        if len(self.step_times) >= 2:
            s0, t0 = self.step_times[0]
            s1, t1 = self.step_times[-1]
            if t1 > t0 and s1 > s0:
                step_rate = (s1 - s0) / (t1 - t0)
                if self.max_steps and val_step:
                    remaining = self.max_steps - val_step
                    if remaining > 0 and step_rate > 0:
                        eta_str = format_duration(remaining / step_rate)

        # Build log entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": now - self.start_time,
            "gpus": gpu_stats,
            "val_step": val_step,
            "val_metrics": val_data,
            "num_checkpoints": num_checkpoints,
            "step_rate": step_rate,
            "eta": eta_str,
        }

        # Write to log
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def print_status(self, entry):
        elapsed = format_duration(entry["elapsed_s"])
        step = entry["val_step"] or "?"
        rate = f"{entry['step_rate']:.1f} steps/s" if entry["step_rate"] else "estimating..."

        # Clear screen and print header
        print("\033[2J\033[H", end="")
        print("=" * 72)
        print(f"  SONARSPLAT TRAINING MONITOR")
        print(f"  Results: {self.results_dir}")
        print(f"  Elapsed: {elapsed}  |  ETA: {entry['eta']}")
        print(f"  Last eval step: {step} / {self.max_steps}  |  Rate: {rate}")
        print("=" * 72)

        # GPU table
        print(f"\n  {'GPU':>4} {'Util':>6} {'Mem':>12} {'Peak Mem':>10} {'Temp':>6} {'Power':>7} {'Energy':>8}")
        print(f"  {'─'*4} {'─'*6} {'─'*12} {'─'*10} {'─'*6} {'─'*7} {'─'*8}")
        for gs in entry["gpus"]:
            gid = gs["gpu_id"]
            mem_str = f"{format_mem(gs['mem_used_mb'])}/{format_mem(gs['mem_total_mb'])}"
            print(
                f"  {gid:>4} {gs['util_pct']:>5.0f}% {mem_str:>12} "
                f"{format_mem(self.peak_mem[gid]):>10} {gs['temp_c']:>5.0f}C "
                f"{gs['power_w']:>6.0f}W {self.total_energy_wh[gid]:>6.1f}Wh"
            )

        # Metrics
        if entry["val_metrics"]:
            m = entry["val_metrics"]
            print(f"\n  Latest Validation (step {step}):")
            print(f"    PSNR:  {m.get('psnr', 0):.2f} dB")
            print(f"    SSIM:  {m.get('ssim', 0):.4f}")
            print(f"    LPIPS: {m.get('lpips', 0):.4f}")
            print(f"    #GS:   {m.get('num_GS', 0):,}")
        else:
            print(f"\n  No validation results yet...")

        print(f"\n  Log: {self.log_path}")
        print(f"  Checkpoints found: {entry['num_checkpoints']}")
        print(f"\n  Press Ctrl+C to stop monitoring")


def main():
    parser = argparse.ArgumentParser(description="SonarSplat Training Monitor")
    parser.add_argument("--results_dir", required=True, help="Path to results directory")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs to monitor (default: 0)")
    parser.add_argument("--poll_interval", type=int, default=5, help="Seconds between polls (default: 5)")
    parser.add_argument("--max_steps", type=int, default=40000, help="Total training steps (default: 40000)")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    monitor = TrainingMonitor(args.results_dir, gpu_ids, args.poll_interval, args.max_steps)

    def handle_signal(sig, frame):
        print("\n\nMonitor stopped. Final stats:")
        print(f"  Total time: {format_duration(time.time() - monitor.start_time)}")
        for gid in gpu_ids:
            print(f"  GPU {gid} peak mem: {format_mem(monitor.peak_mem[gid])}, total energy: {monitor.total_energy_wh[gid]:.1f} Wh")
        print(f"  Log saved: {monitor.log_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)

    print(f"Monitoring {args.results_dir} on GPU(s) {gpu_ids}...")
    print(f"Polling every {args.poll_interval}s. Press Ctrl+C to stop.\n")

    while True:
        entry = monitor.poll()
        monitor.print_status(entry)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
