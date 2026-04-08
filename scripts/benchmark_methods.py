"""
Benchmark comparison: our implementation vs Z-Splat vs WaterSplatting vs SonarSplat.

Four methods:
  ours           -- our new sonar-primary implementation (this repo, zplat branch)
                    sonar loss primary; Gamma NLL; beam pattern; per-Gaussian r_n;
                    camera loss resolves elevation ambiguity only
  zplat          -- Z-Splat baseline (arXiv 2404.04687, Qu et al. 2024)
                    camera-primary, sonar used only as a depth regularizer; uses
                    mu_z (not Euclidean range), no beam pattern, no reflectivity
  watersplatting -- camera-only optical 3DGS (WaterSplatting)
  sonarsplat     -- SonarSplat (Sethuraman et al., IEEE RA-L 2025)
                    sonar-only novel view synthesis with polar rasterization

Usage:
    python scripts/benchmark_methods.py \\
        --ours_dir          /path/to/our_renders \\
        --zplat_dir         /path/to/zplat_renders \\
        --watersplat_dir    /path/to/watersplat_renders \\
        --sonarsplat_dir    /path/to/sonarsplat_renders \\
        --output_dir        ./benchmark_results

Or point at a single root directory with sub-folders named
  ours/, zplat/, watersplatting/, sonarsplat/

Each method dir contains one sub-folder per scene.  Supported layouts
(auto-detected per scene):

  Layout A  — Nerfstudio / ZSplat:
      <scene>/test/ours_<iter>/gt/
      <scene>/test/ours_<iter>/renders/

  Layout B  — SonarSplat:
      <scene>/sonar_renders/test/gt_sonar_images/
      <scene>/sonar_renders/test/sonar_images/

  Layout C  — simple:
      <scene>/gt/
      <scene>/renders/
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
import pandas as pd

# ── metric libraries ─────────────────────────────────────────────────────────
try:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    print("[WARN] torchmetrics not installed — PSNR/SSIM/LPIPS unavailable")

try:
    from lpipsPyTorch import lpips as lpips_pytorch
    HAS_LPIPS_PYTORCH = True
except ImportError:
    HAS_LPIPS_PYTORCH = False

# ── constants ─────────────────────────────────────────────────────────────────
BORDER_CROP = 10   # pixels trimmed on each edge (matches existing eval scripts)

# Display labels for each method key
METHOD_LABELS = {
    "ours":          "Ours (sonar-primary)",
    "zplat":         "Z-Splat (arXiv 2404.04687)",
    "watersplatting": "WaterSplatting (camera-only)",
    "sonarsplat":    "SonarSplat (Sethuraman RA-L 2025)",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# Directory layout helpers
# ═════════════════════════════════════════════════════════════════════════════

def _find_gt_pred_dirs(scene_dir: str):
    """
    Auto-detect gt/pred dirs inside a scene folder.
    Returns (gt_dir, pred_dir) or (None, None).
    """
    scene_dir = Path(scene_dir)

    # Layout A: test/ours_<N>/gt  +  test/ours_<N>/renders
    test_dir = scene_dir / "test"
    if test_dir.is_dir():
        ours_dirs = sorted(
            [d for d in test_dir.iterdir() if d.is_dir() and d.name.startswith("ours_")],
            key=lambda d: int(d.name.split("_")[-1]) if d.name.split("_")[-1].isdigit() else 0,
            reverse=True,
        )
        for ours in ours_dirs:
            gt_dir   = ours / "gt"
            pred_dir = ours / "renders"
            if gt_dir.is_dir() and pred_dir.is_dir():
                return str(gt_dir), str(pred_dir)

    # Layout B: sonar_renders/test/gt_sonar_images  +  sonar_images
    sonar_gt   = scene_dir / "sonar_renders" / "test" / "gt_sonar_images"
    sonar_pred = scene_dir / "sonar_renders" / "test" / "sonar_images"
    if sonar_gt.is_dir() and sonar_pred.is_dir():
        return str(sonar_gt), str(sonar_pred)

    # Layout C: simple gt/ + renders/
    if (scene_dir / "renders").is_dir() and (scene_dir / "gt").is_dir():
        return str(scene_dir / "gt"), str(scene_dir / "renders")

    return None, None


def collect_scenes(method_dir: str, method_key: str):
    """Walk method_dir and return {scene_name: (gt_dir, pred_dir)}."""
    scenes = {}
    if not method_dir or not os.path.isdir(method_dir):
        return scenes
    for entry in sorted(os.scandir(method_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        gt_dir, pred_dir = _find_gt_pred_dirs(entry.path)
        if gt_dir and pred_dir:
            scenes[entry.name] = (gt_dir, pred_dir)
        else:
            print(f"  [SKIP] {entry.name}: unrecognised layout in {entry.path}")
    return scenes


# ═════════════════════════════════════════════════════════════════════════════
# Image helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_chw_float(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW [0,1]


def _crop_border(img: np.ndarray, px: int = BORDER_CROP) -> np.ndarray:
    return img[:, px:-px, px:-px]


def _list_images(folder: str):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(
        [p for p in Path(folder).iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.stem.lstrip("0") or "0",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

def _build_metric_fns(skip_lpips: bool):
    fns = {}
    if HAS_TORCHMETRICS:
        fns["psnr"] = PeakSignalNoiseRatio(data_range=1.0).to(device)
        fns["ssim"] = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        if not HAS_LPIPS_PYTORCH and not skip_lpips:
            fns["lpips"] = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=True
            ).to(device)
    return fns


def evaluate_pair(gt_dir: str, pred_dir: str, fns: dict, skip_lpips: bool):
    gt_paths   = _list_images(gt_dir)
    pred_paths = _list_images(pred_dir)

    if not gt_paths or not pred_paths:
        print(f"    [WARN] empty folder — gt:{len(gt_paths)} pred:{len(pred_paths)}")
        return {}
    if len(gt_paths) != len(pred_paths):
        print(f"    [WARN] count mismatch gt:{len(gt_paths)} pred:{len(pred_paths)} — skip")
        return {}

    psnrs, ssims, lpips_vals = [], [], []

    for gt_p, pred_p in zip(gt_paths, pred_paths):
        try:
            gt_img   = _crop_border(_load_chw_float(str(gt_p)))
            pred_img = _crop_border(_load_chw_float(str(pred_p)))
        except Exception as e:
            print(f"    [WARN] {pred_p.name}: {e}")
            continue

        if gt_img.shape != pred_img.shape:
            pred_img = cv2.resize(
                pred_img.transpose(1, 2, 0),
                (gt_img.shape[2], gt_img.shape[1]),
                interpolation=cv2.INTER_LINEAR,
            ).transpose(2, 0, 1)

        gt_t   = torch.from_numpy(gt_img).unsqueeze(0).float().to(device)
        pred_t = torch.from_numpy(pred_img).unsqueeze(0).float().to(device)

        if "psnr" in fns:
            psnrs.append(fns["psnr"](pred_t, gt_t).item())
        if "ssim" in fns:
            ssims.append(fns["ssim"](pred_t, gt_t).item())
        if not skip_lpips:
            if HAS_LPIPS_PYTORCH:
                lpips_vals.append(lpips_pytorch(pred_t, gt_t, net_type="vgg").item())
            elif "lpips" in fns:
                lpips_vals.append(fns["lpips"](pred_t, gt_t).item())

    out = {}
    if psnrs:      out["PSNR"]  = float(np.mean(psnrs))
    if ssims:      out["SSIM"]  = float(np.mean(ssims))
    if lpips_vals: out["LPIPS"] = float(np.mean(lpips_vals))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Benchmark loop
# ═════════════════════════════════════════════════════════════════════════════

def run_benchmark(method_dirs: dict, output_dir: str, skip_lpips: bool):
    os.makedirs(output_dir, exist_ok=True)
    fns = _build_metric_fns(skip_lpips)

    all_rows = []
    summary  = defaultdict(lambda: defaultdict(list))  # method_key -> metric -> [values]

    for method_key, method_dir in method_dirs.items():
        label = METHOD_LABELS.get(method_key, method_key)
        if not method_dir:
            print(f"\n[SKIP] {label}: no directory provided")
            continue

        print(f"\n{'='*65}")
        print(f"  {label}")
        print(f"  {method_dir}")
        print(f"{'='*65}")

        scenes = collect_scenes(method_dir, method_key)
        if not scenes:
            print(f"  No evaluable scenes found.")
            continue

        for scene_name, (gt_dir, pred_dir) in tqdm.tqdm(scenes.items(), desc=method_key):
            print(f"\n  Scene : {scene_name}")
            print(f"    GT  : {gt_dir}")
            print(f"    Pred: {pred_dir}")

            metrics = evaluate_pair(gt_dir, pred_dir, fns, skip_lpips)
            if not metrics:
                continue

            all_rows.append({"method": label, "method_key": method_key,
                             "scene": scene_name, **metrics})
            for k, v in metrics.items():
                summary[method_key][k].append(v)

            print("    " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    if not all_rows:
        print("\n[ERROR] No results collected. Check your directory paths.")
        sys.exit(1)

    # ── per-scene CSV ─────────────────────────────────────────────────────────
    df_scene = pd.DataFrame(all_rows)
    scene_csv = os.path.join(output_dir, "per_scene_metrics.csv")
    df_scene.to_csv(scene_csv, index=False)
    print(f"\nPer-scene results  → {scene_csv}")

    # ── summary CSV ───────────────────────────────────────────────────────────
    summary_rows = []
    # preserve display order: ours first, then baselines
    for method_key in ["ours", "zplat", "watersplatting", "sonarsplat"]:
        if method_key not in summary:
            continue
        label = METHOD_LABELS.get(method_key, method_key)
        row = {"method": label, "method_key": method_key}
        for metric_name, values in summary[method_key].items():
            row[f"{metric_name}_mean"] = float(np.mean(values))
            row[f"{metric_name}_std"]  = float(np.std(values))
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "summary_metrics.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Summary results    → {summary_csv}")

    # ── console table ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*65}")
    print(df_summary.drop(columns=["method_key"]).to_string(index=False))
    print(f"{'='*65}")

    # ── delta table: ours vs each baseline ────────────────────────────────────
    if "ours" in summary:
        ours_row = next((r for r in summary_rows if r["method_key"] == "ours"), None)
        if ours_row:
            print("\nDELTA  (ours − baseline)  "
                  "positive = ours better for PSNR/SSIM, positive = ours better (lower) for LPIPS")
            print(f"  {'Baseline':<40}  {'PSNR':>8}  {'SSIM':>8}  {'LPIPS':>8}")
            print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}")
            for row in summary_rows:
                if row["method_key"] == "ours":
                    continue
                parts = []
                for m in ("PSNR", "SSIM", "LPIPS"):
                    key = f"{m}_mean"
                    if key in row and key in ours_row:
                        delta = ours_row[key] - row[key]
                        if m == "LPIPS":
                            delta = -delta   # lower LPIPS is better, flip for display
                        parts.append(f"{delta:+8.4f}")
                    else:
                        parts.append(f"{'N/A':>8}")
                print(f"  {row['method']:<40}  {'  '.join(parts)}")

    return df_summary


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark: Ours vs Z-Splat vs WaterSplatting vs SonarSplat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--root_dir", type=str, default=None,
        help=(
            "Root dir containing sub-folders: ours/, zplat/, "
            "watersplatting/, sonarsplat/. Overrides individual flags."
        ),
    )
    parser.add_argument("--ours_dir",        type=str, default=None,
                        help="Our sonar-primary implementation renders")
    parser.add_argument("--zplat_dir",       type=str, default=None,
                        help="Z-Splat renders (arXiv 2404.04687, camera-primary baseline)")
    parser.add_argument("--watersplat_dir",  type=str, default=None,
                        help="WaterSplatting renders (camera-only baseline)")
    parser.add_argument("--sonarsplat_dir",  type=str, default=None,
                        help="SonarSplat renders (Sethuraman RA-L 2025 baseline)")
    parser.add_argument("--output_dir",      type=str, default="./benchmark_results",
                        help="Directory for CSV outputs (default: ./benchmark_results)")
    parser.add_argument("--skip_lpips",      action="store_true",
                        help="Skip LPIPS (faster runs, no VGG network needed)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.root_dir:
        root = Path(args.root_dir)
        def _d(name):
            p = root / name
            return str(p) if p.is_dir() else None
        method_dirs = {
            "ours":          _d("ours"),
            "zplat":         _d("zplat"),
            "watersplatting": _d("watersplatting"),
            "sonarsplat":    _d("sonarsplat"),
        }
    else:
        method_dirs = {
            "ours":          args.ours_dir,
            "zplat":         args.zplat_dir,
            "watersplatting": args.watersplat_dir,
            "sonarsplat":    args.sonarsplat_dir,
        }

    active = {k: v for k, v in method_dirs.items() if v}
    if not active:
        print("[ERROR] No method directories provided.")
        print("  Use --root_dir or individual --ours_dir / --zplat_dir / "
              "--watersplat_dir / --sonarsplat_dir flags.")
        sys.exit(1)

    print("Methods to evaluate:")
    for k, v in method_dirs.items():
        label = METHOD_LABELS.get(k, k)
        status = v if v else "(not provided — will be skipped)"
        print(f"  {label:<42}  {status}")
    print(f"\nOutput: {args.output_dir}")
    print(f"Device: {device}\n")

    run_benchmark(method_dirs, args.output_dir, args.skip_lpips)


if __name__ == "__main__":
    main()
