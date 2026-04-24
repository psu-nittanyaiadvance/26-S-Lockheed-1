#!/usr/bin/env python3
"""
Universal converter: any SonarSplat PKL dataset → COLMAP tree for train_v2.py.

Works with all datasets that have the standard layout:
    <data_dir>/
        Config.json       ← sonar params (RangeBins, AzimuthBins, RangeMax, Azimuth)
        Data/*.pkl        ← per-frame dicts: ImagingSonar (range×az) + PoseSensor (4×4 c2w)

Output (COLMAP format for Z-Splat v2):
    <output_dir>/
        images/           ← sonar intensity as 3-ch PNG (for Scene loader)
        sparse/0/
            cameras.bin   ← one PINHOLE camera (intrinsics unused; camera_loss_weight=0)
            images.bin    ← one pose per frame
            points3D.bin  ← 3-D sonar seed points for Gaussian init

Usage:
    conda activate sonarsplat
    cd "Download Datasets"

    # Any named dataset (auto output dir under ~/datasets/):
    python convert_to_zsplat.py --data_dir "/media/priyanshu/2TB SSD/sonarsplat_dataset/monohansett_3D"
    python convert_to_zsplat.py --data_dir "/media/priyanshu/2TB SSD/sonarsplat_dataset/concrete_piling_3D"
    python convert_to_zsplat.py --data_dir "/media/priyanshu/2TB SSD/sunboat_dataset/processed_session1"

    # Custom output dir:
    python convert_to_zsplat.py \\
        --data_dir   "/media/priyanshu/2TB SSD/sonarsplat_dataset/monohansett_3D" \\
        --output_dir "/home/priyanshu/datasets/monohansett_zsplat"

    # Subsample every 2nd frame:
    python convert_to_zsplat.py --data_dir "/path/to/dataset" --skip_frames 2

After conversion, launch training:
    cd ../gaussian-splatting-with-depth
    bash scripts/run_sonar_v2.sh \\
        ~/datasets/<name>_zsplat \\
        "/media/priyanshu/2TB SSD/sonarsplat_dataset/<name>" \\
        "/media/priyanshu/2TB SSD/results" \\
        30000
"""

import argparse
import json
import math
import os
import pickle
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# ---------------------------------------------------------------------------
# COLMAP binary writers
# ---------------------------------------------------------------------------

def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → COLMAP quaternion (qw, qx, qy, qz)."""
    q = Rotation.from_matrix(R).as_quat()   # scipy: (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def write_cameras_bin(path, cameras):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(cameras)))
        for cam in cameras:
            f.write(struct.pack("I", cam["id"]))
            f.write(struct.pack("I", 1))          # PINHOLE
            f.write(struct.pack("Q", cam["width"]))
            f.write(struct.pack("Q", cam["height"]))
            for p in cam["params"]:
                f.write(struct.pack("d", float(p)))


def write_images_bin(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(images)))
        for img in images:
            f.write(struct.pack("I", img["id"]))
            for q in img["qvec"]:
                f.write(struct.pack("d", float(q)))
            for t in img["tvec"]:
                f.write(struct.pack("d", float(t)))
            f.write(struct.pack("I", img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("Q", 0))          # no 2-D keypoints


def write_points3d_bin(path, xyzs: np.ndarray, rgbs: np.ndarray):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(xyzs)))
        for i, (xyz, rgb) in enumerate(zip(xyzs, rgbs)):
            f.write(struct.pack("Q", i + 1))
            for v in xyz:
                f.write(struct.pack("d", float(v)))
            for c in rgb:
                f.write(struct.pack("B", int(np.clip(c, 0, 255))))
            f.write(struct.pack("d", 0.0))        # reprojection error
            f.write(struct.pack("Q", 0))          # no track entries


# ---------------------------------------------------------------------------
# Sonar → 3-D conversion helpers
# ---------------------------------------------------------------------------

def sonar_pixels_to_world(img_az_range: np.ndarray, c2w: np.ndarray,
                           n_az: int, n_range: int,
                           max_range: float, hfov_deg: float,
                           threshold: float = 0.05,
                           max_pts: int = 300) -> np.ndarray:
    """Convert high-intensity sonar pixels to 3-D world points (elevation=0)."""
    mask = img_az_range > threshold
    az_idx, range_idx = np.where(mask)
    if az_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if az_idx.size > max_pts:
        sel = np.random.choice(az_idx.size, max_pts, replace=False)
        az_idx    = az_idx[sel]
        range_idx = range_idx[sel]

    hfov_rad  = math.radians(hfov_deg)
    range_m   = range_idx.astype(np.float64) / n_range * max_range
    theta_rad = (az_idx.astype(np.float64) / n_az - 0.5) * hfov_rad

    x = range_m * np.cos(theta_rad)
    y = range_m * np.sin(theta_rad)
    z = np.zeros_like(x)

    pts_sensor = np.stack([x, y, z, np.ones_like(x)], axis=1)
    pts_world  = (c2w @ pts_sensor.T).T[:, :3]
    return pts_world


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_config(data_dir: Path) -> dict:
    """
    Read sonar params from Config.json.
    Supports both:
      - Nested format: {"agents": [{"sensors": [..., {"configuration": {...}}]}]}
      - Flat format:   {"RangeBins": N, ...}
    """
    cfg_path = data_dir / "Config.json"
    with open(cfg_path) as f:
        raw = json.load(f)

    if "agents" in raw:
        cfg = raw["agents"][0]["sensors"][-1]["configuration"]
    elif "RangeBins" in raw:
        cfg = raw
    else:
        raise ValueError(f"Unrecognised Config.json layout in {cfg_path}")

    return {
        "n_range":   int(cfg["RangeBins"]),
        "n_az":      int(cfg["AzimuthBins"]),
        "max_range": float(cfg["RangeMax"]),
        "hfov_deg":  float(cfg["Azimuth"]),
    }


def convert(data_dir: str, output_dir: str, skip_frames: int,
            img_threshold: float, max_seed_pts_per_frame: int):

    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)

    cfg = read_config(data_dir)
    n_range   = cfg["n_range"]
    n_az      = cfg["n_az"]
    max_range = cfg["max_range"]
    hfov_deg  = cfg["hfov_deg"]

    print(f"Dataset : {data_dir.name}")
    print(f"Sonar   : {n_az} az × {n_range} range bins")
    print(f"Range   : 0 – {max_range:.1f} m  ({max_range/n_range*100:.2f} cm/bin)")
    print(f"H-FOV   : {hfov_deg}°")

    # enumerate + sort PKL files numerically
    pkl_dir  = data_dir / "Data"
    all_pkls = sorted(pkl_dir.glob("*.pkl"),
                      key=lambda p: int(p.stem) if p.stem.isdigit()
                                    else int(re.sub(r'\D', '', p.stem) or "0"))
    selected = all_pkls[::skip_frames]
    print(f"Frames  : {len(all_pkls)} total → {len(selected)} selected "
          f"(skip_frames={skip_frames})\n")

    # output dirs
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)

    cam_list   = []
    image_list = []
    all_pts    = []
    all_rgb    = []

    for frame_idx, pkl_path in enumerate(tqdm(selected, desc="Converting")):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        sonar_raw = data["ImagingSonar"].astype(np.float64)   # (range, az)
        c2w       = data["PoseSensor"].astype(np.float64)     # (4,4)

        # transpose → (az, range) = (height, width)
        img = sonar_raw.T.astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        img[img < img_threshold] = 0.0
        # suppress edge noise
        img[:,  :10] = 0.0
        img[:, -10:] = 0.0
        img[ :10, :] = 0.0
        img[-10:, :] = 0.0

        stem = pkl_path.stem

        # save as 3-channel grayscale PNG (Scene loader needs RGB shape)
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        img_rgb   = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
        Image.fromarray(img_rgb).save(output_dir / "images" / f"{stem}.png")

        # one shared PINHOLE camera (intrinsics unused; camera_loss_weight=0)
        if frame_idx == 0:
            cam_list.append({
                "id":     1,
                "width":  n_range,
                "height": n_az,
                "params": [float(n_range), float(n_range),
                           float(n_range) / 2.0, float(n_az) / 2.0],
            })

        # COLMAP pose (w2c)
        w2c  = np.linalg.inv(c2w)
        R    = w2c[:3, :3]
        tvec = w2c[:3,  3]
        qvec = rotmat_to_qvec(R)

        image_list.append({
            "id":        frame_idx + 1,
            "qvec":      qvec,
            "tvec":      tvec,
            "camera_id": 1,
            "name":      f"{stem}.png",
        })

        # accumulate 3-D seed points every 10th frame
        if frame_idx % 10 == 0:
            pts = sonar_pixels_to_world(img, c2w, n_az, n_range,
                                        max_range, hfov_deg,
                                        threshold=img_threshold,
                                        max_pts=max_seed_pts_per_frame)
            if pts.shape[0] > 0:
                intensity = img[img > img_threshold].mean()
                grey      = int(np.clip(intensity * 255, 80, 220))
                rgb       = np.full((pts.shape[0], 3), grey, dtype=np.uint8)
                all_pts.append(pts)
                all_rgb.append(rgb)

    # write COLMAP sparse/0/
    sparse = output_dir / "sparse" / "0"
    write_cameras_bin(sparse / "cameras.bin", cam_list)
    write_images_bin (sparse / "images.bin",  image_list)

    if all_pts:
        pts_all = np.concatenate(all_pts, axis=0)
        rgb_all = np.concatenate(all_rgb, axis=0)
        if pts_all.shape[0] > 100_000:
            idx     = np.random.choice(pts_all.shape[0], 100_000, replace=False)
            pts_all = pts_all[idx]
            rgb_all = rgb_all[idx]
        write_points3d_bin(sparse / "points3D.bin", pts_all, rgb_all)
        print(f"\nWrote {pts_all.shape[0]:,} sonar seed points")
    else:
        write_points3d_bin(sparse / "points3D.bin",
                           np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))

    print(f"\nDone.")
    print(f"  Output  : {output_dir}")
    print(f"  Frames  : {len(image_list)}")
    print(f"\nNext step:")
    print(f"  cd ../gaussian-splatting-with-depth")
    print(f"  bash scripts/run_sonar_v2.sh \\")
    print(f"      {output_dir} \\")
    print(f"      {data_dir} \\")
    print(f"      \"/media/priyanshu/2TB SSD/results\" \\")
    print(f"      30000")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert any SonarSplat PKL dataset → Z-Splat COLMAP format")
    p.add_argument("--data_dir", required=True,
                   help="Source dataset directory (contains Data/*.pkl and Config.json)")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: ~/datasets/<dataset_name>_zsplat)")
    p.add_argument("--skip_frames", type=int, default=1,
                   help="Take every Nth frame (1=all, 2=every other, etc.)")
    p.add_argument("--img_threshold", type=float, default=0.02,
                   help="Intensity threshold: pixels below this are treated as no-echo")
    p.add_argument("--max_seed_pts", type=int, default=200,
                   help="Max sonar seed points per sampled frame (for points3D.bin)")
    args = p.parse_args()

    if args.output_dir is None:
        name = Path(args.data_dir).name
        args.output_dir = str(Path.home() / "datasets" / f"{name}_zsplat")

    convert(args.data_dir, args.output_dir, args.skip_frames,
            args.img_threshold, args.max_seed_pts)
