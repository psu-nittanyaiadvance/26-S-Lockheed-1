#!/usr/bin/env python3
"""
Convert monohansett_3D (or any SonarSplat PKL dataset) into the COLMAP tree
that train_v2.py expects, so Z-Splat v2 can run on sonar-only data.

Output layout:
    <output_dir>/
        images/          ← sonar intensity saved as 3-ch PNG (for Scene loader)
        depth/           ← per-image range maps as .npy  (az × range, metres)
        sparse/0/
            cameras.bin  ← one PINHOLE camera (intrinsics don't matter; cam_loss=0)
            images.bin   ← one pose per frame from PKL PoseSensor
            points3D.bin ← 3-D sonar point cloud (seeds for Gaussian init)

The raw PKL directory is passed directly as --sonar_data_dir to train_v2.py;
SonarDataCache matches COLMAP image stems → Data/{stem}.pkl automatically.

Usage:
    conda activate sonarsplat
    cd "Download Datasets"
    python convert_monohansett_zsplat.py \\
        --data_dir   "/media/priyanshu/2TB SSD/sonarsplat_dataset/monohansett_3D" \\
        --output_dir "/media/priyanshu/2TB SSD/sonarsplat_dataset/monohansett_3D_zsplat" \\
        [--skip_frames 1]   # 1 = every frame (default), 2 = every other, etc.
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
# COLMAP binary writers (same format as create_valid_z_splat_scene.py)
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
            f.write(struct.pack("I", 1))          # model_id=1 → PINHOLE
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
            f.write(struct.pack("Q", 0))          # no 2-D points


def write_points3d_bin(path, xyzs: np.ndarray, rgbs: np.ndarray):
    """xyzs: (N,3) float64, rgbs: (N,3) uint8."""
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
    """
    Convert high-intensity sonar pixels to 3-D world points.

    Sonar geometry (2-D imaging, elevation unknown → set to 0):
        range_m   = range_idx / n_range * max_range
        theta_rad = (az_idx / n_az - 0.5) * hfov_rad     (from boresight)
        x_sensor  = range_m * cos(theta)   (forward)
        y_sensor  = range_m * sin(theta)   (lateral)
        z_sensor  = 0                      (no elevation)
    """
    mask = img_az_range > threshold
    az_idx, range_idx = np.where(mask)
    if az_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # subsample to max_pts
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

    pts_sensor = np.stack([x, y, z, np.ones_like(x)], axis=1)  # (N,4)
    pts_world  = (c2w @ pts_sensor.T).T[:, :3]                 # (N,3)
    return pts_world


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(data_dir: str, output_dir: str, skip_frames: int,
            img_threshold: float, write_depth: bool = False):

    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)

    # ---- read sonar config --------------------------------------------------
    cfg_path = data_dir / "Config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)["agents"][0]["sensors"][-1]["configuration"]

    n_range   = int(cfg["RangeBins"])    # 402 for monohansett
    n_az      = int(cfg["AzimuthBins"])  # 260
    max_range = float(cfg["RangeMax"])   # 8.0 m
    hfov_deg  = float(cfg["Azimuth"])    # 130°

    range_resolution = max_range / n_range  # metres per bin

    print(f"Dataset : {data_dir.name}")
    print(f"Sonar   : {n_az} az × {n_range} range bins")
    print(f"Range   : 0 – {max_range:.1f} m  ({range_resolution*100:.2f} cm/bin)")
    print(f"H-FOV   : {hfov_deg}°")

    # ---- enumerate + sort PKL files numerically -----------------------------
    pkl_dir = data_dir / "Data"
    all_pkls = sorted(pkl_dir.glob("*.pkl"),
                      key=lambda p: int(p.stem) if p.stem.isdigit()
                                    else int(re.sub(r'\D', '', p.stem)))
    selected = all_pkls[::skip_frames]
    print(f"Frames  : {len(all_pkls)} total → {len(selected)} selected "
          f"(skip_frames={skip_frames})\n")

    # ---- prepare output dirs ------------------------------------------------
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    if write_depth:
        (output_dir / "depth").mkdir(parents=True, exist_ok=True)
    (output_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    print(f"write_depth={write_depth}  (False = render_sonar_image path, depth.npy not needed)")

    # ---- range depth template (constant per column) -------------------------
    range_values = np.arange(n_range, dtype=np.float32) * range_resolution

    # ---- process frames -----------------------------------------------------
    cam_list   = []
    image_list = []
    all_pts    = []   # for points3D.bin
    all_rgb    = []

    for frame_idx, pkl_path in enumerate(tqdm(selected, desc="Converting")):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        sonar_raw = data["ImagingSonar"].astype(np.float64)  # (range, az)
        c2w       = data["PoseSensor"].astype(np.float64)    # (4,4) c2w

        # transpose → (az, range) = (height, width) for image convention
        img = sonar_raw.T.astype(np.float32)                 # (n_az, n_range)
        img = np.clip(img, 0.0, 1.0)
        img[img < img_threshold] = 0.0
        img[:,  :10] = 0.0   # edge noise suppression
        img[:, -10:] = 0.0
        img[ :10, :] = 0.0
        img[-10:, :] = 0.0

        stem = pkl_path.stem   # e.g. "0", "1000"

        # -- save sonar image as 3-channel grayscale PNG (for Scene loader) --
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        img_rgb   = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
        Image.fromarray(img_rgb).save(output_dir / "images" / f"{stem}.png")

        # -- depth map: range_m per pixel, 0 where no echo -------------------
        # Only written when --write_depth is set; not needed for render_sonar_image() path.
        if write_depth:
            depth = np.tile(range_values, (n_az, 1))   # (n_az, n_range)
            depth[img < img_threshold] = 0.0
            np.save(output_dir / "depth" / f"{stem}.npy", depth)

        # -- COLMAP camera (one shared camera for all frames) -----------------
        if frame_idx == 0:
            # PINHOLE params: fx, fy, cx, cy
            # camera_loss_weight=0 → intrinsics don't affect sonar training;
            # these values give a ~90° diagonal FOV, reasonable for init.
            fx = float(n_range)
            fy = float(n_range)
            cx = float(n_range) / 2.0
            cy = float(n_az)    / 2.0
            cam_list.append({
                "id":     1,
                "width":  n_range,
                "height": n_az,
                "params": [fx, fy, cx, cy],
            })

        # -- COLMAP image pose (w2c → qvec + tvec) ----------------------------
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

        # -- accumulate 3-D sonar points (sample every 10th frame) -----------
        if frame_idx % 10 == 0:
            pts = sonar_pixels_to_world(img, c2w, n_az, n_range,
                                        max_range, hfov_deg,
                                        threshold=img_threshold, max_pts=200)
            if pts.shape[0] > 0:
                # intensity as point colour (white-ish)
                intensity = img[img > img_threshold].mean()
                grey      = int(np.clip(intensity * 255, 80, 220))
                rgb       = np.full((pts.shape[0], 3), grey, dtype=np.uint8)
                all_pts.append(pts)
                all_rgb.append(rgb)

    # ---- write COLMAP sparse/0/ ---------------------------------------------
    sparse = output_dir / "sparse" / "0"
    write_cameras_bin(sparse / "cameras.bin", cam_list)
    write_images_bin (sparse / "images.bin",  image_list)

    if all_pts:
        pts_all = np.concatenate(all_pts, axis=0)
        rgb_all = np.concatenate(all_rgb, axis=0)
        # cap at 100K points
        if pts_all.shape[0] > 100_000:
            idx     = np.random.choice(pts_all.shape[0], 100_000, replace=False)
            pts_all = pts_all[idx]
            rgb_all = rgb_all[idx]
        write_points3d_bin(sparse / "points3D.bin", pts_all, rgb_all)
        print(f"\nWrote {pts_all.shape[0]:,} sonar seed points")
    else:
        # empty fallback
        write_points3d_bin(sparse / "points3D.bin",
                           np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))

    # ---- summary ------------------------------------------------------------
    print(f"\nDone.")
    print(f"  Output  : {output_dir}")
    print(f"  Frames  : {len(image_list)}")
    print(f"\nNext steps:")
    print(f"  cd gaussian-splatting-with-depth")
    print(f"  bash scripts/run_monohansett_v2.sh \"/media/priyanshu/2TB SSD/results\" 30000")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert SonarSplat PKL dataset → Z-Splat COLMAP format")
    p.add_argument("--data_dir",
                   default="/media/priyanshu/2TB SSD/sonarsplat_dataset/monohansett_3D",
                   help="Source dataset directory (contains Data/ and Config.json)")
    p.add_argument("--output_dir",
                   default="/home/priyanshu/datasets/monohansett_3D_zsplat",
                   help="Output COLMAP tree directory (default: NVMe, not 2TB SSD which is full)")
    p.add_argument("--skip_frames", type=int, default=1,
                   help="Take every Nth frame (1=all, 2=every other, etc.)")
    p.add_argument("--img_threshold", type=float, default=0.02,
                   help="Intensity threshold below which pixels are treated as no-echo")
    p.add_argument("--write_depth", action="store_true",
                   help="Write depth.npy files (only needed for histogram fallback path; "
                        "not required when --sonar_data_dir is set in training)")
    args = p.parse_args()

    convert(args.data_dir, args.output_dir, args.skip_frames,
            args.img_threshold, write_depth=args.write_depth)
