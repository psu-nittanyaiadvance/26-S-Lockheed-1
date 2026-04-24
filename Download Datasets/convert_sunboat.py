#!/usr/bin/env python3
"""
Convert sunboat_dataset → SonarSplat pkl format for sonar_simple_trainer_v2.py.

Dataset structure (after unzip):
  Sunboat_03-09-2023/
  ├── 2023-09-03-07-58-37/
  │   ├── camera/   {00001..03829}.png  (1355×1692 RGB)
  │   ├── sonar/    {00001..03829}.png  (497×902 RGB sector-scan)
  │   ├── navigation/navigation.csv
  │   └── samples.json
  └── 2023-09-03-11-37-57/
      └── ...

Sonar geometry (empirically determined):
  - Apex at (row=496, col=450) of the 497×902 image (sensor at bottom-center)
  - Full azimuth: 130° (±65°), matching SonarSplat's Config format
  - Scale: 0.054 m/pixel → max range ≈ 26.8 m
  - Elevation: 20° (assumed, matching monohansett SonarSplat scene)

Navigation:
  - GPS lat/lon/alt → ENU coordinates (1st frame as origin)
  - Yaw/pitch/roll (radians, ZYX convention) → SO3 rotation

Output: SonarSplat pkl format
  Data/{000000..}.pkl: {'PoseSensor': 4×4, 'CameraPose': 4×4, 'ImagingSonar': [R,A]}
  Config.json: SonarSplat sonar config
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation

# ──────────────────────── Sonar geometry constants ────────────────────────────
APEX_ROW       = 496          # pixel row of sonar apex in 497×902 image
APEX_COL       = 450          # pixel column of sonar apex
IMG_HEIGHT     = 497
IMG_WIDTH      = 902
AZIMUTH_DEG    = 130.0        # full horizontal FOV
SCALE_M_PER_PX = 0.054        # meters per pixel from apex
MAX_RANGE_M    = SCALE_M_PER_PX * APEX_ROW   # ≈26.8 m
ELEVATION_DEG  = 20.0         # vertical FOV (assumed)

# Output polar grid (matches monohansett dims)
RANGE_BINS     = 256
AZIMUTH_BINS   = 260

# ──────────────────────────────────────────────────────────────────────────────

def cartesian_to_polar(png_path: str) -> np.ndarray:
    """
    Convert a Cartesian sector-scan PNG to polar [RangeBins × AzimuthBins] in [0,1].

    The sonar PNG has its apex at (APEX_ROW, APEX_COL).  Each range bin r maps to
    a circle of radius  r_px = (r+0.5) * APEX_ROW / RANGE_BINS  pixels from the
    apex.  Each azimuth bin a maps to an angle
        θ = (a/(BINS-1) - 0.5) * AZIMUTH_DEG   (degrees, CCW from bore-sight)
    The image pixel at (apex_row - r_px*cos(θ), apex_col + r_px*sin(θ)) is
    sampled by nearest-neighbour and its luminance used as intensity.
    """
    img = np.array(Image.open(png_path), dtype=np.float32)  # (H, W, 3)

    # Luminance as sonar intensity proxy (colormap is approximately plasma)
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    lum /= 255.0   # → [0, 1]

    # Pre-compute sampling coordinates for all (range, azimuth) pairs (vectorised)
    r_idx  = np.arange(RANGE_BINS, dtype=np.float32)
    a_idx  = np.arange(AZIMUTH_BINS, dtype=np.float32)

    r_px   = (r_idx + 0.5) * APEX_ROW / RANGE_BINS            # [R]
    az_deg = (a_idx / (AZIMUTH_BINS - 1) - 0.5) * AZIMUTH_DEG # [A]  −65° … +65°
    az_rad = np.radians(az_deg)                                 # [A]

    # Broadcast: rows[r, a] = apex_row - r_px[r]*cos(az[a])
    rows = (APEX_ROW - np.outer(r_px, np.cos(az_rad))).astype(np.int32)  # [R, A]
    cols = (APEX_COL + np.outer(r_px, np.sin(az_rad))).astype(np.int32)  # [R, A]

    # Clamp to valid image range
    rows = np.clip(rows, 0, IMG_HEIGHT - 1)
    cols = np.clip(cols, 0, IMG_WIDTH  - 1)

    return lum[rows, cols].astype(np.float64)   # [R, A]


def gps_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """Convert GPS (degrees, metres) to local ENU (metres) using flat-Earth approx."""
    R_earth = 6_371_000.0
    east  = R_earth * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    north = R_earth * np.radians(lat - ref_lat)
    up    = alt - ref_alt
    return np.array([east, north, up], dtype=np.float64)


def build_se3(enu, yaw_rad, pitch_rad, roll_rad):
    """
    Build a 4×4 SE3 pose (world ← sensor) from ENU translation + ZYX Euler angles.

    Convention: yaw rotates around Z (up), pitch around Y, roll around X.
    Scipy ZYX = intrinsic rotations applied in Z, Y, X order.
    """
    R = Rotation.from_euler('ZYX', [yaw_rad, pitch_rad, roll_rad]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = enu
    return T


def convert_session(session_dir: str, output_dir: str,
                    subsample: int = 10, max_frames: int = None):
    """Convert one session directory → SonarSplat pkl dataset."""

    nav_csv    = os.path.join(session_dir, 'navigation', 'navigation.csv')
    sonar_dir  = os.path.join(session_dir, 'sonar')
    data_dir   = os.path.join(output_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)

    nav = pd.read_csv(nav_csv)
    n_nav = len(nav)

    # Reference GPS position (frame 0)
    ref_lat = nav.iloc[0]['latitude']
    ref_lon = nav.iloc[0]['longitude']
    ref_alt = nav.iloc[0]['altitude']

    # Frame indices to convert: every `subsample`-th frame, nav-limited
    indices = list(range(0, n_nav, subsample))
    if max_frames is not None:
        indices = indices[:max_frames]

    print(f'Session: {os.path.basename(session_dir)}')
    print(f'  Navigation rows : {n_nav}')
    print(f'  Sonar frames    : {len(os.listdir(sonar_dir))}')
    print(f'  Subsampling     : every {subsample} → {len(indices)} output frames')
    print(f'  Output dir      : {output_dir}')

    for out_idx, nav_idx in enumerate(indices):
        row = nav.iloc[nav_idx]

        # GPS → ENU → SE3
        enu  = gps_to_enu(row['latitude'], row['longitude'], row['altitude'],
                           ref_lat, ref_lon, ref_alt)
        pose = build_se3(enu, float(row['yaw']), float(row['pitch']), float(row['roll']))

        # Sonar: nav row N → sonar file N+1 (files are 1-based)
        sonar_file = os.path.join(sonar_dir, f'{nav_idx + 1:05d}.png')
        if not os.path.isfile(sonar_file):
            print(f'  [WARN] sonar file missing: {sonar_file}')
            continue

        polar = cartesian_to_polar(sonar_file)   # [RANGE_BINS, AZIMUTH_BINS]

        pkl_data = {
            'PoseSensor' : pose,
            'CameraPose' : pose,    # sonar and camera co-located (approximation)
            'ImagingSonar': polar,
        }

        out_path = os.path.join(data_dir, f'{out_idx:06d}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(pkl_data, f)

        if out_idx % 20 == 0:
            depth = row['depth']
            enu_dist = np.linalg.norm(enu[:2])   # 2D displacement from origin
            print(f'  [{out_idx:4d}/{len(indices)}] nav={nav_idx}, '
                  f'depth={depth:.1f}m, ENU dist={enu_dist:.1f}m')

    # ── Config.json ──────────────────────────────────────────────────────────
    config = {
        'agents': [{
            'sensors': [{
                'sensor_type'  : 'ImagingSonar',
                'socket'       : 'SonarSocket',
                'Hz'           : 10,
                'rotation'     : [0, 0, 0],
                'configuration': {
                    'RangeBins'     : RANGE_BINS,
                    'AzimuthBins'   : AZIMUTH_BINS,
                    'RangeMin'      : 0,
                    'RangeMax'      : MAX_RANGE_M,
                    'InitOctreeRange': -1,
                    'Elevation'     : ELEVATION_DEG,
                    'Azimuth'       : AZIMUTH_DEG,
                    'AzimuthStreaks': -1,
                    'ScaleNoise'    : True,
                    'AddSigma'      : 0.0,
                    'MultSigma'     : 0.0,
                    'RangeSigma'    : 0.0,
                    'MultiPath'     : False,
                    'ViewOctree'    : -1,
                }
            }]
        }]
    }

    config_path = os.path.join(output_dir, 'Config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f'  Done → {len(indices)} pkl files + Config.json')
    return len(indices)


def main():
    ap = argparse.ArgumentParser(description='Convert sunboat dataset to SonarSplat pkl format')
    ap.add_argument('--dataset_dir', type=str,
                    default='/media/priyanshu/2TB SSD/sunboat_dataset/Sunboat_03-09-2023',
                    help='Root of the unzipped Sunboat_03-09-2023 directory')
    ap.add_argument('--output_dir', type=str,
                    default='/media/priyanshu/2TB SSD/sunboat_dataset/processed_session1',
                    help='Where to write pkl files and Config.json')
    ap.add_argument('--session', type=str,
                    default='2023-09-03-07-58-37',
                    help='Which session sub-directory to convert')
    ap.add_argument('--subsample', type=int, default=10,
                    help='Keep every Nth navigation frame (default: 10 → ~380 frames)')
    ap.add_argument('--max_frames', type=int, default=None,
                    help='Cap output at this many frames (for quick testing)')
    args = ap.parse_args()

    session_dir = os.path.join(args.dataset_dir, args.session)
    if not os.path.isdir(session_dir):
        print(f'ERROR: session dir not found: {session_dir}')
        sys.exit(1)

    n = convert_session(session_dir, args.output_dir,
                        subsample=args.subsample, max_frames=args.max_frames)
    print(f'\nConversion complete: {n} frames written to {args.output_dir}')


if __name__ == '__main__':
    main()
