import os
import sys
import struct
import pickle
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False
    print("WARNING: pycolmap not installed. Run: pip install pycolmap")

# ─── Paths ────────────────────────────────────────────────────────────────────
AONEUS_ROOT  = Path("/data/Lockheed1-Spring26/gaussian-splatting-with-depth_data/data")
RGB_FOLDER   = AONEUS_ROOT / "reduced_baseline_0.6x_rgb"
SONAR_FOLDER = AONEUS_ROOT / "reduced_baseline_0.6x_sonar"
OUTPUT_ROOT  = Path("/data/Lockheed1-Spring26/gaussian-splatting-with-depth_data/transformed_data")

SONAR_MAX_RANGE_M = 5.0


def rotmat_to_qvec(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def parse_cameras_sphere(npz_path, image_dir):
    data = np.load(npz_path)
    print("cameras_sphere.npz keys (first 12):", list(data.keys())[:12], "...")

    image_files = sorted(Path(image_dir).glob("*.png")) + \
                  sorted(Path(image_dir).glob("*.jpg"))
    print(f"Found {len(image_files)} images")

    img0 = Image.open(image_files[0])
    W, H = img0.size

    world_mat_keys = sorted(
        [k for k in data.keys() if k.startswith("world_mat_") and "inv" not in k],
        key=lambda k: int(k.split("_")[-1])
    )
    print(f"Found {len(world_mat_keys)} camera poses")

    cam_list   = []
    image_list = []

    for i, wkey in enumerate(world_mat_keys):
        if i >= len(image_files):
            break

        ckey = f"camera_mat_{i}"
        if ckey in data:
            K = data[ckey][:3, :3]
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
        else:
            P  = data[wkey]
            if P.shape == (4, 4):
                P = P[:3, :]
            M  = P[:3, :3]
            Q, R_mat = np.linalg.qr(np.linalg.inv(M))
            K  = np.linalg.inv(R_mat)
            signs = np.sign(np.diag(K))
            K  = K * signs[:, None]
            fx = abs(float(K[0, 0]))
            fy = abs(float(K[1, 1]))
            cx = float(K[0, 2])
            cy = float(K[1, 2])

        P = data[wkey]
        if P.shape == (4, 4):
            P = P[:3, :]
        K_full = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        Rt = np.linalg.inv(K_full) @ P
        R  = Rt[:3, :3]
        t  = Rt[:3,  3]
        qvec = rotmat_to_qvec(R)

        cam_list.append({
            "id": i + 1, "model": "PINHOLE",
            "width": W, "height": H,
            "params": [fx, fy, cx, cy],
            "R": R, "t": t,
        })
        image_list.append({
            "id": i + 1, "qvec": qvec, "tvec": t,
            "camera_id": i + 1, "name": image_files[i].name,
        })

    return cam_list, image_list, image_files[:len(world_mat_keys)]


def run_colmap_triangulation(image_dir, sparse_dir, cam_list, image_list):
    if not PYCOLMAP_AVAILABLE:
        print("  pycolmap not available")
        return False

    db_path = OUTPUT_ROOT / "database.db"

    print("  1. Extracting features...")
    pycolmap.extract_features(str(db_path), str(image_dir))

    print("  2. Matching features...")
    pycolmap.match_exhaustive(str(db_path))

    print("  3. Writing known poses to binary format...")
    # Write known poses so triangulate_point_cloud can use them
    write_cameras_bin(sparse_dir / "cameras.bin", cam_list)
    write_images_bin(sparse_dir  / "images.bin",  image_list)
    write_points3d_bin(sparse_dir / "points3D.bin")  # empty, will be filled

    print("  4. Triangulating 3D points with known poses...")
    try:
        pycolmap.triangulate_point_cloud(
            database_path=str(db_path),
            image_path=str(image_dir),
            input_path=str(sparse_dir),
            output_path=str(sparse_dir),
        )
    except Exception as e:
        print(f"  Triangulation error: {e}")
        return False

    # Check how many points were created
    try:
        from scene.colmap_loader import read_points3D_binary
        xyz, _, _ = read_points3D_binary(str(sparse_dir / "points3D.bin"))
        n_pts = len(xyz)
    except Exception:
        n_pts = -1

    print(f"  Triangulated {n_pts} 3D points")
    return n_pts > 0


def write_cameras_bin(path, cameras):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(cameras)))
        for cam in cameras:
            f.write(struct.pack("I", cam["id"]))
            f.write(struct.pack("I", 1))
            f.write(struct.pack("Q", cam["width"]))
            f.write(struct.pack("Q", cam["height"]))
            for p in cam["params"]:
                f.write(struct.pack("d", p))


def write_images_bin(path, images):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(images)))
        for img in images:
            f.write(struct.pack("I", img["id"]))
            for q in img["qvec"]:
                f.write(struct.pack("d", q))
            for t in img["tvec"]:
                f.write(struct.pack("d", t))
            f.write(struct.pack("I", img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("Q", 0))


def write_points3d_bin(path, points=None):
    with open(path, "wb") as f:
        if not points:
            f.write(struct.pack("Q", 0))
            return
        f.write(struct.pack("Q", len(points)))
        for i, (xyz, rgb) in enumerate(points):
            f.write(struct.pack("Q", i + 1))
            for v in xyz:
                f.write(struct.pack("d", float(v)))
            for c in rgb:
                f.write(struct.pack("B", int(np.clip(c, 0, 255))))
            f.write(struct.pack("d", 0.0))
            f.write(struct.pack("Q", 0))


def write_colmap_fallback(out_sparse, cam_list, image_list):
    write_cameras_bin(out_sparse / "cameras.bin", cam_list)
    write_images_bin(out_sparse  / "images.bin",  image_list)
    rng = np.random.default_rng(42)
    n_pts = 10_000
    xyz = rng.uniform(-0.5, 0.5, (n_pts, 3))
    rgb = rng.integers(100, 200, (n_pts, 3))
    points = list(zip(xyz, rgb))
    write_points3d_bin(out_sparse / "points3D.bin", points)
    print(f"  Wrote {n_pts} random seed points (fallback)")


def extract_sonar_depth(pkl_path, max_range=SONAR_MAX_RANGE_M):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    if not isinstance(d, dict) or "ImagingSonar" not in d:
        print(f"  Warning: no ImagingSonar key in {pkl_path.name}")
        return None

    sonar_img = d["ImagingSonar"]
    n_range, n_az = sonar_img.shape
    range_values = np.linspace(0, max_range, n_range)
    depth_map = np.zeros_like(sonar_img, dtype=np.float32)
    for r in range(n_range):
        depth_map[r, :] = range_values[r]
    depth_map[sonar_img == 0] = 0.0
    return depth_map


def convert():
    print("=" * 60)
    print("AONeuS -> Z-Splat format converter")
    print("=" * 60)

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    out_images = OUTPUT_ROOT / "images"
    out_sparse = OUTPUT_ROOT / "sparse" / "0"
    out_depth  = OUTPUT_ROOT / "depth"

    for d in [out_images, out_sparse, out_depth]:
        d.mkdir(parents=True, exist_ok=True)

    print("\nParsing camera poses...")
    cam_list, image_list, image_files = parse_cameras_sphere(
        RGB_FOLDER / "cameras_sphere.npz",
        RGB_FOLDER / "image"
    )
    c = cam_list[0]
    print(f"Camera 0: {c['width']}x{c['height']}, "
          f"fx={c['params'][0]:.1f}, fy={c['params'][1]:.1f}, "
          f"cx={c['params'][2]:.1f}, cy={c['params'][3]:.1f}")

    print(f"\nCopying {len(image_files)} RGB images...")
    for f in image_files:
        shutil.copy2(f, out_images / f.name)
    print("Done")

    print("\nRunning COLMAP triangulation with known poses...")
    colmap_success = run_colmap_triangulation(
        out_images, out_sparse, cam_list, image_list
    )

    if not colmap_success:
        print("Falling back to manual binary writing + random seed points...")
        write_colmap_fallback(out_sparse, cam_list, image_list)

    sonar_data_dir = SONAR_FOLDER / "Data"
    pkl_files = sorted(sonar_data_dir.glob("*.pkl"))
    print(f"\nConverting {len(pkl_files)} sonar pkl files...")

    success = 0
    for i, pkl_path in enumerate(pkl_files):
        if i >= len(image_files):
            break
        depth = extract_sonar_depth(pkl_path)
        if depth is not None:
            np.save(out_depth / f"{image_files[i].stem}.npy", depth)
            success += 1

    print(f"Saved {success}/{min(len(pkl_files), len(image_files))} depth maps")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"  images/   : {len(list(out_images.glob('*')))} files")
    print(f"  sparse/0/ : cameras.bin, images.bin, points3D.bin")
    print(f"  depth/    : {len(list(out_depth.glob('*.npy')))} .npy files")
    print("\nTo run Z-Splat training:")
    print(f"  cd /data/Lockheed1-Spring26/26-S-Lockheed-1/gaussian-splatting-with-depth")
    print(f"  conda activate gaussian_splatting_with_depth")
    print(f"  python train.py \\")
    print(f"    -s {OUTPUT_ROOT} \\")
    print(f"    -m /data/Lockheed1-Spring26/gaussian-splatting-with-depth_outputs/aoneus \\")
    print(f"    --depth_loss \\")
    print(f"    --eval \\")
    print(f"    -i images")
    print("=" * 60)


if __name__ == "__main__":
    convert()