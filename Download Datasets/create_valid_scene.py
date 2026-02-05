from pathlib import Path
import shutil
import pycolmap
from download_seaThruNerf import split_scene
import numpy as np

DOWNLOAD_DATASETS_PATH = Path(__file__).resolve().parent
REPO_ROOT_FOLDER = Path(DOWNLOAD_DATASETS_PATH.parent)
REPO_PARENT_FOLDER = Path(REPO_ROOT_FOLDER.parent)
WATERSPLATTING_DATA_FOLDER = Path(REPO_PARENT_FOLDER.joinpath("watersplatting_data"))


def convert_images_to_scene(IMAGE_FOLDER_PATH, dataset_name, scene_name):
    WATERSPLATTING_DATA_FOLDER.mkdir(exist_ok=True)
    DATASET_FOLDER = Path(WATERSPLATTING_DATA_FOLDER.joinpath(dataset_name))
    DATASET_FOLDER.mkdir(exist_ok=True)
    SCENE_FOLDER = Path(DATASET_FOLDER.joinpath(scene_name))
    SCENE_FOLDER.mkdir(exist_ok=True)

    SCENE_IMAGES = Path(SCENE_FOLDER.joinpath("images_wb"))
    SCENE_SPARSE = Path(SCENE_FOLDER.joinpath("sparse"))

    SCENE_IMAGES.mkdir(exist_ok=True)

    database_path = SCENE_FOLDER / "database.db"

    file_paths = []
    
    for item in Path(IMAGE_FOLDER_PATH).iterdir(): #save abs image file path strings to list
        if item.is_file():
            file_path = item.resolve()
            file_paths += [file_path]

            shutil.copy2(item, SCENE_IMAGES)

    try:
        # 1. Extract features
        print("  1. Extracting features...")
        pycolmap.extract_features(database_path, SCENE_IMAGES)
        
        # 2. Match features
        print("  2. Matching features...")
        pycolmap.match_exhaustive(database_path)
        
        # 3. Sparse reconstruction (this is all you need!)
        print("  3. Sparse reconstruction...")
        maps = pycolmap.incremental_mapping(database_path, SCENE_IMAGES, SCENE_SPARSE)
        
        if not maps or len(maps) == 0:
            print("  WARNING: Sparse reconstruction failed!")
            return False
        
        # 4. colmap to llff
        print("  4. Converting to LLFF format...")
        colmap_to_llff(SCENE_FOLDER)
        
        print(f"  ✓ Sparse reconstruction complete!")
        
    except Exception as e:
        print(f"  ✗ Error during reconstruction: {e}\n")
        return False
        
    split_scene(str(SCENE_FOLDER))
    return True



def colmap_to_llff(scene_folder):
    """Convert COLMAP output to LLFF poses_bounds.npy format"""
    sparse_dir = scene_folder / "sparse" / "0"
    
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    
    poses_bounds = []
    
    for image_id, image in reconstruction.images.items():
        camera = reconstruction.cameras[image.camera_id]
        
        # FIXED: Call cam_from_world() as a method
        transform = image.cam_from_world()  # ← Added parentheses
        R = transform.rotation.matrix()
        t = transform.translation
        
        # Convert to LLFF format (camera to world)
        pose = np.concatenate([R.T, t.reshape(3, 1)], axis=1)
        
        # Add h, w, f
        h, w = camera.height, camera.width
        f = camera.focal_length_x
        hwf = np.array([[h], [w], [f]])
        pose = np.concatenate([pose, hwf], axis=1)
        
        pose_flat = pose.flatten()
        
        # Compute bounds
        pts3d = reconstruction.points3D
        if len(pts3d) > 0:
            points = np.array([p.xyz for p in pts3d.values()])
            depths = (R @ points.T).T[:, 2]
            near = np.percentile(depths, 1)
            far = np.percentile(depths, 99)
        else:
            near, far = 0.1, 100.0
        
        poses_bounds.append(np.concatenate([pose_flat, [near, far]]))
    
    poses_bounds = np.array(poses_bounds)
    np.save(scene_folder / "poses_bounds.npy", poses_bounds)
    print(f"  ✓ Saved poses_bounds.npy with shape {poses_bounds.shape}")

        
    

    





    

    