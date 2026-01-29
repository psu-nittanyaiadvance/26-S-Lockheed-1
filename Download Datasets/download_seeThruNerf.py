import os
import zipfile
import subprocess


# Paths: data lives next to repo, NOT within

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))#Download Datasets directory
REPO_ROOT = os.path.dirname(SCRIPT_DIR) #26-S-Lockheed-1 repo root
REPO_PARENT = os.path.dirname(REPO_ROOT) #root of the repo, contains the watersplatting_data directory

DATA_ROOT = os.path.join(REPO_PARENT, "watersplatting_data") #watersplatting_data directory

# zip linked from the official SeaThru-NeRF repo
GDRIVE_FILE_ID = "1RzojBFvBWjUUhuJb95xJPSNP3nJwZWaT"
ZIP_NAME = "SeathruNeRF_dataset.zip"
ZIP_PATH = os.path.join(DATA_ROOT, ZIP_NAME)


DATASET_DIR = os.path.join(DATA_ROOT, "SeathruNeRF_dataset") #actual dataset directory


def download_zip_if_needed():
    os.makedirs(DATA_ROOT, exist_ok=True)

    if os.path.exists(ZIP_PATH):
        print(f"Zip file for SeathruNeRFalready exists: {ZIP_PATH}")
        return
    
    print("Downloading the zip, if this fails: pip install --user gdown")
    subprocess.run(
        ["gdown", GDRIVE_FILE_ID, "-O", ZIP_PATH],
        check=True
    )
    print(f"Downloaded zip file at: {ZIP_PATH}")


def extract_if_needed():
    if os.path.exists(DATASET_DIR):
        print(f"Dataset already extracted to: {DATASET_DIR}")
        return

    print("Extracting the SeaThru-NeRFzip file")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_ROOT)
    print(f"Extracted the zip file to: {DATASET_DIR}")


def is_scene_dir(path: str) -> bool:
    """
    checks if a directory within the DATASET_DIR is a valid scene.
    """
    if not os.path.isdir(path):
        return False

    # images dir can be images_wb or Images_wb
    has_images = (
        os.path.isdir(os.path.join(path, "images_wb")) or
        os.path.isdir(os.path.join(path, "Images_wb"))
    )

    # colmap sparse can be either sparse/0 OR colmap/sparse/0 depending on dataset variant
    has_sparse = (
        os.path.isdir(os.path.join(path, "sparse", "0")) or
        os.path.isdir(os.path.join(path, "colmap", "sparse", "0"))
    )

    return has_images and has_sparse


def split_scene(scene_dir: str, eval_interval: int = 8):
    """
    Splits the scene into train and test sets according to eval_interval.
    This doesn't actually affect the training process, just documents the split if needed.
    """
    
    #check if the scene uses images_wb or Images_wb
    if os.path.isdir(os.path.join(scene_dir, "images_wb")):
        images_dir = os.path.join(scene_dir, "images_wb")
    else:
        images_dir = os.path.join(scene_dir, "Images_wb")
        

    all_images = sorted(os.listdir(images_dir)) #get a list ofall the images in images_wb and sort them

    test_images = all_images[::eval_interval] #get a list of every 8th image for testing according to watersplatting README
    train_images = [] #list all other images for training

    for image in all_images:
        if image not in test_images:
            train_images += [image]


    with open(os.path.join(scene_dir, "train_list.txt"), "w") as f:
        f.write("\n".join(train_images))
    with open(os.path.join(scene_dir, "test_list.txt"), "w") as f:
        f.write("\n".join(test_images))

    print(f"  - {os.path.basename(scene_dir)}: total={len(all_images)} train={len(train_images)} test={len(test_images)}")


def main():
    download_zip_if_needed()
    extract_if_needed()

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Expected dataset folder not found: {DATASET_DIR}")

    print("Discovering scenes inside the dataset directory")
    scene_dirs = []
    
    for item in sorted(os.listdir(DATASET_DIR)):
        item_path = os.path.join(DATASET_DIR, item)
        if is_scene_dir(item_path):
            scene_dirs += [item_path]

    if not scene_dirs:
        raise RuntimeError(
            "No scenes found. Expected each scene to contain images_wb/ and sparse/0/. "
            f"Look inside: {DATASET_DIR}"
        )

    print(f"Found {len(scene_dirs)} scenes:")

    print("Writing split files (every 8th frame -> test for documentation")
    for d in scene_dirs:
        split_scene(d, eval_interval=8)

    print("\nDone.")
    print("Example training command for one scene:") #according to the watersplatting dataset
    print(f"  ns-train water-splatting colmap --colmap-path sparse/0 --data {os.path.join(DATASET_DIR, os.path.basename(scene_dirs[0]))} --images-path images_wb")


if __name__ == "__main__":
    main()