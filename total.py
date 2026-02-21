import os

# change this to your dataset root
ROOT_DIR = "/Users/leolu/Desktop/frames"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

category_counts = {}
total_images = 0

for root, _, files in os.walk(ROOT_DIR):
    rel = os.path.relpath(root, ROOT_DIR)
    parts = rel.split(os.sep)

    # images directly in ROOT go into a special bucket
    if parts == ['.']:
        category = "__root__"
    else:
        category = parts[0]

    for f in files:
        if os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS:
            category_counts[category] = category_counts.get(category, 0) + 1
            total_images += 1

print("Image count per category:")
for k in sorted(category_counts):
    print(f"  {k}: {category_counts[k]}")

print(f"\nTotal images: {total_images}")
