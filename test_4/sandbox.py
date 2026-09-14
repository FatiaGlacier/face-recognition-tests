import numpy as np
import os
import cv2
from pathlib import Path
from sixdrepnet import SixDRepNet

dir = "D:\\Projects\\python-opencv-test\\photos\\facial_rotation_test\\"

model = SixDRepNet(gpu_id=-1)

def load_images_from_folder(folder_path):
    """Loads all images from folder"""
    images = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    folder = Path(folder_path)
    if not folder.exists():
        print(f"[!] Folder doesn't exists: {folder_path}")
        return images

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in valid_extensions:
            images[file.stem] = str(file)

    return images

def get_coordinates_from_images(imgs):
    results = []
    print("Getting coordinates...")
    for name, path in imgs.items():
        print(f"Processing image: {name}")
        img = cv2.imread(path)
        pitch, yaw, roll = model.predict(img)

        results.append((name, pitch, yaw, roll))

    return results

def format_result(name, pitch, yaw, roll):
    """Formats the comparison result"""
    return (
        f"{name} : pitch={float(pitch):.4f} | yaw={float(yaw):.4f} | roll={float(roll):.4f}"
    )

def print_results(res):
    n = len(res)
    for i in range(n):
        result_str = format_result(res[i][0], res[i][1], res[i][2], res[i][3])
        print(result_str)

# ============================================================================
# MAIN PROGRAM
# ============================================================================

print("=" * 80)
print("IMAGES LOADING")
print("=" * 80)

# Loading photos
source_folder = dir + "source\\"
source_images = load_images_from_folder(source_folder)
print(f"\n📸 Images found (source): {len(source_images)}")
for name in source_images.keys():
    print(f"  - {name}")

print("=" * 80)
print("PROCESSING IMAGES...")
print("=" * 80)

results = get_coordinates_from_images(source_images)

print("=" * 80)
print("RESULTS")
print("=" * 80)

print_results(results)
