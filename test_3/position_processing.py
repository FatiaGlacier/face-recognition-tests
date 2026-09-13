import numpy as np
import os
import cv2
from pathlib import Path
from sixdrepnet import SixDRepNet
import json

DIR = "D:\\Projects\\python-opencv-test\\photos\\facial_rotation_test\\small_video_test\\frames"

model = SixDRepNet(gpu_id=-1)

DATA = {
    "1":[], # pitch test
    "2":[], # yaw test
    "3":[]  # roll test
}

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
            images[file.name] = str(file)

    return images

def get_coordinates_from_images(imgs):
    results = []
    print("Getting coordinates...")
    for name, path in imgs.items():
        print(f"Processing image: {name}")
        img = cv2.imread(path)
        pitch, yaw, roll = model.predict(img)
        score = abs(pitch) + abs(yaw) + abs(roll)
        results.append((
            name, float(pitch), float(yaw), float(roll), float(score)
        ))

    return results

def format_result(name, pitch, yaw, roll, score):
    """Formats the comparison result"""
    return (
        f"{name} : pitch={float(pitch):.4f} | yaw={float(yaw):.4f} | roll={float(roll):.4f} | score={float(score):.4f}"
    )

def print_results(res):
    n = len(res)
    for i in range(n):
        result_str = format_result(res[i][0], res[i][1], res[i][2], res[i][3], res[i][4])
        print(result_str)

def sort_by(res, index):
    n = len(res)

    for i in range(n):
        min_index = i

        for j in range(i+1, n):
            if res[j][index] < res[min_index][index]:
                min_index = j

        temp = res[i]
        res[i] = res[min_index]
        res[min_index] = temp

    return res

def get_best_for_step_by_pitch(data, step=1):
    best_frames = []
    min_target = round(data[0][1])
    max_target = round(data[-1][1])

    last_seen_index = 0
    for target in range(min_target, max_target + 1):
        sub_arr = [] # img, yaw, score
        min_border = target - step/2
        max_border =  target + step/2
        #print(f"target: {target}, borders [{min_border};{max_border}]")
        for current in range(last_seen_index, len(data)):
            #print(f"current: {current}")
            if(min_border <= data[current][1] <= max_border):
                score = abs(data[current][2]) + abs(data[current][3]) # |yaw| + |roll|
                sub_arr.append((
                    data[current][0], # name
                    data[current][1], # pitch
                    data[current][2], # yaw
                    data[current][3], # roll
                    score # score = |pitch| + |roll|"
                ))
                #print(f"Added {current}")
                continue
            #print(f"Break on current: {current}")
            last_seen_index = current
            break

        sort_by(sub_arr, 4)
        #print(f"length: {len(best_frames)}")
        if(len(sub_arr)>0):
            best_frames.append(sub_arr[0])
        
    return best_frames

def get_best_for_step_by_yaw(data, step=1):
    best_frames = []
    min_target = round(data[0][2])
    max_target = round(data[-1][2])

    last_seen_index = 0
    for target in range(min_target, max_target + 1):
        sub_arr = [] # img, yaw, score
        min_border = target - step/2
        max_border =  target + step/2
        #print(f"target: {target}, borders [{min_border};{max_border}]")
        for current in range(last_seen_index, len(data)):
            #print(f"current: {current}")
            if(min_border <= data[current][2] <= max_border):
                score = abs(data[current][1]) + abs(data[current][3]) # |pitch| + |roll|
                sub_arr.append((
                    data[current][0], # name
                    data[current][1], # pitch
                    data[current][2], # yaw
                    data[current][3], # roll
                    score # score = |pitch| + |roll|"
                ))
                #print(f"Added {current}")
                continue
            #print(f"Break on current: {current}")
            last_seen_index = current
            break

        sort_by(sub_arr, 4)
        #print(f"length: {len(best_frames)}")
        if(len(sub_arr)>0):
            best_frames.append(sub_arr[0])
        
    return best_frames

def get_best_for_step_by_roll(data, step=1):
    best_frames = []
    min_target = round(data[0][3])
    max_target = round(data[-1][3])

    last_seen_index = 0
    for target in range(min_target, max_target + 1):
        sub_arr = [] # img, yaw, score
        min_border = target - step/2
        max_border =  target + step/2
        #print(f"target: {target}, borders [{min_border};{max_border}]")
        for current in range(last_seen_index, len(data)):
            #print(f"current: {current}")
            if(min_border <= data[current][3] <= max_border):
                score = abs(data[current][1]) + abs(data[current][2]) # |pitch| + |yaw|
                sub_arr.append((
                    data[current][0], # name
                    data[current][1], # pitch
                    data[current][2], # yaw
                    data[current][3], # roll
                    score # score = |pitch| + |roll|"
                ))
                #print(f"Added {current}")
                continue
            #print(f"Break on current: {current}")
            last_seen_index = current
            break

        sort_by(sub_arr, 4)
        #print(f"length: {len(best_frames)}")
        if(len(sub_arr)>0):
            best_frames.append(sub_arr[0])
        
    return best_frames

def get_best_for_step_by(data, index, step=1):
    if(index == 1):
        return get_best_for_step_by_pitch(data, step)
    if(index == 2):
        return get_best_for_step_by_yaw(data, step)
    if(index == 3):
        return get_best_for_step_by_roll(data, step)

def save_json(best_pitch, best_yaw, best_roll):
    for img, pitch, yaw, roll, score in best_pitch:
        DATA["1"].append({
            "img": img,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "score": score
        })

    for img, pitch, yaw, roll, score in best_yaw:
            DATA["2"].append({
                "img": img,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "score": score
            })

    for img, pitch, yaw, roll, score in best_roll:
            DATA["3"].append({
                "img": img,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "score": score
            })
    
    with open("results.json", "w") as f:
        json.dump(DATA, f, indent=4)



# ============================================================================
# MAIN PROGRAM
# ============================================================================

print("=" * 80)
print("IMAGES LOADING")
print("=" * 80)

# Loading photos
source_images = load_images_from_folder(DIR)
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

print("=" * 80)
print("Best by pitch")
print("=" * 80)
sort_by(results, 1)
best_pitch = get_best_for_step_by(results, 1)
print_results(best_pitch)

print("=" * 80)
print("Best by yaw")
print("=" * 80)
sort_by(results, 2)
best_yaw = get_best_for_step_by(results, 2)
print_results(best_yaw)

print("=" * 80)
print("Best by roll")
print("=" * 80)
sort_by(results, 3)
best_roll = get_best_for_step_by(results, 3)
print_results(best_roll)

save_json(best_pitch, best_yaw, best_roll)
