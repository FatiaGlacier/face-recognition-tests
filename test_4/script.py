from deepface import DeepFace
import numpy as np
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt

BASE_DIR = "D:\\Projects\\python-opencv-test\\photos\\facial_rotation_test\\big_video_test\\"

ME_BASE = BASE_DIR + "me\\"

ME_PITCH_PATH = ME_BASE + "pitch\\frames\\"
ME_YAW_PATH = ME_BASE + "yaw\\frames\\"
ME_ROLL_PATH = ME_BASE + "roll\\frames\\"

AXIS_MAP = {"1": "pitch", "2": "yaw", "3": "roll"}

# ============================================================================
# FUNCTIONS FOR WORK WITH EMBEDDINGS
# ============================================================================

def normalize_embedding(embedding):
    """Normalizes a vector to unit length"""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def get_embeddings(path, model_name):
    """Receives normalized embedding from image"""
    try:
        embedding = DeepFace.represent(
            img_path=path,
            model_name=model_name,
            enforce_detection=True
        )
        emb = np.array(embedding[0]["embedding"])
        return normalize_embedding(emb)
        #return emb
    except Exception as e:
        #print(f"[!] Exception {path} ({model_name}): {e}")
        print(f"[!] Exception {Path(path).name} ({model_name})")

        # try:
        #     Path(path).unlink()
        #     print(f"Deleted: {path}")
        # except Exception as delete_error:
        #     print(f"Could not delete {path}: {delete_error}")

        return None

def add_all_embeddings_to_records(records, model_name):
    """Takes embeddings from all images in dictionary"""
    for record in records:
        emb = get_embeddings(record["image"], model_name=model_name)
        if emb is not None:
            record["embedding"] = emb

def add_embeddings_to_target(target, model_name):
    target_emb = get_embeddings(target["image"], model_name=model_name)
    if target_emb is not None:
        target["embedding"] = target_emb

# ============================================================================
# SIMILARITY
# ============================================================================

def calculate_distance(emb1, emb2):
    raw = float(np.dot(emb1, emb2))
    distance = 1 - raw
    return distance, raw

# ============================================================================
# AUXILIARY FUNCTIONS
# ============================================================================

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

def format_result(src, trg, distance, raw, category):
    """Formats the comparison result"""
    return (
        f"{src} vs {trg} : dist={float(distance):.4f} | raw={float(raw):.4f} | ({category})"
    )

def load_data_from_json(file):
    with open(file) as f:
        data = json.load(f)

    # pitch_data = data[1]
    # yaw_data = data[2]
    # roll_data = data[3]
    # target_data = data['target'][0]

    return data #pitch_data, yaw_data, roll_data, target_data

def get_target_from_data(data):#, folder_path
    record = data["target"][0]
    #folder = Path(folder_path)
    #image_path = folder / record["img"]
    return {
        "image": record["img"],
        "pitch": record["pitch"],
        "yaw": record["yaw"],
        "roll": record["roll"],
        "score": record["score"],
        "embedding": None
    }

def load_images_by_data(data, axis_key):#, folder_path
    """Loads all images from folder"""
    results = []
    #folder = Path(folder_path)
    # if not folder.exists():
    #     print(f"[!] Folder doesn't exists: {folder_path}")
    #     return results
    
    for record in data[axis_key]:
        image_path = Path(record["img"])#folder / item["img"]
        if not image_path.exists():
            print(f"[!] Image not found: {image_path}")
            continue

        results.append({
            "image": str(image_path),
            "pitch": record["pitch"],
            "yaw": record["yaw"],
            "roll": record["roll"],
            "score": record["score"],
            "raw": None,
            "distance": None,
             "axis": AXIS_MAP[axis_key],
            "embedding": None
        })    

    return results

def compare(target, records, category_name, model_name):
    """Compares racords to target"""

    print(f"\nCOMPARING: {target['image']} vs {category_name} [{model_name}]")
    
    for record in records:
        if record.get("embedding") is None:
            continue

        distance, raw = calculate_distance(target["embedding"], record["embedding"])
        record["raw"] = raw
        record["distance"] = distance


def print_results(res):
    n = len(res)
    for i in range(n):
        result_str = format_result(res[i][0], res[i][1], res[i][2], res[i][3], res[i][4])
        print(result_str)

def sort_by_raw(res):
    n = len(res)

    for i in range(n):
        max_index = i

        for j in range(i+1, n):
            if res[j][3] > res[max_index][3]:
                max_index = j

        temp = res[i]
        res[i] = res[max_index]
        res[max_index] = temp

    return res

def intersection(embs1, embs2):
    if not embs1:
        return []

    if not embs2:
        return []
    
    embs2_min_raw = embs2[-1][3]

    res = []
    n = len(embs1)
    for i in range(n):
        if(embs1[i][3] >= embs2_min_raw):
            res.append(embs1[i])

    return res

# ============================================================================
# PLOTS
# ============================================================================

def generate_plot_angle_results(records, axis_name, model_name, person, output_dir="."):
    angles = [] # x
    raws = [] # y

    for record in records:
        angles.append(record[axis_name])

        if(record.get("raw") is not None):
            raws.append(record["raw"])
        else:
            raws.append(0)

    plt.figure(figsize=(10, 6))
    plt.scatter(angles, raws, alpha=0.6, s=5)
    plt.axhline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    plt.xlabel(f"{axis_name} (°)")
    plt.ylabel("raw similarity")
    plt.title(f"{model_name}: similarity vs {axis_name}")
    plt.grid(True, alpha=0.3)

    filename = f"{output_dir}/{model_name}_{axis_name}_{person}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Збережено: {filename}")

def plot_all_angle_tests(pitch_records, yaw_records, roll_records, model_name, preson, output_dir="."):
    records_map = {
        "pitch": pitch_records,
        "yaw": yaw_records,
        "roll": roll_records,
    }

    for axis_name in AXIS_MAP.values():
        generate_plot_angle_results(records_map[axis_name], axis_name, model_name, preson, output_dir)

# ============================================================================
# MAIN PROGRAM
# ============================================================================

print("=" * 80)
print("IMAGES LOADING")
print("=" * 80)

me_data = load_data_from_json("results_me.json")

# Images for pitch test
me_pitch_records = load_images_by_data(me_data, "1")#, ME_PITCH_PATH
print(f"\n📄 Records found: {len(me_pitch_records)}")
for record in me_pitch_records:
    print(f"  - {record["image"]}")

# Images for pitch test
me_yaw_records = load_images_by_data(me_data, "2")#, ME_YAW_PATH
print(f"\n📄 Records found: {len(me_yaw_records)}")
for record in me_yaw_records:
    print(f"  - {record["image"]}")
  
# Images for pitch test
me_roll_records = load_images_by_data(me_data, "3")#, ME_ROLL_PATH
print(f"\n📄 Records found: {len(me_roll_records)}")
for record in me_roll_records:
    print(f"  - {record["image"]}")

me_target = get_target_from_data(me_data)
# ============================================================================
# TEST 1: FaceNet512
# ============================================================================

print("\n" + "=" * 80)
print("TEST 1: Testing FaceNet512")
print("=" * 80)

print("\n🔄 Image processing FaceNet512...")
print("\n🔄 Target embeddings...")
add_embeddings_to_target(me_target, "Facenet512")
print("\n🔄 Pitch embeddings...")
add_all_embeddings_to_records(me_pitch_records, "Facenet512")
print("\n🔄 Yaw embeddings...")
add_all_embeddings_to_records(me_yaw_records, "Facenet512")
print("\n🔄 Roll embeddings...")
add_all_embeddings_to_records(me_roll_records, "Facenet512")

print("\n" + "=" * 80)
print("RESULTS: FaceNet512")
print("=" * 80)

compare(me_target, me_pitch_records, "me_pitch", "FaceNet512")
compare(me_target, me_yaw_records, "me_yaw", "FaceNet512")
compare(me_target, me_roll_records, "me_roll", "FaceNet512")

# for record in me_pitch_records:
#     print(f"Image: {Path(record['image']).name} | raw: {record["raw"]} | distance: {record["distance"]}")

plot_all_angle_tests(me_pitch_records, me_yaw_records, me_roll_records, "FaceNet512", "me")

# # ============================================================================
# # TEST 2: ArcFace
# # ============================================================================

print("\n" + "=" * 80)
print("TEST 2: Testing ArcFace")
print("=" * 80)



print("\n" + "=" * 80)
print("RESULTS: ArcFace")
print("=" * 80)



# # ============================================================================
# # TEST 3: Facenet
# # ============================================================================

print("\n" + "=" * 80)
print("TEST 3: Testing Facenet")
print("=" * 80)

print("\n" + "=" * 80)
print("RESULTS: Facenet")
print("=" * 80)


# # ============================================================================
# # TEST 4: VGG-Face
# # ============================================================================

print("\n" + "=" * 80)
print("TEST 4: Testing VGG-Face")
print("=" * 80)

print("\n" + "=" * 80)
print("RESULTS: VGG-Face")
print("=" * 80)


# # ============================================================================
# # TEST 5: Dlib
# # ============================================================================

print("\n" + "=" * 80)
print("TEST 5: Testing Dlib")
print("=" * 80)


print("\n" + "=" * 80)
print("RESULTS: Dlib")
print("=" * 80)


