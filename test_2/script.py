from deepface import DeepFace
import numpy as np
import os
from pathlib import Path

dir = "D:\\Projects\\python-opencv-test\\photos\\facial_rotation_test\\"

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
        print(f"[!] Exception {path} ({model_name}): {e}")

        # try:
        #     Path(path).unlink()
        #     print(f"Deleted: {path}")
        # except Exception as delete_error:
        #     print(f"Could not delete {path}: {delete_error}")

        return None

def get_all_embeddings(images_dict, model_name):
    """Takes embeddings from all images in dictionary"""
    embeddings = {}
    for name, path in images_dict.items():
        emb = get_embeddings(path, model_name=model_name)
        if emb is not None:
            embeddings[name] = emb
    return embeddings

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


def compare(doc_emb_dict, user_emb_dict, category_name, model_name):
    """Compares docs to images from category """
    
    results = []
    
    for doc_name, doc_emb in doc_emb_dict.items():

        print(f"\nCOMPARING: {doc_name} vs {category_name} [{model_name}]")

        if doc_emb is None:
            print(f"[!] Embeddings not found: {doc_name}")
            continue

        for image_name, user_emb in user_emb_dict.items():
            if user_emb is None:
                continue

            distance, raw = calculate_distance(doc_emb, user_emb)

            results.append((doc_name, image_name, distance, raw, category_name))

    return results

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
# MAIN PROGRAM
# ============================================================================

print("=" * 80)
print("IMAGES LOADING")
print("=" * 80)

# Завантажуємо документи
docs_folder = dir + "target\\"
docs_images = load_images_from_folder(docs_folder)
print(f"\n📄 Targets found: {len(docs_images)}")
for name in docs_images.keys():
    print(f"  - {name}")

# Завантажуємо фото (гарне освітлення)
me_good_folder = dir + "source\\"
me_good_images = load_images_from_folder(me_good_folder)
print(f"\n📸 Images found (source): {len(me_good_images)}")
for name in me_good_images.keys():
    print(f"  - {name}")

# Завантажуємо фото (гарне освітлення)
not_me_folder = dir + "source\\not_me"
not_me_images = load_images_from_folder(not_me_folder)
print(f"\n📸 Images found (source/not_me): {len(not_me_images)}")
for name in not_me_images.keys():
    print(f"  - {name}")

# ============================================================================
# TEST 1: FaceNet512
# ============================================================================

print("\n" + "=" * 80)
print("TEST 1: Testing FaceNet512")
print("=" * 80)

print("\n🔄 Image processing FaceNet512...")
doc_embeddings_fn512 = get_all_embeddings(docs_images, "Facenet512")
me_embeddings_fn512 = get_all_embeddings(me_good_images, "Facenet512")
not_me_embeddings_fn512 = get_all_embeddings(not_me_images, "Facenet512")

print("\n" + "=" * 80)
print("RESULTS: FaceNet512")
print("=" * 80)

all_results_fn512 = []
all_results_fn512.extend(compare(doc_embeddings_fn512, me_embeddings_fn512, "me", "FaceNet512"))
all_results_fn512.extend(compare(doc_embeddings_fn512, not_me_embeddings_fn512, "not_me", "FaceNet512"))

sort_by_raw(all_results_fn512)
print_results(all_results_fn512)

# # ============================================================================
# # TEST 2: ArcFace
# # ============================================================================

print("\n" + "=" * 80)
print("TEST 2: Testing ArcFace")
print("=" * 80)

print("\n🔄 Image processing ArcFace...")
doc_embeddings_arc = get_all_embeddings(docs_images, "ArcFace")
me_good_embeddings_arc = get_all_embeddings(me_good_images, "ArcFace")
not_me_embeddings_arc= get_all_embeddings(not_me_images, "ArcFace")

print("\n" + "=" * 80)
print("RESULTS: ArcFace")
print("=" * 80)

all_results_arc = []
all_results_arc.extend(compare(doc_embeddings_arc, me_good_embeddings_arc, "me", "ArcFace"))
all_results_arc.extend(compare(doc_embeddings_arc, not_me_embeddings_arc, "not_me", "ArcFace"))

sort_by_raw(all_results_arc)
print_results(all_results_arc)

# # ============================================================================
# # TEST 3: Facenet
# # ============================================================================

print("\n" + "=" * 80)
print("TEST 3: Testing Facenet")
print("=" * 80)

print("\n🔄 Image processing Facenet...")
doc_embeddings_fn = get_all_embeddings(docs_images, "Facenet")
me_good_embeddings_fn = get_all_embeddings(me_good_images, "Facenet")
not_me_embeddings_fn= get_all_embeddings(not_me_images, "Facenet")

print("\n" + "=" * 80)
print("RESULTS: Facenet")
print("=" * 80)

all_results_fn = []
all_results_fn.extend(compare(doc_embeddings_fn, me_good_embeddings_fn, "me", "Facenet"))
all_results_fn.extend(compare(doc_embeddings_fn, not_me_embeddings_fn, "not_me", "Facenet"))

sort_by_raw(all_results_fn)
print_results(all_results_fn)

# # ============================================================================
# # TEST 4: VGG-Face
# # ============================================================================

print("\n" + "=" * 80)
print("TEST 4: Testing VGG-Face")
print("=" * 80)

print("\n🔄 Image processing VGG-Face...")
doc_embeddings_vgg = get_all_embeddings(docs_images, "VGG-Face")
me_good_embeddings_vgg = get_all_embeddings(me_good_images, "VGG-Face")
not_me_embeddings_vgg= get_all_embeddings(not_me_images, "VGG-Face")

print("\n" + "=" * 80)
print("RESULTS: VGG-Face")
print("=" * 80)

all_results_vgg = []
all_results_vgg.extend(compare(doc_embeddings_vgg, me_good_embeddings_vgg, "me", "VGG-Face"))
all_results_vgg.extend(compare(doc_embeddings_vgg, not_me_embeddings_vgg, "not_me", "VGG-Face"))

sort_by_raw(all_results_vgg)
print_results(all_results_vgg)
# # ============================================================================
# # TEST 5: Dlib
# # ============================================================================

print("\n" + "=" * 80)
print("TEST 5: Testing Dlib")
print("=" * 80)

print("\n🔄 Image processing Dlib...")
doc_embeddings_dlib = get_all_embeddings(docs_images, "Dlib")
me_good_embeddings_dlib = get_all_embeddings(me_good_images, "Dlib")
not_me_embeddings_dlib = get_all_embeddings(not_me_images, "Dlib")

print("\n" + "=" * 80)
print("RESULTS: Dlib")
print("=" * 80)

all_results_dlib = []
all_results_dlib.extend(compare(doc_embeddings_dlib, me_good_embeddings_dlib, "me", "Dlib"))
all_results_dlib.extend(compare(doc_embeddings_dlib, not_me_embeddings_dlib, "not_me", "Dlib"))

sort_by_raw(all_results_dlib)
print_results(all_results_dlib)

