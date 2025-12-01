from deepface import DeepFace
import numpy as np
import os
from pathlib import Path

dir = "D:\\Projects\\python-opencv-test\\photos\\"

def normalize_embedding(embedding):
    """Нормалізує вектор до одиничної довжини"""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def get_encoding(path, model_name):
    try:
        embedding = DeepFace.represent(
            img_path=path,
            model_name=model_name,
            enforce_detection=True
        )

        enc = np.array(embedding[0]["embedding"])

        return normalize_embedding(enc)
    except Exception as e:
        print(f"[!] Помилка {path} ({model_name}): {e}")
        return None

def load_images_from_folder(folder_path):
    images = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    folder = Path(folder_path)
    if not folder.exists():
        print(f"[!] Папка не існує: {folder_path}")
        return images

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in valid_extensions:
            images[file.stem] = str(file)

    return images

print("=" * 80)
print("ЗАВАНТАЖЕННЯ ФОТОГРАФІЙ")
print("=" * 80)

# Завантажуємо документи
docs_folder = dir + "docs\\"
docs_images = load_images_from_folder(docs_folder)
print(f"\n📄 Знайдено документів: {len(docs_images)}")
for name in docs_images.keys():
    print(f"  - {name}")

# Завантажуємо фото (гарне освітлення)
me_good_folder = dir + "me_good\\"
me_good_images = load_images_from_folder(me_good_folder)
print(f"\n📸 Знайдено фото (me_good): {len(me_good_images)}")
for name in me_good_images.keys():
    print(f"  - {name}")

def analyze_embedding_norms(encodings_dict, model_name):
    print(f"\n{'=' * 80}")
    print(f"АНАЛІЗ НОРМ EMBEDDINGS: {model_name}")
    print(f"{'=' * 80}")

    norms = []
    for name, enc in encodings_dict.items():
        if enc is not None:
            norm = np.linalg.norm(enc)
            norms.append(norm)
            print(f"{name:<25}: L2 norm = {norm:.4f}")

    if norms:
        print(f"\nСтатистика:")
        print(f"  Min:  {min(norms):.4f}")
        print(f"  Max:  {max(norms):.4f}")
        print(f"  Mean: {np.mean(norms):.4f}")
        print(f"  Std:  {np.std(norms):.4f}")

        mean_norm = np.mean(norms)
        if 0.95 <= mean_norm <= 1.05:
            print("\n  ✅ Вектори виглядають НОРМАЛІЗОВАНИМИ (L2 norm ≈ 1.0)")
        else:
            print(f"\n  ⚠️ Вектори виглядають НЕ нормалізованими (L2 norm ≈ {mean_norm:.2f})")

def get_all_encodings(images_dict, model_name):
    encodings = {}
    for name, path in images_dict.items():
        enc = get_encoding(path, model_name=model_name)
        if enc is not None:
            encodings[name] = enc
    return encodings

print("\n🔄 Обробка зображень з Facenet512...")
doc_encodings_arc = get_all_encodings(docs_images, "Facenet512")
me_good_encodings_arc = get_all_encodings(me_good_images, "Facenet512")

# Аналіз норм для Facenet512
analyze_embedding_norms(doc_encodings_arc, "Facenet512")
analyze_embedding_norms(me_good_encodings_arc, "Facenet512")

print("\n🔄 Обробка зображень з ArcFace...")
doc_encodings_arc_2 = get_all_encodings(docs_images, "ArcFace")
me_good_encodings_arc_2 = get_all_encodings(me_good_images, "ArcFace")

# Аналіз норм для ArcFace
analyze_embedding_norms(doc_encodings_arc_2, "ArcFace")
analyze_embedding_norms(me_good_encodings_arc_2, "ArcFace")
