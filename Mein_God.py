from deepface import DeepFace
import numpy as np
import os
from pathlib import Path

dir = "D:\\Projects\\python-opencv-test\\photos\\"


# ============================================================================
# ФУНКЦІЇ ДЛЯ РОБОТИ З EMBEDDINGS
# ============================================================================

def normalize_embedding(embedding):
    """Нормалізує вектор до одиничної довжини"""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def get_encoding(path, model_name):
    """Отримує нормалізований embedding для зображення"""
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


def get_all_encodings(images_dict, model_name):
    """Отримує embeddings для всіх зображень у словнику"""
    encodings = {}
    for name, path in images_dict.items():
        enc = get_encoding(path, model_name=model_name)
        if enc is not None:
            encodings[name] = enc
    return encodings


# ============================================================================
# ФУНКЦІЇ ДЛЯ ОБЧИСЛЕННЯ СХОЖОСТІ
# ============================================================================

def cosine_similarity(enc1, enc2):
    """
    Обчислює косинусну схожість між двома нормалізованими векторами
    Для нормалізованих векторів це просто скалярний добуток
    """
    raw = float(np.dot(enc1, enc2))
    confidence = (raw + 1) / 2 * 100  # Перетворюємо [-1, 1] -> [0, 100]
    return confidence, raw


def euclidean_similarity(enc1, enc2):
    """
    Обчислює евклідову відстань і конвертує у відсотки схожості
    Для нормалізованих векторів використовує математичний зв'язок з косинусом:
    cos = 1 - d²/2
    """
    distance = float(np.linalg.norm(enc1 - enc2))

    # Обмежуємо відстань в допустимих межах [0, 2] для нормалізованих векторів
    distance = np.clip(distance, 0.0, 2.0)

    # Обчислюємо косинус з відстані: cos = 1 - d²/2
    cosine = 1.0 - (distance ** 2) / 2.0

    # Конвертуємо косинус в проценти: [-1, 1] → [0, 100]
    confidence = (cosine + 1.0) / 2.0 * 100.0

    return confidence, distance


# ============================================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================================

def load_images_from_folder(folder_path):
    """Завантажує всі зображення з папки"""
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


def analyze_embedding_norms(encodings_dict, model_name):
    """Аналізує норми embeddings для перевірки нормалізації"""
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


def format_result(src, dst, cosine_percent, cosine_raw, euclid_percent, distance):
    """Форматує результат порівняння"""
    return (
        f"{src} vs {dst}: "
        f"Cosine={float(cosine_percent):.2f}%, "
        f"Cosine raw={float(cosine_raw):.4f}, "
        f"Euclidean={float(euclid_percent):.2f}% "
        f"(dist={float(distance):.4f})"
    )


def compare_and_print(doc_name, doc_enc, user_encodings_dict, category_name, model_name):
    """Порівнює документ з усіма фото з категорії"""
    print(f"\n{'=' * 80}")
    print(f"ПОРІВНЯННЯ: {doc_name} vs {category_name} [{model_name}]")
    print(f"{'=' * 80}")

    if doc_enc is None:
        print(f"[!] Немає encoding для документа {doc_name}")
        return

    results = []
    for user_name, user_enc in user_encodings_dict.items():
        if user_enc is None:
            continue

        cos, raw = cosine_similarity(doc_enc, user_enc)
        euc, dist = euclidean_similarity(doc_enc, user_enc)

        result_str = format_result(doc_name, user_name, cos, raw, euc, dist)
        print(result_str)
        results.append((user_name, cos, euc))

    return results


# def verify_match(doc_name, doc_enc_fn512, doc_enc_arc, user_encodings_fn512, user_encodings_arc,
#                  category_name, threshold_cosine=80.0, threshold_euclidean=80.0):
#     """
#     Перевіряє, чи збігаються особи за обома моделями та обома метриками
#     """
#     print(f"\n{'=' * 80}")
#     print(f"ВЕРИФІКАЦІЯ: {doc_name} vs {category_name} (ОБИДВІ МОДЕЛІ + ОБИ МЕТРИКИ)")
#     print(f"{'=' * 80}")
#
#     for user_name in user_encodings_fn512.keys():
#         if user_name not in user_encodings_arc:
#             continue
#
#         # FaceNet512 - Cosine
#         cos_fn512, _ = cosine_similarity(doc_enc_fn512, user_encodings_fn512[user_name])
#         # FaceNet512 - Euclidean
#         euc_fn512, _ = euclidean_similarity(doc_enc_fn512, user_encodings_fn512[user_name])
#
#         # ArcFace - Cosine
#         cos_arc, _ = cosine_similarity(doc_enc_arc, user_encodings_arc[user_name])
#         # ArcFace - Euclidean
#         euc_arc, _ = euclidean_similarity(doc_enc_arc, user_encodings_arc[user_name])
#
#         # Перевірка всіх 4 умов
#         match_fn512_cos = cos_fn512 >= threshold_cosine
#         match_fn512_euc = euc_fn512 >= threshold_euclidean
#         match_arc_cos = cos_arc >= threshold_cosine
#         match_arc_euc = euc_arc >= threshold_euclidean
#
#         # Всі 4 метрики мають підтвердити
#         all_match = match_fn512_cos and match_fn512_euc and match_arc_cos and match_arc_euc
#
#         # Підрахунок скільки метрик підтвердили
#         matches_count = sum([match_fn512_cos, match_fn512_euc, match_arc_cos, match_arc_euc])
#
#         if all_match:
#             status = "✅ ПІДТВЕРДЖЕНО (4/4)"
#         elif matches_count >= 3:
#             status = f"⚠️ ЧАСТКОВО ({matches_count}/4)"
#         else:
#             status = f"❌ ВІДХИЛЕНО ({matches_count}/4)"
#
#         print(f"\n{user_name}:")
#         print(f"  FaceNet512: Cosine={cos_fn512:6.2f}% {'✓' if match_fn512_cos else '✗'} | "
#               f"Euclidean={euc_fn512:6.2f}% {'✓' if match_fn512_euc else '✗'}")
#         print(f"  ArcFace:    Cosine={cos_arc:6.2f}% {'✓' if match_arc_cos else '✗'} | "
#               f"Euclidean={euc_arc:6.2f}% {'✓' if match_arc_euc else '✗'}")
#         print(f"  → {status}")

def verify_match( # _crosscheck
    doc_name,
    doc_enc_fn512,
    doc_enc_arc,
    user_encodings_fn512,
    user_encodings_arc,
    category_name,
    arc_strong=83.0,      # % для ArcFace cosine
    arc_mid=75.0,         # середня зона
    fn512_strong=75.0,    # % для FaceNet512 euclidean (пам’ятай: менше = краще)
    fn512_mid=60.0        # середня зона
):
    """
    Cross-check перевірка:
    ArcFace (cosine) + FaceNet512 (euclidean на нормалізованих векторах)
    """

    print(f"\n{'=' * 80}")
    print(f"ВЕРИФІКАЦІЯ (CROSS-CHECK): {doc_name} vs {category_name}")
    print(f"{'=' * 80}")

    for user_name in user_encodings_fn512.keys():
        if user_name not in user_encodings_arc:
            continue

        # ArcFace – cosine (%)
        cos_arc, _ = cosine_similarity(doc_enc_arc, user_encodings_arc[user_name])

        # FaceNet512 – euclidean (%) на нормалізованих
        euc_fn512, _ = euclidean_similarity(doc_enc_fn512, user_encodings_fn512[user_name])

        # Логіка порогів
        arc_str = cos_arc >= arc_strong
        arc_mid_ok = arc_mid <= cos_arc < arc_strong

        fn_str = euc_fn512 >= fn512_strong  # оскільки ти мапиш евклід у %, де більше = краще
        fn_mid_ok = fn512_mid <= euc_fn512 < fn512_strong

        # Рішення
        if arc_str and fn_str:
            status = "✅ APPROVE (обидва сильні)"
        elif (arc_str and fn_mid_ok) or (fn_str and arc_mid_ok):
            status = "⚠ REVIEW (один сильний, другий середній)"
        else:
            status = "❌ REJECT"

        print(f"\n{user_name}:")
        print(f"  ArcFace (cosine):     {cos_arc:6.2f}% {'✓' if arc_str or arc_mid_ok else '✗'}")
        print(f"  FaceNet512 (euclid):  {euc_fn512:6.2f}% {'✓' if fn_str or fn_mid_ok else '✗'}")
        print(f"  → {status}")

# ============================================================================
# ОСНОВНА ПРОГРАМА
# ============================================================================

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

# Завантажуємо фото (під лампою)
me_light_folder = dir + "me_light\\"
me_light_images = load_images_from_folder(me_light_folder)
print(f"\n💡 Знайдено фото (me_light): {len(me_light_images)}")
for name in me_light_images.keys():
    print(f"  - {name}")

# Завантажуємо фото (нормальне)
me_normal_folder = dir + "me_normal\\"
me_normal_images = load_images_from_folder(me_normal_folder)
print(f"\n🙂 Знайдено фото (me_normal): {len(me_normal_images)}")
for name in me_normal_images.keys():
    print(f"  - {name}")

# Завантажуємо фото (не я)
not_me_folder = dir + "not_me\\"
not_me_images = load_images_from_folder(not_me_folder)
print(f"\n🥶 Знайдено фото (not_me): {len(not_me_images)}")
for name in not_me_images.keys():
    print(f"  - {name}")

# ============================================================================
# ТЕСТ 1: FaceNet512
# ============================================================================

print("\n" + "=" * 80)
print("ТЕСТ 1: ТЕСТУВАННЯ З FaceNet512")
print("=" * 80)

print("\n🔄 Обробка зображень з FaceNet512...")
doc_encodings_fn512 = get_all_encodings(docs_images, "Facenet512")
me_good_encodings_fn512 = get_all_encodings(me_good_images, "Facenet512")
me_light_encodings_fn512 = get_all_encodings(me_light_images, "Facenet512")
me_normal_encodings_fn512 = get_all_encodings(me_normal_images, "Facenet512")
not_me_encodings_fn512 = get_all_encodings(not_me_images, "Facenet512")

# Аналіз норм
analyze_embedding_norms(doc_encodings_fn512, "FaceNet512 - Documents")
analyze_embedding_norms(me_good_encodings_fn512, "FaceNet512 - me_good")

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТИ: FaceNet512")
print("=" * 80)

for doc_name, doc_enc in doc_encodings_fn512.items():
    compare_and_print(doc_name, doc_enc, me_good_encodings_fn512, "me_good", "FaceNet512")
    compare_and_print(doc_name, doc_enc, me_light_encodings_fn512, "me_light", "FaceNet512")
    compare_and_print(doc_name, doc_enc, me_normal_encodings_fn512, "me_normal", "FaceNet512")
    compare_and_print(doc_name, doc_enc, not_me_encodings_fn512, "not_me", "FaceNet512")

# ============================================================================
# ТЕСТ 2: ArcFace
# ============================================================================

print("\n" + "=" * 80)
print("ТЕСТ 2: ТЕСТУВАННЯ З ArcFace")
print("=" * 80)

print("\n🔄 Обробка зображень з ArcFace...")
doc_encodings_arc = get_all_encodings(docs_images, "ArcFace")
me_good_encodings_arc = get_all_encodings(me_good_images, "ArcFace")
me_light_encodings_arc = get_all_encodings(me_light_images, "ArcFace")
me_normal_encodings_arc = get_all_encodings(me_normal_images, "ArcFace")
not_me_encodings_arc = get_all_encodings(not_me_images, "ArcFace")

# Аналіз норм
analyze_embedding_norms(doc_encodings_arc, "ArcFace - Documents")
analyze_embedding_norms(me_good_encodings_arc, "ArcFace - me_good")

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТИ: ArcFace")
print("=" * 80)

for doc_name, doc_enc in doc_encodings_arc.items():
    compare_and_print(doc_name, doc_enc, me_good_encodings_arc, "me_good", "ArcFace")
    compare_and_print(doc_name, doc_enc, me_light_encodings_arc, "me_light", "ArcFace")
    compare_and_print(doc_name, doc_enc, me_normal_encodings_arc, "me_normal", "ArcFace")
    compare_and_print(doc_name, doc_enc, not_me_encodings_arc, "not_me", "ArcFace")

# ============================================================================
# ТЕСТ 3: ВЕРИФІКАЦІЯ ОБОМА МОДЕЛЯМИ ТА МЕТРИКАМИ
# ============================================================================

print("\n" + "=" * 80)
print("ТЕСТ 3: ВЕРИФІКАЦІЯ ОБОМА МОДЕЛЯМИ (FaceNet512 + ArcFace) ТА МЕТРИКАМИ (Cosine + Euclidean)")
print("=" * 80)

for doc_name in doc_encodings_fn512.keys():
    if doc_name not in doc_encodings_arc:
        continue

    verify_match(
        doc_name,
        doc_encodings_fn512[doc_name],
        doc_encodings_arc[doc_name],
        me_good_encodings_fn512,
        me_good_encodings_arc,
        "me_good"
        # threshold_cosine=80.0,
        # threshold_euclidean=80.0
    )

    verify_match(
        doc_name,
        doc_encodings_fn512[doc_name],
        doc_encodings_arc[doc_name],
        me_light_encodings_fn512,
        me_light_encodings_arc,
        "me_light"
        # threshold_cosine=80.0,
        # threshold_euclidean=80.0
    )

    verify_match(
        doc_name,
        doc_encodings_fn512[doc_name],
        doc_encodings_arc[doc_name],
        me_normal_encodings_fn512,
        me_normal_encodings_arc,
        "me_normal"
        # threshold_cosine=80.0,
        # threshold_euclidean=80.0
    )

    verify_match(
        doc_name,
        doc_encodings_fn512[doc_name],
        doc_encodings_arc[doc_name],
        not_me_encodings_fn512,
        not_me_encodings_arc,
        "not_me"
        # threshold_cosine=80.0,
        # threshold_euclidean=80.0
    )

print("\n" + "=" * 80)
print("ТЕСТУВАННЯ ЗАВЕРШЕНО")
print("=" * 80)