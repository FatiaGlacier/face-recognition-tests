from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dir = "D:\\Projects\\python-opencv-test\\photos\\treshhold_check\\"


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
    """Обчислює косинусну схожість між двома нормалізованими векторами"""
    raw = float(np.dot(enc1, enc2))
    confidence = (raw + 1) / 2 * 100  # Перетворюємо [-1, 1] -> [0, 100]
    return confidence, raw


def euclidean_similarity(enc1, enc2):
    """Обчислює евклідову відстань і конвертує у відсотки схожості"""
    distance = float(np.linalg.norm(enc1 - enc2))
    distance = np.clip(distance, 0.0, 2.0)
    cosine = 1.0 - (distance ** 2) / 2.0
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


# ============================================================================
# ФУНКЦІЇ ДЛЯ ЗБОРУ ДАНИХ
# ============================================================================

def collect_comparison_data(doc_encodings, user_encodings):
    """Збирає дані порівнянь - повертає списки cosine і euclidean scores"""
    cosine_scores = []
    euclidean_scores = []

    for doc_name, doc_enc in doc_encodings.items():
        if doc_enc is None:
            continue

        for user_name, user_enc in user_encodings.items():
            if user_enc is None:
                continue

            cos_percent, _ = cosine_similarity(doc_enc, user_enc)
            euc_percent, _ = euclidean_similarity(doc_enc, user_enc)

            cosine_scores.append(float(cos_percent))
            euclidean_scores.append(float(euc_percent))

    return cosine_scores, euclidean_scores


# ============================================================================
# ФУНКЦІЇ ДЛЯ ПОБУДОВИ ГРАФІКІВ
# ============================================================================

def plot_single_metric(me_scores, notme_scores, metric_name, model_name,
                       me_color, notme_color, ax):
    """Будує один графік для однієї метрики"""
    bins = np.linspace(0, 100, 30)

    # Гістограми
    ax.hist(me_scores, bins=bins, alpha=0.6, color=me_color,
            label='Я (позитивні)', edgecolor='black', linewidth=0.5)
    ax.hist(notme_scores, bins=bins, alpha=0.6, color=notme_color,
            label='Не я (негативні)', edgecolor='black', linewidth=0.5)

    # Середні значення
    me_mean = np.mean(me_scores)
    me_std = np.std(me_scores)
    notme_mean = np.mean(notme_scores)
    notme_std = np.std(notme_scores)

    ax.axvline(me_mean, color=me_color, linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(notme_mean, color=notme_color, linestyle='--', linewidth=2, alpha=0.8)

    # Оформлення
    ax.set_xlabel('Score (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Кількість', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_name} - {metric_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Статистика
    stats_text = (f'Я: μ={me_mean:.1f}%, σ={me_std:.1f}%\n'
                  f'Не я: μ={notme_mean:.1f}%, σ={notme_std:.1f}%\n'
                  f'Δμ={abs(me_mean - notme_mean):.1f}%')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=9, family='monospace')


def create_plots(me_cosine_fn, me_euc_fn, notme_cosine_fn, notme_euc_fn,
                 me_cosine_arc, me_euc_arc, notme_cosine_arc, notme_euc_arc):
    """Створює 4 графіки: 2 для FaceNet512 і 2 для ArcFace"""

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Face Verification Analysis - Розподіл схожості',
                 fontsize=16, fontweight='bold', y=0.995)

    # Grid layout: 2 rows x 2 columns
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25,
                          left=0.06, right=0.98, top=0.95, bottom=0.06)

    # FaceNet512 - Cosine (верхній лівий)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_single_metric(me_cosine_fn, notme_cosine_fn,
                       'Cosine Similarity', 'FaceNet512',
                       'blue', 'darkred', ax1)

    # FaceNet512 - Euclidean (верхній правий)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_single_metric(me_euc_fn, notme_euc_fn,
                       'Euclidean Distance', 'FaceNet512',
                       'cyan', 'red', ax2)

    # ArcFace - Cosine (нижній лівий)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_single_metric(me_cosine_arc, notme_cosine_arc,
                       'Cosine Similarity', 'ArcFace',
                       'blue', 'darkred', ax3)

    # ArcFace - Euclidean (нижній правий)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_single_metric(me_euc_arc, notme_euc_arc,
                       'Euclidean Distance', 'ArcFace',
                       'cyan', 'red', ax4)

    # Зберігаємо
    plt.savefig('face_verification_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Графік збережено: face_verification_analysis.png")

    #plt.show()


def calculate_metrics_table(me_scores, notme_scores, thresholds, title):
    """Розраховує та виводить таблицю метрик"""
    print(f"\n{'=' * 90}")
    print(f"{title}")
    print(f"{'=' * 90}")
    print(f"{'Threshold':<12} {'FAR (%)':<12} {'FRR (%)':<12} {'Accuracy (%)':<15} "
          f"{'TP':<8} {'TN':<8} {'FP':<8} {'FN':<8}")
    print(f"{'=' * 90}")

    for threshold in thresholds:
        TP = sum(1 for s in me_scores if s >= threshold)
        FN = sum(1 for s in me_scores if s < threshold)
        TN = sum(1 for s in notme_scores if s < threshold)
        FP = sum(1 for s in notme_scores if s >= threshold)

        FAR = (FP / (FP + TN) * 100) if (FP + TN) > 0 else 0
        FRR = (FN / (FN + TP) * 100) if (FN + TP) > 0 else 0
        Accuracy = ((TP + TN) / (TP + TN + FP + FN) * 100) if (TP + TN + FP + FN) > 0 else 0

        # Виділяємо найкращий threshold (FAR < 1%)
        marker = " ⭐" if FAR < 1.0 and Accuracy > 90 else ""

        print(f"{threshold:<12.1f} {FAR:<12.2f} {FRR:<12.2f} {Accuracy:<15.2f} "
              f"{TP:<8} {TN:<8} {FP:<8} {FN:<8}{marker}")


# ============================================================================
# ОСНОВНА ПРОГРАМА
# ============================================================================

print("=" * 80)
print("FACE VERIFICATION ANALYSIS")
print("=" * 80)

# Завантажуємо фотографії
print("\n🔄 Завантаження фотографій...")
docs_images = load_images_from_folder(dir + "docs\\")
me_good_images = load_images_from_folder(dir + "me_good\\")
me_light_images = load_images_from_folder(dir + "me_light\\")
me_normal_images = load_images_from_folder(dir + "me_normal\\")
not_me_images = load_images_from_folder(dir + "not_me\\")

print(f"  📄 Документи: {len(docs_images)}")
print(f"  📸 me_good: {len(me_good_images)}")
print(f"  💡 me_light: {len(me_light_images)}")
print(f"  🙂 me_normal: {len(me_normal_images)}")
print(f"  🥶 not_me: {len(not_me_images)}")

# Обробка FaceNet512
print("\n🔄 Обробка з FaceNet512...")
doc_encodings_fn512 = get_all_encodings(docs_images, "Facenet512")
me_good_encodings_fn512 = get_all_encodings(me_good_images, "Facenet512")
me_light_encodings_fn512 = get_all_encodings(me_light_images, "Facenet512")
me_normal_encodings_fn512 = get_all_encodings(me_normal_images, "Facenet512")
not_me_encodings_fn512 = get_all_encodings(not_me_images, "Facenet512")

# Обробка ArcFace
print("\n🔄 Обробка з ArcFace...")
doc_encodings_arc = get_all_encodings(docs_images, "ArcFace")
me_good_encodings_arc = get_all_encodings(me_good_images, "ArcFace")
me_light_encodings_arc = get_all_encodings(me_light_images, "ArcFace")
me_normal_encodings_arc = get_all_encodings(me_normal_images, "ArcFace")
not_me_encodings_arc = get_all_encodings(not_me_images, "ArcFace")

# Об'єднуємо всі "мої" фото
print("\n🔄 Збір даних для аналізу...")
all_me_encodings_fn512 = {**me_good_encodings_fn512, **me_light_encodings_fn512, **me_normal_encodings_fn512}
all_me_encodings_arc = {**me_good_encodings_arc, **me_light_encodings_arc, **me_normal_encodings_arc}

# Збираємо scores для FaceNet512
me_cosine_fn512, me_euclidean_fn512 = collect_comparison_data(doc_encodings_fn512, all_me_encodings_fn512)
notme_cosine_fn512, notme_euclidean_fn512 = collect_comparison_data(doc_encodings_fn512, not_me_encodings_fn512)

# Збираємо scores для ArcFace
me_cosine_arc, me_euclidean_arc = collect_comparison_data(doc_encodings_arc, all_me_encodings_arc)
notme_cosine_arc, notme_euclidean_arc = collect_comparison_data(doc_encodings_arc, not_me_encodings_arc)

print(f"  ✅ FaceNet512: {len(me_cosine_fn512)} порівнянь (Я) + {len(notme_cosine_fn512)} порівнянь (Не я)")
print(f"  ✅ ArcFace: {len(me_cosine_arc)} порівнянь (Я) + {len(notme_cosine_arc)} порівнянь (Не я)")

# Будуємо графіки
print("\n📊 Створення графіків...")
create_plots(me_cosine_fn512, me_euclidean_fn512, notme_cosine_fn512, notme_euclidean_fn512,
             me_cosine_arc, me_euclidean_arc, notme_cosine_arc, notme_euclidean_arc)

# Виводимо таблиці метрик
thresholds = [50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100]
#[60, 65, 70, 75, 80, 85, 90, 95][50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100]


calculate_metrics_table(me_cosine_fn512, notme_cosine_fn512, thresholds,
                        "FACENET512 - COSINE SIMILARITY")

calculate_metrics_table(me_euclidean_fn512, notme_euclidean_fn512, thresholds,
                        "FACENET512 - EUCLIDEAN DISTANCE")

calculate_metrics_table(me_cosine_arc, notme_cosine_arc, thresholds,
                        "ARCFACE - COSINE SIMILARITY")

calculate_metrics_table(me_euclidean_arc, notme_euclidean_arc, thresholds,
                        "ARCFACE - EUCLIDEAN DISTANCE")

print("\n" + "=" * 80)
print("✅ АНАЛІЗ ЗАВЕРШЕНО")
print("=" * 80)
print("\n💡 Рекомендації:")
print("  - Threshold'и з ⭐ мають FAR < 1% та Accuracy > 90%")
print("  - Для автоапруву вибирай threshold де FAR < 1%")
print("  - Δμ показує наскільки добре розділені розподіли")
print("=" * 80)