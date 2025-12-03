import cv2
import numpy as np
from pathlib import Path
from mtcnn import MTCNN
import os
from tqdm import tqdm

# ============================================================================
# КОНФІГУРАЦІЯ
# ============================================================================

VIDEO_DIR = "D:\\Projects\\python-opencv-test\\photos\\me_good"
OUTPUT_DIR = "D:\\Projects\\python-opencv-test\\photos\\me_good\\extracted_frames\\"

# Параметри
FRAME_SKIP = 1  # Обробляємо кожен N-й фрейм (для швидкості)
TOP_N_FRAMES = 10  # Скільки найкращих фреймів зберегти
MIN_FACE_SIZE = 100  # Мінімальний розмір обличчя (пікселі)
SAVE_WITH_BBOX = False  # ⬅️ FALSE = чисті фрейми, TRUE = з зеленим квадратом
SAVE_FULL_FRAME = True  # ⬅️ TRUE = весь кадр, FALSE = тільки обличчя (cropped)

# Пороги для pose (положення обличчя)
MAX_PITCH = 20#15  # ±15° вгору/вниз (менше = строгіше)
MAX_YAW = 20#20  # ±20° вліво/вправо
MAX_ROLL = 20#15  # ±15° нахил голови


# ============================================================================
# ФУНКЦІЇ
# ============================================================================

def initialize_detector():
    """Ініціалізує MTCNN детектор"""
    print("🔧 Ініціалізація MTCNN детектора...")
    detector = MTCNN()
    print("✅ Детектор готовий\n")
    return detector


def calculate_sharpness(image):
    """
    Розраховує різкість зображення за допомогою Laplacian variance
    Вище значення = більш різке зображення
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def calculate_brightness(image):
    """
    Розраховує середню яскравість зображення
    Оптимальне значення ~128 (середній gray)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def calculate_contrast(image):
    """
    Розраховує контраст зображення (стандартне відхилення яскравості)
    Вище значення = більший контраст
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)


def calculate_face_size_score(face_box, frame_width, frame_height):
    """
    Розраховує оцінку розміру обличчя відносно фрейму
    Більше обличчя = краще для розпізнавання
    """
    x, y, w, h = face_box
    face_area = w * h
    frame_area = frame_width * frame_height

    # Відносний розмір обличчя (0-1)
    relative_size = face_area / frame_area

    # Оптимально коли обличчя займає 10-40% кадру
    if relative_size < 0.05:
        return relative_size / 0.05 * 0.5  # Дуже маленьке
    elif relative_size <= 0.4:
        return 0.5 + (relative_size - 0.05) / 0.35 * 0.5  # Оптимальний розмір
    else:
        return 1.0 - (relative_size - 0.4) / 0.6 * 0.3  # Занадто велике


def calculate_face_position_score(face_box, frame_width, frame_height):
    """
    Розраховує оцінку позиції обличчя в кадрі
    Обличчя по центру = краще
    """
    x, y, w, h = face_box

    face_center_x = x + w / 2
    face_center_y = y + h / 2

    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    # Відстань від центру (нормалізована)
    distance_x = abs(face_center_x - frame_center_x) / (frame_width / 2)
    distance_y = abs(face_center_y - frame_center_y) / (frame_height / 2)

    # Комбінована відстань (0 = центр, 1 = край)
    distance = (distance_x + distance_y) / 2

    # Інвертуємо: центр = 1.0, край = 0.0
    return 1.0 - distance


def calculate_head_pose(keypoints):
    """
    Розраховує pose обличчя з MTCNN keypoints
    Повертає: pitch (вгору/вниз), yaw (вліво/вправо), roll (нахил)
    """
    left_eye = np.array(keypoints['left_eye'])
    right_eye = np.array(keypoints['right_eye'])
    nose = np.array(keypoints['nose'])
    mouth_left = np.array(keypoints['mouth_left'])
    mouth_right = np.array(keypoints['mouth_right'])

    # 1. ROLL (нахил голови) - з кута між очима
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    roll = np.degrees(np.arctan2(dY, dX))

    # 2. YAW (поворот вліво/вправо) - з позиції носа відносно очей
    eye_center = (left_eye + right_eye) / 2
    eye_width = np.linalg.norm(right_eye - left_eye)

    # Відстань носа від лінії між очима
    nose_to_eye_center = nose - eye_center

    # Проекція на горизонтальну вісь
    eye_direction = (right_eye - left_eye) / eye_width
    nose_offset = np.dot(nose_to_eye_center, eye_direction)

    # Конвертуємо в градуси (0 = по центру, + = вправо, - = вліво)
    yaw = nose_offset * 2  # Емпіричний коефіцієнт

    # 3. PITCH (нахил вгору/вниз) - з позиції носа по вертикалі
    mouth_center = (mouth_left + mouth_right) / 2

    # Відстань від очей до рота (висота обличчя)
    face_height = np.linalg.norm(mouth_center - eye_center)

    # Вертикальна позиція носа відносно очей
    nose_y_offset = nose[1] - eye_center[1]

    # Нормалізуємо по висоті обличчя
    nose_y_ratio = nose_y_offset / face_height if face_height > 0 else 0

    # Конвертуємо в градуси (- = вгору, + = вниз)
    # Нормальна позиція носа ~0.3 від очей до рота
    pitch = (nose_y_ratio - 0.3) * 60  # Емпіричний коефіцієнт

    return pitch, yaw, roll


def is_frontal_face(keypoints, pitch_threshold=MAX_PITCH,
                    yaw_threshold=MAX_YAW, roll_threshold=MAX_ROLL):
    """
    Перевіряє чи обличчя в frontal позиції
    Повертає: (is_frontal, pitch, yaw, roll)
    """
    pitch, yaw, roll = calculate_head_pose(keypoints)

    is_frontal = (
            abs(pitch) <= pitch_threshold and
            abs(yaw) <= yaw_threshold and
            abs(roll) <= roll_threshold
    )

    return is_frontal, pitch, yaw, roll


def calculate_pose_score(pitch, yaw, roll):
    """
    Розраховує score якості pose (0-1)
    0° по всіх осях = 1.0, чим більше відхилення = менший score
    """
    # Кожна вісь окремо
    pitch_score = max(0, 1.0 - abs(pitch) / 30)  # 0° = 1.0, 30° = 0.0
    yaw_score = max(0, 1.0 - abs(yaw) / 40)
    roll_score = max(0, 1.0 - abs(roll) / 30)

    # Середнє
    pose_score = (pitch_score + yaw_score + roll_score) / 3

    return pose_score


def calculate_quality_score(frame, face_detection):
    """
    Розраховує загальну якість кадру для face recognition
    Повертає score від 0 до 1 та детальні метрики
    """
    height, width = frame.shape[:2]

    face_box = face_detection['box']
    keypoints = face_detection['keypoints']
    x, y, w, h = face_box

    # Обрізаємо обличчя (з невеликим padding)
    padding = int(max(w, h) * 0.2)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    face_roi = frame[y1:y2, x1:x2]

    if face_roi.size == 0:
        return 0.0, {}

    # 1. Різкість (35% ваги)
    sharpness = calculate_sharpness(face_roi)
    sharpness_score = min(sharpness / 500, 1.0)

    # 2. Розмір обличчя (20% ваги)
    size_score = calculate_face_size_score(face_box, width, height)

    # 3. POSE обличчя (25% ваги) ⬅️ НОВЕ!
    is_frontal, pitch, yaw, roll = is_frontal_face(keypoints)
    pose_score = calculate_pose_score(pitch, yaw, roll)

    # 4. Позиція обличчя в кадрі (10% ваги)
    position_score = calculate_face_position_score(face_box, width, height)

    # 5. Яскравість (5% ваги)
    brightness = calculate_brightness(face_roi)
    brightness_score = 1.0 - abs(brightness - 128) / 128

    # 6. Контраст (5% ваги)
    contrast = calculate_contrast(face_roi)
    contrast_score = min(contrast / 80, 1.0)

    # Загальна оцінка
    total_score = (
            sharpness_score * 0.35 +
            size_score * 0.20 +
            pose_score * 0.25 +  # ⬅️ POSE тепер 25%!
            position_score * 0.10 +
            brightness_score * 0.05 +
            contrast_score * 0.05
    )

    # Повертаємо детальні метрики
    details = {
        'sharpness': sharpness,
        'sharpness_score': sharpness_score,
        'size_score': size_score,
        'pose_score': pose_score,
        'pitch': pitch,
        'yaw': yaw,
        'roll': roll,
        'is_frontal': is_frontal,
        'position_score': position_score,
        'brightness': brightness,
        'brightness_score': brightness_score,
        'contrast': contrast,
        'contrast_score': contrast_score
    }

    return total_score, details


def extract_best_frames_from_video(video_path, detector, output_folder, video_name):
    """
    Витягує найкращі фрейми з відео
    """
    print(f"\n{'=' * 80}")
    print(f"📹 Обробка відео: {video_name}")
    print(f"{'=' * 80}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Не вдалося відкрити відео: {video_path}")
        return

    # Інформація про відео
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📊 Інформація про відео:")
    print(f"   Розмір: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Тривалість: {duration:.2f} секунд")
    print(f"   Всього фреймів: {total_frames}")
    print(f"   Обробляємо кожен {FRAME_SKIP}-й фрейм")

    # Створюємо папку для цього відео
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    # Збір даних
    frame_data = []
    frame_count = 0
    processed_count = 0

    print(f"\n🔍 Пошук та аналіз обличь...")

    pbar = tqdm(total=total_frames, desc="Обробка фреймів")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)

        # Пропускаємо фрейми для швидкості
        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue

        # Детектуємо обличчя
        detections = detector.detect_faces(frame)

        if len(detections) == 0:
            frame_count += 1
            continue

        # Беремо найбільше обличчя (якщо їх кілька)
        largest_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])

        face_box = largest_face['box']
        x, y, w, h = face_box

        # Фільтруємо занадто маленькі обличчя
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            frame_count += 1
            continue

        # Розраховуємо якість
        quality, details = calculate_quality_score(frame, largest_face)

        # Додаткові метрики для відображення
        sharpness = details['sharpness']
        brightness = details['brightness']
        contrast = details['contrast']
        pitch = details['pitch']
        yaw = details['yaw']
        roll = details['roll']
        is_frontal = details['is_frontal']

        # ФІЛЬТРУЄМО не-фронтальні обличчя ⬅️ ВАЖЛИВО!
        if not is_frontal:
            frame_count += 1
            continue

        frame_data.append({
            'frame_number': frame_count,
            'frame': frame.copy(),
            'face_box': face_box,
            'quality_score': quality,
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast,
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'confidence': largest_face['confidence'],
            'details': details
        })

        processed_count += 1
        frame_count += 1

    pbar.close()
    cap.release()

    print(f"\n✅ Оброблено: {processed_count} фреймів з обличчями")
    print(f"   З них фронтальних: {len(frame_data)}")

    if len(frame_data) == 0:
        print("❌ Не знайдено жодного ФРОНТАЛЬНОГО обличчя у відео!")
        print(f"⚠️  Спробуй збільшити пороги: MAX_PITCH, MAX_YAW, MAX_ROLL")
        return

    # Сортуємо по якості і беремо топ N
    frame_data.sort(key=lambda x: x['quality_score'], reverse=True)
    best_frames = frame_data[:TOP_N_FRAMES]

    print(f"\n📸 Збереження {len(best_frames)} найкращих фреймів:")
    print(f"{'=' * 110}")
    print(
        f"{'Rank':<6} {'Frame#':<10} {'Quality':<10} {'Sharp':<8} {'Pose':<8} {'Pitch':<8} {'Yaw':<8} {'Roll':<8} {'Conf':<8}")
    print(f"{'=' * 110}")

    for rank, frame_info in enumerate(best_frames, 1):
        frame_num = frame_info['frame_number']
        quality = frame_info['quality_score']
        sharpness = frame_info['sharpness']
        confidence = frame_info['confidence']
        pitch = frame_info['pitch']
        yaw = frame_info['yaw']
        roll = frame_info['roll']
        details = frame_info['details']

        print(f"{rank:<6} {frame_num:<10} {quality:<10.3f} "
              f"{sharpness:<8.1f} {details['pose_score']:<8.3f} "
              f"{pitch:<8.1f} {yaw:<8.1f} {roll:<8.1f} "
              f"{confidence:<8.3f}")

        # Зберігаємо фрейм
        output_filename = f"{video_name}_frame_{frame_num:06d}_quality_{quality:.3f}.jpg"
        output_path = os.path.join(video_output_folder, output_filename)

        # Вибираємо що зберігати
        if SAVE_FULL_FRAME:
            # Зберігаємо весь кадр
            if SAVE_WITH_BBOX:
                # З зеленим квадратом та метриками
                frame_to_save = frame_info['frame'].copy()
                x, y, w, h = frame_info['face_box']
                cv2.rectangle(frame_to_save, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Додаємо текст з метриками
                text1 = f"Q: {quality:.3f} | S: {sharpness:.0f}"
                text2 = f"P:{pitch:.0f} Y:{yaw:.0f} R:{roll:.0f}"
                cv2.putText(frame_to_save, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.putText(frame_to_save, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
            else:
                # Чистий кадр (без bbox)
                frame_to_save = frame_info['frame']
        else:
            # Зберігаємо тільки обличчя (cropped)
            x, y, w, h = frame_info['face_box']
            padding = int(max(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame_info['frame'].shape[1], x + w + padding)
            y2 = min(frame_info['frame'].shape[0], y + h + padding)
            frame_to_save = frame_info['frame'][y1:y2, x1:x2]

        cv2.imwrite(output_path, frame_to_save)

    print(f"{'=' * 110}")
    print(f"✅ Фрейми збережено в: {video_output_folder}")

    # Розширена статистика
    print(f"\n📊 Детальна статистика по ФРОНТАЛЬНИХ фреймах:")
    print(f"{'=' * 100}")

    all_qualities = [f['quality_score'] for f in frame_data]
    all_sharpness = [f['sharpness'] for f in frame_data]
    all_brightness = [f['brightness'] for f in frame_data]
    all_contrast = [f['contrast'] for f in frame_data]
    all_pitch = [f['pitch'] for f in frame_data]
    all_yaw = [f['yaw'] for f in frame_data]
    all_roll = [f['roll'] for f in frame_data]

    print(f"\n{'Метрика':<20} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    print(f"{'-' * 68}")
    print(
        f"{'Quality Score':<20} {np.min(all_qualities):<12.3f} {np.max(all_qualities):<12.3f} {np.mean(all_qualities):<12.3f} {np.std(all_qualities):<12.3f}")
    print(
        f"{'Sharpness':<20} {np.min(all_sharpness):<12.1f} {np.max(all_sharpness):<12.1f} {np.mean(all_sharpness):<12.1f} {np.std(all_sharpness):<12.1f}")
    print(
        f"{'Brightness':<20} {np.min(all_brightness):<12.1f} {np.max(all_brightness):<12.1f} {np.mean(all_brightness):<12.1f} {np.std(all_brightness):<12.1f}")
    print(
        f"{'Contrast':<20} {np.min(all_contrast):<12.1f} {np.max(all_contrast):<12.1f} {np.mean(all_contrast):<12.1f} {np.std(all_contrast):<12.1f}")
    print(
        f"{'Pitch (°)':<20} {np.min(all_pitch):<12.1f} {np.max(all_pitch):<12.1f} {np.mean(all_pitch):<12.1f} {np.std(all_pitch):<12.1f}")
    print(
        f"{'Yaw (°)':<20} {np.min(all_yaw):<12.1f} {np.max(all_yaw):<12.1f} {np.mean(all_yaw):<12.1f} {np.std(all_yaw):<12.1f}")
    print(
        f"{'Roll (°)':<20} {np.min(all_roll):<12.1f} {np.max(all_roll):<12.1f} {np.mean(all_roll):<12.1f} {np.std(all_roll):<12.1f}")

    print(f"\n⚠️  АНАЛІЗ ЯКОСТІ ВІДЕО:")
    print(f"   📐 Pose пороги: Pitch ±{MAX_PITCH}°, Yaw ±{MAX_YAW}°, Roll ±{MAX_ROLL}°")

    # Аналіз різкості
    avg_sharpness = np.mean(all_sharpness)
    if avg_sharpness < 100:
        print(f"   ❌ РІЗКІСТЬ ДУЖЕ НИЗЬКА ({avg_sharpness:.1f})")
        print(f"      → Відео розмите або не в фокусі")
        print(f"      → Рекомендація: використати камеру з автофокусом")
    elif avg_sharpness < 200:
        print(f"   ⚠️  Різкість низька ({avg_sharpness:.1f})")
        print(f"      → Можна покращити фокусування")
    else:
        print(f"   ✅ Різкість хороша ({avg_sharpness:.1f})")

    # Аналіз яскравості
    avg_brightness = np.mean(all_brightness)
    if avg_brightness < 80:
        print(f"   ❌ ЗАНАДТО ТЕМНО ({avg_brightness:.1f})")
        print(f"      → Додайте освітлення")
    elif avg_brightness > 160:
        print(f"   ❌ ЗАНАДТО ЯСКРАВО ({avg_brightness:.1f})")
        print(f"      → Зменште освітлення або експозицію камери")
    elif avg_brightness < 100 or avg_brightness > 140:
        print(f"   ⚠️  Освітлення не оптимальне ({avg_brightness:.1f})")
        print(f"      → Оптимум: 110-130")
    else:
        print(f"   ✅ Освітлення хороше ({avg_brightness:.1f})")

    # Аналіз контрасту
    avg_contrast = np.mean(all_contrast)
    if avg_contrast < 30:
        print(f"   ⚠️  Контраст низький ({avg_contrast:.1f})")
        print(f"      → Плоске освітлення, може погіршити розпізнавання")
    else:
        print(f"   ✅ Контраст достатній ({avg_contrast:.1f})")

    # Аналіз pose
    avg_pitch = abs(np.mean(all_pitch))
    avg_yaw = abs(np.mean(all_yaw))
    avg_roll = abs(np.mean(all_roll))

    print(f"\n   📐 Середні кути pose:")
    print(f"      Pitch: {np.mean(all_pitch):.1f}° (вгору-/вниз+)")
    print(f"      Yaw:   {np.mean(all_yaw):.1f}° (ліво-/право+)")
    print(f"      Roll:  {np.mean(all_roll):.1f}° (нахил)")

    if avg_pitch > 10:
        print(f"   ⚠️  Голова в середньому нахилена {'вгору' if np.mean(all_pitch) < 0 else 'вниз'}")
    if avg_yaw > 10:
        print(f"   ⚠️  Голова в середньому повернута {'вліво' if np.mean(all_yaw) < 0 else 'вправо'}")
    if avg_roll > 10:
        print(f"   ⚠️  Голова в середньому нахилена")

    print(f"{'=' * 100}")


def process_all_videos():
    """
    Обробляє всі відео з директорії
    """
    print("=" * 80)
    print("VIDEO FRAME EXTRACTOR - CLEAN FRAMES")
    print("=" * 80)
    print(f"⚙️  Налаштування:")
    print(f"   Save with BBox: {SAVE_WITH_BBOX}")
    print(f"   Save full frame: {SAVE_FULL_FRAME}")
    print("=" * 80)

    # Створюємо output директорію
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Ініціалізуємо детектор
    detector = initialize_detector()

    # Знаходимо всі відео файли
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_dir = Path(VIDEO_DIR)

    if not video_dir.exists():
        print(f"❌ Директорія не існує: {VIDEO_DIR}")
        return

    video_files = [f for f in video_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in video_extensions]

    if len(video_files) == 0:
        print(f"❌ Не знайдено відео файлів в: {VIDEO_DIR}")
        return

    print(f"\n📁 Знайдено відео файлів: {len(video_files)}")
    for video_file in video_files:
        print(f"   - {video_file.name}")

    # Обробляємо кожне відео
    for video_file in video_files:
        video_name = video_file.stem
        extract_best_frames_from_video(
            str(video_file),
            detector,
            OUTPUT_DIR,
            video_name
        )

    print(f"\n{'=' * 80}")
    print("✅ ВСІ ВІДЕО ОБРОБЛЕНО")
    print(f"{'=' * 80}")
    print(f"📂 Результати збережено в: {OUTPUT_DIR}")
    print(f"{'=' * 80}")


# ============================================================================
# ОСНОВНА ПРОГРАМА
# ============================================================================

if __name__ == "__main__":
    process_all_videos()