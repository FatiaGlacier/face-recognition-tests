import cv2
import numpy as np
from pathlib import Path

from deepface import DeepFace
from mtcnn import MTCNN
import os
from tqdm import tqdm
import time

from multiprocessing import Process, Queue, cpu_count, Manager
import queue

# ============================================================================
# КОНФІГУРАЦІЯ
# ============================================================================

VIDEO_DIR = "D:\\Projects\\python-opencv-test\\photos\\frames_processing_test\\vid"
OUTPUT_DIR = "D:\\Projects\\python-opencv-test\\photos\\frames_processing_test\\frames\\"
DOC_DIR = "D:\\Projects\\python-opencv-test\\photos\\frames_processing_test\\doc\\"

# Параметри
FRAME_SKIP = 2  # Обробляємо кожен N-й фрейм (для швидкості)
TOP_N_FRAMES = 7  # Скільки найкращих фреймів зберегти
MIN_FACE_SIZE = 100  # Мінімальний розмір обличчя (пікселі)
SAVE_WITH_BBOX = False  # ⬅️ FALSE = чисті фрейми, TRUE = з зеленим квадратом
SAVE_FULL_FRAME = True  # ⬅️ TRUE = весь кадр, FALSE = тільки обличчя (cropped)

# Пороги для pose (положення обличчя)
MAX_PITCH = 20#15  # ±15° вгору/вниз (менше = строгіше)
MAX_YAW = 20#20  # ±20° вліво/вправо
MAX_ROLL = 20#15  # ±15° нахил голови

NUM_PROCESSES = cpu_count() - 1  # Залишаємо 1 ядро системі
QUEUE_MAXSIZE = 50  # Розмір черги для кадрів

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

def extract_frames(video_path, video_name):
    print(f"\n{'=' * 80}")
    print(f"📹 Діставання фреймів з відео: {video_name}")
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

    # Збір даних
    frames = []
    frame_count = 0

    print(f"\n🔍 Читання фреймів...")

    pbar = tqdm(total=total_frames, desc="Читання фреймів")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)
        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue

        frames.append(frame)
        frame_count += 1

    return frames, frame_count

def save_frames(frames_data, output_folder, video_name, ):
    print(f"\n📸 Збереження {len(frames_data)} найкращих фреймів:")
    print(f"{'=' * 110}")
    print(
        f"{'Rank':<6} {'Frame#':<10} {'Quality':<10} {'Sharp':<8} {'Pose':<8} {'Pitch':<8} {'Yaw':<8} {'Roll':<8} {'Conf':<8}")
    print(f"{'=' * 110}")

    for rank, frame_info in enumerate(frames_data, 1):
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
        output_path = os.path.join(output_folder, output_filename)

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
    print(f"✅ Фрейми збережено в: {output_folder}")

DIR_FRAMES = None


def process_worker(frame_queue, result_queue, worker_id):
    """
    Обробляє один chunk фреймів з залоченим прогрес-баром
    """
    detector = MTCNN()

    processed_count = 0

    while True:
        try:
            # Отримуємо кадр з черги (timeout щоб не зависати)
            item = frame_queue.get(timeout=1)

            # None = сигнал завершення
            if item is None:
                break

            frame_number, frame = item

            # Детекція облич
            detections = detector.detect_faces(frame)

            if len(detections) == 0:
                processed_count += 1
                continue

            # Вибираємо найбільше обличчя
            largest_face = max(detections, key=lambda d: d['box'][2] * d['box'][3])
            face_box = largest_face['box']
            x, y, w, h = face_box

            # Фільтр по розміру
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                processed_count += 1
                continue

            # Обчислюємо якість
            quality, details = calculate_quality_score(frame, largest_face)

            # Фільтр поpose
            if not details['is_frontal']:
                processed_count += 1
                continue

            # Відправляємо результат
            result_queue.put({
                'frame_number': frame_number,
                'frame': frame.copy(),
                'face_box': face_box,
                'quality_score': quality,
                'sharpness': details['sharpness'],
                'brightness': details['brightness'],
                'contrast': details['contrast'],
                'pitch': details['pitch'],
                'yaw': details['yaw'],
                'roll': details['roll'],
                'confidence': largest_face['confidence'],
                'details': details
            })

            processed_count += 1

        except queue.Empty:
            # Черга порожня, чекаємо ще
            continue
        except Exception as e:
            print(f"❌ Worker {worker_id} помилка: {e}")
            break

    result_queue.put(None)
    print(f"✅ Worker {worker_id} завершив роботу. Оброблено: {processed_count}")


def result_collector(result_queue, results_list, num_workers):
    """
    Збирає результати у фоні
    """
    completed = 0
    while completed < num_workers:
        try:
            result = result_queue.get(timeout=1)
            if result is None:  # Сигнал завершення від worker
                completed += 1
            else:
                results_list.append(result)
        except queue.Empty:
            continue

def process_frames_multiprocessing(frames, video_name):
    print(f"\n{'=' * 80}")
    print(f"📹 Обробка фреймів (MULTITHREADED): {video_name}")
    print(f"{'=' * 80}")
    print(f"   Всього фреймів: {len(frames)}")
    print(f"   Процесів: {NUM_PROCESSES}")

    start_time = time.time()

    #Queue creation
    frame_queue = Queue(maxsize=QUEUE_MAXSIZE)
    result_queue = Queue()

    # Використовуємо Manager для shared list
    manager = Manager()
    results_list = manager.list()

    #Turning on workers
    processes = []
    for i in range(NUM_PROCESSES):
        p = Process(target=process_worker, args=(frame_queue, result_queue, i))
        p.start()
        processes.append(p)

    # Запускаємо колектор
    collector = Process(target=result_collector, args=(result_queue, results_list, NUM_PROCESSES))
    collector.start()

    print(f"\n🚀 Запущено {NUM_PROCESSES} процесів...")

    print(f"📤 Заповнення черги кадрами...")
    for idx, frame in enumerate(tqdm(frames, desc="Додавання кадрів")):
        frame_queue.put((idx, frame))

    for _ in range(NUM_PROCESSES):
        frame_queue.put(None)

    print(f"✅ Всі кадри додано в чергу")

    print(f"📥 Збір результатів...")
    # Чекаємо завершення workers
    for p in processes:
        p.join()

    # Чекаємо завершення колектора
    collector.join()

    # Конвертуємо Manager.list → звичайний list
    all_results = list(results_list)

    processing_time = time.time() - start_time

    print(f"\n✅ Обробка завершена за {processing_time:.2f} секунд")
    print(f"   Швидкість: {len(frames) / processing_time:.1f} фреймів/сек")
    print(f"   Знайдено фронтальних облич: {len(all_results)}")

    if len(all_results) == 0:
        print("❌ Не знайдено жодного ФРОНТАЛЬНОГО обличчя у відео!")
        print(f"⚠️  Спробуй збільшити пороги: MAX_PITCH, MAX_YAW, MAX_ROLL")
        return []

    return all_results


def process_all_videos_multiprocessing():
    """
    Обробляє всі відео з директорії (з таймінгами)
    """
    print("=" * 80)
    print("VIDEO FRAME EXTRACTOR - MULTIPROCESSING MODE")
    print("=" * 80)
    print(f"⚙️  Налаштування:")
    print(f"   Save with BBox: {SAVE_WITH_BBOX}")
    print(f"   Save full frame: {SAVE_FULL_FRAME}")
    print(f"   Processes: {NUM_PROCESSES}")
    print(f"   Queue size: {QUEUE_MAXSIZE}")
    print("=" * 80)

    # Створюємо output директорію
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Ініціалізуємо детектор (РАЗ!)
    print("\n🔧 Ініціалізація MTCNN...")
    init_start = time.time()
    #detector = initialize_detector()
    init_time = time.time() - init_start
    print(f"✅ MTCNN готовий ({init_time:.2f} сек)")

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
    total_start_time = time.time()
    video_name = None
    video_names = []
    all_dirs = []

    for video_file in video_files:
        video_name = video_file.stem
        video_names.append(video_name)
        video_start_time = time.time()

        # ============== ФАЗА 1: ЧИТАННЯ ==============
        print(f"\n{'=' * 80}")
        print(f"📼 ВІДЕО: {video_name}")
        print(f"{'=' * 80}")

        read_start = time.time()
        frames, count = extract_frames(str(video_file), video_name)
        read_time = time.time() - read_start

        print(f"\n⏱️  Читання: {read_time:.2f} сек")
        print(f"   Прочитано фреймів: {len(frames)}")

        # ============== ФАЗА 2: ОБРОБКА ==============
        process_start = time.time()
        frame_data = process_frames_multiprocessing(frames, video_name)
        process_time = time.time() - process_start

        print(f"\n⏱️  Обробка: {process_time:.2f} сек")

        if len(frame_data) == 0:
            print("⚠️  Пропускаємо відео - немає результатів")
            continue

        # ============== ФАЗА 3: СОРТУВАННЯ ==============
        sort_start = time.time()
        frame_data.sort(key=lambda x: x['quality_score'], reverse=True)
        best_frames = frame_data[:TOP_N_FRAMES]
        sort_time = time.time() - sort_start

        print(f"\n⏱️  Сортування: {sort_time:.3f} сек")
        print(f"   Топ-{TOP_N_FRAMES} вибрано")

        # ============== ФАЗА 4: ЗБЕРЕЖЕННЯ ==============
        save_start = time.time()
        video_output_folder = os.path.join(OUTPUT_DIR, video_name)
        os.makedirs(video_output_folder, exist_ok=True)
        save_frames(best_frames, video_output_folder, video_name)
        save_time = time.time() - save_start

        print(f"\n⏱️  Збереження: {save_time:.2f} сек")

        # ============== ПІДСУМОК ПО ВІДЕО ==============
        video_total_time = time.time() - video_start_time

        all_dirs.append(video_output_folder)
        print(f"\n{'=' * 80}")
        print(f"⏱️  ПІДСУМОК ПО ВІДЕО '{video_name}':")
        print(f"{'=' * 80}")
        print(f"   Читання:     {read_time:>8.2f} сек ({read_time / video_total_time * 100:>5.1f}%)")
        print(f"   Обробка:     {process_time:>8.2f} сек ({process_time / video_total_time * 100:>5.1f}%)")
        print(f"   Сортування:  {sort_time:>8.2f} сек ({sort_time / video_total_time * 100:>5.1f}%)")
        print(f"   Збереження:  {save_time:>8.2f} сек ({save_time / video_total_time * 100:>5.1f}%)")
        print(f"   {'─' * 40}")
        print(f"   ЗАГАЛЬНО:    {video_total_time:>8.2f} сек")
        print(f"{'=' * 80}")

    # ============== ЗАГАЛЬНИЙ ПІДСУМОК ==============
    total_time = time.time() - total_start_time

    print(f"\n{'=' * 80}")
    print("✅ ВСІ ВІДЕО ОБРОБЛЕНО")
    print(f"{'=' * 80}")
    print(f"⏱️  Загальний час: {total_time:.2f} секунд")
    print(f"📂 Результати збережено в: {OUTPUT_DIR}")
    print(f"{'=' * 80}")

    #frames_dir = os.path.join(OUTPUT_DIR, video_name) if video_name else OUTPUT_DIR
    return all_dirs, video_names


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

def format_result(src, dst, cosine_percent, cosine_raw):
    """Форматує результат порівняння"""
    return (
        f"{src} vs {dst}: "
        f"Cosine={float(cosine_percent):.2f}%, "
        f"Cosine raw={float(cosine_raw):.4f} "
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

        result_str = format_result(doc_name, user_name, cos, raw)
        print(result_str)
        results.append((user_name, cos))

    return results

def recognition(doc_dir, frame_dirs, video_names):
    print("=" * 80)
    print("ЗАВАНТАЖЕННЯ ФОТОГРАФІЙ")
    print("=" * 80)

    # Завантажуємо документи
    docs_images = load_images_from_folder(doc_dir)
    print(f"\n📄 Знайдено документів: {len(docs_images)}")
    for name in docs_images.keys():
        print(f"  - {name}")

    vid_pointer = 0
    for dir in frame_dirs:

        # Завантажуємо фото (гарне освітлення)
        frames = load_images_from_folder(dir)
        print(f"\n📸 Знайдено фото {video_names[vid_pointer]}: {len(frames)}")
        vid_pointer+=1
        for name in frames.keys():
            print(f"  - {name}")

        print("\n" + "=" * 80)
        print("ТЕСТ: ПОРІВНЯННЯ З ArcFace")
        print("=" * 80)
        total_start_time = time.time()
        print("\n🔄 Обробка зображень з ArcFace...")
        doc_encodings = get_all_encodings(docs_images, "ArcFace")
        frames_encodings = get_all_encodings(frames, "ArcFace")

        for doc_name, doc_enc in doc_encodings.items():
            compare_and_print(doc_name, doc_enc, frames_encodings, "Frames", "ArcFace")

        total_time = time.time() - total_start_time
        print(f"{'=' * 80}")
        print(f"⏱️  Загальний час: {total_time:.2f} секунд")
        print(f"{'=' * 80}")


def get_frames_and_compare():
    FRAME_DIRs, video_names = process_all_videos_multiprocessing()
    recognition(DOC_DIR, FRAME_DIRs, video_names)

# ============================================================================
# ОСНОВНА ПРОГРАМА
# ============================================================================

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    get_frames_and_compare()