import cv2
from pathlib import Path

from tqdm import tqdm

# ============================================================================
# КОНФІГУРАЦІЯ
# ============================================================================

BASE_DIR = "D:\\Projects\\python-opencv-test\\photos\\facial_rotation_test\\big_video_test\\"

ME_BASE = BASE_DIR + "me\\"

PITCH_VIDEO_PATH = ME_BASE + "pitch\\pitch.mp4"
PITCH_OUTPUT_PATH = ME_BASE + "pitch\\frames\\"

YAW_VIDEO_PATH = ME_BASE + "yaw\\yaw.mp4"
YAW_OUTPUT_PATH = ME_BASE + "yaw\\frames\\"

ROLL_VIDEO_PATH = ME_BASE + "roll\\roll.mp4"
ROLL_OUTPUT_PATH = ME_BASE + "roll\\frames\\"

def extract_frames(video_path, output_path, step=1):
    video_name = Path(video_path).stem

    print(f"\n{'=' * 80}")
    print(f"📹 Processing video: {video_name}")
    print(f"{'=' * 80}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Can not open video: {video_path}")
        return

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Info about video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Info:")
    print(f"Size: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} секунд")
    print(f"Frames: {total_frames}")

    # Збір даних
    frame_index = 0
    saved_count = 0

    print(f"\n Processing frames...")

    pbar = tqdm(total=total_frames, desc="Processing frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step == 0:
            filename = output_dir / f"{video_name}_{frame_index:05d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved_count += 1

        frame_index += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"Saved {saved_count} frames to: {output_dir}")
    return saved_count

extract_frames(PITCH_VIDEO_PATH, PITCH_OUTPUT_PATH)
extract_frames(YAW_VIDEO_PATH, YAW_OUTPUT_PATH)
extract_frames(ROLL_VIDEO_PATH, ROLL_OUTPUT_PATH)