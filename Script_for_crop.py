import os
import json
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

CONFIDENCE_THRESHOLD = 0.5  # Порог уверенности для детекций
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov')
MAX_WORKERS = os.cpu_count() // 2
MIN_INTERVAL_SECONDS = 0.5  # Минимальный интервал между кадрами в секундах


class VideoCache:
    """Кеш для хранения путей к видеофайлам Читаем сразу все видео и запоминаем их расположение"""

    def __init__(self):
        self.cache = {}

    def build(self, video_root):
        """Построение кеша путей к видео"""
        print(f"Начало построения кеша видео из {video_root}...")
        for root_dir, _, files in os.walk(video_root):
            for file in files:
                if file.lower().endswith(VIDEO_EXTENSIONS):
                    name = os.path.splitext(file)[0].lower()
                    if name not in self.cache:
                        self.cache[name] = os.path.join(root_dir, file)
        print(f"Кеш построен. Найдено видео: {len(self.cache)}")


def get_animal_class(json_path):
    """Извлекает название класса из структуры папок"""
    parent_dir = os.path.dirname(json_path)
    return os.path.basename(parent_dir).replace("_json", "")


def process_json(json_path, video_cache, output_root):
    """Обработка JSON с использованием кешированных видео"""
    total_saved = 0
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Ошибка чтения JSON {json_path}: {str(e)}")
        return

    animal_cls = get_animal_class(json_path)
    output_folder = os.path.join(output_root, animal_cls)
    os.makedirs(output_folder, exist_ok=True)

    json_basename = os.path.splitext(os.path.basename(json_path))[0].lower()
    video_path = video_cache.cache.get(json_basename)

    if not video_path:
        print(f"Видео для {json_basename} не найдено")
        return

    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        frame_data = []
        for frame_key, frame_info in data.get('file', {}).items():
            try:
                frame_number = int(os.path.splitext(frame_info.split('_')[-1])[0])
                if 0 <= frame_number < frame_count:
                    detections = data['detections'].get(frame_key, [])
                    if any(d.get('conf', 0) >= CONFIDENCE_THRESHOLD for d in detections):
                        frame_data.append((frame_number, frame_key))
            except:
                continue

        frame_data.sort()

        for frame_number, frame_key in frame_data:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                continue

            detections = [(i, d) for i, d in enumerate(data['detections'][frame_key])
                          if d.get('conf', 0) >= CONFIDENCE_THRESHOLD]

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(
                    lambda item: process_detection(
                        detection=item[1],
                        frame=frame,
                        frame_number=frame_number,
                        output_folder=output_folder,
                        video_name=video_name,
                        det_idx=item[0]
                    ),
                    detections
                ))
                total_saved += sum(results)

        print(f"\n{video_name}:")
        print(f"Всего кадров в видео: {frame_count}")
        print(f"Кадров с детекциями: {len(frame_data)}")
        print(f"Сохранено объектов: {total_saved}")

    except Exception as e:
        print(f"Ошибка: {str(e)}")
    finally:
        cap.release()


def process_detection(detection, frame, frame_number, output_folder, video_name, det_idx):
    try:
        h, w = frame.shape[:2]
        x = int(detection['bbox'][0] * w)
        y = int(detection['bbox'][1] * h)
        crop_w = min(int(detection['bbox'][2] * w), w - x)
        crop_h = min(int(detection['bbox'][3] * h), h - y)

        if crop_w <= 0 or crop_h <= 0:
            return 0

        crop = frame[y:y + crop_h, x:x + crop_w]
        if crop.size == 0:
            return 0

        output_filename = (
            f"{video_name}_"
            f"frame_{frame_number:08d}_"
            f"det_{det_idx:03d}_"
            f"conf_{detection['conf']:.2f}.jpg"
        )
        output_path = os.path.join(output_folder, output_filename)

        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(output_path)
        return 1

    except Exception as e:
        print(f"⚠Ошибка детекции: {str(e)}")
        return 0


def process_all_jsons(json_root, video_root, output_root):
    """Обработка всех JSON с предварительным кешированием"""
    # Инициализация и построение кеша
    cache = VideoCache()
    cache.build(video_root)

    json_files = []
    for root, _, files in os.walk(json_root):
        json_files.extend(os.path.join(root, f) for f in files if f.endswith('.json'))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(
            executor.map(
                lambda j: process_json(j, cache, output_root),
                json_files
            ),
            total=len(json_files),
            desc="Обработка JSON файлов"
        ))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    process_all_jsons(
        json_root=os.path.join(current_dir, "json_final"),
        video_root=os.path.join(current_dir, "all_videos"),
        output_root=os.path.join(current_dir, "results_crop")
    )
