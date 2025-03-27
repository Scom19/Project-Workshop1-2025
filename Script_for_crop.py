import os
import json
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

CONFIDENCE_THRESHOLD = 0.5  # Порог уверенности для детекций
IOU_THRESHOLDS = {
    'Bison': 0.6,
    'Sus': 0.6,
    'Capreolus': 0.65,
    'Nyctereutes': 0.65,
}
DEFAULT_IOU_THRESHOLD = 0.7
MIN_SIZE_FOR = {
    'Bison': 0.06,
    'Sus': 0.06,
}
DEFAULT_MIN_SIZE_RATIO = 0.04
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov')
MAX_WORKERS = os.cpu_count() // 2
MIN_INTERVAL_SECONDS = 0.5  # Минимальный интервал между кадрами в секундах
MAX_HISTORY_SIZE = 150


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


class CropHistory:
    """Класс для отслеживания истории вырезанных объектов для исключения дубликатов"""
    def __init__(self):
        self.history = {}
        self.lock = threading.Lock()

    def add_crop(self, animal_class, x_center_rel, y_center_rel, crop_w_rel, crop_h_rel):
        # Сохранение параметров кропа с ограничением максимального размера истории
        with self.lock:
            if animal_class not in self.history:
                self.history[animal_class] = []

            self.history[animal_class].append({
                'x_center_rel': x_center_rel,
                'y_center_rel': y_center_rel,
                'crop_w_rel': crop_w_rel,
                'crop_h_rel': crop_h_rel
            })

            if len(self.history[animal_class]) > MAX_HISTORY_SIZE:
                self.history[animal_class].pop(0)

    def calculate_iou(self, box1, box2):
        """Вычисление метрики IoU для двух боксов"""
        box1_x1 = box1[0] - box1[2] / 2
        box1_y1 = box1[1] - box1[3] / 2
        box1_x2 = box1[0] + box1[2] / 2
        box1_y2 = box1[1] + box1[3] / 2

        box2_x1 = box2[0] - box2[2] / 2
        box2_y1 = box2[1] - box2[3] / 2
        box2_x2 = box2[0] + box2[2] / 2
        box2_y2 = box2[1] + box2[3] / 2

        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]

        return intersection_area / (area1 + area2 - intersection_area) if (area1 + area2 - intersection_area) > 0 else 0

    def is_new_crop(self, animal_class, current_box):
        # Сравнение с историей кропов для данного класса животных
        with self.lock:
            if animal_class not in self.history:
                return True

            iou_threshold = IOU_THRESHOLDS.get(animal_class, DEFAULT_IOU_THRESHOLD)

            for stored in self.history[animal_class]:
                stored_box = (
                    stored['x_center_rel'],
                    stored['y_center_rel'],
                    stored['crop_w_rel'],
                    stored['crop_h_rel']
                )
                iou = self.calculate_iou(current_box, stored_box)
                if iou > iou_threshold:
                    return False
            return True


def save_detection(detection, frame, output_dir, video_name,
                   frame_num, det_idx, animal_class, crop_history):
    """1. Преобразование относительных координат в абсолютные
    2. Проверка минимальных размеров и соотношения сторон
    3. Проверка уникальности через историю кропов
    4. Сохранение изображения при выполнении всех условий"""
    try:
        h, w = frame.shape[:2]

        x_center_rel = detection['bbox'][0] + detection['bbox'][2] / 2
        y_center_rel = detection['bbox'][1] + detection['bbox'][3] / 2
        crop_w_rel = detection['bbox'][2]
        crop_h_rel = detection['bbox'][3]

        if crop_w_rel <= 0 or crop_h_rel <= 0:
            return 0

        min_size_ratio = MIN_SIZE_FOR.get(animal_class, DEFAULT_MIN_SIZE_RATIO)
        if crop_w_rel < min_size_ratio or crop_h_rel < min_size_ratio:
            return 0

        x = int((x_center_rel - crop_w_rel / 2) * w)
        y = int((y_center_rel - crop_h_rel / 2) * h)
        crop_w = int(crop_w_rel * w)
        crop_h = int(crop_h_rel * h)
        if max(crop_w, crop_h) / min(crop_w, crop_h) >= 5:
            return 0
        x = max(0, x)
        y = max(0, y)
        crop_w = min(crop_w, w - x)
        crop_h = min(crop_h, h - y)

        if crop_w <= 0 or crop_h <= 0:
            return 0

        crop = frame[y:y + crop_h, x:x + crop_w]
        if crop.size == 0:
            return 0

        current_box = (x_center_rel, y_center_rel, crop_w_rel, crop_h_rel)
        is_new = crop_history.is_new_crop(animal_class, current_box)

        if is_new:
            filename = f"{video_name}_f{frame_num:08d}_d{det_idx:03d}_c{detection['conf']:.2f}.jpg"
            output_path = os.path.join(output_dir, filename)
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(output_path)
            crop_history.add_crop(
                animal_class,
                x_center_rel,
                y_center_rel,
                crop_w_rel,
                crop_h_rel
            )
            return 1
        return 0
    except Exception as e:
        print(f"Ошибка сохранения: {str(e)}")
        return 0


def process_json(json_path, video_cache, output_root, crop_history):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Ошибка чтения {json_path}: {str(e)}")
        return

    animal_class = os.path.basename(os.path.dirname(json_path)).replace("_json", "")
    output_dir = os.path.join(output_root, animal_class)
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(json_path))[0].lower()
    video_path = video_cache.cache.get(video_name)
    if not video_path:
        print(f"Видео для {video_name} не найдено")
        return

    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        min_frame_interval = max(1, int(fps * MIN_INTERVAL_SECONDS))
        candidates = []

        for frame_key, frame_info in data.get('file', {}).items():
            try:
                frame_num = int(os.path.splitext(frame_info.split('_')[-1])[0])
                if 0 <= frame_num < frame_count:
                    detections = [d for d in data['detections'].get(frame_key, [])
                                  if d.get('conf', 0) >= CONFIDENCE_THRESHOLD]
                    if detections:
                        candidates.append((frame_num, frame_key))
            except:
                continue

        total_saved = 0
        for fn, fk in candidates[::min_frame_interval]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                continue

            detections = [d for d in data['detections'].get(fk, [])
                          if d.get('conf', 0) >= CONFIDENCE_THRESHOLD]

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for idx, det in enumerate(detections):
                    futures.append(
                        executor.submit(
                            save_detection,
                            det,
                            frame,
                            output_dir,
                            video_name,
                            fn,
                            idx,
                            animal_class,
                            crop_history
                        )
                    )
                total_saved += sum(f.result() for f in futures)

        print(f"\n{video_name}: Сохранено {total_saved} объектов")

    except Exception as e:
        print(f"Ошибка: {str(e)}")
    finally:
        cap.release()


def process_all_jsons(json_root, video_root, output_root):
    """Обработка всех JSON с предварительным кешированием"""
    # Инициализация и построение кеша
    cache = VideoCache()
    cache.build(video_root)
    crop_history = CropHistory()

    json_files = []
    for root, _, files in os.walk(json_root):
        json_files.extend(os.path.join(root, f) for f in files if f.endswith('.json'))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [executor.submit(process_json, jp, cache, output_root, crop_history)
                 for jp in json_files]

        for _ in tqdm(as_completed(tasks), total=len(tasks), desc="Обработка JSON файлов"):
            pass


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    process_all_jsons(
        json_root=os.path.join(current_dir, "json_final"),
        video_root=os.path.join(current_dir, "all_videos"),
        output_root=os.path.join(current_dir, "results_crop")
    )
