import os
import json
import cv2
from PIL import Image

CONFIDENCE_THRESHOLD = 0.5  # Порог уверенности для детекций
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov')  # Поддерживаемые видеоформаты


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

    # Инициализация видеопотока
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка открытия видео {video_path}")
        return

    # Получение префикса из названия видео для имен файлов
    video_prefix = os.path.splitext(os.path.basename(video_path))[0]

    # Получение размеров кадра видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    saved_crops = 0  # Счетчик успешных сохранений
    errors = 0  # Счетчик ошибок

    # Обработка каждого кадра из JSON-аннотаций
    for frame_key, frame_info in data.get('file', {}).items():
        try:
            # Извлечение номера кадра из названия файла в аннотациях
            frame_number = int(os.path.splitext(frame_info.split('_')[-1])[0])

            # Установка позиции воспроизведения видео
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                continue  # Пропуск битых кадров

            # Обработка всех детекций для текущего кадра
            for det_idx, detection in enumerate(data['detections'].get(frame_key, [])):
                # Фильтрация по порогу уверенности
                if detection.get('conf', 0) < CONFIDENCE_THRESHOLD:
                    continue

                try:
                    # Конвертация относительных координат в абсолютные
                    x = int(detection['bbox'][0] * frame_width)
                    y = int(detection['bbox'][1] * frame_height)
                    w = int(detection['bbox'][2] * frame_width)
                    h = int(detection['bbox'][3] * frame_height)

                    # Коррекция размеров bounding box для выхода за границы
                    w = min(w, frame_width - x)
                    h = min(h, frame_height - y)
                    if w <= 0 or h <= 0:
                        continue  # Пропуск некорректных размеров

                    # Вырезаем область интереса (ROI) из кадра
                    crop = frame[y:y + h, x:x + w]
                    if crop.size == 0:
                        continue  # Пропуск пустых кропов

                    # Формирование уникального имени файла с префиксом видео
                    output_filename = f"{video_prefix}_frame_{frame_number:06d}_det_{det_idx:03d}.jpg"
                    output_path = os.path.join(output_folder, output_filename)

                    # Конвертация цветового пространства и сохранение
                    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(output_path)

                    # Проверка успешности сохранения
                    if not os.path.exists(output_path):
                        print(f"Ошибка сохранения: {output_path}")
                        errors += 1
                        continue

                    saved_crops += 1
                except Exception as e:
                    errors += 1
                    print(f"Ошибка обработки детекции {det_idx}: {str(e)}")

        except Exception as e:
            errors += 1
            print(f"Ошибка обработки кадра {frame_key}: {str(e)}")

    # Завершение работы с видеофайлом
    cap.release()
    print(f"Итоги обработки: {saved_crops} сохранено, {errors} ошибок")


def process_all_jsons(json_root, video_root, output_root):
    """Обработка всех JSON с предварительным кешированием"""
    # Инициализация и построение кеша
    cache = VideoCache()
    cache.build(video_root)

    # Обработка JSON-файлов
    for root, _, files in os.walk(json_root):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                print(f"\n{'=' * 50}\nОбработка: {json_path}")
                process_json(json_path, cache, output_root)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    process_all_jsons(
        json_root=os.path.join(current_dir, "json_final"),
        video_root=os.path.join(current_dir, "all_videos"),
        output_root=os.path.join(current_dir, "results_crop")
    )
