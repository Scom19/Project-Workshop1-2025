import os
import json
import cv2
from PIL import Image

CONFIDENCE_THRESHOLD = 0.5
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov')


def get_animal_class(json_path):
    """Берём название класса из названия папки с json-ом"""
    parent_dir = os.path.dirname(json_path)
    return os.path.basename(parent_dir).replace("_json", "")


def find_video_file(json_basename, video_root):
    """Поиск видео по точному совпадению названия"""
    target_name = json_basename.lower()
    for root_dir, _, files in os.walk(video_root):  # Проход по всем видео в папке с видео
        for file in files:
            video_name = os.path.splitext(file)[0].lower()  # Название файла без расширения
            if video_name == target_name and file.lower().endswith(VIDEO_EXTENSIONS):
                return os.path.join(root_dir, file)
    return None


def process_json(json_path, video_root, output_root):
    """Обработка JSON и сохранение кропов"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Ошибка чтения JSON {json_path}: {str(e)}")
        return

    # Создание папки
    animal_cls = get_animal_class(json_path)
    json_basename = os.path.splitext(os.path.basename(json_path))[0]  # Название json-а без расширения
    output_folder = os.path.join(output_root, animal_cls, json_basename)  # Формируем финальный путь
    os.makedirs(output_folder, exist_ok=True)

    # Поиск видео
    video_path = find_video_file(json_basename, video_root)
    if not video_path:
        print(f"Видео для {json_basename} не найдено")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка открытия видео {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    saved_crops = 0
    errors = 0

    # Обработка кадров
    for frame_key, frame_info in data.get('file', {}).items():
        try:
            frame_number = int(os.path.splitext(frame_info.split('_')[-1])[0])  # Извлекаем номер кадра
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Устанавливаем позицию найденного кадра
            ret, frame = cap.read()  # Считываем кадр
            if not ret:
                continue

            # Обработка детекций
            for det_idx, detection in enumerate(data['detections'].get(frame_key, [])):  # Извлекаем детекции
                if detection.get('conf', 0) < CONFIDENCE_THRESHOLD:  # Порог уверенности
                    continue

                try:
                    # Расчет координат
                    x = int(detection['bbox'][0] * frame_width)
                    y = int(detection['bbox'][1] * frame_height)
                    w = int(detection['bbox'][2] * frame_width)
                    h = int(detection['bbox'][3] * frame_height)

                    # Проверка выхода за границы
                    if x + w > frame_width:
                        w = frame_width - x
                    if y + h > frame_height:
                        h = frame_height - y
                    if w <= 0 or h <= 0:
                        continue

                    # Вырезка кропа
                    crop = frame[y:y + h, x:x + w]
                    if crop.size == 0:
                        continue

                    output_path = os.path.join(
                        output_folder,
                        f"frame_{frame_number:06d}_det_{det_idx:03d}.jpg"
                    )
                    # Сохранение через PIL
                    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(output_path)
                    # Cохранен ли файл
                    if os.path.exists(output_path):
                        pass
                    else:
                        print(f"Ошибка: не удалось сохранить кроп по пути: {output_path}")
                        errors += 1
                        continue

                    saved_crops += 1
                except Exception as e:
                    errors += 1
                    print(f"Ошибка кропа {det_idx}: {str(e)}")

        except Exception as e:
            errors += 1
            print(f"Ошибка кадра {frame_key}: {str(e)}")

    cap.release()  # Закрываем видео
    print(f"Сохранено кропов: {saved_crops} | Ошибок: {errors}")


def process_all_jsons(json_root, video_root, output_root):
    """
    Обрабатываем все json-файлы в директории
    """
    for root, _, files in os.walk(json_root):  # Проходиммся по всем json-ам
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                print(f"\n{'=' * 50}\nОбработка: {json_path}")
                process_json(json_path, video_root, output_root)

#Запуск скрипта
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    process_all_jsons(
        json_root=os.path.join(current_dir, "json_final"),
        video_root=os.path.join(current_dir, "all_videos"),
        output_root=os.path.join(current_dir, "results_crop")
    )
