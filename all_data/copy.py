import os
import shutil
import glob

# Пути
IMAGES_SRC_DIR = r'C:\Users\outlo\Desktop\all_data\images'
LABELS_DIR     = r'C:\Users\outlo\Desktop\all_data\train\labels'
TEST_DIR       = r'C:\Users\outlo\Desktop\all_data\test' # Куда скидываем "лишние"

os.makedirs(TEST_DIR, exist_ok=True)

# Собираем имена всех файлов разметки (без расширения) в set для быстрого поиска
label_stems = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(LABELS_DIR, '*.txt'))}
print(f"Загружено имен разметок: {len(label_stems)}")

# Ищем все изображения
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
all_images = [f for f in os.listdir(IMAGES_SRC_DIR) if f.lower().endswith(image_extensions)]

moved = 0

for img_name in all_images:
    stem = os.path.splitext(img_name)[0]
    
    # Если имени картинки нет в списке разметок — это наш кандидат на тест
    if stem not in label_stems:
        src = os.path.join(IMAGES_SRC_DIR, img_name)
        dst = os.path.join(TEST_DIR, img_name)
        
        shutil.move(src, dst) # Используем move, чтобы убрать их из общей папки
        moved += 1

print(f"\nЗавершено!")
print(f"Перенесено в '{TEST_DIR}': {moved} изображений без разметки.")
