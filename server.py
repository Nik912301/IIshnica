# server.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import json
import traceback
from pathlib import Path
from typing import Any, Optional, Dict, List

# Обработка изображений и данные
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from skimage import measure
from scipy.spatial import cKDTree

# ============================================================================
# УТИЛИТА: Конвертация NumPy типов в JSON-сериализуемые
# ============================================================================

def to_json_serializable(obj: Any) -> Any:
    """
    Рекурсивно конвертирует объекты NumPy и другие не-JSON-сериализуемые типы
    в нативные типы Python для корректной отправки через FastAPI.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [to_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Для объектов с __dict__ (но не самих numpy типов)
        try:
            return to_json_serializable(vars(obj))
        except:
            return str(obj)
    else:
        return obj

# ============================================================================
# НАСТРОЙКА FASTAPI
# ============================================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print(f"📁 BASE_DIR: {BASE_DIR}")

app = FastAPI(title="Plant Graph Analyzer 2026")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создаём необходимые папки
for folder in ["uploads", "results", "debug_stages", "temp_calib", "calib_visualizations"]:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

# Монтируем статические файлы
app.mount("/results", StaticFiles(directory=os.path.join(BASE_DIR, "results")), name="results")
app.mount("/debug", StaticFiles(directory=os.path.join(BASE_DIR, "debug_stages")), name="debug")
app.mount("/calib_visualizations", StaticFiles(directory=os.path.join(BASE_DIR, "calib_visualizations")), name="calib_vis")
app.mount("/temp_calib", StaticFiles(directory=os.path.join(BASE_DIR, "temp_calib")), name="temp_calib")

# ============================================================================
# КОНФИГУРАЦИЯ ПО УМОЛЧАНИЮ
# ============================================================================

DEFAULT_CONFIG = {
    # Пути
    "MODEL_PATH": os.path.join(BASE_DIR, "yolo_weights", "plants_optimized_seg", "weights", "best_h.pt"),
    "DEBUG_DIR": os.path.join(BASE_DIR, "debug_stages"),
    
    # Классы YOLO
    "ROOT_CLASSES": {'root', 'корень'},
    "STEM_CLASSES": {'stem', 'стебель'},
    "LEAF_CLASSES": {'leaf', 'лист'},
    
    # YOLO параметры
    "YOLO_CONF_ROOT": 0.001,
    "YOLO_CONF_STEM": 0.10,
    "YOLO_CONF_LEAF": 0.30,
    "YOLO_IOU": 0.50,
    "YOLO_IOU_ROOT": 0.80,
    "YOLO_IMG_SIZE": 1280,
    "YOLO_MAX_DET": 1000,
    "YOLO_AUGMENT": "light",
    
    # Изображение
    "IMG_MAX_WIDTH": 2048,
    "IMG_MAX_HEIGHT": 2048,
    "IMG_INTERPOLATION": "cubic",
    
    # Предобработка
    "PREPROC_BLUR": 0,
    "PREPROC_CONTRAST": 2.0,
    "PREPROC_DENOISE": 0,
    "PREPROC_ADAPTIVE": "gaussian",
    
    # Скелетизация
    "SKELETON_METHOD": "zhang",
    "SKELETON_ITER": 100,
    
    # Граф
    "GRAPH_CONNECT_RADIUS": 1.7,
    "GRAPH_MAX_NODES": 38000,
    "REMOVE_SMALL_COMP_SIZE": 30,
    "GRAPH_PRUNE_BRANCHES": 5,
    
    # Соединение компонент
    "ROOT_CONNECT_MAX_DIST": 50,
    "ROOT_CONNECT_MIN_DIST": 4.0,
    "ROOT_ANCHOR_MAX_DIST": 50,
    "ROOT_MIN_SIZE_RATIO": 0.1,
    "STEM_LEAF_CONNECT_MAX_DIST": 5,
    "STEM_LEAF_CONNECT_MIN_DIST": 1.0,
    "STEM_LEAF_ANGLE_THRESH": 45,
    
    # Морфология
    "COMBINED_MIN_SIZE": 60,
    "MORPH_KERNEL_SIZE": 3,
    "MORPH_CLOSE_SIZE": 0,
    "MORPH_ITER": 1,
    "NOISE_REMOVE_AREA": 20,
    "NOISE_REMOVE_ECCENTRICITY": 0.99,
    
    # Отрисовка
    "EDGE_THICKNESS": 2,
    "NODE_END_RADIUS": 5,
    "NODE_MID_RADIUS": 2,
    "DRAW_NODE_LABELS": "degree",
    "FONT_SIZE": 12,
    "DRAW_BOUNDING_BOX": "yolo",
    "OUTPUT_FORMAT": "png",
    
    # Расширенные
    "MAX_WORKERS": 4,
    "BATCH_SIZE": 1,
    "USE_GPU": "auto",
    "DEBUG_MODE": "off",
    "SAVE_INTERMEDIATE": "all",
    "LOG_LEVEL": "debug",
    "EXPORT_FORMAT": "json",
    "EXPORT_GRAPHML": False,
    "EXPORT_COORDS": True,
    
    # Калибровка
    "CHESSBOARD_SIZE": (7, 4),
    "SQUARE_SIZE_MM": 25.0,
    "CALIB_FLAGS": cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5,
    "CALIB_ALPHA": 1.0,
    
    # Цвета
    "COLORS": {
        'root': {'edge': (255, 0, 180), 'node_end': (0, 0, 255), 'node_mid': (180, 0, 255)},
        'stem': {'edge': (0, 220, 0), 'node_end': (0, 100, 0), 'node_mid': (100, 255, 100)},
        'leaf': {'edge': (0, 180, 255), 'node_end': (0, 80, 180), 'node_mid': (100, 220, 255)},
    },
    
    "SAVE_DEBUG_STAGES": True,
}

# ============================================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================================

model = None
camera_matrix = None
dist_coeffs = None
pixels_per_mm = None

# Загрузка модели
try:
    from ultralytics import YOLO
    model_path = DEFAULT_CONFIG["MODEL_PATH"]
    if os.path.isfile(model_path):
        device = "0" if DEFAULT_CONFIG["USE_GPU"] != "false" else "cpu"
        model = YOLO(model_path)
        print(f"✅ Модель загружена: {model_path} (device={device})")
    else:
        print(f"⚠️ Модель не найдена: {model_path}")
except ImportError:
    print("⚠️ ultralytics не установлен — YOLO отключён")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")

# Загрузка калибровки
def try_load_calibration():
    global camera_matrix, dist_coeffs, pixels_per_mm
    calib_file = os.path.join(BASE_DIR, "calibration_data_single.npz")
    if os.path.exists(calib_file):
        try:
            data = np.load(calib_file)
            camera_matrix = data.get('camera_matrix')
            dist_coeffs = data.get('dist_coeffs')
            pixels_per_mm = data.get('pixels_per_mm', None)
            print(f"✅ Калибровка загружена: {pixels_per_mm:.3f} px/mm" if pixels_per_mm else "✅ Калибровка загружена")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки калибровки: {e}")

try_load_calibration()

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def get_interpolation_method(name: str) -> int:
    """Возвращает константу интерполяции OpenCV по имени."""
    methods = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4,
        'nearest': cv2.INTER_NEAREST,
    }
    return methods.get(name, cv2.INTER_CUBIC)

def preprocess_image(img: np.ndarray, config: dict) -> np.ndarray:
    """Предобработка изображения согласно параметрам."""
    h, w = img.shape[:2]
    max_w = config.get("IMG_MAX_WIDTH", 2048)
    max_h = config.get("IMG_MAX_HEIGHT", 2048)
    
    # Ресайз если нужно
    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        interp = get_interpolation_method(config.get("IMG_INTERPOLATION", "cubic"))
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    
    # Размытие
    blur_k = config.get("PREPROC_BLUR", 0)
    if blur_k > 1 and blur_k % 2 == 1:
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    
    # CLAHE для контраста
    contrast = config.get("PREPROC_CONTRAST", 2.0)
    if contrast > 1.0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Denoise
    denoise_h = config.get("PREPROC_DENOISE", 0)
    if denoise_h > 0:
        img = cv2.fastNlMeansDenoisingColored(img, None, denoise_h, denoise_h, 7, 21)
    
    return img

def build_graph(skeleton: np.ndarray, config: dict) -> tuple:
    """Строит граф из скелета."""
    yx = np.column_stack(np.where(skeleton))
    if len(yx) == 0:
        return nx.Graph(), skeleton
    
    # Downsampling если узлов слишком много
    max_nodes = config.get("GRAPH_MAX_NODES", 38000)
    if len(yx) > max_nodes:
        step = max(1, len(yx) // max_nodes)
        yx = yx[::step]
    
    if len(yx) == 0:
        return nx.Graph(), skeleton
    
    # Построение графа через KD-Tree
    tree = cKDTree(yx)
    radius = config.get("GRAPH_CONNECT_RADIUS", 1.7)
    pairs = tree.query_pairs(r=radius)
    
    G = nx.Graph()
    node_map = {i: (int(yx[i][1]), int(yx[i][0])) for i in range(len(yx))}
    
    for xy in node_map.values():
        G.add_node(xy)
    
    for i, j in pairs:
        n1, n2 = node_map[i], node_map[j]
        dist = float(np.linalg.norm(yx[i] - yx[j]))
        G.add_edge(n1, n2, weight=dist)
    
    return G, skeleton

def connect_nearby_components(G: nx.Graph, max_dist: float, min_dist: float) -> nx.Graph:
    """Соединяет близкие компоненты графа."""
    if len(G) == 0 or nx.is_connected(G):
        return G
    
    components = list(nx.connected_components(G))
    if len(components) < 2:
        return G
    
    # Центроиды компонент
    centroids = []
    for comp in components:
        nodes = np.array(list(comp))
        centroids.append(nodes.mean(axis=0))
    centroids = np.array(centroids)
    
    # Поиск пар для соединения
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=max_dist)
    
    for i, j in pairs:
        comp_i = list(components[i])
        comp_j = list(components[j])
        
        # Находим ближайшие узлы между компонентами
        min_d = float('inf')
        best_pair = None
        for ni in comp_i:
            for nj in comp_j:
                d = np.linalg.norm(np.array(ni) - np.array(nj))
                if min_d < d < max_dist and d >= min_dist:
                    min_d = d
                    best_pair = (ni, nj)
        
        if best_pair:
            G.add_edge(best_pair[0], best_pair[1], weight=min_d)
    
    return G

def prune_small_branches(G: nx.Graph, min_length: int) -> nx.Graph:
    """Удаляет короткие ветви графа."""
    if min_length <= 0:
        return G
    
    # Находим листья
    leaves = [n for n, d in G.degree() if d == 1]
    
    # Для каждого листа идём пока не встретим узел степени != 2
    to_remove = set()
    for leaf in leaves:
        path = [leaf]
        current = leaf
        while True:
            neighbors = list(G.neighbors(current))
            # Убираем предыдущий узел из соседей
            if len(path) > 1:
                neighbors = [n for n in neighbors if n != path[-2]]
            
            if len(neighbors) == 0:
                break
            if len(neighbors) > 1 or G.degree(neighbors[0]) != 2:
                # Встретили разветвление или конец
                if len(path) < min_length:
                    to_remove.update(path)
                break
            
            path.append(neighbors[0])
            current = neighbors[0]
    
    if to_remove:
        G = G.copy()
        G.remove_nodes_from(to_remove)
    
    return G

def save_debug_image(img: np.ndarray, prefix: str, config: dict):
    """Сохраняет отладочное изображение."""
    if not config.get("SAVE_DEBUG_STAGES", True):
        return
    path = os.path.join(config["DEBUG_DIR"], f"{prefix}_{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(path, img)
    print(f"🔍 Debug saved: {path}")

# ============================================================================
# КАЛИБРОВКА
# ============================================================================

def calibrate_single_image(image_path: str, square_size_mm: float = 25.0, 
                          chessboard_size: tuple = (7, 4), calib_flags: str = "fix_k3_k4_k5") -> dict:
    """Калибровка по одному изображению шахматки."""
    global camera_matrix, dist_coeffs, pixels_per_mm

    if not os.path.isfile(image_path):
        return {"status": "error", "message": f"Файл не найден: {image_path}"}

    img = cv2.imread(image_path)
    if img is None:
        return {"status": "error", "message": f"Не удалось прочитать: {image_path}"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    CB = chessboard_size

    ret, corners = cv2.findChessboardCorners(
        gray, CB,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not ret:
        return {"status": "error", "message": f"Шахматка не найдена ({CB[0]}×{CB[1]} углов)"}

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 3D точки
    objp = np.zeros((CB[0] * CB[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CB[0], 0:CB[1]].T.reshape(-1, 2) * square_size_mm

    # Флаги калибровки
    flag_map = {
        "fix_k3_k4_k5": cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5,
        "fix_k3": cv2.CALIB_FIX_K3,
        "none": 0,
        "rational": cv2.CALIB_RATIONAL_MODEL,
    }
    flags = flag_map.get(calib_flags, cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5)

    ret_calib, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [corners], gray.shape[::-1], None, None, flags=flags
    )

    if not ret_calib:
        return {"status": "error", "message": "Калибровка не удалась"}

    # Ошибка репроекции
    mean_error = 0
    for i in range(len([objp])):
        imgpoints2, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(corners, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    rms_error = mean_error

    # Масштаб
    pixel_distances = []
    for row in range(CB[1]):
        for col in range(CB[0] - 1):
            idx = row * CB[0] + col
            p1, p2 = corners[idx][0], corners[idx + 1][0]
            pixel_distances.append(np.linalg.norm(p1 - p2))
    for col in range(CB[0]):
        for row in range(CB[1] - 1):
            idx = row * CB[0] + col
            p1, p2 = corners[idx][0], corners[idx + CB[0]][0]
            pixel_distances.append(np.linalg.norm(p1 - p2))
    
    avg_px = np.mean(pixel_distances) if pixel_distances else 1.0
    pixels_per_mm_local = avg_px / square_size_mm
    mm_per_pixel = 1.0 / pixels_per_mm_local if pixels_per_mm_local > 0 else 0

    # Сохранение глобально
    camera_matrix = mtx
    dist_coeffs = dist
    pixels_per_mm = pixels_per_mm_local

    # Сохранение на диск
    calib_path = os.path.join(BASE_DIR, "calibration_data_single.npz")
    np.savez(calib_path, camera_matrix=mtx, dist_coeffs=dist, 
             rms_error=float(rms_error), pixels_per_mm=float(pixels_per_mm_local),
             square_size_mm=square_size_mm)

    return to_json_serializable({
        "status": "ok",
        "message": "Калибровка успешна",
        "rms_error": float(rms_error),
        "pixels_per_mm": float(pixels_per_mm_local),
        "mm_per_pixel": float(mm_per_pixel),
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist()
    })

def visualize_calibration_single(
    image_path: str,
    calib_data: dict,
    output_dir: str = None,
    square_size_mm: float = 25.0,
    chessboard_size: tuple = (7, 4),
    alpha: float = 1.0,
    figsize: tuple = (14, 6),
    save_result: bool = True
) -> dict:
    """
    Создаёт визуализацию калибровки: исходное изображение с углами + исправленное.
    
    Args:
        image_path: Путь к исходному изображению шахматки
        calib_data: Словарь с калибровочными данными:
            - camera_matrix: матрица камеры 3×3
            - dist_coeffs: коэффициенты дисторсии
            - pixels_per_mm: масштаб (пиксели на мм)
            - rms_error: ошибка репроекции
        output_dir: Директория для сохранения результата
        square_size_mm: Размер одного квадрата шахматки в мм
        chessboard_size: Количество ВНУТРЕННИХ углов (колонки, строки)
        alpha: Параметр crop при устранении дисторсии (0–1)
        figsize: Размер фигуры matplotlib в дюймах
        save_result: Сохранять ли результат в файл
    
    Returns:
        dict со статусом, путём к файлу и метаданными (все значения JSON-сериализуемые)
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # 🔹 1. Проверка входных данных
    if not image_path or not os.path.isfile(image_path):
        print(f"❌ visualize_calibration_single: файл не найден: {image_path}")
        return to_json_serializable({
            "status": "error",
            "message": f"Файл не найден: {image_path}",
            "saved_path": None
        })
    
    print(f"📷 Визуализация калибровки: {image_path}")
    
    # 🔹 2. Извлечение калибровочных данных
    camera_matrix = calib_data.get('camera_matrix')
    dist_coeffs = calib_data.get('dist_coeffs')
    pixels_per_mm = calib_data.get('pixels_per_mm', 0)
    rms_error = calib_data.get('rms_error', 0)
    
    if camera_matrix is None or dist_coeffs is None:
        return to_json_serializable({
            "status": "error",
            "message": "Отсутствуют camera_matrix или dist_coeffs",
            "saved_path": None
        })
    
    # Конвертация в numpy массивы
    camera_matrix = np.array(camera_matrix, dtype=np.float32)
    dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
    
    # 🔹 3. Чтение и подготовка изображения
    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        return to_json_serializable({
            "status": "error",
            "message": f"Не удалось прочитать изображение: {image_path}",
            "saved_path": None
        })
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    CB = chessboard_size  # (внутренние углы: колонки, строки)
    
    # 🔹 4. Поиск углов для отрисовки
    ret, corners = cv2.findChessboardCorners(
        gray, CB,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    img_with_corners = img.copy()
    
    if ret:
        # Уточнение координат углов
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img_with_corners, CB, corners, True)
        print(f"✓ Найдено {len(corners)} углов для отрисовки")
    else:
        print(f"⚠️ Углы не найдены для отрисовки (это нормально после калибровки)")
        # Добавляем поясняющий текст
        cv2.putText(
            img_with_corners,
            "Corners detected during calibration",
            (50, img.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (136, 255, 136),
            2,
            cv2.LINE_AA
        )
    
    # 🔹 5. Устранение дисторсии
    h, w = img.shape[:2]
    
    try:
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h),
            alpha=alpha,  # 0 = обрезать чёрные края, 1 = сохранить все пиксели
            newImgSize=(w, h)
        )
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_mtx)
        
        # Crop по ROI если нужно
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y:y+rh, x:x+rw]
        
        print(f"✓ Дисторсия устранена. ROI: {roi}")
        
    except Exception as e:
        print(f"⚠️ Ошибка при устранении дисторсии: {e}")
        undistorted = img.copy()  # fallback на оригинал
    
    # 🔹 6. Создание визуализации через matplotlib
    try:
        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='#1a1a2e')
        
        # Заголовок с метриками
        title_parts = [f"Калибровка: {pixels_per_mm:.2f} px/mm"]
        if rms_error > 0:
            title_parts.append(f"RMS: {rms_error:.4f}")
        if square_size_mm:
            title_parts.append(f"Квадрат: {square_size_mm} мм")
        
        fig.suptitle(
            " | ".join(title_parts),
            fontsize=13, fontweight='bold', color='#e0e0ff', y=1.02
        )
        
        # Левая панель: с углами
        axes[0].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"📐 С углами ({CB[0]}×{CB[1]})", color='#4CAF50', fontsize=11)
        axes[0].axis('off')
        
        # Правая панель: без дисторсии
        axes[1].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
        axes[1].set_title("✨ Без дисторсии", color='#2196F3', fontsize=11)
        axes[1].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
    except Exception as e:
        print(f"❌ Ошибка при создании графика: {e}")
        return to_json_serializable({
            "status": "error",
            "message": f"Ошибка визуализации: {str(e)}",
            "saved_path": None
        })
    
    # 🔹 7. Сохранение результата
    saved_path = None
    if save_result:
        if output_dir is None:
            output_dir = os.path.join(BASE_DIR, "calib_visualizations")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Уникальное имя файла
        stem = Path(image_path).stem
        out_filename = f"calib_vis_{stem}_{uuid.uuid4().hex[:6]}.png"
        saved_path = os.path.normpath(os.path.join(output_dir, out_filename))
        
        try:
            plt.savefig(
                saved_path,
                dpi=200,
                bbox_inches='tight',
                facecolor='#1a1a2e',
                edgecolor='none',
                format='png'
            )
            print(f"✓ Визуализация сохранена: {saved_path}")
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            saved_path = None
    
    # 🔹 8. Очистка ресурсов
    plt.close(fig)
    
    # 🔹 9. Возврат результата (все значения сериализуемы)
    return to_json_serializable({
        "status": "ok",
        "message": "Визуализация создана успешно",
        "saved_path": saved_path,
        "pixels_per_mm": float(pixels_per_mm) if pixels_per_mm else 0,
        "mm_per_pixel": float(1 / pixels_per_mm) if pixels_per_mm and pixels_per_mm > 0 else 0,
        "chessboard_size": list(chessboard_size),
        "rms_error": float(rms_error) if rms_error else 0,
        "square_size_mm": float(square_size_mm),
        "alpha": float(alpha)
    })


@app.post("/calibrate-single/")
async def calibrate_single(
    file: UploadFile = File(...),
    square_size_mm: float = Form(25.0),
    chessboard_cols: int = Form(7),
    chessboard_rows: int = Form(4),
    calib_flags: str = Form("fix_k3_k4_k5"),
    calib_alpha: float = Form(1.0)
):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            return JSONResponse(content={"status": "error", "message": "Требуется изображение"}, status_code=400)

        ext = Path(file.filename).suffix.lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
            ext = '.jpg'
        
        temp_dir = os.path.join(BASE_DIR, "temp_calib")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.normpath(os.path.join(temp_dir, f"calib_{uuid.uuid4().hex[:8]}{ext}"))

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Калибровка
        calib_result = calibrate_single_image(
            temp_path,
            square_size_mm=square_size_mm,
            chessboard_size=(chessboard_cols, chessboard_rows),
            calib_flags=calib_flags
        )

        if calib_result["status"] != "ok":
            try: os.remove(temp_path)
            except: pass
            return JSONResponse(content=calib_result, status_code=400 if "error" in calib_result.get("status", "") else 200)

        # Визуализация
        vis_result = visualize_calibration_single(
            image_path=temp_path,
            calib_data={
                "camera_matrix": calib_result["camera_matrix"],
                "dist_coeffs": calib_result["dist_coeffs"],
                "pixels_per_mm": calib_result["pixels_per_mm"],
                "rms_error": calib_result["rms_error"]
            },
            square_size_mm=square_size_mm,
            chessboard_size=(chessboard_cols, chessboard_rows),
            alpha=calib_alpha,
            output_dir=os.path.join(BASE_DIR, "calib_visualizations"),
            save_result=True
        )

        # Удаление временного файла
        try: os.remove(temp_path)
        except: pass

        response = {
            "status": "ok",
            "message": calib_result.get("message", "Калибровка успешна"),
            "rms_error": calib_result.get("rms_error"),
            "pixels_per_mm": calib_result.get("pixels_per_mm"),
            "mm_per_pixel": calib_result.get("mm_per_pixel"),
        }

        if vis_result.get("saved_path") and os.path.isfile(vis_result["saved_path"]):
            filename = os.path.basename(vis_result["saved_path"])
            response["visualization_url"] = f"/calib_visualizations/{filename}"

        return JSONResponse(content=to_json_serializable(response))

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": f"Ошибка сервера: {str(e)}"}, status_code=500)

# ============================================================================
# АНАЛИЗ РАСТЕНИЙ
# ============================================================================

def process_plant_image(input_path: str, output_path: str, config: dict) -> tuple:
    """Основная функция анализа изображения растения."""
    import time
    start_time = time.time()
    
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Не удалось прочитать: {input_path}")
    
    # Предобработка
    img = preprocess_image(img, config)
    vis_img = img.copy()
    
    # Инициализация
    masks = {g: np.zeros(img.shape[:2], bool) for g in ['root', 'stem', 'leaf']}
    areas = {g: 0 for g in ['root', 'stem', 'leaf']}
    lengths_px = {g: 0 for g in ['root', 'stem', 'leaf']}
    graph_stats = {g: {'nodes': 0, 'edges': 0} for g in ['root', 'stem', 'leaf']}
    
    # Сегментация YOLO
    if model is not None:
        try:
            imgsz = config.get("YOLO_IMG_SIZE", 1280)
            conf_map = {'root': config["YOLO_CONF_ROOT"], 'stem': config["YOLO_CONF_STEM"], 'leaf': config["YOLO_CONF_LEAF"]}
            class_map = {'root': [0], 'stem': [1], 'leaf': [2]}
            
            for group in ['root', 'stem', 'leaf']:
                results = model.predict(
                    source=img, conf=conf_map[group], iou=config["YOLO_IOU"],
                    imgsz=imgsz, retina_masks=True, classes=class_map[group],
                    max_det=config.get("YOLO_MAX_DET", 1000), verbose=False
                )[0]
                
                if results.masks is not None:
                    for msk in results.masks.data:
                        seg = msk.cpu().numpy() > 0.5
                        if seg.shape[:2] != img.shape[:2]:
                            seg = cv2.resize(seg.astype(np.uint8), (img.shape[1], img.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                        masks[group] |= seg
                        areas[group] += int(seg.sum())
        except Exception as e:
            print(f"⚠️ YOLO ошибка: {e}")
    
    # Обработка каждой группы
    debug_stages = []
    
    for group in ['stem', 'leaf', 'root']:
        if not np.any(masks[group]):
            continue
        
        # Морфологическая очистка
        mask_uint8 = masks[group].astype(np.uint8) * 255
        ksize = config.get("MORPH_KERNEL_SIZE", 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        
        if config.get("MORPH_ITER", 1) > 0:
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=config["MORPH_ITER"])
        if config.get("MORPH_CLOSE_SIZE", 0) > 0:
            kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (config["MORPH_CLOSE_SIZE"], config["MORPH_CLOSE_SIZE"]))
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kclose, iterations=1)
        
        # Удаление мелких объектов
        min_size = config.get("COMBINED_MIN_SIZE", 60)
        labeled = measure.label(mask_uint8 // 255)
        props = measure.regionprops(labeled)
        for prop in props:
            if prop.area < min_size:
                mask_uint8[labeled == prop.label] = 0
        
        # Удаление по эксцентриситету
        ecc_thresh = config.get("NOISE_REMOVE_ECCENTRICITY", 0.99)
        if ecc_thresh < 1.0:
            for prop in props:
                if prop.eccentricity > ecc_thresh and prop.area < min_size * 3:
                    mask_uint8[labeled == prop.label] = 0
        
        masks[group] = mask_uint8 > 0
        
        # Скелетизация
        skeleton = skeletonize(masks[group])
        
        # Построение графа
        G_temp, _ = build_graph(skeleton, config)
        
        # Удаление малых компонент
        small_comps = [c for c in nx.connected_components(G_temp) 
                      if len(c) < config.get("REMOVE_SMALL_COMP_SIZE", 30)]
        skeleton_cleaned = skeleton.copy()
        for comp in small_comps:
            for node in comp:
                skeleton_cleaned[node[1], node[0]] = False
        
        G, _ = build_graph(skeleton_cleaned, config)
        
        # Обрезка ветвей
        G = prune_small_branches(G, config.get("GRAPH_PRUNE_BRANCHES", 5))
        
        # Соединение компонент
        max_d = config["ROOT_CONNECT_MAX_DIST"] if group == 'root' else config["STEM_LEAF_CONNECT_MAX_DIST"]
        min_d = config["ROOT_CONNECT_MIN_DIST"] if group == 'root' else config["STEM_LEAF_CONNECT_MIN_DIST"]
        G = connect_nearby_components(G, max_d, min_d)
        
        # Очистка корней от мусора
        if group == 'root' and len(G) > 0 and not nx.is_connected(G):
            anchor_points = []
            
            if np.any(masks.get('stem', False)):
                props = measure.regionprops(measure.label(masks['stem'].astype(np.uint8)))
                if props:
                    largest = max(props, key=lambda x: x.area)
                    anchor_points.append((largest.centroid[1], largest.centroid[0]))
            
            if np.any(masks.get('leaf', False)):
                props = measure.regionprops(measure.label(masks['leaf'].astype(np.uint8)))
                if props:
                    largest = max(props, key=lambda x: x.area)
                    anchor_points.append((largest.centroid[1], largest.centroid[0]))
            
            if anchor_points:
                keep_comps = []
                for comp in nx.connected_components(G):
                    nodes = np.array(list(comp))
                    for anchor in anchor_points:
                        if np.min(np.linalg.norm(nodes - np.array(anchor), axis=1)) < config.get("ROOT_ANCHOR_MAX_DIST", 50):
                            keep_comps.append(comp)
                            break
                
                if keep_comps:
                    keep_nodes = set().union(*keep_comps)
                    G = G.subgraph(keep_nodes).copy()
            else:
                largest = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest).copy()
        
        # Подсчёт длины
        total_len = sum(d.get('weight', np.linalg.norm(np.array(u) - np.array(v))) 
                       for u, v, d in G.edges(data=True))
        lengths_px[group] = int(total_len)
        
        # Статистика графа
        graph_stats[group] = {'nodes': len(G.nodes()), 'edges': len(G.edges())}
        
        # Отрисовка
        color = DEFAULT_CONFIG["COLORS"].get(group, {'edge': (255,255,255)})
        thickness = config.get("EDGE_THICKNESS", 2)
        
        for u, v, d in G.edges(data=True):
            cv2.line(vis_img, u, v, color['edge'], thickness)
        
        if config.get("DRAW_NODE_LABELS", "none") != "none":
            font_size = config.get("FONT_SIZE", 12)
            for node in G.nodes():
                deg = G.degree(node)
                c = color.get('node_end', (255,0,0)) if deg != 2 else color.get('node_mid', (0,255,0))
                r = config["NODE_END_RADIUS"] if deg != 2 else config["NODE_MID_RADIUS"]
                cv2.circle(vis_img, node, r, c, -1)
                
                if config["DRAW_NODE_LABELS"] in ["degree", "both"]:
                    cv2.putText(vis_img, str(deg), (node[0]+8, node[1]-8), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_size/20, (255,255,255), 1)
        
        # Debug
        if config.get("SAVE_INTERMEDIATE") == "all":
            debug_stages.append((f"{group}_mask", (masks[group].astype(np.uint8) * 255)))
            debug_stages.append((f"{group}_skeleton", (skeleton.astype(np.uint8) * 255)))
    
    # Сохранение результата
    output_format = config.get("OUTPUT_FORMAT", "png")
    if output_format == "jpg":
        cv2.imwrite(output_path, vis_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif output_format == "webp":
        cv2.imwrite(output_path, vis_img, [cv2.IMWRITE_WEBP_QUALITY, 90])
    else:
        cv2.imwrite(output_path, vis_img)
    
    # Сохранение debug
    if debug_stages and config.get("SAVE_DEBUG_STAGES"):
        h, w = img.shape[:2]
        canvas_h = h * len(debug_stages) + 40 * (len(debug_stages) + 1)
        canvas = np.ones((canvas_h, w, 3), np.uint8) * 30
        y_off = 30
        for title, stage_img in debug_stages:
            if stage_img.dtype == bool:
                stage_img = stage_img.astype(np.uint8) * 255
            if len(stage_img.shape) == 2:
                stage_img = cv2.cvtColor(stage_img, cv2.COLOR_GRAY2BGR)
            if stage_img.shape[:2] != (h, w):
                stage_img = cv2.resize(stage_img, (w, h))
            canvas[y_off:y_off+h, :w] = stage_img
            cv2.putText(canvas, title, (15, y_off+28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            y_off += h + 40
        debug_path = os.path.join(config["DEBUG_DIR"], f"debug_{uuid.uuid4().hex[:8]}.png")
        cv2.imwrite(debug_path, canvas)
    
    # Масштаб
    ppm = config.get("pixels_per_mm") or pixels_per_mm
    if ppm and ppm > 0:
        scale_cm_px = 1.0 / ppm / 10.0
    else:
        scale_cm_px = (config.get("SQUARE_SIZE_MM", 25.0) / 100.0) / 10.0
    
    lengths_cm = {k: round(v * scale_cm_px, 2) for k, v in lengths_px.items() if v > 0}
    
    # Статистика
    total_nodes = sum(gs['nodes'] for gs in graph_stats.values())
    total_edges = sum(gs['edges'] for gs in graph_stats.values())
    
    stats = {
        "areas": {k: int(v) for k, v in areas.items()},
        "lengths_px": {k: int(v) for k, v in lengths_px.items()},
        "lengths_cm_approx": {k: float(v) for k, v in lengths_cm.items()},
        "graph_stats": to_json_serializable(graph_stats),
        "total_area_px": int(sum(areas.values())),
        "total_length_px": int(sum(lengths_px.values())),
        "total_nodes": int(total_nodes),
        "total_edges": int(total_edges),
        "model_loaded": model is not None,
        "calibrated": ppm is not None and ppm > 0,
        "pixels_per_mm": float(ppm) if ppm else None,
        "processing_time_sec": round(time.time() - start_time, 2),
    }
    
    return output_path, to_json_serializable(stats)


@app.post("/analyze/")
async def analyze_plant(file: UploadFile = File(...), params: str = Form("{}")):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            return JSONResponse(content={"error": "Только изображения"}, status_code=400)
        
        # Сохранение файла
        ext = Path(file.filename).suffix.lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
            ext = '.jpg'
        
        stem = uuid.uuid4().hex[:12]
        input_path = os.path.join(BASE_DIR, "uploads", f"{stem}{ext}")
        output_path = os.path.join(BASE_DIR, "results", f"result_{stem}.{ext}")
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Парсинг параметров
        user_cfg = {}
        if params:
            try:
                user_cfg = json.loads(params)
            except:
                pass
        
        # Объединение конфигураций
        proc_cfg = DEFAULT_CONFIG.copy()
        proc_cfg.update(user_cfg)
        
        # Обработка
        res_path, stats = process_plant_image(input_path, output_path, proc_cfg)
        
        response = {
            "status": "ok",
            "image_url": f"/results/{Path(res_path).name}",
            "stats": stats
        }
        
        return JSONResponse(content=to_json_serializable(response))
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e), "status": "error"}, status_code=500)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Plant Graph Analyzer API", "endpoints": ["/", "/analyze/", "/calibrate-single/"]}


# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        log_level="info"
    )