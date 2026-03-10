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
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree
import skimage.measure as meas
import time

# ============================================================================
# УТИЛИТА: Конвертация NumPy типов в JSON-сериализуемые
# ============================================================================
def to_json_serializable(obj: Any) -> Any:
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
for folder in ["uploads", "results", "debug_stages", "temp_calib", "calib_visualizations"]:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)
app.mount("/results", StaticFiles(directory=os.path.join(BASE_DIR, "results")), name="results")
app.mount("/debug", StaticFiles(directory=os.path.join(BASE_DIR, "debug_stages")), name="debug")
app.mount("/calib_visualizations", StaticFiles(directory=os.path.join(BASE_DIR, "calib_visualizations")), name="calib_vis")
app.mount("/temp_calib", StaticFiles(directory=os.path.join(BASE_DIR, "temp_calib")), name="temp_calib")

# ============================================================================
# ⚙️ КОНФИГУРАЦИЯ (точно как в локальном скрипте)
# ============================================================================
DEFAULT_CONFIG = {
    # ─── Пути ────────────────────────────────────────────────────────────────
    "MODEL_PATH": os.path.join(BASE_DIR, "yolo_weights", "plants_optimized_seg", "weights", "best_h.pt"),
    "DEBUG_DIR": os.path.join(BASE_DIR, "debug_stages"),
    
    # ─── Классы ─────────────────────────────────────────────────────────────
    "ROOT_CLASSES": {'root', 'корень'},
    "STEM_CLASSES": {'stem', 'стебель'},
    "LEAF_CLASSES": {'leaf', 'лист'},
    
    # ─── 🎯 YOLO параметры (ТОЧНО как в локальном скрипте) ──────────────────
    "YOLO_CONF_ROOT": 0.1,
    "YOLO_CONF_STEM": 0.10,
    "YOLO_CONF_LEAF": 0.30,
    "YOLO_IOU": 0.50,
    "YOLO_IMG_SIZE": (1536, 2048),
    "YOLO_MAX_DET": 1000,
    "YOLO_RETINA_MASKS": True,
    "YOLO_AGNOSTIC_NMS": False,
    
    # ─── 🖼️ Параметры изображения ───────────────────────────────────────────
    "IMG_MAX_WIDTH": 2048,
    "IMG_MAX_HEIGHT": 1536,
    
    # ─── 🦴 Скелетизация ─────────────────────────────────────────────────────
    "SKELETON_METHOD": "zhang",
    
    # ─── 🕸️ Параметры графа (как в локальном скрипте) ───────────────────────
    "GRAPH_CONNECT_RADIUS": 1.7,
    "GRAPH_MAX_NODES": 38000,
    "REMOVE_SMALL_COMP_SIZE": 60,
    
    # ─── 🔗 Соединение компонент (как в локальном скрипте) ──────────────────
    "ROOT_CONNECT_MAX_DIST": 60,
    "ROOT_CONNECT_MIN_DIST": 4.0,
    "ROOT_ANCHOR_MAX_DIST": 300,
    "STEM_LEAF_CONNECT_MAX_DIST": 5,
    "STEM_LEAF_CONNECT_MIN_DIST": 1.0,
    
    # ─── 🧹 Морфология ───────────────────────────────────────────────────────
    "COMBINED_MIN_SIZE": 20,
    "MORPH_KERNEL_SIZE": 3,
    "MORPH_CLOSE_SIZE": 0,
    "MORPH_ITER": 1,
    
    # ─── 🎨 Отрисовка (как в локальном скрипте) ─────────────────────────────
    "EDGE_THICKNESS": 2,
    "NODE_END_RADIUS_STEM_LEAF": 5,
    "NODE_MID_RADIUS_STEM_LEAF": 2,
    "NODE_END_RADIUS_ROOT": 4,
    "NODE_MID_RADIUS_ROOT": 2,
    "NODE_END_RADIUS_DEBUG": 6,
    "NODE_MID_RADIUS_DEBUG": 3,
    "FONT_SIZE": 0.7,
    "FONT_SIZE_ROOT": 0.6,  # для подписи корней как в локальном
    "OUTPUT_FORMAT": "jpg",
    
    # ─── ⚡ Производительность ───────────────────────────────────────────────
    "USE_GPU": "auto",
    "SAVE_DEBUG_STAGES": True,
    
    # ─── 🎨 Цвета (как в локальном скрипте) ─────────────────────────────────
    "COLORS": {
        'root': {'edge': (255, 0, 180), 'node_end': (0, 0, 255), 'node_mid': (180, 0, 255)},
        'stem': {'edge': (0, 220, 0),   'node_end': (0, 100, 0), 'node_mid': (100, 255, 100)},
        'leaf': {'edge': (0, 180, 255), 'node_end': (0, 80, 180),  'node_mid': (100, 220, 255)},
    },
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
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ТОЧНО как в локальном скрипте)
# ============================================================================

def build_graph(skeleton, config: dict, keep_only_largest: bool = False):
    """Строит граф из скелета — логика точно как в локальном скрипте."""
    yx = np.column_stack(np.where(skeleton))
    if len(yx) == 0:
        return nx.Graph(), skeleton
    
    # Downsampling как в локальном скрипте
    max_nodes = config.get("GRAPH_MAX_NODES", 38000)
    if len(yx) > max_nodes:
        step = max(1, len(yx) // max_nodes)
        yx = yx[::step]
    
    if len(yx) == 0:
        return nx.Graph(), skeleton
    
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
    
    if len(G) and keep_only_largest:
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    
    return G, skeleton


def connect_nearby_components(G: nx.Graph, max_dist: float, min_dist_to_connect: float) -> nx.Graph:
    """
    Соединяет близкие компоненты графа по концевым узлам.
    Логика ТОЧНО как в локальном скрипте.
    """
    if len(G) == 0 or nx.is_connected(G):
        return G
    
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return G
    
    # Собираем концы (degree != 2) каждой компоненты
    tips_per_comp = {}
    for i, comp in enumerate(components):
        subgraph = G.subgraph(comp)
        tips = [n for n in subgraph if subgraph.degree(n) != 2]
        if not tips:
            tips = list(comp)[:1]
        tips_per_comp[i] = tips
    
    all_tips = []
    for i, tips in tips_per_comp.items():
        for tip in tips:
            all_tips.append((tip, i))
    
    if len(all_tips) < 2:
        return G
    
    coords = np.array([pt for pt, _ in all_tips])
    tree = cKDTree(coords)
    
    to_add = []
    used = set()
    pairs = tree.query_pairs(r=max_dist)
    
    for idx1, idx2 in pairs:
        pt1, comp1 = all_tips[idx1]
        pt2, comp2 = all_tips[idx2]
        
        if comp1 == comp2:
            continue
        
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist > max_dist or dist < min_dist_to_connect:
            continue
        
        if idx1 not in used and idx2 not in used:
            to_add.append((pt1, pt2, dist))
            used.add(idx1)
            used.add(idx2)
    
    for u, v, d in sorted(to_add, key=lambda x: x[2]):
        G.add_edge(u, v, weight=d)
    
    return G


def get_color(cls_group: str, config: dict = None):
    """Возвращает цвета для группы (как в локальном скрипте)."""
    if config is None:
        config = DEFAULT_CONFIG
    return config["COLORS"].get(cls_group, config["COLORS"]["root"])


def make_bbox_mask(shape, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Создаёт маску по bounding box."""
    mask = np.zeros(shape[:2], dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def save_debug_stages(name_prefix: str, base_img: np.ndarray, stages: list, config: dict):
    """Сохраняет отладочные стадии."""
    if not stages or not config.get("SAVE_DEBUG_STAGES", True):
        return
    
    h, w = base_img.shape[:2]
    total_height = h * len(stages) + 40 * (len(stages) + 1)
    debug_canvas = np.ones((total_height, w, 3), dtype=np.uint8) * 30
    y_offset = 30
    
    for title, stage_img in stages:
        if stage_img is None or stage_img.size == 0:
            continue
        if stage_img.dtype == bool:
            stage_img = stage_img.astype(np.uint8) * 255
        if len(stage_img.shape) == 2:
            stage_img = cv2.cvtColor(stage_img, cv2.COLOR_GRAY2BGR)
        if stage_img.shape[:2] != (h, w):
            stage_img = cv2.resize(stage_img, (w, h), interpolation=cv2.INTER_AREA)
        
        debug_canvas[y_offset:y_offset + h, 0:w] = stage_img
        cv2.putText(debug_canvas, title, (15, y_offset + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y_offset += h + 40
    
    path = os.path.join(config["DEBUG_DIR"], f"DEBUG_{name_prefix}_stages.jpg")
    cv2.imwrite(path, debug_canvas)
    print(f"🔍 Debug saved: {path}")


def keep_longest_component(skeleton: np.ndarray) -> np.ndarray:
    """Оставляет только самую длинную компоненту скелета."""
    if not skeleton.any():
        return skeleton
    labeled = label(skeleton)
    props = regionprops(labeled)
    if not props:
        return np.zeros_like(skeleton, dtype=bool)
    longest = max(props, key=lambda x: x.area)
    return labeled == longest.label


# ============================================================================
# КАЛИБРОВКА (без изменений)
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
# 🌱 АНАЛИЗ РАСТЕНИЙ (СИНХРОНИЗИРОВАНО С ЛОКАЛЬНЫМ СКРИПТОМ)
# ============================================================================

def process_plant_image(input_path: str, output_path: str, config: dict) -> tuple:
    """
    Обрабатывает изображение растения — логика ТОЧНО как в локальном скрипте.
    """
    start_time = time.time()
    
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Не удалось прочитать: {input_path}")
    
    vis_img = img.copy()
    
    # Инициализация масок и статистики
    masks = {g: np.zeros(img.shape[:2], dtype=bool) for g in ['root', 'stem', 'leaf']}
    areas = {g: 0 for g in ['root', 'stem', 'leaf']}
    bbox_by_group = {g: None for g in ['root', 'stem', 'leaf']}
    lengths_px = {g: 0 for g in ['root', 'stem', 'leaf']}
    graph_stats = {g: {'nodes': 0, 'edges': 0} for g in ['root', 'stem', 'leaf']}
    
    # ─── YOLO сегментация (ПАРАМЕТРЫ ТОЧНО как в локальном скрипте) ─────────
    if model is not None:
        try:
            imgsz = config.get("YOLO_IMG_SIZE", (1536, 2048))
            conf_map = {
                'root': config["YOLO_CONF_ROOT"],
                'stem': config["YOLO_CONF_STEM"],
                'leaf': config["YOLO_CONF_LEAF"]
            }
            class_map = {'root': [0], 'stem': [1], 'leaf': [2]}
            iou_val = config.get("YOLO_IOU", 0.50)
            
            for group in ['root', 'stem', 'leaf']:
                results = model.predict(
                    source=img,
                    conf=conf_map[group],
                    iou=iou_val,
                    imgsz=imgsz,
                    retina_masks=config.get("YOLO_RETINA_MASKS", True),
                    agnostic_nms=config.get("YOLO_AGNOSTIC_NMS", False),
                    classes=class_map[group],
                    max_det=config.get("YOLO_MAX_DET", 1000),
                    verbose=False
                )[0]
                
                if results.masks is not None:
                    for idx, (msk, box) in enumerate(zip(results.masks.data, results.boxes.xyxy)):
                        seg = msk.cpu().numpy() > 0.5
                        if seg.shape[:2] != img.shape[:2]:
                            seg = cv2.resize(seg.astype(np.uint8), (img.shape[1], img.shape[0]),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                        
                        cls_name = model.names[int(results.boxes.cls[idx])].lower()
                        
                        # Определяем группу по имени класса
                        target_group = None
                        if cls_name in config["ROOT_CLASSES"]:
                            target_group = 'root'
                        elif cls_name in config["STEM_CLASSES"]:
                            target_group = 'stem'
                        elif cls_name in config["LEAF_CLASSES"]:
                            target_group = 'leaf'
                        
                        if target_group:
                            masks[target_group] |= seg
                            areas[target_group] += int(seg.sum())
                            
                            x1, y1, x2, y2 = map(int, box.cpu().numpy())
                            if bbox_by_group[target_group] is None:
                                bbox_by_group[target_group] = (x1, y1, x2, y2)
                            else:
                                ex1, ey1, ex2, ey2 = bbox_by_group[target_group]
                                bbox_by_group[target_group] = (
                                    min(ex1, x1), min(ey1, y1),
                                    max(ex2, x2), max(ey2, y2)
                                )
        except Exception as e:
            print(f"⚠️ YOLO ошибка: {e}")
    
    debug_stages = []
    
    # ─── ОБРАБОТКА: STEM ───────────────────────────────────────────────────
    if masks['stem'].any():
        crop_orig = img.copy()
        stages = [("Original", crop_orig.copy()), ("Binary mask", masks['stem'])]
        
        skeleton = skeletonize(masks['stem'])
        stages.append(("Full skeleton", skeleton))
        
        # Удаление малых компонент ПЕРЕД финальным построением графа
        G_temp, _ = build_graph(skeleton, config)
        small_comps = [c for c in nx.connected_components(G_temp)
                    if len(c) < config.get("REMOVE_SMALL_COMP_SIZE", 30)]
        for comp in small_comps:
            for node in comp:
                skeleton[node[1], node[0]] = False  # обнуляем пиксели в скелете
        
        # ПЕРЕСТРОЕНИЕ графа после очистки скелета
        G, _ = build_graph(skeleton, config)
        
        # Соединение компонент
        G = connect_nearby_components(G,
                                    config["STEM_LEAF_CONNECT_MAX_DIST"],
                                    config["STEM_LEAF_CONNECT_MIN_DIST"])
        
        total_len = sum(d['weight'] for _, _, d in G.edges(data=True))
        lengths_px['stem'] = int(total_len)
        graph_stats['stem'] = {'nodes': len(G.nodes()), 'edges': len(G.edges())}
        
        # ─── ОТРИСОВКА DEBUG (как в локальном: радиусы 6/3) ────────────────
        crop_graph = crop_orig.copy()
        color = get_color('stem', config)
        for u, v, d in G.edges(data=True):
            cv2.line(crop_graph, u, v, color['edge'], 2)
        for node in G.nodes():
            # В локальном скрипте для debug: end=6, mid=3
            r = 6 if G.degree(node) != 2 else 3
            c = color['node_end'] if G.degree(node) != 2 else color['node_mid']
            cv2.circle(crop_graph, node, r, c, -1)
        stages.append((f"Final graph L:{int(total_len)} px", crop_graph))
        
        save_debug_stages("stem", crop_orig, stages, config)
        
        # ─── ФИНАЛЬНАЯ ОТРИСОВКА на vis_img (радиусы 5/2 как в локальном) ──
        for u, v, d in G.edges(data=True):
            cv2.line(vis_img, u, v, color['edge'], config["EDGE_THICKNESS"])
        for node in G.nodes():
            r = 5 if G.degree(node) != 2 else 2  # как в локальном скрипте
            c = color['node_end'] if G.degree(node) != 2 else color['node_mid']
            cv2.circle(vis_img, node, r, c, -1)
        
        if bbox_by_group['stem']:
            x1, y1, x2, y2 = bbox_by_group['stem']
            info = f"stem | L:{int(total_len)} px | N:{G.number_of_nodes()}"
            cv2.putText(vis_img, info, (x1, max(y1 - 25, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, config["FONT_SIZE"], (255, 255, 255), 2)
    
    # ─── ОБРАБОТКА: LEAF ────────────────────────────────────────────────────
    if masks['leaf'].any():
        crop_orig = img.copy()
        stages = [("Original", crop_orig.copy()), ("Binary mask", masks['leaf'])]
        
        skeleton = skeletonize(masks['leaf'])
        stages.append(("Full skeleton", skeleton))
        stages.append(("Longest component", skeleton))
        
        # В локальном скрипте для leaf НЕТ удаления малых компонент перед connect!
        G, _ = build_graph(skeleton, config)
        
        # Соединение компонент
        G = connect_nearby_components(G,
                                    config["STEM_LEAF_CONNECT_MAX_DIST"],
                                    config["STEM_LEAF_CONNECT_MIN_DIST"])
        
        total_len = sum(d['weight'] for _, _, d in G.edges(data=True))
        lengths_px['leaf'] = int(total_len)
        graph_stats['leaf'] = {'nodes': len(G.nodes()), 'edges': len(G.edges())}
        
        # ─── ОТРИСОВКА DEBUG (радиусы 6/3 как в локальном) ─────────────────
        crop_graph = crop_orig.copy()
        color = get_color('leaf', config)
        for u, v, d in G.edges(data=True):
            cv2.line(crop_graph, u, v, color['edge'], 2)
        for node in G.nodes():
            r = 6 if G.degree(node) != 2 else 3  # как в локальном
            c = color['node_end'] if G.degree(node) != 2 else color['node_mid']
            cv2.circle(crop_graph, node, r, c, -1)
        stages.append((f"Final graph L:{int(total_len)} px", crop_graph))
        
        save_debug_stages("leaf", crop_orig, stages, config)
        
        # ─── ФИНАЛЬНАЯ ОТРИСОВКА (радиусы 5/2) ─────────────────────────────
        for u, v, d in G.edges(data=True):
            cv2.line(vis_img, u, v, color['edge'], config["EDGE_THICKNESS"])
        for node in G.nodes():
            r = 5 if G.degree(node) != 2 else 2  # как в локальном
            c = color['node_end'] if G.degree(node) != 2 else color['node_mid']
            cv2.circle(vis_img, node, r, c, -1)
        
        if bbox_by_group['leaf']:
            x1, y1, x2, y2 = bbox_by_group['leaf']
            info = f"leaf | L:{int(total_len)} px | N:{G.number_of_nodes()}"
            cv2.putText(vis_img, info, (x1, max(y1 - 25, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, config["FONT_SIZE"], (255, 255, 255), 2)
    
    if masks['root'].any():
        seg_mask = masks['root']
        stages = [("Original", img.copy())]
        
        mask_uint8 = (seg_mask).astype(np.uint8) * 255
        stages.append(("YOLO mask", mask_uint8.copy()))
        
        # Морфология
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (config["MORPH_KERNEL_SIZE"], config["MORPH_KERNEL_SIZE"]))
        mask_smooth = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel,
                                    iterations=config["MORPH_ITER"])
        stages.append(("Smoothed mask", mask_smooth.copy()))
        
        skeleton_raw = skeletonize(mask_smooth > 0)
        stages.append(("Skeleton raw", skeleton_raw.copy()))
        
        # Удаление малых компонент
        G_temp, _ = build_graph(skeleton_raw, config)
        small_comps = [c for c in nx.connected_components(G_temp)
                    if len(c) < config.get("REMOVE_SMALL_COMP_SIZE", 30)]
        for comp in small_comps:
            for node in comp:
                skeleton_raw[node[1], node[0]] = False
        
        skeleton_cleaned = skeleton_raw.copy()
        stages.append(("Skeleton cleaned", skeleton_cleaned.copy()))
        
        G, _ = build_graph(skeleton_raw, config)
        
        graph_before = img.copy()
        for u, v in G.edges():
            cv2.line(graph_before, u, v, (255, 0, 180), 1)
        stages.append(("Graph before connect", graph_before))
        
        # Соединение компонент
        G = connect_nearby_components(G,
                                    config["ROOT_CONNECT_MAX_DIST"],
                                    config["ROOT_CONNECT_MIN_DIST"])
        
    
        # ✅ 1. Собираем якоря из ГРАФОВ стебля и листа (по компонентам!)
        anchor_points = []
        dist_threshold = 100

        # --- СТЕБЛЬ ---
        if masks['stem'].any():
            stem_skel = skeletonize(masks['stem'])
            G_stem, _ = build_graph(stem_skel, config)
            if len(G_stem) > 0:
                for comp in nx.connected_components(G_stem):
                    comp_nodes = np.array(list(comp))
                    if len(comp_nodes) > 0:
                        cx, cy = np.mean(comp_nodes, axis=0)
                        anchor_points.append((float(cx), float(cy)))
        
        # --- ЛИСТ ---
        if masks['leaf'].any():
            leaf_skel = skeletonize(masks['leaf'])
            G_leaf, _ = build_graph(leaf_skel, config)
            if len(G_leaf) > 0:
                for comp in nx.connected_components(G_leaf):
                    comp_nodes = np.array(list(comp))
                    if len(comp_nodes) > 0:
                        cx, cy = np.mean(comp_nodes, axis=0)
                        anchor_points.append((float(cx), float(cy)))
        
        # ✅ 2. Фильтрация графа корней (G)
        if anchor_points and len(G) > 0:
            keep_comps = []
            remove_comps = []
            
            for comp in nx.connected_components(G):
                comp_nodes = np.array(list(comp))
                is_near_anchor = False
                
                for anchor in anchor_points:
                    dists = np.linalg.norm(comp_nodes - np.array(anchor), axis=1)
                    if dists.min() < dist_threshold:
                        is_near_anchor = True
                        break
                
                if is_near_anchor:
                    keep_comps.append(comp)
                else:
                    remove_comps.append(comp)
            
            # ✅ 3. Если нашли компоненты рядом с якорем — оставляем ВСЕ их
            if keep_comps:
                keep_nodes = set().union(*keep_comps)
                G = G.subgraph(keep_nodes).copy()
                print(f"   [root] kept {len(keep_comps)} components near anchors, removed {len(remove_comps)} far")
            else:
                # ⚠️ НИ ОДНОЙ компоненты рядом — НЕ удаляем всё, а оставляем ВСЕ компоненты корней!
                # (или можно оставить топ-N по размеру, но не одну)
                print(f"   [root] no components near anchors, keeping ALL {len(list(nx.connected_components(G)))} components")
                # G остаётся без изменений
        elif len(G) > 0:
            # Нет якорей — оставляем ВСЕ компоненты корней (не одну!)
            n_comps = len(list(nx.connected_components(G)))
            print(f"   [root] no anchors, keeping ALL {n_comps} components")
            # G остаётся без изменений


        # ─── РАСЧЁТ ДЛИНЫ ───────────────────────────────────────────────────
        total_len = sum(d.get('weight', np.linalg.norm(np.array(u) - np.array(v)))
                    for u, v, d in G.edges(data=True))
        lengths_px['root'] = int(total_len)
        graph_stats['root'] = {'nodes': len(G.nodes()), 'edges': len(G.edges())}
        
        # Отрисовка debug
        graph_after = img.copy()
        color = get_color('root', config)
        for u, v, d in G.edges(data=True):
            cv2.line(graph_after, u, v, color['edge'], 2)
        for node in G.nodes():
            deg = G.degree(node)
            r = 4 if deg != 2 else 2
            c = color['node_end'] if deg != 2 else color['node_mid']
            cv2.circle(graph_after, node, r, c, -1)
        stages.append((f"Final graph L:{int(total_len)} px", graph_after))
        
        save_debug_stages("root", img.copy(), stages, config)
        
        # Отрисовка на финальное изображение
        for u, v, d in G.edges(data=True):
            cv2.line(vis_img, u, v, color['edge'], config["EDGE_THICKNESS"])
        for node in G.nodes():
            deg = G.degree(node)
            r = 4 if deg != 2 else 2
            c = color['node_end'] if deg != 2 else color['node_mid']
            cv2.circle(vis_img, node, r, c, -1)
        
        # Подпись длины корней — ✅ FONT_SIZE = 0.6 как в локальном!
        labeled = meas.label(seg_mask.astype(bool))
        props = meas.regionprops(labeled)
        if props:
            main_p = max(props, key=lambda x: x.area)
            cy_c, cx_c = main_p.centroid
            cv2.putText(vis_img, f"Roots: {int(total_len)} px", (int(cx_c), int(cy_c)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Bounding box
        if bbox_by_group['root']:
            x1, y1, x2, y2 = bbox_by_group['root']
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 140, 255), 2)

    # ─── ОТРИСОВКА ПЛОЩАДЕЙ И МАСОК (как в локальном скрипте) ─────────────
    overlay = vis_img.copy()
    for group_name, mask_bool in [('leaf', masks['leaf']), ('stem', masks['stem']), ('root', masks['root'])]:
        if np.any(mask_bool):
            color = config["COLORS"][group_name]['edge']
            overlay[mask_bool] = color
    
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
    
    # Панель статистики
    padding = 20
    line_height = 40
    panel_width = 450
    panel_height = line_height * 4 + padding
    margin = 30
    
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + panel_width, margin + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0, vis_img)
    
    stats = [
        (f"Roots Area: {areas['root']:,} px", config["COLORS"]["root"]["edge"]),
        (f"Stem Area:  {areas['stem']:,} px", config["COLORS"]["stem"]["edge"]),
        (f"Leaf Area:  {areas['leaf']:,} px", config["COLORS"]["leaf"]["edge"]),
        (f"Total Area: {sum(areas.values()):,} px", (255, 255, 255))
    ]
    
    for i, (text, color) in enumerate(stats):
        y_pos = margin + padding + (i * line_height) + 10
        cv2.putText(vis_img, text, (margin + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis_img, text, (margin + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2, cv2.LINE_AA)
    
    # ─── СОХРАНЕНИЕ ─────────────────────────────────────────────────────────
    output_format = config.get("OUTPUT_FORMAT", "jpg")
    if output_format == "jpg":
        cv2.imwrite(output_path, vis_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif output_format == "webp":
        cv2.imwrite(output_path, vis_img, [cv2.IMWRITE_WEBP_QUALITY, 90])
    else:
        cv2.imwrite(output_path, vis_img)
    
    # Статистика
    ppm = config.get("pixels_per_mm") or pixels_per_mm
    if ppm and ppm > 0:
        scale_cm_px = 1.0 / ppm / 10.0
    else:
        scale_cm_px = (config.get("SQUARE_SIZE_MM", 25.0) / 100.0) / 10.0
    
    lengths_cm = {k: round(v * scale_cm_px, 2) for k, v in lengths_px.items() if v > 0}
    
    stats = {
        "areas": {k: int(v) for k, v in areas.items()},
        "lengths_px": {k: int(v) for k, v in lengths_px.items()},
        "lengths_cm_approx": {k: float(v) for k, v in lengths_cm.items()},
        "graph_stats": to_json_serializable(graph_stats),
        "total_area_px": int(sum(areas.values())),
        "total_length_px": int(sum(lengths_px.values())),
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
        
        ext = Path(file.filename).suffix.lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
            ext = '.jpg'
        
        stem = uuid.uuid4().hex[:12]
        input_path = os.path.join(BASE_DIR, "uploads", f"{stem}{ext}")
        output_path = os.path.join(BASE_DIR, "results", f"result_{stem}.{ext}")
        
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        user_cfg = {}
        if params:
            try:
                user_cfg = json.loads(params)
            except:
                pass
        
        proc_cfg = DEFAULT_CONFIG.copy()
        proc_cfg.update(user_cfg)
        
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


@app.get("/")
async def root():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Plant Graph Analyzer API", "endpoints": ["/", "/analyze/", "/calibrate-single/"]}




    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        log_level="info"
    )