import os
import uuid
from pathlib import Path
from typing import Dict, List
import io
import zipfile

import cv2
import numpy as np
import networkx as nx
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from starlette.responses import StreamingResponse

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "all_data"
FRONTEND_DIR = BASE_DIR / "frontend"
WEIGHTS_PATH = DATA_DIR / "yolo_weights" / "plants_optimized_seg" / "weights" / "best.pt"
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_FILE = BASE_DIR / "calibration.json"

# Стандартная калибровка (фиксированное значение, без фото).
STANDARD_MM_PER_PX = float(os.environ.get("MM_PER_PX", "0.213317502"))

CALIBRATION_MODE_STANDARD = "standard"
CALIBRATION_MODE_PHOTO = "photo"


def _load_calibration_data() -> dict:
    import json
    if CALIBRATION_FILE.is_file():
        try:
            return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"mode": CALIBRATION_MODE_STANDARD, "mm_per_px": STANDARD_MM_PER_PX}


def _save_calibration_data(data: dict) -> None:
    import json
    CALIBRATION_FILE.write_text(
        json.dumps({"mode": data.get("mode", CALIBRATION_MODE_STANDARD), "mm_per_px": data.get("mm_per_px", STANDARD_MM_PER_PX)}, indent=2),
        encoding="utf-8",
    )


def _load_calibration() -> float:
    """Всегда стандартная калибровка (фиксированный масштаб)."""
    return STANDARD_MM_PER_PX


# Глобальные калибровочные коэффициенты (обновляются при смене режима/калибровке).
_MM_PER_PX = _load_calibration()
_MM2_PER_PX2 = _MM_PER_PX * _MM_PER_PX


def get_mm_per_px() -> float:
    return _MM_PER_PX


def get_calibration_mode() -> str:
    return CALIBRATION_MODE_STANDARD


def set_calibration(mm_per_px: float) -> None:
    """Установить калибровку по фото (режим «по фото»)."""
    global _MM_PER_PX, _MM2_PER_PX2
    _MM_PER_PX = mm_per_px
    _MM2_PER_PX2 = mm_per_px * mm_per_px
    _save_calibration_data({"mode": CALIBRATION_MODE_PHOTO, "mm_per_px": mm_per_px})


def set_calibration_mode(mode: str) -> None:
    """Переключить режим: standard | photo. При standard используется STANDARD_MM_PER_PX."""
    global _MM_PER_PX, _MM2_PER_PX2
    data = _load_calibration_data()
    data["mode"] = mode if mode in (CALIBRATION_MODE_STANDARD, CALIBRATION_MODE_PHOTO) else CALIBRATION_MODE_STANDARD
    _save_calibration_data(data)
    _MM_PER_PX = STANDARD_MM_PER_PX if data["mode"] == CALIBRATION_MODE_STANDARD else float(data.get("mm_per_px", STANDARD_MM_PER_PX))
    _MM2_PER_PX2 = _MM_PER_PX * _MM_PER_PX

# В коде используем get_mm_per_px(), чтобы учитывать обновлённую калибровку.


COLORS: Dict[str, Dict[str, tuple]] = {
    "root": {"edge": (255, 0, 180), "node_end": (0, 0, 255), "node_mid": (180, 0, 255)},
    "stem": {"edge": (0, 220, 0), "node_end": (0, 100, 0), "node_mid": (100, 255, 100)},
    "leaf": {"edge": (0, 180, 255), "node_end": (0, 80, 180), "node_mid": (100, 220, 255)},
}

CLASS_INDEX_TO_NAME = {
    0: "root",
    1: "stem",
    2: "leaf",
}

# ── ГРУППЫ КЛАССОВ И ПАРАМЕТРЫ ГРАФА (как в ноутбуке) ──────────────────────────

ROOT_CLASSES = {"root", "корень"}
STEM_CLASSES = {"stem", "стебель"}
LEAF_CLASSES = {"leaf", "лист"}

MASK_OVERLAP_THRESH = 0.60
COMBINED_MIN_SIZE = 60
SKELETON_MIN_LENGTH = 30

GRAPH_CONNECT_RADIUS = 1.7
GRAPH_MAX_NODES = 38000
MAX_SHORT_BRANCH_LEN_ROOT = 1.42


class AnalyzeResponse(BaseModel):
    job_id: str
    mm_per_px: float
    areas_mm2: Dict[str, float]
    areas_cm2: Dict[str, float]
    areas_px: Dict[str, int]
    lengths_mm: Dict[str, float]
    lengths_cm: Dict[str, float]
    lengths_px: Dict[str, float]
    image_url: str
    json_url: str
    csv_url: str


class ArchiveRequest(BaseModel):
    job_ids: List[str]


# Размеры сетки внутренних углов (cols x rows). 8×5 клеток → (7, 4) внутренних углов (как в рабочей калибровке).
CHESSBOARD_8x5 = (7, 4)
CALIBRATION_PATTERNS: List[tuple] = [
    (7, 4), (4, 7), (5, 8), (8, 5), (5, 4), (4, 5), (6, 4), (4, 6),
    (6, 9), (9, 6), (5, 3), (3, 5), (2, 7), (7, 2), (3, 8), (8, 3),
    (3, 7), (7, 3), (4, 4), (6, 6), (5, 5), (3, 4), (4, 3),
    (2, 5), (5, 2), (2, 6), (6, 2), (3, 6), (6, 3),
]
# Флаги из рабочей калибровки (findChessboardCorners)
CALIB_CB_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def _clahe(gray: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """Усиление контраста (помогает при бликах и тенях)."""
    try:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
        return clahe.apply(gray)
    except Exception:
        return gray


def _corners_to_px_per_square(corners, cols: int, rows: int) -> float | None:
    """По массиву углов считает среднее расстояние между соседними (в px на клетку)."""
    if corners is None or len(corners) != rows * cols:
        return None
    pts = corners.reshape(rows, cols, 2)
    dists: List[float] = []
    for r in range(rows):
        for c in range(cols - 1):
            dists.append(float(np.linalg.norm(pts[r, c] - pts[r, c + 1])))
    for r in range(rows - 1):
        for c in range(cols):
            dists.append(float(np.linalg.norm(pts[r, c] - pts[r + 1, c])))
    if not dists:
        return None
    px = float(np.mean(dists))
    return px if px > 0 else None


def _find_corners_one(gray: np.ndarray, cols: int, rows: int, flags: int) -> float | None:
    """findChessboardCorners; при успехе возвращает px_per_square."""
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok or corners is None:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return _corners_to_px_per_square(corners, cols, rows)


def _find_corners_sb(gray: np.ndarray, cols: int, rows: int) -> float | None:
    """findChessboardCornersSB (более устойчив к частичной видимости/бликам), если есть в OpenCV."""
    try:
        sb = getattr(cv2, "findChessboardCornersSB", None)
        if sb is None:
            return None
        flags = 0
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            flags |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
        ok, corners = sb(gray, (cols, rows), flags)
        if not ok or corners is None:
            return None
        return _corners_to_px_per_square(corners, cols, rows)
    except Exception:
        return None


def _calibrate_chessboard_single(gray: np.ndarray, checker_size_mm: float, patterns: List[tuple]) -> float | None:
    """Калибровка: сначала (7,4) с флагами как в рабочей визуализации, затем SB и остальные паттерны."""
    # 1) Точно как в рабочем коде: (7, 4) + ADAPTIVE_THRESH + NORMALIZE_IMAGE
    cols, rows = CHESSBOARD_8x5
    px = _find_corners_one(gray, cols, rows, CALIB_CB_FLAGS)
    if px is not None:
        return checker_size_mm / px
    # 2) findChessboardCornersSB по всем паттернам (устойчивее к бликам)
    for (cols, rows) in patterns:
        px = _find_corners_sb(gray, cols, rows)
        if px is not None:
            return checker_size_mm / px
    # 3) Классический findChessboardCorners по всем паттернам и флагам
    flags_list = [CALIB_CB_FLAGS, cv2.CALIB_CB_ADAPTIVE_THRESH]
    for flags in flags_list:
        for (cols, rows) in patterns:
            px = _find_corners_one(gray, cols, rows, flags)
            if px is not None:
                return checker_size_mm / px
    return None


def _calibrate_chessboard(gray: np.ndarray, checker_size_mm: float, patterns: List[tuple]) -> float | None:
    """Калибровка: оригинал, CLAHE, bilateral, несколько масштабов."""
    h, w = gray.shape[:2]
    variants: List[tuple] = [(gray, 1.0)]
    variants.append((_clahe(gray), 1.0))
    variants.append((_clahe(gray, clip_limit=3.0, grid=4), 1.0))
    try:
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        variants.append((bilateral, 1.0))
        variants.append((_clahe(bilateral), 1.0))
    except Exception:
        pass
    for img, scale_inv in variants:
        result = _calibrate_chessboard_single(img, checker_size_mm, patterns)
        if result is not None:
            return result / scale_inv
    # Несколько масштабов для больших и маленьких фото
    for target in (800, 1000, 1200, 600, 400):
        if max(h, w) > target + 50:
            scale = target / max(h, w)
            small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            for proc in [lambda g: g, _clahe]:
                try:
                    v = proc(small)
                    result = _calibrate_chessboard_single(v, checker_size_mm, patterns)
                    if result is not None:
                        return result / scale
                except Exception:
                    pass
        elif max(h, w) < target - 50 and target <= 800:
            scale = target / max(h, w)
            big = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            result = _calibrate_chessboard_single(big, checker_size_mm, patterns)
            if result is not None:
                return result / scale
    return None


app = FastAPI(title="Plant Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# РЎС‚Р°С‚РёС‡РµСЃРєР°СЏ СЂР°Р·РґР°С‡Р° С„СЂРѕРЅС‚РµРЅРґР° Рё index.html РЅР° РєРѕСЂРЅРµ
if FRONTEND_DIR.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR, html=False),
        name="static-frontend",
    )


if not WEIGHTS_PATH.is_file():
    raise FileNotFoundError(
        f"YOLO weights not found: {WEIGHTS_PATH}\n"
        "Put best.pt there (see README: section 'Веса модели YOLO'). "
        "Weights are not in the repo due to GitHub 100 MB limit."
    )

model = YOLO(str(WEIGHTS_PATH))


def apply_preprocessing(
    image_bgr: np.ndarray,
    grayscale: bool,
    contrast: float,
    brightness: int,
    saturation: float,
    blur: float,
    enhance_dark_background: bool = False,
) -> np.ndarray:
    img = image_bgr.copy()

    if grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Усиление контраста для тёмного фона (чашки Петри, тёмный фон — лучше видны корни и листья)
    if enhance_dark_background:
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        except Exception:
            pass

    # Контраст и яркость
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    # Насыщенность (только для цветного изображения)
    if not grayscale and abs(saturation - 1.0) > 1e-3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= saturation
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Размытие
    if blur > 0:
        k = int(round(blur)) * 2 + 1
        k = max(3, k)
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


# ── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ГРАФОВ (адаптация из ноутбука) ────────────────


def _keep_best_one_per_class(results):
    if len(results.boxes) == 0:
        return results

    boxes = results.boxes
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    areas = ((boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])).cpu().numpy()
    if areas.max() > 0:
        norm_areas = areas / areas.max()
    else:
        norm_areas = np.zeros_like(areas)

    combined_scores = confs ** 1.2 * (norm_areas ** 0.8 + 1e-6)

    keep_indices = []
    for cls in np.unique(classes):
        mask = classes == cls
        if not mask.any():
            continue
        cls_scores = combined_scores[mask]
        best_local_idx = cls_scores.argmax()
        global_idx = np.where(mask)[0][best_local_idx]
        keep_indices.append(global_idx)

    if not keep_indices:
        results.boxes = results.boxes[:0]
        if results.masks is not None:
            results.masks = results.masks[:0]
        return results

    keep_indices = sorted(keep_indices)
    results.boxes = results.boxes[keep_indices]
    if results.masks is not None:
        results.masks = results.masks[keep_indices]
    return results


def _get_adaptive_params(gray, mask):
    if not np.any(mask):
        return 35, 9
    masked_gray = gray[mask]
    if len(masked_gray) < 100:
        return 25, 7
    contrast = np.std(masked_gray)
    block_size = int(25 + 0.05 * np.sqrt(mask.sum()))
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    block_size = max(15, min(95, block_size))
    C = max(1, min(12, 9 - contrast / 15))
    return block_size, C


def _adaptive_binarize(gray, mask_bool):
    if not np.any(mask_bool):
        return np.zeros_like(gray, dtype=bool)
    block, C = _get_adaptive_params(gray, mask_bool)
    masked = gray.copy()
    masked[~mask_bool] = 255
    binary = cv2.adaptiveThreshold(
        masked,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block,
        C=C,
    )
    binary = (binary == 0) & mask_bool
    binary = remove_small_objects(binary, min_size=COMBINED_MIN_SIZE)
    return binary


def _filter_skeleton_by_mask(skeleton, seg_mask, thresh=0.5):
    if thresh <= 0 or not skeleton.any():
        return skeleton
    labeled = label(skeleton)
    out = np.zeros_like(skeleton, dtype=bool)
    for lbl in range(1, labeled.max() + 1):
        comp = labeled == lbl
        total = comp.sum()
        if total < SKELETON_MIN_LENGTH:
            continue
        inside = (comp & seg_mask).sum()
        if inside / total >= thresh:
            out |= comp
    return out


def _keep_longest_component(skeleton):
    if not skeleton.any():
        return skeleton
    labeled = label(skeleton)
    props = regionprops(labeled)
    if not props:
        return np.zeros_like(skeleton, dtype=bool)
    longest = max(props, key=lambda x: x.area)
    return labeled == longest.label


def _build_graph(skeleton, keep_only_largest=False):
    yx = np.column_stack(np.where(skeleton))
    if len(yx) == 0:
        return nx.Graph(), skeleton
    step = max(1, len(yx) // GRAPH_MAX_NODES)
    yx = yx[::step]
    if len(yx) == 0:
        return nx.Graph(), skeleton
    tree = cKDTree(yx)
    pairs = tree.query_pairs(r=GRAPH_CONNECT_RADIUS)
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


def _remove_short_terminal_branches(G, max_len):
    if max_len <= 0:
        return
    changed = True
    while changed:
        changed = False
        to_remove = []
        for node in list(G.nodes()):
            if G.degree(node) == 1:
                nbrs = list(G.neighbors(node))
                if len(nbrs) == 1:
                    nbr = nbrs[0]
                    if G[node][nbr]["weight"] <= max_len:
                        to_remove.append((node, nbr))
        if to_remove:
            G.remove_edges_from(to_remove)
            G.remove_nodes_from(list(nx.isolates(G)))
            changed = True


def _connect_nearby_components(G, max_dist=25, min_dist_to_connect=3.0):
    if len(G) == 0 or nx.is_connected(G):
        return G

    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return G

    tips_per_comp = {}
    for i, comp in enumerate(components):
        subgraph = G.subgraph(comp)
        tips = [n for n in subgraph if subgraph.degree(n) == 1]
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
    for idx1, idx2 in tree.query_pairs(r=max_dist):
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


def _get_color(group_name: str):
    return COLORS.get(group_name, COLORS["root"])


def _make_bbox_mask(shape, x1, y1, x2, y2):
    mask = np.zeros(shape[:2], dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def run_graph_segmentation(
    image_bgr: np.ndarray,
    conf: float = 0.10,
    connect_dist: float = 100.0,
    root_overlap_thresh: float = MASK_OVERLAP_THRESH,
) -> (np.ndarray, Dict[str, int], Dict[str, float], Dict[str, float], Dict[str, float]):
    """
    Полный пайплайн из ноутбука: сегментация + скелетизация + графы для корней, стебля и листьев.
    Возвращает картинку с графами, площади в пикселях и в мм^2.
    """
    img = image_bgr.copy()
    gray_global = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── YOLO-предсказание как в ноутбуке ─────────────────────────
    results = model.predict(source=img, conf=float(conf), imgsz=960, retina_masks=True, verbose=False)[0]
    results = _keep_best_one_per_class(results)

    vis_img = img.copy()
    masks_by_group = {
        "leaf": np.zeros(img.shape[:2], dtype=bool),
        "stem": np.zeros(img.shape[:2], dtype=bool),
        "root": [],
    }
    bbox_by_group = {"leaf": None, "stem": None}
    areas_px: Dict[str, int] = {"root": 0, "stem": 0, "leaf": 0}
    lengths_px: Dict[str, float] = {"root": 0.0, "stem": 0.0, "leaf": 0.0}

    if results.masks is None or len(results.masks.data) == 0:
        mm2_per_px2 = get_mm_per_px() ** 2
        mm_per_px = get_mm_per_px()
        areas_mm2 = {k: float(v) * mm2_per_px2 for k, v in areas_px.items()}
        lengths_mm = {k: float(v) * mm_per_px for k, v in lengths_px.items()}
        return vis_img, areas_px, areas_mm2, lengths_px, lengths_mm

    for idx, (mask_tensor, box) in enumerate(zip(results.masks.data, results.boxes.xyxy)):
        cls_name = model.names[int(results.boxes.cls[idx])].lower()
        seg_mask = mask_tensor.cpu().numpy() > 0.5
        x1, y1, x2, y2 = map(int, box.cpu().numpy())

        if cls_name in ROOT_CLASSES:
            bbox_mask = _make_bbox_mask(img.shape, x1, y1, x2, y2)
            masks_by_group["root"].append((bbox_mask, (x1, y1, x2, y2), seg_mask))
            areas_px["root"] += int(np.sum(seg_mask))
        elif cls_name in STEM_CLASSES:
            masks_by_group["stem"] |= seg_mask
            areas_px["stem"] += int(np.sum(seg_mask))
            if bbox_by_group["stem"] is None:
                bbox_by_group["stem"] = (x1, y1, x2, y2)
            else:
                ex1, ey1, ex2, ey2 = bbox_by_group["stem"]
                bbox_by_group["stem"] = (min(ex1, x1), min(ey1, y1), max(ex2, x2), max(ey2, y2))
        elif cls_name in LEAF_CLASSES:
            masks_by_group["leaf"] |= seg_mask
            areas_px["leaf"] += int(np.sum(seg_mask))
            if bbox_by_group["leaf"] is None:
                bbox_by_group["leaf"] = (x1, y1, x2, y2)
            else:
                ex1, ey1, ex2, ey2 = bbox_by_group["leaf"]
                bbox_by_group["leaf"] = (min(ex1, x1), min(ey1, y1), max(ex2, x2), max(ey2, y2))

    # ── STEM ──────────────────────────────────────────────────────
    if masks_by_group["stem"].any():
        crop_orig_stem = img.copy()
        skeleton = skeletonize(masks_by_group["stem"])
        longest_skel = _keep_longest_component(skeleton)
        G, _ = _build_graph(longest_skel)
        G = _connect_nearby_components(G, max_dist=connect_dist, min_dist_to_connect=4.0)

        color = _get_color("stem")
        for u, v, d in G.edges(data=True):
            cv2.line(vis_img, u, v, color["edge"], 2)
        for node in G.nodes():
            if G.degree(node) != 2:
                cv2.circle(vis_img, node, 5, color["node_end"], -1)
            else:
                cv2.circle(vis_img, node, 2, color["node_mid"], -1)

        if bbox_by_group["stem"]:
            x1, y1, x2, y2 = bbox_by_group["stem"]
            total_len = float(sum(d["weight"] for _, _, d in G.edges(data=True)))
            lengths_px["stem"] = total_len
            info = f"stem | L:{total_len} px | N:{G.number_of_nodes()}"
            cv2.putText(
                vis_img,
                info,
                (x1, max(y1 - 25, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    # ── LEAF ──────────────────────────────────────────────────────
    if masks_by_group["leaf"].any():
        crop_orig_leaf = img.copy()
        skeleton = skeletonize(masks_by_group["leaf"])
        longest_skel = _keep_longest_component(skeleton)
        G, _ = _build_graph(longest_skel)
        G = _connect_nearby_components(G, max_dist=connect_dist, min_dist_to_connect=4.0)

        color = _get_color("leaf")
        for u, v, d in G.edges(data=True):
            cv2.line(vis_img, u, v, color["edge"], 2)
        for node in G.nodes():
            if G.degree(node) != 2:
                cv2.circle(vis_img, node, 5, color["node_end"], -1)
            else:
                cv2.circle(vis_img, node, 2, color["node_mid"], -1)

        if bbox_by_group["leaf"]:
            x1, y1, x2, y2 = bbox_by_group["leaf"]
            total_len = float(sum(d["weight"] for _, _, d in G.edges(data=True)))
            lengths_px["leaf"] = total_len
            info = f"leaf | L:{total_len} px | N:{G.number_of_nodes()}"
            cv2.putText(
                vis_img,
                info,
                (x1, max(y1 - 25, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    # ── ROOTS ─────────────────────────────────────────────────────
    for root_idx, (bbox_mask, bbox, seg_mask) in enumerate(masks_by_group["root"]):
        x1, y1, x2, y2 = bbox
        pad = 100
        cy1 = max(0, y1 - pad)
        cy2 = min(img.shape[0], y2 + pad)
        cx1 = max(0, x1 - pad)
        cx2 = min(img.shape[1], x2 + pad)
        crop_orig = img[cy1:cy2, cx1:cx2].copy()

        binary = _adaptive_binarize(gray_global, bbox_mask)
        combined = binary & seg_mask

        skeleton_raw = skeletonize(combined)
        skeleton_filt = _filter_skeleton_by_mask(skeleton_raw, seg_mask, root_overlap_thresh)

        G, _ = _build_graph(skeleton_filt)
        to_remove_mask = [
            (u, v) for u, v in G.edges() if not (seg_mask[u[1], u[0]] and seg_mask[v[1], v[0]])
        ]
        G.remove_edges_from(to_remove_mask)
        _remove_short_terminal_branches(G, MAX_SHORT_BRANCH_LEN_ROOT)
        G.remove_nodes_from(list(nx.isolates(G)))
        G = _connect_nearby_components(G, max_dist=connect_dist, min_dist_to_connect=4.0)

        color = _get_color("root")
        total_len = float(sum(d["weight"] for _, _, d in G.edges(data=True)))
        lengths_px["root"] += total_len

        for u, v, d in G.edges(data=True):
            u_adj = (u[0] - cx1, u[1] - cy1)
            v_adj = (v[0] - cx1, v[1] - cy1)
            cv2.line(crop_orig, u_adj, v_adj, color["edge"], 2)

        for node in G.nodes():
            pt = (node[0] - cx1, node[1] - cy1)
            if G.degree(node) != 2:
                cv2.circle(crop_orig, pt, 4, color["node_end"], -1)
            else:
                cv2.circle(crop_orig, pt, 2, color["node_mid"], -1)

        # Рисуем на основном изображении
        for u, v, d in G.edges(data=True):
            cv2.line(vis_img, u, v, color["edge"], 2)
        for node in G.nodes():
            if G.degree(node) != 2:
                cv2.circle(vis_img, node, 4, color["node_end"], -1)
            else:
                cv2.circle(vis_img, node, 2, color["node_mid"], -1)

        info = f"root-{root_idx} | L:{total_len} px | N:{G.number_of_nodes()}"
        cv2.putText(
            vis_img,
            info,
            (x1, max(y1 - 25, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 140, 255), 2)

    # ── ОТРИСОВКА ПЛОЩАДЕЙ И ПАНЕЛИ ──────────────────────────────
    overlay = vis_img.copy()

    root_full_mask = np.zeros(img.shape[:2], dtype=bool)
    for bbox_m, coords, seg_m in masks_by_group["root"]:
        root_full_mask |= seg_m

    for group_name, mask_bool in [
        ("leaf", masks_by_group["leaf"]),
        ("stem", masks_by_group["stem"]),
    ]:
        if np.any(mask_bool):
            color = COLORS[group_name]["edge"]
            overlay[mask_bool] = color

    if np.any(root_full_mask):
        color = COLORS["root"]["edge"]
        overlay[root_full_mask] = color

    alpha = 0.3
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)

    padding = 20
    line_height = 40
    panel_width = 450
    panel_height = line_height * 4 + padding
    margin = 30

    overlay_panel = vis_img.copy()
    cv2.rectangle(
        overlay_panel,
        (margin, margin),
        (margin + panel_width, margin + panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay_panel, 0.6, vis_img, 0.4, 0, vis_img)

    stats = [
        (f"Roots Area:  {areas_px['root']:,} px", COLORS["root"]["edge"]),
        (f"Stem Area:   {areas_px['stem']:,} px", COLORS["stem"]["edge"]),
        (f"Leaf Area:   {areas_px['leaf']:,} px", COLORS["leaf"]["edge"]),
        (f"Total Area:  {sum(areas_px.values()):,} px", (255, 255, 255)),
    ]

    for i, (text, color) in enumerate(stats):
        y_pos = margin + padding + (i * line_height) + 10
        cv2.putText(
            vis_img,
            text,
            (margin + 15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis_img,
            text,
            (margin + 15, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

    mm2_per_px2 = get_mm_per_px() ** 2
    mm_per_px = get_mm_per_px()
    areas_mm2 = {k: float(v) * mm2_per_px2 for k, v in areas_px.items()}
    lengths_mm = {k: float(v) * mm_per_px for k, v in lengths_px.items()}
    return vis_img, areas_px, areas_mm2, lengths_px, lengths_mm


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    grayscale: bool = Form(False),
    contrast: float = Form(1.0),
    brightness: int = Form(0),
    saturation: float = Form(1.0),
    blur: float = Form(0.0),
    enhance_dark: bool = Form(False),
    conf: float = Form(0.07),
    connect_dist: float = Form(100.0),
    root_overlap: float = Form(MASK_OVERLAP_THRESH),
):
    content = await file.read()
    np_arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"detail": "Cannot decode image"})

    img_proc = apply_preprocessing(
        img,
        grayscale=grayscale,
        contrast=contrast,
        brightness=brightness,
        saturation=saturation,
        blur=blur,
        enhance_dark_background=enhance_dark,
    )

    vis_img, areas_px, areas_mm2, lengths_px, lengths_mm = run_graph_segmentation(
        img_proc,
        conf=conf,
        connect_dist=connect_dist,
        root_overlap_thresh=root_overlap,
    )

    job_id = uuid.uuid4().hex
    out_dir = RUNS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "result.png"
    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"

    cv2.imwrite(str(image_path), vis_img)

    import json
    areas_cm2 = {k: v / 100.0 for k, v in areas_mm2.items()}
    lengths_cm = {k: v / 10.0 for k, v in lengths_mm.items()}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "job_id": job_id,
                "file_name": file.filename,
                "mm_per_px": get_mm_per_px(),
                "areas_px": areas_px,
                "areas_mm2": areas_mm2,
                "areas_cm2": areas_cm2,
                "lengths_px": lengths_px,
                "lengths_mm": lengths_mm,
                "lengths_cm": lengths_cm,
                "params": {
                    "grayscale": grayscale,
                    "contrast": contrast,
                    "brightness": brightness,
                    "saturation": saturation,
                    "blur": blur,
                    "enhance_dark": enhance_dark,
                    "conf": conf,
                    "connect_dist": connect_dist,
                    "root_overlap": root_overlap,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    import csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "area_px", "area_mm2", "area_cm2"])
        for cls_name in CLASS_INDEX_TO_NAME.values():
            writer.writerow([
                cls_name,
                areas_px.get(cls_name, 0),
                areas_mm2.get(cls_name, 0.0),
                areas_cm2.get(cls_name, 0.0),
            ])

    base_url = ""
    image_url = f"/api/download/image/{job_id}"
    json_url = f"/api/download/json/{job_id}"
    csv_url = f"/api/download/csv/{job_id}"

    return AnalyzeResponse(
        job_id=job_id,
        mm_per_px=get_mm_per_px(),
        areas_mm2=areas_mm2,
        areas_cm2=areas_cm2,
        areas_px=areas_px,
        lengths_mm=lengths_mm,
        lengths_cm=lengths_cm,
        lengths_px=lengths_px,
        image_url=base_url + image_url,
        json_url=base_url + json_url,
        csv_url=base_url + csv_url,
    )


def _get_paths(job_id: str):
    out_dir = RUNS_DIR / job_id
    image_path = out_dir / "result.png"
    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"
    return out_dir, image_path, json_path, csv_path


@app.get("/api/download/image/{job_id}")
async def download_image(job_id: str):
    _, image_path, _, _ = _get_paths(job_id)
    if not image_path.is_file():
        return JSONResponse(status_code=404, content={"detail": "Image not found"})
    return FileResponse(str(image_path), media_type="image/png", filename=f"{job_id}_result.png")


@app.get("/api/download/json/{job_id}")
async def download_json(job_id: str):
    _, _, json_path, _ = _get_paths(job_id)
    if not json_path.is_file():
        return JSONResponse(status_code=404, content={"detail": "JSON not found"})
    return FileResponse(str(json_path), media_type="application/json", filename=f"{job_id}_results.json")


@app.get("/api/download/csv/{job_id}")
async def download_csv(job_id: str):
    _, _, _, csv_path = _get_paths(job_id)
    if not csv_path.is_file():
        return JSONResponse(status_code=404, content={"detail": "CSV not found"})
    return FileResponse(str(csv_path), media_type="text/csv", filename=f"{job_id}_results.csv")


@app.post("/api/archive/{kind}")
async def download_archive(kind: str, body: ArchiveRequest):
    """Собрать zip-архив по списку job_ids.

    kind: 'image' | 'json' | 'csv'
    """
    kind = kind.lower()
    if kind not in {"image", "json", "csv"}:
        return JSONResponse(status_code=400, content={"detail": "Invalid archive kind"})

    memfile = io.BytesIO()
    added = 0

    with zipfile.ZipFile(memfile, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for job_id in body.job_ids:
            _, image_path, json_path, csv_path = _get_paths(job_id)
            if kind == "image" and image_path.is_file():
                zf.write(image_path, arcname=f"{job_id}_result.png")
                added += 1
            elif kind == "json" and json_path.is_file():
                zf.write(json_path, arcname=f"{job_id}_results.json")
                added += 1
            elif kind == "csv" and csv_path.is_file():
                zf.write(csv_path, arcname=f"{job_id}_results.csv")
                added += 1

    if added == 0:
        return JSONResponse(status_code=404, content={"detail": "No files found for requested jobs"})

    memfile.seek(0)
    filename = f"plants_{kind}_results_{len(body.job_ids)}.zip"
    return StreamingResponse(
        memfile,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/calibration")
async def get_calibration():
    """Текущая калибровка: всегда стандартная (фиксированный масштаб)."""
    mm = get_mm_per_px()
    return {
        "mode": CALIBRATION_MODE_STANDARD,
        "mm_per_px": mm,
        "cm_per_px": mm / 10.0,
        "standard_mm_per_px": STANDARD_MM_PER_PX,
    }


@app.get("/api/jobs")
async def list_jobs(limit: int = 100):
    """Список последних анализов (job_id, file_name, areas, lengths) для таблицы."""
    import json
    jobs: List[dict] = []
    if not RUNS_DIR.is_dir():
        return {"jobs": []}
    subdirs = sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for subdir in subdirs[:limit]:
        if not subdir.is_dir():
            continue
        json_path = subdir / "results.json"
        if not json_path.is_file():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            amm = data.get("areas_mm2", {})
            acm = data.get("areas_cm2") or {k: v / 100.0 for k, v in amm.items()}
            lmm = data.get("lengths_mm", {})
            lcm = data.get("lengths_cm") or {k: v / 10.0 for k, v in lmm.items()}
            jobs.append({
                "job_id": data.get("job_id", subdir.name),
                "file_name": data.get("file_name", ""),
                "mm_per_px": data.get("mm_per_px"),
                "areas_px": data.get("areas_px", {}),
                "areas_mm2": amm,
                "areas_cm2": acm,
                "lengths_px": data.get("lengths_px", {}),
                "lengths_mm": lmm,
                "lengths_cm": lcm,
            })
        except Exception:
            continue
    return {"jobs": jobs}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    """Serve main frontend page on localhost:8000/."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.is_file():
        return JSONResponse(status_code=404, content={"detail": "index.html not found"})
    return FileResponse(str(index_path), media_type="text/html")


