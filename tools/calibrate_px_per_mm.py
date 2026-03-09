from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PatternResult:
    pattern: tuple[int, int]  # (cols, rows) inner corners
    px_per_square_mean: float
    px_per_square_std: float
    n: int


def estimate_px_per_square(gray: np.ndarray, pattern: tuple[int, int]) -> PatternResult | None:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok or corners is None:
        return None

    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    cols, rows = pattern
    pts = corners.reshape(-1, 2).reshape(rows, cols, 2)

    dists: list[float] = []
    # horizontal edges
    for r in range(rows):
        for c in range(cols - 1):
            dists.append(float(np.linalg.norm(pts[r, c] - pts[r, c + 1])))
    # vertical edges
    for r in range(rows - 1):
        for c in range(cols):
            dists.append(float(np.linalg.norm(pts[r, c] - pts[r + 1, c])))

    arr = np.asarray(dists, dtype=np.float64)
    return PatternResult(
        pattern=pattern,
        px_per_square_mean=float(arr.mean()),
        px_per_square_std=float(arr.std()),
        n=int(arr.size),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to chessboard image")
    ap.add_argument("--square-mm", type=float, required=True, help="Real checker square size in millimeters")
    ap.add_argument(
        "--patterns",
        default="5x8,8x5,4x7,7x4,6x9,9x6",
        help="Comma-separated inner-corner patterns like '5x8,8x5'",
    )
    args = ap.parse_args()

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    patterns: list[tuple[int, int]] = []
    for token in str(args.patterns).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise SystemExit(f"Bad pattern token: {token}")
        a, b = token.split("x", 1)
        patterns.append((int(a), int(b)))

    results: list[PatternResult] = []
    for p in patterns:
        r = estimate_px_per_square(gray, p)
        if r is not None:
            results.append(r)

    if not results:
        print("NO_PATTERN_FOUND")
        return 2

    results.sort(key=lambda x: x.px_per_square_std)
    print("Found patterns (sorted by std):")
    for r in results:
        print(
            f"  pattern={r.pattern}  px_per_square_mean={r.px_per_square_mean:.3f}  "
            f"std={r.px_per_square_std:.3f}  n={r.n}"
        )

    best = results[0]
    px_per_mm = best.px_per_square_mean / float(args.square_mm)
    mm_per_px = float(args.square_mm) / best.px_per_square_mean

    print("\nBEST:")
    print(f"pattern={best.pattern}")
    print(f"px_per_mm={px_per_mm:.6f}")
    print(f"mm_per_px={mm_per_px:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

