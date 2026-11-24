"""
Build a slot map for a (near-)static top-down camera from the parked cars themselves.

  1. Detect vehicles on N sampled frames (tiled detector, see src/aerial_detector.py).
  2. Cars that stay in the same place on most sampled frames are parked -> one slot each.
  3. Slots are grouped into columns; gaps along a column that fit whole slots at the
     column's regular pitch become EMPTY slots.

    python tools/auto_slots_from_detections.py \
        --video 4196207-uhd_3840_2160_24fps.mp4 --out configs/drone_parking_lot_auto.json

Output uses the same schema as configs/*.json, in --width coordinates (default 1280).
Always eyeball the preview image it writes before using the config.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aerial_detector import AerialVehicleDetector  # noqa: E402
from src.motion_compensator import CameraMotionCompensator  # noqa: E402


def to_frame0(boxes: np.ndarray, H_0_to_k: np.ndarray) -> np.ndarray:
    """Map boxes seen in frame k back into frame-0 coordinates (slots are stored in frame 0,
    and the app's GMC warps them forward from there)."""
    if not len(boxes):
        return boxes
    Hinv = np.linalg.inv(H_0_to_k)
    corners = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], 1)
    warped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float32), Hinv).reshape(-1, 4, 2)
    return np.concatenate([warped.min(1), warped.max(1)], 1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def persistent_boxes(per_frame: list[np.ndarray], min_ratio: float, iou_thr: float = 0.4) -> np.ndarray:
    """Greedy clustering of boxes across frames; keep clusters present in >= min_ratio of frames."""
    clusters: list[list[np.ndarray]] = []
    centers = np.zeros((0, 4))
    for boxes in per_frame:
        for b in boxes:
            if len(centers):
                ious = iou_matrix(b[None], centers)[0]
                j = int(ious.argmax())
                if ious[j] >= iou_thr:
                    clusters[j].append(b)
                    centers[j] = np.median(clusters[j], axis=0)
                    continue
            clusters.append([b])
            centers = np.vstack([centers, b])
    need = max(1, int(np.ceil(min_ratio * len(per_frame))))
    return np.array([np.median(c, axis=0) for c in clusters if len(c) >= need])


def group_columns(boxes: np.ndarray, gap_frac: float = 0.45) -> list[np.ndarray]:
    """Split boxes into vertical columns by x-centre (cars parked side by side, stacked in y)."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    order = np.argsort(cx)
    med_w = np.median(boxes[:, 2] - boxes[:, 0])
    cols, cur = [], [order[0]]
    for i, j in zip(order[:-1], order[1:]):
        if cx[j] - cx[i] > gap_frac * med_w:
            cols.append(boxes[cur]); cur = []
        cur.append(j)
    cols.append(boxes[cur])
    return cols


def drop_outliers(boxes: np.ndarray) -> np.ndarray:
    """Remove boxes whose area is far from a typical car (trees, buses, merged detections)."""
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    med = np.median(area)
    return boxes[(area > 0.45 * med) & (area < 1.9 * med)]


def vegetation_mask(img: np.ndarray) -> np.ndarray:
    """Grass / trees: green hue with real saturation (asphalt and cars are grey-ish)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] >= 22) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40)


def gap_crosses_vegetation(veg: np.ndarray, x1: float, x2: float, ya: float, yb: float,
                           band_frac: float = 0.25) -> bool:
    """True if any horizontal band between two cars of a column is mostly grass (an island)."""
    H, W = veg.shape
    xa, xb = int(max(0, x1)), int(min(W, x2))
    ya, yb = int(max(0, ya)), int(min(H, yb))
    if xb <= xa or yb <= ya:
        return False
    rows = veg[ya:yb, xa:xb].mean(axis=1)  # green fraction of each pixel row in the gap
    return bool((rows > band_frac).any())


def fill_column(col: np.ndarray, all_boxes: np.ndarray, pitch: float, max_fill: int = 3,
                min_slots: int = 4, veg: np.ndarray | None = None) -> list[tuple[np.ndarray, bool]]:
    """Return [(box, occupied)] for a column, inserting empty slots into small regular gaps.

    Gaps bigger than `max_fill` slots are treated as aisles and left alone, gaps that cross a
    grass island (`veg` mask) end the row, and an inferred slot is skipped if it overlaps any
    detected car.
    """
    col = col[np.argsort((col[:, 1] + col[:, 3]) / 2)]
    out = [(b, True) for b in col]
    if len(col) < min_slots:  # too short to be a parking row (road edge, stray detections)
        return out
    cy = (col[:, 1] + col[:, 3]) / 2
    x1, x2 = np.median(col[:, 0]), np.median(col[:, 2])
    h = float(np.median(col[:, 3] - col[:, 1]))
    for k in range(1, len(col)):
        gap = cy[k] - cy[k - 1]
        n_missing = int(round(gap / pitch)) - 1
        if not 1 <= n_missing <= max_fill:
            continue
        if veg is not None and gap_crosses_vegetation(veg, x1, x2, col[k - 1, 3], col[k, 1]):
            continue
        for m in range(1, n_missing + 1):
            y = cy[k - 1] + m * gap / (n_missing + 1)
            cand = np.array([x1, y - h / 2, x2, y + h / 2])
            if iou_matrix(cand[None], all_boxes).max() < 0.15:
                out.append((cand, False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--samples", type=int, default=12, help="Frames sampled across the video")
    ap.add_argument("--min-ratio", type=float, default=0.5, help="Fraction of samples a car must appear in")
    ap.add_argument("--width", type=int, default=1280, help="Coordinate space of the output config")
    ap.add_argument("--max-dwell", type=int, default=60)
    args = ap.parse_args()

    det = AerialVehicleDetector(args.model)
    gmc = CameraMotionCompensator(max_features=500)  # same settings as the app
    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idx = set(np.linspace(0, n - 1, args.samples).astype(int).tolist())
    per_frame, ref = [], None
    # Work in --width coordinates (what the app runs at). Every frame goes through GMC so the
    # homography frame0 -> k is the one the app will use to warp these slots forward.
    for idx in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        scale = args.width / frame.shape[1]
        small = cv2.resize(frame, (args.width, int(round(frame.shape[0] * scale))))
        gmc.update(small)
        ref = small if ref is None else ref
        if idx not in sample_idx:
            continue
        boxes, _ = det(frame)
        per_frame.append(to_frame0(boxes * scale, gmc.accumulated_H.astype(np.float64)))
        print(f"  frame {idx:4d}: {len(boxes)} vehicles (camera offset {gmc.total_dx:+.1f}, {gmc.total_dy:+.1f} px)")
    cap.release()

    parked = drop_outliers(persistent_boxes(per_frame, args.min_ratio))
    print(f"[INFO] parked (persistent) vehicles: {len(parked)}")
    # Slot pitch along a row ~= one car width across (short side) plus the painted gap
    short = np.minimum(parked[:, 2] - parked[:, 0], parked[:, 3] - parked[:, 1])
    cols = group_columns(parked)
    diffs = np.concatenate([np.diff(np.sort((c[:, 1] + c[:, 3]) / 2)) for c in cols if len(c) > 1])
    diffs = diffs[(diffs > 0.8 * np.median(short)) & (diffs < 2.0 * np.median(short))]
    pitch = float(np.median(diffs))
    print(f"[INFO] slot pitch: {pitch:.1f}px")
    veg = vegetation_mask(ref)  # frame 0, same coordinates as the boxes
    slots = []
    for col in sorted(cols, key=lambda c: np.median(c[:, 0])):
        slots += fill_column(col, parked, pitch, veg=veg)
    n_empty = sum(1 for _, occ in slots if not occ)
    print(f"[INFO] slots: {len(slots)} ({n_empty} inferred empty)")

    scale = 1.0  # boxes are already in --width, frame-0 coordinates
    config = []
    for i, (b, _) in enumerate(slots, 1):
        x1, y1, x2, y2 = (b * scale).round().astype(int).tolist()
        config.append({"id": i, "name": f"Slot_{i:03d}",
                       "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                       "max_dwell_minutes": args.max_dwell})
    Path(args.out).write_text(json.dumps(config, indent=1), encoding="utf-8")

    prev = ref.copy()  # frame 0, which is what the slot coordinates refer to
    for s, (_, occ) in zip(config, slots):
        cv2.polylines(prev, [np.array(s["polygon"])], True, (0, 0, 255) if occ else (0, 255, 0), 1)
    prev_path = Path(args.out).with_suffix(".preview.jpg")
    cv2.imwrite(str(prev_path), prev)
    print(f"[DONE] {args.out}  (preview: {prev_path})")


if __name__ == "__main__":
    main()
