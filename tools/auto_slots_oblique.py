"""
Extend a slot map for an oblique (pole / building) camera with one slot per parked car.

Every tracked car whose ground point stays still (in the GMC-stabilised frame) for at least
--min-still frames is parked; its slot is the ground band of its segmentation mask (the
bottom of the silhouette, where the wheels touch the bay), mapped back to frame 0. Slots
that land on an existing (hand-traced) slot are skipped, so traced bays keep priority and
still provide the empty spaces.

    python tools/auto_slots_oblique.py --video 14292633_3840_2160_30fps.mp4 \
        --base configs/commercial_parking_zones.json --out configs/commercial_parking_zones.json

Writes a preview image next to --out. Only finds spaces that hold a car at some point in the
clip; trace empty rows by hand (or with paint lines) if they matter.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.motion_compensator import CameraMotionCompensator  # noqa: E402
from src.tracker import VehicleTracker  # noqa: E402

W, H = 1280, 720


def ground_band_polygon(mask_xy: np.ndarray, band_frac: float) -> np.ndarray | None:
    """Convex hull of the lowest `band_frac` of the mask in every column (frame coords)."""
    pts = np.asarray(mask_xy, dtype=np.float32)
    if len(pts) < 3:
        return None
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [(pts - [x0, y0]).astype(np.int32)], 1)
    cols = m.any(0)
    if not cols.any():
        return None
    lowest = h - 1 - np.argmax(m[::-1], axis=0)
    band = max(2, int(band_frac * h))
    keep = m.astype(bool) & (np.arange(h)[:, None] >= lowest - band) & cols
    ys, xs = np.nonzero(keep)
    if len(xs) < 10:
        return None
    hull = cv2.convexHull(np.stack([xs + x0, ys + y0], 1).astype(np.int32))
    hull = cv2.approxPolyDP(hull, 1.5, True).reshape(-1, 2)
    return hull.astype(np.float32) if len(hull) >= 3 else None


def to_frame0(poly: np.ndarray, H_0_to_k: np.ndarray) -> np.ndarray:
    Hinv = np.linalg.inv(H_0_to_k.astype(np.float64))
    return cv2.perspectiveTransform(poly.reshape(-1, 1, 2).astype(np.float64), Hinv).reshape(-1, 2)


def poly_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection area / area of the smaller polygon."""
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    small = min(cv2.contourArea(a.astype(np.float32)), cv2.contourArea(b.astype(np.float32)))
    return inter / small if small > 0 else 0.0


def still_flags(o: list, lag: int, tol_frac: float, min_lag: int = 5, smooth: int = 9) -> np.ndarray:
    """Per observation: did the car's ground point stay put over the last ~`lag` frames?

    Compared locally in image space: the point seen `lag` frames earlier is carried to the
    current frame by the camera motion (H_k @ H_{k-lag}^-1), which avoids extrapolating far
    outside frame 0. Short tracks (cars leaving the view) fall back to the longest lag
    >= min_lag they have. Boxes cut by the frame border never count.

    The tolerance is loose on purpose: the bottom-centre of a box is not a fixed ground point
    for a nearby 3D car -- it slides by up to ~0.2 car heights over half a second as the
    viewing angle changes -- while a car creeping through the aisle moves >= ~0.3. A rolling
    median keeps one jittery frame from splitting a parked run."""
    by_frame = {ob["frame"]: ob for ob in o}
    res = np.full(len(o), np.nan)
    for k, ob in enumerate(o):
        if ob["border"]:
            continue
        prev = next((by_frame[ob["frame"] - L] for L in range(lag, min_lag - 1, -1)
                     if ob["frame"] - L in by_frame and not by_frame[ob["frame"] - L]["border"]), None)
        if prev is None:
            continue
        M = ob["H"].astype(np.float64) @ np.linalg.inv(prev["H"].astype(np.float64))
        p = cv2.perspectiveTransform(prev["g"].reshape(1, 1, 2), M).reshape(2)
        res[k] = np.hypot(*(p - ob["g"])) / ob["h"]
    half = smooth // 2
    with np.errstate(all="ignore"):
        med = np.array([np.nanmedian(res[max(0, k - half):k + half + 1])
                        if not np.isnan(res[k]) else np.nan for k in range(len(o))])
    return np.nan_to_num(med, nan=np.inf) < tol_frac


def longest_run(flags: np.ndarray) -> tuple[int, int]:
    best, start = (0, 0), None
    for k, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = k
        elif not f and start is not None:
            if k - start > best[1] - best[0]:
                best = (start, k)
            start = None
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--base", help="Existing config whose slots keep priority")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-still", type=int, default=15, help="Frames a car must stay still")
    ap.add_argument("--min-height", type=float, default=16, help="Ignore cars smaller than this (px)")
    ap.add_argument("--band", type=float, default=0.3, help="Ground band, fraction of mask height")
    ap.add_argument("--max-dwell", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO input size (far rows need 1280)")
    args = ap.parse_args()

    base = json.loads(Path(args.base).read_text()) if args.base else {"slots": []}
    if isinstance(base, list):
        base = {"slots": base}
    base["slots"] = [s for s in base["slots"] if s.get("zone") != "ZONE_P"]  # re-runs replace old output
    # Driving lanes, plus hand-marked "no_slot_zones" (roads, kiosks the detector takes for cars).
    lanes = [np.array(z["polygon"], np.float32) for z in base.get("lane_zones", []) + base.get("no_slot_zones", [])]
    base_polys = [np.array(s["polygon"], np.float32) for s in base["slots"]]

    tracker = VehicleTracker(model_name=args.model, conf_threshold=args.conf, imgsz=args.imgsz)
    gmc = CameraMotionCompensator(max_features=500)
    cap = cv2.VideoCapture(args.video)
    obs = defaultdict(list)  # id -> [(ground point f0, band polygon f0, touches_border, frame)]
    frame0, idx = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H))
        frame0 = frame.copy() if frame0 is None else frame0
        gmc.update(frame)
        Hk = gmc.accumulated_H
        for v in tracker.track(frame):
            x1, y1, x2, y2 = v["bbox"]
            if y2 - y1 < args.min_height or v.get("mask") is None:
                continue
            if (x2 - x1) > 3.5 * (y2 - y1):  # far too long for a car seen at an angle (hedges, kerbs)
                continue
            band = ground_band_polygon(np.array(v["mask"]), args.band)
            if band is None:
                continue
            border = x1 < 3 or y1 < 3 or x2 > W - 3 or y2 > H - 3
            obs[v["id"]].append({"frame": idx, "g": np.array([(x1 + x2) / 2, y2]), "H": Hk.copy(),
                                 "band": band, "border": border, "h": y2 - y1})
        idx += 1
    cap.release()
    print(f"[INFO] {idx} frames, {len(obs)} tracks")

    cands = []
    for tid, o in obs.items():
        if len(o) < args.min_still:
            continue
        i, j = longest_run(still_flags(o, lag=15, tol_frac=0.25))
        if j - i < args.min_still:
            continue
        ob = o[(i + j) // 2]  # mid-window frame, never border-cut (still frames exclude those)
        cands.append((j - i, to_frame0(ob["band"], ob["H"])))
    print(f"[INFO] parked cars: {len(cands)}")

    # Longest-observed first; drop cars waiting in a driving lane and anything landing on a
    # base slot or an already-kept slot (partial overlap would make the car "straddle" both).
    kept = []
    for _, poly in sorted(cands, key=lambda c: -c[0]):
        c = tuple(float(v) for v in poly.mean(0))
        if any(cv2.pointPolygonTest(z, c, False) >= 0 for z in lanes):
            continue
        if any(poly_overlap(poly, p) > 0.1 for p in base_polys + kept):
            continue
        kept.append(poly)
    print(f"[INFO] new slots: {len(kept)} (on top of {len(base_polys)} existing)")

    slots = [dict(s) for s in base["slots"]]
    kept.sort(key=lambda p: (round(p[:, 1].max() / 60), p[:, 0].mean()))  # rough row-major order
    for n, poly in enumerate(kept, 1):
        slots.append({"name": f"Slot_P{n}", "zone": "ZONE_P",
                      "polygon": np.round(poly).astype(int).tolist(), "max_dwell_minutes": args.max_dwell})
    for i, s in enumerate(slots, 1):
        s["id"] = i
    out = dict(base)
    out["slots"] = [{"id": s["id"], **{k: v for k, v in s.items() if k != "id"}} for s in slots]
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    prev = frame0.copy()
    for s in out["slots"]:
        auto = s["zone"] == "ZONE_P"
        cv2.polylines(prev, [np.array(s["polygon"], np.int32)], True, (0, 200, 255) if auto else (0, 255, 0), 2)
    prev_path = Path(args.out).with_suffix(".preview.jpg")
    cv2.imwrite(str(prev_path), prev)
    print(f"[DONE] {args.out}  ({len(out['slots'])} slots, preview: {prev_path})")


if __name__ == "__main__":
    main()
