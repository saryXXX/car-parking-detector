"""
Baseline: run Ultralytics' built-in ParkingManagement solution on one of our videos,
using our existing slot polygons, so we can compare it against our own pipeline.

    python tools/baseline_ultralytics_parking.py \
        --video 4196207-uhd_3840_2160_24fps.mp4 \
        --config configs/drone_parking_lot.json \
        --model yolo11s.pt --max-frames 150

Outputs (in output/baseline/):
  <video>_<model>.mp4   annotated video
  <video>_<model>.csv   per-frame: occupied, free, n_vehicle_detections, ms_per_frame

Note: ParkingManagement assumes a FIXED camera. On moving-camera footage the slot
polygons stay put while the ground moves, so occupancy drifts over time.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO, solutions

# Matched by name so COCO (car=2) and VisDrone (car=3, van=4, ...) weights both work
VEHICLE_NAMES = {"car", "van", "truck", "bus", "motorcycle", "motor", "vehicle", "occupied_slot"}


def vehicle_class_ids(model_path: str) -> list[int]:
    names = YOLO(model_path).names
    ids = [i for i, n in names.items() if n.lower() in VEHICLE_NAMES]
    print(f"[INFO] vehicle classes: {[names[i] for i in ids]}")
    return ids


def convert_slots(config_path: Path, out_path: Path, scale: float) -> int:
    """Our config: [{"polygon": [[x,y],...]}, ...] -> Ultralytics: [{"points": [[x,y],...]}, ...]."""
    slots = json.loads(config_path.read_text(encoding="utf-8"))
    converted = [{"points": [[int(x * scale), int(y * scale)] for x, y in s["polygon"]]} for s in slots]
    out_path.write_text(json.dumps(converted), encoding="utf-8")
    return len(converted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", required=True, help="Our slot JSON (polygon coords at --width)")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--width", type=int, default=1280, help="Frame width the slot polygons were drawn at")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="output/baseline")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.video).stem}_{Path(args.model).stem}"
    slots_json = out_dir / f"{stem}_slots.json"
    n_slots = convert_slots(Path(args.config), slots_json, 1.0)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open {args.video}"
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    size = (args.width, int(h0 * args.width / w0))

    parking = solutions.ParkingManagement(
        model=args.model, json_file=str(slots_json), show=False,
        classes=vehicle_class_ids(args.model), conf=args.conf, verbose=False,
    )
    writer = cv2.VideoWriter(str(out_dir / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    rows, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        res = parking(frame)
        ms = (time.perf_counter() - t0) * 1000
        n_det = len(parking.boxes) if getattr(parking, "boxes", None) is not None else -1
        rows.append([i, res.filled_slots, res.available_slots, n_det, round(ms, 1)])
        writer.write(res.plot_im)
        i += 1
    cap.release()
    writer.release()

    with open(out_dir / f"{stem}.csv", "w", newline="") as f:
        csv.writer(f).writerows([["frame", "occupied", "free", "detections", "ms"]] + rows)

    if rows:
        avg_ms = sum(r[4] for r in rows[1:] or rows) / max(1, len(rows) - 1)
        print(f"[DONE] {i} frames | {n_slots} slots | avg {avg_ms:.0f} ms/frame (~{1000/avg_ms:.1f} FPS)")
        print(f"       detections/frame: first={rows[0][3]} last={rows[-1][3]} | "
              f"occupied: first={rows[0][1]} last={rows[-1][1]}")
        print(f"       -> {out_dir / (stem + '.mp4')}")


if __name__ == "__main__":
    main()
