"""
Compare occupancy modes (rule / cnn / fusion) against hand-labelled ground truth.

Usage:
  python tools/eval_occupancy.py --video VIDEO --config configs/parking_slots.json \
      --labels ground_truth.csv --weights runs/classify/train/weights/best.pt

ground_truth.csv columns: frame,slot_id,occupied (0/1). Only labelled frames are scored,
but ALL frames up to the last labelled one are fed through SpatialEngine.update() so the
hysteresis state is realistic at each labelled point.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `src.*` imports

import cv2

from src.slot_classifier import SlotClassifier
from src.spatial_engine import SpatialEngine


def load_labels(path):
    by_frame = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_frame[int(row["frame"])][int(row["slot_id"])] = int(row["occupied"])
    return by_frame


def run_mode(video_path, config_path, labels, mode, weights):
    classifier = SlotClassifier(weights, every_n_frames=1) if mode != "rule" and weights else None
    engine = SpatialEngine(config_path=config_path, occupancy_mode=mode, slot_classifier=classifier)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    last_frame = max(labels)

    y_true, y_pred = [], []
    flips = defaultdict(int)
    prev_state = {}
    frame_idx = 0
    while frame_idx <= last_frame:
        ok, frame = cap.read()
        if not ok:
            break
        # NOTE: tracked_vehicles would normally come from YOLO+ByteTrack; for this
        # evaluation harness, plug in your tracker's output here per frame.
        tracked_vehicles = get_tracked_vehicles_for_frame(frame_idx)
        result = engine.update(tracked_vehicles, current_time=frame_idx / fps, frame=frame)

        for sid, state in engine.occupancy_state.items():
            if sid in prev_state and prev_state[sid] != state:
                flips[sid] += 1
            prev_state[sid] = state

        if frame_idx in labels:
            for sid, gt in labels[frame_idx].items():
                y_true.append(gt)
                y_pred.append(int(engine.occupancy_state.get(sid, False)))
        frame_idx += 1
    cap.release()

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    minutes = (last_frame / fps) / 60.0
    flips_per_min = (sum(flips.values()) / max(1, len(engine.occupancy_state))) / max(1e-9, minutes)
    return dict(mode=mode, n=len(y_true), accuracy=acc, precision=prec, recall=rec, f1=f1,
                flips_per_slot_per_min=flips_per_min)


def get_tracked_vehicles_for_frame(frame_idx):
    """Placeholder: wire this to src/tracker.py output for your evaluation video."""
    raise NotImplementedError(
        "Plug in your tracker's per-frame output here (list of {'id', 'bbox'} dicts)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--weights", default=None, help="Classifier weights for cnn/fusion modes")
    ap.add_argument("--modes", nargs="*", default=["rule", "cnn", "fusion"])
    args = ap.parse_args()

    labels = load_labels(args.labels)
    rows = []
    for mode in args.modes:
        if mode != "rule" and not args.weights:
            print(f"[skip] mode={mode} needs --weights")
            continue
        rows.append(run_mode(args.video, args.config, labels, mode, args.weights))

    print("\n| mode | n | accuracy | precision | recall | f1 | flips/slot/min |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['mode']} | {r['n']} | {r['accuracy']:.3f} | {r['precision']:.3f} | "
              f"{r['recall']:.3f} | {r['f1']:.3f} | {r['flips_per_slot_per_min']:.2f} |")


if __name__ == "__main__":
    main()
