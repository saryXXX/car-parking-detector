"""
Interactive ground-truth labeler for your own camera (live URL or recorded clip).

Samples frames at a fixed interval, overlays the calibrated slot polygons, and lets
you click each slot to cycle empty -> occupied -> skip. Writes a labels.csv compatible
with tools/build_slot_dataset.py --source video, so you can:

  1. python src/calibrate.py --source <video_or_rtsp_url> --output configs/my_cam.json
  2. python tools/label_slots_interactive.py --source <video_or_rtsp_url> \
         --config configs/my_cam.json --out my_cam_labels.csv --every-sec 5
  3. python tools/build_slot_dataset.py --source video --video <video_or_rtsp_url> \
         --config configs/my_cam.json --labels my_cam_labels.csv --group my_cam \
         --out datasets/slots_mycam
  4. python tools/finetune_slot_classifier.py --data datasets/slots_mycam \
         --base runs/classify/train/weights/best.pt

Controls (per sampled frame):
  Click a slot polygon : cycle its label  empty (red) -> occupied (green) -> unset (gray)
  [N] / [Space]         : accept current labels for this frame, advance to next sample
  [S]                   : save labels.csv and quit
  [Q] / [ESC]            : quit without saving the current (unaccepted) frame
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

UNSET, EMPTY, OCCUPIED = -1, 0, 1
COLORS = {UNSET: (140, 140, 140), EMPTY: (0, 0, 220), OCCUPIED: (0, 200, 0)}


def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(np.array(poly, dtype=np.float32), pt, False) >= 0


class InteractiveLabeler:
    def __init__(self, source, slots, every_sec, max_frames):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.every_n = max(1, int(round(every_sec * self.fps)))
        self.max_frames = max_frames
        self.slots = slots
        self.labels = {s["id"]: UNSET for s in slots}
        self.rows = []  # (frame_idx, slot_id, occupied)
        self.window = "Slot labeler -- click to cycle, N/space=accept, S=save&quit, Q=quit"

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for s in self.slots:
            if point_in_poly((x, y), s["polygon"]):
                cur = self.labels[s["id"]]
                self.labels[s["id"]] = {UNSET: EMPTY, EMPTY: OCCUPIED, OCCUPIED: UNSET}[cur]
                break

    def draw(self, frame):
        img = frame.copy()
        for s in self.slots:
            pts = np.array(s["polygon"], np.int32).reshape((-1, 1, 2))
            color = COLORS[self.labels[s["id"]]]
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
            cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
            cx = int(np.mean([p[0] for p in s["polygon"]]))
            cy = int(np.mean([p[1] for p in s["polygon"]]))
            cv2.putText(img, str(s["id"]), (cx - 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h - 30), (w, h), (20, 20, 20), -1)
        n_labeled = sum(1 for v in self.labels.values() if v != UNSET)
        cv2.putText(img, f"labeled {n_labeled}/{len(self.slots)} slots this frame | rows so far: {len(self.rows)}",
                    (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return img

    def run(self, out_path):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.mouse_callback)
        frame_idx = 0
        n_sampled = 0
        while self.max_frames == 0 or n_sampled < self.max_frames:
            ok, frame = self.cap.read()
            if not ok:
                break
            if frame_idx % self.every_n != 0:
                frame_idx += 1
                continue

            self.labels = {s["id"]: UNSET for s in self.slots}
            while True:
                cv2.imshow(self.window, self.draw(frame))
                key = cv2.waitKey(20) & 0xFF
                if key in (ord('n'), ord(' ')):
                    for sid, lab in self.labels.items():
                        if lab != UNSET:
                            self.rows.append((frame_idx, sid, lab))
                    n_sampled += 1
                    break
                if key == ord('s'):
                    for sid, lab in self.labels.items():
                        if lab != UNSET:
                            self.rows.append((frame_idx, sid, lab))
                    self._save(out_path)
                    cv2.destroyAllWindows()
                    return
                if key in (ord('q'), 27):
                    cv2.destroyAllWindows()
                    self._save(out_path)
                    return
            frame_idx += 1
        self._save(out_path)
        cv2.destroyAllWindows()

    def _save(self, out_path):
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "slot_id", "occupied"])
            w.writerows(self.rows)
        print(f"[OK] Wrote {len(self.rows)} labeled rows to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Video file path or RTSP/HTTP stream URL")
    ap.add_argument("--config", required=True, type=Path, help="Slot polygon config (from src/calibrate.py)")
    ap.add_argument("--out", required=True, type=Path, help="Output labels.csv")
    ap.add_argument("--every-sec", type=float, default=5.0, help="Seconds between sampled frames")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after labeling this many frames (0 = until source ends or you quit)")
    args = ap.parse_args()

    slots = json.load(open(args.config, encoding="utf-8"))
    labeler = InteractiveLabeler(args.source, slots, args.every_sec, args.max_frames)
    labeler.run(args.out)


if __name__ == "__main__":
    main()
