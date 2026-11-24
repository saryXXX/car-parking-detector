"""
Tiled vehicle detector for top-down / drone footage.

Stock COCO YOLO misses top-down cars at full-frame resolution (they are small and look
like "cell phones" to it). Running it on overlapping high-res tiles and accepting any
vehicle-sized box fixes most of that until an aerial-trained model (VisDrone/CARPK)
is available -- pass one via `model_path` and set `class_agnostic=False`.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision
from ultralytics import YOLO

# COCO classes that a top-down car gets mistaken for, plus the real vehicle classes.
COCO_TOPDOWN_VEHICLE = {"car", "truck", "bus", "cell phone", "remote", "suitcase", "skateboard",
                        "motorcycle", "tv", "laptop", "book", "keyboard", "parking meter"}
VEHICLE_NAMES = {"car", "van", "truck", "bus", "motor", "motorcycle", "vehicle"}


class AerialVehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", tile_frac: float = 1 / 3, overlap: float = 0.2,
                 imgsz: int = 1024, conf: float = 0.05, nms_iou: float = 0.4,
                 min_side_frac: float = 0.016, max_side_frac: float = 0.085,
                 class_agnostic: bool | None = None):
        self.model = YOLO(model_path)
        names = self.model.names
        if class_agnostic is None:  # aerial-trained models already know "car"
            class_agnostic = "cell phone" in names.values()
        allowed = COCO_TOPDOWN_VEHICLE if class_agnostic else VEHICLE_NAMES
        self.class_ids = [i for i, n in names.items() if n.lower() in allowed]
        # Aerial-trained weights (geo-trax, VisDrone) see small top-down cars on the full
        # frame: no tiling, normal confidence. Tiling is only the COCO fallback.
        self.aerial_trained = not class_agnostic
        if self.aerial_trained:
            tile_frac, conf, imgsz = 1.0, max(conf, 0.25), max(imgsz, 1280)
        self.tile_frac, self.overlap, self.imgsz, self.conf, self.nms_iou = tile_frac, overlap, imgsz, conf, nms_iou
        # Vehicle size limits, as a fraction of the frame width (4K drone: ~60-330 px long side)
        self.min_side_frac, self.max_side_frac = min_side_frac, max_side_frac

    def _tiles(self, H: int, W: int):
        t = min(int(round(W * self.tile_frac)), H, W)  # 1280 frame -> 427 px tiles
        step = int(t * (1 - self.overlap))
        ys = sorted(set(list(range(0, max(1, H - t), step)) + [H - t]))
        xs = sorted(set(list(range(0, max(1, W - t), step)) + [W - t]))
        return [(x, y, t) for y in ys for x in xs]

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (boxes xyxy float32 [N,4] in frame pixels, scores [N])."""
        H, W = frame.shape[:2]
        if self.tile_frac >= 1.0:
            r = self.model(frame, imgsz=self.imgsz, conf=self.conf, classes=self.class_ids,
                           max_det=1000, verbose=False)[0]
            return self._size_filter(r.boxes.xyxy.cpu().numpy().astype(np.float32),
                                     r.boxes.conf.cpu().numpy(), W)
        tiles = self._tiles(H, W)
        results = self.model([frame[y:y + t, x:x + t] for x, y, t in tiles], imgsz=self.imgsz,
                             conf=self.conf, classes=self.class_ids, verbose=False)
        boxes, scores = [], []
        for (x, y, _), r in zip(tiles, results):
            if len(r.boxes):
                boxes.append(r.boxes.xyxy.cpu().numpy() + [x, y, x, y])
                scores.append(r.boxes.conf.cpu().numpy())
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        b = torch.tensor(np.concatenate(boxes), dtype=torch.float32)
        s = torch.tensor(np.concatenate(scores), dtype=torch.float32)
        keep = torchvision.ops.nms(b, s, self.nms_iou)
        return self._size_filter(b[keep].numpy(), s[keep].numpy(), W)

    def _size_filter(self, b: np.ndarray, s: np.ndarray, W: int) -> tuple[np.ndarray, np.ndarray]:
        """Drop boxes too small/large to be a vehicle (pedestrians, merged rows, trees)."""
        w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        long_side, short_side = np.maximum(w, h), np.minimum(w, h)
        ok = (long_side > self.min_side_frac * W * 1.6) & (long_side < self.max_side_frac * W) \
             & (short_side > self.min_side_frac * W * 0.7)
        return b[ok], s[ok]
