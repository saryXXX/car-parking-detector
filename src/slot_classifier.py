"""
Per-slot occupancy classifier.

Wraps a `yolo classify` model (occupied/empty, 2 classes) trained with
tools/build_slot_dataset.py + `yolo classify train`. Crops and perspective-rectifies
every visible slot from the current frame, runs ONE batched inference per update
(not one call per slot), and caches the result for `every_n_frames` frames so the
dashboard stays real-time even with many slots.
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

CROP_SIZE = (128, 256)  # (width, height), must match tools/build_slot_dataset.py


class SlotClassifier:
    def __init__(self, weights_path: str, device: str = "cpu", every_n_frames: int = 3,
                 imgsz: int = 224):
        from ultralytics import YOLO  # local import: optional dependency

        self.model = YOLO(weights_path)
        self.device = device
        self.every_n_frames = max(1, every_n_frames)
        self.imgsz = imgsz
        self._frame_count = 0
        self._last_result: Dict[int, float] = {}

        names = self.model.names  # {0: "empty", 1: "occupied"} or similar
        self._occupied_idx = next(i for i, n in names.items() if n.lower().startswith("occ"))

    @staticmethod
    def _rectify(frame: np.ndarray, polygon: List[List[int]]) -> Optional[np.ndarray]:
        if len(polygon) < 4:
            return None
        src = np.array(polygon[:4], dtype=np.float32)
        dst = np.array(
            [[0, 0], [CROP_SIZE[0] - 1, 0], [CROP_SIZE[0] - 1, CROP_SIZE[1] - 1], [0, CROP_SIZE[1] - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, CROP_SIZE)

    def predict(self, frame: np.ndarray, slots: List[Dict]) -> Dict[int, float]:
        """Return {slot_id: p(occupied)} for every slot in `slots`.

        Runs real inference only every `every_n_frames` calls; in between, returns
        the cached values from the last real inference (new slot ids default to 0.0
        until the next real pass).
        """
        self._frame_count += 1
        if self._last_result and self._frame_count % self.every_n_frames != 0:
            return {s["id"]: self._last_result.get(s["id"], 0.0) for s in slots}

        crops, ids = [], []
        for slot in slots:
            crop = self._rectify(frame, slot["polygon"])
            if crop is not None:
                crops.append(crop)
                ids.append(slot["id"])

        if not crops:
            return {}

        results = self.model.predict(crops, imgsz=self.imgsz, device=self.device, verbose=False)
        out: Dict[int, float] = {}
        for sid, res in zip(ids, results):
            out[sid] = float(res.probs.data[self._occupied_idx])

        self._last_result = out
        return {s["id"]: out.get(s["id"], 0.0) for s in slots}
