"""
Runtime wrapper around the trained LightGBM collision/near-miss model (add-on F2/F3,
runtime integration step). Applies the same 3-frame hysteresis idea used for slot
occupancy, so a single noisy frame does not flip the alert state.

NOTE on units: the model was trained on metric features from the synthetic simulator
(tools/simulate_trajectories.py + tools/train_collision_model.py). Converting real
tracked pixel positions to metres is add-on A (homography), which is not built yet.
Until then, `pixel_to_metre_scale` applies a single constant scale factor (the same
75 px -> 1.0 m rule-of-thumb used for the ported baseline) as a stand-in: it is NOT a
substitute for a calibrated homography and should be replaced once add-on A exists.
"""

import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np

from src.trajectory_features import FEATURE_NAMES, TrackHistory, TrackSample, pair_features

PX_PER_METRE_FALLBACK = 75.0  # matches SpatialEngine's collision_dist_threshold=75px == 1.0 m


class CollisionModel:
    def __init__(self, model_path: str, pixel_to_metre_scale: float = 1.0 / PX_PER_METRE_FALLBACK,
                 hysteresis_frames: int = 3):
        self.model = lgb.Booster(model_file=model_path)
        meta_path = Path(model_path).with_suffix(".json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.labels = self.meta.get("labels", ["NORMAL", "NEAR_MISS", "COLLISION"])
        self.p_collision_threshold = self.meta.get("alert_threshold_collision", 0.6)
        self.p_near_or_collision_threshold = self.meta.get("alert_threshold_near_or_collision", 0.5)
        self.scale = pixel_to_metre_scale  # metres per pixel
        self.hysteresis_frames = hysteresis_frames

        self._histories: Dict[int, TrackHistory] = {}
        self._pair_alert_streak: Dict[Tuple[int, int], int] = {}

    def _get_history(self, track_id: int) -> TrackHistory:
        if track_id not in self._histories:
            self._histories[track_id] = TrackHistory()
        return self._histories[track_id]

    def update_track(self, track_id: int, bbox, t: float):
        """Push one frame of a track's bottom-centre position (pixels) converted to metres."""
        x_px = (bbox[0] + bbox[2]) / 2.0
        y_px = bbox[3]
        hist = self._get_history(track_id)
        heading = hist.latest.heading if hist.samples else 0.0
        if hist.samples:
            dx, dy = x_px - hist.latest.x / self.scale, y_px - hist.latest.y / self.scale
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                heading = float(np.arctan2(dy, dx))
        hist.push(TrackSample(t=t, x=x_px * self.scale, y=y_px * self.scale, heading=heading))

    def predict(self, pairs: List[Tuple[int, int]], fps: float, t: float) -> List[Dict]:
        """pairs: list of (track_id_a, track_id_b) for tracks closer than ~8 m.
        Returns one dict per pair with the same key shape SpatialEngine's rule alerts use."""
        out = []
        for a_id, b_id in pairs:
            if a_id not in self._histories or b_id not in self._histories:
                continue
            hist_a, hist_b = self._histories[a_id], self._histories[b_id]
            if len(hist_a.samples) < 2 or len(hist_b.samples) < 2:
                continue
            feats = pair_features(hist_a, hist_b, fps=fps)
            X = np.array([[feats[name] for name in FEATURE_NAMES]])
            probs = self.model.predict(X)[0]  # [p_normal, p_near_miss, p_collision]
            p_near, p_coll = probs[1], probs[2]

            key = (min(a_id, b_id), max(a_id, b_id))
            is_alert = p_coll >= self.p_collision_threshold or (p_near + p_coll) >= self.p_near_or_collision_threshold
            self._pair_alert_streak[key] = self._pair_alert_streak.get(key, 0) + 1 if is_alert else 0

            if self._pair_alert_streak[key] >= self.hysteresis_frames:
                alert_type = "COLLISION" if p_coll >= self.p_collision_threshold else "NEAR_MISS"
                out.append({
                    "type": alert_type,
                    "car1_id": a_id,
                    "car2_id": b_id,
                    "p_collision": float(p_coll),
                    "p_near_miss": float(p_near),
                    "distance_m": feats["d"],
                    "ttc_s": feats["ttc"],
                    "source": "ml",
                })
        return out
