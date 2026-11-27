"""
One frame of the full pipeline: GMC -> slot warp -> detect/track -> spatial engine -> render.

GMC (ORB + RANSAC, CPU) and detection (GPU) don't depend on each other, so GMC runs on a
worker thread while the tracker waits on the GPU. OpenCV releases the GIL, so the two really
overlap. Only the scheduling changes: every stage gets the same inputs as before.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.motion_compensator import CameraMotionCompensator


class FramePipeline:
    def __init__(self, engine, tracker, visualizer, max_features: int = 500):
        self.engine = engine
        self.tracker = tracker
        self.visualizer = visualizer
        self.gmc = CameraMotionCompensator(max_features=max_features)
        self.base_slots = [dict(s) for s in engine.slots]
        self.base_lane_zones = [dict(lz) for lz in engine.lane_zones]
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gmc")

    def process(self, frame: np.ndarray, t: float, fps_display: float = 30.0
                ) -> Tuple[np.ndarray, Dict, List[Dict]]:
        h, w = frame.shape[:2]
        gmc_job = self._pool.submit(self.gmc.update, frame)
        tracked = self.tracker.track(frame)          # GPU, overlaps with GMC
        gmc_job.result()

        zone_info = self.gmc.get_active_zone_info()
        comp_slots = self.gmc.transform_polygons(self.base_slots, frame_width=w, frame_height=h)
        self.engine.set_dynamic_slots(comp_slots)
        self.visualizer.set_dynamic_slots(comp_slots)
        if self.base_lane_zones:
            self.engine.set_dynamic_lane_zones(
                self.gmc.transform_polygons(self.base_lane_zones, frame_width=w, frame_height=h))

        analytics = self.engine.update(tracked, current_time=t, camera_dx=self.gmc.camera_dx,
                                       camera_dy=self.gmc.camera_dy, frame=frame,
                                       ground_H=self.gmc.accumulated_H)
        vis = self.visualizer.render(frame, analytics, tracked, zone_info=zone_info, fps=fps_display)
        return vis, analytics, tracked

    def close(self):
        self._pool.shutdown(wait=False)
