"""
Spatial & Temporal Analytics Engine for Smart Parking
Features:
1. Dynamic Slot Geometry with Global Motion Compensation (GMC).
2. Out-of-bounds slot awareness (hides off-screen slots).
3. Temporal hysteresis and Individual Dwell Time.
4. True Ground Velocity (compensating for camera pan/tilt).
5. Physics-Based Collision & Near-Miss Detection:
   - Evaluates windowed inter-vehicle distance closing rate (Delta d <= -20 px over 0.3s).
   - Evaluates relative velocity (v_rel >= 25 px/s).
   - Excludes static cars parked side-by-side (0 false alarms).
6. Line Straddling Detection with multi-wheel sampling.
"""

import json
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class SpatialEngine:
    def __init__(
        self,
        config_path: str = "configs/parking_slots.json",
        hysteresis_window: int = 30,
        occupied_threshold: float = 0.8,
        free_threshold: float = 0.2,
        obstruction_threshold_sec: float = 2.5,
        collision_dist_threshold: float = 75.0,
        near_miss_dist_threshold: float = 120.0,
        fps: float = 30.0,
        occupancy_mode: str = "rule",
        slot_classifier: Optional[object] = None,
        collision_mode: str = "rule",
        collision_model: Optional[object] = None,
        px_per_metre: Optional[float] = None,
        top_down: bool = False,
        oblique: bool = False,
    ):
        self.config_path = config_path
        # Oblique real camera (pole / building mounted, looking across the lot): a car's
        # roof projects onto the bay behind it and boxes of cars at different depths overlap
        # through occlusion. Slot and contact tests then use the ground band of each car.
        self.oblique = oblique and not top_down
        # Top-down (drone) view: the whole bbox is the car's footprint. The oblique-camera
        # sampling (lower 2/3 + bottom edge) puts a row of samples on the boundary with the
        # next slot and flags every tightly parked car as straddling.
        self.top_down = top_down
        # Pixel thresholds below (collision/near-miss distance, 20 px closing) were tuned for
        # ~120 px-long cars. In top-down mode they are rescaled by the observed car length.
        self.REFERENCE_CAR_LENGTH_PX = 120.0
        self.size_scale = 1.0
        # Below this ground speed (car lengths / s) a car counts as stopped for straddling.
        self.STOPPED_SPEED_LEN_PER_S = 0.3
        self.hysteresis_window = hysteresis_window
        self.occupied_threshold = occupied_threshold
        self.free_threshold = free_threshold
        self.obstruction_threshold_sec = obstruction_threshold_sec
        self.collision_dist_threshold = collision_dist_threshold
        self.near_miss_dist_threshold = near_miss_dist_threshold
        self.fps = fps
        assert occupancy_mode in ("rule", "cnn", "fusion")
        self.occupancy_mode = occupancy_mode
        self.slot_classifier = slot_classifier
        assert collision_mode in ("rule", "ml")
        self.collision_mode = collision_mode
        self.collision_model = collision_model

        self.slots = []
        self.lane_zones: List[Dict] = []
        self._config_px_per_metre: Optional[float] = None
        self._load_config()
        # Explicit arg > calibrated value from the config file (src/calibrate.py --ref-metres) > 75px/m fallback.
        self.px_per_metre = px_per_metre or self._config_px_per_metre or 75.0
        self.ml_pair_distance_px = 8.0 * self.px_per_metre  # 8 m pair-distance gate for ML collision mode

        self.history: Dict[int, deque] = {
            s["id"]: deque(maxlen=self.hysteresis_window) for s in self.slots
        }
        self.occupancy_state: Dict[int, bool] = {s["id"]: False for s in self.slots}
        self.dwell_data: Dict[int, Dict] = {
            s["id"]: {"car_id": None, "occupied_since": None, "elapsed_seconds": 0.0}
            for s in self.slots
        }
        self.lane_monitoring: Dict[int, Dict] = {}
        self.tripwire_x = 140
        self.crossed_ids = set()
        self.total_entries = 0

        # Historique des positions et vitesses lissées
        self.prev_positions: Dict[int, Tuple[float, float, float]] = {}
        self.smoothed_velocities: Dict[int, Tuple[float, float]] = {}
        # Historique fenêtré des distances (pair -> deque de (timestamp, dist))
        # Immunise le calcul contre le jitter sous-pixel (+/- 2px de YOLO)
        self.pair_distance_windows: Dict[Tuple[int, int], deque] = {}

    def _load_config(self):
        if isinstance(self.config_path, list):
            raw = self.config_path
        else:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        # Two config shapes: a flat list of slots (legacy), or {"px_per_metre": ..., "slots": [...],
        # "lane_zones": [...]} written by src/calibrate.py once a camera has been scale-calibrated.
        if isinstance(raw, dict) and "slots" in raw:
            self.slots = raw["slots"]
            self._config_px_per_metre = raw.get("px_per_metre")
            self.lane_zones = raw.get("lane_zones", [])
        else:
            self.slots = raw
            self.lane_zones = []

    def set_dynamic_lane_zones(self, updated_lane_zones: List[Dict]):
        """Camera-motion-compensated lane zone polygons (see motion_compensator.transform_polygons)."""
        self.lane_zones = updated_lane_zones

    def set_dynamic_slots(self, updated_slots: List[Dict]):
        """Met à jour dynamiquement les places après compensation de mouvement caméra."""
        self.slots = updated_slots
        for s in self.slots:
            sid = s["id"]
            if sid not in self.history:
                self.history[sid] = deque(maxlen=self.hysteresis_window)
                self.occupancy_state[sid] = False
                self.dwell_data[sid] = {"car_id": None, "occupied_since": None, "elapsed_seconds": 0.0}

    @staticmethod
    def get_bottom_contact_point(bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2.0), int(y2)

    def is_point_in_slot(self, point: Tuple[int, int], slot_polygon: List[List[int]]) -> bool:
        pts = np.array(slot_polygon, np.int32)
        dist = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), measureDist=False)
        return dist >= 0

    @staticmethod
    def get_footprint_samples(bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
        """3x3 grid over the lower 2/3 of the bbox (the car's ground footprint, avoiding
        the windshield/roof area which often leans into the neighboring slot in oblique
        camera angles). Used to measure how MUCH of a vehicle sits in a slot, rather than
        a single sample point -- a point sitting exactly on a shared slot boundary passes
        pointPolygonTest (>=0, "on edge" counts as inside) for BOTH neighboring polygons,
        which previously made every normally-parked car near a slot line register as
        straddling on every frame.
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        ys = (y1 + h * 0.55, y1 + h * 0.78, y2)
        xs = (x1 + w * 0.15, x1 + w * 0.5, x1 + w * 0.85)
        return [(x, y) for y in ys for x in xs]

    @staticmethod
    def get_topdown_samples(bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
        """3x3 grid over the inner 60% of the bbox, away from edges shared with neighbours."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        return [(x1 + w * fx, y1 + h * fy) for fy in (0.2, 0.5, 0.8) for fx in (0.2, 0.5, 0.8)]

    def slot_footprint_overlap(self, bbox: Tuple[float, float, float, float], slot_polygon: List[List[int]]) -> float:
        """Fraction of the vehicle's footprint samples that fall inside this slot polygon."""
        samples = self.get_topdown_samples(bbox) if self.top_down else self.get_footprint_samples(bbox)
        hits = sum(1 for pt in samples if self.is_point_in_slot(pt, slot_polygon))
        return hits / len(samples)

    @staticmethod
    def slot_mask_overlap(vehicle_mask: List[List[float]], slot_polygon: List[List[int]],
                          ground_band: Optional[float] = None) -> float:
        """Fraction of the vehicle's actual pixel mask (from YOLO instance segmentation)
        that falls inside this slot polygon -- the real footprint, not an approximation
        from the bounding box. Rasterized on a small local canvas covering just the two
        polygons' combined bounds, so this stays cheap even with many slots per frame.

        ground_band (oblique cameras): keep only the lowest `ground_band` fraction of the
        mask's height in every column. Seen at an angle, a car's roof and windscreen project
        onto the bay *behind* it; only the bottom silhouette (wheels, sills) touches its own bay.
        """
        veh_pts = np.array(vehicle_mask, dtype=np.float32)
        slot_pts = np.array(slot_polygon, dtype=np.float32)

        all_pts = np.vstack([veh_pts, slot_pts])
        min_x, min_y = all_pts.min(axis=0)
        max_x, max_y = all_pts.max(axis=0)
        pad = 2
        w = max(1, int(max_x - min_x) + 2 * pad)
        h = max(1, int(max_y - min_y) + 2 * pad)
        if w * h > 4_000_000:  # degenerate/far-apart polygons: skip rasterization, no overlap
            return 0.0

        offset = np.array([min_x - pad, min_y - pad], dtype=np.float32)
        veh_local = (veh_pts - offset).astype(np.int32)
        slot_local = (slot_pts - offset).astype(np.int32)

        veh_mask = np.zeros((h, w), dtype=np.uint8)
        slot_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(veh_mask, [veh_local], 1)
        cv2.fillPoly(slot_mask, [slot_local], 1)

        if ground_band:
            cols = veh_mask.any(axis=0)
            if cols.any():
                band = max(2, int(ground_band * (veh_pts[:, 1].max() - veh_pts[:, 1].min())))
                lowest = h - 1 - np.argmax(veh_mask[::-1], axis=0)  # bottom-most mask row per column
                rows = np.arange(h)[:, None]
                veh_mask &= ((rows >= lowest - band) & cols).astype(np.uint8)

        veh_area = int(veh_mask.sum())
        if veh_area == 0:
            return 0.0
        inter_area = int((veh_mask & slot_mask).sum())
        return inter_area / veh_area

    @staticmethod
    def time_to_overlap(b1, v1, b2, v2, horizon: float, margin: float) -> Optional[float]:
        """Earliest t in [0, horizon] s at which b1 and b2 (each grown by `margin` px), moving
        at constant velocities v1/v2 (px/s), overlap -- or None. Swept-box test per axis."""
        lo, hi = 0.0, horizon
        for axis in (0, 1):
            a_min, a_max = b1[axis] - margin, b1[axis + 2] + margin
            c_min, c_max = b2[axis] - margin, b2[axis + 2] + margin
            vr = v1[axis] - v2[axis]
            if abs(vr) < 1e-6:
                if a_max < c_min or a_min > c_max:
                    return None
                continue
            t1, t2 = (c_min - a_max) / vr, (c_max - a_min) / vr
            lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
            if lo > hi:
                return None
        return lo

    def compute_bbox_overlap(self, boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        minArea = min((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]), (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return (interArea / float(minArea)) if minArea > 0 else 0.0

    def update(
        self,
        tracked_vehicles: List[Dict],
        current_time: Optional[float] = None,
        camera_dx: float = 0.0,
        camera_dy: float = 0.0,
        frame: Optional[np.ndarray] = None,
        ground_H: Optional[np.ndarray] = None,
    ) -> Dict:
        """ground_H: homography reference frame -> current frame from the GMC. When given,
        speeds are measured in the stabilised reference frame, which removes camera pan and
        perspective; camera_dx/dy alone only remove a uniform shift."""
        if current_time is None:
            current_time = time.time()

        # Filtrer uniquement les places visibles si l'attribut is_visible existe
        active_slots = [s for s in self.slots if s.get("is_visible", True)]

        # p(occupied) per slot from the CNN classifier, batched once per update().
        # Only computed when a frame is supplied and mode requires it; the classifier
        # itself caches across every_n_frames, so this stays cheap.
        p_cnn: Dict[int, float] = {}
        if self.occupancy_mode in ("cnn", "fusion") and self.slot_classifier is not None and frame is not None:
            p_cnn = self.slot_classifier.predict(frame, active_slots)

        instant_assignments: Dict[int, Optional[int]] = {s["id"]: None for s in active_slots}
        cars_in_slots = set()
        line_straddle_alerts = []

        # Cheap per-slot bounding rect, computed once per update() (not per vehicle) so the
        # expensive mask/footprint overlap test below only runs for slots a vehicle could
        # plausibly be in, instead of all N slots x every tracked vehicle every frame.
        slot_bounds = np.array([
            (min(p[0] for p in s["polygon"]), min(p[1] for p in s["polygon"]),
             max(p[0] for p in s["polygon"]), max(p[1] for p in s["polygon"]))
            for s in active_slots], dtype=np.float32).reshape(-1, 4)

        # 1. Association spatiale et détection à cheval sur la ligne
        # Thresholds: a slot is "present" once a meaningful share of the vehicle's
        # footprint sits in it (not just a boundary-touching sample point), and straddling
        # requires genuine presence in 2+ slots at once -- not one dominant slot plus a
        # sliver that only grazes the line into a neighbor.
        PRESENCE_THRESHOLD = 0.30
        STRADDLE_THRESHOLD = 0.30
        for veh in tracked_vehicles:
            car_id = veh["id"]
            bbox = veh["bbox"]
            mask = veh.get("mask")
            px, py = self.get_bottom_contact_point(bbox)

            if px > self.tripwire_x and car_id not in self.crossed_ids:
                self.crossed_ids.add(car_id)
                self.total_entries += 1

            bx1, by1, bx2, by2 = bbox
            overlaps = []
            # Quick bbox-vs-slot-bbox rejection (vectorised) before the precise, costlier test.
            near = np.nonzero((slot_bounds[:, 2] >= bx1) & (slot_bounds[:, 0] <= bx2) &
                              (slot_bounds[:, 3] >= by1) & (slot_bounds[:, 1] <= by2))[0]
            for idx in near:
                slot = active_slots[idx]
                sid = slot["id"]
                poly = slot["polygon"]
                if mask:
                    frac = self.slot_mask_overlap(mask, poly, ground_band=0.25 if self.oblique else None)
                else:
                    frac = self.slot_footprint_overlap(bbox, poly)
                if frac >= PRESENCE_THRESHOLD:
                    overlaps.append((sid, frac))

            straddling_slots = [sid for sid, frac in overlaps if frac >= STRADDLE_THRESHOLD]
            # A car still manoeuvring crosses bay lines on its way in or out; only a stopped
            # car is "parked across two bays". Speed is from the previous frame (section 3).
            vx, vy = self.smoothed_velocities.get(car_id, (0.0, 0.0))
            car_len = max(bx2 - bx1, by2 - by1)
            is_moving = np.hypot(vx, vy) > self.STOPPED_SPEED_LEN_PER_S * car_len

            if len(straddling_slots) >= 2 and is_moving:
                # Count it in its best slot only, without raising a straddle alert.
                sid = max(overlaps, key=lambda pair: pair[1])[0]
                instant_assignments[sid] = car_id
                cars_in_slots.add(car_id)
            elif len(straddling_slots) >= 2:
                line_straddle_alerts.append({
                    "car_id": car_id,
                    "slot_ids": straddling_slots,
                    "bbox": bbox
                })
                for sid in straddling_slots:
                    instant_assignments[sid] = car_id
                    cars_in_slots.add(car_id)
            elif overlaps:
                # Single best-matching slot: normal parking, no straddle.
                sid = max(overlaps, key=lambda pair: pair[1])[0]
                instant_assignments[sid] = car_id
                cars_in_slots.add(car_id)

        # 2. Hystérésis & Dwell time
        for slot in active_slots:
            sid = slot["id"]
            car_id = instant_assignments.get(sid, None)
            is_present = car_id is not None

            if self.occupancy_mode == "rule" or sid not in p_cnn:
                push_value = float(is_present)
            elif self.occupancy_mode == "cnn":
                push_value = p_cnn[sid]
            else:  # fusion
                push_value = 0.6 * p_cnn[sid] + 0.4 * float(is_present)
            self.history[sid].append(push_value)

            ratio = sum(self.history[sid]) / len(self.history[sid])
            prev_state = self.occupancy_state.get(sid, False)

            if ratio >= self.occupied_threshold:
                new_state = True
            elif ratio <= self.free_threshold:
                new_state = False
            else:
                new_state = prev_state

            self.occupancy_state[sid] = new_state
            dwell = self.dwell_data[sid]

            if new_state:
                if not prev_state or dwell["occupied_since"] is None:
                    dwell["occupied_since"] = current_time
                    dwell["car_id"] = car_id
                dwell["elapsed_seconds"] = current_time - dwell["occupied_since"]
                if car_id is not None:
                    dwell["car_id"] = car_id
            else:
                dwell["occupied_since"] = None
                dwell["car_id"] = None
                dwell["elapsed_seconds"] = 0.0

        # 3. Calcul des vitesses relatives au sol (soustraction caméra + lissage EMA)
        H_inv = None
        if ground_H is not None:
            try:
                H_inv = np.linalg.inv(np.asarray(ground_H, dtype=np.float64))
            except np.linalg.LinAlgError:
                H_inv = None
        for veh in tracked_vehicles:
            cid = veh["id"]
            bbox = veh["bbox"]
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            if H_inv is not None:
                # Track the ground-contact point (bottom centre) in the stabilised frame.
                p = H_inv @ np.array([cx, bbox[3], 1.0])
                cx, cy = p[0] / p[2], p[1] / p[2]

            if cid in self.prev_positions:
                old_x, old_y, old_t = self.prev_positions[cid]
                dt = max(1e-3, current_time - old_t)
                app_vx = (cx - old_x) / dt
                app_vy = (cy - old_y) / dt
                comp = H_inv is None and dt < 0.2  # stabilised coords need no shift correction
                raw_vx = app_vx - (camera_dx / dt if comp else 0.0)
                raw_vy = app_vy - (camera_dy / dt if comp else 0.0)
                if cid in self.smoothed_velocities:
                    old_vx, old_vy = self.smoothed_velocities[cid]
                    self.smoothed_velocities[cid] = (0.6 * old_vx + 0.4 * raw_vx, 0.6 * old_vy + 0.4 * raw_vy)
                else:
                    self.smoothed_velocities[cid] = (raw_vx, raw_vy)
            else:
                self.smoothed_velocities[cid] = (0.0, 0.0)

            self.prev_positions[cid] = (cx, cy, current_time)

        # 4. Surveillance de voie (voitures arrêtées dans la voie de circulation)
        # When lane_zones are configured (see src/calibrate.py), a car only counts as a lane
        # obstruction candidate if it's actually inside a driving-lane polygon -- not merely
        # "not matched to one of our configured slots". On a real lot we rarely have every
        # single parking space defined; without this check, any car legitimately parked in an
        # unconfigured space got flagged as "blocking the lane" just because it wasn't in our
        # (incomplete) slot list. Configs with no lane_zones keep the old permissive behavior
        # (built for the single-lane synthetic simulations, where it's unambiguous).
        obstruction_alerts = []
        active_lane_cars = []

        for veh in tracked_vehicles:
            car_id = veh["id"]
            if car_id in cars_in_slots:
                continue
            if self.lane_zones:
                px, py = self.get_bottom_contact_point(veh["bbox"])
                if not any(self.is_point_in_slot((px, py), lz["polygon"]) for lz in self.lane_zones):
                    continue
            bbox = veh["bbox"]
            vx, vy = self.smoothed_velocities.get(car_id, (0.0, 0.0))
            spd = np.hypot(vx, vy)

            if car_id not in self.lane_monitoring:
                self.lane_monitoring[car_id] = {
                    "first_seen": current_time,
                    "duration": 0.0,
                    "is_stopped": False
                }
            else:
                rec = self.lane_monitoring[car_id]
                if spd < 15.0:
                    rec["duration"] = current_time - rec["first_seen"]
                    rec["is_stopped"] = True
                    if rec["duration"] >= self.obstruction_threshold_sec:
                        obstruction_alerts.append({
                            "car_id": car_id,
                            "duration": rec["duration"],
                            "bbox": bbox
                        })
                else:
                    rec["first_seen"] = current_time
                    rec["duration"] = 0.0
                    rec["is_stopped"] = False

            active_lane_cars.append({
                "car_id": car_id,
                "bbox": bbox,
                "duration": self.lane_monitoring[car_id]["duration"],
                "is_stopped": self.lane_monitoring[car_id]["is_stopped"]
            })

        # 5. Détection de Collision & Near-Miss PHYSIQUE ROBUSTE :
        # RÈGLE 1 : Les voitures garées côte à côte ne s'accrochent pas.
        # RÈGLE 2 : Rapprochement fenêtré sur 8 à 15 frames (Delta d <= -20px sur 0.3s).
        collision_alerts = []
        n_veh = len(tracked_vehicles)

        if self.collision_mode == "ml" and self.collision_model is not None:
            for veh in tracked_vehicles:
                self.collision_model.update_track(veh["id"], veh["bbox"], current_time)
            close_pairs = []
            for i in range(n_veh):
                for j in range(i + 1, n_veh):
                    v1, v2 = tracked_vehicles[i], tracked_vehicles[j]
                    b1, b2 = v1["bbox"], v2["bbox"]
                    cx1, cy1 = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
                    cx2, cy2 = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
                    if np.hypot(cx1 - cx2, cy1 - cy2) < self.ml_pair_distance_px:
                        close_pairs.append((v1["id"], v2["id"]))
            ml_alerts = self.collision_model.predict(close_pairs, fps=self.fps, t=current_time)
            veh_by_id = {v["id"]: v for v in tracked_vehicles}
            for a in ml_alerts:
                v1, v2 = veh_by_id.get(a["car1_id"]), veh_by_id.get(a["car2_id"])
                if v1 is None or v2 is None:
                    continue
                b1, b2 = v1["bbox"], v2["bbox"]
                cx = int((b1[0] + b1[2] + b2[0] + b2[2]) / 4)
                cy = int((b1[1] + b1[3] + b2[1] + b2[3]) / 4)
                collision_alerts.append({
                    "type": a["type"], "car1_id": a["car1_id"], "car2_id": a["car2_id"],
                    "dist": a["distance_m"] * 75.0,  # back to pixel-equivalent for display parity
                    "closing_speed": None, "pos": (cx, cy), "source": "ml",
                    "p_collision": a["p_collision"], "ttc_s": a["ttc_s"],
                })
            n_veh = 0  # skip the rule loop below entirely in ml mode

        if self.top_down and tracked_vehicles:
            lengths = [max(v["bbox"][2] - v["bbox"][0], v["bbox"][3] - v["bbox"][1]) for v in tracked_vehicles]
            target = float(np.median(lengths)) / self.REFERENCE_CAR_LENGTH_PX
            self.size_scale = 0.9 * self.size_scale + 0.1 * target if self.size_scale != 1.0 else target
        k = self.size_scale
        collision_dist, near_miss_dist = self.collision_dist_threshold * k, self.near_miss_dist_threshold * k
        min_closing_px, min_closing_speed = 20.0 * k, 50.0 * k

        # Vectorised pre-filter: only pairs within reach of a near-miss are checked in Python
        # (a 240-car drone lot is ~29k pairs/frame otherwise). A pair that leaves the radius
        # loses its distance window, so it re-enters with fresh history.
        candidate_pairs = []
        if n_veh > 1:
            boxes = np.array([v["bbox"] for v in tracked_vehicles], dtype=np.float32)
            gx = np.maximum(0, np.maximum(boxes[:, None, 0], boxes[None, :, 0]) - np.minimum(boxes[:, None, 2], boxes[None, :, 2]))
            gy = np.maximum(0, np.maximum(boxes[:, None, 1], boxes[None, :, 1]) - np.minimum(boxes[:, None, 3], boxes[None, :, 3]))
            reach = max(near_miss_dist, collision_dist) * 2.0
            ii, jj = np.nonzero(np.triu(np.hypot(gx, gy) < reach, k=1))
            candidate_pairs = list(zip(ii.tolist(), jj.tolist()))
        updated_pairs = set()

        for i, j in candidate_pairs:
            v1, v2 = tracked_vehicles[i], tracked_vehicles[j]
            c1_id, c2_id = v1["id"], v2["id"]

            # Exclure si les deux véhicules sont déjà garés dans leurs places
            if c1_id in cars_in_slots and c2_id in cars_in_slots:
                continue

            b1, b2 = v1["bbox"], v2["bbox"]
            cx1, cy1 = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
            cx2, cy2 = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
            dist_centers = np.hypot(cx1 - cx2, cy1 - cy2)
            if self.top_down:
                # Bird's-eye view: what matters is the gap between the car outlines, not
                # their centres -- a car driving down the aisle past a parked one keeps a
                # full lane width of clearance even though their centres are close.
                gap_x = max(0.0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
                gap_y = max(0.0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
                gap = float(np.hypot(gap_x, gap_y))
                car_len = self.REFERENCE_CAR_LENGTH_PX * k
                # Axis-aligned boxes of angled cars touch long before the cars do, so a
                # COLLISION needs real box overlap. A NEAR_MISS needs the predicted paths
                # to actually meet within 1 s -- a car driving past a parked bus closes
                # the distance but its path never crosses the bus.
                is_contact_zone = False
                ttc = self.time_to_overlap(b1, self.smoothed_velocities.get(c1_id, (0.0, 0.0)),
                                           b2, self.smoothed_velocities.get(c2_id, (0.0, 0.0)),
                                           horizon=1.0, margin=0.05 * car_len)
                is_near_zone = ttc is not None and gap < 0.5 * car_len
                # Track the outline gap (not centre distance) for the closing test: a car
                # driving alongside another keeps a constant gap even as centres approach.
                dist_centers = gap
            elif self.oblique:
                # Seen at an angle, a car passing behind a parked one overlaps it on screen.
                # Compare ground bands (bottom 30% of each box): they only meet when the
                # cars are at the same depth, i.e. actually close on the ground.
                b1 = (b1[0], b1[3] - 0.3 * (b1[3] - b1[1]), b1[2], b1[3])
                b2 = (b2[0], b2[3] - 0.3 * (b2[3] - b2[1]), b2[2], b2[3])
                gap_x = max(0.0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
                gap_y = max(0.0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
                gap = float(np.hypot(gap_x, gap_y))
                is_contact_zone = False
                is_near_zone = gap < 0.3 * min(b1[2] - b1[0], b2[2] - b2[0])
                dist_centers = gap
            else:
                is_contact_zone = dist_centers < collision_dist
                is_near_zone = dist_centers < near_miss_dist

            pair_key = tuple(sorted((c1_id, c2_id)))
            updated_pairs.add(pair_key)
            if pair_key not in self.pair_distance_windows:
                self.pair_distance_windows[pair_key] = deque(maxlen=15)
            self.pair_distance_windows[pair_key].append((current_time, dist_centers))

            win = self.pair_distance_windows[pair_key]
            overlap_ratio = self.compute_bbox_overlap(b1, b2)

            # Calcul du rapprochement fenêtré (sur au moins 6 frames, soit ~0.2s)
            if len(win) >= 6:
                t_old, d_old = win[0]
                dt = max(1e-3, current_time - t_old)
                delta_d = dist_centers - d_old
                closing_speed = delta_d / dt

                # CONDITION COLLISION PHYSIQUE :
                # Rapprochement réel net (>= 20px) ET (contact direct ou overlap significatif)
                if (overlap_ratio > 0.20 or is_contact_zone) and (delta_d < -min_closing_px and closing_speed < -min_closing_speed):
                    collision_alerts.append({
                        "type": "COLLISION",
                        "car1_id": c1_id,
                        "car2_id": c2_id,
                        "dist": dist_centers,
                        "closing_speed": abs(closing_speed),
                        "pos": (int((cx1 + cx2) / 2), int((cy1 + cy2) / 2))
                    })
                # CONDITION NEAR-MISS :
                # Distance d'alerte (< 120px) ET rapprochement rapide net (delta_d < -20px)
                elif is_near_zone and (delta_d < -min_closing_px and closing_speed < -min_closing_speed):
                    collision_alerts.append({
                        "type": "NEAR_MISS",
                        "car1_id": c1_id,
                        "car2_id": c2_id,
                        "dist": dist_centers,
                        "closing_speed": abs(closing_speed),
                        "pos": (int((cx1 + cx2) / 2), int((cy1 + cy2) / 2))
                    })

        if self.collision_mode != "ml" or self.collision_model is None:
            self.pair_distance_windows = {k: w for k, w in self.pair_distance_windows.items() if k in updated_pairs}

        # Count straddled slots as occupied (even if hysteresis hasn't stabilized yet)
        straddled_slot_ids = set()
        for stra in line_straddle_alerts:
            straddled_slot_ids.update(stra.get("slot_ids", []))

        total_slots = len(active_slots)
        active_ids = {s["id"] for s in active_slots}  # set lookup: O(slots) instead of O(slots^2)
        occupied_count = sum(1 for sid, state in self.occupancy_state.items()
                           if (state or sid in straddled_slot_ids) and sid in active_ids)
        free_count = total_slots - occupied_count
        occupancy_rate = (occupied_count / total_slots * 100.0) if total_slots > 0 else 0.0

        return {
            "total_slots": total_slots,
            "occupied_count": occupied_count,
            "free_count": free_count,
            "occupancy_rate": occupancy_rate,
            "states": self.occupancy_state,
            "dwell_data": self.dwell_data,
            "obstruction_alerts": obstruction_alerts,
            "active_lane_cars": active_lane_cars,
            "line_straddle_alerts": line_straddle_alerts,
            "collision_alerts": collision_alerts,
            "total_entries": self.total_entries
        }
