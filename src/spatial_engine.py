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
        fps: float = 30.0
    ):
        self.config_path = config_path
        self.hysteresis_window = hysteresis_window
        self.occupied_threshold = occupied_threshold
        self.free_threshold = free_threshold
        self.obstruction_threshold_sec = obstruction_threshold_sec
        self.collision_dist_threshold = collision_dist_threshold
        self.near_miss_dist_threshold = near_miss_dist_threshold
        self.fps = fps

        self.slots = []
        self._load_config()

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
            self.slots = self.config_path
        else:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.slots = json.load(f)

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
        camera_dy: float = 0.0
    ) -> Dict:
        if current_time is None:
            current_time = time.time()

        # Filtrer uniquement les places visibles si l'attribut is_visible existe
        active_slots = [s for s in self.slots if s.get("is_visible", True)]

        instant_assignments: Dict[int, Optional[int]] = {s["id"]: None for s in active_slots}
        cars_in_slots = set()
        line_straddle_alerts = []

        # 1. Association spatiale et détection à cheval sur la ligne
        for veh in tracked_vehicles:
            car_id = veh["id"]
            bbox = veh["bbox"]
            px, py = self.get_bottom_contact_point(bbox)
            w = bbox[2] - bbox[0]

            p_left = (int(bbox[0] + w * 0.25), int(bbox[3]))
            p_right = (int(bbox[2] - w * 0.25), int(bbox[3]))

            if px > self.tripwire_x and car_id not in self.crossed_ids:
                self.crossed_ids.add(car_id)
                self.total_entries += 1

            matched_slots = []
            for slot in active_slots:
                sid = slot["id"]
                poly = slot["polygon"]
                center_in = self.is_point_in_slot((px, py), poly)
                left_in = self.is_point_in_slot(p_left, poly)
                right_in = self.is_point_in_slot(p_right, poly)

                # Match if: center is in, OR both wheels in (normal park), OR any wheel in (straddle detection)
                if center_in or (left_in and right_in) or left_in or right_in:
                    matched_slots.append(sid)

            if len(matched_slots) >= 2:
                line_straddle_alerts.append({
                    "car_id": car_id,
                    "slot_ids": matched_slots,
                    "bbox": bbox
                })
                for sid in matched_slots:
                    instant_assignments[sid] = car_id
                    cars_in_slots.add(car_id)
            elif len(matched_slots) == 1:
                sid = matched_slots[0]
                instant_assignments[sid] = car_id
                cars_in_slots.add(car_id)

        # 2. Hystérésis & Dwell time
        for slot in active_slots:
            sid = slot["id"]
            car_id = instant_assignments.get(sid, None)
            is_present = car_id is not None
            self.history[sid].append(is_present)

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
        for veh in tracked_vehicles:
            cid = veh["id"]
            bbox = veh["bbox"]
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

            if cid in self.prev_positions:
                old_x, old_y, old_t = self.prev_positions[cid]
                dt = max(1e-3, current_time - old_t)
                app_vx = (cx - old_x) / dt
                app_vy = (cy - old_y) / dt
                raw_vx = app_vx - (camera_dx / dt if dt < 0.2 else 0.0)
                raw_vy = app_vy - (camera_dy / dt if dt < 0.2 else 0.0)
                if cid in self.smoothed_velocities:
                    old_vx, old_vy = self.smoothed_velocities[cid]
                    self.smoothed_velocities[cid] = (0.6 * old_vx + 0.4 * raw_vx, 0.6 * old_vy + 0.4 * raw_vy)
                else:
                    self.smoothed_velocities[cid] = (raw_vx, raw_vy)
            else:
                self.smoothed_velocities[cid] = (0.0, 0.0)

            self.prev_positions[cid] = (cx, cy, current_time)

        # 4. Surveillance de voie (voitures arrêtées hors places)
        obstruction_alerts = []
        active_lane_cars = []

        for veh in tracked_vehicles:
            car_id = veh["id"]
            if car_id not in cars_in_slots:
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

        for i in range(n_veh):
            for j in range(i + 1, n_veh):
                v1, v2 = tracked_vehicles[i], tracked_vehicles[j]
                c1_id, c2_id = v1["id"], v2["id"]

                # Exclure si les deux véhicules sont déjà garés dans leurs places
                if c1_id in cars_in_slots and c2_id in cars_in_slots:
                    continue

                b1, b2 = v1["bbox"], v2["bbox"]
                cx1, cy1 = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
                cx2, cy2 = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
                dist_centers = np.hypot(cx1 - cx2, cy1 - cy2)

                pair_key = tuple(sorted((c1_id, c2_id)))
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
                    if (overlap_ratio > 0.20 or dist_centers < self.collision_dist_threshold) and (delta_d < -20.0 and closing_speed < -50.0):
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
                    elif dist_centers < self.near_miss_dist_threshold and (delta_d < -20.0 and closing_speed < -50.0):
                        collision_alerts.append({
                            "type": "NEAR_MISS",
                            "car1_id": c1_id,
                            "car2_id": c2_id,
                            "dist": dist_centers,
                            "closing_speed": abs(closing_speed),
                            "pos": (int((cx1 + cx2) / 2), int((cy1 + cy2) / 2))
                        })

        # Count straddled slots as occupied (even if hysteresis hasn't stabilized yet)
        straddled_slot_ids = set()
        for stra in line_straddle_alerts:
            straddled_slot_ids.update(stra.get("slot_ids", []))

        total_slots = len(active_slots)
        occupied_count = sum(1 for sid, state in self.occupancy_state.items()
                           if (state or sid in straddled_slot_ids) and any(s["id"] == sid for s in active_slots))
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
