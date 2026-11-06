"""
Global Motion Compensator (GMC) & Multi-Zone Adaptability
Smart Parking Monitoring & Spatial Analytics

Features:
1. Estimates camera ego-motion (pan/tilt/zoom) via ORB feature matching + RANSAC homography.
2. Dynamically warps parking slot polygons (Augmented Reality ground locking).
3. Detects when slots move out of the camera field of view and smoothly deactivates them.
4. Multi-Zone Adaptability: Automatically switches active zones and slot definitions
   (e.g., Zone A: Entrance -> Zone B: Central Aisle -> Zone C: East Wing) as the camera pans.
5. Provides true ground velocity by subtracting camera ego-motion.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class CameraMotionCompensator:
    def __init__(self, max_features: int = 500):
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kp = None
        self.prev_des = None

        # Cumulative transformation from frame 0 to current frame
        self.accumulated_H = np.eye(3, dtype=np.float32)
        # Instantaneous camera displacement (pixels per frame)
        self.camera_dx = 0.0
        self.camera_dy = 0.0
        # Cumulative displacement
        self.total_dx = 0.0
        self.total_dy = 0.0

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes a new frame, computes camera motion step, and accumulates global homography.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp, self.prev_des = self.orb.detectAndCompute(gray, None)
            return self.accumulated_H

        kp, des = self.orb.detectAndCompute(gray, None)

        if des is not None and self.prev_des is not None and len(des) > 10 and len(self.prev_des) > 10:
            matches = self.bf.match(self.prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)[:120]

            if len(matches) >= 8:
                src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                H_step, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

                if H_step is not None:
                    # Instantaneous displacement
                    self.camera_dx = float(H_step[0, 2])
                    self.camera_dy = float(H_step[1, 2])
                    # Accumulate transformation
                    self.accumulated_H = np.dot(H_step, self.accumulated_H)
                    self.total_dx = float(self.accumulated_H[0, 2])
                    self.total_dy = float(self.accumulated_H[1, 2])

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des
        return self.accumulated_H

    def get_active_zone_info(self) -> Dict:
        """
        Determines the active parking zone based on camera horizontal pan displacement (total_dx).
        """
        dx = self.total_dx
        if dx > -180.0:
            zone_id = "ZONE_A"
            zone_name = "Zone A : Entree & Voie Sud"
            description = "Surveillance des places A1 a A5"
        elif dx >= -420.0:
            zone_id = "ZONE_B"
            zone_name = "Zone B : Allee Centrale & PureGym"
            description = "Surveillance de l'allee principale et places B1 a B6"
        else:
            zone_id = "ZONE_C"
            zone_name = "Zone C : Secteur Batiments Est"
            description = "Surveillance du secteur est et places C1 a C6"

        return {
            "zone_id": zone_id,
            "zone_name": zone_name,
            "description": description,
            "pan_dx": self.total_dx,
            "pan_dy": self.total_dy
        }

    def transform_polygons(
        self,
        slot_definitions: List[Dict],
        frame_width: int = 1280,
        frame_height: int = 720
    ) -> List[Dict]:
        """
        Applies camera motion compensation to parking slot polygons.
        Automatically tags out-of-bounds slots as is_visible=False so they don't persist on screen.
        """
        compensated_slots = []
        for slot in slot_definitions:
            poly = np.array(slot["polygon"], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(poly, self.accumulated_H)
            new_poly = [[int(pt[0][0]), int(pt[0][1])] for pt in transformed]

            # Compute centroid
            cx = int(np.mean([p[0] for p in new_poly]))
            cy = int(np.mean([p[1] for p in new_poly]))

            # Check if at least 2 vertices are inside the active frame viewport
            in_view_pts = sum(1 for p in new_poly if 0 <= p[0] <= frame_width and 0 <= p[1] <= frame_height)
            is_visible = (in_view_pts >= 2) and (-100 <= cx <= frame_width + 100) and (-100 <= cy <= frame_height + 100)

            compensated_slots.append({
                "id": slot["id"],
                "name": slot["name"],
                "polygon": new_poly,
                "centroid": (cx, cy),
                "is_visible": is_visible,
                "zone": slot.get("zone", "ZONE_A"),
                "max_dwell_minutes": slot.get("max_dwell_minutes", 15)
            })

        return compensated_slots
