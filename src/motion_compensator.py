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
    def __init__(self, max_features: int = 500, proc_width: int = 640):
        # Features are matched on a downscaled copy (ORB on 1280 px was ~45 ms/frame); the
        # homography is rescaled back to full-frame coordinates.
        self.proc_width = proc_width
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

    # Minimum RANSAC inliers for a homography to be trusted.
    MIN_INLIERS = 25

    def _match_homography(self, kp_a, des_a, kp_b, des_b) -> Optional[np.ndarray]:
        """Homography mapping points of image A onto image B, or None if unreliable."""
        if des_a is None or des_b is None or len(des_a) < 10 or len(des_b) < 10:
            return None
        matches = sorted(self.bf.match(des_a, des_b), key=lambda m: m.distance)[:200]
        if len(matches) < 8:
            return None
        src = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or mask is None or int(mask.sum()) < self.MIN_INLIERS:
            return None
        return H

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimates the homography frame0 -> current frame.

        Each frame is matched against a fixed KEYFRAME (not the previous frame): chaining
        frame-to-frame steps loses slow drift entirely -- a hovering drone drifting ~0.1 px per
        frame produced per-step homographies of ~identity and an accumulated motion of 0,
        while the ground had actually moved 15-25 px. The keyframe is only replaced when the
        view has moved too far to match it reliably (handheld pans), at which point we chain
        once from the previous frame and re-anchor.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s = min(1.0, self.proc_width / gray.shape[1])
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else gray
        kp, des = self.orb.detectAndCompute(small, None)
        self._S = np.diag([s, s, 1.0])
        self._S_inv = np.diag([1.0 / s, 1.0 / s, 1.0])

        if self.prev_gray is None:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            self.key_kp, self.key_des, self.key_H = kp, des, self.accumulated_H.copy()
            return self.accumulated_H

        prev_total = (self.total_dx, self.total_dy)
        H_key = self._match_homography(self.key_kp, self.key_des, kp, des)
        to_full = lambda H: self._S_inv @ H @ self._S  # small-image homography -> full frame
        if H_key is not None:
            new_H = to_full(H_key) @ self.key_H
        else:
            # Lost the keyframe: chain one step from the previous frame and re-anchor on it.
            H_step = self._match_homography(self.prev_kp, self.prev_des, kp, des)
            new_H = (to_full(H_step) @ self.accumulated_H) if H_step is not None else self.accumulated_H
            self.key_kp, self.key_des, self.key_H = kp, des, new_H.copy()

        self.accumulated_H = new_H.astype(np.float32)
        self.total_dx = float(self.accumulated_H[0, 2])
        self.total_dy = float(self.accumulated_H[1, 2])
        self.camera_dx = self.total_dx - prev_total[0]
        self.camera_dy = self.total_dy - prev_total[1]

        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
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
        if not slot_definitions:
            return []
        # One perspectiveTransform for every vertex of every slot (240-slot lots: one call
        # instead of 240), then split back per slot.
        polys = [np.asarray(s["polygon"], dtype=np.float32).reshape(-1, 2) for s in slot_definitions]
        sizes = [len(p) for p in polys]
        warped = cv2.perspectiveTransform(np.concatenate(polys).reshape(-1, 1, 2),
                                          self.accumulated_H).reshape(-1, 2).astype(np.int32)
        compensated_slots = []
        for slot, pts in zip(slot_definitions, np.split(warped, np.cumsum(sizes)[:-1])):
            new_poly = pts.tolist()
            cx, cy = (int(v) for v in pts.mean(axis=0))

            # Check if at least 2 vertices are inside the active frame viewport
            in_view_pts = int(((pts[:, 0] >= 0) & (pts[:, 0] <= frame_width) &
                               (pts[:, 1] >= 0) & (pts[:, 1] <= frame_height)).sum())
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
