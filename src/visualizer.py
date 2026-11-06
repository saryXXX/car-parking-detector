"""
Visualizer Module for Smart Parking Monitoring & Spatial Analytics
High-End, Clean Aesthetic without Visual Clutter:
- Multi-Zone Adaptive HUD (Active Zone, Camera Pan Offset, Occupancy Stats).
- Out-of-bounds slot clipping (only renders slots currently in frame).
- Minimalist bounding boxes and ground contact indicators.
- Physics-backed collision & near-miss indicators (sleek pulse ring, HUD alert ticker).
"""

from typing import Dict, List, Optional
import cv2
import numpy as np
import time


class ParkingVisualizer:
    # Modern, sleek color palette (BGR)
    COLOR_FREE = (46, 204, 113)         # Emerald Green
    COLOR_OCCUPIED = (60, 60, 220)      # Crimson Red
    COLOR_OVERSTAY = (30, 130, 255)     # Deep Orange
    COLOR_STRADDLE = (210, 50, 210)     # Orchid Magenta
    COLOR_ALERT = (30, 30, 235)         # Neon Red
    COLOR_NEAR_MISS = (0, 165, 255)     # Amber Orange
    COLOR_LANE_WARN = (0, 215, 255)     # Warm Yellow
    COLOR_TRIPWIRE = (255, 200, 0)      # Cyan Accent
    COLOR_DOT = (0, 255, 255)           # Yellow Contact Dot
    COLOR_TEXT = (255, 255, 255)        # Clean White

    def __init__(self, slot_definitions: List[Dict]):
        self.slot_defs = slot_definitions

    def set_dynamic_slots(self, updated_slots: List[Dict]):
        self.slot_defs = updated_slots

    def render(
        self,
        frame: np.ndarray,
        analytics: Dict,
        tracked_vehicles: List[Dict],
        zone_info: Optional[Dict] = None,
        fps: float = 30.0
    ) -> np.ndarray:
        vis_frame = frame.copy()
        overlay = frame.copy()
        h, w = vis_frame.shape[:2]

        states = analytics.get("states", {})
        dwell_data = analytics.get("dwell_data", {})
        obstruction_alerts = analytics.get("obstruction_alerts", [])
        straddle_alerts = analytics.get("line_straddle_alerts", [])
        collision_alerts = analytics.get("collision_alerts", [])
        entries = analytics.get("total_entries", 0)

        # 1. Rendu des places de parking visibles
        straddled_slots = set()
        for stra in straddle_alerts:
            straddled_slots.update(stra.get("slot_ids", []))

        for slot in self.slot_defs:
            # Ne pas afficher si la place est hors champ suite au pan caméra
            if not slot.get("is_visible", True):
                continue

            sid = slot["id"]
            pts = np.array(slot["polygon"], np.int32).reshape((-1, 1, 2))
            is_occ = states.get(sid, False)
            d_info = dwell_data.get(sid, {})
            elapsed = d_info.get("elapsed_seconds", 0.0)
            max_dwell_sec = slot.get("max_dwell_minutes", 15) * 60.0

            if sid in straddled_slots:
                color = self.COLOR_STRADDLE
            elif not is_occ:
                color = self.COLOR_FREE
            elif elapsed > max_dwell_sec:
                color = self.COLOR_OVERSTAY
            else:
                color = self.COLOR_OCCUPIED

            # Polygone semi-transparent
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(vis_frame, [pts], True, color, 2, cv2.LINE_AA)

            cx = int(np.mean([p[0] for p in slot["polygon"]]))
            cy = int(np.mean([p[1] for p in slot["polygon"]]))

            # Label de la place (propre et centré)
            if sid in straddled_slots:
                cv2.putText(vis_frame, f"{slot['name']} : BLOQUE", (cx - 45, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
            elif not is_occ:
                cv2.putText(vis_frame, f"{slot['name']} : LIBRE", (cx - 40, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                car_id = d_info.get("car_id", "?")
                mins, secs = divmod(int(elapsed), 60)
                cv2.putText(vis_frame, f"{slot['name']} [Car #{car_id}]", (cx - 50, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(vis_frame, f"{mins:02d}m {secs:02d}s", (cx - 22, cy + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # Transparence douce pour les places
        cv2.addWeighted(overlay, 0.28, vis_frame, 0.72, 0, vis_frame)

        # 2. Dessin épuré des véhicules
        for v in tracked_vehicles:
            x1, y1, x2, y2 = map(int, v["bbox"])
            px, py = int((x1 + x2) / 2), y2

            # Rectangle véhicule fin et élégant
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (220, 200, 100), 1, cv2.LINE_AA)
            # Point de contact au sol
            cv2.circle(vis_frame, (px, py), 3, self.COLOR_DOT, -1, cv2.LINE_AA)
            # Badge identifiant discret
            cv2.putText(vis_frame, f"Car #{v['id']}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)

        # 3. Alertes à cheval sur la ligne (discret sur le véhicule fautif)
        for stra in straddle_alerts[:2]:
            x1, y1, x2, y2 = map(int, stra["bbox"])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), self.COLOR_STRADDLE, 2, cv2.LINE_AA)
            cv2.putText(vis_frame, f"A CHEVAL SUR LA LIGNE (Car #{stra['car_id']})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_STRADDLE, 2, cv2.LINE_AA)

        # 4. Alertes Collisions & Near-Miss (Uniquement pour vrais incidents, max 2)
        for col in collision_alerts[:2]:
            pt = col["pos"]
            c1, c2 = col["car1_id"], col["car2_id"]
            if col["type"] == "COLLISION":
                # COLLISION: Solid red ring
                cv2.circle(vis_frame, pt, 28, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(vis_frame, pt, 14, (0, 80, 255), -1, cv2.LINE_AA)
                cv2.putText(vis_frame, f"💥 COLLISION ! Car #{c1} x #{c2}", (pt[0] - 100, pt[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 3, cv2.LINE_AA)
                # Pulsing effect
                pulse = int((time.time() * 5) % 10)
                cv2.circle(vis_frame, pt, 40 + pulse, (0, 0, 255), 2, cv2.LINE_AA)
            elif col["type"] == "NEAR_MISS":
                # NEAR-MISS: Orange pulsing ring (more dramatic)
                cv2.circle(vis_frame, pt, 24, self.COLOR_NEAR_MISS, 3, cv2.LINE_AA)
                cv2.putText(vis_frame, f"⚡ RISQUE ! Car #{c1} <-> #{c2}", (pt[0] - 95, pt[1] - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_NEAR_MISS, 2, cv2.LINE_AA)
                # Large pulsing warning ring
                pulse = int((time.time() * 6) % 15)
                cv2.circle(vis_frame, pt, 45 + pulse, self.COLOR_NEAR_MISS, 2, cv2.LINE_AA)

        # 5. Bandeau Supérieur HUD Professionnel (Anti-Clutter)
        hud_h = 56
        hud_overlay = vis_frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (18, 18, 18), -1)
        cv2.addWeighted(hud_overlay, 0.85, vis_frame, 0.15, 0, vis_frame)
        cv2.line(vis_frame, (0, hud_h), (w, hud_h), (60, 60, 60), 1)

        # Ligne 1 : Statistiques du parking
        tot = analytics.get("total_slots", len(self.slot_defs))
        occ = analytics.get("occupied_count", 0)
        free = analytics.get("free_count", tot - occ)
        rate = analytics.get("occupancy_rate", 0.0)

        stats_text = f"SMART PARKING | Places Libres: {free}/{tot} | Occupees: {occ} ({rate:.0f}%) | Entrees: {entries} | FPS: {fps:.1f}"
        cv2.putText(vis_frame, stats_text, (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

        # Ligne 2 : Zone Adaptative & Compensation Caméra GMC
        if zone_info is not None:
            z_name = zone_info.get("zone_name", "Zone A")
            pan_dx = zone_info.get("pan_dx", 0.0)
            pan_dy = zone_info.get("pan_dy", 0.0)
            zone_text = f"CAMERA GMC : Pan ({pan_dx:+.0f}px, {pan_dy:+.0f}px) -> ACTIVE : {z_name.upper()}"
        else:
            zone_text = "CAMERA FIXE | ZONE UNIQUE SURVEILLEE"

        cv2.putText(vis_frame, zone_text, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)

        # Alerte Ticker en haut à droite si incident actif
        incident_count = len(straddle_alerts) + len(collision_alerts) + len(obstruction_alerts)
        if incident_count > 0:
            inc_label = f"ALERTE : {incident_count} INCIDENT(S) ACTIF(S)"
            cv2.rectangle(vis_frame, (w - 380, 10), (w - 15, 46), (30, 30, 200), -1)
            cv2.putText(vis_frame, inc_label, (w - 370, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Detailed incident panel on right side
            panel_x = w - 400
            panel_y = 60
            panel_h = 25

            # Straddle violations
            for i, stra in enumerate(straddle_alerts[:3]):
                car_id = stra.get("car_id", "?")
                slots = ", ".join([str(s) for s in stra.get("slot_ids", [])])
                msg = f"PARKING A CHEVAL: Car #{car_id} crosses slots {slots}"
                cv2.rectangle(vis_frame, (panel_x - 20, panel_y + i * panel_h), (w - 15, panel_y + (i+1) * panel_h), (210, 50, 210), -1)
                cv2.putText(vis_frame, msg, (panel_x - 10, panel_y + i * panel_h + 17),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            # Lane obstruction violations
            offset = len(straddle_alerts[:3])
            for i, obs in enumerate(obstruction_alerts[:3]):
                car_id = obs.get("car_id", "?")
                msg = f"LANE BLOCKED: Car #{car_id} obstructing traffic lane"
                cv2.rectangle(vis_frame, (panel_x - 20, panel_y + (offset + i) * panel_h), (w - 15, panel_y + (offset + i + 1) * panel_h), (30, 130, 255), -1)
                cv2.putText(vis_frame, msg, (panel_x - 10, panel_y + (offset + i) * panel_h + 17),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            # Collision alerts
            offset = len(straddle_alerts[:3]) + len(obstruction_alerts[:3])
            for i, col in enumerate(collision_alerts[:3]):
                c1, c2 = col.get("car1_id", "?"), col.get("car2_id", "?")
                col_type = col.get("type", "COLLISION")
                msg = f"{col_type}: Car #{c1} <-> Car #{c2}"
                color = (0, 0, 255) if col_type == "COLLISION" else (0, 165, 255)
                cv2.rectangle(vis_frame, (panel_x - 20, panel_y + (offset + i) * panel_h), (w - 15, panel_y + (offset + i + 1) * panel_h), color, -1)
                cv2.putText(vis_frame, msg, (panel_x - 10, panel_y + (offset + i) * panel_h + 17),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(vis_frame, (w - 210, 10), (w - 15, 46), (30, 140, 40), -1)
            cv2.putText(vis_frame, "SYSTEME NOMINAL", (w - 195, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        return vis_frame
