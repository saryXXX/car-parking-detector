"""
Main Execution Script for Smart Parking Monitoring & Spatial Analytics
Supports video files, camera streams, and 4 distinct simulation scenarios:
  1. Stationnement Conforme & Dwell Time
  2. Stationnement à Cheval sur la Ligne (Mauvais Stationnement)
  3. Risque d'Accrochage Évité de Justesse (Near-Miss)
  4. Collision Réelle Confirmée (Crash & Impact en pleine allée)
"""

import argparse
import os
import time
import cv2
import numpy as np

from src.spatial_engine import SpatialEngine
from src.tracker import VehicleTracker
from src.visualizer import ParkingVisualizer
from src.motion_compensator import CameraMotionCompensator


def run_scenario_1_proper_parking(engine: SpatialEngine, visualizer: ParkingVisualizer):
    """Scénario 1 : Conforme - Insertion fluide dans Slot 01 sans toucher Slot 02."""
    print("[INFO] Lancement Scénario 1 : Stationnement Régulier & Dwell Time...")
    width, height = 1280, 720
    fps = 30.0
    frame_delay = int(1000 / fps)

    car_101 = {"id": 101, "x": 50.0, "y": 360.0}
    car_102 = {"id": 102, "x": 350.0, "y": 190.0}

    frame_idx = 0
    start_time = time.time()
    cv2.namedWindow("Scenario 1 : Stationnement Conforme", cv2.WINDOW_AUTOSIZE)

    while True:
        sim_time = start_time + (frame_idx / fps)
        bg = np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)

        for x in range(0, width, 60):
            cv2.line(bg, (x, 360), (x + 30, 360), (0, 255, 255), 2)

        if frame_idx < 40:
            car_101["x"] += (190.0 - 50.0) / 40.0
            car_101["y"] = 360.0
        elif frame_idx < 80:
            car_101["x"] = 190.0
            car_101["y"] -= (360.0 - 190.0) / 40.0
        else:
            car_101["x"] = 190.0
            car_101["y"] = 190.0

        tracked = []
        for c in [car_101, car_102]:
            cx, cy = int(c["x"]), int(c["y"])
            w, h = 80, 110
            bbox = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
            tracked.append({"id": c["id"], "bbox": bbox, "conf": 0.95, "cls_name": "car"})
            cv2.rectangle(bg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (80, 80, 200), -1)

        analytics = engine.update(tracked, current_time=sim_time)
        vis = visualizer.render(bg, analytics, tracked, fps=fps)

        cv2.putText(vis, "SCENARIO 1 : Stationnement Regulier | Quitter: 'Q'", (20, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow("Scenario 1 : Stationnement Conforme", vis)
        if cv2.waitKey(frame_delay) & 0xFF in [ord('q'), 27]:
            break
        frame_idx += 1

    cv2.destroyAllWindows()


def run_scenario_2_line_straddling(engine: SpatialEngine, visualizer: ParkingVisualizer):
    """Scénario 2 : Mauvais stationnement - Véhicule garé à cheval sur la ligne entre Slot 04 et Slot 05."""
    print("[INFO] Lancement Scénario 2 : Stationnement à Cheval sur la Ligne...")
    width, height = 1280, 720
    fps = 30.0
    frame_delay = int(1000 / fps)

    car_104 = {"id": 104, "x": 60.0, "y": 360.0}

    frame_idx = 0
    start_time = time.time()
    cv2.namedWindow("Scenario 2 : A Cheval sur la Ligne", cv2.WINDOW_AUTOSIZE)

    while True:
        sim_time = start_time + (frame_idx / fps)
        bg = np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)

        for x in range(0, width, 60):
            cv2.line(bg, (x, 360), (x + 30, 360), (0, 255, 255), 2)

        if frame_idx < 45:
            car_104["x"] += (270.0 - 60.0) / 45.0
            car_104["y"] = 360.0
        elif frame_idx < 85:
            car_104["x"] = 270.0
            car_104["y"] += (530.0 - 360.0) / 40.0
        else:
            car_104["x"] = 270.0
            car_104["y"] = 530.0

        cx, cy = int(car_104["x"]), int(car_104["y"])
        w, h = 80, 110
        bbox = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
        tracked = [{"id": car_104["id"], "bbox": bbox, "conf": 0.95, "cls_name": "car"}]
        cv2.rectangle(bg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (80, 80, 200), -1)

        analytics = engine.update(tracked, current_time=sim_time)
        vis = visualizer.render(bg, analytics, tracked, fps=fps)

        cv2.putText(vis, "SCENARIO 2 : A Cheval sur la Ligne (Slots 4 et 5 bloques) | Quitter: 'Q'", (20, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow("Scenario 2 : A Cheval sur la Ligne", vis)
        if cv2.waitKey(frame_delay) & 0xFF in [ord('q'), 27]:
            break
        frame_idx += 1

    cv2.destroyAllWindows()


def run_scenario_3_near_miss(engine: SpatialEngine, visualizer: ParkingVisualizer):
    """Scénario 3 : Risque de collision évité de justesse (Near-Miss avec freinage)."""
    print("[INFO] Lancement Scénario 3 : NEAR-MISS (Évitement de justesse)...")
    width, height = 1280, 720
    fps = 30.0
    frame_delay = int(1000 / fps)

    car_101 = {"id": 101, "x": 190.0, "y": 190.0}
    car_105 = {"id": 105, "x": 50.0, "y": 360.0}

    frame_idx = 0
    start_time = time.time()
    cv2.namedWindow("Scenario 3 : NEAR-MISS (Evitement d'urgence)", cv2.WINDOW_AUTOSIZE)

    while True:
        sim_time = start_time + (frame_idx / fps)
        bg = np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)

        for x in range(0, width, 60):
            cv2.line(bg, (x, 360), (x + 30, 360), (0, 255, 255), 2)

        # Car 105 circule en continu
        car_105["x"] += 4.5

        # Car 101 recule puis freine d'urgence dès que Car 105 s'approche
        if frame_idx >= 15 and frame_idx < 45:
            car_101["y"] += 3.5
        elif frame_idx >= 45:
            car_101["y"] = 295.0 # Freinage d'urgence ! Pas d'impact

        tracked = []
        for c in [car_101, car_105]:
            cx, cy = int(c["x"]), int(c["y"])
            w, h = 80, 110
            bbox = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
            tracked.append({"id": c["id"], "bbox": bbox, "conf": 0.95, "cls_name": "car"})
            cv2.rectangle(bg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (80, 80, 200), -1)

        analytics = engine.update(tracked, current_time=sim_time)
        vis = visualizer.render(bg, analytics, tracked, fps=fps)

        cv2.putText(vis, "SCENARIO 3 : NEAR-MISS (Freinage in extremis -> Collision evitee) | Quitter: 'Q'", (20, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)

        cv2.imshow("Scenario 3 : NEAR-MISS (Evitement d'urgence)", vis)
        if cv2.waitKey(frame_delay) & 0xFF in [ord('q'), 27]:
            break
        frame_idx += 1

    cv2.destroyAllWindows()


def run_scenario_4_actual_collision(engine: SpatialEngine, visualizer: ParkingVisualizer):
    """Scénario 4 : Collision confirmée avec impact physique direct."""
    print("[INFO] Lancement Scénario 4 : COLLISION CONFIRMÉE...")
    width, height = 1280, 720
    fps = 30.0
    frame_delay = int(1000 / fps)

    car_101 = {"id": 101, "x": 190.0, "y": 190.0}
    car_105 = {"id": 105, "x": 50.0, "y": 360.0}

    frame_idx = 0
    start_time = time.time()
    cv2.namedWindow("Scenario 4 : Collision Reelle Confirmée", cv2.WINDOW_AUTOSIZE)
    collision_occurred = False

    while True:
        sim_time = start_time + (frame_idx / fps)
        bg = np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)

        for x in range(0, width, 60):
            cv2.line(bg, (x, 360), (x + 30, 360), (0, 255, 255), 2)

        if not collision_occurred:
            car_105["x"] += 4.0
            if frame_idx >= 15 and car_101["y"] < 360.0:
                car_101["y"] += 4.0

            if abs(car_105["x"] - car_101["x"]) < 70 and abs(car_105["y"] - car_101["y"]) < 50:
                collision_occurred = True

        tracked = []
        for c in [car_101, car_105]:
            cx, cy = int(c["x"]), int(c["y"])
            w, h = 80, 110
            bbox = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
            tracked.append({"id": c["id"], "bbox": bbox, "conf": 0.95, "cls_name": "car"})
            color = (80, 80, 200) if not collision_occurred else (40, 40, 220)
            cv2.rectangle(bg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, -1)

        analytics = engine.update(tracked, current_time=sim_time)
        vis = visualizer.render(bg, analytics, tracked, fps=fps)

        if collision_occurred:
            pulse = int((frame_idx % 10) * 4)
            impact_pt = (int((car_101["x"] + car_105["x"]) / 2), 360)
            cv2.circle(vis, impact_pt, 40 + pulse, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.putText(vis, "SCENARIO 4 : COLLISION CONFIRMEE (Impact physique direct) | Quitter: 'Q'", (20, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2, cv2.LINE_AA)

        cv2.imshow("Scenario 4 : Collision Reelle Confirmée", vis)
        if cv2.waitKey(frame_delay) & 0xFF in [ord('q'), 27]:
            break
        frame_idx += 1

    cv2.destroyAllWindows()


def run_video_pipeline(source: str, engine: SpatialEngine, tracker: VehicleTracker, visualizer: ParkingVisualizer):
    print(f"[INFO] Ouverture du flux vidéo : {source}...")
    src_input = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src_input)

    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo '{source}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = max(1, int(1000 / fps))
    prev_time = time.time()

    # Initialisation du compensateur de mouvement global (GMC)
    gmc = CameraMotionCompensator(max_features=500)
    base_slots = [dict(s) for s in engine.slots]

    cv2.namedWindow("Smart Parking Monitoring - YOLOv8 + ByteTrack + GMC", cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin du flux vidéo.")
            break

        # Redimensionnement automatique pour fluidité temps réel sur vidéos 4K
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (1280, 720))

        curr_time = time.time()
        actual_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else fps
        prev_time = curr_time

        # 1. Mise à jour de la matrice de mouvement caméra et ancrage au sol des places
        gmc.update(frame)
        zone_info = gmc.get_active_zone_info()
        compensated_slots = gmc.transform_polygons(base_slots, frame_width=frame.shape[1], frame_height=frame.shape[0])
        engine.set_dynamic_slots(compensated_slots)
        visualizer.set_dynamic_slots(compensated_slots)

        # 2. Tracking des véhicules
        tracked_vehicles = tracker.track(frame)

        # 3. Moteur spatial avec soustraction de l'ego-motion caméra et alertes physiques strictes
        analytics = engine.update(
            tracked_vehicles,
            current_time=curr_time,
            camera_dx=gmc.camera_dx,
            camera_dy=gmc.camera_dy
        )

        # 4. Rendu visuel haute fidélité avec HUD multi-zones et anti-clutter
        vis_frame = visualizer.render(
            frame,
            analytics,
            tracked_vehicles,
            zone_info=zone_info,
            fps=actual_fps
        )

        cv2.imshow("Smart Parking Monitoring - YOLOv8 + ByteTrack + GMC", vis_frame)
        if cv2.waitKey(frame_delay) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Système Intelligent de Suivi de Parking (Smart Parking Monitoring)")
    parser.add_argument("--source", type=str, default=None, help="Chemin vers la vidéo (.mp4) ou index webcam (0).")
    parser.add_argument("--config", type=str, default="configs/parking_slots.json", help="Chemin de configuration des places.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Modèle YOLOv8 (yolov8n.pt, etc.).")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3, 4],
                        help="1=Conforme, 2=Sur la ligne, 3=Near-Miss (Evitement), 4=Collision Reelle")
    args = parser.parse_args()

    engine = SpatialEngine(config_path=args.config)
    visualizer = ParkingVisualizer(slot_definitions=engine.slots)

    if args.source is not None:
        tracker = VehicleTracker(model_name=args.model)
        run_video_pipeline(args.source, engine, tracker, visualizer)
    else:
        if args.scenario == 1:
            run_scenario_1_proper_parking(engine, visualizer)
        elif args.scenario == 2:
            run_scenario_2_line_straddling(engine, visualizer)
        elif args.scenario == 3:
            run_scenario_3_near_miss(engine, visualizer)
        elif args.scenario == 4:
            run_scenario_4_actual_collision(engine, visualizer)


if __name__ == "__main__":
    main()
