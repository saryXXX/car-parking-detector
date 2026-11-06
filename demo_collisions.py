"""
Standalone Demonstration Script: Near-Miss vs Actual Collision
Visually compares:
1. NEAR-MISS (Risque d'accrochage avec freinage d'urgence / évitement)
2. ACTUAL COLLISION (Accrochage confirmé avec impact physique et arrêt brutal)
"""

import argparse
import time
import cv2
import numpy as np

from src.spatial_engine import SpatialEngine
from src.visualizer import ParkingVisualizer


def run_near_miss_demo(engine: SpatialEngine, visualizer: ParkingVisualizer):
    """
    DÉMONSTRATION DU NEAR-MISS (Évitement de justesse) :
    Car #101 commence à reculer de sa place.
    Car #105 arrive à vive allure dans l'allée centrale.
    La distance passe sous les 130 px -> Alerte RISQUE DE COLLISION (Near-Miss).
    Car #101 freine d'urgence à temps (y=295) et Car #105 passe devant sans contact.
    """
    print("[INFO] Lancement du Scénario NEAR-MISS (Évitement d'urgence)...")
    width, height = 1280, 720
    fps = 30.0
    frame_delay = int(1000 / fps)

    car_101 = {"id": 101, "x": 190.0, "y": 190.0}
    car_105 = {"id": 105, "x": 60.0, "y": 360.0}

    frame_idx = 0
    start_time = time.time()
    cv2.namedWindow("DEMO : NEAR-MISS (Risque & Evitement d'urgence)", cv2.WINDOW_AUTOSIZE)

    while True:
        sim_time = start_time + (frame_idx / fps)
        bg = np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)

        for x in range(0, width, 60):
            cv2.line(bg, (x, 360), (x + 30, 360), (0, 255, 255), 2)

        # Car 105 avance continuellement
        car_105["x"] += 4.5

        # Car 101 recule jusqu'à apercevoir Car 105 puis FREINE à frame 45 (y=295)
        if frame_idx >= 15 and frame_idx < 45:
            car_101["y"] += 3.5  # Recule vers l'allée
        elif frame_idx >= 45:
            # Freinage d'urgence ! La voiture s'arrête net avant la ligne de trajectoire
            car_101["y"] = 295.0

        tracked = []
        for c in [car_101, car_105]:
            cx, cy = int(c["x"]), int(c["y"])
            w, h = 80, 110
            bbox = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
            tracked.append({"id": c["id"], "bbox": bbox, "conf": 0.95, "cls_name": "car"})
            cv2.rectangle(bg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (80, 80, 200), -1)

        analytics = engine.update(tracked, current_time=sim_time)
        vis = visualizer.render(bg, analytics, tracked, fps=fps)

        # Bannière explicative
        cv2.rectangle(vis, (0, height - 55), (width, height), (25, 25, 25), -1)
        cv2.putText(vis, "SCENARIO NEAR-MISS : Car #101 freine d'urgence -> Collision evitee de justesse ! | Quitter: 'Q'",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)

        cv2.imshow("DEMO : NEAR-MISS (Risque & Evitement d'urgence)", vis)
        if cv2.waitKey(frame_delay) & 0xFF in [ord('q'), 27]:
            break
        frame_idx += 1

    cv2.destroyAllWindows()


def run_actual_collision_demo(engine: SpatialEngine, visualizer: ParkingVisualizer):
    """
    DÉMONSTRATION D'UNE VRAIE COLLISION (Crash / Impact) :
    Car #101 recule sans regarder en pleine allée centrale (y=360).
    Car #105 roule dans l'allée à la même hauteur.
    À frame 50, les deux boîtes s'encastrent :
    -> Alerte COLLISION DÉTECTÉE avec flash d'impact rouge et arrêt brutal des deux véhicules.
    """
    print("[INFO] Lancement du Scénario COLLISION RÉELLE (Accrochage avec impact)...")
    width, height = 1280, 720
    fps = 30.0
    frame_delay = int(1000 / fps)

    car_101 = {"id": 101, "x": 190.0, "y": 190.0}
    car_105 = {"id": 105, "x": 60.0, "y": 360.0}

    frame_idx = 0
    start_time = time.time()
    cv2.namedWindow("DEMO : COLLISION REELLE (Impact & Crash)", cv2.WINDOW_AUTOSIZE)

    collision_occurred = False

    while True:
        sim_time = start_time + (frame_idx / fps)
        bg = np.full((height, width, 3), (45, 45, 45), dtype=np.uint8)

        for x in range(0, width, 60):
            cv2.line(bg, (x, 360), (x + 30, 360), (0, 255, 255), 2)

        if not collision_occurred:
            # Car 105 roule
            car_105["x"] += 4.0
            # Car 101 recule jusqu'en plein milieu de l'allée centrale (y=360)
            if frame_idx >= 15 and car_101["y"] < 360.0:
                car_101["y"] += 4.0

            # Vérifie si le point d'impact est atteint
            if abs(car_105["x"] - car_101["x"]) < 70 and abs(car_105["y"] - car_101["y"]) < 50:
                collision_occurred = True
                print("[ALERT] IMPACT DETECTE ! Les deux vehicules sont immobilises.")
        else:
            # Après l'impact, les deux voitures sont accidentées et bloquées
            pass

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

        # Effet visuel d'ondes de choc autour de la collision
        if collision_occurred:
            pulse = int((frame_idx % 10) * 4)
            impact_pt = (int((car_101["x"] + car_105["x"]) / 2), 360)
            cv2.circle(vis, impact_pt, 40 + pulse, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(vis, impact_pt, 20 + pulse, (0, 215, 255), 2, cv2.LINE_AA)

        cv2.rectangle(vis, (0, height - 55), (width, height), (25, 25, 25), -1)
        cv2.putText(vis, "SCENARIO COLLISION REELLE : Impact physique direct dans l'allee ! | Quitter: 'Q'",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2, cv2.LINE_AA)

        cv2.imshow("DEMO : COLLISION REELLE (Impact & Crash)", vis)
        if cv2.waitKey(frame_delay) & 0xFF in [ord('q'), 27]:
            break
        frame_idx += 1

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Demonstrateur Specifique : Near-Miss vs Collision Reelle")
    parser.add_argument("--mode", type=str, default="near_miss", choices=["near_miss", "collision", "both"],
                        help="Mode a executer : near_miss, collision, ou both")
    args = parser.parse_args()

    engine = SpatialEngine("configs/parking_slots.json")
    visualizer = ParkingVisualizer(engine.slots)

    if args.mode in ["near_miss", "both"]:
        run_near_miss_demo(engine, visualizer)
    if args.mode in ["collision", "both"]:
        # Reinitialise pour un run propre
        engine = SpatialEngine("configs/parking_slots.json")
        run_actual_collision_demo(engine, visualizer)


if __name__ == "__main__":
    main()

