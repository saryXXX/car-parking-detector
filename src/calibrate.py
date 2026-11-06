"""
Interactive Parking Slot Calibration Tool
Allows clicking 4 corners to define each parking space polygon and saves to JSON.
"""

import argparse
import json
import os
import cv2
import numpy as np


class ParkingCalibrator:
    def __init__(self, source_path: str = None, output_path: str = "configs/parking_slots.json"):
        self.source_path = source_path
        self.output_path = output_path
        self.slots = []          # List of saved slots: [{"id": 1, "name": "Slot_01", "polygon": [...]}]
        self.current_points = [] # Current polygon being drawn: [(x, y), ...]
        self.frame = None
        self.display_frame = None
        self.window_name = "Calibrateur Interactif de Parking (OpenCV)"

        # Load existing slots if configuration file already exists
        self._load_existing_slots()

    def _load_existing_slots(self):
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    self.slots = json.load(f)
                print(f"[INFO] {len(self.slots)} place(s) existante(s) chargee(s) depuis '{self.output_path}'.")
            except Exception as e:
                print(f"[WARN] Impossible de charger '{self.output_path}': {e}")

    def _create_synthetic_parking_frame(self, width: int = 1280, height: int = 720) -> np.ndarray:
        """Generates a synthetic parking background if no video is provided."""
        img = np.full((height, width, 3), (50, 50, 50), dtype=np.uint8) # Asphalt grey

        # Central driving lane markings
        cv2.line(img, (0, 360), (width, 360), (255, 255, 255), 2, cv2.LINE_AA)
        for x in range(0, width, 60):
            cv2.line(img, (x, 360), (x + 30, 360), (0, 255, 255), 3, cv2.LINE_AA)

        # Draw some guidance parking lines (top row and bottom row)
        for x in range(100, 1150, 160):
            # Top bays
            cv2.line(img, (x, 80), (x, 300), (240, 240, 240), 3)
            # Bottom bays
            cv2.line(img, (x, 420), (x, 640), (240, 240, 240), 3)

        cv2.putText(img, "PARKING SIMULATION (Mode Demonstration)", (360, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return img

    def load_frame(self):
        if self.source_path and os.path.exists(self.source_path):
            ext = os.path.splitext(self.source_path)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                self.frame = cv2.imread(self.source_path)
            else:
                cap = cv2.VideoCapture(self.source_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    if frame.shape[1] > 1280:
                        self.frame = cv2.resize(frame, (1280, 720))
                    else:
                        self.frame = frame
                else:
                    print(f"[ERREUR] Impossible de lire une frame depuis '{self.source_path}'.")
                    self.frame = self._create_synthetic_parking_frame()
        else:
            if self.source_path:
                print(f"[INFO] Source '{self.source_path}' introuvable. Generation d'une frame simulee.")
            self.frame = self._create_synthetic_parking_frame()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([int(x), int(y)])
            print(f"[POINT] Coin #{len(self.current_points)} ajoute : ({x}, {y})")

            # When 4 points are reached, validate slot automatically
            if len(self.current_points) == 4:
                slot_id = len(self.slots) + 1
                new_slot = {
                    "id": slot_id,
                    "name": f"Slot_{slot_id:02d}",
                    "polygon": self.current_points.copy(),
                    "max_dwell_minutes": 15
                }
                self.slots.append(new_slot)
                print(f"[SUCCESS] Place '{new_slot['name']}' enregistree avec 4 coins !")
                self.current_points.clear()

    def draw_hud(self, img: np.ndarray):
        # Draw all saved slots
        for slot in self.slots:
            pts = np.array(slot["polygon"], np.int32).reshape((-1, 1, 2))
            # Semi-transparent overlay
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 0)) # Green
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)

            # Slot label
            cx = int(np.mean([p[0] for p in slot["polygon"]]))
            cy = int(np.mean([p[1] for p in slot["polygon"]]))
            cv2.putText(img, slot["name"], (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw points currently being clicked
        for idx, pt in enumerate(self.current_points):
            cv2.circle(img, tuple(pt), 5, (0, 140, 255), -1) # Orange circle
            cv2.putText(img, str(idx + 1), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            if idx > 0:
                cv2.line(img, tuple(self.current_points[idx - 1]), tuple(pt), (0, 165, 255), 2)

        # Instructions banner at the bottom
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h - 45), (w, h), (20, 20, 20), -1)
        hud_text = "[Clic Gauche] : Coin (4 par place) | [Z/U] : Annuler | [C] : Effacer slot en cours | [Entree/S] : Sauvegarder & Quitter"
        cv2.putText(img, hud_text, (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    def save_to_json(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.slots, f, indent=2, ensure_ascii=False)
        print(f"[OK] Configuration sauvegardee avec succes dans : '{self.output_path}' ({len(self.slots)} places).")

    def run(self):
        self.load_frame()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("=" * 65)
        print(" CALIBRATEUR DE PLACES DE PARKING ")
        print(" - Cliquez sur 4 coins pour chaque place de parking.")
        print(" - Appuyez sur 'Z' ou 'U' pour annuler.")
        print(" - Appuyez sur 'Entree' ou 'S' pour sauvegarder et quitter.")
        print("=" * 65)

        while True:
            self.display_frame = self.frame.copy()
            self.draw_hud(self.display_frame)
            cv2.imshow(self.window_name, self.display_frame)

            key = cv2.waitKey(20) & 0xFF
            if key in [ord('q'), 27]: # q or ESC
                print("[INFO] Fermeture demandee.")
                break
            elif key in [ord('s'), 13]: # s or Enter
                self.save_to_json()
                break
            elif key in [ord('z'), ord('u')]: # Undo
                if self.current_points:
                    removed = self.current_points.pop()
                    print(f"[UNDO] Point retire : {removed}")
                elif self.slots:
                    removed_slot = self.slots.pop()
                    print(f"[UNDO] Place retiree : {removed_slot['name']}")
            elif key == ord('c'): # Clear current slot points
                self.current_points.clear()
                print("[CLEAR] Points en cours effaces.")
            elif key == ord('r'): # Reset all
                self.slots.clear()
                self.current_points.clear()
                print("[RESET] Toutes les places ont ete effacees.")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outil de calibration interactif des places de parking.")
    parser.add_argument("--source", type=str, default=None, help="Chemin vers une video ou image de reference.")
    parser.add_argument("--output", type=str, default="configs/parking_slots.json", help="Chemin du fichier JSON de sortie.")
    args = parser.parse_args()

    calibrator = ParkingCalibrator(source_path=args.source, output_path=args.output)
    calibrator.run()

