"""
Parking Slot Calibration Tool - v2
Interactive tool to define parking slot coordinates for your specific camera/parking lot
"""

import cv2
import json
import os
import numpy as np
from typing import List, Tuple


class SlotCalibrator:
    """Interactive slot calibration tool"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.slots = []
        self.current_slot = {"name": "", "polygon": []}
        self.drawing = False
        self.current_point = None

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_point = (x, y)
            self.current_slot["polygon"].append([x, y])
            print(f"Point added: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Undo last point
            if self.current_slot["polygon"]:
                self.current_slot["polygon"].pop()
                print("Last point removed")

        elif event == cv2.EVENT_MOUSEMOVE:
            self.current_point = (x, y)

    def calibrate_from_video(self, frame_index: int = 0):
        """Calibrate using a frame from video"""
        print("\n" + "="*70)
        print("PARKING SLOT CALIBRATION - INTERACTIVE MODE")
        print("="*70)

        # Load frame
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("[ERROR] Cannot read frame")
            return

        h, w = frame.shape[:2]
        display_frame = frame.copy()

        cv2.namedWindow("Slot Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Slot Calibration", 1280, 720)
        cv2.setMouseCallback("Slot Calibration", self.mouse_callback)

        slot_id = 1

        print(f"\n[INFO] Video frame: {frame_index}")
        print(f"[INFO] Frame size: {w}×{h}")
        print(f"\n[INSTRUCTIONS]:")
        print(f"  LEFT CLICK  - Add point to current slot")
        print(f"  RIGHT CLICK - Remove last point")
        print(f"  SPACE       - Finish current slot")
        print(f"  'S'         - Save and exit")
        print(f"  'C'         - Clear current slot")
        print(f"\n[STARTING] Slot #{slot_id}")

        while True:
            display = frame.copy()

            # Draw completed slots
            for i, slot in enumerate(self.slots):
                poly = [(int(p[0]), int(p[1])) for p in slot["polygon"]]
                if len(poly) > 0:
                    cv2.polylines(display, [np.array(poly, np.int32)], True, (0, 255, 0), 2)
                    cv2.putText(display, slot["name"], (poly[0][0], poly[0][1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw current slot being drawn
            if len(self.current_slot["polygon"]) > 0:
                poly = [(int(p[0]), int(p[1])) for p in self.current_slot["polygon"]]
                cv2.polylines(display, [np.array(poly, np.int32)], False, (0, 0, 255), 2)

                for point in poly:
                    cv2.circle(display, point, 5, (0, 165, 255), -1)

                # Preview line to cursor
                if self.current_point:
                    cv2.line(display, poly[-1], self.current_point, (255, 0, 0), 1)

            # Status bar
            status = f"Slots: {len(self.slots)} | Current points: {len(self.current_slot['polygon'])}"
            cv2.rectangle(display, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(display, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Slot Calibration", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Space - finish current slot
                if len(self.current_slot["polygon"]) >= 4:
                    self.current_slot["name"] = f"Slot_{slot_id:02d}"
                    self.slots.append(self.current_slot.copy())
                    print(f"✅ Slot #{slot_id} completed with {len(self.current_slot['polygon'])} points")

                    slot_id += 1
                    self.current_slot = {"name": "", "polygon": []}
                    print(f"[STARTING] Slot #{slot_id}")
                else:
                    print(f"⚠️  Need at least 4 points! Current: {len(self.current_slot['polygon'])}")

            elif key == ord('c'):  # Clear current
                self.current_slot["polygon"] = []
                print("Current slot cleared")

            elif key == ord('s'):  # Save and exit
                break

            elif key == 27:  # ESC
                print("Calibration cancelled")
                break

        cv2.destroyAllWindows()

        if self.slots:
            self.save_config()

    def save_config(self, output_file: str = "configs/commercial_parking_slots.json"):
        """Save slot configuration"""
        print(f"\n{'='*70}")
        print("SAVING CONFIGURATION")
        print(f"{'='*70}")

        # Add default properties
        config = []
        for i, slot in enumerate(self.slots, 1):
            config.append({
                "id": i,
                "name": slot["name"],
                "polygon": slot["polygon"],
                "max_dwell_minutes": 15
            })

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✅ Saved {len(self.slots)} slots to: {output_file}")
        print(f"\nSlot Summary:")
        for slot in config:
            print(f"  - {slot['name']}: {len(slot['polygon'])} points")

        print(f"\n{'='*70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate parking slots for specific camera")
    parser.add_argument("--video", type=str, default="4196207-uhd_3840_2160_24fps.mp4",
                       help="Video file to calibrate from")
    parser.add_argument("--frame", type=int, default=0,
                       help="Frame index to start calibration")
    parser.add_argument("--output", type=str, default="configs/commercial_parking_slots.json",
                       help="Output config file")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        return

    calibrator = SlotCalibrator(args.video)
    calibrator.calibrate_from_video(args.frame)


if __name__ == "__main__":
    main()
