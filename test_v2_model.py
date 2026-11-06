"""
Test the v2 fine-tuned model on parking video
"""

import cv2
from src.tracker import VehicleTracker
from src.visualizer import ParkingVisualizer
import json
import os

# Load v2 model
print("Loading v2 fine-tuned model...")
tracker = VehicleTracker(model_name="runs/detect/models/parking_yolov8n_v2-4/weights/best.pt")

# Load parking slot configuration
if os.path.exists("configs/commercial_parking_slots.json"):
    with open("configs/commercial_parking_slots.json", 'r') as f:
        slots_config = json.load(f)
    print(f"✅ Loaded {len(slots_config)} parking slots from config")
else:
    print("⚠️  No slot configuration found. Run calibrate_slots.py first.")
    slots_config = []

# Test on video
video_path = "4196207-uhd_3840_2160_24fps.mp4"

if not os.path.exists(video_path):
    print(f"❌ Video not found: {video_path}")
    print("Create a test image instead:")
    print("  python compare_models_v2.py --mode image")
    exit(1)

print(f"\nTesting on video: {video_path}")
cap = cv2.VideoCapture(video_path)

frame_count = 0
max_frames = 50  # Test first 50 frames

print("\nProcessing frames...")
print("-" * 70)

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run detection
    vehicles = tracker.track(frame)

    if frame_count % 10 == 0 or frame_count == 1:
        print(f"Frame {frame_count}: Detected {len(vehicles)} vehicles")

cap.release()

print("-" * 70)
print(f"\n✅ Test Complete!")
print(f"Processed {frame_count} frames")
print(f"\nv2 Model is ready for deployment at:")
print(f"  runs/detect/models/parking_yolov8n_v2-4/weights/best.pt")
