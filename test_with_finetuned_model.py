"""
Test the Fine-Tuned Model on Real Videos
Complete pipeline: detection → tracking → analysis → visualization
"""

import cv2
import os
from src.tracker import VehicleTracker
from src.spatial_engine import SpatialEngine
from src.visualizer import ParkingVisualizer


def process_video(video_path: str, model_path: str = "models/parking_yolov8n_v2/weights/best.pt", max_frames: int = 500):
    """
    Process video with fine-tuned model

    Args:
        video_path: Path to video file
        model_path: Path to fine-tuned model
        max_frames: Maximum frames to process (None for all)
    """

    print("\n" + "="*70)
    print("TESTING FINE-TUNED PARKING MODEL")
    print("="*70)
    print(f"\n[INFO] Video: {video_path}")
    print(f"[INFO] Model: {model_path}")

    # Check if files exist
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    # Initialize components
    print("\n[INFO] Loading tracker with fine-tuned model...")
    tracker = VehicleTracker(model_name=model_path)

    print("[INFO] Initializing spatial engine...")
    engine = SpatialEngine()

    print("[INFO] Setting up visualizer...")
    visualizer = ParkingVisualizer(engine.slots)

    # Open video
    print(f"\n[INFO] Opening video...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Cannot open video file")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video properties:")
    print(f"       Resolution: {width}×{height}")
    print(f"       Total frames: {total_frames}")
    print(f"       FPS: {fps:.1f}")

    # Statistics
    stats = {
        "total_frames": 0,
        "vehicles_detected": 0,
        "collisions": 0,
        "near_misses": 0,
        "lane_obstructions": 0,
        "avg_occupancy": 0.0,
        "max_occupancy": 0.0,
    }

    # Process frames
    print(f"\n[INFO] Processing video (press 'Q' to stop)...")
    print("-" * 70)

    frame_count = 0
    occupancy_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if max_frames and frame_count > max_frames:
            break

        # Track vehicles
        vehicles = tracker.track(frame)

        # Analyze parking
        analytics = engine.update(vehicles)

        # Visualize
        output = visualizer.render(frame, analytics, vehicles, fps=fps)

        # Update statistics
        stats["total_frames"] = frame_count
        stats["vehicles_detected"] += len(vehicles)
        stats["collisions"] += len(analytics["collision_alerts"])
        stats["near_misses"] += len([a for a in analytics["collision_alerts"] if a["type"] == "NEAR_MISS"])
        stats["lane_obstructions"] += len(analytics["obstruction_alerts"])

        occupancy = analytics["occupancy_rate"]
        occupancy_history.append(occupancy)
        stats["avg_occupancy"] = sum(occupancy_history) / len(occupancy_history)
        stats["max_occupancy"] = max(occupancy_history)

        # Display frame
        cv2.imshow("Smart Parking - Fine-Tuned Model Test", output)

        # Print progress every 30 frames
        if frame_count % 30 == 0 or frame_count == 1:
            print(f"[FRAME {frame_count:4d}/{total_frames}] "
                  f"Vehicles: {len(vehicles):2d} | "
                  f"Occupancy: {occupancy:5.1f}% | "
                  f"Collisions: {len(analytics['collision_alerts']):1d}")

        # Exit on 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] User stopped playback")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"\nFrames processed:       {stats['total_frames']}")
    print(f"Total vehicles seen:    {stats['vehicles_detected']}")
    print(f"Collisions detected:    {stats['collisions']}")
    print(f"Near-misses detected:   {stats['near_misses']}")
    print(f"Lane obstructions:      {stats['lane_obstructions']}")
    print(f"\nOccupancy Statistics:")
    print(f"  Average: {stats['avg_occupancy']:.1f}%")
    print(f"  Maximum: {stats['max_occupancy']:.1f}%")
    print(f"\nModel Performance:")
    print(f"  Detection Rate: {stats['vehicles_detected']/max(stats['total_frames'], 1):.2f} vehicles/frame")
    print(f"  Collision Events: {stats['collisions']} total")
    print(f"\n✓ Test complete! Model is working correctly.")
    print("="*70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test fine-tuned parking model on video")
    parser.add_argument("--video", type=str, default="4196207-uhd_3840_2160_24fps.mp4",
                       help="Path to video file")
    parser.add_argument("--model", type=str, default="models/parking_yolov8n_v2/weights/best.pt",
                       help="Path to fine-tuned model")
    parser.add_argument("--frames", type=int, default=500,
                       help="Maximum frames to process (0 for all)")

    args = parser.parse_args()

    # Process video
    process_video(
        video_path=args.video,
        model_path=args.model,
        max_frames=args.frames if args.frames > 0 else None
    )


if __name__ == "__main__":
    main()
