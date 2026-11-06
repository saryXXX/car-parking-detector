"""
Extract Frames from Video for Dataset Labeling
Extracts high-quality frames from parking lot videos for manual annotation
"""

import cv2
import os
from pathlib import Path


def extract_frames(video_path: str, output_dir: str = "data/parking_dataset/images/train",
                   frame_interval: int = 5, max_frames: int = None, resize_scale: float = 1.0):
    """
    Extract frames from video

    Args:
        video_path: Path to video file
        output_dir: Where to save frames
        frame_interval: Extract every N frames (5 = every 5th frame)
        max_frames: Max frames to extract
        resize_scale: Resize factor (1.0 = no resize, 0.5 = half size)
    """

    print("\n" + "="*70)
    print("FRAME EXTRACTION FROM VIDEO")
    print("="*70)

    # Check video exists
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    print(f"\n[INFO] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Cannot open video")
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
    print(f"\n[INFO] Extraction settings:")
    print(f"       Extract every {frame_interval} frames")
    print(f"       Max frames: {max_frames if max_frames else 'All'}")
    print(f"       Resize scale: {resize_scale}")
    print(f"       Output: {output_dir}")

    # Extract frames
    print(f"\n[INFO] Extracting frames...")
    print("-" * 70)

    frame_count = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract every N frames
        if frame_count % frame_interval == 0:
            if max_frames and extracted_count >= max_frames:
                break

            # Resize if needed
            if resize_scale != 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * resize_scale), int(h * resize_scale)))

            # Save frame
            output_filename = f"real_parking_{extracted_count:04d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, frame)

            extracted_count += 1

            if extracted_count % 20 == 0:
                print(f"  Extracted {extracted_count} frames...")

        frame_count += 1

    cap.release()

    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✅ Extracted {extracted_count} frames")
    print(f"   Saved to: {output_dir}")
    print(f"   Total frames processed: {frame_count}")
    print(f"\n📝 NEXT STEPS:")
    print(f"   1. Manually annotate these frames using a YOLO labeling tool:")
    print(f"      - LabelImg (https://github.com/heartexlabs/labelImg)")
    print(f"      - Roboflow (https://roboflow.com)")
    print(f"      - CVAT (https://cvat.org)")
    print(f"   2. Save labels in YOLO format (.txt files)")
    print(f"   3. Place .txt files in: {output_dir.replace('images', 'labels')}")
    print(f"   4. Run fine-tuning: python finetune_model.py --epochs 50")
    print(f"\n{'='*70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from parking video")
    parser.add_argument("--video", type=str, default="4196207-uhd_3840_2160_24fps.mp4",
                       help="Video file path")
    parser.add_argument("--output", type=str, default="data/parking_dataset/images/train",
                       help="Output directory")
    parser.add_argument("--interval", type=int, default=5,
                       help="Extract every N frames")
    parser.add_argument("--max", type=int, default=200,
                       help="Maximum frames to extract")
    parser.add_argument("--scale", type=float, default=0.5,
                       help="Resize scale (0.5 = half size)")

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        frame_interval=args.interval,
        max_frames=args.max,
        resize_scale=args.scale
    )


if __name__ == "__main__":
    main()
