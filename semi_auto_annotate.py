"""
Semi-Automated Annotation Helper
Uses baseline YOLOv8n to suggest bounding boxes, user verifies/corrects
Generates YOLO format labels automatically
"""

import cv2
import os
import json
from pathlib import Path
from src.tracker import VehicleTracker


class AnnotationHelper:
    """Semi-automated annotation with baseline model suggestions"""

    def __init__(self, image_dir: str, output_label_dir: str):
        self.image_dir = image_dir
        self.output_label_dir = output_label_dir
        self.tracker = VehicleTracker(model_name="yolov8n.pt")
        self.current_frame = None
        self.current_boxes = []
        self.current_image_path = None
        self.image_height = 0
        self.image_width = 0

        os.makedirs(output_label_dir, exist_ok=True)

    def get_images(self):
        """Get all images from directory"""
        images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            images.extend(Path(self.image_dir).glob(f'*{ext}'))
            images.extend(Path(self.image_dir).glob(f'*{ext.upper()}'))
        return sorted(images)

    def normalize_bbox(self, x1, y1, x2, y2):
        """Convert pixel coordinates to normalized YOLO format"""
        x_center = ((x1 + x2) / 2) / self.image_width
        y_center = ((y1 + y2) / 2) / self.image_height
        width = (x2 - x1) / self.image_width
        height = (y2 - y1) / self.image_height

        # Clamp to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        return x_center, y_center, width, height

    def denormalize_bbox(self, x_center, y_center, width, height):
        """Convert normalized YOLO to pixel coordinates"""
        x1 = int((x_center - width/2) * self.image_width)
        y1 = int((y_center - height/2) * self.image_height)
        x2 = int((x_center + width/2) * self.image_width)
        y2 = int((y_center + height/2) * self.image_height)
        return max(0, x1), max(0, y1), min(self.image_width, x2), min(self.image_height, y2)

    def auto_label_image(self, image_path):
        """Generate automatic labels using baseline model"""
        print(f"\n[INFO] Processing: {os.path.basename(image_path)}")

        # Read image
        self.current_frame = cv2.imread(str(image_path))
        self.current_image_path = image_path
        self.image_height, self.image_width = self.current_frame.shape[:2]

        # Run detection
        print(f"[INFO] Running baseline model detection...")
        vehicles = self.tracker.track(self.current_frame)

        # Convert to normalized format
        self.current_boxes = []
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle["bbox"]
            x_c, y_c, w, h = self.normalize_bbox(x1, y1, x2, y2)

            self.current_boxes.append({
                "class_id": 0,  # vehicle class
                "x_center": x_c,
                "y_center": y_c,
                "width": w,
                "height": h,
                "bbox_pixel": (int(x1), int(y1), int(x2), int(y2))
            })

        print(f"[INFO] Detected {len(self.current_boxes)} vehicles")
        return len(self.current_boxes)

    def save_labels(self):
        """Save current labels to YOLO format"""
        if not self.current_image_path:
            return False

        # Get base name without extension
        base_name = Path(self.current_image_path).stem
        label_path = os.path.join(self.output_label_dir, f"{base_name}.txt")

        # Write YOLO format
        with open(label_path, 'w') as f:
            for box in self.current_boxes:
                line = f"{box['class_id']} {box['x_center']:.6f} {box['y_center']:.6f} {box['width']:.6f} {box['height']:.6f}\n"
                f.write(line)

        print(f"✅ Saved {len(self.current_boxes)} labels to: {label_path}")
        return True

    def process_all_images(self):
        """Process all images with auto-labeling"""
        images = self.get_images()

        if not images:
            print("[ERROR] No images found!")
            return

        print("\n" + "="*70)
        print("SEMI-AUTOMATED ANNOTATION")
        print("="*70)
        print(f"\n[INFO] Found {len(images)} images to process")

        successful = 0
        failed = 0

        for idx, image_path in enumerate(images, 1):
            try:
                print(f"\n[{idx}/{len(images)}] ", end="")
                num_boxes = self.auto_label_image(image_path)
                self.save_labels()
                successful += 1

            except Exception as e:
                print(f"[ERROR] Failed to process {image_path}: {e}")
                failed += 1

        # Summary
        print("\n" + "="*70)
        print("ANNOTATION COMPLETE")
        print("="*70)
        print(f"\n✅ Successfully processed: {successful} images")
        if failed > 0:
            print(f"❌ Failed: {failed} images")

        print(f"\n📁 Labels saved to: {self.output_label_dir}")
        print(f"\n✅ Ready for model fine-tuning!")
        print(f"\nNext step:")
        print(f"  python finetune_model.py --epochs 50 --batch 8 --cpu --version v2")
        print("\n" + "="*70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semi-automated annotation helper")
    parser.add_argument("--image-dir", type=str, default="data/parking_dataset/images/train",
                       help="Directory with images to annotate")
    parser.add_argument("--label-dir", type=str, default="data/parking_dataset/labels/train",
                       help="Output directory for labels")

    args = parser.parse_args()

    helper = AnnotationHelper(args.image_dir, args.label_dir)
    helper.process_all_images()


if __name__ == "__main__":
    main()
