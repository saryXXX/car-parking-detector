"""
Dataset Debug Script
Diagnoses label format, image/label alignment, and data quality issues
"""

import os
import cv2
import json
from pathlib import Path
from collections import defaultdict


class DatasetDebugger:
    def __init__(self, data_dir: str = "data/parking_dataset"):
        self.data_dir = data_dir
        self.issues = []
        self.warnings = []

    def check_directory_structure(self):
        """Verify correct directory layout"""
        print("\n[CHECK] Directory Structure")
        print("-" * 60)

        required_dirs = [
            ("images/train", "Training images"),
            ("images/val", "Validation images"),
            ("labels/train", "Training labels"),
            ("labels/val", "Validation labels"),
        ]

        for rel_path, desc in required_dirs:
            full_path = os.path.join(self.data_dir, rel_path)
            if os.path.exists(full_path):
                count = len([f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))])
                print(f"✓ {rel_path:<20} ({count} files) - {desc}")
            else:
                print(f"✗ {rel_path:<20} MISSING!")
                self.issues.append(f"Missing directory: {rel_path}")

    def check_image_label_pairs(self):
        """Ensure every image has corresponding label"""
        print("\n[CHECK] Image-Label Pairs")
        print("-" * 60)

        for split in ["train", "val"]:
            img_dir = os.path.join(self.data_dir, "images", split)
            label_dir = os.path.join(self.data_dir, "labels", split)

            if not os.path.exists(img_dir):
                continue

            images = {f.rsplit('.', 1)[0] for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))}
            labels = {f.rsplit('.', 1)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}

            missing_labels = images - labels
            extra_labels = labels - images

            print(f"\n{split.upper()} Split:")
            print(f"  Images: {len(images)}")
            print(f"  Labels: {len(labels)}")

            if missing_labels:
                print(f"  ⚠ Missing labels for: {list(missing_labels)[:5]}")
                self.warnings.append(f"Missing labels in {split}: {len(missing_labels)} images")

            if extra_labels:
                print(f"  ⚠ Extra labels (no image): {list(extra_labels)[:5]}")
                self.warnings.append(f"Extra labels in {split}: {len(extra_labels)} files")

            if not missing_labels and not extra_labels:
                print(f"  ✓ All images have matching labels!")

    def check_label_format(self):
        """Verify YOLO label format"""
        print("\n[CHECK] Label Format (YOLO)")
        print("-" * 60)

        label_dir = os.path.join(self.data_dir, "labels", "val")
        if not os.path.exists(label_dir):
            print("No validation labels found")
            return

        sample_labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')][:3]

        for label_file in sample_labels:
            label_path = os.path.join(label_dir, label_file)

            print(f"\n{label_file}:")
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                if not lines:
                    print(f"  ⚠ EMPTY label file!")
                    self.warnings.append(f"Empty label file: {label_file}")
                    continue

                print(f"  Lines: {len(lines)}")

                # Check format
                for idx, line in enumerate(lines[:3]):  # Show first 3
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"  ✗ Line {idx+1}: Wrong format (expected 5 values, got {len(parts)})")
                        print(f"     Raw: {line.strip()}")
                        self.issues.append(f"Format error in {label_file} line {idx+1}")
                    else:
                        cls_id, x_center, y_center, width, height = map(float, parts)

                        # Validate ranges (normalized 0-1)
                        valid = all(0 <= v <= 1 for v in [x_center, y_center, width, height])
                        if not valid:
                            print(f"  ✗ Line {idx+1}: Values outside [0,1] range!")
                            print(f"     {cls_id} {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}")
                            self.issues.append(f"Out-of-range coordinates in {label_file}")
                        else:
                            print(f"  ✓ Line {idx+1}: cls={int(cls_id)} center=({x_center:.3f},{y_center:.3f}) size=({width:.3f}×{height:.3f})")

            except Exception as e:
                print(f"  ✗ Error reading: {e}")
                self.issues.append(f"Error reading {label_file}: {e}")

    def check_image_label_alignment(self):
        """Verify images and labels are actually aligned"""
        print("\n[CHECK] Image-Label Alignment")
        print("-" * 60)

        img_dir = os.path.join(self.data_dir, "images", "val")
        label_dir = os.path.join(self.data_dir, "labels", "val")

        if not os.path.exists(img_dir):
            return

        sample_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))][:3]

        for image_file in sample_images:
            image_path = os.path.join(img_dir, image_file)
            label_file = image_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)

            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"✗ {image_file}: Cannot read image!")
                self.issues.append(f"Cannot read image: {image_file}")
                continue

            h, w = img.shape[:2]
            print(f"\n{image_file}: {w}×{h}px")

            # Read labels
            try:
                with open(label_path, 'r') as f:
                    annotations = [line.strip().split() for line in f if line.strip()]

                print(f"  Objects: {len(annotations)}")

                if len(annotations) > 0:
                    for idx, parts in enumerate(annotations[:2]):
                        if len(parts) == 5:
                            cls_id, x_c, y_c, box_w, box_h = map(float, parts)

                            # Convert to pixel coordinates
                            x1 = int((x_c - box_w/2) * w)
                            y1 = int((y_c - box_h/2) * h)
                            x2 = int((x_c + box_w/2) * w)
                            y2 = int((y_c + box_h/2) * h)

                            print(f"    Object {idx+1}: bbox=({x1},{y1},{x2},{y2}) class={int(cls_id)}")

                            # Validate bbox is within image
                            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                                print(f"      ⚠ Bbox outside image bounds!")
                                self.warnings.append(f"Out-of-bounds bbox in {image_file}")
                else:
                    print(f"  ⚠ No annotations in label file!")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.issues.append(f"Error in {label_file}: {e}")

    def check_class_distribution(self):
        """Analyze class distribution"""
        print("\n[CHECK] Class Distribution")
        print("-" * 60)

        for split in ["train", "val"]:
            label_dir = os.path.join(self.data_dir, "labels", split)

            if not os.path.exists(label_dir):
                continue

            class_counts = defaultdict(int)
            total_objects = 0

            for label_file in os.listdir(label_dir):
                if not label_file.endswith('.txt'):
                    continue

                with open(os.path.join(label_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            cls_id = int(float(parts[0]))
                            class_counts[cls_id] += 1
                            total_objects += 1

            print(f"\n{split.upper()} Split:")
            print(f"  Total objects: {total_objects}")

            for cls_id in sorted(class_counts.keys()):
                count = class_counts[cls_id]
                pct = 100 * count / total_objects if total_objects > 0 else 0
                print(f"    Class {cls_id}: {count} objects ({pct:.1f}%)")

    def generate_report(self):
        """Print final report"""
        print("\n" + "="*60)
        print("DEBUG REPORT")
        print("="*60)

        if self.issues:
            print("\n🚨 CRITICAL ISSUES (must fix):")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\n✓ No critical issues found")

        if self.warnings:
            print("\n⚠ WARNINGS (may need attention):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        else:
            print("\n✓ No warnings")

        print("\n" + "="*60)

        if not self.issues:
            print("✓ Dataset looks good! Ready for training.")
        else:
            print("✗ Please fix critical issues before training.")

        return len(self.issues) == 0

    def run_all_checks(self):
        """Run all diagnostics"""
        print("\n" + "="*60)
        print("DATASET DIAGNOSTIC SCAN")
        print("="*60)

        self.check_directory_structure()
        self.check_image_label_pairs()
        self.check_label_format()
        self.check_image_label_alignment()
        self.check_class_distribution()

        return self.generate_report()


def main():
    debugger = DatasetDebugger()
    success = debugger.run_all_checks()

    if success:
        print("\n✓ You can proceed with training!")
    else:
        print("\n✗ Please fix the issues above before training.")


if __name__ == "__main__":
    main()
