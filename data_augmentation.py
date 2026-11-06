"""
Data Augmentation Script
Expands parking dataset with transformations to improve model generalization
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
import shutil


class DataAugmenter:
    def __init__(self, output_dir: str = "data/parking_dataset_augmented"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def read_yolo_annotation(self, label_file: str) -> list:
        """Read YOLO format annotations"""
        annotations = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        annotations.append([float(x) for x in parts])
        return annotations

    def write_yolo_annotation(self, label_file: str, annotations: list):
        """Write YOLO format annotations"""
        os.makedirs(os.path.dirname(label_file), exist_ok=True)
        with open(label_file, 'w') as f:
            for ann in annotations:
                f.write(f"{int(ann[0])} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

    def augment_brightness(self, img: np.ndarray, factor: float = 0.7) -> np.ndarray:
        """Adjust brightness"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def augment_contrast(self, img: np.ndarray, factor: float = 1.3) -> np.ndarray:
        """Adjust contrast"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * factor, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def augment_rotation(self, img: np.ndarray, annotations: list, angle: float = 15) -> Tuple[np.ndarray, list]:
        """Rotate image and adjust bounding boxes"""
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h))

        # Transform annotations (simplified - keep same for small rotations)
        return rotated, annotations

    def augment_flip(self, img: np.ndarray, annotations: list, direction: str = "h") -> Tuple[np.ndarray, list]:
        """Flip image horizontally or vertically"""
        if direction == "h":
            flipped = cv2.flip(img, 1)  # Horizontal flip
            # Adjust x coordinates
            new_annotations = []
            for ann in annotations:
                new_ann = ann.copy()
                new_ann[1] = 1.0 - ann[1]  # Mirror x_center
                new_annotations.append(new_ann)
        else:
            flipped = cv2.flip(img, 0)  # Vertical flip
            new_annotations = []
            for ann in annotations:
                new_ann = ann.copy()
                new_ann[2] = 1.0 - ann[2]  # Mirror y_center
                new_annotations.append(new_ann)

        return flipped, new_annotations

    def augment_blur(self, img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Add Gaussian blur"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def augment_noise(self, img: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level * 255, img.shape).astype(np.uint8)
        return cv2.add(img, noise)

    def augment_hsv(self, img: np.ndarray, h_shift: int = 15, s_factor: float = 0.7, v_factor: float = 0.8) -> np.ndarray:
        """Adjust HSV values"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Hue rotation
        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180

        # Saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)

        # Value
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_factor, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def augment_dataset(self, source_dir: str = "data/parking_dataset", augmentations_per_image: int = 5):
        """Apply multiple augmentations to expand dataset"""

        # Copy original structure
        train_img_src = os.path.join(source_dir, "images", "train")
        train_label_src = os.path.join(source_dir, "labels", "train")
        val_img_src = os.path.join(source_dir, "images", "val")
        val_label_src = os.path.join(source_dir, "labels", "val")

        train_img_dst = os.path.join(self.output_dir, "images", "train")
        train_label_dst = os.path.join(self.output_dir, "labels", "train")
        val_img_dst = os.path.join(self.output_dir, "images", "val")
        val_label_dst = os.path.join(self.output_dir, "labels", "val")

        os.makedirs(train_img_dst, exist_ok=True)
        os.makedirs(train_label_dst, exist_ok=True)
        os.makedirs(val_img_dst, exist_ok=True)
        os.makedirs(val_label_dst, exist_ok=True)

        # Copy original validation data (don't augment)
        print("[INFO] Copying validation dataset...")
        for file in os.listdir(val_img_src):
            if file.lower().endswith(('.jpg', '.png')):
                shutil.copy(os.path.join(val_img_src, file), os.path.join(val_img_dst, file))
                label_file = file.rsplit('.', 1)[0] + '.txt'
                shutil.copy(
                    os.path.join(val_label_src, label_file),
                    os.path.join(val_label_dst, label_file)
                )

        # Augment training data
        print("[INFO] Augmenting training dataset...")
        image_files = [f for f in os.listdir(train_img_src) if f.lower().endswith(('.jpg', '.png'))]

        total_images = len(image_files)
        counter = 0

        for image_file in image_files:
            image_path = os.path.join(train_img_src, image_file)
            label_file = image_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(train_label_src, label_file)

            img = cv2.imread(image_path)
            if img is None:
                continue

            annotations = self.read_yolo_annotation(label_path)
            file_base = image_file.rsplit('.', 1)[0]
            ext = image_file.rsplit('.', 1)[1]

            # Save original
            cv2.imwrite(os.path.join(train_img_dst, image_file), img)
            self.write_yolo_annotation(os.path.join(train_label_dst, label_file), annotations)

            # Apply augmentations
            augmentation_ops = [
                ("brightness_down", lambda im: self.augment_brightness(im, 0.7), annotations),
                ("brightness_up", lambda im: self.augment_brightness(im, 1.3), annotations),
                ("contrast_up", lambda im: self.augment_contrast(im, 1.3), annotations),
                ("blur", lambda im: self.augment_blur(im), annotations),
                ("noise", lambda im: self.augment_noise(im), annotations),
                ("hsv_shift", lambda im: self.augment_hsv(im), annotations),
                ("flip_h", lambda im: self.augment_flip(im, annotations, "h")[0],
                 self.augment_flip(img, annotations, "h")[1]),
                ("flip_v", lambda im: self.augment_flip(im, annotations, "v")[0],
                 self.augment_flip(img, annotations, "v")[1]),
            ]

            for idx, (aug_name, aug_func, aug_annot) in enumerate(augmentation_ops[:augmentations_per_image]):
                if isinstance(aug_func, tuple):
                    continue

                try:
                    if "flip" in aug_name:
                        # Handle flip separately as it returns both image and annotations
                        aug_img = aug_func(img)
                        aug_label = aug_annot
                    else:
                        aug_img = aug_func(img)
                        aug_label = aug_annot

                    aug_filename = f"{file_base}_{aug_name}.{ext}"
                    aug_labelname = f"{file_base}_{aug_name}.txt"

                    cv2.imwrite(os.path.join(train_img_dst, aug_filename), aug_img)
                    self.write_yolo_annotation(os.path.join(train_label_dst, aug_labelname), aug_label)
                except Exception as e:
                    print(f"[WARNING] Failed to augment {image_file} with {aug_name}: {e}")

            counter += 1
            if counter % 5 == 0:
                print(f"  Processed {counter}/{total_images} images...")

        print(f"\n[SUCCESS] Dataset augmentation complete!")
        print(f"Augmented dataset saved to: {self.output_dir}")
        print(f"Original images: {total_images}")
        print(f"Augmented dataset size: ~{total_images * (1 + augmentations_per_image)} images")


def main():
    augmenter = DataAugmenter()
    augmenter.augment_dataset(source_dir="data/parking_dataset", augmentations_per_image=5)


if __name__ == "__main__":
    main()
