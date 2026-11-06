"""
Model Evaluation Script
Tests current YOLOv8n model performance on parking dataset
Generates metrics and visualizations for accuracy assessment
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

from ultralytics import YOLO


class ModelEvaluator:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.35):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.results = {
            "total_images": 0,
            "total_ground_truth": 0,
            "total_detections": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "per_class_metrics": {},
            "per_image_results": []
        }

    def load_yolo_labels(self, label_path: str) -> List[Tuple[int, float, float, float, float]]:
        """Load YOLO format labels (class_id, x_center, y_center, width, height in normalized coords)"""
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        labels.append((int(parts[0]), parts[1], parts[2], parts[3], parts[4]))
        return labels

    def denormalize_bbox(self, norm_bbox: Tuple[float, float, float, float],
                         img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized YOLO bbox to pixel coordinates (x1, y1, x2, y2)"""
        x_center, y_center, width, height = norm_bbox
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        return (x1, y1, x2, y2)

    def compute_iou(self, box1: Tuple[int, int, int, int],
                   box2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union (IoU) between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def evaluate_image(self, image_path: str, label_path: str, iou_threshold: float = 0.5) -> Dict:
        """Evaluate model on a single image"""
        img = cv2.imread(image_path)
        if img is None:
            return None

        h, w = img.shape[:2]

        # Load ground truth
        ground_truth = self.load_yolo_labels(label_path)
        gt_boxes = [self.denormalize_bbox(gt[1:], w, h) for gt in ground_truth]

        # Run detection
        results = self.model.predict(image_path, conf=self.conf_threshold, verbose=False)

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box, conf, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
                    det_box = tuple(int(x) for x in box.cpu().tolist())
                    detections.append({
                        "box": det_box,
                        "conf": conf.item(),
                        "cls": int(cls_id.item())
                    })

        # Match detections to ground truth
        matched_gt = set()
        matched_det = set()

        for det_idx, det in enumerate(detections):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = self.compute_iou(det["box"], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                matched_det.add(det_idx)

        tp = len(matched_det)
        fp = len(detections) - tp
        fn = len(gt_boxes) - len(matched_gt)

        return {
            "image": image_path,
            "gt_count": len(gt_boxes),
            "det_count": len(detections),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "detections": detections,
            "ground_truth_count": len(ground_truth)
        }

    def evaluate_dataset(self, data_dir: str = "data/parking_dataset") -> Dict:
        """Evaluate model on entire dataset"""
        val_images_dir = os.path.join(data_dir, "images", "val")
        val_labels_dir = os.path.join(data_dir, "labels", "val")

        if not os.path.exists(val_images_dir):
            print(f"[ERROR] Validation images directory not found: {val_images_dir}")
            return self.results

        all_metrics = {
            "tp": 0, "fp": 0, "fn": 0,
            "images_evaluated": 0,
            "images_with_errors": 0,
            "per_image": []
        }

        image_files = [f for f in os.listdir(val_images_dir) if f.lower().endswith(('.jpg', '.png'))]
        print(f"\n[INFO] Evaluating {len(image_files)} validation images...")

        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(val_images_dir, image_file)
            label_file = image_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(val_labels_dir, label_file)

            result = self.evaluate_image(image_path, label_path)
            if result is None:
                all_metrics["images_with_errors"] += 1
                continue

            all_metrics["tp"] += result["tp"]
            all_metrics["fp"] += result["fp"]
            all_metrics["fn"] += result["fn"]
            all_metrics["images_evaluated"] += 1
            all_metrics["per_image"].append(result)

            if (idx + 1) % 5 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} images...")

        # Calculate metrics
        tp = all_metrics["tp"]
        fp = all_metrics["fp"]
        fn = all_metrics["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        self.results = {
            "total_images": all_metrics["images_evaluated"],
            "total_ground_truth": tp + fn,
            "total_detections": tp + fp,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "per_image_results": all_metrics["per_image"]
        }

        return self.results

    def print_results(self):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS (YOLOv8n)")
        print("="*60)
        print(f"Images Evaluated:      {self.results['total_images']}")
        print(f"Total Ground Truth:    {self.results['total_ground_truth']}")
        print(f"Total Detections:      {self.results['total_detections']}")
        print(f"True Positives:        {self.results['true_positives']}")
        print(f"False Positives:       {self.results['false_positives']}")
        print(f"False Negatives:       {self.results['false_negatives']}")
        print("-"*60)
        print(f"Precision:             {self.results['precision']:.2%}")
        print(f"Recall:                {self.results['recall']:.2%}")
        print(f"F1 Score:              {self.results['f1_score']:.2%}")
        print("="*60)

    def save_results(self, output_file: str = "evaluation_results.json"):
        """Save results to JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n[INFO] Results saved to {output_file}")


def main():
    evaluator = ModelEvaluator(model_path="yolov8n.pt", conf_threshold=0.35)
    results = evaluator.evaluate_dataset()
    evaluator.print_results()
    evaluator.save_results("evaluation_results.json")


if __name__ == "__main__":
    main()
