"""
Model Switcher Utility
Easily switch between different model versions and compare performance
"""

import os
import json
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List
import sys


class ModelSwitcher:
    def __init__(self):
        self.models_dir = "models"
        self.baseline_model = "yolov8n.pt"
        self.available_models = self._discover_models()

    def _discover_models(self) -> Dict[str, str]:
        """Discover all available models in models/ directory"""
        models = {
            "baseline": self.baseline_model
        }

        if os.path.exists(self.models_dir):
            for version_dir in os.listdir(self.models_dir):
                version_path = os.path.join(self.models_dir, version_dir)
                if os.path.isdir(version_path):
                    best_pt = os.path.join(version_path, "weights", "best.pt")
                    if os.path.exists(best_pt):
                        models[version_dir] = best_pt

        return models

    def list_models(self):
        """List all available models"""
        print("\n" + "="*60)
        print("AVAILABLE MODELS")
        print("="*60)

        for name, path in self.available_models.items():
            exists = "✓" if os.path.exists(path) else "✗"
            size = ""
            if os.path.exists(path):
                size_bytes = os.path.getsize(path) / (1024**2)
                size = f" ({size_bytes:.1f}MB)"
            print(f"{exists} {name:<30} {path}{size}")

        print("="*60 + "\n")

    def get_model(self, model_name: str = "baseline") -> YOLO:
        """Load a specific model"""
        if model_name not in self.available_models:
            print(f"[ERROR] Model '{model_name}' not found!")
            print("Available models:")
            for name in self.available_models.keys():
                print(f"  - {name}")
            return None

        model_path = self.available_models[model_name]
        print(f"[INFO] Loading model: {model_name}")
        print(f"       Path: {model_path}")

        model = YOLO(model_path)
        return model

    def validate_model(self, model_name: str, dataset_yaml: str = "data/parking_dataset/dataset.yaml") -> Dict:
        """Validate a model on the dataset"""
        model = self.get_model(model_name)
        if model is None:
            return None

        print(f"\n[INFO] Validating {model_name}...")
        metrics = model.val(data=dataset_yaml, device=0, verbose=False)

        results = {
            "model": model_name,
            "map50": float(metrics.box.map50) if hasattr(metrics, 'box') else None,
            "map": float(metrics.box.map) if hasattr(metrics, 'box') else None,
            "precision": float(metrics.box.mp) if hasattr(metrics, 'box') else None,
            "recall": float(metrics.box.mr) if hasattr(metrics, 'box') else None,
        }

        return results

    def compare_models(self, model_names: List[str] = None, dataset_yaml: str = "data/parking_dataset/dataset.yaml"):
        """Compare performance of multiple models"""
        if model_names is None:
            model_names = list(self.available_models.keys())

        print("\n" + "="*100)
        print("MODEL COMPARISON")
        print("="*100)

        comparison_results = []

        for model_name in model_names:
            metrics = self.validate_model(model_name, dataset_yaml)
            if metrics:
                comparison_results.append(metrics)

        # Print comparison table
        print("\n" + "-"*100)
        print(f"{'Model':<30} {'mAP@0.5':<15} {'mAP@0.5:0.95':<15} {'Precision':<15} {'Recall':<15}")
        print("-"*100)

        for result in comparison_results:
            print(f"{result['model']:<30} "
                  f"{(result['map50'] or 0):<15.4f} "
                  f"{(result['map'] or 0):<15.4f} "
                  f"{(result['precision'] or 0):<15.4f} "
                  f"{(result['recall'] or 0):<15.4f}")

        print("="*100 + "\n")

        # Find best model
        if comparison_results:
            best_model = max(comparison_results, key=lambda x: (x['map50'] or 0))
            print(f"[BEST] Model: {best_model['model']} (mAP@0.5: {best_model['map50']:.4f})\n")

        return comparison_results

    def test_model_inference(self, model_name: str, image_path: str = "assets/video_samples/frame_4196207.jpg"):
        """Test inference on a single image"""
        model = self.get_model(model_name)
        if model is None or not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return

        print(f"\n[INFO] Running inference on: {image_path}")
        results = model.predict(image_path, conf=0.35, verbose=False)

        if results and len(results) > 0:
            boxes = results[0].boxes
            print(f"\n[RESULTS] Detections: {len(boxes)} vehicles found")

            for i, (box, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                x1, y1, x2, y2 = [int(x) for x in box]
                print(f"  {i+1}. Box: ({x1}, {y1}, {x2}, {y2}) | Confidence: {conf:.2%}")
        else:
            print("[RESULTS] No detections")

    def export_model(self, model_name: str, export_format: str = "onnx"):
        """Export model to different format"""
        model = self.get_model(model_name)
        if model is None:
            return None

        print(f"\n[INFO] Exporting {model_name} to {export_format}...")
        export_path = model.export(format=export_format)
        print(f"[SUCCESS] Exported to: {export_path}")
        return export_path

    def generate_report(self, output_file: str = "model_comparison_report.json"):
        """Generate comparison report for all models"""
        print("\n[INFO] Generating model comparison report...")

        results = self.compare_models()

        report = {
            "available_models": self.available_models,
            "comparison_results": results,
            "timestamp": str(Path.cwd())
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"[SUCCESS] Report saved to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Model Switcher - Manage and compare YOLO models")
    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument("--validate", type=str, help="Validate a specific model")
    parser.add_argument("--compare", nargs="+", help="Compare multiple models")
    parser.add_argument("--test", type=str, help="Test inference on an image")
    parser.add_argument("--image", type=str, default="assets/video_samples/frame_4196207.jpg",
                       help="Image path for testing")
    parser.add_argument("--export", type=str, help="Export model to format")
    parser.add_argument("--format", type=str, default="onnx", help="Export format (onnx, tflite, etc.)")
    parser.add_argument("--report", action="store_true", help="Generate comparison report")

    args = parser.parse_args()

    switcher = ModelSwitcher()

    if args.list:
        switcher.list_models()

    elif args.validate:
        result = switcher.validate_model(args.validate)
        if result:
            print(f"\n[VALIDATION RESULTS for {args.validate}]")
            for key, value in result.items():
                if key != 'model':
                    print(f"  {key}: {value:.4f}" if value else f"  {key}: N/A")

    elif args.compare:
        switcher.compare_models(args.compare)

    elif args.test:
        switcher.test_model_inference(args.test, args.image)

    elif args.export:
        switcher.export_model(args.export, args.format)

    elif args.report:
        switcher.generate_report()

    else:
        print("\n[INFO] Model Switcher - Usage Examples:\n")
        print("  List all models:")
        print("    python model_switcher.py --list\n")
        print("  Validate a model:")
        print("    python model_switcher.py --validate parking_yolov8n_v2\n")
        print("  Compare models:")
        print("    python model_switcher.py --compare baseline parking_yolov8n_v2 parking_yolov8n_v3\n")
        print("  Test inference:")
        print("    python model_switcher.py --test parking_yolov8n_v2 --image frame.jpg\n")
        print("  Export model:")
        print("    python model_switcher.py --export parking_yolov8n_v2 --format onnx\n")
        print("  Generate report:")
        print("    python model_switcher.py --report\n")


if __name__ == "__main__":
    main()
