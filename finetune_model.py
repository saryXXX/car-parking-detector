"""
Fine-tuning Pipeline for YOLOv8n on Parking Dataset
Trains an improved model version with better accuracy for parking scenarios
Supports both GPU and CPU training
FIXED: Uses existing dataset.yaml and respects correct class count
"""

import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict


class ParkingModelTrainer:
    def __init__(self, base_model: str = "yolov8n.pt", output_dir: str = "models"):
        self.base_model = base_model
        self.output_dir = output_dir
        self.model_name = None
        os.makedirs(output_dir, exist_ok=True)

    def prepare_dataset_yaml(self, data_dir: str = "data/parking_dataset") -> str:
        """
        Use existing dataset.yaml if valid, otherwise create from scratch
        Never override an existing correct configuration
        """
        dataset_yaml_path = os.path.join(data_dir, "dataset.yaml")

        # Check if dataset.yaml already exists with correct format
        if os.path.exists(dataset_yaml_path):
            print(f"[INFO] Found existing dataset.yaml: {dataset_yaml_path}")
            try:
                with open(dataset_yaml_path, 'r') as f:
                    existing_config = yaml.safe_load(f)

                # Verify it has the required fields
                if all(key in existing_config for key in ['path', 'train', 'val', 'nc', 'names']):
                    print(f"[✓] Using existing config with {existing_config['nc']} classes: {existing_config['names']}")
                    return dataset_yaml_path
            except Exception as e:
                print(f"[WARNING] Could not read existing dataset.yaml: {e}")

        # Auto-detect classes from labels
        print(f"[INFO] Auto-detecting classes from labels...")
        class_counts = defaultdict(int)
        label_dir = os.path.join(data_dir, "labels", "train")

        if os.path.exists(label_dir):
            for label_file in os.listdir(label_dir):
                if label_file.endswith('.txt'):
                    with open(os.path.join(label_dir, label_file), 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                cls_id = int(float(parts[0]))
                                class_counts[cls_id] += 1

        num_classes = len(class_counts) if class_counts else 1
        class_names = [f"class_{i}" for i in range(num_classes)]

        print(f"[INFO] Detected {num_classes} classes:")
        for cls_id, count in sorted(class_counts.items()):
            print(f"       Class {cls_id}: {count} objects")

        # Get absolute paths
        abs_data_dir = os.path.abspath(data_dir)
        train_img_dir = os.path.join(abs_data_dir, "images", "train")
        val_img_dir = os.path.join(abs_data_dir, "images", "val")

        # Create dataset config
        dataset_config = {
            "path": abs_data_dir,
            "train": train_img_dir,
            "val": val_img_dir,
            "nc": num_classes,
            "names": class_names
        }

        # Write YAML file
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"[✓] Dataset YAML created with {num_classes} classes")
        return dataset_yaml_path

    def finetune(self,
                 dataset_yaml: str,
                 epochs: int = 50,
                 imgsz: int = 640,
                 batch_size: int = 8,
                 patience: int = 20,
                 model_version: str = "v2",
                 force_cpu: bool = False) -> str:
        """
        Fine-tune YOLOv8n on parking dataset
        """
        print("\n" + "="*60)
        print("STARTING YOLOV8N FINE-TUNING")
        print("="*60)

        # Detect device
        cuda_available = torch.cuda.is_available()
        if force_cpu:
            device = "cpu"
            print("[INFO] Forcing CPU training (as requested)")
        elif cuda_available:
            device = 0  # Use first GPU
            print(f"[INFO] GPU detected - using CUDA device 0")
        else:
            device = "cpu"
            print("[⚠️  INFO] GPU not detected - using CPU training")
            print("[TIP] GPU training is much faster. To enable:")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

        # Load base model
        print(f"\n[INFO] Loading base model: {self.base_model}")
        model = YOLO(self.base_model)

        # Adjust batch size for CPU if needed
        effective_batch_size = batch_size
        if device == "cpu" and batch_size > 4:
            print(f"[WARNING] Reducing batch size from {batch_size} to 4 for CPU training")
            effective_batch_size = 4

        # Training parameters
        train_params = {
            "data": dataset_yaml,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": effective_batch_size,
            "patience": patience,
            "save": True,
            "device": device,
            "workers": 0 if device == "cpu" else 4,
            "augment": True,
            "mosaic": 1.0,
            "flipud": 0.5,
            "fliplr": 0.5,
            "degrees": 10,
            "translate": 0.1,
            "scale": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "close_mosaic": 10,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "project": self.output_dir,
            "name": f"parking_yolov8n_{model_version}",
            "exist_ok": False,
            "verbose": True
        }

        print(f"\n[INFO] Training parameters:")
        for key, value in train_params.items():
            if key not in ["data"]:
                print(f"  {key}: {value}")

        # Train model
        print("\n[INFO] Starting training...")
        print("[INFO] This may take a while on CPU. Estimated time: 30min-2hrs\n")

        try:
            results = model.train(**train_params)
        except Exception as e:
            print(f"\n[ERROR] Training failed: {e}")
            print("\nTroubleshooting tips:")
            print("1. Check dataset.yaml exists and is valid")
            print("2. Verify images and labels are in correct directories")
            print("3. Run 'python debug_dataset.py' to diagnose dataset issues")
            raise

        # Get the trained model path
        trained_model_path = os.path.join(
            self.output_dir,
            f"parking_yolov8n_{model_version}",
            "weights",
            "best.pt"
        )

        self.model_name = trained_model_path

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print(f"Model saved: {trained_model_path}")
        print("="*60)

        return trained_model_path

    def validate_model(self, model_path: str, dataset_yaml: str):
        """Validate the trained model"""
        print(f"\n[INFO] Validating model: {model_path}")

        device = "cpu" if not torch.cuda.is_available() else 0
        model = YOLO(model_path)
        metrics = model.val(data=dataset_yaml, device=device)
        return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n for parking lot detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--version", type=str, default="v2", help="Model version suffix")
    parser.add_argument("--data-dir", type=str, default="data/parking_dataset", help="Dataset directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")

    args = parser.parse_args()

    # Initialize trainer
    trainer = ParkingModelTrainer(base_model="yolov8n.pt", output_dir="models")

    # Prepare dataset - DOES NOT OVERRIDE existing correct config
    dataset_yaml = trainer.prepare_dataset_yaml(data_dir=args.data_dir)

    # Check if dataset exists
    if not os.path.exists(dataset_yaml):
        print(f"\n[ERROR] Dataset YAML not found: {dataset_yaml}")
        return

    # Fine-tune model
    trained_model = trainer.finetune(
        dataset_yaml=dataset_yaml,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        model_version=args.version,
        force_cpu=args.cpu
    )

    print("\n[SUCCESS] Fine-tuning complete!")
    print(f"New model saved at: {trained_model}")


if __name__ == "__main__":
    main()
