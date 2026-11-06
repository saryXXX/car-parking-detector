"""
Fine-tuning Pipeline for YOLOv8n on Parking Dataset
Trains an improved model version with better accuracy for parking scenarios
Supports both GPU and CPU training
"""

import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import shutil


class ParkingModelTrainer:
    def __init__(self, base_model: str = "yolov8n.pt", output_dir: str = "models"):
        self.base_model = base_model
        self.output_dir = output_dir
        self.model_name = None
        os.makedirs(output_dir, exist_ok=True)

    def prepare_dataset_yaml(self, data_dir: str = "data/parking_dataset") -> str:
        """
        Create dataset.yaml for YOLO training
        YOLO expects: images/train, images/val, labels/train, labels/val
        """
        dataset_yaml_path = os.path.join(data_dir, "dataset.yaml")

        # Get absolute paths
        abs_data_dir = os.path.abspath(data_dir)
        train_img_dir = os.path.join(abs_data_dir, "images", "train")
        val_img_dir = os.path.join(abs_data_dir, "images", "val")
        train_label_dir = os.path.join(abs_data_dir, "labels", "train")
        val_label_dir = os.path.join(abs_data_dir, "labels", "val")

        # Create dataset config
        dataset_config = {
            "path": abs_data_dir,
            "train": train_img_dir,
            "val": val_img_dir,
            "nc": 1,  # Number of classes (only "car" for parking)
            "names": ["car"]
        }

        # Write YAML file
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"[INFO] Dataset YAML created: {dataset_yaml_path}")
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

        Args:
            dataset_yaml: Path to dataset.yaml
            epochs: Number of training epochs
            imgsz: Input image size
            batch_size: Batch size for training
            patience: Early stopping patience (no improvement for N epochs)
            model_version: Version suffix (e.g., v2, v3)
            force_cpu: Force CPU training even if GPU available
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
            print(f"[INFO] GPU detected - CUDA available. Using GPU device 0")
            print(f"       GPU: {torch.cuda.get_device_name(0)}")
            print(f"       VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            device = "cpu"
            print("[⚠️  INFO] GPU not detected - using CPU training")
            print("[TIP] GPU training is much faster. To enable:")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("[NOTE] CPU training will be slower but will work. Reduce batch size if memory is low.")

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
            "workers": 0 if device == "cpu" else 4,  # No workers for CPU
            "augment": True,  # Enable data augmentation
            "mosaic": 1.0,  # Mosaic augmentation
            "flipud": 0.5,  # Flip upside down
            "fliplr": 0.5,  # Flip left-right
            "degrees": 10,  # Rotation
            "translate": 0.1,  # Translation
            "scale": 0.5,  # Scaling
            "hsv_h": 0.015,  # HSV hue
            "hsv_s": 0.7,  # HSV saturation
            "hsv_v": 0.4,  # HSV value
            "close_mosaic": 10,  # Disable mosaic in final epochs
            "optimizer": "SGD",  # SGD or Adam
            "lr0": 0.01,  # Initial learning rate
            "lrf": 0.01,  # Final learning rate ratio
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 7.5,  # Box loss gain
            "cls": 0.5,  # Class loss gain
            "dfl": 1.5,  # DFL loss gain
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
        print("[INFO] This may take a while on CPU. Estimated time: 30min-2hrs depending on epochs\n")

        try:
            results = model.train(**train_params)
        except Exception as e:
            print(f"\n[ERROR] Training failed: {e}")
            print("\nTroubleshooting tips:")
            print("1. Check dataset.yaml exists and is valid")
            print("2. Verify images and labels are in correct directories")
            print("3. Run 'python debug_dataset.py' to diagnose dataset issues")
            print("4. If memory error, reduce --batch or --imgsz")
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

    def validate_model(self, model_path: str, dataset_yaml: str) -> dict:
        """Validate the trained model"""
        print(f"\n[INFO] Validating model: {model_path}")

        device = "cpu" if not torch.cuda.is_available() else 0
        model = YOLO(model_path)
        metrics = model.val(data=dataset_yaml, device=device)
        return metrics

    def export_model(self, model_path: str, export_format: str = "onnx") -> str:
        """Export trained model to different formats"""
        print(f"\n[INFO] Exporting model to {export_format}")
        model = YOLO(model_path)
        export_path = model.export(format=export_format)
        print(f"[INFO] Exported model: {export_path}")
        return export_path

    def compare_models(self, old_model: str, new_model: str, dataset_yaml: str):
        """Compare performance between old and new models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        device = "cpu" if not torch.cuda.is_available() else 0

        print(f"\n[EVALUATION] Old Model: {old_model}")
        old_model_obj = YOLO(old_model)
        old_results = old_model_obj.val(data=dataset_yaml, device=device, verbose=False)

        print(f"\n[EVALUATION] New Model: {new_model}")
        new_model_obj = YOLO(new_model)
        new_results = new_model_obj.val(data=dataset_yaml, device=device, verbose=False)

        print("\n" + "-"*60)
        print("COMPARISON RESULTS")
        print("-"*60)
        print(f"{'Metric':<20} {'Old Model':<15} {'New Model':<15} {'Improvement':<15}")
        print("-"*60)

        # Compare key metrics
        metrics_to_compare = [
            ("mAP@0.5", "map50"),
            ("mAP@0.5:0.95", "map"),
        ]

        for metric_name, metric_key in metrics_to_compare:
            old_val = getattr(old_results, metric_key, None)
            new_val = getattr(new_results, metric_key, None)

            if old_val is not None and new_val is not None:
                improvement = new_val - old_val
                improvement_pct = (improvement / old_val * 100) if old_val != 0 else 0
                print(f"{metric_name:<20} {old_val:<15.4f} {new_val:<15.4f} {improvement_pct:+.2f}%")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n for parking lot detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--version", type=str, default="v2", help="Model version suffix")
    parser.add_argument("--data-dir", type=str, default="data/parking_dataset", help="Dataset directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (slower but works without GPU)")

    args = parser.parse_args()

    # Initialize trainer
    trainer = ParkingModelTrainer(base_model="yolov8n.pt", output_dir="models")

    # Prepare dataset
    dataset_yaml = trainer.prepare_dataset_yaml(data_dir=args.data_dir)

    # Check if dataset exists
    if not os.path.exists(dataset_yaml):
        print(f"[ERROR] Dataset YAML not found: {dataset_yaml}")
        print("[INFO] Make sure your data structure is:")
        print("  data/parking_dataset/")
        print("    ├── images/")
        print("    │   ├── train/")
        print("    │   └── val/")
        print("    └── labels/")
        print("        ├── train/")
        print("        └── val/")
        print("\n[TIP] Run 'python debug_dataset.py' to diagnose dataset issues")
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

    # Validate new model
    print("\n[INFO] Validating trained model...")
    trainer.validate_model(trained_model, dataset_yaml)

    # Compare with base model
    print("\n[INFO] Comparing with base model...")
    trainer.compare_models("yolov8n.pt", trained_model, dataset_yaml)

    print("\n[SUCCESS] Fine-tuning complete!")
    print(f"New model saved at: {trained_model}")
    print(f"Use this model by updating tracker.py to use: {trained_model}")


if __name__ == "__main__":
    main()
