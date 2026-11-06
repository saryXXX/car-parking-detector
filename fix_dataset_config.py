"""
Fix Dataset Configuration
Updates dataset.yaml to match actual class distribution
"""

import os
import yaml
from collections import defaultdict


def analyze_classes(data_dir: str = "data/parking_dataset") -> dict:
    """Analyze what classes exist in labels"""
    class_names = {}
    class_counts = defaultdict(int)

    for split in ["train", "val"]:
        label_dir = os.path.join(data_dir, "labels", split)

        if not os.path.exists(label_dir):
            continue

        for label_file in os.listdir(label_dir):
            if not label_file.endswith('.txt'):
                continue

            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        cls_id = int(float(parts[0]))
                        class_counts[cls_id] += 1

    return class_counts


def create_corrected_dataset_yaml(data_dir: str = "data/parking_dataset"):
    """Create corrected dataset.yaml with correct number of classes"""

    # Analyze actual classes
    class_counts = analyze_classes(data_dir)
    num_classes = len(class_counts)

    print("\n" + "="*60)
    print("FIXING DATASET CONFIGURATION")
    print("="*60)

    print(f"\n[INFO] Found {num_classes} classes in dataset:")
    for cls_id in sorted(class_counts.keys()):
        print(f"  Class {cls_id}: {class_counts[cls_id]} objects")

    # Create class names based on detected classes
    # Common parking lot labels: occupied, empty, reserved
    class_name_map = {
        0: "vehicle_or_slot",
        1: "parked_vehicle",
        2: "empty_slot",
        3: "reserved",
        4: "handicap"
    }

    class_names = [class_name_map.get(i, f"class_{i}") for i in range(num_classes)]

    print(f"\n[INFO] Class names (you can customize these):")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    # Get absolute paths
    abs_data_dir = os.path.abspath(data_dir)
    train_img_dir = os.path.join(abs_data_dir, "images", "train")
    val_img_dir = os.path.join(abs_data_dir, "images", "val")

    # Create corrected dataset config
    dataset_config = {
        "path": abs_data_dir,
        "train": train_img_dir,
        "val": val_img_dir,
        "nc": num_classes,
        "names": class_names
    }

    # Write corrected YAML
    dataset_yaml_path = os.path.join(data_dir, "dataset.yaml")
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"\n[SUCCESS] Fixed dataset.yaml saved to: {dataset_yaml_path}")
    print("\nContent:")
    with open(dataset_yaml_path, 'r') as f:
        print(f.read())

    print("\n" + "="*60)
    print("CONFIGURATION FIXED!")
    print("="*60)
    print("\nYou can now proceed with fine-tuning:")
    print("  python finetune_model.py --epochs 50 --batch 8 --cpu")

    return dataset_yaml_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix dataset configuration")
    parser.add_argument("--data-dir", type=str, default="data/parking_dataset", help="Dataset directory")
    parser.add_argument("--customize", action="store_true", help="Customize class names")

    args = parser.parse_args()

    if args.customize:
        class_counts = analyze_classes(args.data_dir)
        print("\n[INFO] Current classes:")
        for cls_id in sorted(class_counts.keys()):
            print(f"  Class {cls_id}: {class_counts[cls_id]} objects")

        print("\n[INFO] Enter class names (comma-separated):")
        print("Example: vehicle,parking_slot,empty")
        class_names_input = input("> ").strip()

        if class_names_input:
            class_names = [n.strip() for n in class_names_input.split(',')]
            num_classes = len(class_counts)

            if len(class_names) != num_classes:
                print(f"[ERROR] Expected {num_classes} names, got {len(class_names)}")
                return

            abs_data_dir = os.path.abspath(args.data_dir)
            train_img_dir = os.path.join(abs_data_dir, "images", "train")
            val_img_dir = os.path.join(abs_data_dir, "images", "val")

            dataset_config = {
                "path": abs_data_dir,
                "train": train_img_dir,
                "val": val_img_dir,
                "nc": num_classes,
                "names": class_names
            }

            dataset_yaml_path = os.path.join(args.data_dir, "dataset.yaml")
            with open(dataset_yaml_path, 'w') as f:
                yaml.dump(dataset_config, f, default_flow_style=False)

            print(f"\n[SUCCESS] Custom dataset.yaml saved!")
    else:
        create_corrected_dataset_yaml(args.data_dir)


if __name__ == "__main__":
    main()
