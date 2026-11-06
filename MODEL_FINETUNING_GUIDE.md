# YOLOv8n Model Fine-Tuning Guide

## Overview
This guide walks you through improving the parking lot detection model accuracy by fine-tuning YOLOv8n on your parking-specific dataset.

## Step 1: Evaluate Current Model Performance

First, test the baseline YOLOv8n model to identify accuracy issues.

```bash
python evaluate_model.py
```

**Output:**
- `evaluation_results.json` - Detailed metrics
- Console output with Precision, Recall, and F1 Score

**What to look for:**
- **Low Precision** → False positives (detecting non-vehicles)
- **Low Recall** → False negatives (missing vehicles)
- **Low F1** → General accuracy issues

---

## Step 2: (Optional) Augment Dataset

If you want to expand your training data, use data augmentation to create synthetic variations:

```bash
python data_augmentation.py
```

**What it does:**
- Creates 5-8 augmented versions per image (brightness, rotation, flip, noise, HSV shifts)
- Generates `data/parking_dataset_augmented/` with ~10x more training samples
- Preserves validation set unchanged

**Use this if:**
- Dataset is small (<50 images)
- Model is overfitting
- Want better generalization to different lighting/conditions

---

## Step 3: Fine-Tune the Model

Train an improved version of YOLOv8n on your parking dataset:

```bash
# Basic fine-tuning (50 epochs, batch size 8)
python finetune_model.py

# Advanced options
python finetune_model.py --epochs 100 --batch 16 --imgsz 640 --version v3

# Using augmented dataset
python finetune_model.py --data-dir data/parking_dataset_augmented
```

**Parameters:**
- `--epochs` - Training iterations (default: 50, increase for better accuracy)
- `--batch` - Batch size (increase if GPU memory allows, max 16-32)
- `--imgsz` - Input image size (default: 640, can use 1280 for detail)
- `--version` - Model version suffix (v2, v3, etc. for tracking)
- `--data-dir` - Dataset directory (original or augmented)

**Expected output:**
- Training progress with loss metrics
- Validation metrics every epoch
- Best model saved to: `models/parking_yolov8n_v2/weights/best.pt`
- Comparison with baseline model

**Training tips:**
- **GPU Required**: Fine-tuning is memory-intensive (requires NVIDIA GPU or significant compute time)
- **Convergence**: Typically 30-50 epochs needed for convergence
- **Early Stopping**: Training stops if validation metric doesn't improve for 20 epochs
- **Data Format**: Ensure YOLO format (class_id, x_center, y_center, width, height in normalized coords)

---

## Step 4: Validate Improved Model

After training, the script automatically validates and compares:

```bash
# Manual validation (if needed)
python -c "
from ultralytics import YOLO
model = YOLO('models/parking_yolov8n_v2/weights/best.pt')
metrics = model.val(data='data/parking_dataset/dataset.yaml', device=0)
"
```

**Expected metrics improvement:**
- Precision: +5-15% (fewer false positives)
- Recall: +10-20% (catch more vehicles)
- F1 Score: +10-15% (overall accuracy)

---

## Step 5: Integrate New Model into Tracker

Update your vehicle tracker to use the improved model:

### Option A: Update Default Model
Edit `src/tracker.py`:

```python
def __init__(self, 
    model_name: str = "models/parking_yolov8n_v2/weights/best.pt",  # ← Change this
    tracker_type: str = "bytetrack.yaml",
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.5
):
```

### Option B: Pass Model at Runtime
```python
from src.tracker import VehicleTracker

tracker = VehicleTracker(model_name="models/parking_yolov8n_v2/weights/best.pt")
```

---

## Step 6: Test in Real Pipeline

Run your complete parking analysis pipeline with the new model:

```bash
# Demo with improved model
python demo_collisions.py --mode both

# Or integrate into your video processing pipeline
python -c "
import cv2
from src.tracker import VehicleTracker
from src.spatial_engine import SpatialEngine
from src.visualizer import ParkingVisualizer

tracker = VehicleTracker(model_name='models/parking_yolov8n_v2/weights/best.pt')
engine = SpatialEngine()
vis = ParkingVisualizer(engine.slots)

cap = cv2.VideoCapture('video_file.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    vehicles = tracker.track(frame)
    analytics = engine.update(vehicles)
    output = vis.render(frame, analytics, vehicles)
    
    cv2.imshow('Smart Parking', output)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
"
```

---

## Model Versions Tracking

Keep multiple model versions for comparison:

| Version | Dataset | Epochs | Accuracy | Notes |
|---------|---------|--------|----------|-------|
| baseline | parking_dataset | - | baseline | Pre-trained YOLOv8n |
| v2 | parking_dataset | 50 | +12% | First fine-tune |
| v3 | parking_dataset_augmented | 100 | +18% | With augmentation |
| v4 | parking_dataset_augmented | 100 (imgsz=1280) | +20% | Higher resolution |

**Compare models:**
```bash
# The script automatically compares during training
# Manual comparison:
python -c "
from ultralytics import YOLO

old = YOLO('yolov8n.pt')
new = YOLO('models/parking_yolov8n_v2/weights/best.pt')

old.val(data='data/parking_dataset/dataset.yaml')
new.val(data='data/parking_dataset/dataset.yaml')
"
```

---

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size: `--batch 4`
- Use smaller image size: `--imgsz 416`
- Enable mixed precision in training code

### "Low accuracy improvement"
- **Cause**: Dataset too small or too similar
- **Solution**: Use data augmentation, collect more diverse data
- Add challenging scenarios (shadows, angles, lighting variations)

### "Validation metrics worse than training"
- **Cause**: Overfitting
- **Solution**: Increase regularization, more data augmentation, reduce model size

### "Model inference is slow"
- Use YOLOv8n (nano) instead of larger variants
- Reduce input image size: `--imgsz 416`
- Enable GPU inference
- Export to ONNX for faster inference

### "Dataset YAML not found"
- Ensure directory structure:
  ```
  data/parking_dataset/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/
  ```
- Run: `python finetune_model.py` (creates dataset.yaml automatically)

---

## Advanced Techniques

### Transfer Learning with Different Base Model
```python
# Use YOLOv8s (small) instead of nano for better accuracy
from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # Download automatically
results = model.train(
    data="data/parking_dataset/dataset.yaml",
    epochs=50,
    imgsz=640,
    device=0
)
```

### Custom Training Loop with Learning Rate Scheduling
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data="data/parking_dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    optimizer="SGD",
    lr0=0.01,      # Initial LR
    lrf=0.0001,    # Final LR ratio
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5
)
```

### Export Trained Model
```python
from ultralytics import YOLO

model = YOLO("models/parking_yolov8n_v2/weights/best.pt")

# Export to different formats
model.export(format="onnx")      # ONNX (cross-platform)
model.export(format="tflite")    # TensorFlow Lite (mobile)
model.export(format="torchscript") # TorchScript (production)
```

---

## Expected Improvements After Fine-Tuning

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Precision (%) | 65 | 77 | +12% |
| Recall (%) | 70 | 85 | +15% |
| F1 Score (%) | 67 | 81 | +14% |
| False Positives | High | Low | ↓ |
| False Negatives | Medium | Low | ↓ |
| Inference Speed | ~30ms/frame | ~35ms/frame | -15% |

**Note**: Actual improvements depend on dataset quality, size, and diversity.

---

## Next Steps

1. ✅ Evaluate baseline model
2. ✅ Augment dataset (optional but recommended)
3. ✅ Fine-tune YOLOv8n
4. ✅ Compare with baseline
5. ✅ Integrate into production pipeline
6. ✅ Monitor performance in real-world scenarios

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Dataset Format](https://docs.ultralytics.com/datasets/detect/)
- [Transfer Learning Guide](https://docs.ultralytics.com/modes/train/)

---

## Questions?

If you encounter issues:
1. Check error messages carefully
2. Verify data format with `evaluate_model.py`
3. Start with default parameters, then adjust
4. Monitor GPU memory and CPU usage
5. Save intermediate models for comparison
