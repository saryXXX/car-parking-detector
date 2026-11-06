# 🚀 Quick Start: Fine-Tuning Your Parking Model

## 30-Second TL;DR

```bash
# 1. Evaluate current model
python evaluate_model.py

# 2. Augment dataset (optional but recommended)
python data_augmentation.py

# 3. Fine-tune
python finetune_model.py --epochs 50 --batch 8

# 4. Compare with baseline
python model_switcher.py --report

# 5. Use new model
# Edit src/tracker.py and change model_name to your new model path
```

---

## Step-by-Step Process

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
# If you have GPU (recommended for training):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Check Current Model Performance
```bash
python evaluate_model.py
```

This generates:
- `evaluation_results.json` - Detailed metrics
- Console output with precision/recall/F1 scores

**Example output:**
```
============================================================
MODEL EVALUATION RESULTS (YOLOv8n)
============================================================
Images Evaluated:      8
Total Ground Truth:    45
Total Detections:      38
True Positives:        35
False Positives:       3
False Negatives:       10
------------------------------------------------------------
Precision:             92.11%
Recall:                77.78%
F1 Score:              84.42%
============================================================
```

### Step 3: Expand Training Data (Optional)
If your accuracy is low, more data helps. Create augmented versions:

```bash
python data_augmentation.py
```

This creates:
- Brightness/contrast variations
- Rotations and flips
- Noise and blur
- HSV color shifts
- ~5-8 augmented images per original

**Result:** `data/parking_dataset_augmented/` with ~10x more training images

### Step 4: Fine-Tune the Model

**Basic (Simple):**
```bash
python finetune_model.py
```

**Advanced (Better Results):**
```bash
# With augmented data, more epochs, larger batch
python finetune_model.py \
  --data-dir data/parking_dataset_augmented \
  --epochs 100 \
  --batch 16 \
  --version v3
```

**What happens:**
1. Loads pre-trained YOLOv8n
2. Trains on your parking data
3. Automatically validates every epoch
4. Stops early if no improvement (20 epochs patience)
5. Saves best model to: `models/parking_yolov8n_v3/weights/best.pt`

**Expected output:**
```
============================================================
STARTING YOLOV8N FINE-TUNING
============================================================
[INFO] Loading base model: yolov8n.pt
[INFO] Training parameters:
  epochs: 100
  batch: 16
  ...

[INFO] Starting training...
Epoch 1/100: 100%|████| 2/2 [00:45<00:00, 22.73s/it]
  Class     Images     Instances          P          R      mAP50   mAP50-95
  all         8         45           0.85      0.82      0.823      0.621

...

============================================================
TRAINING COMPLETE
Model saved: models/parking_yolov8n_v3/weights/best.pt
============================================================
```

### Step 5: View Model Comparison

See how the new model compares to the baseline:

```bash
# List all models
python model_switcher.py --list

# Compare all versions
python model_switcher.py --compare baseline parking_yolov8n_v2 parking_yolov8n_v3

# Generate detailed report
python model_switcher.py --report
```

**Example comparison:**
```
====================================================================================================
MODEL COMPARISON
====================================================================================================

Model                          mAP@0.5         mAP@0.5:0.95     Precision       Recall         
baseline                       0.8234          0.6210           0.8500          0.7800         
parking_yolov8n_v2             0.8956          0.6875           0.9211          0.8844         
parking_yolov8n_v3             0.9234          0.7123           0.9456          0.9011         

[BEST] Model: parking_yolov8n_v3 (mAP@0.5: 0.9234)
====================================================================================================
```

### Step 6: Use New Model

Update your tracker to use the improved model:

**Option A: Edit `src/tracker.py`**
```python
def __init__(self,
    model_name: str = "models/parking_yolov8n_v3/weights/best.pt",  # ← Your new model
    ...
):
```

**Option B: Use at runtime**
```python
from src.tracker import VehicleTracker

tracker = VehicleTracker(model_name="models/parking_yolov8n_v3/weights/best.pt")
```

**Option C: Test with demo script**
```bash
# This will use the updated tracker automatically
python demo_collisions.py --mode both
```

---

## Fine-Tuning Parameters Explained

| Parameter | Default | Description | When to change |
|-----------|---------|-------------|-----------------|
| `--epochs` | 50 | Training iterations | Increase (100+) for better accuracy, decrease (20) for speed |
| `--batch` | 8 | Batch size | Increase (16) if GPU memory allows, decrease if OOM errors |
| `--imgsz` | 640 | Input image size | Use 1280 for better detail on large parking lots |
| `--version` | v2 | Model name suffix | Use v3, v4, etc. to track multiple versions |
| `--data-dir` | parking_dataset | Dataset path | Use `parking_dataset_augmented` for more data |

**GPU Memory Requirements:**
- Batch 8, imgsz 640: ~4GB VRAM (RTX 3050)
- Batch 16, imgsz 640: ~8GB VRAM (RTX 3060)
- Batch 16, imgsz 1280: ~14GB VRAM (RTX 3070)

If you get "CUDA out of memory", reduce batch size or image size.

---

## Common Scenarios

### Scenario 1: "I need quick improvement"
```bash
# Augment + fine-tune with default settings
python data_augmentation.py
python finetune_model.py --data-dir data/parking_dataset_augmented --epochs 50
```
**Time:** ~30-45 minutes on GPU
**Improvement:** +10-15% accuracy

### Scenario 2: "I need maximum accuracy"
```bash
# Full workflow with all options
python data_augmentation.py
python finetune_model.py \
  --data-dir data/parking_dataset_augmented \
  --epochs 100 \
  --batch 16 \
  --imgsz 1280 \
  --version v4
```
**Time:** ~2-3 hours on good GPU
**Improvement:** +15-25% accuracy

### Scenario 3: "Test without retraining"
```bash
# Just compare existing models
python model_switcher.py --compare baseline parking_yolov8n_v2
python model_switcher.py --test parking_yolov8n_v2 --image your_image.jpg
```
**Time:** ~1 minute
**Cost:** None (no training)

---

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution:**
```bash
# Install PyTorch with GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "CUDA out of memory"
**Solution:**
```bash
# Reduce batch size
python finetune_model.py --batch 4
```

### Issue: "dataset.yaml not found"
**Solution:** The script creates it automatically. Ensure your data structure is:
```
data/parking_dataset/
├── images/
│   ├── train/     (your training images)
│   └── val/       (your validation images)
└── labels/
    ├── train/     (YOLO format txt files)
    └── val/
```

### Issue: "Training is very slow"
**Causes & Solutions:**
- Using CPU instead of GPU: Check CUDA is installed
- Batch size too small: Increase to 16 or 32
- Image resolution too high: Use `--imgsz 416` instead of 1280

### Issue: "New model not better than baseline"
**Causes & Solutions:**
- Dataset too small: Run `data_augmentation.py`
- Not enough epochs: Increase to 100+
- Poor data quality: Ensure labels are accurate

---

## Performance Expectations

After fine-tuning on parking-specific data:

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| Precision | 65% | 80% | +15% |
| Recall | 70% | 85% | +15% |
| F1 Score | 67% | 82% | +15% |
| FPS (inference) | 30 | 28 | -2 FPS |

**Note**: Slight FPS decrease due to higher model capacity, but worth it for accuracy.

---

## Next: Integration

Once you have a model you're happy with:

1. Update tracker to use new model
2. Run full pipeline: `python demo_collisions.py`
3. Test on real videos
4. Deploy to production

For full guide, see: [MODEL_FINETUNING_GUIDE.md](MODEL_FINETUNING_GUIDE.md)

---

## Questions?

**My model accuracy is still low:**
- More training data needed
- Augment more with `data_augmentation.py`
- Train longer: `--epochs 150`
- Check data quality (correct labels)

**I want to try a different model size:**
```bash
# Use YOLOv8s (small) instead of nano
python -c "
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.train(data='data/parking_dataset/dataset.yaml', epochs=50, device=0)
"
```

**How do I monitor training?**
- TensorBoard integration built-in
- Automatic validation plots saved
- Check `models/parking_yolov8n_v3/results.png`

---

**Ready to fine-tune?** Start with Step 1! 🚀
