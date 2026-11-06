# 🔧 Troubleshooting Guide

## Issue: 0% Accuracy in Model Evaluation

**Symptom:**
```
Precision:             0.00%
Recall:                0.00%
F1 Score:              0.00%
```

**Root Causes & Solutions:**

### 1. Label Format Issue (Most Common)
YOLO expects normalized coordinates `[0, 1]` range, not pixel coordinates.

**Wrong format:**
```
0 100 150 200 250
```

**Correct format:**
```
0 0.5 0.5 0.3 0.4
```
Where: `class_id x_center y_center width height` (all normalized to image size)

**Fix:**
```bash
# Diagnose format issues
python debug_dataset.py

# Look for errors like:
# ✗ Line 1: Values outside [0,1] range!
```

### 2. Missing or Mismatched Labels
Every training image needs a corresponding `.txt` label file.

**Check:**
```
data/parking_dataset/
├── images/train/
│   ├── image_001.jpg  ← needs matching label
│   └── image_002.jpg  ← needs matching label
└── labels/train/
    ├── image_001.txt  ✓
    └── image_002.txt  ✓
```

**Fix:**
```bash
python debug_dataset.py
# Look for: "Missing labels for: [...]"
```

### 3. Empty Label Files
Label files exist but contain no annotations.

**Fix:**
```bash
# Check if labels are actually populated
ls -lh data/parking_dataset/labels/train/

# If files are 0 bytes or very small, labels are missing
```

### 4. Class ID Mismatch
YOLO expects class IDs to start from 0, and `nc` (number of classes) must match.

**Check `data/parking_dataset/dataset.yaml`:**
```yaml
nc: 1  # ← Must be correct!
names: ["car"]  # ← Only 1 class (0-indexed)
```

**Fix:**
```bash
# Run debug to check class distribution
python debug_dataset.py
# Should show: Class 0: XXX objects (100.0%)
```

---

## Issue: "Invalid CUDA device=0" / GPU Not Available

**Symptom:**
```
ValueError: Invalid CUDA 'device=0' requested. 
torch.cuda.is_available(): False
```

**Solution: Use CPU Training**

Option 1 - Force CPU (recommended for now):
```bash
python finetune_model.py --epochs 50 --batch 8 --cpu
```

Option 2 - Install CUDA-enabled PyTorch:
```bash
# Uninstall CPU version
pip uninstall torch torchvision -y

# Install CUDA 11.8 version (for RTX 30/40 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

**Expected with CPU training:**
- 1-3 hours for 50 epochs (vs 5-15 minutes on GPU)
- Training still works fine, just slower
- Reduce batch size if running out of memory

---

## Issue: "CUDA out of memory"

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate ...
```

**Solutions (in order of preference):**

1. **Reduce batch size:**
   ```bash
   python finetune_model.py --batch 4
   ```

2. **Reduce image size:**
   ```bash
   python finetune_model.py --imgsz 416
   ```

3. **Use smaller model:**
   ```bash
   # Train YOLOv8s instead of YOLOv8n
   # Edit finetune_model.py: self.base_model = "yolov8s.pt"
   ```

4. **Use CPU instead:**
   ```bash
   python finetune_model.py --cpu --batch 4
   ```

---

## Issue: Dataset Not Found

**Symptom:**
```
[ERROR] Dataset YAML not found: data/parking_dataset/dataset.yaml
```

**Check directory structure:**
```bash
# On Windows PowerShell:
Get-ChildItem -Recurse data/parking_dataset/ | Select-Object Name

# Should show:
# images/train/ (with .jpg files)
# images/val/ (with .jpg files)
# labels/train/ (with .txt files)
# labels/val/ (with .txt files)
```

**Fix:**
```bash
# Manually verify structure
python debug_dataset.py
# It will tell you exactly what's wrong
```

---

## Issue: Evaluation Shows Different Results Each Time

**Symptom:**
Results vary between runs of `evaluate_model.py`

**Cause:**
YOLO has random augmentation during detection. This is normal.

**Solution:**
Run multiple times and take average:
```bash
for i in {1..3}; do python evaluate_model.py; done
```

---

## Issue: Training Stops Early / Low Final Loss

**Symptom:**
```
Training stops at epoch 10/50
Best epoch: 5
```

**Cause:**
Early stopping triggered (20 epochs with no improvement)

**Solutions:**

1. **Disable early stopping:**
   ```bash
   python finetune_model.py --patience 50  # Increase patience
   ```

2. **More data:**
   ```bash
   python data_augmentation.py
   python finetune_model.py --data-dir data/parking_dataset_augmented
   ```

3. **Different learning rate:**
   Edit `finetune_model.py`:
   ```python
   "lr0": 0.001,  # Lower initial LR
   "lrf": 0.0001,  # Lower final LR
   ```

---

## Issue: Model Performance Not Improving

**Symptom:**
After fine-tuning, accuracy is same as baseline or worse.

**Causes & Solutions:**

1. **Overfitting to small dataset**
   ```bash
   # Add more augmentation
   python data_augmentation.py
   
   # Increase augmentation strength in finetune_model.py:
   "degrees": 20,  # More rotation
   "scale": 1.0,   # More scaling
   ```

2. **Poor quality labels**
   ```bash
   # Verify labels manually
   python debug_dataset.py
   
   # Check a few images with overlaid boxes:
   # Edit evaluate_model.py to save visualized boxes
   ```

3. **Too few training epochs**
   ```bash
   python finetune_model.py --epochs 100 --patience 30
   ```

4. **Learning rate too high**
   ```python
   # In finetune_model.py:
   "lr0": 0.001,  # Reduce from 0.01
   ```

---

## Issue: "ImportError: No module named 'ultralytics'"

**Solution:**
```bash
pip install -r requirements.txt
# or
pip install ultralytics opencv-python
```

---

## Issue: YOLOv8 Model File Corrupted

**Symptom:**
```
OSError: Can't load 'yolov8n.pt'. File may be corrupted.
```

**Solution:**
```bash
# Remove corrupted model
rm yolov8n.pt

# YOLO will automatically re-download next run
python finetune_model.py

# Or manually download:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## Complete Diagnostic Workflow

Run this if everything seems broken:

```bash
# 1. Diagnose dataset
python debug_dataset.py

# 2. Check model loads
python -c "from ultralytics import YOLO; m = YOLO('yolov8n.pt'); print('✓ Model loads OK')"

# 3. Test evaluation
python evaluate_model.py

# 4. Check PyTorch
python -c "import torch; print(f'✓ PyTorch OK. CUDA: {torch.cuda.is_available()}')"

# 5. Try training on CPU
python finetune_model.py --epochs 5 --batch 4 --cpu

# 6. Check results
ls -lh models/parking_yolov8n_v2/weights/best.pt
```

---

## Getting Help

**If problems persist:**

1. Run complete diagnostic:
   ```bash
   python debug_dataset.py
   ```

2. Share the output showing:
   - Directory structure (✓ or ✗ for each folder)
   - Label format validation
   - Class distribution
   - Any error messages

3. Check:
   - Python version: `python --version` (need 3.8+)
   - PyTorch: `python -c "import torch; print(torch.__version__)"`
   - YOLO: `pip show ultralytics`

4. Common files to check:
   - `data/parking_dataset/dataset.yaml`
   - `data/parking_dataset/labels/val/` (check 1-2 files)
   - `evaluation_results.json` (after running evaluate_model.py)

---

## Quick Reference

| Issue | Command |
|-------|---------|
| Diagnose everything | `python debug_dataset.py` |
| Test current model | `python evaluate_model.py` |
| Fine-tune (CPU) | `python finetune_model.py --cpu --batch 4` |
| Fine-tune (GPU) | `python finetune_model.py --epochs 50` |
| Compare models | `python model_switcher.py --compare baseline parking_yolov8n_v2` |
| Augment data | `python data_augmentation.py` |

---

**Still stuck?** The `debug_dataset.py` script usually identifies 90% of issues. Run it first!
