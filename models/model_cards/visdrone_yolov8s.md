---
license: agpl-3.0

pipeline_tag: object-detection

library_name: ultralytics

datasets:
  - Voxel51/VisDrone2019-DET

tags:
  - object-detection
  - detectionbench
  - ultralytics
  - pytorch
  - computer-vision
  - aerial-imagery
  - drone
  - visdrone
  - detr
  - roboflow
  - transformer
metrics:
  - map50
  - map50-95
  - precision
  - recall
  - f1

base_model: "Ultralytics/YOLOv8"

model-index:
  - name: YOLOv8s Finetuned on VisDrone-DET
    results:
      - task:
          type: object-detection
          name: Object Detection
        dataset:
          name: VisDrone-DET
          type: visdrone
        metrics:
          - type: mAP50
            value: 43.47
            name: mAP@50 (test split)
          - type: mAP50-95
            value: 25.77
            name: mAP@50-95 (test split)
          - type: precision
            value: 56.09
            name: Precision (test split)
          - type: recall
            value: 44.51
            name: Recall (test split)
        source:
          url: https://github.com/dronefreak/DetectionBench
          name: DetectionBench
---


# YOLOv8s Finetuned on VisDrone-DET

Fine-tuned YOLOv8s object detector on the **VisDrone-DET** benchmark dataset, trained and evaluated as part of [DetectionBench](https://github.com/dronefreak/DetectionBench) -- a framework for reproducibly benchmarking modern object detectors with identical training recipes and evaluation metrics across multiple real-world datasets.

<!--
  Demo banner: side-by-side video of this checkpoint's detections on two
  VisDrone-DET test clips. Media lives under assets/ in this repo. The <video>
  renders on the Hugging Face model page (absolute resolve/ URL); on GitHub
  the nested <img> poster is shown instead.
-->
<p align="center"><video controls autoplay loop muted playsinline width="900" poster="https://huggingface.co/dronefreak/visdrone-yolov8s/resolve/main/assets/demo_banner_poster.jpg" src="https://huggingface.co/dronefreak/visdrone-yolov8s/resolve/main/assets/demo_banner.mp4"><img src="https://huggingface.co/dronefreak/visdrone-yolov8s/resolve/main/assets/demo_banner_poster.jpg" alt="YOLOv8s detections on two VisDrone-DET test clips" width="900"></video></p>

<br>

<!-- ROW 1: Identity & Tech Stack -->
<div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;">
  <img src="https://img.shields.io/badge/Task-Object_Detection-blue?style=flat-square" alt="Task">
  <img src="https://img.shields.io/badge/Framework-Ultralytics_YOLO-0aa1a7?style=flat-square" alt="Framework">
  <img src="https://img.shields.io/badge/Base_Model-YOLOv8s-purple?style=flat-square" alt="Base Model">
</div>

<!-- ROW 2: Performance Metrics -->
<div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;">
  <img src="https://img.shields.io/badge/mAP@50-43.47%25-success?style=flat-square" alt="mAP@50">
  <img src="https://img.shields.io/badge/mAP@50:95-25.77%25-orange?style=flat-square" alt="mAP@50:95">
  <img src="https://img.shields.io/badge/Params-11.2M-lightgrey?style=flat-square" alt="Params">
</div>

<!-- ROW 3: Metadata -->
<div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-bottom: 24px; flex-wrap: wrap;">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-lightgrey?style=flat-square" alt="License">
  <a href="https://github.com/dronefreak/DetectionBench"><img src="https://img.shields.io/badge/Source-DetectionBench-black?style=flat-square" alt="Source"></a>
</div>

---

## Usage

### Install Dependencies

```bash
pip install ultralytics huggingface_hub
```

### Load Model from Hugging Face

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download(
    repo_id="dronefreak/visdrone-yolov8s",
    filename="best.pt"
)

model = YOLO(weights)
```

### Run Inference

```python
results = model.predict(
    source="image.jpg",
    conf=0.25
)

results[0].show()
```
---

## Performance

Evaluated on the VisDrone-DET **test** split, using DetectionBench's standard evaluation pipeline (`detectionbench-evaluate`).

| Metric     | Score (%)       |
| ---------- | --------------- |
| mAP@50     | 43.47     |
| mAP@50-95  | 25.77   |
| Precision  | 56.09 |
| Recall     | 44.51    |
| F1 Score   | 49.63        |
| Parameters | 11.2M    |
| FLOPs      | 28.6B (at 640 px)     |

---

## VisDrone-DET Model Zoo

Every model DetectionBench has trained and evaluated on VisDrone-DET so far, for full transparency -- see [DetectionBench](https://github.com/dronefreak/DetectionBench) for the smaller, curated comparison set used on the project README.

| Model                | mAP@50        | mAP@50-95       | Precision         | Recall         |
| --------------------- | ------------- | --------------- | ----------------- | -------------- |
| YOLO26m | 49.11 | 29.24 | 60.21 | 49.76 |
| YOLO11m | 48.03 | 28.56 | 59.34 | 48.3 |
| YOLOv9m | 47.62 | 28.52 | 59.26 | 48.4 |
| YOLOv10m | 46.73 | 27.72 | 59.23 | 47.35 |
| YOLOv8m | 45.47 | 26.94 | 57.86 | 46.39 |
| YOLOv9s | 45.38 | 26.95 | 56.95 | 46.19 |
| YOLO26s | 44.87 | 26.43 | 56.44 | 45.41 |
| YOLOv10s | 44.51 | 26.28 | 55.97 | 45.61 |
| YOLO11s | 43.64 | 25.88 | 54.48 | 45.01 |
| **YOLOv8s** | **43.47** | **25.77** | **56.09** | **44.51** |
| YOLOv9t | 40.67 | 23.73 | 52.84 | 41.78 |
| YOLO26n | 39.9 | 22.95 | 51.14 | 42.08 |
| YOLOv10n | 39.8 | 23.08 | 51.22 | 41.5 |
| YOLOv8n | 39.69 | 23.03 | 52.02 | 41.36 |
| RF-DETR Medium | 39.62 | 21.99 | 70.62 | 46.76 |
| YOLO11n | 39.52 | 23.0 | 51.49 | 41.02 |
| RF-DETR Small | 39.13 | 21.68 | 64.38 | 49.47 |
| RF-DETR Nano | 37.92 | 20.88 | 69.02 | 46.21 |
---

## Per-Class Performance

| Class                      | mAP@50          | mAP@50-95         |
| -------------------------- | --------------- | ----------------- |
| pedestrian       | 43.83 | 18.75 |
| people       | 27.18 | 10.09 |
| bicycle       | 22.34 | 10.13 |
| car       | 81.99 | 53.56 |
| van       | 47.71 | 33.21 |
| truck       | 50.0 | 33.68 |
| tricycle       | 30.14 | 17.7 |
| awning-tricycle       | 23.52 | 14.78 |
| bus       | 62.63 | 45.78 |
| motor       | 45.32 | 20.06 |
| others       | 0.0 | 0.0 |

![Normalized Confusion Matrix](confusion_matrix_normalized.png)

---

## Dataset

This model was trained on **VisDrone-DET**. For the full dataset description, provenance, license, and citation, see the dataset card:

https://huggingface.co/datasets/Voxel51/VisDrone2019-DET

### Classes

* pedestrian
* people
* bicycle
* car
* van
* truck
* tricycle
* awning-tricycle
* bus
* motor
* others
---

## Training Configuration

| Setting          | Value                           |
| ---------------- | -------------------------------- |
| Dataset          | VisDrone-DET       |
| Framework        | Ultralytics YOLO                  |
| Training Toolkit | DetectionBench                   |
| Epochs (configured max) | 100 |
| Epochs (actually trained) | 100 |
| Early Stopping Patience | 25 |
| Batch Size | auto (Ultralytics AutoBatch) |
| Image Size | 1280 |
| Optimizer | SGD |
| Initial Learning Rate | 0.01 |
| Seed | 0 |
---

## Repository Contents

```text
best.pt
results.csv
args.yaml
BoxPR_curve.png
BoxF1_curve.png
BoxP_curve.png
BoxR_curve.png
confusion_matrix.png
confusion_matrix_normalized.png
val_batch0_pred.jpg
visdrone_yolov8s_showcase.jpg
assets/demo_banner.mp4
assets/demo_banner_poster.jpg
README.md
```

---

## Related Resources

* [VisDrone-DET dataset card](https://huggingface.co/datasets/Voxel51/VisDrone2019-DET) on Hugging Face
* [DetectionBench](https://github.com/dronefreak/DetectionBench) -- reproducible benchmarks for modern object detectors on real-world datasets

---

## Training Framework

This model was trained using [DetectionBench](https://github.com/dronefreak/DetectionBench), an open-source framework for benchmarking object detectors across multiple real-world datasets with a common pipeline.

Features include:

* A dataset-adapter registry for converting real-world datasets into a canonical format
* Identical training/evaluation recipes across model families (Ultralytics YOLO/RT-DETR, RF-DETR)
* Hardware profiling (latency, FPS, VRAM, parameters, FLOPs)
* One-command reproducibility via versioned Hydra configs

If you find this model useful, please consider starring the repository.

---

## Known Limitations

* Severe class imbalance: `car` (42.21%) and `pedestrian` (23.12%) account for two-thirds of all annotated boxes in the training set, while `awning-tricycle` (0.95%) and `tricycle` (1.40%) are rare -- the `others` class has zero annotated instances in the training set entirely and is effectively unusable (always 0 AP).
* Extreme small-object density: ~53 annotated boxes per image on average, with roughly 69% of boxes covering under 0.1% of the image area -- consistent with VisDrone's aerial small-object detection challenge (objects captured from significant altitude).
* The original authors license VisDrone under CC BY-NC-SA 3.0 -- non-commercial research use only (see the dataset's homepage); this applies to any model trained on it, not only the raw images.
* These RF-DETR checkpoints were trained/evaluated directly through DetectionBench. The YOLO/RT-DETR rows in the External VisDrone Model Zoo comparison below were trained via a separate companion codebase, not reproduced inside DetectionBench -- see that collection for their own training details and caveats.
---

## Citation

If you use this model in your research, please consider citing the dataset and the model architecture:

```
@article{zhu2018vision,
  title={Vision meets drones: A challenge},
  author={Zhu, Pengfei and Wen, Longyin and Bian, Xiao and Ling, Haibin and Hu, Qinghua},
  journal={arXiv preprint arXiv:1804.07437},
  year={2018}
}
```

```bibtex
No official YOLOv8 research paper has been published by Ultralytics; this is their own recommended software citation instead:

@software{jocher2023yolov8,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```