---
license: cc-by-4.0
base_model:
- Ultralytics/YOLOv8
base_model_relation: finetune
pipeline_tag: object-detection
library_name: ultralytics
num_parameters: 11137922
github: https://github.com/rfonod/geo-trax
language:
- en
tags:
- ultralytics
- yolov8
- object-detection
- aerial-imagery
- drone
- vehicle-detection
- birds-eye-view
- geo-trax
- trajectory
- urban-traffic
- tracking
- arxiv:2411.02136
- arxiv:2611.17190
datasets:
- rfonod/songdo-vision
- Voxel51/VisDrone2019-DET
- detection-datasets/coco
model-index:
- name: rfonod/geo-trax/geotrax_hbb_yolov8s_1920_v1
  results:
  - task:
      type: object-detection
    dataset:
      type: rfonod/songdo-vision
      name: Songdo Vision (v1)
      revision: v1
      split: test
    metrics:
      - type: precision  # mAP@0.5 not available as a standard metric type on HF
        value: 0.951
        name: mAP@0.5
      - type: precision
        value: 0.711
        name: mAP@0.5:0.95
      - type: precision
        value: 0.911
        name: Precision
      - type: recall
        value: 0.935
        name: Recall
---

# Geo-trax: YOLOv8s Vehicle Detector for Drone BEV Imagery

[![GitHub](https://img.shields.io/badge/GitHub-geo--trax-blue?logo=github)](https://github.com/rfonod/geo-trax)
[![PyPI](https://img.shields.io/pypi/v/geo-trax)](https://pypi.org/project/geo-trax/)
[![Docs](https://img.shields.io/badge/docs-rfonod.github.io%2Fgeo--trax-blue)](https://rfonod.github.io/geo-trax/)
[![Demo Space](https://img.shields.io/badge/🤗%20Space-Live%20Demo-yellow)](https://huggingface.co/spaces/rfonod/geo-trax)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](LICENSE)
[![Paper](https://img.shields.io/badge/Journal-10.1016%2Fj.trc.2025.105205-blue)](https://doi.org/10.1016/j.trc.2025.105205)
[![arXiv](https://img.shields.io/badge/arXiv-2411.02136-b31b1b)](https://arxiv.org/abs/2411.02136)
[![Songdo Vision](https://img.shields.io/badge/🤗%20Dataset-rfonod%2Fsongdo--vision-yellow)](https://huggingface.co/datasets/rfonod/songdo-vision)
[![Songdo Traffic](https://img.shields.io/badge/🤗%20Dataset-rfonod%2Fsongdo--traffic-yellow)](https://huggingface.co/datasets/rfonod/songdo-traffic)
[![Collection](https://img.shields.io/badge/🤗%20Collection-Trajectory%20Extraction%20%26%20Stabilization-yellow)](https://huggingface.co/collections/rfonod/trajectory-extraction-and-stabilization-68aa0f4c5372ae5a8edf22ce)
[![Website](https://img.shields.io/badge/REAL%20Lab-Geo--trax-informational)](https://www.real-lab.ch/geo-trax)
[![YouTube](https://img.shields.io/badge/YouTube-Demo-red?logo=youtube&logoColor=red)](https://youtu.be/gOGivL9FFLk)
[![Geo Export](https://img.shields.io/badge/Geo%20Export-GeoJSON%20%7C%20GeoPackage-4c8)](https://rfonod.github.io/geo-trax/guides/export/)

This is the default detection model for **[Geo-trax](https://github.com/rfonod/geo-trax)**, a
comprehensive pipeline for extracting georeferenced vehicle trajectories from high-altitude drone
(bird's-eye view) video footage. The model detects vehicles in aerial imagery and underpins the
results reported in the associated [publication](https://doi.org/10.1016/j.trc.2025.105205).

![Geo-trax Output Visualization](https://raw.githubusercontent.com/rfonod/geo-trax/main/assets/geo-trax_visualization.webp)

🎬 This accelerated animation previews some of the capabilities of Geo-trax. Watch the full
demonstration (~4 min) on [YouTube](https://youtu.be/gOGivL9FFLk).

**Highlights**

- 🎯 **0.951 mAP@50** on the Songdo Vision test split, with mAP@50 ≥ 0.988 on cars and buses.
- 🛰️ Built for **high-altitude, top-down drone video** (140–150 m), where generic detectors miss
  small vehicles; native at **1920 px**.
- 📦 **Drop-in**: `pip install geo-trax` pulls it automatically, or load it with three lines of
  Ultralytics; an **ONNX** export is included.
- 🗺️ It is the detector behind **[Songdo Traffic](https://huggingface.co/datasets/rfonod/songdo-traffic)**,
  ~700,000 georeferenced vehicle trajectories, and the vehicle masks of the
  **[Stabilo Benchmark](https://huggingface.co/datasets/rfonod/stabilo-benchmark)**.
- 🚀 **No install needed to try it**: open the [live demo](https://huggingface.co/spaces/rfonod/geo-trax).

## Model Details

| Property | Value |
|---|---|
| Architecture | YOLOv8s (HBB, horizontal bounding boxes) |
| Input resolution | 1920 × 1920 px |
| Classes | 6 trained (4 primary + 2 auxiliary; see below) |
| Parameters | 11.1 M (11,137,922) |
| Framework | [Ultralytics](https://github.com/ultralytics/ultralytics) 8.x (Geo-trax itself requires ≥ 8.4.184) |
| Trained on | 19,339 annotated aerial images (679,306 labeled instances); multi-stage, see [publication](https://doi.org/10.1016/j.trc.2025.105205) |
| Validated on | [Songdo Vision **v1**](https://doi.org/10.5281/zenodo.13828408) test set (1,084 images, 55,124 vehicle instances) |

### Classes and Detection Performance

Metrics reported on the [Songdo Vision **v1**](https://doi.org/10.5281/zenodo.13828408) test split
(1,084 images, 55,124 labeled vehicle instances). The `Instances` column is the per-class
support in that test set. See Table 3 of the
[publication](https://doi.org/10.1016/j.trc.2025.105205) for full results.

| ID | Label | Notes | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---:|---|---|---|---|
| 0 | Car | incl. vans | 49,508 | 0.979 | 0.981 | 0.992 | 0.835 |
| 1 | Bus | | 1,759 | 0.952 | 0.977 | 0.988 | 0.826 |
| 2 | Truck | | 3,052 | 0.887 | 0.916 | 0.935 | 0.722 |
| 3 | Motorcycle | | 805 | 0.827 | 0.866 | 0.888 | 0.463 |
| 4 | Pedestrian | not evaluated | n/a | n/a | n/a | n/a | n/a |
| 5 | Bicycle | not evaluated | n/a | n/a | n/a | n/a | n/a |
| **All** | | | **55,124** | **0.911** | **0.935** | **0.951** | **0.711** |

The model reaches **0.951 mAP@50** and **0.711 mAP@50-95** overall, with near-saturated accuracy
on cars and buses (mAP@50 ≥ 0.988). Trucks and especially motorcycles are harder: motorcycles are
small, sparse in the test set (805 instances), and the main driver of the lower mAP@50-95.

> **Dataset version:** these metrics were measured against **Songdo Vision v1**. Songdo Vision
> [v2](https://doi.org/10.5281/zenodo.21728430) is an annotation-quality release that removed 196
> invalid boxes and added 1,951 vehicles the original pass had missed, so its test split holds 55,528
> instances rather than 55,124. **This model has not been retrained or re-evaluated on v2**, and the
> numbers above should not be compared directly against results reported on v2. Note also that the v2
> additions were surfaced with a Geo-trax detector before human confirmation, so evaluating this model
> family on the v2 test split carries a small favourable bias; filtering COCO `id <= 272435` restores
> the v1-provenance test set.

### Evaluation Plots

Precision-recall curves and the normalized confusion matrix on the Songdo Vision v1 test set:

<table>
  <tr>
    <td align="center"><img src="assets/PR_curve.png" alt="Precision-Recall Curve" width="420"><br><sub>Precision-Recall Curve</sub></td>
    <td align="center"><img src="assets/confusion_matrix_normalized.png" alt="Normalized Confusion Matrix" width="420"><br><sub>Normalized Confusion Matrix</sub></td>
  </tr>
</table>

> **Note on pedestrian and bicycle classes:** The model was trained on pedestrian and bicycle
> instances; however, these classes are **not evaluated and not recommended for use**. They were
> underrepresented in the training data, are not annotated in the Songdo Vision dataset (making
> reliable evaluation impossible), and achieve poor detection performance in practice.

## How to Use

> 🚀 **Try it first, no install:** the [interactive 🤗 Space](https://huggingface.co/spaces/rfonod/geo-trax) runs this detector in your browser on your own aerial image or short clip.

### With Geo-trax (recommended)

This model is the default in Geo-trax and downloads automatically on first use:

```bash
python -m pip install geo-trax
geotrax extract video.mp4         # detect, track, and stabilize; auto-downloads the model
geotrax batch video.mp4 --no-geo  # full pipeline in pixel coordinates; skip georeferencing
geotrax batch video.mp4           # full pipeline including georeferencing (requires orthophotos)
geotrax batch video.mp4 --sahi    # optional SAHI inference for better small-object detection
geotrax extract video.mp4 --set 'class_conf={1: 0.15, 2: 0.4}'  # per-class confidence thresholds
geotrax export video.mp4          # georeferenced trajectories to GeoPackage (or --format geojson) for QGIS/kepler.gl
```

The Geo-trax [documentation](https://rfonod.github.io/geo-trax/) has a guide per pipeline stage
(including [georeferencing](https://rfonod.github.io/geo-trax/guides/georeference/),
[SAHI](https://rfonod.github.io/geo-trax/guides/sahi/) and
[GIS export](https://rfonod.github.io/geo-trax/guides/export/)), plus a
[config reference](https://rfonod.github.io/geo-trax/reference/config/) and a
[CLI reference](https://rfonod.github.io/geo-trax/reference/cli/) generated from the code.

### Direct Ultralytics inference (PyTorch)

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

weights = hf_hub_download(repo_id="rfonod/geo-trax", filename="geotrax_hbb_yolov8s_1920_v1.pt")
model = YOLO(weights)

results = model("drone_frame.jpg", imgsz=1920, conf=0.25, iou=0.7, classes=[0, 1, 2, 3])
results[0].show()
```

> **Tip:** The model was trained and validated at **1920 px** input resolution. Downscaling to
> 1280 px is possible with a small accuracy trade-off; going below 960 px significantly degrades
> detection of small vehicles (motorcycles, distant cars). Pass `classes=[0, 1, 2, 3]` to
> restrict inference to the four evaluated classes and suppress unreliable predictions.

### ONNX inference

An ONNX export (opset 12, static 1920 × 1920 input) is available for deployment without a PyTorch dependency:

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

onnx_path = hf_hub_download(repo_id="rfonod/geo-trax", filename="geotrax_hbb_yolov8s_1920_v1.onnx")
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# Prepare input: RGB image resized/padded to 1920×1920, normalized to [0, 1]
img = np.random.rand(1, 3, 1920, 1920).astype(np.float32)  # replace with real image
outputs = session.run(None, {"images": img})
# outputs[0] shape: (1, 10, 75600); 10 = 4 bbox coords + 6 class scores
```

## Training Data

Training followed a multi-stage strategy starting from **YOLOv8s weights pretrained on COCO**
as the initial foundation. Two successive stages were applied:

**Stage 1 (BASE):** The model was trained on a large, diverse collection drawn from eight
public aerial and drone datasets (CARPK, PUCPR+, CyCAR, UAVDT, HARPY, RAI4VD, UIT-ADrone,
and VisDrone) combined with the [Songdo Vision **v1**](https://doi.org/10.5281/zenodo.13828408)
dataset, totalling 19,339 training images with 679,306 annotations across 6 classes
(car, bus, truck, motorcycle, pedestrian, bicycle).

**Stage 2 (FINE):** The BASE-trained model was subsequently fine-tuned on a curated,
high-quality subset of 9,004 images with 321,368 annotations, emphasising accurate annotations
and higher-resolution images, again combined with
[Songdo Vision **v1**](https://doi.org/10.5281/zenodo.13828408), to yield the final weights released here.

**Training set composition** (annotations per class):

| Stage | Images | Annotations | Car | Bus | Truck | Motorcycle | Pedestrian | Bicycle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE | 19,339 | 679,306 | 561,666 | 15,587 | 28,830 | 44,512 | 24,239 | 4,472 |
| FINE |  9,004 | 321,368 | 266,745 |  8,047 | 14,305 | 30,925 |  1,260 |    86 |

[Songdo Vision](https://doi.org/10.5281/zenodo.13828407) comprises 5,419 annotated drone frames
(4,335 training / 1,084 test; 80/20 split) collected during a large-scale urban traffic
monitoring experiment in Songdo, South Korea. It covers four primary vehicle classes captured
at 140-150 m altitude by DJI Mavic 3 drones. The counts above come from **v1**, which
contributed 217,311 training and 55,124 test instances; the current release is
[v2](https://doi.org/10.5281/zenodo.21728430) with 218,662 and 55,528 respectively.

**Training configuration:**

| Setting | Value |
|---|---|
| Initialization | YOLOv8s pretrained on COCO |
| Optimizer | SGD |
| Learning rate (initial / final factor) | 0.01 / 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Batch size | 8 |
| Early stopping | 50-epoch patience |
| Input resolution | 1920 × 1920 px (letterbox padding) |
| Mixed precision | AMP enabled |
| Augmentation | random scaling, translation, horizontal flip, mosaic, colour jitter, Gaussian/median blur, grayscale, CLAHE |

See the [publication](https://doi.org/10.1016/j.trc.2025.105205) for complete dataset statistics,
training details, and ablation results.

## Intended Use and Limitations

- **GSD assumption:** The bundled Geo-trax config assumes a ground sampling distance (GSD) of
  ~0.055 m/px (DJI Mavic 3, 4K, 140-150 m altitude). Adjust
  [`extraction.dimension_estimation.gsd`](https://rfonod.github.io/geo-trax/reference/config/extraction/#extraction.dimension_estimation.gsd)
  for different hardware or flight altitudes.
- **Supported classes:** Car, bus, truck, and motorcycle (class IDs 0-3). The model was also
  trained on pedestrian and bicycle instances; however, these classes achieve poor detection
  performance and are not recommended for use (see the class table above). Geo-trax filters to
  the four primary classes by default; when using Ultralytics directly, pass
  `classes=[0, 1, 2, 3]` to suppress unreliable predictions.

## The Geo-trax and Songdo family

All six Hub repos come from one multi-drone traffic experiment over Songdo, South Korea, and
are gathered in the 🤗 collection [Trajectory Extraction and Stabilization](https://huggingface.co/collections/rfonod/trajectory-extraction-and-stabilization-68aa0f4c5372ae5a8edf22ce).

| | Repo | What it is |
|---|---|---|
| 🧠 Model | **`rfonod/geo-trax`** (this page) | YOLOv8s vehicle detector for high-altitude drone imagery (0.951 mAP@50) |
| 🚀 Space | [`rfonod/geo-trax`](https://huggingface.co/spaces/rfonod/geo-trax) | Run the detector in your browser on your own image or clip |
| 🛣️ Dataset | [`rfonod/songdo-traffic`](https://huggingface.co/datasets/rfonod/songdo-traffic) | ~700,000 georeferenced vehicle trajectories from 10 drones over 20 intersections |
| 🖼️ Dataset | [`rfonod/songdo-vision`](https://huggingface.co/datasets/rfonod/songdo-vision) | 5,419 4K frames with 274,190 human-verified vehicle boxes |
| 🔄 Dataset | [`rfonod/songdo-vision-obb`](https://huggingface.co/datasets/rfonod/songdo-vision-obb) | The same frames with oriented (rotated) vehicle boxes |
| 🎯 Dataset | [`rfonod/stabilo-benchmark`](https://huggingface.co/datasets/rfonod/stabilo-benchmark) | Benchmark data for reference-based UAV video stabilization |

## Related datasets and resources

- **Live demo**: interactive 🤗 Space: [`rfonod/geo-trax` (Spaces)](https://huggingface.co/spaces/rfonod/geo-trax)
- **Songdo Traffic**: the georeferenced vehicle-trajectory dataset this model helps produce via
  the Geo-trax pipeline:
  [`10.5281/zenodo.13828383`](https://doi.org/10.5281/zenodo.13828383) ·
  HF [`rfonod/songdo-traffic`](https://huggingface.co/datasets/rfonod/songdo-traffic)
- **Songdo Vision**: the vehicle-detection (annotated image) dataset used to train and validate
  this model: [`10.5281/zenodo.13828407`](https://doi.org/10.5281/zenodo.13828407) ·
  HF [`rfonod/songdo-vision`](https://huggingface.co/datasets/rfonod/songdo-vision)
- **Songdo Vision OBB**: the same frames as Songdo Vision v2 with oriented bounding boxes,
  introduced with the stabilization benchmark below:
  [`10.5281/zenodo.15050578`](https://doi.org/10.5281/zenodo.15050578) ·
  HF [`rfonod/songdo-vision-obb`](https://huggingface.co/datasets/rfonod/songdo-vision-obb)
- **Stabilization benchmark**: [Stabilo Benchmark](https://github.com/rfonod/stabilo-benchmark),
  which evaluates the registration step behind Geo-trax's stabilization on dynamic urban scenes and
  annotates its vehicles with this model, via HBB2OBB:
  [arXiv:2611.17190](https://arxiv.org/abs/2611.17190) · data record
  [`10.5281/zenodo.23623540`](https://doi.org/10.5281/zenodo.23623540) ·
  HF [`rfonod/stabilo-benchmark`](https://huggingface.co/datasets/rfonod/stabilo-benchmark)
- **Source video recordings** (not open access):
  [`10.5075/EPFL.20.500.14299/253923`](https://doi.org/10.5075/EPFL.20.500.14299/253923)
- **Publication**: *Transportation Research Part C* (2025):
  [`10.1016/j.trc.2025.105205`](https://doi.org/10.1016/j.trc.2025.105205) ·
  [arXiv:2411.02136](https://arxiv.org/abs/2411.02136)
- **Software**: Geo-trax: [github.com/rfonod/geo-trax](https://github.com/rfonod/geo-trax) ·
  [documentation](https://rfonod.github.io/geo-trax/) ·
  Zenodo [`10.5281/zenodo.12119542`](https://doi.org/10.5281/zenodo.12119542) ·
  [demo video](https://youtu.be/gOGivL9FFLk)

## Citation

If you use this model, please cite the associated publication:

```bibtex
@article{fonod2025advanced,
  title   = {Advanced computer vision for extracting georeferenced vehicle trajectories from drone imagery},
  author  = {Fonod, Robert and Cho, Haechan and Yeo, Hwasoo and Geroliminis, Nikolas},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume  = {178},
  pages   = {105205},
  year    = {2025},
  doi     = {10.1016/j.trc.2025.105205}
}
```

If you additionally use the [Geo-trax software](https://github.com/rfonod/geo-trax), please
also cite its Zenodo archive, replacing the version and year with the release you used (the
example shows the latest at the time of writing). The DOI below is the
*concept* DOI, which always resolves to the latest release; to cite one release specifically,
use that version's own DOI from the [Zenodo record](https://doi.org/10.5281/zenodo.12119542).

```bibtex
@software{fonod_geotrax,
  author  = {Fonod, Robert},
  title   = {Geo-trax: A Comprehensive Framework for Georeferenced Vehicle Trajectory Extraction from Drone Imagery},
  url     = {https://github.com/rfonod/geo-trax},
  doi     = {10.5281/zenodo.12119542},
  version = {1.6.2},
  year    = {2026}
}
```

## License

This model is released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license; see the [LICENSE](LICENSE) file for the full terms. The Geo-trax codebase is distributed
separately under the [MIT License](https://github.com/rfonod/geo-trax/blob/main/LICENSE).
