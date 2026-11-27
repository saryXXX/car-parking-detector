"""
YOLOv8 + ByteTrack Vehicle Tracking Pipeline
Extracts persistent vehicle track IDs, bounding boxes, and ground contact points.

When given a YOLO segmentation checkpoint (e.g. yolov8n-seg.pt), also extracts each
vehicle's pixel mask polygon -- this lets SpatialEngine compute true footprint overlap
against a slot polygon instead of approximating from the bounding box, which matters
most on oblique/angled cameras where a car's bbox includes empty background corners
that don't belong to the vehicle at all.
"""

from typing import Dict, List, Optional
import cv2
import numpy as np


def mask_polygons(masks) -> List[np.ndarray]:
    """Same output as Ultralytics' `masks.xy`, without its two big costs.

    `masks.xy` copies the whole (N, H, W) mask stack to the CPU (~66 MB for 70 cars at
    imgsz 1280) and traces contours on N full-size images. Here each mask's extent is found
    on the device, only that crop (+1 px of zeros) is copied, and it is traced with an offset.
    Contour tracing is translation-invariant, so the points are identical.
    """
    from ultralytics.data.converter import merge_multi_segment
    from ultralytics.utils import ops

    data = masks.data
    if data.shape[0] == 0:
        return []
    H, W = data.shape[1:]
    nz = data > 0
    rows = nz.any(dim=2).cpu().numpy()  # (N, H): small transfers
    cols = nz.any(dim=1).cpu().numpy()  # (N, W)
    import torch

    boxes = []
    for i in range(data.shape[0]):
        ys, xs = np.flatnonzero(rows[i]), np.flatnonzero(cols[i])
        if not len(ys):
            boxes.append(None)
            continue
        boxes.append((max(0, ys[0] - 1), min(H, ys[-1] + 2), max(0, xs[0] - 1), min(W, xs[-1] + 2)))
    # All crops gathered on the device and copied in ONE transfer (one small copy per mask
    # forces a device sync each time, ~70 per frame).
    pieces = [nz[i, y0:y1, x0:x1].reshape(-1) for i, b in enumerate(boxes) if b for y0, y1, x0, x1 in [b]]
    flat = torch.cat(pieces).byte().cpu().numpy() if pieces else np.zeros(0, np.uint8)

    out, pos = [], 0
    for b in boxes:
        if b is None:
            out.append(np.zeros((0, 2), np.float32))
            continue
        y0, y1, x0, x1 = b
        n = (y1 - y0) * (x1 - x0)
        crop = flat[pos:pos + n].reshape(y1 - y0, x1 - x0)
        pos += n
        c = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(int(x0), int(y0)))[0]
        if not c:
            seg = np.zeros((0, 2))
        elif len(c) > 1:  # masks2segments(strategy="all"): merge every contour
            seg = np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in c]))
        else:
            seg = c[0].reshape(-1, 2)
        out.append(ops.scale_coords((H, W), seg.astype("float32"), masks.orig_shape, normalize=False))
    return out


class VehicleTracker:
    # COCO dataset default fallback class indices for road vehicles
    DEFAULT_VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_type: str = "bytetrack.yaml",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        aerial: bool = False,
        fps: float = 30.0,
        imgsz: int = 640,
    ):
        self.model_name = model_name
        # Inference size. 640 on a 1280 frame halves every car; far rows of a wide oblique
        # view (30-40 px cars) then drop below what the detector can see.
        self.imgsz = imgsz
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.active_vehicle_classes = {}
        # Aerial mode: tiled detection (src/aerial_detector.py) + standalone ByteTrack.
        # Needed for top-down drone footage, where full-frame YOLO finds almost no cars.
        self.aerial = aerial
        self.fps = fps
        self._aerial_detector = None
        self._byte_tracker = None

    def _load_aerial(self):
        if self._aerial_detector is None:
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import IterableSimpleNamespace, YAML
            from ultralytics.utils.checks import check_yaml
            from src.aerial_detector import AerialVehicleDetector

            print(f"[INFO] Mode aerien (tuiles) : {self.model_name}")
            self._aerial_detector = AerialVehicleDetector(self.model_name)
            cfg = IterableSimpleNamespace(**YAML.load(check_yaml(self.tracker_type)))
            cfg.track_buffer = max(cfg.track_buffer, int(self.fps * 2))  # frames: keep IDs through ~2 s of misses
            # Tiled COCO detections of top-down cars score low (0.05-0.4); default 0.25 gates
            # would never start a track for most of them.
            if not self._aerial_detector.aerial_trained:
                cfg.track_high_thresh, cfg.track_low_thresh, cfg.new_track_thresh = 0.1, 0.05, 0.1
            cfg.fuse_score = False  # IoU*score would sink low-score parked cars below the match gate
            self._byte_tracker = BYTETracker(cfg)

    def _track_aerial(self, frame: np.ndarray) -> List[Dict]:
        from ultralytics.engine.results import Boxes

        self._load_aerial()
        boxes, scores = self._aerial_detector(frame)
        data = np.zeros((len(boxes), 6), np.float32)
        if len(boxes):
            data[:, :4], data[:, 4] = boxes, scores
        tracks = self._byte_tracker.update(Boxes(data, frame.shape[:2]), frame)
        return [
            {"id": int(t[4]), "bbox": tuple(map(float, t[:4])), "conf": float(t[5]),
             "cls_name": "car", "mask": None}
            for t in tracks
        ]

    def load_model(self):
        """Lazy load of the YOLO model and dynamic vehicle class detection."""
        if self.model is None:
            from src.model_cache import get_yolo
            print(f"[INFO] Chargement du modele YOLO : {self.model_name}...")
            # Cached per process: weights are loaded (and the GPU warmed up) once, not per run.
            self.model = get_yolo(self.model_name, "track")
            
            # Résolution dynamique intelligente des classes selon le modèle chargé.
            # Vehicle classes only -- "occupied_slot"/"occupied" are slot-state labels from
            # specialized classifiers (see src/slot_classifier.py), not individual vehicles.
            # Treating them as trackable vehicles produced giant per-row boxes that spanned
            # multiple slot polygons at once, which looked like (but wasn't) straddle-parking.
            target_keywords = {"car", "motorcycle", "bus", "truck", "van", "vehicle", "automobile"}
            self.active_vehicle_classes = {}
            for cid, cname in self.model.names.items():
                name_clean = str(cname).lower().strip()
                tokens = name_clean.replace("-", " ").replace("_", " ").split()
                if any(kw in tokens for kw in target_keywords) or name_clean in target_keywords:
                    self.active_vehicle_classes[int(cid)] = str(cname)
            
            if not self.active_vehicle_classes:
                # Modèle custom : on suit toutes les classes disponibles
                self.active_vehicle_classes = {int(k): str(v) for k, v in self.model.names.items()}
                
            print(f"[INFO] Modele YOLO charge. Classes actives suivies : {self.active_vehicle_classes}")

    def track(self, frame: np.ndarray) -> List[Dict]:
        """
        Runs YOLO + ByteTrack on a single video frame.
        
        Returns:
            List of detected vehicles:
            [
                {
                    "id": int,
                    "bbox": (x1, y1, x2, y2),
                    "conf": float,
                    "cls_name": str
                },
                ...
            ]
        """
        if self.aerial:
            return self._track_aerial(frame)
        self.load_model()

        # Classes ciblées
        classes_filter = list(self.active_vehicle_classes.keys()) if self.active_vehicle_classes else None

        # Run tracking inference
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_type,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=classes_filter,
            imgsz=self.imgsz,
            verbose=False
        )

        tracked_vehicles = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            masks = getattr(results[0], "masks", None)
            mask_polys = mask_polygons(masks) if masks is not None else None  # one (N,2) array per detection, original-frame coords

            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                xyxy_coords = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                cls_indices = boxes.cls.int().cpu().tolist()

                for i, (tid, bbox, conf, cls_idx) in enumerate(zip(track_ids, xyxy_coords, confs, cls_indices)):
                    cls_name = self.active_vehicle_classes.get(cls_idx, "vehicle")
                    entry = {
                        "id": int(tid),
                        "bbox": tuple(map(float, bbox)),
                        "conf": float(conf),
                        "cls_name": cls_name,
                        "mask": None,
                    }
                    if mask_polys is not None and i < len(mask_polys) and len(mask_polys[i]) >= 3:
                        entry["mask"] = mask_polys[i].tolist()
                    tracked_vehicles.append(entry)

        return tracked_vehicles

