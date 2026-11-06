"""
YOLOv8 + ByteTrack Vehicle Tracking Pipeline
Extracts persistent vehicle track IDs, bounding boxes, and ground contact points.
"""

from typing import Dict, List, Optional
import numpy as np


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
        iou_threshold: float = 0.5
    ):
        self.model_name = model_name
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.active_vehicle_classes = {}

    def load_model(self):
        """Lazy load of the YOLO model and dynamic vehicle class detection."""
        if self.model is None:
            from ultralytics import YOLO
            print(f"[INFO] Chargement du modele YOLO : {self.model_name}...")
            self.model = YOLO(self.model_name)
            
            # Résolution dynamique intelligente des classes selon le modèle chargé
            # Inclut les véhicules et les places occupées (pour les modèles spécialisés PKLot / Drone)
            target_keywords = {"car", "motorcycle", "bus", "truck", "van", "vehicle", "automobile", "occupied_slot", "occupied"}
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
            verbose=False
        )

        tracked_vehicles = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                xyxy_coords = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                cls_indices = boxes.cls.int().cpu().tolist()

                for tid, bbox, conf, cls_idx in zip(track_ids, xyxy_coords, confs, cls_indices):
                    cls_name = self.active_vehicle_classes.get(cls_idx, "vehicle")
                    tracked_vehicles.append({
                        "id": int(tid),
                        "bbox": tuple(map(float, bbox)),
                        "conf": float(conf),
                        "cls_name": cls_name
                    })

        return tracked_vehicles

