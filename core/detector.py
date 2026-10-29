"""
VisionSpace - YOLOv11 Real-Time Object Detection Module
Detects objects from camera frames and extracts 2D bounding boxes, classes, and confidences.
"""

import cv2
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO
from config.settings import (
    YOLO_MODEL_NAME, DETECTION_CONFIDENCE,
    DETECTION_IOU_THRESHOLD, INTEREST_CLASSES
)


class ObjectDetector:
    """Wrapper around YOLOv11 for real-time spatial object detection."""

    def __init__(self, model_name: str = YOLO_MODEL_NAME, conf_thresh: float = DETECTION_CONFIDENCE):
        self.conf_thresh = conf_thresh
        print(f"[VisionSpace] Initializing YOLOv11 Object Detector ({model_name})...")
        self.model = YOLO(model_name)
        # Warmup model with dummy tensor to avoid initial inference latency
        self._warmup()

    def _warmup(self):
        """Warm up the detector to preload weights into memory."""
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False, conf=self.conf_thresh)
            print("[VisionSpace] YOLOv11 Detector loaded and warmed up.")
        except Exception as e:
            print(f"[VisionSpace] Warning during YOLO warmup: {e}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a BGR image frame.

        Returns:
            List of detection dicts containing:
            - 'bbox': [x1, y1, x2, y2]
            - 'center': (cx, cy)
            - 'width': w
            - 'height': h
            - 'class_id': int
            - 'class_name': str
            - 'confidence': float
        """
        if frame is None:
            return []

        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=DETECTION_IOU_THRESHOLD,
            verbose=False
        )

        detections = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = self.model.names.get(cls_id, f"obj_{cls_id}")

            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            cx = int(x1 + w / 2.0)
            cy = int(y1 + h / 2.0)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "width": w,
                "height": h,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf
            })

        return detections
