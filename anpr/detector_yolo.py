from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from ultralytics import YOLO

from .utils import Detection, clamp_bbox

@dataclass
class YoloDetectorConfig:
    weights: str
    conf: float = 0.25
    iou: float = 0.45
    img_size: int = 640

class YoloPlateDetector:
    def __init__(self, cfg: YoloDetectorConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        h, w = image_bgr.shape[:2]
        # Ultralytics działa na RGB
        image_rgb = image_bgr[:, :, ::-1]
        results = self.model.predict(
            source=image_rgb,
            imgsz=self.cfg.img_size,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            verbose=False
        )
        dets: List[Detection] = []
        if not results:
            return dets
        r0 = results[0]
        if r0.boxes is None:
            return dets
        boxes = r0.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls).astype(int)

        for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
            bb = clamp_bbox((int(x1), int(y1), int(x2), int(y2)), w=w, h=h)
            dets.append(Detection(bbox=bb, conf=float(c), cls=int(cls)))
        # Najpewniejsza tablica jako pierwsza
        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets
