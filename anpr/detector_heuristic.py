from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np

from .utils import Detection, clamp_bbox

@dataclass
class HeuristicDetectorConfig:
    min_area_ratio: float = 0.002  # względem całego obrazu
    max_area_ratio: float = 0.20
    aspect_min: float = 2.0
    aspect_max: float = 6.5

class HeuristicPlateDetector:
    """Detektor 'na start' – działa na prostych scenach, demo, synthetic.
    Nie zastępuje modelu uczonego, ale pozwala uruchomić aplikację bez wag YOLO.
    """
    def __init__(self, cfg: HeuristicDetectorConfig = HeuristicDetectorConfig()):
        self.cfg = cfg

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        h, w = image_bgr.shape[:2]
        img = image_bgr.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets: List[Detection] = []
        area_img = float(w * h)

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            x, y, ww, hh = cv2.boundingRect(approx)
            area = ww * hh
            area_ratio = area / area_img
            if area_ratio < self.cfg.min_area_ratio or area_ratio > self.cfg.max_area_ratio:
                continue
            aspect = ww / float(hh + 1e-6)
            if aspect < self.cfg.aspect_min or aspect > self.cfg.aspect_max:
                continue

            # preferuj jasne prostokąty (tablica)
            crop = gray[y:y+hh, x:x+ww]
            if crop.size == 0:
                continue
            mean = float(np.mean(crop))
            # confidence heurystyczne: jasność + area
            conf = min(1.0, 0.5 * (mean / 255.0) + 0.5 * (area_ratio / self.cfg.max_area_ratio))
            bbox = clamp_bbox((x, y, x+ww, y+hh), w=w, h=h)
            dets.append(Detection(bbox=bbox, conf=conf, cls=0))

        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets
