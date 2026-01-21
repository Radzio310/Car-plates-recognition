from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import time

import cv2
import numpy as np
import yaml

from .detector_yolo import YoloPlateDetector, YoloDetectorConfig
from .detector_heuristic import HeuristicPlateDetector, HeuristicDetectorConfig
from .ocr import make_ocr_engine
from .db import PlateDB
from .utils import normalize_plate, validate_plate, BBox


@dataclass
class PipelineOutput:
    plate_text_raw: str
    plate_text_norm: str
    plate_valid_format: bool
    ocr_conf: float
    detected: bool
    bbox: Optional[BBox]
    access_granted: Optional[bool]
    error: Optional[str]
    timing_ms: Dict[str, float]


class ANPRPipeline:
    def __init__(self, config_path: str = "configs/app_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        det_cfg = cfg.get("detector", {}) or {}
        det_type = (det_cfg.get("type") or "heuristic").lower().strip()

        if det_type == "yolo":
            self.detector = YoloPlateDetector(
                YoloDetectorConfig(
                    weights=det_cfg["weights"],
                    conf=float(det_cfg.get("conf", 0.25)),
                    iou=float(det_cfg.get("iou", 0.45)),
                    img_size=int(det_cfg.get("img_size", 640)),
                )
            )
        elif det_type == "heuristic":
            self.detector = HeuristicPlateDetector(
                HeuristicDetectorConfig(
                    min_area_ratio=float(det_cfg.get("min_area_ratio", 0.002)),
                    max_area_ratio=float(det_cfg.get("max_area_ratio", 0.20)),
                    aspect_min=float(det_cfg.get("aspect_min", 2.0)),
                    aspect_max=float(det_cfg.get("aspect_max", 6.5)),
                )
            )
        else:
            raise ValueError(f"Unsupported detector.type: {det_type}. Use yolo or heuristic.")

        ocr_cfg = cfg.get("ocr", {}) or {}

        fast_cfg = ocr_cfg.get("fast", {})
        if not isinstance(fast_cfg, dict):
            fast_cfg = {}

        fast_try_psm8 = bool(fast_cfg.get("try_psm8", True))
        fast_cut_blue_auto = bool(fast_cfg.get("cut_blue_auto", False))
        fast_resize_fx = float(fast_cfg.get("resize_fx", 2.1))
        fast_oem = int(fast_cfg.get("oem", 3))

        self.ocr = make_ocr_engine(
            engine=ocr_cfg.get("engine", "easyocr"),
            languages=ocr_cfg.get("languages", ["en"]),
            tesseract_lang=ocr_cfg.get("tesseract_lang", "eng"),
            fast_try_psm8=fast_try_psm8,
            fast_cut_blue_auto=fast_cut_blue_auto,
            fast_resize_fx=fast_resize_fx,
            fast_oem=fast_oem,
        )

        pp = cfg.get("postprocess", {}) or {}
        self.allowed_chars = pp.get("allowed_chars", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        self.uppercase = bool(pp.get("uppercase", True))
        self.strip_spaces = bool(pp.get("strip_spaces", True))
        self.plate_regex = pp.get("plate_regex", "^[A-Z]{1,3}[A-Z0-9]{4,5}$")

        db_path = (cfg.get("access_control", {}) or {}).get("sqlite_path", "data/plates.db")
        self.db = PlateDB(path=db_path)

        # KLUCZ: domyślnie tight-crop jak w demo
        crop_cfg = cfg.get("crop", {}) or {}
        self.pad_x_ratio = float(crop_cfg.get("pad_x_ratio", 0.0))
        self.pad_y_ratio = float(crop_cfg.get("pad_y_ratio", 0.0))

    def run(self, image_bgr: np.ndarray) -> PipelineOutput:
        t0 = time.perf_counter()
        dets = self.detector.detect(image_bgr)
        t1 = time.perf_counter()

        if not dets:
            return PipelineOutput(
                plate_text_raw="",
                plate_text_norm="",
                plate_valid_format=False,
                ocr_conf=0.0,
                detected=False,
                bbox=None,
                access_granted=None,
                error="Nie wykryto tablicy rejestracyjnej na obrazie.",
                timing_ms={"detect": (t1 - t0) * 1000.0, "ocr": 0.0, "db": 0.0, "total": (t1 - t0) * 1000.0},
            )

        best = dets[0]
        x1, y1, x2, y2 = best.bbox

        H, W = image_bgr.shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        padx = int(max(0.0, self.pad_x_ratio) * bw)
        pady = int(max(0.0, self.pad_y_ratio) * bh)

        x1p = max(0, x1 - padx)
        y1p = max(0, y1 - pady)
        x2p = min(W, x2 + padx)
        y2p = min(H, y2 + pady)

        # UWAGA: slice jest [y1:y2), więc x2p/y2p mogą być równe W/H
        if x2p <= x1p or y2p <= y1p:
            crop = image_bgr[y1:y2, x1:x2].copy()
        else:
            crop = image_bgr[y1p:y2p, x1p:x2p].copy()

        t2 = time.perf_counter()
        ocr_res = self.ocr.read(crop)
        t3 = time.perf_counter()

        norm = normalize_plate(
            ocr_res.text,
            allowed_chars=self.allowed_chars,
            uppercase=self.uppercase,
            strip_spaces=self.strip_spaces,
        )
        is_valid = validate_plate(norm, self.plate_regex)

        t4 = time.perf_counter()
        access: Optional[bool] = None
        if norm and is_valid:
            access = bool(self.db.exists(norm))
        t5 = time.perf_counter()

        err = None
        if not norm:
            err = "OCR nie zwrócił tekstu (spróbuj innego OCR lub popraw pre-processing)."
        elif not is_valid:
            err = "OCR zwrócił tekst, ale nie pasuje do formatu (regex) – nie sprawdzono w bazie."

        return PipelineOutput(
            plate_text_raw=ocr_res.text,
            plate_text_norm=norm,
            plate_valid_format=is_valid,
            ocr_conf=float(ocr_res.confidence),
            detected=True,
            bbox=best.bbox,
            access_granted=access,
            error=err,
            timing_ms={
                "detect": (t1 - t0) * 1000.0,
                "ocr": (t3 - t2) * 1000.0,
                "db": (t5 - t4) * 1000.0,
                "total": (t5 - t0) * 1000.0,
            },
        )

    @staticmethod
    def draw_bbox(image_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
        out = image_bgr.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return out
