from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

BBox = Tuple[int, int, int, int]  # x1,y1,x2,y2

@dataclass
class Detection:
    bbox: BBox
    conf: float
    cls: int = 0

def clamp_bbox(bbox: BBox, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def iou_xyxy(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0

def normalize_plate(text: str, allowed_chars: str, uppercase: bool=True, strip_spaces: bool=True) -> str:
    if text is None:
        return ""
    s = text
    if strip_spaces:
        s = re.sub(r"\s+", "", s)
    if uppercase:
        s = s.upper()
    s = "".join(ch for ch in s if ch in allowed_chars)
    return s

def validate_plate(text: str, plate_regex: str) -> bool:
    if not text:
        return False
    return re.match(plate_regex, text) is not None
