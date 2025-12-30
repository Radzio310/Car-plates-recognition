from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

from anpr.pipeline import ANPRPipeline
from anpr.utils import iou_xyxy

def read_gt_csv(path: Path) -> Dict[str, str]:
    # Oczekiwane kolumny: filename, plate_text (nazwy mogą się różnić; obsługujemy kilka wariantów)
    data: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("filename") or row.get("file") or row.get("image") or row.get("img") or ""
            txt = row.get("plate_text") or row.get("plate") or row.get("text") or row.get("label") or ""
            if fn:
                data[Path(fn).name] = txt.strip()
    return data

def read_yolo_label(txt_path: Path, img_w: int, img_h: int):
    # tylko pierwsza tablica (w praktyce często 1 na obraz)
    if not txt_path.exists():
        return None
    lines = [l.strip() for l in txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return None
    parts = lines[0].split()
    if len(parts) != 5:
        return None
    _, xc, yc, w, h = parts
    xc, yc, w, h = float(xc), float(yc), float(w), float(h)
    x1 = int((xc - w/2.0) * img_w)
    y1 = int((yc - h/2.0) * img_h)
    x2 = int((xc + w/2.0) * img_w)
    y2 = int((yc + h/2.0) * img_h)
    return (x1, y1, x2, y2)

def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Katalog z obrazami test (np. data/yolo/images/test)")
    ap.add_argument("--labels", required=True, help="Katalog z labelami YOLO dla test (np. data/yolo/labels/test)")
    ap.add_argument("--gt-csv", required=True, help="CSV z ground-truth tekstem tablicy dla obrazów")
    ap.add_argument("--config", default="configs/app_config.yaml")
    ap.add_argument("--weights", default="", help="Opcjonalnie: podmień weights w runtime (nadpisze config)")
    ap.add_argument("--limit", type=int, default=100, help="Ile obrazów ocenić (wymagane: 100 do czasu)")
    args = ap.parse_args()

    pipeline = ANPRPipeline(config_path=args.config)
    if args.weights:
        # dynamiczna podmiana wag YOLO
        pipeline.detector.model = pipeline.detector.model.__class__(args.weights)

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    gt = read_gt_csv(Path(args.gt_csv))

    imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not imgs:
        raise SystemExit("Brak obrazów w --images")

    imgs = imgs[: max(1, args.limit)]
    start = time.perf_counter()

    correct = 0
    total = 0
    ious: List[float] = []
    failures = 0

    for img_path in tqdm(imgs, desc="Evaluating"):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt_text = gt.get(img_path.name, "")
        out = pipeline.run(im)

        # IoU tylko jeśli mamy bbox gt
        gt_box = read_yolo_label(labels_dir / f"{img_path.stem}.txt", w, h)
        if gt_box and out.bbox:
            ious.append(iou_xyxy(gt_box, out.bbox))

        if not gt_text:
            # brak GT tekstu => nie liczymy accuracy na tym przykładzie
            continue

        total += 1
        if out.plate_text_norm and out.plate_text_norm == out.plate_text_norm.__class__(out.plate_text_norm)(out.plate_text_norm):
            pass
        # porównanie po normalizacji: tu używamy pipeline.norm jako wynik
        # oraz oczekujemy, że GT może zawierać spacje
        from anpr.utils import normalize_plate
        gt_norm = normalize_plate(gt_text, allowed_chars=pipeline.allowed_chars)
        if out.plate_text_norm == gt_norm:
            correct += 1
        else:
            failures += 1

    elapsed = time.perf_counter() - start

    acc = (correct / total * 100.0) if total else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    grade = calculate_final_grade(acc, elapsed)

    print("==== RESULTS ====")
    print(f"OCR accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"Mean IoU (detection): {mean_iou:.4f} (n={len(ious)})")
    print(f"Processing time for {len(imgs)} images: {elapsed:.2f} s")
    if len(imgs) != 100:
        print("NOTE: Wymaganie czasu dotyczy 100 zdjęć – ustaw --limit 100, o ile masz dostępne 100 testowych obrazów.")
    print(f"Final grade (PDF formula): {grade:.1f}")

if __name__ == "__main__":
    main()
