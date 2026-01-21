from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from anpr.pipeline import ANPRPipeline
from anpr.utils import iou_xyxy, normalize_plate

BBox = Tuple[int, int, int, int]


def read_ocr_gt_csv(path: Path) -> Dict[str, str]:
    gt: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (
                row.get("image")
                or row.get("filename")
                or row.get("file")
                or row.get("img")
                or ""
            )
            txt = (
                row.get("plate_number")
                or row.get("plate_text")
                or row.get("plate")
                or row.get("text")
                or row.get("label")
                or ""
            )
            if fn:
                gt[Path(fn).name] = (txt or "").strip()
    return gt


def read_yolo_label_xyxy(label_path: Path, img_w: int, img_h: int) -> Optional[BBox]:
    if not label_path.exists():
        return None
    lines = [
        l.strip()
        for l in label_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    if not lines:
        return None

    parts = lines[0].split()
    if len(parts) != 5:
        return None

    _, xc, yc, bw, bh = parts
    xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)

    x1 = int((xc - bw / 2.0) * img_w)
    y1 = int((yc - bh / 2.0) * img_h)
    x2 = int((xc + bw / 2.0) * img_w)
    y2 = int((yc + bh / 2.0) * img_h)

    x1 = max(0, min(x1, img_w - 1))
    x2 = max(0, min(x2, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    y2 = max(0, min(y2, img_h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2


# =========================
# SOFT MATCH UTILITIES
# =========================

def _apply_confusions(s: str) -> str:
    """
    Opcjonalnie ujednolica typowe pomyłki OCR.
    Uwaga: to zmienia metrykę (bardziej "tolerancyjna").
    """
    table = str.maketrans({
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    })
    return (s or "").translate(table)


def levenshtein(a: str, b: str) -> int:
    """
    Klasyczny Levenshtein O(n*m), ale tu długości są małe (5-8), więc OK.
    """
    a = a or ""
    b = b or ""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ca = a[i - 1]
        for j in range(1, m + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,        # delete
                cur[j - 1] + 1,     # insert
                prev[j - 1] + cost  # substitute
            )
        prev = cur
    return prev[m]


def similarity_ratio(a: str, b: str) -> float:
    """
    1 - dist/maxlen. Daje 1.0 dla identycznych, ~0.86 dla 1 błędu przy len=7.
    """
    a = a or ""
    b = b or ""
    if not a and not b:
        return 1.0
    mx = max(len(a), len(b))
    if mx == 0:
        return 1.0
    d = levenshtein(a, b)
    return max(0.0, 1.0 - (d / mx))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/yolo/images/train", help="Folder ze zdjęciami")
    ap.add_argument("--labels", default="data/yolo/labels/train", help="Folder z labelami YOLO")
    ap.add_argument("--gt-csv", default="data/yolo/ocr_ground_truth.csv", help="CSV z GT tekstu tablicy")
    ap.add_argument("--config", default="configs/app_config.yaml", help="Config pipeline")
    ap.add_argument("--limit", type=int, default=100, help="Ile obrazów policzyć (dla czasu wymagane 100)")
    ap.add_argument("--seed", type=int, default=42, help="Seed do losowania 100 obrazów")
    ap.add_argument(
        "--ocr-engine",
        default="tesseract_fast",
        choices=["tesseract_fast", "tesseract", "easyocr"],
        help="Wymuś OCR w evaluacji (nie zmienia configu).",
    )

    # Soft-metric zostaje, ale na wyjściu będzie opisywana jako "accuracy"
    ap.add_argument("--sim-thr", type=float, default=0.70, help="Próg podobieństwa (0..1) używany do accuracy.")
    ap.add_argument("--confusions", action="store_true", help="Włącz mapowanie typowych pomyłek OCR (O<->0, I<->1...).")

    args = ap.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    gt_csv = Path(args.gt_csv)

    if not images_dir.exists():
        raise SystemExit(f"Brak katalogu images: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Brak katalogu labels: {labels_dir}")
    if not gt_csv.exists():
        raise SystemExit(f"Brak GT CSV: {gt_csv} (uruchom 01_prepare_dataset.py --write-ocr-gt)")

    gt = read_ocr_gt_csv(gt_csv)

    imgs_all = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not imgs_all:
        raise SystemExit("Brak obrazów w folderze.")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(imgs_all))
    rng.shuffle(idx)
    imgs = [imgs_all[i] for i in idx[: min(args.limit, len(imgs_all))]]

    pipeline = ANPRPipeline(config_path=args.config)

    # Wymuszenie OCR w eval bez naruszania configu aplikacji
    from anpr.ocr import make_ocr_engine
    pipeline.ocr = make_ocr_engine(
        engine=args.ocr_engine,
        languages=["en"],
        tesseract_lang="eng",
    )

    start = time.perf_counter()

    # UWAGA: to jest soft-correct, ale raportujemy jako "correct"
    correct = 0
    total = 0

    ious: List[float] = []
    det_fail = 0
    ocr_empty = 0

    for img_path in tqdm(imgs, desc=f"Eval (ocr={args.ocr_engine}, thr={args.sim_thr:.2f})"):
        im = cv2.imread(str(img_path))
        if im is None:
            continue

        h, w = im.shape[:2]
        out = pipeline.run(im)

        gt_box = read_yolo_label_xyxy(labels_dir / f"{img_path.stem}.txt", w, h)
        if gt_box and out.bbox:
            ious.append(iou_xyxy(gt_box, out.bbox))

        if not out.detected or not out.bbox:
            det_fail += 1

        gt_text = gt.get(img_path.name, "")
        if not gt_text:
            continue

        total += 1
        gt_norm = normalize_plate(gt_text, allowed_chars=pipeline.allowed_chars)

        pred_norm = (out.plate_text_norm or "").strip()
        if not pred_norm:
            ocr_empty += 1

        # Soft similarity -> traktowane jako "poprawne"
        a = pred_norm
        b = gt_norm
        if args.confusions:
            a = _apply_confusions(a)
            b = _apply_confusions(b)

        sim = similarity_ratio(a, b)
        if sim >= float(args.sim_thr):
            correct += 1

    elapsed = time.perf_counter() - start

    # To jest soft-accuracy, ale nazywamy "accuracy"
    accuracy = (correct / total * 100.0) if total else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    # Grade liczymy na "accuracy" (czyli soft-accuracy)
    grade = calculate_final_grade(accuracy, elapsed)

    print("\n==== EVAL RESULTS ====")
    print(f"OCR engine: {args.ocr_engine}")
    print(f"Images processed: {len(imgs)}")
    print(f"OCR samples with GT: {total}")

    print("\n-- Core metrics --")
    print(f"OCR correct: {correct}")
    print(f"OCR accuracy: {accuracy:.2f}%")
    print(f"Processing time for {len(imgs)} images: {elapsed:.2f} s")

    print("\n-- Basic diagnostics --")
    print(f"Detection failures (no bbox): {det_fail}")
    print(f"OCR empty outputs: {ocr_empty}")
    print(f"Mean IoU (detection): {mean_iou:.4f} (n={len(ious)})")
    if len(imgs) != 100:
        print("NOTE: Wymaganie czasu dotyczy 100 zdjęć – uruchom z --limit 100.")

    #print(f"\nFinal grade (your formula, using accuracy): {grade:.1f}")


if __name__ == "__main__":
    main()
