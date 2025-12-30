from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

# =========
# CVAT -> YOLO dataset builder
# =========

# YOLO label format per line:
# class_id x_center y_center width height  (all normalized 0..1)

BBox = Tuple[float, float, float, float]  # x1,y1,x2,y2


@dataclass
class ImageAnn:
    name: str
    width: int
    height: int
    boxes: List[BBox]
    plate_texts: List[str]


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def find_images_root(raw: Path) -> Path:
    """
    Dataset expected:
      raw/annotations.xml
      raw/photos/*.jpg
    But we try to be flexible.
    """
    if (raw / "photos").is_dir():
        return raw / "photos"
    if (raw / "images").is_dir():
        return raw / "images"
    # fallback: raw itself
    return raw


def parse_cvat_annotations(xml_path: Path) -> Dict[str, ImageAnn]:
    """
    Parses CVAT annotations.xml with structure:
      <image id="..." name="31.jpg" width="..." height="...">
        <box label="plate" xtl="..." ytl="..." xbr="..." ybr="...">
          <attribute name="plate number">...</attribute>
        </box>
      </image>
    """
    txt = xml_path.read_text(encoding="utf-8", errors="ignore")
    root = ET.fromstring(txt)

    out: Dict[str, ImageAnn] = {}
    for im in root.findall("image"):
        name = im.get("name") or ""
        if not name:
            continue

        # sometimes CVAT stores name like "photos/31.jpg"
        name_basename = Path(name).name

        try:
            w = int(im.get("width", "0"))
            h = int(im.get("height", "0"))
        except ValueError:
            w, h = 0, 0

        boxes: List[BBox] = []
        texts: List[str] = []

        for box in im.findall("box"):
            if box.get("label") != "plate":
                continue

            x1 = float(box.get("xtl", "0"))
            y1 = float(box.get("ytl", "0"))
            x2 = float(box.get("xbr", "0"))
            y2 = float(box.get("ybr", "0"))

            # Fix inverted coords just in case
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            boxes.append((x1, y1, x2, y2))

            # plate number text (optional)
            plate_text = ""
            for attr in box.findall("attribute"):
                if (attr.get("name") or "").strip().lower() == "plate number":
                    plate_text = (attr.text or "").strip()
                    break
            if plate_text:
                texts.append(plate_text)

        out[name_basename] = ImageAnn(
            name=name_basename,
            width=w,
            height=h,
            boxes=boxes,
            plate_texts=texts,
        )

    return out


def resolve_image_path(images_root: Path, filename: str) -> Optional[Path]:
    """
    Resolve filename under images_root.
    Tries:
      images_root/filename
      scan by basename if not found
    """
    p = images_root / filename
    if p.exists():
        return p

    # fallback: search by basename
    base = Path(filename).name
    candidates = list(images_root.rglob(base))
    if candidates:
        return candidates[0]

    return None


def bbox_to_yolo(b: BBox, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b

    # clamp to image bounds
    x1 = clamp(x1, 0, img_w - 1)
    x2 = clamp(x2, 0, img_w - 1)
    y1 = clamp(y1, 0, img_h - 1)
    y2 = clamp(y2, 0, img_h - 1)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    # normalize
    return (xc / img_w, yc / img_h, bw / img_w, bh / img_h)


def write_yolo_label(label_path: Path, boxes: List[BBox], img_w: int, img_h: int, class_id: int = 0) -> None:
    lines: List[str] = []
    for b in boxes:
        xc, yc, w, h = bbox_to_yolo(b, img_w, img_h)
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    """
    mode: copy | hardlink
    """
    safe_mkdir(dst.parent)
    if dst.exists():
        return

    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except Exception:
            # fallback to copy
            pass

    shutil.copy2(src, dst)


def write_ultralytics_yaml(yaml_path: Path, yolo_root: Path, class_name: str = "plate") -> None:
    """
    Ultralytics dataset.yaml:
      path: <root>
      train: images/train
      val: images/val
      test: images/test
      names:
        0: plate
    """
    content = "\n".join(
        [
            f'path: "{yolo_root.as_posix()}"',
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    safe_mkdir(yaml_path.parent)
    yaml_path.write_text(content, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare YOLO dataset from CVAT annotations.xml + photos/")
    ap.add_argument("--raw", default="data/plates", help="Folder with annotations.xml and photos/")
    ap.add_argument("--out", default="data/yolo", help="Output YOLO dataset folder")
    ap.add_argument("--test-ratio", type=float, default=0.30, help="Test split ratio (min 0.30 recommended)")
    ap.add_argument("--val-ratio", type=float, default=0.10, help="Val split ratio")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["copy", "hardlink"], default="copy", help="How to place images in output")
    ap.add_argument("--class-name", default="plate")
    ap.add_argument("--yaml-out", default="configs/dataset.yaml", help="Where to write ultralytics dataset yaml")
    ap.add_argument("--write-ocr-gt", action="store_true", help="Write OCR ground-truth CSV (image -> plate_number)")
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    yaml_out = Path(args.yaml_out)

    if not raw.exists():
        raise SystemExit(f"RAW path not found: {raw}")

    xml_path = raw / "annotations.xml"
    if not xml_path.exists():
        # fallback: first xml
        xmls = list(raw.rglob("*.xml"))
        if not xmls:
            raise SystemExit(f"annotations.xml not found in: {raw}")
        xml_path = xmls[0]

    images_root = find_images_root(raw)

    print("RAW:", raw)
    print("XML:", xml_path)
    print("IMAGES ROOT:", images_root)

    anns = parse_cvat_annotations(xml_path)

    # Keep only images that exist
    items: List[Tuple[Path, ImageAnn]] = []
    missing = 0
    for fname, ann in anns.items():
        ip = resolve_image_path(images_root, fname)
        if not ip:
            missing += 1
            continue
        # ignore non-images
        if ip.suffix.lower() not in IMG_EXTS:
            continue
        items.append((ip, ann))

    if not items:
        raise SystemExit("No images resolved from annotations. Check folder structure (photos/) and XML names.")

    if missing:
        print(f"Warning: missing images referenced by XML: {missing}")

    # Split
    test_ratio = float(args.test_ratio)
    val_ratio = float(args.val_ratio)
    if test_ratio < 0.30:
        print("Warning: --test-ratio < 0.30 (your assignment requires test >= 30%).")

    if test_ratio + val_ratio >= 1.0:
        raise SystemExit("Invalid split: test_ratio + val_ratio must be < 1.0")

    random.seed(args.seed)
    random.shuffle(items)

    n = len(items)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise SystemExit("Split produced empty train set. Adjust ratios.")

    test_items = items[:n_test]
    val_items = items[n_test : n_test + n_val]
    train_items = items[n_test + n_val :]

    # Prepare output dirs
    for split in ["train", "val", "test"]:
        safe_mkdir(out / "images" / split)
        safe_mkdir(out / "labels" / split)

    # Optional OCR GT CSV
    ocr_rows: List[Tuple[str, str]] = []

    def process_split(split_name: str, split_items: List[Tuple[Path, ImageAnn]]) -> None:
        for img_path, ann in split_items:
            dst_img = out / "images" / split_name / img_path.name
            dst_lbl = out / "labels" / split_name / (img_path.stem + ".txt")

            # copy image
            copy_or_link(img_path, dst_img, args.mode)

            # label
            if ann.width <= 0 or ann.height <= 0:
                # fallback: try reading image size via PIL if available
                try:
                    from PIL import Image
                    with Image.open(img_path) as im:
                        w, h = im.size
                except Exception:
                    raise SystemExit(f"Missing width/height in XML and cannot read image: {img_path}")
            else:
                w, h = ann.width, ann.height

            write_yolo_label(dst_lbl, ann.boxes, w, h, class_id=0)

            # OCR ground truth (if available)
            if ann.plate_texts:
                # keep first text or join if multiple
                gt = ann.plate_texts[0]
                ocr_rows.append((img_path.name, gt))

    print(f"Total resolved images: {n}")
    print(f"Split: train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    process_split("train", train_items)
    process_split("val", val_items)
    process_split("test", test_items)

    # Write dataset.yaml for ultralytics
    write_ultralytics_yaml(yaml_out, out.resolve(), class_name=args.class_name)
    print("Wrote dataset YAML:", yaml_out)

    # Write optional OCR GT CSV
    if args.write_ocr_gt:
        gt_path = out / "ocr_ground_truth.csv"
        with gt_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "plate_number"])
            for row in ocr_rows:
                w.writerow(row)
        print("Wrote OCR ground truth:", gt_path, f"(rows={len(ocr_rows)})")

    print("Done. YOLO dataset at:", out)


if __name__ == "__main__":
    main()
