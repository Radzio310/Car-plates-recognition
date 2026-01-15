from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

def evaluate_val_set():
    model_path = Path("runs/classify/runs/train/emergency_classification_v2/weights/best.pt")

    val_dir = Path("data/emergency_yolo_cls/val")
    emg_dir = val_dir / "emergency"
    non_dir = val_dir / "non_emergency"

    if not model_path.exists():
        print(f"BŁĄD: Nie znaleziono modelu w {model_path}")
        return
    if not val_dir.exists():
        print(f"BŁĄD: Nie znaleziono walidacji w {val_dir}")
        return

    model = YOLO(model_path)

    samples = []
    for p in emg_dir.glob("*.jpg"):
        samples.append((p, 1))
    for p in non_dir.glob("*.jpg"):
        samples.append((p, 0))

    if not samples:
        print("BŁĄD: Brak obrazów w val.")
        return

    correct = 0
    total = 0

    print(f"Testuję na val: {len(samples)} obrazów...")

    for img_path, gt_label in tqdm(samples):
        result = model.predict(source=str(img_path), verbose=False)[0]
        pred_idx = int(result.probs.top1)
        pred_conf = float(result.probs.top1conf)
        pred_class_name = result.names[pred_idx]

        threshold = 0.6
        pred_label = 1 if (pred_class_name == "emergency" and pred_conf >= threshold) else 0

        if pred_label == gt_label:
            correct += 1
        total += 1

    acc = (correct / total) * 100
    print(f"Accuracy (val): {acc:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate_val_set()
