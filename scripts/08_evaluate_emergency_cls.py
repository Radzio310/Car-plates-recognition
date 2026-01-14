import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def evaluate_test_set():
    # 1. Paths
    model_path = Path("runs/train/emergency_classification_v2/weights/best.pt")
    test_images_dir = Path("data/emergency_vehicles/Emergency_Vehicles/test")
    gt_csv = Path("data/emergency_vehicles/Emergency_Vehicles/ground_truth.csv")
    output_csv = Path("data/emergency_vehicles/Emergency_Vehicles/test_predictions.csv")

    if not model_path.exists():
        print(f"BŁĄD: Nie znaleziono modelu w {model_path}")
        return

    if not gt_csv.exists():
        print(f"BŁĄD: Nie znaleziono ground truth w {gt_csv}")
        return

    # 2. Load model and ground truth
    model = YOLO(model_path)
    gt_df = pd.read_csv(gt_csv)

    gt_map = dict(zip(gt_df["image_names"], gt_df["emergency_or_not"]))

    all_test_images = list(test_images_dir.glob("*.jpg"))

    results_rows = []

    correct = 0
    total = 0
    skipped = 0

    print(f"Rozpoczynam testowanie na {len(all_test_images)} zdjęciach...")

    for img_path in tqdm(all_test_images):
        img_name = img_path.name

        if img_name not in gt_map:
            skipped += 1
            continue

        gt_label = int(gt_map[img_name])

        # Prediction
        result = model.predict(source=str(img_path), verbose=False)[0]

        pred_idx = int(result.probs.top1)
        pred_conf = float(result.probs.top1conf)
        pred_class_name = result.names[pred_idx]

        threshold = 0.6
        pred_label = 1 if (pred_class_name == "emergency" and pred_conf >= threshold) else 0

        is_correct = int(pred_label == gt_label)

        results_rows.append({
            "image_name": img_name,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "confidence": round(pred_conf, 4),
            "is_correct": is_correct
        })

        if is_correct:
            correct += 1
        total += 1

    # Save CSV
    df_out = pd.DataFrame(results_rows)
    df_out.to_csv(output_csv, index=False)

    accuracy = (correct / total) * 100 if total > 0 else 0

    print("\n==== WYNIKI TESTU (GROUND TRUTH) ====")
    print(f"Przetworzonych zdjęć: {total}")
    print(f"Pominiętych (brak GT): {skipped}")
    print(f"Poprawnych predykcji: {correct}")
    print(f"Dokładność (Accuracy): {accuracy:.2f}%")
    print(f"Zapisano predykcje do: {output_csv.resolve()}")


if __name__ == "__main__":
    evaluate_test_set()
