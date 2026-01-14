from ultralytics import YOLO
from pathlib import Path

def train_emergency():
    # 1. Pobierz pełną (bezwzględną) ścieżkę do folderu z danymi
    dataset_path = Path("data/emergency_yolo_cls").resolve()

    if not dataset_path.exists():
        print(f"BŁĄD: Folder {dataset_path} nie istnieje!")
        print("Upewnij się, że najpierw uruchomiłeś skrypt 06_prepare_emergency_classification.py")
        return

    # 2. Ładowanie modelu klasyfikacji
    model = YOLO('yolov8n-cls.pt')

    # 3. Rozpoczęcie trenowania z użyciem pełnej ścieżki
    model.train(
        data=str(dataset_path),
        epochs=50,  # Zwiększ liczbę epok dla lepszego dotarcia modelu
        imgsz=448,  # Większy rozmiar obrazu poprawi trafność
        batch=16,
        project='runs/train',
        name='emergency_classification_v2'
    )

if __name__ == "__main__":
    train_emergency()