# Raport – ANPR (detekcja + OCR)

## 1. Dataset (Kaggle)
- Nazwa: Poland vehicle license plate dataset
- Split: train/val/test (test >= 30%)

## 2. Metody
### 2.1 Detekcja tablic
- Model: YOLOv8 (Ultralytics)
- Parametry: imgsz=640, conf=..., iou=...

### 2.2 OCR
- Silnik: EasyOCR / Tesseract
- Preprocessing: (opis)

## 3. Ewaluacja
### 3.1 Dokładność OCR
- accuracy = poprawne / wszystkie (test)

### 3.2 Czas 100 zdjęć
- czas = ... s

### 3.3 IoU (detekcja)
- mean IoU = ...

## 4. Wynik końcowy
- ocena wg funkcji z PDF: ...
