# ANPR (Automatic Number Plate Recognition) + Emergency Vehicle Classification  
Detekcja tablicy + OCR + kontrola dostępu + rozpoznawanie pojazdów uprzywilejowanych (ambulans itp.)

Projekt realizuje kompletny „flow bramy”:
1) **Najpierw** klasyfikacja, czy pojazd jest **uprzywilejowany** (*Emergency / Non-Emergency*).  
2) Jeśli **tak** → brama otwiera się automatycznie (**bez OCR**).  
3) Jeśli **nie** → uruchamiamy **ANPR**: detekcja tablicy + OCR + sprawdzenie whitelisty w SQLite.

---

## Spis treści
- [1) Szybki start (aplikacja Streamlit)](#1-szybki-start-aplikacja-streamlit)
- [2) Testowe zdjęcia + whitelist (baza zaufanych tablic)](#2-testowe-zdjęcia--whitelist-baza-zaufanych-tablic)
- [3) ANPR: Kaggle → YOLO → trening detektora tablic](#3-anpr-kaggle--yolo--trening-detektora-tablic)
- [4) ANPR: Ewaluacja (OCR accuracy, czas, IoU)](#4-anpr-ewaluacja-ocr-accuracy-czas-iou)
- [5) Emergency Vehicles: Kaggle → przygotowanie → trening → ewaluacja](#5-emergency-vehicles-kaggle--przygotowanie--trening--ewaluacja)
- [6) Integracja w aplikacji i konfiguracja progu](#6-integracja-w-aplikacji-i-konfiguracja-progu)
- [7) (Opcjonalnie) Demo z kamerą / symulacja środowiska](#7-opcjonalnie-demo-z-kamerą--symulacja-środowiska)
- [Struktura projektu](#struktura-projektu)
- [Uwagi praktyczne](#uwagi-praktyczne)

---

## 1) Szybki start (aplikacja Streamlit)

### Instalacja
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

> Jeśli instalujesz `easyocr` / `ultralytics` i środowisko nie dobierze automatycznie PyTorch,
> doinstaluj PyTorch zgodnie z instrukcją dla Twojej platformy (CPU/CUDA).

### Uruchom aplikację
```bash
streamlit run app/streamlit_app.py
```

### Jak działa aplikacja
- **Tryb Użytkownik**: upload lub wybór zdjęcia testowego → uruchomienie analizy → podgląd wyniku i decyzji bramy.
- **Tryb Administrator**: zmiana configu pipeline, ustawienie progu klasyfikacji *Emergency*, zarządzanie whitelistą w bazie SQLite.

---

## 2) Testowe zdjęcia + whitelist (baza zaufanych tablic)

- Testowe obrazy: `data/test_images/`
- Baza zaufanych tablic (SQLite): `data/plates.db`
- Lista startowa (CSV): `data/whitelist.csv`

Dodaj wpisy do bazy:
```bash
python scripts/db_manage.py add "SK12345"
python scripts/db_manage.py add "WX9A2BC"
python scripts/db_manage.py list
```

---

## 3) ANPR: Kaggle → YOLO → trening detektora tablic

### 3.1 Pobranie bazy z Kaggle
Wymaga skonfigurowania Kaggle API (plik `~/.kaggle/kaggle.json`).

```bash
python scripts/00_download_kaggle_dataset.py --out data/kaggle_raw
```

### 3.2 Przygotowanie danych pod YOLO (konwersja + split)
Skrypt `scripts/01_prepare_dataset.py` próbuje wykryć format adnotacji (YOLO txt / PascalVOC XML / COCO JSON).

Wynik zapisuje do:
- `data/yolo/images/{train,val,test}`
- `data/yolo/labels/{train,val,test}`
- `configs/dataset.yaml`

```bash
python scripts/01_prepare_dataset.py --raw data/kaggle_raw --out data/yolo --test-ratio 0.30
```

### 3.3 Trenowanie detektora tablic (YOLOv8)
```bash
python scripts/02_train_detector.py --data configs/dataset.yaml --epochs 50 --imgsz 640
```

Po treningu wagi są w:
- `runs/detect/train/weights/`

---

## 4) ANPR: Ewaluacja (OCR accuracy, czas, IoU)

Zakładamy, że dataset ma ground-truth tekst tablicy (np. CSV mapujący `filename -> plate_text`).
Wskaż plik mapowania przez `--gt-csv`.

```bash
python scripts/03_evaluate.py \
  --images data/yolo/images/test \
  --labels data/yolo/labels/test \
  --gt-csv data/kaggle_raw/ground_truth.csv \
  --weights runs/detect/train/weights/best.pt \
  --limit 100
```

Skrypt wypisze:
- OCR accuracy (%),
- średnie IoU dla detekcji,
- czas na 100 zdjęć,
- ocenę końcową wg funkcji z PDF.

---

## 5) Emergency Vehicles: Kaggle → przygotowanie → trening → ewaluacja

Ten moduł pozwala wykryć pojazd uprzywilejowany (np. ambulans) **zanim** uruchomimy ANPR.

### 5.1 Pobranie datasetu (Kaggle)
Dataset: `abhisheksinghblr/emergency-vehicles-identification`

```bash
python scripts/05_download_emergency_data.py
```

Docelowo dane powinny znaleźć się w:
- `data/emergency_vehicles/`

### 5.2 Przygotowanie danych do klasyfikacji YOLO (train/val, foldery klas)
Skrypt buduje strukturę:
- `data/emergency_yolo_cls/train/emergency`
- `data/emergency_yolo_cls/train/non_emergency`
- `data/emergency_yolo_cls/val/emergency`
- `data/emergency_yolo_cls/val/non_emergency`

```bash
python scripts/06_prepare_emergency_classification.py
```

> Uwaga: skrypt kopiuje obrazy na podstawie `train.csv`. Jeśli chcesz zawsze przebudowywać od zera,
> możesz odkomentować czyszczenie katalogu w skrypcie.

### 5.3 Trening klasyfikatora (YOLOv8-cls)
```bash
python scripts/07_train_emergency_cls.py
```

Model bazowy: `yolov8n-cls.pt`  
Wynik: `best.pt` w katalogu treningowym (zależnie od ustawień Ultralytics – zwykle w `runs/classify/...`).

### 5.4 Ewaluacja na walidacji
```bash
python scripts/08_evaluate_emergency_classes.py
```

Skrypt liczy accuracy na `val/` dla progowania predykcji (próg ustawiony w skrypcie).

---

## 6) Integracja w aplikacji i konfiguracja progu

### 6.1 Flow decyzyjny
Aplikacja działa według logiki:
1) **Emergency classifier** → jeśli klasa `emergency` i confidence ≥ próg → **brama otwarta automatycznie**  
2) inaczej → klasyczne ANPR (detekcja tablicy + OCR + whitelist)

W trybie „Emergency” aplikacja celowo:
- nie uruchamia OCR,
- nie wymaga wykrycia tablicy,
- zapisuje w `timing_ms` informacje o:
  - `emergency_cls_conf`,
  - `emergency_cls_name`,
  - `emergency_threshold`.

### 6.2 Lokalizacja modelu emergency
Aplikacja sama szuka modelu w typowych miejscach, priorytetowo:
1) `runs/classify/runs/train/emergency_classification_v2/weights/best.pt`
2) następnie rekurencyjnie:
   - `runs/**/emergency_classification*/weights/best.pt`
   - `runs/**/emergency*/weights/best.pt`
(ostatnio modyfikowany plik wygrywa)

Jeśli model nie zostanie znaleziony:
- sidebar pokaże status **BRAK**,
- klasyfikacja emergency zwróci `model_missing` i system przejdzie do ANPR.

### 6.3 Próg „Emergency” (administrator)
Próg jest ustawiany w aplikacji (Tryb Administrator) i zapisywany do:
- `data/emergency_settings.json`

Format:
```json
{ "emergency_threshold": 0.80 }
```

Dodatkowo w aplikacji jest bezpieczny clamp:
- min: `0.50`
- max: `0.99`

---

## 7) (Opcjonalnie) Demo z kamerą / symulacja środowiska
```bash
python scripts/04_camera_demo.py --config configs/app_config.yaml
```

Możesz ustawić `detector.type=yolo` po treningu, aby demo działało na realnych kadrach.

---

## Struktura projektu

```
app/
  streamlit_app.py

anpr/
  pipeline.py
  utils.py
  ...

configs/
  app_config.yaml
  dataset.yaml

data/
  plates.db
  whitelist.csv
  test_images/
  kaggle_raw/
  yolo/
  emergency_vehicles/
  emergency_yolo_cls/
  emergency_settings.json

scripts/
  00_download_kaggle_dataset.py
  01_prepare_dataset.py
  02_train_detector.py
  03_evaluate.py
  04_camera_demo.py
  05_download_emergency_data.py
  06_prepare_emergency_classification.py
  07_train_emergency_cls.py
  08_evaluate_emergency_classes.py
  db_manage.py
```

---

## Uwagi praktyczne

### OCR (Tesseract)
Jeśli używasz `pytesseract`, to poza `pip install` musisz mieć zainstalowany systemowy Tesseract (binarka).
W przeciwnym razie OCR na Tesseract nie zadziała.

### PyTorch
`easyocr` i `ultralytics` mogą wymagać PyTorch. W wielu środowiskach instalacja PyTorch jest zależna od platformy (CPU/CUDA),
dlatego najlepiej instalować go zgodnie z oficjalną instrukcją dla Twojej konfiguracji.

### Reprodukowalność
Zalecenie: nie instalować bibliotek w runtime w skryptach (np. `pip install kaggle` w trakcie działania).  
Docelowo wszystkie zależności powinny być w `requirements.txt`, a środowisko stawiane raz.

### Stabilność wyników
Jeżeli celem jest dobra jakość:
- YOLO detektor trenowany na Twoim dataspec,
- OCR na cropie tablicy + preprocessing (kontrast, odszumianie, deskew),
- normalizacja znaków i walidacja regex dla tablic PL.
