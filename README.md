# ANPR (Automatic Number Plate Recognition) – detekcja + OCR + kontrola dostępu

Ten projekt jest szkieletem do zaliczenia: detekcja tablicy + OCR + ewaluacja (dokładność, czas) oraz przygotowanie pod trenowanie detektora na bazie z Kaggle.

Wymagania z PDF (skrót):
- użycie bazy Kaggle,
- detekcja tablic + OCR,
- ocena: dokładność OCR oraz czas 100 zdjęć,
- przy trenowaniu detektora: test >= 30% + IoU,
- praca na branchu `automatic_plate_number_recognition`.

## 1) Szybki start (aplikacja z uploadem zdjęcia)

### Instalacja
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

### Uruchom aplikację
```bash
streamlit run app/streamlit_app.py
```

W aplikacji:
1. Klikasz „Wgraj zdjęcie”.
2. Dostajesz:
   - wykrytą tablicę (bbox + crop),
   - odczytany numer (lub błąd),
   - wynik weryfikacji w bazie (Access Granted/Denied).

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

## 3) Pobranie bazy z Kaggle

Wymaga skonfigurowania Kaggle API (plik `~/.kaggle/kaggle.json`).
```bash
python scripts/00_download_kaggle_dataset.py --out data/kaggle_raw
```

## 4) Przygotowanie danych pod YOLO (konwersja + split)

Skrypt `scripts/01_prepare_dataset.py` próbuje wykryć format adnotacji (YOLO txt / PascalVOC XML / COCO JSON).
Wynik zapisuje do:
- `data/yolo/images/{train,val,test}`
- `data/yolo/labels/{train,val,test}`
- `configs/dataset.yaml`

```bash
python scripts/01_prepare_dataset.py --raw data/kaggle_raw --out data/yolo --test-ratio 0.30
```

## 5) Trenowanie detektora tablic (YOLOv8)

```bash
python scripts/02_train_detector.py --data configs/dataset.yaml --epochs 50 --imgsz 640
```

Po treningu wagi są w katalogu `runs/detect/train/weights/`.

## 6) Ewaluacja (OCR accuracy, czas 100 zdjęć, IoU)

Zakładamy, że dataset ma ground-truth tekst tablicy (np. CSV mapujący filename -> plate_text).
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

## Konfiguracja pipeline

`configs/app_config.yaml`:
- ścieżka do wag YOLO,
- wybór OCR (`easyocr`/`tesseract`),
- próg confidence,
- walidacja/normalizacja numeru.

## Uwaga dot. jakości
Jeżeli chcesz uzyskać stabilną dokładność, rekomendacja:
- detektor YOLO trenowany na tej bazie,
- OCR na cropie tablicy + pre-processing (deskew/contrast),
- prosta normalizacja znaków i walidacja wzorca tablic PL.

## 7) (Opcjonalnie) Demo z kamerą / symulacja środowiska

```bash
python scripts/04_camera_demo.py --config configs/app_config.yaml
```

Możesz ustawić `detector.type=yolo` po treningu, aby demo działało na realnych kadrach.
