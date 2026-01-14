import os
import zipfile
from pathlib import Path


def download_and_setup():
    # 1. Definicja ścieżek
    dataset_slug = "abhisheksinghblr/emergency-vehicles-identification"
    target_dir = Path("data/emergency_vehicles")
    zip_path = target_dir / "dataset.zip"

    # Tworzenie folderu jeśli nie istnieje
    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. Instalacja biblioteki kaggle jeśli jej brak
    try:
        import kaggle
    except ImportError:
        print("Instalowanie biblioteki kaggle...")
        os.system("pip install kaggle")
        import kaggle

    # 3. Pobieranie
    print(f"Pobieranie datasetu {dataset_slug}...")
    try:
        # Pobieranie plików datasetu
        kaggle.api.dataset_download_files(dataset_slug, path=target_dir, unzip=False)

        # Znalezienie pobranego pliku zip (nazwa może się różnić)
        downloaded_zip = list(target_dir.glob("*.zip"))[0]

        # 4. Rozpakowywanie
        print("Rozpakowywanie plików...")
        with zipfile.ZipFile(downloaded_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        # Usuwamy zip po rozpakowaniu
        downloaded_zip.unlink()

        print(f"Sukces! Dane znajdują się w: {target_dir.absolute()}")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        print("\nUpewnij się, że masz plik kaggle.json w ~/.kaggle/")


if __name__ == "__main__":
    download_and_setup()