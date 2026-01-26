import argparse
import subprocess
import sys
from pathlib import Path

# Definicja datasetu
DATASET_SLUG = "abhisheksinghblr/emergency-vehicles-identification"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/emergency_vehicles", help="Katalog docelowy")
    ap.add_argument("--dataset", default=DATASET_SLUG)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Budujemy komendę tak samo jak w skrypcie 00
    cmd = [
        "kaggle", "datasets", "download",
        "-d", args.dataset,
        "-p", str(out)
    ]

    if args.force:
        cmd.append("--force")

    print(f"Running: {' '.join(cmd)}")

    try:
        # Wywołanie systemowe (omija błąd importu kaggle)
        subprocess.check_call(cmd)

        # Rozpakowywanie wszystkich plików zip w folderze docelowym
        for z in out.glob("*.zip"):
            print(f"Unzipping: {z}")
            subprocess.check_call([
                sys.executable, "-m", "zipfile", "-e", str(z), str(out)
            ])
            # Usuwamy zip po rozpakowaniu
            z.unlink()

        print(f"Done. Dataset in: {out}")

    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas pobierania: {e}")
        print("Upewnij się, że masz skonfigurowane CLI Kaggle lub plik kaggle.json")


if __name__ == "__main__":
    main()