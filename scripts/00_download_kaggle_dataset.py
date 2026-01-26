from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASET = "piotrstefaskiue/poland-vehicle-license-plate-dataset"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/kaggle_raw", help="Docelowy katalog na dane")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--local", default="", help="Ścieżka do lokalnego folderu z datasetem (np. plates_raw)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # TRYB LOKALNY (bez Kaggle)
    if args.local:
        src = Path(args.local)
        if not src.exists():
            raise SystemExit(f"Nie znaleziono --local: {src}")

        # kopiujemy zawartość folderu do out
        for item in src.iterdir():
            dst = out / item.name
            if dst.exists() and args.force:
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)

        print("Done. Local dataset copied to:", out)
        return

    # TRYB KAGGLE (jeśli kiedyś wrócisz)
    cmd = ["kaggle", "datasets", "download", "-d", "piotrstefaskiue/poland-vehicle-license-plate-dataset", "-p",
           "data/kaggle_raw"]
    if args.force:
        cmd.append("--force")
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    for z in out.glob("*.zip"):
        print("Unzipping:", z)
        subprocess.check_call([sys.executable, "-m", "zipfile", "-e", str(z), str(out)])

    print("Done. Raw dataset in:", out)

if __name__ == "__main__":
    main()
