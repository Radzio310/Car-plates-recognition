from __future__ import annotations
import argparse
import yaml
from datetime import datetime, timezone
from pathlib import Path
from anpr.db import PlateDB
from anpr.utils import normalize_plate


def load_db_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku konfiguracji: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("access_control", {})


def main():
    ap = argparse.ArgumentParser(description="ANPR Database Management (PostgreSQL)")
    ap.add_argument("cmd", choices=["init", "add", "exists", "list"], help="Komenda do wykonania")
    ap.add_argument("--config", default="configs/app_config.yaml", help="Ścieżka do app_config.yaml")
    ap.add_argument("--plate", default="", help="Numer tablicy do dodania/sprawdzenia")
    args = ap.parse_args()

    try:
        # Wczytanie konfiguracji
        conf = load_db_config(args.config)

        # Inicjalizacja bazy danych
        db = PlateDB(
            host=conf.get("host", "localhost"),
            database=conf.get("database", "anpr_db"),
            user=conf.get("user", "user"),
            password=conf.get("password", "password123"),
            port=int(conf.get("port", 5432))
        )

        if args.cmd == "init":
            db.init()
            print(f"✅ Baza PostgreSQL '{db.database}' została zainicjalizowana.")

        elif args.cmd == "add":
            if not args.plate:
                print("❌ Błąd: Musisz podać numer tablicy: --plate 'NUMER'")
                return
            plate = normalize_plate(args.plate, allowed_chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            db.add(plate, datetime.now(timezone.utc).isoformat())
            print(f"✅ Dodano tablicę: {plate}")

        elif args.cmd == "exists":
            if not args.plate:
                print("❌ Błąd: Podaj numer tablicy: --plate 'NUMER'")
                return
            plate = normalize_plate(args.plate, allowed_chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            found = db.exists(plate)
            print(f"Tablica {plate}: {'ZNAJDZIONA' if found else 'BRAK W BAZIE'}")

        elif args.cmd == "list":
            plates = db.list()
            print(f"\nLista zaufanych tablic ({len(plates)}):")
            for p in plates:
                print(f" - {p}")

    except Exception as e:
        print(f"\n❌ WYSTĄPIŁ BŁĄD: {e}")
        print("\nSprawdź:")
        print("1. Czy 'docker-compose up -d' został wykonany?")
        print("2. Czy w app_config.yaml host to 'localhost' (jeśli uruchamiasz skrypt lokalnie)?")


if __name__ == "__main__":
    main()