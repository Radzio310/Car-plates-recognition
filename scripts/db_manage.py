from __future__ import annotations
import argparse
from datetime import datetime, timezone

import yaml
from anpr.db import PlateDB
from anpr.utils import normalize_plate

def load_db_path(config_path: str) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["access_control"]["sqlite_path"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "add", "exists", "list"])
    ap.add_argument("--config", default="configs/app_config.yaml")
    ap.add_argument("--plate", default="")
    args = ap.parse_args()

    db_path = load_db_path(args.config)
    db = PlateDB(path=db_path)

    if args.cmd == "init":
        db.init()
        print(f"OK: initialized {db_path}")
        return

    if args.cmd == "add":
        if not args.plate:
            raise SystemExit("Podaj --plate")
        plate = normalize_plate(args.plate, allowed_chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        db.add(plate, datetime.now(timezone.utc).isoformat())
        print(f"OK: added {plate}")
        return

    if args.cmd == "exists":
        if not args.plate:
            raise SystemExit("Podaj --plate")
        plate = normalize_plate(args.plate, allowed_chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        print(db.exists(plate))
        return

    if args.cmd == "list":
        for p in db.list():
            print(p)

if __name__ == "__main__":
    main()
