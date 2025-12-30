from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

DDL = """
CREATE TABLE IF NOT EXISTS trusted_plates (
  plate TEXT PRIMARY KEY,
  added_at TEXT NOT NULL
);
"""

@dataclass
class PlateDB:
    path: str

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def init(self) -> None:
        with self.connect() as con:
            con.execute(DDL)
            con.commit()

    def add(self, plate: str, added_at_iso: str) -> None:
        self.init()
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO trusted_plates (plate, added_at) VALUES (?, ?)",
                (plate, added_at_iso)
            )
            con.commit()

    def exists(self, plate: str) -> bool:
        self.init()
        with self.connect() as con:
            cur = con.execute("SELECT 1 FROM trusted_plates WHERE plate = ? LIMIT 1", (plate,))
            return cur.fetchone() is not None

    def list(self, limit: int = 200) -> List[str]:
        self.init()
        with self.connect() as con:
            cur = con.execute("SELECT plate FROM trusted_plates ORDER BY added_at DESC LIMIT ?", (limit,))
            return [r[0] for r in cur.fetchall()]
