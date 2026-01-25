from __future__ import annotations
from dataclasses import dataclass
from typing import List

import psycopg2
import time

DDL = """
CREATE TABLE IF NOT EXISTS trusted_plates (
  plate TEXT PRIMARY KEY,
  added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

@dataclass
class PlateDB:
    # Teraz path zastępujemy parametrami połączenia (lub jednym stringiem URL)
    host: str = "localhost"
    database: str = "anpr_db"
    user: str = "user"
    password: str = "password123"
    port: int = 5432

    def connect(self):
        host_to_try = "127.0.0.1" if self.host == "localhost" else self.host
        dsn = f"host={host_to_try} dbname={self.database} user={self.user} password={self.password} port={self.port} connect_timeout=5"

        for i in range(5):
            try:
                conn = psycopg2.connect(dsn)
                conn.set_client_encoding('UTF8')
                return conn
            except Exception as e:
                print(f"Próba {i + 1}/5: Błąd połączenia (sprawdź hasło/użytkownika)")
                time.sleep(2)

        raise Exception("Ostateczny błąd połączenia. Sprawdź logi Dockera: docker logs anpr_postgres")

    def init(self) -> None:
        with self.connect() as con:
            with con.cursor() as cur:
                cur.execute(DDL)
            con.commit()

    def add(self, plate: str, added_at_iso: str) -> None:
        with self.connect() as con:
            with con.cursor() as cur:
                cur.execute(
                    "INSERT INTO trusted_plates (plate, added_at) VALUES (%s, %s) ON CONFLICT (plate) DO UPDATE SET added_at = EXCLUDED.added_at",
                    (plate, added_at_iso)
                )
            con.commit()

    def exists(self, plate: str) -> bool:
        with self.connect() as con:
            with con.cursor() as cur:
                cur.execute("SELECT 1 FROM trusted_plates WHERE plate = %s", (plate,))
                return cur.fetchone() is not None

    def list(self, limit: int = 200) -> List[str]:
        """Pobiera listę zaufanych tablic z bazy PostgreSQL."""
        with self.connect() as con:
            with con.cursor() as cur:
                # Wybieramy kolumnę 'plate' z tabeli 'trusted_plates'
                cur.execute("SELECT plate FROM trusted_plates ORDER BY added_at DESC LIMIT %s", (limit,))
                rows = cur.fetchall()
                # Zwracamy listę samych numerów (bez krotek)
                return [r[0] for r in rows]