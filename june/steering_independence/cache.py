"""SQLite cache for judge API calls."""

import json
import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JudgeCache:
    """Thread-safe SQLite cache for judge scores."""

    def __init__(self, db_path: str = "outputs/judge_cache.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Init table on first connection
        conn = self._conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(cache_key TEXT PRIMARY KEY, score REAL, timestamp TEXT)"
        )
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
        return self._local.conn

    @staticmethod
    def make_key(**kwargs) -> str:
        """Create a deterministic cache key from keyword arguments."""
        raw = json.dumps(kwargs, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[float]:
        row = self._conn().execute(
            "SELECT score FROM cache WHERE cache_key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put(self, key: str, score: float) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO cache (cache_key, score, timestamp) VALUES (?, ?, ?)",
            (key, score, datetime.now(timezone.utc).isoformat()),
        )
        self._conn().commit()
