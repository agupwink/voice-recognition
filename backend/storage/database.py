"""
SQLite-backed user store.

Schema
──────
users
  username        TEXT  PRIMARY KEY
  display_name    TEXT
  passphrase_text TEXT
  voice_transcript TEXT
  voice_embedding BLOB   (numpy float32 array serialised with np.save)
  engine          TEXT
  created_at      REAL
  last_login      REAL

Thread safety: sqlite3 connections are NOT shared across threads.
Each call opens its own connection with WAL mode for concurrent reads.
"""

from __future__ import annotations

import io
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "users.db"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def _emb_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _blob_to_emb(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data))


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "username":         row["username"],
        "display_name":     row["display_name"],
        "passphrase_text":  row["passphrase_text"],
        "voice_transcript": row["voice_transcript"],
        "voice_embedding":  _blob_to_emb(row["voice_embedding"]),
        "engine":           row["engine"],
        "created_at":       row["created_at"],
        "last_login":       row["last_login"],
    }


# ── Public API ────────────────────────────────────────────────────────────────

def init() -> None:
    """Create the database and tables if they don't exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username         TEXT PRIMARY KEY,
                display_name     TEXT NOT NULL,
                passphrase_text  TEXT NOT NULL,
                voice_transcript TEXT NOT NULL,
                voice_embedding  BLOB NOT NULL,
                engine           TEXT NOT NULL,
                created_at       REAL NOT NULL,
                last_login       REAL
            )
        """)
        c.commit()
    count = get_user_count()
    logger.info("Database ready — %s  (%d user(s))", _DB_PATH, count)


def exists(username: str) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM users WHERE username = ?", (username.lower(),)
        ).fetchone()
    return row is not None


def create_user(
    username: str,
    display_name: str,
    passphrase_text: str,
    voice_transcript: str,
    voice_embedding: np.ndarray,
    engine: str,
) -> None:
    with _conn() as c:
        c.execute(
            """INSERT INTO users
               (username, display_name, passphrase_text, voice_transcript,
                voice_embedding, engine, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                username.lower(),
                display_name,
                passphrase_text,
                voice_transcript,
                _emb_to_blob(voice_embedding),
                engine,
                time.time(),
            ),
        )
        c.commit()


def get_user(username: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM users WHERE username = ?", (username.lower(),)
        ).fetchone()
    return _row_to_dict(row) if row else None


def touch_login(username: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE users SET last_login = ? WHERE username = ?",
            (time.time(), username.lower()),
        )
        c.commit()


def list_users() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT username, display_name, engine, created_at, last_login FROM users"
            " ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_user_count() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def delete_user(username: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM users WHERE username = ?", (username.lower(),))
        c.commit()
    return cur.rowcount > 0
