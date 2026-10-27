"""
DGSense SQLite Persistence Layer

Stores everything the UI previously kept only in React state (lost on refresh):
  - recognition_events     : Direction A prediction history
  - conversation_messages  : two-way conversation transcripts
  - practice_attempts      : Practice Mode results
  - recorded_samples       : Sign Recorder clips (30x144 landmarks) for future retraining

Uses the Python stdlib sqlite3 (no extra dependency). The database file lives next to
the backend unless DGSENSE_DB_PATH overrides it (tests point it at a temp file).
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from backend.config import settings, BASE_DIR

DB_PATH = Path(os.getenv("DGSENSE_DB_PATH", str(BASE_DIR / "dgsense.db")))

SCHEMA = """
CREATE TABLE IF NOT EXISTS recognition_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    canonical_label TEXT NOT NULL,
    german_text TEXT,
    english_text TEXT,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    request_id TEXT
);
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    speaker TEXT NOT NULL,
    speaker_name TEXT,
    text TEXT NOT NULL,
    translation TEXT,
    glosses TEXT,
    match_type TEXT,
    clips TEXT
);
CREATE TABLE IF NOT EXISTS practice_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    target_sign TEXT NOT NULL,
    recognized_sign TEXT,
    correct INTEGER NOT NULL,
    confidence REAL
);
CREATE TABLE IF NOT EXISTS recorded_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    label TEXT NOT NULL,
    signer_id TEXT,
    notes TEXT,
    frames TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recognition_created ON recognition_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversation_created ON conversation_messages(id);
CREATE INDEX IF NOT EXISTS idx_samples_label ON recorded_samples(label);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def get_db():
    """Yield a committed connection; rolls back on exception."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_db():
        pass


# --- recognition history ----------------------------------------------------

def insert_recognition_event(canonical_label: str, german_text: Optional[str],
                             english_text: Optional[str], confidence: float,
                             status: str, request_id: Optional[str] = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO recognition_events (created_at, canonical_label, german_text, "
            "english_text, confidence, status, request_id) VALUES (?,?,?,?,?,?,?)",
            (_now(), canonical_label, german_text, english_text, confidence, status, request_id),
        )
        return int(cur.lastrowid)


def list_recognition_events(limit: int = 50) -> List[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM recognition_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def count_recognition_events() -> int:
    with get_db() as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM recognition_events").fetchone()["n"])


def clear_recognition_events() -> int:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM recognition_events")
        return cur.rowcount


# --- conversation -----------------------------------------------------------

def insert_conversation_message(speaker: str, speaker_name: Optional[str], text: str,
                                translation: Optional[str] = None,
                                glosses: Optional[list] = None,
                                match_type: Optional[str] = None,
                                clips: Optional[list] = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO conversation_messages (created_at, speaker, speaker_name, text, "
            "translation, glosses, match_type, clips) VALUES (?,?,?,?,?,?,?,?)",
            (_now(), speaker, speaker_name, text, translation,
             json.dumps(glosses or [], ensure_ascii=False), match_type,
             json.dumps(clips or [], ensure_ascii=False)),
        )
        return int(cur.lastrowid)


def list_conversation_messages(limit: int = 500) -> List[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM conversation_messages ORDER BY id ASC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["glosses"] = json.loads(d.get("glosses") or "[]")
            d["clips"] = json.loads(d.get("clips") or "[]")
            out.append(d)
        return out


def clear_conversation_messages() -> int:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM conversation_messages")
        return cur.rowcount


# --- practice attempts ------------------------------------------------------

def insert_practice_attempt(target_sign: str, recognized_sign: Optional[str],
                            correct: bool, confidence: Optional[float]) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO practice_attempts (created_at, target_sign, recognized_sign, "
            "correct, confidence) VALUES (?,?,?,?,?)",
            (_now(), target_sign, recognized_sign, 1 if correct else 0, confidence),
        )
        return int(cur.lastrowid)


def list_practice_attempts(limit: int = 100) -> List[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM practice_attempts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def clear_practice_attempts() -> int:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM practice_attempts")
        return cur.rowcount


# --- recorded samples (recorder upload for future retraining) ---------------

def insert_recorded_sample(label: str, frames: list, signer_id: Optional[str] = None,
                           notes: Optional[str] = None) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO recorded_samples (created_at, label, signer_id, notes, frames) "
            "VALUES (?,?,?,?,?)",
            (_now(), label, signer_id, notes, json.dumps(frames)),
        )
        return int(cur.lastrowid)


def list_recorded_samples(include_frames: bool = False) -> List[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, created_at, label, signer_id, notes, frames "
            "FROM recorded_samples ORDER BY id ASC"
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["frame_count"] = len(json.loads(d.pop("frames")))
            if include_frames:
                d["frames"] = json.loads(conn.execute(
                    "SELECT frames FROM recorded_samples WHERE id = ?", (d["id"],)
                ).fetchone()["frames"])
            out.append(d)
        return out


def get_recorded_sample(sample_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, created_at, label, signer_id, notes, frames "
            "FROM recorded_samples WHERE id = ?", (sample_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["frames"] = json.loads(d["frames"])
        d["frame_count"] = len(d["frames"])
        return d


def delete_recorded_sample(sample_id: int) -> int:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM recorded_samples WHERE id = ?", (sample_id,))
        return cur.rowcount


def clear_recorded_samples() -> int:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM recorded_samples")
        return cur.rowcount