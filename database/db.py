"""
Small SQLite helper. Stores ONLY analytical results (timestamp, emotion,
confidence, engagement score) - never webcam frames or images.

Two tables:
  sessions           one row per session (id, start_time, end_time)
  session_analytics  many rows per session (one reading every few seconds)
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "database.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _run(sql, params=(), fetch=None):
    """Runs one SQL statement safely (always closes the connection)."""
    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute(sql, params)
            if fetch == "all":
                return [dict(row) for row in cursor.fetchall()]
            if fetch == "one":
                row = cursor.fetchone()
                return dict(row) if row else None
            return cursor.lastrowid
    finally:
        conn.close()


def init_db():
    # Older prototype versions had no session_id column: recreate that table
    # (only old test readings are lost).
    columns = [row["name"] for row in _run("PRAGMA table_info(session_analytics)", fetch="all")]
    if columns and "session_id" not in columns:
        _run("DROP TABLE session_analytics")

    _run("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            end_time TEXT
        )
    """)
    _run("""
        CREATE TABLE IF NOT EXISTS session_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            detected_emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            engagement_score REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    close_unfinished_sessions()


def close_unfinished_sessions():
    """If the app was closed in the middle of a session, give that session an end time."""
    _run("""
        UPDATE sessions
        SET end_time = COALESCE(
            (SELECT MAX(timestamp) FROM session_analytics WHERE session_id = sessions.id),
            start_time)
        WHERE end_time IS NULL
    """)


# ---------------- sessions ----------------
def create_session(start_time):
    return _run("INSERT INTO sessions (start_time) VALUES (?)", (start_time,))


def end_session(session_id, end_time):
    _run("UPDATE sessions SET end_time = ? WHERE id = ?", (end_time, session_id))


def get_session(session_id):
    return _run("SELECT * FROM sessions WHERE id = ?", (session_id,), fetch="one")


def get_latest_session_id():
    row = _run("SELECT MAX(id) AS latest FROM sessions", fetch="one")
    return row["latest"] if row else None


def get_sessions():
    """All sessions, newest first, with reading count and average engagement."""
    return _run("""
        SELECT s.id, s.start_time, s.end_time,
               COUNT(a.id) AS observation_count,
               AVG(a.engagement_score) AS average_engagement
        FROM sessions s
        LEFT JOIN session_analytics a ON a.session_id = s.id
        GROUP BY s.id
        ORDER BY s.id DESC
    """, fetch="all")


# ---------------- readings ----------------
def insert_record(session_id, timestamp, emotion, confidence, engagement_score):
    _run(
        "INSERT INTO session_analytics "
        "(session_id, timestamp, detected_emotion, confidence, engagement_score) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, timestamp, emotion, float(confidence), float(engagement_score)),
    )


def get_records(session_id):
    return _run("SELECT * FROM session_analytics WHERE session_id = ? ORDER BY id ASC",
                (session_id,), fetch="all")


def clear_history():
    """Deletes every session and reading (used by the 'Clear history' button)."""
    _run("DELETE FROM session_analytics")
    _run("DELETE FROM sessions")
