"""SQLite database models for player profiles and analysis sessions."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


DB_PATH = Path("data/pitch_analyzer.db")


@dataclass
class Player:
    id: Optional[int] = None
    name: str = ""
    throws: str = "R"
    age: Optional[float] = None
    height_inches: Optional[float] = None
    weight_lbs: Optional[float] = None
    team: str = ""
    notes: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class AnalysisSession:
    id: Optional[int] = None
    player_id: int = 0
    video_path: str = ""
    video_filename: str = ""
    backend: str = "yolov8"
    model_size: str = "m"
    confidence_threshold: float = 0.3
    output_dir: str = ""
    report_html: str = ""
    results_json: str = ""
    status: str = "pending"
    error_message: str = ""
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class Database:
    """SQLite database manager for the desktop application."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def initialize(self) -> None:
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                throws TEXT NOT NULL DEFAULT 'R',
                age REAL,
                height_inches REAL,
                weight_lbs REAL,
                team TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS analysis_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER NOT NULL REFERENCES players(id),
                video_path TEXT NOT NULL,
                video_filename TEXT NOT NULL,
                backend TEXT DEFAULT 'yolov8',
                model_size TEXT DEFAULT 'm',
                confidence_threshold REAL DEFAULT 0.3,
                output_dir TEXT,
                report_html TEXT,
                results_json TEXT,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
        """)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- Player CRUD --

    def add_player(self, player: Player) -> int:
        cur = self.conn.execute(
            """INSERT INTO players (name, throws, age, height_inches, weight_lbs, team, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (player.name, player.throws, player.age, player.height_inches,
             player.weight_lbs, player.team, player.notes),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_player(self, player: Player) -> None:
        self.conn.execute(
            """UPDATE players SET name=?, throws=?, age=?, height_inches=?, weight_lbs=?,
               team=?, notes=?, updated_at=CURRENT_TIMESTAMP WHERE id=?""",
            (player.name, player.throws, player.age, player.height_inches,
             player.weight_lbs, player.team, player.notes, player.id),
        )
        self.conn.commit()

    def delete_player(self, player_id: int) -> None:
        self.conn.execute("DELETE FROM analysis_sessions WHERE player_id=?", (player_id,))
        self.conn.execute("DELETE FROM players WHERE id=?", (player_id,))
        self.conn.commit()

    def get_all_players(self) -> list[Player]:
        rows = self.conn.execute("SELECT * FROM players ORDER BY name").fetchall()
        return [self._row_to_player(r) for r in rows]

    def get_player(self, player_id: int) -> Optional[Player]:
        row = self.conn.execute("SELECT * FROM players WHERE id=?", (player_id,)).fetchone()
        return self._row_to_player(row) if row else None

    # -- Session CRUD --

    def add_session(self, session: AnalysisSession) -> int:
        cur = self.conn.execute(
            """INSERT INTO analysis_sessions
               (player_id, video_path, video_filename, backend, model_size,
                confidence_threshold, output_dir, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session.player_id, session.video_path, session.video_filename,
             session.backend, session.model_size, session.confidence_threshold,
             session.output_dir, session.status),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_session_completed(
        self, session_id: int, report_html: str, results_json: str, output_dir: str,
    ) -> None:
        self.conn.execute(
            """UPDATE analysis_sessions
               SET status='completed', report_html=?, results_json=?, output_dir=?,
                   completed_at=CURRENT_TIMESTAMP
               WHERE id=?""",
            (report_html, results_json, output_dir, session_id),
        )
        self.conn.commit()

    def update_session_failed(self, session_id: int, error_message: str) -> None:
        self.conn.execute(
            """UPDATE analysis_sessions
               SET status='failed', error_message=?, completed_at=CURRENT_TIMESTAMP
               WHERE id=?""",
            (error_message, session_id),
        )
        self.conn.commit()

    def update_session_running(self, session_id: int) -> None:
        self.conn.execute(
            "UPDATE analysis_sessions SET status='running' WHERE id=?",
            (session_id,),
        )
        self.conn.commit()

    def get_sessions_for_player(self, player_id: int) -> list[AnalysisSession]:
        rows = self.conn.execute(
            "SELECT * FROM analysis_sessions WHERE player_id=? ORDER BY created_at DESC",
            (player_id,),
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def get_session(self, session_id: int) -> Optional[AnalysisSession]:
        row = self.conn.execute(
            "SELECT * FROM analysis_sessions WHERE id=?", (session_id,),
        ).fetchone()
        return self._row_to_session(row) if row else None

    def delete_session(self, session_id: int) -> None:
        self.conn.execute("DELETE FROM analysis_sessions WHERE id=?", (session_id,))
        self.conn.commit()

    # -- Helpers --

    @staticmethod
    def _row_to_player(row: sqlite3.Row) -> Player:
        return Player(
            id=row["id"], name=row["name"], throws=row["throws"],
            age=row["age"], height_inches=row["height_inches"],
            weight_lbs=row["weight_lbs"], team=row["team"] or "",
            notes=row["notes"] or "",
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> AnalysisSession:
        return AnalysisSession(
            id=row["id"], player_id=row["player_id"],
            video_path=row["video_path"], video_filename=row["video_filename"],
            backend=row["backend"], model_size=row["model_size"] or "m",
            confidence_threshold=row["confidence_threshold"] or 0.3,
            output_dir=row["output_dir"] or "",
            report_html=row["report_html"] or "",
            results_json=row["results_json"] or "",
            status=row["status"] or "pending",
            error_message=row["error_message"] or "",
            created_at=row["created_at"], completed_at=row["completed_at"],
        )
