"""SQLite database models for the Pitch Mechanics Analyzer desktop app.

Schema v2: players, physical_snapshots, sessions, pitches, pitch_events,
pitch_metrics, schema_version.  Includes migration from the v1 schema
(players + analysis_sessions).
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional


DB_PATH = Path("data/pitch_analyzer.db")

SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Player:
    id: Optional[int] = None
    name: str = ""
    throws: str = "R"
    team: str = ""
    notes: str = ""
    photo_path: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class PhysicalSnapshot:
    id: Optional[int] = None
    player_id: int = 0
    measured_date: str = ""          # ISO date string YYYY-MM-DD
    age_years: float = 0.0
    height_inches: float = 0.0
    weight_lbs: float = 0.0
    arm_length_inches: Optional[float] = None
    notes: str = ""
    created_at: Optional[str] = None


@dataclass
class Session:
    id: Optional[int] = None
    player_id: int = 0
    session_date: str = ""           # ISO date string YYYY-MM-DD
    location: str = ""
    session_type: str = "bullpen"
    notes: str = ""
    physical_snapshot_id: Optional[int] = None
    created_at: Optional[str] = None


@dataclass
class Pitch:
    id: Optional[int] = None
    session_id: int = 0
    pitch_number: Optional[int] = None
    pitch_type: str = "fastball"
    video_path: str = ""
    video_filename: str = ""
    backend: str = "yolov8"
    model_size: str = "m"
    confidence_threshold: float = 0.3
    pose_mode: str = "2d"
    status: str = "pending"
    error_message: str = ""
    output_dir: str = ""
    report_html: str = ""
    coaching_report: str = ""
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    frames_with_poses: Optional[int] = None
    avg_confidence: Optional[float] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class PitchEvent:
    pitch_id: int = 0
    leg_lift_frame: Optional[int] = None
    foot_plant_frame: Optional[int] = None
    max_er_frame: Optional[int] = None
    ball_release_frame: Optional[int] = None
    max_ir_frame: Optional[int] = None
    windup_to_fp: Optional[float] = None
    fp_to_mer: Optional[float] = None
    mer_to_release: Optional[float] = None
    arm_cocking: Optional[float] = None
    arm_acceleration: Optional[float] = None
    arm_deceleration: Optional[float] = None


@dataclass
class PitchMetric:
    id: Optional[int] = None
    pitch_id: int = 0
    metric_name: str = ""
    display_name: str = ""
    value: float = 0.0
    unit: str = "deg"
    obp_median: Optional[float] = None
    obp_percentile: Optional[float] = None
    obp_flag: Optional[str] = None
    youth_p25: Optional[float] = None
    youth_p50: Optional[float] = None
    youth_p75: Optional[float] = None
    youth_percentile: Optional[float] = None
    youth_flag: Optional[str] = None


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------

_NEW_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    throws TEXT NOT NULL DEFAULT 'R' CHECK (throws IN ('R', 'L')),
    team TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    photo_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS physical_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    measured_date DATE NOT NULL,
    age_years REAL NOT NULL,
    height_inches REAL NOT NULL,
    weight_lbs REAL NOT NULL,
    arm_length_inches REAL,
    notes TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id, measured_date)
);
CREATE INDEX IF NOT EXISTS idx_snapshots_player_date
    ON physical_snapshots(player_id, measured_date);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    location TEXT DEFAULT '',
    session_type TEXT DEFAULT 'bullpen'
        CHECK (session_type IN ('bullpen', 'game', 'lesson', 'warmup', 'other')),
    notes TEXT DEFAULT '',
    physical_snapshot_id INTEGER REFERENCES physical_snapshots(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sessions_player_date
    ON sessions(player_id, session_date DESC);

CREATE TABLE IF NOT EXISTS pitches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    pitch_number INTEGER,
    pitch_type TEXT DEFAULT 'fastball'
        CHECK (pitch_type IN ('fastball','curveball','changeup','slider','cutter','sinker','other')),
    video_path TEXT NOT NULL,
    video_filename TEXT NOT NULL,
    backend TEXT DEFAULT 'yolov8',
    model_size TEXT DEFAULT 'm',
    confidence_threshold REAL DEFAULT 0.3,
    pose_mode TEXT DEFAULT '2d',
    status TEXT DEFAULT 'pending'
        CHECK (status IN ('pending','running','completed','failed')),
    error_message TEXT DEFAULT '',
    output_dir TEXT DEFAULT '',
    report_html TEXT DEFAULT '',
    coaching_report TEXT DEFAULT '',
    fps REAL,
    total_frames INTEGER,
    frames_with_poses INTEGER,
    avg_confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pitches_session ON pitches(session_id, pitch_number);

CREATE TABLE IF NOT EXISTS pitch_events (
    pitch_id INTEGER PRIMARY KEY REFERENCES pitches(id) ON DELETE CASCADE,
    leg_lift_frame INTEGER,
    foot_plant_frame INTEGER,
    max_er_frame INTEGER,
    ball_release_frame INTEGER,
    max_ir_frame INTEGER,
    windup_to_fp REAL,
    fp_to_mer REAL,
    mer_to_release REAL,
    arm_cocking REAL,
    arm_acceleration REAL,
    arm_deceleration REAL
);

CREATE TABLE IF NOT EXISTS pitch_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pitch_id INTEGER NOT NULL REFERENCES pitches(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    display_name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT DEFAULT 'deg',
    obp_median REAL,
    obp_percentile REAL,
    obp_flag TEXT,
    youth_p25 REAL,
    youth_p50 REAL,
    youth_p75 REAL,
    youth_percentile REAL,
    youth_flag TEXT,
    UNIQUE(pitch_id, metric_name)
);
CREATE INDEX IF NOT EXISTS idx_metrics_pitch ON pitch_metrics(pitch_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON pitch_metrics(metric_name, value);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);
"""


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
        """Create tables if they don't exist, run migrations if needed."""
        if self._needs_migration():
            self._migrate_v1_to_v2()
        else:
            self.conn.executescript(_NEW_SCHEMA_SQL)
            self._ensure_schema_version()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Schema detection & migration
    # ------------------------------------------------------------------

    def _table_exists(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
        ).fetchone()
        return row is not None

    def _needs_migration(self) -> bool:
        """Return True if the old v1 schema is present (analysis_sessions table)."""
        return self._table_exists("analysis_sessions") and not self._table_exists("schema_version")

    def _ensure_schema_version(self) -> None:
        row = self.conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None or row["version"] < SCHEMA_VERSION:
            self.conn.execute(
                "INSERT OR REPLACE INTO schema_version (version, description) VALUES (?, ?)",
                (SCHEMA_VERSION, "6-table normalised schema"),
            )
            self.conn.commit()

    def _migrate_v1_to_v2(self) -> None:
        """Migrate from the old 2-table schema to the new 6-table schema."""
        c = self.conn

        # 1. Read all old data first
        old_players = c.execute("SELECT * FROM players").fetchall()
        old_sessions = c.execute("SELECT * FROM analysis_sessions").fetchall()

        # 2. Drop old tables
        c.execute("DROP TABLE IF EXISTS analysis_sessions")
        c.execute("DROP TABLE IF EXISTS players")

        # 3. Create new tables
        c.executescript(_NEW_SCHEMA_SQL)

        # 4. Migrate players → players + physical_snapshots
        for op in old_players:
            c.execute(
                """INSERT INTO players (id, name, throws, team, notes, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (op["id"], op["name"], op["throws"], op["team"] or "",
                 op["notes"] or "", op["created_at"], op["updated_at"]),
            )
            # If old player had age/height/weight, create a snapshot
            has_body = False
            try:
                has_body = op["age"] is not None or op["height_inches"] is not None or op["weight_lbs"] is not None
            except (IndexError, KeyError):
                pass
            if has_body:
                measured = op["created_at"][:10] if op["created_at"] else str(date.today())
                c.execute(
                    """INSERT INTO physical_snapshots
                       (player_id, measured_date, age_years, height_inches, weight_lbs)
                       VALUES (?, ?, ?, ?, ?)""",
                    (op["id"], measured,
                     op["age"] or 0,
                     op["height_inches"] or 0,
                     op["weight_lbs"] or 0),
                )

        # 5. Migrate analysis_sessions → sessions + pitches + pitch_events + pitch_metrics
        for os_ in old_sessions:
            session_date = os_["created_at"][:10] if os_["created_at"] else str(date.today())

            # Find closest snapshot for this player/date
            snap_row = c.execute(
                """SELECT id FROM physical_snapshots
                   WHERE player_id = ?
                   ORDER BY ABS(julianday(measured_date) - julianday(?))
                   LIMIT 1""",
                (os_["player_id"], session_date),
            ).fetchone()
            snap_id = snap_row["id"] if snap_row else None

            # Create session
            cur = c.execute(
                """INSERT INTO sessions (player_id, session_date, session_type, physical_snapshot_id, created_at)
                   VALUES (?, ?, 'bullpen', ?, ?)""",
                (os_["player_id"], session_date, snap_id, os_["created_at"]),
            )
            session_id = cur.lastrowid

            # Create pitch
            cur = c.execute(
                """INSERT INTO pitches
                   (session_id, pitch_number, video_path, video_filename, backend, model_size,
                    confidence_threshold, status, error_message, output_dir, report_html,
                    created_at, completed_at)
                   VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, os_["video_path"], os_["video_filename"],
                 os_["backend"] or "yolov8", os_["model_size"] or "m",
                 os_["confidence_threshold"] or 0.3,
                 os_["status"] or "pending", os_["error_message"] or "",
                 os_["output_dir"] or "", os_["report_html"] or "",
                 os_["created_at"], os_["completed_at"]),
            )
            pitch_id = cur.lastrowid

            # Parse results_json → pitch_metrics
            results_json = os_["results_json"] or ""
            if results_json:
                try:
                    metrics = json.loads(results_json)
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                c.execute(
                                    """INSERT OR IGNORE INTO pitch_metrics
                                       (pitch_id, metric_name, display_name, value, unit)
                                       VALUES (?, ?, ?, ?, 'deg')""",
                                    (pitch_id, metric_name,
                                     metric_name.replace("_", " ").title(),
                                     float(value)),
                                )
                except (json.JSONDecodeError, TypeError):
                    pass

        # 6. Record schema version
        c.execute(
            "INSERT OR REPLACE INTO schema_version (version, description) VALUES (?, ?)",
            (SCHEMA_VERSION, "Migrated from v1 (2-table) to v2 (6-table)"),
        )
        c.commit()

    # ------------------------------------------------------------------
    # Player CRUD
    # ------------------------------------------------------------------

    def add_player(self, player: Player) -> int:
        cur = self.conn.execute(
            """INSERT INTO players (name, throws, team, notes, photo_path)
               VALUES (?, ?, ?, ?, ?)""",
            (player.name, player.throws, player.team, player.notes, player.photo_path),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_player(self, player: Player) -> None:
        self.conn.execute(
            """UPDATE players SET name=?, throws=?, team=?, notes=?, photo_path=?,
               updated_at=CURRENT_TIMESTAMP WHERE id=?""",
            (player.name, player.throws, player.team, player.notes,
             player.photo_path, player.id),
        )
        self.conn.commit()

    def delete_player(self, player_id: int) -> None:
        self.conn.execute("DELETE FROM players WHERE id=?", (player_id,))
        self.conn.commit()

    def get_all_players(self) -> list[Player]:
        rows = self.conn.execute("SELECT * FROM players ORDER BY name").fetchall()
        return [self._row_to_player(r) for r in rows]

    def get_player(self, player_id: int) -> Optional[Player]:
        row = self.conn.execute("SELECT * FROM players WHERE id=?", (player_id,)).fetchone()
        return self._row_to_player(row) if row else None

    # ------------------------------------------------------------------
    # PhysicalSnapshot CRUD
    # ------------------------------------------------------------------

    def add_snapshot(self, snap: PhysicalSnapshot) -> int:
        cur = self.conn.execute(
            """INSERT INTO physical_snapshots
               (player_id, measured_date, age_years, height_inches, weight_lbs,
                arm_length_inches, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (snap.player_id, snap.measured_date, snap.age_years,
             snap.height_inches, snap.weight_lbs, snap.arm_length_inches, snap.notes),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_snapshot(self, snap: PhysicalSnapshot) -> None:
        self.conn.execute(
            """UPDATE physical_snapshots SET measured_date=?, age_years=?, height_inches=?,
               weight_lbs=?, arm_length_inches=?, notes=? WHERE id=?""",
            (snap.measured_date, snap.age_years, snap.height_inches,
             snap.weight_lbs, snap.arm_length_inches, snap.notes, snap.id),
        )
        self.conn.commit()

    def delete_snapshot(self, snapshot_id: int) -> None:
        self.conn.execute("DELETE FROM physical_snapshots WHERE id=?", (snapshot_id,))
        self.conn.commit()

    def get_snapshots_for_player(self, player_id: int) -> list[PhysicalSnapshot]:
        rows = self.conn.execute(
            "SELECT * FROM physical_snapshots WHERE player_id=? ORDER BY measured_date DESC",
            (player_id,),
        ).fetchall()
        return [self._row_to_snapshot(r) for r in rows]

    def get_snapshot(self, snapshot_id: int) -> Optional[PhysicalSnapshot]:
        row = self.conn.execute(
            "SELECT * FROM physical_snapshots WHERE id=?", (snapshot_id,),
        ).fetchone()
        return self._row_to_snapshot(row) if row else None

    def get_closest_snapshot(self, player_id: int, target_date: str) -> Optional[PhysicalSnapshot]:
        """Find the physical snapshot with the nearest measured_date to target_date."""
        row = self.conn.execute(
            """SELECT * FROM physical_snapshots
               WHERE player_id = ?
               ORDER BY ABS(julianday(measured_date) - julianday(?))
               LIMIT 1""",
            (player_id, target_date),
        ).fetchone()
        return self._row_to_snapshot(row) if row else None

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def add_session(self, session: Session) -> int:
        cur = self.conn.execute(
            """INSERT INTO sessions
               (player_id, session_date, location, session_type, notes, physical_snapshot_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session.player_id, session.session_date, session.location,
             session.session_type, session.notes, session.physical_snapshot_id),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_session(self, session: Session) -> None:
        self.conn.execute(
            """UPDATE sessions SET session_date=?, location=?, session_type=?, notes=?,
               physical_snapshot_id=? WHERE id=?""",
            (session.session_date, session.location, session.session_type,
             session.notes, session.physical_snapshot_id, session.id),
        )
        self.conn.commit()

    def delete_session(self, session_id: int) -> None:
        self.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        self.conn.commit()

    def get_sessions_for_player(self, player_id: int) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE player_id=? ORDER BY session_date DESC",
            (player_id,),
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def get_session(self, session_id: int) -> Optional[Session]:
        row = self.conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return self._row_to_session(row) if row else None

    # ------------------------------------------------------------------
    # Pitch CRUD
    # ------------------------------------------------------------------

    def add_pitch(self, pitch: Pitch) -> int:
        cur = self.conn.execute(
            """INSERT INTO pitches
               (session_id, pitch_number, pitch_type, video_path, video_filename,
                backend, model_size, confidence_threshold, pose_mode, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pitch.session_id, pitch.pitch_number, pitch.pitch_type,
             pitch.video_path, pitch.video_filename,
             pitch.backend, pitch.model_size, pitch.confidence_threshold,
             pitch.pose_mode, pitch.status),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_pitch_completed(
        self, pitch_id: int, report_html: str, coaching_report: str, output_dir: str,
    ) -> None:
        self.conn.execute(
            """UPDATE pitches
               SET status='completed', report_html=?, coaching_report=?, output_dir=?,
                   completed_at=CURRENT_TIMESTAMP
               WHERE id=?""",
            (report_html, coaching_report, output_dir, pitch_id),
        )
        self.conn.commit()

    def update_pitch_failed(self, pitch_id: int, error_message: str) -> None:
        self.conn.execute(
            """UPDATE pitches SET status='failed', error_message=?,
               completed_at=CURRENT_TIMESTAMP WHERE id=?""",
            (error_message, pitch_id),
        )
        self.conn.commit()

    def update_pitch_running(self, pitch_id: int) -> None:
        self.conn.execute("UPDATE pitches SET status='running' WHERE id=?", (pitch_id,))
        self.conn.commit()

    def update_pitch_diagnostics(
        self, pitch_id: int, fps: float, total_frames: int,
        frames_with_poses: int, avg_confidence: float,
    ) -> None:
        self.conn.execute(
            """UPDATE pitches SET fps=?, total_frames=?, frames_with_poses=?, avg_confidence=?
               WHERE id=?""",
            (fps, total_frames, frames_with_poses, avg_confidence, pitch_id),
        )
        self.conn.commit()

    def get_pitch(self, pitch_id: int) -> Optional[Pitch]:
        row = self.conn.execute("SELECT * FROM pitches WHERE id=?", (pitch_id,)).fetchone()
        return self._row_to_pitch(row) if row else None

    def get_pitches_for_session(self, session_id: int) -> list[Pitch]:
        rows = self.conn.execute(
            "SELECT * FROM pitches WHERE session_id=? ORDER BY pitch_number",
            (session_id,),
        ).fetchall()
        return [self._row_to_pitch(r) for r in rows]

    def delete_pitch(self, pitch_id: int) -> None:
        self.conn.execute("DELETE FROM pitches WHERE id=?", (pitch_id,))
        self.conn.commit()

    def get_next_pitch_number(self, session_id: int) -> int:
        row = self.conn.execute(
            "SELECT MAX(pitch_number) AS mx FROM pitches WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return (row["mx"] or 0) + 1

    # ------------------------------------------------------------------
    # PitchEvent CRUD
    # ------------------------------------------------------------------

    def add_pitch_event(self, event: PitchEvent) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO pitch_events
               (pitch_id, leg_lift_frame, foot_plant_frame, max_er_frame,
                ball_release_frame, max_ir_frame, windup_to_fp, fp_to_mer,
                mer_to_release, arm_cocking, arm_acceleration, arm_deceleration)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.pitch_id, event.leg_lift_frame, event.foot_plant_frame,
             event.max_er_frame, event.ball_release_frame, event.max_ir_frame,
             event.windup_to_fp, event.fp_to_mer, event.mer_to_release,
             event.arm_cocking, event.arm_acceleration, event.arm_deceleration),
        )
        self.conn.commit()

    def get_pitch_event(self, pitch_id: int) -> Optional[PitchEvent]:
        row = self.conn.execute(
            "SELECT * FROM pitch_events WHERE pitch_id=?", (pitch_id,),
        ).fetchone()
        return self._row_to_pitch_event(row) if row else None

    # ------------------------------------------------------------------
    # PitchMetric CRUD
    # ------------------------------------------------------------------

    def add_pitch_metric(self, metric: PitchMetric) -> int:
        cur = self.conn.execute(
            """INSERT OR REPLACE INTO pitch_metrics
               (pitch_id, metric_name, display_name, value, unit,
                obp_median, obp_percentile, obp_flag,
                youth_p25, youth_p50, youth_p75, youth_percentile, youth_flag)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (metric.pitch_id, metric.metric_name, metric.display_name,
             metric.value, metric.unit,
             metric.obp_median, metric.obp_percentile, metric.obp_flag,
             metric.youth_p25, metric.youth_p50, metric.youth_p75,
             metric.youth_percentile, metric.youth_flag),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_metrics_for_pitch(self, pitch_id: int) -> list[PitchMetric]:
        rows = self.conn.execute(
            "SELECT * FROM pitch_metrics WHERE pitch_id=? ORDER BY metric_name",
            (pitch_id,),
        ).fetchall()
        return [self._row_to_pitch_metric(r) for r in rows]

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_metric_trend(
        self, player_id: int, metric_name: str,
    ) -> list[tuple[str, str, float]]:
        """Return (session_date, pitch_type, value) tuples over time."""
        rows = self.conn.execute(
            """SELECT s.session_date, p.pitch_type, pm.value
               FROM pitch_metrics pm
               JOIN pitches p ON pm.pitch_id = p.id
               JOIN sessions s ON p.session_id = s.id
               WHERE s.player_id = ? AND pm.metric_name = ?
               ORDER BY s.session_date, p.pitch_number""",
            (player_id, metric_name),
        ).fetchall()
        return [(r["session_date"], r["pitch_type"], r["value"]) for r in rows]

    def get_pitch_type_averages(
        self, player_id: int, metric_name: str,
    ) -> list[tuple[str, float, int]]:
        """Return (pitch_type, avg_value, count) grouped by pitch type."""
        rows = self.conn.execute(
            """SELECT p.pitch_type, AVG(pm.value) AS avg_val, COUNT(*) AS n
               FROM pitch_metrics pm
               JOIN pitches p ON pm.pitch_id = p.id
               JOIN sessions s ON p.session_id = s.id
               WHERE s.player_id = ? AND pm.metric_name = ?
               GROUP BY p.pitch_type""",
            (player_id, metric_name),
        ).fetchall()
        return [(r["pitch_type"], r["avg_val"], r["n"]) for r in rows]

    def get_session_summary(self, session_id: int) -> list[dict]:
        """Return all pitches with their metrics for a session."""
        rows = self.conn.execute(
            """SELECT p.id AS pitch_id, p.pitch_number, p.pitch_type, p.status,
                      pm.metric_name, pm.display_name, pm.value, pm.obp_percentile,
                      pm.youth_percentile
               FROM pitches p
               LEFT JOIN pitch_metrics pm ON pm.pitch_id = p.id
               WHERE p.session_id = ?
               ORDER BY p.pitch_number, pm.metric_name""",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Row → dataclass converters
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_player(row: sqlite3.Row) -> Player:
        return Player(
            id=row["id"], name=row["name"], throws=row["throws"],
            team=row["team"] or "", notes=row["notes"] or "",
            photo_path=row["photo_path"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> PhysicalSnapshot:
        return PhysicalSnapshot(
            id=row["id"], player_id=row["player_id"],
            measured_date=row["measured_date"],
            age_years=row["age_years"], height_inches=row["height_inches"],
            weight_lbs=row["weight_lbs"],
            arm_length_inches=row["arm_length_inches"],
            notes=row["notes"] or "",
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"], player_id=row["player_id"],
            session_date=row["session_date"],
            location=row["location"] or "",
            session_type=row["session_type"] or "bullpen",
            notes=row["notes"] or "",
            physical_snapshot_id=row["physical_snapshot_id"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_pitch(row: sqlite3.Row) -> Pitch:
        return Pitch(
            id=row["id"], session_id=row["session_id"],
            pitch_number=row["pitch_number"],
            pitch_type=row["pitch_type"] or "fastball",
            video_path=row["video_path"], video_filename=row["video_filename"],
            backend=row["backend"] or "yolov8",
            model_size=row["model_size"] or "m",
            confidence_threshold=row["confidence_threshold"] or 0.3,
            pose_mode=row["pose_mode"] or "2d",
            status=row["status"] or "pending",
            error_message=row["error_message"] or "",
            output_dir=row["output_dir"] or "",
            report_html=row["report_html"] or "",
            coaching_report=row["coaching_report"] or "",
            fps=row["fps"], total_frames=row["total_frames"],
            frames_with_poses=row["frames_with_poses"],
            avg_confidence=row["avg_confidence"],
            created_at=row["created_at"], completed_at=row["completed_at"],
        )

    @staticmethod
    def _row_to_pitch_event(row: sqlite3.Row) -> PitchEvent:
        return PitchEvent(
            pitch_id=row["pitch_id"],
            leg_lift_frame=row["leg_lift_frame"],
            foot_plant_frame=row["foot_plant_frame"],
            max_er_frame=row["max_er_frame"],
            ball_release_frame=row["ball_release_frame"],
            max_ir_frame=row["max_ir_frame"],
            windup_to_fp=row["windup_to_fp"],
            fp_to_mer=row["fp_to_mer"],
            mer_to_release=row["mer_to_release"],
            arm_cocking=row["arm_cocking"],
            arm_acceleration=row["arm_acceleration"],
            arm_deceleration=row["arm_deceleration"],
        )

    @staticmethod
    def _row_to_pitch_metric(row: sqlite3.Row) -> PitchMetric:
        return PitchMetric(
            id=row["id"], pitch_id=row["pitch_id"],
            metric_name=row["metric_name"],
            display_name=row["display_name"],
            value=row["value"], unit=row["unit"] or "deg",
            obp_median=row["obp_median"],
            obp_percentile=row["obp_percentile"],
            obp_flag=row["obp_flag"],
            youth_p25=row["youth_p25"], youth_p50=row["youth_p50"],
            youth_p75=row["youth_p75"],
            youth_percentile=row["youth_percentile"],
            youth_flag=row["youth_flag"],
        )
