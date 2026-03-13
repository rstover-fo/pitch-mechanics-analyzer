"""Tests for the v2 data model and v1→v2 migration."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.desktop.models import (
    Database,
    PhysicalSnapshot,
    Pitch,
    PitchEvent,
    PitchMetric,
    Player,
    Session,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Fresh v2 database in a temp directory."""
    db = Database(db_path=tmp_path / "test.db")
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def player(db):
    """A saved player."""
    p = Player(name="Test Pitcher", throws="R", team="Rockets")
    p.id = db.add_player(p)
    return p


@pytest.fixture
def snapshot(db, player):
    """A saved physical snapshot."""
    s = PhysicalSnapshot(
        player_id=player.id,
        measured_date="2025-06-15",
        age_years=14.5,
        height_inches=68.0,
        weight_lbs=145.0,
        arm_length_inches=28.0,
    )
    s.id = db.add_snapshot(s)
    return s


@pytest.fixture
def session(db, player, snapshot):
    """A saved session linked to the player and snapshot."""
    s = Session(
        player_id=player.id,
        session_date="2025-06-15",
        location="Field A",
        session_type="bullpen",
        physical_snapshot_id=snapshot.id,
    )
    s.id = db.add_session(s)
    return s


@pytest.fixture
def pitch(db, session):
    """A saved pitch in the session."""
    p = Pitch(
        session_id=session.id,
        pitch_number=1,
        pitch_type="fastball",
        video_path="/videos/pitch1.mp4",
        video_filename="pitch1.mp4",
    )
    p.id = db.add_pitch(p)
    return p


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

class TestSchemaCreation:
    def test_all_tables_exist(self, db):
        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {
            "players", "physical_snapshots", "sessions", "pitches",
            "pitch_events", "pitch_metrics", "schema_version",
        }
        assert expected.issubset(tables)

    def test_schema_version_is_2(self, db):
        row = db.conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        assert row["version"] == 2


# ---------------------------------------------------------------------------
# Player CRUD
# ---------------------------------------------------------------------------

class TestPlayerCRUD:
    def test_add_and_get(self, db):
        p = Player(name="Alice", throws="L", team="Aces", notes="lefty")
        p.id = db.add_player(p)
        assert p.id is not None

        loaded = db.get_player(p.id)
        assert loaded.name == "Alice"
        assert loaded.throws == "L"
        assert loaded.team == "Aces"
        assert loaded.notes == "lefty"

    def test_update(self, db, player):
        player.name = "Updated Name"
        player.team = "New Team"
        db.update_player(player)
        loaded = db.get_player(player.id)
        assert loaded.name == "Updated Name"
        assert loaded.team == "New Team"

    def test_delete(self, db, player):
        db.delete_player(player.id)
        assert db.get_player(player.id) is None

    def test_get_all(self, db):
        db.add_player(Player(name="Bravo"))
        db.add_player(Player(name="Alpha"))
        players = db.get_all_players()
        names = [p.name for p in players]
        assert names == sorted(names)  # alphabetical order


# ---------------------------------------------------------------------------
# PhysicalSnapshot CRUD
# ---------------------------------------------------------------------------

class TestSnapshotCRUD:
    def test_add_and_get(self, db, player):
        snap = PhysicalSnapshot(
            player_id=player.id,
            measured_date="2025-01-01",
            age_years=13.0,
            height_inches=60.0,
            weight_lbs=120.0,
        )
        snap.id = db.add_snapshot(snap)
        loaded = db.get_snapshot(snap.id)
        assert loaded.age_years == 13.0
        assert loaded.height_inches == 60.0

    def test_get_for_player(self, db, player, snapshot):
        snaps = db.get_snapshots_for_player(player.id)
        assert len(snaps) == 1
        assert snaps[0].id == snapshot.id

    def test_closest_snapshot(self, db, player):
        db.add_snapshot(PhysicalSnapshot(
            player_id=player.id, measured_date="2025-01-01",
            age_years=13.0, height_inches=60.0, weight_lbs=120.0,
        ))
        db.add_snapshot(PhysicalSnapshot(
            player_id=player.id, measured_date="2025-06-01",
            age_years=13.5, height_inches=62.0, weight_lbs=130.0,
        ))
        db.add_snapshot(PhysicalSnapshot(
            player_id=player.id, measured_date="2025-12-01",
            age_years=14.0, height_inches=64.0, weight_lbs=140.0,
        ))
        closest = db.get_closest_snapshot(player.id, "2025-05-15")
        assert closest.measured_date == "2025-06-01"

    def test_delete(self, db, player, snapshot):
        db.delete_snapshot(snapshot.id)
        assert db.get_snapshot(snapshot.id) is None

    def test_cascade_on_player_delete(self, db, player, snapshot):
        db.delete_player(player.id)
        assert db.get_snapshot(snapshot.id) is None


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

class TestSessionCRUD:
    def test_add_and_get(self, db, session):
        loaded = db.get_session(session.id)
        assert loaded.location == "Field A"
        assert loaded.session_type == "bullpen"

    def test_get_for_player(self, db, player, session):
        sessions = db.get_sessions_for_player(player.id)
        assert len(sessions) == 1
        assert sessions[0].id == session.id

    def test_delete(self, db, session):
        db.delete_session(session.id)
        assert db.get_session(session.id) is None

    def test_cascade_on_player_delete(self, db, player, session):
        db.delete_player(player.id)
        assert db.get_session(session.id) is None


# ---------------------------------------------------------------------------
# Pitch CRUD
# ---------------------------------------------------------------------------

class TestPitchCRUD:
    def test_add_and_get(self, db, pitch):
        loaded = db.get_pitch(pitch.id)
        assert loaded.pitch_type == "fastball"
        assert loaded.pitch_number == 1
        assert loaded.status == "pending"

    def test_update_completed(self, db, pitch):
        db.update_pitch_completed(pitch.id, "<html>report</html>", "coaching text", "/out")
        loaded = db.get_pitch(pitch.id)
        assert loaded.status == "completed"
        assert loaded.report_html == "<html>report</html>"
        assert loaded.completed_at is not None

    def test_update_failed(self, db, pitch):
        db.update_pitch_failed(pitch.id, "Pose detection failed")
        loaded = db.get_pitch(pitch.id)
        assert loaded.status == "failed"
        assert loaded.error_message == "Pose detection failed"

    def test_update_running(self, db, pitch):
        db.update_pitch_running(pitch.id)
        loaded = db.get_pitch(pitch.id)
        assert loaded.status == "running"

    def test_get_for_session(self, db, session, pitch):
        pitches = db.get_pitches_for_session(session.id)
        assert len(pitches) == 1

    def test_next_pitch_number(self, db, session, pitch):
        assert db.get_next_pitch_number(session.id) == 2

    def test_next_pitch_number_empty_session(self, db, session):
        # session fixture doesn't create pitches directly; we need an empty one
        s2 = Session(
            player_id=session.player_id,
            session_date="2025-07-01",
            session_type="bullpen",
        )
        s2.id = db.add_session(s2)
        assert db.get_next_pitch_number(s2.id) == 1

    def test_delete(self, db, pitch):
        db.delete_pitch(pitch.id)
        assert db.get_pitch(pitch.id) is None

    def test_cascade_on_session_delete(self, db, session, pitch):
        db.delete_session(session.id)
        assert db.get_pitch(pitch.id) is None


# ---------------------------------------------------------------------------
# PitchEvent CRUD
# ---------------------------------------------------------------------------

class TestPitchEventCRUD:
    def test_add_and_get(self, db, pitch):
        event = PitchEvent(
            pitch_id=pitch.id,
            leg_lift_frame=10,
            foot_plant_frame=30,
            max_er_frame=40,
            ball_release_frame=45,
            max_ir_frame=55,
            windup_to_fp=0.5,
            fp_to_mer=0.2,
            mer_to_release=0.05,
        )
        db.add_pitch_event(event)
        loaded = db.get_pitch_event(pitch.id)
        assert loaded.leg_lift_frame == 10
        assert loaded.ball_release_frame == 45
        assert loaded.windup_to_fp == 0.5

    def test_replace_on_rerun(self, db, pitch):
        db.add_pitch_event(PitchEvent(pitch_id=pitch.id, leg_lift_frame=10))
        db.add_pitch_event(PitchEvent(pitch_id=pitch.id, leg_lift_frame=20))
        loaded = db.get_pitch_event(pitch.id)
        assert loaded.leg_lift_frame == 20  # replaced

    def test_cascade_on_pitch_delete(self, db, pitch):
        db.add_pitch_event(PitchEvent(pitch_id=pitch.id, leg_lift_frame=10))
        db.delete_pitch(pitch.id)
        assert db.get_pitch_event(pitch.id) is None


# ---------------------------------------------------------------------------
# PitchMetric CRUD
# ---------------------------------------------------------------------------

class TestPitchMetricCRUD:
    def test_add_and_get(self, db, pitch):
        m = PitchMetric(
            pitch_id=pitch.id,
            metric_name="peak_external_rotation",
            display_name="Peak External Rotation",
            value=178.5,
            unit="deg",
            obp_median=170.0,
            obp_percentile=85.0,
            obp_flag="elite",
        )
        m.id = db.add_pitch_metric(m)
        metrics = db.get_metrics_for_pitch(pitch.id)
        assert len(metrics) == 1
        assert metrics[0].value == 178.5
        assert metrics[0].obp_percentile == 85.0

    def test_unique_constraint_replaces(self, db, pitch):
        db.add_pitch_metric(PitchMetric(
            pitch_id=pitch.id, metric_name="stride_length",
            display_name="Stride Length", value=5.0,
        ))
        db.add_pitch_metric(PitchMetric(
            pitch_id=pitch.id, metric_name="stride_length",
            display_name="Stride Length", value=5.5,
        ))
        metrics = db.get_metrics_for_pitch(pitch.id)
        stride = [m for m in metrics if m.metric_name == "stride_length"]
        assert len(stride) == 1
        assert stride[0].value == 5.5

    def test_cascade_on_pitch_delete(self, db, pitch):
        db.add_pitch_metric(PitchMetric(
            pitch_id=pitch.id, metric_name="test_metric",
            display_name="Test", value=1.0,
        ))
        db.delete_pitch(pitch.id)
        assert db.get_metrics_for_pitch(pitch.id) == []


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

class TestQueryHelpers:
    def _setup_multi_pitch(self, db, player, snapshot):
        """Create 2 sessions with multiple pitches and metrics."""
        s1 = Session(
            player_id=player.id, session_date="2025-06-01",
            session_type="bullpen", physical_snapshot_id=snapshot.id,
        )
        s1.id = db.add_session(s1)
        s2 = Session(
            player_id=player.id, session_date="2025-07-01",
            session_type="game", physical_snapshot_id=snapshot.id,
        )
        s2.id = db.add_session(s2)

        p1 = Pitch(session_id=s1.id, pitch_number=1, pitch_type="fastball",
                    video_path="/v1.mp4", video_filename="v1.mp4", status="completed")
        p1.id = db.add_pitch(p1)
        p2 = Pitch(session_id=s1.id, pitch_number=2, pitch_type="curveball",
                    video_path="/v2.mp4", video_filename="v2.mp4", status="completed")
        p2.id = db.add_pitch(p2)
        p3 = Pitch(session_id=s2.id, pitch_number=1, pitch_type="fastball",
                    video_path="/v3.mp4", video_filename="v3.mp4", status="completed")
        p3.id = db.add_pitch(p3)

        for pid, val in [(p1.id, 170.0), (p2.id, 155.0), (p3.id, 175.0)]:
            db.add_pitch_metric(PitchMetric(
                pitch_id=pid, metric_name="peak_er",
                display_name="Peak ER", value=val, unit="deg",
            ))
        return s1, s2, p1, p2, p3

    def test_metric_trend(self, db, player, snapshot):
        self._setup_multi_pitch(db, player, snapshot)
        trend = db.get_metric_trend(player.id, "peak_er")
        assert len(trend) == 3
        dates = [t[0] for t in trend]
        assert dates == sorted(dates)

    def test_pitch_type_averages(self, db, player, snapshot):
        self._setup_multi_pitch(db, player, snapshot)
        avgs = db.get_pitch_type_averages(player.id, "peak_er")
        by_type = {a[0]: a for a in avgs}
        assert "fastball" in by_type
        assert "curveball" in by_type
        assert by_type["fastball"][2] == 2  # count
        assert by_type["curveball"][2] == 1

    def test_session_summary(self, db, player, snapshot):
        s1, _, _, _, _ = self._setup_multi_pitch(db, player, snapshot)
        summary = db.get_session_summary(s1.id)
        assert len(summary) == 2  # 2 pitches with metrics


# ---------------------------------------------------------------------------
# Migration: v1 → v2
# ---------------------------------------------------------------------------

class TestMigration:
    """Test migrating from the old 2-table schema to the new 6-table schema."""

    V1_SCHEMA = """\
    CREATE TABLE players (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        throws TEXT NOT NULL DEFAULT 'R',
        age REAL,
        height_inches REAL,
        weight_lbs REAL,
        team TEXT DEFAULT '',
        notes TEXT DEFAULT '',
        photo_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE analysis_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
        video_path TEXT NOT NULL,
        video_filename TEXT NOT NULL,
        backend TEXT DEFAULT 'yolov8',
        model_size TEXT DEFAULT 'm',
        confidence_threshold REAL DEFAULT 0.3,
        status TEXT DEFAULT 'pending',
        error_message TEXT DEFAULT '',
        output_dir TEXT DEFAULT '',
        report_html TEXT DEFAULT '',
        results_json TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP
    );
    """

    def _create_v1_db(self, tmp_path):
        """Create a v1 database with sample data and return its path."""
        db_path = tmp_path / "migrate_test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(self.V1_SCHEMA)

        # Insert players
        conn.execute(
            """INSERT INTO players (id, name, throws, age, height_inches, weight_lbs, team, created_at, updated_at)
               VALUES (1, 'Mike', 'R', 14.5, 68.0, 145.0, 'Tigers', '2025-03-01 10:00:00', '2025-03-01 10:00:00')"""
        )
        conn.execute(
            """INSERT INTO players (id, name, throws, age, height_inches, weight_lbs, team, created_at, updated_at)
               VALUES (2, 'Sam', 'L', NULL, NULL, NULL, 'Eagles', '2025-04-01 10:00:00', '2025-04-01 10:00:00')"""
        )

        # Insert analysis sessions
        metrics_data = json.dumps({
            "peak_external_rotation": 175.5,
            "stride_length": 5.2,
        })
        conn.execute(
            """INSERT INTO analysis_sessions
               (id, player_id, video_path, video_filename, backend, model_size,
                confidence_threshold, status, report_html, results_json,
                created_at, completed_at)
               VALUES (1, 1, '/vids/pitch1.mp4', 'pitch1.mp4', 'yolov8', 'm',
                       0.3, 'completed', '<html>report</html>', ?, '2025-03-15 12:00:00', '2025-03-15 12:05:00')""",
            (metrics_data,),
        )
        conn.execute(
            """INSERT INTO analysis_sessions
               (id, player_id, video_path, video_filename, status,
                error_message, created_at)
               VALUES (2, 1, '/vids/pitch2.mp4', 'pitch2.mp4', 'failed',
                       'No poses detected', '2025-03-15 12:10:00')"""
        )

        conn.commit()
        conn.close()
        return db_path

    def test_migration_creates_all_tables(self, tmp_path):
        db_path = self._create_v1_db(tmp_path)
        db = Database(db_path=db_path)
        db.initialize()

        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {
            "players", "physical_snapshots", "sessions", "pitches",
            "pitch_events", "pitch_metrics", "schema_version",
        }
        assert expected.issubset(tables)
        # Old table should be gone
        assert "analysis_sessions" not in tables
        db.close()

    def test_migration_preserves_players(self, tmp_path):
        db_path = self._create_v1_db(tmp_path)
        db = Database(db_path=db_path)
        db.initialize()

        players = db.get_all_players()
        assert len(players) == 2
        mike = db.get_player(1)
        assert mike.name == "Mike"
        assert mike.throws == "R"
        assert mike.team == "Tigers"
        db.close()

    def test_migration_creates_snapshots_for_body_data(self, tmp_path):
        db_path = self._create_v1_db(tmp_path)
        db = Database(db_path=db_path)
        db.initialize()

        # Mike had body data → should have a snapshot
        mike_snaps = db.get_snapshots_for_player(1)
        assert len(mike_snaps) == 1
        assert mike_snaps[0].age_years == 14.5
        assert mike_snaps[0].height_inches == 68.0
        assert mike_snaps[0].weight_lbs == 145.0

        # Sam had no body data → no snapshot
        sam_snaps = db.get_snapshots_for_player(2)
        assert len(sam_snaps) == 0
        db.close()

    def test_migration_creates_sessions_and_pitches(self, tmp_path):
        db_path = self._create_v1_db(tmp_path)
        db = Database(db_path=db_path)
        db.initialize()

        sessions = db.get_sessions_for_player(1)
        # Each old analysis_session creates a new session + pitch
        # But they're on the same date, so might be 2 sessions (one per old analysis_session)
        assert len(sessions) >= 1

        total_pitches = 0
        for s in sessions:
            pitches = db.get_pitches_for_session(s.id)
            total_pitches += len(pitches)
        assert total_pitches == 2  # 2 old analysis_sessions → 2 pitches
        db.close()

    def test_migration_preserves_pitch_status(self, tmp_path):
        db_path = self._create_v1_db(tmp_path)
        db = Database(db_path=db_path)
        db.initialize()

        # Find the completed pitch
        sessions = db.get_sessions_for_player(1)
        completed = []
        failed = []
        for s in sessions:
            for p in db.get_pitches_for_session(s.id):
                if p.status == "completed":
                    completed.append(p)
                elif p.status == "failed":
                    failed.append(p)

        assert len(completed) == 1
        assert completed[0].report_html == "<html>report</html>"

        assert len(failed) == 1
        assert failed[0].error_message == "No poses detected"
        db.close()

    def test_migration_parses_results_json_to_metrics(self, tmp_path):
        db_path = self._create_v1_db(tmp_path)
        db = Database(db_path=db_path)
        db.initialize()

        # Find the completed pitch and check its metrics
        sessions = db.get_sessions_for_player(1)
        for s in sessions:
            for p in db.get_pitches_for_session(s.id):
                if p.status == "completed":
                    metrics = db.get_metrics_for_pitch(p.id)
                    metric_names = {m.metric_name for m in metrics}
                    assert "peak_external_rotation" in metric_names
                    assert "stride_length" in metric_names

                    er = next(m for m in metrics if m.metric_name == "peak_external_rotation")
                    assert er.value == 175.5
                    db.close()
                    return

        pytest.fail("No completed pitch found after migration")

    def test_migration_sets_schema_version(self, tmp_path):
        db_path = self._create_v1_db(tmp_path)
        db = Database(db_path=db_path)
        db.initialize()

        row = db.conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        assert row["version"] == 2
        db.close()

    def test_migration_links_snapshot_to_session(self, tmp_path):
        db_path = self._create_v1_db(tmp_path)
        db = Database(db_path=db_path)
        db.initialize()

        sessions = db.get_sessions_for_player(1)
        # Mike's sessions should reference his snapshot
        for s in sessions:
            assert s.physical_snapshot_id is not None
            snap = db.get_snapshot(s.physical_snapshot_id)
            assert snap is not None
            assert snap.player_id == 1
        db.close()

    def test_no_migration_on_fresh_db(self, tmp_path):
        """A fresh database should not trigger migration."""
        db = Database(db_path=tmp_path / "fresh.db")
        db.initialize()
        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "analysis_sessions" not in tables
        assert "schema_version" in tables
        db.close()

    def test_idempotent_initialize(self, tmp_path):
        """Calling initialize() twice should be safe."""
        db = Database(db_path=tmp_path / "idem.db")
        db.initialize()
        db.initialize()  # should not raise
        assert db.get_all_players() == []
        db.close()
