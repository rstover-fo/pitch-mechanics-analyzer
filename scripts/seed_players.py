"""Seed the database with baseline player profiles and sessions.

Reads ground truth data from data/ground_truth/ and creates:
- 2 players (Jack Stover, Hank Stover)
- 1 session per player
- 2 pitches per session (one per video clip)
- pitch_events from hand-labeled ground truth

Idempotent: skips creation if players already exist (checked by name).
"""

import json
import sys
from datetime import date
from pathlib import Path

# Allow running from repo root: python scripts/seed_players.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.desktop.models import (
    Database,
    PhysicalSnapshot,
    Pitch,
    PitchEvent,
    Player,
    Session,
)

GROUND_TRUTH_DIR = Path("data/ground_truth")
DEFAULT_FPS = 240.0  # iPhone slo-mo


def load_ground_truth(clip_name: str) -> dict:
    """Load hand-labeled ground truth for a clip."""
    gt_path = GROUND_TRUTH_DIR / f"{clip_name}_ground_truth.json"
    if not gt_path.exists():
        print(f"  WARNING: ground truth not found: {gt_path}")
        return {}
    with open(gt_path) as f:
        return json.load(f)


def compute_phase_durations(events: dict, fps: float) -> dict:
    """Compute phase durations in seconds from frame indices."""
    leg_lift = events.get("leg_lift")
    foot_plant = events.get("foot_plant")
    max_er = events.get("max_er")
    ball_release = events.get("ball_release")

    def delta(start, end):
        if start is None or end is None:
            return None
        return (end - start) / fps

    return {
        "windup_to_fp": delta(leg_lift, foot_plant),
        "fp_to_mer": delta(foot_plant, max_er),
        "mer_to_release": delta(max_er, ball_release),
        "arm_cocking": delta(foot_plant, max_er),
        "arm_acceleration": delta(max_er, ball_release),
        "arm_deceleration": None,  # max_ir not in ground truth
    }


def seed(db: Database) -> None:
    """Create players, sessions, pitches, and pitch_events."""
    # Load pitcher definitions
    pitchers_path = GROUND_TRUTH_DIR / "pitchers.json"
    with open(pitchers_path) as f:
        pitchers = json.load(f)

    today = str(date.today())

    player_defs = [
        {
            "key": "player_a",
            "name": "Jack Stover",
            "notes": pitchers["player_a"]["description"],
            "throws": pitchers["player_a"]["throws"],
            "clips": pitchers["player_a"]["clips"],
        },
        {
            "key": "player_b",
            "name": "Hank Stover",
            "notes": pitchers["player_b"]["description"],
            "throws": pitchers["player_b"]["throws"],
            "clips": pitchers["player_b"]["clips"],
        },
    ]

    # Check for existing players
    existing = {p.name for p in db.get_all_players()}

    for pdef in player_defs:
        if pdef["name"] in existing:
            print(f"SKIP: {pdef['name']} already exists")
            continue

        # Create player
        player = Player(
            name=pdef["name"],
            throws=pdef["throws"],
            team="",
            notes=pdef["notes"],
        )
        player_id = db.add_player(player)
        print(f"Created player: {pdef['name']} (id={player_id})")

        # Create physical snapshot
        snapshot_id = None
        if pdef["key"] == "player_a":
            snapshot = PhysicalSnapshot(
                player_id=player_id,
                measured_date=today,
                age_years=10,
                height_inches=58,       # 4'10"
                weight_lbs=90,
                notes="Initial baseline",
            )
            snapshot_id = db.add_snapshot(snapshot)
            print(f"  Physical snapshot id={snapshot_id} (age=10, 4'10\", 90 lbs)")
        elif pdef["key"] == "player_b":
            snapshot = PhysicalSnapshot(
                player_id=player_id,
                measured_date=today,
                age_years=12,
                height_inches=59.5,     # 4'11.5"
                weight_lbs=78,
                notes="Initial baseline",
            )
            snapshot_id = db.add_snapshot(snapshot)
            print(f"  Physical snapshot id={snapshot_id} (age=12, 4'11.5\", 78 lbs)")

        # Create session
        session = Session(
            player_id=player_id,
            session_date=today,
            session_type="bullpen",
            location="Driveway",
            notes="Initial baseline session",
            physical_snapshot_id=snapshot_id,
        )
        session_id = db.add_session(session)
        print(f"  Session id={session_id} ({today}, bullpen)")

        # Create pitches
        for pitch_num, clip_name in enumerate(pdef["clips"], start=1):
            video_filename = f"{clip_name}.MOV"
            video_path = f"data/uploads/{video_filename}"

            pitch = Pitch(
                session_id=session_id,
                pitch_number=pitch_num,
                pitch_type="fastball",
                video_path=video_path,
                video_filename=video_filename,
                backend="yolov8",
                model_size="m",
                status="pending",
            )
            pitch_id = db.add_pitch(pitch)
            print(f"  Pitch {pitch_num}: {video_filename} (id={pitch_id})")

            # Load ground truth and create pitch_events
            gt = load_ground_truth(clip_name)
            if gt:
                events = gt.get("events", {})
                fps = gt.get("fps", DEFAULT_FPS)
                durations = compute_phase_durations(events, fps)

                pe = PitchEvent(
                    pitch_id=pitch_id,
                    leg_lift_frame=events.get("leg_lift"),
                    foot_plant_frame=events.get("foot_plant"),
                    max_er_frame=events.get("max_er"),
                    ball_release_frame=events.get("ball_release"),
                    windup_to_fp=durations["windup_to_fp"],
                    fp_to_mer=durations["fp_to_mer"],
                    mer_to_release=durations["mer_to_release"],
                    arm_cocking=durations["arm_cocking"],
                    arm_acceleration=durations["arm_acceleration"],
                    arm_deceleration=durations["arm_deceleration"],
                )
                db.add_pitch_event(pe)
                print(f"    Events: leg_lift={events.get('leg_lift')}, "
                      f"foot_plant={events.get('foot_plant')}, "
                      f"max_er={events.get('max_er')}, "
                      f"ball_release={events.get('ball_release')}")

    # Summary
    players = db.get_all_players()
    total_sessions = 0
    total_pitches = 0
    total_events = 0
    for p in players:
        sessions = db.get_sessions_for_player(p.id)
        total_sessions += len(sessions)
        for s in sessions:
            pitches = db.get_pitches_for_session(s.id)
            total_pitches += len(pitches)
            for pitch in pitches:
                if db.get_pitch_event(pitch.id):
                    total_events += 1

    print(f"\nDatabase summary:")
    print(f"  Players:  {len(players)}")
    print(f"  Sessions: {total_sessions}")
    print(f"  Pitches:  {total_pitches}")
    print(f"  Events:   {total_events}")


def main():
    db = Database()
    db.initialize()
    try:
        seed(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
