"""Run analysis on all pending pitches in the database."""

import argparse
import sys
import traceback
from pathlib import Path

# Allow running from repo root: python scripts/run_batch_analysis.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.desktop.models import Database, Pitch, PitchEvent, PitchMetric
from src.pipeline import PipelineConfig, PitchAnalysisPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis on pending pitches.")
    parser.add_argument("--player", type=str, default=None,
                        help="Filter by player name")
    parser.add_argument("--backend", type=str, default="yolov8",
                        help="Pose estimation backend (default: yolov8)")
    parser.add_argument("--model-size", type=str, default="m",
                        help="Model size (default: m)")
    return parser.parse_args()


def get_pending_pitches(db: Database, player_name: str | None = None) -> list[dict]:
    """Return pending pitches with player and session info."""
    results = []
    for player in db.get_all_players():
        if player_name and player.name != player_name:
            continue
        for session in db.get_sessions_for_player(player.id):
            for pitch in db.get_pitches_for_session(session.id):
                if pitch.status == "pending":
                    results.append({
                        "player": player,
                        "session": session,
                        "pitch": pitch,
                    })
    return results


def save_results(db: Database, pitch_id: int, result) -> None:
    """Save pipeline results to the database."""
    # Update pitch as completed
    db.update_pitch_completed(
        pitch_id,
        report_html=result.report_html,
        coaching_report=result.coaching_report,
        output_dir=str(result.output_dir) if result.output_dir else "",
    )

    # Save diagnostics
    video_info = result.pose_sequence.video_info
    frames_with_poses = len(result.pose_sequence.frames)
    avg_conf = 0.0
    if frames_with_poses > 0:
        total_conf = sum(
            sum(f.confidence.values()) / max(len(f.confidence), 1)
            for f in result.pose_sequence.frames
        )
        avg_conf = total_conf / frames_with_poses
    db.update_pitch_diagnostics(
        pitch_id,
        fps=video_info.fps,
        total_frames=video_info.total_frames,
        frames_with_poses=frames_with_poses,
        avg_confidence=avg_conf,
    )

    # Save events
    events = result.events
    durations = events.phase_durations()
    pe = PitchEvent(
        pitch_id=pitch_id,
        leg_lift_frame=events.leg_lift_apex,
        foot_plant_frame=events.foot_plant,
        max_er_frame=events.max_external_rotation,
        ball_release_frame=events.ball_release,
        max_ir_frame=events.max_internal_rotation,
        windup_to_fp=durations.get("windup_to_foot_plant"),
        fp_to_mer=durations.get("foot_plant_to_mer"),
        mer_to_release=durations.get("mer_to_release"),
        arm_cocking=durations.get("arm_cocking"),
        arm_acceleration=durations.get("arm_acceleration"),
        arm_deceleration=durations.get("arm_deceleration"),
    )
    db.add_pitch_event(pe)

    # Save metrics
    bench_lookup: dict[str, dict] = {}
    for bc in (result.benchmark_comparisons or []):
        bench_lookup[bc.get("metric", "")] = bc

    youth_lookup: dict[str, dict] = {}
    for yc in (result.youth_comparisons or []):
        if isinstance(yc, dict):
            youth_lookup[yc.get("metric", "")] = yc

    for attr in vars(result.metrics):
        val = getattr(result.metrics, attr)
        if val is None or attr.startswith("_") or not isinstance(val, (int, float)):
            continue

        bc = bench_lookup.get(attr, {})
        yc = youth_lookup.get(attr, {})

        pm = PitchMetric(
            pitch_id=pitch_id,
            metric_name=attr,
            display_name=bc.get("display_name", attr.replace("_", " ").title()),
            value=float(val),
            unit=bc.get("unit", "deg"),
            obp_median=bc.get("benchmark_median"),
            obp_percentile=bc.get("percentile_rank"),
            obp_flag=bc.get("flag"),
            youth_p25=yc.get("p25"),
            youth_p50=yc.get("p50"),
            youth_p75=yc.get("p75"),
            youth_percentile=yc.get("percentile"),
            youth_flag=yc.get("flag"),
        )
        db.add_pitch_metric(pm)


def main():
    args = parse_args()

    db = Database()
    db.initialize()

    try:
        pending = get_pending_pitches(db, args.player)
        if not pending:
            print("No pending pitches found.")
            return

        print(f"Found {len(pending)} pending pitch(es)\n")

        completed = 0
        failed = 0

        for item in pending:
            player = item["player"]
            session = item["session"]
            pitch = item["pitch"]
            video_path = Path(pitch.video_path)

            print(f"Analyzing {player.name} - pitch {pitch.pitch_number} "
                  f"({pitch.video_filename})...", end=" ", flush=True)

            if not video_path.exists():
                msg = f"Video not found: {video_path}"
                print(f"SKIP ({msg})")
                db.update_pitch_failed(pitch.id, msg)
                failed += 1
                continue

            # Get closest physical snapshot for youth mode
            snap = db.get_closest_snapshot(player.id, session.session_date)

            config = PipelineConfig(
                backend=args.backend,
                model_size=args.model_size,
                throws=player.throws,
            )
            if snap:
                config.age = int(snap.age_years) if snap.age_years else None
                config.height_inches = snap.height_inches or None
                config.weight_lbs = snap.weight_lbs or None

            try:
                db.update_pitch_running(pitch.id)
                pipeline = PitchAnalysisPipeline(config)
                result = pipeline.run(video_path)
                save_results(db, pitch.id, result)
                completed += 1
                print("OK")
            except Exception as e:
                db.update_pitch_failed(pitch.id, str(e))
                failed += 1
                print(f"FAILED: {e}")
                traceback.print_exc()

        total = completed + failed
        print(f"\nCompleted {completed}/{total} pitches"
              + (f" ({failed} failed)" if failed else ""))
    finally:
        db.close()


if __name__ == "__main__":
    main()
