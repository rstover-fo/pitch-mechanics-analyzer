"""Tests for pipeline validation and sanity checks."""

import pytest
from src.biomechanics.events import DeliveryEvents
from src.biomechanics.validation import validate_pipeline_output


class TestEventOrdering:
    def test_correct_ordering_no_warnings(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 50
        events.foot_plant = 70
        events.max_external_rotation = 75
        events.ball_release = 80
        warnings = validate_pipeline_output(events, avg_confidence=0.7)
        error_warnings = [w for w in warnings if w["severity"] == "error"]
        assert len(error_warnings) == 0

    def test_out_of_order_events(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 80
        events.foot_plant = 70
        events.max_external_rotation = 75
        events.ball_release = 90
        warnings = validate_pipeline_output(events, avg_confidence=0.7)
        codes = [w["code"] for w in warnings]
        assert "events_out_of_order" in codes

    def test_missing_event_flagged(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 50
        events.foot_plant = 70
        events.max_external_rotation = None
        events.ball_release = 80
        warnings = validate_pipeline_output(events, avg_confidence=0.7)
        codes = [w["code"] for w in warnings]
        assert "event_not_detected" in codes


class TestPhaseTiming:
    def test_plausible_timing_no_warnings(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 50
        events.foot_plant = 65
        events.max_external_rotation = 69
        events.ball_release = 72
        warnings = validate_pipeline_output(events, avg_confidence=0.7)
        timing_warnings = [w for w in warnings if "duration" in w["code"]]
        assert len(timing_warnings) == 0

    def test_implausible_stride_duration(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 10
        events.foot_plant = 100  # 3.0s stride — too long
        events.max_external_rotation = 105
        events.ball_release = 108
        warnings = validate_pipeline_output(events, avg_confidence=0.7)
        codes = [w["code"] for w in warnings]
        assert "stride_duration_implausible" in codes


class TestConfidence:
    def test_low_confidence_flagged(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 50
        events.foot_plant = 70
        events.max_external_rotation = 75
        events.ball_release = 80
        warnings = validate_pipeline_output(events, avg_confidence=0.15)
        codes = [w["code"] for w in warnings]
        assert "low_tracking_confidence" in codes

    def test_good_confidence_no_warning(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 50
        events.foot_plant = 70
        events.max_external_rotation = 75
        events.ball_release = 80
        warnings = validate_pipeline_output(events, avg_confidence=0.8)
        codes = [w["code"] for w in warnings]
        assert "low_tracking_confidence" not in codes
