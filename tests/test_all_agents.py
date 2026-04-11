"""
ARIA Test Suite
===============
Unit tests for all four agents.
Run: pytest tests/ -v --tb=short
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.drone_model import DroneState, DroneConfig, DronePhysics, MotorID, inject_motor_failure
from agents.llm_pilot import LLMPilot, FlightMode
from agents.ethical_guardrails import EthicalGuardrailEngine, GeoZone, SocialContext
from agents.digital_twin import DigitalTwin, MissionSegment, LiPoModel, WindEstimator


# ─────────────────────────────────────────────
# Core Physics Tests
# ─────────────────────────────────────────────

class TestDronePhysics:
    def test_hover(self):
        """Drone should maintain altitude at hover RPM."""
        cfg = DroneConfig()
        physics = DronePhysics(cfg)
        state = DroneState()
        state.position = np.array([0.0, 0.0, 5.0])
        hover_w = np.sqrt(cfg.mass * cfg.gravity / (4 * cfg.k_thrust))
        hover_rpm = hover_w * 60.0 / (2 * np.pi)
        state.motor_rpms = np.ones(4) * hover_rpm

        for _ in range(100):  # 2 seconds at 50Hz
            state = physics.step(state, np.ones(4) * hover_rpm, dt=0.02)

        assert abs(state.position[2] - 5.0) < 1.0, "Should maintain approx altitude"

    def test_motor_failure_injection(self):
        state = DroneState()
        state = inject_motor_failure(state, MotorID.FRONT_RIGHT, severity=1.0)
        assert state.motor_health[1] == 0.0

    def test_state_vector_shape(self):
        state = DroneState()
        vec = state.to_vector()
        assert vec.shape == (21,)

    def test_battery_drains(self):
        cfg = DroneConfig()
        physics = DronePhysics(cfg)
        state = DroneState()
        state.position = np.array([0.0, 0.0, 5.0])
        hover_w = np.sqrt(cfg.mass * cfg.gravity / (4 * cfg.k_thrust))
        hover_rpm = hover_w * 60.0 / (2 * np.pi)
        for _ in range(500):
            state = physics.step(state, np.ones(4) * hover_rpm, dt=0.02)
        assert state.battery_soc < 1.0

    def test_ground_collision_stops(self):
        physics = DronePhysics()
        state = DroneState()
        state.position = np.array([0.0, 0.0, 0.1])
        state.velocity = np.array([0.0, 0.0, -5.0])
        state = physics.step(state, np.zeros(4), dt=0.02)
        assert state.position[2] >= 0.0


# ─────────────────────────────────────────────
# LLM Pilot Tests
# ─────────────────────────────────────────────

class TestLLMPilot:
    def setup_method(self):
        self.pilot = LLMPilot()  # Mock mode

    def test_follow_command(self):
        plan = self.pilot.parse_command("Follow the red car")
        assert plan.mode == FlightMode.FOLLOW

    def test_stealth_follow_distance(self):
        plan = self.pilot.parse_command("Follow the blue truck but stay unnoticed")
        assert plan.follow_distance >= 25.0, "Stealth follow should maintain large distance"

    def test_return_home(self):
        plan = self.pilot.parse_command("Return home immediately")
        assert plan.mode == FlightMode.RETURN_HOME

    def test_search_generates_waypoints(self):
        plan = self.pilot.parse_command("Search the field for survivors")
        assert plan.mode == FlightMode.SEARCH
        assert len(plan.waypoints) > 0

    def test_land_command(self):
        plan = self.pilot.parse_command("Land now")
        assert plan.mode == FlightMode.LAND


# ─────────────────────────────────────────────
# Ethical Guardrails Tests
# ─────────────────────────────────────────────

class TestEthicalGuardrails:
    def setup_method(self):
        self.engine = EthicalGuardrailEngine()
        self.engine.register_static_zone(
            GeoZone(50, 50, 30, SocialContext.FUNERAL, "Test Funeral Zone")
        )

    def test_safe_position_allowed(self):
        pos = np.array([0.0, 0.0, 15.0])
        decision = self.engine.check_position(pos)
        assert decision.allowed

    def test_funeral_zone_blocked(self):
        pos = np.array([50.0, 50.0, 15.0])  # Inside funeral zone
        decision = self.engine.check_position(pos)
        assert not decision.allowed
        assert "FUNERAL" in decision.reason

    def test_reroute_computed_for_blocked(self):
        pos = np.array([50.0, 50.0, 15.0])
        decision = self.engine.check_position(pos)
        assert decision.suggested_reroute is not None

    def test_path_sanitization(self):
        path = [
            np.array([0.0, 0.0, 15.0]),
            np.array([50.0, 50.0, 15.0]),  # blocked
            np.array([100.0, 100.0, 15.0]),
        ]
        clean = self.engine.sanitize_path(path)
        assert len(clean) == 3
        # Middle waypoint should be rerouted (not the same as input)


# ─────────────────────────────────────────────
# Digital Twin / Energy Tests
# ─────────────────────────────────────────────

class TestDigitalTwin:
    def setup_method(self):
        self.twin = DigitalTwin()

    def test_hover_thrust(self):
        thrust = self.twin.hover_thrust()
        assert 9.0 < thrust < 12.0  # ~10.3 N for 1.05 kg

    def test_forecast_completes(self):
        segs = [MissionSegment(np.zeros(3), np.array([100, 0, 15]), speed=5.0)]
        forecast = self.twin.forecast(segs, current_soc=1.0)
        assert forecast.total_energy_wh > 0
        assert forecast.can_complete

    def test_low_battery_triggers_warning(self):
        segs = [MissionSegment(np.zeros(3), np.array([5000, 0, 15]), speed=5.0)]
        forecast = self.twin.forecast(segs, current_soc=0.05)
        assert not forecast.can_complete
        assert forecast.warning_message != ""

    def test_wind_estimator_updates(self):
        we = WindEstimator()
        cmd_vel = np.array([3.0, 0.0, 0.0])
        actual_vel = np.array([5.0, 1.0, 0.0])
        we.update(cmd_vel, actual_vel)
        assert we.wind_speed > 0

    def test_battery_model_ocv(self):
        bat = LiPoModel()
        ocv_full = bat.ocv(1.0)
        ocv_empty = bat.ocv(0.0)
        assert ocv_full > ocv_empty


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
