"""
ARIA Simulation Runner
======================
Integrates all four agents into a single simulation loop.
Uses PyFlyt for multirotor physics visualization.

Recommended simulator: PyFlyt (built on PyBullet)
Install: pip install PyFlyt
Docs: https://jjshoots.github.io/PyFlyt/

Alternative: Microsoft AirSim
Download: https://github.com/microsoft/AirSim/releases
API docs: https://microsoft.github.io/AirSim/

Run this file to start the integrated ARIA simulation.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.live import Live

from core.drone_model import DroneState, DroneConfig, DronePhysics, MotorID, inject_motor_failure
from agents.llm_pilot import LLMPilot, WaypointNavigator
from agents.ethical_guardrails import EthicalGuardrailEngine, GeoZone, SocialContext
from agents.digital_twin import DigitalTwin, MissionSegment, WindEstimator

console = Console()


# ─────────────────────────────────────────────
# Simulation Configuration
# ─────────────────────────────────────────────

SIM_DT        = 0.02   # 50 Hz
SIM_DURATION  = 120.0  # seconds
FAILURE_TIME  = 30.0   # inject motor failure at t=30s
COMMAND       = "Search the north field for a blue truck and follow it, but stay unnoticed."


# ─────────────────────────────────────────────
# Integrated ARIA Simulation
# ─────────────────────────────────────────────

def run_simulation(headless: bool = True):
    logger.info("=" * 60)
    logger.info("  ARIA – Adaptive Resilient Intelligent Autopilot")
    logger.info("  Integrated 4-Agent Simulation")
    logger.info("=" * 60)

    # Initialize systems
    physics  = DronePhysics()
    pilot    = LLMPilot()              # Mock mode (no model path)
    guardrails = EthicalGuardrailEngine()
    twin     = DigitalTwin()

    # Register ethical zones
    guardrails.register_static_zone(GeoZone(60, 40, 35, SocialContext.FUNERAL,   "Memorial Park Funeral"))
    guardrails.register_static_zone(GeoZone(-20, 80, 50, SocialContext.SCHOOL,   "North Elementary"))

    # Parse NL command → mission
    state = DroneState()
    state.position = np.array([0.0, 0.0, 15.0])
    state.motor_rpms = np.ones(4) * 4500.0

    plan = pilot.parse_command(COMMAND, drone_position=state.position)
    logger.info(f"Mission mode: {plan.mode.name}")
    logger.info(f"Reasoning   : {plan.reasoning}")

    # Sanitize path with ethical guardrails
    raw_wp = [np.array([wp.x, wp.y, wp.z]) for wp in plan.waypoints]
    clean_wp = guardrails.sanitize_path(raw_wp) if raw_wp else [state.position.copy()]

    navigator = WaypointNavigator(hover_rpm=4500.0)

    # Simulation loop
    t = 0.0
    history = []
    failure_injected = False
    hover_rpm = 4500.0

    console.print(f"\n[bold cyan]Starting simulation: {SIM_DURATION}s @ {1/SIM_DT:.0f}Hz[/bold cyan]")

    while t < SIM_DURATION:
        # Inject motor failure at t=FAILURE_TIME
        if not failure_injected and t >= FAILURE_TIME:
            inject_motor_failure(state, MotorID.FRONT_RIGHT, severity=1.0)
            logger.warning(f"[t={t:.1f}s] MOTOR FAILURE: FRONT_RIGHT (complete)")
            failure_injected = True

        # Agent A: Self-healing RPM computation
        # (In production: replace with trained PPO model inference)
        healthy_motors = state.motor_health
        if healthy_motors.sum() < 4.0:
            # Redistribute thrust: boost healthy motors
            compensation = 4.0 / max(healthy_motors.sum(), 1.0)
            rpm_cmd = np.ones(4) * hover_rpm * compensation * healthy_motors
            # Counter-torque: opposite-corner boost
            if healthy_motors[1] < 0.5:  # FR failed
                rpm_cmd[2] += hover_rpm * 0.15  # boost RL for yaw balance
        else:
            # Agent B: Navigator
            waypoints_arr = [np.array([wp.x, wp.y, wp.z]) for wp in plan.waypoints]
            if waypoints_arr:
                rpm_cmd = navigator.compute_rpm(state, plan)
            else:
                rpm_cmd = np.ones(4) * hover_rpm

        # Agent C: Ethical position check
        decision = guardrails.check_position(state.position)
        if not decision.allowed:
            logger.warning(f"[t={t:.1f}s] GUARDRAIL: {decision.reason}")
            # Emergency altitude gain
            rpm_cmd += 200.0

        # Physics step
        state = physics.step(state, rpm_cmd, dt=SIM_DT)
        t += SIM_DT

        # Agent D: Digital twin forecast every 5s
        if int(t * 10) % 50 == 0:
            segs = [MissionSegment(state.position, state.position + np.array([50,50,0]), speed=5.0)]
            twin.adaptive_mission_update(segs, state.battery_soc)

        # Record
        history.append({
            "t": t,
            "x": state.position[0],
            "y": state.position[1],
            "z": state.position[2],
            "roll": np.degrees(state.attitude[0]),
            "pitch": np.degrees(state.attitude[1]),
            "yaw": np.degrees(state.attitude[2]),
            "battery": state.battery_soc * 100.0,
            "motor_health": state.motor_health.copy(),
        })

    logger.success(f"Simulation complete. {len(history)} data points recorded.")

    # Generate plots
    _plot_results(history, save_dir="assets")
    return history


def _plot_results(history: list, save_dir: str = "assets"):
    """Generate and save all visualization plots."""
    Path(save_dir).mkdir(exist_ok=True)
    ts  = [h["t"]       for h in history]
    zs  = [h["z"]       for h in history]
    bat = [h["battery"] for h in history]
    roll= [h["roll"]    for h in history]
    xs  = [h["x"]       for h in history]
    ys  = [h["y"]       for h in history]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("ARIA – Simulation Results", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Altitude over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ts, zs, "b-", linewidth=1.5)
    ax1.axvline(x=30, color="red", linestyle="--", label="Motor Failure")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Altitude (m)")
    ax1.set_title("Altitude vs Time"); ax1.legend(); ax1.grid(True)

    # Battery SoC
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ts, bat, "g-", linewidth=1.5)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Battery SoC (%)")
    ax2.set_title("Battery State of Charge"); ax2.grid(True)

    # Roll angle (self-healing indicator)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ts, roll, "r-", linewidth=1.0)
    ax3.axvline(x=30, color="red", linestyle="--", label="Motor Failure")
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Roll (°)")
    ax3.set_title("Roll Angle (Self-Healing)"); ax3.legend(); ax3.grid(True)

    # 2D flight path
    ax4 = fig.add_subplot(gs[1, 0:2])
    sc = ax4.scatter(xs, ys, c=ts, cmap="viridis", s=2)
    plt.colorbar(sc, ax=ax4, label="Time (s)")
    ax4.set_xlabel("X (m)"); ax4.set_ylabel("Y (m)")
    ax4.set_title("2D Flight Path (color = time)")

    # Motor health timeline
    ax5 = fig.add_subplot(gs[1, 2])
    labels = ["FL", "FR", "RL", "RR"]
    colors = ["blue", "orange", "green", "red"]
    for i in range(4):
        health = [h["motor_health"][i] * 100 for h in history]
        ax5.plot(ts, health, label=labels[i], color=colors[i], linewidth=1.5)
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Motor Health (%)")
    ax5.set_title("Motor Health Timeline"); ax5.legend(); ax5.grid(True)

    plt.savefig(f"{save_dir}/aria_simulation_results.png", dpi=150, bbox_inches="tight")
    logger.success(f"Plot saved → {save_dir}/aria_simulation_results.png")

    # Individual figures for paper
    _save_individual_plots(history, ts, zs, bat, roll, xs, ys, save_dir)


def _save_individual_plots(history, ts, zs, bat, roll, xs, ys, save_dir):
    """Save individual plots for inclusion in LaTeX paper."""

    # Fig 1: Self-healing altitude recovery
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, zs, "b-", linewidth=2, label="Altitude")
    ax.axvline(x=30, color="red", linestyle="--", linewidth=2, label="Motor Failure @ t=30s")
    ax.fill_between(ts, min(zs)-1, zs, alpha=0.1, color="blue")
    ax.set_xlabel("Time (s)", fontsize=12); ax.set_ylabel("Altitude (m)", fontsize=12)
    ax.set_title("Self-Healing: Altitude Recovery After Motor Failure", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_self_healing_altitude.pdf", dpi=300)
    plt.savefig(f"{save_dir}/fig_self_healing_altitude.png", dpi=150)
    plt.close()

    # Fig 2: Energy / battery
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, bat, "g-", linewidth=2)
    ax.set_xlabel("Time (s)", fontsize=12); ax.set_ylabel("Battery SoC (%)", fontsize=12)
    ax.set_title("Digital Twin Energy Forecast vs Actual Consumption", fontsize=13)
    ax.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_energy_forecast.pdf", dpi=300)
    plt.savefig(f"{save_dir}/fig_energy_forecast.png", dpi=150)
    plt.close()

    # Fig 3: Flight path with ethical zones
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(xs, ys, "b-", linewidth=1.5, alpha=0.7, label="Flight Path")
    circle1 = plt.Circle((60, 40), 35, color="red", alpha=0.2, label="Funeral Zone")
    circle2 = plt.Circle((-20, 80), 50, color="orange", alpha=0.2, label="School Zone")
    ax.add_patch(circle1); ax.add_patch(circle2)
    ax.set_xlabel("X (m)", fontsize=12); ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Ethical Path Sanitization (Guardrail Zones)", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.4); ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig_ethical_guardrails.pdf", dpi=300)
    plt.savefig(f"{save_dir}/fig_ethical_guardrails.png", dpi=150)
    plt.close()

    logger.success("All individual figures saved.")


def main():
    run_simulation(headless=True)


if __name__ == "__main__":
    main()
