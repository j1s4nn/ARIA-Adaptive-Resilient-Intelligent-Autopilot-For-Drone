"""
ARIA Simulation Runner - Integrated 4-Agent Demonstration
==========================================================
Integrates all four ARIA agents into a single, fully autonomous mission
and automatically saves every output:

  output/figures/      14 PNG figures (2-4 per core motive) + PDF copies
  output/telemetry/    aria_telemetry.csv (full 50 Hz flight log)
  output/logs/         aria_simulation.log (complete run log)
  output/summary_report.md  human-readable mission summary

The four core motives demonstrated:
  Agent A - Self-Healing     : motor failure at t=45s, fault-tolerant
                               thrust re-allocation keeps the drone flying
  Agent B - LLM Pilot        : natural language command -> mission plan
                               -> waypoint navigation
  Agent C - Ethical Guardrails: funeral / school exclusion zones reroute
                               the planned path and block violations online
  Agent D - Digital Twin     : 60 s energy forecasting, Kalman wind
                               estimation, speed/altitude adaptation

Run it (from anywhere):
    python simulation/run_simulation.py
Optional arguments:
    --duration 150 --failure-time 45 --seed 7
"""

import sys
import csv
import shutil
import argparse
from pathlib import Path

# --- Make "python simulation/run_simulation.py" work without installation ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering -> figures saved as files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from loguru import logger
from rich.console import Console
from rich.table import Table

from core.drone_model import (DroneState, DroneConfig, DronePhysics,
                              MotorID, inject_motor_failure)
from agents.llm_pilot import LLMPilot, FlightMode
from agents.ethical_guardrails import EthicalGuardrailEngine, GeoZone, SocialContext
from agents.digital_twin import DigitalTwin, MissionSegment

console = Console()

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SIM_DT       = 0.02    # 50 Hz control loop
PHYS_DT      = 0.001   # 1 kHz physics integration (sub-stepped for stability)
FAILURE_MOTOR = MotorID.FRONT_RIGHT

COMMAND = ("Search the north field for a blue truck and follow it, "
           "but stay unnoticed.")

BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR    = OUTPUT_DIR / "figures"
LOG_DIR    = OUTPUT_DIR / "logs"
TEL_DIR    = OUTPUT_DIR / "telemetry"

MOTOR_LABELS = ["FL", "FR", "RL", "RR"]
MOTOR_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def setup_output_dirs() -> None:
    for d in (FIG_DIR, LOG_DIR, TEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Cascaded attitude controller with fault-tolerant control allocation
# ---------------------------------------------------------------------
class AttitudeController:
    """
    Position -> attitude -> motor mixer cascade.

    This controller stands in for Agent A's trained PPO policy at demo
    time. Its fault-tolerance mechanism mirrors what the RL agent learns:
    when a motor fails, the control-allocation mixer is rebuilt with only
    the healthy motors (least-squares), sacrificing yaw authority when the
    platform becomes under-actuated - exactly like a real quadrotor.
    """

    def __init__(self, cfg: DroneConfig):
        self.cfg = cfg
        self.kp_pos = np.array([2.2, 2.2, 4.5])
        self.kd_pos = np.array([2.8, 2.8, 4.5])
        # Attitude gains sized for the airframe inertia (bandwidth ~5 rad/s,
        # damping ratio ~0.9): kp = wn^2 * I, kd = 2*zeta*wn*I
        self.kp_att = np.array([0.18, 0.18, 0.30])
        self.kd_att = np.array([0.065, 0.065, 0.11])
        self.max_tilt = np.radians(20.0)
        self.max_acc_xy = 3.5   # matches max_tilt: g*tan(20 deg) ~= 3.57 m/s^2
        # Weighted wrench fit: prioritize collective thrust + roll/pitch,
        # sacrifice yaw first when a motor is lost.
        self.wrench_weights = np.array([1.0, 4.0, 4.0, 0.3])

    def _mixer_matrix(self) -> np.ndarray:
        """Rows: [thrust, roll, pitch, yaw]; columns: [FL, FR, RL, RR]."""
        L = self.cfg.arm_length
        kappa = self.cfg.k_drag / self.cfg.k_thrust
        return np.array([
            [1.0,    1.0,    1.0,    1.0   ],
            [L,     -L,      L,     -L     ],
            [L,      L,     -L,     -L     ],
            [kappa, -kappa, -kappa,  kappa ],  # diagonal spin pairs
        ])

    def compute_rpm(self, state: DroneState, target_pos: np.ndarray,
                    cruise_speed: float = 4.0, yaw_setpoint: float = 0.0) -> np.ndarray:
        cfg = self.cfg
        g = cfg.gravity

        # --- Outer loop: velocity-limited position control ---
        # Desired velocity points at the target and shrinks on approach
        # (smooth arrival, no overshoot limit cycles).
        err = np.asarray(target_pos, float) - state.position
        dist_xy = float(np.hypot(err[0], err[1]))
        v_des = np.zeros(3)
        if dist_xy > 1e-3:
            v_max_xy = min(cruise_speed, 0.2 + 0.8 * dist_xy)
            v_des[0] = err[0] / dist_xy * v_max_xy
            v_des[1] = err[1] / dist_xy * v_max_xy
        v_des[2] = np.clip(1.5 * err[2], -2.5, 2.5)

        acc_des = self.kd_pos * (v_des - state.velocity)
        acc_des[0] = np.clip(acc_des[0], -self.max_acc_xy, self.max_acc_xy)
        acc_des[1] = np.clip(acc_des[1], -self.max_acc_xy, self.max_acc_xy)
        acc_des[2] = np.clip(acc_des[2], -6.0, 6.0)

        # --- Desired tilt from horizontal acceleration (small-angle) ---
        psi = state.attitude[2]
        phi_des = (acc_des[0] * np.sin(psi) - acc_des[1] * np.cos(psi)) / g
        theta_des = (acc_des[0] * np.cos(psi) + acc_des[1] * np.sin(psi)) / g
        att_des = np.array([
            np.clip(phi_des, -self.max_tilt, self.max_tilt),
            np.clip(theta_des, -self.max_tilt, self.max_tilt),
            yaw_setpoint,
        ])

        # --- Collective thrust (tilt compensated) ---
        cos_tilt = max(np.cos(state.attitude[0]) * np.cos(state.attitude[1]), 0.5)
        collective = cfg.mass * (g + acc_des[2]) / cos_tilt

        # --- Inner loop: attitude -> torques ---
        att_err = att_des - state.attitude
        att_err[2] = np.arctan2(np.sin(att_err[2]), np.cos(att_err[2]))
        torque = self.kp_att * att_err - self.kd_att * state.angular_rate

        # --- Fault-tolerant control allocation ---
        health = state.motor_health
        M = self._mixer_matrix() * health[None, :]       # dead motors contribute nothing
        wrench = np.array([collective, torque[0], torque[1], torque[2]])
        W = np.diag(self.wrench_weights)
        if health.sum() > 3.99:
            forces = np.linalg.solve(self._mixer_matrix(), wrench)
        else:
            forces, *_ = np.linalg.lstsq(W @ M, W @ wrench, rcond=None)
            forces = forces * health
        max_force = cfg.k_thrust * (cfg.max_rpm * 2.0 * np.pi / 60.0) ** 2
        forces = np.clip(forces, 0.0, max_force)

        omega = np.sqrt(np.maximum(forces, 0.0) / cfg.k_thrust)
        rpm = omega * 60.0 / (2.0 * np.pi)
        return np.where(health > 1e-3, np.clip(rpm, 0.0, cfg.max_rpm), 0.0)


# ---------------------------------------------------------------------
# Wind field used by the simulation (unknown to the controller)
# ---------------------------------------------------------------------
def true_wind_at(t: float) -> np.ndarray:
    """Ambient wind: steady NE breeze plus slow gusts."""
    return np.array([
        2.0 + 0.8 * np.sin(2.0 * np.pi * t / 30.0),
        1.2 + 0.5 * np.sin(2.0 * np.pi * t / 23.0 + 1.3),
        0.0,
    ])


# ---------------------------------------------------------------------
# Integrated ARIA Simulation
# ---------------------------------------------------------------------
def run_simulation(duration: float = 150.0, failure_time: float = 45.0,
                   command: str = COMMAND, seed: int = 42):
    """Run the integrated 4-agent mission and save every output artifact."""
    setup_output_dirs()
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")
    logger.add(LOG_DIR / "aria_simulation.log", level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {message}")

    np.random.seed(seed)

    logger.info("=" * 64)
    logger.info("  ARIA - Adaptive Resilient Intelligent Autopilot")
    logger.info("  Integrated 4-Agent Simulation (headless, auto-saving outputs)")
    logger.info("=" * 64)

    cfg      = DroneConfig()
    physics  = DronePhysics(cfg)
    pilot    = LLMPilot()                    # mock mode unless a GGUF model is provided
    guardrails = EthicalGuardrailEngine()
    twin     = DigitalTwin()
    controller = AttitudeController(cfg)

    # ---- Agent C: register known social exclusion zones ----
    zones = [
        GeoZone(10, 20, 18, SocialContext.FUNERAL, "Memorial Park Funeral"),
        GeoZone(-20, 70, 45, SocialContext.SCHOOL, "North Elementary"),
    ]
    for z in zones:
        guardrails.register_static_zone(z)

    # ---- Agent B: natural language command -> mission plan ----
    state = DroneState()
    state.position = np.zeros(3)             # takeoff from the ground pad
    plan = pilot.parse_command(command, drone_position=state.position)
    logger.info(f"[Agent B] Mission mode : {plan.mode.name}")
    logger.info(f"[Agent B] Reasoning    : {plan.reasoning}")
    logger.info(f"[Agent B] Waypoints    : {len(plan.waypoints)}")

    raw_wp = [np.array([wp.x, wp.y, wp.z]) for wp in plan.waypoints]

    # ---- Agent C: sanitize the planned path before flight ----
    clean_wp = guardrails.sanitize_path(raw_wp) if raw_wp else [state.position.copy()]
    n_rerouted = sum(1 for a, b in zip(raw_wp, clean_wp) if np.linalg.norm(a - b) > 1e-6)
    logger.info(f"[Agent C] Path sanitized: {n_rerouted}/{len(raw_wp)} waypoints rerouted")

    # ---- Simulation loop ----
    t, failure_injected = 0.0, False
    wp_idx, cruise_speed = 0, 4.0
    yaw_hold = 0.0
    history, twin_log, guardrail_events = [], [], []
    last_guardrail_event_t = -10.0

    console.print(f"\n[bold cyan]Mission: {command}[/bold cyan]")
    console.print(f"[cyan]Simulating {duration:.0f}s @ {1/SIM_DT:.0f} Hz "
                  f"(motor failure scheduled at t={failure_time:.0f}s)...[/cyan]\n")

    while t < duration:
        # ---- Agent A: inject motor failure mid-mission ----
        if not failure_injected and t >= failure_time:
            inject_motor_failure(state, FAILURE_MOTOR, severity=1.0)
            logger.warning(f"[Agent A] t={t:.1f}s MOTOR FAILURE: "
                           f"{FAILURE_MOTOR.name} - re-allocating thrust to healthy motors")
            failure_injected = True

        # ---- Waypoint selection ----
        if wp_idx < len(clean_wp):
            target_pos = clean_wp[wp_idx]
            dist = float(np.linalg.norm(target_pos - state.position))
            if dist < 2.0:
                logger.info(f"[Agent B] Waypoint {wp_idx + 1}/{len(clean_wp)} reached "
                            f"at t={t:.1f}s")
                wp_idx = min(wp_idx + 1, len(clean_wp) - 1)
                target_pos = clean_wp[wp_idx]
                dist = float(np.linalg.norm(target_pos - state.position))
            direction = (target_pos - state.position) / max(dist, 1e-3)
            target_vel = direction * min(cruise_speed, 1.0 + 0.6 * dist)
        else:
            target_pos = state.position.copy()
            target_vel = np.zeros(3)

        # ---- Agent C: online position veto with reroute ----
        decision = guardrails.check_position(state.position)
        blocked = not decision.allowed
        if blocked:
            if decision.suggested_reroute is not None:
                target_pos = decision.suggested_reroute
                target_vel = np.zeros(3)
            if t - last_guardrail_event_t > 2.0:
                guardrail_events.append({
                    "t": t, "x": state.position[0], "y": state.position[1],
                    "z": state.position[2], "reason": decision.reason,
                })
                logger.warning(f"[Agent C] t={t:.1f}s GUARDRAIL VETO: {decision.reason}")
                last_guardrail_event_t = t

        # ---- Fly: controller -> physics (sub-stepped integration) ----
        wind_true = true_wind_at(t)
        rpm_cmd = controller.compute_rpm(state, target_pos, cruise_speed, yaw_hold)
        for _ in range(int(round(SIM_DT / PHYS_DT))):
            state = physics.step(state, rpm_cmd, dt=PHYS_DT, wind=wind_true)
        t += SIM_DT

        # ---- Agent D: wind estimation + 60 s energy forecast ----
        twin.wind.update(target_vel, state.velocity)
        if int(round(t * 10)) % 50 == 0 and wp_idx < len(clean_wp):
            segs = []
            segs.append(MissionSegment(state.position.copy(),
                                       clean_wp[wp_idx].copy(), speed=cruise_speed))
            for k in range(wp_idx, min(wp_idx + 3, len(clean_wp) - 1)):
                segs.append(MissionSegment(clean_wp[k].copy(),
                                           clean_wp[k + 1].copy(), speed=cruise_speed))
            forecast = twin.forecast(segs, state.battery_soc)
            cap_wh = twin.battery.capacity_ah * twin.battery.ocv(state.battery_soc)
            pred_soc_60 = max(0.0, state.battery_soc - forecast.total_energy_wh / cap_wh)
            twin_log.append({
                "t": t, "energy_wh": forecast.total_energy_wh,
                "remaining_wh": forecast.estimated_remaining_wh,
                "can_complete": forecast.can_complete,
                "rec_speed": forecast.recommended_speed,
                "rec_alt": forecast.recommended_altitude,
                "pred_soc_60s": pred_soc_60,
            })
            if not forecast.can_complete and cruise_speed > forecast.recommended_speed:
                cruise_speed = forecast.recommended_speed
                logger.warning(f"[Agent D] t={t:.1f}s Energy budget tight - "
                               f"reducing cruise speed to {cruise_speed:.1f} m/s")

        # ---- Telemetry record ----
        history.append({
            "t": t,
            "x": state.position[0], "y": state.position[1], "z": state.position[2],
            "vx": state.velocity[0], "vy": state.velocity[1], "vz": state.velocity[2],
            "roll": np.degrees(state.attitude[0]),
            "pitch": np.degrees(state.attitude[1]),
            "yaw": np.degrees(state.attitude[2]),
            "battery": state.battery_soc * 100.0,
            "health": state.motor_health.copy(),
            "rpms": state.motor_rpms.copy(),
            "tx": target_pos[0], "ty": target_pos[1], "tz": target_pos[2],
            "wp_idx": wp_idx, "cruise": cruise_speed, "blocked": blocked,
            "wind_true": wind_true.copy(),
            "wind_est": twin.wind.wind_vector.copy(),
        })

    logger.success(f"Simulation complete: {len(history)} telemetry samples recorded.")
    results = {
        "history": history, "twin_log": twin_log,
        "guardrail_events": guardrail_events,
        "plan": plan, "command": command,
        "raw_wp": raw_wp, "clean_wp": clean_wp, "zones": zones,
        "duration": duration, "failure_time": failure_time,
        "failure_motor": FAILURE_MOTOR, "n_rerouted": n_rerouted,
    }
    _save_telemetry_csv(history, TEL_DIR / "aria_telemetry.csv")
    _generate_all_figures(results)
    _write_summary_report(results)
    _print_console_summary(results)
    return results


# ---------------------------------------------------------------------
# Telemetry CSV
# ---------------------------------------------------------------------
def _save_telemetry_csv(history: list, path: Path) -> None:
    cols = ["t", "x", "y", "z", "vx", "vy", "vz", "roll_deg", "pitch_deg",
            "yaw_deg", "battery_pct",
            "health_FL", "health_FR", "health_RL", "health_RR",
            "rpm_FL", "rpm_FR", "rpm_RL", "rpm_RR",
            "target_x", "target_y", "target_z", "wp_idx", "cruise_speed",
            "guardrail_blocked",
            "wind_true_x", "wind_true_y", "wind_true_z",
            "wind_est_x", "wind_est_y", "wind_est_z"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for h in history:
            w.writerow([
                f"{h['t']:.3f}", f"{h['x']:.4f}", f"{h['y']:.4f}", f"{h['z']:.4f}",
                f"{h['vx']:.4f}", f"{h['vy']:.4f}", f"{h['vz']:.4f}",
                f"{h['roll']:.4f}", f"{h['pitch']:.4f}", f"{h['yaw']:.4f}",
                f"{h['battery']:.4f}",
                *[f"{v:.3f}" for v in h["health"]],
                *[f"{v:.1f}" for v in h["rpms"]],
                f"{h['tx']:.3f}", f"{h['ty']:.3f}", f"{h['tz']:.3f}",
                h["wp_idx"], f"{h['cruise']:.2f}", int(h["blocked"]),
                *[f"{v:.4f}" for v in h["wind_true"]],
                *[f"{v:.4f}" for v in h["wind_est"]],
            ])
    logger.success(f"Telemetry CSV saved -> {path}")


# ---------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------
def _col(history: list, key: str) -> np.ndarray:
    return np.array([h[key] for h in history])


def _fig(path_stem: str, fig) -> Path:
    png = FIG_DIR / f"{path_stem}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    logger.success(f"Figure saved -> {png.name}")
    return png


def _mark_failure(ax, t_fail: float):
    ax.axvline(x=t_fail, color="red", linestyle="--", linewidth=1.4, alpha=0.8,
               label=f"Motor failure @ t={t_fail:.0f}s")


# ---------------------------------------------------------------------
# Fig 00 - System architecture diagram
# ---------------------------------------------------------------------
def _fig_architecture(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("ARIA System Architecture - Four Cooperative AI Agents",
                 fontsize=15, fontweight="bold", pad=16)

    def box(x, y, w, h, text, color, fs=10):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           linewidth=1.6, edgecolor=color, facecolor=color + "22")
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="#222222")
        return (x, y, w, h)

    def arrow(p1, p2, color="#444444", style="-|>", lw=1.8, connectionstyle="arc3,rad=0"):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                                     color=color, linewidth=lw,
                                     connectionstyle=connectionstyle))

    box(0.3, 6.4, 2.6, 1.0, "Operator\nNatural Language\nCommand", "#9467bd")
    box(4.2, 6.4, 3.0, 1.0, "Agent B - LLM Pilot\n(local Mistral-7B / rule fallback)\nNL -> MissionPlan JSON", "#1f77b4")
    box(9.0, 6.4, 3.4, 1.0, "Agent C - Ethical Guardrails\nCLIP scene classifier\n+ social geo-zone database", "#d62728")
    box(4.2, 3.6, 3.0, 1.1, "Flight Controller\nposition / attitude cascade\nfault-tolerant mixer", "#17becf")
    box(0.3, 3.6, 2.6, 1.1, "Agent A - Self-Healing\nPPO policy re-allocates\nthrust after motor loss", "#2ca02c")
    box(9.0, 3.6, 3.4, 1.1, "Agent D - Digital Twin\n60 s energy forecast\nKalman wind estimator", "#ff7f0e")
    box(4.2, 0.9, 3.0, 1.1, "Quadrotor Physics\n6-DOF dynamics, motors\nbattery, wind field", "#7f7f7f")
    box(0.3, 0.9, 2.6, 1.1, "Motor Failure\nInjection\n(t = 45 s)", "#8c564b")
    box(9.0, 0.9, 3.4, 1.1, "Outputs\nfigures / telemetry CSV\nlogs / report", "#bcbd22")

    arrow((2.9, 6.9), (4.2, 6.9))
    arrow((7.2, 6.9), (9.0, 6.9))
    arrow((10.7, 6.4), (5.7, 4.7), color="#d62728", connectionstyle="arc3,rad=-0.25")
    ax.text(8.6, 5.9, "sanitized path", fontsize=8.5, color="#d62728", style="italic")
    arrow((5.7, 6.4), (5.7, 4.7), color="#1f77b4")
    ax.text(5.85, 5.6, "mission plan", fontsize=8.5, color="#1f77b4", style="italic")
    arrow((2.9, 4.15), (4.2, 4.15), color="#2ca02c")
    ax.text(3.0, 4.35, "re-allocation", fontsize=8.5, color="#2ca02c", style="italic")
    arrow((5.7, 3.6), (5.7, 2.0))
    ax.text(5.85, 2.7, "RPM commands", fontsize=8.5, style="italic")
    arrow((2.9, 1.45), (4.2, 1.45), color="#8c564b")
    arrow((7.2, 1.45), (9.0, 1.45), color="#7f7f7f")
    arrow((10.7, 3.6), (6.6, 2.0), color="#ff7f0e", connectionstyle="arc3,rad=0.2")
    ax.text(9.2, 2.7, "speed / altitude advice", fontsize=8.5, color="#ff7f0e", style="italic")
    arrow((7.2, 1.2), (9.0, 1.2), color="#7f7f7f")
    ax.text(7.5, 0.95, "telemetry", fontsize=8.5, style="italic")

    fig.tight_layout()
    _fig("00_system_architecture", fig)
    shutil.copyfile(FIG_DIR / "00_system_architecture.png",
                    FIG_DIR / "fig_architecture.png")


# ---------------------------------------------------------------------
# Fig 01 - Agent A: altitude recovery after motor failure
# ---------------------------------------------------------------------
def _fig_agentA_altitude(results: dict) -> None:
    h = results["history"]
    ts, zs = _col(h, "t"), _col(h, "z")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ts, zs, color="#2ca02c", linewidth=2.0, label="Altitude")
    _mark_failure(ax, results["failure_time"])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude (m)")
    ax.set_title("Agent A - Self-Healing: Altitude Recovery After Motor Failure")
    ax.legend(loc="best"); fig.tight_layout()
    _fig("01_agentA_altitude_recovery", fig)
    shutil.copyfile(FIG_DIR / "01_agentA_altitude_recovery.png",
                    FIG_DIR / "fig_self_healing_altitude.png")
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    ax2.plot(ts, zs, color="#2ca02c", linewidth=2.0)
    ax2.axvline(x=results["failure_time"], color="red", linestyle="--", linewidth=1.4)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Altitude (m)")
    ax2.set_title("Agent A - Self-Healing: Altitude Recovery After Motor Failure")
    ax2.grid(True, alpha=0.35); fig2.tight_layout()
    fig2.savefig(FIG_DIR / "fig_self_healing_altitude.pdf", dpi=300)
    plt.close(fig2)


# ---------------------------------------------------------------------
# Fig 02 - Agent A: attitude stabilization
# ---------------------------------------------------------------------
def _fig_agentA_attitude(results: dict) -> None:
    h = results["history"]
    ts = _col(h, "t")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, key, name, color in zip(
            axes, ("roll", "pitch", "yaw"),
            ("Roll", "Pitch", "Yaw"), ("#d62728", "#1f77b4", "#9467bd")):
        ax.plot(ts, _col(h, key), color=color, linewidth=1.2)
        _mark_failure(ax, results["failure_time"])
        ax.set_ylabel(f"{name} (deg)")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title("Agent A - Attitude Stabilization During / After Motor Failure")
    axes[2].set_xlabel("Time (s)")
    fig.tight_layout(); _fig("02_agentA_attitude_stabilization", fig)


# ---------------------------------------------------------------------
# Fig 03 - Agent A: motor RPM redistribution
# ---------------------------------------------------------------------
def _fig_agentA_motors(results: dict) -> None:
    h = results["history"]
    ts = _col(h, "t")
    rpms = np.array([x["rpms"] for x in h])
    health = np.array([x["health"] for x in h])
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for i in range(4):
        axes[0].plot(ts, rpms[:, i], color=MOTOR_COLORS[i], linewidth=1.4,
                     label=f"{MOTOR_LABELS[i]}")
    _mark_failure(axes[0], results["failure_time"])
    axes[0].set_ylabel("Motor RPM")
    axes[0].set_title("Agent A - Fault-Tolerant Thrust Redistribution (RPM per Motor)")
    axes[0].legend(ncol=4, fontsize=9)
    for i in range(4):
        axes[1].plot(ts, health[:, i] * 100.0, color=MOTOR_COLORS[i],
                     linewidth=2.0, label=f"{MOTOR_LABELS[i]}")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Motor Health (%)")
    axes[1].set_ylim(-5, 105); axes[1].legend(ncol=4, fontsize=9)
    fig.tight_layout(); _fig("03_agentA_motor_rpm_redistribution", fig)


# ---------------------------------------------------------------------
# Fig 04 - Agent B: mission plan parsed from natural language
# ---------------------------------------------------------------------
def _fig_agentB_plan(results: dict) -> None:
    plan = results["plan"]
    fig = plt.figure(figsize=(11, 5.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.25)

    ax_txt = fig.add_subplot(gs[0, 0]); ax_txt.axis("off")
    ax_txt.set_title("Agent B - Natural Language Command Parsing",
                     fontsize=12, fontweight="bold")
    lines = [
        ("Operator command", f'"{results["command"]}"'),
        ("Parsed mode", plan.mode.name),
        ("Target", plan.target_description or "-"),
        ("Follow distance", f"{plan.follow_distance:.0f} m "
                            + ("(stealth)" if plan.follow_distance >= 25 else "(normal)")),
        ("Waypoints generated", str(len(plan.waypoints))),
        ("Reasoning", plan.reasoning),
        ("Parser backend", "local LLM (GGUF)" if getattr(plan, "_llm", False)
                            else "rule-based fallback (no model installed)"),
    ]
    y = 0.92
    for label, value in lines:
        ax_txt.text(0.02, y, f"{label}:", fontsize=10.5, fontweight="bold",
                    va="top")
        ax_txt.text(0.38, y, str(value), fontsize=10, va="top",
                    wrap=True, color="#333333")
        y -= 0.13

    ax_map = fig.add_subplot(gs[0, 1])
    for z in results["zones"]:
        ax_map.add_patch(Circle((z.center_x, z.center_y), z.radius,
                                color="#d62728", alpha=0.15))
        ax_map.annotate(z.label, (z.center_x, z.center_y), fontsize=8,
                        ha="center", color="#a02020")
    wps = np.array([[wp.x, wp.y] for wp in plan.waypoints]) if plan.waypoints else None
    if wps is not None and len(wps):
        ax_map.plot(wps[:, 0], wps[:, 1], "o-", color="#1f77b4",
                    markersize=5, linewidth=1.4, label="Generated search grid")
        ax_map.plot(wps[0, 0], wps[0, 1], "s", color="#2ca02c", markersize=9,
                    label="Entry waypoint")
    ax_map.plot([0], [0], "^", color="black", markersize=10, label="Launch pad")
    ax_map.set_xlabel("X (m)"); ax_map.set_ylabel("Y (m)")
    ax_map.set_title("Generated Mission Waypoints")
    ax_map.legend(fontsize=8, loc="lower left")
    ax_map.set_aspect("equal")
    _fig("04_agentB_mission_plan", fig)


# ---------------------------------------------------------------------
# Fig 05 - Agent B: NL -> MissionPlan pipeline diagram
# ---------------------------------------------------------------------
def _fig_agentB_pipeline(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.set_title("Agent B - LLM Pilot Processing Pipeline",
                 fontsize=14, fontweight="bold", pad=12)
    stages = [
        ("Natural language\ncommand", "#9467bd"),
        ("Prompt build\n(drone state + rules)", "#1f77b4"),
        ("Local LLM / rule\nfallback parser", "#17becf"),
        ("JSON MissionPlan\nmode + waypoints", "#2ca02c"),
        ("Waypoint\nnavigator", "#ff7f0e"),
        ("Motor commands\n(via controller)", "#d62728"),
    ]
    w, h_, gap = 1.85, 1.5, 0.32
    x = 0.25
    for i, (label, color) in enumerate(stages):
        b = FancyBboxPatch((x, 1.3), w, h_, boxstyle="round,pad=0.07",
                           linewidth=1.6, edgecolor=color, facecolor=color + "22")
        ax.add_patch(b)
        ax.text(x + w / 2, 1.3 + h_ / 2, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, 1.05, f"step {i + 1}", ha="center",
                fontsize=8, color="#666666")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, 1.3 + h_ / 2),
                                         (x + w + gap - 0.03, 1.3 + h_ / 2),
                                         arrowstyle="-|>", mutation_scale=14,
                                         color="#444444", linewidth=1.6))
        x += w + gap
    ax.text(6.5, 0.45, "Safety constraints embedded in the system prompt: minimum altitude, "
            "stealth follow distance >= 25 m, conservative speeds.",
            ha="center", fontsize=9, style="italic", color="#555555")
    _fig("05_agentB_llm_pipeline", fig)
    shutil.copyfile(FIG_DIR / "05_agentB_llm_pipeline.png",
                    FIG_DIR / "fig_llm_pipeline.png")


# ---------------------------------------------------------------------
# Fig 06 - Agent B: planned vs actual trajectory
# ---------------------------------------------------------------------
def _fig_agentB_tracking(results: dict) -> None:
    h = results["history"]
    xs, ys = _col(h, "x"), _col(h, "y")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    ax = axes[0]
    cw = np.array(results["clean_wp"])
    if len(cw):
        ax.plot(cw[:, 0], cw[:, 1], "--", color="#999999", linewidth=1.5,
                label="Sanitized plan")
        ax.plot(cw[:, 0], cw[:, 1], "o", color="#1f77b4", markersize=6,
                markerfacecolor="white", markeredgewidth=1.5)
    sc = ax.scatter(xs, ys, c=_col(h, "t"), cmap="viridis", s=5)
    fig.colorbar(sc, ax=ax, label="Time (s)")
    ax.plot([0], [0], "^", color="black", markersize=10, label="Launch pad")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Planned Waypoints vs Flown Trajectory")
    ax.legend(fontsize=8); ax.set_aspect("equal")

    ax = axes[1]
    ts = _col(h, "t")
    wp_idx = _col(h, "wp_idx")
    ax.step(ts, wp_idx + 1, where="post", color="#ff7f0e", linewidth=2.0)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Waypoint index")
    ax.set_title("Waypoint Progression Over Time")
    ax.set_ylim(0, max(len(results["clean_wp"]), 1) + 1)
    fig.suptitle("Agent B - LLM Pilot: Waypoint Navigation Performance",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); _fig("06_agentB_trajectory_tracking", fig)


# ---------------------------------------------------------------------
# Fig 07 - Agent C: ethical zone map with sanitized path
# ---------------------------------------------------------------------
def _fig_agentC_map(results: dict) -> None:
    h = results["history"]
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    zone_colors = {SocialContext.FUNERAL: "#d62728", SocialContext.SCHOOL: "#ff7f0e"}
    for z in results["zones"]:
        c = zone_colors.get(z.context, "#d62728")
        ax.add_patch(Circle((z.center_x, z.center_y), z.radius, color=c, alpha=0.18))
        ax.add_patch(Circle((z.center_x, z.center_y), z.radius, fill=False,
                            edgecolor=c, linewidth=2.0, linestyle="--"))
        ax.annotate(f"{z.label}\n({z.context.name})", (z.center_x, z.center_y),
                    ha="center", fontsize=9, color=c, fontweight="bold")
    rw = np.array(results["raw_wp"]) if results["raw_wp"] else None
    cw = np.array(results["clean_wp"]) if results["clean_wp"] else None
    if rw is not None and len(rw):
        ax.plot(rw[:, 0], rw[:, 1], "-", color="#aaaaaa", linewidth=1.3,
                label="Original plan (blocked segments)")
        ax.plot(rw[:, 0], rw[:, 1], ".", color="#aaaaaa", markersize=6)
    if cw is not None and len(cw):
        ax.plot(cw[:, 0], cw[:, 1], "o--", color="#1f77b4", markersize=5,
                linewidth=1.4, label="Sanitized plan")
    ax.plot(_col(h, "x"), _col(h, "y"), "-", color="#2ca02c", linewidth=2.0,
            alpha=0.9, label="Actual flight path")
    ev = results["guardrail_events"]
    if ev:
        ax.plot([e["x"] for e in ev], [e["y"] for e in ev], "x", color="red",
                markersize=10, markeredgewidth=2.5, label="Online guardrail veto")
    ax.plot([0], [0], "^", color="black", markersize=11, label="Launch pad")
    ax.set_xlabel("X (m)", fontsize=11); ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title("Agent C - Ethical Guardrails: Path Sanitization Around Social Zones",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left"); ax.set_aspect("equal")
    fig.tight_layout(); _fig("07_agentC_ethical_zone_map", fig)
    shutil.copyfile(FIG_DIR / "07_agentC_ethical_zone_map.png",
                    FIG_DIR / "fig_ethical_guardrails.png")
    fig2, ax2 = plt.subplots(figsize=(9.5, 7.5))
    ax2.plot(_col(h, "x"), _col(h, "y"), "-", color="#2ca02c", linewidth=2.0)
    for z in results["zones"]:
        ax2.add_patch(Circle((z.center_x, z.center_y), z.radius,
                             color="#d62728", alpha=0.15))
    ax2.set_aspect("equal"); ax2.grid(True, alpha=0.35)
    ax2.set_title("Agent C - Ethical Path Sanitization (Guardrail Zones)")
    fig2.savefig(FIG_DIR / "fig_ethical_guardrails.pdf", dpi=300)
    plt.close(fig2)


# ---------------------------------------------------------------------
# Fig 08 - Agent C: distance to zones over time
# ---------------------------------------------------------------------
def _fig_agentC_proximity(results: dict) -> None:
    h = results["history"]
    ts, xs, ys = _col(h, "t"), _col(h, "x"), _col(h, "y")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = ["#d62728", "#ff7f0e", "#9467bd"]
    for i, z in enumerate(results["zones"]):
        d = np.sqrt((xs - z.center_x) ** 2 + (ys - z.center_y) ** 2) - z.radius
        ax.plot(ts, d, color=colors[i % 3], linewidth=1.6,
                label=f"{z.label} ({z.context.name})")
    ax.axhline(0, color="black", linewidth=1.0, linestyle=":")
    ax.text(ts[-1] * 0.01, 1.2, "inside zone", fontsize=8.5, color="#666666")
    ev = results["guardrail_events"]
    if ev:
        ax.plot([e["t"] for e in ev], [0] * len(ev), "rx", markersize=9,
                markeredgewidth=2.2, label="Guardrail veto events")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Signed distance to zone edge (m)")
    ax.set_title("Agent C - Proximity to Ethical Exclusion Zones")
    ax.legend(fontsize=8.5); fig.tight_layout()
    _fig("08_agentC_zone_proximity", fig)


# ---------------------------------------------------------------------
# Fig 09 - Agent D: battery SoC with 60 s-ahead forecast
# ---------------------------------------------------------------------
def _fig_agentD_battery(results: dict) -> None:
    h, tl = results["history"], results["twin_log"]
    ts, bat = _col(h, "t"), _col(h, "battery")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(ts, bat, color="#2ca02c", linewidth=2.2, label="Actual battery SoC")
    if tl:
        ft = np.array([e["t"] + 60.0 for e in tl])
        fp = np.array([e["pred_soc_60s"] * 100.0 for e in tl])
        keep = ft <= ts[-1]
        ax.plot(ft[keep], fp[keep], "--", color="#ff7f0e", linewidth=1.6,
                label="Digital-twin forecast (made 60 s earlier)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Battery SoC (%)")
    ax.set_title("Agent D - Digital Twin: Battery Forecast vs Actual Consumption")
    ax.legend(fontsize=9); fig.tight_layout()
    _fig("09_agentD_battery_soc", fig)
    shutil.copyfile(FIG_DIR / "09_agentD_battery_soc.png",
                    FIG_DIR / "fig_energy_forecast.png")
    fig2, ax2 = plt.subplots(figsize=(9.5, 4.8))
    ax2.plot(ts, bat, color="#2ca02c", linewidth=2.2)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Battery SoC (%)")
    ax2.set_title("Digital Twin Energy Forecast vs Actual Consumption")
    ax2.grid(True, alpha=0.35); fig2.tight_layout()
    fig2.savefig(FIG_DIR / "fig_energy_forecast.pdf", dpi=300)
    plt.close(fig2)


# ---------------------------------------------------------------------
# Fig 10 - Agent D: energy forecast + adaptation decisions
# ---------------------------------------------------------------------
def _fig_agentD_energy(results: dict) -> None:
    h, tl = results["history"], results["twin_log"]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True)
    if tl:
        tt = np.array([e["t"] for e in tl])
        ax = axes[0]
        ax.plot(tt, [e["energy_wh"] for e in tl], "-", color="#d62728",
                linewidth=1.8, label="Energy needed (next 60 s horizon)")
        ax.plot(tt, [e["remaining_wh"] * 0.8 for e in tl], "--", color="#2ca02c",
                linewidth=1.8, label="Usable remaining energy (80% of pack)")
        ax.fill_between(tt, 0, [e["energy_wh"] for e in tl],
                        color="#d62728", alpha=0.08)
        ax.set_ylabel("Energy (Wh)")
        ax.set_title("Agent D - Rolling 60 s Energy Forecast")
        ax.legend(fontsize=9)
        ax = axes[1]
        ax.plot(tt, [e["rec_speed"] for e in tl], "-", color="#1f77b4",
                linewidth=1.8, label="Recommended speed")
        ax.plot(tt, [e["rec_alt"] for e in tl], "-", color="#9467bd",
                linewidth=1.8, label="Recommended altitude")
        ax.set_ylabel("Speed (m/s) / Altitude (m)")
        ax.legend(fontsize=9)
    axes[1].plot(_col(h, "t"), _col(h, "cruise"), ":", color="#333333",
                 linewidth=1.6, label="Applied cruise speed")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Agent D - Mission Adaptation Recommendations")
    axes[1].legend(fontsize=9)
    fig.tight_layout(); _fig("10_agentD_energy_forecast", fig)


# ---------------------------------------------------------------------
# Fig 11 - Agent D: Kalman wind estimation
# ---------------------------------------------------------------------
def _fig_agentD_wind(results: dict) -> None:
    h = results["history"]
    ts = _col(h, "t")
    wt = np.array([x["wind_true"] for x in h])
    we = np.array([x["wind_est"] for x in h])
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    ax = axes[0]
    ax.plot(ts, wt[:, 0], color="#1f77b4", linewidth=1.6, label="True wind X")
    ax.plot(ts, we[:, 0], "--", color="#1f77b4", linewidth=1.4,
            alpha=0.85, label="Estimated wind X")
    ax.plot(ts, wt[:, 1], color="#d62728", linewidth=1.6, label="True wind Y")
    ax.plot(ts, we[:, 1], "--", color="#d62728", linewidth=1.4,
            alpha=0.85, label="Estimated wind Y")
    ax.set_ylabel("Wind component (m/s)")
    ax.set_title("Agent D - Kalman Wind Estimator vs Ground Truth")
    ax.legend(fontsize=8.5, ncol=2)
    ax = axes[1]
    ax.plot(ts, np.linalg.norm(wt[:, :2], axis=1), color="#2ca02c",
            linewidth=1.8, label="True horizontal wind speed")
    ax.plot(ts, np.linalg.norm(we[:, :2], axis=1), "--", color="#ff7f0e",
            linewidth=1.6, label="Estimated wind speed")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Wind speed (m/s)")
    ax.legend(fontsize=9)
    fig.tight_layout(); _fig("11_agentD_wind_estimation", fig)


# ---------------------------------------------------------------------
# Fig 12 - combined dashboard
# ---------------------------------------------------------------------
def _fig_dashboard(results: dict) -> None:
    h = results["history"]
    ts = _col(h, "t")
    t_fail = results["failure_time"]
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("ARIA - Integrated 4-Agent Simulation Dashboard",
                 fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ts, _col(h, "z"), "b-", linewidth=1.6)
    _mark_failure(ax, t_fail)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude (Agent A recovery)"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ts, _col(h, "battery"), "g-", linewidth=1.6)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Battery SoC (%)")
    ax.set_title("Battery (Agent D forecast)")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ts, _col(h, "roll"), "r-", linewidth=1.0, label="roll")
    ax.plot(ts, _col(h, "pitch"), "b-", linewidth=1.0, label="pitch")
    _mark_failure(ax, t_fail)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (deg)")
    ax.set_title("Attitude (Agent A)"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0:2])
    for z in results["zones"]:
        ax.add_patch(Circle((z.center_x, z.center_y), z.radius,
                            color="#d62728", alpha=0.12))
    sc = ax.scatter(_col(h, "x"), _col(h, "y"), c=ts, cmap="viridis", s=4)
    fig.colorbar(sc, ax=ax, label="Time (s)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Flight path + ethical zones (Agent B / C)")
    ax.set_aspect("equal")

    ax = fig.add_subplot(gs[1, 2])
    rpms = np.array([x["rpms"] for x in h])
    for i in range(4):
        ax.plot(ts, rpms[:, i], color=MOTOR_COLORS[i], linewidth=1.2,
                label=MOTOR_LABELS[i])
    _mark_failure(ax, t_fail)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("RPM")
    ax.set_title("Motor RPMs (Agent A)"); ax.legend(fontsize=8, ncol=2)

    png = FIG_DIR / "12_aria_dashboard.png"
    fig.savefig(png, dpi=150); plt.close(fig)
    logger.success(f"Figure saved -> {png.name}")
    shutil.copyfile(png, FIG_DIR / "aria_simulation_results.png")


# ---------------------------------------------------------------------
# Fig 13 - 3D flight path
# ---------------------------------------------------------------------
def _fig_3d_path(results: dict) -> None:
    h = results["history"]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    ts = _col(h, "t")
    sc = ax.scatter(_col(h, "x"), _col(h, "y"), _col(h, "z"),
                    c=ts, cmap="plasma", s=5)
    cw = np.array(results["clean_wp"]) if results["clean_wp"] else None
    if cw is not None and len(cw):
        ax.plot(cw[:, 0], cw[:, 1], cw[:, 2], "o--", color="#1f77b4",
                markersize=4, linewidth=1.2, label="Sanitized waypoints")
    ax.scatter([0], [0], [0], color="black", marker="^", s=70, label="Launch pad")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("ARIA 3D Flight Path (color = time)", fontweight="bold")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Time (s)")
    ax.legend(fontsize=8, loc="upper left")
    _fig("13_flight_path_3d", fig)


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------
def _generate_all_figures(results: dict) -> None:
    logger.info("Generating figures ...")
    _fig_architecture(results)
    _fig_agentA_altitude(results)
    _fig_agentA_attitude(results)
    _fig_agentA_motors(results)
    _fig_agentB_plan(results)
    _fig_agentB_pipeline(results)
    _fig_agentB_tracking(results)
    _fig_agentC_map(results)
    _fig_agentC_proximity(results)
    _fig_agentD_battery(results)
    _fig_agentD_energy(results)
    _fig_agentD_wind(results)
    _fig_dashboard(results)
    _fig_3d_path(results)
    logger.success(f"All figures saved to {FIG_DIR}")


# ---------------------------------------------------------------------
# Summary report + console output
# ---------------------------------------------------------------------
def _mission_stats(results: dict) -> dict:
    h = results["history"]
    ts, zs = _col(h, "t"), _col(h, "z")
    idx_fail = int(np.searchsorted(ts, results["failure_time"]))
    post = slice(idx_fail, None)
    min_alt_after_failure = float(np.min(zs[post])) if idx_fail < len(ts) else float("nan")
    last = h[-1]
    dist_flown = float(np.sum(np.linalg.norm(
        np.diff(np.column_stack([_col(h, "x"), _col(h, "y")]), axis=0), axis=1)))
    return {
        "samples": len(h),
        "final_pos": (last["x"], last["y"], last["z"]),
        "final_battery": last["battery"],
        "min_alt_after_failure": min_alt_after_failure,
        "final_alt": last["z"],
        "waypoints_reached": last["wp_idx"] + 1,
        "waypoints_total": len(results["clean_wp"]),
        "dist_flown": dist_flown,
        "guardrail_events": len(results["guardrail_events"]),
        "rerouted": results["n_rerouted"],
    }


def _write_summary_report(results: dict) -> None:
    s = _mission_stats(results)
    plan = results["plan"]
    lines = [
        "# ARIA Simulation - Mission Summary Report",
        "",
        f"- **Command:** {results['command']}",
        f"- **Parsed mode:** {plan.mode.name} (Agent B)",
        f"- **Reasoning:** {plan.reasoning}",
        f"- **Duration:** {results['duration']:.0f} s @ {1/SIM_DT:.0f} Hz",
        f"- **Motor failure:** {results['failure_motor'].name} at t="
        f"{results['failure_time']:.0f} s (Agent A)",
        "",
        "## Key results",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Telemetry samples | {s['samples']} |",
        f"| Distance flown | {s['dist_flown']:.1f} m |",
        f"| Waypoints reached | {s['waypoints_reached']} / {s['waypoints_total']} |",
        f"| Waypoints rerouted by guardrails | {s['rerouted']} (Agent C) |",
        f"| Online guardrail vetoes | {s['guardrail_events']} (Agent C) |",
        f"| Min altitude after motor failure | {s['min_alt_after_failure']:.1f} m (Agent A) |",
        f"| Final altitude | {s['final_alt']:.1f} m |",
        f"| Final battery SoC | {s['final_battery']:.1f} % (Agent D) |",
        "",
        "## Generated artifacts",
        "",
        f"- Figures: `output/figures/` ({len(list(FIG_DIR.glob('*.png')))} PNG files)",
        "- Telemetry: `output/telemetry/aria_telemetry.csv`",
        "- Log: `output/logs/aria_simulation.log`",
        "",
        "All figures are auto-generated on every run - no manual screenshots needed.",
    ]
    path = OUTPUT_DIR / "summary_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"Summary report saved -> {path}")


def _print_console_summary(results: dict) -> None:
    s = _mission_stats(results)
    table = Table(title="ARIA Mission Summary", show_lines=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_row("Command", results["command"])
    table.add_row("Mission mode (Agent B)", results["plan"].mode.name)
    table.add_row("Waypoints reached (Agent B)",
                  f"{s['waypoints_reached']} / {s['waypoints_total']}")
    table.add_row("Waypoints rerouted (Agent C)", str(s["rerouted"]))
    table.add_row("Online guardrail vetoes (Agent C)", str(s["guardrail_events"]))
    table.add_row("Min altitude after failure (Agent A)",
                  f"{s['min_alt_after_failure']:.1f} m")
    table.add_row("Final battery SoC (Agent D)", f"{s['final_battery']:.1f} %")
    table.add_row("Distance flown", f"{s['dist_flown']:.1f} m")
    table.add_row("Telemetry samples", str(s["samples"]))
    console.print(table)

    files = sorted(p.name for p in FIG_DIR.glob("*.png"))
    console.print(f"\n[bold]Outputs saved:[/bold]")
    console.print(f"  figures   : {FIG_DIR}  ({len(files)} PNG files)")
    console.print(f"  telemetry : {TEL_DIR / 'aria_telemetry.csv'}")
    console.print(f"  log       : {LOG_DIR / 'aria_simulation.log'}")
    console.print(f"  report    : {OUTPUT_DIR / 'summary_report.md'}\n")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARIA integrated 4-agent drone simulation (auto-saves all outputs)")
    parser.add_argument("--duration", type=float, default=150.0,
                        help="simulation duration in seconds (default: 150)")
    parser.add_argument("--failure-time", type=float, default=45.0,
                        help="time at which the motor failure is injected (default: 45)")
    parser.add_argument("--command", type=str, default=COMMAND,
                        help="natural language mission command")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    run_simulation(duration=args.duration, failure_time=args.failure_time,
                   command=args.command, seed=args.seed)


if __name__ == "__main__":
    main()
