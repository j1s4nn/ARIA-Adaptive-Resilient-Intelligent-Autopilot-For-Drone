"""
ARIA Core: Drone State Management & Physics Model
Handles quadrotor dynamics, motor failure simulation, and state estimation.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from loguru import logger


class MotorID(Enum):
    FRONT_LEFT  = 0
    FRONT_RIGHT = 1
    REAR_LEFT   = 2
    REAR_RIGHT  = 3


@dataclass
class DroneConfig:
    """Physical parameters of the quadrotor platform."""
    mass: float = 0.85           # kg
    arm_length: float = 0.175    # m (motor-to-center)
    Ixx: float = 7.0e-3          # kg·m²
    Iyy: float = 7.0e-3
    Izz: float = 1.2e-2
    k_thrust: float = 3.13e-5    # N/(rad/s)²
    k_drag: float = 7.5e-7       # Nm/(rad/s)²
    max_rpm: float = 8600.0
    min_rpm: float = 1200.0
    gravity: float = 9.81        # m/s²
    drag_coeff: float = 0.1      # translational drag
    battery_capacity: float = 4.0  # Ah (typical 4S LiPo)
    battery_voltage: float = 14.8  # V


@dataclass
class DroneState:
    """Full 6-DOF state of the drone."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))    # [x, y, z] meters
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))    # [vx, vy, vz] m/s
    attitude: np.ndarray = field(default_factory=lambda: np.zeros(3))    # [roll, pitch, yaw] rad
    angular_rate: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [p, q, r] rad/s
    motor_rpms: np.ndarray = field(default_factory=lambda: np.zeros(4))  # per motor
    battery_soc: float = 1.0    # State of Charge [0,1]
    timestamp: float = 0.0       # seconds

    # Motor health: 1.0 = healthy, 0.0 = failed
    motor_health: np.ndarray = field(default_factory=lambda: np.ones(4))

    def to_vector(self) -> np.ndarray:
        """Flatten state to a 19-dim observation vector."""
        return np.concatenate([
            self.position,
            self.velocity,
            self.attitude,
            self.angular_rate,
            self.motor_rpms / 8600.0,  # normalize
            [self.battery_soc],
            self.motor_health,
        ])


class DronePhysics:
    """
    Simplified quadrotor physics engine.
    Computes forces/torques from motor commands and integrates state forward.
    """

    def __init__(self, config: Optional[DroneConfig] = None):
        self.cfg = config or DroneConfig()
        self.g = np.array([0.0, 0.0, -self.cfg.gravity])

    def rotation_matrix(self, attitude: np.ndarray) -> np.ndarray:
        """ZYX Euler → rotation matrix R (body→world)."""
        r, p, y = attitude
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        return np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,    cp*sr,              cp*cr           ],
        ])

    def motor_forces_torques(self, rpms: np.ndarray, health: np.ndarray):
        """Compute thrust and torques from motor RPMs and health status."""
        ω = rpms * (2 * np.pi / 60.0)  # RPM → rad/s
        thrusts = self.cfg.k_thrust * ω**2 * health
        drags   = self.cfg.k_drag   * ω**2 * health
        L = self.cfg.arm_length

        total_thrust = np.sum(thrusts)
        # Standard + configuration:
        #  M0(FL) M1(FR) M2(RL) M3(RR)
        tau_x = L * (thrusts[0] + thrusts[2] - thrusts[1] - thrusts[3])  # roll
        tau_y = L * (thrusts[0] + thrusts[1] - thrusts[2] - thrusts[3])  # pitch
        # yaw: CCW motors minus CW motors
        tau_z = drags[0] - drags[1] + drags[2] - drags[3]

        return total_thrust, np.array([tau_x, tau_y, tau_z])

    def step(self, state: DroneState, rpm_cmd: np.ndarray, dt: float = 0.01) -> DroneState:
        """
        Integrate dynamics one time step forward.
        rpm_cmd: desired RPM per motor (4,)
        """
        cfg = self.cfg
        # Clamp commands
        rpms = np.clip(rpm_cmd * state.motor_health, cfg.min_rpm, cfg.max_rpm)
        state.motor_rpms = rpms

        thrust, torques = self.motor_forces_torques(rpms, state.motor_health)

        R = self.rotation_matrix(state.attitude)
        thrust_world = R @ np.array([0, 0, thrust])

        # Translational drag
        drag_force = -cfg.drag_coeff * state.velocity

        # Linear acceleration (world frame)
        accel = (thrust_world + drag_force) / cfg.mass + self.g

        # Angular acceleration (body frame)
        I = np.diag([cfg.Ixx, cfg.Iyy, cfg.Izz])
        omega = state.angular_rate
        alpha = np.linalg.inv(I) @ (torques - np.cross(omega, I @ omega))

        # Integrate
        new_state = DroneState()
        new_state.position     = state.position + state.velocity * dt
        new_state.velocity     = state.velocity + accel * dt
        new_state.attitude     = state.attitude + state.angular_rate * dt
        new_state.angular_rate = state.angular_rate + alpha * dt
        new_state.motor_rpms   = rpms
        new_state.motor_health = state.motor_health.copy()
        new_state.timestamp    = state.timestamp + dt

        # Simple battery drain model (power ∝ thrust²)
        power = thrust**2 / (cfg.mass * cfg.battery_voltage * 3600)
        new_state.battery_soc  = max(0.0, state.battery_soc - power * dt)

        # Ground collision
        if new_state.position[2] < 0.0:
            new_state.position[2] = 0.0
            new_state.velocity[2] = 0.0

        return new_state


def inject_motor_failure(state: DroneState, motor_id: MotorID, severity: float = 1.0) -> DroneState:
    """
    Simulate motor failure.
    severity=1.0 → complete failure | severity=0.5 → 50% degradation
    """
    state.motor_health[motor_id.value] = max(0.0, 1.0 - severity)
    logger.warning(f"Motor {motor_id.name} health set to {state.motor_health[motor_id.value]:.1%}")
    return state
