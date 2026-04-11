"""
ARIA Agent A: Self-Healing Reinforcement Learning Controller
============================================================
Uses PPO (Proximal Policy Optimization) to learn flight recovery
after motor failure. Trains a policy that observes motor health
and re-distributes thrust in real-time.

GPU: RTX 3060 12GB is more than sufficient for this model size.
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from loguru import logger
from pathlib import Path
from typing import Optional

from core.drone_model import DroneState, DroneConfig, DronePhysics, MotorID, inject_motor_failure


class SelfHealingEnv(gym.Env):
    """
    Custom Gymnasium environment for motor-failure recovery training.

    Observation (19-dim):
        position (3), velocity (3), attitude (3), angular_rate (3),
        motor_rpms_normalized (4), battery_soc (1), motor_health (4) = 21-dim

    Action (4-dim):
        Normalized RPM delta for each motor [-1, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[DroneConfig] = None, render_mode=None):
        super().__init__()
        self.cfg = config or DroneConfig()
        self.physics = DronePhysics(self.cfg)
        self.render_mode = render_mode

        self.dt = 0.02          # 50 Hz control loop
        self.max_steps = 500
        self.step_count = 0

        # Target hover altitude
        self.target_altitude = 5.0

        # Observation space
        obs_dim = 21
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: RPM correction per motor
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Hover RPM (approx sqrt(mg / (4 * k_thrust)))
        hover_w = np.sqrt(self.cfg.mass * self.cfg.gravity / (4 * self.cfg.k_thrust))
        self.hover_rpm = hover_w * 60.0 / (2 * np.pi)

        self.state = self._init_state()

    def _init_state(self) -> DroneState:
        state = DroneState()
        state.position = np.array([0.0, 0.0, self.target_altitude])
        state.motor_rpms = np.ones(4) * self.hover_rpm
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = self._init_state()

        # Randomly inject motor failure during training (curriculum)
        if self.np_random.random() < 0.7:
            failed_motor = self.np_random.integers(0, 4)
            severity = self.np_random.uniform(0.3, 1.0)
            inject_motor_failure(self.state, MotorID(failed_motor), severity)

        return self.state.to_vector().astype(np.float32), {}

    def step(self, action: np.ndarray):
        self.step_count += 1

        # Map action to RPM command
        rpm_range = self.cfg.max_rpm - self.cfg.min_rpm
        rpm_cmd = self.hover_rpm + action * (rpm_range * 0.3)

        self.state = self.physics.step(self.state, rpm_cmd, dt=self.dt)
        obs = self.state.to_vector().astype(np.float32)

        # Reward shaping
        reward = self._compute_reward()

        # Termination
        pos = self.state.position
        terminated = (
            pos[2] <= 0.1 or                    # crashed
            abs(pos[0]) > 20 or abs(pos[1]) > 20 or  # out of bounds
            np.any(np.abs(self.state.attitude[:2]) > np.pi / 2)  # flipped
        )
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _compute_reward(self) -> float:
        pos   = self.state.position
        vel   = self.state.velocity
        att   = self.state.attitude

        # Altitude tracking
        alt_error = abs(pos[2] - self.target_altitude)
        r_altitude = np.exp(-0.5 * alt_error)

        # Stability (penalize roll/pitch and angular rates)
        r_stability = np.exp(-2.0 * np.sum(att[:2]**2))
        r_ang_rate  = np.exp(-0.5 * np.sum(self.state.angular_rate**2))

        # Position drift
        r_position  = np.exp(-0.1 * (pos[0]**2 + pos[1]**2))

        # Survival bonus
        r_survive   = 0.1

        # Battery efficiency (avoid unnecessary power)
        r_battery   = 0.05 * self.state.battery_soc

        total = r_altitude + r_stability + r_ang_rate + r_position + r_survive + r_battery
        return float(total)


def build_ppo_agent(env_id, device="cuda"):
    """Build PPO with network size appropriate for RTX 3060."""
    env = DummyVecEnv([lambda: SelfHealingEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log="./runs/self_healing/",
    )
    return model, env


def train(total_timesteps: int = 2_000_000, save_path: str = "models/self_healing_ppo"):
    """
    Train the self-healing agent.
    ~2M steps takes ~30–60 min on RTX 3060.
    """
    Path("models").mkdir(exist_ok=True)
    model, env = build_ppo_agent("SelfHealingEnv")

    eval_env = DummyVecEnv([lambda: SelfHealingEnv()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    callbacks = [
        EvalCallback(eval_env, best_model_save_path=save_path,
                     eval_freq=50_000, n_eval_episodes=20, verbose=1),
        CheckpointCallback(save_freq=100_000, save_path=save_path, name_prefix="aria_heal"),
    ]

    logger.info(f"Training self-healing agent for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    model.save(save_path + "/final_model")
    env.save(save_path + "/vec_normalize.pkl")
    logger.success("Training complete.")


def load_agent(model_path: str, vec_norm_path: str):
    env = DummyVecEnv([lambda: SelfHealingEnv()])
    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(model_path, env=env, device="cuda")
    return model, env


if __name__ == "__main__":
    # Quick environment sanity check
    env = SelfHealingEnv()
    check_env(env, warn=True)
    logger.info("Environment check passed. Starting training...")
    train(total_timesteps=2_000_000)
