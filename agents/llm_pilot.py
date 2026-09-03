"""
ARIA Agent B: Natural Language Command & Control (LLM-Pilot)
=============================================================
Translates high-level human intent (e.g., "follow the blue truck
at a safe distance") into low-level waypoint sequences and
behavioral modes using a local quantized LLM (LLaMA / Mistral).

Compatible with llama-cpp-python for CPU/GPU inference.
RTX 3060: Use Q4_K_M quantized model (~4GB VRAM for 7B params).
"""

import json
import re
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional
from loguru import logger

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. Using mock LLM for testing.")


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

class FlightMode(Enum):
    HOVER       = auto()
    GOTO        = auto()
    FOLLOW      = auto()
    SEARCH      = auto()
    ORBIT       = auto()
    RETURN_HOME = auto()
    LAND        = auto()


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    speed: float = 3.0       # m/s
    loiter_time: float = 0.0 # seconds


@dataclass
class MissionPlan:
    mode: FlightMode
    waypoints: list[Waypoint]
    target_description: Optional[str] = None
    follow_distance: float = 15.0   # m (safe follow range)
    orbit_radius: float = 10.0
    raw_command: str = ""
    reasoning: str = ""


# ─────────────────────────────────────────────
# LLM Interface
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are ARIA's flight planner. Parse the operator's natural language command 
and output a JSON flight plan. Be conservative for safety.

Output ONLY valid JSON in this exact format:
{
  "mode": "GOTO|FOLLOW|SEARCH|ORBIT|HOVER|RETURN_HOME|LAND",
  "waypoints": [{"x": float, "y": float, "z": float, "speed": float, "loiter_time": float}],
  "target_description": "string or null",
  "follow_distance": float,
  "orbit_radius": float,
  "reasoning": "brief explanation"
}

Rules:
- Default altitude is 15 meters unless stated otherwise
- "unnoticed" or "stealth" follow → follow_distance >= 25m, altitude +10m above target
- Never fly below 5m altitude in autonomous mode
- "fast" → speed 8 m/s, "slow/careful" → speed 2 m/s, default → 4 m/s
"""


class LLMPilot:
    """
    Wraps a local quantized LLM to parse operator text commands
    into structured MissionPlan objects.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 n_gpu_layers: int = 35,
                 context_length: int = 2048):
        """
        model_path: Path to GGUF model file.
          Recommended: Mistral-7B-Instruct-v0.2-Q4_K_M.gguf (~4GB)
          Download: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
        n_gpu_layers: Offload layers to GPU. 35 fits in RTX 3060 12GB with Q4_K_M.
        """
        self.model_path = model_path
        self.llm = None

        if LLAMA_AVAILABLE and model_path and Path(model_path).exists():
            logger.info(f"Loading LLM from {model_path} ({n_gpu_layers} GPU layers)")
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=context_length,
                temperature=0.1,    # Low temp for deterministic commands
                verbose=False,
            )
            logger.success("LLM loaded successfully.")
        else:
            logger.warning("Running in MOCK mode (no LLM model file). For real use, "
                           "download a GGUF model and pass model_path.")

    def parse_command(self, command: str, drone_position: Optional[np.ndarray] = None) -> MissionPlan:
        """
        Convert a natural language command to a structured MissionPlan.

        Example commands:
          "Go find the blue truck, follow it but stay at a distance where you won't be noticed"
          "Fly to the north parking lot at 20 meters altitude"
          "Circle the building three times then come home"
        """
        logger.info(f"Parsing command: '{command}'")
        pos_str = f"[{drone_position[0]:.1f}, {drone_position[1]:.1f}, {drone_position[2]:.1f}]" \
                  if drone_position is not None else "[0, 0, 15]"

        user_prompt = f"Current drone position: {pos_str}\nCommand: {command}"

        if self.llm is not None:
            plan = self._llm_parse(user_prompt)
        else:
            plan = self._rule_based_parse(command)

        plan.raw_command = command
        return plan

    def _llm_parse(self, user_prompt: str) -> MissionPlan:
        """Call the local LLM and parse its JSON output."""
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        raw = response["choices"][0]["message"]["content"].strip()
        return self._json_to_plan(raw)

    def _json_to_plan(self, json_str: str) -> MissionPlan:
        """Parse LLM JSON output into MissionPlan."""
        # Extract JSON block if model added preamble
        match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if match:
            json_str = match.group()

        try:
            data = json.loads(json_str)
            waypoints = [Waypoint(**wp) for wp in data.get("waypoints", [])]
            mode = FlightMode[data.get("mode", "HOVER")]
            return MissionPlan(
                mode=mode,
                waypoints=waypoints,
                target_description=data.get("target_description"),
                follow_distance=float(data.get("follow_distance", 15.0)),
                orbit_radius=float(data.get("orbit_radius", 10.0)),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.error(f"Failed to parse LLM output: {e}\nRaw: {json_str}")
            return MissionPlan(mode=FlightMode.HOVER, waypoints=[])

    def _rule_based_parse(self, command: str) -> MissionPlan:
        """
        Fallback rule-based parser for testing without a model.
        Handles the most common intent patterns.
        """
        cmd = command.lower()
        waypoints = []

        if any(k in cmd for k in ["search", "find", "look for"]):
            # Checked before "follow": "search X and follow it" commands
            # should first fly the search pattern.
            waypoints = self._generate_search_grid(grid_size=80, step=20, altitude=15)
            stealth = any(k in cmd for k in ["unnoticed", "hidden", "stealthy"])
            return MissionPlan(
                mode=FlightMode.SEARCH,
                waypoints=waypoints,
                target_description=self._extract_target(cmd),
                follow_distance=30.0 if stealth else 15.0,
                reasoning="Search grid generated"
                          + (" (stealth constraints applied)" if stealth else ""),
            )
        elif any(k in cmd for k in ["follow", "track", "tail"]):
            stealth = any(k in cmd for k in ["unnoticed", "hidden", "stealthy", "far"])
            dist = 30.0 if stealth else 15.0
            return MissionPlan(
                mode=FlightMode.FOLLOW,
                waypoints=[Waypoint(0, 0, 20)],
                target_description=self._extract_target(cmd),
                follow_distance=dist,
                reasoning=f"Follow mode detected. Distance {'(stealth)' if stealth else '(normal)'}: {dist}m",
            )
        elif any(k in cmd for k in ["orbit", "circle", "loop"]):
            return MissionPlan(
                mode=FlightMode.ORBIT,
                waypoints=[Waypoint(0, 0, 15)],
                orbit_radius=float(re.search(r'(\d+)\s*m', cmd).group(1)) if re.search(r'(\d+)\s*m', cmd) else 20.0,
                reasoning="Orbit pattern commanded.",
            )
        elif any(k in cmd for k in ["home", "return", "come back", "rtl"]):
            return MissionPlan(mode=FlightMode.RETURN_HOME, waypoints=[Waypoint(0, 0, 20)])
        elif any(k in cmd for k in ["land", "descend"]):
            return MissionPlan(mode=FlightMode.LAND, waypoints=[Waypoint(0, 0, 0)])
        elif any(k in cmd for k in ["go to", "fly to", "move to"]):
            # Try to extract coordinates
            nums = re.findall(r'[-\d.]+', cmd)
            x, y, z = (float(nums[0]), float(nums[1]), float(nums[2])) if len(nums) >= 3 else (10, 10, 15)
            return MissionPlan(
                mode=FlightMode.GOTO,
                waypoints=[Waypoint(x, y, z, speed=4.0)],
                reasoning=f"Goto waypoint ({x}, {y}, {z})",
            )
        else:
            return MissionPlan(mode=FlightMode.HOVER, waypoints=[Waypoint(0, 0, 15)],
                               reasoning="No specific command recognized. Hovering.")

    @staticmethod
    def _extract_target(cmd: str) -> str:
        """Heuristically extract target description from command."""
        patterns = [
            r'(?:find|follow|track|the)\s+([a-z\s]+?)(?:\s+and|\s+but|,|$)',
        ]
        for p in patterns:
            m = re.search(p, cmd)
            if m:
                return m.group(1).strip()
        return "unknown target"

    @staticmethod
    def _generate_search_grid(grid_size: float, step: float, altitude: float) -> list[Waypoint]:
        """Generate lawnmower search pattern."""
        waypoints = []
        y = -grid_size / 2
        while y <= grid_size / 2:
            xs = np.arange(-grid_size / 2, grid_size / 2, step)
            if (y // step) % 2 == 1:
                xs = xs[::-1]
            for x in xs:
                waypoints.append(Waypoint(x=float(x), y=float(y), z=altitude, speed=5.0))
            y += step
        return waypoints


# ─────────────────────────────────────────────
# Waypoint Navigator
# ─────────────────────────────────────────────

class WaypointNavigator:
    """
    Converts MissionPlan waypoints into motor thrust commands
    via a simple proportional controller.
    """

    def __init__(self, kp_pos=0.8, kp_vel=2.0, hover_rpm=4500.0):
        self.kp_pos = kp_pos
        self.kp_vel = kp_vel
        self.hover_rpm = hover_rpm
        self.current_wp_idx = 0

    def compute_rpm(self, state, plan: MissionPlan) -> np.ndarray:
        """Compute RPM commands toward current waypoint."""
        if not plan.waypoints or self.current_wp_idx >= len(plan.waypoints):
            return np.ones(4) * self.hover_rpm

        wp = plan.waypoints[self.current_wp_idx]
        target = np.array([wp.x, wp.y, wp.z])
        error = target - state.position
        dist  = np.linalg.norm(error)

        if dist < 1.0:
            self.current_wp_idx = min(self.current_wp_idx + 1, len(plan.waypoints) - 1)

        # Simple altitude controller
        z_error = error[2]
        delta_rpm = np.clip(z_error * self.kp_pos * 200.0, -500, 500)

        return np.clip(np.ones(4) * (self.hover_rpm + delta_rpm), 1200, 8600)


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pilot = LLMPilot()  # No model path → runs in mock mode

    test_commands = [
        "Go find the blue truck, follow it, but stay at a distance where you won't be noticed.",
        "Search the north field for survivors and mark their positions.",
        "Orbit the warehouse at 25 meters radius.",
        "Return home immediately.",
    ]

    for cmd in test_commands:
        plan = pilot.parse_command(cmd, drone_position=np.array([0.0, 0.0, 20.0]))
        print(f"\nCommand : {cmd}")
        print(f"Mode    : {plan.mode.name}")
        print(f"Reason  : {plan.reasoning}")
        print(f"Waypoints: {len(plan.waypoints)}")
