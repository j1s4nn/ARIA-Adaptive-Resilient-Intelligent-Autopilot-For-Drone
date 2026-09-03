"""
ARIA agents package.

Heavy dependencies (torch / stable-baselines3 / gymnasium) are imported
lazily so that the lightweight agents (LLM-Pilot rule mode, Ethical
Guardrails heuristic mode, Digital Twin) work even on machines that do
not have the full reinforcement-learning stack installed.

Importing ``SelfHealingEnv`` / ``train_self_healing`` / ``load_agent``
requires the ``ml`` extras::

    pip install -e .[ml]
"""

from .llm_pilot import LLMPilot, MissionPlan, FlightMode, Waypoint, WaypointNavigator
from .ethical_guardrails import EthicalGuardrailEngine, GeoZone, SocialContext
from .digital_twin import DigitalTwin, MissionSegment, EnergyForecast

__all__ = [
    "LLMPilot", "MissionPlan", "FlightMode", "Waypoint", "WaypointNavigator",
    "EthicalGuardrailEngine", "GeoZone", "SocialContext",
    "DigitalTwin", "MissionSegment", "EnergyForecast",
    # Lazy (require the `ml` extras)
    "SelfHealingEnv", "train_self_healing", "load_agent",
]


def __getattr__(name):
    """Lazily import the reinforcement-learning symbols on first access."""
    if name in {"SelfHealingEnv", "train_self_healing", "load_agent"}:
        try:
            from . import self_healing_agent as _sha
        except ImportError as exc:  # pragma: no cover - optional dep path
            raise ImportError(
                "The self-healing RL agent requires the optional ML stack. "
                "Install it with:  pip install -e .[ml]"
            ) from exc
        return {
            "SelfHealingEnv": _sha.SelfHealingEnv,
            "train_self_healing": _sha.train,
            "load_agent": _sha.load_agent,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
