# Contributing to ARIA

Thank you for your interest in contributing to ARIA! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- A clear, descriptive title
- Detailed steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version, GPU if applicable)
- Relevant logs from `output/logs/`

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Include:

- A clear description of the proposed feature
- Rationale and use cases
- If applicable, mockups or diagrams

### Pull Requests

1. **Fork** the repository and create your branch from `main`
2. **Write tests** for new features or bug fixes
3. **Run the full test suite**: `pytest tests/`
4. **Run a simulation** to verify integration: `python simulation/run_simulation.py`
5. **Update documentation** if you change APIs or add features
6. **Follow the code style**: use `black` and `isort` for formatting
7. **Commit with clear messages** describing what and why, not just how

```bash
# Example workflow
git checkout -b feature/improved-wind-estimation
# ... make changes ...
pytest tests/
python simulation/run_simulation.py --duration 60
git add -p
git commit -m "feat(digital-twin): add Kalman-filtered wind estimation"
git push origin feature/improved-wind-estimation
```

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ARIA-Adaptive-Resilient-Intelligent-Autopilot-For-Drone.git
cd ARIA-Adaptive-Resilient-Intelligent-Autopilot-For-Drone

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run simulation
python simulation/run_simulation.py
```

### Optional: ML/LLM Dependencies

For full Agent A (RL self-healing) and Agent B (LLM pilot) functionality:

```bash
pip install torch stable-baselines3 gymnasium llama-cpp-python
```

Download a GGUF model (e.g., Llama 3.2 1B) and pass `model_path` when creating `LLMPilot`.

## Project Structure

```
agents/
  self_healing_agent.py    # Agent A: RL-based fault recovery
  llm_pilot.py             # Agent B: NL command → waypoints
  ethical_guardrails.py    # Agent C: Geofencing + social context
  digital_twin.py          # Agent D: Energy + wind forecasting
core/
  drone_model.py           # 6-DOF quadcopter dynamics
simulation/
  run_simulation.py        # Integrated 4-agent demo
tests/                     # Unit tests for all agents
output/                    # Auto-generated figures, logs, telemetry
```

## Coding Standards

- **Python 3.10+** required
- Use **type hints** wherever possible
- **Docstrings** for all public functions/classes (Google or NumPy style)
- **Loguru** for logging (avoid print statements)
- Keep functions **small and focused** (prefer composition over large monoliths)
- **No hard-coded paths** — use `pathlib.Path` and relative references

## Testing

- Unit tests live in `tests/`
- Integration test: `python simulation/run_simulation.py` (should complete without crashes)
- All PRs must pass CI checks (GitHub Actions)

## Questions?

Open an issue with the `question` label or reach out in Discussions.

---

**Happy hacking!** 🚁✨
