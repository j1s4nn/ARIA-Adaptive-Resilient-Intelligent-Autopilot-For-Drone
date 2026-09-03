# Changelog

All notable changes to ARIA are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-03

### Added
- `simulation/run_simulation.py` now **auto-saves every output**:
  14+ PNG figures (2-4 per core motive) + paper aliases (PNG/PDF),
  full 50 Hz telemetry CSV, run log, and a markdown mission summary report
  under `output/`.
- Fault-tolerant cascaded attitude controller with least-squares control
  allocation (healthy-motor-only mixer) standing in for Agent A's PPO policy.
- Wind-aware physics: `DronePhysics.step()` accepts an ambient wind vector.
- CLI options for the simulation: `--duration`, `--failure-time`,
  `--command`, `--seed`.
- Professional project files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, issue/PR templates, GitHub Actions CI.
- Lean `requirements.txt` (installs cleanly) + optional `requirements-ml.txt`.
- Modern packaging via `pyproject.toml` with optional extras (`[ml]`, `[cv]`).

### Fixed
- **Installer-breaking `requirements.txt`**: pinned local-version torch wheel
  (`torch==2.1.0+cu118`) could not be installed with plain pip and was
  incompatible with modern Python; moved heavy ML deps to optional extras.
- **Import crash**: `agents/__init__.py` eagerly imported torch /
  stable-baselines3 / gymnasium, breaking the whole package on machines
  without the RL stack; now lazy-imported with a helpful error message.
- **Import crash**: `cv2` was a hard dependency of Agent C; now optional.
- **Import crash**: unused `scipy.optimize` import removed from Agent D.
- **Physics bug**: failed motors were clipped to `min_rpm` and kept spinning
  at 1200 RPM; dead motors now spin at 0 RPM.
- **Control bug**: post-failure thrust compensation was linear in RPM
  (thrust grows with RPM^2), launching the drone skyward; replaced with a
  proper fault-tolerant mixer.
- **Design bug**: `DigitalTwin.forecast()` judged whole-mission feasibility
  from a 60 s energy horizon only; long missions on a nearly-empty pack are
  now correctly reported as infeasible.
- **Broken entry point**: `aria-train` console script pointed to the
  non-existent `agents.train` module; now targets
  `agents.self_healing_agent:main`.
- CI workflow no longer installs multi-GB torch just to run unit tests.

### Changed
- Default demo mission: search pattern crossing ethical exclusion zones with
  a FRONT_RIGHT motor failure injected at t = 45 s.
- `SelfHealingEnv` training device auto-detects CUDA/CPU.

## [1.0.0] - 2025-12-01

### Added
- Initial release: four-agent ARIA stack (Self-Healing RL, LLM Pilot,
  Ethical Guardrails, Digital Twin), simplified 6-DOF quadrotor physics,
  integrated simulation, unit tests, and LaTeX paper skeleton.
