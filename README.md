# ARIA – Adaptive Resilient Intelligent Autopilot

> *An advanced autonomous drone system with self-healing, natural language control, ethical awareness, and energy-intelligent mission planning.*

---

##  Overview

ARIA integrates **four AI agents** into a unified drone autopilot stack:

| Agent | Innovation |
|-------|-----------|
| **A – Self-Healing** | RL agent re-calculates flight physics for a crippled drone in real-time |
| **B – LLM-Pilot** | Local LLM translates natural language commands to flight maneuvers |
| **C – Ethical Guardrails** | CLIP-based scene classifier prevents socially unacceptable overflights |
| **D – Digital Twin** | Simulates energy 60 seconds ahead, adjusts altitude to catch tailwinds |

---

##  Hardware Compatibility

| Component | Requirement | Your Setup (RTX 3060) |
|-----------|------------|----------------------|
| GPU | CUDA-capable |  RTX 3060 12GB |
| VRAM | ≥6GB for RL training |  12GB |
| VRAM | ~4GB for Q4 LLM (7B) |  Fits in 12GB |
| RAM | ≥12GB |  16GB |
| CUDA | 11.8+ | Use cu118 wheels |

**Does the code work on RTX 3060 + 16GB RAM?**  Yes, with these caveats:
- RL training (Agent A): ~30–60 min for 2M steps on RTX 3060
- LLM inference (Agent B): Mistral 7B Q4_K_M uses ~4GB VRAM, leaving 8GB for RL
- CLIP (Agent C): ViT-B/32 uses ~1GB VRAM, very fast
- Digital Twin (Agent D): CPU-only, no GPU needed

---

## 🛰️ Simulation Software

### Primary (Recommended): PyFlyt
```bash
pip install PyFlyt
```
- Built on PyBullet physics
- Native multirotor support
- Works on all platforms
- Docs: https://jjshoots.github.io/PyFlyt/

### Alternative: Microsoft AirSim
- Download: https://github.com/microsoft/AirSim/releases
- Unreal Engine-based, photorealistic
- API: https://microsoft.github.io/AirSim/
- Python: `pip install airsim`

### Alternative: Gazebo + PX4 SITL
- Best for real-hardware transfer
- Guide: https://docs.px4.io/main/en/simulation/gazebo_classic.html

---

##  Installation

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/ARIA-drone.git
cd ARIA-drone

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install PyTorch with CUDA 11.8 (RTX 3060)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. (Optional) Install LLM for Agent B
#    Download: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
#    Get: Mistral-7B-Instruct-v0.2.Q4_K_M.gguf (~4.1 GB)
pip install llama-cpp-python --extra-index-url \
    https://abetlen.github.io/llama-cpp-python/whl/cu118

# 6. Install CLIP
pip install git+https://github.com/openai/CLIP.git

# 7. Install ARIA
pip install -e .
```

---

##  Quick Start

```bash
# Run integrated simulation (no GPU required for demo)
python simulation/run_simulation.py

# Train self-healing RL agent
python agents/self_healing_agent.py

# Run tests
pytest tests/ -v
```

---

##  Project Structure

```
ARIA/
├── core/
│   └── drone_model.py         # Physics model, state, motor dynamics
├── agents/
│   ├── self_healing_agent.py  # Agent A: RL-based fault recovery
│   ├── llm_pilot.py           # Agent B: NL command → flight plan
│   ├── ethical_guardrails.py  # Agent C: CLIP social context filter
│   └── digital_twin.py        # Agent D: Energy forecasting
├── simulation/
│   └── run_simulation.py      # Integrated 4-agent simulation loop
├── tests/
│   └── test_all_agents.py     # Pytest unit tests
├── assets/                    # Generated figures for paper
├── docs/                      # LaTeX paper files
├── requirements.txt
├── setup.py
└── README.md
```

---

##  Results Visualization

After running `run_simulation.py`, figures are saved to `assets/`:

- `aria_simulation_results.png` — Combined dashboard
- `fig_self_healing_altitude.pdf` — Agent A recovery plot
- `fig_energy_forecast.pdf` — Agent D energy timeline
- `fig_ethical_guardrails.pdf` — Agent C zone avoidance map

Use these directly in the LaTeX paper.

---

##  LLM Model Setup (Agent B)

Agent B requires a local GGUF model. Without it, a rule-based fallback is used automatically.

```python
from agents.llm_pilot import LLMPilot

# With local model
pilot = LLMPilot(model_path="models/Mistral-7B-Instruct-v0.2.Q4_K_M.gguf")

# Without model (mock/rule-based)
pilot = LLMPilot()

plan = pilot.parse_command("Find the blue truck and follow it unnoticed")
```

---

##  Citation

If you use ARIA in your research:

```bibtex
@misc{aria2025,
  title  = {ARIA: Adaptive Resilient Intelligent Autopilot},
  author = {Your Name},
  year   = {2025},
  note   = {GitHub: https://github.com/YOUR_USERNAME/ARIA-drone}
}
```
