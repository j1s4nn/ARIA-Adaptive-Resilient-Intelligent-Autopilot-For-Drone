from setuptools import setup, find_packages

setup(
    name="aria-drone",
    version="1.0.0",
    author="Your Name",
    description="ARIA: Adaptive Resilient Intelligent Autopilot – Advanced AI-powered drone system",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.1.0",
        "loguru>=0.7.2",
        "pydantic>=2.4.0",
        "torch>=2.1.0",
    ],
    entry_points={
        "console_scripts": [
            "aria-sim=simulation.run_simulation:main",
            "aria-train=agents.train:main",
        ]
    },
)
