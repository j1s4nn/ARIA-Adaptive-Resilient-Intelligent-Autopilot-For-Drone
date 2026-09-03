"""
ARIA simulation package.

``run_simulation`` is imported lazily because it pulls in matplotlib and
the full agent stack; importing the package itself stays lightweight.
"""

__all__ = ["run_simulation"]


def __getattr__(name):
    if name == "run_simulation":
        from .run_simulation import run_simulation
        return run_simulation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
