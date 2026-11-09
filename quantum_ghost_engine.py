"""Backward-compatible alias: the code now lives in multi_timeline.py.

The old function names still work and forward to the new ones."""
from multi_timeline import (  # noqa: F401
    _run_one_timeline,
    run_multi_timeline,
    run_multi_timeline_analysis,
)


def run_quantum_analysis(width=20, height=20, seed=42, iterations=3, max_workers=None):
    result = run_multi_timeline_analysis(width, height, seed, iterations, max_workers)
    result["quantum_computing"] = False      # key kept so older callers do not break
    return result


def run_quantum_superposition(width=20, height=20, seed=42, iterations=3):
    return run_multi_timeline(width, height, seed, iterations)


if __name__ == "__main__":
    run_multi_timeline()
