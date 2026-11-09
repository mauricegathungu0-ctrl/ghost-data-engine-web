"""
Ghost Data Engine multi-timeline extension.

Runs several independently seeded 2D engine runs in parallel worker processes and
reports the mean and population variance of their summary metrics. This is ordinary
seeded simulation on a classical CPU.
"""
import contextlib
import io
import statistics
from concurrent.futures import ProcessPoolExecutor
from main import run_ghost_engine


def _run_one_timeline(task):
    index, width, height, seed = task
    # Suppress each worker's verbose core log; the report is printed once in the parent.
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_ghost_engine(
            width=width, height=height, seed=seed, num_voids=None,
            max_void_size=5, keep_border=True, intensity=10,
            intensity_jitter=2,
        )
    summary = result["report"]["summary"]
    return {
        "timeline_id": index + 1,
        "seed": seed,
        "tension": summary.get("cumulative_structural_tension", 0),
        "reconstructed": summary.get("cells_reconstructed", 0),
        "stalled": summary.get("cells_stalled", 0),
        "hottest_coord": summary.get("highest_tension_coordinate"),
        "mean_confidence": summary.get("mean_confidence", 0.0),
        "propagation_waves": summary.get("propagation_waves", 0),
    }


def run_multi_timeline_analysis(width=20, height=20, seed=42, iterations=3, max_workers=None):
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(width, int) or not isinstance(height, int) or width < 3 or height < 3:
        raise ValueError("width and height must be integers of at least 3")
    if max_workers is not None and (not isinstance(max_workers, int) or max_workers < 1):
        raise ValueError("max_workers must be a positive integer or None")
    tasks = [
        (i, width, height, (seed + i * 17) if seed is not None else None)
        for i in range(iterations)
    ]
    workers = max_workers or min(iterations, 4)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        timelines = list(pool.map(_run_one_timeline, tasks))

    metrics = {
        key: [float(t[key]) for t in timelines]
        for key in ("tension", "reconstructed", "stalled", "mean_confidence")
    }
    variance = {
        key: round(statistics.pvariance(values), 6) if len(values) > 1 else 0.0
        for key, values in metrics.items()
    }
    means = {key: round(statistics.mean(values), 6) for key, values in metrics.items()}
    return {
        "timelines": timelines,
        "mean": means,
        "population_variance": variance,
        "timeline_count": len(timelines),
        "execution": "parallel_processes",
    }


def run_multi_timeline(width=20, height=20, seed=42, iterations=3):
    """Prints a report and returns the list of per-timeline dicts."""
    result = run_multi_timeline_analysis(width, height, seed, iterations)
    print("=== GHOST DATA ENGINE MULTI-TIMELINE REPORT ===")
    print(f"Parallel timelines: {result['timeline_count']}")
    for timeline in result["timelines"]:
        print(
            f"Timeline #{timeline['timeline_id']} seed={timeline['seed']} "
            f"tension={timeline['tension']} reconstructed={timeline['reconstructed']} "
            f"stalled={timeline['stalled']} confidence={timeline['mean_confidence']}"
        )
    print("Means:", result["mean"])
    print("Population variance:", result["population_variance"])
    print("Execution:", result["execution"])
    return result["timelines"]


if __name__ == "__main__":
    run_multi_timeline()
