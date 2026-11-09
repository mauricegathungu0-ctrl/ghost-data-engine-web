"""
Independent synthetic benchmark: Ghost tension-weighted recursive diffusion vs two baselines.

Metrics are computed only on coordinates deliberately hidden from a known
synthetic field. This is a reproducible prototype benchmark.
"""
import math
import random
from generator import generate_dataset
from scanner import scan_for_void_clusters
from tension import calculate_boundary_tension
from reconstructor import reconstruct_void


def _field(x, y):
    return 10.0 + 2.0 * math.sin(x / 3.0) + 1.5 * math.cos(y / 4.0)


def _metrics(predictions, truth):
    errors = [predictions[c] - truth[c] for c in truth if c in predictions]
    if not errors:
        return {"coverage": 0.0, "mae": None, "rmse": None}
    return {
        "coverage": round(len(errors) / len(truth), 4),
        "mae": round(sum(abs(e) for e in errors) / len(errors), 6),
        "rmse": round(math.sqrt(sum(e * e for e in errors) / len(errors)), 6),
    }


def run_benchmark(width=30, height=30, missing_fraction=0.2, seed=7):
    if not 0 < missing_fraction < 1:
        raise ValueError("missing_fraction must be between 0 and 1")
    rng = random.Random(seed)
    truth_field = {(x, y): _field(x, y)
                   for x in range(1, width + 1)
                   for y in range(1, height + 1)}
    hidden_count = max(1, int(width * height * missing_fraction))
    hidden = set(rng.sample(list(truth_field), hidden_count))
    observed = {
        c: {"status": "Recorded", "intensity": value}
        for c, value in truth_field.items() if c not in hidden
    }

    clusters = scan_for_void_clusters(
        observed, connectivity=4, bounds=(1, width, 1, height)
    )
    tension = calculate_boundary_tension(clusters, observed, void_level=0)
    report = reconstruct_void(
        clusters, tension, observed, pressure_threshold=0.01,
        threshold_ratio=0.5, inward_decay=0.95, void_level=0
    )
    ghost_predictions = {
        tuple(map(int, key.strip("()").split(", "))): item["intensity"]
        for key, item in report["audit_trail"].items()
    }

    # Baseline 1: one global observed mean.
    global_mean = sum(v["intensity"] for v in observed.values()) / len(observed)
    mean_predictions = {c: global_mean for c in hidden}

    # Baseline 2: iterative neighbor averaging from observed/reconstructed cells.
    neighbor_predictions = {}
    unresolved = set(hidden)
    values = {c: v["intensity"] for c, v in observed.items()}
    while unresolved:
        proposals = {}
        for x, y in unresolved:
            adjacent = [(x+dx, y+dy) for dx,dy in
                        ((1,0),(-1,0),(0,1),(0,-1)) if (x+dx,y+dy) in values]
            if adjacent:
                proposals[(x,y)] = sum(values[c] for c in adjacent) / len(adjacent)
        if not proposals:
            break
        for c, value in proposals.items():
            values[c] = value
            neighbor_predictions[c] = value
            unresolved.remove(c)

    truth = {c: truth_field[c] for c in hidden}
    return {
        "configuration": {
            "width": width, "height": height,
            "missing_fraction": missing_fraction, "seed": seed,
            "hidden_cells": len(hidden),
        },
        "methods": {
            "ghost_engine_3_tension_weighted": _metrics(ghost_predictions, truth),
            "global_mean_baseline": _metrics(mean_predictions, truth),
            "iterative_neighbor_average_baseline": _metrics(neighbor_predictions, truth),
        },
        "interpretation": (
            "Compare coverage, MAE and RMSE on this synthetic field. "
            "Lower MAE/RMSE is better; coverage indicates how many hidden cells "
            "a method supplied."
        ),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_benchmark(), indent=2))
