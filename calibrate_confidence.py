"""
calibrate_confidence.py - does gde_spatiotemporal's confidence score track real error?

For each of N seeds this script
  1. builds a synthetic 3D ground-truth field truth[t][y][x] (smooth in space and
     time plus noise),
  2. carves drifting cloud-shaped voids out of it (the mask spans several frames),
  3. hands ONLY the masked grid to gde_spatiotemporal.reconstruct_3d,
  4. compares every reconstructed cell's heuristic confidence with its true
     absolute error |predicted - actual|.

Cells are pooled over all seeds, grouped into confidence buckets, and each bucket
reports its empirical 95th-percentile error: "when the engine says confidence is
in this bucket, 95% of its cells were off by no more than this much".
Unresolved cells (the engine refused to fill them) are counted, never scored.

The ground truth is only touched inside the scoring step, after the engine has
returned. test_calibrate_confidence.py proves the engine never sees it.

Pure standard library. Usage:
    python calibrate_confidence.py
    python calibrate_confidence.py --seeds 30 --size 24 --frames 8 \
        --markdown calibration.md --json calibration.json
"""
import argparse
import contextlib
import io
import json
import math
import random

from gde_spatiotemporal import reconstruct_3d

MIN_RELIABLE_CELLS = 30   # buckets with fewer cells get a "*" next to their P95


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def make_truth(rng, frames, height, width, noise=1.5):
    """Smooth 3D field (random waves in x, y, t) plus Gaussian noise."""
    a1, a2 = rng.uniform(10, 25), rng.uniform(8, 20)
    fx, fy = rng.uniform(0.10, 0.30), rng.uniform(0.10, 0.30)
    ft1, ft2 = rng.uniform(0.10, 0.35), rng.uniform(0.10, 0.35)
    p1, p2 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
    base = rng.uniform(80, 140)
    return [[[base + a1 * math.sin(fx * x + ft1 * t + p1)
              + a2 * math.cos(fy * y - ft2 * t + p2)
              + rng.gauss(0.0, noise)
              for x in range(width)] for y in range(height)] for t in range(frames)]


def make_cloud_mask(rng, frames, height, width, num_clouds=3, max_fraction=0.5):
    """True = hidden. Each cloud is an ellipse that drifts across several frames."""
    mask = [[[False] * width for _ in range(height)] for _ in range(frames)]
    total = frames * height * width
    for _ in range(num_clouds):
        cx, cy = rng.uniform(0, width - 1), rng.uniform(0, height - 1)
        vx, vy = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        rx = rng.uniform(2.0, max(2.5, 0.28 * width))
        ry = rng.uniform(2.0, max(2.5, 0.28 * height))
        t0 = rng.randrange(frames)
        t1 = min(frames, t0 + rng.randint(2, frames))
        trial = [[row[:] for row in frame] for frame in mask]
        for t in range(t0, t1):
            mx, my = cx + vx * (t - t0), cy + vy * (t - t0)
            for y in range(height):
                for x in range(width):
                    if ((x - mx) / rx) ** 2 + ((y - my) / ry) ** 2 <= 1.0:
                        trial[t][y][x] = True
        if sum(v for f in trial for r in f for v in r) <= max_fraction * total:
            mask = trial
    return mask


def mask_grid(truth, mask):
    """Copy of truth with every hidden cell replaced by None."""
    return [[[None if mask[t][y][x] else truth[t][y][x]
              for x in range(len(truth[0][0]))]
             for y in range(len(truth[0]))]
            for t in range(len(truth))]


# ---------------------------------------------------------------------------
# Statistics (no numpy)
# ---------------------------------------------------------------------------
def percentile(values, q):
    """Linear-interpolated percentile, q in [0, 100]. None for an empty list."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rank correlation; None if undefined (n<3 or a constant series)."""
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def evaluate_seed(seed, size=24, frames=8, noise=1.5, num_clouds=3,
                  max_fraction=0.5, **engine_kwargs):
    """Run one seed. Returns {"cells": [(confidence, abs_error, wave)], "missing", "unresolved"}."""
    rng = random.Random(seed)
    truth = make_truth(rng, frames, size, size, noise)
    mask = make_cloud_mask(rng, frames, size, size, num_clouds, max_fraction)
    masked = mask_grid(truth, mask)            # <- the ONLY thing the engine gets
    report = reconstruct_3d(masked, **engine_kwargs)

    # Scoring: ground truth is read here, after reconstruction is finished.
    cells = []
    for row in report["audit_trail"].values():
        t, y, x = row["coordinate"]
        cells.append((row["confidence"], abs(row["intensity"] - truth[t][y][x]), row["wave"]))
    return {"cells": cells,
            "missing": report["summary"]["missing_cells"],
            "unresolved": report["summary"]["cells_unresolved"]}


def bucketize(cells, bucket_width=0.1):
    """Group (confidence, error, wave) tuples into [lo, hi) buckets (last includes 1.0)."""
    n = int(round(1.0 / bucket_width))
    buckets = []
    for i in range(n):
        lo, hi = round(i * bucket_width, 6), round((i + 1) * bucket_width, 6)
        last = i == n - 1
        errs = [e for c, e, _ in cells if lo <= c < hi or (last and c == hi)]
        confs = [c for c, e, _ in cells if lo <= c < hi or (last and c == hi)]
        buckets.append({
            "low": lo, "high": hi, "cells": len(errs),
            "mean_confidence": sum(confs) / len(confs) if confs else None,
            "mean_error": sum(errs) / len(errs) if errs else None,
            "median_error": percentile(errs, 50),
            "p95_error": percentile(errs, 95),
            "max_error": max(errs) if errs else None,
        })
    return buckets[::-1]          # highest confidence first


def run(seeds=30, base_seed=0, bucket_width=0.1, **kwargs):
    all_cells, missing, unresolved = [], 0, 0
    for i in range(seeds):
        with contextlib.redirect_stdout(io.StringIO()):
            res = evaluate_seed(base_seed + i, **kwargs)
        all_cells.extend(res["cells"])
        missing += res["missing"]
        unresolved += res["unresolved"]
    errs = [e for _, e, _ in all_cells]
    return {
        "config": dict(seeds=seeds, base_seed=base_seed, **kwargs),
        "seeds": seeds,
        "missing_cells": missing,
        "reconstructed_cells": len(all_cells),
        "unresolved_cells": unresolved,
        "overall_mean_error": sum(errs) / len(errs) if errs else None,
        "overall_p95_error": percentile(errs, 95),
        "spearman_confidence_vs_error": spearman([c for c, _, _ in all_cells], errs),
        "buckets": bucketize(all_cells, bucket_width),
    }


def _f(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def to_markdown(result):
    cfg = result["config"]
    rho = result["spearman_confidence_vs_error"]
    lines = [
        f"Confidence calibration: {result['seeds']} seeds, "
        f"{cfg.get('size', 24)}x{cfg.get('size', 24)} px x {cfg.get('frames', 8)} frames, "
        f"drifting cloud masks. Error = |predicted - actual| in intensity units.\n",
        f"- Missing cells: {result['missing_cells']} | reconstructed: "
        f"{result['reconstructed_cells']} | unresolved (not scored): {result['unresolved_cells']}",
        f"- Overall mean error {_f(result['overall_mean_error'])}, "
        f"overall 95th-percentile error {_f(result['overall_p95_error'])}",
        f"- Spearman rho(confidence, error) = {_f(rho)} "
        "(negative = higher confidence goes with lower error)\n",
        "| Confidence | Cells | Mean conf | Mean error | Median | P95 error | Max |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in result["buckets"]:
        flag = "*" if 0 < b["cells"] < MIN_RELIABLE_CELLS else ""
        lines.append(f"| {b['low']:.1f}-{b['high']:.1f} | {b['cells']} | "
                     f"{_f(b['mean_confidence'])} | {_f(b['mean_error'])} | "
                     f"{_f(b['median_error'])} | {_f(b['p95_error'])}{flag} | "
                     f"{_f(b['max_error'])} |")
    populated = [b for b in result["buckets"] if b["cells"] >= MIN_RELIABLE_CELLS]
    p95s = [b["p95_error"] for b in populated]
    monotone = all(p95s[i] <= p95s[i + 1] for i in range(len(p95s) - 1)) if len(p95s) > 1 else None
    lines.append("")
    lines.append(f"`*` = fewer than {MIN_RELIABLE_CELLS} cells: P95 is not reliable.")
    if monotone is True:
        lines.append("Reading: P95 error rises as confidence falls, so the score ranks cells sensibly.")
    elif monotone is False:
        lines.append("Reading: P95 error is NOT monotone in confidence across well-populated "
                     "buckets, so the score is not a dependable ranking of error.")
    lines.append("The score is a heuristic, not a probability; use the P95 column, "
                 "not the confidence value itself, as the error margin.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=24, help="grid width = height")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--num-clouds", type=int, default=3)
    ap.add_argument("--max-fraction", type=float, default=0.5,
                    help="max share of cells hidden by clouds")
    ap.add_argument("--bucket-width", type=float, default=0.1)
    ap.add_argument("--connectivity", type=int, choices=(4, 8), default=4)
    ap.add_argument("--temporal-decay", type=float, default=0.5)
    ap.add_argument("--inward-decay", type=float, default=0.9)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.seeds < 1 or a.size < 3 or a.frames < 1 or not 0 < a.bucket_width <= 1:
        ap.error("need seeds>=1, size>=3, frames>=1 and 0 < bucket-width <= 1")

    result = run(seeds=a.seeds, base_seed=a.base_seed, bucket_width=a.bucket_width,
                 size=a.size, frames=a.frames, noise=a.noise,
                 num_clouds=a.num_clouds, max_fraction=a.max_fraction,
                 spatial_connectivity=a.connectivity,
                 temporal_decay=a.temporal_decay, inward_decay=a.inward_decay)
    md = to_markdown(result)
    print(md)
    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as fh:
            fh.write(md)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
