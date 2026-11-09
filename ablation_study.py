"""
ablation_study.py - Ghost Data Engine ablation on identical masked voids.

Reuses real_benchmark.py for images, masks and the grid builder, and the
unmodified pure-Python engine (scanner / tension / reconstructor). Every
configuration sees exactly the same hidden-pixel masks and is scored against
the real pixels on those masks.

Groups
------
  A. Refinement   : waves-only (refine_max_iter=0)  vs  waves + thin-plate
  B. Tension      : tension weights ON vs OFF, with and without refinement
                    (OFF = every perimeter wall gets weight 1.0; cells_pressed
                    and all other report fields are left untouched)
  C. Connectivity : 4-connected vs 8-connected void scanning

Runtime covers scan + tension + reconstruct for each configuration (the shared
grid-dict build is excluded).

Usage:
    python ablation_study.py
    python ablation_study.py --size 128 --seeds 8 --void-size 14 --markdown ablation.md

Requires numpy, scipy and scikit-image only because real_benchmark.py imports
them (image loading); the engine itself stays dependency-free.
"""
import argparse
import contextlib
import copy
import io
import statistics
import time

import numpy as np

import real_benchmark as rb
from scanner import scan_for_void_clusters
from tension import calculate_boundary_tension
from reconstructor import reconstruct_void

INWARD_DECAY = 0.85   # same as ghost_accurate in real_benchmark.py
VOID_LEVEL = 0.0


def _uniform_tension(report):
    """Return a copy of a tension report with every wall weight set to 1.0."""
    flat = copy.deepcopy(report)
    for void_report in flat.values():
        for wall in void_report["edge_tension"].values():
            wall["weight"] = 1.0
    return flat


def run_config(grid, mask, connectivity=4, tension_on=True, refine=True):
    """Run the engine once; return (predictions for img[mask], seconds)."""
    h, w = mask.shape
    extra = {"relax": False}          # keep 3.1 meaning of these ablations (relaxation is a separate option)
    if not refine:
        extra["refine_max_iter"] = 0
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # silence engine logs
        clusters = scan_for_void_clusters(grid, connectivity=connectivity,
                                          bounds=(1, w, 1, h))
        tension = calculate_boundary_tension(clusters, grid, void_level=VOID_LEVEL)
        if not tension_on:
            tension = _uniform_tension(tension)
        report = reconstruct_void(clusters, tension, grid, pressure_threshold=0.0,
                                  inward_decay=INWARD_DECAY, void_level=VOID_LEVEL,
                                  mode="accurate", **extra)
    dt = time.perf_counter() - t0

    trail = report["audit_trail"]
    ys, xs = np.nonzero(mask)
    pred = np.full(len(ys), np.nan)
    for i, (y, x) in enumerate(zip(ys, xs)):
        item = trail.get(str((int(x) + 1, int(y) + 1)))
        if item is not None:
            pred[i] = item["intensity"]
    return pred, dt


# (group, label, kwargs for run_config)
CONFIGS = [
    ("A. Refinement",   "waves only (no thin-plate)",        dict(refine=False)),
    ("A. Refinement",   "waves + thin-plate refinement",     dict(refine=True)),
    ("B. Tension",      "tension ON,  waves only",           dict(tension_on=True,  refine=False)),
    ("B. Tension",      "tension OFF, waves only",           dict(tension_on=False, refine=False)),
    ("B. Tension",      "tension ON,  waves + thin-plate",   dict(tension_on=True,  refine=True)),
    ("B. Tension",      "tension OFF, waves + thin-plate",   dict(tension_on=False, refine=True)),
    ("C. Connectivity", "4-connectivity",                    dict(connectivity=4)),
    ("C. Connectivity", "8-connectivity",                    dict(connectivity=8)),
]


def run(size, seeds, num_voids, void_size, images):
    rows = []
    for name in images:
        img = rb.load_image(name, size)
        for seed in range(seeds):
            mask = rb.carve_blob_mask(size, num_voids, void_size, seed)
            truth = img[mask]
            # Hidden pixels are removed BEFORE the grid is built: engine never sees them.
            grid = rb._grid_dict(np.where(mask, np.nan, img), mask)
            for group, label, kw in CONFIGS:
                pred, dt = run_config(grid, mask, **kw)
                covered = ~np.isnan(pred)
                err = pred[covered] - truth[covered]
                rows.append({
                    "image": name, "seed": seed, "group": group, "label": label,
                    "hidden": int(mask.sum()),
                    "coverage": float(covered.mean()),
                    "mae": float(np.mean(np.abs(err))) if err.size else None,
                    "ms": dt * 1000.0,
                })
    return rows


def _agg(rs):
    maes = [r["mae"] for r in rs if r["mae"] is not None]
    return {
        "mae": statistics.mean(maes) if maes else float("nan"),
        "sd": statistics.pstdev(maes) if len(maes) > 1 else 0.0,
        "cov": statistics.mean(r["coverage"] for r in rs),
        "ms": statistics.mean(r["ms"] for r in rs),
    }


def _table(rows, title):
    lines = [f"### {title}\n", "| Group | Setting | MAE (mean ± sd) | Coverage | Time (ms) |",
             "|---|---|---|---|---|"]
    for group, label, _ in CONFIGS:
        rs = [r for r in rows if r["label"] == label and r["group"] == group]
        if not rs:
            continue
        a = _agg(rs)
        lines.append(f"| {group} | {label} | {a['mae']:.3f} ± {a['sd']:.3f} | "
                     f"{a['cov']:.0%} | {a['ms']:.1f} |")
    lines.append("")
    return "\n".join(lines)


def to_markdown(rows, cfg):
    out = [f"Ablation study: {cfg['size']}x{cfg['size']} px, {cfg['seeds']} seeds, "
           f"{cfg['num_voids']} blob voids/seed (max_void_size={cfg['void_size']}). "
           f"Identical masks for every setting. Intensity 0-255; lower MAE is better. "
           f"Time = scan + tension + reconstruct.\n"]
    for image in dict.fromkeys(r["image"] for r in rows):
        hid = statistics.mean(r["hidden"] for r in rows if r["image"] == image)
        out.append(_table([r for r in rows if r["image"] == image],
                          f"{image} (~{hid:.0f} hidden px per run)"))
    out.append(_table(rows, "All images pooled"))
    out.append("Differences smaller than the standard deviation are within "
               "seed-to-seed noise and should not be read as wins.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--num-voids", type=int, default=6)
    ap.add_argument("--void-size", type=int, default=8)
    ap.add_argument("--images", nargs="+", choices=list(rb.IMAGES), default=None)
    ap.add_argument("--markdown", default=None, help="also write the tables to this file")
    a = ap.parse_args()

    cfg = {"size": a.size, "seeds": a.seeds, "num_voids": a.num_voids,
           "void_size": a.void_size}
    rows = run(a.size, a.seeds, a.num_voids, a.void_size, a.images or list(rb.IMAGES))
    md = to_markdown(rows, cfg)
    print(md)
    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as fh:
            fh.write(md)


if __name__ == "__main__":
    main()
