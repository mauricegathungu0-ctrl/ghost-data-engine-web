"""app_core.py - Streamlit-free helpers that wrap the Ghost Data Engine for the web app.

Everything here is plain Python/NumPy so it can be tested without Streamlit.
The engine modules (scanner, tension, reconstructor, engine_v3, ...) are used unchanged.
"""
import contextlib
import io
import os
import random
import tempfile

import numpy as np

import engine_v3 as E
from generator import SHAPE_BUILDERS, _carve_voids
from gde_spatiotemporal import reconstruct_3d
import calibrate_confidence as cc

MAX_CELLS_2D = 6000          # pure-Python engine: keep the hosted app responsive
MAX_CELLS_3D = 12000


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ---------------------------------------------------------------------------
# 2D repair
# ---------------------------------------------------------------------------
def load_grid_from_bytes(data: bytes, delimiter=","):
    """CSV bytes -> (grid dict, width, height). Uses the engine's own loader."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "upload.csv")
        with open(path, "wb") as fh:
            fh.write(data)
        grid, w, h = _quiet(E.load_csv_with_gaps, path, delimiter)
    return grid, w, h


def grid_to_matrix(grid, w, h):
    """Engine grid -> numpy (h, w) with NaN for missing. Row 0 = top (CSV order)."""
    m = np.full((h, w), np.nan)
    for (x, y), info in grid.items():
        m[h - y, x - 1] = info["intensity"]
    return m


def repair_matrix(grid, w, h, connectivity=4, inward_decay=1.0, blend=True,
                  mode="accurate", relax=True):
    """Run scan + tension + reconstruct (+ blends). Returns a result dict."""
    voids, report = _quiet(E.repair_grid, grid, w, h, connectivity, inward_decay, 0.0, mode, relax)
    original = grid_to_matrix(grid, w, h)
    if not voids:
        return {"voids": [], "report": None, "original": original, "repaired": original.copy(),
                "filled_mask": np.zeros_like(original, dtype=bool), "summary": None}
    if blend:
        E.apply_blends(grid, report, w, h, knn_blend=E.DEFAULT_KNN_BLEND, row_blend=E.DEFAULT_ROW_BLEND)
    repaired = original.copy()
    filled = np.zeros(original.shape, dtype=bool)
    for key, row in report["audit_trail"].items():
        x, y = (int(v) for v in key.strip("()").split(","))
        repaired[h - y, x - 1] = row["intensity"]
        filled[h - y, x - 1] = True
    return {"voids": voids, "report": report, "original": original, "repaired": repaired,
            "filled_mask": filled, "summary": report["summary"]}


def audit_rows(report, w, h):
    """Flat list of dicts (one per reconstructed cell) suitable for a DataFrame."""
    rows = []
    for key, r in report["audit_trail"].items():
        x, y = (int(v) for v in key.strip("()").split(","))
        rows.append({"row": h - y + 1, "col": x, "void_id": r["void_id"], "value": r["intensity"],
                     "wave": r["wave"], "confidence (heuristic)": r["confidence"],
                     "sources": r["walls_used"], "method": r["interpolation"]})
    rows.sort(key=lambda d: (d["void_id"], d["wave"], d["row"], d["col"]))
    return rows


def matrix_to_csv(matrix):
    lines = []
    for row in matrix:
        lines.append(",".join("NaN" if np.isnan(v) else repr(float(v)) for v in row))
    return "\n".join(lines) + "\n"


def sample_csv_text():
    return ("12.5,13.1,missing,14.2\n11.9,bad,12.8,13.0\n10.1,10.5,11.2,NaN\n")


# ---------------------------------------------------------------------------
# 2D demo with ground truth
# ---------------------------------------------------------------------------
def smooth_field(w, h, kind="waves"):
    y, x = np.mgrid[0:h, 0:w]
    if kind == "plane":
        return 3.0 * x + 2.0 * y + 5.0
    if kind == "bump":
        return 100.0 * np.exp(-(((x - w / 2) / (w / 3)) ** 2 + ((y - h / 2) / (h / 3)) ** 2)) + 20.0
    return 100.0 + 15.0 * np.sin(x / 4.0) + 10.0 * np.cos(y / 5.0)


def run_demo_2d(width=40, height=40, seed=1, num_voids=5, void_size=6, shapes=None,
                field="waves", connectivity=4, inward_decay=1.0, blend=True):
    shapes = shapes or list(SHAPE_BUILDERS)
    truth = smooth_field(width, height, field)
    rng = random.Random(seed)
    cells, _ = _carve_voids(rng, width, height, num_voids, void_size, shapes, True)
    hidden = np.zeros((height, width), dtype=bool)
    grid = {}
    for yy in range(height):
        for xx in range(width):
            x, y = xx + 1, height - yy          # engine coords: y counts up from the bottom row
            if (x, y) in cells:
                hidden[yy, xx] = True
            else:
                grid[(x, y)] = {"status": "Recorded", "intensity": float(truth[yy, xx])}
    res = repair_matrix(grid, width, height, connectivity, inward_decay, blend)
    res["truth"], res["hidden"] = truth, hidden
    if res["report"] is not None:
        err = np.abs(res["repaired"] - truth)[res["filled_mask"]]
        res["mae"] = float(err.mean()) if err.size else None
        res["max_err"] = float(err.max()) if err.size else None
        res["value_range"] = float(truth.max() - truth.min())
    return res


# ---------------------------------------------------------------------------
# 3D demo
# ---------------------------------------------------------------------------
def run_demo_3d(size=16, frames=6, seed=3, num_clouds=3, max_fraction=0.4,
                temporal_decay=0.5, connectivity=4, inward_decay=0.9, max_waves=None):
    rng = random.Random(seed)
    truth = cc.make_truth(rng, frames, size, size, 1.5)
    mask = cc.make_cloud_mask(rng, frames, size, size, num_clouds, max_fraction)
    masked = cc.mask_grid(truth, mask)
    rep = _quiet(reconstruct_3d, masked, spatial_connectivity=connectivity,
                 temporal_decay=temporal_decay, inward_decay=inward_decay, max_waves=max_waves)
    t = np.array(truth, dtype=float)
    hidden = np.array(mask, dtype=bool)
    pred = np.array([[[np.nan if v is None else v for v in r] for r in f] for f in rep["grid"]], dtype=float)
    obs = np.where(hidden, np.nan, t)
    filled = hidden & np.isfinite(pred)
    err = np.abs(pred - t)[filled]
    return {"truth": t, "hidden": hidden, "observed": obs, "predicted": pred, "report": rep,
            "mae": float(err.mean()) if err.size else None,
            "p95": float(np.percentile(err, 95)) if err.size else None,
            "range": float(t.max() - t.min())}


def run_calibration(seeds=8, size=16, frames=6):
    return cc.run(seeds=seeds, size=size, frames=frames)
