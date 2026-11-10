"""app_core.py - Streamlit-free helpers that wrap the Ghost Data Engine for the web app.

Everything here is plain Python/NumPy so it can be tested without Streamlit.
The engine modules (scanner, tension, reconstructor, engine_v3, ...) are used unchanged.
"""
import contextlib
import csv
import io
import os
import random
import tempfile

import numpy as np
import pandas as pd

import engine_v3 as E
from generator import SHAPE_BUILDERS, _carve_voids
from gde_spatiotemporal import reconstruct_3d
import calibrate_confidence as cc

def _cap(name, default):
    """Processing cap in cells; override with an environment variable on a bigger server."""
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


# Upload size (500 MB) is set in .streamlit/config.toml. These caps limit how many cells the
# pure-Python engine will repair in one request. Measured: ~1s at 13k cells, ~9s at 50k on one core.
MAX_CELLS_2D = _cap("GDE_MAX_CELLS_2D", 60000)
MAX_CELLS_3D = _cap("GDE_MAX_CELLS_3D", 12000)
MAX_TABLE_CELLS = _cap("GDE_MAX_TABLE_CELLS", 60000)
SENTINELS = {"", "nan", "NaN", "NAN", "null", "NULL", "missing", "-999", "-999.0", "bad", "gap",
             "NA", "N/A", "n/a"}


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ---------------------------------------------------------------------------
# 2D repair
# ---------------------------------------------------------------------------
def decode_bytes(data: bytes) -> str:
    """Decode an upload; utf-8 with or without BOM, then Windows-1252 (Excel exports)."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_grid_from_bytes(data: bytes, delimiter=","):
    """CSV bytes -> (grid dict, width, height). Uses the engine's own loader."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "upload.csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(decode_bytes(data))
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


def sample_table_text():
    """Small telemetry-style table with text columns and gaps in the numeric ones."""
    return ("Orbital_Regime,Anomaly_Type,Latitude_Deg,Longitude_Deg,Solar_Radiation_Wm2\n"
            "Low Earth Orbit (LEO),Ground Station Loss,-12.4502,45.1025,1361.2\n"
            "Sun-Synchronous,Cloud Coverage Blackout,52.52,13.405,1359.8\n"
            "Sun-Synchronous,Packet Drop,,,1360.15\n"
            "Sun-Synchronous,Sensor Signal Noise,52.61,13.521,\n"
            "Geostationary (GEO),Solar Flare Noise,0.012,-75.201,1385.1\n"
            "Geostationary (GEO),Missing Values,,-75.202,1362\n")


# ---------------------------------------------------------------------------
# Table mode: files with a header row and/or text columns
# ---------------------------------------------------------------------------
def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def looks_like_table(data: bytes, delimiter=","):
    """True when the first row holds a real text label (a header), not just numbers/gap markers."""
    first = next(csv.reader(io.StringIO(decode_bytes(data)), delimiter=delimiter), [])
    return any(c.strip() not in SENTINELS and not _is_number(c.strip()) for c in first)


def read_table(data: bytes, delimiter=","):
    """CSV bytes -> (DataFrame, numeric column names). Numeric columns become floats (gaps = NaN);
    text columns are kept exactly as written."""
    text = decode_bytes(data)
    skipped = 0
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False)
    except pd.errors.ParserError:
        # Malformed lines (more fields than the header): drop them, but count them so the app can say so.
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        width = len(rows[0]) if rows else 0
        kept = [r for r in rows if len(r) <= width]
        skipped = len(rows) - len(kept)
        buf = io.StringIO()
        csv.writer(buf, delimiter=delimiter, lineterminator="\n").writerows(kept)
        df = pd.read_csv(io.StringIO(buf.getvalue()), sep=delimiter, dtype=str, keep_default_na=False)
    df.attrs["skipped_lines"] = skipped
    df.columns = [str(c).strip() or f"col_{i + 1}" for i, c in enumerate(df.columns)]
    numeric = []
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        gap = s.isin(SENTINELS)
        vals = pd.to_numeric(s.where(~gap), errors="coerce")
        if vals.notna().any() and not (vals.isna() & ~gap).any():
            df[c] = vals.replace([np.inf, -np.inf], np.nan)
            numeric.append(c)
    return df, numeric


def repair_table(df, numeric_cols, inward_decay=1.0, blend=False):
    """Repair every numeric column of `df` with the real engine, one column at a time.

    Each numeric column is a 1-wide grid whose rows are the records, so a gap is a void along the
    column and is filled from the recorded values above/below it. Columns are never mixed (latitude
    is never filled from a radiation value). Text columns and recorded cells are never changed.
    The post-fill blends were tuned on 2D grids, so they are off by default here: on a single
    column they pull values toward far-away rows.
    """
    n = len(df)
    out = df.copy()
    filled = pd.DataFrame(False, index=df.index, columns=df.columns)
    audit, per_column, skipped = [], [], []
    tot = {"voids": 0, "reconstructed": 0, "stalled": 0}
    confs = []
    for col in numeric_cols:
        series = df[col].astype(float)
        grid = {(1, n - i): {"status": "Recorded", "intensity": float(v)}
                for i, v in enumerate(series) if np.isfinite(v)}
        missing = n - len(grid)
        if missing == 0:
            continue
        if not grid:
            skipped.append(col)
            continue
        voids, report = _quiet(E.repair_grid, grid, 1, n, 4, inward_decay, 0.0, "accurate", True)
        if blend:
            E.apply_blends(grid, report, 1, n, knn_blend=E.DEFAULT_KNN_BLEND, row_blend=E.DEFAULT_ROW_BLEND)
        for key, r in report["audit_trail"].items():
            x, y = (int(v) for v in key.strip("()").split(","))
            i = n - y
            out.loc[out.index[i], col] = r["intensity"]
            filled.loc[filled.index[i], col] = True
            confs.append(r["confidence"])
            audit.append({"row": i, "column": col, "void_id": r["void_id"], "value": r["intensity"],
                          "wave": r["wave"], "confidence (heuristic)": r["confidence"],
                          "sources": r["walls_used"], "method": r["interpolation"]})
        s = report["summary"]
        tot["voids"] += s["total_voids"]
        tot["reconstructed"] += s["cells_reconstructed"]
        tot["stalled"] += s["cells_stalled"]
        per_column.append({"column": col, "missing": missing, "filled": s["cells_reconstructed"],
                           "not filled": s["cells_stalled"]})
    summary = {"total_voids": tot["voids"], "cells_reconstructed": tot["reconstructed"],
               "cells_stalled": tot["stalled"] + sum(int(df[c].isna().sum()) for c in skipped),
               "mean_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0}
    audit.sort(key=lambda d: (d["column"], d["row"]))
    return {"repaired": out, "filled": filled, "summary": summary, "audit": audit,
            "per_column": per_column, "skipped": skipped, "any_missing": bool(per_column or skipped)}


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


# ---------------------------------------------------------------------------
# Live accuracy test: hide known cells, repair, score against the truth
# ---------------------------------------------------------------------------
def load_complete_matrix(data: bytes, delimiter=","):
    """Complete numeric CSV bytes -> (h, w) float matrix. Raises ValueError if not a clean numeric grid."""
    text = decode_bytes(data)
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delimiter) if r]
    if not rows:
        raise ValueError("The file is empty.")
    try:
        m = np.array([[float(x) for x in r] for r in rows], dtype=float)
    except ValueError:
        raise ValueError("This test needs a complete, all-numeric CSV with no header and no blanks.")
    if m.ndim != 2 or not np.isfinite(m).all():
        raise ValueError("This test needs a complete CSV (every cell a number, rows of equal length).")
    return m


def make_hide_mask(shape, scatter=0.05, blocks=6, bmin=30, bmax=90, seed=0, custom=None):
    """True = hidden. Scattered single cells + random multi-day outages (+ one optional custom block).

    custom = (row_from, row_to, col_from, col_to), 1-based inclusive.
    """
    h, w = shape
    rng = np.random.default_rng(int(seed))
    mask = rng.random(shape) < scatter
    bmax = max(1, min(int(bmax), w - 1))
    bmin = max(1, min(int(bmin), bmax))
    for _ in range(int(blocks)):
        bh = int(rng.integers(1, min(3, h) + 1))
        bw = int(rng.integers(bmin, bmax + 1))
        r = int(rng.integers(0, h - bh + 1))
        c = int(rng.integers(0, w - bw + 1))
        mask[r:r + bh, c:c + bw] = True
    if custom:
        r0, r1, c0, c1 = custom
        mask[max(0, r0 - 1):min(h, r1), max(0, c0 - 1):min(w, c1)] = True
    mask[mask.all(axis=1)] = False          # keep some context in every row
    mask[:, mask.all(axis=0)] = False       # ... and every column
    return mask


def _to_grid(truth, mask):
    h, w = truth.shape
    return {(x + 1, h - i): {"status": "Recorded", "intensity": float(truth[i, x])}
            for i in range(h) for x in range(w) if not mask[i, x]}


def _base_row_linear(truth, mask):
    out = truth.copy()
    xs = np.arange(truth.shape[1])
    for i in range(truth.shape[0]):
        ok = ~mask[i]
        if ok.any() and mask[i].any():
            out[i, mask[i]] = np.interp(xs[mask[i]], xs[ok], truth[i, ok])
    return out


def _base_same_col_mean(truth, mask):
    out = truth.copy()
    vals = np.where(mask, np.nan, truth)
    with np.errstate(all="ignore"):
        col = np.nanmean(vals, axis=0)
    col = np.where(np.isnan(col), np.nanmean(vals), col)
    ii, jj = np.nonzero(mask)
    out[ii, jj] = col[jj]
    return out


def accuracy_test(truth, mask, blend=True):
    """Repair `truth` with `mask` hidden. Methods only read cells where ~mask. Returns a result dict."""
    import time
    h, w = truth.shape
    t0 = time.time()
    res = repair_matrix(_to_grid(truth, mask), w, h, blend=blend)
    gde_sec = time.time() - t0
    repaired = np.where(mask, res["repaired"], truth)
    not_filled = mask & ~res["filled_mask"]
    preds = {"Ghost Data Engine": repaired}
    secs = {"Ghost Data Engine": gde_sec}
    for name, fn in (("Linear interpolation (along each row)", _base_row_linear),
                     ("Column mean (same position in other rows)", _base_same_col_mean)):
        t0 = time.time()
        preds[name] = fn(truth, mask)
        secs[name] = time.time() - t0
    rows = []
    for name, p in preds.items():
        sel = mask & ~not_filled if name == "Ghost Data Engine" else mask
        err = p[sel] - truth[sel]
        rows.append({"Method": name, "Mean abs. error": float(np.mean(np.abs(err))) if err.size else float("nan"),
                     "RMSE": float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan"),
                     "Seconds": secs[name]})
    return {"repaired": repaired, "filled_mask": res["filled_mask"], "not_filled": not_filled,
            "scores": rows, "summary": res["summary"], "n_hidden": int(mask.sum()),
            "n_filled": int((mask & res["filled_mask"]).sum())}


# ---------------------------------------------------------------------------
# Labelled grids: first row = headings (e.g. "Jan 01"), first column = row labels (e.g. a year)
# ---------------------------------------------------------------------------
def _num(text):
    try:
        float(text.strip())
        return True
    except ValueError:
        return False


def _read_rows(data: bytes, delimiter=","):
    return [r for r in csv.reader(io.StringIO(decode_bytes(data)), delimiter=delimiter) if r]


def detect_labelled(data: bytes, delimiter=","):
    """True if the file looks like a wide numeric grid with a heading row and a label column."""
    rows = _read_rows(data, delimiter)
    if len(rows) < 3 or len(rows[0]) < 12:
        return False
    head = rows[0][1:]
    if sum(_num(c) for c in head if c.strip()) > 0.5 * max(1, len(head)):
        return False                       # headings are numbers: not a heading row
    body = rows[1:]
    if len(rows[0]) <= len(body):
        return False                       # more rows than columns looks like a normal table
    ok = total = 0
    for r in body[:20]:
        for c in r[1:]:
            if c.strip() and c.strip().lower() not in {"nan", "null", "missing", "bad", "gap", "-999"}:
                total += 1
                ok += _num(c)
    return total > 0 and ok / total > 0.98


def _unique(labels):
    seen, out = {}, []
    for lab in labels:
        n = seen.get(lab, 0)
        seen[lab] = n + 1
        out.append(lab if n == 0 else f"{lab} ({n + 1})")
    return out


def split_labelled(data: bytes, delimiter=","):
    """Return (corner, column_labels, row_labels, numeric_bytes) for a labelled grid."""
    rows = _read_rows(data, delimiter)
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    corner = rows[0][0]
    cols = _unique([c.strip() or f"col_{i + 1}" for i, c in enumerate(rows[0][1:])])
    labels = _unique([r[0].strip() or f"row_{i + 1}" for i, r in enumerate(rows[1:])])
    buf = io.StringIO()
    csv.writer(buf, delimiter=delimiter, lineterminator="\n").writerows([r[1:] for r in rows[1:]])
    return corner, cols, labels, buf.getvalue().encode("utf-8")


def load_complete_with_labels(data: bytes, delimiter=","):
    """Complete numeric CSV (labelled or plain) -> (matrix, row_labels, col_labels)."""
    if detect_labelled(data, delimiter):
        _corner, cols, labels, body = split_labelled(data, delimiter)
        return load_complete_matrix(body, delimiter), labels, cols
    m = load_complete_matrix(data, delimiter)
    return m, [f"Row {i + 1}" for i in range(m.shape[0])], [str(j + 1) for j in range(m.shape[1])]
