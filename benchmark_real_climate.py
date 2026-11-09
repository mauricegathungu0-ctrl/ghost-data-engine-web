"""
benchmark_real_climate.py - GDE 3.1 Spatiotemporal vs temporal / scipy interpolation
on a gridded climate or vegetation time series under realistic cloud-cover masks.

What it does
  1. Loads a time-series grid with shape (time, y, x):
       * NetCDF   (.nc/.nc4)  e.g. an ERA5 2 m temperature or total precipitation file
       * GeoTIFF  (.tif/.tiff) multi-band stack, e.g. Sentinel-2 NDVI with one band per
                  date, or a directory of single-band GeoTIFFs sorted by file name
       * NumPy    (.npy / .npz) array of shape (time, y, x)
     Cells that are already NaN / fill value / nodata in the source are "invalid"
     and are never hidden, scored or counted.
  2. Hides 20%-40% of the VALID cells with cloud-like masks: spatially smooth blobs that
     drift between frames (see make_cloud_mask), thresholded so that exactly the
     requested fraction of valid cells disappears.
  3. Hands each method ONLY the masked array (hidden cells = NaN). The hidden truth is
     touched only afterwards, by the scorer.
  4. Compares
       gde_3d            gde_spatiotemporal.reconstruct_3d (pure-Python engine, defaults)
       gde_3d_bounded    the same engine with max_waves=G, so cells deeper than G waves
                         inside a void are declined instead of guessed
       linear_temporal   per-pixel linear interpolation along time; cells with no valid
                         observation on BOTH sides in time are declined (no extrapolation)
       scipy_linear3d    scipy LinearNDInterpolator over (t, y, x); NaN outside the hull
                         is declined
       scipy_nearest3d   scipy nearest neighbour over (t, y, x); never declines
     on MAE, RMSE, runtime and the share of cells flagged unresolved.

Ungrounded cells
  A hidden cell is "ungrounded" when its distance to the nearest recorded cell in
  (t, y, x) index space exceeds --ground-radius (default 4). Ideally a method declines
  these instead of inventing a value. The report gives
     ungrounded flagged %   of ungrounded cells the method left unresolved (higher = safer)
     grounded flagged %     of grounded cells it left unresolved (lower = more useful)

Honesty notes
  * Without --data the script falls back to a labelled SYNTHETIC climate-like field.
    Those numbers are NOT real-data results; use --require-real to forbid the fallback.
  * "MAE common" / "RMSE common" score every method on the same cells (those resolved by
    all methods in that run) so a method cannot look better just by skipping hard cells.
  * gde_spatiotemporal.py itself stays pure Python; numpy / scipy / netCDF4 / xarray /
    rasterio / tifffile are used here only, and each reader is optional.

Usage
    python benchmark_real_climate.py --data era5_t2m.nc --var t2m
    python benchmark_real_climate.py --data ndvi_stack.tif --max-frames 10 --crop 48
    python benchmark_real_climate.py --data ndvi_dir/ --fractions 0.2 0.3 0.4 --seeds 5
    python benchmark_real_climate.py                      # labelled synthetic fallback
    python benchmark_real_climate.py --json out.json --markdown out.md
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from gde_spatiotemporal import reconstruct_3d

DEFAULT_FRACTIONS = (0.20, 0.30, 0.40)
MIN_FRACTION, MAX_FRACTION = 0.20, 0.40


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _pick_variable(names, var, path):
    if var:
        if var not in names:
            raise ValueError(f"variable {var!r} not found in {path}; available: {sorted(names)}")
        return var
    skip = {"time", "latitude", "longitude", "lat", "lon", "x", "y", "expver", "number",
            "valid_time", "level", "spatial_ref", "crs"}
    cands = [n for n in names if n.lower() not in skip and not n.lower().endswith("_bnds")]
    if not cands:
        raise ValueError(f"no data variable found in {path}")
    return cands[0]


def _to_3d(arr, source):
    arr = np.squeeze(np.asarray(arr, dtype=float))
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"{source}: expected (time, y, x) after squeezing, got shape {arr.shape}")
    return arr


def _load_netcdf(path, var):
    errors = []
    try:  # xarray decodes scale/offset/fill values for us
        import xarray as xr
        with xr.open_dataset(path) as ds:
            name = _pick_variable(list(ds.data_vars), var, path)
            return _to_3d(ds[name].values, path), f"netcdf:{name} (xarray)"
    except ImportError as e:
        errors.append(str(e))
    try:
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            name = _pick_variable(list(ds.variables), var, path)
            data = ds.variables[name][:]
            data = np.ma.filled(np.ma.asarray(data, dtype=float), np.nan)
            return _to_3d(data, path), f"netcdf:{name} (netCDF4)"
    except ImportError as e:
        errors.append(str(e))
    try:  # NetCDF3 classic only
        from scipy.io import netcdf_file
        with netcdf_file(path, mmap=False) as ds:
            name = _pick_variable(list(ds.variables), var, path)
            v = ds.variables[name]
            data = np.array(v.data, dtype=float)
            fill = getattr(v, "_FillValue", getattr(v, "missing_value", None))
            if fill is not None:
                data[data == float(np.squeeze(fill))] = np.nan
            data = data * float(getattr(v, "scale_factor", 1.0)) + float(getattr(v, "add_offset", 0.0))
            return _to_3d(data, path), f"netcdf:{name} (scipy.io, NetCDF3 only)"
    except Exception as e:  # noqa: BLE001 - report every reader's failure together
        errors.append(f"scipy.io.netcdf_file: {e}")
    raise RuntimeError("could not read NetCDF file; install xarray or netCDF4 "
                       f"(or convert to .npy). Reader errors: {errors}")


def _read_tiff(path, nodata):
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read().astype(float)
            nd = src.nodata if nodata is None else nodata
        reader = "rasterio"
    except ImportError:
        try:
            import tifffile
        except ImportError as e:
            raise RuntimeError("reading GeoTIFF needs rasterio or tifffile") from e
        arr = tifffile.imread(path).astype(float)
        nd = nodata
        reader = "tifffile"
    if nd is not None:
        arr[arr == nd] = np.nan
    return arr, reader


def _load_geotiff(path, nodata):
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.lower().endswith((".tif", ".tiff")))
        if not files:
            raise ValueError(f"no .tif files in {path}")
        frames, reader = [], ""
        for f in files:
            a, reader = _read_tiff(os.path.join(path, f), nodata)
            frames.append(np.squeeze(a))
        return _to_3d(np.stack(frames), path), f"geotiff dir ({len(files)} files, {reader})"
    arr, reader = _read_tiff(path, nodata)
    return _to_3d(arr, path), f"geotiff stack ({reader})"


def load_dataset(path, var=None, nodata=None):
    """Return (array[t][y][x] with NaN for invalid cells, description)."""
    low = path.lower()
    if os.path.isdir(path) or low.endswith((".tif", ".tiff")):
        arr, desc = _load_geotiff(path, nodata)
    elif low.endswith((".nc", ".nc4", ".cdf")):
        arr, desc = _load_netcdf(path, var)
    elif low.endswith(".npy"):
        arr, desc = _to_3d(np.load(path), path), "numpy .npy"
    elif low.endswith(".npz"):
        z = np.load(path)
        name = var or z.files[0]
        arr, desc = _to_3d(z[name], path), f"numpy .npz:{name}"
    else:
        raise ValueError(f"unsupported dataset {path!r} (use .nc, .tif/.tiff, a folder of tifs, .npy, .npz)")
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr, desc


def synthetic_climate(frames=12, height=40, width=40, seed=0):
    """Labelled stand-in for a 2 m temperature field (kelvin): latitude gradient,
    diurnal cycle, a travelling weather system and spatially correlated noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames)[:, None, None]
    y = np.linspace(0, 1, height)[None, :, None]
    x = np.linspace(0, 1, width)[None, None, :]
    noise = ndimage.gaussian_filter(rng.normal(size=(frames, height, width)), (1.0, 2.5, 2.5))
    field = (285.0 - 15.0 * y + 4.0 * np.sin(2 * np.pi * (t / 8.0) + 3 * x)
             + 5.0 * np.exp(-(((x - 0.2 - 0.05 * t) / 0.18) ** 2 + ((y - 0.5) / 0.25) ** 2))
             + 1.5 * noise * 3)
    return field


# ---------------------------------------------------------------------------
# Cloud masks
# ---------------------------------------------------------------------------
def make_cloud_mask(valid, fraction, rng, blob_sigma=None, persistence=0.8, drift=(0.7, 1.1)):
    """Cloud-cover mask (True = hidden) over `valid` (t, y, x bool).

    A cloudiness field is built frame by frame: the previous frame advected by `drift`
    pixels (wind), blended with fresh spatially smooth noise (`persistence`), so clouds
    form coherent blobs that move and evolve. It is then thresholded so that exactly
    round(fraction * valid.sum()) VALID cells are hidden. Invalid cells are never hidden.
    """
    if not (0.0 <= fraction < 1.0):
        raise ValueError("fraction must be in [0, 1)")
    frames, height, width = valid.shape
    sigma = blob_sigma or max(1.5, min(height, width) / 10.0)
    field = np.empty(valid.shape, dtype=float)
    prev = None
    for t in range(frames):
        fresh = ndimage.gaussian_filter(rng.normal(size=(height, width)), sigma, mode="wrap")
        fresh /= fresh.std() or 1.0
        if prev is None:
            cur = fresh
        else:
            moved = ndimage.shift(prev, drift, order=1, mode="wrap")
            cur = persistence * moved + math.sqrt(1 - persistence ** 2) * fresh
            cur /= cur.std() or 1.0
        field[t] = cur
        prev = cur
    n_valid = int(valid.sum())
    k = int(round(fraction * n_valid))
    mask = np.zeros(valid.shape, dtype=bool)
    if k == 0:
        return mask
    scores = np.where(valid, field, -np.inf)
    flat = scores.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    mask.ravel()[idx] = True
    return mask


# ---------------------------------------------------------------------------
# Methods: each takes a masked array (NaN = hidden or invalid) and returns a
# prediction array with NaN wherever it declines to answer. None sees the truth.
# ---------------------------------------------------------------------------
def method_gde(masked, max_waves=None, temporal_decay=0.5, temporal_reach=1, inward_decay=0.9,
               connectivity=4):
    grid = np.where(np.isfinite(masked), masked, np.nan)
    nested = [[[None if math.isnan(v) else v for v in row] for row in frame]
              for frame in grid.tolist()]
    with contextlib.redirect_stdout(io.StringIO()):
        report = reconstruct_3d(nested, spatial_connectivity=connectivity,
                                temporal_decay=temporal_decay, temporal_reach=temporal_reach,
                                inward_decay=inward_decay, max_waves=max_waves)
    pred = np.full(masked.shape, np.nan)
    for row in report["audit_trail"].values():
        t, y, x = row["coordinate"]
        pred[t, y, x] = row["intensity"]
    return pred


def method_linear_temporal(masked):
    frames, height, width = masked.shape
    pred = np.full(masked.shape, np.nan)
    tt = np.arange(frames, dtype=float)
    for y in range(height):
        for x in range(width):
            series = masked[:, y, x]
            ok = np.isfinite(series)
            if ok.sum() == 0:
                continue
            pred[:, y, x] = np.interp(tt, tt[ok], series[ok], left=np.nan, right=np.nan)
    return pred


def _points(known, time_scale):
    t, y, x = np.nonzero(known)
    return np.column_stack([t * time_scale, y, x])


def method_scipy_linear3d(masked, time_scale=1.0):
    known = np.isfinite(masked)
    pred = np.full(masked.shape, np.nan)
    if known.sum() < 5:
        return pred
    interp = LinearNDInterpolator(_points(known, time_scale), masked[known])
    q = ~known
    pred[q] = interp(_points(q, time_scale))
    return pred


def method_scipy_nearest3d(masked, time_scale=1.0):
    known = np.isfinite(masked)
    pred = np.full(masked.shape, np.nan)
    if known.sum() == 0:
        return pred
    interp = NearestNDInterpolator(_points(known, time_scale), masked[known])
    q = ~known
    pred[q] = interp(_points(q, time_scale))
    return pred


def build_methods(args):
    gde_kw = dict(temporal_decay=args.temporal_decay, temporal_reach=args.temporal_reach,
                  inward_decay=args.inward_decay)
    return {
        "gde_3d": lambda m: method_gde(m, max_waves=None, **gde_kw),
        "gde_3d_bounded": lambda m: method_gde(m, max_waves=max(1, int(args.ground_radius)), **gde_kw),
        "linear_temporal": method_linear_temporal,
        "scipy_linear3d": lambda m: method_scipy_linear3d(m, args.time_scale),
        "scipy_nearest3d": lambda m: method_scipy_nearest3d(m, args.time_scale),
    }


# ---------------------------------------------------------------------------
# Scoring (the only place the hidden truth is used)
# ---------------------------------------------------------------------------
def ungrounded_cells(observed, hidden, radius, time_scale=1.0):
    """Hidden cells farther than `radius` (index units) from any recorded cell."""
    if not observed.any():
        return hidden.copy()
    dist = ndimage.distance_transform_edt(~observed, sampling=(time_scale, 1.0, 1.0))
    return hidden & (dist > radius)


def run_once(truth, valid, fraction, seed, methods, ground_radius, time_scale):
    rng = np.random.default_rng(seed)
    hidden = make_cloud_mask(valid, fraction, rng)
    masked = np.where(hidden, np.nan, np.where(valid, truth, np.nan))
    observed = np.isfinite(masked)
    ungrounded = ungrounded_cells(observed, hidden, ground_radius, time_scale)
    grounded = hidden & ~ungrounded

    preds, times = {}, {}
    for name, fn in methods.items():
        start = time.perf_counter()
        preds[name] = fn(masked.copy())
        times[name] = time.perf_counter() - start

    resolved = {n: hidden & np.isfinite(p) for n, p in preds.items()}
    common = hidden.copy()
    for r in resolved.values():
        common &= r

    out = {"fraction_requested": fraction, "seed": seed,
           "hidden_cells": int(hidden.sum()), "valid_cells": int(valid.sum()),
           "hidden_fraction": float(hidden.sum() / max(1, valid.sum())),
           "ungrounded_cells": int(ungrounded.sum()), "common_cells": int(common.sum()),
           "methods": {}}
    for name, pred in preds.items():
        err = pred - truth
        res = resolved[name]
        out["methods"][name] = {
            "runtime_s": times[name],
            "resolved": int(res.sum()),
            "unresolved": int((hidden & ~res).sum()),
            "abs_err_sum": float(np.abs(err[res]).sum()),
            "sq_err_sum": float((err[res] ** 2).sum()),
            "common_abs_err_sum": float(np.abs(err[common]).sum()),
            "common_sq_err_sum": float((err[common] ** 2).sum()),
            "ungrounded_flagged": int((ungrounded & ~res).sum()),
            "grounded_flagged": int((grounded & ~res).sum()),
            "ungrounded_filled_abs_err_sum": float(np.abs(err[ungrounded & res]).sum()),
            "ungrounded_filled": int((ungrounded & res).sum()),
        }
    return out


def _ratio(a, b):
    return a / b if b else None


def aggregate(runs, method_names):
    """Pool all runs (sums, not means of means) into one row per method."""
    rows = {}
    hidden = sum(r["hidden_cells"] for r in runs)
    ungr = sum(r["ungrounded_cells"] for r in runs)
    common = sum(r["common_cells"] for r in runs)
    for name in method_names:
        ms = [r["methods"][name] for r in runs]
        resolved = sum(m["resolved"] for m in ms)
        ung_flagged = sum(m["ungrounded_flagged"] for m in ms)
        gr_flagged = sum(m["grounded_flagged"] for m in ms)
        ung_filled = sum(m["ungrounded_filled"] for m in ms)
        rows[name] = {
            "mae": _ratio(sum(m["abs_err_sum"] for m in ms), resolved),
            "rmse": (math.sqrt(sum(m["sq_err_sum"] for m in ms) / resolved) if resolved else None),
            "mae_common": _ratio(sum(m["common_abs_err_sum"] for m in ms), common),
            "rmse_common": (math.sqrt(sum(m["common_sq_err_sum"] for m in ms) / common) if common else None),
            "runtime_s": sum(m["runtime_s"] for m in ms) / len(ms),
            "resolved_pct": 100.0 * resolved / hidden if hidden else None,
            "unresolved_pct": 100.0 * (hidden - resolved) / hidden if hidden else None,
            "ungrounded_flagged_pct": (100.0 * ung_flagged / ungr) if ungr else None,
            "grounded_flagged_pct": (100.0 * gr_flagged / (hidden - ungr)) if hidden > ungr else None,
            "mae_ungrounded_filled": _ratio(sum(m["ungrounded_filled_abs_err_sum"] for m in ms), ung_filled),
            "ungrounded_filled": ung_filled,
        }
    return {"runs": len(runs), "hidden_cells": hidden, "ungrounded_cells": ungr,
            "common_cells": common, "methods": rows}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _f(v, nd=4, suffix=""):
    return "n/a" if v is None else f"{v:.{nd}f}{suffix}"


def format_table(agg):
    head = ("| method | MAE | RMSE | MAE common | RMSE common | runtime s | unresolved % | "
            "ungrounded flagged % | grounded flagged % |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for name, r in agg["methods"].items():
        lines.append(f"| {name} | {_f(r['mae'])} | {_f(r['rmse'])} | {_f(r['mae_common'])} | "
                     f"{_f(r['rmse_common'])} | {_f(r['runtime_s'], 3)} | {_f(r['unresolved_pct'], 1)} | "
                     f"{_f(r['ungrounded_flagged_pct'], 1)} | {_f(r['grounded_flagged_pct'], 1)} |")
    return "\n".join(lines)


def build_markdown(result):
    d = result["dataset"]
    lines = ["# Real-climate benchmark: GDE 3.1 Spatiotemporal vs interpolation", ""]
    if d["synthetic"]:
        lines += ["> **SYNTHETIC DATA.** No real dataset was supplied, so these numbers come from a "
                  "labelled climate-like stand-in and are NOT real-data results.", ""]
    lines += [f"- Dataset: `{d['name']}` ({d['description']}), shape (t, y, x) = {tuple(d['shape'])}, "
              f"valid cells = {d['valid_cells']}",
              f"- Seeds: {result['config']['seeds']}, hidden fractions: {result['config']['fractions']}, "
              f"ground radius G = {result['config']['ground_radius']}",
              "- Cloud masks hide the stated fraction of valid cells; methods see only the masked array.",
              "- MAE/RMSE are in the data's own units, over cells each method resolved; the *common* "
              "columns score every method on the same cells.",
              "- Ungrounded = hidden cell farther than G from any recorded cell. 'ungrounded flagged' "
              "is the share of those a method left unresolved (higher = safer).", ""]
    lines += ["## All runs pooled", "", format_table(result["overall"]), ""]
    for frac, agg in result["by_fraction"].items():
        lines += [f"## Hidden fraction {float(frac):.0%}", "", format_table(agg), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def prepare_dataset(args):
    if args.data:
        arr, desc = load_dataset(args.data, args.var, args.nodata)
        name, synthetic = os.path.basename(os.path.normpath(args.data)), False
    else:
        if args.require_real:
            raise SystemExit("--require-real set but no --data given")
        arr = synthetic_climate(args.max_frames or 12, args.crop or 40, args.crop or 40, seed=0)
        name, desc, synthetic = "synthetic_climate", "SYNTHETIC stand-in, not real data", True
        print("[!] No --data supplied: using a labelled SYNTHETIC field. "
              "These are NOT real-data results.", file=sys.stderr)
    if args.max_frames:
        arr = arr[:args.max_frames]
    if args.crop:
        arr = arr[:, :args.crop, :args.crop]
    if args.stride > 1:
        arr = arr[:, ::args.stride, ::args.stride]
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        raise SystemExit("dataset has no valid cells after loading/cropping")
    return arr, valid, name, desc, synthetic


def run(args):
    for f in args.fractions:
        if not (MIN_FRACTION - 1e-9 <= f <= MAX_FRACTION + 1e-9) and not args.allow_any_fraction:
            raise SystemExit(f"fraction {f} outside {MIN_FRACTION}-{MAX_FRACTION} "
                             "(use --allow-any-fraction to override)")
    arr, valid, name, desc, synthetic = prepare_dataset(args)
    truth = np.where(valid, arr, 0.0)
    methods = build_methods(args)
    runs, by_fraction = [], {}
    for frac in args.fractions:
        fruns = []
        for s in range(args.seeds):
            fruns.append(run_once(truth, valid, frac, args.seed_base + s, methods,
                                  args.ground_radius, args.time_scale))
        by_fraction[f"{frac}"] = aggregate(fruns, list(methods))
        runs.extend(fruns)
    return {
        "dataset": {"name": name, "description": desc, "synthetic": synthetic,
                    "shape": list(arr.shape), "valid_cells": int(valid.sum())},
        "config": {"seeds": args.seeds, "fractions": list(args.fractions),
                   "ground_radius": args.ground_radius, "time_scale": args.time_scale,
                   "temporal_decay": args.temporal_decay, "temporal_reach": args.temporal_reach,
                   "inward_decay": args.inward_decay},
        "overall": aggregate(runs, list(methods)),
        "by_fraction": by_fraction,
        "runs": runs,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help=".nc, .tif/.tiff, folder of .tif, .npy or .npz with shape (time, y, x)")
    p.add_argument("--var", help="variable name (NetCDF / .npz); default: first data variable")
    p.add_argument("--nodata", type=float, help="value to treat as invalid")
    p.add_argument("--require-real", action="store_true", help="refuse the synthetic fallback")
    p.add_argument("--max-frames", type=int, default=12, help="use the first N time steps (default 12)")
    p.add_argument("--crop", type=int, default=40, help="crop to N x N cells from the corner (default 40)")
    p.add_argument("--stride", type=int, default=1, help="spatial subsampling step")
    p.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS),
                   help="fractions of valid cells to hide (each within 0.20-0.40)")
    p.add_argument("--allow-any-fraction", action="store_true", help="allow fractions outside 0.20-0.40")
    p.add_argument("--seeds", type=int, default=3, help="cloud masks per fraction (default 3)")
    p.add_argument("--seed-base", type=int, default=100)
    p.add_argument("--ground-radius", type=float, default=4.0,
                   help="G: hidden cells farther than this from any recorded cell are 'ungrounded'; "
                        "also max_waves for gde_3d_bounded")
    p.add_argument("--time-scale", type=float, default=1.0, help="weight of the time axis in scipy 3D methods and in distances")
    p.add_argument("--temporal-decay", type=float, default=0.5)
    p.add_argument("--temporal-reach", type=int, default=1)
    p.add_argument("--inward-decay", type=float, default=0.9)
    p.add_argument("--json", help="write full results to this JSON file")
    p.add_argument("--markdown", help="write the report to this Markdown file")
    args = p.parse_args(argv)
    if args.seeds < 1:
        p.error("--seeds must be >= 1")
    if args.ground_radius < 1:
        p.error("--ground-radius must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    result = run(args)
    md = build_markdown(result)
    print(md)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    main()
