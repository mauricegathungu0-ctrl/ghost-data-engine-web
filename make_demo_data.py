"""Make a demo file + accuracy report from a complete matrix CSV.

Takes a COMPLETE headerless numeric CSV (the "answer key"), hides some values the
way real outages happen (scattered single cells + runs of consecutive days),
repairs them with Ghost Data Engine, and scores every method against the true
values it never saw.

Usage:
    python make_demo_data.py real_power_FULL_answer_key.csv
    python make_demo_data.py key.csv --seeds 10 --scatter 0.05 --blocks 6 --out demo_out

Writes (in --out): gapped.csv  (upload this to the demo), answer_key.csv,
repaired.csv, results.md, results.json
"""
import argparse
import csv
import json
import os
import statistics
import time

import numpy as np
from scipy.interpolate import griddata

import app_core as core


def load_matrix(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [[float(x) for x in r] for r in csv.reader(fh) if r]
    return np.array(rows, dtype=float)


def make_mask(shape, scatter, blocks, rng, bmin=5, bmax=20):
    """True = hidden. Scattered cells plus rectangular outages (rows x consecutive columns)."""
    h, w = shape
    mask = rng.random(shape) < scatter
    for _ in range(blocks):
        bh = int(rng.integers(1, 3))            # 1-2 rows (years)
        bw = int(rng.integers(bmin, bmax + 1))  # consecutive columns (days)
        r = int(rng.integers(0, h - bh + 1))
        c = int(rng.integers(0, w - bw + 1))
        mask[r:r + bh, c:c + bw] = True
    # never hide a whole row or column: every method needs some context
    mask[mask.all(axis=1)] = False
    return mask


def to_grid(truth, mask):
    h, w = truth.shape
    return {(x + 1, h - i): {"status": "Recorded", "intensity": float(truth[i, x])}
            for i in range(h) for x in range(w) if not mask[i, x]}


# ---- baselines: each reads only values where ~mask -------------------------------
def m_row_linear(truth, mask):
    out = truth.copy()
    xs = np.arange(truth.shape[1])
    for i in range(truth.shape[0]):
        ok = ~mask[i]
        if ok.any() and mask[i].any():
            out[i, mask[i]] = np.interp(xs[mask[i]], xs[ok], truth[i, ok])
    return out


def m_same_day_mean(truth, mask):
    """Mean of the same day-of-year across the other (recorded) years."""
    out = truth.copy()
    vals = np.where(mask, np.nan, truth)
    col = np.nanmean(vals, axis=0)
    fallback = np.nanmean(vals)
    col = np.where(np.isnan(col), fallback, col)
    for i, j in zip(*np.nonzero(mask)):
        out[i, j] = col[j]
    return out


def m_griddata(truth, mask):
    ii, jj = np.nonzero(~mask)
    pts = np.column_stack([ii, jj]).astype(float)
    vals = truth[~mask]
    qi, qj = np.nonzero(mask)
    q = np.column_stack([qi, qj]).astype(float)
    lin = griddata(pts, vals, q, method="linear")
    near = griddata(pts, vals, q, method="nearest")
    lin = np.where(np.isnan(lin), near, lin)
    out = truth.copy()
    out[mask] = lin
    return out


def m_ghost(truth, mask):
    h, w = truth.shape
    res = core.repair_matrix(to_grid(truth, mask), w, h, blend=True)
    return res["repaired"]


METHODS = {
    "Ghost Data Engine": m_ghost,
    "Linear interpolation (along days)": m_row_linear,
    "Same-day mean of other years": m_same_day_mean,
    "2D linear (scipy griddata)": m_griddata,
}


def score(truth, mask, pred):
    err = pred[mask] - truth[mask]
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key", help="complete headerless numeric CSV")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--scatter", type=float, default=0.05, help="share of single cells hidden at random")
    ap.add_argument("--blocks", type=int, default=6, help="number of multi-day outage blocks")
    ap.add_argument("--bmin", type=int, default=5, help="shortest outage (days)")
    ap.add_argument("--bmax", type=int, default=20, help="longest outage (days)")
    ap.add_argument("--out", default="demo_out")
    a = ap.parse_args()

    truth = load_matrix(a.key)
    h, w = truth.shape
    if h * w > core.MAX_CELLS_2D:
        print(f"note: {h*w} cells is above the hosted demo limit ({core.MAX_CELLS_2D}); "
              "the command line is fine, the web demo will refuse it.")
    os.makedirs(a.out, exist_ok=True)

    scores = {m: {"mae": [], "rmse": [], "sec": []} for m in METHODS}
    hidden_share = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        mask = make_mask(truth.shape, a.scatter, a.blocks, rng, a.bmin, a.bmax)
        hidden_share.append(mask.mean())
        for name, fn in METHODS.items():
            t0 = time.time()
            pred = fn(truth, mask)
            sec = time.time() - t0
            mae, rmse = score(truth, mask, pred)
            scores[name]["mae"].append(mae)
            scores[name]["rmse"].append(rmse)
            scores[name]["sec"].append(sec)

    # demo files from seed 0
    mask0 = make_mask(truth.shape, a.scatter, a.blocks, np.random.default_rng(0), a.bmin, a.bmax)
    pred0 = m_ghost(truth, mask0)
    repaired = np.where(mask0, pred0, truth)

    def write(path, mat, blank=None):
        with open(path, "w", newline="") as fh:
            wr = csv.writer(fh)
            for i in range(mat.shape[0]):
                wr.writerow(["" if (blank is not None and blank[i, j]) else f"{mat[i, j]:.2f}"
                             for j in range(mat.shape[1])])

    write(os.path.join(a.out, "gapped.csv"), truth, blank=mask0)
    write(os.path.join(a.out, "answer_key.csv"), truth)
    write(os.path.join(a.out, "repaired.csv"), repaired)

    rows = []
    for name, s in scores.items():
        rows.append((name, statistics.mean(s["mae"]), statistics.stdev(s["mae"]) if a.seeds > 1 else 0.0,
                     statistics.mean(s["rmse"]), statistics.mean(s["sec"])))
    rows.sort(key=lambda r: r[1])
    best_base = min(r[1] for r in rows if r[0] != "Ghost Data Engine")
    gde_mae = next(r[1] for r in rows if r[0] == "Ghost Data Engine")

    md = [f"# Accuracy on {h}x{w} matrix ({h*w} cells)", "",
          f"Hidden: about {100*statistics.mean(hidden_share):.1f}% of cells per trial "
          f"({a.scatter:.0%} scattered + {a.blocks} outage blocks of {a.bmin}-{a.bmax} days). "
          f"{a.seeds} random trials. Errors are measured only on hidden cells, against the true values. "
          "Every method sees only the recorded cells.", "",
          "| Method | MAE (mean ± sd) | RMSE | Seconds |", "|---|---|---|---|"]
    for n, mae, sd, rmse, sec in rows:
        md.append(f"| {n} | {mae:.3f} ± {sd:.3f} | {rmse:.3f} | {sec:.2f} |")
    md += ["", f"GDE MAE vs best baseline: {gde_mae:.3f} vs {best_base:.3f} "
               f"({'better' if gde_mae < best_base else 'NOT better'}, "
               f"{100*(best_base-gde_mae)/best_base:+.1f}% change).", ""]
    text = "\n".join(md)
    with open(os.path.join(a.out, "results.md"), "w") as fh:
        fh.write(text)
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump({"shape": [h, w], "seeds": a.seeds, "scores": scores}, fh, indent=1)
    print(text)
    print(f"Files written to {a.out}/")


if __name__ == "__main__":
    main()
