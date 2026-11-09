"""evaluate_blends.py - reproduces the table in RESULTS.md (needs numpy + scipy; ~1 minute).
Usage (from research/):  python evaluate_blends.py
Evaluate post-fill blends. Tune on masks 1-5, report on masks 6-10.
The filler (engine_v3) only ever reads the *_MISSING.csv file; the answer key is read here, for scoring only."""
import sys, os, io, csv, copy, json, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..")
DATA_ROOT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "blend_results.json")
sys.path.insert(0, ENGINE_DIR)
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import engine_v3 as E

DATASETS = {"temperature": ("temperature", "real_power_FULL_answer_key.csv", "real_mask_{:02d}_MISSING.csv"),
            "humidity":    ("humidity",    "humidity_FULL_answer_key.csv",   "humidity_mask_{:02d}_MISSING.csv")}
TUNE, TEST = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
BASE = "engine_v3 (knn 5, w0.75)"
CANDS = {"engine 3.2 (no blend)": dict(),
         BASE: dict(knn_blend=(5, 0.75)),
         "knn (3, w0.90)": dict(knn_blend=(3, 0.9)),
         "v3 + row blend 0.25": dict(knn_blend=(5, 0.75), row_blend=0.25),
         "v3 + row blend 0.50": dict(knn_blend=(5, 0.75), row_blend=0.5),
         "v3 + row blend 0.75": dict(knn_blend=(5, 0.75), row_blend=0.75),
         "v3 + row blend 0.90": dict(knn_blend=(5, 0.75), row_blend=0.9)}
NEW = [n for n in CANDS if n not in ("engine 3.2 (no blend)", BASE)]
rd = lambda p: list(csv.reader(open(p)))

def baselines(full, m):
    R, C = full.shape; yy, xx = np.mgrid[0:R, 0:C]; obs = ~m
    out = {}
    e = []
    x = np.arange(C)
    pred = full.copy()
    for i in range(R):
        if m[i].any():
            ok = ~m[i]; pred[i, m[i]] = np.interp(x[m[i]], x[ok], full[i, ok])
    out["row linear"] = pred
    p = np.c_[yy[obs] / (R - 1), xx[obs] / (C - 1)]; q = np.c_[yy[m] / (R - 1), xx[m] / (C - 1)]
    d, i = cKDTree(p).query(q, k=10); w = 1 / np.maximum(d, 1e-12)
    pk = full.copy(); pk[m] = (w * full[obs][i]).sum(1) / w.sum(1); out["KNN k=10"] = pk
    pts = np.c_[xx[obs], yy[obs]]; qq = np.c_[xx[m], yy[m]]
    pl = griddata(pts, full[obs], qq, method="linear"); nn = griddata(pts, full[obs], qq, method="nearest")
    ps = full.copy(); ps[m] = np.where(np.isnan(pl), nn, pl); out["scipy 2D linear"] = ps
    return out

results = {}
for ds, (sub, key, mfmt) in DATASETS.items():
    D = os.path.join(DATA_ROOT, sub)
    full = np.array([[float(v) for v in r] for r in rd(os.path.join(D, key))]); R, C = full.shape
    per = {}   # method -> {mask: mae}
    changed_known = 0
    for k in TUNE + TEST:
        src = os.path.join(D, mfmt.format(k))
        m = np.array([[v == "" for v in r] for r in rd(src)])
        with contextlib.redirect_stdout(io.StringIO()):
            grid, w, h = E.load_csv_with_gaps(src)          # filler input = missing file only
            voids, report = E.repair_grid(grid, w, h)
        for name, kw in CANDS.items():
            rep = {"audit_trail": {a: dict(b) for a, b in report["audit_trail"].items()}}
            E.apply_blends(grid, rep, w, h, **kw)
            pred = np.full((R, C), np.nan)
            for (x, y), info in grid.items(): pred[h - y, x - 1] = info["intensity"]
            for a, b in rep["audit_trail"].items():
                x, y = eval(a); pred[h - y, x - 1] = b["intensity"]
            changed_known += int((np.abs(pred[~m] - full[~m]) > 1e-9).sum())
            assert not np.isnan(pred).any()
            per.setdefault(name, {})[k] = float(np.abs(pred - full)[m].mean())
        for name, pred in baselines(full, m).items():
            per.setdefault(name, {})[k] = float(np.abs(pred - full)[m].mean())
        # consistency: row blend 1.0 with no knn must equal independent numpy row linear
        rep = {"audit_trail": {a: dict(b) for a, b in report["audit_trail"].items()}}
        E.apply_blends(grid, rep, w, h, row_blend=1.0)
        pr = np.full((R, C), np.nan)
        for (x, y), info in grid.items(): pr[h - y, x - 1] = info["intensity"]
        for a, b in rep["audit_trail"].items():
            x, y = eval(a); pr[h - y, x - 1] = b["intensity"]
        per.setdefault("_rowblend1_vs_rowlinear_maxdiff", {})[k] = float(np.abs(pr - baselines(full, m)["row linear"]).max())
    results[ds] = {"per": per, "known_cells_changed": changed_known}
json.dump(results, open(OUT, "w"), indent=1)

mean = lambda d, ks: float(np.mean([d[k] for k in ks])); sd = lambda d, ks: float(np.std([d[k] for k in ks]))
print("known cells changed in any candidate output:", {d: r["known_cells_changed"] for d, r in results.items()})
print("row-blend(1.0) vs independent row linear, max diff:", {d: max(r["per"]["_rowblend1_vs_rowlinear_maxdiff"].values()) for d, r in results.items()})
print("\n=== TUNING (masks 1-5): mean MAE and ratio to engine_v3 ===")
score = {}
for name in NEW:
    ratios = []
    for ds, r in results.items():
        ratios.append(mean(r["per"][name], TUNE) / mean(r["per"][BASE], TUNE))
    score[name] = float(np.mean(ratios))
for ds, r in results.items():
    print(f" {ds}: " + " | ".join(f"{n} {mean(r['per'][n], TUNE):.4f}" for n in [BASE] + NEW))
for n, v in score.items(): print(f"  {n}: mean ratio to engine_v3 {v:.4f}")
best = min(score, key=score.get)
print("chosen on masks 1-5 only:", best, "(ratio %.4f)" % score[best] + ("  -> NOT better than engine_v3 on tuning masks" if score[best] >= 1 else ""))
print("\n=== UNSEEN masks 6-10 ===")
order = ["engine 3.2 (no blend)", BASE, "knn (3, w0.90)", "v3 + row blend 0.25", "v3 + row blend 0.50", "v3 + row blend 0.75", "v3 + row blend 0.90", "row linear", "KNN k=10", "scipy 2D linear"]
verdict = {}
for ds, r in results.items():
    print(f"\n{ds}:")
    print(f" {'method':26s}" + " ".join(f"m{k:02d}   " for k in TEST) + "|  mean    sd")
    for n in order:
        v = r["per"][n]; print(f" {n:26s}" + " ".join(f"{v[k]:.3f} " if ds == 'temperature' else f"{v[k]:.2f}  " for k in TEST) + f"| {mean(v, TEST):.4f} {sd(v, TEST):.4f}")
    b, c, rl = r["per"][BASE], r["per"][best], r["per"]["row linear"]
    drop = mean(b, TEST) - mean(c, TEST); spread = max(sd(b, TEST), sd(c, TEST))
    wins = sum(c[k] < b[k] for k in TEST)
    d2 = mean(rl, TEST) - mean(c, TEST); sp2 = max(sd(rl, TEST), sd(c, TEST)); wins2 = sum(c[k] < rl[k] for k in TEST)
    verdict[ds] = dict(drop=drop, spread=spread, passes=bool(drop > spread), wins=wins, vs_row_gain=d2, vs_row_spread=sp2, wins_vs_row=wins2)
    print(f" chosen ({best}) vs engine_v3: improvement {drop:.4f} | spread {spread:.4f} | beats on {wins}/5 masks -> {'PASS' if drop > spread else 'FAIL'}")
    print(f" chosen vs row linear: gain {d2:+.4f} | spread {sp2:.4f} | beats on {wins2}/5 masks -> {'beats row linear' if d2 > sp2 else ('ties (within spread)' if d2 > -sp2 else 'row linear wins')}")
final = all(v["passes"] for v in verdict.values())
print("\nOVERALL:", f"KEEP {best}" if final else "REVERT: keep engine_v3 (knn 5, w0.75) unchanged")
json.dump({"best": best, "keep": final, "verdict": verdict}, open(OUT + ".decision.json", "w"), indent=1)
