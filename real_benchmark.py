"""
Real-data benchmark: Ghost Data Engine vs scipy / scikit-image interpolators.

Real images (lunar surface, Hubble deep field, a photograph, a grass texture,
all shipped inside scikit-image) are downsampled and then have *artificial blob
voids* cut out of them. Every method sees only the pixels outside the voids and
is scored (MAE / RMSE) on the hidden pixels against the real pixel values.

Usage:
    python real_benchmark.py                     # default: 64x64, 5 seeds
    python real_benchmark.py --size 96 --seeds 10 --void-size 10
    python real_benchmark.py --json results.json --markdown results.md

Requires: numpy, scipy, scikit-image
"""
import argparse
import contextlib
import io
import json
import random
import statistics
import time

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from skimage import data as skdata
from skimage.color import rgb2gray
from skimage.restoration import inpaint_biharmonic
from skimage.transform import resize

from generator import _carve_voids
from scanner import scan_for_void_clusters
from tension import calculate_boundary_tension
from reconstructor import reconstruct_void

# name -> (loader, description)
IMAGES = {
    "moon": (skdata.moon, "lunar surface (remote-sensing-like)"),
    "hubble": (skdata.hubble_deep_field, "Hubble deep field (astronomy)"),
    "camera": (skdata.camera, "classic photograph"),
    "grass": (skdata.grass, "grass texture (hard, high-frequency)"),
}


def load_image(name, size):
    img = IMAGES[name][0]()
    if img.ndim == 3:
        img = rgb2gray(img[..., :3]) * 255.0
    img = resize(img.astype(float), (size, size), anti_aliasing=True, preserve_range=True)
    return img


def carve_blob_mask(size, num_voids, void_size, seed):
    """Boolean mask (rows=y-1, cols=x-1) of hidden pixels, using generator.py's blob shape."""
    rng = random.Random(seed)
    cells, _ = _carve_voids(rng, size, size, num_voids, void_size, ["blob"], True)
    mask = np.zeros((size, size), dtype=bool)
    for x, y in cells:
        mask[y - 1, x - 1] = True
    return mask


# ---------------------------------------------------------------------------
# Methods: each takes (img, mask) and returns predicted values for img[mask]
# (in np.nonzero(mask) order). They must only read img where ~mask.
# ---------------------------------------------------------------------------
def _grid_dict(img, mask):
    size_y, size_x = img.shape
    return {(x + 1, y + 1): {"status": "Recorded", "intensity": float(img[y, x])}
            for y in range(size_y) for x in range(size_x) if not mask[y, x]}


def _ghost(img, mask, inward_decay, void_level, mode="legacy", **extra):
    grid = _grid_dict(img, mask)
    h, w = img.shape
    with contextlib.redirect_stdout(io.StringIO()):  # silence the engine's logs
        clusters = scan_for_void_clusters(grid, connectivity=4, bounds=(1, w, 1, h))
        tension = calculate_boundary_tension(clusters, grid, void_level=void_level)
        report = reconstruct_void(clusters, tension, grid, pressure_threshold=0.0,
                                  inward_decay=inward_decay, void_level=void_level,
                                  mode=mode, **extra)
    trail = report["audit_trail"]
    ys, xs = np.nonzero(mask)
    pred = np.full(len(ys), np.nan)
    for i, (y, x) in enumerate(zip(ys, xs)):
        item = trail.get(str((int(x) + 1, int(y) + 1)))
        if item is not None:
            pred[i] = item["intensity"]
    return pred


def ghost_default(img, mask):       # 2.0/3.0 defaults: decay 0.85, void_level 0
    return _ghost(img, mask, 0.85, 0.0)


def ghost_nodecay(img, mask):       # decay disabled: no pull toward void_level
    return _ghost(img, mask, 1.0, 0.0)


def ghost_accurate(img, mask):      # 3.1 behaviour, pinned (relax=False) so the published 3.1 numbers reproduce
    return _ghost(img, mask, 0.85, 0.0, mode="accurate", relax=False)


def ghost_accurate_relax(img, mask):   # 3.2 default: 3.1 + neighbour-average relaxation after the fill
    return _ghost(img, mask, 0.85, 0.0, mode="accurate", relax=True)


def ghost_accurate_blend(img, mask):   # 3.2 optional: depth-blended relaxation (edge cells unchanged)
    return _ghost(img, mask, 0.85, 0.0, mode="accurate", relax=True, relax_depth_scale=2.0)


# Ablation variants (run with --ablation): isolate what each 3.1 change contributes.
def ghost_acc_norefine(img, mask):  # accurate wave fill only (no thin-plate refinement)
    return _ghost(img, mask, 0.85, 0.0, mode="accurate", refine_max_iter=0, relax=False)


def ghost_acc_nodecay(img, mask):   # accurate + refinement, decay disabled
    return _ghost(img, mask, 1.0, 0.0, mode="accurate", relax=False)


def _pts(img, mask):
    ky, kx = np.nonzero(~mask)
    qy, qx = np.nonzero(mask)
    return np.c_[kx, ky].astype(float), img[~mask], np.c_[qx, qy].astype(float)


def scipy_nearest(img, mask):
    p, v, q = _pts(img, mask)
    return NearestNDInterpolator(p, v)(q)


def scipy_linear(img, mask):
    p, v, q = _pts(img, mask)
    out = LinearNDInterpolator(p, v)(q)
    if np.isnan(out).any():  # outside convex hull -> nearest, so coverage is fair
        out = np.where(np.isnan(out), NearestNDInterpolator(p, v)(q), out)
    return out


def scipy_cubic(img, mask):
    p, v, q = _pts(img, mask)
    out = CloughTocher2DInterpolator(p, v)(q)
    if np.isnan(out).any():
        out = np.where(np.isnan(out), NearestNDInterpolator(p, v)(q), out)
    return out


def scipy_rbf(img, mask):
    p, v, q = _pts(img, mask)
    return RBFInterpolator(p, v, neighbors=64, kernel="thin_plate_spline", smoothing=0.0)(q)


def skimage_biharmonic(img, mask):
    filled = inpaint_biharmonic(np.where(mask, 0.0, img), mask)
    return filled[mask]


METHODS = {
    "ghost_default (decay .85)": ghost_default,
    "ghost_nodecay (decay 1.0)": ghost_nodecay,
    "ghost_accurate (3.1)": ghost_accurate,
    "ghost_accurate+relax (3.2 default)": ghost_accurate_relax,
    "ghost_accurate+blend (3.2 option)": ghost_accurate_blend,
    "scipy nearest": scipy_nearest,
    "scipy linear (griddata)": scipy_linear,
    "scipy cubic (griddata)": scipy_cubic,
    "scipy RBF thin-plate": scipy_rbf,
    "skimage biharmonic": skimage_biharmonic,
}


ABLATION_METHODS = {
    "ghost_acc_norefine (waves only)": ghost_acc_norefine,
    "ghost_acc_nodecay (decay 1.0)": ghost_acc_nodecay,
}


def run(size=64, seeds=5, num_voids=6, void_size=8, images=None, ablation=False):
    images = images or list(IMAGES)
    methods = dict(METHODS)
    if ablation:
        methods.update(ABLATION_METHODS)
    rows = []
    for name in images:
        img = load_image(name, size)
        for seed in range(seeds):
            mask = carve_blob_mask(size, num_voids, void_size, seed)
            truth = img[mask]
            for mname, fn in methods.items():
                masked_img = np.where(mask, np.nan, img)  # methods get no hidden values
                t0 = time.perf_counter()
                pred = fn(np.nan_to_num(masked_img, nan=0.0) if mname.startswith("skimage") else masked_img, mask)
                dt = time.perf_counter() - t0
                covered = ~np.isnan(pred)
                err = pred[covered] - truth[covered]
                rows.append({
                    "image": name, "seed": seed, "method": mname,
                    "hidden_pixels": int(mask.sum()),
                    "coverage": float(covered.mean()),
                    "mae": float(np.mean(np.abs(err))) if err.size else None,
                    "rmse": float(np.sqrt(np.mean(err ** 2))) if err.size else None,
                    "seconds": dt,
                })
    return rows


def summarize(rows):
    out = {}
    for r in rows:
        out.setdefault(r["image"], {}).setdefault(r["method"], []).append(r)
    summary = {}
    for image, methods in out.items():
        summary[image] = {}
        for m, rs in methods.items():
            maes = [r["mae"] for r in rs if r["mae"] is not None]
            summary[image][m] = {
                "mae_mean": statistics.mean(maes),
                "mae_std": statistics.pstdev(maes),
                "rmse_mean": statistics.mean(r["rmse"] for r in rs if r["rmse"] is not None),
                "coverage": statistics.mean(r["coverage"] for r in rs),
                "seconds_mean": statistics.mean(r["seconds"] for r in rs),
                "hidden_pixels_mean": statistics.mean(r["hidden_pixels"] for r in rs),
            }
    return summary


def to_markdown(summary, cfg):
    lines = [f"Real-image benchmark: {cfg['size']}x{cfg['size']} px, {cfg['seeds']} seeds, "
             f"{cfg['num_voids']} blob voids/seed (max_void_size={cfg['void_size']}). "
             f"Intensity range 0-255. Lower MAE is better.\n"]
    for image, methods in summary.items():
        hid = next(iter(methods.values()))["hidden_pixels_mean"]
        lines.append(f"### {image} — {IMAGES[image][1]} (~{hid:.0f} hidden px per run)\n")
        lines.append("| Method | MAE (mean ± sd) | RMSE | Coverage | Time (ms) |")
        lines.append("|---|---|---|---|---|")
        best = min(v["mae_mean"] for v in methods.values())
        for m, v in sorted(methods.items(), key=lambda kv: kv[1]["mae_mean"]):
            mark = " **(best)**" if v["mae_mean"] == best else ""
            lines.append(f"| {m}{mark} | {v['mae_mean']:.3f} ± {v['mae_std']:.3f} | "
                         f"{v['rmse_mean']:.3f} | {v['coverage']:.0%} | {v['seconds_mean']*1000:.1f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--num-voids", type=int, default=6)
    ap.add_argument("--void-size", type=int, default=8)
    ap.add_argument("--images", nargs="+", choices=list(IMAGES), default=None)
    ap.add_argument("--ablation", action="store_true",
                    help="also run ablation variants of the accurate mode")
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None)
    a = ap.parse_args()
    cfg = {"size": a.size, "seeds": a.seeds, "num_voids": a.num_voids, "void_size": a.void_size}
    rows = run(a.size, a.seeds, a.num_voids, a.void_size, a.images, a.ablation)
    summary = summarize(rows)
    md = to_markdown(summary, cfg)
    print(md)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"config": cfg, "summary": summary, "runs": rows}, fh, indent=2)
    if a.markdown:
        with open(a.markdown, "w") as fh:
            fh.write(md)


if __name__ == "__main__":
    main()
