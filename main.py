# main.py
import argparse
import json
import sys

from generator import generate_dataset, SHAPE_BUILDERS
from scanner import scan_for_void_clusters
from tension import calculate_boundary_tension
from reconstructor import reconstruct_void
from engine_v3 import apply_blends, DEFAULT_KNN_BLEND, DEFAULT_ROW_BLEND


def run_ghost_engine(
    width=20,
    height=20,
    seed=None,
    num_voids=None,
    max_void_size=5,
    shapes=None,
    keep_border=True,
    intensity=10,
    intensity_jitter=0,
    connectivity=4,
    pressure_threshold=None,
    threshold_ratio=0.5,
    inward_decay=0.85,
    void_level=0,
    blend=True,
):
    """Runs generate -> scan -> tension -> reconstruct (-> post-fill blend) and returns everything.

    blend=True (default) applies the engine_v3 post-fill blends (DEFAULT_KNN_BLEND, DEFAULT_ROW_BLEND)
    to the filled cells using recorded cells only; blend=False gives the plain engine output.
    """
    print("=== INITIALIZING GHOST DATA ENGINE 3.0 ===")

    # Step 1: generate a grid (with ground truth so the scanner can be checked)
    grid, truth = generate_dataset(
        width=width, height=height, num_voids=num_voids,
        max_void_size=max_void_size, shapes=shapes, seed=seed,
        keep_border=keep_border, intensity=intensity,
        intensity_jitter=intensity_jitter, return_ground_truth=True,
    )

    # Step 2: scan the FULL grid so voids touching the outer edge are caught too
    voids = scan_for_void_clusters(
        grid, connectivity=connectivity, bounds=(1, width, 1, height)
    )
    found = {c for v in voids for c in v["cells"]}
    scanner_ok = found == truth

    # Step 3: boundary pressure / tension (dict keyed by void id)
    tension_data = calculate_boundary_tension(voids, grid, void_level=void_level)

    # Step 4: reconstruct the missing space
    report = reconstruct_void(
        voids, tension_data, grid,
        pressure_threshold=pressure_threshold,
        threshold_ratio=threshold_ratio,
        inward_decay=inward_decay,
        void_level=void_level,
    )
    if blend:
        apply_blends(grid, report, width, height,
                     knn_blend=DEFAULT_KNN_BLEND, row_blend=DEFAULT_ROW_BLEND)
    return {"grid": grid, "truth": truth, "voids": voids, "tension": tension_data,
            "report": report, "scanner_ok": scanner_ok, "blend": blend}


def print_report(result, audit_limit=15):
    report = result["report"]
    s = report["summary"]

    print("\n=== GHOST DATA ENGINE 3.0 — FINAL STRUCTURAL REPORT ===")
    print(f"Scanner vs ground truth : {'OK' if result['scanner_ok'] else 'MISMATCH'}")
    print(f"Total voids             : {s.get('total_voids', 0)}")
    print(f"Total void cells        : {s.get('total_void_cells', 0)}")
    print(f"Cells reconstructed     : {s.get('cells_reconstructed', 0)}")
    print(f"Cells stalled           : {s.get('cells_stalled', 0)}")
    print(f"Pressure threshold      : {s.get('pressure_threshold', 'n/a')}")
    print(f"Cumulative tension      : {s.get('cumulative_structural_tension', 0)}")
    print(f"Hottest coordinate      : {s.get('highest_tension_coordinate')}")
    print(f"Inference               : {s.get('inference', '')}")
    print(f"Propagation waves       : {s.get('propagation_waves', 0)}")
    print(f"Mean confidence         : {s.get('mean_confidence', 0.0)} (heuristic)")
    print(f"Post-fill blend         : {'ON knn=%s row=%s' % (DEFAULT_KNN_BLEND, DEFAULT_ROW_BLEND) if result.get('blend') else 'off'}")

    if report["voids"]:
        print("\n--- Per-void results ---")
        for vid, v in report["voids"].items():
            print(f"Void #{vid}: {v['classification']} ({v['rule_fired']}) | "
                  f"size={v['metrics']['size']} filled={v['cells_filled']} "
                  f"stalled={v['cells_stalled']} strongest_side={v['strongest_side']} "
                  f"net_pull={v['net_pull']}")

    trail = report["audit_trail"]
    if trail:
        ordered = sorted(trail.items(), key=lambda kv: (kv[1]["void_id"], kv[1]["fill_order"]))
        shown = ordered if audit_limit == 0 else ordered[:audit_limit]
        print(f"\n--- Audit trail ({len(shown)} of {len(trail)} reconstructed cells) ---")
        for cell, r in shown:
            print(f"{cell} void#{r['void_id']} order={r['fill_order']} depth={r['ingrowth_depth']} "
                  f"I={r['intensity']} | pressure {r['source_tension']} >= "
                  f"threshold {r['pressure_threshold']} -> {r['interpolation']} "
                  f"over {r['walls_used']} source(s) | confidence={r.get('confidence', 0.0)}")
    if report["stalled_cells"]:
        print(f"\n{len(report['stalled_cells'])} cell(s) stalled below the threshold: "
              f"{dict(list(report['stalled_cells'].items())[:10])}"
              + (" ..." if len(report["stalled_cells"]) > 10 else ""))


def _add(parser, name, **kw):
    """Registers --some-flag and its --some_flag alias."""
    names = [f"--{name}"]
    if "-" in name:
        names.append(f"--{name.replace('-', '_')}")
    parser.add_argument(*names, dest=name.replace("-", "_"), **kw)


def build_parser():
    p = argparse.ArgumentParser(description="Ghost Data Engine 3.0: robust, tension-informed 2D reconstruction.")
    _add(p, "width", type=int, default=20, help="grid width (default 20)")
    _add(p, "height", type=int, default=20, help="grid height (default 20)")
    _add(p, "seed", type=int, default=None, help="random seed for reproducible grids")
    _add(p, "num-voids", type=int, default=None, help="number of voids to carve (default: random)")
    _add(p, "max-void-size", type=int, default=5, help="rough max extent of one void")
    _add(p, "shapes", nargs="+", choices=list(SHAPE_BUILDERS), default=None,
         help="void shapes to use (default: all)")
    _add(p, "no-border", action="store_true",
         help="allow voids to touch the grid's outer edge")
    _add(p, "intensity", type=int, default=10, help="base point intensity")
    _add(p, "jitter", type=int, default=0, help="+/- intensity jitter per point")
    _add(p, "connectivity", type=int, choices=(4, 8), default=4, help="void adjacency (4 or 8)")
    _add(p, "pressure-threshold", type=float, default=None,
         help="absolute pressure needed to fill a cell (default: ratio * mean intensity)")
    _add(p, "threshold-ratio", type=float, default=0.5,
         help="used when --pressure-threshold is not given")
    _add(p, "inward-decay", type=float, default=0.85, help="how strongly filled cells press on neighbors")
    _add(p, "void-level", type=float, default=0, help="assumed intensity of missing data")
    _add(p, "no-blend", action="store_true",
         help="switch the post-fill blend off (default: on)")
    _add(p, "audit-limit", type=int, default=15,
         help="audit-trail lines to print (0 = all)")
    _add(p, "audit-json", type=str, default=None, help="write the full audit trail to this JSON file")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.width < 3 or args.height < 3:
        parser.error("width and height must be at least 3")
    if args.max_void_size < 1:
        parser.error("max-void-size must be at least 1")
    if args.num_voids is not None and args.num_voids < 0:
        parser.error("num-voids must be non-negative")
    if args.jitter < 0:
        parser.error("jitter must be non-negative")
    if args.threshold_ratio < 0:
        parser.error("threshold-ratio must be non-negative")
    if args.pressure_threshold is not None and args.pressure_threshold < 0:
        parser.error("pressure-threshold must be non-negative")
    if not 0 <= args.inward_decay <= 1:
        parser.error("inward-decay must be between 0 and 1")

    result = run_ghost_engine(
        width=args.width, height=args.height, seed=args.seed,
        num_voids=args.num_voids, max_void_size=args.max_void_size,
        shapes=args.shapes, keep_border=not args.no_border,
        intensity=args.intensity, intensity_jitter=args.jitter,
        connectivity=args.connectivity, pressure_threshold=args.pressure_threshold,
        threshold_ratio=args.threshold_ratio, inward_decay=args.inward_decay,
        void_level=args.void_level, blend=not args.no_blend,
    )
    print_report(result, audit_limit=args.audit_limit)

    if args.audit_json:
        trail = {str(k): v for k, v in result["report"]["audit_trail"].items()}
        with open(args.audit_json, "w") as fh:
            json.dump(trail, fh, indent=2)
        print(f"\n[+] Full audit trail written to {args.audit_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
