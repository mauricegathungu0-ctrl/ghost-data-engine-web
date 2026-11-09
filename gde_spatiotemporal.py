"""
Ghost Data Engine - 3D grid reconstruction  (grid[time][y][x]).

Pure standard library. Independent of the dict-based 2D pipeline and of the
older spatiotemporal.py (which works on {(x, y): {...}} frames); this module
takes plain nested lists and returns plain nested lists.

Input
-----
    grid[t][y][x] = number           recorded value
                  = None / NaN       missing (a void cell)
    (extra sentinels such as -999 can be declared with `missing_values`)

How it works
------------
1. Missing cells are grouped into 3D voids (connected components). Two cells are
   connected through the same neighbour rules used for propagation:
     * spatial  : 4 neighbours (x+-1, y+-1) or 8 with diagonals
     * temporal : t-1 and t+1 (optionally up to `temporal_reach` frames away)
2. Each void is filled by synchronous inward waves from its recorded boundary,
   exactly like the 2D engine: wave 1 fills cells touching recorded data, wave 2
   the cells touching wave-1 cells, and so on.
3. A cell's value is the weighted mean of its already-known neighbours, so it can
   never leave the range of its sources and is never pulled toward zero. Weights:
     * spatial source              : 1 / distance  (1.0 side-on, 0.707 diagonal)
     * temporal source, gap k      : temporal_decay ** k   (closer dates count more)
     * inferred (not recorded)     : additionally x inward_decay ** (wave it was filled)
4. No fabrication: a void with no recorded spatial or temporal boundary cell at
   all (e.g. an entirely empty grid) is reported as unresolved and its cells stay
   None. Cells that a `max_waves` cap never reaches are also left unresolved.

Every reconstructed cell gets an audit row: source (t, y, x) coordinates, wave
number and a heuristic confidence score. The confidence is a bounded diagnostic
(more sources, more recorded weight, shallower wave -> higher); it is NOT a
calibrated probability.

Public API
----------
    reconstruct_3d(grid, ...) -> report dict
"""
import math
from collections import deque

CONFIDENCE_WAVE_DECAY = 0.9   # confidence shrinks by this factor per wave of depth


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _is_missing(value, coord, missing_values):
    """True for None / NaN / declared sentinels. Rejects non-numeric and +-inf."""
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"cell {coord} must be a number or None, got {type(value).__name__}")
    if isinstance(value, float) and math.isnan(value):
        return True
    if math.isinf(value):
        raise ValueError(f"cell {coord} holds an infinite value")
    return value in missing_values


def _check_shape(grid):
    if not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError("grid must be a non-empty list of frames: grid[t][y][x]")
    height = width = None
    for t, frame in enumerate(grid):
        if not isinstance(frame, (list, tuple)) or not frame:
            raise ValueError(f"frame {t} must be a non-empty list of rows")
        for y, row in enumerate(frame):
            if not isinstance(row, (list, tuple)) or not row:
                raise ValueError(f"row (t={t}, y={y}) must be a non-empty list")
            if width is None:
                width = len(row)
            if len(row) != width:
                raise ValueError(f"ragged grid: row (t={t}, y={y}) has {len(row)} cells, expected {width}")
        if height is None:
            height = len(frame)
        if len(frame) != height:
            raise ValueError(f"ragged grid: frame {t} has {len(frame)} rows, expected {height}")
    return len(grid), height, width


def _check_params(spatial_connectivity, temporal_decay, temporal_reach,
                  inward_decay, max_waves):
    if spatial_connectivity not in (4, 8):
        raise ValueError("spatial_connectivity must be 4 or 8")
    for name, v in (("temporal_decay", temporal_decay), ("inward_decay", inward_decay)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v) or not 0 < v <= 1:
            raise ValueError(f"{name} must be a number in (0, 1]")
    if not isinstance(temporal_reach, int) or isinstance(temporal_reach, bool) or temporal_reach < 1:
        raise ValueError("temporal_reach must be a positive integer")
    if max_waves is not None and (not isinstance(max_waves, int)
                                  or isinstance(max_waves, bool) or max_waves < 1):
        raise ValueError("max_waves must be a positive integer or None")


def _build_offsets(spatial_connectivity, temporal_decay, temporal_reach):
    """List of (dt, dy, dx, axis, base_weight)."""
    spatial = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if spatial_connectivity == 8:
        spatial += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    offsets = [(0, dy, dx, "spatial", 1.0 / math.hypot(dx, dy)) for dx, dy in spatial]
    for k in range(1, temporal_reach + 1):
        for sign in (-1, 1):
            offsets.append((sign * k, 0, 0, "temporal", float(temporal_decay) ** k))
    return offsets


def _confidence(source_count, recorded_share, wave):
    support = 1.0 - 0.5 ** max(1, source_count)
    score = support * (0.5 + 0.5 * recorded_share) * (CONFIDENCE_WAVE_DECAY ** (wave - 1))
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def reconstruct_3d(grid, spatial_connectivity=4, temporal_decay=0.5,
                   temporal_reach=1, inward_decay=0.9, max_waves=None,
                   missing_values=()):
    """
    Reconstruct missing cells of grid[t][y][x].

    Parameters
    ----------
    grid                 : nested lists; None / NaN (or `missing_values`) = missing.
    spatial_connectivity : 4 (default) or 8 in-frame neighbours.
    temporal_decay       : weight of a temporal source, relative to a side-on
                           spatial source (1.0). Source `k` frames away weighs
                           temporal_decay ** k. In (0, 1]; lower = temporal
                           neighbours matter less; 1.0 = same as spatial.
    temporal_reach       : how many frames either side may act as sources
                           (default 1 = strictly t-1 and t+1).
    inward_decay         : down-weights *inferred* sources by inward_decay ** depth
                           so recorded data dominates. It never shrinks values.
    max_waves            : optional cap per void; cells beyond it stay unresolved.
    missing_values       : extra sentinels treated as missing, e.g. (-999,).

    Returns
    -------
    dict with:
      "grid"        : copy of the input; reconstructed cells filled, unresolved None
      "audit_trail" : {"(t, y, x)": row} for every reconstructed cell. Row keys:
                      coordinate, void_id, fill_order, wave, intensity,
                      source_coordinates [(t, y, x), ...],
                      sources [{coordinate, axis, recorded, weight}, ...],
                      spatial_sources, temporal_sources, recorded_sources,
                      confidence
      "unresolved"  : {"(t, y, x)": {"void_id", "reason"}}
                      reason: "no_valid_boundary" | "max_waves_reached"
      "voids"       : {void_id: {...size, status, boundary counts, waves...}}
      "summary"     : totals and the parameters used
    The input grid is never modified.
    """
    _check_params(spatial_connectivity, temporal_decay, temporal_reach,
                  inward_decay, max_waves)
    T, H, W = _check_shape(grid)
    missing_values = tuple(missing_values)

    known, missing = {}, set()
    for t in range(T):
        for y in range(H):
            for x in range(W):
                coord = (t, y, x)
                value = grid[t][y][x]
                if _is_missing(value, coord, missing_values):
                    missing.add(coord)
                else:
                    known[coord] = float(value)

    offsets = _build_offsets(spatial_connectivity, temporal_decay, temporal_reach)

    # --- 1. label 3D voids (deterministic: scan order t, y, x) -------------
    voids, seen = [], set()
    for start in sorted(missing):
        if start in seen:
            continue
        cells, queue = set(), deque([start])
        seen.add(start)
        spatial_b, temporal_b = set(), set()
        while queue:
            c = queue.popleft()
            cells.add(c)
            for dt, dy, dx, axis, _ in offsets:
                n = (c[0] + dt, c[1] + dy, c[2] + dx)
                if n in known:
                    (spatial_b if axis == "spatial" else temporal_b).add(n)
                elif n in missing and n not in seen:
                    seen.add(n)
                    queue.append(n)
        voids.append({"id": len(voids) + 1, "cells": cells,
                      "spatial_boundary": spatial_b, "temporal_boundary": temporal_b})

    # --- 2. fill each void by inward waves ---------------------------------
    audit, unresolved, void_report = {}, {}, {}
    inferred = {}                       # cell -> (value, wave filled)
    fill_order = 0
    confidences = []
    max_wave_seen = 0

    for void in voids:
        vid, cells = void["id"], void["cells"]
        t_lo = min(c[0] for c in cells)
        t_hi = max(c[0] for c in cells)
        info = {"size": len(cells), "waves": 0, "cells_filled": 0,
                "cells_unresolved": 0,
                "spatial_boundary_cells": len(void["spatial_boundary"]),
                "temporal_boundary_cells": len(void["temporal_boundary"]),
                "time_span": (t_lo, t_hi)}

        if not void["spatial_boundary"] and not void["temporal_boundary"]:
            for c in sorted(cells):
                unresolved[str(c)] = {"void_id": vid, "reason": "no_valid_boundary"}
            info.update(status="Unresolved", cells_unresolved=len(cells))
            void_report[vid] = info
            continue

        unfilled, wave = set(cells), 0
        while unfilled and (max_waves is None or wave < max_waves):
            wave += 1
            proposals = []
            for cell in sorted(unfilled):
                sources = []            # (coord, axis, recorded, value, weight)
                for dt, dy, dx, axis, base in offsets:
                    n = (cell[0] + dt, cell[1] + dy, cell[2] + dx)
                    if n in known:
                        sources.append((n, axis, True, known[n], base))
                    elif n in inferred:
                        value, depth = inferred[n]
                        sources.append((n, axis, False, value,
                                        max(1e-12, base * inward_decay ** depth)))
                if not sources:
                    continue
                total_w = sum(s[4] for s in sources)
                value = sum(s[3] * s[4] for s in sources) / total_w
                share = sum(s[4] for s in sources if s[2]) / total_w
                proposals.append((cell, value, sources, share))
            if not proposals:
                break
            # synchronous: cells filled in this wave cannot feed each other
            for cell, value, sources, share in proposals:
                fill_order += 1
                value = round(value, 6)
                inferred[cell] = (value, wave)
                unfilled.discard(cell)
                conf = _confidence(len(sources), share, wave)
                confidences.append(conf)
                audit[str(cell)] = {
                    "coordinate": cell, "void_id": vid, "fill_order": fill_order,
                    "wave": wave, "intensity": value,
                    "source_coordinates": [s[0] for s in sources],
                    "sources": [{"coordinate": s[0], "axis": s[1], "recorded": s[2],
                                 "weight": round(s[4], 6)} for s in sources],
                    "spatial_sources": sum(1 for s in sources if s[1] == "spatial"),
                    "temporal_sources": sum(1 for s in sources if s[1] == "temporal"),
                    "recorded_sources": sum(1 for s in sources if s[2]),
                    "confidence": conf,
                }

        for c in sorted(unfilled):
            unresolved[str(c)] = {"void_id": vid, "reason": "max_waves_reached"
                                  if max_waves is not None and wave >= max_waves
                                  else "no_valid_boundary"}
        filled = len(cells) - len(unfilled)
        info.update(waves=wave, cells_filled=filled, cells_unresolved=len(unfilled),
                    status="Healed" if not unfilled else
                    ("Partial" if filled else "Unresolved"))
        max_wave_seen = max(max_wave_seen, wave)
        void_report[vid] = info

    # --- 3. assemble output -------------------------------------------------
    out = []
    for t in range(T):
        frame = []
        for y in range(H):
            row = []
            for x in range(W):
                c = (t, y, x)
                if c in known:
                    row.append(grid[t][y][x])
                elif c in inferred:
                    row.append(inferred[c][0])
                else:
                    row.append(None)
            frame.append(row)
        out.append(frame)

    summary = {
        "dimensions": {"t": T, "y": H, "x": W},
        "total_cells": T * H * W,
        "recorded_cells": len(known),
        "missing_cells": len(missing),
        "total_voids": len(voids),
        "voids_healed": sum(1 for v in void_report.values() if v["status"] == "Healed"),
        "voids_unresolved_or_partial": sum(1 for v in void_report.values()
                                           if v["status"] != "Healed"),
        "cells_reconstructed": len(inferred),
        "cells_unresolved": len(unresolved),
        "propagation_waves": max_wave_seen,
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "confidence_is_heuristic": True,
        "parameters": {"spatial_connectivity": spatial_connectivity,
                       "temporal_decay": temporal_decay,
                       "temporal_reach": temporal_reach,
                       "inward_decay": inward_decay, "max_waves": max_waves,
                       "missing_values": list(missing_values)},
    }
    return {"grid": out, "audit_trail": audit, "unresolved": unresolved,
            "voids": void_report, "summary": summary}


if __name__ == "__main__":
    # Tiny demo: a value ramp in time (value = 10 * t), middle frame has a hole.
    demo = [[[10.0 * t for _ in range(5)] for _ in range(5)] for t in range(3)]
    for y in range(1, 4):
        for x in range(1, 4):
            demo[1][y][x] = None
    rep = reconstruct_3d(demo)
    s = rep["summary"]
    print(f"voids={s['total_voids']} reconstructed={s['cells_reconstructed']} "
          f"unresolved={s['cells_unresolved']} waves={s['propagation_waves']} "
          f"mean_confidence={s['mean_confidence']} (heuristic)")
    centre = rep["audit_trail"]["(1, 2, 2)"]
    print("centre cell:", centre["intensity"], "wave", centre["wave"],
          "sources", centre["source_coordinates"])
    empty = reconstruct_3d([[[None, None], [None, None]]])
    print("empty grid ->", empty["summary"]["cells_reconstructed"], "reconstructed,",
          empty["summary"]["cells_unresolved"], "unresolved (nothing fabricated)")
