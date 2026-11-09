"""Ghost Data Engine 3.0: robust, tension-informed multi-wave reconstruction.

Compatibility: reconstruct_void(...) and the original report keys are retained.
The algorithm remains a classical numerical heuristic, not a guarantee of truth.
"""
from __future__ import annotations

import math
import statistics


def _cell_neighbors(cell):
    x, y = cell
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _parse_cell(key):
    """'(x, y)' audit-trail key -> (x, y) tuple."""
    x, y = key.strip("()").split(",")
    return int(x), int(y)


def _finite_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _confidence(pressure, threshold, source_count, tension_support=0.0):
    """Bounded diagnostic score; explicitly not a calibrated probability."""
    if threshold <= 0:
        strength = 1.0 if pressure > 0 else 0.0
    else:
        strength = max(0.0, min(1.0, pressure / (pressure + threshold)))
    support = 1.0 - (0.5 ** max(1, source_count))
    score = strength * (0.5 + 0.5 * support)
    score *= 0.85 + 0.15 * max(0.0, min(1.0, tension_support))
    return round(max(0.0, min(1.0, score)), 4)


def _weighted_mean(items):
    total_weight = sum(weight for _, weight in items)
    if total_weight <= 0 or not math.isfinite(total_weight):
        return sum(value for value, _ in items) / len(items)
    return sum(value * weight for value, weight in items) / total_weight


def _local_anchor(cells, values):
    """Mean of recorded values touching a void (the void's own 'surroundings')."""
    seen, total = set(), 0.0
    for cell in cells:
        for n in _cell_neighbors(cell):
            if n in values and n not in seen:
                seen.add(n)
                total += values[n]
    return total / len(seen) if seen else None


def _refine_smooth(fixed, unknown_cells, start, max_iter, tol, lo, hi, membrane=0.02):
    """Dependency-free thin-plate (biharmonic) relaxation of one void.

    Minimises the squared 5-point Laplacian over every grid cell whose stencil
    touches the void, solved by conjugate gradients on the normal equations.
    `fixed` holds recorded values (and other voids' current estimates); the void
    cells in `unknown_cells` start from `start` (the wave-filled estimate).
    A small membrane term (first differences, weight `membrane`) keeps the system
    well-posed at grid edges and for tiny voids, where curvature alone is
    under-constrained. Results are clamped to [lo, hi] (the recorded range).
    Returns (values_dict, iterations_used, converged).
    """
    unknown = sorted(unknown_cells)
    index = {c: i for i, c in enumerate(unknown)}
    n = len(unknown)
    if n == 0:
        return {}, 0, True

    def available(c):
        return c in index or c in fixed

    # Cells on the grid edge / next to a hole lack a full neighbourhood; curvature
    # would only extrapolate there, so they are governed by the membrane term alone.
    edge_unknown = {c for c in unknown if not all(available(nb) for nb in _cell_neighbors(c))}
    centers = set(unknown)
    for c in unknown:
        centers.update(_cell_neighbors(c))
    rows = []   # each row: (list of (unknown_index, coef), constant_from_fixed)
    for c in sorted(centers):
        stencil = [(c, 4.0)] + [(nb, -1.0) for nb in _cell_neighbors(c)]
        if not all(available(cell) for cell, _ in stencil):
            continue                       # incomplete stencil (grid edge / hole): skip
        if edge_unknown and any(cell in edge_unknown for cell, _ in stencil):
            continue
        terms, const = [], 0.0
        for cell, coef in stencil:
            if cell in index:
                terms.append((index[cell], coef))
            else:
                const += coef * fixed[cell]
        if terms:
            rows.append((terms, const))

    if membrane > 0:
        w = math.sqrt(membrane)
        seen_pairs = set()
        for c in unknown:
            for nb in _cell_neighbors(c):
                if not available(nb):
                    continue
                pair = (c, nb) if nb not in index or index[c] < index[nb] else (nb, c)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                terms, const = [(index[c], w)], 0.0
                if nb in index:
                    terms.append((index[nb], -w))
                else:
                    const -= w * fixed[nb]
                rows.append((terms, const))

    x = [float(start[c]) for c in unknown]
    if not rows:
        return {c: x[i] for i, c in enumerate(unknown)}, 0, True

    eps = 1e-6        # tiny pull toward the wave estimate keeps the system well-posed

    def matvec(vec):
        out = [eps * v for v in vec]
        for terms, _ in rows:
            t = 0.0
            for i, coef in terms:
                t += coef * vec[i]
            for i, coef in terms:
                out[i] += coef * t
        return out

    b = [eps * v for v in x]
    for terms, const in rows:
        for i, coef in terms:
            b[i] -= coef * const

    ax = matvec(x)
    r = [b[i] - ax[i] for i in range(n)]
    p = r[:]
    rs = sum(v * v for v in r)
    bnorm = math.sqrt(sum(v * v for v in b)) or 1.0
    used, converged = 0, math.sqrt(rs) <= tol * bnorm
    while not converged and used < max_iter:
        ap = matvec(p)
        denom = sum(p[i] * ap[i] for i in range(n))
        if denom <= 0 or not math.isfinite(denom):
            break
        alpha = rs / denom
        x = [x[i] + alpha * p[i] for i in range(n)]
        r = [r[i] - alpha * ap[i] for i in range(n)]
        rs_new = sum(v * v for v in r)
        used += 1
        if math.sqrt(rs_new) <= tol * bnorm:
            converged = True
            break
        beta = rs_new / rs
        p = [r[i] + beta * p[i] for i in range(n)]
        rs = rs_new

    out = {}
    for i, c in enumerate(unknown):
        v = x[i]
        if not math.isfinite(v):
            v = float(start[c])
        out[c] = min(hi, max(lo, v))
    return out, used, converged


def _relax_laplace(fixed, unknown_cells, start, tol, max_iter, lo, hi):
    """Neighbour-average relaxation of one void (Laplace / harmonic fill).

    Repeatedly replaces every void cell with the mean of its 4 neighbours while
    every recorded cell (and every cell in `fixed`) stays put, until the largest
    single-cell change in a sweep drops below `tol`. Uses Gauss-Seidel sweeps with
    over-relaxation (SOR); that only speeds up convergence, the fixed point is
    exactly "each cell equals the average of its neighbours". Neighbours that do
    not exist (grid edge) or were never filled are simply left out of the average.
    Results stay inside [lo, hi]. Returns (values_dict, sweeps_used, converged).
    """
    cells = sorted(unknown_cells)
    n = len(cells)
    if n == 0:
        return {}, 0, True
    index = {c: i for i, c in enumerate(cells)}
    table = []                       # per cell: (unknown neighbour indices, fixed sum, count)
    for c in cells:
        ui, fsum, cnt = [], 0.0, 0
        for nb in _cell_neighbors(c):
            if nb in index:
                ui.append(index[nb])
                cnt += 1
            elif nb in fixed:
                fsum += fixed[nb]
                cnt += 1
        table.append((ui, fsum, cnt))
    x = [float(start[c]) for c in cells]
    span = max(max(c[0] for c in cells) - min(c[0] for c in cells),
               max(c[1] for c in cells) - min(c[1] for c in cells)) + 1
    omega = 2.0 / (1.0 + math.sin(math.pi / (span + 1)))   # SOR optimum for a span-wide patch
    omega = min(1.95, max(1.0, omega))
    used, converged = 0, False
    while used < max_iter:
        used += 1
        delta = 0.0
        for i in range(n):
            ui, fsum, cnt = table[i]
            if cnt == 0:
                continue
            s = fsum
            for j in ui:
                s += x[j]
            new = x[i] + omega * (s / cnt - x[i])
            d = abs(new - x[i])
            if d > delta:
                delta = d
            x[i] = new
        if delta < tol:
            converged = True
            break
    out = {}
    for i, c in enumerate(cells):
        v = x[i]
        if not math.isfinite(v):
            v = float(start[c])
        out[c] = min(hi, max(lo, v))
    return out, used, converged


def reconstruct_void(
    void_clusters,
    tension_data,
    data_grid,
    pressure_threshold=None,
    threshold_ratio=0.5,
    inward_decay=0.85,
    void_level=0,
    mode="accurate",
    refine_max_iter=600,
    refine_tol=1e-5,
    relax=True,
    relax_tol=0.001,
    relax_max_iter=20000,
    relax_depth_scale=None,
):
    """Reconstruct voids in synchronous waves using measured boundary tension.

    Boundary wall weights now affect source influence. If the configured
    pressure threshold blocks progress, a conservative weighted-neighbor
    fallback is used for cells that have at least one valid source. Cells with
    no path to any recorded source remain explicitly unresolved.

    mode
        "accurate" (default): same wave fill and audit trail, but
          * wave weights use wall geometry, not brightness above void_level;
          * inward decay relaxes toward the void's own surroundings, not
            toward void_level (so 0.85 no longer darkens real data);
          * a thin-plate (biharmonic) relaxation then settles every void so
            slopes and edges continue smoothly across it. Each audit row keeps
            the original wave value as "wave_intensity".
        "legacy": the exact 3.0 behaviour, unchanged.
    refine_max_iter / refine_tol bound the relaxation (accurate mode only).

    relax (default True) adds a final neighbour-average relaxation after the fill:
    each filled cell is repeatedly replaced by the mean of its 4 neighbours, with
    recorded cells held fixed, until the largest change is below relax_tol
    (default 0.001) or relax_max_iter sweeps have run. Set relax=False to get the
    pre-relaxation behaviour. It applies in mode="accurate" only; mode="legacy" is
    the frozen 3.0 algorithm and ignores it. Each relaxed audit row keeps the
    value from before this step as "pre_relax_intensity".

    relax_depth_scale (default None = off) blends instead of replacing: a cell at
    depth d (its wave number, 1 = touching recorded data) ends at
    w * pre_relax + (1 - w) * relaxed with w = exp(-(d - 1) / relax_depth_scale),
    so cells at the edge of a gap keep the pre-relaxation value and the deep
    centre follows the neighbour-average value. Only used when relax is on.
    """
    inward_decay = _finite_number(inward_decay, "inward_decay")
    threshold_ratio = _finite_number(threshold_ratio, "threshold_ratio")
    void_level = _finite_number(void_level, "void_level")
    if not 0 <= inward_decay <= 1:
        raise ValueError("inward_decay must be between 0 and 1")
    if mode not in ("accurate", "legacy"):
        raise ValueError("mode must be 'accurate' or 'legacy'")
    if not isinstance(refine_max_iter, int) or refine_max_iter < 0:
        raise ValueError("refine_max_iter must be a non-negative integer")
    refine_tol = _finite_number(refine_tol, "refine_tol")
    if refine_tol <= 0:
        raise ValueError("refine_tol must be positive")
    if not isinstance(relax, bool):
        raise ValueError("relax must be True or False")
    relax_tol = _finite_number(relax_tol, "relax_tol")
    if relax_tol <= 0:
        raise ValueError("relax_tol must be positive")
    if not isinstance(relax_max_iter, int) or isinstance(relax_max_iter, bool) or relax_max_iter < 1:
        raise ValueError("relax_max_iter must be a positive integer")
    if relax_depth_scale is not None:
        relax_depth_scale = _finite_number(relax_depth_scale, "relax_depth_scale")
        if relax_depth_scale <= 0:
            raise ValueError("relax_depth_scale must be positive or None")
    accurate = mode == "accurate"
    if threshold_ratio < 0:
        raise ValueError("threshold_ratio must be non-negative")
    if not isinstance(data_grid, dict):
        raise TypeError("data_grid must be a dictionary keyed by (x, y)")
    if not isinstance(void_clusters, (list, tuple)):
        raise TypeError("void_clusters must be a list or tuple")

    values = {}
    for coord, info in data_grid.items():
        if not (isinstance(coord, tuple) and len(coord) == 2):
            raise ValueError(f"Invalid coordinate: {coord!r}")
        if not isinstance(info, dict) or "intensity" not in info:
            raise ValueError(f"Missing intensity for coordinate {coord!r}")
        values[coord] = _finite_number(info["intensity"], f"intensity at {coord}")

    intensities = list(values.values())
    scale = statistics.median(abs(v - void_level) for v in intensities) if intensities else 1.0
    if scale <= 1e-12:
        scale = max((abs(v - void_level) for v in intensities), default=1.0)
    if pressure_threshold is None:
        pressure_threshold = scale * threshold_ratio
    else:
        pressure_threshold = _finite_number(pressure_threshold, "pressure_threshold")
    if pressure_threshold < 0:
        raise ValueError("pressure_threshold must be non-negative")

    audit_trail, stalled_cells, per_void_results = {}, {}, {}
    all_confidence = {}
    wave_store = []
    total_void_cells = sum(len(v.get("cells", [])) for v in void_clusters)
    cells_reconstructed = cells_stalled = total_waves = 0
    cumulative_tension = 0.0
    highest_tension_coord, max_tension_val = None, -1.0

    for void in void_clusters:
        if not isinstance(void, dict) or "id" not in void or "cells" not in void:
            raise ValueError("Each void must contain 'id' and 'cells'")
        vid = void["id"]
        cells = set(void["cells"])
        if any(c in values for c in cells):
            raise ValueError(f"Void {vid} overlaps recorded coordinates")
        if any(not (isinstance(c, tuple) and len(c) == 2) for c in cells):
            raise ValueError(f"Void {vid} contains an invalid coordinate")
        unfilled = set(cells)
        filled_values, filled_depths = {}, {}
        anchor = void_level
        if accurate:
            local = _local_anchor(cells, values)
            if local is not None:
                anchor = local
        v_tension = tension_data.get(vid, {}) if isinstance(tension_data, dict) else {}
        edge_tension = v_tension.get("edge_tension", {})
        directional = v_tension.get("directional_pressure", {})
        max_direction = max((abs(_finite_number(v, "directional pressure")) for v in directional.values()), default=0.0)
        fill_order = wave = filled_in_void = 0
        used_fallback = 0

        # A hard upper bound protects against accidental non-progress loops.
        max_waves = max(1, len(unfilled) + 1)
        while unfilled and wave < max_waves:
            wave += 1
            total_waves += 1
            proposals = []
            for cell in sorted(unfilled):
                source_items = []
                pressure = 0.0
                for neighbor in _cell_neighbors(cell):
                    if neighbor in filled_values:
                        source_value = filled_values[neighbor]
                        is_inferred = True
                        source_depth = filled_depths[neighbor]
                    elif neighbor in values:
                        source_value = values[neighbor]
                        is_inferred = False
                        source_depth = 0
                    else:
                        continue
                    magnitude = abs(source_value - void_level)
                    pressure += magnitude
                    # The original tension calculator's wall weights now alter
                    # how strongly perimeter observations influence a cell.
                    wall = edge_tension.get(neighbor, {})
                    raw_wall_weight = abs(float(wall.get("weight", 0.0))) if wall else 0.0
                    pressed = max(1, int(wall.get("cells_pressed", 1))) if wall else 1
                    if accurate:
                        # geometry only: how many void cells the wall touches
                        tension_factor = math.sqrt(pressed)
                    else:
                        tension_factor = 1.0 + min(4.0, raw_wall_weight / pressed / max(scale, 1e-12))
                    depth_factor = inward_decay ** max(0, source_depth)
                    weight = max(1e-9, tension_factor * depth_factor)
                    source_items.append((source_value, weight, neighbor, is_inferred))

                if not source_items:
                    continue
                weighted = _weighted_mean([(v, w) for v, w, _, _ in source_items])
                has_inferred = any(is_inferred for _, _, _, is_inferred in source_items)
                if has_inferred:
                    weighted = anchor + (weighted - anchor) * inward_decay
                threshold_passed = pressure >= pressure_threshold
                # Safe fallback avoids threshold-induced stalls while retaining
                # source-connectedness: no source means no invented value.
                fallback = not threshold_passed
                confidence = _confidence(pressure, pressure_threshold, len(source_items), max_direction / (max_direction + scale + 1e-12))
                proposals.append((cell, weighted, pressure, source_items, confidence, fallback))

            if not proposals:
                break
            # Synchronous updates: proposals in this wave cannot feed each other.
            for cell, inferred, pressure, sources, conf, fallback in proposals:
                fill_order += 1
                inferred = _finite_number(inferred, f"reconstruction at {cell}")
                filled_values[cell] = round(inferred, 6)
                filled_depths[cell] = wave
                unfilled.remove(cell)
                filled_in_void += 1
                cells_reconstructed += 1
                if fallback:
                    used_fallback += 1
                all_confidence[str(cell)] = conf
                source_coords = [coord for _, _, coord, _ in sources]
                audit_trail[str(cell)] = {
                    "void_id": vid, "fill_order": fill_order,
                    "ingrowth_depth": wave, "wave": wave,
                    "intensity": round(inferred, 6),
                    "source_tension": round(pressure, 6),
                    "pressure_threshold": pressure_threshold,
                    "interpolation": "tension_weighted_recursive_diffusion",
                    "walls_used": len(sources), "source_coordinates": source_coords,
                    "confidence": conf,
                    "threshold_fallback_used": fallback,
                }
                if pressure > max_tension_val:
                    max_tension_val, highest_tension_coord = pressure, cell
                cumulative_tension += pressure

        for cell in sorted(unfilled):
            stalled_cells[str(cell)] = {"void_id": vid, "tension": 0.0, "wave_checked": wave}
        stalled_in_void = len(unfilled)
        wave_store.append((vid, dict(filled_values)))
        cells_stalled += stalled_in_void
        confidence_values = [all_confidence[str(c)] for c in filled_values]
        per_void_results[vid] = {
            "classification": "Healed" if not unfilled else "Partial",
            "rule_fired": "tension_weighted_recursive_diffusion_v3",
            "metrics": {
                "size": len(cells), "waves": wave,
                "mean_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else 0.0,
            },
            "cells_filled": filled_in_void, "cells_stalled": stalled_in_void,
            "strongest_side": v_tension.get("strongest_side"),
            "net_pull": v_tension.get("net_direction"),
            "threshold_fallback_cells": used_fallback,
        }

    refinement = {"mode": mode, "applied": False, "voids_refined": 0,
                  "iterations": 0, "not_converged_voids": 0}
    if accurate and refine_max_iter > 0 and wave_store and values:
        lo, hi = min(values.values()), max(values.values())
        current = dict(values)
        for _, wv in wave_store:                 # every void starts from its wave fill
            current.update(wv)
        for vid, wv in wave_store:
            if not wv:
                continue
            fixed = {c: v for c, v in current.items() if c not in wv}
            new_vals, used, converged = _refine_smooth(
                fixed, set(wv), wv, refine_max_iter, refine_tol, lo, hi)
            current.update(new_vals)
            refinement["voids_refined"] += 1
            refinement["iterations"] += used
            if not converged:
                refinement["not_converged_voids"] += 1
            for cell, new_value in new_vals.items():
                row = audit_trail[str(cell)]
                row["wave_intensity"] = row["intensity"]
                row["intensity"] = round(new_value, 6)
                row["interpolation"] = "tension_weighted_recursive_diffusion+thin_plate_refine"
            per_void_results[vid]["refined"] = True
        refinement["applied"] = refinement["voids_refined"] > 0

    relaxation = {"enabled": bool(accurate and relax), "applied": False, "voids_relaxed": 0,
                  "sweeps": 0, "not_converged_voids": 0, "tol": relax_tol,
                  "depth_scale": relax_depth_scale}
    if accurate and relax and wave_store and values:
        lo, hi = min(values.values()), max(values.values())
        estimate = dict(values)                  # recorded + every void's current estimate
        for cell_key, row in audit_trail.items():
            estimate[_parse_cell(cell_key)] = row["intensity"]
        for vid, wv in wave_store:
            if not wv:
                continue
            fixed = {c: v for c, v in estimate.items() if c not in wv}
            start = {c: estimate[c] for c in wv}
            new_vals, used, converged = _relax_laplace(
                fixed, set(wv), start, relax_tol, relax_max_iter, lo, hi)
            estimate.update(new_vals)
            relaxation["voids_relaxed"] += 1
            relaxation["sweeps"] += used
            if not converged:
                relaxation["not_converged_voids"] += 1
            for cell, new_value in new_vals.items():
                row = audit_trail[str(cell)]
                row["pre_relax_intensity"] = row["intensity"]
                if relax_depth_scale is not None:
                    w = math.exp(-(row["wave"] - 1) / relax_depth_scale)
                    new_value = w * row["intensity"] + (1.0 - w) * new_value
                row["intensity"] = round(new_value, 6)
                row["interpolation"] += ("+depth_blended_relax" if relax_depth_scale is not None
                                         else "+neighbour_average_relax")
            per_void_results[vid]["relaxed"] = True
        relaxation["applied"] = relaxation["voids_relaxed"] > 0

    filled_count = len(all_confidence)
    summary = {
        "total_voids": len(void_clusters), "total_void_cells": total_void_cells,
        "cells_reconstructed": cells_reconstructed, "cells_stalled": cells_stalled,
        "pressure_threshold": pressure_threshold,
        "cumulative_structural_tension": round(cumulative_tension, 6),
        "highest_tension_coordinate": highest_tension_coord,
        "inference": "Tension-weighted recursive multi-wave diffusion with conservative source fallback.",
        "propagation_waves": total_waves,
        "mean_confidence": round(sum(all_confidence.values()) / filled_count, 4) if filled_count else 0.0,
        "confidence_is_heuristic": True,
        "threshold_fallback_cells": sum(v["threshold_fallback_cells"] for v in per_void_results.values()),
        "unresolved_without_source": cells_stalled,
        "mode": mode,
        "refinement": refinement,
        "relaxation": relaxation,
    }
    return {"summary": summary, "voids": per_void_results,
            "audit_trail": audit_trail, "stalled_cells": stalled_cells,
            "cell_confidence": all_confidence}
