"""
Optional 3D space-time reconstruction module.

Input:
    frames = [
        {(x, y): {"intensity": value}, ...},  # frame/time 0
        {(x, y): {"intensity": value}, ...},  # frame/time 1
        ...
    ]

The module models (x, y, t) as a 3D lattice and propagates values through
six-connected spatial and temporal neighbors. It is separate from the original
2D API and does not modify generator/scanner/tension behavior.
"""


def _neighbors3(coord):
    x, y, t = coord
    return (
        (x + 1, y, t), (x - 1, y, t),
        (x, y + 1, t), (x, y - 1, t),
        (x, y, t + 1), (x, y, t - 1),
    )


def reconstruct_spatiotemporal(
    frames, bounds=None, pressure_threshold=1.0, inward_decay=0.9, max_waves=None
):
    import math

    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError("frames must contain at least one frame")
    if not math.isfinite(float(inward_decay)) or not 0 <= inward_decay <= 1:
        raise ValueError("inward_decay must be finite and between 0 and 1")
    if not math.isfinite(float(pressure_threshold)) or pressure_threshold < 0:
        raise ValueError("pressure_threshold must be finite and non-negative")
    if max_waves is not None and (not isinstance(max_waves, int) or max_waves < 1):
        raise ValueError("max_waves must be a positive integer or None")

    known, all_coords = {}, []
    for t, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise TypeError(f"frame {t} must be a dictionary")
        for (x, y), info in frame.items():
            try:
                value = float(info["intensity"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid intensity at {(x, y, t)}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite intensity at {(x, y, t)}")
            coord = (x, y, t)
            known[coord] = value
            all_coords.append(coord)

    if bounds is None:
        if not all_coords:
            raise ValueError("Cannot infer bounds from empty frames; supply bounds")
        min_x, max_x = min(c[0] for c in all_coords), max(c[0] for c in all_coords)
        min_y, max_y = min(c[1] for c in all_coords), max(c[1] for c in all_coords)
    else:
        if len(bounds) != 4 or not all(isinstance(v, int) for v in bounds):
            raise ValueError("bounds must be four integer coordinates")
        min_x, max_x, min_y, max_y = bounds
        if min_x > max_x or min_y > max_y:
            raise ValueError("bounds minimums must not exceed maximums")
        if any(not (min_x <= x <= max_x and min_y <= y <= max_y) for x, y, _ in all_coords):
            raise ValueError("frame contains coordinates outside bounds")

    missing = {(x, y, t) for t in range(len(frames))
               for x in range(min_x, max_x + 1)
               for y in range(min_y, max_y + 1) if (x, y, t) not in known}
    inferred, audit = {}, {}
    wave = 0
    while missing and (max_waves is None or wave < max_waves):
        wave += 1
        proposals = []
        for coord in sorted(missing):
            sources = [inferred[n] if n in inferred else known[n]
                       for n in _neighbors3(coord) if n in inferred or n in known]
            if not sources:
                continue
            pressure = sum(abs(v) for v in sources)
            mean_value = sum(sources) / len(sources)
            has_inferred = any(n in inferred for n in _neighbors3(coord))
            value = mean_value * (inward_decay if has_inferred else 1.0)
            fallback = pressure < pressure_threshold
            confidence = min(1.0, pressure / (pressure + max(pressure_threshold, 1e-12)))
            proposals.append((coord, value, pressure, confidence, len(sources), fallback))
        if not proposals:
            break
        for coord, value, pressure, confidence, count, fallback in proposals:
            inferred[coord] = float(value)
            audit[coord] = {"wave": wave, "intensity": round(value, 6),
                            "pressure": round(pressure, 6),
                            "confidence_heuristic": round(confidence, 4),
                            "source_count": count, "threshold_fallback_used": fallback}
            missing.remove(coord)

    return {
        "known_voxels": known, "reconstructed_voxels": inferred,
        "unresolved_voxels": sorted(missing), "audit": audit,
        "space_time_dimensions": (max_x - min_x + 1, max_y - min_y + 1, len(frames)),
        "propagation_waves": wave, "confidence_is_heuristic": True,
    }
