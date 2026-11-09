# scanner.py
from collections import deque

NEIGHBORS_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
NEIGHBORS_8 = NEIGHBORS_4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _find_void_coordinates(data_grid, bounds=None):
    """Every coordinate inside the bounding box that has no recorded data."""
    if bounds is None:
        if not data_grid:
            return set()
        xs = [c[0] for c in data_grid]
        ys = [c[1] for c in data_grid]
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    else:
        if len(bounds) != 4:
            raise ValueError("bounds must be (min_x, max_x, min_y, max_y)")
        min_x, max_x, min_y, max_y = bounds
        if not all(isinstance(v, int) for v in bounds):
            raise ValueError("bounds must contain integer coordinates")
        if min_x > max_x or min_y > max_y:
            raise ValueError("bounds minimums must not exceed maximums")
        outside = [c for c in data_grid if not (min_x <= c[0] <= max_x and min_y <= c[1] <= max_y)]
        if outside:
            raise ValueError(f"data_grid contains coordinates outside bounds: {outside[:3]}")

    return {
        (x, y)
        for x in range(min_x, max_x + 1)
        for y in range(min_y, max_y + 1)
        if (x, y) not in data_grid
    }


def _flood_fill(start, unvisited, offsets):
    """Breadth-first search: collect every void cell connected to `start`."""
    cluster = set()
    queue = deque([start])
    unvisited.discard(start)

    while queue:
        x, y = queue.popleft()
        cluster.add((x, y))
        for dx, dy in offsets:
            nxt = (x + dx, y + dy)
            if nxt in unvisited:       # adjacent AND still a void AND not yet claimed
                unvisited.discard(nxt)
                queue.append(nxt)
    return cluster


def _classify_boundaries(cluster, data_grid, offsets):
    """
    Splits one cluster into:
      interior_cells : void cells completely surrounded by other void cells
      edge_cells     : void cells that touch at least one recorded cell
      perimeter      : the recorded cells forming the 'walls' of the hole
    """
    edge_cells = set()
    perimeter = set()

    for x, y in cluster:
        for dx, dy in offsets:
            neighbor = (x + dx, y + dy)
            if neighbor in data_grid:
                edge_cells.add((x, y))
                perimeter.add(neighbor)

    return cluster - edge_cells, edge_cells, perimeter


def scan_for_void_clusters(data_grid, connectivity=4, bounds=None):
    """
    Finds missing coordinates and groups touching ones into distinct void objects
    (connected-component labeling).

    Parameters
    ----------
    data_grid    : dict of {(x, y): {...}} recorded points.
    connectivity : 4 = cells touch only via sides (default);
                   8 = diagonal contact also joins cells into one void.
    bounds       : optional (min_x, max_x, min_y, max_y) to scan. Defaults to the
                   bounding box of the recorded data. Pass the full grid size to
                   catch voids that touch the dataset's outer edge.

    Returns
    -------
    List of dicts, one per void, numbered from 1 in scan order:
        id, size, cells, interior_cells, edge_cells, perimeter, bounding_box
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if not data_grid and bounds is None:
        return []

    offsets = NEIGHBORS_4 if connectivity == 4 else NEIGHBORS_8
    unvisited = _find_void_coordinates(data_grid, bounds)
    total_void_cells = len(unvisited)

    voids = []
    while unvisited:
        start = min(unvisited)  # deterministic numbering: left-to-right, bottom-to-top
        cluster = _flood_fill(start, unvisited, offsets)
        interior, edge, perimeter = _classify_boundaries(cluster, data_grid, offsets)

        xs = [c[0] for c in cluster]
        ys = [c[1] for c in cluster]
        voids.append({
            "id": len(voids) + 1,
            "size": len(cluster),
            "cells": sorted(cluster),
            "interior_cells": sorted(interior),
            "edge_cells": sorted(edge),
            "perimeter": sorted(perimeter),
            "bounding_box": (min(xs), max(xs), min(ys), max(ys)),
        })

    print(f"[+] Scan complete. Detected {total_void_cells} ghost data coordinates "
          f"grouped into {len(voids)} distinct void(s).")
    for v in voids:
        print(f"    Void #{v['id']}: {v['size']} cells "
              f"({len(v['interior_cells'])} interior, {len(v['edge_cells'])} edge), "
              f"{len(v['perimeter'])} perimeter walls")
    return voids


def scan_for_voids(data_grid):
    """
    Backward-compatible flat list of void coordinates, so the existing
    tension.py / main.py keep working until they are upgraded to use clusters.
    """
    voids = scan_for_void_clusters(data_grid)
    return sorted(cell for v in voids for cell in v["cells"])
