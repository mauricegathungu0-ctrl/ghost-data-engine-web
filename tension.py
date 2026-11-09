# tension.py
"""
Boundary tension calculator.

Produces exactly the data contract reconstructor.py consumes:

    {void_id: {
        "total_tension"        : sum of every perimeter wall's tension,
        "edge_tension"         : {wall_coord: {"weight": ..., "intensity": ...,
                                               "cells_pressed": ...}},
        "directional_pressure" : {"North": p, "South": p, "East": p, "West": p},
        "net_direction"        : (dx, dy) resultant push on the void,
        "strongest_side"       : compass side pressing hardest (or None),
        ...extra diagnostics...
    }}

Conventions (consistent with scanner.py, which numbers cells bottom-to-top):
    +x = East, +y = North.
    A wall lying NORTH of a void cell presses that cell toward the SOUTH, so its
    pressure is booked under "North" (where the pressure *comes from*) and its
    push vector is (0, -1).

A wall's tension ("weight"):
    weight = (number of void cells the wall presses on) * (intensity - void_level)
reconstructor.py reads this weight (and "cells_pressed") from "edge_tension".
"""

SIDES = ("North", "South", "East", "West")

# offset from a void cell to the wall, -> side the wall sits on
_SIDE_OF_OFFSET = {(0, 1): "North", (0, -1): "South", (1, 0): "East", (-1, 0): "West"}

# unit vector a wall on that side pushes the void along
_PUSH = {"North": (0, -1), "South": (0, 1), "East": (-1, 0), "West": (1, 0)}


def calculate_boundary_tension(void_clusters, data_grid, void_level=0):
    """
    Parameters
    ----------
    void_clusters : list of void dicts from scanner.scan_for_void_clusters().
    data_grid     : the recorded grid {(x, y): {"intensity": ...}}.
    void_level    : assumed intensity of missing data. Must match the value
                    passed to reconstruct_void().

    Returns
    -------
    dict keyed by void id (reconstructor does tension_report[void["id"]]).
    """
    reports = {}

    for void in void_clusters:
        cells = set(void["cells"])
        edge_tension = {}
        directional = {side: 0.0 for side in SIDES}

        for wall in void["perimeter"]:
            if wall not in data_grid:
                continue
            intensity = data_grid[wall].get("intensity", 0)
            gradient = intensity - void_level

            pressed = 0
            for (dx, dy), side in _SIDE_OF_OFFSET.items():
                # wall = cell + (dx, dy)  ->  cell = wall - (dx, dy)
                if (wall[0] - dx, wall[1] - dy) in cells:
                    pressed += 1
                    directional[side] += gradient

            edge_tension[wall] = {
                "weight": round(pressed * gradient, 3),
                "intensity": intensity,
                "cells_pressed": pressed,
            }

        total = sum(e["weight"] for e in edge_tension.values())
        net_x = sum(directional[s] * _PUSH[s][0] for s in SIDES)
        net_y = sum(directional[s] * _PUSH[s][1] for s in SIDES)

        strongest = max(SIDES, key=lambda s: directional[s])
        if directional[strongest] <= 0:
            strongest = None

        peak = max(edge_tension, key=lambda c: edge_tension[c]["weight"], default=None)

        reports[void["id"]] = {
            "void_id": void["id"],
            "total_tension": round(total, 3),
            "edge_tension": edge_tension,
            "directional_pressure": {s: round(v, 3) for s, v in directional.items()},
            "net_direction": (round(net_x, 3), round(net_y, 3)),
            "strongest_side": strongest,
            # diagnostics
            "highest_tension_coordinate": peak,
            "average_tension": round(total / len(edge_tension), 2) if edge_tension else 0.0,
            "perimeter_wall_count": len(edge_tension),
        }

    print(f"[+] Boundary tension calculated across {len(void_clusters)} void cluster(s).")
    return reports


if __name__ == "__main__":
    from generator import generate_dataset
    from scanner import scan_for_void_clusters

    grid = generate_dataset(seed=42)
    clusters = scan_for_void_clusters(grid)
    reports = calculate_boundary_tension(clusters, grid)
    for vid, r in reports.items():
        print(f"    - Void #{vid}: total tension = {r['total_tension']}, "
              f"peak wall = {r['highest_tension_coordinate']}, "
              f"strongest side = {r['strongest_side']}, net push = {r['net_direction']}")
