# generator.py
import random

# ---------------------------------------------------------------------------
# Shape builders
# Each returns a set of (dx, dy) offsets relative to an anchor point (0, 0).
# ---------------------------------------------------------------------------

def _rectangle(rng, size):
    w = rng.randint(1, max(1, size))
    h = rng.randint(1, max(1, size))
    return {(dx, dy) for dx in range(w) for dy in range(h)}


def _l_shape(rng, size):
    arm = max(2, size)
    thick = rng.randint(1, max(1, arm // 2))
    cells = {(dx, dy) for dx in range(arm) for dy in range(thick)}   # horizontal arm
    cells |= {(dx, dy) for dx in range(thick) for dy in range(arm)}  # vertical arm
    return cells


def _cross(rng, size):
    r = max(1, size // 2)
    cells = {(dx, 0) for dx in range(-r, r + 1)}
    cells |= {(0, dy) for dy in range(-r, r + 1)}
    return cells


def _diagonal(rng, size):
    length = max(2, size)
    step = rng.choice([1, -1])
    cells = {(i, i * step) for i in range(length)}
    if rng.random() < 0.5:  # sometimes thicken the line
        cells |= {(i + 1, i * step) for i in range(length)}
    return cells


def _blob(rng, size):
    """Irregular organic shape grown cell by cell from a seed."""
    target = rng.randint(max(2, size), max(3, size * size // 2 + 1))
    cells = {(0, 0)}
    while len(cells) < target:
        cx, cy = rng.choice(list(cells))
        dx, dy = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        cells.add((cx + dx, cy + dy))
    return cells


def _speckle(rng, size):
    """A few isolated dropped points, like sensor glitches."""
    count = rng.randint(1, max(1, size))
    return {(rng.randint(0, size), rng.randint(0, size)) for _ in range(count)}


SHAPE_BUILDERS = {
    "rectangle": _rectangle,
    "l_shape": _l_shape,
    "cross": _cross,
    "diagonal": _diagonal,
    "blob": _blob,
    "speckle": _speckle,
}


# ---------------------------------------------------------------------------
# Void placement
# ---------------------------------------------------------------------------

def _carve_voids(rng, width, height, num_voids, max_void_size, shapes, keep_border):
    """Returns (set of all void coordinates, list of per-void metadata)."""
    # Keep a 1-cell recorded frame so the scanner's bounding box stays intact.
    lo = 2 if keep_border else 1
    x_max = width - 1 if keep_border else width
    y_max = height - 1 if keep_border else height

    void_cells = set()
    void_info = []

    for _ in range(num_voids):
        shape_name = rng.choice(shapes)
        offsets = SHAPE_BUILDERS[shape_name](rng, max_void_size)

        # Normalize so the shape's top-left sits at (0, 0)
        min_dx = min(dx for dx, _ in offsets)
        min_dy = min(dy for _, dy in offsets)
        offsets = {(dx - min_dx, dy - min_dy) for dx, dy in offsets}

        # Random anchor; clip anything that would fall outside the allowed zone
        ax = rng.randint(lo, max(lo, x_max))
        ay = rng.randint(lo, max(lo, y_max))
        placed = {
            (ax + dx, ay + dy)
            for dx, dy in offsets
            if lo <= ax + dx <= x_max and lo <= ay + dy <= y_max
        }

        if placed:
            void_cells |= placed
            void_info.append({"shape": shape_name, "anchor": (ax, ay), "size": len(placed)})

    return void_cells, void_info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(
    width=20,
    height=20,
    num_voids=None,
    max_void_size=5,
    shapes=None,
    seed=None,
    keep_border=True,
    intensity=10,
    intensity_jitter=0,
    return_ground_truth=False,
):
    """
    Generates a 2D coordinate grid (dict) with randomized, irregular voids.

    Parameters
    ----------
    width, height     : grid dimensions (coordinates run 1..width / 1..height).
    num_voids         : how many voids to scatter. None = random, scaled to grid size.
    max_void_size     : rough upper bound on a single void's extent.
    shapes            : list of shape names to use. None = all of
                        rectangle, l_shape, cross, diagonal, blob, speckle.
    seed              : set for reproducible grids (great for testing).
    keep_border       : keep the outer ring recorded so voids stay *inside*
                        the dataset's bounding box (what scanner.py detects).
    intensity         : base intensity of recorded points.
    intensity_jitter  : +/- random variation applied to each point's intensity.
    return_ground_truth : if True, returns (grid, set_of_true_void_coords)
                        so you can check the scanner found everything.

    Returns
    -------
    data_grid  (or (data_grid, true_voids) when return_ground_truth=True)
    """
    for name, value in (("width", width), ("height", height), ("max_void_size", max_void_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if num_voids is not None and (not isinstance(num_voids, int) or isinstance(num_voids, bool) or num_voids < 0):
        raise ValueError("num_voids must be a non-negative integer or None")
    if not isinstance(intensity_jitter, int) or isinstance(intensity_jitter, bool) or intensity_jitter < 0:
        raise ValueError("intensity_jitter must be a non-negative integer")

    rng = random.Random(seed)

    shapes = shapes or list(SHAPE_BUILDERS)
    if not shapes:
        raise ValueError("shapes must contain at least one supported shape")
    unknown = [s for s in shapes if s not in SHAPE_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown shapes {unknown}. Choose from {list(SHAPE_BUILDERS)}")

    if num_voids is None:
        num_voids = rng.randint(2, max(3, (width * height) // 40))

    void_cells, void_info = _carve_voids(
        rng, width, height, num_voids, max_void_size, shapes, keep_border
    )

    data_grid = {}
    for x in range(1, width + 1):
        for y in range(1, height + 1):
            if (x, y) in void_cells:
                continue
            value = intensity + (rng.randint(-intensity_jitter, intensity_jitter)
                                 if intensity_jitter else 0)
            data_grid[(x, y)] = {"status": "Recorded", "intensity": value}

    shape_summary = ", ".join(f"{v['shape']}({v['size']})" for v in void_info) or "none"
    print(f"[+] Dataset generated: {len(data_grid)} recorded points, "
          f"{len(void_cells)} hidden void cells across {len(void_info)} voids "
          f"[{shape_summary}]" + (f" | seed={seed}" if seed is not None else ""))

    if return_ground_truth:
        return data_grid, void_cells
    return data_grid


if __name__ == "__main__":
    generate_dataset(seed=42)
