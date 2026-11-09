# engine_v3.py
"""
Ghost Data Engine 3.3 - file pipeline with optional post-fill blends.

CSV in -> scan voids -> tension -> reconstruct (unchanged 3.2 engine) -> optional blends -> CSV out.

Post-fill blends (pure Python, no dependencies; they read ONLY recorded cells, so recorded values
can never change and nothing outside the input file is ever used):

  knn_blend=(k, weight)   each filled cell becomes (1-weight)*filled + weight*KNN, where KNN is the
                          1/distance-weighted mean of the k nearest recorded cells. Distance is
                          measured with the column (x, e.g. day) and the row (y, e.g. year)
                          coordinates each scaled to 0..1.
  row_blend=weight        after the KNN blend, each filled cell becomes
                          (1-weight)*value + weight*ROW_LINEAR, where ROW_LINEAR is linear
                          interpolation along the same row (constant beyond the first/last recorded
                          cell of that row). A row with no recorded cell is left as it was.

Set either to None to switch it off; both None gives exactly the 3.2 output.
"""
import ast
import bisect
import csv
import math
import os
import sys

# Ensure the script can import local engine files cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scanner import scan_for_void_clusters
from tension import calculate_boundary_tension
from reconstructor import reconstruct_void

# Defaults used when the caller does not choose (set from the tuning run; see RESULTS.md).
DEFAULT_KNN_BLEND = (5, 0.75)
DEFAULT_ROW_BLEND = 0.75


def load_csv_with_gaps(file_path, delimiter=",", null_values=None):
    """Reads a CSV matrix file and translates it into a GDE data_grid dictionary."""
    if null_values is None:
        null_values = {"", "nan", "NaN", "null", "NULL", "missing", "-999", "-999.0", "bad", "gap"}

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target data file not found: {file_path}")

    raw_matrix = []
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if row:
                raw_matrix.append(row)

    if not raw_matrix:
        raise ValueError("The provided file is empty.")

    height = len(raw_matrix)
    width = max(len(row) for row in raw_matrix)   # ragged rows: use the widest row

    data_grid = {}
    unparsed = 0

    for row_idx, row in enumerate(raw_matrix):
        y = height - row_idx
        for col_idx, raw_val in enumerate(row):
            x = col_idx + 1
            clean_val = raw_val.strip()

            if clean_val in null_values:
                continue

            try:
                intensity_value = float(clean_val)
                data_grid[(x, y)] = {"status": "Recorded", "intensity": intensity_value}
            except ValueError:
                unparsed += 1      # non-numeric text is treated as a gap (as before)
                continue

    if unparsed:
        print(f"[!] {unparsed} non-numeric cell(s) not in the null list were treated as missing. "
              f"If the file has a header row or text column, remove it first.")
    print(f"[+] Loaded file matrix: {width}x{height} pixels. Detected {len(data_grid)} recorded points.")
    return data_grid, width, height


def export_reconstructed_csv(output_path, original_grid, report, width, height):
    """Recompiles the fixed GDE grid back into a clean standard CSV file."""
    combined_values = {}
    for coord, info in original_grid.items():
        combined_values[coord] = info["intensity"]

    for coord_str, info in report["audit_trail"].items():
        coord_tuple = ast.literal_eval(coord_str)
        combined_values[coord_tuple] = info["intensity"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row_idx in range(height):
            y = height - row_idx
            row_data = []
            for col_idx in range(width):
                x = col_idx + 1
                val = combined_values.get((x, y), "NaN")
                row_data.append(val)
            writer.writerow(row_data)

    print(f"[+] Clean, repaired dataset successfully written to: {output_path}")


# ---------------------------------------------------------------------------
# Post-fill blends
# ---------------------------------------------------------------------------
_OFFSET_CACHE = {}


def _sorted_offsets(width, height):
    """All (dx, dy) offsets ordered by scaled distance (ties: dy, then dx). Cached per grid size."""
    key = (width, height)
    if key not in _OFFSET_CACHE:
        sx, sy = max(1, width - 1), max(1, height - 1)
        offs = [(math.hypot(dx / sx, dy / sy), dy, dx)
                for dy in range(-(height - 1), height) for dx in range(-(width - 1), width)
                if (dx, dy) != (0, 0)]
        offs.sort()
        _OFFSET_CACHE[key] = offs
    return _OFFSET_CACHE[key]


def knn_value(data_grid, cell, width, height, k):
    """1/distance-weighted mean of the k nearest RECORDED cells to `cell` (None if no data)."""
    x, y = cell
    num = den = 0.0
    found = 0
    for dist, dy, dx in _sorted_offsets(width, height):
        c = (x + dx, y + dy)
        info = data_grid.get(c)
        if info is None:
            continue
        w = 1.0 / dist
        num += w * info["intensity"]
        den += w
        found += 1
        if found >= k:
            break
    return num / den if den else None


def _row_tables(data_grid):
    rows = {}
    for (x, y), info in data_grid.items():
        rows.setdefault(y, []).append((x, info["intensity"]))
    tables = {}
    for y, items in rows.items():
        items.sort()
        tables[y] = ([i[0] for i in items], [i[1] for i in items])
    return tables


def row_linear_value(tables, cell):
    """Linear interpolation along the cell's row from recorded cells only (None if row has none)."""
    x, y = cell
    t = tables.get(y)
    if not t:
        return None
    xs, vs = t
    i = bisect.bisect_left(xs, x)
    if i == 0:
        return vs[0]
    if i >= len(xs):
        return vs[-1]
    x0, x1, v0, v1 = xs[i - 1], xs[i], vs[i - 1], vs[i]
    return v0 + (v1 - v0) * (x - x0) / (x1 - x0)


def apply_blends(data_grid, report, width, height, knn_blend=None, row_blend=None):
    """Blend the filled cells of `report` in place using ONLY recorded cells of `data_grid`.

    Returns the number of cells changed. Recorded cells are never touched (they are not in the
    audit trail). knn_blend=(k, weight) is applied first, then row_blend=weight.
    """
    if knn_blend is not None:
        k, wt = knn_blend
        if not (isinstance(k, int) and not isinstance(k, bool) and k >= 1) or not 0 <= wt <= 1:
            raise ValueError("knn_blend must be (k >= 1, 0 <= weight <= 1)")
    if row_blend is not None and not 0 <= row_blend <= 1:
        raise ValueError("row_blend must be a weight between 0 and 1")
    if knn_blend is None and row_blend is None:
        return 0
    tables = _row_tables(data_grid) if row_blend is not None else None
    changed = 0
    for key, row in report["audit_trail"].items():
        cell = ast.literal_eval(key)
        value = row["intensity"]
        if knn_blend is not None:
            kv = knn_value(data_grid, cell, width, height, knn_blend[0])
            if kv is not None:
                value = (1.0 - knn_blend[1]) * value + knn_blend[1] * kv
        if row_blend is not None:
            rv = row_linear_value(tables, cell)
            if rv is not None:
                value = (1.0 - row_blend) * value + row_blend * rv
        row["intensity"] = round(value, 6)
        changed += 1
    return changed


def repair_grid(data_grid, width, height, connectivity=4, inward_decay=1.0, void_level=0.0,
                mode="accurate", relax=True):
    """Scan + tension + reconstruct. Returns (voids, report); the report is the raw engine fill."""
    voids = scan_for_void_clusters(data_grid, connectivity=connectivity, bounds=(1, width, 1, height))
    if not voids:
        return voids, None
    tension_data = calculate_boundary_tension(voids, data_grid, void_level=void_level)
    report = reconstruct_void(
        voids, tension_data, data_grid,
        pressure_threshold=0.0,
        inward_decay=inward_decay,
        void_level=void_level,
        mode=mode,
        relax=relax,
    )
    return voids, report


_KEEP = object()


def process_and_fix_file(input_file, output_file, connectivity=4, inward_decay=1.0, void_level=0.0,
                         mode="accurate", relax=True, knn_blend=_KEEP, row_blend=_KEEP):
    """Executes the complete end-to-end load -> scan -> repair -> blend -> export pipeline on a user file.

    knn_blend / row_blend: see the module docstring. Left unspecified they take the module defaults
    (DEFAULT_KNN_BLEND / DEFAULT_ROW_BLEND); pass None to switch a blend off. Passing None for both
    reproduces the plain 3.2 engine output exactly.

    Returns a dict: {"written": bool, "output_file": path or None, "reconstructed": int,
    "stalled": int, "message": str}. (Earlier versions returned None.)
    """
    if knn_blend is _KEEP:
        knn_blend = DEFAULT_KNN_BLEND
    if row_blend is _KEEP:
        row_blend = DEFAULT_ROW_BLEND
    print(f"\n=== PROCESSING USER DATA DATASET: {os.path.basename(input_file)} ===")

    data_grid, w, h = load_csv_with_gaps(input_file)
    voids, report = repair_grid(data_grid, w, h, connectivity, inward_decay, void_level, mode, relax)

    if not voids:
        print("[!] No missing cells or voids detected in this file! Exiting without changes.")
        return {"written": False, "output_file": None, "reconstructed": 0, "stalled": 0,
                "message": "No missing cells were found, so no output file was written."}

    apply_blends(data_grid, report, w, h, knn_blend=knn_blend, row_blend=row_blend)
    export_reconstructed_csv(output_file, data_grid, report, w, h)
    print(f"    Summary: Reconstructed {report['summary']['cells_reconstructed']} cells. Stalled: {report['summary']['cells_stalled']} cells.")
    return {"written": True, "output_file": output_file,
            "reconstructed": report["summary"]["cells_reconstructed"],
            "stalled": report["summary"]["cells_stalled"],
            "knn_blend": knn_blend, "row_blend": row_blend,
            "message": f"Reconstructed {report['summary']['cells_reconstructed']} cells "
                       f"({report['summary']['cells_stalled']} could not be filled)."}


if __name__ == "__main__":
    # Built-in automatic test
    dummy_input = "sample_corrupted_data.csv"
    dummy_output = "sample_fixed_data.csv"

    with open(dummy_input, "w") as f:
        f.write("12.5,13.1,missing,14.2\n")
        f.write("11.9,bad,12.8,13.0\n")
        f.write("10.1,10.5,11.2,NaN\n")

    process_and_fix_file(dummy_input, dummy_output, inward_decay=1.0)
