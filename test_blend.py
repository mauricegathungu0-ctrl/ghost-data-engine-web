"""Tests for the engine_v3 post-fill blends (knn_blend, row_blend). Dependency-free."""
import ast
import copy
import csv
import inspect
import os
import tempfile
import unittest

import engine_v3 as E

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(os.path.dirname(HERE), "v3")
if not os.path.exists(os.path.join(V3, "gui_app.py")):   # flat layout (e.g. the web-app repo)
    V3 = HERE


def _write(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _read(path):
    with open(path, newline="") as f:
        return [[float(v) for v in r] for r in csv.reader(f)]


def _ramp_file(d, missing):
    rows = [[float(10 * r + c) for c in range(12)] for r in range(6)]
    shown = [[("" if (r, c) in missing else v) for c, v in enumerate(row)] for r, row in enumerate(rows)]
    p = os.path.join(d, "in.csv")
    _write(p, shown)
    return p, rows


class BlendTests(unittest.TestCase):
    MISSING = {(2, 3), (2, 4), (2, 5), (4, 8), (0, 6), (5, 11)}

    def test_known_cells_never_change_for_any_setting(self):
        with tempfile.TemporaryDirectory() as d:
            src, truth = _ramp_file(d, self.MISSING)
            for kw in (dict(knn_blend=None, row_blend=None), dict(), dict(knn_blend=(3, 0.9), row_blend=1.0),
                       dict(knn_blend=(20, 1.0), row_blend=None), dict(knn_blend=None, row_blend=0.5)):
                out = os.path.join(d, "out.csv")
                E.process_and_fix_file(src, out, **kw)
                got = _read(out)
                for r in range(6):
                    for c in range(12):
                        if (r, c) not in self.MISSING:
                            self.assertEqual(got[r][c], truth[r][c], (kw, r, c))

    def test_both_off_matches_plain_engine(self):
        with tempfile.TemporaryDirectory() as d:
            src, _ = _ramp_file(d, self.MISSING)
            grid, w, h = E.load_csv_with_gaps(src)
            _, plain = E.repair_grid(grid, w, h)
            expect = {k: v["intensity"] for k, v in plain["audit_trail"].items()}
            out = os.path.join(d, "o.csv")
            E.process_and_fix_file(src, out, knn_blend=None, row_blend=None)
            got = _read(out)
            for key, val in expect.items():
                x, y = ast.literal_eval(key)
                self.assertAlmostEqual(got[h - y][x - 1], val, places=6)

    def test_row_blend_one_is_exact_row_linear(self):
        with tempfile.TemporaryDirectory() as d:
            src, truth = _ramp_file(d, {(2, 3), (2, 4), (2, 5)})
            out = os.path.join(d, "o.csv")
            E.process_and_fix_file(src, out, knn_blend=None, row_blend=1.0)
            got = _read(out)
            for c in (3, 4, 5):                      # ramp is linear along the row -> exact
                self.assertAlmostEqual(got[2][c], truth[2][c], places=5)

    def test_row_linear_extends_constant_beyond_row_ends_and_skips_empty_rows(self):
        tables = E._row_tables({(2, 1): {"intensity": 5.0}, (4, 1): {"intensity": 9.0}})
        self.assertEqual(E.row_linear_value(tables, (1, 1)), 5.0)
        self.assertEqual(E.row_linear_value(tables, (6, 1)), 9.0)
        self.assertAlmostEqual(E.row_linear_value(tables, (3, 1)), 7.0)
        self.assertIsNone(E.row_linear_value(tables, (3, 2)))

    def test_knn_on_constant_data_is_constant_and_uses_only_recorded_cells(self):
        grid = {(x, y): {"status": "Recorded", "intensity": 7.0} for x in range(1, 9) for y in range(1, 6)
                if (x, y) not in {(3, 3), (4, 3)}}
        self.assertAlmostEqual(E.knn_value(grid, (3, 3), 8, 5, 5), 7.0)
        # a value placed ONLY on a missing cell must be invisible to the blend
        poisoned = copy.deepcopy(grid)
        self.assertEqual(E.knn_value(grid, (3, 3), 8, 5, 5), E.knn_value(poisoned, (3, 3), 8, 5, 5))

    def test_invalid_blend_arguments_rejected(self):
        report = {"audit_trail": {}}
        for kw in (dict(knn_blend=(0, 0.5)), dict(knn_blend=(5, 1.5)), dict(row_blend=-0.1), dict(row_blend=2)):
            with self.assertRaises(ValueError, msg=repr(kw)):
                E.apply_blends({}, report, 3, 3, **kw)

    def test_blend_code_takes_no_truth_argument(self):
        for fn in (E.apply_blends, E.knn_value, E.row_linear_value, E.process_and_fix_file):
            for name in inspect.signature(fn).parameters:
                self.assertNotIn(name, ("truth", "answer_key", "ground_truth", "actual", "reference"))

    def test_defaults_are_on_and_can_be_switched_off(self):
        self.assertIsNotNone(E.DEFAULT_KNN_BLEND)
        with tempfile.TemporaryDirectory() as d:
            src, _ = _ramp_file(d, self.MISSING)
            res = E.process_and_fix_file(src, os.path.join(d, "a.csv"))
            self.assertEqual((res["knn_blend"], res["row_blend"]), (E.DEFAULT_KNN_BLEND, E.DEFAULT_ROW_BLEND))
            res = E.process_and_fix_file(src, os.path.join(d, "b.csv"), knn_blend=None, row_blend=None)
            self.assertEqual((res["knn_blend"], res["row_blend"]), (None, None))

    def test_main_and_gui_expose_the_option(self):
        with open(os.path.join(V3, "gui_app.py")) as f:
            gui_src = f.read()
        self.assertIn("blend_select", gui_src)
        self.assertIn('"knn_blend": None', gui_src)       # "Off" switches both blends off
        compile(gui_src, "gui_app.py", "exec")           # tkinter may be absent, so compile only
        import main
        parser = main.build_parser()
        self.assertFalse(parser.parse_args([]).no_blend)  # blend is ON unless --no-blend is given
        self.assertTrue(parser.parse_args(["--no-blend"]).no_blend)
        result = main.run_ghost_engine(width=12, height=9, seed=3, num_voids=2, max_void_size=3)
        self.assertTrue(result["blend"])
        plain = main.run_ghost_engine(width=12, height=9, seed=3, num_voids=2, max_void_size=3, blend=False)
        self.assertFalse(plain["blend"])
        self.assertEqual(set(result["report"]["audit_trail"]), set(plain["report"]["audit_trail"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
