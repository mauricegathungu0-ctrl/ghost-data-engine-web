"""Tests for benchmark_real_climate.py: loaders, cloud masks, scoring, and no truth leakage."""
import copy
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import benchmark_real_climate as brc

SHAPE = (5, 14, 14)


def _field(seed=0):
    return brc.synthetic_climate(*SHAPE, seed=seed)


def _args(**kw):
    base = ["--max-frames", str(SHAPE[0]), "--crop", str(SHAPE[1]), "--seeds", "1"]
    a = brc.parse_args(base)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class MaskTests(unittest.TestCase):
    def test_hides_exact_fraction_of_valid_cells_only(self):
        valid = np.ones(SHAPE, bool)
        valid[:, :3, :] = False                       # permanently invalid band
        for frac in (0.2, 0.3, 0.4):
            mask = brc.make_cloud_mask(valid, frac, np.random.default_rng(1))
            self.assertEqual(int(mask.sum()), round(frac * valid.sum()))
            self.assertFalse((mask & ~valid).any(), "an invalid cell was hidden")

    def test_clouds_are_clustered_not_salt_and_pepper(self):
        valid = np.ones((6, 30, 30), bool)
        mask = brc.make_cloud_mask(valid, 0.3, np.random.default_rng(2))
        # neighbours of hidden cells are hidden far more often than the 30% base rate
        same = (mask[:, 1:, :] & mask[:, :-1, :]).sum() / max(1, mask[:, :-1, :].sum())
        self.assertGreater(same, 0.7)

    def test_deterministic_per_seed(self):
        valid = np.ones(SHAPE, bool)
        a = brc.make_cloud_mask(valid, 0.3, np.random.default_rng(7))
        b = brc.make_cloud_mask(valid, 0.3, np.random.default_rng(7))
        self.assertTrue((a == b).all())

    def test_fraction_outside_20_40_rejected_by_default(self):
        with self.assertRaises(SystemExit):
            brc.run(_args(fractions=[0.6]))


class LoaderTests(unittest.TestCase):
    def test_npy_and_nodata(self):
        arr = _field()
        arr[0, 0, 0] = -9999.0
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.npy")
            np.save(p, arr)
            out, _ = brc.load_dataset(p, nodata=-9999.0)
        self.assertTrue(np.isnan(out[0, 0, 0]))
        self.assertEqual(out.shape, SHAPE)

    def test_netcdf3_roundtrip_with_fill_and_scale(self):
        from scipy.io import netcdf_file
        raw = np.arange(2 * 3 * 4, dtype="i2").reshape(2, 3, 4)
        raw[0, 0, 0] = -32767
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t2m.nc")
            f = netcdf_file(p, "w")
            f.createDimension("time", 2)
            f.createDimension("latitude", 3)
            f.createDimension("longitude", 4)
            v = f.createVariable("t2m", "i2", ("time", "latitude", "longitude"))
            v[:] = raw
            v._FillValue = np.int16(-32767)
            v.scale_factor = 0.5
            v.add_offset = 250.0
            f.close()
            try:
                out, desc = brc.load_dataset(p, var="t2m")
            except RuntimeError as e:   # only if no reader at all
                self.skipTest(str(e))
        self.assertEqual(out.shape, (2, 3, 4))
        self.assertTrue(np.isnan(out[0, 0, 0]))
        self.assertAlmostEqual(out[0, 0, 1], 250.5)

    def test_geotiff_stack_and_directory(self):
        try:
            import tifffile
        except ImportError:
            self.skipTest("tifffile not installed")
        arr = _field().astype("float32")
        with tempfile.TemporaryDirectory() as d:
            stack = os.path.join(d, "ndvi.tif")
            tifffile.imwrite(stack, arr)
            out, _ = brc.load_dataset(stack)
            self.assertEqual(out.shape, SHAPE)
            self.assertTrue(np.allclose(out, arr, atol=1e-4))
            folder = os.path.join(d, "bands")
            os.mkdir(folder)
            for i, frame in enumerate(arr):
                tifffile.imwrite(os.path.join(folder, f"b{i:02d}.tif"), frame)
            out2, _ = brc.load_dataset(folder)
            self.assertEqual(out2.shape, SHAPE)

    def test_unsupported_and_bad_variable(self):
        with self.assertRaises(ValueError):
            brc.load_dataset("data.csv")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.npy")
            np.save(p, np.zeros((4, 4)))        # 2D is promoted to one frame
            out, _ = brc.load_dataset(p)
            self.assertEqual(out.shape, (1, 4, 4))


class MethodTests(unittest.TestCase):
    def test_linear_temporal_exact_on_linear_series_and_declines_outside(self):
        t = np.arange(6, dtype=float)[:, None, None]
        arr = np.broadcast_to(2.0 * t + 1.0, (6, 2, 2)).copy()
        arr[2:4] = np.nan          # interior gap
        arr[5] = np.nan            # trailing gap: no observation after it
        pred = brc.method_linear_temporal(arr)
        self.assertTrue(np.allclose(pred[2:4], (2.0 * np.arange(2, 4) + 1.0)[:, None, None]))
        self.assertTrue(np.isnan(pred[5]).all())

    def test_gde_declines_when_nothing_is_recorded(self):
        pred = brc.method_gde(np.full((2, 3, 3), np.nan))
        self.assertTrue(np.isnan(pred).all())

    def test_scipy_linear_declines_outside_hull_nearest_does_not(self):
        arr = np.full((3, 5, 5), np.nan)
        arr[:, 1:4, 1:4] = 10.0
        lin = brc.method_scipy_linear3d(arr)
        near = brc.method_scipy_nearest3d(arr)
        self.assertTrue(np.isnan(lin[:, 0, :]).all())
        self.assertTrue(np.isfinite(near[np.isnan(arr)]).all())

    def test_ungrounded_distance(self):
        observed = np.zeros((1, 1, 12), bool)
        observed[0, 0, 0] = True
        hidden = ~observed
        ung = brc.ungrounded_cells(observed, hidden, 4.0)
        self.assertEqual(int(ung.sum()), 7)           # cells at x = 5..11
        self.assertFalse(ung[0, 0, 4])
        self.assertTrue(brc.ungrounded_cells(np.zeros_like(observed), hidden, 4.0)[hidden].all())


class NoLeakTests(unittest.TestCase):
    def _run(self, truth, valid, methods):
        return brc.run_once(truth, valid, 0.3, 11, methods, 4.0, 1.0)

    def test_methods_receive_nan_at_every_hidden_cell(self):
        truth = _field()
        valid = np.ones(SHAPE, bool)
        seen = {}

        def spy(masked):
            seen["m"] = masked.copy()
            return np.full(masked.shape, np.nan)

        self._run(truth, valid, {"spy": spy})
        hidden = brc.make_cloud_mask(valid, 0.3, np.random.default_rng(11))
        self.assertTrue(np.isnan(seen["m"][hidden]).all())
        self.assertTrue(np.array_equal(seen["m"][~hidden], truth[~hidden]))

    def test_predictions_do_not_depend_on_hidden_truth(self):
        truth = _field()
        valid = np.ones(SHAPE, bool)
        hidden = brc.make_cloud_mask(valid, 0.3, np.random.default_rng(11))
        poisoned = truth.copy()
        poisoned[hidden] = 987654.321
        captured = []
        methods = {"gde": brc.method_gde, "lin": brc.method_linear_temporal,
                   "sci": brc.method_scipy_linear3d}
        for tr in (truth, poisoned):
            row = {}
            for name, fn in methods.items():
                row[name] = fn(np.where(hidden, np.nan, tr))
            captured.append(row)
        for name in methods:
            a, b = captured[0][name], captured[1][name]
            self.assertTrue(np.array_equal(np.isnan(a), np.isnan(b)))
            self.assertTrue(np.allclose(a[np.isfinite(a)], b[np.isfinite(b)]))
            self.assertFalse((b[np.isfinite(b)] > 1e5).any())

    def test_gde_engine_input_has_none_at_hidden_cells(self):
        truth = _field()
        valid = np.ones(SHAPE, bool)
        captured = {}
        real = brc.reconstruct_3d

        def spy(grid, **kw):
            captured["grid"] = copy.deepcopy(grid)
            return real(grid, **kw)

        with mock.patch.object(brc, "reconstruct_3d", spy):
            self._run(truth, valid, {"gde": brc.method_gde})
        hidden = brc.make_cloud_mask(valid, 0.3, np.random.default_rng(11))
        flat = np.array([[[v is None for v in r] for r in f] for f in captured["grid"]])
        self.assertTrue(np.array_equal(flat, hidden))


class ScoringTests(unittest.TestCase):
    def test_constant_offset_method_scores_exactly(self):
        truth = _field()
        valid = np.ones(SHAPE, bool)
        off = lambda m: np.where(np.isnan(m), 0.0, m) * 0 + truth + 2.0   # noqa: E731 (oracle on purpose)
        run = brc.run_once(truth, valid, 0.3, 3, {"off": off}, 4.0, 1.0)
        agg = brc.aggregate([run], ["off"])["methods"]["off"]
        self.assertAlmostEqual(agg["mae"], 2.0)
        self.assertAlmostEqual(agg["rmse"], 2.0)
        self.assertEqual(agg["unresolved_pct"], 0.0)

    def test_declining_method_counts_unresolved_and_is_not_scored(self):
        truth = _field()
        valid = np.ones(SHAPE, bool)
        none = lambda m: np.full(m.shape, np.nan)   # noqa: E731
        run = brc.run_once(truth, valid, 0.3, 3, {"none": none}, 4.0, 1.0)
        agg = brc.aggregate([run], ["none"])["methods"]["none"]
        self.assertEqual(agg["unresolved_pct"], 100.0)
        self.assertIsNone(agg["mae"])

    def test_run_is_deterministic_and_reports_all_methods(self):
        args = _args(fractions=[0.2, 0.4])
        a, b = brc.run(args), brc.run(args)
        for r in (a, b):
            for run in r["runs"]:
                for m in run["methods"].values():
                    m["runtime_s"] = 0.0
        for r in (a["overall"], b["overall"]):
            for m in r["methods"].values():
                m["runtime_s"] = 0.0
        self.assertEqual(a["overall"], b["overall"])
        self.assertEqual(set(a["overall"]["methods"]),
                         {"gde_3d", "gde_3d_bounded", "linear_temporal",
                          "scipy_linear3d", "scipy_nearest3d"})
        self.assertTrue(a["dataset"]["synthetic"])
        self.assertIn("SYNTHETIC", brc.build_markdown(a))

    def test_require_real_refuses_synthetic(self):
        with self.assertRaises(SystemExit):
            brc.run(_args(require_real=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
