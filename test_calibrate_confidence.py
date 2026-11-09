"""Tests for calibrate_confidence.py, chiefly: the engine never sees hidden ground truth."""
import copy
import inspect
import random
import unittest
from unittest import mock

import calibrate_confidence as cc
import gde_spatiotemporal as gst

SEED, SIZE, FRAMES = 5, 10, 4
SENTINEL = 987654.321


def _rebuild(seed=SEED):
    """Recreate the truth and mask evaluate_seed() builds for a seed (same RNG order)."""
    rng = random.Random(seed)
    truth = cc.make_truth(rng, FRAMES, SIZE, SIZE, 1.5)
    mask = cc.make_cloud_mask(rng, FRAMES, SIZE, SIZE, 3, 0.5)
    return truth, mask


def _spied_run(seed=SEED):
    captured = {}
    real = cc.reconstruct_3d

    def spy(grid, **kw):
        captured["grid"] = copy.deepcopy(grid)
        captured["report"] = real(grid, **kw)
        return captured["report"]

    with mock.patch.object(cc, "reconstruct_3d", spy):
        result = cc.evaluate_seed(seed, size=SIZE, frames=FRAMES)
    return result, captured


class NoGroundTruthLeakTests(unittest.TestCase):
    def test_engine_input_has_none_at_every_hidden_cell(self):
        _, cap = _spied_run()
        truth, mask = _rebuild()
        hidden = 0
        for t in range(FRAMES):
            for y in range(SIZE):
                for x in range(SIZE):
                    if mask[t][y][x]:
                        hidden += 1
                        self.assertIsNone(cap["grid"][t][y][x], (t, y, x))
                    else:
                        self.assertEqual(cap["grid"][t][y][x], truth[t][y][x])
        self.assertGreater(hidden, 0, "test is vacuous: nothing was hidden")

    def test_poisoned_hidden_truth_never_reaches_engine_or_its_output(self):
        truth, mask = _rebuild()
        poisoned = copy.deepcopy(truth)
        n = 0
        for t in range(FRAMES):
            for y in range(SIZE):
                for x in range(SIZE):
                    if mask[t][y][x]:
                        poisoned[t][y][x] = SENTINEL
                        n += 1
        self.assertGreater(n, 0)
        masked = cc.mask_grid(poisoned, mask)
        flat_in = [v for f in masked for r in f for v in r]
        self.assertNotIn(SENTINEL, flat_in)
        rep = gst.reconstruct_3d(masked)
        flat_out = [v for f in rep["grid"] for r in f for v in r]
        self.assertNotIn(SENTINEL, flat_out)
        for row in rep["audit_trail"].values():
            self.assertLess(row["intensity"], 1e4)

    def test_predictions_identical_whatever_the_hidden_truth_is(self):
        truth_a, mask = _rebuild()
        rng = random.Random(99)
        truth_b = copy.deepcopy(truth_a)
        for t in range(FRAMES):
            for y in range(SIZE):
                for x in range(SIZE):
                    if mask[t][y][x]:
                        truth_b[t][y][x] = rng.uniform(-5000, 5000)
        rep_a = gst.reconstruct_3d(cc.mask_grid(truth_a, mask))
        rep_b = gst.reconstruct_3d(cc.mask_grid(truth_b, mask))
        self.assertEqual(rep_a["audit_trail"], rep_b["audit_trail"])
        self.assertEqual(rep_a["grid"], rep_b["grid"])
        # ...while the scored error DOES depend on truth, so truth is used only to score.
        err = lambda truth, rep: sum(abs(r["intensity"] - truth[r["coordinate"][0]][r["coordinate"][1]][r["coordinate"][2]])
                                     for r in rep["audit_trail"].values())
        self.assertNotAlmostEqual(err(truth_a, rep_a), err(truth_b, rep_b))

    def test_engine_signature_accepts_no_truth_or_mask(self):
        params = set(inspect.signature(gst.reconstruct_3d).parameters)
        for forbidden in ("truth", "mask", "ground_truth", "actual", "reference"):
            self.assertNotIn(forbidden, params)


class ScoringTests(unittest.TestCase):
    def test_error_is_abs_predicted_minus_actual(self):
        result, cap = _spied_run()
        truth, _ = _rebuild()
        expected = sorted(
            (r["confidence"], abs(r["intensity"] - truth[r["coordinate"][0]][r["coordinate"][1]][r["coordinate"][2]]), r["wave"])
            for r in cap["report"]["audit_trail"].values())
        self.assertEqual(sorted(result["cells"]), expected)
        self.assertGreater(len(result["cells"]), 0)

    def test_unresolved_cells_are_counted_not_scored(self):
        with mock.patch.object(cc, "make_cloud_mask",
                               lambda rng, f, h, w, *a, **k:
                               [[[True] * w for _ in range(h)] for _ in range(f)]):
            res = cc.evaluate_seed(1, size=6, frames=2)
        self.assertEqual(res["cells"], [])
        self.assertEqual(res["unresolved"], res["missing"])
        self.assertEqual(res["missing"], 6 * 6 * 2)

    def test_runs_are_deterministic_and_bucket_counts_add_up(self):
        a = cc.run(seeds=3, size=SIZE, frames=FRAMES)
        b = cc.run(seeds=3, size=SIZE, frames=FRAMES)
        self.assertEqual(a, b)
        self.assertEqual(sum(bk["cells"] for bk in a["buckets"]), a["reconstructed_cells"])
        self.assertEqual(a["reconstructed_cells"] + a["unresolved_cells"], a["missing_cells"])
        for bk in a["buckets"]:
            if bk["cells"]:
                self.assertLessEqual(bk["median_error"], bk["p95_error"] + 1e-12)
                self.assertLessEqual(bk["p95_error"], bk["max_error"] + 1e-12)


class StatsTests(unittest.TestCase):
    def test_percentile(self):
        self.assertAlmostEqual(cc.percentile(list(range(1, 102)), 95), 96.0)
        self.assertEqual(cc.percentile([4.0], 95), 4.0)
        self.assertIsNone(cc.percentile([], 95))
        self.assertAlmostEqual(cc.percentile([0, 10], 50), 5.0)

    def test_spearman(self):
        self.assertAlmostEqual(cc.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertAlmostEqual(cc.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertIsNone(cc.spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(cc.spearman([1, 2], [1, 2]))

    def test_bucket_edges(self):
        cells = [(0.9, 1.0, 1), (1.0, 2.0, 1), (0.8999, 3.0, 1), (0.0, 4.0, 1), (0.5, 5.0, 1)]
        by_low = {b["low"]: b["cells"] for b in cc.bucketize(cells)}
        self.assertEqual(by_low[0.9], 2)      # 0.9 and 1.0 (last bucket is closed)
        self.assertEqual(by_low[0.8], 1)
        self.assertEqual(by_low[0.5], 1)
        self.assertEqual(by_low[0.0], 1)
        self.assertEqual(sum(by_low.values()), len(cells))


if __name__ == "__main__":
    unittest.main(verbosity=2)
