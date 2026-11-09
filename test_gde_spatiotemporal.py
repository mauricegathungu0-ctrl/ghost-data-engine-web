"""Dependency-free tests for gde_spatiotemporal.py (3D [time][y][x] reconstruction)."""
import copy
import json
import math
import random
import unittest

from gde_spatiotemporal import reconstruct_3d


def _ramp(T=3, H=7, W=7):
    return [[[10.0 * t for _ in range(W)] for _ in range(H)] for t in range(T)]


class SpatioTemporalTests(unittest.TestCase):
    def test_temporal_ramp_is_recovered_and_input_untouched(self):
        g = _ramp()
        for y in range(2, 5):
            for x in range(2, 5):
                g[1][y][x] = None
        before = copy.deepcopy(g)
        rep = reconstruct_3d(g)
        self.assertEqual(g, before)
        for y in range(2, 5):
            for x in range(2, 5):
                self.assertAlmostEqual(rep["grid"][1][y][x], 10.0, places=6)
        self.assertEqual(rep["summary"]["cells_unresolved"], 0)

    def test_empty_grid_is_unresolved_not_fabricated(self):
        rep = reconstruct_3d([[[None, None], [None, None]]] * 2)
        self.assertEqual(rep["summary"]["cells_reconstructed"], 0)
        self.assertEqual(rep["summary"]["cells_unresolved"], 8)
        self.assertEqual(rep["audit_trail"], {})
        self.assertTrue(all(v is None for f in rep["grid"] for r in f for v in r))
        self.assertTrue(all(u["reason"] == "no_valid_boundary"
                            for u in rep["unresolved"].values()))

    def test_audit_trail_contents_and_json(self):
        g = _ramp()
        g[1][3][3] = None
        rep = reconstruct_3d(g)
        row = rep["audit_trail"]["(1, 3, 3)"]
        self.assertEqual(row["wave"], 1)
        self.assertEqual(row["coordinate"], (1, 3, 3))
        self.assertIn((0, 3, 3), row["source_coordinates"])
        self.assertIn((2, 3, 3), row["source_coordinates"])
        self.assertEqual(row["temporal_sources"], 2)
        self.assertEqual(row["spatial_sources"], 4)
        self.assertTrue(0.0 <= row["confidence"] <= 1.0)
        json.dumps(rep)

    def test_values_stay_in_source_range_and_runs_are_deterministic(self):
        rng = random.Random(7)
        g = [[[rng.uniform(5, 50) for _ in range(10)] for _ in range(10)] for _ in range(5)]
        lo = min(v for f in g for r in f for v in r)
        hi = max(v for f in g for r in f for v in r)
        for _ in range(40):
            g[rng.randrange(5)][rng.randrange(10)][rng.randrange(10)] = None
        for conn in (4, 8):
            a = reconstruct_3d(g, spatial_connectivity=conn)
            b = reconstruct_3d(g, spatial_connectivity=conn)
            self.assertEqual(a["audit_trail"], b["audit_trail"])
            self.assertEqual(a["summary"]["cells_unresolved"], 0)
            for row in a["audit_trail"].values():
                self.assertTrue(lo - 1e-9 <= row["intensity"] <= hi + 1e-9)

    def test_temporal_decay_shifts_influence(self):
        g = [[[0.0] * 5 for _ in range(5)] for _ in range(3)]
        for t in (0, 2):
            g[t] = [[100.0] * 5 for _ in range(5)]
        g[1][2][2] = None
        low = reconstruct_3d(g, temporal_decay=0.01)["grid"][1][2][2]
        high = reconstruct_3d(g, temporal_decay=1.0)["grid"][1][2][2]
        self.assertLess(low, high)

    def test_max_waves_leaves_remaining_cells_unresolved(self):
        g = [[[1.0] * 9]]
        g[0][0][3:8] = [None] * 5
        rep = reconstruct_3d(g, max_waves=1)
        self.assertEqual(rep["summary"]["cells_reconstructed"], 2)
        self.assertEqual({u["reason"] for u in rep["unresolved"].values()},
                         {"max_waves_reached"})

    def test_missing_sentinels_and_nan(self):
        rep = reconstruct_3d([[[1, -999, 3]]], missing_values=(-999,))
        self.assertEqual(rep["grid"][0][0][1], 2.0)
        rep = reconstruct_3d([[[1.0, float("nan"), 3.0]]])
        self.assertEqual(rep["grid"][0][0][1], 2.0)

    def test_bridging_fully_missing_frames(self):
        g = [[[5.0] * 3 for _ in range(3)]]
        g += [[[None] * 3 for _ in range(3)] for _ in range(2)]
        g += [[[5.0] * 3 for _ in range(3)]]
        rep = reconstruct_3d(g)
        self.assertEqual(rep["summary"]["cells_reconstructed"], 18)
        self.assertEqual(rep["summary"]["cells_unresolved"], 0)

    def test_invalid_inputs_rejected(self):
        for bad in ([], [[]], [[[]]], [[[1, 2], [3]]], [[[1, "a"]]],
                    [[[float("inf")]]], [[[True]]]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                reconstruct_3d(bad)
        for kw in (dict(spatial_connectivity=6), dict(temporal_decay=0),
                   dict(temporal_decay=1.5), dict(inward_decay=0),
                   dict(temporal_reach=0), dict(max_waves=0)):
            with self.assertRaises(ValueError, msg=repr(kw)):
                reconstruct_3d([[[1.0]]], **kw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
