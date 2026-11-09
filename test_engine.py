"""Dependency-free regression and stress tests for Ghost Data Engine 3.0."""
import math
import random
import unittest

from generator import generate_dataset
from scanner import scan_for_void_clusters
from tension import calculate_boundary_tension
from reconstructor import reconstruct_void
from spatiotemporal import reconstruct_spatiotemporal
from benchmark import run_benchmark


class GhostEngineTests(unittest.TestCase):
    def test_seeded_pipeline_reconstructs_every_generated_void(self):
        for seed in range(50):
            grid, truth = generate_dataset(width=24, height=19, seed=seed,
                                           num_voids=5, max_void_size=6,
                                           intensity=10, intensity_jitter=3,
                                           return_ground_truth=True)
            voids = scan_for_void_clusters(grid, bounds=(1, 24, 1, 19))
            found = {c for v in voids for c in v["cells"]}
            self.assertEqual(found, truth, f"scanner mismatch at seed {seed}")
            tension = calculate_boundary_tension(voids, grid)
            report = reconstruct_void(voids, tension, grid, pressure_threshold=1e9)
            self.assertEqual(report["summary"]["cells_reconstructed"], len(truth))
            self.assertEqual(report["summary"]["cells_stalled"], 0)
            for row in report["audit_trail"].values():
                self.assertTrue(math.isfinite(row["intensity"]))

    def test_empty_grid_with_bounds_is_detected(self):
        voids = scan_for_void_clusters({}, bounds=(1, 3, 1, 2))
        self.assertEqual(sum(v["size"] for v in voids), 6)

    def test_empty_grid_without_bounds_is_safe(self):
        self.assertEqual(scan_for_void_clusters({}), [])

    def test_invalid_connectivity_rejected(self):
        with self.assertRaises(ValueError):
            scan_for_void_clusters({(1, 1): {"intensity": 1}}, connectivity=6)

    def test_invalid_bounds_rejected(self):
        with self.assertRaises(ValueError):
            scan_for_void_clusters({(1, 1): {"intensity": 1}}, bounds=(3, 1, 1, 2))

    def test_no_source_is_reported_not_fabricated(self):
        voids = [{"id": 1, "size": 1, "cells": [(1, 1)]}]
        report = reconstruct_void(voids, {}, {}, pressure_threshold=0)
        self.assertEqual(report["summary"]["cells_stalled"], 1)
        self.assertEqual(report["audit_trail"], {})

    def test_non_finite_input_rejected(self):
        voids = [{"id": 1, "size": 1, "cells": [(2, 1)]}]
        with self.assertRaises(ValueError):
            reconstruct_void(voids, {}, {(1, 1): {"intensity": float("nan")}})

    def test_generator_rejects_bad_dimensions(self):
        with self.assertRaises(ValueError):
            generate_dataset(width=0)

    def test_spatiotemporal_reconstruction(self):
        frames = [
            {(1, 1): {"intensity": 8.0}, (2, 1): {"intensity": 8.0}},
            {(1, 1): {"intensity": 8.0}},
        ]
        result = reconstruct_spatiotemporal(frames, bounds=(1, 2, 1, 1))
        self.assertIn((2, 1, 1), result["reconstructed_voxels"])
        self.assertEqual(result["unresolved_voxels"], [])

    def test_benchmark_has_full_coverage(self):
        result = run_benchmark(width=15, height=15, missing_fraction=0.25, seed=3)
        for method in result["methods"].values():
            self.assertEqual(method["coverage"], 1.0)

    def test_deterministic_reconstruction(self):
        grid, _ = generate_dataset(seed=2026, return_ground_truth=True)
        voids = scan_for_void_clusters(grid, bounds=(1, 20, 1, 20))
        tension = calculate_boundary_tension(voids, grid)
        a = reconstruct_void(voids, tension, grid)
        b = reconstruct_void(voids, tension, grid)
        self.assertEqual(a["audit_trail"], b["audit_trail"])


    # ---- 3.1 additions -------------------------------------------------
    def test_legacy_mode_reproduces_3_0_sample(self):
        # Values the shipped 3.0 sample_fixed_data.csv was produced with.
        grid = {(1, 3): 12.5, (2, 3): 13.1, (4, 3): 14.2,
                (1, 2): 11.9, (3, 2): 12.8, (4, 2): 13.0,
                (1, 1): 10.1, (2, 1): 10.5, (3, 1): 11.2}
        grid = {k: {"status": "Recorded", "intensity": v} for k, v in grid.items()}
        voids = scan_for_void_clusters(grid, bounds=(1, 4, 1, 3))
        tension = calculate_boundary_tension(voids, grid)
        rep = reconstruct_void(voids, tension, grid, pressure_threshold=0.0,
                               inward_decay=1.0, mode="legacy")
        got = {k: v["intensity"] for k, v in rep["audit_trail"].items()}
        self.assertAlmostEqual(got["(3, 3)"], 13.38067, places=4)
        self.assertAlmostEqual(got["(2, 2)"], 12.116582, places=4)
        self.assertAlmostEqual(got["(4, 1)"], 12.132927, places=4)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            reconstruct_void([], {}, {}, mode="turbo")

    def test_accurate_mode_recovers_a_plane_and_keeps_audit(self):
        # A linear ramp must be reproduced almost exactly inside a hole.
        w = h = 14
        grid = {(x, y): {"status": "Recorded", "intensity": 3.0 * x + 2.0 * y + 5}
                for x in range(1, w + 1) for y in range(1, h + 1)
                if not (5 <= x <= 9 and 5 <= y <= 9)}
        voids = scan_for_void_clusters(grid, bounds=(1, w, 1, h))
        tension = calculate_boundary_tension(voids, grid)
        rep = reconstruct_void(voids, tension, grid)
        self.assertTrue(rep["summary"]["refinement"]["applied"])
        for key, row in rep["audit_trail"].items():
            x, y = eval(key)
            self.assertAlmostEqual(row["intensity"], 3.0 * x + 2.0 * y + 5, delta=0.5)
            self.assertIn("wave_intensity", row)
            self.assertTrue(row["source_coordinates"])
        self.assertEqual(rep["summary"]["cells_stalled"], 0)

    def test_accurate_never_leaves_recorded_range(self):
        for seed in range(10):
            grid, _ = generate_dataset(width=20, height=20, seed=seed, num_voids=4,
                                       max_void_size=5, intensity=100,
                                       intensity_jitter=20, return_ground_truth=True)
            voids = scan_for_void_clusters(grid, bounds=(1, 20, 1, 20))
            rep = reconstruct_void(voids, calculate_boundary_tension(voids, grid), grid)
            lo = min(v["intensity"] for v in grid.values())
            hi = max(v["intensity"] for v in grid.values())
            for row in rep["audit_trail"].values():
                self.assertTrue(lo - 1e-9 <= row["intensity"] <= hi + 1e-9)

    def test_no_decay_bias_toward_zero_in_accurate_mode(self):
        grid = {(x, y): {"status": "Recorded", "intensity": 200.0}
                for x in range(1, 13) for y in range(1, 13)
                if not (4 <= x <= 9 and 4 <= y <= 9)}
        voids = scan_for_void_clusters(grid, bounds=(1, 12, 1, 12))
        rep = reconstruct_void(voids, calculate_boundary_tension(voids, grid), grid)
        for row in rep["audit_trail"].values():
            self.assertAlmostEqual(row["intensity"], 200.0, delta=1e-3)

    def test_wrapper_roundtrip_and_status(self):
        import os, tempfile
        from data_wrapper import process_and_fix_file
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in.csv"), os.path.join(d, "out.csv")
            with open(src, "w") as f:
                f.write("1,2,3\n4,,6\n7,8,9\n")
            res = process_and_fix_file(src, dst)
            self.assertTrue(res["written"])
            self.assertTrue(os.path.exists(dst))
            with open(src, "w") as f:
                f.write("1,2,3\n4,5,6\n")
            res = process_and_fix_file(src, os.path.join(d, "none.csv"))
            self.assertFalse(res["written"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
