"""Smoke test for real_benchmark.py. Skipped when numpy/scipy/scikit-image are absent."""
import unittest

try:
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import skimage  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy, scipy and scikit-image are required")
class RealBenchmarkTests(unittest.TestCase):
    def test_all_methods_run_with_full_coverage_and_finite_error(self):
        import math
        import real_benchmark as rb
        rows = rb.run(size=32, seeds=2, num_voids=3, void_size=5, images=["moon"])
        self.assertEqual({r["method"] for r in rows}, set(rb.METHODS))
        for r in rows:
            self.assertEqual(r["coverage"], 1.0, r["method"])
            self.assertTrue(math.isfinite(r["mae"]), r["method"])

    def test_methods_never_see_hidden_pixels(self):
        # Changing the hidden pixels must not change any prediction.
        import numpy as np
        import real_benchmark as rb
        img = rb.load_image("camera", 32)
        mask = rb.carve_blob_mask(32, 3, 5, seed=1)
        altered = img.copy()
        altered[mask] = 255.0 - altered[mask]
        for name, fn in rb.METHODS.items():
            a = fn(np.where(mask, np.nan, img), mask)
            b = fn(np.where(mask, np.nan, altered), mask)
            np.testing.assert_allclose(a, b, err_msg=name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
