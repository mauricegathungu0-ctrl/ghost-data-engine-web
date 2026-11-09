"""Tests for app_core.py (the Streamlit-free helpers). Needs numpy + pandas."""
import unittest

import numpy as np

import app_core as core


class TableModeTests(unittest.TestCase):
    def setUp(self):
        self.data = core.sample_table_text().encode()
        self.df, self.numeric = core.read_table(self.data)

    def test_detects_header_and_numeric_columns(self):
        self.assertTrue(core.looks_like_table(self.data))
        self.assertFalse(core.looks_like_table(core.sample_csv_text().encode()))
        self.assertEqual(self.numeric, ["Latitude_Deg", "Longitude_Deg", "Solar_Radiation_Wm2"])

    def test_text_and_recorded_cells_never_change(self):
        res = core.repair_table(self.df, self.numeric)
        out, filled = res["repaired"], res["filled"]
        for c in ("Orbital_Regime", "Anomaly_Type"):
            self.assertTrue((out[c] == self.df[c]).all())
        for c in self.numeric:
            rec = ~self.df[c].isna()
            self.assertTrue((out.loc[rec, c] == self.df.loc[rec, c]).all())
            self.assertFalse(filled.loc[rec, c].any())
        self.assertFalse(out[self.numeric].isna().any().any())
        self.assertEqual(int(filled.values.sum()), res["summary"]["cells_reconstructed"])

    def test_columns_are_not_mixed(self):
        res = core.repair_table(self.df, self.numeric)
        lat = res["repaired"].loc[2, "Latitude_Deg"]
        self.assertTrue(52.5 <= lat <= 52.7)          # between its own neighbours, not pulled by radiation

    def test_all_missing_column_is_reported_not_invented(self):
        df = self.df.copy()
        df["Latitude_Deg"] = np.nan
        res = core.repair_table(df, self.numeric)
        self.assertIn("Latitude_Deg", res["skipped"])
        self.assertTrue(res["repaired"]["Latitude_Deg"].isna().all())

    def test_malformed_lines_are_skipped_and_counted_not_fatal(self):
        raw = b"name,val\na,1\nb,2,EXTRA,FIELDS\nc,\nd,4\n"
        df, numeric = core.read_table(raw)
        self.assertEqual(df.attrs["skipped_lines"], 1)
        self.assertEqual(list(df["name"]), ["a", "c", "d"])
        self.assertEqual(numeric, ["val"])
        clean, _ = core.read_table(b"name,val\na,1\n")
        self.assertEqual(clean.attrs["skipped_lines"], 0)

    def test_no_gaps_means_nothing_to_do(self):
        full = core.read_table(b"a,b\nx,1\ny,2\n")
        res = core.repair_table(*full)
        self.assertFalse(res["any_missing"])


class GridModeTests(unittest.TestCase):
    def test_bom_and_cp1252_uploads_load(self):
        raw = "﻿1,2,3\n4,,6\n7,8,9\n".encode("utf-8")
        grid, w, h = core.load_grid_from_bytes(raw)
        self.assertEqual((w, h, len(grid)), (3, 3, 8))
        grid, w, h = core.load_grid_from_bytes("1,2,3\n4,,6\n7,8,é\n".encode("cp1252"))
        self.assertEqual(w, 3)

    def test_sample_grid_repairs_without_changing_recorded_cells(self):
        grid, w, h = core.load_grid_from_bytes(core.sample_csv_text().encode())
        before = {k: v["intensity"] for k, v in grid.items()}
        res = core.repair_matrix(grid, w, h)
        self.assertEqual(res["summary"]["cells_stalled"], 0)
        self.assertFalse(np.isnan(res["repaired"]).any())
        for (x, y), v in before.items():
            self.assertEqual(res["repaired"][h - y, x - 1], v)


if __name__ == "__main__":
    unittest.main(verbosity=2)
