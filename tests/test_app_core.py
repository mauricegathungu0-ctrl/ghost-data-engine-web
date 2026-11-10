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


class LiveAccuracyTest(unittest.TestCase):
    def _truth(self):
        i, j = np.meshgrid(np.arange(20), np.arange(60), indexing="ij")
        return 20 + np.sin(j / 7.0) + 0.3 * np.cos(i / 3.0)

    def test_mask_is_reproducible_and_keeps_context(self):
        t = self._truth()
        a = core.make_hide_mask(t.shape, 0.05, 3, 5, 15, seed=4)
        b = core.make_hide_mask(t.shape, 0.05, 3, 5, 15, seed=4)
        self.assertTrue((a == b).all())
        self.assertFalse(a.all(axis=1).any())
        self.assertFalse(a.all(axis=0).any())

    def test_custom_block_is_hidden(self):
        t = self._truth()
        m = core.make_hide_mask(t.shape, 0, 0, seed=0, custom=(3, 4, 10, 20))
        self.assertTrue(m[2:4, 9:20].all())
        self.assertEqual(int(m.sum()), 2 * 11)

    def test_engine_never_reads_hidden_values(self):
        t = self._truth()
        m = core.make_hide_mask(t.shape, 0.05, 2, 5, 15, seed=1)
        t2 = t.copy()
        t2[m] += 1000.0                      # change only the hidden truth
        r1 = core.accuracy_test(t, m)["repaired"]
        r2 = core.accuracy_test(t2, m)["repaired"]
        self.assertTrue(np.allclose(r1[m], r2[m]))
        self.assertTrue(np.allclose(r1[~m], t[~m]))     # recorded cells untouched

    def test_scores_present_and_reasonable(self):
        t = self._truth()
        m = core.make_hide_mask(t.shape, 0.05, 2, 5, 15, seed=2)
        res = core.accuracy_test(t, m)
        names = [r["Method"] for r in res["scores"]]
        self.assertEqual(names[0], "Ghost Data Engine")
        self.assertEqual(res["n_hidden"], int(m.sum()))
        self.assertLess(res["scores"][0]["Mean abs. error"], 0.5)

    def test_rejects_incomplete_or_text(self):
        with self.assertRaises(ValueError):
            core.load_complete_matrix(b"1,2,3\n4,,6\n")
        with self.assertRaises(ValueError):
            core.load_complete_matrix(b"a,b\n1,2\n")




class LabelledGridTest(unittest.TestCase):
    def _file(self, blank=True):
        head = "Year," + ",".join(f"d{j}" for j in range(1, 21))
        rows = []
        for i in range(5):
            vals = [f"{20 + 0.1 * i + 0.05 * j:.2f}" for j in range(20)]
            if blank and i == 2:
                vals[5] = ""
                vals[6] = ""
            rows.append(f"{1990 + i}," + ",".join(vals))
        return ("\n".join([head] + rows) + "\n").encode()

    def test_detect_and_split(self):
        data = self._file()
        self.assertTrue(core.detect_labelled(data))
        corner, cols, labels, body = core.split_labelled(data)
        self.assertEqual(corner, "Year")
        self.assertEqual(labels, ["1990", "1991", "1992", "1993", "1994"])
        self.assertEqual(len(cols), 20)
        self.assertEqual(len(body.decode().strip().splitlines()), 5)

    def test_plain_numeric_and_normal_tables_not_labelled(self):
        self.assertFalse(core.detect_labelled(core.sample_csv_text().encode()))
        self.assertFalse(core.detect_labelled(core.sample_table_text().encode()))

    def test_labelled_repair_keeps_labels_and_fills(self):
        _c, cols, labels, body = core.split_labelled(self._file())
        grid, w, h = core.load_grid_from_bytes(body)
        self.assertEqual((w, h), (20, 5))
        res = core.repair_matrix(grid, w, h, blend=True)
        self.assertEqual(int(res["filled_mask"].sum()), 2)
        self.assertFalse(np.isnan(res["repaired"]).any())

    def test_complete_with_labels(self):
        m, rows, cols = core.load_complete_with_labels(self._file(blank=False))
        self.assertEqual(m.shape, (5, 20))
        self.assertEqual(rows[0], "1990")
        self.assertEqual(cols[0], "d1")


if __name__ == "__main__":
    unittest.main()
