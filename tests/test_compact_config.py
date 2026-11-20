"""Compact manual inputs and legacy compatibility; no measured data loaded."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from rsoxs_unified_workup import config
from rsoxs_unified_workup.outputs import build_manifest
from rsoxs_unified_workup import anisotropy
from test_multi_polarization import product_fixture


def scan_set(pols=(0, 90, 45, 135), layout="separate_runs"):
    row = {"scan_set_id": "synthetic", "sample_id": "sample", "scan_type": "long",
           "energy_region": "C", "polarization_layout": layout}
    for i, (pol, column) in enumerate(config.POLARIZATION_SCAN_COLUMNS.items()):
        row[column] = (42 if layout == "within_run" else 42 + i) if pol in pols else None
    return row


def write_csv(root, name, rows):
    with (root / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inputs(root, row):
    write_csv(root, "samples.csv", [{"sample_id": "sample", "sample_name": "Example", "notes": "N/A"}])
    write_csv(root, "scan_sets.csv", [row])
    (root / "processing").mkdir()
    (root / "processing/synthetic.toml").write_text("[anisotropy]\nchi_halfwidth_degrees = 45\n")


class GenerationTests(unittest.TestCase):
    def test_standard_two_four_and_shared_runs(self):
        for pols in ((0, 90), (45, 135), (0, 90, 45, 135)):
            for layout in ("separate_runs", "within_run"):
                with self.subTest(pols=pols, layout=layout):
                    row = scan_set(pols, layout)
                    before = copy.deepcopy(row)
                    records = config.generate_scan_set_records(row)
                    self.assertEqual(len(records["scans"]), len(pols))
                    self.assertEqual(len(records["polarization_pairs"]), len(pols) // 2)
                    self.assertEqual(len(records["anisotropy_sectors"]), len(pols) * 2)
                    self.assertEqual(row, before)
                    self.assertEqual(set(config.get_mask_pairs({"scan_set": row, **records}).values()),
                                     {(first, second) for first, second in ((0, 90), (45, 135)) if first in pols})

    def test_legacy_sector_ids_and_labels(self):
        records = config.generate_scan_set_records(scan_set())
        sectors = records["anisotropy_sectors"]
        expected = [("12oclock", 90), ("9oclock", 180), ("6oclock", -90), ("3oclock", 0),
                    ("10p30oclock", 135), ("7p30oclock", -135), ("4p30oclock", -45), ("1p30oclock", 45)]
        self.assertEqual([(s["label"], s["chi_center_deg"]) for s in sectors], expected)
        self.assertEqual([s["sector_id"] for s in sectors], ["synthetic_" + label for label, _ in expected])
        self.assertTrue(all(s["chi_halfwidth_deg"] == 45 for s in sectors))

    def test_missing_tokens_and_integer_strings(self):
        for missing in (None, np.nan, "", " ", "N/A", "n/a", "None", "none", "NA", "nan"):
            with self.subTest(missing=missing):
                row = scan_set((0, 90))
                row.update(pol0_scan_id=" 42 ", pol90_scan_id="43.0", pol45_scan_id=missing, pol135_scan_id=missing)
                self.assertEqual(len(config.generate_scan_set_records(row)["scans"]), 2)

    def test_incomplete_pairs_and_no_runs(self):
        for pols in ((0,), (90,), (45,), (135,), (0, 90, 45)):
            with self.subTest(pols=pols), self.assertRaisesRegex(ValueError, "incomplete"):
                config.generate_scan_set_records(scan_set(pols))
        with self.assertRaisesRegex(ValueError, "no polarization run IDs"):
            config.generate_scan_set_records(scan_set(()))

    def test_invalid_ids_and_layout(self):
        for value in (0, -1, 42.5, True, False, np.inf, "bad", "True"):
            row = scan_set()
            row["pol0_scan_id"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "pol0_scan_id"):
                config.generate_scan_set_records(row)
        row = scan_set()
        row["pol90_scan_id"] = row["pol0_scan_id"]
        with self.assertRaisesRegex(ValueError, "within_run"):
            config.generate_scan_set_records(row)
        row = scan_set(layout="within_run")
        row["pol45_scan_id"] = row["pol135_scan_id"] = 99
        self.assertEqual(len(config.generate_scan_set_records(row)["scans"]), 4)

    def test_sector_halfwidth_and_independent_settings(self):
        settings = {"anisotropy": {"chi_halfwidth_degrees": 30},
                    "beam_center_fit": {"chi_half_width_degrees": 40},
                    "normalization": {"half_ring_halfwidth_degrees": 90}}
        result = config.generate_scan_set_records(scan_set(), settings)
        self.assertTrue(all(s["chi_halfwidth_deg"] == 30 for s in result["anisotropy_sectors"]))
        for value in (0, -1, 45.1, 90, True, np.nan, np.inf, "bad"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "chi_halfwidth_degrees"):
                config.generate_scan_set_records(scan_set(), {"anisotropy": {"chi_halfwidth_degrees": value}})


class LoadingTests(unittest.TestCase):
    def test_generated_records_preserve_pointwise_scientific_results(self):
        for pols in ((0, 90), (45, 135), (0, 90, 45, 135)):
            for layout in ("separate_runs", "within_run"):
                with self.subTest(pols=pols, layout=layout):
                    products = product_fixture(pols, layout)
                    before = products["configuration"]
                    row = scan_set(pols, layout)
                    for column in config.POLARIZATION_SCAN_COLUMNS.values():
                        row[column] = None
                    for record in before["scans"]:
                        row[config.POLARIZATION_SCAN_COLUMNS[record["polarization_deg"]]] = record["scan_id"]
                    after = {**before, "scan_set": row,
                             **config.generate_scan_set_records(row, before["processing"])}
                    pointwise = anisotropy.build_pointwise_anisotropy_maps(after, products["scans"])
                    for mode, pairs in pointwise.items():
                        for pair_id, result in pairs.items():
                            for field in ("parallel", "perpendicular", "difference", "anisotropy"):
                                xr.testing.assert_equal(
                                    result[field], products["pointwise_anisotropy_results"][mode][pair_id][field])

    def test_only_two_csvs_needed_and_legacy_files_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs(root, scan_set())
            self.assertEqual(set(config.read_config_tables(root)), {"samples", "scan_sets"})
            first = config.load_scan_set("synthetic", root)
            # Even malformed legacy files must not be consulted in compact mode.
            for name in ("scans.csv", "polarization_pairs.csv", "anisotropy_sectors.csv"):
                (root / name).write_text('malformed,"unterminated\n')
            second = config.load_scan_set("synthetic", root)
            self.assertEqual(first["scans"], second["scans"])
            self.assertEqual(first["polarization_pairs"], second["polarization_pairs"])
            self.assertEqual(first["anisotropy_sectors"], second["anisotropy_sectors"])

    def test_missing_values_survive_csv_reading(self):
        for token in ("", "N/A", "None"):
            with self.subTest(token=token), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                row = scan_set((0, 90), "within_run")
                row["pol45_scan_id"] = row["pol135_scan_id"] = token
                inputs(root, row)
                self.assertEqual(len(config.load_scan_set("synthetic", root)["scans"]), 2)

    def test_partial_headers_and_duplicate_scan_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = scan_set()
            del row["pol135_scan_id"]
            inputs(root, row)
            with self.assertRaisesRegex(ValueError, "all four"):
                config.read_config_tables(root)
            write_csv(root, "scan_sets.csv", [scan_set(), scan_set()])
            with self.assertRaisesRegex(ValueError, "exactly one"):
                config.load_scan_set("synthetic", root)

    def test_missing_sample_and_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs(root, scan_set())
            (root / "processing/synthetic.toml").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "processing"):
                config.load_scan_set("synthetic", root)
            row = scan_set()
            row["sample_id"] = "absent"
            write_csv(root, "scan_sets.csv", [row])
            with self.assertRaisesRegex(ValueError, "sample"):
                config.load_scan_set("synthetic", root)

    def test_legacy_files_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = scan_set((0, 90))
            generated = config.generate_scan_set_records(row)
            legacy_row = {key: value for key, value in row.items()
                          if key not in config.POLARIZATION_SCAN_COLUMNS.values()}
            inputs(root, legacy_row)
            for table in ("scans", "polarization_pairs", "anisotropy_sectors"):
                write_csv(root, config.TABLE_FILES[table], generated[table])
            loaded = config.load_scan_set("synthetic", root)
            self.assertEqual(loaded["scans"], generated["scans"])
            self.assertEqual(loaded["anisotropy_sectors"], generated["anisotropy_sectors"])

    def test_manifest_keeps_manual_and_generated_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs(root, scan_set((0, 90), "within_run"))
            loaded = config.load_scan_set("synthetic", root)
            with patch("rsoxs_unified_workup.outputs._git_value", return_value=None):
                manifest = build_manifest(loaded, {})
            saved = json.loads(json.dumps(manifest))["configuration"]
            self.assertEqual(saved["scan_set"]["pol0_scan_id"], 42)
            self.assertIsNone(saved["scan_set"]["pol45_scan_id"])
            self.assertEqual(len(saved["scans"]), 2)
            self.assertEqual(len(saved["anisotropy_sectors"]), 4)
            self.assertEqual(saved["polarization_pairs"][0]["polarization_deg_2"], 90)


if __name__ == "__main__":
    unittest.main()
