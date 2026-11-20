"""Synthetic checks for explicitly choosing TOML centers without fitting."""
import tempfile
import unittest
from unittest.mock import patch

import xarray as xr

from rsoxs_unified_workup import beam_center, config, integration, masks, outputs
from test_multi_polarization import product_fixture


class ConfiguredCenterTests(unittest.TestCase):
    def fixture(self, pols=(0, 90), layout="within_run"):
        args = product_fixture(pols, layout)
        records = args["scans"]
        for record in records.values():
            record.pop("beam_center_approval", None)
        return args

    def approve(self, args):
        return beam_center.approve_configured_beam_centers(
            args["scans"], args["configuration"]["processing"],
            config.get_mask_pairs(args["configuration"]), approve=True,
        )

    def test_explicit_approval_required(self):
        args = self.fixture()
        pairs = config.get_mask_pairs(args["configuration"])
        with self.assertRaises(ValueError):
            beam_center.approve_configured_beam_centers(
                args["scans"], args["configuration"]["processing"], pairs,
            )
        self.assertTrue(all("beam_center_approval" not in r for r in args["scans"].values()))

    def test_two_and_four_polarizations_without_fit(self):
        for pols in ((0, 90), (45, 135), (0, 90, 45, 135)):
            for layout in ("within_run", "separate_runs"):
                with self.subTest(pols=pols, layout=layout):
                    args = self.fixture(pols, layout)
                    centers = self.approve(args)
                    self.assertEqual(len(centers), len(pols) // 2)
                    processing = args["configuration"]["processing"]
                    for record in args["scans"].values():
                        self.assertNotIn("beam_center_fit", record)
                        self.assertEqual(record["beam_center_approval"]["selection_method"], "configured_no_fit")
                        self.assertEqual(
                            beam_center.get_approved_beam_center(record, processing),
                            {"x": .5, "y": .5},
                        )
                        # Integration's imported getter must also accept this path.
                        self.assertEqual(integration.get_approved_beam_center(record, processing), {"x": .5, "y": .5})

    def test_validation_is_atomic(self):
        args = self.fixture((0, 90, 45, 135))
        args["scans"][135]["final_mask"].attrs["beam_center"] = {"x": 9., "y": 9.}
        with self.assertRaises(ValueError):
            self.approve(args)
        self.assertTrue(all("beam_center_approval" not in r for r in args["scans"].values()))

    def test_distinct_pair_centers_are_preserved(self):
        args = self.fixture((0, 90, 45, 135))
        processing = args["configuration"]["processing"]
        processing["beam_centers"]["synthetic_pair_45_135"] = {
            "center_x": 1.25, "center_y": .75,
        }
        for pol in (45, 135):
            args["scans"][pol]["final_mask"].attrs["beam_center"] = {"x": 1.25, "y": .75}
        centers = self.approve(args)
        self.assertEqual(centers["synthetic_pair_0_90"], {"x": .5, "y": .5})
        self.assertEqual(centers["synthetic_pair_45_135"], {"x": 1.25, "y": .75})
        for pol in (45, 135):
            self.assertEqual(beam_center.get_approved_beam_center(args["scans"][pol], processing), {"x": 1.25, "y": .75})

    def test_changed_center_mask_or_scan_rejected(self):
        modifications = (
            lambda a: a["configuration"]["processing"]["beam_centers"]["synthetic_pair_0_90"].update(center_x=.6),
            lambda a: a["scans"][0]["final_mask"].attrs.update(mask_id="rebuilt"),
            lambda a: a["configuration"]["processing"]["center_mask"].update(outer_usable_radius=3.),
            lambda a: a["scans"][0].update(scan_id=999),
            lambda a: a["scans"][0].update(polarization_deg=90),
            lambda a: a["scans"][0]["final_mask"].attrs.update(mask_pair_id="different_pair"),
        )
        for change in modifications:
            args = self.fixture()
            self.approve(args)
            change(args)
            with self.assertRaises(ValueError):
                beam_center.get_approved_beam_center(args["scans"][0], args["configuration"]["processing"])

    def test_search_settings_do_not_invalidate_unfitted_center(self):
        args = self.fixture()
        self.approve(args)
        args["configuration"]["processing"]["beam_center_fit"] = {"fit_energy": 12345., "reflection_mode": "none"}
        self.assertEqual(
            beam_center.get_approved_beam_center(args["scans"][0], args["configuration"]["processing"]),
            {"x": .5, "y": .5},
        )

    def test_applying_masks_revokes_approval(self):
        args = self.fixture()
        self.approve(args)
        masks.apply_final_masks(args["scans"], {p: r["final_mask"] for p, r in args["scans"].items()})
        with self.assertRaises(ValueError):
            beam_center.get_approved_beam_center(args["scans"][0], args["configuration"]["processing"])

    def test_fit_based_approval_still_requires_real_fit(self):
        args = self.fixture()
        with self.assertRaises(ValueError):
            beam_center.get_approved_beam_center(args["scans"][0], args["configuration"]["processing"])
        record = args["scans"][0]
        entry = {"mask_pair_id": "synthetic_pair_0_90", "mask_pair_polarizations": [0, 90], "mask_id": "mask", "pass_id": "fit-pass"}
        record["beam_center_approval"] = {
            **entry, "center": {"x": .5, "y": .5}, "selection_method": "fitted_review",
        }
        with patch.object(beam_center, "_current_center_fit", return_value=entry) as check_fit:
            self.assertEqual(beam_center.get_approved_beam_center(record, args["configuration"]["processing"]), {"x": .5, "y": .5})
            check_fit.assert_called_once()

    def test_manual_exports_and_stale_products(self):
        from numcodecs import blosc
        blosc.use_threads = False
        args = self.fixture()
        self.approve(args)
        for record in args["scans"].values():
            pass_id = record["beam_center_approval"]["pass_id"]
            for section in ("integration", "dark_integration", "intensity"):
                record[section]["beam_center_pass_id"] = pass_id
        datasets = outputs.build_pair_workup_datasets(**args)
        for group, dataset in datasets.items():
            if "/scans/" not in group:
                self.assertEqual(dataset.attrs["beam_center_selection_methods"], {"0": "configured_no_fit", "90": "configured_no_fit"})
        with tempfile.TemporaryDirectory(prefix="rsoxs-manual-center-test-") as directory:
            summary = outputs.save_workup_results(**args, aggregate_anisotropy_results=None, output_root=directory)
            for group, dataset in datasets.items():
                with xr.open_zarr(summary["zarr_store"], group=group, consolidated=True, chunks=None) as saved:
                    xr.testing.assert_identical(outputs._prepare_zarr_dataset(dataset), saved.load())
        self.approve(args)
        with self.assertRaisesRegex(ValueError, "stale or missing"):
            outputs.build_pair_workup_datasets(**args)


if __name__ == "__main__":
    unittest.main()
