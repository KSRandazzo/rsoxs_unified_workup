import ast
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from rsoxs_unified_workup import anisotropy, angular_fit, config, plotting, workflow


def fixture(pols=(0, 90), layout="within_run"):
    energies = [280., 284.8, 286., 290.]
    chi = np.arange(-175., 180., 10.)
    q = np.array([0.01, 0.02, 0.03, 0.04])
    scans = [{"scan_set_id": "test", "scan_id": 42 if layout == "within_run" else 42+i,
              "polarization_deg": pol} for i, pol in enumerate(pols)]
    pairs = []
    records = {}
    results = {"full_ring": {}, "half_ring": {}}
    settings = {"period_degrees": 90, "harmonics": 1, "q_bins": 3,
                "q_value": 0.02, "energy_ev": 280.}
    for i in range(0, len(pols), 2):
        a, b = scans[i:i+2]
        pair_id = f"pair_{pols[i]}_{pols[i+1]}"
        pair = {"pair_id": pair_id, "scan_set_id": "test", "scan_id_1": a["scan_id"], "scan_id_2": b["scan_id"]}
        if layout == "within_run":
            pair.update(polarization_deg_1=pols[i], polarization_deg_2=pols[i+1])
        pairs.append(pair)
        values = np.array([0.1, 0.02, 0.04, 0.06])[:, None, None] + 0.08*np.cos(2*np.pi*chi/90)[None, :, None] + np.zeros((4, chi.size, 4))
        values[:, :2, :] = np.nan
        values[:, :, -1] = np.nan
        data = xr.DataArray(values, dims=("energy", "chi", "q"),
                            coords={"energy": energies, "chi": chi, "q": q},
                            attrs={"q_cap_applied": True, "source_product": "clean"})
        cap = xr.DataArray([0.03]*4, dims="energy", coords={"energy": energies})
        for mode in results:
            results[mode][pair_id] = {"anisotropy": data.copy(deep=True)}
        for record in (a, b):
            records[record["polarization_deg"]] = {**record, "scan_type": "long", "data": data,
                                                   "integration": {"q_cap": cap.copy(deep=True), "mask_pair_id": pair_id}}
    configuration = {"scan_set": {"scan_set_id": "test", "scan_type": "long", "polarization_layout": layout},
                     "scans": scans, "polarization_pairs": pairs, "anisotropy_sectors": [],
                     "processing": {"plotting": {"energies_ev": [284.8, 286.]},
                                    "beam_center_fit": {"fit_energy": 280.},
                                    "normalization": {"half_ring_halfwidth_degrees": 90},
                                    "angular_anisotropy_fit": settings}}
    return configuration, records, results


class EnergySummaryTests(unittest.TestCase):
    def test_companion_order_manual_reference_and_unchanged_source(self):
        _, _, results = fixture()
        before = results["half_ring"]["pair_0_90"]["anisotropy"].copy(deep=True)
        summary = anisotropy.summarize_pointwise_energy(results, (0.01, 0.03))
        for mode in results:
            ds = summary[mode]["pair_0_90"]
            raw = results[mode]["pair_0_90"]["anisotropy"].isel(q=slice(0, 3))
            expected = raw.mean("chi", skipna=True)
            companion = abs(raw).mean("chi", skipna=True)
            xr.testing.assert_allclose(ds.q_reduced_chi_mean, abs(expected).mean("q"))
            xr.testing.assert_allclose(ds.q_reduced_mean_abs, companion.mean("q"))
            self.assertTrue((ds.q_reduced_mean_abs >= ds.q_reduced_chi_mean - 1e-12).all())
            self.assertGreater(float(ds.q_reduced_mean_abs[1]), float(ds.q_reduced_chi_mean[1]))
            self.assertIs(ds.attrs["periodic_fit_used"], False)
        xr.testing.assert_identical(results["half_ring"]["pair_0_90"]["anisotropy"], before)

    def test_sum_missing_signed_and_complete_window(self):
        _, _, results = fixture()
        for pairs in results.values():
            values = pairs["pair_0_90"]["anisotropy"].values
            values[0, :, :] = np.nan
            values[1, :, 1] = np.nan
        partial = anisotropy.summarize_pointwise_energy(results, (0.01, 0.03), q_reduction="sum", absolute_q_values=False)
        strict = anisotropy.summarize_pointwise_energy(results, (0.01, 0.03), require_complete_q_range=True)
        ds = partial["half_ring"]["pair_0_90"]
        self.assertTrue(np.isnan(ds.q_reduced_chi_mean[0]))
        self.assertTrue(np.isnan(ds.q_reduced_mean_abs[0]))
        np.testing.assert_array_equal(ds.valid_q_count, [0, 2, 3, 3])
        self.assertTrue(np.isnan(strict["half_ring"]["pair_0_90"].q_reduced_chi_mean[1]))
        self.assertTrue(np.isfinite(ds.q_reduced_chi_mean[1]))

    def test_not_fit_data_and_bad_options(self):
        _, _, source = fixture()
        for options in ({"q_reduction": "bad"}, {"absolute_q_values": "yes"}):
            with self.assertRaises(ValueError):
                anisotropy.summarize_pointwise_energy(source, (0.01, 0.03), **options)
        for bounds in ((0.03, 0.01), (0.05, 0.06), (0.011, 0.019)):
            with self.assertRaises(ValueError):
                anisotropy.summarize_pointwise_energy(source, bounds)
        source["half_ring"]["pair_0_90"]["anisotropy"] = source["half_ring"]["pair_0_90"]["anisotropy"].mean("chi")
        with self.assertRaisesRegex(ValueError, "not fit maps"):
            anisotropy.summarize_pointwise_energy(source, (0.01, 0.03))


class WorkflowTests(unittest.TestCase):
    def test_missing_csv_metadata_stable_across_reload(self):
        cfg, _, _ = fixture()
        cfg["scan_set"]["optional_metadata"] = float("nan")
        other = deepcopy(cfg)
        other["scan_set"]["optional_metadata"] = float("nan")
        workflow.assert_processing_current(workflow.numerical_configuration(cfg), other)
        workflow.assert_stage_current(workflow.stage_configuration(cfg, "integration"), other, "integration")

    def test_shared_energies_include_center_without_mutation(self):
        cfg, scans, _ = fixture()
        original = deepcopy(cfg)
        self.assertEqual(config.get_review_energies(cfg["processing"]), [284.8, 286., 280.])
        self.assertEqual(config.get_review_energies(cfg["processing"], include_beam_center=False), [284.8, 286.])
        self.assertEqual(workflow.validate_review_energies(cfg, scans), [284.8, 286., 280.])
        self.assertEqual(cfg, original)
        cfg["processing"]["plotting"]["energies_ev"] = [284.8, 284.8, 286.]
        self.assertEqual(config.get_review_energies(cfg["processing"]), [284.8, 286., 280.])
        cfg["processing"]["plotting"]["energies_ev"] = [np.nan]
        with self.assertRaises(ValueError):
            config.get_review_energies(cfg["processing"])

    def test_physical_vs_display_changes_and_review_staleness(self):
        cfg, _, data = fixture()
        snapshot = workflow.numerical_configuration(cfg)
        token = workflow.mark_anisotropy_reviewed(cfg, data, (0.01, 0.03))
        workflow.require_anisotropy_review(token, cfg, data, (0.01, 0.03))
        cfg["processing"]["plotting"]["energies_ev"] = [286.]
        workflow.assert_processing_current(snapshot, cfg)
        with self.assertRaises(RuntimeError):
            workflow.require_anisotropy_review(token, cfg, data, (0.01, 0.03))
        cfg["processing"]["normalization"]["half_ring_halfwidth_degrees"] = 80
        with self.assertRaisesRegex(RuntimeError, "Numerical configuration changed"):
            workflow.assert_processing_current(snapshot, cfg)

    def test_stage_checks_and_recomputed_source(self):
        cfg, _, data = fixture()
        snapshot = workflow.stage_configuration(cfg, "integration")
        cfg["processing"]["normalization"]["half_ring_halfwidth_degrees"] = 80
        workflow.assert_stage_current(snapshot, cfg, "integration")
        token = workflow.mark_anisotropy_reviewed(cfg, data, (0.01, 0.03))
        for other, bounds in ((deepcopy(data), (0.01, 0.03)), (data, (0.02, 0.03))):
            with self.assertRaises(RuntimeError):
                workflow.require_anisotropy_review(token, cfg, other, bounds)
        cfg["processing"]["beam_center_fit"]["fit_energy"] = 284.8
        with self.assertRaises(RuntimeError):
            workflow.assert_stage_current(snapshot, cfg, "integration")


class MultiEnergyFitTests(unittest.TestCase):
    def test_two_four_pols_and_layouts_fit_every_selected_energy(self):
        for pols, layout in (((0, 90), "within_run"), ((45, 135), "within_run"),
                             ((0, 90, 45, 135), "within_run"), ((0, 90), "separate_runs")):
            cfg, scans, data = fixture(pols, layout)
            with self.subTest(pols=pols, layout=layout):
                result = angular_fit.fit_review_energy_waveforms(cfg, scans, data)
                for mode, pairs in result.items():
                    self.assertEqual(set(pairs), set(data[mode]))
                    for entries in pairs.values():
                        self.assertEqual([e["requested_energy_ev"] for e in entries], [284.8, 286.])
                        self.assertTrue(all(e["status"] == "fitted" for e in entries))
                        for entry in entries:
                            comparison = entry["comparison"]
                            xr.testing.assert_identical(comparison["sinusoid"]["observed"], comparison["triangle"]["observed"])
                            self.assertEqual(entry["selected_energy_ev"], entry["requested_energy_ev"])

    def test_cap_or_missing_data_is_reported_not_fabricated(self):
        cfg, scans, data = fixture()
        for record in scans.values():
            record["integration"]["q_cap"].values[1] = 0.015
        for pairs in data.values():
            pairs["pair_0_90"]["anisotropy"].values[2, :, :] = np.nan
        result = angular_fit.fit_review_energy_waveforms(cfg, scans, data)
        for pairs in result.values():
            entries = pairs["pair_0_90"]
            self.assertTrue(all(e["status"] == "not_fitted" for e in entries))
            self.assertIn("q cap", entries[0]["reason"])
            self.assertIsNone(entries[0]["sinusoid"])

    def test_all_selected_slice_parameters_cross_check_full_maps(self):
        cfg, scans, data = fixture()
        reviews = angular_fit.fit_review_energy_waveforms(cfg, scans, data)
        maps = angular_fit.fit_scan_set_angular_maps(cfg, scans, data, progress=False)
        refs = workflow.map_references_from_energy_reviews(reviews)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, "show"):
            plotting.plot_angular_anisotropy_map_review(maps, slice_fits=refs,
                                                       energy_range_ev=(284., 287.), energy_markers_ev=[284.8, 286.])
        refs["half_ring"]["pair_0_90"][1]["sinusoid_parameters"]["offset"] += 1
        with self.assertRaises(AssertionError), contextlib.redirect_stdout(io.StringIO()):
            plotting.plot_angular_anisotropy_map_review(maps, slice_fits=refs)


class PlotAndNotebookTests(unittest.TestCase):
    def test_workup_preview_and_fit_map_windows_follow_the_energy_region(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/workup.ipynb").read_text())
        sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        preview_source = next(source for source in sources if "# Display windows for the measured" in source)
        map_source = next(source for source in sources if "FIT_MAP_ENERGY_RANGES_EV =" in source)
        for region, energies, preview_bounds, fit_bounds in (
            ("C", [284.8, 286.0], (280, 290), (282, 290)),
            ("O", [533.0, 535.0], (528, 538), (528, 538)),
        ):
            cfg, scans, _ = fixture()
            cfg["scan_set"]["energy_region"] = region
            cfg["processing"]["plotting"]["energies_ev"] = energies
            cfg["processing"]["beam_center_fit"]["fit_energy"] = energies[0]
            for record in scans.values():
                record["data"] = record["data"].assign_coords(energy=[energies[0] - 1, *energies, energies[-1] + 1])
            values = xr.DataArray(np.ones((2, 2)), dims=("energy", "q"),
                                  coords={"energy": energies, "q": [0.01, 0.02]})
            namespace = {
                "SCAN_SET_ID": "test", "load_scan_set": lambda _: cfg,
                "workup_plotting": plotting, "workflow": workflow, "scan": scans,
                "_processed_settings": workflow.numerical_configuration(cfg),
                "NORMALIZATION_MODE": "half_ring", "pointwise_energy_summary_results": {},
                "chi_averaged_pointwise_results": {"half_ring": {"pair": {"anisotropy": values}}},
                "angular_anisotropy_map_results": {"fixture": True},
            }
            figures = []
            with self.subTest(region=region), patch("importlib.reload", side_effect=lambda module: module), \
                 patch.object(plotting, "plot_pointwise_energy_diagnostics"), \
                 patch.object(plotting, "plot_angular_anisotropy_map_review") as map_plot, \
                 patch.object(plt, "show", side_effect=lambda: figures.append(plt.gcf())), patch.object(plt, "close"):
                exec(compile(preview_source, "<workup energy preview>", "exec"), namespace)
                self.assertEqual(tuple(figures[-1].axes[0].get_ylim()), preview_bounds)
                self.assertEqual([float(line.get_ydata()[0]) for line in figures[-1].axes[0].lines], energies)
                exec(compile(map_source, "<workup fit-map zoom>", "exec"), namespace)
                self.assertEqual(map_plot.call_args.kwargs["energy_range_ev"], fit_bounds)
                self.assertEqual(map_plot.call_args.kwargs["energy_markers_ev"], energies)
            for figure in figures:
                plt.close(figure)

    def test_real_derived_products_refresh_without_reprocessing(self):
        from test_multi_polarization import product_fixture
        arguments = product_fixture()
        cfg = arguments["configuration"]
        cfg["processing"].update(plotting={"energies_ev": [284.8, 286.]},
                                 beam_center_fit={"fit_energy": 280.})
        before = {pol: record["integration"]["clean"].copy(deep=True) for pol, record in arguments["scans"].items()}
        with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, "show"):
            plotting.refresh_selected_energy_reviews(
                cfg, arguments["scans"], arguments["pointwise_anisotropy_results"],
                arguments["chi_averaged_pointwise_results"], arguments["sector_anisotropy_results"],
                arguments["sector_averaged_anisotropy_results"], arguments["global_intensity_ratio_results"],
                include_image_reviews=False,
            )
        for pol, record in arguments["scans"].items():
            xr.testing.assert_identical(record["integration"]["clean"], before[pol])

    def test_map_window_markers_and_empty_range_error(self):
        cfg, scans, data = fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            maps = angular_fit.fit_scan_set_angular_maps(cfg, scans, data, progress=False)
        figures = []
        with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, "show", side_effect=lambda: figures.append(plt.gcf())), patch.object(plt, "close"):
            plotting.plot_angular_anisotropy_map_review(maps, energy_range_ev=(284., 287.), energy_markers_ev=[284.8, 286., 290.])
        self.addCleanup(lambda: [plt.close(fig) for fig in figures])
        for ax in [ax for ax in figures[0].axes if ax.get_xlabel() == "q (Å⁻¹)"]:
            np.testing.assert_allclose(ax.get_ylim(), [284., 287.])
            self.assertEqual(len(ax.lines), 2)
            for line, energy in zip(ax.lines, [284.8, 286.]):
                np.testing.assert_allclose(line.get_ydata(), [energy, energy])
                self.assertEqual(line.get_linestyle(), ":")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "available energy range"):
            plotting.plot_angular_anisotropy_map_review(maps, energy_range_ev=(300., 310.))

    def test_data_energy_plot_and_both_waveforms(self):
        cfg, scans, data = fixture()
        summary = anisotropy.summarize_pointwise_energy(data, (0.01, 0.03))
        figures = []
        with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, "show", side_effect=lambda: figures.append(plt.gcf())), patch.object(plt, "close"):
            plotting.plot_pointwise_energy_diagnostics(summary, energy_markers_ev=[284.8, 286.])
            plotting.plot_review_energy_waveforms(angular_fit.fit_review_energy_waveforms(cfg, scans, data))
        self.addCleanup(lambda: [plt.close(fig) for fig in figures])
        self.assertEqual(len(figures), 3)
        np.testing.assert_allclose(figures[0].axes[0].lines[0].get_ydata(), summary["half_ring"]["pair_0_90"].q_reduced_chi_mean)
        np.testing.assert_allclose(figures[0].axes[0].lines[1].get_ydata(), summary["half_ring"]["pair_0_90"].q_reduced_mean_abs)
        self.assertEqual(len(figures[1].axes[0].lines), 2)

    def test_refresh_only_calls_plots(self):
        cfg, scans, data = fixture()
        plot_names = ["plot_raw_frames", "plot_saturation_overlays", "plot_final_masked_frames", "plot_integration_review",
                      "plot_dark_integration_review", "plot_normalization_review", "plot_flux_correction_review",
                      "plot_ring_mean_intensity_review", "plot_pointwise_anisotropy_review", "plot_chi_averaged_pointwise_review",
                      "plot_sector_anisotropy_review", "plot_anisotropy_sector_overlays", "plot_sector_averaged_anisotropy_review",
                      "plot_global_intensity_ratio_review", "plot_anisotropy_method_comparison"]
        with contextlib.ExitStack() as stack:
            mocks = {name: stack.enter_context(patch.object(plotting, name)) for name in plot_names}
            plotting.refresh_selected_energy_reviews(cfg, scans, data, data, data, data, data, saturation_masks={})
        self.assertTrue(all(mock.called for mock in mocks.values()))
        self.assertEqual(mocks["plot_anisotropy_sector_overlays"].call_count, 3)

    def test_notebook_order_single_energy_source_and_export_guard(self):
        notebook = json.loads((Path(__file__).resolve().parents[1]/"notebooks/workup.ipynb").read_text())
        cells = notebook["cells"]
        codes = [(i, "".join(c["source"])) for i, c in enumerate(cells) if c["cell_type"] == "code"]
        position = lambda phrase: next(i for i, s in codes if phrase in s)
        self.assertLess(position("plot_anisotropy_robustness_overview("), position("fit_review_energy_waveforms("))
        self.assertLess(position("summarize_pointwise_energy("), position("fit_review_energy_waveforms("))
        self.assertLess(position("plot_review_energy_waveforms("), position("fit_scan_set_angular_maps("))
        self.assertFalse(any("plot_angular_anisotropy_fit_review(" in s for _, s in codes))
        self.assertFalse(any("ENERGY_MARKERS_EV = [" in s for _, s in codes))
        for _, s in codes:
            ast.parse(s)
        self.assertTrue(all(c["outputs"] == [] and c["execution_count"] is None for c in cells if c["cell_type"] == "code"))


if __name__ == "__main__":
    unittest.main()
