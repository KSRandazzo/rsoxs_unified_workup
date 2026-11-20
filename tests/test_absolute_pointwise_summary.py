import contextlib
import io
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from rsoxs_unified_workup.anisotropy import summarize_absolute_pointwise_anisotropy
from rsoxs_unified_workup.plotting import plot_absolute_pointwise_energy_review


def fixture(q=(0.01, 0.02, 0.03)):
    values = np.array([
        [[0.2, -0.4, 0.6], [-0.2, 0.4, -0.6]],
        [[0.1, np.nan, 0.5], [-0.1, np.inf, -0.5]],
        [[0.3, -0.6, 0.9], [-0.3, 0.6, -0.9]],
        [[np.nan, np.inf, np.nan], [np.nan, np.nan, np.nan]],
    ])
    array = xr.DataArray(
        values, dims=("energy", "chi", "q"),
        coords={"energy": [280., 281., 282., 283.], "chi": [-0.5, 0.5], "q": list(q)},
        attrs={"q_cap_applied": True, "source_product": "clean"},
    )
    return {mode: {"pair_0_90": {"anisotropy": array.copy(deep=True)}}
            for mode in ("full_ring", "half_ring")}


class SummaryTests(unittest.TestCase):
    def test_absolute_before_mean_and_complete_window(self):
        source = fixture()
        before = source["half_ring"]["pair_0_90"]["anisotropy"].copy(deep=True)
        result = summarize_absolute_pointwise_anisotropy(source, (0.01, 0.03))
        for mode in source:
            data = result[mode]["pair_0_90"]
            np.testing.assert_allclose(data.mean_abs_anisotropy, [0.4, np.nan, 0.6, np.nan], equal_nan=True)
            np.testing.assert_allclose(data.sum_abs_anisotropy, data.mean_abs_anisotropy * 3, equal_nan=True)
            np.testing.assert_array_equal(data.valid_q_count, [3, 2, 3, 0])
            np.testing.assert_array_equal(data.valid_point_count, [6, 4, 6, 0])
            np.testing.assert_array_equal(data.eligible, [True, False, True, False])
            self.assertTrue(data.attrs["q_cap_applied"])
        xr.testing.assert_identical(before, source["half_ring"]["pair_0_90"]["anisotropy"])

    def test_partial_window_and_equal_q_weights(self):
        source = fixture()
        array = source["half_ring"]["pair_0_90"]["anisotropy"]
        array.values[0, 1, 0] = np.nan
        result = summarize_absolute_pointwise_anisotropy(source, (0.01, 0.03), require_complete_q_range=False)
        data = result["half_ring"]["pair_0_90"]
        self.assertAlmostEqual(data.mean_abs_anisotropy.values[0], 0.4)
        self.assertNotAlmostEqual(data.mean_abs_anisotropy.values[0], np.nanmean(abs(array.values[0])))
        self.assertAlmostEqual(data.mean_abs_anisotropy.values[1], 0.3)
        self.assertAlmostEqual(data.sum_abs_anisotropy.values[1], 0.6)
        self.assertTrue(np.isnan(data.sum_abs_anisotropy.values[3]))

    def test_native_pair_grids_and_transposition(self):
        source = fixture()
        for mode in source:
            array = source[mode]["pair_0_90"]["anisotropy"]
            source[mode]["pair_45_135"] = {"anisotropy": array.assign_coords(q=[0.011, 0.021, 0.031]).transpose("q", "energy", "chi")}
        result = summarize_absolute_pointwise_anisotropy(source, (0.015, 0.03))
        for mode in source:
            np.testing.assert_allclose(result[mode]["pair_0_90"].q, [0.02, 0.03])
            np.testing.assert_allclose(result[mode]["pair_45_135"].q, [0.021])

    def test_bad_ranges(self):
        for bounds in ((0.03, 0.01), (-1, 0.03), (0.01, np.nan), (0.01,), "bad"):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                summarize_absolute_pointwise_anisotropy(fixture(), bounds)
        for bounds in ((0.04, 0.05), (0.012, 0.013)):
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, "available q-grid range"):
                summarize_absolute_pointwise_anisotropy(fixture(), bounds)

    def test_bad_source(self):
        for q in ([0.01, 0.01, 0.03], [0.02, 0.01, 0.03], [0.01, np.nan, 0.03]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                summarize_absolute_pointwise_anisotropy(fixture(q=q), (0.01, 0.03))
        source = fixture()
        source["half_ring"]["pair_0_90"]["anisotropy"].attrs.clear()
        with self.assertRaisesRegex(ValueError, "q-capped"):
            summarize_absolute_pointwise_anisotropy(source, (0.01, 0.03))
        source = fixture()
        source["half_ring"]["pair_0_90"]["anisotropy"] = source["half_ring"]["pair_0_90"]["anisotropy"].mean("chi")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            summarize_absolute_pointwise_anisotropy(source, (0.01, 0.03))


class PlotTests(unittest.TestCase):
    def capture(self, data, **kwargs):
        figures = []
        with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, "show", side_effect=lambda: figures.append(plt.gcf())), patch.object(plt, "close"):
            plot_absolute_pointwise_energy_review(data, **kwargs)
        self.addCleanup(lambda: [plt.close(figure) for figure in figures])
        return figures

    def test_median_reference_and_missing_gaps(self):
        data = summarize_absolute_pointwise_anisotropy(fixture(), (0.01, 0.03))
        figures = self.capture(data)
        self.assertEqual(len(figures), 1)
        axis = figures[0].axes[0]
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [0.4, np.nan, 0.6, np.nan], equal_nan=True)
        self.assertEqual(axis.lines[1].get_linestyle(), ":")
        np.testing.assert_allclose(axis.lines[1].get_ydata(), [0.5, 0.5])

    def test_sum_manual_reference_and_all_missing(self):
        data = summarize_absolute_pointwise_anisotropy(fixture(), (0.01, 0.03))
        figures = self.capture(data, statistic="sum", reference_level=1.0)
        np.testing.assert_allclose(figures[0].axes[0].lines[0].get_ydata(), [1.2, np.nan, 1.8, np.nan], equal_nan=True)
        np.testing.assert_allclose(figures[0].axes[0].lines[1].get_ydata(), [1., 1.])
        source = fixture()
        for mode in source:
            source[mode]["pair_0_90"]["anisotropy"].values[:] = np.nan
        missing = summarize_absolute_pointwise_anisotropy(source, (0.01, 0.03))
        self.assertEqual(len(self.capture(missing)[0].axes[0].lines), 1)

    def test_bad_plot_options(self):
        data = summarize_absolute_pointwise_anisotropy(fixture(), (0.01, 0.03))
        for kwargs in ({"statistic": "median"}, {"normalization_mode": "other"}, {"reference_level": -1}, {"reference_level": np.inf}, {"reference_level": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plot_absolute_pointwise_energy_review(data, **kwargs)


if __name__ == "__main__":
    unittest.main()
