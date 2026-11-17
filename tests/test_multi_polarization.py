"""Synthetic tests only: no BNL connection or measured image loading."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from rsoxs_unified_workup import (
    config, loading, darks, anisotropy, integration, masks,
    normalization, intensity, outputs, plotting,
)


def configuration(polarizations=(0, 90), layout="within_run", shared_id=True):
    scans = [
        {"scan_id": 42 if shared_id else 42 + index,
         "scan_set_id": "synthetic", "polarization_deg": pol}
        for index, pol in enumerate(polarizations)
    ]
    pairs = []
    sectors = []
    for index in range(0, len(scans), 2):
        a, b = scans[index:index + 2]
        pair_id = f"synthetic_pair_{a['polarization_deg']}_{b['polarization_deg']}"
        pair = {"pair_id": pair_id, "scan_set_id": "synthetic",
                "scan_id_1": a["scan_id"], "scan_id_2": b["scan_id"]}
        if layout == "within_run":
            pair.update(polarization_deg_1=a["polarization_deg"], polarization_deg_2=b["polarization_deg"])
        pairs.append(pair)
        for sector_index, angle in enumerate((a['polarization_deg'], a['polarization_deg'] + 90,
                                               a['polarization_deg'] + 180, a['polarization_deg'] - 90)):
            sectors.append({"sector_id": f"{pair_id}_sector_{sector_index}", "scan_set_id": "synthetic",
                            "pair_id": pair_id, "label": str(angle), "chi_center_deg": angle,
                            "chi_halfwidth_deg": 45})
    return {
        "sample": {"sample_id": "synthetic_sample", "sample_name": "Synthetic",
                   "diameter_nm": np.nan, "mw_g_mol": np.nan,
                   "polymer": np.nan, "wash_state": np.nan},
        "scan_set": {"scan_set_id": "synthetic", "scan_type": "long", "energy_region": "C",
                     "polarization_layout": layout},
        "scans": scans, "polarization_pairs": pairs, "anisotropy_sectors": sectors,
        "processing": {"normalization": {"half_ring_center_degrees": 180,
                                             "half_ring_halfwidth_degrees": 90}},
    }


def source_array(polarizations=(0, 90), blocked=False, dtype=np.int64):
    energies = np.array([280., 284.8, 286.])
    pols = np.repeat(polarizations, energies.size) if blocked else np.tile(polarizations, energies.size)
    energy_values = np.tile(energies, len(polarizations)) if blocked else np.repeat(energies, len(polarizations))
    n = pols.size
    index = pd.MultiIndex.from_arrays([energy_values, pols], names=["energy", "polarization"])
    array = xr.DataArray(np.arange(n * 6, dtype=dtype).reshape(n, 2, 3),
                         dims=("system", "pix_y", "pix_x"),
                         coords=xr.Coordinates.from_pandas_multiindex(index, "system"),
                         name="Wide Angle CCD Detector_image",
                         attrs={"polarization": pols.copy(), "energy": energy_values.copy(),
                                "RSoXS Au Mesh Current": 10. + np.arange(n),
                                "NSLS-II Ring Current": 300. + np.arange(n),
                                "exposure": 1. + np.arange(n), "sdd": 39.19,
                                "pixel1": 6e-5, "meas_time": "synthetic-time",
                                "start": {"scan_id": 42, "sample_name": "Synthetic"}})
    return array.assign_coords(dark_id=("system", np.arange(n) % 2),
                               frame_time=("system", 1000. + np.arange(n)))


class FakeLoader:
    def __init__(self, sources):
        self.sources = sources
        self.calls = []

    def loadRun(self, scan_id, **kwargs):
        self.calls.append((scan_id, kwargs))
        return self.sources[scan_id]


class ConfigTests(unittest.TestCase):
    def test_legacy_csvs_and_default_layout(self):
        for layout in (None, "", np.nan, "separate_runs"):
            c = configuration(layout="separate_runs", shared_id=False)
            c["scan_set"]["polarization_layout"] = layout
            self.assertEqual(config.get_mask_pairs(c), {"synthetic_pair_0_90": (0, 90)})
            for pair in c["polarization_pairs"]:
                pair.update(polarization_deg_1=np.nan, polarization_deg_2=np.nan)
            self.assertEqual(anisotropy.pair_polarizations(c, c["polarization_pairs"][0]), (0, 90))

    def test_two_and_four_polarizations_same_source(self):
        for pols in ((0, 90), (45, 135), (0, 90, 45, 135)):
            c = configuration(pols)
            self.assertEqual(len(config.get_mask_pairs(c)), len(pols) // 2)
            self.assertEqual(loading.scan_ids_by_polarization(c), {pol: 42 for pol in pols})
            for pair in c["polarization_pairs"]:
                self.assertEqual(anisotropy.pair_polarizations(c, pair),
                                 (pair["polarization_deg_1"], pair["polarization_deg_2"]))

    def test_different_mixed_source_runs(self):
        c = configuration((0, 90, 45, 135))
        for row in c["scans"][2:]:
            row["scan_id"] = 43
        c["polarization_pairs"][1].update(scan_id_1=43, scan_id_2=43)
        self.assertEqual(len(config.get_mask_pairs(c)), 2)

    def test_configuration_errors(self):
        cases = []
        c = configuration(); c['scan_set']['polarization_layout'] = 'bad'; cases.append(c)
        c = configuration(); c['scan_set']['polarization_layout'] = 'separate_runs'; cases.append(c)
        c = configuration(); del c['polarization_pairs'][0]['polarization_deg_2']; cases.append(c)
        c = configuration(); c['polarization_pairs'][0]['polarization_deg_2'] = 45; cases.append(c)
        c = configuration(); c['polarization_pairs'][0]['polarization_deg_2'] = 0; cases.append(c)
        c = configuration(); c['polarization_pairs'][0]['scan_id_2'] = 99; cases.append(c)
        c = configuration(); c['scans'].append(copy.deepcopy(c['scans'][0])); cases.append(c)
        c = configuration(); c['polarization_pairs'] = []; cases.append(c)
        c = configuration(); p = copy.deepcopy(c['polarization_pairs'][0]); p['pair_id'] = 'other'; c['polarization_pairs'].append(p); cases.append(c)
        c = configuration(); c['scans'][0]['scan_id'] = 42.5; cases.append(c)
        c = configuration(); c['scans'][0]['polarization_deg'] = np.nan; cases.append(c)
        for c in cases:
            with self.subTest(c=c), self.assertRaises(ValueError):
                config.get_mask_pairs(c)
        for value in (True, np.inf, -1, 'nonsense'):
            with self.subTest(scan_id=value), self.assertRaises(ValueError):
                config.scan_identity({'scan_id': value, 'polarization_deg': 0})


class LoadingTests(unittest.TestCase):
    def test_exact_selection_and_one_read_per_source(self):
        for pols in ((0, 90), (45, 135), (0, 90, 45, 135)):
            for blocked in (False, True):
                source = source_array(pols, blocked=blocked)
                before = source.copy(deep=True)
                loader = FakeLoader({42: source})
                records = loading.load_scans(loader, configuration(pols))
                self.assertEqual(loader.calls, [(42, {'dims': ['energy', 'polarization']})])
                for pol, record in records.items():
                    indices = np.flatnonzero(source.polarization.values == pol)
                    actual = record['data']
                    self.assertEqual(actual.dtype, source.dtype)
                    self.assertEqual(actual.dims, ('system', 'pix_y', 'pix_x'))
                    self.assertEqual(actual.indexes['system'].names, ['energy'])
                    np.testing.assert_array_equal(actual.values, source.values[indices])
                    np.testing.assert_array_equal(actual.energy, source.energy.values[indices])
                    np.testing.assert_array_equal(actual.dark_id, source.dark_id.values[indices])
                    np.testing.assert_array_equal(actual.frame_time, source.frame_time.values[indices])
                    np.testing.assert_array_equal(actual.source_frame_index, indices)
                    np.testing.assert_array_equal(record['au_mesh_current'], source.attrs['RSoXS Au Mesh Current'][indices])
                    np.testing.assert_array_equal(record['ring_current'], source.attrs['NSLS-II Ring Current'][indices])
                    np.testing.assert_array_equal(actual.attrs['exposure'], source.attrs['exposure'][indices])
                    self.assertEqual(actual.attrs['polarization'], float(pol))
                    self.assertEqual(record['polarization_selection']['source_frame_indices'], indices.tolist())
                    actual.unstack('system').transpose('pix_y', 'pix_x', 'energy')
                    prepared, prepared_masks = integration.prepare_integration_inputs(record, xr.zeros_like(actual, dtype=bool))
                    self.assertEqual(prepared.shape, (2, 3, 3))
                    self.assertEqual(prepared.shape, prepared_masks.shape)
                xr.testing.assert_identical(source, before)

    def test_legacy_scalar_attribute_and_energy_only_index(self):
        c = configuration(layout='separate_runs', shared_id=False)
        sources = {}
        for row in c['scans']:
            pol = row['polarization_deg']
            image = source_array((pol,)).reset_index('system').drop_vars('polarization')
            image = image.drop_vars('energy').assign_coords(
                xr.Coordinates.from_pandas_multiindex(pd.MultiIndex.from_arrays([[280., 284.8, 286.]], names=['energy']), 'system'))
            image.attrs['polarization'] = float(pol)
            sources[row['scan_id']] = image
        loader = FakeLoader(sources)
        records = loading.load_scans(loader, c)
        self.assertEqual(loader.calls, [(42, {}), (43, {})])
        for pol, record in records.items():
            np.testing.assert_array_equal(record['data'].values, sources[record['scan_id']].values)
            self.assertNotIn('source_frame_index', record['data'].coords)

    def test_missing_sample_descriptors_are_not_a_blocker(self):
        label = loading.build_base_description(configuration()['sample'])
        self.assertEqual(label, 'Synthetic')
        self.assertEqual(loading.build_base_description({'sample_name': 'Bare', 'diameter_nm': 50, 'polymer': 'none'}), 'Bare, D=50nm SiNPs')
        label = loading.build_base_description({'diameter_nm': 50, 'mw_g_mol': 259000, 'polymer': 'PS', 'wash_state': 'unwashed'})
        self.assertEqual(label, 'Nanocomposite with D=50nm SiNPs, Mw=259kDa PS, unwashed')

    def test_dtype_and_only_explicit_polarizations_selected(self):
        source = source_array((0, 90, 45, 135), dtype=np.uint32)
        records = loading.load_scans(FakeLoader({42: source}), configuration())
        self.assertEqual(set(records), {0, 90})
        self.assertEqual(records[0]['data'].dtype, np.uint32)
        self.assertEqual(records[0]['polarization_selection']['available_polarizations'], [0., 45., 90., 135.])

    def test_invalid_pair_rejected_before_download(self):
        c = configuration(); del c['polarization_pairs'][0]['polarization_deg_1']
        loader = FakeLoader({42: source_array()})
        with self.assertRaises(ValueError):
            loading.load_scans(loader, c)
        self.assertEqual(loader.calls, [])

    def test_bad_measured_metadata_is_rejected(self):
        def check(source, c=None):
            with self.assertRaises(ValueError):
                loading.load_scans(FakeLoader({42: source}), c or configuration())
        source = source_array((0,)); check(source)
        source = source_array(); source.attrs['RSoXS Au Mesh Current'] = np.ones(5); check(source)
        source = source_array(); source.attrs['RSoXS Au Mesh Current'] = 1; check(source)
        source = source_array(); del source.attrs['RSoXS Au Mesh Current']; check(source)
        source = source_array().reset_index('system').drop_vars('polarization'); check(source)
        source = source_array().reset_index('system').assign_coords(polarization=('system', [0, 90, 0, 90, 0, np.nan])); check(source)
        source = source_array().reset_index('system').assign_coords(energy=('system', [280, 280, 280, 284.8, 286, 286])); check(source)
        source = source_array(); c = configuration(layout='separate_runs', shared_id=False); check(source, c)


class DarkTests(unittest.TestCase):
    def test_complete_stack_read_once_and_single_dark_preserved(self):
        for frame_count in (1, 2):
            class Reader:
                calls = 0
                def read(self):
                    self.calls += 1
                    return xr.Dataset({'Wide Angle CCD Detector_image': xr.DataArray(
                        np.arange(frame_count * 6, dtype=np.uint32).reshape(frame_count, 1, 2, 3),
                        dims=('time', 'dim_0', 'dim_1', 'dim_2'))})
            reader = Reader()
            loader = FakeLoader({42: source_array((0, 90, 45, 135))})
            loader.c = {42: {'dark': {'data': reader}}}
            records = loading.load_scans(loader, configuration((0, 90, 45, 135)))
            with redirect_stdout(io.StringIO()):
                darks.load_dark_frames(loader, records)
            self.assertEqual(reader.calls, 1)
            for record in records.values():
                self.assertEqual(record['darks'].dims, ('time', 'pix_y', 'pix_x'))
                self.assertEqual(record['darks'].shape, (frame_count, 2, 3))
                self.assertEqual(record['darks'].dtype, np.uint32)
                np.testing.assert_array_equal(record['darks'], np.arange(frame_count * 6).reshape(frame_count, 2, 3))


def product_fixture(pols=(0, 90), layout='within_run'):
    """Real downstream calculations on tiny synthetic integrated arrays."""
    c = configuration(pols, layout, shared_id=(layout == 'within_run'))
    source = source_array(pols)
    sources = {42: source}
    if layout == 'separate_runs':
        sources = {
            row['scan_id']: source.isel(system=np.flatnonzero(source.polarization.values == row['polarization_deg']))
            for row in c['scans']
        }
        for data in sources.values():
            indices = np.flatnonzero(source.polarization.values == data.polarization.values[0])
            data.attrs = copy.deepcopy(source.attrs)
            for key in ('energy', 'exposure', 'RSoXS Au Mesh Current', 'NSLS-II Ring Current'):
                data.attrs[key] = data.attrs[key][indices]
    records = loading.load_scans(FakeLoader(sources), c)
    c['processing']['flux_correction'] = {'enabled': True}
    c['processing']['beam_centers'] = {
        pair_id: {'center_x': 0.5, 'center_y': 0.5}
        for pair_id in config.get_mask_pairs(c)
    }
    c['processing']['center_mask'] = {
        'outer_usable_radius': 4., 'saturation_search_width': 1.,
    }
    pairs = config.get_mask_pairs(c)
    small_mask = xr.DataArray(np.zeros((2, 3), dtype=bool), dims=('pix_y', 'pix_x'))
    static = masks.combine_static_masks(
        {pol: small_mask.copy() for pol in pols},
        np.zeros((2, 3), dtype=bool), np.zeros((2, 3), dtype=bool), pairs,
    )
    masks.apply_static_masks(records, static)
    saturated = masks.build_saturation_masks(records, pairs)
    centered, _ = masks.build_center_masks(
        saturated, np.array([[0.5, 0.5], [0.75, 0.5]]), c['processing'], pairs,
    )
    final = masks.build_final_masks(static, saturated, centered, pairs)
    masks.apply_final_masks(records, final)
    for pair_index, (pair_id, pair) in enumerate(pairs.items()):
        q = np.linspace(.01, .08, 8) + pair_index * 1e-6
        chi = np.arange(-179.5, 180, 1.)
        energy = records[pair[0]]['data'].energy.values
        cap = xr.DataArray([.06, .05, .04], dims='energy', coords={'energy': energy})
        for pol in pair:
            record = records[pol]
            record['_approved'] = {'x': .5, 'y': .5}
            record['beam_center_approval'] = {'pass_id': 'synthetic-pass'}
            provenance = {
                'beam_center_x': .5, 'beam_center_y': .5,
                'mask_pair_id': pair_id, 'mask_id': final[pol].attrs['mask_id'],
                'beam_center_pass_id': 'synthetic-pass',
            }
            values = np.broadcast_to(
                10 + np.cos(np.deg2rad(2 * (chi[None, :, None] - pol))),
                (len(energy), len(chi), len(q)),
            ).copy()
            uncapped = xr.DataArray(values, dims=('energy', 'chi', 'q'),
                                   coords={'energy': energy, 'chi': chi, 'q': q})
            clean = uncapped.where(uncapped.q <= cap)
            record['integration'] = {
                **provenance, 'remeshed': uncapped, 'clean': clean,
                'clean_uncapped_q': uncapped, 'source_fraction': xr.ones_like(uncapped),
                'postmask': xr.zeros_like(uncapped, dtype=np.int8), 'q_cap': cap,
            }
            record['dark_integration'] = {
                **provenance, 'darks_remeshed': uncapped / 100,
                'darks_clean': clean / 100, 'dark_frame_index': 0,
            }
            record['darks'] = xr.DataArray(
                np.arange(12, dtype=np.uint32).reshape(2, 2, 3), dims=('time', 'pix_y', 'pix_x'))
            record['normalization'] = normalization.normalize_scan(record, c['processing'])
            record['flux_correction'] = normalization.flux_correct_scan(record, c['processing'])
            record['intensity'] = intensity.calculate_ring_mean_intensities(record, c['processing'])
    sectors = anisotropy.calculate_sector_anisotropies(c, records)
    pointwise = anisotropy.build_pointwise_anisotropy_maps(c, records)
    return dict(
        configuration=c, scans=records, sector_anisotropy_results=sectors,
        pointwise_anisotropy_results=pointwise,
        chi_averaged_pointwise_results=anisotropy.average_pointwise_anisotropy(pointwise),
        sector_averaged_anisotropy_results=anisotropy.average_sector_anisotropy(sectors),
        global_intensity_ratio_results=anisotropy.calculate_global_intensity_ratio(pointwise),
    )


class DownstreamTests(unittest.TestCase):
    def test_pair_masks_and_scientific_results_match_legacy_layout(self):
        for pols in ((0, 90), (45, 135), (0, 90, 45, 135)):
            mixed = product_fixture(pols)
            separate = product_fixture(pols, 'separate_runs')
            for pair in config.get_mask_pairs(mixed['configuration']).values():
                np.testing.assert_array_equal(
                    mixed['scans'][pair[0]]['final_mask'], mixed['scans'][pair[1]]['final_mask'])
            for pol in pols:
                for section in ('integration', 'normalization', 'flux_correction', 'intensity'):
                    for key, value in mixed['scans'][pol][section].items():
                        if isinstance(value, xr.DataArray):
                            xr.testing.assert_equal(value, separate['scans'][pol][section][key])
            for section in ('pointwise_anisotropy_results', 'chi_averaged_pointwise_results',
                            'sector_averaged_anisotropy_results', 'global_intensity_ratio_results'):
                for mode, pairs in mixed[section].items():
                    for pair_id, result in pairs.items():
                        for key, value in result.items():
                            if isinstance(value, xr.DataArray):
                                xr.testing.assert_equal(value, separate[section][mode][pair_id][key])
            for mode, sectors in mixed['sector_anisotropy_results'].items():
                for sector_id, result in sectors.items():
                    xr.testing.assert_equal(result['anisotropy'], separate['sector_anisotropy_results'][mode][sector_id]['anisotropy'])

    def test_real_zarr_roundtrip_does_not_overwrite_same_run_polarizations(self):
        from numcodecs import blosc
        blosc.use_threads = False
        for pols in ((0, 90), (45, 135), (0, 90, 45, 135)):
            arguments = product_fixture(pols)
            with patch.object(outputs, 'get_approved_beam_center', side_effect=lambda record, _: record['_approved']):
                datasets = outputs.build_pair_workup_datasets(**arguments)
                self.assertEqual(len(datasets), 11 * (len(pols) // 2))
                for pair_id, pair in config.get_mask_pairs(arguments['configuration']).items():
                    for pol in pair:
                        path = f'pairs/{pair_id}/scans/42/polarizations/{pol}/data'
                        actual = datasets[path]
                        self.assertEqual(actual['data'].dtype, np.int64)
                        self.assertIn('source_frame_index', actual.coords)
                        self.assertEqual(actual.attrs['polarization_layout'], 'within_run')
                        np.testing.assert_array_equal(actual['data'], arguments['scans'][pol]['data'])
                        dark_path = path[:-4] + 'darks'
                        self.assertEqual(datasets[dark_path].attrs['dark_frame_scope'], 'complete_source_run')
                with tempfile.TemporaryDirectory(prefix='rsoxs-synthetic-roundtrip-') as directory:
                    summary = outputs.save_workup_results(
                        **arguments, aggregate_anisotropy_results=None, output_root=directory)
                    self.assertEqual(set(summary['groups']), set(datasets))
                    with summary['manifest'].open() as stream:
                        manifest = json.load(stream)
                    self.assertEqual(manifest['schema_version'], 7)
                    self.assertEqual(manifest['configuration']['scan_set']['polarization_layout'], 'within_run')
                    for group, dataset in datasets.items():
                        with xr.open_zarr(summary['zarr_store'], group=group, consolidated=True, chunks=None) as saved:
                            xr.testing.assert_identical(outputs._prepare_zarr_dataset(dataset), saved.load())
                    with self.assertRaises(FileExistsError):
                        outputs.save_workup_results(**arguments, aggregate_anisotropy_results=None, output_root=directory)
            legacy = product_fixture(pols, 'separate_runs')
            with patch.object(outputs, 'get_approved_beam_center', side_effect=lambda record, _: record['_approved']):
                datasets = outputs.build_pair_workup_datasets(**legacy)
            for row in legacy['configuration']['scans']:
                pair_id = next(pair_id for pair_id, pair in config.get_mask_pairs(legacy['configuration']).items()
                               if row['polarization_deg'] in pair)
                self.assertIn(f"pairs/{pair_id}/scans/{row['scan_id']}/data", datasets)

    def test_sector_overlay_rows_use_correct_polarizations_with_shared_ids(self):
        import matplotlib.pyplot as plt
        arguments = product_fixture((0, 90, 45, 135))
        figures = []
        try:
            with patch.object(plt, 'show', side_effect=lambda: figures.append(plt.gcf())):
                plotting.plot_anisotropy_sector_overlays(
                    arguments['scans'], arguments['configuration'], arguments['sector_anisotropy_results'])
            self.assertEqual(len(figures), 2)
            for figure, pair in zip(figures, ((0, 90), (45, 135))):
                for row, pol in enumerate(pair):
                    for axis in figure.axes[row * 4:(row + 1) * 4]:
                        self.assertIn(f'polarization={pol}°', axis.get_title())
        finally:
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
