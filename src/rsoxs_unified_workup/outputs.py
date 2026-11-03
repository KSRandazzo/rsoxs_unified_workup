import json
import platform
import shutil
import subprocess
from datetime import date, datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from rsoxs_unified_workup.config import PROJECT_ROOT


def _polarization_value(value):
    """Keep integer-valued polarizations readable."""
    value = float(value)

    if value.is_integer():
        return int(value)

    return value


def _use_reference_grid(
    data,
    reference,
    tolerance=1e-3,
):
    """Safely assign a positional product to its reference grid."""
    corrected = data

    for dimension in ("energy", "chi", "q"):
        if dimension not in data.dims:
            continue

        if data.sizes[dimension] != reference.sizes[dimension]:
            raise ValueError(
                f"{dimension} sizes differ: "
                f"{data.sizes[dimension]} and "
                f"{reference.sizes[dimension]}"
            )

        difference = np.max(
            np.abs(
                data[dimension].values
                - reference[dimension].values
            )
        )

        if difference > tolerance:
            raise ValueError(
                f"{dimension} coordinates differ by "
                f"{difference:g}, above tolerance "
                f"{tolerance:g}"
            )

        corrected = corrected.assign_coords(
            {
                dimension: reference[dimension],
            }
        )

    return corrected


def _stack_by_polarization(
    scans,
    section,
    key,
    name,
    align_to_clean=False,
):
    """Stack one stored scan product across polarizations."""
    arrays = []

    for polarization in sorted(scans, key=float):
        scan_record = scans[polarization]
        array = scan_record[section][key]

        if align_to_clean:
            array = _use_reference_grid(
                array,
                scan_record["integration"]["clean"],
            )

        arrays.append(
            array.expand_dims(
                polarization=[
                    _polarization_value(polarization)
                ]
            )
        )

    combined = xr.concat(
        arrays,
        dim="polarization",
    )
    combined.name = name

    return combined


def build_scan_products_dataset(scans):
    """Collect final and diagnostic products from nested scan records."""
    if not scans:
        raise ValueError("No scans supplied")

    products = [
        (
            "integration",
            (
                "remeshed",
                "source_fraction",
                "postmask",
                "clean_uncapped_q",
                "clean",
                "q_cap",
            ),
        ),
        (
            "dark_integration",
            (
                "darks_remeshed",
                "darks_clean",
            ),
        ),
        (
            "normalization",
            (
                "full_ring_mean",
                "half_ring_mean",
                "full_ring_normalized",
                "half_ring_normalized",
            ),
        ),
    ]

    data_vars = {}

    for section, keys in products:
        for key in keys:
            data_vars[key] = _stack_by_polarization(
                scans=scans,
                section=section,
                key=key,
                name=key,
                align_to_clean=key in (
                    "source_fraction",
                    "postmask",
                ),
            )

    has_flux = [
        bool(scan_record["flux_correction"])
        for scan_record in scans.values()
    ]

    if any(has_flux):
        if not all(has_flux):
            raise ValueError(
                "Flux correction is present for only some scans. "
                "Rerun the flux-correction cell for the whole set."
            )

        data_vars["au_mesh_current"] = _stack_by_polarization(
            scans=scans,
            section="flux_correction",
            key="flux",
            name="au_mesh_current",
        )
        data_vars["flux_corrected"] = _stack_by_polarization(
            scans=scans,
            section="flux_correction",
            key="flux_corrected",
            name="flux_corrected",
        )

    dataset = xr.Dataset(data_vars)
    dataset.attrs["description"] = (
        "Integrated, dark, normalized, and "
        "flux-corrected RSoXS scan products"
    )

    return dataset


def build_sector_anisotropy_dataset(
    anisotropy_results,
):
    """Collect every configured sector calculation."""
    modes = list(anisotropy_results)

    if not modes:
        raise ValueError("No anisotropy results supplied")

    sector_ids = list(anisotropy_results[modes[0]])

    for mode in modes:
        if list(anisotropy_results[mode]) != sector_ids:
            raise ValueError(
                "Sector order differs between "
                "normalization modes"
            )

    def stack_variable(key, name):
        mode_arrays = []

        for mode in modes:
            sector_arrays = [
                anisotropy_results[mode][sector_id][
                    key
                ].expand_dims(sector=[sector_id])
                for sector_id in sector_ids
            ]
            mode_array = xr.concat(
                sector_arrays,
                dim="sector",
            ).expand_dims(
                normalization_mode=[mode]
            )
            mode_arrays.append(mode_array)

        combined = xr.concat(
            mode_arrays,
            dim="normalization_mode",
        )
        combined.name = name

        return combined

    first_mode = anisotropy_results[modes[0]]
    dataset = xr.Dataset(
        {
            "parallel_intensity": stack_variable(
                "parallel_intensity",
                "parallel_intensity",
            ),
            "perpendicular_intensity": stack_variable(
                "perpendicular_intensity",
                "perpendicular_intensity",
            ),
            "anisotropy": stack_variable(
                "anisotropy",
                "anisotropy",
            ),
        }
    )

    dataset = dataset.assign_coords(
        pair_id=(
            "sector",
            [
                str(first_mode[sector_id]["pair_id"])
                for sector_id in sector_ids
            ],
        ),
        sector_label=(
            "sector",
            [
                str(first_mode[sector_id]["label"])
                for sector_id in sector_ids
            ],
        ),
        chi_center_deg=(
            "sector",
            [
                float(
                    first_mode[sector_id][
                        "chi_center_deg"
                    ]
                )
                for sector_id in sector_ids
            ],
        ),
        chi_halfwidth_deg=(
            "sector",
            [
                float(
                    first_mode[sector_id][
                        "chi_halfwidth_deg"
                    ]
                )
                for sector_id in sector_ids
            ],
        ),
        parallel_polarization=(
            "sector",
            [
                float(
                    first_mode[sector_id][
                        "parallel_polarization"
                    ]
                )
                for sector_id in sector_ids
            ],
        ),
        perpendicular_polarization=(
            "sector",
            [
                float(
                    first_mode[sector_id][
                        "perpendicular_polarization"
                    ]
                )
                for sector_id in sector_ids
            ],
        ),
    )
    dataset.attrs["description"] = (
        "Configured sector-level anisotropy results"
    )

    return dataset


def build_aggregate_anisotropy_dataset(
    aggregate_results,
):
    """Collect all three aggregate anisotropy methods."""
    method_descriptions = {
        "method_1_global_intensity_ratio": (
            "Global intensity ratio"
        ),
        "method_2_pointwise_anisotropy": (
            "Mean of pointwise anisotropy"
        ),
        "method_3_sector_anisotropy_mean": (
            "Mean of sector anisotropies"
        ),
    }
    modes = list(aggregate_results)
    methods = list(method_descriptions)

    mode_arrays = []

    for mode in modes:
        missing = [
            method
            for method in methods
            if method not in aggregate_results[mode]
        ]

        if missing:
            raise ValueError(
                f"Missing aggregate methods for "
                f"{mode!r}: {missing}"
            )

        method_arrays = [
            aggregate_results[mode][method].expand_dims(
                aggregation_method=[method]
            )
            for method in methods
        ]
        mode_array = xr.concat(
            method_arrays,
            dim="aggregation_method",
        ).expand_dims(
            normalization_mode=[mode]
        )
        mode_arrays.append(mode_array)

    anisotropy = xr.concat(
        mode_arrays,
        dim="normalization_mode",
    )
    anisotropy.name = "anisotropy"

    dataset = anisotropy.to_dataset()
    dataset = dataset.assign_coords(
        method_description=(
            "aggregation_method",
            [
                method_descriptions[method]
                for method in methods
            ],
        )
    )
    dataset.attrs["description"] = (
        "Three alternative aggregate anisotropy methods"
    )

    return dataset


def build_pointwise_anisotropy_dataset(
    pointwise_results,
):
    """Collect stitched chi-resolved anisotropy maps."""
    modes = list(pointwise_results)

    if not modes:
        raise ValueError(
            "No pointwise anisotropy results supplied"
        )

    pair_ids = list(pointwise_results[modes[0]])

    for mode in modes:
        if list(pointwise_results[mode]) != pair_ids:
            raise ValueError(
                "Polarization-pair order differs between "
                "normalization modes"
            )

    def stack_variable(key, name):
        mode_arrays = []

        for mode in modes:
            pair_arrays = [
                pointwise_results[mode][pair_id][
                    key
                ].expand_dims(
                    polarization_pair=[pair_id]
                )
                for pair_id in pair_ids
            ]
            pair_array = xr.concat(
                pair_arrays,
                dim="polarization_pair",
            ).expand_dims(
                normalization_mode=[mode]
            )
            mode_arrays.append(pair_array)

        combined = xr.concat(
            mode_arrays,
            dim="normalization_mode",
        )
        combined.name = name

        return combined

    dataset = xr.Dataset(
        {
            "parallel_intensity": stack_variable(
                "parallel",
                "parallel_intensity",
            ),
            "perpendicular_intensity": stack_variable(
                "perpendicular",
                "perpendicular_intensity",
            ),
            "intensity_difference": stack_variable(
                "difference",
                "intensity_difference",
            ),
            "anisotropy": stack_variable(
                "anisotropy",
                "anisotropy",
            ),
        }
    )

    first_mode = pointwise_results[modes[0]]
    dataset = dataset.assign_coords(
        sector_ids=(
            "polarization_pair",
            [
                json.dumps(
                    first_mode[pair_id]["sector_ids"]
                )
                for pair_id in pair_ids
            ],
        )
    )
    dataset.attrs["description"] = (
        "Stitched chi-resolved parallel, perpendicular, "
        "difference, and pointwise anisotropy maps"
    )

    return dataset


def _json_ready(value):
    """Convert scientific values to JSON-safe values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]

    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())

    if isinstance(value, np.generic):
        return _json_ready(value.item())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def _git_value(*arguments):
    """Run a read-only Git query."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        return None

    return completed.stdout.strip()


def build_manifest(configuration, datasets):
    """Build provenance metadata for one saved workup."""
    package_names = [
        "rsoxs-unified-workup",
        "numpy",
        "xarray",
        "pandas",
        "PyHyperScattering",
        "zarr",
        "scipy",
    ]
    package_versions = {}

    for package_name in package_names:
        try:
            package_versions[package_name] = version(
                package_name
            )
        except Exception:
            package_versions[package_name] = None

    git_status = _git_value("status", "--porcelain")

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "scan_set_id": configuration[
            "scan_set"
        ]["scan_set_id"],
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_working_tree_clean": (
            git_status == ""
            if git_status is not None
            else None
        ),
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "zarr_store": "workup.zarr",
        "zarr_groups": {
            group_name: list(dataset.data_vars)
            for group_name, dataset in datasets.items()
        },
        "configuration": _json_ready(configuration),
    }


def _zarr_encoding(dataset):
    """Choose practical chunks for saved arrays."""
    preferred_chunks = {
        "polarization": 1,
        "normalization_mode": 1,
        "polarization_pair": 1,
        "sector": 1,
        "aggregation_method": 1,
        "energy": 1,
        "chi": 180,
        "q": 250,
    }
    encoding = {}

    for name, variable in dataset.data_vars.items():
        encoding[name] = {
            "chunks": tuple(
                min(
                    variable.sizes[dimension],
                    preferred_chunks.get(
                        dimension,
                        variable.sizes[dimension],
                    ),
                )
                for dimension in variable.dims
            )
        }

    return encoding


def _prepare_zarr_dataset(dataset):
    """Convert metadata to JSON-safe values for Zarr."""
    prepared = dataset.copy(deep=False)
    prepared.attrs = _json_ready(dataset.attrs)

    for name in prepared.variables:
        prepared[name].attrs = _json_ready(
            dataset[name].attrs
        )

    return prepared


def save_workup_results(
    configuration,
    scans,
    anisotropy_results,
    aggregate_anisotropy_results,
    pointwise_anisotropy_results,
    output_root=None,
    overwrite=False,
):
    """Save one complete workup as Zarr plus JSON metadata."""
    datasets = {
        "scan_products": build_scan_products_dataset(scans),
        "sector_anisotropy": (
            build_sector_anisotropy_dataset(
                anisotropy_results
            )
        ),
        "aggregate_anisotropy": (
            build_aggregate_anisotropy_dataset(
                aggregate_anisotropy_results
            )
        ),
        "pointwise_anisotropy": (
            build_pointwise_anisotropy_dataset(
                pointwise_anisotropy_results
            )
        ),
    }

    scan_set_id = str(
        configuration["scan_set"]["scan_set_id"]
    )

    if output_root is None:
        output_root = PROJECT_ROOT / "results"
    else:
        output_root = Path(output_root)

        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root

    output_directory = output_root / scan_set_id
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    store_path = output_directory / "workup.zarr"
    manifest_path = output_directory / "manifest.json"
    temporary_store = (
        output_directory / ".workup.zarr.incomplete"
    )
    temporary_manifest = (
        output_directory / ".manifest.json.incomplete"
    )

    existing_outputs = (
        store_path.exists()
        or manifest_path.exists()
    )
    if existing_outputs and not overwrite:
        raise FileExistsError(
            f"Results already exist in "
            f"{output_directory}. Set overwrite=True "
            f"to replace generated results."
        )

    if temporary_store.exists():
        shutil.rmtree(temporary_store)

    if temporary_manifest.exists():
        temporary_manifest.unlink()

    try:
        for index, (
            group_name,
            dataset,
        ) in enumerate(datasets.items()):
            dataset = _prepare_zarr_dataset(dataset)
            dataset.to_zarr(
                temporary_store,
                group=group_name,
                mode="w" if index == 0 else "a",
                consolidated=False,
                encoding=_zarr_encoding(dataset),
            )

        zarr.consolidate_metadata(temporary_store)

        manifest = build_manifest(
            configuration,
            datasets,
        )
        temporary_manifest.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )

        if store_path.exists():
            shutil.rmtree(store_path)

        temporary_store.replace(store_path)
        temporary_manifest.replace(manifest_path)

    except Exception:
        if temporary_store.exists():
            shutil.rmtree(temporary_store)

        if temporary_manifest.exists():
            temporary_manifest.unlink()

        raise

    return {
        "output_directory": output_directory,
        "zarr_store": store_path,
        "manifest": manifest_path,
        "groups": list(datasets),
    }