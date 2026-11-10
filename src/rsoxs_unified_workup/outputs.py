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
import pandas as pd

from rsoxs_unified_workup.config import (
    PROJECT_ROOT,
    get_mask_pairs,
)
from rsoxs_unified_workup.beam_center import (
    get_approved_beam_center,
)


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


def build_detector_image_dataset(scan_record, key):
    """Preserve one scan's loaded detector data or complete dark stack."""
    if key not in ("data", "darks"):
        raise ValueError("Detector export key must be 'data' or 'darks'")

    array = scan_record[key]
    if not isinstance(array, xr.DataArray):
        raise TypeError(
            f"Scan {scan_record['scan_id']}: {key} must be a DataArray"
        )

    original_name = array.name
    array = array.copy(deep=False)
    multi_indexes = {}

    # Zarr cannot directly store a pandas MultiIndex. Preserve its
    # levels as coordinates without sorting, unstacking, or padding.
    for dimension in array.dims:
        index = array.indexes.get(dimension)
        if isinstance(index, pd.MultiIndex):
            multi_indexes[dimension] = list(index.names)
            array = array.reset_index(dimension)

    dataset = array.rename(key).to_dataset()
    dataset.attrs.update({
        "description": (
            "Loaded detector images before workup masking and integration"
            if key == "data"
            else "All loaded detector dark frames"
        ),
        "scan_id": int(scan_record["scan_id"]),
        "polarization_deg": float(scan_record["polarization_deg"]),
        "polarization_pair_id": (
            scan_record["final_mask"].attrs["mask_pair_id"]
        ),
        "source_scan_key": key,
        "original_array_name": original_name,
        "original_multi_indexes": multi_indexes,
    })

    if key == "darks":
        dataset.attrs["integrated_dark_frame_index"] = (
            scan_record["dark_integration"]["dark_frame_index"]
        )

    return dataset


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


def _check_export_grid(value, reference, label, dimensions):
    """Require matching native coordinates without resampling."""
    if isinstance(value, (dict, xr.Dataset)):
        for name, item in value.items():
            _check_export_grid(
                item,
                reference,
                f"{label}/{name}",
                dimensions,
            )

    elif isinstance(value, xr.DataArray):
        for dimension in dimensions:
            if dimension in value.dims and not np.array_equal(
                value[dimension].values,
                reference[dimension].values,
            ):
                raise ValueError(
                    f"{label} has different {dimension} coordinates. "
                    "Recalculate this product on its pair's native grid."
                )


def build_pair_workup_datasets(
    configuration,
    scans,
    anisotropy_results,
    pointwise_anisotropy_results,
):
    """Build separate export datasets for each pair's native q grid."""
    mask_pairs = get_mask_pairs(configuration)
    processing = configuration["processing"]

    if set(scans) != {
        polarization
        for pair in mask_pairs.values()
        for polarization in pair
    }:
        raise ValueError(
            "Loaded scans do not match the configured pairs"
        )

    if (
        not anisotropy_results
        or set(anisotropy_results)
        != set(pointwise_anisotropy_results)
    ):
        raise ValueError(
            "Sector and pointwise normalization modes do not match"
        )

    for mode in anisotropy_results:
        if set(pointwise_anisotropy_results[mode]) != set(mask_pairs):
            raise ValueError(
                f"Pointwise pairs do not match configuration in {mode}"
            )

        if any(
            sector["pair_id"] not in mask_pairs
            for sector in anisotropy_results[mode].values()
        ):
            raise ValueError(f"Unknown sector pair in {mode}")

    datasets = {}

    for pair_id, pair in mask_pairs.items():
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or pair_id in (".", "..")
            or "/" in pair_id
            or "\\" in pair_id
        ):
            raise ValueError(
                f"Pair ID is not a safe Zarr group name: {pair_id!r}"
            )

        pair_scans = {
            polarization: scans[polarization]
            for polarization in pair
        }
        reference = pair_scans[pair[0]]["integration"]["remeshed"]
        centers = []

        for polarization, record in pair_scans.items():
            if float(record["polarization_deg"]) != float(polarization):
                raise ValueError(
                    "Scan dictionary keys do not match "
                    "polarization metadata"
                )

            mask = record["final_mask"]

            if (
                mask.attrs.get("mask_pair_id") != pair_id
                or set(
                    mask.attrs.get("mask_pair_polarizations", [])
                ) != set(pair)
            ):
                raise ValueError(
                    f"Scan {record['scan_id']} has the wrong mask pair"
                )

            center = get_approved_beam_center(record, processing)
            centers.append(center)

            expected = {
                "beam_center_x": center["x"],
                "beam_center_y": center["y"],
                "mask_pair_id": pair_id,
                "mask_id": mask.attrs["mask_id"],
                "beam_center_pass_id": (
                    record["beam_center_approval"]["pass_id"]
                ),
            }

            for section in ("integration", "dark_integration"):
                result = record[section]

                if any(
                    result.get(key) != value
                    for key, value in expected.items()
                ):
                    raise ValueError(
                        f"Scan {record['scan_id']} has stale or missing "
                        f"{section} provenance. "
                        "Recalculate it before exporting."
                    )

            for section in (
                "integration",
                "dark_integration",
                "normalization",
                "flux_correction",
            ):
                _check_export_grid(
                    record[section],
                    reference,
                    f"{pair_id}/{polarization}/{section}",
                    ("energy", "chi", "q"),
                )

        if centers[0] != centers[1]:
            raise ValueError(
                f"The two scans in pair {pair_id!r} "
                "use different centers"
            )

        pair_sectors = {
            mode: {
                sector_id: sector
                for sector_id, sector in results.items()
                if sector["pair_id"] == pair_id
            }
            for mode, results in anisotropy_results.items()
        }

        if any(
            not sectors
            for sectors in pair_sectors.values()
        ):
            raise ValueError(
                f"Sector results are missing for pair {pair_id!r}"
            )

        pair_pointwise = {
            mode: {pair_id: results[pair_id]}
            for mode, results in pointwise_anisotropy_results.items()
        }

        _check_export_grid(
            pair_sectors,
            reference,
            f"{pair_id}/sectors",
            ("energy", "q"),
        )
        _check_export_grid(
            pair_pointwise,
            reference,
            f"{pair_id}/pointwise",
            ("energy", "q"),
        )

        products = build_scan_products_dataset(pair_scans)
        polarizations = products["polarization"].values

        products = products.assign_coords(
            scan_id=(
                "polarization",
                [
                    int(pair_scans[pol]["scan_id"])
                    for pol in polarizations
                ],
            ),
            beam_center_x=(
                "polarization",
                [centers[0]["x"]] * len(pair),
            ),
            beam_center_y=(
                "polarization",
                [centers[0]["y"]] * len(pair),
            ),
            mask_id=(
                "polarization",
                [
                    pair_scans[pol]["final_mask"].attrs["mask_id"]
                    for pol in polarizations
                ],
            ),
            beam_center_pass_id=(
                "polarization",
                [
                    pair_scans[pol]["beam_center_approval"]["pass_id"]
                    for pol in polarizations
                ],
            ),
        )

        for coordinate in ("beam_center_x", "beam_center_y"):
            products[coordinate].attrs["units"] = "pixel"

        pair_datasets = {
            "scan_products": products,
            "sector_anisotropy": (
                build_sector_anisotropy_dataset(pair_sectors)
            ),
            "pointwise_anisotropy": (
                build_pointwise_anisotropy_dataset(pair_pointwise)
            ),
        }

        for name, dataset in pair_datasets.items():
            _check_export_grid(
                dataset,
                reference,
                f"{pair_id}/{name}",
                ("energy", "q"),
            )

            dataset.attrs.update({
                "polarization_pair_id": pair_id,
                "polarizations": list(pair),
                "beam_center": dict(centers[0]),
                "q_grid_scope": "polarization_pair",
            })

            datasets[f"pairs/{pair_id}/{name}"] = dataset

        # Preserve each scan separately: no alignment across scans'
        # detector coordinates, timestamps, or dark-frame counts.
        for record in pair_scans.values():
            scan_id = int(record["scan_id"])
            for key in ("data", "darks"):
                group_name = f"pairs/{pair_id}/scans/{scan_id}/{key}"
                datasets[group_name] = build_detector_image_dataset(
                    record,
                    key,
                )

    return datasets


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
        "schema_version": 3,
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
        "system": 1,
        "time": 1,
        "pix_y": 512,
        "pix_x": 512,
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
    if aggregate_anisotropy_results is not None:
        raise ValueError(
            "Cross-pair aggregate export is paused. "
            "Pass aggregate_anisotropy_results=None."
        )

    datasets = build_pair_workup_datasets(
        configuration=configuration,
        scans=scans,
        anisotropy_results=anisotropy_results,
        pointwise_anisotropy_results=pointwise_anisotropy_results,
    )

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


def print_workup_export_tree(
    save_summary,
    *,
    include_array_attributes=False,
):
    """Print a saved export's structure without loading its data."""
    manifest_path = Path(save_summary["manifest"])
    store_path = Path(save_summary["zarr_store"])

    with manifest_path.open() as file:
        manifest = json.load(file)

    store = zarr.open_consolidated(
        str(store_path),
        mode="r",
    )

    def metadata_node(name, value):
        if isinstance(value, dict):
            return (
                f"{name} [dictionary]",
                [
                    metadata_node(str(key), item)
                    for key, item in value.items()
                ],
            )

        if isinstance(value, list):
            if any(
                isinstance(item, (dict, list))
                for item in value
            ):
                return (
                    f"{name} [list: {len(value)} entries]",
                    [
                        metadata_node(f"[{index}]", item)
                        for index, item in enumerate(value)
                    ],
                )

            if len(value) > 8:
                preview = json.dumps(
                    value[:3],
                    ensure_ascii=False,
                )
                return (
                    f"{name}: {preview[:-1]}, ...] "
                    f"({len(value)} entries)",
                    [],
                )

        text = json.dumps(value, ensure_ascii=False)
        if len(text) > 160:
            text = text[:157] + "..."

        return (f"{name}: {text}", [])

    def array_node(name, array):
        dimensions = tuple(
            array.attrs.get("_ARRAY_DIMENSIONS", [])
        )
        children = []

        if include_array_attributes:
            attributes = dict(array.attrs)
            attributes.pop("_ARRAY_DIMENSIONS", None)

            if attributes:
                children.append(
                    metadata_node("attributes", attributes)
                )

        return (
            f"{name}: dims={dimensions}, "
            f"shape={array.shape}, dtype={array.dtype}",
            children,
        )

    def group_node(group_name, group):
        array_names = sorted(group.array_keys())
        variable_names = manifest["zarr_groups"].get(group.path)
        children = []

        if variable_names is not None:
            variable_names = set(variable_names)
            label = f"{group_name}/ [xarray Dataset]"

            children.extend([
                (
                    "data variables",
                    [
                        array_node(name, group[name])
                        for name in array_names
                        if name in variable_names
                    ],
                ),
                (
                    "coordinates",
                    [
                        array_node(name, group[name])
                        for name in array_names
                        if name not in variable_names
                    ],
                ),
            ])

        else:
            label = f"{group_name}/ [Zarr group]"

            if array_names:
                children.append((
                    "arrays",
                    [
                        array_node(name, group[name])
                        for name in array_names
                    ],
                ))

        if group.attrs:
            children.append(
                metadata_node("attributes", dict(group.attrs))
            )

        children.extend(
            group_node(name, group[name])
            for name in sorted(group.group_keys())
        )

        return label, children

    group_nodes = [
        group_node(name, store[name])
        for name in sorted(store.group_keys())
    ]

    nodes = [
        metadata_node(manifest_path.name, manifest),
        (f"{store_path.name}/ [Zarr store]", group_nodes),
    ]

    def print_nodes(items, prefix=""):
        for index, (label, children) in enumerate(items):
            last = index == len(items) - 1
            connector = "└── " if last else "├── "
            print(prefix + connector + label)

            child_prefix = "    " if last else "│   "
            print_nodes(children, prefix + child_prefix)

    print(f"{store_path.parent}/")
    print_nodes(nodes)