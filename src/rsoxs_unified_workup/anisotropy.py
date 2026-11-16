import numpy as np
import xarray as xr

from rsoxs_unified_workup.normalization import (
    select_chi_region,
)


NORMALIZATION_MODES = {
    "full_ring": "full_ring_normalized",
    "half_ring": "half_ring_normalized",
}


def _polarization_value(value):
    """Return integer-valued polarizations as integers."""
    value = float(value)

    if value.is_integer():
        return int(value)

    return value


def axial_angle_distance(angle_1, angle_2):
    """Return the smallest separation between unoriented axes."""
    return abs(
        ((float(angle_1) - float(angle_2) + 90) % 180)
        - 90
    )


def pair_polarizations(configuration, pair_record):
    """Resolve a configured scan pair to its polarizations."""
    polarization_by_scan_id = {
        int(scan["scan_id"]): _polarization_value(
            scan["polarization_deg"]
        )
        for scan in configuration["scans"]
    }

    scan_id_1 = int(pair_record["scan_id_1"])
    scan_id_2 = int(pair_record["scan_id_2"])

    try:
        polarization_1 = polarization_by_scan_id[scan_id_1]
        polarization_2 = polarization_by_scan_id[scan_id_2]
    except KeyError as error:
        raise ValueError(
            f"Polarization pair {pair_record['pair_id']!r} "
            f"refers to a scan outside this scan set"
        ) from error

    separation = axial_angle_distance(
        polarization_1,
        polarization_2,
    )
    if not np.isclose(separation, 90):
        raise ValueError(
            f"Polarization pair {pair_record['pair_id']!r} "
            f"is separated by {separation:g}°, not 90°"
        )

    return polarization_1, polarization_2


def assign_parallel_perpendicular(
    sector_center,
    polarization_1,
    polarization_2,
):
    """Assign pair members relative to one sector direction."""
    distance_1 = axial_angle_distance(
        sector_center,
        polarization_1,
    )
    distance_2 = axial_angle_distance(
        sector_center,
        polarization_2,
    )

    if np.isclose(distance_1, distance_2):
        raise ValueError(
            f"Sector center {sector_center}° is equally "
            f"aligned with polarizations {polarization_1}° "
            f"and {polarization_2}°"
        )

    if distance_1 < distance_2:
        return polarization_1, polarization_2

    return polarization_2, polarization_1


def calculate_sector_anisotropy(
    scans,
    normalization_key,
    sector,
    parallel_polarization,
    perpendicular_polarization,
):
    """Calculate anisotropy for one normalization and sector."""
    parallel_slice = select_chi_region(
        scans[parallel_polarization]["normalization"][normalization_key],
        center_degrees=sector["chi_center_deg"],
        halfwidth_degrees=sector[
            "chi_halfwidth_deg"
        ],
    )
    perpendicular_slice = select_chi_region(
        scans[perpendicular_polarization]["normalization"][normalization_key],
        center_degrees=sector["chi_center_deg"],
        halfwidth_degrees=sector[
            "chi_halfwidth_deg"
        ],
    )

    parallel_intensity = parallel_slice.mean(
        "chi",
        skipna=True,
    )
    perpendicular_intensity = (
        perpendicular_slice.mean(
            "chi",
            skipna=True,
        )
    )
    denominator = (
        parallel_intensity
        + perpendicular_intensity
    )
    anisotropy = (
        parallel_intensity
        - perpendicular_intensity
    ) / denominator.where(denominator != 0)
    anisotropy.name = "anisotropy"

    return {
        "parallel_polarization": parallel_polarization,
        "perpendicular_polarization": (
            perpendicular_polarization
        ),
        "parallel_slice": parallel_slice,
        "perpendicular_slice": perpendicular_slice,
        "parallel_intensity": parallel_intensity,
        "perpendicular_intensity": (
            perpendicular_intensity
        ),
        "anisotropy": anisotropy,
    }


def calculate_sector_anisotropies(
    configuration,
    scans,
):
    """Calculate every configured sector for both normalizations."""
    pairs_by_id = {
        pair["pair_id"]: pair
        for pair in configuration["polarization_pairs"]
    }
    results = {
        mode: {}
        for mode in NORMALIZATION_MODES
    }

    for sector in configuration["anisotropy_sectors"]:
        pair_id = sector["pair_id"]

        if pair_id not in pairs_by_id:
            raise ValueError(
                f"Sector {sector['sector_id']!r} refers "
                f"to unknown pair {pair_id!r}"
            )

        polarization_1, polarization_2 = (
            pair_polarizations(
                configuration,
                pairs_by_id[pair_id],
            )
        )
        parallel, perpendicular = (
            assign_parallel_perpendicular(
                sector_center=sector["chi_center_deg"],
                polarization_1=polarization_1,
                polarization_2=polarization_2,
            )
        )

        for mode, normalization_key in NORMALIZATION_MODES.items():
            calculation = calculate_sector_anisotropy(
                scans=scans,
                normalization_key=normalization_key,
                sector=sector,
                parallel_polarization=parallel,
                perpendicular_polarization=perpendicular,
            )
            results[mode][sector["sector_id"]] = {
                **sector,
                **calculation,
            }

    return results


def _relative_chi_slice(data, center_degrees):
    """Express a sector's chi coordinates relative to its center."""
    relative_chi = (
        (
            data["chi"].values
            - float(center_degrees)
            + 180
        )
        % 360
    ) - 180

    return (
        data.assign_coords(chi=relative_chi)
        .sortby("chi")
        .rename({"chi": "relative_chi"})
    )


def aggregate_one_mode(sector_results):
    """Calculate all three aggregate anisotropy methods."""
    if not sector_results:
        raise ValueError("No sector results were supplied")

    sector_ids = list(sector_results)
    reference = sector_results[sector_ids[0]]

    reference_chi = _relative_chi_slice(
        reference["parallel_slice"],
        reference["chi_center_deg"],
    )["relative_chi"]

    parallel_slices = []
    perpendicular_slices = []
    sector_anisotropies = []

    for sector_id in sector_ids:
        sector = sector_results[sector_id]

        parallel = _relative_chi_slice(
            sector["parallel_slice"],
            sector["chi_center_deg"],
        ).interp(relative_chi=reference_chi)

        perpendicular = _relative_chi_slice(
            sector["perpendicular_slice"],
            sector["chi_center_deg"],
        ).interp(relative_chi=reference_chi)

        parallel_slices.append(parallel)
        perpendicular_slices.append(perpendicular)
        sector_anisotropies.append(
            sector["anisotropy"]
        )

    sector_coordinate = xr.IndexVariable(
        "sector",
        sector_ids,
    )
    parallel_stack = xr.concat(
        parallel_slices,
        dim=sector_coordinate,
    )
    perpendicular_stack = xr.concat(
        perpendicular_slices,
        dim=sector_coordinate,
    )
    anisotropy_stack = xr.concat(
        sector_anisotropies,
        dim=sector_coordinate,
    )

    pooled_parallel = parallel_stack.mean(
        "sector",
        skipna=True,
    ).mean(
        "relative_chi",
        skipna=True,
    )
    pooled_perpendicular = perpendicular_stack.mean(
        "sector",
        skipna=True,
    ).mean(
        "relative_chi",
        skipna=True,
    )
    pooled_denominator = (
        pooled_parallel + pooled_perpendicular
    )
    method_1 = (
        pooled_parallel - pooled_perpendicular
    ) / pooled_denominator.where(
        pooled_denominator != 0
    )

    pointwise_denominator = (
        parallel_stack + perpendicular_stack
    )
    pointwise_anisotropy = (
        parallel_stack - perpendicular_stack
    ) / pointwise_denominator.where(
        pointwise_denominator != 0
    )
    method_2 = pointwise_anisotropy.mean(
        ("sector", "relative_chi"),
        skipna=True,
    )

    method_3 = anisotropy_stack.mean(
        "sector",
        skipna=True,
    )

    method_1.name = "anisotropy_method_1"
    method_2.name = "anisotropy_method_2"
    method_3.name = "anisotropy_method_3"

    return {
        "method_1_global_intensity_ratio": method_1,
        "method_2_pointwise_anisotropy": method_2,
        "method_3_sector_anisotropy_mean": method_3,
        "parallel_stack": parallel_stack,
        "perpendicular_stack": perpendicular_stack,
        "pointwise_anisotropy": pointwise_anisotropy,
        "sector_anisotropy_stack": anisotropy_stack,
    }


def aggregate_anisotropy(sector_anisotropy_results):
    """Calculate all aggregation methods for every mode."""
    return {
        mode: aggregate_one_mode(sector_results)
        for mode, sector_results
        in sector_anisotropy_results.items()
    }


def build_pointwise_anisotropy_maps(configuration, scans):
    """Build stitched pointwise maps from q-capped normalization."""
    pairs_by_id = {
        pair["pair_id"]: pair
        for pair in configuration["polarization_pairs"]
    }
    sectors_by_pair = {}

    for sector in configuration["anisotropy_sectors"]:
        sectors_by_pair.setdefault(
            sector["pair_id"], []
        ).append(sector)

    all_results = {
        mode: {}
        for mode in NORMALIZATION_MODES
    }

    def sector_patch(polarization, normalization_key, sector):
        normalization = scans[polarization].get("normalization", {})

        if normalization_key not in normalization:
            raise ValueError(
                "Q-capped normalization is missing. "
                "Rerun normalization for every scan first."
            )

        return select_chi_region(
            normalization[normalization_key],
            center_degrees=sector["chi_center_deg"],
            halfwidth_degrees=sector["chi_halfwidth_deg"],
        ).transpose("energy", "chi", "q")

    for pair_id, sectors in sectors_by_pair.items():
        if pair_id not in pairs_by_id:
            raise ValueError(
                f"Unknown polarization pair {pair_id!r}"
            )

        polarization_1, polarization_2 = pair_polarizations(
            configuration,
            pairs_by_id[pair_id],
        )
        sectors = sorted(
            sectors,
            key=lambda sector: float(sector["chi_center_deg"]),
        )

        for mode, normalization_key in NORMALIZATION_MODES.items():
            parallel_patches = []
            perpendicular_patches = []

            for sector in sectors:
                parallel_pol, perpendicular_pol = (
                    assign_parallel_perpendicular(
                        sector["chi_center_deg"],
                        polarization_1,
                        polarization_2,
                    )
                )
                parallel_patches.append(
                    sector_patch(
                        parallel_pol,
                        normalization_key,
                        sector,
                    )
                )
                perpendicular_patches.append(
                    sector_patch(
                        perpendicular_pol,
                        normalization_key,
                        sector,
                    )
                )

            parallel = xr.concat(
                parallel_patches,
                dim="chi",
                join="exact",
            ).sortby("chi")
            perpendicular = xr.concat(
                perpendicular_patches,
                dim="chi",
                join="exact",
            ).sortby("chi")

            parallel, perpendicular = xr.align(
                parallel,
                perpendicular,
                join="exact",
                copy=False,
            )

            chi_values = parallel["chi"].values
            if len(np.unique(chi_values)) != len(chi_values):
                raise ValueError(
                    f"Overlapping sectors produced duplicate "
                    f"chi coordinates for pair {pair_id!r}"
                )

            difference = parallel - perpendicular
            denominator = parallel + perpendicular
            pointwise_anisotropy = (
                difference / denominator.where(denominator != 0)
            )

            parallel.name = "parallel_intensity"
            perpendicular.name = "perpendicular_intensity"
            difference.name = "intensity_difference"
            pointwise_anisotropy.name = "pointwise_anisotropy"

            for array in (
                parallel,
                perpendicular,
                difference,
                pointwise_anisotropy,
            ):
                array.attrs = {
                    **array.attrs,
                    "source_product": "clean",
                    "q_cap_applied": True,
                }

            all_results[mode][pair_id] = {
                "parallel": parallel,
                "perpendicular": perpendicular,
                "difference": difference,
                "anisotropy": pointwise_anisotropy,
                "sector_ids": [
                    sector["sector_id"]
                    for sector in sectors
                ],
            }

    return all_results


def average_pointwise_anisotropy(pointwise_results):
    """Average pointwise anisotropy over chi, separately for each pair."""
    averaged_results = {}

    for mode, pair_results in pointwise_results.items():
        averaged_results[mode] = {}

        for pair_id, result in pair_results.items():
            values = result["anisotropy"]

            if set(values.dims) != {"energy", "chi", "q"}:
                raise ValueError(
                    f"{mode}/{pair_id}: expected energy, chi, and q dimensions"
                )
            if values.attrs.get("q_cap_applied") is not True:
                raise ValueError(
                    f"{mode}/{pair_id}: rerun q-capped pointwise anisotropy first"
                )

            values = values.transpose("energy", "chi", "q")
            mean = values.mean("chi", skipna=True, keep_attrs=True)
            count = values.count("chi")
            mean.name = "chi_averaged_pointwise_anisotropy"
            count.name = "valid_chi_count"

            mean.attrs = {
                **mean.attrs,
                "aggregation_method": "mean_pointwise_anisotropy",
                "averaged_dimension": "chi",
                "angular_weighting": "equal weight per non-NaN chi bin",
            }
            count.attrs = {
                "description": "Number of non-NaN pointwise anisotropy bins used in the chi mean",
            }

            averaged_results[mode][pair_id] = {
                "anisotropy": mean,
                "valid_chi_count": count,
            }

    return averaged_results


def average_sector_anisotropy(sector_anisotropy_results):
    """Average sector anisotropies, separately for each pair and mode."""
    averaged_results = {}

    for mode, sector_results in sector_anisotropy_results.items():
        sectors_by_pair = {}

        for sector_id, sector in sector_results.items():
            if set(sector["anisotropy"].dims) != {"energy", "q"}:
                raise ValueError(
                    f"{mode}/{sector_id}: expected energy and q dimensions"
                )
            sectors_by_pair.setdefault(
                sector["pair_id"], [],
            ).append((sector_id, sector))

        averaged_results[mode] = {}

        for pair_id, sectors in sectors_by_pair.items():
            stack = xr.concat(
                [
                    sector["anisotropy"].transpose("energy", "q")
                    for _, sector in sectors
                ],
                dim=xr.IndexVariable(
                    "sector", [sector_id for sector_id, _ in sectors],
                ),
                join="exact",
                coords="minimal",
                compat="equals",
            ).transpose("energy", "sector", "q")
            stack = stack.assign_coords(
                sector_label=(
                    "sector", [sector["label"] for _, sector in sectors],
                ),
            )
            stack.name = "sector_anisotropy"

            mean = stack.mean("sector", skipna=True, keep_attrs=True)
            count = stack.count("sector")
            mean.name = "sector_averaged_anisotropy"
            count.name = "valid_sector_count"
            mean.attrs = {
                **mean.attrs,
                "aggregation_method": "mean_sector_anisotropy",
                "averaged_dimension": "sector",
                "sector_weighting": "equal weight per non-NaN sector",
            }
            count.attrs = {
                "description": "Number of non-NaN sector anisotropies used in the sector mean",
            }

            averaged_results[mode][pair_id] = {
                "sector_anisotropy": stack,
                "anisotropy": mean,
                "valid_sector_count": count,
            }

    return averaged_results


def calculate_global_intensity_ratio(pointwise_results):
    """Pool stitched intensities over chi, then calculate anisotropy."""
    global_results = {}

    for mode, pair_results in pointwise_results.items():
        global_results[mode] = {}

        for pair_id, result in pair_results.items():
            parallel = result["parallel"]
            perpendicular = result["perpendicular"]

            for values in (parallel, perpendicular):
                if set(values.dims) != {"energy", "chi", "q"}:
                    raise ValueError(
                        f"{mode}/{pair_id}: expected energy, chi, and q dimensions"
                    )
                if values.attrs.get("q_cap_applied") is not True:
                    raise ValueError(
                        f"{mode}/{pair_id}: rerun q-capped pointwise maps first"
                    )

            parallel, perpendicular = xr.align(
                parallel.transpose("energy", "chi", "q"),
                perpendicular.transpose("energy", "chi", "q"),
                join="exact", copy=False,
            )
            valid = np.isfinite(parallel) & np.isfinite(perpendicular)
            count = valid.sum("chi")

            parallel_mean = parallel.where(valid).mean(
                "chi", skipna=True, keep_attrs=True,
            )
            perpendicular_mean = perpendicular.where(valid).mean(
                "chi", skipna=True, keep_attrs=True,
            )
            denominator = parallel_mean + perpendicular_mean
            anisotropy = (
                (parallel_mean - perpendicular_mean)
                / denominator.where(denominator != 0)
            )

            parallel_mean.name = "global_parallel_intensity"
            perpendicular_mean.name = "global_perpendicular_intensity"
            anisotropy.name = "global_intensity_ratio_anisotropy"
            count.name = "valid_chi_count"

            metadata = {
                "aggregation_method": "global_intensity_ratio",
                "normalization_mode": mode,
                "averaged_dimension": "chi",
                "angular_weighting": "equal weight per jointly finite chi bin",
                "source_product": "clean",
                "q_cap_applied": True,
            }
            for values in (parallel_mean, perpendicular_mean, anisotropy):
                values.attrs = {**values.attrs, **metadata}
            count.attrs = {
                "description": "Number of chi bins with finite parallel and perpendicular intensities",
            }

            global_results[mode][pair_id] = {
                "parallel_intensity": parallel_mean,
                "perpendicular_intensity": perpendicular_mean,
                "anisotropy": anisotropy,
                "valid_chi_count": count,
            }

    return global_results