import numpy as np

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
    normalized_data,
    sector,
    parallel_polarization,
    perpendicular_polarization,
):
    """Calculate anisotropy for one normalization and sector."""
    parallel_slice = select_chi_region(
        normalized_data[parallel_polarization],
        center_degrees=sector["chi_center_deg"],
        halfwidth_degrees=sector[
            "chi_halfwidth_deg"
        ],
    )
    perpendicular_slice = select_chi_region(
        normalized_data[perpendicular_polarization],
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
        "parallel_intensity": parallel_intensity,
        "perpendicular_intensity": (
            perpendicular_intensity
        ),
        "anisotropy": anisotropy,
    }


def calculate_anisotropy(
    configuration,
    normalization_results,
):
    """Calculate every configured sector for both normalizations."""
    pairs_by_id = {
        pair["pair_id"]: pair
        for pair in configuration["polarization_pairs"]
    }
    normalized_by_mode = {
        mode: {
            polarization: result[data_key]
            for polarization, result
            in normalization_results.items()
        }
        for mode, data_key in NORMALIZATION_MODES.items()
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

        for mode, normalized_data in (
            normalized_by_mode.items()
        ):
            calculation = calculate_sector_anisotropy(
                normalized_data=normalized_data,
                sector=sector,
                parallel_polarization=parallel,
                perpendicular_polarization=perpendicular,
            )
            results[mode][sector["sector_id"]] = {
                **sector,
                **calculation,
            }

    return results
