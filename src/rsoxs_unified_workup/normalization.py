import numpy as np


def wrap_chi(values):
    """Wrap angles into the interval [-180, 180)."""
    return ((np.asarray(values) + 180) % 360) - 180


def select_chi_region(
    data,
    center_degrees,
    halfwidth_degrees,
):
    """Select a chi region, including one crossing ±180°."""
    if not 0 < halfwidth_degrees <= 180:
        raise ValueError(
            "halfwidth_degrees must be above 0 and at most 180"
        )

    if np.isclose(halfwidth_degrees, 180):
        return data

    chi_values = wrap_chi(data["chi"].values)
    lower = wrap_chi(
        center_degrees - halfwidth_degrees
    )
    upper = wrap_chi(
        center_degrees + halfwidth_degrees
    )

    if lower <= upper:
        selected = (
            (chi_values >= lower)
            & (chi_values <= upper)
        )
    else:
        selected = (
            (chi_values >= lower)
            | (chi_values <= upper)
        )

    return data.isel(chi=np.flatnonzero(selected))


def normalize_scan(integration_result, settings):
    """Calculate full-ring and configured half-ring normalization."""
    clean = integration_result["clean"]
    half_ring = select_chi_region(
        clean,
        center_degrees=settings[
            "half_ring_center_degrees"
        ],
        halfwidth_degrees=settings[
            "half_ring_halfwidth_degrees"
        ],
    )

    full_ring_mean = clean.mean(
        "chi",
        skipna=True,
    )
    half_ring_mean = half_ring.mean(
        "chi",
        skipna=True,
    )

    full_denominator = full_ring_mean.where(
        full_ring_mean != 0
    )
    half_denominator = half_ring_mean.where(
        half_ring_mean != 0
    )

    return {
        "full_ring_mean": full_ring_mean,
        "half_ring_mean": half_ring_mean,
        "full_ring_normalized": (
            clean / full_denominator
        ),
        "half_ring_normalized": (
            clean / half_denominator
        ),
        "half_ring_center_degrees": float(
            settings["half_ring_center_degrees"]
        ),
        "half_ring_halfwidth_degrees": float(
            settings["half_ring_halfwidth_degrees"]
        ),
    }


def normalize_scan_set(
    integration_results,
    processing,
):
    """Normalize every polarization in a scan set."""
    settings = processing["normalization"]

    return {
        polarization: normalize_scan(
            integration_result=result,
            settings=settings,
        )
        for polarization, result
        in integration_results.items()
    }
