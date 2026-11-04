import numpy as np
import xarray as xr


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


def normalize_scan(scan_record, processing):
    """Return capped and uncapped chi-normalization products for one scan."""
    settings = processing["normalization"]
    integration = scan_record["integration"]

    def normalize_intensity(data):
        half_ring = select_chi_region(
            data,
            center_degrees=settings["half_ring_center_degrees"],
            halfwidth_degrees=settings["half_ring_halfwidth_degrees"],
        )

        full_ring_mean = data.mean("chi", skipna=True)
        half_ring_mean = half_ring.mean("chi", skipna=True)

        full_denominator = full_ring_mean.where(
            full_ring_mean != 0
        )
        half_denominator = half_ring_mean.where(
            half_ring_mean != 0
        )

        return {
            "full_ring_mean": full_ring_mean,
            "half_ring_mean": half_ring_mean,
            "full_ring_normalized": data / full_denominator,
            "half_ring_normalized": data / half_denominator,
        }

    return {
        **normalize_intensity(integration["clean"]),
        "uncapped_q": normalize_intensity(
            integration["clean_uncapped_q"]
        ),
        "half_ring_center_degrees": float(
            settings["half_ring_center_degrees"]
        ),
        "half_ring_halfwidth_degrees": float(
            settings["half_ring_halfwidth_degrees"]
        ),
    }


def flux_correct_scan(scan_record, processing):
    """Optionally flux-correct the integrated intensity of one scan."""
    settings = processing.get("flux_correction", {})

    if not settings.get("enabled", False):
        return {}

    clean = scan_record["integration"]["clean"]
    monitor = scan_record["au_mesh_current"]
    monitor_values = np.asarray(
        monitor.values,
        dtype=float,
    ).reshape(-1)
    energies = np.asarray(
        clean.energy.values,
        dtype=float,
    )

    if len(monitor_values) != len(energies):
        raise ValueError(
            f"Scan {scan_record['scan_id']} has "
            f"{len(monitor_values)} monitor values for "
            f"{len(energies)} energies"
        )
    if not np.all(np.isfinite(monitor_values)):
        raise ValueError(
            f"Scan {scan_record['scan_id']} has "
            f"non-finite Au mesh current"
        )
    if np.any(monitor_values == 0):
        raise ValueError(
            f"Scan {scan_record['scan_id']} has zero "
            f"Au mesh current"
        )

    if "energy" in monitor.coords:
        monitor_energies = np.asarray(
            monitor["energy"].values,
            dtype=float,
        ).reshape(-1)

        if not np.allclose(
            monitor_energies,
            energies,
        ):
            raise ValueError(
                f"Scan {scan_record['scan_id']} Au mesh "
                f"energies do not match integrated energies"
            )

    flux = xr.DataArray(
        monitor_values,
        coords={"energy": energies},
        dims=("energy",),
        name="au_mesh_current",
    )
    flux_corrected = clean / flux
    flux_corrected.name = "flux_corrected_intensity"

    return {
        "flux": flux,
        "flux_corrected": flux_corrected,
    }
