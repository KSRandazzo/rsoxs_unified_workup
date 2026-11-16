import numpy as np

from rsoxs_unified_workup.normalization import select_chi_region


def calculate_ring_mean_intensities(scan_record, processing):
    """Average flux-corrected intensity over full and configured half rings."""
    correction = scan_record.get("flux_correction", {})
    if "flux_corrected" not in correction:
        raise ValueError(
            "Flux-corrected data are missing. Enable flux correction "
            "and run the flux-correction cell first."
        )

    integration = scan_record["integration"]
    data = correction["flux_corrected"]
    if set(data.dims) != {"energy", "chi", "q"}:
        raise ValueError("Intensity data must have energy, chi, and q dimensions")

    data = data.transpose("energy", "chi", "q")
    for dimension in ("energy", "chi", "q"):
        if not np.array_equal(
            data[dimension].values,
            integration["clean"][dimension].values,
        ):
            raise ValueError(
                f"Flux-corrected and integrated {dimension} coordinates differ"
            )

    q_cap = integration["q_cap"]
    if q_cap.dims != ("energy",) or not np.array_equal(
        q_cap.energy.values, data.energy.values
    ):
        raise ValueError("Q caps must match the intensity energy coordinates")

    data = data.where(np.isfinite(data) & (data.q <= q_cap))
    settings = processing["normalization"]
    center = float(settings["half_ring_center_degrees"])
    halfwidth = float(settings["half_ring_halfwidth_degrees"])
    if not np.isfinite(center) or not np.isfinite(halfwidth):
        raise ValueError("Half-ring center and halfwidth must be finite")

    regions = {
        "full_ring": data,
        "half_ring": select_chi_region(data, center, halfwidth),
    }
    metadata = {
        "source_product": "flux_correction.flux_corrected",
        "flux_correction_applied": True,
        "ring_normalization_applied": False,
        "exposure_time_correction_applied": False,
        "q_cap_applied": True,
        "intensity_scale": "relative detector intensity / Au mesh current",
        "scan_id": int(scan_record["scan_id"]),
        "polarization_deg": float(scan_record["polarization_deg"]),
        **{
            key: integration[key]
            for key in (
                "beam_center_x", "beam_center_y",
                "mask_pair_id", "mask_id", "beam_center_pass_id",
            )
        },
    }
    results = {
        **metadata,
        "half_ring_center_degrees": center,
        "half_ring_halfwidth_degrees": halfwidth,
    }

    for mode, region in regions.items():
        mean = region.mean("chi", skipna=True)
        count = region.count("chi")
        mean.name = f"flux_corrected_{mode}_mean_intensity"
        count.name = f"{mode}_valid_chi_count"
        mean.attrs = {
            **metadata,
            "angular_reduction": "equal weight per finite chi bin",
            "angular_region": mode,
            "selected_chi_bin_count": region.sizes["chi"],
        }
        if mode == "half_ring":
            mean.attrs.update({
                "half_ring_center_degrees": center,
                "half_ring_halfwidth_degrees": halfwidth,
            })
        count.attrs = {
            "description": "Number of finite chi bins contributing to intensity",
            "angular_region": mode,
        }
        results[f"{mode}_mean"] = mean
        results[f"{mode}_valid_chi_count"] = count

    return results