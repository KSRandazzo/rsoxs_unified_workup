"""Review-only workflow helpers; no loading, integration, fitting, or writes."""
from copy import deepcopy
import numpy as np

from rsoxs_unified_workup.config import get_review_energies, validate_requested_energies


def _canonical(value):
    """Stable comparisons when CSV metadata contains NaN or NumPy scalars."""
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_metadata": str(value)}
    return value


def numerical_configuration(configuration):
    """Capture conservative numerical settings while excluding review-only knobs."""
    settings = deepcopy(configuration["processing"])
    for key in ("plotting", "runtime", "angular_anisotropy_fit"):
        settings.pop(key, None)
    return _canonical({
        "processing": settings,
        **{key: deepcopy(configuration.get(key)) for key in (
            "scan_set", "scans", "polarization_pairs", "anisotropy_sectors",
        )},
    })


def assert_processing_current(snapshot, configuration):
    if snapshot != numerical_configuration(configuration):
        raise RuntimeError(
            "Numerical configuration changed since processing. Do not refresh plots "
            "or fit stale results. Rerun the affected masks/center/integration/"
            "normalization/anisotropy calculations, then repeat the review."
        )


STAGE_SETTINGS = {
    "integration": ("darks", "beamstop", "detector_borders", "center_mask", "beam_centers", "beam_center_fit", "integration"),
    "normalization": ("normalization",),
    "flux": ("flux_correction",),
}


def stage_configuration(configuration, stage):
    return _canonical({
        "scan_set": deepcopy(configuration["scan_set"]),
        "scans": deepcopy(configuration["scans"]),
        "processing": {key: deepcopy(configuration["processing"].get(key)) for key in STAGE_SETTINGS[stage]},
    })


def assert_stage_current(snapshot, configuration, stage):
    if snapshot != stage_configuration(configuration, stage):
        raise RuntimeError(f"{stage} settings changed. Rerun {stage} and affected downstream cells before continuing.")


def validate_review_energies(configuration, scans):
    energies = get_review_energies(configuration["processing"])
    if not energies:
        raise ValueError("Set [plotting] energies_ev in this scan set's TOML")
    for record in scans.values():
        validate_requested_energies(
            scan_record=record, energy_values=record["data"].energy.values,
            requested_energies=energies, setting_name="Shared review energies (including centering)",
        )
    return energies


def review_signature(configuration, q_range):
    return {
        "numerical": numerical_configuration(configuration),
        "energies": get_review_energies(configuration["processing"]),
        "q_range": tuple(float(value) for value in q_range),
    }


def mark_anisotropy_reviewed(configuration, pointwise_results, q_range):
    return {"settings": review_signature(configuration, q_range), "source": pointwise_results}


def require_anisotropy_review(token, configuration, pointwise_results, q_range):
    if (
        not token or token.get("source") is not pointwise_results
        or token.get("settings") != review_signature(configuration, q_range)
    ):
        raise RuntimeError(
            "Review is missing or stale. Refresh the selected-energy plots and "
            "data-derived energy curves, then confirm the anisotropy review before fitting."
        )


def map_references_from_energy_reviews(energy_reviews):
    """Adapt successful multi-energy slice fits to the existing map cross-check."""
    result = {}
    for mode, pairs in energy_reviews.items():
        result[mode] = {}
        for pair_id, entries in pairs.items():
            successful = [item["sinusoid"] for item in entries if item["status"] == "fitted"]
            # Every successful selected energy is checked by the caller.
            result[mode][pair_id] = successful
    return result
