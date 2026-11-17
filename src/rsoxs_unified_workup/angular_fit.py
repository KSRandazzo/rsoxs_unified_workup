import numpy as np
import xarray as xr


def fit_pointwise_angular_slice(data, q_cap, settings):
    """Fit a fixed-period Fourier model to one measured pointwise A(chi) slice."""
    if not isinstance(data, xr.DataArray) or set(data.dims) != {
        "energy", "chi", "q",
    }:
        raise ValueError("Pointwise anisotropy must have energy, chi, and q dimensions")
    if data.attrs.get("q_cap_applied") is not True:
        raise ValueError("Use the q-capped pointwise anisotropy maps")
    data = data.transpose("energy", "chi", "q")

    if not isinstance(q_cap, xr.DataArray) or q_cap.dims != ("energy",):
        raise ValueError("The pair q cap must have an energy dimension")
    if not np.array_equal(q_cap.energy.values, data.energy.values):
        raise ValueError("The pair q cap and anisotropy energy coordinates differ")

    def positive_number(key):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"angular_anisotropy_fit.{key} must be numeric")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"angular_anisotropy_fit.{key} must be positive and finite")
        return value

    requested_energy = positive_number("energy_ev")
    requested_q = positive_number("q_value")
    period = positive_number("period_degrees")
    harmonics = settings["harmonics"]
    if isinstance(harmonics, bool) or not isinstance(harmonics, (int, np.integer)):
        raise ValueError("angular_anisotropy_fit.harmonics must be a positive integer")
    harmonics = int(harmonics)
    if harmonics < 1:
        raise ValueError("angular_anisotropy_fit.harmonics must be a positive integer")

    cycles = 360.0 / period
    if cycles < 1 or not np.isclose(cycles, round(cycles), rtol=0, atol=1e-9):
        raise ValueError("The period must divide 360 degrees, e.g. 90, 180, or 360")

    def nearest_index(dimension, requested):
        available = np.asarray(data[dimension].values, dtype=float)
        if available.ndim != 1 or not available.size or not np.isfinite(available).all():
            raise ValueError(f"Available {dimension} coordinates must be finite and nonempty")
        lower, upper = float(available.min()), float(available.max())
        tolerance = 1e-9 * max(1.0, abs(lower), abs(upper))
        if requested < lower - tolerance or requested > upper + tolerance:
            raise ValueError(
                f"Requested {dimension}={requested:g} is outside "
                f"the available range {lower:g} to {upper:g}"
            )
        return int(np.argmin(np.abs(available - requested)))

    q_bins = settings.get("q_bins", 1)
    if (
        isinstance(q_bins, bool)
        or not isinstance(q_bins, (int, np.integer))
        or q_bins < 1
        or q_bins % 2 == 0
    ):
        raise ValueError(
            "angular_anisotropy_fit.q_bins must be a positive odd integer"
        )
    q_bins = int(q_bins)

    energy_index = nearest_index("energy", requested_energy)
    q_index = nearest_index("q", requested_q)
    energy = float(data.energy.values[energy_index])
    q_value = float(data.q.values[q_index])

    available_q = np.asarray(data.q.values, dtype=float)
    if not np.all(np.diff(available_q) > 0):
        raise ValueError("Q coordinates must be strictly increasing")

    half_bins = q_bins // 2
    first_bin = q_index - half_bins
    last_bin = q_index + half_bins
    if first_bin < 0 or last_bin >= available_q.size:
        raise ValueError(
            f"Cannot select {q_bins} centered q bins around q={q_value:g}; "
            "choose a q farther from the grid edge or fewer bins."
        )
    selected_q_values = available_q[first_bin:last_bin + 1]

    cap = float(q_cap.isel(energy=energy_index).item())
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError(f"No finite positive pair q cap exists at {energy:g} eV")
    if selected_q_values[-1] > cap:
        raise ValueError(
            f"Selected q range {selected_q_values[0]:g} to "
            f"{selected_q_values[-1]:g} exceeds the pair q cap={cap:g} "
            f"at {energy:g} eV. Choose a lower q or fewer bins."
        )

    patch = data.isel(
        energy=energy_index,
        q=slice(first_bin, last_bin + 1),
    )
    finite = np.isfinite(patch)
    q_valid_bin_count = finite.sum("q").assign_coords(q=q_value)
    q_valid_bin_count.name = "q_valid_bin_count"

    # Require the complete q window at every fitted chi direction.
    profile = patch.where(finite).mean(
        "q", skipna=False, keep_attrs=True,
    ).assign_coords(q=q_value)
    profile.name = "q_averaged_pointwise_anisotropy"

    fit = _fit_angular_profile(profile, period, harmonics)
    fit.update({
        "requested_energy_ev": requested_energy,
        "selected_energy_ev": energy,
        "requested_q": requested_q,
        "selected_q": q_value,
        "q_bins": q_bins,
        "selected_q_values": selected_q_values.tolist(),
        "q_valid_bin_count": q_valid_bin_count,
        "q_averaging": "equal q-bin mean; all selected bins must be finite",
        "pair_q_cap": cap,
        "q_cap_applied": True,
        "source_attributes": dict(data.attrs),
    })
    return fit


def _fit_angular_profile(profile, period, harmonics):
    """Shared least-squares scoring on the actual finite chi coordinates."""
    if not isinstance(profile, xr.DataArray) or profile.dims != ("chi",):
        raise ValueError("The angular profile must be a DataArray with a chi dimension")
    chi = np.asarray(profile.chi.values, dtype=float)
    observed = np.asarray(profile.values, dtype=float)
    if not np.isfinite(chi).all():
        raise ValueError("Chi coordinates must be finite")
    if np.unique((chi + 180) % 360).size != chi.size:
        raise ValueError("Repeated chi directions would give some bins extra weight")

    used = np.isfinite(observed)
    parameter_count = 1 + 2 * harmonics
    if int(used.sum()) <= parameter_count:
        raise ValueError(
            f"Only {int(used.sum())} finite chi bins; "
            f"need more than {parameter_count} for this model"
        )

    def design_matrix(angles):
        phase = 2 * np.pi * np.asarray(angles, dtype=float) / period
        columns = [np.ones(phase.size)]
        for harmonic in range(1, harmonics + 1):
            columns.extend((
                np.cos(harmonic * phase),
                np.sin(harmonic * phase),
            ))
        return np.column_stack(columns)

    design = design_matrix(chi)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design[used], observed[used], rcond=None
    )
    if rank != parameter_count:
        raise ValueError("The available chi coverage cannot determine all fit coefficients")

    predicted = xr.DataArray(
        design @ coefficients, dims=("chi",), coords=profile.coords,
        name="fitted_angular_anisotropy",
    )
    fit_mask = xr.DataArray(
        used, dims=("chi",), coords=profile.coords, name="fit_mask",
    )
    residual = (profile - predicted).where(fit_mask)
    residual.name = "angular_fit_residual"
    errors = residual.values[used]
    squared_error = float(np.sum(errors ** 2))
    centered = observed[used] - observed[used].mean()
    total_variation = float(np.sum(centered ** 2))
    r_squared = (
        1 - squared_error / total_variation
        if total_variation > 0 else np.nan
    )

    terms = ["offset"]
    for harmonic in range(1, harmonics + 1):
        terms.extend((f"cos_{harmonic}", f"sin_{harmonic}"))
    coefficient_array = xr.DataArray(
        coefficients, dims=("term",), coords={"term": terms},
        name="fourier_coefficient",
    )
    model_angles = np.linspace(-180.0, 180.0, 1441)
    model_curve = xr.DataArray(
        design_matrix(model_angles) @ coefficients,
        dims=("chi_model",), coords={"chi_model": model_angles},
        name="angular_fit_model_curve",
    )

    sample_count = int(used.sum())
    mean_squared_error = squared_error / sample_count

    # Prevent roundoff-only residuals from deciding between exact-fit models.
    scale = float(np.max(np.abs(observed[used])))
    variance_floor = max(
        np.finfo(float).tiny,
        (100 * np.finfo(float).eps * scale) ** 2,
    )
    if total_variation <= sample_count * variance_floor:
        # R² is undefined for a profile that is constant to numerical precision.
        r_squared = np.nan
    bic = (
        sample_count * np.log(max(mean_squared_error, variance_floor))
        + parameter_count * np.log(sample_count)
    )
    model_name = (
        "constant baseline"
        if harmonics == 0
        else f"{period:g} deg, H={harmonics}"
    )

    # Re-express the single harmonic; this does not perform another fit.
    sinusoid_parameters = None
    if harmonics == 1:
        offset, cosine, sine = map(float, coefficients)
        amplitude = float(np.hypot(cosine, sine))
        phase_threshold = max(
            np.finfo(float).tiny,
            100 * np.finfo(float).eps * scale,
        )
        phase_defined = amplitude > phase_threshold
        phase_shift = np.nan
        if phase_defined:
            phase_shift = float(
                (np.arctan2(sine, cosine) * period / (2 * np.pi)) % period
            )
            # Modulo can round a tiny negative angle to exactly the period.
            if phase_shift >= period:
                phase_shift = 0.0
        sinusoid_parameters = {
            "offset": offset,
            "amplitude": amplitude,
            "phase_shift_degrees": phase_shift,
            "phase_defined": bool(phase_defined),
            "phase_zero_amplitude_threshold": float(phase_threshold),
            "phase_convention": (
                "A(chi) = offset + amplitude * "
                "cos(2*pi*(chi - phase_shift_degrees)/period_degrees); "
                "amplitude >= 0; phase shift in [0, period_degrees)"
            ),
        }

    return {
        "observed": profile.copy(deep=False),
        "fitted": predicted,
        "residual": residual,
        "fit_mask": fit_mask,
        "model_curve": model_curve,
        "coefficients": coefficient_array,
        "sinusoid_parameters": sinusoid_parameters,
        "period_degrees": None if harmonics == 0 else float(period),
        "harmonics": int(harmonics),
        "model_name": model_name,
        "valid_chi_count": sample_count,
        "parameter_count": parameter_count,
        "residual_degrees_of_freedom": sample_count - parameter_count,
        "residual_sum_of_squares": squared_error,
        "rmse": float(np.sqrt(mean_squared_error)),
        "r_squared": float(r_squared),
        "bic": float(bic),
        "bic_variance_floor": float(variance_floor),
        "design_condition_number": float(singular_values[0] / singular_values[-1]),
        "fit_method": "ordinary least squares",
        "angular_weighting": "equal weight per finite chi bin",
        "period_fitted": False,
    }


def compare_angular_models(reference_fits, max_cycles=8, max_harmonics=2):
    """Rank constant and Fourier models on each unchanged q-averaged profile.

    Lower BIC is preferred under an equal-variance independent Gaussian
    residual model. Correlated angular bins make this an exploratory ranking,
    not a formal confidence statement about a physical period.
    """
    for name, value in (
        ("max_cycles", max_cycles),
        ("max_harmonics", max_harmonics),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")

    comparisons = {}
    for mode, pair_results in reference_fits.items():
        comparisons[mode] = {}

        for pair_id, reference in pair_results.items():
            profile = reference["observed"]
            if reference.get("q_cap_applied") is not True:
                raise ValueError(f"{mode}/{pair_id}: use a q-capped reference fit")
            if not np.array_equal(
                reference["fit_mask"].values, np.isfinite(profile.values)
            ):
                raise ValueError(f"{mode}/{pair_id}: reference fit mask is inconsistent")

            baseline = {
                **reference,
                **_fit_angular_profile(profile, 360.0, 0),
            }
            candidates = [baseline]
            failed_models = []

            for cycles in range(1, int(max_cycles) + 1):
                period = 360.0 / cycles
                for harmonics in range(1, int(max_harmonics) + 1):
                    try:
                        fitted = _fit_angular_profile(profile, period, harmonics)
                    except ValueError as error:
                        failed_models.append({
                            "period_degrees": period,
                            "harmonics": harmonics,
                            "reason": str(error),
                        })
                        continue

                    np.testing.assert_array_equal(
                        fitted["fit_mask"].values, baseline["fit_mask"].values
                    )
                    candidates.append({**reference, **fitted})

            candidates.sort(
                key=lambda fitted: (fitted["bic"], fitted["parameter_count"])
            )
            best = candidates[0].copy()
            delta_vs_constant = baseline["bic"] - best["bic"]
            runner_up_gap = (
                candidates[1]["bic"] - best["bic"]
                if len(candidates) > 1 else np.nan
            )
            best.update({
                "selection_method": "minimum BIC among tested models",
                "period_selected_from_candidates": best["harmonics"] > 0,
                "delta_bic_vs_constant": float(delta_vs_constant),
                "runner_up_delta_bic": float(runner_up_gap),
            })

            ranking = [
                {
                    "rank": index + 1,
                    "model_name": fitted["model_name"],
                    "period_degrees": fitted["period_degrees"],
                    "harmonics": fitted["harmonics"],
                    "parameter_count": fitted["parameter_count"],
                    "valid_chi_count": fitted["valid_chi_count"],
                    "rmse": fitted["rmse"],
                    "r_squared": fitted["r_squared"],
                    "bic": fitted["bic"],
                    "delta_bic": fitted["bic"] - best["bic"],
                }
                for index, fitted in enumerate(candidates)
            ]
            comparisons[mode][pair_id] = {
                "best_fit": best,
                "ranking": ranking,
                "baseline_bic": baseline["bic"],
                "failed_models": failed_models,
                "max_cycles": int(max_cycles),
                "max_harmonics": int(max_harmonics),
                "bic_assumption": (
                    "Equal-variance independent Gaussian residuals; "
                    "exploratory ranking when angular bins are correlated"
                ),
            }

    return comparisons


def triangle_wave(chi_degrees, phase_shift_degrees, period_degrees=90.0):
    """Symmetric triangle in [-1, 1], with a maximum at the phase shift."""
    distance = (
        (np.asarray(chi_degrees, dtype=float) - phase_shift_degrees)
        / period_degrees + 0.5
    ) % 1.0 - 0.5
    return 1.0 - 4.0 * np.abs(distance)


def _fit_triangle_profile(profile, period=90.0):
    """Global least squares over the phase intervals of a fixed triangle.

    Between corner-crossing phases the wave is affine in phase. An
    unconstrained three-column linear fit provides the interior candidate;
    each interval's endpoints are also tested. This avoids initial guesses
    and does not quantize the fitted phase onto a sampling grid.
    """
    baseline = _fit_angular_profile(profile, period, 0)
    chi = np.asarray(profile.chi.values, dtype=float)
    observed = np.asarray(profile.values, dtype=float)
    used = np.isfinite(observed)
    sample_count = int(used.sum())
    if sample_count <= 3:
        raise ValueError("Need more than three finite chi bins for the triangle fit")

    angles = chi[used]
    values = observed[used]
    value_mean = float(values.mean())
    centered_values = values - value_mean
    scale = float(np.max(np.abs(values)))
    phase_threshold = max(
        np.finfo(float).tiny, 100 * np.finfo(float).eps * scale,
    )
    # Use fractions of a cycle so the derivative column is dimensionless.
    crossings = np.unique(np.concatenate((
        [0.0, 1.0], (angles / period) % 1.0,
        (angles / period - 0.5) % 1.0,
    )))
    best = None

    def consider_phase(phase_fraction):
        nonlocal best
        phase = float((phase_fraction % 1.0) * period)
        if phase >= period:
            phase = 0.0
        wave = triangle_wave(angles, phase, period)
        design = np.column_stack((np.ones(sample_count), wave))
        coefficients, _, rank, singular_values = np.linalg.lstsq(
            design, centered_values, rcond=None,
        )
        if rank != 2:
            return
        errors = centered_values - design @ coefficients
        squared_error = float(np.sum(errors ** 2))
        if best is None or squared_error < best["squared_error"]:
            best = {
                "offset": value_mean + float(coefficients[0]),
                "signed_amplitude": float(coefficients[1]),
                "phase": phase,
                "squared_error": squared_error,
                "linear_condition_number": float(singular_values[0] / singular_values[-1]),
            }

    for crossing in crossings:
        consider_phase(crossing)
    tested_intervals = 0
    for lower, upper in zip(crossings[:-1], crossings[1:]):
        if upper - lower <= 1e-12:
            continue
        tested_intervals += 1
        midpoint = float((lower + upper) / 2)
        relative = (angles / period - midpoint + 0.5) % 1.0 - 0.5
        wave = triangle_wave(angles, midpoint * period, period)
        slope = 4.0 * np.sign(relative)
        design = np.column_stack((np.ones(sample_count), wave, slope))
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, centered_values, rcond=None,
        )
        if rank != 3 or abs(coefficients[1]) <= phase_threshold:
            continue
        candidate = midpoint + coefficients[2] / coefficients[1]
        if lower <= candidate <= upper:
            consider_phase(candidate)

    if best is None:
        raise ValueError("The finite angular coverage cannot determine a triangle wave")

    offset = best["offset"]
    amplitude = abs(best["signed_amplitude"])
    phase = best["phase"]
    if best["signed_amplitude"] < 0:
        phase = (phase + period / 2) % period
    phase_defined = amplitude > phase_threshold
    if not phase_defined:
        # A numerically flat wave has no identifiable phase.
        offset, amplitude, phase = value_mean, 0.0, 0.0

    predicted = xr.DataArray(
        offset + amplitude * triangle_wave(chi, phase, period),
        dims=("chi",), coords=profile.coords,
        name="fitted_triangle_anisotropy",
    )
    residual = (profile - predicted).where(baseline["fit_mask"])
    residual.name = "triangle_fit_residual"
    squared_error = float(np.sum(residual.values[used] ** 2))
    total_variation = float(np.sum(centered_values ** 2))
    mean_squared_error = squared_error / sample_count
    variance_floor = baseline["bic_variance_floor"]
    r_squared = (
        1.0 - squared_error / total_variation
        if total_variation > sample_count * variance_floor else np.nan
    )
    model_angles = np.linspace(-180.0, 180.0, 1441)
    # Include exact fitted corners in the display curve, not in measured data.
    corners = phase + np.arange(-10, 11) * period / 2
    corners = corners[(corners >= -180) & (corners <= 180)]
    model_angles = np.unique(np.concatenate((model_angles, corners)))
    curve = xr.DataArray(
        offset + amplitude * triangle_wave(model_angles, phase, period),
        dims=("chi_model",), coords={"chi_model": model_angles},
        name="triangle_fit_model_curve",
    )
    parameters = {
        "offset": float(offset),
        "amplitude": float(amplitude),
        "phase_shift_degrees": float(phase) if phase_defined else np.nan,
        "phase_defined": bool(phase_defined),
        "phase_zero_amplitude_threshold": float(phase_threshold),
        "phase_convention": (
            "Triangle maximum at phase_shift_degrees modulo period_degrees; "
            "amplitude >= 0; wave ranges from offset-amplitude to offset+amplitude"
        ),
    }
    return {
        "observed": profile.copy(deep=False),
        "fitted": predicted,
        "residual": residual,
        "fit_mask": baseline["fit_mask"],
        "model_curve": curve,
        "triangle_parameters": parameters,
        "period_degrees": float(period),
        "waveform": "triangle",
        "model_name": f"{period:g} deg triangle",
        "valid_chi_count": sample_count,
        "parameter_count": 3,
        "residual_degrees_of_freedom": sample_count - 3,
        "residual_sum_of_squares": squared_error,
        "rmse": float(np.sqrt(mean_squared_error)),
        "r_squared": float(r_squared),
        "bic": float(sample_count * np.log(max(mean_squared_error, variance_floor)) + 3 * np.log(sample_count)),
        "bic_variance_floor": variance_floor,
        "linear_condition_number": best["linear_condition_number"],
        "phase_intervals_tested": tested_intervals,
        "fit_method": "global piecewise-linear least squares over phase",
        "angular_weighting": "equal weight per finite chi bin",
        "period_fitted": False,
    }


def fit_fixed_90_waveforms(reference_fits):
    """Compare a 90-degree sinusoid and triangle on the identical profile.

    Reuse the existing single-harmonic sinusoid without refitting or
    changing the q window. Both waveforms fit offset, amplitude and phase;
    their phase shift is the chi location of a maximum, modulo 90 degrees.
    """
    results = {}
    for mode, pair_results in reference_fits.items():
        results[mode] = {}
        for pair_id, reference in pair_results.items():
            if (
                reference.get("period_degrees") != 90.0
                or reference.get("harmonics") != 1
                or reference.get("sinusoid_parameters") is None
            ):
                raise ValueError(
                    f"{mode}/{pair_id}: set period_degrees=90 and harmonics=1, "
                    "then reload angular_fit and rerun the angular-fitting calculation"
                )
            if reference.get("q_cap_applied") is not True:
                raise ValueError(f"{mode}/{pair_id}: use the q-capped reference fit")
            np.testing.assert_array_equal(
                reference["fit_mask"].values,
                np.isfinite(reference["observed"].values),
            )
            if (
                len(reference["selected_q_values"]) != reference["q_bins"]
                or max(reference["selected_q_values"]) > reference["pair_q_cap"]
            ):
                raise ValueError(f"{mode}/{pair_id}: the reference q window is inconsistent")
            triangle = _fit_triangle_profile(reference["observed"], period=90.0)
            np.testing.assert_array_equal(
                triangle["fit_mask"].values, reference["fit_mask"].values,
            )
            metadata_keys = (
                "requested_energy_ev", "selected_energy_ev", "requested_q",
                "selected_q", "q_bins", "selected_q_values", "q_valid_bin_count",
                "q_averaging", "pair_q_cap", "q_cap_applied", "source_attributes",
                "normalization_mode", "pair_id",
            )
            triangle.update({
                key: reference[key] for key in metadata_keys if key in reference
            })
            results[mode][pair_id] = {
                "sinusoid": reference,
                "triangle": triangle,
                "delta_bic_triangle_minus_sinusoid": triangle["bic"] - reference["bic"],
                "comparison_note": (
                    "Both models have three fitted parameters and identical finite "
                    "chi bins; smaller RMSE is a better fit to these same data. "
                    "No uncertainty estimate or automatic model selection is applied."
                ),
            }
    return results


ANGULAR_MAP_STATUS = {
    0: "fit successful",
    1: "centered q window crosses the grid edge",
    2: "q window exceeds the pair q cap",
    3: "not enough finite chi bins (need more than three)",
    4: "angular coverage cannot determine all three coefficients",
    5: "numerical fit failure",
    6: "no finite positive pair q cap at this energy",
}


def fit_pointwise_angular_map(data, q_cap, settings, progress=False, description=None):
    """Fit a fixed-90-degree sinusoid at every eligible energy/q center.

    Uses the same centered q-bin mean and complete-window finite-chi
    requirement as fit_pointwise_angular_slice. Columns with identical
    finite-chi coverage are solved together using the same least-squares
    design, without normal equations, interpolation or chi smoothing.
    """
    if not isinstance(data, xr.DataArray) or set(data.dims) != {"energy", "chi", "q"}:
        raise ValueError("Pointwise anisotropy must have energy, chi, and q dimensions")
    if data.attrs.get("q_cap_applied") is not True:
        raise ValueError("Use the q-capped pointwise anisotropy maps")
    harmonics = settings.get("harmonics")
    if (
        settings.get("period_degrees") != 90
        or isinstance(harmonics, bool)
        or not isinstance(harmonics, (int, np.integer))
        or harmonics != 1
    ):
        raise ValueError("Angular parameter maps require period_degrees=90 and harmonics=1")
    data = data.transpose("energy", "chi", "q")
    if not isinstance(q_cap, xr.DataArray) or q_cap.dims != ("energy",):
        raise ValueError("The pair q cap must have an energy dimension")
    if not np.array_equal(q_cap.energy.values, data.energy.values):
        raise ValueError("The pair q cap and anisotropy energy coordinates differ")
    energy = np.asarray(data.energy.values, dtype=float)
    chi = np.asarray(data.chi.values, dtype=float)
    q = np.asarray(data.q.values, dtype=float)
    if not energy.size or not np.isfinite(energy).all() or np.unique(energy).size != energy.size:
        raise ValueError("Energy coordinates must be nonempty, finite, and unique")
    if not chi.size or not np.isfinite(chi).all() or np.unique((chi + 180) % 360).size != chi.size:
        raise ValueError("Chi coordinates must contain finite, unique directions")
    if not q.size or not np.isfinite(q).all() or not np.all(np.diff(q) > 0):
        raise ValueError("Q coordinates must be finite and strictly increasing")
    q_bins = settings.get("q_bins", 1)
    if isinstance(q_bins, bool) or not isinstance(q_bins, (int, np.integer)) or q_bins < 1 or q_bins % 2 == 0:
        raise ValueError("angular_anisotropy_fit.q_bins must be a positive odd integer")
    q_bins = int(q_bins)
    if q_bins > q.size:
        raise ValueError(f"q_bins={q_bins} exceeds the available {q.size} q bins")

    caps = np.asarray(q_cap.values, dtype=float)
    shape = (energy.size, q.size)
    float_fields = (
        "offset", "amplitude", "phase_shift_degrees", "cos_coefficient",
        "sin_coefficient", "rmse", "r_squared", "design_condition_number",
    )
    maps = {name: np.full(shape, np.nan, dtype=float) for name in float_fields}
    counts = np.zeros(shape, dtype=np.int64)
    status = np.full(shape, 1, dtype=np.int8)
    phase_defined = np.zeros(shape, dtype=bool)
    eligible = np.zeros(shape, dtype=bool)
    half = q_bins // 2
    centers = np.arange(half, q.size - half)
    lower_q = np.full(q.size, np.nan)
    upper_q = np.full(q.size, np.nan)
    lower_q[centers] = q[centers - half]
    upper_q[centers] = q[centers + half]
    theta = 2 * np.pi * chi / 90.0
    design = np.column_stack((np.ones(chi.size), np.cos(theta), np.sin(theta)))
    iterator = range(energy.size)
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, desc=description or "Angular sinusoid maps", unit="energy")

    for energy_index in iterator:
        cap = caps[energy_index]
        if not np.isfinite(cap) or cap <= 0:
            status[energy_index, centers] = 6
            continue
        allowed = upper_q[centers] <= cap
        status[energy_index, centers] = 2
        eligible_centers = centers[allowed]
        if not eligible_centers.size:
            continue
        eligible[energy_index, eligible_centers] = True
        status[energy_index, eligible_centers] = 3
        raw = np.asarray(data.isel(energy=energy_index).values, dtype=float)
        finite_raw = np.where(np.isfinite(raw), raw, np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(finite_raw, q_bins, axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            profiles = windows.mean(axis=-1)[:, allowed]
        finite = np.isfinite(profiles)
        count = finite.sum(axis=0)
        counts[energy_index, eligible_centers] = count
        fit_candidates = np.flatnonzero(count > 3)
        if not fit_candidates.size:
            continue
        packed = np.packbits(finite[:, fit_candidates].T, axis=1)
        _, first, groups = np.unique(packed, axis=0, return_index=True, return_inverse=True)
        for group, first_index in enumerate(first):
            local_columns = fit_candidates[groups == group]
            output_columns = eligible_centers[local_columns]
            used = finite[:, fit_candidates[first_index]]
            x = design[used]
            y = profiles[used][:, local_columns]
            try:
                coefficients, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
            except np.linalg.LinAlgError:
                status[energy_index, output_columns] = 5
                continue
            if rank != 3:
                status[energy_index, output_columns] = 4
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                errors = y - x @ coefficients
                squared_error = np.sum(errors ** 2, axis=0)
                total_variation = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
                scale = np.max(np.abs(y), axis=0)
                variance_floor = np.maximum(np.finfo(float).tiny, (100 * np.finfo(float).eps * scale) ** 2)
                rmse = np.sqrt(squared_error / int(used.sum()))
                amplitude = np.hypot(coefficients[1], coefficients[2])
            valid_solution = np.isfinite(coefficients).all(axis=0) & np.isfinite(rmse) & np.isfinite(amplitude)
            status[energy_index, output_columns[~valid_solution]] = 5
            if not valid_solution.any():
                continue
            output_columns = output_columns[valid_solution]
            beta = coefficients[:, valid_solution]
            amplitude = amplitude[valid_solution]
            squared_error = squared_error[valid_solution]
            total_variation = total_variation[valid_solution]
            variance_floor = variance_floor[valid_solution]
            scale = scale[valid_solution]
            r_squared = np.full(output_columns.size, np.nan)
            varied = total_variation > int(used.sum()) * variance_floor
            r_squared[varied] = 1 - squared_error[varied] / total_variation[varied]
            phase_threshold = np.maximum(np.finfo(float).tiny, 100 * np.finfo(float).eps * scale)
            defined = amplitude > phase_threshold
            phase = (np.arctan2(beta[2], beta[1]) * 90 / (2 * np.pi)) % 90
            phase[phase >= 90] = 0.0
            phase[~defined] = np.nan
            maps["offset"][energy_index, output_columns] = beta[0]
            maps["cos_coefficient"][energy_index, output_columns] = beta[1]
            maps["sin_coefficient"][energy_index, output_columns] = beta[2]
            maps["amplitude"][energy_index, output_columns] = amplitude
            maps["phase_shift_degrees"][energy_index, output_columns] = phase
            maps["rmse"][energy_index, output_columns] = rmse[valid_solution]
            maps["r_squared"][energy_index, output_columns] = r_squared
            maps["design_condition_number"][energy_index, output_columns] = singular[0] / singular[-1]
            status[energy_index, output_columns] = 0
            phase_defined[energy_index, output_columns] = defined

    coordinates = {"energy": data.energy, "q": data.q}
    dataset = xr.Dataset({
        **{name: xr.DataArray(values, dims=("energy", "q"), coords=coordinates) for name, values in maps.items()},
        "valid_chi_count": xr.DataArray(counts, dims=("energy", "q"), coords=coordinates),
        "fit_status": xr.DataArray(status, dims=("energy", "q"), coords=coordinates),
        "fit_success": xr.DataArray(status == 0, dims=("energy", "q"), coords=coordinates),
        "phase_defined": xr.DataArray(phase_defined, dims=("energy", "q"), coords=coordinates),
        "window_eligible": xr.DataArray(eligible, dims=("energy", "q"), coords=coordinates),
        "q_window_lower": xr.DataArray(lower_q, dims="q", coords={"q": data.q}),
        "q_window_upper": xr.DataArray(upper_q, dims="q", coords={"q": data.q}),
        "pair_q_cap": q_cap.copy(deep=False),
    })
    dataset.attrs = {
        "description": "Fixed-90-degree sinusoid parameters of sector-relative pointwise anisotropy",
        "model": "A(chi) = offset + amplitude*cos(2*pi*(chi-phase_shift_degrees)/90)",
        "period_degrees": 90.0,
        "harmonics": 1,
        "parameter_count": 3,
        "q_bins": q_bins,
        "q_averaging": "equal centered q-bin mean; all selected bins must be finite at each chi",
        "q_cap_applied": True,
        "angular_weighting": "equal weight per finite chi bin",
        "source_chi_bin_count": int(chi.size),
        "source_attributes": dict(data.attrs),
        "phase_convention": "Location of a fitted maximum, modulo 90 degrees; amplitude is nonnegative",
        "phase_caution": "phase_defined tests numerical nonzero amplitude, not statistical significance or uncertainty",
        "fit_status_codes": {str(code): meaning for code, meaning in ANGULAR_MAP_STATUS.items()},
        "fit_method": "ordinary least squares; identical finite-chi masks batched together",
        "energy_q_scope": "all measured energies and eligible native q centers; slice energy_ev and q_value do not limit this map",
    }
    dataset["phase_shift_degrees"].attrs["units"] = "degrees"
    dataset["fit_status"].attrs["codes"] = dataset.attrs["fit_status_codes"]
    return dataset


def fit_scan_set_angular_maps(configuration, scans, pointwise_results, progress=True):
    """Fit both normalization modes separately on each pair's native grid."""
    from rsoxs_unified_workup.config import get_mask_pairs

    pairs = get_mask_pairs(configuration)
    settings = configuration["processing"]["angular_anisotropy_fit"]
    if set(pointwise_results) != {"full_ring", "half_ring"}:
        raise ValueError("Provide full_ring and half_ring pointwise anisotropy results")
    results = {}
    for mode, pair_results in pointwise_results.items():
        if set(pair_results) != set(pairs):
            raise ValueError(f"{mode}: pointwise pair IDs do not match the configuration")
        results[mode] = {}
        for pair_id, polarizations in pairs.items():
            records = [scans[polarization] for polarization in polarizations]
            if any(record["integration"]["mask_pair_id"] != pair_id for record in records):
                raise ValueError(f"{pair_id}: scan integration pair metadata is inconsistent")
            cap = xr.concat(
                [record["integration"]["q_cap"] for record in records],
                dim="pair_member", join="exact",
            ).min("pair_member", skipna=False)
            dataset = fit_pointwise_angular_map(
                pair_results[pair_id]["anisotropy"], cap, settings,
                progress=progress, description=f"{mode}/{pair_id}",
            )
            dataset.attrs.update({
                "polarization_pair_id": pair_id,
                "polarizations": list(polarizations),
                "normalization_mode": mode,
                "scan_set_id": configuration["scan_set"]["scan_set_id"],
                "q_grid_scope": "polarization_pair",
            })
            results[mode][pair_id] = dataset
            print(f"\n{mode}/{pair_id}: {int(dataset.fit_success.sum().item()):,} fitted energy/q points")
            for code, meaning in ANGULAR_MAP_STATUS.items():
                count = int((dataset.fit_status == code).sum().item())
                if code and count:
                    print(f"  Not fitted: {count:,} — {meaning}")
    return results
