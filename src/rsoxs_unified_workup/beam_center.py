import numpy as np
import PyHyperScattering as phs
import xarray as xr
from joblib import Parallel, delayed, parallel_backend
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4


def wrap_chi(values):
    """Wrap angles into the interval [-180, 180)."""
    return ((np.asarray(values) + 180) % 360) - 180


def get_chi_slice(data, chi_center, chi_halfwidth):
    """Select a chi interval, including intervals crossing ±180°."""
    chi_values = data["chi"].values
    chi_min = wrap_chi(chi_center - chi_halfwidth)
    chi_max = wrap_chi(chi_center + chi_halfwidth)
    wrapped_values = wrap_chi(chi_values)

    if chi_min <= chi_max:
        selected = (
            (wrapped_values >= chi_min)
            & (wrapped_values <= chi_max)
        )
    else:
        selected = (
            (wrapped_values >= chi_min)
            | (wrapped_values <= chi_max)
        )

    return data.sel(chi=selected)


def azimuth_centers(polarization):
    """Return the four chi regions used for one polarization."""
    polarization = float(polarization) % 180

    if np.isclose(polarization, 0) or np.isclose(
        polarization,
        90,
    ):
        return [90, 180, -90, 0]

    if np.isclose(polarization, 45) or np.isclose(
        polarization,
        135,
    ):
        return [45, 135, -45, -135]

    raise ValueError(
        "Beam-center fitting currently supports polarizations "
        "0°, 45°, 90°, and 135°"
    )


def gaussian(x, amplitude, center, sigma, offset):
    """Evaluate a Gaussian peak plus a constant offset."""
    return (
        amplitude
        * np.exp(-((x - center) ** 2) / (2 * sigma**2))
        + offset
    )


def fit_peak(q, intensity, q_target, q_window_width):
    """Fit a Gaussian peak near the configured q target."""
    q = np.asarray(q)
    intensity = np.asarray(intensity)

    selected = (
        (q > q_target - q_window_width)
        & (q < q_target + q_window_width)
        & np.isfinite(q)
        & np.isfinite(intensity)
    )
    q_fit = q[selected]
    intensity_fit = intensity[selected]

    if len(q_fit) < 5:
        return np.nan, (q_fit, intensity_fit), None

    initial_parameters = [
        intensity_fit.max() - intensity_fit.min(),
        q_fit[np.argmax(intensity_fit)],
        (q_fit.max() - q_fit.min()) / 10,
        intensity_fit.min(),
    ]

    try:
        fitted_parameters, _ = curve_fit(
            gaussian,
            q_fit,
            intensity_fit,
            p0=initial_parameters,
        )
    except (RuntimeError, ValueError):
        return np.nan, (q_fit, intensity_fit), None

    fitted_center = float(fitted_parameters[1])
    q_smooth = np.linspace(q_fit.min(), q_fit.max(), 300)
    intensity_smooth = gaussian(q_smooth, *fitted_parameters)

    return (
        fitted_center,
        (q_fit, intensity_fit),
        (q_smooth, intensity_smooth),
    )


def locate_peak_q(
    clean_data,
    polarization,
    chi_halfwidth,
    energy_value,
    q_target,
    q_window_width,
):
    """Locate the fitted q peak in four azimuthal regions."""
    if energy_value == "all":
        selected_data = clean_data.mean("energy")
    else:
        selected_data = clean_data.sel(
            energy=energy_value,
            method="nearest",
        )

    intensity_curves = {}
    fit_curves = {}
    peaks = {}

    for center in azimuth_centers(polarization):
        intensity_vs_q = get_chi_slice(
            selected_data,
            center,
            chi_halfwidth,
        ).mean("chi")

        q_values = intensity_vs_q["q"].values
        intensity_values = intensity_vs_q.values
        peak_q, fit_data, fit_curve = fit_peak(
            q_values,
            intensity_values,
            q_target,
            q_window_width,
        )

        if fit_curve is None:
            q_smooth = None
            intensity_smooth = None
        else:
            q_smooth, intensity_smooth = fit_curve

        intensity_curves[center] = {
            "q": q_values,
            "intensity": intensity_values,
        }
        fit_curves[center] = {
            "q_fit": fit_data[0],
            "intensity_fit": fit_data[1],
            "q_smooth": q_smooth,
            "intensity_smooth": intensity_smooth,
        }
        peaks[center] = peak_q

    return selected_data, intensity_curves, fit_curves, peaks


def prepare_fit_inputs(data, final_mask, energy_value):
    """Select one energy and its matching 2D integrator mask."""
    loaded_data = data.unstack("system").transpose(
        "pix_y",
        "pix_x",
        "energy",
    )
    loaded_mask = final_mask.unstack("system").transpose(
        "pix_y",
        "pix_x",
        "energy",
    )

    energy_values = loaded_data["energy"].values
    energy_index = int(
        np.argmin(np.abs(energy_values - energy_value))
    )
    actual_energy = float(energy_values[energy_index])

    selected_data = loaded_data.isel(
        energy=slice(energy_index, energy_index + 1)
    )
    selected_mask = loaded_mask.isel(
        energy=energy_index
    ).transpose("pix_y", "pix_x")

    expected_shape = (
        selected_data.sizes["pix_y"],
        selected_data.sizes["pix_x"],
    )
    if selected_mask.shape != expected_shape:
        raise ValueError(
            f"Integrator mask shape {selected_mask.shape} "
            f"does not match image shape {expected_shape}"
        )

    return selected_data, selected_mask, actual_energy


def reflect_integrated_mask(masked_pixels, reflection_mode):
    """Reflect a post-integration mask around configured symmetries."""
    if reflection_mode in (None, "none"):
        return masked_pixels

    chi_shifts = {
        "twofold": [180],
        "fourfold": [0, 90, 180, -90],
        "eightfold": [
            0,
            45,
            90,
            135,
            180,
            -45,
            -90,
            -135,
        ],
    }
    if reflection_mode not in chi_shifts:
        raise ValueError(
            f"Unknown reflection mode: {reflection_mode!r}"
        )

    mask_2d = masked_pixels.any(dim="energy")
    chi_values = mask_2d["chi"].values
    q_values = mask_2d["q"].values
    masked_chi_indexes, masked_q_indexes = np.where(
        mask_2d.values
    )
    masked_chi = chi_values[masked_chi_indexes]
    masked_q = q_values[masked_q_indexes]

    reflected_values = np.zeros_like(
        mask_2d.values,
        dtype=bool,
    )

    for shift in chi_shifts[reflection_mode]:
        shifted_chi = wrap_chi(masked_chi + shift)

        for chi_value, q_value in zip(
            shifted_chi,
            masked_q,
        ):
            chi_index = np.argmin(
                np.abs(chi_values - chi_value)
            )
            q_index = np.argmin(
                np.abs(q_values - q_value)
            )
            reflected_values[chi_index, q_index] = True

    full_mask_2d = mask_2d.copy(
        data=mask_2d.values | reflected_values
    )
    full_mask = full_mask_2d.broadcast_like(
        masked_pixels
    ).transpose(*masked_pixels.dims)

    return full_mask


def integrate_single_energy(integrator, data):
    """Integrate one energy without a one-item progress bar."""
    if data.sizes.get("energy") != 1:
        raise ValueError(
            "Expected exactly one energy for beam-center fitting"
        )

    energy_value = float(data["energy"].values[0])
    image = data.isel(energy=0)
    integrated = integrator.integrateImageStack(image)

    if "energy" not in integrated.dims:
        integrated = integrated.expand_dims(
            energy=[energy_value]
        )

    return integrated.transpose("energy", "chi", "q")


def integrate_at_center(
    selected_data,
    selected_mask,
    beam_center_x,
    beam_center_y,
    postmask_threshold,
    reflection_mode,
    apply_postmask,
):
    """Integrate one energy using one candidate beam center."""
    mask_values = np.asarray(
        selected_mask.values,
        dtype=bool,
    )

    integrator = phs.integrate.PFEnergySeriesIntegrator(
        geomethod="template_xr",
        template_xr=selected_data,
        integration_method="csr",
        maskmethod="numpy",
        mask=mask_values,
    )
    integrator.ni_beamcenter_x = beam_center_x
    integrator.ni_beamcenter_y = beam_center_y
    integrator.mask = mask_values

    remeshed = integrate_single_energy(
        integrator,
        selected_data,
    )

    if not apply_postmask:
        return {
            "remeshed": remeshed,
            "clean": remeshed,
            "remeshed_mask": None,
            "remeshed_ones": None,
            "source_fraction": None,
            "postmask": None,
            "symmetry_mask": None,
            "symmetry_clean": remeshed,
        }

    eligible_values = np.broadcast_to(
        (~mask_values)[..., np.newaxis],
        selected_data.shape,
    ).copy()
    eligible_data = xr.DataArray(
        eligible_values,
        dims=selected_data.dims,
        coords={
            dimension: selected_data.coords[dimension]
            for dimension in selected_data.dims
        },
    )

    integrator.mask = np.zeros_like(
        mask_values,
        dtype=bool,
    )
    remeshed_mask = integrate_single_energy(
        integrator,
        eligible_data,
    )

    ones_input = xr.ones_like(
        selected_data,
        dtype=float,
    )
    remeshed_ones = integrate_single_energy(
        integrator,
        ones_input,
    )

    source_fraction = remeshed_mask / remeshed_ones
    postmask = (
        (source_fraction < postmask_threshold)
        | source_fraction.isnull()
    )
    ordered_postmask = postmask.transpose(
        *remeshed.dims
    )
    if ordered_postmask.shape != remeshed.shape:
        raise ValueError(
            "Postmask shape does not match remeshed data"
        )

    clean = remeshed.copy()
    clean.values[ordered_postmask.values] = np.nan

    symmetry_mask = reflect_integrated_mask(
        postmask,
        reflection_mode,
    )
    ordered_symmetry_mask = symmetry_mask.transpose(
        *remeshed.dims
    )
    if ordered_symmetry_mask.shape != remeshed.shape:
        raise ValueError(
            "Symmetry-mask shape does not match remeshed data"
        )

    symmetry_clean = remeshed.copy()
    symmetry_clean.values[
        ordered_symmetry_mask.values
    ] = np.nan

    return {
        "remeshed": remeshed,
        "clean": clean,
        "remeshed_mask": remeshed_mask,
        "remeshed_ones": remeshed_ones,
        "source_fraction": source_fraction,
        "postmask": postmask,
        "symmetry_mask": symmetry_mask,
        "symmetry_clean": symmetry_clean,
    }


def beam_center_cost(
    selected_data,
    selected_mask,
    polarization,
    beam_center_x,
    beam_center_y,
    fit_config,
):
    """Measure disagreement among the four fitted q peaks."""
    integration = integrate_at_center(
        selected_data=selected_data,
        selected_mask=selected_mask,
        beam_center_x=beam_center_x,
        beam_center_y=beam_center_y,
        postmask_threshold=fit_config[
            "postmask_threshold"
        ],
        reflection_mode=None,
        apply_postmask=fit_config["apply_postmask"],
    )

    _, _, _, peaks = locate_peak_q(
        clean_data=integration["clean"],
        polarization=polarization,
        chi_halfwidth=fit_config[
            "chi_half_width_degrees"
        ],
        energy_value="all",
        q_target=fit_config["q_target"],
        q_window_width=fit_config["q_window_width"],
    )

    peak_values = np.asarray(
        list(peaks.values()),
        dtype=float,
    )
    peak_values = peak_values[np.isfinite(peak_values)]

    if len(peak_values) < 2:
        return np.nan

    return float(
        np.mean(
            (peak_values - np.mean(peak_values)) ** 2
        )
    )


def search_values(center, radius, step):
    """Create an inclusive, stable search axis."""
    if radius < 0:
        raise ValueError("Search radius cannot be negative")
    if step <= 0:
        raise ValueError("Search step must be positive")

    values = np.arange(
        center - radius,
        center + radius + step / 2,
        step,
    )
    return np.round(values, 6)


def evaluate_cost_grid(
    selected_data,
    selected_mask,
    polarization,
    x_values,
    y_values,
    fit_config,
    n_jobs,
    progress_description,
):
    """Evaluate every candidate center with one overall progress bar."""
    total_candidates = len(x_values) * len(y_values)

    tasks = (
        delayed(beam_center_cost)(
            selected_data=selected_data,
            selected_mask=selected_mask,
            polarization=polarization,
            beam_center_x=x_value,
            beam_center_y=y_value,
            fit_config=fit_config,
        )
        for y_value in y_values
        for x_value in x_values
    )

    costs = []

    with parallel_backend("threading"):
        cost_iterator = Parallel(
            n_jobs=n_jobs,
            return_as="generator",
        )(tasks)

        with tqdm(
            total=total_candidates,
            desc=progress_description,
            unit="center",
        ) as progress:
            for cost in cost_iterator:
                costs.append(cost)
                progress.update(1)

    return xr.DataArray(
        np.asarray(costs).reshape(
            len(y_values),
            len(x_values),
        ),
        coords={
            "y": y_values,
            "x": x_values,
        },
        dims=("y", "x"),
        name="beam_center_cost",
    )


def best_grid_point(cost_grid):
    """Return the lowest finite-cost point in a search grid."""
    values = np.asarray(cost_grid.values)

    if not np.any(np.isfinite(values)):
        raise ValueError(
            "No finite beam-center costs were produced"
        )

    flat_index = int(np.nanargmin(values))
    y_index, x_index = np.unravel_index(
        flat_index,
        values.shape,
    )

    return {
        "x": float(cost_grid["x"].values[x_index]),
        "y": float(cost_grid["y"].values[y_index]),
        "cost": float(values[y_index, x_index]),
    }


def _fit_history_entry(
    result,
    processing,
    median_suggestion,
    pass_id,
    started_at_utc,
):
    """Capture a compact record of one fit and its actual mask."""
    mask = result["selected_mask"]
    actual_energy = float(result["actual_fit_energy"])

    if float(mask["energy"].item()) != actual_energy:
        raise ValueError("Fit data and mask energies do not match")

    settings = mask.attrs.get("center_mask_settings")
    summaries = mask.attrs.get("center_mask_summaries")

    if settings is None or summaries is None:
        raise ValueError(
            "Center-mask metadata is missing. "
            "Rebuild the center and final masks before fitting."
        )

    matching = [
        summary
        for summary in summaries
        if float(summary["energy"]) == actual_energy
    ]

    if len(matching) != 1:
        raise ValueError(
            "Expected one center-mask summary for the fit energy"
        )

    return deepcopy({
        "pass_id": pass_id,
        "started_at_utc": started_at_utc,
        "scan_id": int(result["scan_id"]),
        "polarization_deg": float(result["polarization_deg"]),
        "actual_fit_energy": actual_energy,
        "fit_settings": processing["beam_center_fit"],
        "center_mask_settings": settings,
        "center_mask_summary": matching[0],
        "coarse_best": result["coarse_best"],
        "fine_best": result["fine_best"],
        "median_center_suggestion": median_suggestion,
    })


def fit_scan_beam_center(
    scan_record,
    final_mask,
    polarization,
    processing,
):
    """Fit one polarization using coarse and fine searches."""
    fit_config = processing["beam_center_fit"]
    n_jobs = int(
        processing.get("runtime", {}).get(
            "beam_center_jobs",
            1,
        )
    )

    selected_data, selected_mask, actual_energy = (
        prepare_fit_inputs(
            data=scan_record["data"],
            final_mask=final_mask,
            energy_value=fit_config["fit_energy"],
        )
    )

    coarse_x = search_values(
        fit_config["initial_center_x"],
        fit_config["coarse_search_radius"],
        fit_config["coarse_step"],
    )
    coarse_y = search_values(
        fit_config["initial_center_y"],
        fit_config["coarse_search_radius"],
        fit_config["coarse_step"],
    )

    print(
        f"Scan {scan_record['scan_id']}, "
        f"polarization={polarization}°: "
        f"starting coarse search at "
        f"{actual_energy:g} eV"
    )
    coarse_grid = evaluate_cost_grid(
        selected_data=selected_data,
        selected_mask=selected_mask,
        polarization=polarization,
        x_values=coarse_x,
        y_values=coarse_y,
        fit_config=fit_config,
        n_jobs=n_jobs,
        progress_description=(
            f"Scan {scan_record['scan_id']} coarse"
        ),
    )
    coarse_best = best_grid_point(coarse_grid)

    fine_x = search_values(
        coarse_best["x"],
        fit_config["fine_search_radius"],
        fit_config["fine_step"],
    )
    fine_y = search_values(
        coarse_best["y"],
        fit_config["fine_search_radius"],
        fit_config["fine_step"],
    )

    print(
        f"Scan {scan_record['scan_id']}: "
        f"coarse center=({coarse_best['x']}, "
        f"{coarse_best['y']}); starting fine search"
    )
    fine_grid = evaluate_cost_grid(
        selected_data=selected_data,
        selected_mask=selected_mask,
        polarization=polarization,
        x_values=fine_x,
        y_values=fine_y,
        fit_config=fit_config,
        n_jobs=n_jobs,
        progress_description=(
            f"Scan {scan_record['scan_id']} fine"
        ),
    )
    fine_best = best_grid_point(fine_grid)

    final_integration = integrate_at_center(
        selected_data=selected_data,
        selected_mask=selected_mask,
        beam_center_x=fine_best["x"],
        beam_center_y=fine_best["y"],
        postmask_threshold=fit_config[
            "postmask_threshold"
        ],
        reflection_mode=fit_config["reflection_mode"],
        apply_postmask=fit_config["apply_postmask"],
    )
    (
        selected_remesh,
        intensity_curves,
        fit_curves,
        peaks,
    ) = locate_peak_q(
        clean_data=final_integration["clean"],
        polarization=polarization,
        chi_halfwidth=fit_config[
            "chi_half_width_degrees"
        ],
        energy_value="all",
        q_target=fit_config["q_target"],
        q_window_width=fit_config["q_window_width"],
    )

    print(
        f"Scan {scan_record['scan_id']}: "
        f"fine center=({fine_best['x']}, "
        f"{fine_best['y']})"
    )

    return {
        "scan_id": int(scan_record["scan_id"]),
        "polarization_deg": polarization,
        "actual_fit_energy": actual_energy,
        "selected_data": selected_data,
        "selected_mask": selected_mask,
        "coarse_grid": coarse_grid,
        "coarse_best": coarse_best,
        "fine_grid": fine_grid,
        "fine_best": fine_best,
        "integration": final_integration,
        "selected_remesh": selected_remesh,
        "intensity_curves": intensity_curves,
        "fit_curves": fit_curves,
        "peaks": peaks,
    }


def fit_scan_set_beam_centers(scans, processing):
    """Store each scan's latest fit and append its pass history."""
    if not scans:
        raise ValueError("No scans were supplied for beam-center fitting")

    processing_snapshot = deepcopy(processing)
    pass_id = uuid4().hex
    started_at_utc = datetime.now(timezone.utc).isoformat()
    histories = {}

    for polarization, scan_record in scans.items():
        if "final_mask" not in scan_record:
            raise ValueError(
                f"Scan {scan_record['scan_id']} has no final mask"
            )

        mask = scan_record["final_mask"]
        for key in ("center_mask_settings", "center_mask_summaries"):
            if key not in mask.attrs:
                raise ValueError(
                    "Center-mask metadata is missing. "
                    "Rebuild the center and final masks before fitting."
                )

        previous_fit = scan_record.get("beam_center_fit", {})
        history = previous_fit.get("history", [])

        if not isinstance(history, list):
            raise ValueError("Beam-center history must be a list")

        histories[polarization] = deepcopy(history)

    pending_results = {}

    for polarization, scan_record in scans.items():
        pending_results[polarization] = fit_scan_beam_center(
            scan_record=scan_record,
            final_mask=scan_record["final_mask"],
            polarization=polarization,
            processing=processing_snapshot,
        )

    median_suggestion = {
        "x": float(np.median([
            result["fine_best"]["x"]
            for result in pending_results.values()
        ])),
        "y": float(np.median([
            result["fine_best"]["y"]
            for result in pending_results.values()
        ])),
    }

    for polarization, result in pending_results.items():
        histories[polarization].append(
            _fit_history_entry(
                result=result,
                processing=processing_snapshot,
                median_suggestion=median_suggestion,
                pass_id=pass_id,
                started_at_utc=started_at_utc,
            )
        )
        result["history"] = histories[polarization]

    # Update scan records only after the whole pass succeeds.
    for polarization, result in pending_results.items():
        scans[polarization]["beam_center_fit"] = result

    return median_suggestion


def get_shared_beam_center(processing):
    """Return the manually reviewed shared beam center."""
    if "shared_beam_center" not in processing:
        raise ValueError(
            "No reviewed shared beam center is configured. "
            "Review the per-scan fits and add "
            "[shared_beam_center] to the processing TOML."
        )

    settings = processing["shared_beam_center"]

    try:
        center_x = float(settings["center_x"])
        center_y = float(settings["center_y"])
    except KeyError as error:
        raise ValueError(
            "The shared beam center requires center_x "
            "and center_y"
        ) from error

    return {
        "x": center_x,
        "y": center_y,
    }
