import numpy as np
import PyHyperScattering as phs
import xarray as xr
from tqdm.auto import tqdm

from rsoxs_unified_workup.beam_center import (
    get_shared_beam_center,
)


def calculate_q_cap(
    scan_record,
    shared_center,
    outer_radius_pixels,
):
    """Calculate the fully usable maximum q at each energy."""
    data = scan_record["data"]
    energies = np.asarray(data.energy.values, dtype=float)
    detector_height = data.sizes["pix_y"]
    detector_width = data.sizes["pix_x"]

    pixel_size_angstrom = (
        float(scan_record["pixel_size"]) * 1e10
    )
    detector_distance_angstrom = (
        float(scan_record["sdd"]) * 1e7
    )

    x = (
        np.arange(detector_width) - shared_center["x"]
    ) * pixel_size_angstrom
    y = (
        np.arange(detector_height) - shared_center["y"]
    ) * pixel_size_angstrom
    x_grid, y_grid = np.meshgrid(x, y)

    radius_map = np.sqrt(x_grid**2 + y_grid**2)
    two_theta = np.arctan2(
        radius_map,
        detector_distance_angstrom,
    )
    radius_limit = (
        outer_radius_pixels * pixel_size_angstrom
    )
    inside_radius = radius_map <= radius_limit

    q_maximum = []

    for energy in energies:
        wavelength_angstrom = (
            1239.841984 / energy
        ) * 10
        q_map = (
            4
            * np.pi
            / wavelength_angstrom
            * np.sin(two_theta / 2)
        )
        q_maximum.append(
            float(np.max(q_map[inside_radius]))
        )

    return xr.DataArray(
        q_maximum,
        coords={"energy": energies},
        dims=("energy",),
        name="q_cap",
        attrs={
            "units": "angstrom^-1",
            "outer_radius_pixels": outer_radius_pixels,
            "beam_center_x": shared_center["x"],
            "beam_center_y": shared_center["y"],
        },
    )


def prepare_integration_inputs(scan_record, final_mask):
    """Prepare the full energy stack and energy-specific masks."""
    loaded_data = scan_record["data"].unstack(
        "system"
    ).transpose("pix_y", "pix_x", "energy")
    loaded_masks = final_mask.unstack(
        "system"
    ).transpose("pix_y", "pix_x", "energy")

    expected_shape = (
        loaded_data.sizes["pix_y"],
        loaded_data.sizes["pix_x"],
        loaded_data.sizes["energy"],
    )
    if loaded_masks.shape != expected_shape:
        raise ValueError(
            f"Mask stack shape {loaded_masks.shape} does "
            f"not match data shape {expected_shape}"
        )

    if not np.allclose(
        loaded_data.energy.values,
        loaded_masks.energy.values,
    ):
        raise ValueError(
            "Data and mask energy coordinates do not match"
        )

    return loaded_data, loaded_masks


def build_energy_series_integrator(
    loaded_data,
    initial_mask,
    shared_center,
):
    """Build a CSR integrator with one q grid for all energies."""
    mask_values = np.asarray(
        initial_mask.values,
        dtype=bool,
    )
    integrator = phs.integrate.PFEnergySeriesIntegrator(
        geomethod="template_xr",
        template_xr=loaded_data,
        integration_method="csr",
        maskmethod="numpy",
        mask=mask_values,
    )
    integrator.ni_beamcenter_x = shared_center["x"]
    integrator.ni_beamcenter_y = shared_center["y"]

    energies = np.asarray(
        loaded_data.energy.values,
        dtype=float,
    )
    integrator.setupIntegrators(energies)
    integrator.setupDestQ(energies)

    return integrator


def integrate_single_frame(
    integrator,
    image,
    energy,
):
    """Integrate fixed chi bins while retaining existing q resampling."""
    if (
        integrator.do_1d_integration
        or integrator.use_log_ish_binning
        or integrator.return_sigma
    ):
        raise ValueError(
            "This helper requires 2D integration, linear q, "
            "and return_sigma=False"
        )

    energy = float(energy)
    image_2d = image.transpose("pix_y", "pix_x")
    mask = np.asarray(integrator.mask, dtype=bool)

    if mask.shape != image_2d.shape:
        raise ValueError("Mask and image shapes do not match")

    chi_bin_count = 360
    chi_range = (-180.0, 180.0)

    frame = integrator.createIntegrator(energy).integrate2d(
        image_2d.values,
        npt_rad=integrator.npts,
        npt_azim=chi_bin_count,
        azimuth_range=chi_range,
        correctSolidAngle=integrator.correctSolidAngle,
        error_model="azimuthal",
        dummy=-8675309 if integrator.maskToNan else 0,
        mask=mask,
        unit="q_A^-1",
        method=integrator.integration_method,
    )

    intensity = frame.intensity.copy()

    if integrator.maskToNan:
        intensity[intensity == -8675309] = np.nan

    integrated = xr.DataArray(
        intensity[np.newaxis, ...],
        dims=("energy", "chi", "q"),
        coords={
            "energy": [energy],
            "chi": frame.azimuthal,
            "q": frame.radial,
            "image_num": 0,
        },
        attrs=dict(image.attrs),
    )

    integrated.attrs["chi_bin_count"] = chi_bin_count
    integrated.attrs["chi_range_degrees"] = list(chi_range)

    return integrated.interp(
        q=integrator.dest_q,
    ).transpose("energy", "chi", "q")


def apply_positional_mask(data, mask):
    """Apply a mask only when dimensions and coordinates match."""
    ordered_mask = mask.transpose(*data.dims)

    if ordered_mask.shape != data.shape:
        raise ValueError(
            f"Mask shape {ordered_mask.shape} does not "
            f"match data shape {data.shape}"
        )

    xr.align(
        data,
        ordered_mask,
        join="exact",
        copy=False,
    )

    masked_data = data.copy()
    masked_data.values[ordered_mask.values] = np.nan
    return masked_data


def integrate_scan(scan_record, processing):
    """Integrate one scan using its stored mask and processing settings."""
    final_mask = scan_record["final_mask"]
    shared_center = get_shared_beam_center(processing)
    outer_radius_pixels = float(
        processing["center_mask"]["outer_usable_radius"]
    )
    postmask_threshold = float(
        processing["integration"]["postmask_threshold"]
    )
    if not 0 <= postmask_threshold <= 1:
        raise ValueError(
            "postmask_threshold must be between 0 and 1"
        )

    loaded_data, loaded_masks = (
        prepare_integration_inputs(
            scan_record,
            final_mask,
        )
    )
    integrator = build_energy_series_integrator(
        loaded_data=loaded_data,
        initial_mask=loaded_masks.isel(energy=0),
        shared_center=shared_center,
    )

    remeshed_frames = []
    remeshed_mask_frames = []
    remeshed_ones_frames = []
    source_fraction_frames = []
    postmask_frames = []
    clean_frames = []

    energies = loaded_data.energy.values

    for energy_index, energy in enumerate(
        tqdm(
            energies,
            total=len(energies),
            desc=(
                f"Scan {scan_record['scan_id']}, "
                f"polarization={scan_record['polarization_deg']:g}°"
            ),
            unit="energy",
        )
    ):
        energy = float(energy)
        image = loaded_data.isel(energy=energy_index)
        mask = loaded_masks.isel(
            energy=energy_index
        ).transpose("pix_y", "pix_x")
        mask_values = np.asarray(
            mask.values,
            dtype=bool,
        )

        integrator.mask = mask_values
        remeshed = integrate_single_frame(
            integrator,
            image,
            energy,
        )

        eligible_image = image.copy(
            data=(~mask_values).astype(float)
        )
        integrator.mask = np.zeros_like(
            mask_values,
            dtype=bool,
        )
        remeshed_mask = integrate_single_frame(
            integrator,
            eligible_image,
            energy,
        )

        ones_image = xr.ones_like(
            image,
            dtype=float,
        )
        remeshed_ones = integrate_single_frame(
            integrator,
            ones_image,
            energy,
        )

        source_fraction = (
            remeshed_mask / remeshed_ones
        )
        postmask = (
            (source_fraction < postmask_threshold)
            | source_fraction.isnull()
        )
        clean_frame = apply_positional_mask(
            remeshed,
            postmask,
        )

        remeshed_frames.append(remeshed)
        remeshed_mask_frames.append(remeshed_mask)
        remeshed_ones_frames.append(remeshed_ones)
        source_fraction_frames.append(source_fraction)
        postmask_frames.append(postmask)
        clean_frames.append(clean_frame)

    def combine(frames):
        return xr.concat(
            frames,
            dim="energy",
            join="exact",
        )

    remeshed = combine(remeshed_frames)
    remeshed_mask = combine(remeshed_mask_frames)
    remeshed_ones = combine(remeshed_ones_frames)
    source_fraction = combine(
        source_fraction_frames
    )
    postmask = combine(postmask_frames)
    clean_uncapped_q = combine(clean_frames)

    q_cap = calculate_q_cap(
        scan_record=scan_record,
        shared_center=shared_center,
        outer_radius_pixels=outer_radius_pixels,
    )
    clean = clean_uncapped_q.where(
        clean_uncapped_q["q"] <= q_cap
    )

    return {
        "beam_center_x": shared_center["x"],
        "beam_center_y": shared_center["y"],
        "postmask_threshold": postmask_threshold,
        "q_cap": q_cap,
        "remeshed": remeshed,
        "remeshed_mask": remeshed_mask,
        "remeshed_ones": remeshed_ones,
        "source_fraction": source_fraction,
        "postmask": postmask,
        "clean_uncapped_q": clean_uncapped_q,
        "clean": clean,
    }


def integrate_dark_scan(scan_record, processing):
    """Integrate the first dark frame using this scan's stored results."""
    final_mask = scan_record["final_mask"]
    shared_center = get_shared_beam_center(processing)
    integration_result = scan_record["integration"]
    darks = scan_record["darks"]

    if "time" in darks.dims:
        dark_frame = darks.isel(time=0)
        dark_frame_index = 0
    else:
        dark_frame = darks
        dark_frame_index = None

    dark_frame = dark_frame.transpose(
        "pix_y",
        "pix_x",
    )
    loaded_data, loaded_masks = (
        prepare_integration_inputs(
            scan_record,
            final_mask,
        )
    )
    integrator = build_energy_series_integrator(
        loaded_data=loaded_data,
        initial_mask=loaded_masks.isel(energy=0),
        shared_center=shared_center,
    )

    remeshed_frames = []
    clean_frames = []

    energies = loaded_data.energy.values

    for energy_index, energy in enumerate(
        tqdm(
            energies,
            total=len(energies),
            desc=(
                f"Dark scan {scan_record['scan_id']}, "
                f"polarization={scan_record['polarization_deg']:g}°"
            ),
            unit="energy",
        )
    ):
        energy = float(energy)

        mask = loaded_masks.isel(
            energy=energy_index
        ).transpose("pix_y", "pix_x")
        integrator.mask = np.asarray(
            mask.values,
            dtype=bool,
        )

        dark_image = dark_frame.assign_coords(
            energy=energy
        )
        dark_image.attrs = loaded_data.attrs

        remeshed = integrate_single_frame(
            integrator,
            dark_image,
            energy,
        )
        postmask = integration_result[
            "postmask"
        ].isel(
            energy=slice(
                energy_index,
                energy_index + 1,
            )
        )
        clean = apply_positional_mask(
            remeshed,
            postmask,
        )

        remeshed_frames.append(remeshed)
        clean_frames.append(clean)

    darks_remeshed = xr.concat(
        remeshed_frames,
        dim="energy",
        join="exact",
    )
    darks_clean = xr.concat(
        clean_frames,
        dim="energy",
        join="exact",
    )

    return {
        "dark_frame_index": dark_frame_index,
        "original_dark": dark_frame,
        "darks_remeshed": darks_remeshed,
        "darks_remeshed_mask": integration_result[
            "remeshed_mask"
        ],
        "darks_remeshed_ones": integration_result[
            "remeshed_ones"
        ],
        "darks_clean": darks_clean,
    }
