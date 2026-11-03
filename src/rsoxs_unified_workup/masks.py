import numpy as np
from skimage.draw import polygon2mask
import xarray as xr
from copy import deepcopy


def rotated_rectangle(
    center_x,
    center_y,
    length,
    width,
    angle_degrees,
):
    """Return the four vertices of a rotated rectangle."""
    angle = np.deg2rad(angle_degrees)
    half_length = length / 2
    half_width = width / 2

    corners = np.array(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ]
    )

    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    vertices = corners @ rotation.T
    vertices[:, 0] += center_x
    vertices[:, 1] += center_y

    return vertices


def create_beamstop_mask(detector_shape, settings):
    """Create a beamstop mask from configured rectangle geometry."""
    vertices = rotated_rectangle(
        center_x=settings["center_x"],
        center_y=settings["center_y"],
        length=settings["length"],
        width=settings["width"],
        angle_degrees=settings["angle_degrees"],
    )

    polygon_yx = [
        (y, x)
        for x, y in np.round(vertices).astype(int)
    ]
    mask = polygon2mask(detector_shape, polygon_yx)

    return mask, vertices


def create_border_mask(detector_shape, settings):
    """Create a configurable detector-border mask."""
    height, width = detector_shape

    left = int(settings["left_pixels"])
    right = int(settings["right_pixels"])
    bottom = int(settings["bottom_pixels"])
    top = int(settings["top_pixels"])

    widths = {
        "left_pixels": left,
        "right_pixels": right,
        "bottom_pixels": bottom,
        "top_pixels": top,
    }

    if any(value < 0 for value in widths.values()):
        raise ValueError("Detector border widths cannot be negative")

    if left + right >= width:
        raise ValueError(
            "Left and right borders cover the detector width"
        )

    if bottom + top >= height:
        raise ValueError(
            "Bottom and top borders cover the detector height"
        )

    mask = np.zeros(detector_shape, dtype=bool)

    if left:
        mask[:, :left] = True

    if right:
        mask[:, -right:] = True

    if bottom:
        mask[:bottom, :] = True

    if top:
        mask[-top:, :] = True

    return mask


def combine_static_masks(
    dark_masks,
    beamstop_mask,
    border_mask,
    union_across_polarizations=True,
):
    """Combine dark, beamstop, and detector-border masks."""
    if not dark_masks:
        raise ValueError("No dark masks were supplied")

    expected_shape = beamstop_mask.shape

    if border_mask.shape != expected_shape:
        raise ValueError(
            "Beamstop and border masks have different shapes"
        )

    for polarization, dark_mask in dark_masks.items():
        if dark_mask.shape != expected_shape:
            raise ValueError(
                f"Dark mask for polarization {polarization}° "
                f"has shape {dark_mask.shape}; "
                f"expected {expected_shape}"
            )

    if union_across_polarizations:
        shared_dark_mask = np.any(
            [
                dark_mask.values
                for dark_mask in dark_masks.values()
            ],
            axis=0,
        )

    static_masks = {}

    for polarization, dark_mask in dark_masks.items():
        if union_across_polarizations:
            dark_values = shared_dark_mask
        else:
            dark_values = dark_mask.values

        combined_values = (
            dark_values
            | beamstop_mask
            | border_mask
        )

        static_masks[polarization] = dark_mask.copy(
            data=combined_values
        )

    return static_masks


def apply_static_masks(scans, static_masks):
    """Apply static masks and store masked data as NaN."""
    for polarization, scan_data in scans.items():
        if polarization not in static_masks:
            raise ValueError(
                f"No static mask for polarization "
                f"{polarization}°"
            )

        static_mask = static_masks[polarization]

        scan_data["static_mask"] = static_mask
        scan_data["masked_data_static"] = (
            scan_data["data"].where(~static_mask)
        )

    return scans


def build_saturation_masks(scans):
    """Union negative saturated pixels across polarizations."""
    first_scan = next(iter(scans.values()))
    reference = first_scan["masked_data_static"]
    reference_energies = reference.energy.values

    combined = xr.zeros_like(reference, dtype=bool)

    for polarization, scan_data in scans.items():
        masked_data = scan_data["masked_data_static"]

        if not np.array_equal(
            masked_data.energy.values,
            reference_energies,
        ):
            raise ValueError(
                f"Energy values for polarization "
                f"{polarization}° do not match the scan set"
            )

        saturated = (masked_data < 0).fillna(False)
        combined = combined | saturated

    combined.name = "saturation_mask"

    return combined


def build_center_masks(
    saturation_masks,
    beamstop_vertices,
    settings,
):
    """Build circular center masks for every energy."""
    center_x = float(settings["beam_center_x_guess"])
    center_y = float(settings["beam_center_y_guess"])
    search_width = float(
        settings["saturation_search_width"]
    )

    energy_dimension = saturation_masks.energy.dims[0]
    detector_height = saturation_masks.sizes["pix_y"]
    detector_width = saturation_masks.sizes["pix_x"]

    half_width = search_width / 2
    x_min = max(0, int(center_x - half_width))
    x_max = min(
        detector_width,
        int(center_x + half_width),
    )
    y_min = max(0, int(center_y - half_width))
    y_max = min(
        detector_height,
        int(center_y + half_width),
    )

    visible_vertices = np.array(
        [
            (x, y)
            for x, y in beamstop_vertices
            if x >= 0 and y >= 0
        ]
    )

    if visible_vertices.size == 0:
        raise ValueError(
            "No nonnegative beamstop vertices are available"
        )

    beamstop_distances = np.sqrt(
        (visible_vertices[:, 0] - center_x) ** 2
        + (visible_vertices[:, 1] - center_y) ** 2
    )
    farthest_beamstop_index = int(
        np.argmax(beamstop_distances)
    )
    farthest_beamstop = visible_vertices[
        farthest_beamstop_index
    ]
    beamstop_radius = float(
        beamstop_distances[farthest_beamstop_index]
    )

    y_grid, x_grid = np.indices(
        (detector_height, detector_width)
    )
    distance_grid = np.sqrt(
        (x_grid - center_x) ** 2
        + (y_grid - center_y) ** 2
    )

    center_mask_values = np.zeros(
        saturation_masks.shape,
        dtype=bool,
    )
    energy_axis = saturation_masks.get_axis_num(
        energy_dimension
    )
    summaries = []

    for index in range(
        saturation_masks.sizes[energy_dimension]
    ):
        saturation = saturation_masks.isel(
            {energy_dimension: index}
        )
        cropped = saturation.values[
            y_min:y_max,
            x_min:x_max,
        ]
        cropped_y, cropped_x = np.where(cropped)

        selected_radius = beamstop_radius
        selected_point = farthest_beamstop
        radius_source = "beamstop"

        if len(cropped_x):
            full_x = cropped_x + x_min
            full_y = cropped_y + y_min
            saturated_points = np.column_stack(
                [full_x, full_y]
            )
            saturated_distances = np.sqrt(
                (saturated_points[:, 0] - center_x) ** 2
                + (saturated_points[:, 1] - center_y) ** 2
            )
            farthest_saturated_index = int(
                np.argmax(saturated_distances)
            )
            saturation_radius = float(
                saturated_distances[
                    farthest_saturated_index
                ]
            )

            if saturation_radius >= beamstop_radius:
                selected_radius = saturation_radius
                selected_point = saturated_points[
                    farthest_saturated_index
                ]
                radius_source = "saturation"

        center_mask = distance_grid < selected_radius
        indexer = [slice(None)] * saturation_masks.ndim
        indexer[energy_axis] = index
        center_mask_values[tuple(indexer)] = center_mask

        summaries.append(
            {
                "energy": float(
                    saturation_masks.energy.values[index]
                ),
                "radius": selected_radius,
                "radius_source": radius_source,
                "farthest_x": float(selected_point[0]),
                "farthest_y": float(selected_point[1]),
                "search_bounds": (
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                ),
            }
        )

    center_masks = saturation_masks.copy(
        data=center_mask_values
    )
    center_masks.name = "center_mask"
    center_masks.attrs["center_mask_settings"] = deepcopy(settings)
    center_masks.attrs["center_mask_summaries"] = deepcopy(summaries)

    return center_masks, summaries


def build_final_masks(
    static_masks,
    saturation_masks,
    center_masks,
):
    """Combine masks and preserve the center-mask settings."""
    if saturation_masks.shape != center_masks.shape:
        raise ValueError(
            "Saturation and center masks have different shapes"
        )

    metadata_keys = (
        "center_mask_settings",
        "center_mask_summaries",
    )
    for key in metadata_keys:
        if key not in center_masks.attrs:
            raise ValueError(
                "Center-mask metadata is missing. "
                "Rerun center-mask construction first."
            )

    final_masks = {}

    for polarization, static_mask in static_masks.items():
        final_mask = (
            static_mask
            | saturation_masks
            | center_masks
        )
        final_mask = final_mask.transpose(
            *saturation_masks.dims
        )
        final_mask.name = "final_mask"

        for key in metadata_keys:
            final_mask.attrs[key] = deepcopy(
                center_masks.attrs[key]
            )

        final_masks[polarization] = final_mask

    return final_masks


def apply_final_masks(scans, final_masks):
    """Apply final masks and attach results to each scan."""
    for polarization, scan_data in scans.items():
        if polarization not in final_masks:
            raise ValueError(
                f"No final mask for polarization "
                f"{polarization}°"
            )

        final_mask = final_masks[polarization]

        scan_data["final_mask"] = final_mask
        scan_data["masked_data_final"] = (
            scan_data["data"].where(~final_mask)
        )

    return scans