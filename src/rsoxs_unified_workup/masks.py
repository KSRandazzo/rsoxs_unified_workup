import numpy as np
from skimage.draw import polygon2mask
import xarray as xr
from copy import deepcopy
from rsoxs_unified_workup.config import get_beam_centers
from uuid import uuid4


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


def _validate_mask_pairs(polarizations, mask_pairs):
    """Require one complete perpendicular pair per polarization."""
    available = set(polarizations)
    assigned = set()

    if not mask_pairs:
        raise ValueError("No mask pairs were supplied")

    for pair_id, pair in mask_pairs.items():
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(
                f"Mask pair {pair_id!r} must contain two polarizations"
            )

        members = set(pair)
        missing = members - available
        overlap = members & assigned

        if missing:
            raise ValueError(
                f"Mask pair {pair_id!r} has unavailable polarizations: "
                f"{sorted(missing)}"
            )
        if overlap:
            raise ValueError(
                f"Polarizations {sorted(overlap)} belong to multiple pairs"
            )

        separation = abs(
            ((float(pair[0]) - float(pair[1]) + 90) % 180) - 90
        )
        if abs(separation - 90) > 1e-6:
            raise ValueError(f"Mask pair {pair_id!r} is not perpendicular")

        assigned.update(members)

    if assigned != available:
        raise ValueError(
            f"Polarizations without a mask pair: {sorted(available - assigned)}"
        )


def combine_static_masks(
    dark_masks,
    beamstop_mask,
    border_mask,
    mask_pairs,
):
    """Combine dark masks within pairs, plus shared geometric masks."""
    if not dark_masks:
        raise ValueError("No dark masks were supplied")

    _validate_mask_pairs(dark_masks, mask_pairs)
    expected_shape = beamstop_mask.shape

    if beamstop_mask.ndim != 2:
        raise ValueError("The beamstop mask must be two-dimensional")
    if border_mask.shape != expected_shape:
        raise ValueError(
            "Beamstop and border masks have different shapes"
        )

    ordered_dark_masks = {}
    for polarization, dark_mask in dark_masks.items():
        ordered = dark_mask.transpose("pix_y", "pix_x")

        if ordered.shape != expected_shape:
            raise ValueError(
                f"Dark mask for polarization {polarization}° "
                f"has shape {ordered.shape}; expected {expected_shape}"
            )
        if ordered.dtype.kind != "b":
            raise ValueError("Dark masks must contain boolean values")

        ordered_dark_masks[polarization] = ordered

    static_masks = {}

    for pair_id, polarizations in mask_pairs.items():
        pol_1, pol_2 = polarizations
        first, second = xr.align(
            ordered_dark_masks[pol_1],
            ordered_dark_masks[pol_2],
            join="exact",
            copy=False,
        )

        combined_values = (
            first.values
            | second.values
            | beamstop_mask
            | border_mask
        )

        for polarization in polarizations:
            mask = ordered_dark_masks[polarization].copy(
                data=combined_values.copy()
            )
            mask.name = "static_mask"
            mask.attrs["mask_pair_id"] = pair_id
            mask.attrs["mask_pair_polarizations"] = list(polarizations)
            mask.attrs["mask_union_scope"] = "polarization_pair"
            static_masks[polarization] = mask

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


def _on_reference_detector_grid(data, reference, label):
    """Validate a detector grid; only reorder dimensions, never interpolate."""
    ordered = data.transpose(*reference.dims)

    if ordered.shape != reference.shape:
        raise ValueError(
            f"{label} has shape {ordered.shape}; "
            f"expected {reference.shape}"
        )

    for coordinate in ("energy", "pix_y", "pix_x"):
        if not np.array_equal(
            ordered[coordinate].values,
            reference[coordinate].values,
        ):
            raise ValueError(
                f"{label} has different {coordinate} coordinates"
            )

    return ordered


def build_saturation_masks(scans, mask_pairs):
    """Union negative saturated pixels within each pair, at each energy."""
    _validate_mask_pairs(scans, mask_pairs)
    saturation_masks = {}

    for pair_id, polarizations in mask_pairs.items():
        pol_1, pol_2 = polarizations
        reference = scans[pol_1]["masked_data_static"]
        energy_dimension = reference.energy.dims[0]
        reference = reference.transpose(
            energy_dimension, "pix_y", "pix_x"
        )
        combined_values = np.zeros(reference.shape, dtype=bool)
        pair_data = {}

        for polarization in polarizations:
            scan_record = scans[polarization]
            static_mask = scan_record.get("static_mask")
            if (
                static_mask is None
                or static_mask.attrs.get("mask_pair_id") != pair_id
                or set(
                    static_mask.attrs.get("mask_pair_polarizations", [])
                ) != set(polarizations)
                or static_mask.attrs.get("mask_union_scope")
                != "polarization_pair"
            ):
                raise ValueError(
                    f"Scan {scan_record['scan_id']} has no current "
                    f"static mask for pair {pair_id!r}. "
                    "Rebuild and apply the pairwise static masks first."
                )

            ordered = _on_reference_detector_grid(
                scan_record["masked_data_static"],
                reference,
                f"Scan {scan_record['scan_id']}",
            )
            pair_data[polarization] = ordered

            # NaN values compare False; keep the existing < 0 criterion.
            combined_values |= ordered.values < 0

        for polarization in polarizations:
            mask = pair_data[polarization].copy(
                data=combined_values.copy()
            )
            mask.name = "saturation_mask"
            mask.attrs = {
                "mask_pair_id": pair_id,
                "mask_pair_polarizations": list(polarizations),
                "mask_union_scope": "polarization_pair",
            }
            saturation_masks[polarization] = mask

    return saturation_masks


def _build_center_mask_for_pair(
    saturation_masks,
    beamstop_vertices,
    processing,
    center,
):
    """Build circular center masks using this pair's configured center."""
    settings = processing["center_mask"]
    center_x = center["x"]
    center_y = center["y"]

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
    center_masks.attrs["beam_center"] = deepcopy(center)
    center_masks.attrs["center_mask_settings"] = deepcopy(settings)
    center_masks.attrs["center_mask_summaries"] = deepcopy(summaries)

    return center_masks, summaries


def build_center_masks(
    saturation_masks,
    beamstop_vertices,
    processing,
    mask_pairs,
):
    """Build one center mask per pair and energy, using that pair's center."""
    _validate_mask_pairs(saturation_masks, mask_pairs)
    beam_centers = get_beam_centers(processing, mask_pairs)
    center_masks = {}
    center_mask_summaries = {}

    for pair_id, polarizations in mask_pairs.items():
        pol_1, pol_2 = polarizations
        reference = saturation_masks[pol_1]
        energy_dimension = reference.energy.dims[0]
        reference = reference.transpose(
            energy_dimension, "pix_y", "pix_x"
        )
        pair_masks = {}

        for polarization in polarizations:
            mask = saturation_masks[polarization]

            if (
                mask.attrs.get("mask_pair_id") != pair_id
                or set(
                    mask.attrs.get("mask_pair_polarizations", [])
                ) != set(polarizations)
                or mask.attrs.get("mask_union_scope")
                != "polarization_pair"
            ):
                raise ValueError(
                    f"Polarization {polarization}° has no current "
                    f"saturation mask for pair {pair_id!r}"
                )

            ordered = _on_reference_detector_grid(
                mask,
                reference,
                f"Saturation mask for polarization {polarization}°",
            )
            if ordered.dtype.kind != "b":
                raise ValueError("Saturation masks must be boolean")

            pair_masks[polarization] = ordered

        if not np.array_equal(
            pair_masks[pol_1].values,
            pair_masks[pol_2].values,
        ):
            raise ValueError(
                f"Saturation masks differ within pair {pair_id!r}. "
                "Rebuild the pairwise saturation masks first."
            )

        pair_center_mask, summaries = _build_center_mask_for_pair(
            saturation_masks=reference,
            beamstop_vertices=beamstop_vertices,
            processing=processing,
            center=beam_centers[pair_id],
        )

        for polarization in polarizations:
            mask = pair_masks[polarization].copy(
                data=pair_center_mask.values.copy()
            )
            mask.name = "center_mask"
            mask.attrs = deepcopy(pair_center_mask.attrs)

            center_masks[polarization] = mask
            center_mask_summaries[polarization] = deepcopy(summaries)

    return center_masks, center_mask_summaries


def build_final_masks(
    static_masks,
    saturation_masks,
    center_masks,
    mask_pairs,
):
    """Combine each scan's pairwise masks without coordinate alignment."""
    _validate_mask_pairs(static_masks, mask_pairs)

    if (
        set(static_masks) != set(saturation_masks)
        or set(static_masks) != set(center_masks)
    ):
        raise ValueError("Static, saturation, and center-mask keys must match")

    center_metadata = (
        "beam_center",
        "center_mask_settings",
        "center_mask_summaries",
    )
    pair_metadata = (
        "mask_pair_id",
        "mask_pair_polarizations",
        "mask_union_scope",
    )
    final_masks = {}

    for pair_id, polarizations in mask_pairs.items():
        for polarization in polarizations:
            saturation = saturation_masks[polarization]
            energy_dimension = saturation.energy.dims[0]
            saturation = saturation.transpose(
                energy_dimension, "pix_y", "pix_x"
            )

            center = _on_reference_detector_grid(
                center_masks[polarization],
                saturation,
                f"Center mask for polarization {polarization}°",
            )
            static = static_masks[polarization].transpose(
                "pix_y", "pix_x"
            )

            if static.shape != saturation.shape[-2:]:
                raise ValueError("Static mask does not match detector shape")

            for coordinate in ("pix_y", "pix_x"):
                if not np.array_equal(
                    static[coordinate].values,
                    saturation[coordinate].values,
                ):
                    raise ValueError(
                        f"Polarization {polarization}°: "
                        f"static-mask {coordinate} coordinates differ"
                    )

            for mask in (static, saturation, center):
                if mask.dtype.kind != "b":
                    raise ValueError("All detector masks must be boolean")

                if (
                    mask.attrs.get("mask_pair_id") != pair_id
                    or set(
                        mask.attrs.get("mask_pair_polarizations", [])
                    ) != set(polarizations)
                    or mask.attrs.get("mask_union_scope")
                    != "polarization_pair"
                ):
                    raise ValueError(
                        f"Polarization {polarization}° has a mask "
                        f"that does not belong to pair {pair_id!r}"
                    )

            for key in center_metadata:
                if key not in center.attrs:
                    raise ValueError(
                        f"Center-mask metadata {key!r} is missing. "
                        "Rebuild the center masks first."
                    )

            combined_values = (
                static.values[np.newaxis, :, :]
                | saturation.values
                | center.values
            )
            final_mask = saturation.copy(data=combined_values)
            final_mask.name = "final_mask"
            final_mask.attrs = {
                key: deepcopy(center.attrs[key])
                for key in center_metadata + pair_metadata
            }
            final_mask.attrs["mask_id"] = uuid4().hex
            final_masks[polarization] = final_mask

        pol_1, pol_2 = polarizations
        first = final_masks[pol_1]
        second = _on_reference_detector_grid(
            final_masks[pol_2],
            first,
            f"Final mask for polarization {pol_2}°",
        )

        if not np.array_equal(first.values, second.values):
            raise ValueError(
                f"Final masks differ within pair {pair_id!r}. "
                "Rebuild that pair's component masks first."
            )

        for key in center_metadata:
            if first.attrs[key] != second.attrs[key]:
                raise ValueError(
                    f"Center-mask metadata {key!r} differs "
                    f"within pair {pair_id!r}"
                )

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

        scan_data.pop("beam_center_approval", None)
        scan_data["final_mask"] = final_mask
        scan_data["masked_data_final"] = (
            scan_data["data"].where(~final_mask)
        )

    return scans