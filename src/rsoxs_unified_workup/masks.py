import numpy as np
from skimage.draw import polygon2mask


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
