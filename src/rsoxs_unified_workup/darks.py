def load_dark_frames(loader, scans):
    """Load and attach dark frames for every configured scan."""
    for polarization, scan_data in scans.items():
        scan_id = scan_data["scan_id"]
        run = loader.c[scan_id]

        dark_dataset = run["dark"]["data"].read()
        darks = dark_dataset[
            "Wide Angle CCD Detector_image"
        ].squeeze().load()

        if darks.ndim not in (2, 3):
            raise ValueError(
                f"Unexpected dark-frame shape for scan "
                f"{scan_id}: {darks.shape}"
            )

        spatial_dimensions = darks.dims[-2:]
        rename_dimensions = {}

        if spatial_dimensions[0] != "pix_y":
            rename_dimensions[spatial_dimensions[0]] = "pix_y"

        if spatial_dimensions[1] != "pix_x":
            rename_dimensions[spatial_dimensions[1]] = "pix_x"

        if darks.ndim == 3 and darks.dims[0] != "time":
            rename_dimensions[darks.dims[0]] = "time"

        darks = darks.rename(rename_dimensions)

        if darks.ndim == 2:
            darks = darks.expand_dims("time")

        scan_data["darks"] = darks

        print(
            f"Scan {scan_id}, polarization={polarization}°: "
            f"loaded {darks.sizes['time']} dark frame(s), "
            f"shape={darks.shape}"
        )

    return scans


def build_dark_masks(scans, cutoffs):
    """Build one hot-pixel mask per polarization."""
    masks = {}
    summaries = {}

    for polarization, scan_data in scans.items():
        if polarization not in cutoffs:
            raise ValueError(
                f"No dark cutoff supplied for "
                f"polarization {polarization}°"
            )

        darks = scan_data["darks"]
        cutoff = float(cutoffs[polarization])

        maximum_by_pixel = darks.max(
            dim="time",
            skipna=True,
        )
        mask = maximum_by_pixel > cutoff

        masked_pixels = int(mask.sum().item())
        total_pixels = int(mask.size)

        masks[polarization] = mask
        summaries[polarization] = {
            "scan_id": scan_data["scan_id"],
            "cutoff": cutoff,
            "masked_pixels": masked_pixels,
            "masked_percent": (
                100 * masked_pixels / total_pixels
            ),
        }

    return masks, summaries
