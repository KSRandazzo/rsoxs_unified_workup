import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle


def plot_raw_frames(
    scans,
    energy_indices=(0, 1),
    vmin=1e2,
    vmax=1e6,
    cmap="jet",
):
    """Plot selected raw detector frames for every polarization."""
    with plt.style.context("dark_background"):
        for polarization, scan_data in scans.items():
            data = scan_data["data"]
            energy_values = data.energy.values
            frame_count = len(energy_indices)

            fig, axes = plt.subplots(
                1,
                frame_count,
                figsize=(7.5 * frame_count, 8),
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes).flatten()

            for axis, index in zip(axes, energy_indices):
                image = axis.imshow(
                    data.isel(system=index).values,
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    cmap=cmap,
                    origin="lower",
                )
                axis.set_title(
                    f"Energy = {energy_values[index]:.1f} eV"
                )
                axis.axis("off")

            fig.colorbar(
                image,
                ax=axes,
                shrink=0.75,
                label="Intensity",
            )
            fig.suptitle(
                f"Selected energy frames for Scan "
                f"{scan_data['scan_id']}: "
                f"{scan_data['sample_name']}\n"
                f"{scan_data['description']}",
                fontsize=16,
            )

            plt.show()
            plt.close(fig)


def plot_dark_histograms(
    scans,
    upper_percentile,
    bins=200,
):
    """Plot full and upper-tail-trimmed dark histograms."""
    if not 0 < upper_percentile <= 100:
        raise ValueError(
            "upper_percentile must be between 0 and 100"
        )

    cutoffs = {}

    for polarization, scan_data in scans.items():
        if "darks" not in scan_data:
            raise ValueError(
                f"Dark frames have not been loaded for "
                f"polarization {polarization}°"
            )

        values = scan_data["darks"].values.ravel()
        values = values[np.isfinite(values)]

        if values.size == 0:
            raise ValueError(
                f"No finite dark values for "
                f"polarization {polarization}°"
            )

        lower_bound = np.min(values)
        cutoff = float(np.percentile(values, upper_percentile))
        cutoffs[polarization] = cutoff

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(15, 4),
            constrained_layout=True,
        )

        axes[0].hist(values, bins=bins)
        axes[0].axvline(
            cutoff,
            color="red",
            linestyle="--",
            label=f"{upper_percentile:g}th percentile",
        )
        axes[0].set_title("Full intensity range")
        axes[0].set_xlabel("Dark intensity")
        axes[0].set_ylabel("Pixel count")
        axes[0].legend()

        axes[1].hist(
            values,
            range=(lower_bound, cutoff),
            bins=bins,
        )
        axes[1].set_title(
            f"Intensity through the "
            f"{upper_percentile:g}th percentile"
        )
        axes[1].set_xlabel("Dark intensity")
        axes[1].set_ylabel("Pixel count")

        fig.suptitle(
            f"Dark histogram for Scan "
            f"{scan_data['scan_id']}\n"
            f"{scan_data['sample_name']}: "
            f"{scan_data['description']}\n"
            f"Candidate cutoff={cutoff:.2f}; "
            f"upper {100 - upper_percentile:g}% excluded"
        )

        plt.show()
        plt.close(fig)

    return cutoffs


def plot_dark_masks(scans, dark_masks):
    """Plot the spatial locations of masked dark pixels."""
    polarization_count = len(scans)
    column_count = 2
    row_count = int(np.ceil(polarization_count / column_count))

    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(12, 6 * row_count),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).flatten()

    for axis, (polarization, scan_data) in zip(
        axes,
        scans.items(),
    ):
        mask = dark_masks[polarization]
        y_positions, x_positions = np.where(mask.values)

        axis.scatter(
            x_positions,
            y_positions,
            s=3,
            color="cyan",
        )
        axis.set_xlim(0, mask.sizes["pix_x"])
        axis.set_ylim(0, mask.sizes["pix_y"])
        axis.set_aspect("equal")
        axis.set_title(
            f"{scan_data['sample_name']}: "
            f"Scan {scan_data['scan_id']}, "
            f"polarization={polarization}°\n"
            f"{len(x_positions):,} masked pixels"
        )
        axis.set_xlabel("Pixel x")
        axis.set_ylabel("Pixel y")

    for unused_axis in axes[polarization_count:]:
        unused_axis.remove()

    first_scan = next(iter(scans.values()))
    fig.suptitle(
        f"Dark-derived hot-pixel masks\n"
        f"{first_scan['sample_name']}: "
        f"{first_scan['base_description']}",
        fontsize=16,
    )

    plt.show()
    plt.close(fig)


def plot_static_mask_overlays(
    scans,
    static_masks,
    beamstop_vertices,
    vmin=1e0,
    vmax=1e9,
    cmap="plasma",
):
    """Compare raw summed frames with their static masks."""
    row_count = len(scans)

    fig, axes = plt.subplots(
        row_count,
        2,
        figsize=(14, 6 * row_count),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    closed_vertices = np.vstack(
        [beamstop_vertices, beamstop_vertices[0]]
    )

    for row, (polarization, scan_data) in enumerate(
        scans.items()
    ):
        data = scan_data["data"]
        frame_dimension = data.energy.dims[0]
        summed_frame = data.sum(frame_dimension).values
        mask = static_masks[polarization].values

        raw_image = axes[row, 0].imshow(
            summed_frame,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap,
            origin="lower",
        )
        axes[row, 0].plot(
            closed_vertices[:, 0],
            closed_vertices[:, 1],
            color="fuchsia",
            linewidth=2,
        )
        axes[row, 0].set_title(
            f"{scan_data['sample_name']}: "
            f"Scan {scan_data['scan_id']}, "
            f"polarization={polarization}°\n"
            f"Raw sum with beamstop outline"
        )

        masked_frame = np.ma.masked_where(
            mask,
            summed_frame,
        )
        axes[row, 1].imshow(
            masked_frame,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap,
            origin="lower",
        )
        axes[row, 1].set_title(
            f"{scan_data['sample_name']}: "
            f"Scan {scan_data['scan_id']}, "
            f"polarization={polarization}°\n"
            f"After static mask"
        )

        for axis in axes[row]:
            axis.set_aspect("equal")
            axis.set_xlabel("Pixel x")
            axis.set_ylabel("Pixel y")

    fig.colorbar(
        raw_image,
        ax=axes,
        shrink=0.5,
        label="Summed intensity",
    )

    first_scan = next(iter(scans.values()))
    fig.suptitle(
        f"Static-mask review\n"
        f"{first_scan['sample_name']}: "
        f"{first_scan['base_description']}",
        fontsize=16,
    )

    plt.show()
    plt.close(fig)


def plot_saturation_overlays(
    scans,
    saturation_masks,
    vmin=1e1,
    vmax=1e6,
    cmap="jet",
):
    """Overlay unioned saturated pixels on every scan frame."""
    row_count = len(scans)
    energy_dimension = saturation_masks.energy.dims[0]
    energy_count = saturation_masks.sizes[energy_dimension]

    fig, axes = plt.subplots(
        row_count,
        energy_count,
        figsize=(7 * energy_count, 5 * row_count),
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(
        row_count,
        energy_count,
    )

    for row, (polarization, scan_data) in enumerate(
        scans.items()
    ):
        masked_data = scan_data["masked_data_static"]
        frame_dimension = masked_data.energy.dims[0]

        for index in range(energy_count):
            axis = axes[row, index]
            frame = masked_data.isel(
                {frame_dimension: index}
            ).values
            saturation_mask = saturation_masks.isel(
                {energy_dimension: index}
            ).values
            y_positions, x_positions = np.where(
                saturation_mask
            )

            image = axis.imshow(
                frame,
                norm=LogNorm(vmin=vmin, vmax=vmax),
                cmap=cmap,
                origin="lower",
            )
            axis.scatter(
                x_positions,
                y_positions,
                color="white",
                s=1,
            )
            axis.set_title(
                f"{scan_data['sample_name']}: "
                f"Scan {scan_data['scan_id']}, "
                f"polarization={polarization}°\n"
                f"Energy="
                f"{masked_data.energy.values[index]:.1f} eV; "
                f"{len(x_positions):,} saturated pixels"
            )
            axis.set_xlabel("Pixel x")
            axis.set_ylabel("Pixel y")
            axis.set_aspect("equal")

    fig.colorbar(
        image,
        ax=axes,
        shrink=0.5,
        label="Intensity",
    )

    first_scan = next(iter(scans.values()))
    fig.suptitle(
        f"Saturation-mask review\n"
        f"{first_scan['sample_name']}: "
        f"{first_scan['base_description']}",
        fontsize=16,
    )

    plt.show()
    plt.close(fig)


def plot_center_mask_review(
    scans,
    saturation_masks,
    center_masks,
    summaries,
    settings,
):
    """Review center-mask radius selection for each energy."""
    energy_dimension = saturation_masks.energy.dims[0]
    energy_count = saturation_masks.sizes[energy_dimension]

    fig, axes = plt.subplots(
        energy_count,
        2,
        figsize=(12, 5 * energy_count),
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(
        energy_count,
        2,
    )

    center_x = settings["beam_center_x_guess"]
    center_y = settings["beam_center_y_guess"]

    for index, summary in enumerate(summaries):
        saturation = saturation_masks.isel(
            {energy_dimension: index}
        ).values
        center_mask = center_masks.isel(
            {energy_dimension: index}
        ).values

        x_min, x_max, y_min, y_max = summary[
            "search_bounds"
        ]

        axes[index, 0].imshow(
            saturation,
            origin="lower",
            cmap="gray",
        )
        axes[index, 0].scatter(
            summary["farthest_x"],
            summary["farthest_y"],
            color="gold",
            s=80,
            label="Radius point",
        )
        axes[index, 0].scatter(
            center_x,
            center_y,
            color="red",
            s=80,
            label="Beam-center guess",
        )
        axes[index, 0].set_title(
            f"Energy={summary['energy']:.1f} eV\n"
            f"Radius={summary['radius']:.2f} pixels "
            f"from {summary['radius_source']}"
        )
        axes[index, 0].legend()

        axes[index, 1].imshow(
            center_mask,
            origin="lower",
            cmap="gray",
        )
        axes[index, 1].scatter(
            center_x,
            center_y,
            color="red",
            s=80,
        )
        axes[index, 1].set_title("Resulting center mask")

        for axis in axes[index]:
            axis.set_xlim(x_min, x_max)
            axis.set_ylim(y_min, y_max)
            axis.set_xlabel("Pixel x")
            axis.set_ylabel("Pixel y")
            axis.set_aspect("equal")

    first_scan = next(iter(scans.values()))
    fig.suptitle(
        f"Center-mask review\n"
        f"{first_scan['sample_name']}: "
        f"{first_scan['base_description']}",
        fontsize=16,
    )

    plt.show()
    plt.close(fig)


def plot_final_masked_frames(
    scans,
    center_settings,
    vmin=1e1,
    vmax=1e6,
    cmap="jet",
):
    """Review final masked frames and outer usable radius."""
    row_count = len(scans)
    first_scan = next(iter(scans.values()))
    energy_count = first_scan[
        "masked_data_final"
    ].sizes["system"]

    fig, axes = plt.subplots(
        row_count,
        energy_count,
        figsize=(7 * energy_count, 5 * row_count),
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(
        row_count,
        energy_count,
    )

    center_x = center_settings["beam_center_x_guess"]
    center_y = center_settings["beam_center_y_guess"]
    outer_radius = center_settings["outer_usable_radius"]

    for row, (polarization, scan_data) in enumerate(
        scans.items()
    ):
        data = scan_data["masked_data_final"]

        for index in range(energy_count):
            axis = axes[row, index]
            frame = data.isel(system=index)

            image = axis.imshow(
                frame.values,
                norm=LogNorm(vmin=vmin, vmax=vmax),
                cmap=cmap,
                origin="lower",
            )
            axis.add_patch(
                Circle(
                    (center_x, center_y),
                    outer_radius,
                    fill=False,
                    edgecolor="fuchsia",
                    linestyle="--",
                    linewidth=2,
                )
            )
            axis.set_title(
                f"{scan_data['sample_name']}: "
                f"Scan {scan_data['scan_id']}, "
                f"polarization={polarization}°\n"
                f"Energy={data.energy.values[index]:.1f} eV"
            )
            axis.set_xlabel("Pixel x")
            axis.set_ylabel("Pixel y")
            axis.set_aspect("equal")

    fig.colorbar(
        image,
        ax=axes,
        shrink=0.5,
        label="Intensity",
    )
    fig.suptitle(
        f"Final masked frames and outer usable radius\n"
        f"{first_scan['sample_name']}: "
        f"{first_scan['base_description']}",
        fontsize=16,
    )

    plt.show()
    plt.close(fig)