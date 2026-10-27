import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import matplotlib.ticker as mticker


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


def plot_beam_center_fit(fit_result, scan_record):
    """Review the fitted center, q peaks, and search grids."""
    fine_center = fit_result["fine_best"]
    coarse_center = fit_result["coarse_best"]
    remeshed = fit_result["selected_remesh"]
    detector_frame = fit_result["selected_data"].isel(
        energy=0
    )
    coarse_grid = fit_result["coarse_grid"]
    fine_grid = fit_result["fine_grid"]

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(18, 10),
            constrained_layout=True,
        )

        remesh_image = axes[0, 0].pcolormesh(
            remeshed["q"],
            remeshed["chi"],
            remeshed,
            shading="auto",
        )
        axes[0, 0].set_title("Integrated data after postmask")
        axes[0, 0].set_xlabel("q")
        axes[0, 0].set_ylabel("chi")
        fig.colorbar(
            remesh_image,
            ax=axes[0, 0],
            shrink=0.7,
            label="Intensity",
        )

        for center, curve in fit_result[
            "intensity_curves"
        ].items():
            axes[0, 1].plot(
                curve["q"],
                curve["intensity"],
                label=f"{center}°",
            )
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_title("Azimuthal-region intensity curves")
        axes[0, 1].set_xlabel("q")
        axes[0, 1].set_ylabel("Intensity")
        axes[0, 1].legend()

        for center, curve in fit_result[
            "fit_curves"
        ].items():
            axes[0, 2].scatter(
                curve["q_fit"],
                curve["intensity_fit"],
                s=10,
                label=f"{center}° data",
            )
            if curve["q_smooth"] is not None:
                axes[0, 2].plot(
                    curve["q_smooth"],
                    curve["intensity_smooth"],
                    label=f"{center}° fit",
                )
        axes[0, 2].set_title("Gaussian peak fits")
        axes[0, 2].set_xlabel("q")
        axes[0, 2].set_ylabel("Intensity")
        axes[0, 2].legend(fontsize=8)

        axes[1, 0].imshow(
            detector_frame,
            norm=LogNorm(vmin=1, vmax=1e5),
            origin="lower",
            cmap="jet",
        )
        axes[1, 0].scatter(
            fine_center["x"],
            fine_center["y"],
            color="deeppink",
            s=40,
        )
        for radius in (5, 50, 150):
            axes[1, 0].add_patch(
                Circle(
                    (fine_center["x"], fine_center["y"]),
                    radius,
                    fill=False,
                    edgecolor="deeppink",
                )
            )
        axes[1, 0].set_title("Fitted center on detector")
        axes[1, 0].set_xlabel("Pixel x")
        axes[1, 0].set_ylabel("Pixel y")
        axes[1, 0].set_aspect("equal")

        coarse_image = axes[1, 1].pcolormesh(
            coarse_grid["x"],
            coarse_grid["y"],
            coarse_grid,
            shading="auto",
        )
        axes[1, 1].scatter(
            coarse_center["x"],
            coarse_center["y"],
            marker="x",
            color="cyan",
            s=100,
        )
        axes[1, 1].set_title("Coarse-search cost")
        axes[1, 1].set_xlabel("Center x")
        axes[1, 1].set_ylabel("Center y")
        axes[1, 1].set_aspect("equal")
        axes[1, 1].xaxis.set_major_locator(
            mticker.MaxNLocator(integer=True)
        )
        axes[1, 1].yaxis.set_major_locator(
            mticker.MaxNLocator(integer=True)
        )
        fig.colorbar(
            coarse_image,
            ax=axes[1, 1],
            shrink=0.7,
            label="Peak-variance cost",
        )

        fine_image = axes[1, 2].pcolormesh(
            fine_grid["x"],
            fine_grid["y"],
            fine_grid,
            shading="auto",
        )
        axes[1, 2].scatter(
            fine_center["x"],
            fine_center["y"],
            marker="x",
            color="cyan",
            s=100,
        )
        axes[1, 2].set_title("Fine-search cost")
        axes[1, 2].set_xlabel("Center x")
        axes[1, 2].set_ylabel("Center y")
        axes[1, 2].set_aspect("equal")
        axes[1, 2].xaxis.set_major_locator(
            mticker.MultipleLocator(0.1)
        )
        axes[1, 2].yaxis.set_major_locator(
            mticker.MultipleLocator(0.1)
        )
        fig.colorbar(
            fine_image,
            ax=axes[1, 2],
            shrink=0.7,
            label="Peak-variance cost",
        )

        fig.suptitle(
            f"Beam-center fit for Scan "
            f"{scan_record['scan_id']}, "
            f"polarization="
            f"{fit_result['polarization_deg']}°\n"
            f"{scan_record['sample_name']}: "
            f"{scan_record['base_description']}\n"
            f"Fine center=({fine_center['x']}, "
            f"{fine_center['y']}); "
            f"energy={fit_result['actual_fit_energy']:g} eV",
            fontsize=15,
        )

        plt.show()
        plt.close(fig)


def plot_integration_review(scan_record, result):
    """Review each stage of final integration at every energy."""
    datasets = [
        ("remeshed", "Integrated data", "Intensity"),
        (
            "remeshed_mask",
            "Integrated eligible-pixel image",
            "Source-pixel count",
        ),
        (
            "remeshed_ones",
            "Integrated ones",
            "Source-pixel count",
        ),
        (
            "source_fraction",
            "Fraction of eligible source pixels",
            "Fraction",
        ),
        (
            "clean_uncapped_q",
            "Postmasked data without q cap",
            "Intensity",
        ),
        (
            "clean",
            "Postmasked data with q cap",
            "Intensity",
        ),
    ]

    sample_points = [
        (0.015, -50, "red"),
        (0.04, 50, "cyan"),
        (0.11, -100, "lawngreen"),
    ]

    for energy_index, energy in enumerate(
        result["remeshed"].energy.values
    ):
        q_cap = float(
            result["q_cap"].sel(
                energy=energy,
                method="nearest",
            )
        )

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(
                2,
                3,
                figsize=(16, 10),
                constrained_layout=True,
            )

            for axis, (
                key,
                title,
                colorbar_label,
            ) in zip(axes.flat, datasets):
                frame = result[key].isel(
                    energy=energy_index
                )
                image = axis.pcolormesh(
                    frame["q"],
                    frame["chi"],
                    frame,
                    shading="auto",
                )
                axis.set_title(title)
                axis.set_xlabel("q (Å⁻¹)")
                axis.set_ylabel("chi (degrees)")
                fig.colorbar(
                    image,
                    ax=axis,
                    shrink=0.7,
                    label=colorbar_label,
                )
                if key == "clean":
                    axis.axvline(
                        q_cap,
                        color="fuchsia",
                        linestyle="--",
                        label=f"q cap={q_cap:.4f} Å⁻¹",
                    )

                for q_value, chi_value, color in sample_points:
                    sampled_intensity = frame.sel(
                        q=q_value,
                        chi=chi_value,
                        method="nearest",
                    ).item()

                    axis.plot(
                        q_value,
                        chi_value,
                        marker="o",
                        color=color,
                        markersize=7,
                        linestyle="none",
                        label=(
                            f"q={q_value:g}, "
                            f"χ={chi_value:g}: "
                            f"{sampled_intensity:.2f}"
                        ),
                    )

                axis.legend(
                    loc="upper right",
                    fontsize=7,
                )

            fig.suptitle(
                f"Final integration for Scan "
                f"{scan_record['scan_id']}, "
                f"polarization="
                f"{scan_record['polarization_deg']}°\n"
                f"{scan_record['sample_name']}: "
                f"{scan_record['base_description']}\n"
                f"Energy={float(energy):g} eV; "
                f"shared center="
                f"({result['beam_center_x']}, "
                f"{result['beam_center_y']}); "
                f"postmask threshold="
                f"{result['postmask_threshold']:g}",
                fontsize=15,
            )

            plt.show()
            plt.close(fig)


def plot_dark_integration_review(
    scan_record,
    result,
    upper_percentile,
):
    """Compare the selected raw dark with its integrated result."""
    original_dark = result["original_dark"]
    vmin = float(np.nanmin(original_dark.values))
    vmax = float(
        np.nanpercentile(
            original_dark.values,
            upper_percentile,
        )
    )
    sample_points = [
        (0.02, -50, "red"),
        (0.005, 170, "cyan"),
        (0.08, -90, "lawngreen"),
        (0.12, -90, "goldenrod"),
    ]

    for energy_index, energy in enumerate(
        result["darks_clean"].energy.values
    ):
        integrated_dark = result[
            "darks_clean"
        ].isel(energy=energy_index)

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(14, 6),
                constrained_layout=True,
            )

            original_image = axes[0].pcolormesh(
                original_dark["pix_x"],
                original_dark["pix_y"],
                original_dark,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            axes[0].set_title(
                f"Original dark, time="
                f"{result['dark_frame_index']}"
            )
            axes[0].set_xlabel("Pixel x")
            axes[0].set_ylabel("Pixel y")
            axes[0].set_aspect("equal")
            fig.colorbar(
                original_image,
                ax=axes[0],
                shrink=0.7,
                label="Dark counts",
            )

            integrated_image = axes[1].pcolormesh(
                integrated_dark["q"],
                integrated_dark["chi"],
                integrated_dark,
                shading="auto",
            )
            axes[1].set_title(
                "Integrated dark with postmask"
            )
            axes[1].set_xlabel("q (Å⁻¹)")
            axes[1].set_ylabel("chi (degrees)")
            fig.colorbar(
                integrated_image,
                ax=axes[1],
                shrink=0.7,
                label="Dark counts",
            )

            for q_value, chi_value, color in sample_points:
                sampled_intensity = integrated_dark.sel(
                    q=q_value,
                    chi=chi_value,
                    method="nearest",
                ).item()
                axes[1].plot(
                    q_value,
                    chi_value,
                    marker="o",
                    color=color,
                    linestyle="none",
                    markersize=7,
                    label=(
                        f"q={q_value:g}, "
                        f"χ={chi_value:g}: "
                        f"{sampled_intensity:.2f}"
                    ),
                )

            axes[1].legend(
                loc="upper right",
                fontsize=7,
            )
            fig.suptitle(
                f"Dark integration for Scan "
                f"{scan_record['scan_id']}, "
                f"polarization="
                f"{scan_record['polarization_deg']}°\n"
                f"{scan_record['sample_name']}: "
                f"{scan_record['base_description']}\n"
                f"Energy={float(energy):g} eV",
                fontsize=15,
            )

            plt.show()
            plt.close(fig)


def plot_normalization_review(
    scans,
    normalization_results,
):
    """Compare full-ring and half-ring normalization factors."""
    first_scan = next(iter(scans.values()))
    first_result = next(
        iter(normalization_results.values())
    )

    for energy in first_result[
        "full_ring_mean"
    ].energy.values:
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(
                1,
                3,
                figsize=(18, 5),
                constrained_layout=True,
            )

            for polarization, result in (
                normalization_results.items()
            ):
                full_mean = result[
                    "full_ring_mean"
                ].sel(energy=energy)
                half_mean = result[
                    "half_ring_mean"
                ].sel(energy=energy)
                label = (
                    f"{polarization}°; "
                    f"scan {scans[polarization]['scan_id']}"
                )

                axes[0].plot(
                    full_mean["q"],
                    full_mean,
                    label=label,
                )
                axes[1].plot(
                    half_mean["q"],
                    half_mean,
                    label=label,
                )
                axes[2].plot(
                    full_mean["q"],
                    half_mean / full_mean,
                    label=label,
                )

            axes[0].set_title("Full-ring mean intensity")
            axes[1].set_title("Half-ring mean intensity")
            axes[2].set_title(
                "Half-ring mean / full-ring mean"
            )

            axes[0].set_yscale("log")
            axes[1].set_yscale("log")

            for axis in axes:
                axis.set_xlabel("q (Å⁻¹)")
                axis.legend(fontsize=8)

            axes[0].set_ylabel("Mean intensity")
            axes[1].set_ylabel("Mean intensity")
            axes[2].set_ylabel("Intensity ratio")

            fig.suptitle(
                f"Normalization review at "
                f"{float(energy):g} eV\n"
                f"{first_scan['sample_name']}: "
                f"{first_scan['base_description']}\n"
                f"Half ring: center="
                f"{first_result['half_ring_center_degrees']:g}°, "
                f"halfwidth="
                f"{first_result['half_ring_halfwidth_degrees']:g}°",
                fontsize=15,
            )

            plt.show()
            plt.close(fig)


def plot_flux_correction_review(
    scans,
    integration_results,
    flux_results,
):
    """Review Au-mesh flux values and corrected intensities."""
    first_scan = next(iter(scans.values()))
    first_result = next(iter(flux_results.values()))

    with plt.style.context("dark_background"):
        fig, axis = plt.subplots(
            1,
            1,
            figsize=(7, 5),
            constrained_layout=True,
        )

        for polarization, result in flux_results.items():
            flux = result["flux"]
            axis.plot(
                flux.energy,
                flux,
                marker="o",
                label=(
                    f"{polarization}°; "
                    f"scan {scans[polarization]['scan_id']}"
                ),
            )

        axis.set_title("Au mesh current")
        axis.set_xlabel("Energy (eV)")
        axis.set_ylabel("Au mesh current")
        axis.legend()
        fig.suptitle(
            f"Incident-flux monitor review\n"
            f"{first_scan['sample_name']}: "
            f"{first_scan['base_description']}",
            fontsize=15,
        )

        plt.show()
        plt.close(fig)

    for energy in first_result[
        "flux_corrected"
    ].energy.values:
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(14, 5),
                constrained_layout=True,
            )

            for polarization, flux_result in (
                flux_results.items()
            ):
                raw_mean = integration_results[
                    polarization
                ]["clean"].sel(
                    energy=energy
                ).mean("chi", skipna=True)
                corrected_mean = flux_result[
                    "flux_corrected"
                ].sel(
                    energy=energy
                ).mean("chi", skipna=True)
                label = (
                    f"{polarization}°; "
                    f"scan {scans[polarization]['scan_id']}"
                )

                axes[0].plot(
                    raw_mean["q"],
                    raw_mean,
                    label=label,
                )
                axes[1].plot(
                    corrected_mean["q"],
                    corrected_mean,
                    label=label,
                )

            axes[0].set_title("Before flux correction")
            axes[1].set_title("After flux correction")

            for axis in axes:
                axis.set_yscale("log")
                axis.set_xlabel("q (Å⁻¹)")
                axis.set_ylabel("Chi-averaged intensity")
                axis.legend(fontsize=8)

            fig.suptitle(
                f"Flux-correction review at "
                f"{float(energy):g} eV\n"
                f"{first_scan['sample_name']}: "
                f"{first_scan['base_description']}",
                fontsize=15,
            )

            plt.show()
            plt.close(fig)