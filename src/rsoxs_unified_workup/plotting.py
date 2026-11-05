import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Patch, Wedge
import matplotlib.ticker as mticker


def review_energy_indices(scan_record, processing, energy_values):
    """Select frames nearest the requested review energies, in eV."""
    available = np.asarray(energy_values, dtype=float)

    if (
        available.ndim != 1
        or available.size == 0
        or not np.all(np.isfinite(available))
    ):
        raise ValueError(
            "Available energies must be a nonempty, finite 1D array"
        )

    plotting = processing.get("plotting", {})

    if "energy_indices" in plotting:
        raise ValueError(
            "Replace plotting.energy_indices with plotting.energies_ev"
        )

    requested = plotting.get("energies_ev")

    if requested is None:
        if scan_record["scan_type"] == "repeats":
            return list(range(available.size))

        raise ValueError(
            "Specify [plotting] energies_ev for "
            f"scan_type={scan_record['scan_type']!r}"
        )

    if not isinstance(requested, list) or not requested:
        raise ValueError(
            "plotting.energies_ev must be a nonempty list"
        )

    if any(
        isinstance(energy, bool)
        or not isinstance(energy, (int, float))
        or not np.isfinite(energy)
        or energy <= 0
        for energy in requested
    ):
        raise ValueError(
            "Requested energies must be positive, finite numbers in eV"
        )

    indices = []

    for energy in requested:
        index = int(np.argmin(np.abs(available - energy)))

        print(
            f"Scan {scan_record['scan_id']}: "
            f"requested {energy:g} eV "
            f"-> selected {available[index]:g} eV"
        )

        if index not in indices:
            indices.append(index)

    return indices


def plot_raw_frames(
    scans,
    processing,
    vmin=1e2,
    vmax=1e6,
    cmap="jet",
):
    """Plot selected raw detector frames for every polarization."""
    with plt.style.context("dark_background"):
        for polarization, scan_data in scans.items():
            data = scan_data["data"]
            energy_values = data.energy.values
            energy_indices = review_energy_indices(
                scan_record=scan_data,
                processing=processing,
                energy_values=energy_values,
            )
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
                    f"Energy = {energy_values[index]:g} eV"
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
    processing,
    vmin=1e1,
    vmax=1e6,
    cmap="jet",
):
    """Overlay each scan's pairwise saturation mask at review energies."""
    if not scans or set(scans) != set(saturation_masks):
        raise ValueError("Scans and saturation masks must have matching keys")

    rows = []

    for polarization, scan_data in scans.items():
        mask = saturation_masks[polarization]
        energy_dimension = mask.energy.dims[0]
        mask = mask.transpose(energy_dimension, "pix_y", "pix_x")

        data = scan_data["masked_data_static"]
        frame_dimension = data.energy.dims[0]
        data = data.transpose(frame_dimension, "pix_y", "pix_x")

        if data.shape != mask.shape:
            raise ValueError(
                f"Scan {scan_data['scan_id']}: data and mask shapes differ"
            )

        for coordinate in ("energy", "pix_y", "pix_x"):
            if not np.array_equal(
                data[coordinate].values,
                mask[coordinate].values,
            ):
                raise ValueError(
                    f"Scan {scan_data['scan_id']}: "
                    f"data and mask {coordinate} coordinates differ"
                )

        indices = review_energy_indices(
            scan_record=scan_data,
            processing=processing,
            energy_values=data.energy.values,
        )
        rows.append((polarization, scan_data, data, mask, indices))

    row_count = len(rows)
    column_count = max(len(row[4]) for row in rows)

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(7 * column_count, 5 * row_count),
            constrained_layout=True,
            squeeze=False,
        )
        active_axes = []

        for row, (polarization, scan_data, data, mask, indices) in enumerate(rows):
            frame_dimension = data.energy.dims[0]
            energy_dimension = mask.energy.dims[0]
            pair = mask.attrs["mask_pair_polarizations"]
            pair_label = "/".join(f"{value:g}°" for value in pair)

            for column, index in enumerate(indices):
                axis = axes[row, column]
                frame = data.isel({frame_dimension: index}).values
                saturation = mask.isel({energy_dimension: index}).values
                y_positions, x_positions = np.where(saturation)

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
                    label="Pairwise saturated pixels",
                )
                axis.set_title(
                    f"{scan_data['sample_name']}: "
                    f"Scan {scan_data['scan_id']}, "
                    f"polarization={polarization}°\n"
                    f"Pair={pair_label}; "
                    f"Energy={data.energy.values[index]:g} eV; "
                    f"{len(x_positions):,} saturated pixels"
                )
                axis.set_xlabel("Pixel x")
                axis.set_ylabel("Pixel y")
                axis.set_aspect("equal")
                axis.legend(loc="upper right", fontsize=8)
                active_axes.append(axis)

            for column in range(len(indices), column_count):
                axes[row, column].set_visible(False)

        fig.colorbar(
            image,
            ax=active_axes,
            shrink=0.5,
            label="Intensity",
        )

        first_scan = next(iter(scans.values()))
        fig.suptitle(
            f"Pairwise saturation-mask review\n"
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
    processing,
):
    """Review center-mask radius selection once per polarization pair."""
    if (
        not scans
        or set(scans) != set(saturation_masks)
        or set(scans) != set(center_masks)
        or set(scans) != set(summaries)
    ):
        raise ValueError("Scans, masks, and summaries must have matching keys")

    reviewed_pairs = set()

    for polarization, scan_record in scans.items():
        center_mask = center_masks[polarization]
        pair_id = center_mask.attrs["mask_pair_id"]

        if pair_id in reviewed_pairs:
            continue
        reviewed_pairs.add(pair_id)

        pair = center_mask.attrs["mask_pair_polarizations"]
        pair_label = "/".join(f"{value:g}°" for value in pair)
        scan_label = ", ".join(
            str(scans[value]["scan_id"]) for value in pair
        )

        energy_dimension = center_mask.energy.dims[0]
        center_mask = center_mask.transpose(
            energy_dimension, "pix_y", "pix_x"
        )
        saturation_mask = saturation_masks[polarization].transpose(
            *center_mask.dims
        )

        if saturation_mask.shape != center_mask.shape:
            raise ValueError("Saturation and center-mask shapes differ")

        for coordinate in ("energy", "pix_y", "pix_x"):
            if not np.array_equal(
                saturation_mask[coordinate].values,
                center_mask[coordinate].values,
            ):
                raise ValueError(
                    f"Pair {pair_id!r}: mask {coordinate} coordinates differ"
                )

        pair_summaries = summaries[polarization]
        if len(pair_summaries) != center_mask.sizes[energy_dimension]:
            raise ValueError("Center-mask summary count does not match energies")

        center = center_mask.attrs["beam_center"]
        center_x = center["x"]
        center_y = center["y"]

        energy_indices = review_energy_indices(
            scan_record=scan_record,
            processing=processing,
            energy_values=center_mask.energy.values,
        )

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(
                len(energy_indices),
                2,
                figsize=(12, 5 * len(energy_indices)),
                constrained_layout=True,
                squeeze=False,
            )

            for row, energy_index in enumerate(energy_indices):
                summary = pair_summaries[energy_index]
                saturation = saturation_mask.isel(
                    {energy_dimension: energy_index}
                ).values
                center_frame = center_mask.isel(
                    {energy_dimension: energy_index}
                ).values

                x_min, x_max, y_min, y_max = summary["search_bounds"]

                axes[row, 0].imshow(
                    saturation,
                    origin="lower",
                    cmap="gray",
                )
                axes[row, 0].scatter(
                    summary["farthest_x"],
                    summary["farthest_y"],
                    color="gold",
                    s=80,
                    label="Radius point",
                )
                axes[row, 0].scatter(
                    center_x,
                    center_y,
                    color="red",
                    s=80,
                    label="Configured beam center",
                )
                axes[row, 0].set_title(
                    f"Energy={summary['energy']:g} eV\n"
                    f"Radius={summary['radius']:.2f} pixels "
                    f"from {summary['radius_source']}"
                )
                axes[row, 0].legend()

                axes[row, 1].imshow(
                    center_frame,
                    origin="lower",
                    cmap="gray",
                )
                axes[row, 1].scatter(
                    center_x,
                    center_y,
                    color="red",
                    s=80,
                    label="Configured beam center",
                )
                axes[row, 1].set_title("Resulting pairwise center mask")
                axes[row, 1].legend()

                for axis in axes[row]:
                    axis.set_xlim(x_min, x_max)
                    axis.set_ylim(y_min, y_max)
                    axis.set_xlabel("Pixel x")
                    axis.set_ylabel("Pixel y")
                    axis.set_aspect("equal")

            fig.suptitle(
                f"Center-mask review: pair {pair_label}\n"
                f"Scans {scan_label}\n"
                f"{scan_record['sample_name']}: "
                f"{scan_record['base_description']}",
                fontsize=16,
            )

            plt.show()
            plt.close(fig)


def plot_final_masked_frames(
    scans,
    processing,
    vmin=1e1,
    vmax=1e6,
    cmap="jet",
):
    """Review selected final masked frames and outer usable radius."""
    row_count = len(scans)
    first_scan = next(iter(scans.values()))

    energy_indices = review_energy_indices(
        scan_record=first_scan,
        processing=processing,
        energy_values=first_scan["masked_data_final"].energy.values,
    )
    energy_count = len(energy_indices)

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
        data = scan_data["masked_data_final"]
        center = scan_data["final_mask"].attrs["beam_center"]
        center_settings = scan_data["final_mask"].attrs["center_mask_settings"]
        center_x = center["x"]
        center_y = center["y"]
        outer_radius = center_settings["outer_usable_radius"]

        for column, index in enumerate(energy_indices):
            axis = axes[row, column]
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
                f"Energy={data.energy.values[index]:g} eV"
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


def plot_integration_review(scan_record, processing):
    """Review integration results stored inside one scan record."""
    result = scan_record["integration"]
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

    energy_values = result["remeshed"].energy.values
    energy_indices = review_energy_indices(
        scan_record=scan_record,
        processing=processing,
        energy_values=energy_values,
    )

    for energy_index in energy_indices:
        energy = float(energy_values[energy_index])
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
    processing,
):
    """Compare the selected raw dark with its stored integrated result."""
    result = scan_record["dark_integration"]
    original_dark = result["original_dark"]
    upper_percentile = processing["darks"]["upper_percentile"]
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

    energy_values = result["darks_clean"].energy.values
    energy_indices = review_energy_indices(
        scan_record,
        processing,
        energy_values,
    )

    for energy_index in energy_indices:
        energy = float(energy_values[energy_index])
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


def plot_normalization_review(scans, processing):
    """Compare normalization factors stored inside the scan records."""
    first_scan = next(iter(scans.values()))
    first_result = first_scan["normalization"]

    energy_values = first_result["full_ring_mean"].energy.values
    energy_indices = review_energy_indices(
        first_scan,
        processing,
        energy_values,
    )

    for energy_index in energy_indices:
        energy = float(energy_values[energy_index])
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(
                1,
                3,
                figsize=(18, 5),
                constrained_layout=True,
            )

            for polarization, scan_record in scans.items():
                result = scan_record["normalization"]
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


def plot_flux_correction_review(scans, processing):
    """Review flux correction stored inside the scan records."""
    first_scan = next(iter(scans.values()))
    first_result = first_scan["flux_correction"]

    if not first_result:
        print("Flux correction is disabled; skipping its review plots.")
        return

    with plt.style.context("dark_background"):
        fig, axis = plt.subplots(
            1,
            1,
            figsize=(7, 5),
            constrained_layout=True,
        )

        for polarization, scan_record in scans.items():
            result = scan_record["flux_correction"]
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

    energy_values = first_result["flux_corrected"].energy.values
    energy_indices = review_energy_indices(
        first_scan,
        processing,
        energy_values,
    )

    for energy_index in energy_indices:
        energy = float(energy_values[energy_index])
        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(14, 5),
                constrained_layout=True,
            )

            for polarization, scan_record in scans.items():
                flux_result = scan_record["flux_correction"]
                raw_mean = scan_record["integration"]["clean"].sel(
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


def plot_anisotropy_review(
    scans,
    anisotropy_results,
    vmin=-0.05,
    vmax=0.05,
):
    """Plot every configured sector for each normalization."""
    first_scan = next(iter(scans.values()))

    for mode, sector_results in (
        anisotropy_results.items()
    ):
        sector_count = len(sector_results)
        column_count = min(4, sector_count)
        row_count = int(
            np.ceil(sector_count / column_count)
        )

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(
                row_count,
                column_count,
                figsize=(
                    6 * column_count,
                    5 * row_count,
                ),
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes).flatten()

            for axis, sector in zip(
                axes,
                sector_results.values(),
            ):
                values = sector["anisotropy"]
                image = axis.pcolormesh(
                    values["q"],
                    values["energy"],
                    values,
                    shading="auto",
                    cmap="coolwarm",
                    vmin=vmin,
                    vmax=vmax,
                )
                axis.set_title(
                    f"{sector['label']}\n"
                    f"χ center="
                    f"{sector['chi_center_deg']:g}°, "
                    f"halfwidth="
                    f"{sector['chi_halfwidth_deg']:g}°\n"
                    f"parallel="
                    f"{sector['parallel_polarization']}°, "
                    f"perpendicular="
                    f"{sector['perpendicular_polarization']}°"
                )
                axis.set_xlabel("q (Å⁻¹)")
                axis.set_ylabel("Energy (eV)")

            for unused_axis in axes[sector_count:]:
                unused_axis.remove()

            fig.colorbar(
                image,
                ax=axes[:sector_count].tolist(),
                shrink=0.7,
                label="Anisotropy",
            )
            fig.suptitle(
                f"Anisotropy by sector using "
                f"{mode.replace('_', ' ')} normalization\n"
                f"{first_scan['sample_name']}: "
                f"{first_scan['base_description']}",
                fontsize=16,
            )

            plt.show()
            plt.close(fig)


def plot_anisotropy_sector_overlays(
    scans,
    configuration,
    anisotropy_results,
    energy_value=None,
):
    """Overlay configured anisotropy sectors on detector frames."""
    processing = configuration["processing"]
    scan_by_id = {
        int(record["scan_id"]): record
        for record in configuration["scans"]
    }
    full_ring_sectors = anisotropy_results["full_ring"]

    for pair in configuration["polarization_pairs"]:
        sectors = [
            sector
            for sector in full_ring_sectors.values()
            if sector["pair_id"] == pair["pair_id"]
        ]
        if not sectors:
            continue

        polarizations = [
            scan_by_id[int(pair["scan_id_1"])][
                "polarization_deg"
            ],
            scan_by_id[int(pair["scan_id_2"])][
                "polarization_deg"
            ],
        ]
        polarizations = [
            int(value)
            if float(value).is_integer()
            else float(value)
            for value in polarizations
        ]

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(
                2,
                len(sectors),
                figsize=(5 * len(sectors), 10),
                constrained_layout=True,
            )
            axes = np.asarray(
                axes,
                dtype=object,
            ).reshape(2, len(sectors))

            for row, polarization in enumerate(
                polarizations
            ):
                scan_record = scans[polarization]
                center = (
                    scan_record["integration"]["beam_center_x"],
                    scan_record["integration"]["beam_center_y"],
                )
                data = scan_record["masked_data_final"]
                energies = np.asarray(
                    data.energy.values,
                    dtype=float,
                )

                if energy_value is None:
                    energy_index = 0
                else:
                    energy_index = int(
                        np.argmin(
                            np.abs(
                                energies - energy_value
                            )
                        )
                    )

                frame = data.isel(system=energy_index)
                radius = float(
                    np.hypot(
                        frame.sizes["pix_x"],
                        frame.sizes["pix_y"],
                    )
                )

                for column, sector in enumerate(
                    sectors
                ):
                    axis = axes[row, column]
                    axis.imshow(
                        frame,
                        norm=LogNorm(vmin=1e2, vmax=1e5),
                        cmap="jet",
                        origin="lower",
                    )

                    is_parallel = (
                        polarization
                        == sector[
                            "parallel_polarization"
                        ]
                    )
                    color = (
                        "purple"
                        if is_parallel
                        else "white"
                    )
                    axis.add_patch(
                        Wedge(
                            center=center,
                            r=radius,
                            theta1=(
                                sector["chi_center_deg"]
                                - sector[
                                    "chi_halfwidth_deg"
                                ]
                            ),
                            theta2=(
                                sector["chi_center_deg"]
                                + sector[
                                    "chi_halfwidth_deg"
                                ]
                            ),
                            facecolor=color,
                            alpha=0.4,
                        )
                    )
                    axis.set_title(
                        f"{sector['label']}; "
                        f"polarization={polarization}°\n"
                        f"{'parallel' if is_parallel else 'perpendicular'}; "
                        f"halfwidth="
                        f"{sector['chi_halfwidth_deg']:g}°"
                    )
                    axis.set_xlim(0, frame.sizes["pix_x"])
                    axis.set_ylim(0, frame.sizes["pix_y"])
                    axis.set_aspect("equal")
                    axis.set_xticks([])
                    axis.set_yticks([])

            first_scan = scans[polarizations[0]]
            fig.legend(
                handles=[
                    Patch(
                        facecolor="purple",
                        alpha=0.4,
                        label="Parallel",
                    ),
                    Patch(
                        facecolor="white",
                        edgecolor="gray",
                        alpha=0.4,
                        label="Perpendicular",
                    ),
                ],
                loc="upper right",
                fontsize=11,
            )
            fig.suptitle(
                f"Anisotropy-sector review for "
                f"{pair['pair_id']}\n"
                f"{first_scan['sample_name']}: "
                f"{first_scan['base_description']}\n"
                f"Energy={energies[energy_index]:g} eV; "
                f"purple=parallel, white=perpendicular",
                fontsize=16,
            )

            plt.show()
            plt.close(fig)


def plot_aggregate_anisotropy_comparison(
    scans,
    aggregate_results,
    anisotropy_limit=0.05,
    difference_limit=0.0005,
):
    """Compare three aggregate anisotropy methods."""
    method_specs = [
        (
            "method_1_global_intensity_ratio",
            "Method 1: global intensity ratio",
        ),
        (
            "method_2_pointwise_anisotropy",
            "Method 2: pointwise anisotropy",
        ),
        (
            "method_3_sector_anisotropy_mean",
            "Method 3: sector anisotropy mean",
        ),
    ]
    first_scan = next(iter(scans.values()))

    for mode, results in aggregate_results.items():
        methods = {
            key: results[key]
            for key, _ in method_specs
        }
        differences = [
            (
                methods[method_specs[0][0]]
                - methods[method_specs[1][0]],
                "Method 1 − Method 2",
            ),
            (
                methods[method_specs[1][0]]
                - methods[method_specs[2][0]],
                "Method 2 − Method 3",
            ),
            (
                methods[method_specs[0][0]]
                - methods[method_specs[2][0]],
                "Method 1 − Method 3",
            ),
        ]

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(
                2,
                3,
                figsize=(18, 10),
                constrained_layout=True,
            )

            for axis, (key, title) in zip(
                axes[0],
                method_specs,
            ):
                data = methods[key]
                image = axis.pcolormesh(
                    data["q"],
                    data["energy"],
                    data,
                    shading="auto",
                    cmap="coolwarm",
                    vmin=-anisotropy_limit,
                    vmax=anisotropy_limit,
                )
                axis.set_title(title)
                axis.set_xlabel("q (Å⁻¹)")
                axis.set_ylabel("Energy (eV)")
                fig.colorbar(
                    image,
                    ax=axis,
                    shrink=0.8,
                    label="Anisotropy",
                )

            for axis, (data, title) in zip(
                axes[1],
                differences,
            ):
                image = axis.pcolormesh(
                    data["q"],
                    data["energy"],
                    data,
                    shading="auto",
                    cmap="coolwarm",
                    vmin=-difference_limit,
                    vmax=difference_limit,
                )
                axis.set_title(title)
                axis.set_xlabel("q (Å⁻¹)")
                axis.set_ylabel("Energy (eV)")
                fig.colorbar(
                    image,
                    ax=axis,
                    shrink=0.8,
                    label="Anisotropy difference",
                )

            fig.suptitle(
                f"Aggregate anisotropy comparison: "
                f"{mode.replace('_', ' ')}\n"
                f"{first_scan['sample_name']}: "
                f"{first_scan['base_description']}",
                fontsize=16,
            )

            plt.show()
            plt.close(fig)


def plot_pointwise_anisotropy_review(
    scans,
    pointwise_results,
    processing,
    anisotropy_limit=0.15,
    difference_limit=0.15,
):

    """Review stitched pointwise anisotropy maps."""
    first_scan = next(iter(scans.values()))

    for mode, pair_results in pointwise_results.items():
        for pair_id, result in pair_results.items():
            energies = result["anisotropy"].energy.values
            energy_indices = review_energy_indices(
                first_scan,
                processing,
                energies,
            )

            for energy_index in energy_indices:
                energy = float(energies[energy_index])
                parallel = result["parallel"].isel(
                    energy=energy_index
                )
                perpendicular = result[
                    "perpendicular"
                ].isel(energy=energy_index)
                difference = result["difference"].isel(
                    energy=energy_index
                )
                anisotropy = result["anisotropy"].isel(
                    energy=energy_index
                )

                intensity_values = np.concatenate([
                    parallel.values.ravel(),
                    perpendicular.values.ravel(),
                ])
                intensity_values = intensity_values[
                    np.isfinite(intensity_values)
                ]
                intensity_min, intensity_max = (
                    np.nanpercentile(
                        intensity_values,
                        [5, 95],
                    )
                )

                with plt.style.context("dark_background"):
                    fig, axes = plt.subplots(
                        1,
                        4,
                        figsize=(24, 6),
                        constrained_layout=True,
                    )

                    panels = [
                        (
                            parallel,
                            "Stitched parallel intensity",
                            "viridis",
                            intensity_min,
                            intensity_max,
                            "Normalized intensity",
                        ),
                        (
                            perpendicular,
                            "Stitched perpendicular intensity",
                            "viridis",
                            intensity_min,
                            intensity_max,
                            "Normalized intensity",
                        ),
                        (
                            difference,
                            "Parallel − perpendicular",
                            "coolwarm",
                            -difference_limit,
                            difference_limit,
                            "Intensity difference",
                        ),
                        (
                            anisotropy,
                            "Pointwise anisotropy",
                            "coolwarm",
                            -anisotropy_limit,
                            anisotropy_limit,
                            "Anisotropy",
                        ),
                    ]

                    for axis, (
                        data,
                        title,
                        cmap,
                        vmin,
                        vmax,
                        colorbar_label,
                    ) in zip(axes, panels):
                        image = axis.pcolormesh(
                            data["q"],
                            data["chi"],
                            data,
                            shading="auto",
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                        )
                        axis.set_title(title)
                        axis.set_xlabel("q (Å⁻¹)")
                        axis.set_ylabel("chi (degrees)")
                        fig.colorbar(
                            image,
                            ax=axis,
                            shrink=0.8,
                            label=colorbar_label,
                        )

                    fig.suptitle(
                        f"Pointwise anisotropy review\n"
                        f"{pair_id}; "
                        f"{mode.replace('_', ' ')}; "
                        f"Energy={float(energy):g} eV\n"
                        f"{first_scan['sample_name']}: "
                        f"{first_scan['base_description']}",
                        fontsize=15,
                    )

                    plt.show()
                    plt.close(fig)
