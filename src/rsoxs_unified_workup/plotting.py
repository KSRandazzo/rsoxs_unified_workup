import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


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
