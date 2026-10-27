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
