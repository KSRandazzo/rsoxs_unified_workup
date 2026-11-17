from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr

from rsoxs_unified_workup.config import (
    configured_scan_records,
    get_mask_pairs,
    get_polarization_layout,
    scan_identity,
)


def _compact_number(value):
    """Format a number without unnecessary trailing zeros."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"{number:g}" if np.isfinite(number) else None


def build_base_description(sample):
    """Build the descriptive text used in plots."""
    diameter = _compact_number(sample.get("diameter_nm"))
    mw = _compact_number(sample.get("mw_g_mol"))
    mw_kda = _compact_number(float(mw) / 1000) if mw is not None else None

    def text(key):
        value = sample.get(key)
        return str(value) if isinstance(value, str) and value.strip() else None

    polymer = text("polymer")
    wash_state = text("wash_state")
    if diameter and mw_kda and polymer and polymer.casefold() != "none" and wash_state:
        return (
            f"Nanocomposite with D={diameter}nm SiNPs, "
            f"Mw={mw_kda}kDa {polymer}, {wash_state}"
        )
    # Missing descriptors are metadata, not a reason to block image loading.
    parts = [text("sample_name") or text("sample_id") or "Sample"]
    if diameter:
        parts.append(f"D={diameter}nm SiNPs")
    if polymer and polymer.casefold() != "none":
        parts.append(f"Mw={mw_kda}kDa {polymer}" if mw_kda else polymer)
    if wash_state:
        parts.append(wash_state)
    return ", ".join(parts)


def scan_ids_by_polarization(scan_set_data):
    """Return actual run IDs; different polarizations may share one ID."""
    return {
        polarization: scan_id
        for scan_id, polarization in configured_scan_records(scan_set_data)
    }


def _frame_polarizations(data, frame_dim, scan_id, layout):
    """Use measured coordinates for mixed runs, never a baseline scalar."""
    if "polarization" in data.coords:
        coordinate = data["polarization"]
        if coordinate.dims == (frame_dim,):
            values = np.asarray(coordinate.values, dtype=float)
        elif coordinate.ndim == 0 and layout == "separate_runs":
            values = np.full(data.sizes[frame_dim], float(coordinate.item()))
        else:
            raise ValueError(f"Scan {scan_id}: polarization must be a coordinate along {frame_dim}")
    elif layout == "separate_runs":
        try:
            scalar = np.asarray(data.attrs["polarization"], dtype=float)
            if scalar.ndim != 0:
                raise ValueError("Non-scalar polarization without a frame coordinate")
            values = np.full(data.sizes[frame_dim], float(scalar))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Scan {scan_id}: missing scalar polarization metadata") from error
    else:
        raise ValueError(f"Scan {scan_id}: within_run requires a per-frame polarization coordinate")
    if values.shape != (data.sizes[frame_dim],) or not np.isfinite(values).all():
        raise ValueError(f"Scan {scan_id}: polarization coordinates must be finite and match the frame count")
    return values


def _select_polarization(data, scan_id, polarization, layout):
    """Select images and matching metadata, with no averaging or pixel conversion."""
    if not isinstance(data, xr.DataArray) or "energy" not in data.coords:
        raise ValueError(f"Scan {scan_id}: loader must return a DataArray with an energy coordinate")
    if data.energy.ndim != 1:
        raise ValueError(f"Scan {scan_id}: expected a one-dimensional frame energy coordinate")
    frame_dim = data.energy.dims[0]
    if set(data.dims) != {frame_dim, "pix_y", "pix_x"}:
        raise ValueError(f"Scan {scan_id}: expected a frame stack with pix_y and pix_x dimensions")
    total = data.sizes[frame_dim]
    if total == 0:
        raise ValueError(f"Scan {scan_id}: no measured frames")
    measured = _frame_polarizations(data, frame_dim, scan_id, layout)
    selected = np.isclose(measured, polarization, atol=1e-6, rtol=0)
    available = np.unique(measured).tolist()
    if layout == "separate_runs" and not selected.all():
        raise ValueError(
            f"Scan {scan_id}: configured {polarization}°, measured {available}. "
            "If this run contains multiple polarizations, use polarization_layout=within_run."
        )
    indices = np.flatnonzero(selected)
    if indices.size == 0:
        raise ValueError(
            f"Scan {scan_id}: requested polarization {polarization}° is absent. "
            f"Available measured polarizations: {available}."
        )
    energies = np.asarray(data.energy.values, dtype=float)[indices]
    if not np.isfinite(energies).all() or np.any(energies <= 0):
        raise ValueError(f"Scan {scan_id}, {polarization}°: energies must be positive and finite")
    if np.unique(energies).size != energies.size:
        raise ValueError(
            f"Scan {scan_id}, {polarization}°: multiple frames have the same energy. "
            "This workflow requires one loaded frame per energy/polarization; no automatic averaging was applied."
        )

    try:
        flux = np.asarray(data.attrs["RSoXS Au Mesh Current"])
    except KeyError as error:
        raise ValueError(f"Scan {scan_id}: missing Au mesh current metadata") from error
    if flux.ndim != 1 or flux.size != total:
        raise ValueError(
            f"Scan {scan_id}: Au mesh current has shape {flux.shape}, "
            f"expected ({total},) to match the loaded image frames."
        )
    image = data.isel({frame_dim: indices}).copy(deep=False)
    source_index = data.indexes.get(frame_dim)
    source_levels = list(source_index.names) if isinstance(source_index, pd.MultiIndex) else []
    if isinstance(image.indexes.get(frame_dim), pd.MultiIndex):
        image = image.reset_index(frame_dim)
    # Keep original auxiliary coordinates, but replace the frame index with
    # an energy-only MultiIndex so existing unstack('system') code works.
    image = image.drop_vars([name for name in (frame_dim, "energy") if name in image.coords])
    if frame_dim != "system":
        image = image.rename({frame_dim: "system"})
    index = pd.MultiIndex.from_arrays([energies], names=["energy"])
    image = image.assign_coords(xr.Coordinates.from_pandas_multiindex(index, "system"))
    image = image.transpose("system", "pix_y", "pix_x")
    image.energy.attrs = dict(data.energy.attrs)
    attrs = deepcopy(data.attrs)
    # These are per-frame metadata arrays used by the workup/loader.
    for key in ("energy", "exposure", "RSoXS Au Mesh Current", "NSLS-II Ring Current"):
        value = attrs.get(key)
        if value is not None:
            array = np.asarray(value)
            if array.ndim == 1 and array.size == total:
                attrs[key] = array[indices].copy()
    attrs["polarization"] = float(polarization)
    image.attrs = attrs
    provenance = {
        "polarization_layout": layout,
        "source_scan_id": scan_id,
        "source_frame_count": total,
        "source_frame_indices": indices.tolist(),
        "available_polarizations": available,
        "original_system_levels": source_levels,
    }
    if layout == "within_run":
        image = image.assign_coords(source_frame_index=("system", indices))
        image.attrs["polarization_selection"] = deepcopy(provenance)
    return image, flux[indices].copy(), provenance


def load_scans(loader, scan_set_data):
    """Load each physical run once and return one record per polarization."""
    sample = scan_set_data["sample"]
    scan_set = scan_set_data["scan_set"]
    layout = get_polarization_layout(scan_set)
    records = configured_scan_records(scan_set_data)
    get_mask_pairs(scan_set_data)  # Reject ambiguous pairs before downloading.
    base_description = build_base_description(sample)
    loaded_scans = {}
    for scan_id in dict.fromkeys(identity[0] for identity in records):
        # Explicit dimensions avoid a loader heuristic dropping polarization
        # for plans whose name contains 'full' or 'short'.
        if layout == "within_run":
            source = loader.loadRun(scan_id, dims=["energy", "polarization"])
        else:
            source = loader.loadRun(scan_id)
        for configured_id, polarization in records:
            if configured_id != scan_id:
                continue
            data, flux, provenance = _select_polarization(source, scan_id, polarization, layout)
            loaded_scans[polarization] = _loaded_record(
                sample, scan_set, scan_id, polarization, data, flux, provenance, base_description,
            )
    return loaded_scans


def _loaded_record(sample, scan_set, scan_id, polarization, data, flux, provenance, base_description):
    """Keep the downstream scan dictionary independent of source layout."""
    energy_dimension = data.energy.dims[0]
    return {
        **sample,
        **scan_set,
        "scan_id": scan_id,
        "polarization_deg": polarization,
        "polarization_layout": provenance["polarization_layout"],
        "polarization_selection": provenance,
        "base_description": base_description,
        "description": f"{base_description}, polarization={polarization}°",
        "data": data,
        "sdd": data.attrs["sdd"],
        "pixel_size": data.attrs["pixel1"],
        "au_mesh_current": xr.DataArray(
            flux,
            dims=[energy_dimension],
            coords={"energy": (energy_dimension, data.energy.values)},
        ),
        "ring_current": data.attrs.get("NSLS-II Ring Current"),
        "date_time": data.attrs["meas_time"],
    }
