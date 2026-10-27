import xarray as xr


def _compact_number(value):
    """Format a number without unnecessary trailing zeros."""
    return f"{float(value):g}"


def build_base_description(sample):
    """Build the descriptive text used in plots."""
    diameter = _compact_number(sample["diameter_nm"])
    mw_kda = _compact_number(sample["mw_g_mol"] / 1000)

    return (
        f"Nanocomposite with D={diameter}nm SiNPs, "
        f"Mw={mw_kda}kDa {sample['polymer']}, "
        f"{sample['wash_state']}"
    )


def scan_ids_by_polarization(scan_set_data):
    """Return scan IDs indexed by their measured polarization."""
    scan_ids = {}

    for scan in scan_set_data["scans"]:
        polarization = float(scan["polarization_deg"])

        if polarization.is_integer():
            polarization = int(polarization)

        if polarization in scan_ids:
            raise ValueError(
                f"Duplicate polarization in scan set: {polarization}"
            )

        scan_ids[polarization] = int(scan["scan_id"])

    return scan_ids


def load_scans(loader, scan_set_data):
    """Load every scan in a configured scan set."""
    sample = scan_set_data["sample"]
    scan_set = scan_set_data["scan_set"]
    base_description = build_base_description(sample)
    loaded_scans = {}

    for scan_record in scan_set_data["scans"]:
        scan_id = int(scan_record["scan_id"])
        polarization = float(scan_record["polarization_deg"])

        if polarization.is_integer():
            polarization = int(polarization)

        data = loader.loadRun(scan_id)
        measured_polarization = data.attrs.get("polarization")

        if measured_polarization is None:
            raise ValueError(
                f"Scan {scan_id} has no polarization metadata"
            )

        if abs(float(measured_polarization) - float(polarization)) > 1e-6:
            raise ValueError(
                f"Scan {scan_id} was configured as {polarization}°, "
                f"but its data reports {measured_polarization}°"
            )

        energy_dimension = data.energy.dims[0]

        loaded_scans[polarization] = {
            **sample,
            **scan_set,
            "scan_id": scan_id,
            "polarization_deg": polarization,
            "base_description": base_description,
            "description": (
                f"{base_description}, polarization={polarization}°"
            ),
            "data": data,
            "sdd": data.attrs["sdd"],
            "pixel_size": data.attrs["pixel1"],
            "au_mesh_current": xr.DataArray(
                data.attrs["RSoXS Au Mesh Current"],
                dims=[energy_dimension],
                coords={
                    "energy": (
                        energy_dimension,
                        data.energy.values,
                    )
                },
            ),
            "ring_current": data.attrs["NSLS-II Ring Current"],
            "date_time": data.attrs["meas_time"],
        }

    return loaded_scans
