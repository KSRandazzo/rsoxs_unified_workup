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
