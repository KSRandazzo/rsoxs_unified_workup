from pathlib import Path
import tomllib
import pandas as pd
from math import isfinite
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"

TABLE_FILES = {
    "samples": "samples.csv",
    "scan_sets": "scan_sets.csv",
    "scans": "scans.csv",
    "polarization_pairs": "polarization_pairs.csv",
    "anisotropy_sectors": "anisotropy_sectors.csv",
}


def get_polarization_layout(scan_set):
    """Read acquisition layout; older CSVs default to separate runs."""
    value = scan_set.get("polarization_layout")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "separate_runs"
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return "separate_runs"
    if value not in ("separate_runs", "within_run"):
        raise ValueError(
            "polarization_layout must be 'separate_runs' or 'within_run', "
            f"not {value!r}"
        )
    return value


def scan_identity(record):
    """Identify a logical scan by actual run ID AND polarization."""
    try:
        raw_id = record["scan_id"]
        raw_pol = record["polarization_deg"]
        if isinstance(raw_id, (bool, np.bool_)) or isinstance(raw_pol, (bool, np.bool_)):
            raise ValueError("Scan IDs and polarizations cannot be booleans")
        scan_id = int(raw_id)
        polarization = float(raw_pol)
        if scan_id <= 0 or float(raw_id) != scan_id or not isfinite(polarization):
            raise ValueError("Invalid scan ID or polarization")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Every scan row needs a positive integer scan_id and finite "
            "numeric polarization_deg"
        ) from error
    return scan_id, int(polarization) if polarization.is_integer() else polarization


def configured_scan_records(configuration):
    """Validate logical identities without reading any measured data."""
    layout = get_polarization_layout(configuration.get("scan_set", {}))
    records = {}
    run_ids = set()
    polarizations = set()
    for record in configuration["scans"]:
        identity = scan_identity(record)
        scan_id, polarization = identity
        if identity in records:
            raise ValueError(f"Duplicate scan ID/polarization: {identity}")
        if polarization in polarizations:
            raise ValueError("The scan set must have distinct polarizations")
        if layout == "separate_runs" and scan_id in run_ids:
            raise ValueError(
                f"Scan {scan_id} supplies multiple polarizations. Set "
                "polarization_layout=within_run in scan_sets.csv."
            )
        records[identity] = record
        run_ids.add(scan_id)
        polarizations.add(polarization)
    if not records:
        raise ValueError("No scans were supplied")
    return records


def pair_scan_records(configuration, pair):
    """Resolve pair members, requiring polarization for ambiguous run IDs."""
    available = configured_scan_records(configuration)
    members = []
    for member in (1, 2):
        try:
            raw_id = pair[f"scan_id_{member}"]
            if isinstance(raw_id, (bool, np.bool_)):
                raise ValueError("Boolean scan ID")
            scan_id = int(raw_id)
            if scan_id <= 0 or float(raw_id) != scan_id:
                raise ValueError("Invalid scan ID")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Pair {pair.get('pair_id')!r}: invalid scan_id_{member}") from error
        matches = [record for identity, record in available.items() if identity[0] == scan_id]
        criterion = pair.get(f"polarization_deg_{member}")
        missing = criterion is None or (
            isinstance(criterion, float) and np.isnan(criterion)
        ) or (isinstance(criterion, str) and not criterion.strip())
        if not missing:
            _, polarization = scan_identity({"scan_id": scan_id, "polarization_deg": criterion})
            matches = [record for record in matches if scan_identity(record)[1] == polarization]
        if len(matches) != 1:
            raise ValueError(
                f"Pair {pair.get('pair_id')!r}: scan {scan_id} matches "
                f"{len(matches)} configured members. Specify the matching "
                f"polarization_deg_{member} in polarization_pairs.csv."
            )
        members.append(matches[0])
    identities = [scan_identity(record) for record in members]
    if identities[0] == identities[1]:
        raise ValueError(f"Pair {pair.get('pair_id')!r} refers to the same run/polarization twice")
    pol_1, pol_2 = (identity[1] for identity in identities)
    separation = abs(((pol_1 - pol_2 + 90) % 180) - 90)
    if abs(separation - 90) > 1e-6:
        raise ValueError(f"Pair {pair.get('pair_id')!r} is separated by {separation:g}°, not 90°")
    return tuple(members)


def get_beam_center(processing, pair_id):
    """Read the configured beam center for one polarization pair."""
    obsolete_keys = {
        "center_mask": ("beam_center_x_guess", "beam_center_y_guess"),
        "beam_center_fit": ("initial_center_x", "initial_center_y"),
    }

    if (
        "beam_center" in processing
        or "shared_beam_center" in processing
        or any(
            key in processing.get(section, {})
            for section, keys in obsolete_keys.items()
            for key in keys
        )
    ):
        raise ValueError(
            "Remove the single or obsolete center-coordinate settings. "
            "Use a separate [beam_centers.<pair_id>] table for each pair."
        )

    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("A nonempty polarization pair_id is required")

    try:
        settings = processing["beam_centers"][pair_id]

        if not isinstance(settings, dict):
            raise TypeError("Center settings must be a table")

        if any(
            isinstance(settings.get(key), bool)
            for key in ("center_x", "center_y")
        ):
            raise ValueError("Center coordinates cannot be true or false")

        center = {
            "x": float(settings["center_x"]),
            "y": float(settings["center_y"]),
        }
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Add numeric center_x and center_y under "
            f"[beam_centers.{pair_id}]."
        ) from error

    if not all(isfinite(value) for value in center.values()):
        raise ValueError(
            f"Beam-center coordinates for pair {pair_id!r} must be finite"
        )

    return center


def get_beam_centers(processing, mask_pairs):
    """Require exactly one configured center for every mask-sharing pair."""
    if not mask_pairs:
        raise ValueError("No polarization pairs were supplied")

    configured = processing.get("beam_centers")
    if not isinstance(configured, dict):
        raise ValueError(
            "Add a [beam_centers.<pair_id>] table for every polarization pair"
        )

    missing = set(mask_pairs) - set(configured)
    unknown = set(configured) - set(mask_pairs)

    if missing or unknown:
        raise ValueError(
            "Beam-center entries must match the configured polarization pairs. "
            f"Missing: {sorted(missing)}; unknown: {sorted(unknown)}"
        )

    return {
        pair_id: get_beam_center(processing, pair_id)
        for pair_id in mask_pairs
    }


def read_config_tables(config_dir=None):
    """Read all configuration CSV files."""
    if config_dir is None:
        config_dir = DEFAULT_CONFIG_DIR
    config_dir = Path(config_dir)
    tables = {}

    for table_name, filename in TABLE_FILES.items():
        path = config_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing configuration file: {path}")

        tables[table_name] = pd.read_csv(path)

    return tables


def read_processing_config(scan_set_id, config_dir=None):
    """Read processing settings for one scan set."""
    if config_dir is None:
        config_dir = DEFAULT_CONFIG_DIR

    path = (
        Path(config_dir)
        / "processing"
        / f"{scan_set_id}.toml"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing processing configuration file: {path}"
        )

    with path.open("rb") as file:
        return tomllib.load(file)


def _single_record(rows, description):
    """Return exactly one row as a dictionary."""
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one {description}, but found {len(rows)}"
        )

    return rows.iloc[0].to_dict()


def _validate_scan_set_metadata(scan_set):
    """Require separate acquisition-type and energy-region metadata."""
    scan_set_id = scan_set["scan_set_id"]
    get_polarization_layout(scan_set)
    scan_type = scan_set.get("scan_type")

    if scan_type not in ("repeats", "long"):
        raise ValueError(
            f"Scan set {scan_set_id!r}: scan_type must be "
            f"'repeats' or 'long', not {scan_type!r}. "
            "Record the energy region separately in energy_region."
        )

    energy_region = scan_set.get("energy_region")

    if (
        not isinstance(energy_region, str)
        or not energy_region.strip()
    ):
        raise ValueError(
            f"Scan set {scan_set_id!r}: energy_region must "
            "contain a nonempty label, such as C or O."
        )


def validate_requested_energies(
    scan_record,
    energy_values,
    requested_energies,
    setting_name,
):
    """Reject out-of-range long-scan requests before nearest selection."""
    scan_id = scan_record["scan_id"]
    scan_type = scan_record.get("scan_type")

    if scan_type == "repeats":
        return

    if scan_type != "long":
        raise ValueError(
            f"Scan {scan_id}: scan_type must be 'repeats' or 'long', "
            f"not {scan_type!r}. Reload the updated configuration "
            "and scan records."
        )

    available = np.asarray(energy_values, dtype=float)
    requested = np.asarray(requested_energies, dtype=float)

    if (
        available.ndim != 1
        or available.size == 0
        or not np.all(np.isfinite(available))
        or np.any(available <= 0)
    ):
        raise ValueError(
            f"Scan {scan_id}: measured energies must be a "
            "nonempty, positive, finite 1D array."
        )

    lower = float(available.min())
    upper = float(available.max())
    available_range = (
        f"Available measured energy range: "
        f"{lower:.12g} to {upper:.12g} eV."
    )

    if (
        requested.ndim != 1
        or requested.size == 0
        or not np.all(np.isfinite(requested))
        or np.any(requested <= 0)
    ):
        raise ValueError(
            f"Scan {scan_id}: {setting_name} must contain "
            f"positive, finite energies. {available_range}"
        )

    # Allow numerical roundoff, not a meaningful extension of the range.
    tolerance = (
        16 * np.finfo(float).eps
        * max(1.0, abs(lower), abs(upper))
    )
    outside = (
        (requested < lower - tolerance)
        | (requested > upper + tolerance)
    )

    if np.any(outside):
        invalid = ", ".join(
            f"{value:.12g} eV" for value in requested[outside]
        )
        raise ValueError(
            f"Scan {scan_id}: {setting_name} requests "
            f"out-of-range energies: {invalid}. "
            f"{available_range} Choose energies within this range."
        )


def load_scan_set(scan_set_id, config_dir=None):
    """Load a scan set and its linked sample, scans, and pairs."""
    tables = read_config_tables(config_dir)

    scan_set = _single_record(
        tables["scan_sets"].loc[
            tables["scan_sets"]["scan_set_id"] == scan_set_id
        ],
        f"scan set named {scan_set_id!r}",
    )

    _validate_scan_set_metadata(scan_set)
    scan_set["polarization_layout"] = get_polarization_layout(scan_set)

    sample = _single_record(
        tables["samples"].loc[
            tables["samples"]["sample_id"] == scan_set["sample_id"]
        ],
        f"sample named {scan_set['sample_id']!r}",
    )

    scans = tables["scans"].loc[
        tables["scans"]["scan_set_id"] == scan_set_id
    ]

    pairs = tables["polarization_pairs"].loc[
        tables["polarization_pairs"]["scan_set_id"] == scan_set_id
    ]

    sectors = tables["anisotropy_sectors"].loc[
        tables["anisotropy_sectors"]["scan_set_id"]
        == scan_set_id
    ]

    if scans.empty:
        raise ValueError(f"No scans found for scan set {scan_set_id!r}")

    return {
        "sample": sample,
        "scan_set": scan_set,
        "scans": scans.to_dict(orient="records"),
        "polarization_pairs": pairs.to_dict(orient="records"),
        "processing": read_processing_config(
            scan_set_id,
            config_dir,
        ),
        "anisotropy_sectors": sectors.to_dict(
            orient="records"
        ),
    }


def get_mask_pairs(configuration):
    """Return disjoint perpendicular pairs of logical run/polarization members."""
    available = configured_scan_records(configuration)
    mask_pairs = {}
    used = set()
    for pair in configuration["polarization_pairs"]:
        pair_id = pair["pair_id"]
        if pair_id in mask_pairs:
            raise ValueError(f"Duplicate polarization pair ID: {pair_id}")
        members = pair_scan_records(configuration, pair)
        identities = [scan_identity(record) for record in members]
        overlap = set(identities) & used
        if overlap:
            raise ValueError(
                f"Run/polarization members {sorted(overlap)} belong to multiple pairs"
            )
        mask_pairs[pair_id] = tuple(identity[1] for identity in identities)
        used.update(identities)
    unpaired = set(available) - used
    if unpaired:
        raise ValueError(f"Run/polarization members without a pair: {sorted(unpaired)}")
    return mask_pairs
