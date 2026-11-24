from pathlib import Path
import tomllib
import pandas as pd
from math import isfinite
from decimal import Decimal, InvalidOperation
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

POLARIZATION_SCAN_COLUMNS = {
    0: "pol0_scan_id",
    90: "pol90_scan_id",
    45: "pol45_scan_id",
    135: "pol135_scan_id",
}

STANDARD_SECTORS = {
    (0, 90): (("12oclock", 90), ("9oclock", 180),
              ("6oclock", -90), ("3oclock", 0)),
    (45, 135): (("10p30oclock", 135), ("7p30oclock", -135),
               ("4p30oclock", -45), ("1p30oclock", 45)),
}


def get_review_energies(processing, *, include_beam_center=True):
    """Read the shared TOML energy list without changing any settings.

    The independent centering energy is included for plots, but not added
    to the user's interesting-energy list for angular slice fitting.
    """
    plotting = processing.get("plotting", {})
    if "energy_indices" in plotting:
        raise ValueError("Replace plotting.energy_indices with plotting.energies_ev")
    configured = plotting.get("energies_ev")
    if configured is None:
        energies = []
    elif not isinstance(configured, list) or not configured:
        raise ValueError("plotting.energies_ev must be a nonempty list")
    else:
        energies = list(configured)
    if include_beam_center:
        fit_energy = processing.get("beam_center_fit", {}).get("fit_energy")
        if fit_energy is not None:
            energies.append(fit_energy)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value) or value <= 0
        for value in energies
    ):
        raise ValueError("Review energies must be positive, finite numbers in eV")
    return list(dict.fromkeys(float(value) for value in energies))


def uses_compact_scan_sets(columns):
    """The four run-ID columns opt a configuration into generated records."""
    required = set(POLARIZATION_SCAN_COLUMNS.values())
    present = required.intersection(columns)
    if present and present != required:
        raise ValueError(
            "scan_sets.csv must include all four polarization run-ID columns. "
            f"Missing: {sorted(required - present)}. Leave absent polarizations blank."
        )
    return bool(present)


def _optional_run_id(value, column, scan_set_id):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ("", "none", "n/a", "na", "nan"):
            return None
    try:
        if isinstance(value, str):
            number = Decimal(value)
            if not number.is_finite() or number != number.to_integral_value():
                raise ValueError("Noninteger run ID")
            value = int(number)
        scan_id, _ = scan_identity({"scan_id": value, "polarization_deg": 0})
    except (ValueError, InvalidOperation) as error:
        raise ValueError(
            f"Scan set {scan_set_id!r}: {column} must be a positive integer "
            "run ID, or blank/N/A/None."
        ) from error
    return scan_id


def get_anisotropy_sector_halfwidth(processing):
    """Generated nonoverlapping sectors default to a 45-degree halfwidth."""
    settings = processing.get("anisotropy", {})
    if not isinstance(settings, dict):
        raise ValueError("[anisotropy] must be a TOML table")
    value = settings.get("chi_halfwidth_degrees", 45)
    try:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Boolean halfwidth")
        halfwidth = float(value)
        if not isfinite(halfwidth) or not 0 < halfwidth <= 45:
            raise ValueError("Invalid halfwidth")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "[anisotropy] chi_halfwidth_degrees must be finite and greater "
            "than 0 but no greater than 45; wider standard sectors overlap."
        ) from error
    return halfwidth


def generate_scan_set_records(scan_set, processing=None):
    """Derive logical scans, perpendicular pairs, and standard sectors.

    Only run IDs explicitly entered in scan_sets.csv are used. Nothing is
    downloaded and no files are written. IDs match the legacy conventions.
    """
    if not uses_compact_scan_sets(scan_set.keys()):
        raise ValueError("The four polarization run-ID columns are required")
    _validate_scan_set_metadata(scan_set)
    scan_set_id = scan_set["scan_set_id"]
    run_ids = {
        pol: _optional_run_id(scan_set[column], column, scan_set_id)
        for pol, column in POLARIZATION_SCAN_COLUMNS.items()
    }
    scans = [
        {"scan_id": run_id, "scan_set_id": scan_set_id, "polarization_deg": pol}
        for pol, run_id in run_ids.items() if run_id is not None
    ]
    if not scans:
        raise ValueError(f"Scan set {scan_set_id!r}: no polarization run IDs supplied")
    pairs, sectors = [], []
    halfwidth = get_anisotropy_sector_halfwidth(processing or {})
    for (first, second), sector_definitions in STANDARD_SECTORS.items():
        first_id, second_id = run_ids[first], run_ids[second]
        if first_id is None and second_id is None:
            continue
        if first_id is None or second_id is None:
            missing = first if first_id is None else second
            raise ValueError(
                f"Scan set {scan_set_id!r}: incomplete {first}°/{second}° pair. "
                f"Supply {POLARIZATION_SCAN_COLUMNS[missing]}, or leave both members blank."
            )
        pair_id = f"{scan_set_id}_pair_{first}_{second}"
        pairs.append({
            "pair_id": pair_id, "scan_set_id": scan_set_id,
            "scan_id_1": first_id, "scan_id_2": second_id,
            "polarization_deg_1": first, "polarization_deg_2": second,
        })
        sectors.extend({
            "sector_id": f"{scan_set_id}_{label}", "scan_set_id": scan_set_id,
            "pair_id": pair_id, "label": label, "chi_center_deg": center,
            "chi_halfwidth_deg": halfwidth,
        } for label, center in sector_definitions)
    records = {
        "scans": scans, "polarization_pairs": pairs, "anisotropy_sectors": sectors,
    }
    # Reuse the existing identity/pair guards, including acquisition layout.
    get_mask_pairs({"scan_set": scan_set, **records})
    return records


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
    """Read manual CSVs; read legacy linked tables only without run-ID columns.

    In compact mode only samples.csv and scan_sets.csv are inputs. The old
    three linked CSVs may remain on disk, but are not read or used as fallback.
    """
    if config_dir is None:
        config_dir = DEFAULT_CONFIG_DIR
    config_dir = Path(config_dir)
    tables = {}

    for table_name in ("samples", "scan_sets"):
        filename = TABLE_FILES[table_name]
        path = config_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing configuration file: {path}")

        tables[table_name] = pd.read_csv(path)

    if not uses_compact_scan_sets(tables["scan_sets"].columns):
        for table_name in ("scans", "polarization_pairs", "anisotropy_sectors"):
            path = config_dir / TABLE_FILES[table_name]
            if not path.exists():
                raise FileNotFoundError(f"Missing legacy configuration file: {path}")
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
    """Load manual inputs and generate records, or read the legacy layout."""
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

    if uses_compact_scan_sets(tables["scan_sets"].columns):
        processing = read_processing_config(scan_set_id, config_dir)
        return {
            "sample": sample, "scan_set": scan_set, "processing": processing,
            **generate_scan_set_records(scan_set, processing),
        }

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
