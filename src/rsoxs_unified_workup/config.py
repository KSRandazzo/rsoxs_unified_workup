from pathlib import Path
import tomllib
import pandas as pd
from math import isfinite


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"

TABLE_FILES = {
    "samples": "samples.csv",
    "scan_sets": "scan_sets.csv",
    "scans": "scans.csv",
    "polarization_pairs": "polarization_pairs.csv",
    "anisotropy_sectors": "anisotropy_sectors.csv",
}


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


def load_scan_set(scan_set_id, config_dir=None):
    """Load a scan set and its linked sample, scans, and pairs."""
    tables = read_config_tables(config_dir)

    scan_set = _single_record(
        tables["scan_sets"].loc[
            tables["scan_sets"]["scan_set_id"] == scan_set_id
        ],
        f"scan set named {scan_set_id!r}",
    )

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
    """Return non-overlapping perpendicular pairs for mask sharing."""
    scan_records = configuration["scans"]
    if not scan_records:
        raise ValueError("No scans were supplied for mask pairing")

    polarization_by_scan_id = {}
    for scan_record in scan_records:
        scan_id = int(scan_record["scan_id"])
        polarization = float(scan_record["polarization_deg"])

        if scan_id in polarization_by_scan_id:
            raise ValueError(f"Duplicate scan ID: {scan_id}")
        if not isfinite(polarization):
            raise ValueError(
                f"Scan {scan_id} has a nonfinite polarization"
            )

        if polarization.is_integer():
            polarization = int(polarization)

        polarization_by_scan_id[scan_id] = polarization

    if len(set(polarization_by_scan_id.values())) != len(scan_records):
        raise ValueError(
            "Mask pairing requires distinct polarizations in the scan set"
        )

    mask_pairs = {}
    used_scan_ids = set()

    for pair in configuration["polarization_pairs"]:
        pair_id = pair["pair_id"]
        scan_ids = (
            int(pair["scan_id_1"]),
            int(pair["scan_id_2"]),
        )

        if pair_id in mask_pairs:
            raise ValueError(f"Duplicate polarization pair ID: {pair_id}")
        if scan_ids[0] == scan_ids[1]:
            raise ValueError(f"Pair {pair_id!r} refers to the same scan twice")

        missing = set(scan_ids) - set(polarization_by_scan_id)
        if missing:
            raise ValueError(
                f"Pair {pair_id!r} refers to scans outside this scan set: "
                f"{sorted(missing)}"
            )

        overlap = set(scan_ids) & used_scan_ids
        if overlap:
            raise ValueError(
                f"Scans {sorted(overlap)} belong to more than one pair. "
                "Pairwise masks require one pair per scan."
            )

        pol_1, pol_2 = (
            polarization_by_scan_id[scan_id]
            for scan_id in scan_ids
        )
        separation = abs(((pol_1 - pol_2 + 90) % 180) - 90)
        if abs(separation - 90) > 1e-6:
            raise ValueError(
                f"Pair {pair_id!r} is separated by "
                f"{separation:g}°, not 90°"
            )

        mask_pairs[pair_id] = (pol_1, pol_2)
        used_scan_ids.update(scan_ids)

    unpaired = set(polarization_by_scan_id) - used_scan_ids
    if unpaired:
        raise ValueError(
            f"Scans without a polarization pair: {sorted(unpaired)}"
        )

    return mask_pairs