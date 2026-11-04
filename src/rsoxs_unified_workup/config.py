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


def get_beam_center(processing):
    """Read the single configured center used throughout the workup."""
    obsolete_keys = {
        "center_mask": ("beam_center_x_guess", "beam_center_y_guess"),
        "beam_center_fit": ("initial_center_x", "initial_center_y"),
    }

    if "shared_beam_center" in processing or any(
        key in processing.get(section, {})
        for section, keys in obsolete_keys.items()
        for key in keys
    ):
        raise ValueError(
            "Remove the old center-coordinate settings. "
            "Use only [beam_center] center_x and center_y."
        )

    try:
        settings = processing["beam_center"]
        center = {
            "x": float(settings["center_x"]),
            "y": float(settings["center_y"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Add numeric center_x and center_y under [beam_center]."
        ) from error

    if not all(isfinite(value) for value in center.values()):
        raise ValueError("Beam-center coordinates must be finite")

    return center


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