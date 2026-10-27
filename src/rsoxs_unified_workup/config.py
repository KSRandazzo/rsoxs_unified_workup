from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"

TABLE_FILES = {
    "samples": "samples.csv",
    "scan_sets": "scan_sets.csv",
    "scans": "scans.csv",
    "polarization_pairs": "polarization_pairs.csv",
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

    if scans.empty:
        raise ValueError(f"No scans found for scan set {scan_set_id!r}")

    return {
        "sample": sample,
        "scan_set": scan_set,
        "scans": scans.to_dict(orient="records"),
        "polarization_pairs": pairs.to_dict(orient="records"),
    }
