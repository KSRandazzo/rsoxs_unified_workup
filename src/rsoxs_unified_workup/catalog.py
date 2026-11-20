"""Browse saved RSoXS exports using manifest metadata, not image arrays.

The catalog is rebuilt on request. It does not modify exports, current
configuration files, or a persistent index. Latest means newest save time
within each scan_set_id; it does not mean scientifically preferred.
"""

import json
import hashlib
import os
from pathlib import Path
import warnings

import pandas as pd

# Metadata browsing does not import processing configuration or beamline code.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


CATALOG_COLUMNS = (
    "sample_description", "sample_id", "sample_name", "polymer",
    "diameter_nm", "wash_state", "np_vendor", "mw_g_mol", "mn_g_mol", "pdi",
    "scan_set_id", "scan_type", "energy_region", "scan_type_recorded",
    "beam_center_reflection_mode",
    "review_status", "reviewed_at_utc", "reviewer", "review_note",
    "saved_at_utc", "run_id", "is_latest", "scan_ids", "polarizations",
    "scan_count", "pair_count", "group_count", "has_angular_fit",
    "schema_version", "git_commit", "git_working_tree_clean",
    "export_directory", "manifest_path", "zarr_store", "metadata_notes",
)


def _text(value):
    if value is None or isinstance(value, (dict, list)):
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _number_label(value):
    text = _text(value)
    try:
        number = float(text)
    except ValueError:
        return text
    return f"{number:g}" if pd.notna(number) else ""


def _beam_center_reflection_mode(configuration):
    """Read the saved setting, never infer it from current code defaults.

    An explicit JSON null means no reflection, just as in beam centering.
    Missing or malformed metadata is unknown, not evidence of no reflection.
    The setting alone does not establish that center fitting was performed.
    """
    processing = configuration.get("processing")
    if not isinstance(processing, dict):
        return "not_recorded"
    fit_settings = processing.get("beam_center_fit")
    if not isinstance(fit_settings, dict) or "reflection_mode" not in fit_settings:
        return "not_recorded"
    mode = fit_settings["reflection_mode"]
    if mode is None:
        return "none"
    if isinstance(mode, str) and mode.strip():
        return mode.strip().casefold()
    return "not_recorded"


def _export_review_metadata(manifest_path, manifest):
    """Read a separate review history, failing closed on invalid decisions.

    Old manifests without a status are development exports, never approved.
    The latest decision must identify the exact unmodified saved manifest.
    """
    baseline = manifest.get("review_status", "development")
    result = {
        "review_status": baseline if isinstance(baseline, str) and baseline in {"draft", "development"} else "unknown",
        "reviewed_at_utc": None, "reviewer": None, "review_note": "",
    }
    reviews = manifest_path.parent / "reviews"
    try:
        if reviews.is_symlink():
            raise ValueError("Review history is a symbolic link")
        if reviews.exists() and not reviews.is_dir():
            raise ValueError("Review history is not a directory")
        events = sorted(reviews.glob("*.json"))
        if not events:
            return result
        latest = events[-1]
        if latest.is_symlink():
            raise ValueError("Review event is a symbolic link")
        event = json.loads(latest.read_text(encoding="utf-8"))
        if not isinstance(event, dict) or event.get("schema_version") != 1:
            raise ValueError("Unrecognized review record")
        if event.get("event_id") != latest.stem:
            raise ValueError("Review event identity does not match its filename")
        if event.get("scan_set_id") != manifest["scan_set_id"]:
            raise ValueError("Review belongs to a different scan set")
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if event.get("manifest_sha256") != digest:
            raise ValueError("Saved manifest has changed since its latest review")
        status = event.get("status")
        if status not in {"draft", "approved"}:
            raise ValueError("Unrecognized review decision")
        reviewer, note = event.get("reviewer"), event.get("note")
        if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(note, str):
            raise ValueError("Review is missing its reviewer or note")
        if status == "approved" and not note.strip():
            raise ValueError("Approval is missing its review note")
        timestamp = event.get("reviewed_at_utc")
        if not isinstance(timestamp, str):
            raise ValueError("Review is missing its timestamp")
        from datetime import datetime
        reviewed = datetime.fromisoformat(timestamp)
        if reviewed.tzinfo is None:
            raise ValueError("Review timestamp needs a timezone")
        if not latest.stem.startswith(reviewed.strftime("%Y%m%dT%H%M%S%fZ") + "_"):
            raise ValueError("Review timestamp does not match its event identity")
        return {
            "review_status": status, "reviewed_at_utc": timestamp,
            "reviewer": reviewer.strip(), "review_note": note.strip(),
        }
    except (OSError, ValueError, TypeError, KeyError) as error:
        result.update(review_status="unknown", review_note=f"Review needs inspection: {error}")
        return result


def _manifest_paths(root, issues):
    """Find manifests while pruning chunk stores and incomplete directories."""
    def walk_error(error):
        issues.append({"manifest_path": str(error.filename), "reason": str(error)})

    for current, directories, files in os.walk(root, topdown=True, followlinks=False, onerror=walk_error):
        directories[:] = sorted(
            name for name in directories
            if not name.startswith(".")
            and not name.endswith((".zarr", ".incomplete"))
            and not (Path(current) / name).is_symlink()
        )
        if "manifest.json" in files:
            yield Path(current) / "manifest.json"


def _catalog_row(manifest_path):
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must contain a dictionary")
    configuration = manifest.get("configuration", {})
    if not isinstance(configuration, dict):
        raise ValueError("Manifest configuration must be a dictionary")
    sample = configuration.get("sample", {})
    scan_set = configuration.get("scan_set", {})
    if not isinstance(sample, dict) or not isinstance(scan_set, dict):
        raise ValueError("Sample and scan-set metadata must be dictionaries")
    scan_set_id = _text(manifest.get("scan_set_id") or scan_set.get("scan_set_id"))
    if not scan_set_id:
        raise ValueError("Missing scan_set_id")
    configured_id = _text(scan_set.get("scan_set_id"))
    if configured_id and configured_id != scan_set_id:
        raise ValueError("Manifest and configuration scan_set_id disagree")
    generated = manifest.get("generated_at_utc")
    if not isinstance(generated, str) or not generated.strip():
        raise ValueError("Missing save timestamp")
    saved_at = pd.to_datetime(generated, utc=True, errors="raise")
    if pd.isna(saved_at):
        raise ValueError("Invalid save timestamp")

    store_name = manifest.get("zarr_store", "workup.zarr")
    if not isinstance(store_name, str) or not store_name:
        raise ValueError("Invalid Zarr store path")
    relative_store = Path(store_name)
    if relative_store.is_absolute() or ".." in relative_store.parts:
        raise ValueError("Zarr store path must remain inside the export directory")
    store = (manifest_path.parent / relative_store).resolve()
    try:
        store.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise ValueError("Zarr store points outside the export directory") from error
    if not store.is_dir() or not (store / ".zmetadata").is_file() or not (store / ".zgroup").is_file():
        raise ValueError("Missing finalized Zarr store or consolidated metadata")
    groups = manifest.get("zarr_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Missing exported-group dictionary")
    if any(not isinstance(name, str) or not isinstance(variables, list) for name, variables in groups.items()):
        raise ValueError("Malformed exported-group dictionary")

    recorded_type = _text(scan_set.get("scan_type"))
    scan_type = recorded_type.casefold()
    region = _text(scan_set.get("energy_region"))
    notes = []
    # Older manifests used longC/longO before these fields were separated.
    # Normalize the catalog view only; never rewrite the historical manifest.
    if scan_type.startswith("long") and len(recorded_type) > 4:
        if not region:
            region = recorded_type[4:].upper()
        scan_type = "long"
        notes.append(f"Legacy scan_type {recorded_type!r} displayed as long/{region}")
    if not region:
        notes.append("Energy region was not recorded")
    reflection_mode = _beam_center_reflection_mode(configuration)
    if reflection_mode == "not_recorded":
        notes.append("Beam-center mask reflection mode was not recorded or is unreadable")

    sample_id = _text(sample.get("sample_id") or scan_set.get("sample_id"))
    sample_name = _text(sample.get("sample_name"))
    polymer = _text(sample.get("polymer"))
    wash_state = _text(sample.get("wash_state"))
    details = [polymer]
    diameter = _number_label(sample.get("diameter_nm"))
    if diameter:
        details.append(f"{diameter} nm")
    details.append(wash_state)
    description = " | ".join(part for part in (
        sample_name or sample_id or scan_set_id,
        ", ".join(part for part in details if part),
        " ".join(part for part in (scan_type, region) if part),
    ) if part)
    scans = configuration.get("scans", [])
    pairs = configuration.get("polarization_pairs", [])
    if not isinstance(scans, list) or any(not isinstance(scan, dict) for scan in scans):
        raise ValueError("Scan metadata must be a list of dictionaries")
    if not isinstance(pairs, list):
        raise ValueError("Polarization-pair metadata must be a list")

    return {
        "sample_description": description,
        "sample_id": sample_id or None,
        "sample_name": sample_name or None,
        "polymer": polymer or None,
        "diameter_nm": sample.get("diameter_nm"),
        "wash_state": wash_state or None,
        **{name: sample.get(name) for name in ("np_vendor", "mw_g_mol", "mn_g_mol", "pdi")},
        "scan_set_id": scan_set_id,
        "scan_type": scan_type or None,
        "energy_region": region or None,
        "scan_type_recorded": recorded_type or None,
        "beam_center_reflection_mode": reflection_mode,
        **_export_review_metadata(manifest_path, manifest),
        "saved_at_utc": saved_at,
        "run_id": manifest_path.parent.parent.name,
        "is_latest": False,
        "scan_ids": ", ".join(_number_label(scan.get("scan_id")) for scan in scans),
        "polarizations": ", ".join(
            f"{_number_label(scan['polarization_deg'])}°"
            for scan in scans if scan.get("polarization_deg") is not None
        ),
        "scan_count": len(scans),
        "pair_count": len(pairs),
        "group_count": len(groups),
        "has_angular_fit": any(name.endswith("/angular_anisotropy_fit") for name in groups),
        "schema_version": manifest.get("schema_version"),
        "git_commit": manifest.get("git_commit"),
        "git_working_tree_clean": manifest.get("git_working_tree_clean"),
        "export_directory": str(manifest_path.parent.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "zarr_store": str(store),
        "metadata_notes": "; ".join(notes),
    }


def build_results_catalog(results_root=None):
    """Return one row per saved export, including every historical version.

    Read only manifest JSON and check for finalized store metadata files;
    do not open arrays or walk Zarr chunk directories. This is discovery,
    not a replacement for the workup's full export-verification cell.

    Missing, malformed, and unfinished exports are skipped and reported in
    catalog.attrs['skipped_exports'], with a summary warning. Metadata comes
    from each saved snapshot, not from the current configuration CSV files.
    """
    root = Path(results_root) if results_root is not None else PROJECT_ROOT / "results"
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {root}")
    if root.name.endswith((".zarr", ".incomplete")):
        raise ValueError("Select a results directory, not a Zarr store or incomplete export")
    rows = []
    issues = []
    for path in _manifest_paths(root, issues):
        try:
            rows.append(_catalog_row(path))
        except (OSError, ValueError, TypeError) as error:
            issues.append({"manifest_path": str(path), "reason": str(error)})
    catalog = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    catalog["saved_at_utc"] = pd.to_datetime(catalog["saved_at_utc"], utc=True)
    for name in ("is_latest", "has_angular_fit"):
        catalog[name] = catalog[name].astype(bool)
    if not catalog.empty:
        # Manifest path makes equal-timestamp ties deterministic. This does
        # not imply that either export is scientifically preferable.
        catalog = catalog.sort_values(
            ["saved_at_utc", "manifest_path"], ascending=[False, True], kind="stable",
        ).reset_index(drop=True)
        catalog["is_latest"] = ~catalog["scan_set_id"].duplicated()
    catalog.attrs = {"results_root": str(root), "skipped_exports": issues}
    if issues:
        warnings.warn(
            f"Skipped {len(issues)} unavailable or malformed export locations. "
            "See catalog.attrs['skipped_exports'] for paths and reasons.",
            UserWarning, stacklevel=2,
        )
    return catalog


def select_results(
    catalog, *, latest_only=True, sample_id=None, sample_name=None,
    scan_type=None, energy_region=None, polymer=None, wash_state=None, search=None,
    reflection_mode=None,
    review_status=None,
):
    """Filter a catalog without modifying it or deleting saved versions.

    Named metadata filters are case-insensitive exact matches. Each accepts
    a single value or a list/tuple/set of values. Search is a literal,
    case-insensitive substring over the description and sample/scan-set IDs.
    Review status is selected BEFORE latest: an approved export is not
    hidden by a newer draft. Latest is then determined within that status
    selection BEFORE the other filters (including reflection mode).
    reflection_mode=None includes all settings; the string "none" selects
    no reflection and "not_recorded" selects missing/unreadable metadata.
    """
    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("catalog must be a pandas DataFrame")
    if not isinstance(latest_only, bool):
        raise ValueError("latest_only must be True or False")
    required = set(CATALOG_COLUMNS)
    if not required.issubset(catalog.columns):
        raise ValueError("Use the DataFrame returned by build_results_catalog")
    selected = catalog.copy()
    if review_status is not None:
        values = review_status if isinstance(review_status, (list, tuple, set)) else [review_status]
        if any(isinstance(value, (dict, list, tuple, set)) for value in values):
            raise ValueError("review_status: use one value or a flat list of values")
        accepted = {_text(value).casefold() for value in values}
        selected = selected.loc[selected["review_status"].isin(accepted)]
    if latest_only:
        selected = selected.sort_values(
            ["saved_at_utc", "manifest_path"], ascending=[False, True], kind="stable",
        ).drop_duplicates("scan_set_id", keep="first")
    for name, criterion in (
        ("sample_id", sample_id), ("sample_name", sample_name),
        ("scan_type", scan_type), ("energy_region", energy_region),
        ("polymer", polymer), ("wash_state", wash_state),
        ("beam_center_reflection_mode", reflection_mode),
    ):
        if criterion is None:
            continue
        values = criterion if isinstance(criterion, (list, tuple, set)) else [criterion]
        if any(isinstance(value, (dict, list, tuple, set)) for value in values):
            raise ValueError(f"{name}: use one value or a flat list of values")
        accepted = {_text(value).casefold() for value in values}
        mask = selected[name].fillna("").astype(str).str.casefold().isin(accepted)
        selected = selected.loc[mask]
    if search is not None:
        if not isinstance(search, str):
            raise TypeError("search must be a string or None")
        needle = search.strip().casefold()
        mask = pd.Series(False, index=selected.index)
        for name in ("sample_description", "sample_id", "sample_name", "scan_set_id"):
            mask |= selected[name].fillna("").astype(str).str.casefold().str.contains(needle, regex=False)
        selected = selected.loc[mask]
    selected = selected.sort_values(
        ["sample_description", "scan_set_id", "saved_at_utc", "manifest_path"],
        ascending=[True, True, False, True], kind="stable",
    ).reset_index(drop=True)
    selected.attrs = dict(catalog.attrs)
    return selected
