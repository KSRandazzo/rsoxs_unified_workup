"""Preview or archive the old development results without deleting data.

Stop all workup saves before applying. A fixed inventory token prevents
archiving a different batch from the one you inspected. Only the project's
exact results directory is moved; configurations and raw source data stay put.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid


def development_inventory(project_root):
    """Inspect small metadata only; never read Zarr chunks or change files."""
    project = Path(project_root).resolve(strict=True)
    if not (project / "config").is_dir() or not (project / "notebooks").is_dir():
        raise ValueError("Select the workup project containing config/ and notebooks/")
    root = project / "results"
    if root.is_symlink() or not root.is_dir():
        raise ValueError("The exact project results/ directory must exist and not be a symbolic link")
    entries = []
    manifests = []
    incomplete = []
    def walk_error(error):
        raise error

    for current, directories, files in os.walk(root, topdown=True, followlinks=False, onerror=walk_error):
        for name in sorted(directories + files):
            path = Path(current) / name
            if path.is_symlink():
                raise ValueError(f"Inspect the symbolic link before archiving: {path}")
            relative = str(path.relative_to(root))
            if name.endswith(".incomplete"):
                incomplete.append(relative)
            # Include small metadata and tree identities; never descend into
            # finalized or incomplete Zarr chunk stores.
            if path.is_file():
                entries.append([relative, path.stat().st_size, path.stat().st_mtime_ns])
            else:
                entries.append([relative, "directory"])
        directories[:] = sorted(name for name in directories if not name.endswith((".zarr", ".incomplete")))
        if "manifest.json" in files:
            path = Path(current) / "manifest.json"
            content = path.read_bytes()
            manifest = json.loads(content)
            if not isinstance(manifest, dict):
                raise ValueError(f"Inspect the unreadable manifest: {path}")
            if manifest.get("review_status", "development") != "development" or (path.parent / "reviews").exists():
                raise ValueError(f"Refusing to archive a new or reviewed export as development: {path}")
            manifests.append({
                "manifest": str(path.relative_to(root)),
                "sha256": hashlib.sha256(content).hexdigest(),
                "scan_set_id": manifest.get("scan_set_id"),
            })
    if not entries:
        raise ValueError("The active results directory is already empty; nothing to archive")
    payload = {"entries": sorted(entries), "manifests": sorted(manifests, key=lambda row: row["manifest"])}
    token = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {
        "results_root": str(root), "inventory_sha256": token,
        "manifest_count": len(manifests), "manifests": manifests,
        "incomplete_locations": incomplete,
        "top_level_entries": sorted(path.name for path in root.iterdir()),
    }


def archive_development_results(project_root, *, expected_inventory, confirm=False):
    """Move the inspected batch as a unit and create a fresh results/ folder.

    Nothing is deleted or overwritten. On ordinary failure the move is
    reversed when possible; the archive index records recovery paths.
    A directory rename requires the archive to be on the same filesystem.
    """
    if confirm is not True:
        raise ValueError("Explicit confirm=True is required to move development results")
    project = Path(project_root).resolve(strict=True)
    inventory = development_inventory(project)
    if expected_inventory != inventory["inventory_sha256"]:
        raise ValueError("The results inventory changed. Preview again before archiving; nothing was moved")
    archive_root = project / "results_development"
    if archive_root.is_symlink():
        raise ValueError("results_development/ must not be a symbolic link")
    archive_root.mkdir(exist_ok=True)
    lock = archive_root / ".archive.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("An archive operation is active, or results_development/.archive.lock needs inspection") from error
    os.close(descriptor)
    batch = None
    source = project / "results"
    moved = False
    try:
        if development_inventory(project)["inventory_sha256"] != expected_inventory:
            raise ValueError("Results changed before the move; stop all workup saves and preview again")
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
        batch = archive_root / batch_id
        batch.mkdir()
        destination = batch / "results"
        index = {
            "schema_version": 1, "classification": "development",
            "archived_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_results_root": str(source), "archived_results_root": str(destination),
            "inventory": inventory,
            "recovery": "Archived files are intact. Browse the archived_results_root or move individual export directories back deliberately.",
        }
        with (batch / "archive_manifest.json").open("x", encoding="utf-8") as file:
            json.dump(index, file, indent=2, sort_keys=True)
            file.write("\n")
        source.rename(destination)
        moved = True
        source.mkdir()
        return {
            "archived_results_root": destination, "active_results_root": source,
            "archive_manifest": batch / "archive_manifest.json",
            "manifest_count": inventory["manifest_count"],
        }
    except Exception:
        if moved and not source.exists():
            (batch / "results").rename(source)
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--expected-inventory")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-development", action="store_true")
    arguments = parser.parse_args()
    if arguments.apply:
        result = archive_development_results(
            arguments.project_root, expected_inventory=arguments.expected_inventory,
            confirm=arguments.confirm_development,
        )
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        inventory = development_inventory(arguments.project_root)
        print(json.dumps(inventory, indent=2))
        print("Preview only; no files moved or written. Stop all workup saves before applying.")


if __name__ == "__main__":
    main()
