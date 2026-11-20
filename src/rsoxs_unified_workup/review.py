"""Explicit, append-only review decisions for one saved export.

Uses only the standard library. Never rewrites a manifest, Zarr store, or
figure archive. Approval is a human decision, not a fit-quality calculation.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid


def _checked_export(export_directory):
    directory = Path(export_directory).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Select one export directory, not a results root or file")
    path = directory / "manifest.json"
    if path.is_symlink():
        raise ValueError("The export manifest must not be a symbolic link")
    content = path.read_bytes()
    manifest = json.loads(content)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("scan_set_id"), str):
        raise ValueError("Missing export scan_set_id")
    if not manifest["scan_set_id"].strip() or not manifest.get("zarr_groups"):
        raise ValueError("Missing export identity or groups")
    store_name = manifest.get("zarr_store", "workup.zarr")
    if not isinstance(store_name, str) or not store_name:
        raise ValueError("Invalid Zarr store path")
    relative = Path(store_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Zarr store must remain inside the export directory")
    store = (directory / relative).resolve()
    if not store.is_relative_to(directory):
        raise ValueError("Zarr store points outside the export directory")
    if not (store / ".zgroup").is_file() or not (store / ".zmetadata").is_file():
        raise ValueError("The export does not have finalized Zarr metadata")
    return directory, manifest, hashlib.sha256(content).hexdigest()


def set_export_review_status(
    export_directory, status, *, reviewer, note="", confirm=False,
    expected_manifest_sha256=None,
):
    """Append an approved/draft decision; draft can revoke an earlier approval.

    confirm=True is required for either decision. approved additionally
    requires a review note. The optional expected manifest digest ties a
    decision to a previously verified snapshot. This does not verify arrays
    or establish scientific suitability automatically.
    """
    if confirm is not True:
        raise ValueError("Explicit confirm=True is required to record a review")
    if not isinstance(status, str) or status.strip().casefold() not in {"draft", "approved"}:
        raise ValueError("Review status must be 'draft' or 'approved'")
    status = status.strip().casefold()
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Supply the reviewer's name or username")
    if not isinstance(note, str) or (status == "approved" and not note.strip()):
        raise ValueError("Approval requires a note describing what you reviewed")
    directory, manifest, digest = _checked_export(export_directory)
    if expected_manifest_sha256 is not None and expected_manifest_sha256 != digest:
        raise ValueError("The export manifest changed since verification; review this export again")
    reviews = directory / "reviews"
    if reviews.is_symlink():
        raise ValueError("The reviews directory must not be a symbolic link")
    reviews.mkdir(exist_ok=True)
    lock = reviews / ".review.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("Another review is being written, or a stale reviews/.review.lock needs inspection") from error
    os.close(descriptor)
    temporary = None
    try:
        # Keep event names monotonically increasing even if the clock moves
        # backwards. Refuse to write over an unreadable history.
        previous = sorted(reviews.glob("*.json"))
        if previous:
            last = previous[-1]
            if last.is_symlink():
                raise ValueError("Review events must not be symbolic links")
            previous_event = json.loads(last.read_text(encoding="utf-8"))
            previous_time = datetime.fromisoformat(previous_event["reviewed_at_utc"])
            if previous_time.tzinfo is None or previous_event.get("event_id") != last.stem:
                raise ValueError("Unreadable review history; inspect it before recording another decision")
        else:
            previous_time = None
        now = datetime.now(timezone.utc)
        if previous_time is not None and now <= previous_time:
            from datetime import timedelta
            now = previous_time + timedelta(microseconds=1)
        event_id = now.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex
        event = {
            "schema_version": 1,
            "event_id": event_id,
            "scan_set_id": manifest["scan_set_id"],
            "manifest_sha256": digest,
            "reviewed_at_utc": now.isoformat(),
            "status": status,
            "reviewer": reviewer.strip(),
            "note": note.strip(),
        }
        temporary = reviews / ("." + event_id + ".incomplete")
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(event, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        if _checked_export(directory)[2] != digest:
            raise ValueError("The manifest changed while recording the review; no decision was published")
        destination = reviews / (event_id + ".json")
        # An exclusive hard link publishes complete JSON atomically and never
        # overwrites another event. Both paths are on the same filesystem.
        os.link(temporary, destination)
        return {"export_directory": directory, "review_status": status, "review_record": destination}
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_directory", help="Exact directory containing manifest.json")
    parser.add_argument("status", choices=("draft", "approved"))
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--confirm", action="store_true", help="Explicitly record the decision")
    arguments = parser.parse_args()
    result = set_export_review_status(
        arguments.export_directory, arguments.status, reviewer=arguments.reviewer,
        note=arguments.note, confirm=arguments.confirm,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
