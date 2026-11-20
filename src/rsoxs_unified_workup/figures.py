"""Archive saved notebook PNG outputs beside one completed workup export.

No scientific calculations are rerun. This preserves rendered notebook
outputs, not Matplotlib objects, and cannot prove each plot's data provenance.
"""

import base64
import binascii
import hashlib
import json
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _text(value):
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "".join(value)
    if isinstance(value, str):
        return value
    raise ValueError("Notebook text and PNG data must be strings or lists of strings")


def _check_notebook_export(notebook, paths):
    save_cells = [
        cell for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "save_summary" in _text(cell.get("source", ""))
        and "save_workup_results" in _text(cell.get("source", ""))
    ]
    if not save_cells:
        raise ValueError(
            "The saved notebook has no workup-save cell. "
            "Choose the notebook that produced this export."
        )
    save_cell = save_cells[-1]
    recorded = {}
    for output in save_cell.get("outputs", []):
        if output.get("output_type") == "error":
            raise ValueError("The saved notebook's last workup-save cell contains an error")
        if output.get("output_type") == "stream":
            text = _text(output.get("text", ""))
        else:
            text = _text(output.get("data", {}).get("text/plain", ""))
        for line in text.splitlines():
            key, separator, value = line.partition(": ")
            if separator and key in paths:
                recorded[key] = value.strip()
    if set(recorded) != set(paths):
        raise ValueError(
            "The saved notebook does not contain the completed save_summary output. "
            "Save the notebook with its outputs (Ctrl+S) after the Zarr save cell, "
            "then rerun only the figure-archive cell. Do not rerun the Zarr save."
        )
    if any(Path(recorded[key]).resolve() != path for key, path in paths.items()):
        raise ValueError(
            "The saved notebook records a different export from save_summary. "
            "Save the correct notebook with its current outputs (Ctrl+S), "
            "then rerun only the figure-archive cell."
        )


def _png_outputs(notebook):
    figures = []
    other_rich_outputs = []
    for cell_number, cell in enumerate(notebook["cells"], 1):
        for output_number, output in enumerate(cell.get("outputs", []), 1):
            data = output.get("data", {})
            if "image/png" not in data:
                mime_types = sorted(key for key in data if key != "text/plain")
                if mime_types:
                    other_rich_outputs.append({
                        "cell_number": cell_number,
                        "output_number": output_number,
                        "mime_types": mime_types,
                    })
                continue
            try:
                encoded = "".join(_text(data["image/png"]).split())
                png = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError(f"Invalid PNG encoding in notebook cell {cell_number}") from error
            if len(png) < 33 or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
                raise ValueError(f"Invalid PNG header in notebook cell {cell_number}")
            width, height = struct.unpack(">II", png[16:24])
            if not width or not height:
                raise ValueError(f"Invalid PNG dimensions in notebook cell {cell_number}")
            filename = f"cell_{cell_number:03d}_output_{output_number:03d}.png"
            figures.append((png, {
                "filename": filename,
                "cell_number": cell_number,
                "cell_id": cell.get("id"),
                "execution_count": cell.get("execution_count"),
                "output_number": output_number,
                "width_pixels": width,
                "height_pixels": height,
                "byte_count": len(png),
                "sha256": hashlib.sha256(png).hexdigest(),
            }))
    if not figures:
        raise ValueError(
            "No PNG outputs are stored in this notebook. "
            "Save it with the displayed figure outputs retained first."
        )
    return figures, other_rich_outputs


def save_workup_figures(*, save_summary, configuration, notebook_path):
    """Save an immutable PNG archive tied to the supplied completed export.

    The caller must first save the executed notebook with its outputs. The
    saved notebook's last workup-save output, live configuration, and on-disk
    export manifest must agree. Existing figure archives are never replaced.
    Interactive/widget/HTML-only outputs stay in the notebook snapshot; they
    are not rasterized. PNGs retain their original displayed resolution.
    """
    # Reuse the export's scientific-to-JSON conversion, including NaN handling.
    from rsoxs_unified_workup.outputs import _json_ready

    paths = {
        key: Path(save_summary[key]).resolve()
        for key in ("output_directory", "manifest", "zarr_store")
    }
    export_directory = paths["output_directory"]
    if paths["manifest"] != export_directory / "manifest.json" or paths["zarr_store"] != export_directory / "workup.zarr":
        raise ValueError("save_summary paths do not identify one workup export directory")
    store = paths["zarr_store"]
    if not store.is_dir() or not (store / ".zgroup").is_file() or not (store / ".zmetadata").is_file():
        raise ValueError("save_summary does not point to a finalized, consolidated Zarr export")
    manifest_bytes = paths["manifest"].read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("zarr_store") != "workup.zarr":
        raise ValueError("The export manifest identifies a different Zarr store")
    if set(save_summary["groups"]) != set(manifest["zarr_groups"]):
        raise ValueError("save_summary groups do not match the export manifest")
    if manifest["configuration"] != _json_ready(configuration):
        raise ValueError(
            "The current configuration differs from this export's saved configuration. "
            "Archive figures before switching scan sets or changing processing settings."
        )
    scan_set_id = str(configuration["scan_set"]["scan_set_id"])
    if manifest["scan_set_id"] != scan_set_id:
        raise ValueError("The export and current configuration identify different scan sets")

    notebook_path = Path(notebook_path).resolve()
    notebook_bytes = notebook_path.read_bytes()
    notebook = json.loads(notebook_bytes)
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("The notebook file does not contain a valid cells list")
    _check_notebook_export(notebook, paths)
    figures, other_rich_outputs = _png_outputs(notebook)

    destination = export_directory / "figures"
    if destination.exists():
        raise FileExistsError(
            f"A figure archive already exists at {destination}. "
            "It has been kept unchanged; no files were overwritten."
        )
    figure_manifest = {
        "schema_version": 1,
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "scan_set_id": scan_set_id,
        "workup_export": {
            "export_directory": str(export_directory),
            "manifest": "../manifest.json",
            "zarr_store": "../workup.zarr",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "generated_at_utc": manifest["generated_at_utc"],
            "git_commit": manifest.get("git_commit"),
        },
        "source_notebook": {
            "original_path": str(notebook_path),
            "snapshot": "workup.ipynb",
            "sha256": hashlib.sha256(notebook_bytes).hexdigest(),
        },
        "figure_count": len(figures),
        "figures": [metadata for png, metadata in figures],
        "non_png_rich_outputs": other_rich_outputs,
        "provenance_note": (
            "Archived PNG outputs from the saved notebook whose workup-save output "
            "matches this export. This does not establish that every plotting cell "
            "was rerun for the current data; clear stale outputs before archiving."
        ),
    }
    # Publish only after every PNG, snapshot, and index is written successfully.
    with tempfile.TemporaryDirectory(prefix=".figures.incomplete-", dir=export_directory) as temporary:
        staging = Path(temporary)
        for png, metadata in figures:
            (staging / metadata["filename"]).write_bytes(png)
        (staging / "workup.ipynb").write_bytes(notebook_bytes)
        (staging / "figure_manifest.json").write_text(
            json.dumps(figure_manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(f"A figure archive already exists at {destination}")
        staging.rename(destination)
    return {
        "figure_directory": destination,
        "figure_count": len(figures),
        "notebook_snapshot": destination / "workup.ipynb",
        "figure_manifest": destination / "figure_manifest.json",
        "non_png_rich_output_count": len(other_rich_outputs),
    }
