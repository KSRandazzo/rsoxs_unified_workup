"""Synthetic archive checks: no measured data loading or plotting reruns."""

import base64
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rsoxs_unified_workup import figures


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQAAAABJRU5ErkJggg=="
)


class FigureArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.export = self.root / "results" / "runs" / "timestamp" / "synthetic"
        self.export.mkdir(parents=True)
        store = self.export / "workup.zarr"
        store.mkdir()
        (store / ".zgroup").write_text('{"zarr_format": 2}')
        (store / ".zmetadata").write_text('{"zarr_consolidated_format": 1, "metadata": {}}')
        self.configuration = {
            "sample": {"sample_id": "sample"},
            "scan_set": {"scan_set_id": "synthetic"},
            "processing": {"beam_centers": {"pair_0_90": {"center_x": 1., "center_y": 2.}}},
        }
        self.manifest = {
            "scan_set_id": "synthetic", "configuration": self.configuration,
            "generated_at_utc": "2026-11-18T12:00:00+00:00", "git_commit": "abc123",
            "zarr_store": "workup.zarr", "zarr_groups": {"pairs/pair_0_90/scan_products": ["clean"]},
        }
        self.summary = {
            "output_directory": self.export,
            "manifest": self.export / "manifest.json",
            "zarr_store": store,
            "groups": list(self.manifest["zarr_groups"]),
        }
        self.summary["manifest"].write_text(json.dumps(self.manifest))
        save_text = "".join(f"{key}: {self.summary[key]}\n" for key in ("output_directory", "manifest", "zarr_store"))
        self.notebook = {
            "nbformat": 4, "nbformat_minor": 5, "metadata": {},
            "cells": [
                {"cell_type": "code", "id": "plot-cell", "execution_count": 1,
                 "metadata": {}, "source": ["plot_review()"], "outputs": [
                     {"output_type": "display_data", "metadata": {},
                      "data": {"image/png": base64.b64encode(PNG).decode(), "text/plain": "a plot"}},
                     {"output_type": "display_data", "metadata": {},
                      "data": {"image/png": [base64.b64encode(PNG).decode()[:20] + "\n", base64.b64encode(PNG).decode()[20:]]}},
                     {"output_type": "display_data", "metadata": {},
                      "data": {"text/html": "<div>interactive output</div>"}},
                 ]},
                {"cell_type": "code", "id": "save-cell", "execution_count": 2,
                 "metadata": {}, "source": ["save_summary = outputs.save_workup_results(...)"],
                 "outputs": [{"output_type": "stream", "name": "stdout", "text": save_text}]},
            ],
        }
        self.notebook_path = self.root / "workup.ipynb"
        self.write_notebook()

    def write_notebook(self):
        self.notebook_path.write_text(json.dumps(self.notebook))

    def archive(self, **overrides):
        return figures.save_workup_figures(
            save_summary=overrides.get("save_summary", self.summary),
            configuration=overrides.get("configuration", self.configuration),
            notebook_path=self.notebook_path,
        )

    def assert_no_archive(self):
        self.assertFalse((self.export / "figures").exists())
        self.assertEqual(list(self.export.glob(".figures.incomplete-*")), [])

    def test_exact_images_snapshot_and_export_hash_without_export_mutation(self):
        original_export = {path.relative_to(self.export): path.read_bytes()
                           for path in self.export.rglob("*") if path.is_file()}
        original_notebook = self.notebook_path.read_bytes()
        summary = self.archive()
        self.assertEqual(summary["figure_count"], 2)
        self.assertEqual(summary["non_png_rich_output_count"], 1)
        folder = self.export / "figures"
        self.assertEqual((folder / "workup.ipynb").read_bytes(), original_notebook)
        index = json.loads((folder / "figure_manifest.json").read_text())
        self.assertEqual(index["source_notebook"]["sha256"], hashlib.sha256(original_notebook).hexdigest())
        self.assertEqual(index["workup_export"]["manifest_sha256"], hashlib.sha256(self.summary["manifest"].read_bytes()).hexdigest())
        self.assertEqual(index["workup_export"]["manifest"], "../manifest.json")
        self.assertEqual(index["workup_export"]["zarr_store"], "../workup.zarr")
        self.assertEqual(index["scan_set_id"], "synthetic")
        for image in index["figures"]:
            self.assertEqual((folder / image["filename"]).read_bytes(), PNG)
            self.assertEqual(image["sha256"], hashlib.sha256(PNG).hexdigest())
            self.assertEqual(image["cell_id"], "plot-cell")
            self.assertEqual((image["width_pixels"], image["height_pixels"]), (1, 1))
        for relative, content in original_export.items():
            self.assertEqual((self.export / relative).read_bytes(), content)
        self.assertEqual(list(self.export.glob(".figures.incomplete-*")), [])

    def test_existing_archive_is_never_overwritten(self):
        self.archive()
        before = {path.name: path.read_bytes() for path in (self.export / "figures").iterdir()}
        with self.assertRaisesRegex(FileExistsError, "kept unchanged"):
            self.archive()
        self.assertEqual(before, {path.name: path.read_bytes() for path in (self.export / "figures").iterdir()})

    def test_configuration_change_and_wrong_scan_set_rejected(self):
        for key, value in (("sample", {"sample_id": "different"}), ("scan_set", {"scan_set_id": "other"}), ("processing", {})):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.configuration)
                changed[key] = value
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    self.archive(configuration=changed)
                self.assert_no_archive()

    def test_changed_groups_and_mixed_paths_rejected(self):
        for key, value in (("groups", []), ("manifest", self.root / "manifest.json"), ("zarr_store", self.root / "workup.zarr")):
            with self.subTest(key=key):
                changed = dict(self.summary)
                changed[key] = value
                with self.assertRaises(ValueError):
                    self.archive(save_summary=changed)
                self.assert_no_archive()

    def test_unfinished_zarr_rejected(self):
        (self.summary["zarr_store"] / ".zmetadata").unlink()
        with self.assertRaisesRegex(ValueError, "finalized"):
            self.archive()
        self.assert_no_archive()

    def test_old_export_in_notebook_rejected(self):
        self.notebook["cells"][-1]["outputs"][0]["text"] = self.notebook["cells"][-1]["outputs"][0]["text"].replace("/timestamp/", "/old_timestamp/")
        self.write_notebook()
        with self.assertRaisesRegex(ValueError, "different export"):
            self.archive()
        self.assert_no_archive()

    def test_missing_save_output_requests_notebook_save(self):
        self.notebook["cells"][-1]["outputs"] = []
        self.write_notebook()
        with self.assertRaisesRegex(ValueError, "Ctrl\\+S"):
            self.archive()
        self.assert_no_archive()

    def test_failed_last_save_cell_rejected(self):
        self.notebook["cells"][-1]["outputs"].append({"output_type": "error", "ename": "ValueError", "evalue": "failed", "traceback": []})
        self.write_notebook()
        with self.assertRaisesRegex(ValueError, "contains an error"):
            self.archive()
        self.assert_no_archive()

    def test_missing_save_cell_rejected(self):
        self.notebook["cells"].pop()
        self.write_notebook()
        with self.assertRaisesRegex(ValueError, "no workup-save cell"):
            self.archive()
        self.assert_no_archive()

    def test_no_png_outputs_rejected(self):
        self.notebook["cells"][0]["outputs"] = []
        self.write_notebook()
        with self.assertRaisesRegex(ValueError, "No PNG outputs"):
            self.archive()
        self.assert_no_archive()

    def test_invalid_png_is_rejected_before_writing(self):
        for invalid in ("not-base64!", base64.b64encode(b"not png").decode()):
            with self.subTest(invalid=invalid):
                self.notebook["cells"][0]["outputs"][0]["data"]["image/png"] = invalid
                self.write_notebook()
                with self.assertRaisesRegex(ValueError, "Invalid PNG"):
                    self.archive()
                self.assert_no_archive()

    def test_failed_publish_leaves_no_partial_archive(self):
        with patch.object(Path, "rename", side_effect=OSError("simulated publish failure")):
            with self.assertRaisesRegex(OSError, "simulated"):
                self.archive()
        self.assert_no_archive()


if __name__ == "__main__":
    unittest.main()
