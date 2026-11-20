"""Synthetic review and development-archive tests; no measured arrays read."""

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsoxs_unified_workup import catalog, development, review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        for name in ("config", "notebooks"):
            (self.project / name).mkdir()
        self.export = self.project / "results/runs/time/example"
        self.export.mkdir(parents=True)
        (self.export / "workup.zarr").mkdir()
        (self.export / "workup.zarr/.zgroup").write_text('{"zarr_format": 2}')
        (self.export / "workup.zarr/.zmetadata").write_text("{}")
        self.manifest = {
            "scan_set_id": "example", "generated_at_utc": "2026-11-19T12:00:00+00:00",
            "zarr_store": "workup.zarr", "zarr_groups": {"pairs/one/products": ["clean"]},
            "configuration": {"sample": {}, "scan_set": {"scan_set_id": "example"}},
        }
        self.path = self.export / "manifest.json"
        self.path.write_text(json.dumps(self.manifest))

    def decision(self, status="approved", **options):
        return review.set_export_review_status(
            self.export, status, reviewer=options.pop("reviewer", "tester"),
            note=options.pop("note", "Center, masks, normalization and plots checked"),
            confirm=options.pop("confirm", True), **options,
        )

    def test_explicit_approval_append_only_and_revocation(self):
        original = self.path.read_bytes()
        first = self.decision()
        first_content = first["review_record"].read_bytes()
        second = self.decision("draft", note="Mask requires another pass")
        self.assertNotEqual(first["review_record"], second["review_record"])
        self.assertEqual(first["review_record"].read_bytes(), first_content)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(catalog._catalog_row(self.path)["review_status"], "draft")
        third = self.decision()
        self.assertEqual(catalog._catalog_row(self.path)["review_status"], "approved")
        self.assertEqual(len(list((self.export / "reviews").glob("*.json"))), 3)
        self.assertFalse((self.export / "reviews/.review.lock").exists())

    def test_missing_confirmation_reviewer_note_and_bad_status(self):
        for kwargs in ({"confirm": False}, {"reviewer": ""}, {"note": ""}):
            with self.assertRaises(ValueError):
                self.decision(**kwargs)
        with self.assertRaises(ValueError):
            self.decision("good")
        self.assertFalse((self.export / "reviews").exists())

    def test_changed_snapshot_and_missing_store_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "changed since verification"):
            self.decision(expected_manifest_sha256="incorrect")
        (self.export / "workup.zarr/.zmetadata").unlink()
        with self.assertRaisesRegex(ValueError, "finalized"):
            self.decision()
        self.assertFalse((self.export / "reviews").exists())

    def test_review_for_wrong_export_or_invalid_newest_is_never_used(self):
        event = self.decision()["review_record"]
        payload = json.loads(event.read_text())
        payload["scan_set_id"] = "another"
        event.write_text(json.dumps(payload))
        self.assertEqual(catalog._catalog_row(self.path)["review_status"], "unknown")

    def test_manifest_baselines_never_auto_approve(self):
        self.assertEqual(catalog._catalog_row(self.path)["review_status"], "development")
        for value, expected in (("draft", "draft"), ("approved", "unknown"), (None, "unknown"), ([], "unknown")):
            self.manifest["review_status"] = value
            self.path.write_text(json.dumps(self.manifest))
            self.assertEqual(catalog._catalog_row(self.path)["review_status"], expected)

    def test_atomic_publication_failure_and_concurrent_lock(self):
        with patch.object(review.os, "link", side_effect=OSError("fixture failure")):
            with self.assertRaises(OSError):
                self.decision()
        self.assertEqual(list((self.export / "reviews").iterdir()), [])
        lock = self.export / "reviews/.review.lock"
        lock.write_text("another writer")
        with self.assertRaisesRegex(RuntimeError, "Another review"):
            self.decision()
        self.assertEqual(lock.read_text(), "another writer")

    def test_review_history_survives_relocation_with_figure_hashes(self):
        self.decision()
        figures = self.export / "figures"
        figures.mkdir()
        original_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        (figures / "figure_manifest.json").write_text(json.dumps({"manifest_sha256": original_hash}))
        destination = self.project / "copied-export"
        self.export.rename(destination)
        row = catalog._catalog_row(destination / "manifest.json")
        self.assertEqual(row["review_status"], "approved")
        self.assertEqual(hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest(), original_hash)

    def test_new_manifest_and_workup_approval_guards(self):
        from rsoxs_unified_workup.outputs import build_manifest
        import xarray as xr

        result = build_manifest(self.manifest["configuration"], {"pairs/one/products": xr.Dataset()})
        self.assertEqual(result["review_status"], "draft")
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/workup.ipynb").read_text())
        cell = next(c for c in notebook["cells"] if c["id"] == "export-review-decision")
        tree = ast.parse("".join(cell["source"]))
        namespace = {}
        exec(compile(tree, "<approval disabled>", "exec"), namespace)
        self.assertNotIn("review_summary", namespace)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "APPROVE_EXPORT" for t in node.targets):
                node.value = ast.Constant(True)
        ast.fix_missing_locations(tree)
        with self.assertRaisesRegex(RuntimeError, "verification"):
            exec(compile(tree, "<approval missing verification>", "exec"), namespace)

    def test_notebook_requires_fresh_confirmation_and_accepts_case_variants(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/workup.ipynb").read_text())
        cell = next(c for c in notebook["cells"] if c["id"] == "export-review-decision")
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "APPROVE_EXPORT":
                        node.value = ast.Constant(True)
                    if isinstance(target, ast.Name) and target.id == "REVIEW_NOTE":
                        node.value = ast.Constant("Checked this saved export")
        ast.fix_missing_locations(tree)
        namespace = {
            "save_summary": {"output_directory": self.export},
            "_verified_export_snapshot": {
                "export_directory": str(self.export),
                "manifest_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            },
        }
        with patch("builtins.input", return_value="cancel") as prompt:
            exec(compile(tree, "<approval cancelled>", "exec"), namespace)
            prompt.assert_called_once()
        self.assertFalse((self.export / "reviews").exists())
        with patch("builtins.input", return_value=" Approve ") as prompt:
            exec(compile(tree, "<approval confirmed>", "exec"), namespace)
            prompt.assert_called_once()
        self.assertEqual(catalog._catalog_row(self.path)["review_status"], "approved")


class DevelopmentArchiveTests(unittest.TestCase):
    setUp = ReviewTests.setUp
    decision = ReviewTests.decision
    def test_inventory_and_recoverable_move_preserve_all_bytes(self):
        partial = self.project / "results/runs/failed/.workup.zarr.incomplete"
        partial.mkdir(parents=True)
        (partial / "chunk").write_bytes(b"old partial data")
        before = {str(path.relative_to(self.project / "results")): path.read_bytes()
                  for path in (self.project / "results").rglob("*") if path.is_file()}
        inventory = development.development_inventory(self.project)
        self.assertEqual(inventory["manifest_count"], 1)
        self.assertEqual(len(inventory["incomplete_locations"]), 1)
        with self.assertRaisesRegex(ValueError, "Explicit"):
            development.archive_development_results(self.project, expected_inventory=inventory["inventory_sha256"])
        result = development.archive_development_results(
            self.project, expected_inventory=inventory["inventory_sha256"], confirm=True,
        )
        self.assertEqual(list((self.project / "results").iterdir()), [])
        archived = result["archived_results_root"]
        after = {str(path.relative_to(archived)): path.read_bytes() for path in archived.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(len(catalog.build_results_catalog(archived)), 1)
        self.assertTrue(catalog.build_results_catalog(self.project / "results").empty)

    def test_changed_inventory_or_new_draft_refuses_move(self):
        inventory = development.development_inventory(self.project)
        (self.export / "new-note.txt").write_text("changed batch")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            development.archive_development_results(self.project, expected_inventory=inventory["inventory_sha256"], confirm=True)
        self.assertTrue(self.path.exists())
        self.manifest["review_status"] = "draft"
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "new or reviewed"):
            development.development_inventory(self.project)

    def test_archive_refuses_reviewed_export(self):
        self.decision()
        with self.assertRaisesRegex(ValueError, "new or reviewed"):
            development.development_inventory(self.project)

    def test_failed_active_folder_creation_restores_original(self):
        inventory = development.development_inventory(self.project)
        original_mkdir = Path.mkdir

        def failing_mkdir(path, *args, **kwargs):
            if path == self.project / "results":
                raise OSError("fixture mkdir failure")
            return original_mkdir(path, *args, **kwargs)

        with patch.object(Path, "mkdir", failing_mkdir):
            with self.assertRaises(OSError):
                development.archive_development_results(self.project, expected_inventory=inventory["inventory_sha256"], confirm=True)
        self.assertTrue(self.path.exists())
        self.assertFalse((self.project / "results_development/.archive.lock").exists())


if __name__ == "__main__":
    unittest.main()
