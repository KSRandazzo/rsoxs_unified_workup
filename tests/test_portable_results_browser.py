"""Portable reader tests: deliberately block all workup/beamline imports."""

import ast
import builtins
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "browse_results.ipynb"
CATALOG_SOURCE = ROOT / "src" / "rsoxs_unified_workup" / "catalog.py"
FORBIDDEN_IMPORTS = {
    "rsoxs_unified_workup", "PyHyperScattering", "pyFAI", "tiled",
    "joblib", "scipy", "matplotlib", "skimage",
}


@contextmanager
def unrelated_working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def blocked_imports(extra=()):
    original_import = builtins.__import__
    forbidden = FORBIDDEN_IMPORTS | set(extra)

    def checked_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            raise AssertionError(f"Reader attempted a prohibited import: {name}")
        return original_import(name, *args, **kwargs)

    # Simulate a lightweight environment even when these optional packages
    # happen to be installed in the environment used to run the tests.
    # xarray probes optional array backends; CuPy otherwise imports SciPy.
    absent_optional = {name: None for name in ("cupy", "cupyx", "sparse", "pint", "dask")}
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in absent_optional}
    sys.modules.update(absent_optional)
    try:
        with patch("builtins.__import__", checked_import):
            yield
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def fingerprints(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


class PortableReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.results = self.base / "copied-results"
        self.results.mkdir()
        self.unrelated = self.base / "independent-notebook"
        self.unrelated.mkdir()
        self.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.cells = {
            cell["id"]: cell for cell in self.notebook["cells"]
        }
        self.displayed = []
        self.namespace = {
            "__name__": "__main__", "display": self.displayed.append,
        }

    def run_cell(self, cell_id, **settings):
        """Execute the actual notebook source, overriding only user settings."""
        # These existing reader tests intentionally inspect all historical
        # exports; the approved-only notebook default is tested separately.
        if cell_id == "catalog-filter-table":
            settings.setdefault("REVIEW_STATUS", None)
        tree = ast.parse("".join(self.cells[cell_id]["source"]))
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in settings:
                statement.value = ast.parse(repr(settings[target.id]), mode="eval").body
        ast.fix_missing_locations(tree)
        with redirect_stdout(io.StringIO()):
            exec(compile(tree, f"<{cell_id}>", "exec"), self.namespace)

    def save_export(self, scan_set_id="sample_longC", timestamp="2026-11-01T12:00:00Z",
                    *, legacy=False, real=False, reflection="not_recorded"):
        directory = self.results / "runs" / timestamp.replace(":", "") / scan_set_id
        directory.mkdir(parents=True)
        store = directory / "workup.zarr"
        group_name = "pairs/pair_0_90/pointwise_anisotropy"
        if real:
            import numpy as np
            import zarr

            group = zarr.open_group(str(store), mode="w").require_group(group_name)
            for name, values, dimensions in (
                ("energy", np.array([284.8, 286.0]), ["energy"]),
                ("q", np.array([0.01, 0.02, 0.03]), ["q"]),
                ("anisotropy", np.arange(6, dtype=float).reshape(2, 3), ["energy", "q"]),
            ):
                array = group.create_dataset(
                    name, data=values, compressor=None, fill_value=None,
                )
                array.attrs["_ARRAY_DIMENSIONS"] = dimensions
            group.attrs["description"] = "Independent ordinary Zarr fixture"
            zarr.consolidate_metadata(str(store))
        else:
            store.mkdir()
            (store / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
            (store / ".zmetadata").write_text("{}", encoding="utf-8")
        scan_set = {
            "scan_set_id": scan_set_id, "sample_id": "sample",
            "scan_type": "longO" if legacy else "long",
        }
        if not legacy:
            scan_set["energy_region"] = "C"
        manifest = {
            "scan_set_id": scan_set_id, "generated_at_utc": timestamp,
            "zarr_store": "workup.zarr", "schema_version": 2,
            "git_commit": "example", "git_working_tree_clean": True,
            "zarr_groups": {group_name: ["anisotropy"]},
            "configuration": {
                "sample": {
                    "sample_id": "sample", "sample_name": "Readable sample",
                    "polymer": "PS", "diameter_nm": 50,
                },
                "scan_set": scan_set,
                "scans": [
                    {"scan_id": 123, "polarization_deg": 0},
                    {"scan_id": 123, "polarization_deg": 90},
                ],
                "polarization_pairs": [{"pair_id": "pair_0_90"}],
            },
        }
        if reflection != "not_recorded":
            manifest["configuration"]["processing"] = {
                "beam_center_fit": {"reflection_mode": reflection},
            }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return directory, group_name

    def browse(self):
        self.run_cell("portable-reader-setup")
        self.run_cell("catalog-refresh", RESULTS_ROOT=str(self.results))
        # User filter defaults can change without changing test expectations.
        self.run_cell("catalog-filter-table", LATEST_ONLY=True)
        self.run_cell("catalog-selection")

    def test_self_contained_notebook_and_python_kernel(self):
        # Editors update the kernel label/version when a user selects their
        # own environment. Portability depends on imports, not that label.
        self.assertEqual(self.notebook["metadata"]["kernelspec"]["language"], "python")
        self.assertEqual(len(self.cells), len(self.notebook["cells"]))
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            # Saved execution/output state is allowed. Running and saving the
            # notebook must not fail a test of its independent reader code.
            tree = ast.parse("".join(cell["source"]))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse({alias.name.split(".")[0] for alias in node.names} & FORBIDDEN_IMPORTS)
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], FORBIDDEN_IMPORTS)
        setup = "".join(self.cells["portable-reader-setup"]["source"])
        self.assertNotIn("__file__", setup)
        self.assertNotIn("sys.path", setup)
        self.assertNotIn("load_scan_set", setup)

    def test_metadata_only_from_unrelated_folder_without_workup_or_xarray(self):
        self.save_export(timestamp="2026-11-01T12:00:00Z")
        self.save_export(timestamp="2026-11-02T12:00:00Z")
        self.save_export("sample_longO", legacy=True)
        before = fingerprints(self.results)
        with unrelated_working_directory(self.unrelated), blocked_imports({"xarray", "zarr", "numcodecs"}):
            self.browse()
            self.run_cell("catalog-open-dataset")
        catalog = self.namespace["catalog"]
        self.assertEqual(len(catalog), 3)
        self.assertEqual(len(self.namespace["view"]), 2)
        self.assertEqual(catalog["is_latest"].sum(), 2)
        latest_c = catalog.loc[(catalog.scan_set_id == "sample_longC") & catalog.is_latest].iloc[0]
        self.assertEqual(str(latest_c.saved_at_utc), "2026-11-02 12:00:00+00:00")
        oxygen = catalog.loc[catalog.scan_set_id == "sample_longO"].iloc[0]
        self.assertEqual((oxygen.scan_type, oxygen.energy_region), ("long", "O"))
        self.assertNotIn("selected_dataset", self.namespace)
        self.assertEqual(before, fingerprints(self.results))

    def test_all_versions_filters_and_selection_staleness(self):
        self.save_export(timestamp="2026-11-01T12:00:00Z")
        self.save_export(timestamp="2026-11-02T12:00:00Z")
        self.save_export("sample_longO", legacy=True)
        with unrelated_working_directory(self.unrelated), blocked_imports():
            self.browse()
            self.run_cell("catalog-filter-table", LATEST_ONLY=False)
            self.assertEqual(len(self.namespace["view"]), 3)
            self.assertNotIn("selected_export", self.namespace)
            self.run_cell("catalog-filter-table", ENERGY_REGION="o", SAMPLE_NAME="readable sample")
            self.assertEqual(len(self.namespace["view"]), 1)
            self.run_cell("catalog-selection")
            self.assertEqual(self.namespace["selected_export"]["scan_set_id"], "sample_longO")
            self.run_cell("catalog-refresh", RESULTS_ROOT=str(self.results))
            self.assertNotIn("view", self.namespace)
            self.assertNotIn("selected_export", self.namespace)

    def test_automatic_results_discovery_without_repository(self):
        independent_base = self.base / "no-repository"
        (independent_base / "notebooks").mkdir(parents=True)
        self.results.rename(independent_base / "results")
        self.results = independent_base / "results"
        self.save_export()
        with unrelated_working_directory(independent_base / "notebooks"), blocked_imports():
            self.run_cell("portable-reader-setup")
            self.run_cell("catalog-refresh")
            self.assertEqual(self.namespace["catalog"].attrs["results_root"], str(self.results))
        with unrelated_working_directory(self.results), blocked_imports():
            self.run_cell("portable-reader-setup")
            self.run_cell("catalog-refresh")
            self.assertEqual(len(self.namespace["catalog"]), 1)

    def test_missing_root_message_is_actionable(self):
        with unrelated_working_directory(self.unrelated), blocked_imports():
            self.run_cell("portable-reader-setup")
            with self.assertRaisesRegex(FileNotFoundError, "Set RESULTS_ROOT"):
                self.run_cell("catalog-refresh")

    def test_malformed_incomplete_and_store_path_escape_are_not_opened(self):
        self.save_export()
        broken = self.results / "broken"
        broken.mkdir()
        (broken / "manifest.json").write_text("not JSON", encoding="utf-8")
        incomplete, _ = self.save_export("incomplete", "2026-11-03T12:00:00Z")
        (incomplete / "workup.zarr" / ".zmetadata").unlink()
        escaped, _ = self.save_export("escaped", "2026-11-04T12:00:00Z")
        path = escaped / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["zarr_store"] = "../external.zarr"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        hidden = self.results / "ignored.incomplete"
        hidden.mkdir()
        (hidden / "manifest.json").write_text("not JSON", encoding="utf-8")
        before = fingerprints(self.results)
        with unrelated_working_directory(self.unrelated), blocked_imports(), warnings.catch_warnings(record=True):
            self.browse()
        self.assertEqual(len(self.namespace["catalog"]), 1)
        self.assertEqual(len(self.namespace["catalog"].attrs["skipped_exports"]), 3)
        self.assertEqual(before, fingerprints(self.results))

    def test_open_requires_inspection_and_valid_group(self):
        with blocked_imports({"xarray"}):
            with self.assertRaisesRegex(ValueError, "inspection"):
                self.run_cell("catalog-open-dataset", OPEN_DATASET=True)
            self.save_export()
            self.browse()
            with self.assertRaisesRegex(ValueError, "GROUP_NAME"):
                self.run_cell("catalog-open-dataset", OPEN_DATASET=True, GROUP_NAME="absent")

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("xarray", "zarr", "numcodecs")),
        "Optional array-opening packages are not installed",
    )
    def test_real_zarr_open_values_metadata_and_files_are_unchanged(self):
        import numpy as np

        _, group_name = self.save_export(real=True)
        before = fingerprints(self.results)
        with unrelated_working_directory(self.unrelated), blocked_imports():
            self.browse()
            self.run_cell("catalog-open-dataset", OPEN_DATASET=True, GROUP_NAME=group_name)
            dataset = self.namespace["selected_dataset"]
            np.testing.assert_array_equal(dataset.anisotropy.values, np.arange(6).reshape(2, 3))
            self.assertEqual(dict(dataset.sizes), {"energy": 2, "q": 3})
            self.assertEqual(dataset.attrs["description"], "Independent ordinary Zarr fixture")
            self.assertIsNone(dataset.anisotropy.chunks)
            self.run_cell("catalog-open-dataset", OPEN_DATASET=False)
            self.assertNotIn("selected_dataset", self.namespace)
        self.assertEqual(before, fingerprints(self.results))

    def test_standalone_catalog_module_has_no_processing_dependency(self):
        self.save_export()
        module = types.ModuleType("standalone_metadata_catalog")
        module.__file__ = str(CATALOG_SOURCE)
        with blocked_imports({"xarray", "zarr", "numcodecs"}):
            exec(compile(CATALOG_SOURCE.read_text(), str(CATALOG_SOURCE), "exec"), module.__dict__)
            table = module.build_results_catalog(self.results)
            self.assertEqual(len(module.select_results(table, energy_region="C")), 1)


    def test_saved_modes_none_missing_null_and_future_names(self):
        modes = {
            "fourfold": (" FoUrFoLd ", "fourfold"),
            "twofold": ("twofold", "twofold"),
            "eightfold": ("eightfold", "eightfold"),
            "none": ("none", "none"),
            "null": (None, "none"),
            "missing": ("not_recorded", "not_recorded"),
            "empty": ("", "not_recorded"),
            "invalid": (False, "not_recorded"),
            "future": ("future_mode", "future_mode"),
        }
        for scan_set_id, (saved, _) in modes.items():
            self.save_export(scan_set_id, reflection=saved)
        before = fingerprints(self.results)
        with unrelated_working_directory(self.unrelated), blocked_imports({"xarray", "zarr", "numcodecs"}):
            self.browse()
            catalog = self.namespace["catalog"]
            by_id = catalog.set_index("scan_set_id")
            for scan_set_id, (_, expected) in modes.items():
                self.assertEqual(by_id.loc[scan_set_id, "beam_center_reflection_mode"], expected)
            self.run_cell("catalog-filter-table", REFLECTION_MODE="NONE")
            self.assertEqual(set(self.namespace["view"].scan_set_id), {"none", "null"})
            self.run_cell("catalog-filter-table", REFLECTION_MODE="not_recorded")
            self.assertEqual(set(self.namespace["view"].scan_set_id), {"missing", "empty", "invalid"})
            self.run_cell("catalog-filter-table", REFLECTION_MODE=["none", "fourfold"])
            self.assertEqual(set(self.namespace["view"].scan_set_id), {"none", "null", "fourfold"})
            self.run_cell("catalog-filter-table", REFLECTION_MODE="fourfold")
            self.run_cell("catalog-selection")
            self.assertEqual(self.namespace["selected_export"]["beam_center_reflection_mode"], "fourfold")
        self.assertEqual(before, fingerprints(self.results))

    def test_reflection_filter_does_not_substitute_older_latest(self):
        self.save_export(timestamp="2026-11-01T12:00:00Z", reflection="fourfold")
        self.save_export(timestamp="2026-11-02T12:00:00Z", reflection="none")
        with unrelated_working_directory(self.unrelated), blocked_imports():
            self.browse()
            self.run_cell("catalog-filter-table", LATEST_ONLY=True, REFLECTION_MODE="fourfold")
            self.assertTrue(self.namespace["view"].empty)
            self.run_cell("catalog-filter-table", LATEST_ONLY=False, REFLECTION_MODE="fourfold")
            self.assertEqual(len(self.namespace["view"]), 1)
            self.assertFalse(self.namespace["view"].iloc[0].is_latest)
            self.run_cell("catalog-filter-table", LATEST_ONLY=False, REFLECTION_MODE=None)
            self.assertEqual(len(self.namespace["view"]), 2)

    def test_embedded_reader_functions_match_catalog_module(self):
        embedded = ast.parse("".join(self.cells["portable-reader-setup"]["source"]))
        canonical = ast.parse(CATALOG_SOURCE.read_text())
        embedded_functions = {
            node.name: ast.dump(node, include_attributes=False)
            for node in embedded.body if isinstance(node, ast.FunctionDef)
        }
        for node in canonical.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            # Only local default-root discovery/error wording differs.
            if node.name == "build_results_catalog":
                continue
            self.assertEqual(embedded_functions[node.name], ast.dump(node, include_attributes=False))
        canonical_columns = next(
            node.value for node in canonical.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "CATALOG_COLUMNS" for target in node.targets)
        )
        embedded_columns = next(
            node.value for node in embedded.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "CATALOG_COLUMNS" for target in node.targets)
        )
        self.assertEqual(ast.literal_eval(embedded_columns), ast.literal_eval(canonical_columns))

    def test_default_approved_view_and_newer_draft(self):
        from rsoxs_unified_workup.review import set_export_review_status

        old, _ = self.save_export(timestamp="2026-11-01T12:00:00Z")
        new, _ = self.save_export(timestamp="2026-11-02T12:00:00Z")
        path = new / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["review_status"] = "draft"
        path.write_text(json.dumps(manifest))
        set_export_review_status(old, "approved", reviewer="tester", note="Reviewed plots", confirm=True)
        before = fingerprints(self.results)
        with unrelated_working_directory(self.unrelated), blocked_imports():
            self.browse()
            # Execute the real default filter, without overriding its status.
            tree = ast.parse("".join(self.cells["catalog-filter-table"]["source"]))
            with redirect_stdout(io.StringIO()):
                exec(compile(tree, "<default approved filter>", "exec"), self.namespace)
            self.assertEqual(self.namespace["REVIEW_STATUS"], "approved")
            self.assertEqual(len(self.namespace["view"]), 1)
            self.run_cell("catalog-filter-table", REVIEW_STATUS="approved", LATEST_ONLY=True)
            row = self.namespace["view"].iloc[0]
            self.assertEqual(row.export_directory, str(old))
            self.assertFalse(row.is_latest)  # newest overall is still a draft
            self.assertEqual(row.reviewer, "tester")
            self.run_cell("catalog-selection")
        self.assertEqual(before, fingerprints(self.results))

    def test_review_statuses_and_invalid_latest_decision(self):
        from rsoxs_unified_workup.review import set_export_review_status

        old, _ = self.save_export("development")
        approved, _ = self.save_export("approved")
        draft, _ = self.save_export("draft")
        set_export_review_status(approved, "approved", reviewer="tester", note="Checked", confirm=True)
        set_export_review_status(draft, "approved", reviewer="tester", note="Checked", confirm=True)
        result = set_export_review_status(draft, "draft", reviewer="tester", note="Needs another mask", confirm=True)
        with blocked_imports():
            self.browse()
            by_id = self.namespace["catalog"].set_index("scan_set_id")
            self.assertEqual(by_id.loc["development", "review_status"], "development")
            self.assertEqual(by_id.loc["approved", "review_status"], "approved")
            self.assertEqual(by_id.loc["draft", "review_status"], "draft")
        result["review_record"].write_text("broken JSON")
        with blocked_imports():
            self.browse()
            self.run_cell("catalog-filter-table", REVIEW_STATUS="approved")
            self.assertEqual(set(self.namespace["view"].scan_set_id), {"approved"})
            by_id = self.namespace["catalog"].set_index("scan_set_id")
            self.assertEqual(by_id.loc["draft", "review_status"], "unknown")

    def test_changed_manifest_is_not_approved_and_review_filter_lists(self):
        from rsoxs_unified_workup.review import set_export_review_status

        directory, _ = self.save_export("changed")
        self.save_export("development")
        set_export_review_status(directory, "approved", reviewer="tester", note="Checked", confirm=True)
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["git_commit"] = "changed"
        path.write_text(json.dumps(manifest))
        with blocked_imports():
            self.browse()
            self.run_cell("catalog-filter-table", REVIEW_STATUS="approved")
            self.assertTrue(self.namespace["view"].empty)
            self.run_cell("catalog-filter-table", REVIEW_STATUS=["UNKNOWN", "Development"])
            self.assertEqual(len(self.namespace["view"]), 2)


if __name__ == "__main__":
    unittest.main()
