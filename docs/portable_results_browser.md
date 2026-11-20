# Portable, read-only saved-results browser

`notebooks/browse_results.ipynb` is self-contained. It includes its metadata
reader functions and does not import `rsoxs_unified_workup`, processing
configuration, PyHyperScattering, or a beamline client. You can copy just
the notebook to another computer and point it at copied saved exports.

Use any Python notebook environment with the ordinary reader dependencies
listed in `requirements-browse.txt`. To install them into your chosen
environment, run:

```bash
python -m pip install -r requirements-browse.txt
```

Do not install the full workup project merely to browse results. Exact
writer-environment versions are not required. The Zarr reader must support
the saved version-2 format and standard numcodecs codecs. These exports use
ordinary Blosc/LZ4 compression and UTF-8 strings, not pickled Python objects.
Catalog-only browsing needs pandas and Python's standard library; xarray,
Zarr, and numcodecs are needed for the optional array-opening cell.

## Notebook sequence

1. Select your chosen Python environment as this notebook's kernel. Use a
   separate kernel process from any running workup notebook; an environment
   name describes installed packages, not shared variables or memory.
2. Run the reader-setup cell once; its functions are included in the notebook.
3. Set `RESULTS_ROOT` in the refresh cell to the directory containing saved
   exports. `None` searches for a `results` folder relative to the current
   working directory and its parents. On another machine, set an explicit
   path to your copied results folder. It may contain multiple run folders.
4. Refresh, filter, and inspect an export. The default `REVIEW_STATUS =
   "approved"` shows explicitly reviewed exports; use `"draft"`,
   `"development"`, or None to inspect other statuses. Metadata comes from
   its saved manifest and review JSON, never current configuration CSVs/TOMLs.
5. Optional: set `OPEN_DATASET = True` and `GROUP_NAME` to an exact exported
   group listed by the inspect cell, then run the array-opening cell.

`xr.open_zarr(..., consolidated=True, chunks=None)` opens an existing group
for reading. `chunks=None` avoids requiring dask. Numeric arrays are loaded
on access rather than all at once; displaying the dataset shows its structure
and may load small coordinate arrays. No re-integration, fitting, alignment,
normalization, or export is performed. Each pair keeps its native q grid.

## Parallel work and safety

All provided browser cells read files; they do not modify saved exports,
configuration files, or a workup kernel's variables. Separate kernels may
share CPU, memory, and disk bandwidth, so large later analysis can slow a
running workup without changing its results. Do not explicitly attach both
notebooks to one running kernel. The operating system does not enforce
read-only permissions here; adding your own write calls would change this
behavior. Saving the browser notebook itself does not change any export.

Refresh only discovers exports with finalized manifests and Zarr metadata.
It skips incomplete stores and does not repeat full array verification.
Most-recent save time is not a scientific fit-quality judgment. Optional
array opening allows later analysis of already saved results, not another
workup pipeline.

## Beam-center mask reflection

The catalog includes `beam_center_reflection_mode`, shown in the notebook as
`beam-center mask reflection`. It reads the saved manifest's
`configuration.processing.beam_center_fit.reflection_mode`; it never reads
today's processing TOML or assumes a current code default.

Set `REFLECTION_MODE = None` for every mode, or use a string such as `"none"`,
`"twofold"`, `"fourfold"`, or `"eightfold"`. A list selects several modes.
`"none"` means no reflection; `"not_recorded"` means missing/unreadable metadata,
not no reflection. An explicit JSON null is interpreted as no reflection,
matching the centering function. Future named settings remain visible.

Keep `LATEST_ONLY = False` to compare older exports with different reflection
settings for one scan set. With `True`, review status is selected first, then
the newest save per scan set within that status, then reflection/other filters.
A newer draft does not hide an approved result. An older matching reflection
export is not substituted if the newest approved export fails that filter.

Review records are append-only and tied to an exact manifest hash. Old exports
without status metadata are development; they are never automatically approved.
Malformed or mismatched latest review records show `unknown` and cannot enter
the approved view. The browser does not write review decisions. To inspect the
segregated archive, point `RESULTS_ROOT` at its archived `results/` directory and
change `REVIEW_STATUS` to `"development"` or None. See [results_review.md](results_review.md)
for approval, withdrawal, and archive instructions.

This reports the saved centering setting, not proof that center fitting ran.
Configured-center approval without fitting may still record the setting.
It does not imply that final intensity/anisotropy arrays were reflected.
The export-inspection cell also prints the selected saved reflection mode.

The browser's kernel metadata now says generic Python, not the environment
that produced the files. You still choose an installed environment with the
reader dependencies in your notebook application's kernel selector. Creating
a separate environment is optional; using the existing environment in a
separate kernel also works, but is no longer a reader requirement.
