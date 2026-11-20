# Figures associated with a saved workup

Run the figure-archive cell after the Zarr save and export-verification cells.
First save `notebooks/workup.ipynb` with its outputs retained (Ctrl+S). Do not
clear outputs until after archiving. No integration, fitting, normalization,
or plotting cells need to be rerun merely to extract their saved PNG outputs.

The destination comes from the current kernel's `save_summary`, not from the
latest folder on disk or a hard-coded sample/run:

```text
results/runs/<export timestamp>/<scan_set_id>/
├── manifest.json
├── workup.zarr/
└── figures/
    ├── cell_001_output_001.png
    ├── ...
    ├── workup.ipynb
    └── figure_manifest.json
```

`workup.ipynb` is an exact snapshot of the saved notebook. The figure index
records the export timestamp, scan-set ID, source notebook path, source cell
IDs/numbers, and SHA-256 hashes of the original export manifest, notebook, and
each PNG. Relative links to `../manifest.json` and `../workup.zarr` remain
usable if the whole export folder is moved. The original workup manifest and
Zarr store are not modified.

Safety checks require agreement between the live configuration, the saved
export manifest, and the saved notebook's last successful workup-save output.
An unsaved notebook, changed configuration, different export, malformed PNG,
or incomplete Zarr store stops the archive before any figures are published.
If prompted to save the notebook, press Ctrl+S and rerun only the archive cell,
not the Zarr-save cell. Existing `figures/` archives are never overwritten.

For each new workup, restart the kernel and clear old notebook outputs before
running cells. Only keep plots that belong to the current workup. A raster
image does not reliably identify which inputs produced it: the archive checks
the export link, not the scientific provenance of every individual plot.
Skipping a plotting cell while retaining its output from an earlier workup
would archive that stale output too.

All stored `image/png` outputs are extracted at their existing resolution,
without resizing or recomputing the data. Interactive, HTML-only, SVG-only,
and widget-only outputs are not converted to PNG; they are listed separately
in the figure index and retained in the notebook snapshot. Interactive output
may still need its original software/kernel to work when reopening the copy.

The archive is optional and independent of a successful numerical export.
Its absence does not invalidate the Zarr results. It shares the export's
retention policy: `results/` is excluded from Git, so back up the complete
export folder separately.
