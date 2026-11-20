# RSoXS Unified Workup

A reusable and reproducible workflow for processing RSoXS scans without duplicating notebooks.

The raw data in `../sst1_2026apr` will remain unchanged and will not be stored in this Git repository.

## Current configuration and notebooks

Maintain sample metadata in `config/samples.csv` and scan sets/run IDs in
`config/scan_sets.csv`. Each scan set also needs a processing TOML under
`config/processing/`. Scans, standard polarization pairs, and sectors are
generated in memory; the former linked CSVs are archived in `config/legacy/`
and should not be edited for new workups. See [configuration instructions](docs/compact_configuration.md).

`notebooks/workup.ipynb` performs processing, reviews, and verified Zarr
exports. Configured-center approval without fitting and PNG figure archives
are supported; see [figure archive instructions](docs/notebook_figure_archives.md).

`notebooks/browse_results.ipynb` is an independent, read-only saved-results
reader. It does not require the workup package or processing environment.
Use an environment with `requirements-browse.txt` dependencies. It includes
filters for the beam-centering mask reflection setting saved in each export.
See [portable browser instructions](docs/portable_results_browser.md).

The notebook's anisotropy-versus-energy review first averages signed
pointwise anisotropy over chi, optionally takes the absolute value, and
then averages or sums across a chosen q range. User-defined vertical energy
markers aid comparison. See [energy-summary instructions](docs/absolute_pointwise_energy_summary.md).

Saved data and figures live under `results/`, which is excluded from Git.
Committing code/configuration does not back up those exports.
