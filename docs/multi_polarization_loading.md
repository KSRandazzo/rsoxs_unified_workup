# Runs containing more than one polarization

The workup still creates one dictionary entry per polarization:
`scan[0]`, `scan[90]`, and, when configured, `scan[45]` and `scan[135]`.
The true BNL run ID is retained. A logical member is identified by its
**run ID and polarization together**, not by run ID alone.

## CSV configuration

Add an optional `polarization_layout` column to `config/scan_sets.csv`:

- `separate_runs`: each physical run contains one configured polarization.
  This is the default for existing rows with no column or a blank value.
- `within_run`: a physical run contains multiple polarizations, identified
  by its measured per-frame polarization coordinate.

Continue to use `scan_type` (`long` or `repeats`) and `energy_region`
(`C`, `O`, etc.) independently of this layout.

In `config/scan_sets.csv`, enter each desired polarization's physical run ID
in the four columns `pol0_scan_id`, `pol90_scan_id`, `pol45_scan_id`, and
`pol135_scan_id`. For a two-polarization carbon run, the relevant fields are:

```csv
scan_set_id,polarization_layout,pol0_scan_id,pol90_scan_id,pol45_scan_id,pol135_scan_id
DS_EB1A_longC_01,within_run,67456,67456,,
```

The loader generates separate logical scan records and an explicitly
polarization-qualified pair from these entries. No manual entries in
`scans.csv`, `polarization_pairs.csv`, or `anisotropy_sectors.csv` are needed.
Blanks, `N/A`, and `None` mark absent polarizations; incomplete perpendicular
pairs are rejected. Repeating a run ID is valid with `within_run`.

The example shows only relevant fields, not a replacement CSV: retain all
existing scan-set metadata columns. See `compact_configuration.md` for the
complete manual-input workflow and backwards-compatible legacy behavior.

Keep one processing TOML per scan set, with `[anisotropy]
chi_halfwidth_degrees = 45` controlling generated sector halfwidths. Generated
pair IDs match the existing `[beam_centers.<pair_id>]` TOML tables. A carbon
run and an oxygen run belong in separate scan sets with their respective
energy regions, fit/plot energies, and reviewed beam centers.

## Loading behavior

The loader reads each physical run once per `load_scans` call. It selects
the configured polarizations by frame index and selects the matching Au
mesh monitor readings, energies, exposure metadata, and auxiliary frame
coordinates with the same indices. Selection does not average images,
interpolate pixels, or change their dtype. Only explicitly configured
polarizations are processed.

Each selected stack has an energy-only `system` MultiIndex, matching the
existing integration interface. Mixed-run records include
`polarization_selection` provenance and a `source_frame_index` coordinate
mapping selected images to frames in the physical loaded run.

The complete source-run dark stack is loaded once and attached to each
selected polarization record. It is not split by polarization. Any
per-image `dark_id` coordinate remains on the selected data. Existing dark
masking and dark integration are unchanged; the integrated dark diagnostic
still uses the first dark frame.

The underlying PyHyperScattering loader's existing preprocessing is
unchanged. “No averaging” here describes the new polarization-selection
step, not upstream operations performed by that loader.

Ambiguous pairs, absent measured polarizations, incorrect monitor lengths,
and duplicate energies within a selected polarization raise errors. This
workflow expects one loaded image per energy/polarization; it does not add
automatic repeat averaging. Downstream pair members must still have
matching energy and detector coordinates.

## Processing and exports

Masks, reviewed beam centers, normalization, anisotropy, and angular fits
retain their existing pairwise behavior. Different perpendicular pairs do
not acquire a common center or q grid just because they came from one run.

For mixed runs, detector data and darks use distinct export groups:

```text
pairs/<pair_id>/scans/<run_id>/polarizations/<polarization>/data
pairs/<pair_id>/scans/<run_id>/polarizations/<polarization>/darks
```

Existing separate-run raw export paths remain unchanged. Scientific product
groups keep their polarization axis; their `scan_id` coordinates may now
repeat legitimately. The manifest records `polarization_layout`, and
mixed-run detector groups include selection provenance. The manifest
schema version is 7.

After installing the code, restart the notebook kernel before the first
measured test. Run the normal workflow with the newly configured scan set,
review its two- or four-polarization results, and verify saved groups before
merging the feature branch.
