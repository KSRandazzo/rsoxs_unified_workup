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

In `config/scans.csv`, write one row per desired polarization. For example,
a two-polarization carbon run would have these relevant fields:

```csv
scan_set_id,scan_id,polarization_deg
DS_EB1A_longC_01,67456,0
DS_EB1A_longC_01,67456,90
```

In `config/polarization_pairs.csv`, add `polarization_deg_1` and
`polarization_deg_2` to distinguish pair members with the same run ID:

```csv
scan_set_id,pair_id,scan_id_1,scan_id_2,polarization_deg_1,polarization_deg_2
DS_EB1A_longC_01,DS_EB1A_longC_01_pair_0_90,67456,67456,0,90
```

Old pair rows can leave the new columns blank when each run ID uniquely
identifies its member. Existing columns in either CSV must be retained.
The examples above show the relevant fields, not replacement files.

Configure sectors and one processing TOML per scan set as before. Pair IDs
must match the sectors and `[beam_centers.<pair_id>]` TOML tables. A carbon
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
