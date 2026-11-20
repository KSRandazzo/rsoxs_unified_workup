# Configuration with two manually maintained CSVs

Maintain sample descriptions in `config/samples.csv`. Maintain one row per
scan set in `config/scan_sets.csv`, including its existing metadata and four
polarization run-ID columns:

| Column | What to enter |
|---|---|
| `pol0_scan_id` | Physical run ID supplying the 0-degree images |
| `pol90_scan_id` | Physical run ID supplying the 90-degree images |
| `pol45_scan_id` | Physical run ID supplying the 45-degree images |
| `pol135_scan_id` | Physical run ID supplying the 135-degree images |

Keep all four columns, even if a set has only two polarizations. Blank,
`N/A`, or `None` means absent; zero is not a missing-value marker. Run IDs
must be positive integers. A set may contain 0/90, 45/135, or both pairs.
An incomplete pair or a set with no run IDs raises an error.

For example, these are only the relevant fields, not replacement files:

```csv
scan_set_id,polarization_layout,pol0_scan_id,pol90_scan_id,pol45_scan_id,pol135_scan_id
ksr216_2026apr_longC_01,separate_runs,110713,110714,110715,110716
DS_EB1A_2023jul_longC_01,within_run,67456,67456,,
```

Retain `sample_id`, `scan_type`, `energy_region`, proposal/cycle, ADC mode,
beam-decay metadata, and other existing scan-set columns. Each `sample_id`
must identify exactly one row in `samples.csv`; scan-set IDs must be unique.

`separate_runs` means each physical run supplies a single configured
polarization; reused run IDs within that set are rejected. `within_run`
means the loader selects the requested polarizations from measured frame
metadata. It permits repeated IDs, including two mixed runs supplying one
perpendicular pair each. Only explicitly entered polarizations are loaded.

## Generated records

`load_scan_set(...)` generates logical scans, the standard perpendicular
pairs, and four sectors per pair in memory. It does not download measured
data, rewrite CSV files, or guess missing runs. It preserves the existing
pair IDs (`<scan_set_id>_pair_0_90` and `_pair_45_135`) and sector IDs/labels.
The notebook, masks, integration, anisotropy, and export interfaces remain
unchanged. The saved manifest includes the four manual run-ID values and
all generated scan, pair, and sector records.

The old `scans.csv`, `polarization_pairs.csv`, and `anisotropy_sectors.csv`
have been archived in `config/legacy/` as historical reference. They are NOT
read, synchronized, or used as fallback. Do not keep editing them for new
scan sets: change the run-ID columns instead. All archived rows are retained.
Existing results/exports are not changed by this migration.

For backwards compatibility, a configuration folder whose scan_sets.csv
has none of the four run-ID columns still uses the old five-CSV layout,
including custom legacy sectors. A partial set of run-ID columns raises
an error. In compact mode, every row must explicitly specify its runs.
Nonstandard angles or custom sector centers require the legacy layout;
compact mode uses only the standard 0/90 and 45/135 pair definitions.
The archive alone is not a runnable legacy configuration folder: the old
layout requires all five linked CSVs together in the selected configuration
directory. It is not selected automatically from `config/legacy/`.

## Processing TOML and sector halfwidth

Keep one `config/processing/<scan_set_id>.toml` per scan set. The CSV
migration does not invent missing processing settings or beam centers.
Every existing processing TOML receives the following section:

```toml
[anisotropy]
chi_halfwidth_degrees = 45
```

This controls the generated sector halfwidth for both pairs and both ring
normalization modes. It may be greater than zero and no greater than 45
degrees; wider sectors would overlap. Omission defaults to 45 degrees.
Changing it to 30 creates 60-degree sectors with gaps between them; it does
not fill those gaps or change the normalization region.

This is independent of `[beam_center_fit] chi_half_width_degrees` and
`[normalization] half_ring_halfwidth_degrees`. Those settings are unchanged.

After changing only the anisotropy halfwidth, save the TOML, reload
`configuration = workup_config.load_scan_set(SCAN_SET_ID)`, and rerun the
sector and pointwise calculations plus all dependent averages, fits, and
plots. Existing in-memory results do not update automatically. Integration
or center fitting need not be repeated solely for a sector-width change.

When changing physical run IDs or switching scan sets, restart the kernel
and run the normal workup from the beginning. Review before exporting.
