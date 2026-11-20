# Archived configuration tables

These three CSVs preserve the last manually maintained scan, pair, and
sector records, including the recent additions. They are reference copies,
not inputs to the current workup and not automatically synchronized.

For active configuration, edit only `config/samples.csv`,
`config/scan_sets.csv`, and `config/processing/<scan_set_id>.toml`.
Scans, standard polarization pairs, and sectors are generated in memory.
Saved export manifests already contain their own configuration snapshots;
moving these CSVs does not change or invalidate any saved results.

Legacy loading remains supported for an explicitly selected configuration
folder containing all five old-format linked CSVs. This archive by itself
does not contain such a complete configuration and is not a fallback.
