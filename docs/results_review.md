# Development exports, drafts, and approved results

Every new save records `review_status: "draft"` in `manifest.json`. Successful
export verification is not scientific approval. A review decision belongs to
one particular saved export, never to every attempt for a scan set.

## Review in the workup notebook

1. Save as usual. The notebook reports that the new export is a draft.
2. Run the export-verification cell. Review the center, masks, normalization,
   and relevant plots for the scientific analysis you intend to perform.
3. In **Approve this saved export**, set `APPROVE_EXPORT = True`, supply a
   meaningful `REVIEW_NOTE`, and check `REVIEWER`.
4. Run that cell. It displays the exact export directory and asks you to type
   `approve` (capitalization does not matter). Anything else cancels.

The default is False. Even if you leave True saved in the notebook, a fresh
confirmation is required each time: Run All cannot silently approve new data.
The cell requires a successful verification token for the same export and
unchanged manifest. It does not rerun fitting, create another export, or alter
the original manifest, arrays, or figure archive.

## Review a saved export later, or revoke approval

You can record a decision without re-running the workup. Replace the example
path with the **exact export directory** shown by the browser. Do not use the
top-level results folder or automatically select whatever is newest.

```bash
python -m rsoxs_unified_workup.review "/exact/export/directory" approved \
  --reviewer ksr2 --note "Reviewed center, masks, normalization and plots" --confirm
```

To withdraw approval, append a draft decision:

```bash
python -m rsoxs_unified_workup.review "/exact/export/directory" draft \
  --reviewer ksr2 --note "Needs another masking pass" --confirm
```

Both commands require explicit confirmation. Approval additionally requires a
nonempty review note. They only check that the saved export is finalized; you
are responsible for reviewing its scientific suitability. Approval is not a
claim that every possible analysis is valid or that an automated quality test
passed. Do not edit the saved Zarr data in place; export a new version instead.

## Storage and portability

Decisions are small append-only JSON files under:

```text
results/runs/<timestamp>/<scan_set_id>/
  manifest.json                   # immutable provenance; initially draft
  workup.zarr/
  figures/                        # if archived
  reviews/<timestamp>_<uuid>.json # status, reviewer, note, manifest SHA-256
```

The latest review event determines the current status. Earlier decisions are
never overwritten. A draft decision revokes an earlier approval; another
explicit approval can restore it. A malformed/latest mismatched event produces
`unknown`, not a silent fallback to an earlier approval. A changed manifest
invalidates its latest decision. This hashes provenance, not every Zarr chunk;
in-place array changes are unsupported and are not detected by this status
check. Full verification remains the workup notebook's responsibility.

Copy the **whole export directory**, including reviews, when transferring it.
Review records use a manifest hash and scan-set identity, not absolute paths,
so decisions and relative figure links survive relocation. An approved export
whose reviews are omitted reverts to its baseline draft/development status.
The portable browser remains read-only and needs no new specialized packages.

## Browser behavior

The browser defaults to `REVIEW_STATUS = "approved"`.

- `"draft"`: future exports not yet approved, or approval explicitly withdrawn.
- `"development"`: historical exports without a review-status field.
- `"unknown"`: unreadable, unsupported, or mismatched review metadata.
- `None`: every status; a list can select several statuses.

With `LATEST_ONLY = True`, status is selected first, then the newest save in
that status selection, then the other filters. A newer draft does not hide an
older approved result. Reflection filtering still does not substitute an older
approved export if the newest approved export has a different reflection mode.
The table's `is_latest` flag continues to mean newest overall, across statuses.

Until you explicitly approve an export, the default browser view is empty.
That is expected; change the review filter to inspect drafts or development.
After recording or withdrawing a decision, rerun the browser refresh and filter
cells; an already displayed table is a snapshot, not a live status monitor.

## One-time development archive

Stop all running workup saves and close datasets opened from the old results
before archiving. The archive utility previews first:

```bash
python -m rsoxs_unified_workup.development
```

Inspect that inventory, then supply its exact `inventory_sha256` token:

```bash
python -m rsoxs_unified_workup.development --apply --confirm-development \
  --expected-inventory "TOKEN_FROM_PREVIEW"
```

It renames the exact project `results/` folder into
`results_development/<archive_timestamp>/results/` and creates a new empty
active `results/`. This keeps every old export, figures, notebook snapshots,
and incomplete development saves. It does not delete or duplicate large data,
rewrite historical manifests, or move raw beamline/source data or configuration.
New/previously reviewed exports are refused by this development-only utility.
An inventory mismatch stops before moving anything; do not substitute a new
token without checking what changed. The archive must be on the same filesystem.

The archive prints its exact location and writes `archive_manifest.json` with
original/recovery paths and original manifest hashes. Set the browser's
`RESULTS_ROOT` to that archived `results/` folder and `REVIEW_STATUS =
"development"` (or None) to inspect it. Original absolute paths printed in old
notebook snapshots are historical; the browser reports the current locations.
Do not rerun an old save-summary/figure-archive cell using its pre-move paths.

Moving is recoverable: files remain intact at the printed archive location.
Do not move the archive over a nonempty active results folder. Restore selected
exports deliberately if needed. Neither `results/` nor `results_development/`
is tracked by Git; make a separate backup of any data you need to retain.
