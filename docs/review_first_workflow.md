# Review-first workup

Order: geometry/masks -> all-energy processing -> all three anisotropy
estimators -> robustness/energy selection -> optional angular fitting ->
verified draft export. Energy diagnostics never depend on periodic fits.

## Shared energies and the review loop

`[plotting] energies_ev` in the scan set's TOML is the only saved
interesting-energy list. Early plots can use provisional choices. The
independent `beam_center_fit.fit_energy` is automatically added for plotting
and markers without changing either TOML setting.

After inspecting the energy curves, edit energies_ev, save the TOML, and
rerun **Energy selection and review loop**. This refreshes selected-energy
image/mask/integration/intensity and anisotropy figures without loading or
recomputing physical processing. Set INCLUDE_IMAGE_REVIEWS=False for a
quicker anisotropy/intensity-only review. The accepted center is unchanged.

Changing Q_RANGE, Q_REDUCTION, ABSOLUTE_Q_VALUES or REQUIRE_COMPLETE_Q_RANGE
requires only rerunning the inexpensive energy-summary calculation first.
Both energy curves use identical finite angular bins and native q windows:

- Main: reduce_q(abs(mean_chi(A))), or signed chi mean.
- Companion: reduce_q(mean_chi(abs(A))).

The companion avoids angular cancellation, but abs values have a positive
noise contribution. Q-bin coverage is displayed. Partial windows remain
allowed by default; require complete support to blank energies missing any
selected q bin. All-missing windows stay NaN, including sums. Bin sums are
not physical q integrals. Both diagnostics come directly from the original
measured, q-capped pointwise anisotropy, not fitted offset/amplitude/phase.

Physical processing changes invalidate review-only refreshes. Rerun the
affected calculations and downstream results. Review energies and fit-slice
requests are excluded from the numerical snapshot. Missing CSV metadata is
handled without false stale-setting errors.

## Robustness and fitting

Three-estimator reviews precede fitting. Both normalizations and individual
pairs stay separate. Compare feature locations and coverage, not exact
equality. These estimators share processing assumptions; agreement is not
independent validation against common processing artifacts.

Type REVIEWED at the checkpoint after inspecting the data. This authorizes
fitting for the current review settings/source, not export approval. Changing
the energies, q range or pointwise source invalidates the checkpoint.

Every configured interesting energy receives a combined 90-degree sinusoid
and triangle review, using the same measured bins and q_value/q_bins.
Requested and selected energies are reported. A centering energy is fitted
here only if it is also in energies_ev. Ineligible caps/windows or missing
angular data are explicitly reported NOT FITTED, never extrapolated.
The redundant sinusoid-only figure is removed. Old scalar angular-fit
energy_ev remains unchanged for standalone callers, but does not select
the notebook's new multi-energy reviews.

Full parameter maps remain sinusoidal, not triangle maps. Set
RUN_ANGULAR_MAPS=False to skip expensive map fitting. Every successful
selected sinusoid is cross-checked against the full map. ENERGY_RANGE_EV
changes plotting only, with horizontal energy markers because energy is
vertical. Carbon fit-map zoom defaults to 282–290 eV and oxygen to
528–538 eV. The data-derived energy-diagnostic map uses 280–290 eV for
carbon and 528–538 eV for oxygen. These are display bounds, not changes to
the processing TOML or fitted data. Display changes do not require refitting maps.

## Saving and safety

Existing Zarr groups are unchanged. The two energy summaries and selected
waveform comparisons remain notebook review results, not new Zarr groups;
their measured source is already exported and the figure archive retains
their PNGs. Full sinusoidal maps are exported as before when calculated.
Skip fitting cells to export a draft without maps. The map variable starts
as None and is invalidated before refitting, so a failed rerun cannot reuse
an old map. Export approval remains a separate explicit step.

Before committing the notebook template, retain any wanted figure archives
and clear saved execution outputs. The merge-cleanup installer backs up
both complete notebooks under `results/notebook_checkpoints/` before clearing
their outputs. Source cells and their order are preserved except for the
explicit energy-window correction above. Scientific exports, figure archives,
CSV files and all TOML parameter values are unchanged by that cleanup.
