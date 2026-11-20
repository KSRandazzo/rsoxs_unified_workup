# Chi-averaged pointwise anisotropy versus energy

The current notebook review starts from the q-capped pointwise anisotropy
maps, separately for each polarization pair and normalization mode. It does
not rerun integration, normalization, or angular fitting.

Its calculation order is:

1. Average finite signed A(E, chi, q) over chi to obtain A_chi_mean(E, q).
2. Select the native q bins in the inclusive `Q_RANGE`.
3. Optionally take abs(A_chi_mean), controlled by `ABSOLUTE_Q_VALUES`.
4. Average or sum over finite q bins using `Q_REDUCTION`, giving a curve in E.

```text
A_chi_mean(E, q) = mean_chi(A(E, chi, q))
B(E, q) = abs(A_chi_mean(E, q))  # if ABSOLUTE_Q_VALUES = True
curve(E) = mean_q(B(E, q)) or sum_q(B(E, q))
```

Signed angular values can cancel before the absolute value is taken. This
is deliberately different from averaging abs(A) over chi.

Set `Q_RANGE`, `Q_REDUCTION = "mean"` or `"sum"`, and
`ABSOLUTE_Q_VALUES` in the calculation cell. Both normalization modes are
calculated. In the plotting cell, set `NORMALIZATION_MODE` and edit the
`ENERGY_MARKERS_EV` list to position the dotted vertical energy markers.
Change calculation options and rerun the calculation and plot cells;
changing only energy markers requires rerunning just the plot cell.

Each contributing chi bin has equal weight within q, and each contributing
q bin has equal weight in the q mean. Missing/invalid points are ignored;
an entirely missing q window stays NaN, not zero. The existing q cap and
angular masks remain applied, and no missing arcs are filled or smoothed.
Pairs retain their own q grids. Partial q windows are allowed, so changing
coverage can affect comparisons across energies.

For N contributing q bins, sum = N * mean. They rank energies identically
when N is constant; the mean is less sensitive to window width and bin
count. A bin sum is not a physical integral over q. Absolute values also
have a positive noise contribution.

These notebook review curves remain in memory and are not automatically
included in Zarr exports. Their displayed plots can be included in a
notebook figure archive using the existing workflow.

## Separate magnitude diagnostic retained in the code

`summarize_absolute_pointwise_anisotropy` and
`plot_absolute_pointwise_energy_review` implement an alternative diagnostic
developed earlier: abs(A) is taken BEFORE the chi mean, complete q-window
support is required by default, and the plot uses a horizontal reference.
Their tests cover that distinct definition. The current notebook review
does NOT call those functions; they should not be mistaken for the
calculation and vertical-marker plot described above.
