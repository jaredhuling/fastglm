# Post-confirmation precision check

This follow-up was specified after examining the frozen confirmation results.
It does not replace, enlarge retrospectively, or tune the original evaluation.
The joint correlated-six-category test at n=1600 rejected 70/1000 null datasets;
the selected-first-predictor narrow-interior-category test at n=1600 rejected
66/1000. Their pointwise Wilson intervals excluded 5%. Because these departures
occurred at the largest sample size, we check whether they persist with more
replications. Modest finite-sample inflation is compatible with a useful
first-order procedure; equality to 5% in each cell is not an acceptance rule.

Reuse the exact frozen production method and generator, without changes or
additional variants. Generate 5000 fresh datasets for each of original cells 21
and 22. Record both joint and selected-x1 tests, although the two elevated
comparisons motivated the follow-up. Dataset seed is
`120361101 + 10000 * original_cell + replication`, using the same RNG as the
original study. These seeds do not overlap the original confirmation or
development studies. Two single-threaded processes run the two cells.

Report 5% and 1% rejection rates, 95% Wilson Monte Carlo intervals, availability,
failure reasons, and bounds over missing decisions. Keep these post-selection
precision results separate from the original confirmation tables. Do not tune
the method or choose further scenarios according to these results. Preserve
all raw records and audit their identities, seeds, degrees of freedom, and
p-values. Run from the repository root with the same frozen library:

    Rscript benchmarks/ordinal-score-confirmation/precision/run.R

The command first audits the original study, then creates or audits the new
records. Frozen source and follow-up protocol hashes are checked before and
after execution. `ASSESSMENT.md` in the parent directory interprets both stages.
