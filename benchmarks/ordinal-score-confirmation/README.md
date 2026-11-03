# Confirmation of analytic ordinal score inference

`ASSESSMENT.md` gives the scientific findings and implementation decision.
`METHODS.md` specifies the statistic, frequency convention, assumptions, and
computational scope. `PROTOCOL.md` was frozen before the main evaluation;
`precision/PROTOCOL.md` records the later targeted precision check separately.

The main bank contains 27,000 datasets and 54,000 joint/selected-term records.
The precision bank contains 10,000 new datasets and 20,000 records. Raw results,
including failures, are stored per cell. `rates.csv` reports rejection rates,
Monte Carlo intervals, availability, and bounds over missing decisions.
`null-rejection.pdf` plots the original null results. `timing-summary.csv`
reports local public-interface timings; `validation.csv` records prototype
agreement. `checks/` retains local test, build, check, and rendering records.

## Reproduction

From the repository root, select the matching frozen fastglm installation with
`FASTGLM_CONFIRM_LIBRARY`. The original local path was
`/private/tmp/fastglm-score-confirm-lib`. Set numerical-library thread counts
to one when reproducing timings or parallel runs.

    Rscript benchmarks/ordinal-score-confirmation/run.R audit
    Rscript benchmarks/ordinal-score-confirmation/precision/run.R
    Rscript benchmarks/ordinal-score-confirmation/interpret.R
    Rscript benchmarks/ordinal-score-confirmation/plot.R
    python3 benchmarks/ordinal-score-confirmation/check-regeneration.py

The first two commands verify the frozen code and installation and regenerate
summary tables. With existing precision records, the precision command audits
them without rerunning datasets. Starting a new study requires a separate
directory and an explicit protocol; the original raw bank is not overwritten.
The main runner deliberately refuses to overwrite any completed cell.

`manifest.rds` and `manifest.txt` identify protocol, generator, method source,
installed-package files, R version, and RNG convention. `source-snapshot.rds`
contains the exact source bytes, including the uncommitted implementation added
to base revision b5e08af. Rebuilding a package can change installed-file hashes;
this is distinguished from source and numerical reproducibility.

`provenance.csv` inventories final evidence by SHA-256. The relocation check
verifies that inventory, reconstructs the source tree from the snapshot, audits
both banks, and reproduces summaries and the assessment outside this checkout.
It compares tabular results and prose byte for byte; PDF metadata can vary with
render time, so the plot is inventoried but is not a byte-reproduction target.
No simulations are used when the installed test computes an applied p-value.
