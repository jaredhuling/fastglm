# Benchmark result storage

Raw benchmark results are regenerable and are not tracked in this repository. An archive is kept on Box:

`Box-Box/Research/fastglm/benchmarks-archive-20261106/`

On macOS with Box Drive, this folder is at `~/Library/CloudStorage/Box-Box/Research/fastglm/benchmarks-archive-20261106/`. The location is machine specific.

The archive is a complete copy of the tracked `benchmarks/` directory at commit `5c2efb9`, with its paths unchanged (3,774 files, 129 MB). Each file was checked against its git object hash when the copy was made. The files also remain available in the history of this repository.

Raw replicates from `benchmarks/ordinal-covariance/run.R` are in `Box-Box/Research/fastglm/ordinal-covariance-raw-v2-20261106/` (30 files).

Raw per-cell results from `benchmarks/benchmark-ordinal-comparison.R` are in `Box-Box/Research/fastglm/ordinal-comparison-raw-20261106/`.

## What stays in the repository

`.gitignore` defines the rule. These files remain tracked:

- Code at any depth under `benchmarks/` (`.R`, `.Rmd`, `.py`, `.sh`).
- Documentation at any depth under `benchmarks/` (`.md`).
- Summary `.csv` and `.txt` files directly inside a study directory, such as `benchmarks/<study>/summary.csv`.
- All of `benchmarks/readme-results/`, which `README.Rmd` reads when it is knit.

Everything else is ignored, including `.rds` files, logs, memory profiles, and results in subdirectories of a study. Output that a script writes to these locations stays local and is not added to git. Copy results that should be kept to a new dated archive folder on Box and add it to this file.

## Rerunning a summary from archived results

Some study scripts summarize or audit raw files that are now only in the archive. Restore those files before running such a script. From the repository root:

```sh
rsync -a --ignore-existing \
  ~/Library/CloudStorage/Box-Box/Research/fastglm/benchmarks-archive-20261106/benchmarks/ \
  benchmarks/
```

Restored files are ignored by git, and `--ignore-existing` leaves current tracked files unchanged.
