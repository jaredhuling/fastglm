# Ordinal fitting benchmark

These results compare cumulative-logit models with three common slopes and four
ordered categories, using 2,000 or 20,000 observations. The two data sets use fixed
seeds, normally distributed predictors, integer frequency weights including zeros,
and known location offsets. ML and mean bias reduction use the same data within
each sample size. This small panel does not establish scaling to large numbers of
predictors or outcome categories, nor performance under separation or ill conditioning.

## Calls and verification

The ML comparison uses the public formula interfaces of `fastglm_ordinal()` and
`MASS::polr()`. The mean-bias-reduction comparison uses the public matrix interface
of `fastglm_ordinal()` and the internal matrix fitter in released **WeightIt 2.1.0**.
The latter avoids unrelated weighting and inference work in WeightIt's public
wrapper. Both implementations solve the cumulative-logit mean-bias-reduced score
equations; a log-determinant penalty is not substituted for those equations.

Every timed call produces fitted category probabilities. The fastglm calls include
public `predict()` because the reference fitters produce these matrices during
fitting. fastglm also returns inverse expected Fisher information. Both reference
calls disable their optional Hessian calculation (`Hess = FALSE` or `hess = FALSE`).
The ML formula calls retain model frames. These are comparisons of complete calls
with these specified outputs, rather than isolated linear-algebra kernels.

The fitters use their own stopping rules, with all controls saved in `metadata.rds`.
Agreement thresholds were set before execution: maximum absolute coefficient
difference at most `1e-6`, and probability difference at most `2e-7`. Convergence,
threshold ordering, and agreement between reported probabilities and the model
formula are also checked. WeightIt's internal fitter does not return its BR
convergence flag. Its criterion is reconstructed from the returned adjusted-score
matrix: slope columns precede thresholds; dividing slope columns by their original
unweighted predictor SDs recovers the solver coordinates. The maximum absolute
score must be at most `1e-10 * max(1, sum(weights))`, and all category probabilities
must exceed `1e-10`. The script asserts the column order and preservation of the
unnormalized input weights.

The initial fits provide both accuracy checks and warm-up. Four repetitions then
alternate implementation order. Each repetition times ten complete fits and
reports the batch elapsed time divided by ten. All timed fits are checked outside
the timed interval. A failed, nonconverged, inaccurate, or warning-producing fit
prevents a timing summary for that implementation/case; its evidence remains saved.

## Recorded results

The run used R 4.5.1 on macOS arm64, fastglm 0.1.3, MASS 7.3-65, and WeightIt 2.1.0.
Thread environment variables were set to one. No other package tests or benchmarks
ran during the timing window. The table reports medians across the four batches.

| Estimation | Observations | fastglm, ms | Reference, ms | Reference / fastglm |
|:--|--:|--:|--:|--:|
| ML | 2,000 | 3.45 | 13.95 (MASS) | 4.04 |
| ML | 20,000 | 27.60 | 133.65 (MASS) | 4.84 |
| Mean bias reduction | 2,000 | 5.10 | 21.40 (WeightIt) | 4.20 |
| Mean bias reduction | 20,000 | 44.15 | 191.45 (WeightIt) | 4.34 |

All **328** preflight and timed fits passed validation, with no warnings or errors.
The largest coefficient/probability differences were `4.54e-8`/`1.21e-8` against
MASS and `8.59e-11`/`3.25e-11` against WeightIt. These local warm-run measurements
describe this panel and software configuration; they are not general speed or
statistical-accuracy guarantees. Batch-level variation is retained in `timings.csv`.

## Reproduction and files

From the repository root, using installed versions of the packages above:

```sh
Rscript benchmarks/benchmark-ordinal.R \
  --library=/path/to/fastglm-library \
  --reference-library=/path/to/WeightIt-2.1.0-library \
  --output=/path/to/new-results
```

The script refuses to overwrite a directory containing previous metadata. It needs
no benchmark-specific dependency beyond MASS and the pinned WeightIt installation.

- `cases.csv` records seeds and dimensions; `metadata.rds` records controls,
  versions, library locations, platform, and thread settings.
- `hashes.csv` contains MD5 fingerprints of the benchmark and relevant repository
  sources, installed R databases, and installed binaries. These fingerprints
  identify the measured artifacts even though the development package version is
  shared by several snapshots.
- `validation.csv` retains each convergence and accuracy result;
  `conditions.csv` retains warnings and errors, including a header when there are none.
- `timings.csv` contains every batch timing; `summaries.csv` contains eligible
  medians and ranges. `estimates.rds` retains preflight coefficients and diagnostics.
- `session-info.txt` and `run.log` retain session details and execution output.

`initial-forwarding-error/` preserves the first run and its exact script. An extra
helper forwarded formula arguments through `...`, causing `..3` evaluation errors
in the ML preflight cells. The same wrapper behavior was reproduced with
`stats::glm()`, `MASS::polr()`, and fastglm's GLM and ordinal formula interfaces;
it was a benchmark harness issue. No ML timings were reported for that run.
The completed run uses direct public calls, with unchanged data, seeds, controls,
and accuracy thresholds. The initial partial timings are not pooled into the table.
