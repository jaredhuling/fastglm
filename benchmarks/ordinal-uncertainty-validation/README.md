# Ordinal uncertainty: bounded development study

This study examines the availability and finite-sample behavior of the uncertainty procedures in fastglm 0.1.3. It is a development comparison, not a formal evaluation or a guarantee of coverage. The estimator, covariance formulas, convergence rules, bootstrap failure policy, and interval construction are unchanged throughout the study. Coverage results are not used to tune any method.

The package's computational purpose is to make repeated ordinal fits practical while retaining explicit numerical failures. This panel examines the statistical interpretation of those fits at modest dimensions. It does not evaluate high-dimensional or large-category scaling.

## Scientific specification

Each dataset follows the marginal cumulative-logit model

`Pr(Y <= j | X, offset) = plogis(alpha[j] - X beta - offset)`.

There are four ordered categories and two location predictors. The common slopes are `(0.5, -0.35)`. The default thresholds are `(-1, 0, 1)`. Offsets are independent uniforms on `[-0.2, 0.2]`, are observed, and enter every fit with coefficient one. Predictors are jointly standard normal with the correlation specified below. Predictors and offsets are redrawn for each dataset. Both maximum likelihood (ML) and mean bias reduction (meanBR) use the same generated dataset.

| Scenario | Rows | Predictor correlation | Change and scientific question |
| --- | ---: | ---: | --- |
| Regular | 250 | 0.25 | Reference sampling behavior under a correctly specified model. |
| Small | 60 | 0.25 | Accuracy and interval availability at a smaller sample size. |
| Rare endpoint | 120 | 0.25 | First threshold is -4, making the first category rare; record sample-empty categories and bootstrap support changes. |
| Correlated predictors | 120 | 0.90 | Uncertainty in weak predictor contrasts despite full population rank. |
| Independent clusters | 150 | 0.25 | Thirty independent clusters of five observations, with dependent predictors and outcomes within clusters. |

For the clustered scenario, each predictor is a sum of independent cluster and individual Gaussian components, giving within-cluster predictor correlation 0.5. The two predictors retain marginal correlation 0.25. The outcome noise is `qlogis(pnorm(Z))`, where `Z = sqrt(0.6) U_cluster + sqrt(0.4) U_individual` and all Gaussian errors are independent of predictors and offsets. Each `Z` has a standard normal marginal, so this construction gives a standard logistic marginal error and the stated **exact marginal proportional-odds target**. It does not substitute conditional random-intercept slopes for marginal slopes. The clusters are independent; the cluster asymptotic interpretation requires the number of clusters to grow. Thirty clusters is a finite-sample assessment of that approximation.

The targets are all three thresholds and both slopes. The main display uses the first threshold and first slope, selected before the study. All parameters are retained in the results. Every scenario has finite population thresholds and positive category probabilities. Sample-empty categories remain possible and are not removed or replaced. No probability-model misspecification scenario is included: performance of the sandwich under a misspecified cumulative-probability model remains an untested gap in this panel.

## Fixed methods and interval rules

Each estimator uses the public `fastglm_ordinal()` defaults, including its data-derived initialization, convergence requirements, `tol = 1e-8`, and maximum 200 iterations. No true parameter is passed to an estimator or used as a starting value.

Each fitted model is evaluated using four marginal 95% interval procedures:

1. Expected-Fisher Wald intervals, `estimate +/- qnorm(.975) * SE`.
2. HC0 ordinary-score sandwich Wald intervals.
3. HC1 ordinary-score sandwich Wald intervals.
4. Bootstrap percentile intervals from 199 public `fastglm_bootstrap()` refits, using its default R quantile interpolation.

HC0/HC1 use independent individual scores for independent datasets. For the clustered scenario, they use whole-cluster score sums, the package's `G/(G-1)` cluster correction, and the package's HC1 correction when requested. The critical value remains normal; no small-cluster t adjustment is added. For meanBR, the sandwich is the package's explicit first-order ordinary-score convention evaluated at the bias-reduced estimate, not the covariance obtained by differentiating its complete adjusted equations. Its regular asymptotic interpretation is conditional on the assumptions described in the package documentation.

The independent scenarios use the parametric bootstrap, conditional on observed predictors and offsets. The clustered scenario resamples entire clusters. The model-based Wald intervals in that scenario are a diagnostic comparator whose independence-based variance does not account for within-cluster dependence. The marginal coefficient target remains correct.

The public bootstrap withholds all standard errors and intervals if the original fit or any refit has empty categories, a failed statistic, or nonconvergence. All finite point replicates, empty-category patterns, warnings, errors, and convergence diagnostics are retained. This policy can make bootstrap availability substantially lower than original-fit convergence. The study evaluates that existing behavior without replacing unavailable intervals or conditioning the advertised procedure on successful refits. An original fit can itself fail, separately from a bootstrap with unavailable summaries.

## Replications, precision, and provenance

The main panel uses 200 independent datasets per scenario, with ML and meanBR paired on each dataset. At coverage 0.95 with all 200 intervals available, the binomial Monte Carlo standard error is about 0.0154. Conditional coverage with fewer available intervals is less precise. The 199 bootstrap draws also introduce simulation error into percentile endpoints; no claim of precise tail calibration is based on this bounded panel.

Dataset and bootstrap seeds are deterministic functions of stage, scenario, and replicate. Both estimators receive the same bootstrap seed; their parametric distributions differ because they have different fitted parameters. The generator is explicitly Mersenne-Twister, with Inversion normal generation and Rejection sampling. Runs use one process and one requested numerical-library thread. There is no separate scheduled task; the runner computes summaries and audits after completion.

A timing-only pilot uses three datasets per scenario and 39 bootstrap refits with disjoint seed blocks. Its records are retained under `pilot-v1`. The pilot was inspected for runtime and execution correctness, not used to select methods, scenarios, or interval constants. The full run uses `development-v1`. The full runner adds paired-estimator summaries and an archived-runner hash audit; neither change affects a fit or interval. The exact pilot runner is retained in its directory.

The measured installation is recorded by library path, version, R database and native-library hashes, session information, and source hashes. The portable baseline source archive is `method-source/fastglm_0.1.3.tar.gz`, SHA256 `e7d06592082b5a33317e4d2e1ce4acb786f4bec142b0ce8892dc97c317d6307b`. Its R files, namespace, and native ordinal source match the immutable study snapshot. Its DESCRIPTION includes normal metadata added by `R CMD build`. The immutable local installation used for this run is `/private/tmp/fastglm-ordinal-reuse/lib`; that path can be replaced by an installation built from the retained archive. Cross-platform numerical results may differ slightly, and timings are environment-specific.

## Outputs and interpretation

Each `records/*.rds` file contains one dataset identity, seeds, truth, observed category counts, both point estimates, fit diagnostics, all intervals, and both full bootstrap result objects. It does not retain the complete generated design; the generator, seed, and saved data hash identify it. No dataset is replaced after failure.

`summary.csv` distinguishes availability, coverage conditional on availability, and the probability that an interval is both available and covers the target. The last quantity counts an unavailable interval as absence of a covering result; it is not ordinary conditional coverage. Widths and standard errors summarize available intervals. Coverage, availability, widths, and paired differences include Monte Carlo standard errors. `point-summary.csv` reports converged finite estimates and separately summarizes all finite point estimates, so nonconverged numerical results remain visible without being treated as validated fits.

`paired-differences.csv` compares each interval procedure with the model-based interval on paired datasets. `paired-estimators.csv` compares meanBR with ML. Common-available and complete-dataset comparisons are reported separately. `simultaneous-diagnostics.csv` describes inclusion of all five parameters; the procedures consist of marginal 95% intervals and do not have a nominal 95% simultaneous guarantee. `fit-records.csv` and `bootstrap-diagnostics.rds` retain failure and empty-category details. No aggregate ranking across scenarios is defined.

The decision supported by this study is whether current finite-sample behavior and availability justify the existing scope statements, or identify a specific limitation requiring a new theoretical or implementation investigation. A coverage discrepancy is evidence to investigate assumptions, approximation, or code; it is not a reason to inflate an interval or change a numerical tolerance.

## Reproduction

Run commands from the repository root. R, fastglm and its declared dependencies are required; the runner installs nothing and uses only base/recommended R functions beyond fastglm. To rebuild the method in a separate library:

```sh
mkdir -p /path/to/study-library
R CMD INSTALL -l /path/to/study-library benchmarks/ordinal-uncertainty-validation/method-source/fastglm_0.1.3.tar.gz
```

The main run can be resumed without overwriting completed or failed records. A mismatching runner or installed-method fingerprint rejects a resume. Choose a new run ID to rerun on a newly built installation; summaries and auditing never launch fits.

```sh
Rscript benchmarks/ordinal-uncertainty-validation/run.R --stage=pilot --library=/path/to/study-library --run-id=pilot-rerun
Rscript benchmarks/ordinal-uncertainty-validation/run.R --stage=run --library=/path/to/study-library --run-id=development-rerun
Rscript benchmarks/ordinal-uncertainty-validation/run.R --stage=summarize --run-id=development-v1
Rscript benchmarks/ordinal-uncertainty-validation/run.R --stage=audit --run-id=development-v1
Rscript benchmarks/ordinal-uncertainty-validation/verify.R
Rscript benchmarks/ordinal-uncertainty-validation/report.R
```

`--output` changes the result root; `--scenario` can run one scenario at a time within the same compatible run. `--source-root` records source hashes when a source tree is available. The installed method hashes still apply if a source tree is not supplied. The saved runner and manifest identify the computation that produced each result.

`verify.R` regenerates all dataset hashes, checks bootstrap endpoints and standard errors against retained replicates, and verifies byte-for-byte regeneration of the derived CSV tables. `report.R` generates the interpreted result tables and static coverage/availability figure. Both accept a run-directory path as their first argument. `check-endpoints.R` records the separate endpoint-arithmetic check. An initial audit using literal probabilities `.025` and `.975` differed from the package's stored-level expressions by at most `1.110223e-15`; two strict relative comparisons failed near small endpoints. Recomputing `c((1-level)/2, (1+level)/2)` at the stored level reproduces all available endpoints exactly. The initial audit log and comparison records are retained; no fitted result or interval was modified.

## Primary references

The design and reporting separate aims, mechanisms, targets, methods, and performance measures as described by [Morris, White, and Crowther (2019)](https://doi.org/10.1002/sim.8086); Monte Carlo uncertainty is reported rather than interpreting a point coverage estimate as exact. The cumulative-link bias-reduction target is from [Kosmidis (2014)](https://doi.org/10.1111/rssb.12025). Ordinary-score sandwich construction follows [Zeileis (2006)](https://www.jstatsoft.org/article/view/v016i09), with cluster sums and corrections described by [Zeileis, Köll, and Graham (2020)](https://www.jstatsoft.org/article/view/v095i01). These references support the method definitions and study practice, not a finite-sample coverage guarantee for this panel.
