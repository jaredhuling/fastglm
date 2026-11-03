# What causes the inflated proportional-odds rejection rates?

The evidence supports a combination of inaccurate finite-sample covariance, unstable covariance inversion, and a poor chi-square tail approximation. Correcting the derivative of the Firth-adjusted score helps the rare-endpoint case but does not solve it, and does not improve the small five-category joint test. Mean bias reduction of coefficients is not sufficient to calibrate their Wald test.

## Design and scope

The frozen main experiment contains 36,000 response datasets over nine fixed predictor designs. Each design has independent banks for moments (1,000), reference calibration (1,000), and evaluation (2,000). All 36,000 binary-fit datasets completed; the largest adjusted-score residual was 1.1e-10. Analytic derivatives agree with numerical differentiation, the original covariance is reproduced, and selected extreme and empty-category fits replay correctly.

The scenarios reproduce the earlier study's model parameters while holding predictors fixed. They isolate conditional sampling behavior, rather than averaging over changing designs. One predictor design per scenario/sample-size cell limits generalization. This is a mechanism investigation, not a confirmation study or a search for the covariance formula closest to nominal size. No production code changed.

The table reports rejection at nominal 5%, conditional on every category being observed. The three-category settings have one predictor; the five-category setting has three predictors and nine slope contrasts. All columns use n=80. Oracle rows require information unavailable in an application.

| Calculation | Regular, 3 categories | Rare endpoint | Rare interior | Regular, 5 categories |
|---|---:|---:|---:|---:|
| Original empirical covariance | 4.8% | 24.8% | 1.5% | 14.3% |
| Exact adjusted-score derivative | 5.1% | 18.2% | 2.0% | 15.2% |
| Independent mean empirical covariance (oracle) | 4.7% | 16.2% | 5.3% | 9.8% |
| Independent sampling covariance (oracle) | 5.5% | 4.6% | 5.5% | 7.8% |
| Independent sampling covariance and mean (oracle) | 5.3% | 4.8% | 5.5% | 7.8% |

Full results, including 95% Monte Carlo intervals, nominal 1%, both selection populations and larger samples, are in summary.csv. For independent-bank procedures the evaluation intervals condition on the realized moment/reference bank; they do not include uncertainty in those banks.

## Covariance scale is a major part of the endpoint failure

For the rare endpoint with all categories observed, the contrast's independent moment-bank standard deviation is 1.735, compared with 0.956 for the square root of the mean original estimated variance. The mean variance estimate is only 30.3% of the independently estimated sampling variance. Its mean contrast is -0.086 (Monte Carlo standard error 0.059), providing little evidence that a large mean shift explains the problem.

Using the exact adjusted-score derivative changes rejection from 24.8% (22.8%–26.9%) to 18.2% (16.4%–20.1%). The paired change is -6.6 percentage points (Monte Carlo standard error 0.6 points). However, its mean estimated variance becomes 2.03 times the independent sampling variance while rejection remains inflated. A correct derivative alone does not make a local covariance approximation accurate; averages can also be dominated by extreme covariance estimates.

Replacing the empirical score products by the true score variance while retaining the fitted ordinary sensitivity gives 1.7% rejection in the rare-endpoint case and 0.2% in the five-category case. Using a fitted proportional-odds score variance gives 0.3% and 0.4%, respectively. These formulas substantially overcorrect. Knowing the score variance does not remove errors from evaluating nonlinear sensitivities at noisy estimates. This rules out presenting a simple score-variance substitution as the solution.

## Covariance randomness and chi-square shape also matter

For the five-category joint test, replacing each dataset's covariance by an independent mean of that same covariance formula reduces rejection from 14.3% to 9.8%. This changes covariance variability and its dependence on the slope contrasts together. It does not separate those two effects. The mean covariance still underestimates variance in some contrast directions: its eigenvalues relative to the independent sampling covariance range from 0.72 to 1.17.

Even with an independently estimated sampling covariance, the five-category chi-square test rejects 7.8% (6.7%–9.1%) at nominal 5% and 2.9% (2.3%–3.7%) at nominal 1%. Removing the independently estimated mean changes these to 7.8% and 2.8%. Distributional shape therefore remains relevant after covariance and mean corrections, subject to the finite precision of the moment bank.

For the rare endpoint, the sampling-covariance calculation gives 4.6% (3.7%–5.7%) rejection at nominal 5%, but 2.7% (2.0%–3.5%) at nominal 1%. Centering the contrast does little. Agreement at one nominal level can conceal an incorrect tail shape; a multiplicative variance adjustment would not establish a valid test.

## Sparsity and empty categories

The post-study count breakdown below uses the same rare-endpoint n=80 evaluation sample. It is descriptive and was not used to select an eligibility cutoff. The number of observations supporting a cumulative split is more informative here than total sample size alone.

| Observations in rare endpoint | Datasets | Original rejection at 5% | Exact-derivative rejection |
|---|---:|---:|---:|
| 0 | 267 | 56.9% | 48.3% |
| 1 | 563 | 36.2% | 28.4% |
| 2 | 528 | 24.6% | 16.5% |
| 3 | 375 | 17.9% | 11.7% |
| 4 | 177 | 12.4% | 10.7% |
| 5+ | 90 | 7.8% | 5.6% |

Retaining all datasets increases original rare-endpoint rejection from 24.8% to 29.1%. Thus the earlier all-categories-observed restriction did not create the inflation. In the sparse-interior case the original test is conservative (1.5% among observed-category datasets), so there is no single directional correction across sparsity patterns.

For an empty interior category, adjacent binary responses are identical. Their fitted slope contrast and its theoretical empirical covariance are zero. In 49 of the 85 such evaluation datasets, floating-point cancellation left a tiny positive covariance (at most 3.0e-33), which passed a Cholesky factorization. These are not genuine nonsingular covariance estimates. The frozen results preserve that behavior and label it here; the observed-category comparisons above avoid it. A future test must handle the mathematical support explicitly, not infer it solely from numerical factorization.

## Oracle simulation calibration and reference uncertainty

The oracle experiment uses the true model to generate independent reference statistics. The rank p-value has a finite-sample exchangeability argument, including conservative treatment of ties. This argument averages over the random reference sample; many evaluation datasets sharing one reference bank do not give independent unconditional p-values. A fitted-null bootstrap does not inherit the exact argument because its nuisance parameters are estimated.

One original reference bank gave an unusually low endpoint critical value. A separately specified post-study check generated exactly 20,000 new reference datasets for each of the endpoint and five-category n=80 designs, keeping the original moment and evaluation banks fixed. The original results remain available. This adds 40,000 response datasets and measures reference precision; it is not fresh confirmation on new evaluation data.

| Scenario | Reference bank | Selected reference size | Rejection at 5% | Rejection at 1% |
|---|---|---:|---:|---:|
| Rare endpoint | Original | 874 | 8.3% (7.0%–9.6%) | 2.1% (1.5%–2.9%) |
| Rare endpoint | Precision check | 17386 | 5.6% (4.6%–6.8%) | 1.5% (1.0%–2.2%) |
| Regular, 5 categories | Original | 1000 | 4.8% (3.9%–5.8%) | 1.0% (0.6%–1.5%) |
| Regular, 5 categories | Precision check | 20000 | 4.9% (4.0%–5.9%) | 0.9% (0.5%–1.4%) |

Intervals in this table condition on the reference and moment banks. The additional reference bank reduces its own tail-probability uncertainty substantially, but does not eliminate evaluation uncertainty. The endpoint 1% estimate remains somewhat high. These results support simulation calibration as a mechanism worth pursuing; the rank argument, rather than closeness of one observed rate to nominal size, establishes validity of the known-parameter oracle construction.

## Consequences for the next method

Keep the public proportional-odds assessment descriptive for now. Do not substitute the adjusted sensitivity, multiply the covariance by an empirical constant, or add a universal minimum-n rule. The next development candidate should compare the fixed-dimensional meanBR slope contrasts with a fitted-null simulation distribution and avoid a new covariance inversion for each simulated dataset.

NEXT-METHOD.md defines a concrete fixed-metric statistic and its regular-interior asymptotic argument, together with the unresolved nuisance-parameter and original-empty-category issues. In particular, simulations with empty categories must not be silently discarded or have their contrast dimension reduced. Unknown null parameters are the principal remaining obstacle to a rigorously calibrated sparse-sample test.

The present study concerns the development stacked-binary proportional-odds test. It does not invalidate the package's ordinal fitting, marginal-effects API, or existing separate bootstrap procedures.

## Reproduction

PROTOCOL.md, THEORY.md and manifest.rds define the main study. REFERENCE-CHECK.md and reference-manifest.rds define the additional precision check. run.R audits the original 36,000 datasets and regenerates its tables; reference-check.R audits and summarizes the additional 40,000. diagnose.R verifies independent design regeneration and exact replays and records support diagnostics. interpret.R generates this assessment and the figures. README.md gives complete commands. Raw-file hashes and report-input hashes record the complete evidence set.
