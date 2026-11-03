# Assessment of the proportional-odds test candidate

The current chi-square test should remain a development prototype. It has substantial false-positive inflation in sparse endpoint samples and in some small samples with several slope contrasts. Mean bias reduction improves some coefficient behavior but does not consistently calibrate this test. The public graphical diagnostic should remain descriptive.

## Evidence and scope

The frozen development study completed 30,000 datasets in 30 cells, with 90,000 method records. Every dataset is represented by its seed and method outcomes, including unsuccessful results. The three methods were the empirical-score covariance at ML and meanBR binary fits, and the installed car::poTest() comparator. There were 1,000 datasets per scenario/sample-size cell. Results support decisions within this grid; they do not establish general calibration or a minimum safe sample size.

The table below reports the meanBR candidate's false-positive rate at nominal 5%, conditional on an available test. Intervals quantify Monte Carlo uncertainty. The complete comparison, including correlated predictors and five-category interior sparsity, is in RESULTS.md and summary.csv.

| Scenario | n | Available | False-positive rate | 95% MC interval |
|---|---:|---:|---:|---|
| regular3 | 80 | 1000/1000 | 5.6% | 4.3% to 7.2% |
| regular3 | 250 | 1000/1000 | 5.2% | 4.0% to 6.8% |
| regular3 | 1000 | 1000/1000 | 5.1% | 3.9% to 6.6% |
| regular5 | 80 | 1000/1000 | 12.0% | 10.1% to 14.2% |
| regular5 | 250 | 1000/1000 | 6.1% | 4.8% to 7.8% |
| regular5 | 1000 | 1000/1000 | 3.9% | 2.9% to 5.3% |
| rare_lower3 | 80 | 848/1000 | 24.1% | 21.3% to 27.0% |
| rare_lower3 | 250 | 997/1000 | 14.5% | 12.5% to 16.9% |
| rare_lower3 | 1000 | 1000/1000 | 5.5% | 4.2% to 7.1% |
| rare_upper3 | 80 | 852/1000 | 24.2% | 21.4% to 27.2% |
| rare_upper3 | 250 | 997/1000 | 14.4% | 12.4% to 16.8% |
| rare_upper3 | 1000 | 1000/1000 | 6.1% | 4.8% to 7.8% |
| rare_interior3 | 80 | 956/1000 | 1.9% | 1.2% to 3.0% |
| rare_interior3 | 250 | 1000/1000 | 4.1% | 3.0% to 5.5% |
| rare_interior3 | 1000 | 1000/1000 | 5.4% | 4.2% to 7.0% |

The endpoint failure appears for both outcome orders. Inflation also remains at the intermediate sample size, where almost all tests are available, so excluding empty-category datasets cannot explain it. The regular five-category result demonstrates a limitation beyond extreme category sparsity. Sparse interior categories produce a different pattern, including conservative behavior for the three-category case. A single inflation factor would not address these distinct behaviors.

Across the eight null scenarios at n=1000, meanBR rejection rates ranged from 3.9% to 6.1%. These results are encouraging for regular large-sample operation, but this grid cannot justify a universal n cutoff.

## Diagnosis

In the rare-lower-endpoint setting at n=80, the empirical standard deviation of the first x1 slope contrast was 1.572, while the square root of the mean estimated variance was 0.902. The corresponding interval coverage was 75.9%. This supports underestimated sampling variability as a component of the failure; it does not isolate all sources of poor approximation.

MeanBR reduced false-positive rejection relative to ML in that cell by 9.9 percentage points (paired Monte Carlo standard error 1.4 points), on the same 848 available datasets. That improvement leaves substantial inflation. In some other cells meanBR increased rejection slightly; coefficient bias reduction does not imply improvement in test calibration.

Post-study implementation checks replayed 30 meanBR datasets and rebuilt covariance from direct individual scores and a numerical score Jacobian. The maximum relative covariance discrepancy was 1.7e-09; statistic replays agreed within numerical precision. Together with the earlier grouped-frequency and outcome-reversal checks, this gives evidence against a score-product or normalization coding error in these cases.

Some rare-endpoint ML fits returned very large coefficients and fitted-probability warnings despite satisfying the optimizer's convergence flag. Such a flag does not establish the existence of a finite ML solution. These records were retained under the frozen eligibility rule. The meanBR failure is also present with finite estimates, so excluding problematic ML fits would not resolve the main finding.

The observed problem is compatible with poor finite-sample covariance and chi-square approximations when only a few observations identify a binary slope. It does not falsify the conditional asymptotic argument under a fixed, regular interior model. No covariance multiplier, category-count threshold, or nominal level was tuned after seeing these results.

## Power and comparator limits

Power increased with sample size under the prespecified nonparallel alternatives:

| Scenario | n | MeanBR rejection probability | 95% MC interval |
|---|---:|---:|---|
| alternative3 | 80 | 58.3% | 55.2% to 61.3% |
| alternative3 | 250 | 97.6% | 96.5% to 98.4% |
| alternative3 | 1000 | 100.0% | 99.6% to 100.0% |
| alternative5 | 80 | 23.1% | 20.6% to 25.8% |
| alternative5 | 250 | 53.1% | 50.0% to 56.2% |
| alternative5 | 1000 | 99.9% | 99.4% to 100.0% |

The elevated null rejection in the small five-category samples prevents interpreting their power as a calibrated advantage. The alternatives studied here use moderate category frequencies; they do not measure sparse-endpoint power.

The conventional car comparator was conservative in the sparse-endpoint cases, and it sometimes returned finite statistics when an endpoint was unobserved. Its interface does not expose all auxiliary convergence statuses. The stored diagnostics also show nonsymmetric contrast covariance matrices in some multipredictor cases for the installed implementation. Its results were preserved unchanged as an external reference, with these limitations recorded; it is not an infallible numerical reference. See comparator-support-diagnostics.csv and car-source.txt.

## Package decision and next experiment

Keep the public proportional-odds diagnostic descriptive. These findings concern the new stacked-score hypothesis-test prototype; they do not evaluate or invalidate the existing ordinal marginal-effects API or the separate endpoint-bootstrap procedure. No production fitting or inference code changed in this study.

The next bounded investigation is parametric calibration under a fitted proportional-odds null. It needs a discrepancy statistic that remains defined when simulated categories are empty, and an explicit rule for genuine refit failures. Simply bootstrapping the present covariance-inverted statistic can produce singular contrasts and unavailable results in exactly the sparse settings of interest. Derive that statistic and its regularity argument first, then compare a frozen version using fresh seeds. A fitted-null bootstrap would still require empirical calibration checks and would not be an exact finite-sample test.

No test has been promoted. Partial proportional-odds models remain a separate later extension.

## Reproduction and provenance

run.R audits the frozen method and scenario sources and the complete replication grid. summarize.R regenerates the tables and figures. diagnose.R performs the post-study numerical audit, and interpret.R generates this assessment from saved summaries. The raw-data hashes are recorded in results-manifest.txt. This assessment is a development finding, not an independent confirmation study.
