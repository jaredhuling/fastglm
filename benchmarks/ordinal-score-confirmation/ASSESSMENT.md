# Confirmation assessment: analytic proportional-odds score inference

The results support integrating the expected-information score test as an
optional part of ordinal assessment. It supplies deterministic joint and
selected-term inference with grouped frequencies and offsets. The default
assessment stays descriptive. The inferential claim remains a regular
first-order chi-square approximation, including when the original fitted
model used mean bias reduction. A usable ML null remains necessary.

## Independent confirmation

The frozen implementation was evaluated on 27,000 fresh datasets across
nine scenarios and sample sizes 100, 400, and 1600. Each dataset supplies
joint and selected-first-predictor tests, giving 54,000 records. The
parameter values and seed blocks differ from development. The method,
generator, protocol, source snapshot, and installed library were frozen
before generating confirmation outcomes. This evaluates the public package
function, rather than a harness-local substitute.

At nominal 5%, null rejection rates among available tests ranged from 3.6%
to 7.0%. The 7.0% joint result for correlated predictors and the 6.6%
selected-predictor result for a narrow interior category occurred at n=1600.
Their pointwise Monte Carlo intervals excluded 5%; they were therefore
examined in a separate precision study below. The original results remain
unchanged. Occasional pointwise departures across many comparisons do not
by themselves establish a persistent size distortion.

The table reports available datasets and rejection rates with 95% Wilson
Monte Carlo intervals. These intervals concern simulation precision, not
confidence intervals returned to an applied analyst. Joint and selected
tests had the same availability in this grid.

| Null scenario | Individual frequency | Available | Joint test | First predictor |
| --- | ---: | ---: | --- | --- |
| Balanced, four categories | 100 | 1000/1000 | 4.30% (3.21% to 5.74%) | 4.80% (3.64% to 6.31%) |
| Rare endpoint | 100 | 987/1000 | 5.67% (4.39% to 7.30%) | 5.47% (4.22% to 7.07%) |
| Narrow interior category | 100 | 997/1000 | 5.62% (4.35% to 7.22%) | 4.31% (3.22% to 5.76%) |
| Correlated predictors, six categories | 100 | 1000/1000 | 5.30% (4.07% to 6.87%) | 4.40% (3.29% to 5.86%) |
| Uncommon binary predictor | 100 | 884/1000 | 4.98% (3.73% to 6.62%) | 4.64% (3.44% to 6.23%) |
| High leverage | 100 | 1000/1000 | 5.10% (3.90% to 6.64%) | 5.00% (3.81% to 6.53%) |
| Grouped frequencies | 100 | 1000/1000 | 4.60% (3.47% to 6.08%) | 5.00% (3.81% to 6.53%) |
| Balanced, four categories | 400 | 1000/1000 | 5.60% (4.34% to 7.20%) | 5.20% (3.99% to 6.76%) |
| Rare endpoint | 400 | 1000/1000 | 5.50% (4.25% to 7.09%) | 5.80% (4.51% to 7.42%) |
| Narrow interior category | 400 | 1000/1000 | 3.60% (2.61% to 4.94%) | 3.80% (2.78% to 5.17%) |
| Correlated predictors, six categories | 400 | 1000/1000 | 4.90% (3.73% to 6.42%) | 5.30% (4.07% to 6.87%) |
| Uncommon binary predictor | 400 | 999/1000 | 5.31% (4.08% to 6.87%) | 5.21% (3.99% to 6.76%) |
| High leverage | 400 | 1000/1000 | 4.00% (2.95% to 5.40%) | 3.60% (2.61% to 4.94%) |
| Grouped frequencies | 400 | 1000/1000 | 4.50% (3.38% to 5.97%) | 5.30% (4.07% to 6.87%) |
| Balanced, four categories | 1600 | 1000/1000 | 4.40% (3.29% to 5.86%) | 4.80% (3.64% to 6.31%) |
| Rare endpoint | 1600 | 1000/1000 | 5.80% (4.51% to 7.42%) | 5.50% (4.25% to 7.09%) |
| Narrow interior category | 1600 | 1000/1000 | 4.70% (3.55% to 6.19%) | 6.60% (5.22% to 8.31%) |
| Correlated predictors, six categories | 1600 | 1000/1000 | 7.00% (5.58% to 8.75%) | 5.80% (4.51% to 7.42%) |
| Uncommon binary predictor | 1600 | 1000/1000 | 4.80% (3.64% to 6.31%) | 4.50% (3.38% to 5.97%) |
| High leverage | 1600 | 1000/1000 | 5.00% (3.81% to 6.53%) | 4.40% (3.29% to 5.86%) |
| Grouped frequencies | 1600 | 1000/1000 | 4.90% (3.73% to 6.42%) | 4.90% (3.73% to 6.42%) |

At nominal 1%, null rejection rates ranged from 0.10% to 1.72%. The
0.10% selected-predictor rate occurred in the balanced n=100 sample; its
Wilson interval was below 1%, indicating conservatism in that cell. The
1.72% rate occurred for the rare endpoint at n=100. The complete 1% results
and their Monte Carlo intervals are retained in `rates.csv`. The evidence
does not justify an exact-tail-accuracy claim, especially for small samples.

## Precision follow-up

The two elevated n=1600 comparisons motivated 5000 new replications in
each corresponding scenario, with the same frozen method and generator.
This is a post-confirmation diagnostic selected after seeing the first
results, not a replacement confirmation grid. No method choice changed.

| Scenario | Test | Available | Rejection at 5% (95% MC interval) |
| --- | --- | ---: | --- |
| Narrow interior category | joint | 5000/5000 | 4.80% (4.24% to 5.43%) |
| Correlated predictors, six categories | joint | 5000/5000 | 4.82% (4.26% to 5.45%) |
| Narrow interior category | x1 | 5000/5000 | 5.08% (4.51% to 5.72%) |
| Correlated predictors, six categories | x1 | 5000/5000 | 4.94% (4.37% to 5.58%) |

The previously elevated correlated-predictor joint rate was 4.82%
(95% MC interval 4.26% to 5.45%), and the narrow-interior selected rate was
5.08% (4.51% to 5.72%). All 10,000 new datasets yielded both tests. The
four 1% rates were 0.86% to 1.12%. This follow-up is consistent with Monte
Carlo variation in the two original elevated rates; it provides no evidence
of a persistent distortion of their original magnitude. Modest finite-sample
distortion elsewhere remains possible and is not a reason to add an empirical
correction or tune the reference distribution.

## Availability and power

There were 304 unavailable test records on 152 original datasets. At n=100,
the uncommon binary predictor produced 20 deficient designs and 96 failed
ML null fits out of 1000 datasets. Its tests were available on 884 datasets.
Conditional 5% rejection was 4.98% jointly and 4.64% for the first predictor.
The joint rejection bounds over all datasets were 4.4% to 16.0% when
unavailable decisions are counted as all nonrejections or all rejections.
These are missing-decision bounds, not proposed operating rules. Conditional
size alone would conceal this practical limitation. At n=400, one ML null
fit failed in the same scenario; availability was complete at n=1600.

The remaining failures were empty categories: 13 rare-endpoint null samples,
three narrow-interior null samples, and 19 rare-endpoint alternative samples,
all at n=100. No category was removed to manufacture an available test.
No other failure reason occurred in the confirmation grid. All failures
retain their original warnings, details, category totals, seeds, and records.

The alternatives change only the first predictor's slopes. Testing that
predictor alone avoids spending degrees of freedom on unchanged slopes and
has higher power in these examples. That is a consequence of the specified
alternative, not evidence that term selection after inspecting the data
would retain the reported error rate.

| Alternative | Individual frequency | Available | Joint power | First-predictor power |
| --- | ---: | ---: | --- | --- |
| Nonproportional, balanced | 100 | 1000/1000 | 13.90% (11.89% to 16.18%) | 19.60% (17.26% to 22.17%) |
| Nonproportional, rare endpoint | 100 | 981/1000 | 25.28% (22.66% to 28.09%) | 32.82% (29.96% to 35.82%) |
| Nonproportional, balanced | 400 | 1000/1000 | 54.70% (51.60% to 57.76%) | 67.10% (64.13% to 69.94%) |
| Nonproportional, rare endpoint | 400 | 1000/1000 | 76.70% (73.98% to 79.21%) | 85.60% (83.29% to 87.64%) |
| Nonproportional, balanced | 1600 | 1000/1000 | 99.70% (99.12% to 99.90%) | 100.00% (99.62% to 100.00%) |
| Nonproportional, rare endpoint | 1600 | 1000/1000 | 100.00% (99.62% to 100.00%) | 100.00% (99.62% to 100.00%) |

## Numerical validation and computation

The 40 targeted test assertions passed. They cover numerical full-likelihood
derivatives and expected information, grouped/expanded equivalence, offsets,
frequency doubling, outcome reversal, predictor scaling, factor and interaction
terms, prepared-design serialization, newdata, row exclusions, and explicit
unsupported cases. Twenty additional global production/prototype comparisons
agreed within 8.9e-15. The complete package test suite passed, with only the
optional Arrow test skipped because Arrow was unavailable.

The local source-package check compiled and installed successfully. It found
one existing Rd usage markup note, which was corrected. Rechecking the updated
documentation against the frozen installation returned Status: OK. Tests and
vignettes were excluded from those check invocations because they were run
separately; the updated ordinal vignette rendered successfully to HTML. This
is local validation, not a Windows or release check.

The following local medians use five paired repetitions with three predictors
and five categories. The score column times the entire public assessment,
including prediction scores, calibration summaries, the ML null refit, and
score accumulation. Initial model fitting is excluded. The incremental column
is the median paired difference and can differ from the difference of medians.
Timings were collected alongside other local work and are not isolated speed
benchmarks. The implementation does not expand grouped frequencies.

| Rows | Descriptive assessment (seconds) | With score test (seconds) | Increment (seconds) |
| ---: | ---: | ---: | ---: |
| 1000 | 0.002 | 0.005 | 0.004 |
| 10000 | 0.006 | 0.042 | 0.036 |
| 100000 | 0.058 | 0.376 | 0.319 |

## Decision and remaining scope

Keep the optional analytic score test and its explicit availability diagnostics.
The former inflated stacked-binary Wald tests remain development comparators
and are not exposed as formal inference. The earlier paired development
experiments explain why expected information was selected; the independent
confirmation provides new evidence on the frozen production implementation.

The main remaining limitation is inference when an ML null is unavailable
despite finite mean-bias-reduced estimates. A principled analytic test based
on adjusted scores would need its own nuisance treatment and distributional
justification. Simply substituting a bias-reduced null or an empirical
variance multiplier would not supply that justification. It is a separate
methods project, not a prerequisite for this explicitly scoped optional test.

Future package work can build on the established ordinal interface and
diagnostics. Partial proportional-odds fitting is a natural extension if its
parameterization and ordered-probability constraints are addressed explicitly.
The present score test does not fit that alternative model.
