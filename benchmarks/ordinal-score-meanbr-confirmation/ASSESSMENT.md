# Explicit null estimation and accessible test results

The work resolves two different problems. Shared print methods now make
model, summary, bootstrap, and test results readable without manually
selecting nested list components. Separately, a sufficient separation
diagnostic explains a set of unavailable ML score tests, and the explicitly
requested meanBR null supplies a first-order alternative in those samples.
The ML default, category scale, and prohibition on silent estimator switching
are retained. Inference is deterministic; simulation evaluates its behavior.

## Diagnosis of the previous failures

All 96 of the original n=100 ML nonconvergence cases had a two-valued
predictor with a group observed only in an endpoint category. There is an
explicit direction along which the likelihood keeps increasing, so no finite
ML estimate exists. This is not repaired by increasing the iteration limit.
The explicit meanBR option returned a test on all 96 of those datasets.
The remaining 20 original failures had no predictor variation. They remain
unsupported, with a direct explanation rather than a numerical failure.
The replay is developmental diagnosis; it does not estimate the new
procedure's rejection rate on independent samples. METHODS.md gives the
exact separation certificate and its limited detection scope.

## Fresh confirmation

The frozen public implementation was evaluated on 13,500 new datasets
across nine scenarios and n=80, 300, and 1200, giving 54,000 records for
two estimators and two tests per dataset. Methods, tuning and reference
distributions were fixed before generating outcomes. Both tests use all
declared categories; failed datasets are retained without replacement.

At nominal 5%, conditional null rejection ranged from 2.09% to 6.20% for meanBR,
and from 2.35% to 6.83% for ML. These ranges do not establish uniform calibration.
Small-sample conservatism is visible, especially with uncommon predictors.
The finite-sample approximation is not claimed to be exact; modest inflation
is compatible with the intended first-order scope.

At nominal 1%, both methods had cell rates ranging from 0% to 2.2%.
The narrow-interior joint test at n=300 rejected 11/500 datasets with both
methods (95% MC interval 1.23% to 3.90%). The selected test in that scenario
at n=1200 rejected 0/500 with both methods (interval 0% to 0.76%). These
pointwise departures are shared by the two estimators and do not isolate
a meanBR-specific issue. With 500 replications per cell, tail precision is
limited. The results do not support a uniform 1% tail-accuracy claim, and
the method was not tuned in response to these comparisons.

The tables give available datasets out of 500 and rejection percentages
with 95% Wilson Monte Carlo intervals. Intervals are pointwise across cells.

### Joint null tests

| Scenario | n | Available ML / meanBR | ML rejection | meanBR rejection |
| --- | ---: | ---: | --- | --- |
| balanced4 | 80 | 500 / 500 | 6.00% (4.23% to 8.44%) | 5.80% (4.07% to 8.21%) |
| endpoint4 | 80 | 483 / 483 | 4.55% (3.03% to 6.80%) | 3.73% (2.37% to 5.81%) |
| narrow_middle4 | 80 | 492 / 492 | 3.86% (2.49% to 5.95%) | 3.66% (2.33% to 5.71%) |
| correlated6 | 80 | 499 / 499 | 4.21% (2.77% to 6.35%) | 4.01% (2.61% to 6.11%) |
| sparse_binary4 | 80 | 383 / 478 | 2.35% (1.24% to 4.40%) | 2.09% (1.14% to 3.81%) |
| leverage4 | 80 | 500 / 500 | 4.00% (2.60% to 6.10%) | 3.80% (2.45% to 5.86%) |
| grouped4 | 80 | 500 / 500 | 4.80% (3.25% to 7.04%) | 4.80% (3.25% to 7.04%) |
| balanced4 | 300 | 500 / 500 | 4.80% (3.25% to 7.04%) | 4.60% (3.08% to 6.81%) |
| endpoint4 | 300 | 500 / 500 | 4.20% (2.76% to 6.34%) | 3.80% (2.45% to 5.86%) |
| narrow_middle4 | 300 | 500 / 500 | 6.00% (4.23% to 8.44%) | 5.80% (4.07% to 8.21%) |
| correlated6 | 300 | 500 / 500 | 6.20% (4.40% to 8.67%) | 5.80% (4.07% to 8.21%) |
| sparse_binary4 | 300 | 499 / 500 | 6.41% (4.58% to 8.91%) | 6.00% (4.23% to 8.44%) |
| leverage4 | 300 | 500 / 500 | 5.40% (3.74% to 7.74%) | 5.20% (3.57% to 7.51%) |
| grouped4 | 300 | 500 / 500 | 5.60% (3.90% to 7.97%) | 5.60% (3.90% to 7.97%) |
| balanced4 | 1200 | 500 / 500 | 5.00% (3.41% to 7.28%) | 5.00% (3.41% to 7.28%) |
| endpoint4 | 1200 | 500 / 500 | 5.00% (3.41% to 7.28%) | 4.80% (3.25% to 7.04%) |
| narrow_middle4 | 1200 | 500 / 500 | 4.00% (2.60% to 6.10%) | 4.00% (2.60% to 6.10%) |
| correlated6 | 1200 | 500 / 500 | 3.20% (1.98% to 5.13%) | 3.20% (1.98% to 5.13%) |
| sparse_binary4 | 1200 | 500 / 500 | 5.20% (3.57% to 7.51%) | 5.00% (3.41% to 7.28%) |
| leverage4 | 1200 | 500 / 500 | 5.20% (3.57% to 7.51%) | 5.00% (3.41% to 7.28%) |
| grouped4 | 1200 | 500 / 500 | 5.20% (3.57% to 7.51%) | 5.20% (3.57% to 7.51%) |

### Selected-first-predictor null tests

| Scenario | n | Available ML / meanBR | ML rejection | meanBR rejection |
| --- | ---: | ---: | --- | --- |
| balanced4 | 80 | 500 / 500 | 6.00% (4.23% to 8.44%) | 6.00% (4.23% to 8.44%) |
| endpoint4 | 80 | 483 / 483 | 6.83% (4.91% to 9.44%) | 5.38% (3.70% to 7.77%) |
| narrow_middle4 | 80 | 492 / 492 | 4.88% (3.30% to 7.16%) | 4.47% (2.97% to 6.68%) |
| correlated6 | 80 | 499 / 499 | 4.61% (3.09% to 6.82%) | 4.21% (2.77% to 6.35%) |
| sparse_binary4 | 80 | 383 / 478 | 2.35% (1.24% to 4.40%) | 2.51% (1.44% to 4.34%) |
| leverage4 | 80 | 500 / 500 | 4.80% (3.25% to 7.04%) | 4.00% (2.60% to 6.10%) |
| grouped4 | 80 | 500 / 500 | 5.40% (3.74% to 7.74%) | 5.40% (3.74% to 7.74%) |
| balanced4 | 300 | 500 / 500 | 4.40% (2.92% to 6.57%) | 4.40% (2.92% to 6.57%) |
| endpoint4 | 300 | 500 / 500 | 4.40% (2.92% to 6.57%) | 4.20% (2.76% to 6.34%) |
| narrow_middle4 | 300 | 500 / 500 | 6.20% (4.40% to 8.67%) | 6.20% (4.40% to 8.67%) |
| correlated6 | 300 | 500 / 500 | 3.60% (2.29% to 5.62%) | 3.60% (2.29% to 5.62%) |
| sparse_binary4 | 300 | 499 / 500 | 5.01% (3.42% to 7.29%) | 4.80% (3.25% to 7.04%) |
| leverage4 | 300 | 500 / 500 | 5.00% (3.41% to 7.28%) | 5.00% (3.41% to 7.28%) |
| grouped4 | 300 | 500 / 500 | 5.80% (4.07% to 8.21%) | 5.80% (4.07% to 8.21%) |
| balanced4 | 1200 | 500 / 500 | 4.40% (2.92% to 6.57%) | 4.40% (2.92% to 6.57%) |
| endpoint4 | 1200 | 500 / 500 | 4.80% (3.25% to 7.04%) | 4.80% (3.25% to 7.04%) |
| narrow_middle4 | 1200 | 500 / 500 | 3.80% (2.45% to 5.86%) | 3.80% (2.45% to 5.86%) |
| correlated6 | 1200 | 500 / 500 | 3.40% (2.13% to 5.38%) | 3.40% (2.13% to 5.38%) |
| sparse_binary4 | 1200 | 500 / 500 | 6.20% (4.40% to 8.67%) | 6.20% (4.40% to 8.67%) |
| leverage4 | 1200 | 500 / 500 | 5.40% (3.74% to 7.74%) | 5.00% (3.41% to 7.28%) |
| grouped4 | 1200 | 500 / 500 | 5.00% (3.41% to 7.28%) | 5.00% (3.41% to 7.28%) |

In the sparse-predictor n=80 cell, availability was 383/500 (76.6%) for ML
and 478/500 (95.6%) for meanBR. All 95 separated samples were recovered by
the explicit meanBR option; 22 datasets had deficient predictor rank. At
n=300, meanBR recovered one additional separated sample. It lost no test
available to ML in this grid. The remaining common failures were empty
categories, all at n=80: 17 rare-endpoint null samples, eight narrow-interior
null samples, one correlated-six-category sample, and 31 rare-endpoint
alternative samples. No other failure occurred in this confirmation grid.

Availability and rejection need separate interpretation. On the 383 shared
sparse-predictor samples at n=80, joint rejection was 2.35% for ML and 2.61%
for meanBR. Over all 478 available meanBR results it was 2.09%. The latter
rate reflects a different retained sample as well as a different statistic.
`paired.csv` retains every common-availability comparison; `rates.csv` also
contains 1% tails and bounds that count unavailable decisions as respectively
nonrejections or rejections. No automatic post-failure switch was evaluated.

### Power

The alternatives vary only the first predictor's slopes. Selecting that
predictor reduces the test's degrees of freedom and generally increases
power here; this does not justify choosing terms after inspecting p-values.

| Alternative | n | Available ML / meanBR | ML joint power | meanBR joint power |
| --- | ---: | ---: | --- | --- |
| alternative_balanced4 | 80 | 500 / 500 | 12.00% (9.44% to 15.14%) | 12.00% (9.44% to 15.14%) |
| alternative_endpoint4 | 80 | 469 / 469 | 21.96% (18.45% to 25.93%) | 16.84% (13.73% to 20.50%) |
| alternative_balanced4 | 300 | 500 / 500 | 42.40% (38.14% to 46.77%) | 42.40% (38.14% to 46.77%) |
| alternative_endpoint4 | 300 | 500 / 500 | 65.80% (61.54% to 69.82%) | 64.40% (60.11% to 68.47%) |
| alternative_balanced4 | 1200 | 500 / 500 | 98.40% (96.87% to 99.19%) | 98.40% (96.87% to 99.19%) |
| alternative_endpoint4 | 1200 | 500 / 500 | 100.00% (99.24% to 100.00%) | 100.00% (99.24% to 100.00%) |

| Alternative | n | Available ML / meanBR | ML selected power | meanBR selected power |
| --- | ---: | ---: | --- | --- |
| alternative_balanced4 | 80 | 500 / 500 | 14.40% (11.59% to 17.75%) | 13.80% (11.05% to 17.10%) |
| alternative_endpoint4 | 80 | 469 / 469 | 27.93% (24.06% to 32.16%) | 24.95% (21.24% to 29.06%) |
| alternative_balanced4 | 300 | 500 / 500 | 54.20% (49.82% to 58.52%) | 54.00% (49.62% to 58.32%) |
| alternative_endpoint4 | 300 | 500 / 500 | 75.20% (71.23% to 78.78%) | 74.40% (70.40% to 78.03%) |
| alternative_balanced4 | 1200 | 500 / 500 | 99.20% (97.96% to 99.69%) | 99.20% (97.96% to 99.69%) |
| alternative_endpoint4 | 1200 | 500 / 500 | 100.00% (99.24% to 100.00%) | 100.00% (99.24% to 100.00%) |

At n=80 in the rare-endpoint alternative, meanBR joint power was 16.84%,
compared with 21.96% for ML. Selected-term power was 24.95% versus 27.93%.
The availability gain under separation therefore does not establish general
inferential superiority. ML remains the default, while meanBR is an explicit
choice with clearly stated first-order and small-sample limitations.

## Package interface and validation

`fastglm_po_test(fit)` prints the test directly. The component returned by
`fastglm_ordinal_assess(..., po.test="score")` has the same class and print
method. `as.data.frame()` and optional `broom::tidy()` return full-precision
numeric columns; `details=TRUE` adds explanatory fields. Firth profile tests
share these conventions and retain `htest` inheritance. Model summaries
retain their numeric contents and, for GLMs, `summary.glm` inheritance.

Shared formatting covers ordinary GLMs, negative-binomial, hurdle,
zero-inflated, truncated, two-part, ordinal and streaming models, along with
bootstrap summaries and hypothesis tests. It standardizes headings, table
alignment, p-value display, precision controls, and fitting status without
changing estimates or inference through rounding.

The 60 new targeted assertions and the complete existing test suite passed.
Only the optional Arrow test was skipped because Arrow was not installed.
Validation includes numerical likelihood derivatives and expected information
at the meanBR root, nonzero ordinary nuisance scores, grouped expansion,
outcome reversal, separation certificates, preserved numeric data, S3 dispatch,
and reporting extraction. The numerical Hessian check uses an adequate finite-
difference step to avoid cancellation in the second derivatives.

The updated package installed successfully. The local package check against
that installation returned Status: OK, with tests and vignettes checked
separately. Both modified vignettes rendered to HTML in the vignettes folder.
No Windows or release submission was made.

## Interpretation and remaining scope

The new option improves usability for the diagnosed separated samples without
pretending that a finite ML solution exists. It does not fix absent predictor
information, empty declared categories, every form of separation detection,
or every possible numerical failure. Nor does it provide uniform tail accuracy
or a higher-order Firth test. Those remain distinct statistical questions.

The ordinary projected-score statistic at a bias-reduced null is a first-order
adaptation of established ingredients. Kosmidis (2014, Sections 4 and 12)
provides the null estimator's asymptotic properties and discusses a different
larger-model adjusted-score test. The present implementation does not claim
to implement that latter statistic. Its implementation and a comparison of
small-sample behavior would be a separate bounded extension, requiring the
full alternative-model adjustment and fresh evidence before public claims.
