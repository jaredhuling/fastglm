# Endpoint-empty meanBR bootstrap: exact intercept-only development study

## Question

The ordinal bootstrap currently withholds uncertainty summaries if an
original fit or any bootstrap refit has an empty category. That rule treats an
empty endpoint and an empty interior category alike. The implementation
retains empty endpoints under mean bias reduction (meanBR), while it removes
empty interior categories and ties thresholds. This study asks whether
allowing endpoint emptiness alone could recover useful bootstrap intervals.

## Method

The study uses a three-category cumulative-logit model with no predictors.
For this saturated intercept-only case, the meanBR thresholds are

\[
\widehat\alpha_j = \operatorname{logit}\left\{
  \frac{Y_{\leq j}+1/2}{n+1}\right\}, \quad j=1,2.
\]

This follows from the package's adjusted-score and expected-information
equations in `src/fit_ordinal.cpp` (Kosmidis, 2014). An empty endpoint still
gives finite values. An empty interior category collapses the fitted scale, so
those observations are excluded from threshold-vector inference.

For each generating parameter pair and sample size, the script enumerates all
three-category count vectors and weights them by their exact multinomial
probabilities. It computes ideal percentile limits from the exact parametric
bootstrap distribution, using inverse-CDF weighted quantiles. Two availability
rules are compared for `B = 199`:

* Current rule: the original sample and all 199 bootstrap samples have all
  three categories represented.
* Endpoint allowance: the original sample and all 199 bootstrap samples have
  the interior category represented; empty endpoints are retained under
  meanBR.

The exact availability calculation includes the probability of a valid
original sample and all 199 valid bootstrap draws. Coverage values average
over original samples with a represented interior category. “Coverage given
available” further conditions on the candidate availability event. Because
the bootstrap outcome space is enumerated, these figures have no Monte Carlo
error. They describe ideal infinite-refit percentile limits, not the finite
`R = 199` sample quantiles returned in practice.

## Results

| Generating thresholds | n | Current availability | Endpoint-allowance availability | Threshold 1 coverage | Threshold 2 coverage | Joint coverage | Joint coverage given available |
|---|---:|---:|---:|---:|---:|---:|---:|
| (-2.5, 1.0), rare endpoint | 10 | 0.0000001 | 0.8443 | 0.9651 | 0.9282 | 0.8967 | 0.9287 |
| (-2.5, 1.0), rare endpoint | 20 | 0.0084 | 0.9991 | 0.9852 | 0.9610 | 0.9468 | 0.9475 |
| (-2.5, 1.0), rare endpoint | 40 | 0.1454 | 1.0000 | 0.9281 | 0.9520 | 0.8845 | 0.8845 |
| (-2.0, -1.6), sparse interior | 10 | 0.0000000007 | 0.000138 | 0.9720 | 0.8621 | 0.8621 | 0.3293 |
| (-2.0, -1.6), sparse interior | 20 | 0.000070 | 0.00175 | 0.9098 | 0.9580 | 0.8730 | 0.6034 |
| (-2.0, -1.6), sparse interior | 40 | 0.0083 | 0.0230 | 0.9435 | 0.9552 | 0.9232 | 0.7433 |
| (-1.0, 1.0), moderate | 10 | 0.0000012 | 0.4514 | 0.9270 | 0.9270 | 0.8666 | 0.8448 |
| (-1.0, 1.0), moderate | 20 | 0.2708 | 0.9462 | 0.9608 | 0.9608 | 0.9259 | 0.9379 |
| (-1.0, 1.0), moderate | 40 | 0.9454 | 0.9999 | 0.9520 | 0.9520 | 0.9097 | 0.9097 |

An endpoint allowance substantially increases availability when the interior
category is well represented. For the rare-endpoint setting at `n = 10`, the
estimated availability rises from effectively zero to 84%; at `n = 20`, it
rises from 0.8% to 99.9%. The gains are small when the interior category is
sparse: at `n = 40`, candidate availability is only 2.3%. Coverage among
samples eligible for inference is 92.3% jointly, but it falls to 74.3% when
conditioning again on the rare event that all 199 bootstrap refits preserve
the interior category. This warns that the proposed eligibility condition
selects highly unusual bootstrap distributions.

Marginal coverage in the rare-endpoint case is near 95% at `n = 20`, but is
below nominal for one threshold at `n = 40`. This nonmonotone result and the
sparse-interior failures do not support a general relaxation. Joint coverage
is reported descriptively; the percentile intervals target marginal coverage
for each threshold, not simultaneous coverage of both.

## Interpretation and next step

The structural distinction is meaningful, and endpoint emptiness alone need
not change the meanBR parameter dimension. However, these results do not
establish validity for regression models, slope parameters, other links,
cluster or pairs bootstraps, or finite-refit percentile limits. In addition,
the current study's exact multinomial enumeration uses ideal bootstrap
quantiles; production uses a finite random sample of refits.

At the time of this exact intercept-only study, the evidence did not justify
changing the public rule. The subsequent regression study used the package's
actual fit and bootstrap path, included rare endpoints over increasing sample
sizes, and retained sparse-interior cases as a negative control. Its higher-
precision follow-up is documented in [`wave2/RESULTS.md`](wave2/RESULTS.md).
That evidence supports allowing endpoint-empty fits for meanBR parametric
bootstrap summaries while continuing to reject interior-empty original fits
and refits.

## References

Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*,
80(1), 27–38. <https://doi.org/10.1093/biomet/80.1.27>

Kosmidis, I. (2014). Improved estimation in cumulative link models. *Journal of
the Royal Statistical Society: Series B*, 76(1), 169–196.
<https://doi.org/10.1111/rssb.12025>
