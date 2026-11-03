# Wave 2 protocol: ordinal endpoint-bootstrap validation

## Aim

Increase precision for mean-bias-reduced (meanBR) parametric bootstrap
inference when an ordinal endpoint is empty, check response-scale symmetry,
and assess a two-predictor regression. The wave-1 sparse-interior scenario
remains the negative control; its results are preserved in the parent
development report. No public inference rule or nominal level will be tuned
from these results.

## Models and sample sizes

All scenarios use a three-category proportional-odds logit model,
\(\Pr(Y_i\le j\mid x_i)=\operatorname{logit}^{-1}(\alpha_j-x_i^T\beta)\).
For one predictor, `x1 = -1, 0, 1` is balanced at each sample size. For the
two-predictor case, all nine combinations of `x1, x2 = -1, 0, 1` are equally
represented.

| Scenario | Thresholds | Slopes | Predictors | n |
|---|---:|---:|---:|---:|
| Rare lower endpoint | (-4.0, 1.2) | (0.5) | 1 | 60, 120, 240 |
| Rare upper endpoint | (-1.2, 4.0) | (-0.5) | 1 | 60, 120, 240 |
| Moderate control | (-1.0, 1.0) | (0.5) | 1 | 60, 120, 240 |
| Rare lower endpoint, two predictors | (-4.0, 1.2) | (0.35, 0.35) | 2 | 90, 180, 360 |

The upper-endpoint scenario reverses category order and slope sign relative to
the lower-endpoint scenario. The two-predictor scenario uses balanced
factorial covariates. All models use meanBR and fixed declared category levels.

## Replication

There are 500 outer datasets per scenario/sample-size cell and 499 parametric
refits for each original fit that retains the interior category. The main
study has 12 cells (6,000 outer datasets). Data are generated from independent
uniform random variables with master seed `20261105`; deterministic seeds are
stored per outer dataset. The wave-1 seed and outputs are not reused.

## Paired inference rules

Both rules use each original fit and its same bootstrap refits:

1. Current rule: the original and all refits have every category represented,
   and all refits and statistics succeed.
2. Endpoint allowance: the original and all refits retain the interior
   category; empty endpoints are allowed under meanBR; all refits and
   statistics succeed.

Empty interior refits remain ineligible. Compare availability and 95%
type-7 percentile intervals for both thresholds and all slope coefficients.
Report conditional coverage only with its available denominator, Monte Carlo
standard error, median interval width, bias/empirical SD on original fits with
the interior category retained, directional misses, and all fit/refit
failures. Joint threshold coverage is descriptive, not a nominal target.

## Scope

This is a development evaluation of a fixed procedure. It covers parametric
bootstrap, a logit link, balanced fixed designs, and at most two predictors.
It does not evaluate pairs/cluster resampling, model misspecification, other
links, or simultaneous intervals. The wave-1 sparse-interior results remain
the negative control; no additional sparse-interior scenario is added here.
