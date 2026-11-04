# Partial ordinal covariance and marginal-effect validation

This development audit extends established sandwich and delta-method calculations
to the full partial cumulative-link parameter vector. It is an implementation
validation, not a coverage study or a claim of a new inferential method. The
baseline is commit `351d808`. Fast computation with bounded row blocks is a
package requirement: these calculations retain matrices of order `d x d` and
`block.size x d`, rather than an observations-by-categories-by-parameters array.
Cluster aggregation additionally retains one score vector per observed cluster.

## Parameterization and exact derivatives

Let `eta_j = alpha_j - x beta - z gamma_j - offset`, with `r_j` its derivative
with respect to all working coefficients. The vector `r_j` contains a one in
threshold position `j`, `-x` in the common-slope positions, and `-z` in the
threshold-specific block `j`. Let `F`, `f`, and `h = f'/f` denote the inverse
link, its density, and the density's logarithmic derivative. An interior
category bounded by `l = eta_(k-1)` and `u = eta_k` has probability
`P = F(u) - F(l)` and ordinary score

```
a = f(u)/P; b = f(l)/P; t = a-b
score = a*r_upper - b*r_lower
H = h_upper*r_upper*r_upper' + h_lower*r_lower*r_lower'
    + shared*(r_upper-r_lower)*(r_upper-r_lower)'
h_upper = a*(t-h(u)); h_lower = b*(h(l)-t); shared = a*b.
```

Here `H` is the observed negative Hessian, not expected Fisher information.
The formula follows by differentiating `log(P)` twice. Common-slope columns
cancel from the last direction exactly. Common-slope scores use `-x*t`, with
`t` evaluated directly, avoiding cancellation between large threshold scores.
Endpoints retain the single applicable threshold term.

For logit, the equivalent factors are `h_upper=F(u)*(1-F(u))`,
`h_lower=F(l)*(1-F(l))`, and `shared=v*(1+v)`, where
`v=1/expm1(u-l)`. For complementary log-log, write
`D=exp(u)*(1-exp(l-u))`. Then
`h_upper=a*(D/(1-exp(-D))-1)` and
`h_lower=b*(1-D/expm1(D))`. Small-`D` power series and explicit limiting
values avoid cancellation and `0*Inf`. Probit uses the density log-ratio
`-(u-l)*(u+l)/2` to calculate `t` and positive truncated-normal endpoint
distances to calculate the Hessian factors. Its numerical screen rejects
endpoint distances unresolved at the scale of the preceding arithmetic.
The screen is a roundoff diagnostic, not certified interval arithmetic.

Let `m_ik` be effective category frequency, `N=sum(m_ik)`, and `s_ik` the
ordinary category score. Independent-observation covariance is

```
B = inverse(sum_ik(m_ik * H_ik))
M = sum_ik(m_ik * s_ik * s_ik')
V = B * M * B
```

For clusters, first sum `m_ik*s_ik` within each cluster and then take outer
products. The implementation divides observed information and independent
meat by `N`; clustered score sums are divided by `N` before squaring. The
final scaling restores exactly the expressions above. HC1 and cluster factors
retain the existing package conventions. The entire working covariance is
transformed with `T V T'`; threshold/common/varying cross-covariances are retained.

For a continuous variable, let `s_j = dx*beta + dz*gamma_j`, where `dx` and `dz`
are derivatives of the design columns with respect to that variable. The
cumulative marginal effect and its full parameter gradient are

```
effect_j = -f(eta_j) * s_j
gradient_j = -f'(eta_j)*s_j*r_j - f(eta_j)*derivative(s_j).
```

Category effects and gradients are adjacent differences, including zero
endpoints. Weighted averages hold the evaluation covariates and averaging
weights fixed. Formula design derivatives retain the prediction basis and use
centered differences; matrix-column derivatives are exact. Offsets remain fixed.

## Assumptions, guarantees, and failures

The derivative and frequency-aggregation identities above are finite-sample
algebra. First-order inference additionally assumes identifiable interior
parameters, a fixed parameter dimension, nonsingular limiting ordinary-score
derivative, appropriate moment/central-limit conditions, and independent
observations or independent clusters. These sampling assumptions are supplied
by the analyst; numerical convergence does not verify them. Public ordinal
fitting requires integer effective category counts, representing repeated
individuals. Fractional raw weights can be combined with grouped counts when
their products satisfy that requirement. The algebra does not authorize survey
weighting or inverse-probability-weighted inference. An observed frequency row
belongs entirely to its supplied cluster.

For mean bias reduction, covariance is the ordinary-score sandwich evaluated at
the adjusted estimate. It does not differentiate the global adjustment. Its
first-order interpretation requires that adjustment to be negligible on the
square-root sample-size scale. This does not establish bias reduction under
misspecification or finite-sample calibration. HC1 and cluster multipliers are
conventional finite-sample factors, not corrections chosen from this audit.

Public calculations reject nonconvergence, empty-category boundaries, singular
or indefinite observed bread, unsupported HC2/HC3 corrections, fewer than two
positive-frequency clusters, invalid prediction domains, and nonfinite or
numerically unresolved curvature. Source fits do not yet provide robust
covariance. No generalized inverse, ridge, probability clipping, sign repair,
or substitution of expected information is applied.

## Reproducible checks

The focused tests are `tests/testthat/test-ordinal-partial-robust.R` and
`tests/testthat/test-ordinal-partial-marginal-effects.R`. They compare category
scores, observed Hessians, individual meat, and cluster sums with independent
numerical derivatives; grouped and expanded data; ML and meanBR; all three
links; full affine covariance transformations; missing rows and cluster labels;
and marginal-effect standard errors under supplied full cluster covariance.
Marginal-effect parameter derivatives use Richardson differences with a larger
starting step than the inner probability difference, avoiding nested rounding
noise. This oracle change does not alter production methods or tolerances.

Tail cases include logit gaps of `1e-8`, complementary log-log predictors at
`-800` and `800`, probit predictors of magnitude 40, and explicit rejection of
unresolved probit curvature at magnitude `1e6` or an extreme narrow interval.
`normal-tail-reference.py` reproduces the independent high-precision probit
score and curvature fixture using a continued fraction. It is test reference
code, not an operating-method dependency.

Run the focused and existing common-slope regressions with an installed native
library available to `pkgload`:

```r
pkgload::load_all(".", compile = FALSE)
testthat::test_dir("tests/testthat",
  filter = "ordinal-partial-robust|ordinal-partial-marginal-effects|ordinal-inference|ordinal-marginal-effects",
  load_package = "none")
```

## Primary reference provenance

The sandwich construction is established M-estimation machinery, adapted here
to this joint ordinal parameterization. The author's current version of
[Zeileis (2006), *Object-oriented Computation of Sandwich Estimators*,
JSS 16(9), 1–16](https://doi.org/10.18637/jss.v016.i09) was read in Sections 2
and 4, equations (3)–(10). That downloadable version identifies itself as
slightly modified from the journal article; journal metadata were checked at
the DOI record. It states ordinary-score Jacobian and outer-product conventions
and does not supply an ordinal meanBR-specific theorem.

For clustering, Sections 2 and 3, equations (11)–(15), of
[Zeileis, Köll, and Graham (2020), *Various Versatile Variances: An
Object-Oriented Implementation of Clustered Covariances in R*, JSS 95(1),
1–36](https://doi.org/10.18637/jss.v095.i01) give cluster-summed scores and
the conventional correction factors. The authors' downloadable version labels
itself slightly modified from the 2020 journal article; that version and journal
metadata were checked on 2026-11-04. This implementation supports one-way
clustering only.

The independent normal-tail reference uses
[NIST DLMF 7.9.1](https://dlmf.nist.gov/7.9.E1), read on 2026-11-04
(DLMF version 1.2.8), after converting the complementary-error-function
continued fraction to the normal inverse Mills ratio. No novelty claim,
coverage simulation, or coverage-driven adjustment is part of this extension.
