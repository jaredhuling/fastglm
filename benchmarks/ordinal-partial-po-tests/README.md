# Term-level common-slope tests in partial ordinal models

This development validation starts from commit `17c4a8a`. It adds natural term
selection to the previously validated partial-model inference machinery. It
does not introduce a new estimator, calibration correction, or simulation
method. The package's scalability requirement is retained: grouped frequencies
are processed without expanding observations in the operating implementation.

## Hypothesis and parameter mapping

For `K` categories there are `q=K-1` cumulative thresholds. If a selected
predictor column has slopes `gamma_1,...,gamma_q`, its common-slope restriction
is `gamma_j-gamma_1=0`, for `j=2,...,q`. Each row of the generated contrast
matrix contains a one in the later threshold's coefficient position and a
minus one in the first threshold's position. Rows for distinct predictor
columns occupy disjoint coefficient positions. Consequently `s` varying
columns impose exactly `s*(K-2)` independent restrictions.

Formula terms select all their design columns, including complete factors and
interactions. Matrix fits select named design columns. Already-common columns
add no restrictions. Selecting a common-only term therefore returns an
unavailable test with zero degrees of freedom, not a test of zero effect.
Binary outcomes also have no equal-slope restrictions to test.

For a partial fit, the default selects all varying terms. Every unselected
varying effect remains unrestricted in the null and alternative. With
`by.term=TRUE`, each term-specific null restricts only that term; it requires a
separate null fit. The joint test constrains all selected varying terms. This
differs intentionally from the existing common-model score API, where the
shared null is the fully common model and each term-specific alternative
relaxes only its selected term. Existing common-model default behavior is
unchanged. Term-specific p-values are not adjusted for multiple testing.

## Score and likelihood-ratio conventions

The score path first fits the reduced null by ML or mean bias reduction,
according to `null.estimation`. Selected varying columns move into its common
design; other varying columns retain their threshold-specific effects. The
estimate is embedded in the original full partial parameterization by copying
each newly common slope into all its threshold positions. Thresholds and
unrestricted slopes retain their fitted values. Fitting-coordinate centering
and scaling are transformed consistently through the full model's `T` matrix.

Let `U` be the ordinary full-model likelihood score at that embedded estimate,
`I` its full-model expected information, and `B` a basis for the tangent space
of the equality restrictions in working coordinates. The calculation is

```
e = U - I B (B' I B)^(-1) B' U
statistic = e' I^(-1) e.
```

The nuisance projection is retained even at a converged reduced-null meanBR
estimate, because its ordinary nuisance score generally is not zero. This is
the established `fastglm_po_test()` null-estimation convention. It differs
from `fastglm_score_test(..., null.estimation="meanBR")`, which constrains the
full-model adjusted equations and uses the full-model adjusted score. Neither
the fitted root nor the statistic is silently interchanged between procedures.

For `method="lr"`, the full partial model is refitted by ML and the existing
constrained profile engine maximizes the ordinary likelihood under the equality
contrasts. The statistic is twice the full-minus-null log likelihood. Both fits
must converge; both diagnostic records are retained. Supplying an original
meanBR or nonconverged fit does not waive the ML refits. A meanBR LR null is
rejected. The new LR option is limited to partial-model objects; common-model
objects retain the established score interface.

## Scientific conditions and numerical failure contract

Contrast rank, reduced-model embedding, and nuisance projection are algebraic
identities. The chi-square reference is conditional on a correctly specified,
identifiable interior null, fixed parameter dimension, nonsingular limiting
information, consistent nuisance estimates, and the usual independent ordinal
or integer-frequency score central limit theorem. A bounded design with
strictly positive category probabilities and increasing integer cell counts
provides a concrete nondegenerate regime. A smooth order-one mean-bias
adjustment changes a regular null estimate by order `1/N`; its first-order
effect lies in the nuisance tangent space and is removed by the projection.
This establishes only the usual first-order reference under those assumptions,
not a finite-sample or higher-order calibration claim.

All native probability-domain and information checks remain active. Declared
categories are not dropped. Missing and zero-frequency assessment rows are
excluded under the existing API. Missing categories, rank deficiency, a
sufficient endpoint-separation certificate, unsuccessful null/full fits, or
unavailable information produce an unavailable result with an explanation.
The two-valued-predictor separation certificate also applies when the slope
is represented by equal shifts to all of its threshold-specific coefficients.
It is sufficient, not a complete separation detector. No alternative
estimator, degrees-of-freedom reduction, generalized inverse, or Wald fallback
is substituted after failure. Cluster dependence, survey/IPW inference,
boundary-null calibration, and increasing parameter dimension are not covered.

Source-backed objects require explicit evaluation data for these tests. Once
supplied, the dense evaluation design is refitted under the same conventions;
the source's retained coefficients are not used as a substitute for that fit.
Assessment can request the partial score test together with prediction losses.
The descriptive binary-slope comparison remains limited to common-slope fits.

## Validation and primary provenance

`tests/testthat/test-ordinal-partial-po-test.R` checks all three links against
explicit constrained score and LR contrasts and separately fitted reduced ML
models. It verifies reduced-null meanBR estimates, nonzero ordinary nuisance
scores and their projection, factors, interactions, term-specific nuisance
freedom, grouped/expanded equivalence, frequency scaling, offsets, affine
recoding, missingness, unavailable results, source evaluation data, and shared
printing/data-frame extraction. Existing common-model PO and partial
assessment regression tests are also run. These are deterministic development
and regression checks, not calibration evidence. No timing claim is made here.

The inference foundations and earlier independent confirmation are retained in
[`../ordinal-partial-inference/methods.txt`](../ordinal-partial-inference/methods.txt).
The nearest primary ordinal source is [Kosmidis (2014), *Improved estimation in
cumulative link models*, JRSS B 76(1), 169–196](https://doi.org/10.1111/rssb.12025).
The published version in the [Warwick repository](https://wrap.warwick.ac.uk/id/eprint/98828/7/WRAP-improved-estimation-cumulative-link-models-Kosmidis-2014.pdf)
was inspected on 2026-11-04: Section 11 gives equality contrasts between
threshold-specific slopes; Section 12 discusses adjusted scores and their
first-order motivation. The paper's adjusted-score statistic is not the
ordinary projected-score convention used here. The present change is an API
and parameter-mapping adaptation of established methods, with no novelty or
small-sample improvement claim.

From the repository root, with the compiled package library available:

```r
pkgload::load_all(".", compile = FALSE)
testthat::test_dir("tests/testthat",
  filter = "ordinal-partial-po-test|ordinal-partial-integration|ordinal-po-test|ordinal-assessment",
  load_package = "none")
```
