# Covariance for threshold-specific slope comparisons

This is a development assessment, dated November 3, 2026. The immediate purpose
is to determine whether the existing descriptive ordinal diagnostic can support
joint comparisons of slopes. No proportional-odds hypothesis test has been
added to the public API. The prototype evaluates a candidate covariance and
Wald statistic; it has no demonstrated small-sample calibration.

## Target and covariance

Let z_i contain an intercept and the location predictors, let c_ik be the
integer frequency in category k, and let m_i=sum_k c_ik. For threshold j, define
H_ij=sum_{k>j} c_ik and p_ij=logistic(z_i' theta_j + offset_i). Separate binary
fits estimate theta_j. The null hypothesis equates their slope components and
leaves their intercepts free.

An individual in category k contributes the ordinary binary score
u_ijk=z_i {I(k>j)-p_ij}. Stack these scores across thresholds. The sensitivity
matrix A is block diagonal, with A_j=sum_i m_i p_ij(1-p_ij) z_i z_i'. The
empirical score-product matrix has blocks

    B_jl = sum_i z_i z_i' [H_i,max(j,l) - p_il H_ij - p_ij H_il
                          + m_i p_ij p_il].

This identity follows by expanding the two residuals and using
I(k>j)I(k>l)=I(k>max(j,l)). It sums individual score products without expanding
frequency rows. Squaring a grouped row's aggregate score would estimate a
different covariance. The candidate covariance is V=A^{-1} B A^{-T}, using
unscaled sums consistently. Cross-threshold blocks are essential.

Let D compare each threshold's slopes with those of the first threshold.
The statistic is (D theta)' (D V D')^{-1} (D theta), with p(K-2) candidate
degrees of freedom. Intercepts enter the full covariance before selection of
slope contrasts. The prototype requires a positive definite contrast covariance
and stops on a failed factorization. It does not use a generalized inverse to
silently change the hypothesis.

## Conditional asymptotic justification

For a fixed number of categories and predictors, assume independent individuals,
correct binary logistic marginals under the null, a finite interior parameter,
convergent nonsingular normalized sensitivity and score-product matrices, and
a joint central limit theorem for the stacked scores. Also require consistency
and a valid local score expansion. For bias-reduced fits, the stacked adjustment
must be o_p(sqrt(N)), and its local derivative must be negligible relative to N.
Under these conditions the ordinary-score linearization applies to the
bias-reduced fits, and the Wald statistic has the usual limiting chi-square law
when the contrast covariance has full rank. A regular fixed-dimensional model
with bounded predictors, positive probabilities for each category, and full
design rank supplies a concrete candidate regime for these conditions.

This is a conditional argument, not a sparse-sample theorem. Finite Firth
estimates under separation do not verify the regularity assumptions. When a
category is empty, some binary outcomes coincide and contrast rank can fail.
The prototype requires every category to be represented. Cluster dependence,
survey weights, growing parameter dimension, and shrinking category
probabilities remain outside its scope. With misspecified binary marginals,
equal pseudo-true slopes need not characterize proportional odds.

The score-product identity is exact. Covariance consistency and the chi-square
reference are conditional asymptotic claims. Smoothness, independence, correct
marginals, and the limiting regime are assumptions, not guarantees from the
software. Convergence, finite inputs, positive category frequency, and matrix
factorizations can be checked in an implementation.

## Relationship to existing methods

Brant (1990), *Assessing proportionality in the proportional odds model for
ordinal logistic regression*, Biometrics 46, 1171–1178,
[doi:10.2307/2532457](https://doi.org/10.2307/2532457), is the closest antecedent.
Its [PubMed abstract](https://pubmed.ncbi.nlm.nih.gov/2085632/) describes comparing
correlated binary fits and constructing asymptotic measures. The original
full text was not obtained in this pass, so this prototype is not described as
an exact reproduction of Brant's statistic.

The [car implementation](https://rdrr.io/cran/car/src/R/poTest.R), inspected on
November 3, 2026, uses separate maximum-likelihood binary fits and a model-based
cross-threshold covariance. The proposed empirical score-product covariance
can differ in finite samples; numerical equality with `car::poTest()` is
therefore not an appropriate acceptance criterion.

Kosmidis (2014), *Improved estimation in cumulative link models*, JRSS B 76,
169–196, [doi:10.1111/rssb.12025](https://doi.org/10.1111/rssb.12025), discusses
first-order asymptotic inference for bias-reduced estimates in Section 4.2,
page 175, after equation (9). The published PDF is available through
[Warwick's repository](https://wrap.warwick.ac.uk/id/eprint/98828/7/WRAP-improved-estimation-cumulative-link-models-Kosmidis-2014.pdf).
Reading for this assessment covered Sections 1.1 and 4.2 and the discussion
of coverage on page 188. These passages support further investigation, but
do not establish calibration for the proposed stacked comparison.
The downloaded published PDF has SHA-256
`dd66691df64cc60ec764460dfff70f426195e69c6a749a6c434a5f206c9776e2`.
The covariance construction above is an adaptation of ordinary
estimating-equation inference, with no claim of methodological novelty.

## Numerical assessment and next decision

`prototype.R` implements grouped-frequency covariance and a joint statistic.
`validate.R` compares grouped and explicitly expanded representations, checks
the meat against direct individual score products, and checks reversal of the
outcome order under ML and mean bias reduction. `validation.txt` records the
results. These checks assess algebra and implementation, not rejection rates.

Before exposing p-values, freeze the candidate and assess null rejection rates,
power, rank failures, and convergence across moderate and sparse category
frequencies, several sample sizes, numbers of categories, and predictor designs.
Use common datasets for ML and meanBR comparisons and retain every failed fit.
Compare a conventional Brant implementation in regular cases and consider
parametric null calibration as a separate candidate. The next decision is
whether the first-order comparison is reliable enough to expose, or whether
joint slope differences should remain descriptive until calibration improves.
