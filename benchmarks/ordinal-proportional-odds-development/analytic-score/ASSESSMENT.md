# Analytic inference for the proportional-odds diagnostic

The expected-information score test is the most promising next candidate. It substantially improves false-positive control in the development comparisons, requires one null-model fit, and needs no bootstrap or simulation when analyzing a dataset. A mean-bias-reduced null fit is not uniformly better for this test. Production remains unchanged pending a separate confirmation study and a complete public-interface specification.

## What changed the diagnosis

A score test asks whether the fitted proportional-odds model leaves evidence of threshold-specific slope changes. It evaluates that evidence at the common-slope null model, avoiding the unstable separate binary slope estimates used by the earlier Wald prototype. Its uncertainty is computed from the ordinal likelihood.

The information matrix is a second consequential choice. Expected information averages likelihood curvature over possible outcomes under the fitted model. Observed information uses the outcomes actually seen. These are both established score-test constructions under regular asymptotics; they need not behave alike in small samples. Our candidate used expected information before any rejection results were inspected.

A post-study replay held each ML null fit and its ordinary score fixed, changing only the information matrix. Results at nominal 5%, n=80, are below. All comparisons use shared available datasets; they are targeted development diagnoses, not independent confirmation.

| Scenario | Expected information | Observed information | Shared datasets |
|---|---:|---:|---:|
| regular5 | 4.1% (3.0%–5.5%) | 12.1% (10.2%–14.3%) | 1000 |
| rare_lower3 | 4.2% (3.1%–5.8%) | 4.3% (3.2%–5.9%) | 852 |
| many_predictors5 | 5.3% (4.1%–6.9%) | 37.1% (34.2%–40.1%) | 1000 |
| leverage3 | 3.6% (2.6%–4.9%) | 7.7% (6.2%–9.5%) | 1000 |

An external numerical check used the published SAS Cheese example. Our expected-information statistic is 20.937936. An observed-Hessian generalized-inverse calculation gives 17.286815, compared with the published 17.2866. The observed matrix has rank 28 of 32; the expected matrix is nonsingular. This supports the information-choice explanation for that example, without claiming access to SAS internals. The generalized inverse was used only for this external diagnosis, never for a candidate test. INFORMATION-AUDIT.md records the derivation and source locations.

## Main development comparison

The main frozen study completed 30,000 datasets, comparing three methods on identical inputs. There are 1,000 datasets in each of ten scenarios at n=80,250,1000. The ordinary and meanBR-root score calculations use the same expected information and nuisance projection. All methods decline the regular full-dimensional test when an original category is empty. Intervals below are 95% Wilson Monte Carlo intervals; rates condition on availability.

The n=80 null cases show the mechanism clearly. Availability entries are ordered Wald; score at ML; score at meanBR.

| Scenario | Earlier binary meanBR Wald | Expected score at ML | Expected score at meanBR | Available / attempted |
|---|---:|---:|---:|---|
| regular3 | 3.9% (2.9%–5.3%) | 4.3% (3.2%–5.7%) | 4.3% (3.2%–5.7%) | 1000/1000; 1000/1000; 1000/1000 |
| regular5 | 11.3% (9.5%–13.4%) | 4.1% (3.0%–5.5%) | 3.5% (2.5%–4.8%) | 1000/1000; 1000/1000; 1000/1000 |
| correlated5 | 8.9% (7.3%–10.8%) | 5.0% (3.8%–6.5%) | 4.6% (3.5%–6.1%) | 1000/1000; 1000/1000; 1000/1000 |
| rare_lower3 | 23.6% (20.9%–26.6%) | 4.2% (3.1%–5.8%) | 2.3% (1.5%–3.6%) | 852/1000; 852/1000; 852/1000 |
| rare_upper3 | 24.1% (21.4%–27.1%) | 2.9% (1.9%–4.2%) | 1.3% (0.7%–2.3%) | 841/1000; 841/1000; 841/1000 |
| rare_interior3 | 0.9% (0.5%–1.8%) | 5.3% (4.1%–6.9%) | 5.5% (4.2%–7.1%) | 972/1000; 972/1000; 972/1000 |
| rare_interior5 | 6.6% (5.2%–8.4%) | 3.7% (2.7%–5.1%) | 3.4% (2.4%–4.7%) | 950/1000; 950/1000; 950/1000 |
| strong_signal3 | 5.1% (3.9%–6.6%) | 4.8% (3.6%–6.3%) | 3.8% (2.8%–5.2%) | 1000/1000; 1000/1000; 1000/1000 |

The score approach improves the sparse-interior case as well as the endpoint cases: it avoids the earlier test's strong conservatism when an interior category is rare. At nominal 1%, some score comparisons remain conservative. MeanBR-root scoring is particularly conservative for rare endpoints, so coefficient bias reduction does not supply an automatic inferential improvement.

The meanBR-root calculation explicitly projects out the ordinary nuisance score, which is nonzero at a meanBR estimate. Omitting this projection adds a nonnegative nuisance-estimation component that is not evidence of nonparallel slopes. This distinction is algebraically verified. The meanBR-root candidate has a first-order argument only; it is not labeled a higher-order Firth test.

## Power

The alternatives are the two coherent nonparallel models fixed in the earlier study. These power comparisons are developmental. In particular, the earlier Wald test's higher rejection in the small five-category alternative comes with inflated null rejection and cannot be interpreted as a calibrated power advantage.

| Alternative and sample size | Earlier Wald | Expected score at ML | Expected score at meanBR |
|---|---:|---:|---:|
| alternative3, n=80 | 61.1% | 62.2% | 62.1% |
| alternative3, n=250 | 97.4% | 97.3% | 97.3% |
| alternative3, n=1000 | 100.0% | 100.0% | 100.0% |
| alternative5, n=80 | 23.8% | 15.0% | 13.8% |
| alternative5, n=250 | 52.0% | 50.5% | 49.9% |
| alternative5, n=1000 | 99.8% | 99.8% | 99.8% |

## Separate stress comparison

After the main results, a fixed additional comparison added 8,000 datasets with normal predictors, eight predictors, uncommon binary predictor levels, and occasional high-leverage observations. Its protocol was recorded before generating these data. No method was changed, and no correction or eligibility cutoff was tuned. The n=80 results are:

| Scenario | Earlier binary meanBR Wald | Expected score at ML | Expected score at meanBR | Available / attempted |
|---|---:|---:|---:|---|
| normal5 | 13.2% (11.2%–15.4%) | 5.5% (4.2%–7.1%) | 5.2% (4.0%–6.8%) | 1000/1000; 1000/1000; 1000/1000 |
| many_predictors5 | 38.6% (35.6%–41.7%) | 5.3% (4.1%–6.9%) | 4.8% (3.6%–6.3%) | 1000/1000; 1000/1000; 1000/1000 |
| uncommon_binary5 | 78.5% (75.8%–81.0%) | 4.6% (3.5%–6.2%) | 3.7% (2.7%–5.1%) | 978/1000; 930/1000; 978/1000 |
| leverage3 | 41.0% (38.0%–44.1%) | 3.6% (2.6%–4.9%) | 2.0% (1.3%–3.2%) | 999/1000; 1000/1000; 929/1000 |

At n=250 the expected-information score rates remained near or below nominal in this panel; the complete tables include both nominal levels. These successes are evidence for further work, not a finite-sample theorem or proof of uniform calibration. The uncommon-binary ML score is available on only 930/1000 n=80 datasets: 22 have deficient predictor rank and 48 fail null convergence. The meanBR fit improves availability there, but fails convergence in 71/1000 high-leverage n=80 datasets. No silent fallback between methods was used.

These convergence failures need separate investigation. Availability and unconditional rejection bounds are recorded alongside conditional rates. A favorable rate on successful fits cannot establish reliability on the missing cases. The expected-information construction should therefore receive a clear failure result when its required fit is unavailable.

## Computational evidence

The implementation accumulates category scores and information in blocks. It stores no observation-by-observation matrix and fits no unrestricted ordinal alternative. The representative timing exercise below used three predictors and five categories; entries are medians of three repetitions on one dataset per size. These are local prototype timings, not a cross-platform performance guarantee.

| n | Null estimate | Fit time (s) | Score calculation (s) | Total (s) |
|---:|---|---:|---:|---:|
| 10000 | ML | 0.014 | 0.017 | 0.031 |
| 10000 | meanBR | 0.026 | 0.016 | 0.041 |
| 100000 | ML | 0.148 | 0.167 | 0.317 |
| 100000 | meanBR | 0.273 | 0.162 | 0.428 |

## Decision and remaining work

Continue with the expected-information score at an ML null fit as the primary analytic candidate. It is the simplest supported route in these results. Preserve the meanBR-root variant as a labeled development comparison; do not automatically substitute it when ML fails. No new Bartlett factor, adjusted degrees of freedom, or simulation-calibrated threshold is justified by this study.

Before public integration, freeze an independent confirmation panel that includes sparse-alternative power and fit failures, extend and validate grouped-frequency/offset/formula-term handling, and specify which original fits support testing. An assessment of a meanBR-fitted model must state explicitly whether it refits the ML null for the test. Empty categories and deficient design rank must have explicit outcomes. If integrated, document the test naturally in the ordinal vignette, including its distinction from the descriptive binary-slope plots.

The exact score and information identities are checked; asymptotic chi-square validity is conditional on the regularity stated in METHODS.md. Finite-sample uniform validity, behavior outside the studied designs, and higher-order accuracy remain open. The score method and information choices are established ingredients; this work contributes implementation and development evidence, without claiming a new statistical test.

The earlier simulation-calibration proposal remains archived as a diagnostic alternative. It is not the active direction after the user's preference for analytic methods. No public package function, version, vignette or inference default changed in this pass.

## Audit and reproduction

The evidence comprises 38,000 distinct development datasets, 114,000 main/stress method records, and 4,000 post-study replay records. Frozen source/install manifests, complete-grid audits, analytic/numerical identities, outcome reversal, affine predictor invariance, block-size checks and the external-example diagnosis are retained. README.md gives the dispatcher commands. interpret.R generates this assessment directly from saved summaries; provenance.csv records source and result hashes. Rebuilding in a relocated directory is checked separately.
