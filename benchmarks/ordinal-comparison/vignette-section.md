# Ordinal models

Unlike the historical sections above, these times come from a run on 2026-11-06 (UTC) of *fastglm* commit `7f2ae40`, created by `benchmarks/benchmark-ordinal-comparison.R` in the package repository. Each design has 10 predictors, and each time includes coefficient covariance. The ratio is the comparison time divided by the *fastglm* time, shown only when that fit converged and agreed with *fastglm*.

|Link    |Observations | Categories|Method                       |  Seconds| Time ratio|
|:-------|:------------|----------:|:----------------------------|--------:|----------:|
|logit   |100,000      |          5|`fastglm`                    |   0.0673|      1.000|
|logit   |100,000      |          5|`rms::orm`                   |   0.1960|      2.910|
|logit   |100,000      |          5|`ordinal::clm`               |   0.6950|     10.300|
|logit   |100,000      |          5|`MASS::polr`                 |   1.3900|     20.700|
|logit   |100,000      |          5|`VGAM::vglm`                 |   1.7800|     26.500|
|logit   |100,000      |          5|`WeightIt::ordinal_weightit` |   3.0200|     44.900|
|probit  |100,000      |          5|`fastglm`                    |   0.1000|      1.000|
|probit  |100,000      |          5|`rms::orm`                   |   0.2190|      2.180|
|probit  |100,000      |          5|`ordinal::clm`               |   0.6880|      6.870|
|probit  |100,000      |          5|`MASS::polr`                 |   1.7900|     17.900|
|probit  |100,000      |          5|`VGAM::vglm`                 |   1.9200|     19.200|
|probit  |100,000      |          5|`WeightIt::ordinal_weightit` |   6.3000|     62.900|
|cloglog |100,000      |          5|`fastglm`                    |   0.0984|      1.000|
|cloglog |100,000      |          5|`ordinal::clm`               |   0.7470|      7.590|
|cloglog |100,000      |          5|`MASS::polr`                 |   1.4000|     14.300|
|cloglog |100,000      |          5|`VGAM::vglm`                 |   6.5100|         NA|
|cloglog |100,000      |          5|`WeightIt::ordinal_weightit` |   5.7800|     58.800|
|logit   |20,000       |         30|`fastglm`                    |   0.0169|      1.000|
|logit   |20,000       |         30|`rms::orm`                   |   0.0423|      2.510|
|logit   |20,000       |         30|`ordinal::clm`               |   0.6430|     38.000|
|logit   |20,000       |         30|`MASS::polr`                 |   0.8880|     52.600|
|logit   |20,000       |         30|`VGAM::vglm`                 |   5.3300|    315.000|
|logit   |20,000       |         30|`WeightIt::ordinal_weightit` |   2.3500|    139.000|
|logit   |20,000       |        300|`fastglm`                    |   0.0959|      1.000|
|logit   |20,000       |        300|`rms::orm`                   |   0.0472|      0.492|
|logit   |20,000       |        300|`ordinal::clm`               |  40.5000|    422.000|
|logit   |20,000       |        300|`MASS::polr`                 |  29.1000|    304.000|
|logit   |20,000       |        300|`WeightIt::ordinal_weightit` |  48.6000|    507.000|
|logit   |20,000       |       1000|`fastglm`                    |   0.3150|      1.000|
|logit   |20,000       |       1000|`rms::orm`                   |   0.0878|      0.279|
|logit   |20,000       |       1000|`ordinal::clm`               | 484.0000|   1540.000|
|logit   |20,000       |       1000|`MASS::polr`                 | 326.0000|   1030.000|
|logit   |20,000       |       1000|`WeightIt::ordinal_weightit` | 334.0000|   1060.000|

`rms::orm` is designed for outcomes with very many distinct values and is the faster choice at 300 or more categories. `VGAM::vglm` was limited to 30 categories by memory. For cloglog, `VGAM::vglm` warned about intersecting predictors and returned a different fit, so it has no ratio.

