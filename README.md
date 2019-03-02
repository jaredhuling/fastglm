
[![Build
Status](https://travis-ci.org/jaredhuling/fastglm.svg?branch=master)](https://travis-ci.org/jaredhuling/fastglm)

# Overview of ‘fastglm’

The ‘fastglm’ package is a re-write of `glm()` using `RcppEigen`
designed to be computationally efficient.

# Installing the ‘fastglm’ package

Install the development version using the **devtools** package:

``` r
devtools::install_github("jaredhuling/fastglm")
```

or by cloning and building using `R CMD INSTALL`

# Quick Usage Overview

Load the package:

``` r
library(fastglm)
```

A (not comprehensive) comparison with `glm.fit()` and `speedglm.wfit()`:

``` r
library(speedglm)
library(microbenchmark)
library(ggplot2)

set.seed(123)
n.obs  <- 10000
n.vars <- 100
x <- matrix(rnorm(n.obs * n.vars, sd = 3), n.obs, n.vars)
Sigma <- 0.99 ^ abs(outer(1:n.vars, 1:n.vars, FUN = "-"))
x <- MASS::mvrnorm(n.obs, mu = runif(n.vars, min = -1), Sigma = Sigma)

y <- 1 * ( drop(x[,1:25] %*% runif(25, min = -0.1, max = 0.10)) > rnorm(n.obs))

ct <- microbenchmark(
    glm.fit = {gl1 <- glm.fit(x, y, family = binomial())},
    speedglm.eigen  = {sg1 <- speedglm.wfit(y, x, intercept = FALSE,
                                            family = binomial())},
    speedglm.chol   = {sg2 <- speedglm.wfit(y, x, intercept = FALSE, 
                                            family = binomial(), method = "Chol")},
    speedglm.qr     = {sg3 <- speedglm.wfit(y, x, intercept = FALSE,
                                            family = binomial(), method = "qr")},
    fastglm.qr.cpiv = {gf1 <- fastglm(x, y, family = binomial())},
    fastglm.qr      = {gf2 <- fastglm(x, y, family = binomial(), method = 1)},
    fastglm.LLT     = {gf3 <- fastglm(x, y, family = binomial(), method = 2)},
    fastglm.LDLT    = {gf4 <- fastglm(x, y, family = binomial(), method = 3)},
    fastglm.qr.fpiv = {gf5 <- fastglm(x, y, family = binomial(), method = 4)},
    times = 25L
)

autoplot(ct, log = FALSE) + stat_summary(fun.y = median, geom = 'point', size = 2)
```

<img src="vignettes/gen_data-1.png" width="100%" />

``` r
# comparison of estimates
max(abs(coef(gl1) - gf1$coef))
```

    ## [1] 2.590289e-14

``` r
max(abs(coef(gl1) - gf2$coef))
```

    ## [1] 2.546921e-14

``` r
max(abs(coef(gl1) - gf3$coef))
```

    ## [1] 1.140078e-13

``` r
max(abs(coef(gl1) - gf4$coef))
```

    ## [1] 1.094264e-13

``` r
max(abs(coef(gl1) - gf5$coef))
```

    ## [1] 2.776945e-14

``` r
# now between glm and speedglm
max(abs(coef(gl1) - sg1$coef))
```

    ## [1] 1.359413e-12

``` r
max(abs(coef(gl1) - sg2$coef))
```

    ## [1] 1.359413e-12

``` r
max(abs(coef(gl1) - sg3$coef))
```

    ## [1] 1.191977e-12

# Stability

The `fastglm` package does not compromise computational stability for
speed. In fact, for many situations where `glm()` and even `glm2()` do
not converge, `fastglm()` does converge.

As an example, consider the following data scenario, where the response
distribution is (mildly) misspecified, but the link function is quite
badly misspecified. In such scenarios, the standard IRLS algorithm tends
to have convergence issues. The `glm2()` package was designed to handle
such cases, however, it still can have convergence issues. The
`fastglm()` package uses a similar step-halving technique as `glm2()`,
but it starts at better initialized values and thus tends to have better
convergence properties in practice.

``` r
set.seed(1)
x <- matrix(rnorm(10000 * 100), ncol = 100)
y <- (exp(0.25 * x[,1] - 0.25 * x[,3] + 0.5 * x[,4] - 0.5 * x[,5] + rnorm(10000)) ) + 0.1


system.time(gfit1 <- fastglm(cbind(1, x), y, family = Gamma(link = "sqrt")))
```

    ##    user  system elapsed 
    ##   1.081   0.028   1.111

``` r
system.time(gfit2 <- glm(y~x, family = Gamma(link = "sqrt")) )
```

    ##    user  system elapsed 
    ##   4.196   0.166   4.380

``` r
system.time(gfit3 <- glm2::glm2(y~x, family = Gamma(link = "sqrt")) )
```

    ##    user  system elapsed 
    ##   2.767   0.106   2.894

``` r
## Note that fastglm() returns estimates with the
## largest likelihood
logLik(gfit1)
```

    ## 'log Lik.' -16030.81 (df=102)

``` r
logLik(gfit2)
```

    ## 'log Lik.' -16704.05 (df=102)

``` r
logLik(gfit3)
```

    ## 'log Lik.' -16046.66 (df=102)

``` r
coef(gfit1)[1:5]
```

    ##  (Intercept)           X1           X2           X3           X4 
    ##  1.429256009  0.125873599  0.005321164 -0.129389740  0.238937255

``` r
coef(gfit2)[1:5]
```

    ##   (Intercept)            x1            x2            x3            x4 
    ##  1.431168e+00  1.251936e-01 -6.896739e-05 -1.281857e-01  2.366473e-01

``` r
coef(gfit3)[1:5]
```

    ##   (Intercept)            x1            x2            x3            x4 
    ##  1.426864e+00  1.242616e-01 -9.860241e-05 -1.254873e-01  2.361301e-01

``` r
## check convergence of fastglm
gfit1$converged
```

    ## [1] TRUE

``` r
## number of IRLS iterations
gfit1$iter
```

    ## [1] 17

``` r
## now check convergence for glm()
gfit2$converged
```

    ## [1] FALSE

``` r
gfit2$iter
```

    ## [1] 25

``` r
## check convergence for glm2()
gfit3$converged
```

    ## [1] TRUE

``` r
gfit3$iter
```

    ## [1] 19

``` r
## increasing number of IRLS iterations for glm() does not help
system.time(gfit2 <- glm(y~x, family = Gamma(link = "sqrt"), control = list(maxit = 100)) )
```

    ##    user  system elapsed 
    ##  14.285   0.602  15.133

``` r
gfit2$converged
```

    ## [1] FALSE

``` r
gfit2$iter
```

    ## [1] 100

``` r
logLik(gfit1)
```

    ## 'log Lik.' -16030.81 (df=102)

``` r
logLik(gfit2)
```

    ## 'log Lik.' -16054.15 (df=102)
