





[![Build Status](https://travis-ci.org/jaredhuling/fastglm.svg?branch=master)](https://travis-ci.org/jaredhuling/fastglm)

# Overview of 'fastglm'

The 'fastglm' package is a re-write of `glm()` using `RcppEigen` designed to be computationally efficient.



# Installing the 'fastglm' package


Install the development version using the **devtools** package:

```r
devtools::install_github("jaredhuling/fastglm")
```

or by cloning and building using `R CMD INSTALL`

# Quick Usage Overview

Load the package:

```r
library(fastglm)
```


```r
library(speedglm)

set.seed(123)
n.obs  <- 10000
n.vars <- 100
x <- matrix(rnorm(n.obs * n.vars, sd = 3), n.obs, n.vars)

y <- 1 * (0.25 * x[,1] - 0.25 * x[,3] > rnorm(n.obs))

system.time(gl1 <- glm.fit(x, y, family = binomial()))
```

```
##    user  system elapsed 
##    1.06    0.00    1.08
```

```r
system.time(gf1 <- fastglm(x, y, family = binomial()))
```

```
##    user  system elapsed 
##    0.49    0.00    0.49
```

```r
system.time(gf2 <- fastglm(x, y, family = binomial(), method = 1))
```

```
##    user  system elapsed 
##    0.42    0.02    0.44
```

```r
system.time(gf3 <- fastglm(x, y, family = binomial(), method = 2))
```

```
##    user  system elapsed 
##    0.13    0.00    0.13
```

```r
system.time(gf4 <- fastglm(x, y, family = binomial(), method = 3))
```

```
##    user  system elapsed 
##    0.12    0.00    0.13
```

```r
system.time(sg1 <- speedglm.wfit(y, x, intercept = FALSE, family = binomial()))
```

```
##    user  system elapsed 
##    0.40    0.00    0.44
```

```r
system.time(sg2 <- speedglm.wfit(y, x, intercept = FALSE, family = binomial(), method = "Chol"))
```

```
##    user  system elapsed 
##    0.59    0.00    0.61
```

```r
system.time(sg3 <- speedglm.wfit(y, x, intercept = FALSE, family = binomial(), method = "qr"))
```

```
##    user  system elapsed 
##    0.39    0.03    0.43
```

```r
max(abs(coef(gl1) - gf1$coef))
```

```
## [1] 4.132162e-10
```

```r
max(abs(coef(gl1) - gf2$coef))
```

```
## [1] 4.132166e-10
```

```r
max(abs(coef(gl1) - gf3$coef))
```

```
## [1] 4.132164e-10
```

```r
max(abs(coef(gl1) - gf4$coef))
```

```
## [1] 4.132162e-10
```

```r
# now between glm and speedglm
max(abs(coef(gl1) - sg1$coef))
```

```
## [1] 2.164935e-15
```

```r
max(abs(coef(gl1) - sg2$coef))
```

```
## [1] 2.164935e-15
```

```r
max(abs(coef(gl1) - sg3$coef))
```

```
## [1] 2.220446e-15
```
