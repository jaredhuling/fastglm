






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
set.seed(123)
n.obs  <- 10000
n.vars <- 100
x <- matrix(rnorm(n.obs * n.vars, sd = 3), n.obs, n.vars)

y <- 1 * (0.25 * x[,1] - 0.25 * x[,3] > rnorm(10000))

system.time(gl1 <- glm.fit(x, y, family = binomial()))
```

```
##    user  system elapsed 
##    0.89    0.00    0.93
```

```r
system.time(gf1 <- fastglm(x, y, family = binomial()))
```

```
##    user  system elapsed 
##    0.43    0.00    0.45
```

```r
system.time(gf2 <- fastglm(x, y, family = binomial(), method = 1))
```

```
##    user  system elapsed 
##    0.40    0.00    0.42
```

```r
system.time(gf3 <- fastglm(x, y, family = binomial(), method = 2))
```

```
##    user  system elapsed 
##    0.09    0.04    0.13
```

```r
system.time(gf4 <- fastglm(x, y, family = binomial(), method = 3))
```

```
##    user  system elapsed 
##    0.12    0.00    0.13
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
