# Extracted from test-wald-components.R:113

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "fastglm", path = "..")
attach(test_env, warn.conflicts = FALSE)

# test -------------------------------------------------------------------------
set.seed(111705)
x <- cbind("(Intercept)"=1,x=rnorm(240))
x <- cbind(x,duplicate=x[,2])
y <- rgamma(240,2)*rbinom(240,1,.6)
fit <- fastglm_twopart(y~x+duplicate,data=data.frame(y,x=x[,2],duplicate=x[,3]),method="auto",tol=1e-11)
reduced <- fastglm_twopart(x[,1:2],response=y,tol=1e-11)
expect_true(anyNA(coef(fit)))
expect_identical(fastglm_wald_test(fit,terms="x",component="positive")$reason,"nonestimable_contrast")
out <- fastglm_wald_test(fit,contrast=c(count_x=1,count_duplicate=1))
