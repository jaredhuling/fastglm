# Extracted from test-wald-components.R:45

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "fastglm", path = "..")
attach(test_env, warn.conflicts = FALSE)

# test -------------------------------------------------------------------------
set.seed(111702)
d <- data.frame(x=rnorm(300),g=factor(rep(letters[1:3],100)))
d$y <- rgamma(300,2,scale=exp(.3+.2*d$x)/2)*rbinom(300,1,.6)
fit <- fastglm_twopart(y~x+g|x,d,tol=1e-11)
fit$x <- fit$model <- NULL
expect_true(fastglm_wald_test(fit,terms="g",component="positive")$available)
expect_equal(fastglm_wald_test(fit,terms="x",component="binary")$statistic,
        fastglm_wald_test(fit,terms="x",component="zero")$statistic)
fit <- fastglm_twopart(y~x+g|x,d,tol=1e-11)
V <- fastglm_vcov(fit,cluster=rep(1:30,each=10),type="HC1")
L <- c(count_x=1,zero_x=-1)
i <- names(L)
expect_gt(abs(V["count_x","zero_x"]),1e-8)
