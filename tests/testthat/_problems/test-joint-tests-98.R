# Extracted from test-joint-tests.R:98

# test -------------------------------------------------------------------------
set.seed(111402)
x <- rnorm(100)
design <- cbind("(Intercept)"=1,x=x,duplicate=x,z=rnorm(100))
y <- rpois(100,exp(.2+.3*x))
fit <- fastglm(design,y,poisson(),method="auto")
expect_true(anyNA(coef(fit)))
expect_false(fastglm_wald_test(fit,terms="x")$available)
sum <- fastglm_wald_test(fit,contrast=c(x=1,duplicate=1))
reduced <- fastglm(design[,c(1,2,4)],y,poisson())
expect_true(sum$available)
expect_equal(sum$statistic,fastglm_wald_test(reduced,terms="x")$statistic,tolerance=1e-8)
