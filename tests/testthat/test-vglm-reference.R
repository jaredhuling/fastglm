test_that("reference fit matches tightened nnet and VGAM", {
    skip_if_not_installed("nnet")
    skip_if_not_installed("VGAM")
    d <- vglm_sim_data(2000L, 3L, 4L, seed = 1L)
    fit <- .vglm_reference_fit(d$x, d$counts)
    expect_true(fit$converged)
    expect_equal(dim(fit$coefficients), c(3L, 4L))
    expect_equal(rownames(fit$coefficients), c("B", "C", "D"))
    expect_equal(colnames(fit$coefficients), colnames(d$x))

    dat <- data.frame(y = d$y, d$x[, -1])
    nn <- nnet::multinom(y ~ x1 + x2 + x3, data = dat, maxit = 10000,
                         reltol = 1e-14, abstol = 1e-14, MaxNWts = 1e6,
                         trace = FALSE)
    expect_equal(fit$loglik, -nn$deviance / 2, tolerance = 1e-8)
    cn <- coef(nn)
    colnames(cn)[1] <- "(Intercept)"
    expect_equal(fit$coefficients, cn, tolerance = 1e-6, ignore_attr = TRUE)

    # epsilon = 1e-12 is at the floating-point limit, so VGAM warns that it
    # stopped at a half-step; the estimates are still accurate.
    vg <- suppressWarnings(
        VGAM::vglm(y ~ x1 + x2 + x3, VGAM::multinomial(refLevel = 1),
                   data = dat,
                   control = VGAM::vglm.control(epsilon = 1e-12, maxit = 200)))
    expect_equal(as.vector(fit$coefficients), unname(coef(vg)), tolerance = 1e-6)
    expect_equal(unname(sqrt(diag(fit$cov.unscaled))),
                 unname(sqrt(diag(VGAM::vcov(vg)))), tolerance = 1e-6)
})

test_that("reference bias reduction matches brglm2", {
    skip_if_not_installed("brglm2")
    check <- function(d) {
        fit <- .vglm_reference_fit(d$x, d$counts, firth = TRUE)
        expect_true(fit$converged)
        dat <- data.frame(y = d$y, d$x[, -1])
        fm <- stats::reformulate(colnames(d$x)[-1], "y")
        br <- brglm2::brmultinom(fm, data = dat, type = "AS_mean",
                                 epsilon = 1e-12)
        bc <- coef(br)
        expect_equal(as.vector(fit$coefficients), as.vector(unname(bc)),
                     tolerance = 1e-6)
    }
    # brmultinom fits through a Poisson representation with one nuisance
    # parameter per row, so the simulated check uses n = 300 to stay fast.
    check(vglm_sim_data(300L, 3L, 4L, seed = 1L))
    # On separated data the agreement depends on brglm2's own convergence
    # near the boundary, so this external comparison is not run on CRAN.
    skip_on_cran()
    check(vglm_separated_data())
})

test_that("reference fit handles weights, offsets, and zero-weight rows", {
    d <- vglm_sim_data(300L, 2L, 3L, seed = 2L)
    n <- nrow(d$x)
    w <- rep(1, n); w[1:10] <- 0
    fit <- .vglm_reference_fit(d$x, d$counts, weights = w)
    fit2 <- .vglm_reference_fit(d$x[-(1:10), ], d$counts[-(1:10), ])
    f1 <- .vglm_reference_fit(d$x, d$counts)
    expect_equal(fit$coefficients, fit2$coefficients, tolerance = 1e-9)
    expect_equal(fit$loglik, fit2$loglik, tolerance = 1e-9)
    # a constant offset is absorbed by the intercepts
    off <- c(0.3, -0.2)
    fo <- .vglm_reference_fit(d$x, d$counts,
                                        offset = matrix(off, n, 2, byrow = TRUE))
    expect_true(fo$converged)
    shift <- f1$coefficients
    shift[, 1] <- shift[, 1] - off
    expect_equal(fo$coefficients, shift, tolerance = 1e-8)
    # a different reference level gives the same likelihood
    f3 <- .vglm_reference_fit(d$x, d$counts, reference = 3L)
    expect_equal(f3$loglik, f1$loglik, tolerance = 1e-10)
    expect_equal(rownames(f3$coefficients), c("A", "B"))
})

test_that("separated data: ML estimate diverges, bias-reduced estimate is finite", {
    d <- vglm_separated_data()
    # With 20 iterations the unpenalized coefficient for the separating
    # predictor is still growing (it would reach about 90 by iteration 34).
    ml <- .vglm_reference_fit(d$x, d$counts, maxit = 20L)
    expect_false(ml$converged)
    expect_gt(max(abs(ml$coefficients)), 20)
    br <- .vglm_reference_fit(d$x, d$counts, firth = TRUE)
    expect_true(br$converged)
    expect_lt(max(abs(br$coefficients)), 10)
})
