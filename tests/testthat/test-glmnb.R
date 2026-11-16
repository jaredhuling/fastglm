skip_if_not_installed("MASS")

# Match MASS::glm.nb(y ~ x[,-1]) on simulated NB data.

test_that("fastglm_nb matches MASS::glm.nb across realistic theta", {
    skip_if_not_installed("numDeriv")
    for (th_true in c(0.5, 2.0)) {
        set.seed(2024 + th_true * 10)
        n  <- 500
        X  <- cbind(1, matrix(rnorm(n * 2), n, 2))
        mu <- exp(X %*% c(0.3, 0.5, -0.2))
        y  <- MASS::rnegbin(n, mu = mu, theta = th_true)

        m  <- MASS::glm.nb(y ~ X[, -1])
        f  <- fastglm_nb(X, y)

        expect_equal(unname(coef(f)), unname(coef(m)),
                     tolerance = 1e-7,
                     info = paste0("theta_true = ", th_true))
        expect_equal(f$theta, m$theta, tolerance = 1e-6)
        objective <- function(par) -sum(dnbinom(y,
            mu=exp(drop(X%*%par[seq_len(ncol(X))])),size=exp(par[ncol(X)+1L]),log=TRUE))
        information <- numDeriv::hessian(objective,c(coef(f),log(f$theta)))
        j <- ncol(X)+1L
        # MASS holds beta fixed for its theta SE; fastglm reports the joint SE.
        expect_equal(f$theta/sqrt(information[j,j]),m$SE.theta,tolerance=1e-4)
        expect_equal(f$SE.theta,f$theta*sqrt(solve(information)[j,j]),tolerance=1e-5)
        expect_equal(f$twologlik, m$twologlik, tolerance = 1e-6)
        # Coefficient SE should match too.
        expect_equal(unname(f$se), unname(sqrt(diag(vcov(m)))),
                     tolerance = 1e-6)
    }
})

test_that("fastglm_nb matches MASS::glm.nb on the quine dataset", {
    skip_if_not(requireNamespace("MASS", quietly = TRUE))
    quine <- tryCatch({
        utils::data("quine", package = "MASS", envir = environment())
        get("quine", envir = environment())
    }, error = function(e) NULL)
    skip_if(is.null(quine), "quine dataset unavailable")
    f <- fastglm_nb(model.matrix(~ Sex + Age, data = quine), quine$Days)
    m <- MASS::glm.nb(Days ~ Sex + Age, data = quine)
    expect_equal(unname(coef(f)), unname(coef(m)), tolerance = 1e-6)
    expect_equal(f$theta, m$theta, tolerance = 1e-5)
    expect_equal(f$twologlik, m$twologlik, tolerance = 1e-6)
})

test_that("init.theta is honored", {
    set.seed(2025)
    n  <- 300
    X  <- cbind(1, matrix(rnorm(n * 2), n, 2))
    y  <- MASS::rnegbin(n, mu = exp(X %*% c(0.3, 0.5, -0.2)), theta = 1.5)
    f1 <- fastglm_nb(X, y)
    f2 <- fastglm_nb(X, y, init.theta = 5)
    expect_equal(f1$theta, f2$theta, tolerance = 1e-6)
    expect_equal(unname(coef(f1)), unname(coef(f2)), tolerance = 1e-7)
})

test_that("fastglm_nb converges on overdispersed data where MASS::glm.nb fails", {
    set.seed(4)
    n  <- 200
    p  <- 9
    X  <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    beta <- c(4, 1.5, -1.2, 1.5, -1.2, 1.5, -1.2, 1.5, -1.2)
    mu <- exp(X %*% beta)
    y  <- MASS::rnegbin(n, mu = mu, theta = 1)

    f <- fastglm_nb(X, y, link = "log")
    expect_true(f$converged)
    expect_true(all(is.finite(coef(f))))
    expect_true(is.finite(f$theta))
    expect_equal(f$theta, 1, tolerance = 0.2)

    # Verify log-likelihood against glm at the converged theta.
    fam <- MASS::negative.binomial(theta = f$theta)
    g <- glm(y ~ X[, -1], family = fam)
    expect_equal(unname(coef(f)), unname(coef(g)), tolerance = 1e-5)
    expect_equal(as.numeric(logLik(f)), as.numeric(logLik(g)),
                 tolerance = 1e-3)
})

test_that("fastglm_nb returns a fastglm-classed object with the expected slots", {
    set.seed(33)
    n  <- 200
    X  <- cbind(1, matrix(rnorm(n * 2), n, 2))
    y  <- MASS::rnegbin(n, mu = exp(X %*% c(0.3, 0.4, -0.2)), theta = 2)
    f  <- fastglm_nb(X, y)
    expect_s3_class(f, "fastglm_nb")
    expect_s3_class(f, "fastglm")
    expect_named(f$coefficients)
    expect_true(is.finite(f$theta))
    expect_true(is.finite(f$SE.theta))
    expect_true(f$converged)
    # vcov / SE work via the standard fastglm S3 dispatch.
    v <- vcov(f)
    expect_equal(dim(v), rep(ncol(X), 2))
})

test_that("fits with a flat likelihood in the coefficients are reported as converged", {
    # nb-stability vignette, scenario tiny_theta (theta = 0.05), replication 14.
    # Until 0.1.5 the outer loop required the raw coefficient change between
    # outer iterations to fall below outer.tol; here the log-likelihood is
    # flat in one direction, so the coefficients kept moving by 1e-5 while the
    # log-likelihood changed by 1e-9, and the fit stopped at outer.maxit as
    # not converged. Reference: the log-likelihood that 50 outer iterations
    # of the old rule reached (-235.1419368241), and theta 0.05120323528.
    set.seed(40014L); n <- 200L; p <- 3L
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    beta <- c(1.0, rnorm(p - 1, sd = 0.6))
    y <- MASS::rnegbin(n, mu = exp(pmin(drop(X %*% beta), 12)), theta = 0.05)
    f <- fastglm_nb(X, y)
    expect_true(f$converged)
    expect_lt(f$outer.iter, 10)
    expect_lt(abs(as.numeric(logLik(f)) - (-235.1419368241)), 1e-6)
    expect_equal(f$theta, 0.05120323528, tolerance = 1e-5)
})

nb_stability_data <- function(seed, n, p, theta, coef_sd = 0.5, intercept = 1.0) {
    # Generator of the nb-stability vignette (no collinear column, no outliers).
    set.seed(seed)
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    beta <- c(intercept, rnorm(p - 1, sd = coef_sd))
    y <- MASS::rnegbin(n, mu = exp(pmin(as.vector(X %*% beta), 12)), theta = theta)
    list(X = X, y = y)
}

test_that("a fit whose likelihood is highest in the Poisson limit returns theta = Inf", {
    # nb-stability vignette, near_poisson replication 24: the profile
    # likelihood increases toward theta = Inf. Until 0.1.5 the fit stopped
    # at the cap theta = 1e8.
    d <- nb_stability_data(60024L, 200L, 3L, 5000)
    expect_warning(f <- fastglm_nb(d$X, d$y), "theta estimate is infinite")
    pf <- glm.fit(d$X, d$y, family = poisson(), control = list(epsilon = 1e-12))
    expect_identical(f$theta, Inf)
    expect_identical(f$convergence.reason, "theta_infinite")
    expect_true(f$converged)
    expect_true(is.na(f$SE.theta))
    expect_equal(unname(coef(f)), unname(pf$coefficients), tolerance = 1e-7)
    expect_equal(as.numeric(logLik(f)), sum(dpois(d$y, pf$fitted.values, log = TRUE)), tolerance = 1e-10)
    expect_output(print(f), "Poisson limit")
    sparse <- suppressWarnings(fastglm_nb(as(d$X, "dgCMatrix"), d$y))
    expect_identical(sparse$theta, Inf)
    expect_equal(unname(coef(sparse)), unname(coef(f)), tolerance = 1e-10)
    g <- suppressWarnings(fastglm_nb(d$X, d$y, poisson.limit = FALSE))
    expect_equal(g$theta, 1e8)
})

test_that("a local maximum of the profile likelihood is replaced by the Poisson limit", {
    # nb-stability vignette, high_dim replication 10 (n = 120, p = 60): the
    # profile likelihood of theta has a local maximum near theta = 22
    # (log likelihood -324.765) and its supremum in the Poisson limit
    # (-307.434); profile values computed with glm.fit at fixed theta.
    d <- nb_stability_data(80010L, 120L, 60L, 2)
    f <- suppressWarnings(fastglm_nb(d$X, d$y))
    pf <- glm.fit(d$X, d$y, family = poisson(), control = list(epsilon = 1e-12, maxit = 100))
    expect_identical(f$theta, Inf)
    expect_equal(as.numeric(logLik(f)), sum(dpois(d$y, pf$fitted.values, log = TRUE)), tolerance = 1e-9)
    expect_lt(as.numeric(logLik(suppressWarnings(fastglm_nb(d$X, d$y, poisson.limit = FALSE)))), -324)
})

nb_link_data <- function(link, seed) {
    set.seed(seed); n <- 600
    z <- matrix(runif(n * 2), n, 2)
    mu <- if (link == "sqrt") pmax(0.4 + drop(z %*% c(1.6, -1.0)), 0.05)^2
          else pmax(0.5 + drop(z %*% c(3, -2.2)), 0.02)
    list(x = cbind(1, z), y = MASS::rnegbin(n, mu = mu, theta = 2))
}

test_that("sqrt and identity links converge to the constrained optimum", {
    # Until 0.1.5 these fits stopped with "no valid set of coefficients":
    # the least-squares start gave invalid linear predictors, and the optimum
    # lies on the boundary eta = 0. References: the maximum log likelihood
    # under X beta >= 0 from an active-set method (nloptr SLSQP over beta and
    # log theta, from fastglm's solution and an intercept-only start;
    # fastglm-jss dev/nb/run_nb_links.R).
    refs <- list(sqrt = list(seed = 910001, ll = -579.211328491),
                 identity = list(seed = 915001, ll = -810.944379883))
    for (lk in names(refs)) {
        d <- nb_link_data(lk, refs[[lk]]$seed)
        expect_warning(f <- fastglm_nb(d$x, d$y, link = lk), "boundary of the valid region")
        expect_true(f$converged)
        expect_identical(f$convergence.reason, "converged_boundary")
        ll <- sum(dnbinom(d$y, size = f$theta, mu = f$fitted.values, log = TRUE))
        expect_lt(abs(ll - refs[[lk]]$ll), 1e-6)
    }
    d <- nb_link_data("sqrt", 910001)
    sparse <- suppressWarnings(fastglm_nb(as(d$x, "dgCMatrix"), d$y, link = "sqrt"))
    expect_true(sparse$converged)
    expect_lt(abs(sum(dnbinom(d$y, size = sparse$theta, mu = sparse$fitted.values, log = TRUE)) -
                  (-579.211328491)), 1e-6)
})
