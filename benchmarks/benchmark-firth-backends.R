# Rscript benchmarks/benchmark-firth-backends.R <output-directory> [library-directory]
# Run each frozen package library separately on an idle machine. The saved
# coefficients, covariance matrices, iteration counts, and source-read counts
# permit numerical and work comparisons before interpreting elapsed times.
args <- commandArgs(TRUE)
if (!length(args)) stop("Supply an output directory")
if (length(args) > 1L) .libPaths(c(args[2L], .libPaths()))
dir.create(args[1L], recursive = TRUE, showWarnings = FALSE)
library(fastglm)
stopifnot(requireNamespace("microbenchmark", quietly = TRUE),
          requireNamespace("Matrix", quietly = TRUE))
set.seed(81041)
records <- estimates <- list()
for (dimensions in list(c(10000L, 8L), c(50000L, 30L), c(10000L, 80L))) {
    n <- dimensions[1L]; p <- dimensions[2L]
    X <- matrix(rnorm(n * (p - 1L), sd = .15), n)
    X[matrix(runif(length(X)) > .15, n)] <- 0
    X <- cbind(1, X)
    sparse <- Matrix::Matrix(X, sparse = TRUE)
    big <- bigmemory::as.big.matrix(X)
    offset <- runif(n, -.02, .02)
    cases <- list(gaussian = list(gaussian(), "unit"),
                  Gamma_unit = list(Gamma("log"), "unit"),
                  Gamma_repeated = list(Gamma("log"), "repeated"),
                  Gamma_distinct = list(Gamma("log"), "distinct"),
                  poisson_sqrt = list(poisson("sqrt"), "repeated"),
                  poisson_log = list(poisson("log"), "unit"))
    for (case in names(cases)) {
        family <- cases[[case]][[1L]]
        weight_kind <- cases[[case]][[2L]]
        w <- switch(weight_kind, unit = rep(1, n),
                    repeated = rep(c(0, .25, 1.5, 3), length.out = n),
                    distinct = runif(n, .3, 2))
        beta <- c(if (family$link == "sqrt") 3 else .8, rep(.1, p - 1L))
        mu <- family$linkinv(drop(X %*% beta) + offset)
        y <- switch(family$family, gaussian = rnorm(n, mu, .7),
                    Gamma = rgamma(n, shape = 1.5, scale = mu / 1.5),
                    poisson = rpois(n, mu))
        # Each backend receives the same observable intercept-only start.
        start <- c(family$linkfun(weighted.mean(y, w)), rep(0, p - 1L))
        chunk_size <- 2000L
        callback <- function(k) {
            ii <- ((k - 1L) * chunk_size + 1L):min(k * chunk_size, n)
            list(X = X[ii, , drop = FALSE], y = y[ii],
                 weights = w[ii], offset = offset[ii])
        }
        for (backend in c("dense", "sparse", "big.matrix", "streaming")) {
            fit_once <- switch(backend,
                dense = function() fastglm(X, y, family, weights = w, offset = offset,
                    start = start, method = 2L, firth = TRUE, tol = 1e-9),
                sparse = function() fastglm(sparse, y, family, weights = w, offset = offset,
                    start = start, method = 2L, firth = TRUE, tol = 1e-9),
                big.matrix = function() fastglm(big, y, family, weights = w, offset = offset,
                    start = start, method = 2L, firth = TRUE, tol = 1e-9),
                streaming = function() fastglm_streaming(callback, ceiling(n/chunk_size),
                    family, start = start, method = 2L, firth = TRUE, tol = 1e-9))
            fit <- fit_once()
            stopifnot(fit$converged)
            timing <- microbenchmark::microbenchmark(fit_once(), times = 5L)
            id <- paste(case, n, p, backend, sep = "_")
            records[[id]] <- data.frame(case = case, backend = backend, n = n, p = p,
                weights = weight_kind, median_ms = median(timing$time)/1e6,
                min_ms = min(timing$time)/1e6, iterations = fit$iter,
                passes = if (is.null(fit$passes)) NA_integer_ else fit$passes,
                reads = if (is.null(fit$chunk.reads)) NA_integer_ else fit$chunk.reads)
            estimates[[id]] <- list(coefficients = coef(fit), covariance = vcov(fit),
                information.inverse = fit$cov.unscaled, dispersion = fit$dispersion,
                fitted = family$linkinv(drop(X %*% coef(fit)) + offset),
                deviance = fit$deviance)
        }
    }
}
write.csv(do.call(rbind, records), file.path(args[1L], "timings.csv"), row.names = FALSE)
saveRDS(estimates, file.path(args[1L], "estimates.rds"))
writeLines(capture.output(sessionInfo()), file.path(args[1L], "session-info.txt"))
