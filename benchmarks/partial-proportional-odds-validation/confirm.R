# Run from the repository root; use a new output directory on every run.
# Rscript confirm.R FASTGLM_LIBRARY OUTPUT REFERENCE_CACHE REFERENCE_LIBRARY
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 4L)
.libPaths(c(normalizePath(args[1L]), normalizePath(args[4L]), .libPaths()))
library(fastglm)
stopifnot(requireNamespace("numDeriv", quietly = TRUE))
out <- args[2L]
if (dir.exists(out)) stop("Use a new confirmation output directory")
dir.create(out, recursive = TRUE)
source("benchmarks/partial-proportional-odds-validation/reference.R")
checks <- list()
record <- function(name, error, tolerance) {
    checks[[length(checks) + 1L]] <<- data.frame(check = name, error = error,
        tolerance = tolerance, passed = is.finite(error) && error < tolerance)
}
evaluate <- function(a, link, ...) fastglm:::ordinal_partial_native_evaluate(
    a$x, a$z, integer(), a$counts, a$weights, a$offset,
    ncol(a$counts), a$theta, firth = TRUE, link = link, ...)
compare <- function(name, a, b, tolerance)
    record(name, max(abs(as.numeric(a) - as.numeric(b))), tolerance)

for (wide in c(FALSE, TRUE)) for (link in 0:2) {
    a <- partial_confirmation_fixture(wide)
    label <- paste(c("logit", "probit", "cloglog")[link + 1L],
                   if (wide) "wide" else "small")
    got <- evaluate(a, link)
    ref <- partial_cumulant_reference(a$theta, a$x, a$z, a$counts,
        a$weights, a$offset, c("logit", "probit", "cloglog")[link + 1L])
    for (field in c("ordinary.score", "information", "cov.unscaled"))
        compare(paste(label, field), got[[field]], ref[[field]], 2e-6)
    compare(paste(label, "expected negative Hessian"), got$information,
            ref$expected.hessian, 2e-6)
    compare(paste(label, "generic cumulant adjustment"), got$adjustment,
            ref$adjustment, 2e-5)
    compare(paste(label, "log likelihood"), got$loglik, ref$loglik, 1e-10)
}

for (link in 0:2) {
    a <- partial_confirmation_fixture()
    label <- c("logit", "probit", "cloglog")[link + 1L]
    got <- evaluate(a, link)
    common <- a; common$z <- matrix(numeric(), nrow(a$x), 0L)
    common$theta <- a$theta[1:3]
    po <- fastglm:::ordinal_native_evaluate(common$x, integer(), common$counts,
        common$weights, common$offset, 3L, common$theta, TRUE, link = link)
    partial <- evaluate(common, link)
    binary <- a; binary$counts <- cbind(a$counts[, 1L], rowSums(a$counts[, 2:3]))
    binary$theta <- a$theta[c(1L, 3L, 4L)]
    binary_po <- fastglm:::ordinal_native_evaluate(cbind(a$x, a$z), integer(),
        binary$counts, a$weights, a$offset, 2L, binary$theta, TRUE, link = link)
    binary_partial <- evaluate(binary, link)
    rows <- rep(seq_len(nrow(a$x)), rowSums(a$counts * a$weights))
    y <- unlist(lapply(seq_len(nrow(a$x)), function(i)
        rep(seq_len(ncol(a$counts)), a$counts[i, ] * a$weights[i])))
    expanded <- fastglm:::ordinal_partial_native_evaluate(a$x[rows, , drop = FALSE],
        a$z[rows, , drop = FALSE], as.integer(y), matrix(numeric(), 0L, 0L),
        rep(1, length(rows)), a$offset[rows], 3L, a$theta, TRUE, block_size = 7L,
        link = link)
    changed <- a; changed$z[3L, 1L] <- -1e5
    zero <- evaluate(changed, link)
    for (field in c("score", "ordinary.score", "information", "adjustment", "cov.unscaled")) {
        compare(paste(label, "common reduction", field), partial[[field]], po[[field]], 2e-9)
        compare(paste(label, "binary reduction", field), binary_partial[[field]], binary_po[[field]], 2e-9)
        compare(paste(label, "expanded frequencies", field), expanded[[field]], got[[field]], 2e-9)
        compare(paste(label, "zero-frequency domain", field), zero[[field]], got[[field]], 2e-9)
    }
    changed$weights[3L] <- 1
    invalid <- tryCatch(evaluate(changed, link), error = identity)
    record(paste(label, "positive-frequency crossing rejected"),
           as.numeric(!(inherits(invalid, "error") || identical(invalid$valid, FALSE))), .5)
    multiplied <- a; multiplied$weights <- a$weights * 3
    more <- evaluate(multiplied, link)
    compare(paste(label, "frequency-scale adjustment"), more$adjustment, got$adjustment, 2e-9)
    compare(paste(label, "frequency-scale score"), more$ordinary.score, 3 * got$ordinary.score, 2e-9)
    compare(paste(label, "frequency-scale information"), more$information, 3 * got$information, 2e-9)
    T <- diag(c(1, 1, 1 / 1.7, 1 / .8, 1 / .8))
    T[1:2, 3L] <- .4 / 1.7; T[1L, 4L] <- -.3 / .8; T[2L, 5L] <- -.3 / .8
    changed <- a; changed$x <- a$x * 1.7 + .4; changed$z <- a$z * .8 - .3
    changed$theta <- drop(T %*% a$theta)
    recoded <- evaluate(changed, link); inv <- solve(T)
    compare(paste(label, "recoded score"), recoded$score, t(inv) %*% got$score, 2e-9)
    compare(paste(label, "recoded information"), recoded$information,
            t(inv) %*% got$information %*% inv, 2e-9)
    compare(paste(label, "recoded adjustment"), recoded$adjustment,
            t(inv) %*% got$adjustment, 2e-9)
    S <- diag(5L); S[5L, 4L] <- 1
    ref <- partial_cumulant_reference(drop(solve(S, a$theta)), a$x, a$z,
        a$counts, a$weights, a$offset, label, map = S)
    compare(paste(label, "baseline-deviation adjustment"), ref$adjustment,
            t(S) %*% got$adjustment, 2e-5)
}

# Original external implementation; do not copy or alter its source.
reference_source <- file.path(args[3L], "supplementary", "bpolr.R")
source(reference_source)
fitted_checks <- list()
for (wide in c(FALSE, TRUE)) for (link in 0:2) for (br in c(FALSE, TRUE)) {
    a <- partial_confirmation_fixture(wide)
    colnames(a$z) <- paste0("varying", seq_len(ncol(a$z)))
    K <- ncol(a$counts); q <- K - 1L
    rows <- rep(seq_len(nrow(a$x)), rowSums(a$counts * a$weights))
    y <- unlist(lapply(seq_len(nrow(a$x)), function(i)
        rep(seq_len(K), a$counts[i, ] * a$weights[i])))
    dat <- data.frame(y = ordered(y, levels = seq_len(K)), common = a$x[rows, 1L],
                      off = a$offset[rows], a$z[rows, , drop = FALSE])
    formula <- as.formula(paste("y ~ common + offset(off) | 0 |",
                               paste(colnames(a$z), collapse = "+")))
    label <- paste(c("logit", "probit", "cloglog")[link + 1L],
                   if (wide) "wide" else "small", if (br) "BR root" else "ML root")
    external <- bpolr(formula, data = dat, method = if (br) "BR" else "ML",
                       link = c("logit", "probit", "cloglog")[link + 1L], epsilon = 1e-11)
    fitted <- fastglm:::ordinal_partial_native_fit(a$x, a$z, integer(), a$counts,
        a$weights, a$offset, K, a$theta, firth = br, score_tol = 1e-10,
        step_tol = 1e-10, maxit = 1000L, link = link)
    cut <- paste(seq_len(q), 1L + seq_len(q), sep = "|")
    nm <- c(cut, "common", unlist(lapply(cut, function(j) paste(j, colnames(a$z), sep = "."))))
    order <- match(nm, names(coef(external)))
    stopifnot(!anyNA(order))
    compare(paste(label, "coefficients"), fitted$coefficients, coef(external)[order], 2e-6)
    compare(paste(label, "covariance"), fitted$cov.unscaled, external$vcov[order, order], 2e-6)
    record(paste(label, "scaled score"), fitted$scaled.score, 1e-8)
    record(paste(label, "predictor update"), fitted$predictor.step, 1e-8)
    record(paste(label, "both converged"),
           as.numeric(!(isTRUE(fitted$converged) && isTRUE(external$convergence))), .5)
    fitted_checks[[label]] <- list(native = fitted, external = external)
}
saveRDS(fitted_checks, file.path(out, "fixture-fits.rds"))
data("wine", package = "ordinal")
external <- bpolr(rating ~ contact | 0 | temp, data = wine, method = "BR", epsilon = 1e-11)
counts <- rbind(c(4, 9, 5, 0, 0), c(1, 7, 8, 2, 0),
                c(0, 5, 8, 3, 2), c(0, 1, 5, 7, 5))
x <- matrix(c(0, 1, 0, 1), 4L, 1L); z <- matrix(c(0, 0, 1, 1), 4L, 1L)
start <- c(qlogis(cumsum(colSums(counts))[1:4] / sum(counts)), rep(0, 5))
fit <- fastglm:::ordinal_partial_native_fit(x, z, integer(), counts, rep(1, 4),
    numeric(4), 5L, start, firth = TRUE, score_tol = 1e-10, step_tol = 1e-10,
    maxit = 1000L)
order <- c(6:9, 1:5)
compare("wine external coefficients", fit$coefficients, coef(external)[order], 2e-6)
compare("wine external covariance", fit$cov.unscaled, external$vcov[order, order], 2e-6)
record("wine standardized adjusted score", fit$scaled.score, 1e-8)
record("wine remaining predictor update", fit$predictor.step, 1e-8)
record("wine convergence", as.numeric(!isTRUE(fit$converged)), .5)
write.csv(data.frame(parameter = c(paste0("alpha", 1:4), "contact", paste0("temp", 1:4)),
    native = as.numeric(fit$coefficients), external = unname(coef(external)[order]),
    native.se = sqrt(diag(fit$cov.unscaled)), external.se = sqrt(diag(external$vcov[order, order]))),
    file.path(out, "wine-fit.csv"), row.names = FALSE)
saveRDS(list(native = fit, external = external), file.path(out, "wine-fits.rds"))
result <- do.call(rbind, checks)
write.csv(result, file.path(out, "checks.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "session.txt"))
sources <- c("benchmarks/partial-proportional-odds-validation/protocol.txt",
    "benchmarks/partial-proportional-odds-validation/reference.R",
    "benchmarks/partial-proportional-odds-validation/confirm.R",
    "inst/include/ordinal_partial.h", "src/fit_ordinal_partial.cpp", reference_source)
write.csv(data.frame(path = sources, md5 = unname(tools::md5sum(sources))),
          file.path(out, "source-hashes.csv"), row.names = FALSE)
print(result[!result$passed, , drop = FALSE])
cat(sum(result$passed), "of", nrow(result), "comparisons passed\n")
if (!all(result$passed)) stop("Confirmation failed; retain these results")
