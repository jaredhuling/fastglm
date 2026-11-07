# Compare expected, observed, and robust standard errors for ordinal thresholds and
# slopes, under the true link and under misspecification.
# Rscript benchmarks/ordinal-covariance/extended.R --library=dir [--output=dir] [--cores=3] [--quick]
# Each design writes one results file, so an interrupted run resumes.
args <- commandArgs(TRUE)
argument <- function(name, default = "") {
    value <- args[startsWith(args, paste0("--", name, "="))]
    if (length(value)) sub(paste0("^--", name, "="), "", value[1L]) else default
}
quick <- "--quick" %in% args
if (nzchar(argument("library"))) .libPaths(c(argument("library"), .libPaths()))
library(fastglm)
here <- if (quick) tempfile("covariance-") else "benchmarks/ordinal-covariance"
raw <- argument("output", file.path(here, "raw-extended"))
dir.create(raw, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(OMP_NUM_THREADS = "1")
cores <- as.integer(argument("cores", "3"))
make_seed <- function(...) strtoi(substr(tools::md5sum(bytes = charToRaw(paste(..., sep = "|"))), 1L, 7L), 16L)
beta <- c(0.5, -0.5, 0.25)
inverse <- function(link) switch(link, logit = qlogis, probit = qnorm, cloglog = function(u) log(-log1p(-u)))

# Errors come from `truth`, a heavy-tailed mixture, or a skewed gamma distribution.
errors <- function(n, truth) switch(truth,
    logit = , probit = , cloglog = inverse(truth)(runif(n)),
    heavy = rt(n, df = 3),
    skewed = rgamma(n, shape = 2) - 2)
# Cut points are fixed population quantiles, so thresholds have a fixed target.
cut_points <- function(K, truth) {
    set.seed(make_seed("cuts", K, truth))
    latent <- drop(matrix(rnorm(3e6), ncol = 3L) %*% beta) + errors(1e6, truth)
    unname(quantile(latent, seq_len(K - 1L) / K))
}
simulate <- function(n, cuts, truth, seed) {
    set.seed(seed)
    x <- matrix(rnorm(n * 3L), n, 3L, dimnames = list(NULL, c("a", "b", "c")))
    latent <- drop(x %*% beta) + errors(n, truth)
    list(x = x, y = ordered(findInterval(latent, cuts) + 1L, levels = seq_len(length(cuts) + 1L)))
}
# Under misspecification the target is the fit the link would reach in the limit.
targets <- function(design, cuts) {
    file <- file.path(raw, paste0("target_", design$truth, "_", design$link, "_K", design$K, ".rds"))
    if (file.exists(file)) return(readRDS(file))
    data <- simulate(if (quick) 2e4L else 2e6L, cuts, design$truth, make_seed("target", design$label))
    fit <- suppressWarnings(fastglm_ordinal(data$x, data$y, link = design$link))
    value <- c(unname(fit$alpha), unname(fit$beta))
    saveRDS(value, file)
    value
}
one_replicate <- function(design, cuts, target, r) {
    data <- simulate(design$n, cuts, design$truth, make_seed(design$label, r))
    fit <- tryCatch(suppressWarnings(fastglm_ordinal(data$x, data$y, link = design$link)), error = function(e) NULL)
    if (is.null(fit) || !isTRUE(fit$converged) || length(fit$coefficients) != length(target)) return(NULL)
    code <- match(design$link, c("logit", "probit", "cloglog")) - 1L
    observed <- fastglm:::ordinal_observed_evaluate(data$x, as.integer(data$y), matrix(numeric(), 0L, 0L),
        rep(1, design$n), numeric(design$n), design$K, unname(coef(fit)), 2048L, code)
    expected_se <- sqrt(diag(vcov(fit)))
    observed_se <- sqrt(diag(solve(observed$information)))
    robust_se <- sqrt(diag(fastglm_vcov(fit, type = "HC0")))
    c(unname(coef(fit)), expected_se, observed_se, robust_se)
}
grid <- rbind(
    expand.grid(truth = c("logit", "probit", "cloglog"), n = c(500L, 2000L), K = c(5L, 30L), stringsAsFactors = FALSE),
    expand.grid(truth = c("logit", "probit"), n = 2000L, K = 200L, stringsAsFactors = FALSE))
grid$link <- grid$truth
misspecified <- rbind(
    data.frame(truth = c("probit", "cloglog", "heavy", "skewed"), link = "logit"),
    data.frame(truth = c("logit", "heavy"), link = "probit"))
misspecified <- merge(misspecified, expand.grid(n = c(500L, 2000L), K = c(5L, 30L)))
grid <- rbind(grid[, c("truth", "link", "n", "K")], misspecified[, c("truth", "link", "n", "K")])
if (quick) grid <- grid[grid$n == 500L & grid$K == 5L, ][1:2, ]
grid$label <- sprintf("%s_fit-%s_n%d_K%d", grid$truth, grid$link, grid$n, grid$K)
replicates <- if (quick) 12L else 1000L
rows <- list()
for (i in seq_len(nrow(grid))) {
    design <- grid[i, ]
    gc(FALSE)
    file <- file.path(raw, paste0("results_", design$label, ".rds"))
    cuts <- cut_points(design$K, design$truth)
    target <- targets(design, cuts)
    if (file.exists(file) && file.size(file) > 100L) values <- readRDS(file) else {
        # One error would otherwise end every replicate on that worker, and dropped
        # replicates matter for coverage, so each failure is kept with its reason.
        attempts <- parallel::mclapply(seq_len(replicates), function(r) tryCatch(
            one_replicate(design, cuts, target, r), error = function(e) conditionMessage(e)),
            mc.cores = cores, mc.preschedule = TRUE)
        values <- do.call(rbind, Filter(is.numeric, attempts))
        attr(values, "failures") <- table(vapply(attempts, function(a)
            if (is.numeric(a)) "completed" else if (is.null(a)) "not converged or category lost" else substr(a, 1L, 80L), ""))
        saveRDS(values, file)
    }
    d <- ncol(values) %/% 4L; q <- design$K - 1L
    for (j in c(1L, (q + 1L) %/% 2L, q, q + 1:3)) {
        b <- values[, j]; ses <- values[, d * 1:3 + j]; z <- qnorm(0.975)
        rows[[length(rows) + 1L]] <- data.frame(truth = design$truth, link = design$link, n = design$n, K = design$K,
            parameter = if (j <= q) paste0("threshold ", j, " of ", q) else paste0("slope ", j - q),
            kind = if (j <= q) "threshold" else "slope", replicates = nrow(values), bias = mean(b) - target[j],
            empirical_sd = sd(b), expected_se = mean(ses[, 1L]), observed_se = mean(ses[, 2L]), robust_se = mean(ses[, 3L]),
            coverage_expected = mean(abs(b - target[j]) <= z * ses[, 1L]),
            coverage_observed = mean(abs(b - target[j]) <= z * ses[, 2L]),
            coverage_robust = mean(abs(b - target[j]) <= z * ses[, 3L]),
            monte_carlo_se = sqrt(0.95 * 0.05 / nrow(values)))
    }
    message(design$label, " replicates ", nrow(values), if (!is.null(attr(values, "failures")) && length(attr(values, "failures")) > 1L)
        paste0(" | ", paste(names(attr(values, "failures")), attr(values, "failures"), collapse = "; ")))
}
summary <- do.call(rbind, rows)
write.csv(summary, file.path(here, "extended-summary.csv"), row.names = FALSE)
print(summary[, c("truth", "link", "n", "K", "parameter", "empirical_sd", "expected_se", "observed_se", "robust_se",
                  "coverage_expected", "coverage_observed", "coverage_robust")], digits = 3L, row.names = FALSE)
