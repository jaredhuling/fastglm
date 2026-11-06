# Compare expected and observed information for ordinal standard errors as the
# number of categories grows, including one category per observation.
# Rscript benchmarks/ordinal-covariance/run.R --library=dir --extra-library=dir [--output=dir] [--quick]
# Each design writes one results file, so an interrupted run resumes.
args <- commandArgs(TRUE)
argument <- function(name, default = "") {
    value <- args[startsWith(args, paste0("--", name, "="))]
    if (length(value)) sub(paste0("^--", name, "="), "", value[1L]) else default
}
quick <- "--quick" %in% args
.libPaths(c(c(argument("library"), argument("extra-library"))[nzchar(c(argument("library"), argument("extra-library")))], .libPaths()))
for (package in c("fastglm", "Matrix", "parallel", "rms", "microbenchmark")) stopifnot(requireNamespace(package, quietly = TRUE))
library(fastglm)
here <- if (quick) tempfile("covariance-") else "benchmarks/ordinal-covariance"
raw <- argument("output", file.path(here, "raw"))
dir.create(raw, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(OMP_NUM_THREADS = "1")
cores <- as.integer(argument("cores", if (quick) "2" else "4"))
beta <- c(0.5, -0.5, 0.25)
# Hash the label: a sum of character codes gives many replicates the same seed.
make_seed <- function(...) {
    hash <- tools::md5sum(bytes = charToRaw(paste(..., sep = "|")))
    strtoi(substr(hash, 1L, 7L), 16L)
}

# Categories are equal-size bins of a latent outcome. The slopes are unchanged by
# binning under the model, and no category is empty even when there is one per row.
simulate <- function(n, K, link, seed) {
    set.seed(seed)
    x <- matrix(rnorm(n * 3L), n, 3L, dimnames = list(NULL, c("a", "b", "c")))
    inverse <- switch(link, logit = qlogis, probit = qnorm, cloglog = function(u) log(-log1p(-u)))
    latent <- drop(x %*% beta) + inverse(runif(n))
    list(x = x, y = ordered(ceiling(rank(latent, ties.method = "first") * K / n), levels = seq_len(K)))
}

# Slope block of the inverse information, by the Schur complement of the
# tridiagonal threshold block.
slope_covariance <- function(information, q) {
    index <- seq_len(q)
    d <- nrow(information)
    threshold <- Matrix::bandSparse(q, k = 0:1, symmetric = TRUE,
        diagonals = list(diag(information)[index], if (q > 1L) information[cbind(index[-q], index[-1L])] else numeric()))
    cross <- information[index, -index, drop = FALSE]
    solve(information[-index, -index, drop = FALSE] - crossprod(cross, as.matrix(Matrix::solve(threshold, cross))))
}

one_replicate <- function(design, r) {
    data <- simulate(design$n, design$K, design$link, make_seed(design$label, r))
    fit <- suppressWarnings(fastglm_ordinal(data$x, data$y, link = design$link, tol = 1e-8))
    if (!isTRUE(fit$converged)) return(NULL)
    q <- design$K - 1L
    code <- match(design$link, c("logit", "probit", "cloglog")) - 1L
    observed <- fastglm:::ordinal_observed_evaluate(data$x, as.integer(data$y), matrix(numeric(), 0L, 0L),
        rep(1, design$n), numeric(design$n), design$K, unname(coef(fit)), 2048L, code)
    estimate <- unname(fit$beta)
    se_expected <- sqrt(diag(vcov(fit))[q + 1:3])
    se_observed <- sqrt(diag(slope_covariance(observed$information, q)))
    c(estimate, se_expected, se_observed)
}

run_design <- function(design) {
    file <- file.path(raw, paste0("results_", design$label, ".rds"))
    if (file.exists(file)) return(readRDS(file))
    values <- parallel::mclapply(seq_len(design$replicates), function(r) one_replicate(design, r),
                                 mc.cores = cores, mc.preschedule = FALSE)
    values <- do.call(rbind, Filter(is.numeric, values))
    saveRDS(values, file)
    values
}

grid <- expand.grid(n = if (quick) 300L else c(500L, 2000L), fraction = c(5, 30, 0.1, 0.5, 1),
                    link = if (quick) "logit" else c("logit", "probit", "cloglog"), stringsAsFactors = FALSE)
grid$K <- ifelse(grid$fraction >= 5, grid$fraction, grid$n * grid$fraction)
grid$K <- as.integer(grid$K)
grid$replicates <- if (quick) 12L else ifelse(grid$K >= 1000L, 200L, 1000L)
grid$label <- sprintf("%s_n%d_K%d", grid$link, grid$n, grid$K)
rows <- list()
for (i in seq_len(nrow(grid))) {
    design <- grid[i, ]
    gc(FALSE)
    values <- run_design(design)
    for (j in 1:3) {
        b <- values[, j]; e <- values[, j + 3L]; o <- values[, j + 6L]
        z <- qnorm(0.975)
        rows[[length(rows) + 1L]] <- data.frame(link = design$link, n = design$n, K = design$K, slope = j,
            replicates = nrow(values), bias = mean(b) - beta[j], empirical_sd = sd(b),
            mean_se_expected = mean(e), mean_se_observed = mean(o), ratio_expected_observed = mean(e / o),
            maximum_relative_difference = max(abs(e / o - 1)),
            coverage_expected = mean(abs(b - beta[j]) <= z * e), coverage_observed = mean(abs(b - beta[j]) <= z * o),
            monte_carlo_se = sqrt(0.95 * 0.05 / nrow(values)))
    }
    message(design$label, " done")
}
summary <- do.call(rbind, rows)
write.csv(summary, file.path(here, "summary.csv"), row.names = FALSE)
print(summary[, c("link", "n", "K", "slope", "replicates", "empirical_sd", "mean_se_expected", "mean_se_observed",
                  "maximum_relative_difference", "coverage_expected", "coverage_observed")], digits = 3L, row.names = FALSE)

# Cost of each step at n = 20,000, one process, median of three runs.
now <- microbenchmark::get_nanotime
clock <- function(f, r = 3L) median(vapply(seq_len(r), function(i) {gc(FALSE); s <- now(); f(); (now() - s) / 1e9}, 0))
costs <- list()
for (K in if (quick) 10L else c(30L, 300L, 1000L, 2000L, 5000L)) {
    n <- if (quick) 500L else 20000L
    data <- simulate(n, K, "logit", make_seed("cost", K))
    code <- as.integer(data$y)
    d <- data.frame(y = code, data$x)
    start <- c(qlogis(cumsum(tabulate(code, K))[-K] / n), numeric(3L))
    empty <- matrix(numeric(), 0L, 0L)
    newton <- function() fastglm:::ordinal_observed_evaluate(data$x, code, empty, rep(1, n), numeric(n), K, start, 2048L, 0L)
    state <- newton()
    expected <- function() fastglm:::ordinal_native_evaluate(data$x, code, empty, rep(1, n), numeric(n), K, start, FALSE, 2048L, 1e-14, 0L)
    row <- data.frame(n = n, K = K,
        observed_pass = clock(newton), schur_slope_block = clock(function() slope_covariance(state$information, K - 1L)),
        expected_pass_with_inverse = if (K <= 2000L) clock(expected) else NA_real_,
        dense_inverse = if (K <= 2000L) clock(function() chol2inv(chol(state$information))) else NA_real_,
        public_fit = if (K <= 2000L) clock(function() {f <- suppressWarnings(fastglm_ordinal(data$x, data$y, tol = 1e-8)); vcov(f)}) else NA_real_,
        orm = clock(function() {o <- rms::orm(y ~ a + b + c, data = d, eps = 1e-8, maxit = 100L); vcov(o, intercepts = "all")}))
    costs[[length(costs) + 1L]] <- row
    print(row, digits = 3L, row.names = FALSE)
}
write.csv(do.call(rbind, costs), file.path(here, "costs.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(here, "session-info.txt"))
