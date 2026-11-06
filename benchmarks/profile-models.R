# Profile fitting and inference across model classes to locate avoidable cost.
# Rscript benchmarks/profile-models.R --library=dir [--n=100000] [--output=dir]
# Each task reports median elapsed time and how its sampled time divides between
# compiled calls and R code, with the R functions that use the most time.
args <- commandArgs(TRUE)
argument <- function(name, default = "") {
    value <- args[startsWith(args, paste0("--", name, "="))]
    if (length(value)) sub(paste0("^--", name, "="), "", value[1L]) else default
}
if (nzchar(argument("library"))) .libPaths(c(argument("library"), .libPaths()))
library(fastglm)
n <- as.integer(argument("n", "100000"))
output <- argument("output", "benchmarks/model-profile")
dir.create(output, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(OMP_NUM_THREADS = "1")
set.seed(as.integer(argument("seed", "123")))
p <- 10L
X <- matrix(rnorm(n * p), n, p, dimnames = list(NULL, paste0("x", seq_len(p))))
beta <- 0.3 * (-1)^seq_len(p) / sqrt(seq_len(p))
eta <- drop(X %*% beta)
D <- cbind(`(Intercept)` = 1, X)
y_binary <- rbinom(n, 1L, plogis(eta))
y_count <- rpois(n, exp(0.5 + eta))
y_nb <- rnbinom(n, size = 1.5, mu = exp(0.5 + eta))
y_zi <- y_count * rbinom(n, 1L, 0.7)
y_gamma <- rgamma(n, shape = 2, rate = 2 / exp(0.2 + eta))
y_part <- y_gamma * rbinom(n, 1L, 0.7)
y_ordinal <- ordered(cut(eta + rlogis(n), c(-Inf, -1, 0, 1, Inf), labels = FALSE))
frame <- data.frame(y_count, y_nb, y_zi, y_part, X)
rhs <- paste(colnames(X), collapse = "+")
blocks <- split(seq_len(n), ceiling(seq_len(n) / 10000L))
source_of <- function(design, response) fastglm_source(function(k) {
    i <- blocks[[k]]
    list(X = design[i, , drop = FALSE], y = response[i])
}, length(blocks))
ordinal_fit <- fastglm_ordinal(X, y_ordinal)
partial_fit <- fastglm_ordinal(X, y_ordinal, nonparallel = "x2")
glm_fit <- fastglm(D, y_binary, family = binomial())
small <- seq_len(min(n, 5000L))
small_fit <- fastglm_ordinal(X[small, ], y_ordinal[small])
tasks <- list(
    `GLM logistic` = function() fastglm(D, y_binary, family = binomial()),
    `GLM Poisson` = function() fastglm(D, y_count, family = poisson()),
    `GLM Gamma` = function() fastglm(D, y_gamma, family = Gamma("log")),
    `GLM logistic Firth` = function() fastglm(D, y_binary, family = binomial(), firth = TRUE),
    `GLM Gamma Firth` = function() fastglm(D, y_gamma, family = Gamma("log"), firth = TRUE),
    `GLM streamed` = function() fastglm_streaming(source_of(D, y_binary), family = binomial()),
    `GLM summary` = function() summary(glm_fit),
    `GLM predict` = function() predict(glm_fit, newdata = D),
    `Negative binomial` = function() fastglm_nb(D, y_nb),
    `Negative binomial streamed` = function() fastglm_nb(source_of(D, y_nb)),
    `Hurdle Poisson` = function() fastglm_hurdle(D, response = y_zi),
    `Hurdle streamed` = function() fastglm_hurdle(source_of(D, y_zi)),
    `Zero-inflated Poisson` = function() fastglm_zi(as.formula(paste("y_zi~", rhs)), data = frame),
    `Two-part` = function() fastglm_twopart(D, response = y_part),
    `Truncated Poisson` = function() fastglm_truncated(D[y_count > 0L, ], y_count[y_count > 0L]),
    `Ordinal ML` = function() fastglm_ordinal(X, y_ordinal),
    `Ordinal ML probit` = function() fastglm_ordinal(X, y_ordinal, link = "probit"),
    `Ordinal mean BR` = function() fastglm_ordinal(X, y_ordinal, firth = TRUE),
    `Ordinal formula` = function() fastglm_ordinal(as.formula(paste("y_ordinal~", rhs)), data = data.frame(y_ordinal, X)),
    `Ordinal streamed ML` = function() fastglm_ordinal_streaming(source_of(X, y_ordinal)),
    `Ordinal streamed mean BR` = function() fastglm_ordinal_streaming(source_of(X, y_ordinal), firth = TRUE),
    `Partial ML` = function() fastglm_ordinal(X, y_ordinal, nonparallel = "x2"),
    `Partial mean BR` = function() fastglm_ordinal(X, y_ordinal, nonparallel = "x2", firth = TRUE),
    `Partial streamed ML` = function() fastglm_ordinal_streaming(source_of(X, y_ordinal), nonparallel = "x2"),
    `Ordinal summary` = function() summary(ordinal_fit),
    `Ordinal predict` = function() predict(ordinal_fit, newdata = X),
    `Ordinal robust covariance` = function() fastglm_vcov(ordinal_fit, type = "HC0"),
    `Ordinal profile interval` = function() confint(ordinal_fit, parm = "x1", method = "profile"),
    `Ordinal score test` = function() fastglm_score_test(ordinal_fit, terms = "x1"),
    `Ordinal common-slope test` = function() fastglm_po_test(ordinal_fit),
    `Ordinal marginal effects` = function() fastglm_marginal_effects(ordinal_fit, variable = "x1"),
    `Ordinal assessment` = function() fastglm_ordinal_assess(ordinal_fit),
    `Ordinal bootstrap 5,000 rows, 20 refits` = function() fastglm_bootstrap(small_fit, R = 20L),
    `Partial profile interval` = function() confint(partial_fit, parm = "x1", method = "profile"))
wrappers <- Filter(function(name) {
    value <- get(name, envir = asNamespace("fastglm"))
    is.function(value) && any(grepl(".Call(", deparse(body(value)), fixed = TRUE)) && length(deparse(body(value))) <= 4L
}, ls(asNamespace("fastglm"), all.names = TRUE))
native <- c(".Call", ".External", ".External2", ".Fortran", ".C", "La_solve", "La_chol", "La_chol2inv", "La_qr", "qr.default",
            "%*%", "crossprod", "tcrossprod", "backsolve")
profile_task <- function(name, task) {
    message(name)
    clock <- function() {
        started <- Sys.time()
        suppressWarnings(suppressMessages(task()))
        as.numeric(difftime(Sys.time(), started, units = "secs"))
    }
    gc(FALSE)
    first <- tryCatch(clock(), error = function(e) conditionMessage(e))
    if (is.character(first))
        return(data.frame(task = name, status = substr(first, 1L, 120L), median_seconds = NA_real_, compiled_share = NA_real_,
                          dense_algebra_share = NA_real_, top_r = ""))
    runs <- max(3L, min(25L, ceiling(2 / max(first, 1e-3))))
    if (first > 20) runs <- 1L
    trace <- tempfile()
    Rprof(trace, interval = 0.005, gc.profiling = TRUE)
    elapsed <- vapply(seq_len(runs), function(i) clock(), 0)
    Rprof(NULL)
    self <- tryCatch(summaryRprof(trace)$by.self, error = function(e) NULL)
    unlink(trace)
    names_self <- gsub('"', "", rownames(self))
    total <- sum(self$self.time)
    # The profiler can credit compiled time to the wrapper function's name.
    compiled <- names_self %in% c(".Call", ".External", ".External2", ".Fortran", ".C", wrappers)
    algebra <- names_self %in% setdiff(native, c(".Call", ".External", ".External2", ".Fortran", ".C"))
    other <- self[!compiled & !algebra, ]
    data.frame(task = name, status = "completed", median_seconds = median(elapsed),
               compiled_share = sum(self$self.time[compiled]) / total,
               dense_algebra_share = sum(self$self.time[algebra]) / total,
               top_r = paste(sprintf("%s %.0f%%", gsub('"', "", rownames(other))[seq_len(min(4L, nrow(other)))],
                                     other$self.pct[seq_len(min(4L, nrow(other)))]), collapse = "; "))
}
rows <- list()
for (name in names(tasks)) {
    rows[[name]] <- profile_task(name, tasks[[name]])
    write.csv(do.call(rbind, rows), file.path(output, "summary.csv"), row.names = FALSE)
}
writeLines(capture.output(sessionInfo()), file.path(output, "session-info.txt"))
print(do.call(rbind, rows)[, 2:6], digits = 3L)
