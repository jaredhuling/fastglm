# Compare ordinal fitting times with other packages that fit the same model.
# Run from the repository root:
# Rscript benchmarks/benchmark-ordinal-comparison.R --library=/path/to/library
# Options: --extra-library=dir --output=dir --summary=dir --figure=file --quick
#          --merge-only --plot-only --budget=seconds
#
# Each cell runs in its own process, one at a time, and writes one results file,
# so an interrupted study resumes where it stopped. A reference time is compared
# only when that fit agrees with fastglm. Mean bias-reduced cumulative-link fits
# and streamed fits are timed against fastglm's own ML and dense fits because no
# comparator here fits those models: brglm2::bracl() is an adjacent-category model.
args <- commandArgs(TRUE)
argument <- function(name, default = NULL) {
    value <- args[startsWith(args, paste0("--", name, "="))]
    if (length(value)) sub(paste0("^--", name, "="), "", value[1L]) else default
}
quick <- "--quick" %in% args
merge_only <- "--merge-only" %in% args
lib <- argument("library", "")
extra <- argument("extra-library", "")
.libPaths(c(c(lib, extra)[nzchar(c(lib, extra))], .libPaths()))
summary_dir <- argument("summary", if (quick) tempfile("ordinal-comparison-") else "benchmarks/ordinal-comparison")
figure_file <- argument("figure", if (quick) tempfile(fileext = ".png") else "man/figures/README-ordinal-benchmarks.png")
plot_only <- "--plot-only" %in% args
raw_dir <- argument("output", file.path(summary_dir, "raw"))
budget <- as.numeric(argument("budget", if (quick) "120" else "1800"))
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1")

packages <- c("fastglm", "MASS", "ordinal", "VGAM", "rms", "WeightIt")
stopifnot(all(vapply(c(packages, "microbenchmark"), requireNamespace, logical(1), quietly = TRUE)))
if (nzchar(lib)) stopifnot(normalizePath(find.package("fastglm")) == normalizePath(file.path(lib, "fastglm")))
tolerance <- 1e-8
maxit <- 500L
loglik_tolerance <- 1e-4
slope_tolerance <- 1e-3

# Hash the label: a sum of character codes gives many replicates the same seed.
make_seed <- function(...) {
    hash <- tools::md5sum(bytes = charToRaw(paste(..., sep = "|")))
    strtoi(substr(hash, 1L, 7L), 16L)
}

# DGP1 has common slopes. DGP2 draws from a partial proportional-odds model whose
# x2 and x3 slopes change across thresholds slowly enough that cumulative
# probabilities stay ordered except in extreme predictor tails.
simulate <- function(n, p, K, link, nonparallel, seed) {
    set.seed(seed)
    x <- matrix(rnorm(n * p), n, p, dimnames = list(NULL, paste0("x", seq_len(p))))
    eta <- drop(x %*% (0.6 * (-1)^seq_len(p) / sqrt(seq_len(p))))
    forward <- switch(link, logit = plogis, probit = pnorm, cloglog = function(z) -expm1(-exp(z)))
    inverse <- switch(link, logit = qlogis, probit = qnorm, cloglog = function(u) log(-log1p(-u)))
    alpha <- inverse(seq_len(K - 1L) / K)
    u <- runif(n)
    code <- integer(n) + K
    for (j in rev(seq_len(K - 1L))) {
        shift <- if (nonparallel) 0.08 * (j - K / 2) * (x[, 2L] - x[, 3L]) else 0
        code[u <= forward(alpha[j] - eta - shift)] <- j
    }
    list(x = x, y = ordered(code, levels = seq_len(K)))
}

ordinal_loglik <- function(alpha, slopes, x, y, link) {
    forward <- switch(link, logit = plogis, probit = pnorm, cloglog = function(z) -expm1(-exp(z)))
    eta <- drop(x %*% slopes)
    bounds <- c(-Inf, alpha, Inf)
    code <- as.integer(y)
    sum(log(forward(bounds[code + 1L] - eta) - forward(bounds[code] - eta)))
}

# A method returns log likelihood, common slopes on the fastglm sign convention,
# covariance, and convergence. Fitting and covariance are both inside the timing.
METHOD_REGISTRY <- list(
    fastglm = list(fit = function(x, y, link, nonparallel) {
        fit <- fastglm::fastglm_ordinal(x, y, link = link, tol = tolerance, maxit = maxit,
            nonparallel = if (nonparallel) c("x2", "x3"))
        fit$.covariance <- vcov(fit)
        fit
    }, read = function(fit, x, nonparallel) list(loglik = as.numeric(logLik(fit)),
        slopes = if (nonparallel) fit$beta else coef(fit)[colnames(x)],
        covariance = fit$.covariance, converged = isTRUE(fit$converged)),
    links = c("logit", "probit", "cloglog"), partial = TRUE),
    `MASS::polr` = list(fit = function(x, y, link, nonparallel) {
        method <- c(logit = "logistic", probit = "probit", cloglog = "cloglog")[[link]]
        fit <- MASS::polr(y ~ x, method = method, Hess = TRUE, control = list(reltol = 1e-12, maxit = maxit))
        fit$.covariance <- vcov(fit)
        fit
    }, read = function(fit, x, nonparallel) list(loglik = as.numeric(logLik(fit)),
        slopes = setNames(coef(fit), colnames(x)), covariance = fit$.covariance,
        converged = identical(fit$convergence, 0L)),
    links = c("logit", "probit", "cloglog"), partial = FALSE),
    `ordinal::clm` = list(fit = function(x, y, link, nonparallel) {
        # An absolute gradient tolerance cannot be reached at large n, so clm keeps
        # its defaults. The agreement check below still guards the estimates.
        control <- ordinal::clm.control(maxIter = maxit)
        fit <- if (nonparallel) {
            common <- x[, setdiff(colnames(x), c("x2", "x3")), drop = FALSE]
            varying <- x[, c("x2", "x3"), drop = FALSE]
            ordinal::clm(y ~ common, nominal = ~varying, link = link, control = control)
        } else ordinal::clm(y ~ x, link = link, control = control)
        fit$.covariance <- vcov(fit)
        fit
    }, read = function(fit, x, nonparallel) {
        b <- coef(fit)
        slopes <- b[!is.na(b) & !grepl("\\|", names(b))]
        names(slopes) <- sub("^(common|x)", "", names(slopes))
        list(loglik = as.numeric(logLik(fit)), slopes = slopes, covariance = fit$.covariance,
             converged = fit$convergence$code == 0L)
    }, links = c("logit", "probit", "cloglog"), partial = TRUE),
    `VGAM::vglm` = list(fit = function(x, y, link, nonparallel) {
        family <- if (nonparallel) VGAM::cumulative(link = paste0(link, "link"),
                parallel = FALSE ~ xx2 + xx3) else
            VGAM::cumulative(link = paste0(link, "link"), parallel = TRUE)
        d <- data.frame(y = y, x)
        names(d) <- c("y", paste0("x", colnames(x)))
        formula <- reformulate(paste0("x", colnames(x)), "y")
        fit <- VGAM::vglm(formula, family = family, data = d,
            control = VGAM::vglm.control(epsilon = tolerance, maxit = 100L))
        list(fit = fit, covariance = VGAM::vcov(fit))
    }, read = function(fit, x, nonparallel) {
        b <- VGAM::coef(fit$fit)
        slopes <- -b[!grepl("Intercept|:", names(b))]
        names(slopes) <- sub("^x", "", names(slopes))
        list(loglik = as.numeric(VGAM::logLik(fit$fit)), slopes = slopes, covariance = fit$covariance,
             converged = isTRUE(fit$fit@iter < 100L))
    # The working design has about n(K-1) rows and K-1+p columns, which would
    # reach 160 GB at n = 20,000 and K = 1,000. Memory limits this comparator.
    }, links = c("logit", "probit", "cloglog"), partial = TRUE, max_categories = 30L),
    `rms::orm` = list(fit = function(x, y, link, nonparallel) {
        family <- c(logit = "logistic", probit = "probit")[[link]]
        d <- data.frame(y = as.integer(y), x)
        fit <- rms::orm(reformulate(colnames(x), "y"), data = d, family = family, eps = 1e-8, maxit = 100L)
        fit$.covariance <- vcov(fit, intercepts = "all")
        fit
    }, read = function(fit, x, nonparallel) list(loglik = -fit$deviance[[2L]] / 2,
        slopes = coef(fit)[colnames(x)], covariance = fit$.covariance, converged = isTRUE(fit$fail == FALSE)),
    links = c("logit", "probit"), partial = FALSE),
    `WeightIt::ordinal_weightit` = list(fit = function(x, y, link, nonparallel) {
        d <- data.frame(y = y, x)
        fit <- WeightIt::ordinal_weightit(reformulate(colnames(x), "y"), data = d, link = link,
            vcov = "const", control = list(reltol = 1e-12, maxit = maxit))
        fit$.covariance <- vcov(fit)
        fit$.link <- link
        fit
    }, read = function(fit, x, nonparallel) {
        b <- coef(fit)
        slopes <- b[colnames(x)]
        # This class has no logLik() method, so the value is computed outside the timing.
        list(loglik = ordinal_loglik(unname(b[grepl("\\|", names(b))]), slopes, x, fit$y, fit$.link),
             slopes = slopes, covariance = fit$.covariance, converged = !isFALSE(fit$converged))
    }, links = c("logit", "probit", "cloglog"), partial = FALSE))

sizes <- if (quick) 1000L else c(1000L, 10000L, 100000L)
CELLS <- rbind(
    expand.grid(design = "DGP1", panel = "Common slopes", n = sizes, p = 10L, K = 5L,
                link = c("logit", "probit", "cloglog"), nonparallel = FALSE, stringsAsFactors = FALSE),
    if (!quick) expand.grid(design = "DGP1", panel = "Many categories", n = 20000L, p = 10L, K = c(30L, 300L, 1000L),
                            link = "logit", nonparallel = FALSE, stringsAsFactors = FALSE),
    expand.grid(design = "DGP2", panel = "Threshold-specific slopes", n = sizes, p = 10L, K = 5L,
                link = "logit", nonparallel = TRUE, stringsAsFactors = FALSE))

# Run in a forked child so a time limit also stops compiled code.
isolated <- function(work, seconds) {
    job <- parallel::mcparallel(work(), silent = TRUE)
    result <- parallel::mccollect(job, wait = FALSE, timeout = seconds)
    if (is.null(result)) {
        tools::pskill(job$pid, tools::SIGKILL)
        parallel::mccollect(job, wait = TRUE)
        return(list(status = "time limit"))
    }
    value <- result[[1L]]
    if (inherits(value, "try-error")) return(list(status = "error", message = substr(as.character(value), 1L, 300L)))
    value
}

run_method <- function(method, cell, data) {
    spec <- METHOD_REGISTRY[[method]]
    isolated(function() {
        fit_once <- function() spec$fit(data$x, data$y, cell$link, cell$nonparallel)
        conditions <- character()
        started <- Sys.time()
        first <- withCallingHandlers(fit_once(), warning = function(w) {
            conditions <<- c(conditions, conditionMessage(w))
            invokeRestart("muffleWarning")
        })
        first_seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))
        value <- spec$read(first, data$x, cell$nonparallel)
        rm(first)
        # A fit slower than five minutes counts once, using the fit already timed.
        repetitions <- if (quick) 2L else if (first_seconds > 300) 0L else if (first_seconds > 60) 1L else
            if (cell$n >= 100000L) 3L else if (cell$n >= 10000L) 5L else 25L
        elapsed <- if (repetitions) numeric(repetitions) else first_seconds
        for (r in seq_len(repetitions)) {
            gc(FALSE)
            begin <- microbenchmark::get_nanotime()
            suppressWarnings(fit_once())
            elapsed[r] <- (microbenchmark::get_nanotime() - begin) / 1e9
        }
        list(status = "completed", elapsed = elapsed, first_seconds = first_seconds, loglik = value$loglik,
             slopes = value$slopes, converged = value$converged,
             covariance_finite = all(is.finite(value$covariance)), warnings = unique(conditions))
    }, budget)
}

run_cell <- function(cell) {
    label <- sprintf("%s_%s_n%d_p%d_K%d", cell$design, cell$link, cell$n, cell$p, cell$K)
    file <- file.path(raw_dir, sprintf("results_%s.rds", label))
    if (file.exists(file)) return(invisible(file))
    seed <- make_seed(label)
    data <- simulate(cell$n, cell$p, cell$K, cell$link, cell$nonparallel, seed)
    methods <- Filter(function(m) cell$link %in% METHOD_REGISTRY[[m]]$links &&
        (!cell$nonparallel || METHOD_REGISTRY[[m]]$partial) &&
        (is.null(METHOD_REGISTRY[[m]]$max_categories) || cell$K <= METHOD_REGISTRY[[m]]$max_categories),
        names(METHOD_REGISTRY))
    results <- list()
    for (method in methods) {
        results[[method]] <- run_method(method, cell, data)
        message(label, " ", method, ": ", results[[method]]$status, " ",
                format(median(results[[method]]$elapsed), digits = 3L))
    }
    temporary <- paste0(file, ".tmp")
    # Concurrent work can distort timings, so each cell records the load it ended under.
    load <- sub(".*load averages?: *", "", system2("uptime", stdout = TRUE))
    saveRDS(list(cell = cell, label = label, seed = seed, results = results, load = load), temporary)
    file.rename(temporary, file)
    invisible(file)
}

# Bias reduction relative to ML, and streaming relative to a dense fit.
run_internal <- function() {
    file <- file.path(raw_dir, "results_internal.rds")
    if (file.exists(file)) return(invisible(file))
    out <- list()
    block <- 10000L
    for (n in if (quick) 2000L else c(100000L, 1000000L)) {
        label <- sprintf("DGP1_logit_n%d_p10_K5", n)
        data <- simulate(n, 10L, 5L, "logit", FALSE, make_seed("internal", label))
        chunks <- split(seq_len(n), ceiling(seq_len(n) / min(block, n / 4L)))
        source <- function() fastglm::fastglm_source(function(k) {
            i <- chunks[[k]]
            list(X = data$x[i, , drop = FALSE], y = data$y[i])
        }, length(chunks))
        ways <- list(
            `Dense ML` = function() fastglm::fastglm_ordinal(data$x, data$y, tol = tolerance),
            `Dense mean bias reduction` = function() fastglm::fastglm_ordinal(data$x, data$y, firth = TRUE, tol = tolerance),
            `Streamed ML` = function() fastglm::fastglm_ordinal_streaming(source(), tol = tolerance),
            `Streamed mean bias reduction` = function() fastglm::fastglm_ordinal_streaming(source(), firth = TRUE, tol = tolerance))
        for (way in names(ways)) {
            value <- isolated(function() {
                fit <- ways[[way]]()
                list(status = "completed", coefficients = coef(fit), loglik = as.numeric(logLik(fit)),
                     converged = isTRUE(fit$converged),
                     elapsed = vapply(seq_len(if (quick) 2L else 3L),
                                      function(r) system.time(ways[[way]](), gcFirst = TRUE)[["elapsed"]], 0))
            }, budget)
            out[[length(out) + 1L]] <- c(list(n = n, way = way, block = min(block, n / 4L)), value)
            message("internal ", label, " ", way, ": ", value$status)
        }
    }
    saveRDS(out, file)
    invisible(file)
}

merge_results <- function() {
    files <- setdiff(list.files(raw_dir, "^results_.*\\.rds$", full.names = TRUE),
                     file.path(raw_dir, "results_internal.rds"))
    rows <- lapply(files, function(file) {
        block <- readRDS(file)
        base <- block$results$fastglm
        do.call(rbind, lapply(names(block$results), function(method) {
            value <- block$results[[method]]
            completed <- identical(value$status, "completed")
            slope_error <- NA_real_
            if (completed && identical(base$status, "completed")) {
                shared <- intersect(names(base$slopes), names(value$slopes))
                if (length(shared) && !block$cell$nonparallel)
                    slope_error <- max(abs(base$slopes[shared] - value$slopes[shared]))
            }
            loglik_difference <- if (completed && identical(base$status, "completed")) abs(value$loglik - base$loglik) else NA_real_
            agrees <- completed && isTRUE(value$converged) && isTRUE(value$covariance_finite) &&
                isTRUE(loglik_difference < loglik_tolerance) &&
                (is.na(slope_error) || slope_error < slope_tolerance)
            data.frame(block$cell[c("design", "panel", "n", "p", "K", "link")], method,
                status = value$status, seed = block$seed, system_load = if (is.null(block$load)) NA_character_ else block$load,
                repetitions = if (completed) length(value$elapsed) else 0L,
                median_seconds = if (completed) median(value$elapsed) else NA_real_,
                minimum_seconds = if (completed) min(value$elapsed) else NA_real_,
                loglik_difference, slope_error, agrees,
                warnings = if (completed) paste(value$warnings, collapse = " | ") else
                    if (is.null(value$message)) "" else value$message,
                stringsAsFactors = FALSE)
        }))
    })
    summary <- do.call(rbind, rows)
    if (is.null(summary)) return(invisible(NULL))
    base <- summary[summary$method == "fastglm", c("design", "n", "K", "link", "median_seconds")]
    names(base)[5L] <- "fastglm_seconds"
    summary <- merge(summary, base, sort = FALSE)
    summary$speedup <- ifelse(summary$agrees, summary$median_seconds / summary$fastglm_seconds, NA_real_)
    summary <- summary[order(summary$panel, summary$link, summary$n, summary$K, summary$method), ]
    dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)
    write.csv(summary, file.path(summary_dir, "summary.csv"), row.names = FALSE)
    internal <- file.path(raw_dir, "results_internal.rds")
    if (file.exists(internal)) {
        values <- readRDS(internal)
        # Each streamed fit is compared with the dense fit of the same estimator and size.
        dense <- function(v) Filter(function(d) d$n == v$n &&
            d$way == sub("^Streamed", "Dense", v$way), values)[[1L]]
        table <- do.call(rbind, lapply(values, function(v) data.frame(n = v$n, way = v$way, block = v$block,
            status = v$status, converged = isTRUE(v$converged),
            median_seconds = if (identical(v$status, "completed")) median(v$elapsed) else NA_real_,
            loglik = if (identical(v$status, "completed")) v$loglik else NA_real_,
            coefficient_difference = if (identical(v$status, "completed"))
                max(abs(v$coefficients - dense(v)$coefficients)) else NA_real_)))
        write.csv(table, file.path(summary_dir, "internal-summary.csv"), row.names = FALSE)
        print(table)
    }
    versions <- data.frame(package = packages, version = vapply(packages, function(p) as.character(packageVersion(p)), ""),
                           row.names = NULL)
    write.csv(versions, file.path(summary_dir, "packages.csv"), row.names = FALSE)
    writeLines(capture.output(sessionInfo()), file.path(summary_dir, "session-info.txt"))
    # The fingerprint identifies the measured package source even before it is committed.
    files <- system2("git", c("ls-files", "-co", "--exclude-standard", "R", "src", "inst/include"), stdout = TRUE)
    write.csv(data.frame(item = c("commit", "package_source_md5", "system_load", "date_utc"),
        value = c(system2("git", c("rev-parse", "HEAD"), stdout = TRUE),
                  tools::md5sum(bytes = unlist(lapply(sort(files), function(f) readBin(f, "raw", file.size(f))))),
                  system2("uptime", stdout = TRUE), format(Sys.time(), tz = "UTC"))),
        file.path(summary_dir, "metadata.csv"), row.names = FALSE)
    display <- summary[, c("panel", "link", "n", "K", "method", "status", "median_seconds",
                           "speedup", "loglik_difference", "slope_error", "agrees")]
    print(display, row.names = FALSE, digits = 3L)
    invisible(summary)
}

# Time against sample size, with fastglm dominant and comparators recessive. A fit
# that disagrees with fastglm has no comparable time, so it is omitted.
plot_results <- function() {
    timing <- read.csv(file.path(summary_dir, "summary.csv"))
    timing <- timing[timing$agrees & timing$K == 5L, ]
    timing$facet <- ifelse(timing$panel == "Common slopes",
        paste0("Common slopes, ", timing$link), "Threshold-specific slopes, logit")
    timing$facet <- factor(timing$facet, levels = c("Common slopes, logit", "Common slopes, probit",
        "Common slopes, cloglog", "Threshold-specific slopes, logit"))
    order <- c("fastglm", "rms::orm", "ordinal::clm", "MASS::polr", "VGAM::vglm", "WeightIt::ordinal_weightit")
    timing$method <- factor(timing$method, levels = order)
    palette <- c("#0B5CAD", "#D1661F", "#1F9E8F", "#9A62C4", "#9C8200", "#D0628E")
    plot <- ggplot2::ggplot(timing, ggplot2::aes(n, median_seconds, colour = method, shape = method,
                                                 linewidth = method, group = method)) +
        ggplot2::geom_line() + ggplot2::geom_point(size = 2.4) +
        ggplot2::scale_x_log10(breaks = c(1000, 10000, 100000), labels = c("1,000", "10,000", "100,000")) +
        ggplot2::scale_y_log10(breaks = 10^(-3:1), labels = c("0.001", "0.01", "0.1", "1", "10")) +
        ggplot2::scale_colour_manual(values = palette, limits = order, drop = FALSE) +
        ggplot2::scale_shape_manual(values = c(16, 17, 15, 18, 3, 4), limits = order, drop = FALSE) +
        ggplot2::scale_linewidth_manual(values = c(1.2, rep(0.5, 5L)), limits = order, drop = FALSE) +
        ggplot2::facet_wrap(~facet, ncol = 2L) +
        ggplot2::labs(x = "Number of observations", y = "Median seconds per fit, log scale",
                      colour = NULL, shape = NULL, linewidth = NULL) +
        ggplot2::theme_bw(base_size = 12) +
        ggplot2::theme(panel.grid.minor = ggplot2::element_blank(), legend.position = "bottom",
                       panel.spacing.x = ggplot2::unit(1.1, "lines"), strip.background = ggplot2::element_blank(),
                       strip.text = ggplot2::element_text(hjust = 0))
    ggplot2::ggsave(figure_file, plot, width = 9, height = 6.6, dpi = 160, bg = "white")
    invisible(figure_file)
}

if (plot_only) {
    plot_results()
    quit(save = "no")
}
if (!merge_only) {
    for (package in packages) loadNamespace(package)
    for (i in seq_len(nrow(CELLS))) {
        run_cell(CELLS[i, ])
        gc(FALSE)
    }
    run_internal()
}
merge_results()
if (requireNamespace("ggplot2", quietly = TRUE)) plot_results()
