summarize_partial_performance <- function(out) {
    workers <- c("candidate-partial", "baseline-common", "candidate-common")
    fits <- do.call(rbind, lapply(workers, function(worker) {
        path <- file.path(out, worker, "fits.csv")
        if (!file.exists(path)) return(NULL)
        d <- read.csv(path); d$worker <- worker; d
    }))
    if (is.null(fits)) stop("No fitting records were produced; inspect worker logs")
    write.csv(fits, file.path(out, "all-fits.csv"), row.names = FALSE)
    summary <- do.call(rbind, lapply(split(fits, interaction(fits$worker, fits$n, fits$link,
        fits$engine, fits$estimation, drop = TRUE)), function(d) {
        good <- d$status == "ok" & d$phase == "measured"
        data.frame(worker = d$worker[1L], n = d$n[1L], link = d$link[1L],
            engine = d$engine[1L], estimation = d$estimation[1L], repeats = sum(good),
            median.seconds = if (any(good)) median(d$seconds[good]) else NA_real_,
            minimum.seconds = if (any(good)) min(d$seconds[good]) else NA_real_,
            maximum.seconds = if (any(good)) max(d$seconds[good]) else NA_real_,
            all.valid = all(d$status == "ok"), status = paste(unique(d$status), collapse = ";"))
    }))
    rownames(summary) <- NULL
    write.csv(summary, file.path(out, "summary.csv"), row.names = FALSE)
    comparisons <- list()
    for (n in c(10000L, 100000L)) for (link in c("logit", "probit", "cloglog")) {
        a <- summary[summary$worker == "candidate-partial" & summary$n == n &
                     summary$link == link & summary$estimation == "ML", ]
        native <- a[a$engine == "fastglm", ]; reference <- a[a$engine == "clm", ]
        valid <- nrow(native) == 1L && nrow(reference) == 1L &&
            native$all.valid && reference$all.valid && native$repeats > 0 && reference$repeats > 0
        comparisons[[length(comparisons) + 1L]] <- data.frame(comparison = "partial ML clm/fastglm",
            n = n, link = link, estimation = "ML", agreement = valid,
            ratio = if (valid) reference$median.seconds / native$median.seconds else NA_real_)
    }
    paths <- file.path(out, c("baseline-common", "candidate-common"), "solutions.rds")
    solutions <- lapply(paths, function(p) if (file.exists(p)) readRDS(p) else list())
    for (link in c("logit", "probit", "cloglog")) for (method in c("ML", "meanBR")) {
        key <- paste(100000L, link, method, sep = "-")
        b <- solutions[[1L]][[key]]; a <- solutions[[2L]][[key]]
        valid <- !is.null(a) && !is.null(b) && a$valid && b$valid
        if (valid) valid <- all(vapply(c("coefficients", "covariance", "loglik", "probabilities"),
            function(nm) max(abs(a[[nm]] - b[[nm]])) < 1e-9, logical(1)))
        btime <- summary[summary$worker == "baseline-common" & summary$link == link & summary$estimation == method, ]
        atime <- summary[summary$worker == "candidate-common" & summary$link == link & summary$estimation == method, ]
        valid <- valid && nrow(atime) == 1L && nrow(btime) == 1L &&
            atime$all.valid && btime$all.valid && atime$repeats > 0 && btime$repeats > 0
        comparisons[[length(comparisons) + 1L]] <- data.frame(comparison = "common candidate/baseline",
            n = 100000L, link = link, estimation = method, agreement = valid,
            ratio = if (valid) atime$median.seconds / btime$median.seconds else NA_real_)
    }
    write.csv(do.call(rbind, comparisons), file.path(out, "comparisons.csv"), row.names = FALSE)
    print(summary, row.names = FALSE)
    invisible(summary)
}
