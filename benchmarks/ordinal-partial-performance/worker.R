args <- commandArgs(trailingOnly = TRUE); stopifnot(length(args) == 4L)
mode <- match.arg(args[1L], c("partial", "common")); out <- args[3L]
.libPaths(c(normalizePath(args[2L]), .libPaths())); library(fastglm)
if (mode == "partial") stopifnot(requireNamespace("ordinal", quietly = TRUE))
deadline <- as.numeric(args[4L]); remaining <- function() deadline - as.numeric(Sys.time())
writeLines(capture.output(sessionInfo()), file.path(out, "session.txt"))
files <- c(system.file("DESCRIPTION", package = "fastglm"),
    list.files(system.file("libs", package = "fastglm"), full.names = TRUE),
    list.files(system.file("R", package = "fastglm"), full.names = TRUE),
    file.path("benchmarks/ordinal-partial-performance", c("protocol.txt", "run.R", "worker.R", "summarize.R")))
write.csv(data.frame(path = files, md5 = unname(tools::md5sum(files))),
          file.path(out, "hashes.csv"), row.names = FALSE)
records <- inputs <- solutions <- list()
record <- function(row) {
    records[[length(records) + 1L]] <<- row
    write.csv(do.call(rbind, records), file.path(out, "fits.csv"), row.names = FALSE)
}
diagnose <- function(fit, covariance, engine, dat) {
    if (engine == "fastglm") {
        probability <- predict(fit)
        converged <- isTRUE(fit$converged)
        score <- fit$scaled.score; step <- fit$predictor.step
        condition <- fit$information.rcond; iterations <- fit$iterations
        evaluations <- fit$evaluations; fallbacks <- fit$fallbacks
    } else {
        probability <- predict(fit, newdata = dat[c("x1", "x2", "z", "off")], type = "prob")$fit
        converged <- isTRUE(fit$convergence$code == 0L)
        score <- max(abs(fit$gradient)); step <- NA_real_
        condition <- rcond(fit$Hessian); iterations <- sum(fit$niter)
        evaluations <- NA_real_; fallbacks <- NA_real_
    }
    positive <- all(is.finite(covariance)) &&
        min(eigen(covariance, symmetric = TRUE, only.values = TRUE)$values) > 0
    valid <- all(is.finite(probability)) && min(probability) > 0 &&
        max(abs(rowSums(probability) - 1)) < 1e-10
    solved <- converged && (engine != "fastglm" || (score < 1e-7 && step < 1e-7))
    list(probability = probability, converged = converged, score = score, step = step,
         condition = condition, iterations = iterations, evaluations = evaluations,
         fallbacks = fallbacks, covariance.positive = positive, probability.valid = valid,
         valid = solved && positive && valid, loglik = as.numeric(logLik(fit)))
}
for (n in if (mode == "partial") c(10000L, 100000L) else 100000L) for (link.index in 1:3) {
    link <- c("logit", "probit", "cloglog")[link.index]
    seed <- (if (mode == "partial") 761000L else 881000L) + n + 101L * link.index
    set.seed(seed)
    dat <- data.frame(x1 = runif(n, -1, 1), x2 = runif(n, -1, 1), z = runif(n, -1, 1))
    dat$off <- .1 * dat$x1 * dat$z
    alpha <- c(-1.1, .1, 1.3); gamma <- if (mode == "partial") c(.2, -.15, .3) else rep(.25, 3L)
    eta <- matrix(alpha, n, 3L, byrow = TRUE) - .45 * dat$x1 + .35 * dat$x2 -
        dat$z %o% gamma - dat$off
    stopifnot(min(eta[, 2:3] - eta[, 1:2]) > .75)
    cumulative <- switch(link, logit = plogis(eta), probit = pnorm(eta), cloglog = -expm1(-exp(eta)))
    u <- runif(n); dat$y <- ordered(1L + rowSums(u > cumulative), levels = 1:4)
    tmp <- tempfile(tmpdir = out); saveRDS(dat, tmp)
    inputs[[length(inputs) + 1L]] <- data.frame(n = n, link = link, seed = seed,
        data.md5 = unname(tools::md5sum(tmp)), counts = paste(tabulate(as.integer(dat$y), 4L), collapse = ";"))
    unlink(tmp)
    write.csv(do.call(rbind, inputs), file.path(out, "inputs.csv"), row.names = FALSE)
    ml.reference <- NULL
    for (method in c("ML", "meanBR", if (mode == "partial") "clm ML")) {
        engine <- if (method == "clm ML") "clm" else "fastglm"
        br <- method == "meanBR"; estimation <- if (br) "meanBR" else "ML"
        invoke <- if (engine == "clm") function() ordinal::clm(y ~ x1 + x2 + offset(off),
            nominal = ~ z, data = dat, link = link,
            control = ordinal::clm.control(gradTol = 1e-7, maxIter = 200L)) else if (mode == "partial")
            function() fastglm_ordinal(y ~ x1 + x2 + z + offset(off), data = dat,
                nonparallel = ~ z, firth = br, link = link, tol = 1e-8, maxit = 200L) else
            function() fastglm_ordinal(y ~ x1 + x2 + z + offset(off), data = dat,
                firth = br, link = link, tol = 1e-8, maxit = 200L)
        target <- 3L; repetition <- 0L
        repeat {
            row <- data.frame(mode = mode, n = n, K = 4L, link = link, engine = engine,
                estimation = estimation, repetition = repetition,
                phase = if (repetition == 0L) "warmup" else "measured", requested.repeats = target,
                seconds = NA_real_, status = "budget skipped", converged = FALSE,
                iterations = NA_real_, evaluations = NA_real_, fallbacks = NA_real_,
                score = NA_real_, predictor.step = NA_real_, information.rcond = NA_real_,
                covariance.positive = FALSE, minimum.probability = NA_real_, normalization.error = NA_real_,
                loglik = NA_real_, reference.probability.error = NA_real_, reference.loglik.error = NA_real_,
                message = "")
            if (remaining() < 2) { record(row); break }
            warnings <- character(); fit <- NULL; covariance <- NULL; gc()
            setTimeLimit(elapsed = max(1, remaining()), transient = TRUE)
            timing <- system.time(value <- tryCatch(withCallingHandlers({
                fit <- invoke(); covariance <- vcov(fit); TRUE
            }, warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }),
                error = identity))
            setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE)
            row$seconds <- unname(timing[["elapsed"]]); row$message <- paste(warnings, collapse = " | ")
            if (inherits(value, "error")) {
                row$status <- "error"; row$message <- paste(row$message, conditionMessage(value), sep = " | ")
                record(row); break
            }
            d <- tryCatch(diagnose(fit, covariance, engine, dat), error = identity)
            if (inherits(d, "error")) {
                row$status <- "diagnostic error"; row$message <- paste(row$message, conditionMessage(d), sep = " | ")
                record(row); break
            }
            row$converged <- d$converged; row$iterations <- d$iterations
            row$evaluations <- d$evaluations; row$fallbacks <- d$fallbacks
            row$score <- d$score; row$predictor.step <- d$step; row$information.rcond <- d$condition
            row$covariance.positive <- d$covariance.positive
            row$minimum.probability <- min(d$probability)
            row$normalization.error <- max(abs(rowSums(d$probability) - 1)); row$loglik <- d$loglik
            agreement <- TRUE
            if (engine == "fastglm" && !br && repetition == 0L) ml.reference <- d
            if (engine == "clm") {
                agreement <- !is.null(ml.reference) && isTRUE(ml.reference$valid)
                if (agreement) {
                    row$reference.probability.error <- max(abs(d$probability - ml.reference$probability))
                    row$reference.loglik.error <- abs(d$loglik - ml.reference$loglik)
                    agreement <- row$reference.probability.error < 2e-6 && row$reference.loglik.error < 1e-5
                }
            }
            row$status <- if (d$valid && agreement && !length(warnings)) "ok" else "invalid or warning"
            record(row)
            key <- paste(n, link, method, sep = "-")
            solutions[[key]] <- list(coefficients = coef(fit), covariance = covariance, loglik = d$loglik,
                probabilities = d$probability[unique(round(seq(1, n, length.out = 25L))), , drop = FALSE],
                valid = row$status == "ok")
            saveRDS(solutions, file.path(out, "solutions.rds"))
            if (row$status != "ok") break
            if (repetition == 0L && 3 * row$seconds > .75 * remaining()) target <- 1L
            if (repetition == 0L && row$seconds > .75 * remaining()) {
                row$phase <- "measured"; row$repetition <- 1L; row$status <- "budget skipped"
                row$seconds <- NA_real_; row$message <- "Warm-up predicts insufficient remaining budget"
                record(row); break
            }
            if (repetition >= target) break
            repetition <- repetition + 1L
        }
        cat(n, link, method, "finished\n"); flush.console()
    }
}
