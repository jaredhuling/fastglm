# Reproduce the README benchmarks from the repository root:
# Rscript benchmarks/benchmark-readme.R --library=/path/to/library
# Optional: --extra-library=/path/to/dependencies --validate-only --plot-only
# All fitting times include coefficient covariance computation. Profile
# intervals and the additional numerical-Hessian validation are not timed.
args <- commandArgs(TRUE)
argument <- function(name, default = NULL) {
    value <- args[startsWith(args, paste0("--", name, "="))]
    if (length(value)) sub(paste0("^--", name, "="), "", value[1L]) else default
}
lib <- argument("library", Sys.getenv("FASTGLM_README_LIB", ""))
extra <- argument("extra-library", Sys.getenv("FASTGLM_README_EXTRA_LIB", ""))
.libPaths(c(c(lib, extra)[nzchar(c(lib, extra))], .libPaths()))
outdir <- argument("output", "benchmarks/readme-results")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
library(fastglm)
required <- c("fastglm", "glm2", "speedglm", "MASS", "pscl", "brglm2",
              "microbenchmark", "ggplot2", "numDeriv")
stopifnot(all(vapply(required, requireNamespace, logical(1), quietly = TRUE)))
expected_version <- argument("expected-version")
if (!is.null(expected_version)) stopifnot(as.character(packageVersion("fastglm")) == expected_version)
if (nzchar(lib)) stopifnot(normalizePath(find.package("fastglm")) ==
                            normalizePath(file.path(lib, "fastglm")))
commit <- system2("git", c("rev-parse", "HEAD"), stdout = TRUE)
expected_commit <- argument("expected-commit")
if (!is.null(expected_commit)) stopifnot(startsWith(commit, expected_commit))

plot_results <- function() {
    timing <- read.csv(file.path(outdir, "raw-timings.csv"))
    focus <- subset(timing, study == "focused")
    labels <- c("glm.fit", "fastglm QR", "fastglm LLT", "fastglm LDLT", "fastglm auto")
    focus$method <- factor(focus$method, levels = rev(labels))
    p1 <- ggplot2::ggplot(focus, ggplot2::aes(x = elapsed_ms, y = method)) +
        ggplot2::geom_boxplot(fill = "#BDD7E7", width = .5, outlier.shape = NA) +
        ggplot2::geom_point(position = ggplot2::position_jitter(height = .10, width = 0, seed = 14),
                            alpha = .45, size = 1.4, colour = "#265B78") +
        ggplot2::stat_summary(fun = median, geom = "point", size = 2.7, colour = "#B44B30") +
        ggplot2::labs(x = "Elapsed time (milliseconds)", y = NULL,
            title = "Logistic regression: 10,000 observations and 100 predictors",
            subtitle = "25 fits per method; tolerance 1e-10; coefficient covariance included") +
        ggplot2::theme_bw(base_size = 12) +
        ggplot2::theme(panel.grid.minor = ggplot2::element_blank(),
            panel.grid.major.y = ggplot2::element_blank(), plot.title.position = "plot")
    ggplot2::ggsave("man/figures/README-glm_bench-1.png", p1, width = 9, height = 4.8,
                    dpi = 160, bg = "white")
    scaling <- read.csv(file.path(outdir, "scaling-summary.csv"))
    scaling$panel <- factor(scaling$panel, levels = c("Ordinary likelihood fits", "Mean bias-reduced fits"))
    p2 <- ggplot2::ggplot(scaling, ggplot2::aes(x = n, y = speedup, colour = model,
                                               shape = model, group = model)) +
        ggplot2::geom_hline(yintercept = 1, linetype = 2, colour = "grey50") +
        ggplot2::geom_line(linewidth = .75) + ggplot2::geom_point(size = 2.6) +
        ggplot2::scale_x_log10(breaks = c(1000, 10000, 100000), labels = c("1,000", "10,000", "100,000")) +
        ggplot2::scale_y_log10() +
        ggplot2::scale_shape_manual(values = c(16, 17, 15, 18, 8, 0, 1, 2, 5)) +
        ggplot2::scale_colour_manual(values = c("#0072B2", "#D55E00", "#009E73", "#CC79A7",
                                               "#6F4E7C", "#56B4E9", "#A6761D", "#444444", "#E69F00")) +
        ggplot2::facet_wrap(~panel, ncol = 2, scales = "free_y") +
        ggplot2::labs(x = "Number of observations", y = "Reference time / fastglm time",
            colour = NULL, shape = NULL,
            title = "Fitting-time comparisons across model families",
            subtitle = "Ratios above 1 favor fastglm; medians of 5 fits; covariance included",
            caption = "GLM references: fastest of glm.fit, glm.fit2, and speedglm.wfit. Other references: MASS, pscl, brglm2.\nPredictor counts differ by model; see the saved comparison and accuracy tables.") +
        ggplot2::theme_bw(base_size = 11) +
        ggplot2::theme(legend.position = "bottom", legend.box = "vertical",
            panel.grid.minor = ggplot2::element_blank(), plot.title.position = "plot",
            plot.caption = ggplot2::element_text(hjust = 0, size = 9)) +
        ggplot2::guides(colour = ggplot2::guide_legend(nrow = 3), shape = ggplot2::guide_legend(nrow = 3))
    ggplot2::ggsave("man/figures/README-model-benchmarks.png", p2, width = 11, height = 6.4,
                    dpi = 160, bg = "white")
}
if ("--plot-only" %in% args) {
    plot_results()
} else {
    tolerance <- 1e-10
    count_tolerance <- 1e-12
    maxit <- 100L
    accuracy <- list(); timings <- list(); summaries <- list(); inputs <- list()
    validate_only <- "--validate-only" %in% args
    complete <- function(fit, cls = NULL) {
        if (!is.null(cls)) class(fit) <- cls
        fit$.readme_covariance <- stats::vcov(fit)
        fit
    }
    converged <- function(fit) {
        if (inherits(fit, "speedglm")) return(isTRUE(fit$convergence))
        if (!is.null(fit$converged)) return(isTRUE(fit$converged))
        if (!is.null(fit$optim$convergence)) return(all(fit$optim$convergence == 0))
        FALSE
    }
    mean_values <- function(fit, X) {
        if (inherits(fit, c("hurdle", "zeroinfl", "fastglm_hurdle", "fastglm_zi")))
            return(as.numeric(predict(fit, type = "response")))
        as.numeric(fit$family$linkinv(drop(X %*% coef(fit))))
    }
    relative_error <- function(a, b) max(abs(a - b)) / max(abs(b), .Machine$double.eps)
    compare <- function(study, model, method, reference, fit, ref, X, seed) {
        stopifnot(converged(fit), converged(ref))
        v <- fit$.readme_covariance; vr <- ref$.readme_covariance
        stopifnot(all(is.finite(v)), all(is.finite(vr)),
                  min(eigen(v, symmetric = TRUE, only.values = TRUE)$values) > 0,
                  min(eigen(vr, symmetric = TRUE, only.values = TRUE)$values) > 0)
        a <- mean_values(fit, X); b <- mean_values(ref, X)
        stopifnot(length(a) == nrow(X), length(b) == nrow(X),
                  all(is.finite(a)), all(is.finite(b)),
                  length(coef(fit)) == length(coef(ref)))
        row <- data.frame(study, model, n = nrow(X), p = ncol(X), seed, method, reference,
            coefficient_error = max(abs(unname(coef(fit)) - unname(coef(ref)))),
            mean_error = max(abs(a - b)), mean_scaled_error = max(abs(a-b))/(1+max(abs(b))),
            covariance_relative_error = relative_error(v, vr),
            standard_error_relative_error = relative_error(sqrt(diag(v)), sqrt(diag(vr))),
            dispersion_relative_error = NA_real_, theta_relative_error = NA_real_,
            joint_covariance_relative_error = NA_real_,
            theta_se_conditional_relative_difference = NA_real_,
            converged = TRUE, passed = TRUE)
        if (inherits(fit, "fastglm_firth")) {
            row$dispersion_relative_error <- abs(fit$dispersion-ref$dispersion)/max(abs(ref$dispersion),1e-12)
        }
        if (inherits(fit, "fastglm_nb")) {
            row$theta_relative_error <- abs(fit$theta-ref$theta)/ref$theta
            pars <- c(coef(ref), log(ref$theta))
            negll <- function(par) -sum(dnbinom(ref$y, mu = exp(drop(X %*% par[seq_len(ncol(X))])),
                                               size = exp(tail(par, 1L)), log = TRUE))
            vh <- solve(numDeriv::hessian(negll, pars))
            scale <- c(rep(1, ncol(X)), ref$theta)
            vh <- vh * outer(scale, scale)
            row$joint_covariance_relative_error <- relative_error(vcov(fit, full = TRUE), vh)
            row$theta_se_conditional_relative_difference <- abs(fit$SE.theta-ref$SE.theta)/ref$SE.theta
        }
        row$passed <- row$coefficient_error < 5e-5 && row$mean_scaled_error < 1e-5 &&
            row$covariance_relative_error < 2e-4 &&
            (is.na(row$dispersion_relative_error) || row$dispersion_relative_error < 2e-6) &&
            (is.na(row$theta_relative_error) || row$theta_relative_error < 2e-5) &&
            (is.na(row$joint_covariance_relative_error) || row$joint_covariance_relative_error < 2e-4)
        accuracy[[length(accuracy) + 1L]] <<- row
        write.csv(do.call(rbind, accuracy), file.path(outdir, "accuracy.csv"), row.names = FALSE)
        if (!row$passed) stop("Accuracy check failed for ", model, ", n=", nrow(X), ", ", method,
                              "; inspect accuracy.csv before proceeding")
        row
    }
    measure <- function(study, model, method, n, p, fun, repetitions) {
        if (validate_only) return(NA_real_)
        elapsed <- microbenchmark::microbenchmark(fun(), times = repetitions)$time / 1e6
        timings[[length(timings) + 1L]] <<- data.frame(study, model, n, p, method,
            repetition = seq_along(elapsed), elapsed_ms = elapsed)
        write.csv(do.call(rbind, timings), file.path(outdir, "raw-timings.csv"), row.names = FALSE)
        median(elapsed)
    }
    fingerprint <- function(model, X, y, seed) {
        tmp <- tempfile(fileext = ".rds")
        saveRDS(list(X = X, y = y), tmp, compress = FALSE)
        inputs[[length(inputs) + 1L]] <<- data.frame(model, n = nrow(X), p = ncol(X), seed,
            data_md5 = unname(tools::md5sum(tmp)))
        unlink(tmp)
    }
    glm_functions <- function(X, y, family, start = NULL, all_fast = FALSE, references = TRUE) {
        intercept <- all(X[, 1L] == 1)
        f <- list()
        if (references) {
            f[["glm.fit"]] <- function() complete(stats::glm.fit(x = X, y = y, family = family,
                start = start, intercept = intercept, control = list(epsilon = tolerance, maxit = maxit)), c("glm", "lm"))
            f[["glm.fit2"]] <- function() complete(glm2::glm.fit2(x = X, y = y, family = family,
                start = start, intercept = intercept, control = list(epsilon = tolerance, maxit = maxit)), c("glm", "lm"))
            f[["speedglm.wfit"]] <- function() complete(speedglm::speedglm.wfit(y = y, X = X,
                family = family, start = start, intercept = intercept, acc = tolerance,
                maxit = maxit, method = "Cholesky"))
        }
        methods <- if (all_fast) list(QR = 0L, LLT = 2L, LDLT = 3L, auto = "auto") else list(auto = "auto")
        for (label in names(methods)) {
            f[[paste("fastglm", label)]] <- local({
                solver <- methods[[label]]
                function() complete(fastglm::fastglm(X, y, family, start = start,
                    method = solver, tol = tolerance, maxit = maxit, inference = "full"))
            })
        }
        f
    }

    # Original README's focused logistic design: intentionally no intercept.
    set.seed(123)
    n <- 10000L; p <- 100L
    X <- matrix(rnorm(n*p, sd = 3), n, p)
    y <- 1 * (drop(X[,1:25] %*% runif(25,-.1,.1)) > rnorm(n))
    fingerprint("Focused logistic", X, y, 123L)
    fun <- glm_functions(X, y, binomial(), all_fast = TRUE)
    fun <- fun[c("glm.fit", "fastglm QR", "fastglm LLT", "fastglm LDLT", "fastglm auto")]
    fitted <- lapply(fun, function(f) f())
    focused <- list()
    for (method in names(fun)) {
        a <- compare("focused", "Logistic", method, "glm.fit", fitted[[method]], fitted[[1L]], X, 123L)
        focused[[method]] <- data.frame(method,
            median_ms = measure("focused", "Logistic", method, n, p, fun[[method]], 25L),
            coefficient_error = a$coefficient_error, mean_error = a$mean_error)
    }
    write.csv(do.call(rbind, focused), file.path(outdir, "focused-summary.csv"), row.names = FALSE)
    cat("Focused logistic accuracy", if (validate_only) "validated" else "and timing complete", "\n")

    models <- c("Logistic", "Poisson", "Negative binomial", "Hurdle Poisson", "Zero-inflated Poisson",
                "Firth binomial-logit", "Firth Poisson-log", "Firth Gamma-log", "Firth Gaussian-identity")
    for (n in c(1000L, 10000L, 100000L)) for (m in seq_along(models)) {
        model <- models[m]
        seed <- 420000L + n + m
        set.seed(seed)
        p <- if (m <= 2L) 30L else if (m == 3L) 4L else if (m <= 5L) 3L else 6L
        X <- cbind(1, matrix(rnorm(n*(p-1L)), n))
        colnames(X) <- c("(Intercept)", paste0("x", seq_len(p-1L)))
        beta <- c(.3, seq(.3,-.2,length.out=p-1L)/sqrt(p-1L))
        eta <- drop(X %*% beta)
        family <- switch(as.character(m), "1"=binomial(), "2"=poisson(),
                         "6"=binomial(), "7"=poisson(), "8"=Gamma("log"), "9"=gaussian(), NULL)
        if (m %in% c(1L,6L)) y <- rbinom(n,1,plogis(eta))
        if (m %in% c(2L,7L)) y <- rpois(n,exp(eta))
        if (m == 3L) y <- rnbinom(n,mu=exp(eta),size=1.7)
        if (m %in% 4:5) {
            lambda <- exp(.6+.3*X[,2L]-.2*X[,3L])
            if (m == 4L) {
                positive <- rbinom(n,1,plogis(.3+.25*X[,2L]+.1*X[,3L]))
                y <- positive * qpois(exp(-lambda)+(1-exp(-lambda))*runif(n),lambda)
            } else {
                inflation <- rbinom(n,1,plogis(-1+.2*X[,2L]+.15*X[,3L]))
                y <- ifelse(inflation==1L,0L,rpois(n,lambda))
            }
        }
        if (m == 8L) y <- rgamma(n,shape=2,scale=exp(eta)/2)
        if (m == 9L) y <- rnorm(n,eta,sd=.8)
        fingerprint(model,X,y,seed)
        dat <- data.frame(y=y,X[,-1L,drop=FALSE])
        form <- reformulate(colnames(X)[-1L],response="y")
        if (m <= 2L) {
            start <- c(family$linkfun(mean(y)),rep(0,p-1L))
            fun <- glm_functions(X,y,family,start=start)
        } else if (m == 3L) {
            fun <- list(
                "MASS::glm.nb" = function() complete(MASS::glm.nb(form,data=dat,
                    control=glm.control(epsilon=tolerance,maxit=maxit))),
                "fastglm_nb" = function() complete(fastglm_nb(X,y,tol=tolerance,maxit=maxit,
                    outer.tol=tolerance,outer.maxit=maxit,theta.tol=tolerance,theta.maxit=maxit)))
        } else if (m == 4L) {
            fun <- list(
                "pscl::hurdle" = function() complete(pscl::hurdle(form,data=dat,dist="poisson",
                    control=pscl::hurdle.control(maxit=1000L,reltol=count_tolerance))),
                "fastglm_hurdle" = function() complete(fastglm_hurdle(form,data=dat,dist="poisson",
                    tol=count_tolerance,maxit=maxit,outer.tol=count_tolerance,outer.maxit=maxit)))
        } else if (m == 5L) {
            fun <- list(
                "pscl::zeroinfl" = function() complete(pscl::zeroinfl(form,data=dat,dist="poisson",
                    control=pscl::zeroinfl.control(maxit=1000L,reltol=count_tolerance))),
                "fastglm_zi" = function() complete(fastglm_zi(form,data=dat,dist="poisson",
                    tol=count_tolerance,maxit=maxit,em.tol=count_tolerance,em.maxit=1000L)))
        } else {
            start <- c(family$linkfun(mean(y)),rep(0,p-1L))
            fun <- list(
                "brglm2::brglmFit" = function() complete(brglm2::brglmFit(x=X,y=y,family=family,
                    start=start,control=brglm2::brglmControl(type="AS_mean",epsilon=tolerance,maxit=maxit)),
                    c("brglmFit","glm","lm")),
                "fastglm auto" = function() complete(fastglm(X,y,family,start=start,method="auto",
                    firth=TRUE,tol=tolerance,maxit=maxit,inference="full")))
        }
        fits <- lapply(fun,function(f) f())
        for (method in names(fun)) compare("scaling",model,method,names(fun)[1L],
                                          fits[[method]],fits[[1L]],X,seed)
        elapsed <- vapply(names(fun),function(method)
            measure("scaling",model,method,n,p,fun[[method]],5L),numeric(1))
        ref_names <- names(fun)[!startsWith(names(fun),"fastglm")]
        reference <- if (validate_only) ref_names[1L] else ref_names[which.min(elapsed[ref_names])]
        fast_name <- names(fun)[startsWith(names(fun),"fastglm")]
        summaries[[length(summaries)+1L]] <- data.frame(model,n,p,fastglm_ms=elapsed[fast_name],
            reference,reference_ms=elapsed[reference],speedup=elapsed[reference]/elapsed[fast_name],
            fastglm_method=fast_name,panel=if(m<=5L) "Ordinary likelihood fits" else "Mean bias-reduced fits",
            repetitions=5L,seed)
        write.csv(do.call(rbind,summaries),file.path(outdir,"scaling-summary.csv"),row.names=FALSE)
        cat(model,"n=",n,if(validate_only) "validated" else "complete","\n")
    }
    write.csv(do.call(rbind,inputs),file.path(outdir,"inputs.csv"),row.names=FALSE)
    packages <- data.frame(package=required,version=vapply(required,function(p)as.character(packageVersion(p)),character(1)),
                           path=vapply(required,find.package,character(1)))
    write.csv(packages,file.path(outdir,"packages.csv"),row.names=FALSE)
    protocol <- c(
        "Focused data reproduce README seed 123: n=10000,p=100,normal predictors sd=3,25-predictor latent response plus normal noise; no intercept.",
        "Focused fits use each implementation's observable family initialization; no coefficient starts are supplied.",
        "Scaling GLM/Firth fits use a common observable intercept-only start, family$linkfun(mean(y)), with zero slopes.",
        "Dedicated count models retain their observable package initializers; no simulation-truth starts are used.",
        "GLM/Firth/NB tolerances are 1e-10; hurdle/ZI tolerances are 1e-12 for both implementations to obtain comparable fitted-mean accuracy. GLM/Firth/inner/size caps100; EM/BFGS caps1000. Stopping criteria differ across solvers.",
        "speedglm.wfit uses named y/X arguments, acc=1e-10, and Cholesky; no tol argument is supplied.",
        "Every timed wrapper computes coefficient covariance. No profile confidence intervals are timed.",
        "GLM/Firth comparisons use matrix interfaces. MASS::glm.nb uses a formula interface versus fastglm_nb's matrix interface; both two-component implementations use formulas.",
        "NB coefficient covariance follows the expected-information GLM convention. fastglm additionally computes joint observed beta/theta covariance; that target is checked against a numerical Hessian outside timing.",
        "MASS's theta standard error is conditional; its difference is reported separately and is not treated as an error in the joint target.",
        "p counts all design columns, including the intercept for scaling cases; hurdle/ZI use p columns in each component.",
        "Five warmed fitting repetitions per scaling method;25 per focused method. Standard GLM reference is the lowest measured median among glm.fit,glm.fit2,speedglm.wfit.",
        "Fixed seeds are reset for every dataset; input fingerprints, raw timing rows, and accuracy checks are retained. Performance is machine- and design-specific."
    )
    write.csv(data.frame(item=seq_along(protocol),description=protocol),file.path(outdir,"protocol.csv"),row.names=FALSE)
    dll <- file.path(find.package("fastglm"),"libs","fastglm.so")
    metadata <- list(timestamp_utc=format(Sys.time(),tz="UTC",usetz=TRUE),source_commit=commit,
        fastglm_version=as.character(packageVersion("fastglm")),installed_path=find.package("fastglm"),
        installed_binary_md5=unname(tools::md5sum(dll)),script_md5=unname(tools::md5sum("benchmarks/benchmark-readme.R")),
        R=R.version.string,system=Sys.info(),threads=Sys.getenv(c("OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")),
        packages=packages,protocol=protocol,validation_only=validate_only)
    saveRDS(metadata,file.path(outdir,"metadata.rds"))
    writeLines(capture.output(sessionInfo()),file.path(outdir,"session-info.txt"))
    if (!validate_only) plot_results()
}
