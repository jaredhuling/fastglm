## Likelihood ratio comparison of nested models: anova(m0, m1, ...) for every
## fitted class. The engine (.fastglm_lrt) works from the comparison records
## of compare-core.R; the anova methods only collect the models and their
## labels.

#' Likelihood ratio tests for nested fastglm models
#'
#' `anova(m0, m1, ...)` compares two or more nested models fitted to the same
#' data by likelihood ratio tests, for every model class of the package:
#' generalized linear models (matrix, sparse, `big.matrix`, and streamed fits,
#' and formula fits), negative binomial, truncated, zero-inflated, hurdle,
#' two-part, ordinal (including partial proportional odds), and multinomial
#' fits.
#'
#' @param object a fitted model.
#' @param ... further fitted models. Other arguments named here are ignored
#'   with a warning, as in [stats::anova.glm()]. With no further model,
#'   `anova(object)` gives the sequential table of the terms of `object`
#'   (see [drop1.fastglm()]).
#' @param trace for `anova()` of one model, `TRUE` traces each refit of the
#'   sequence, after a line naming it; refits are otherwise silent.
#' @param test `"Chisq"` (or its synonym `"LRT"`) for a chi-square reference,
#'   or `"F"` for an F reference. The default is `"F"` for generalized linear
#'   models whose dispersion is estimated by moments (Gaussian, Gamma, inverse
#'   Gaussian, and quasi families) and `"Chisq"` otherwise, as in
#'   [stats::anova.glm()]. The F test is available only for those models.
#' @param dispersion for generalized linear models, the dispersion used to
#'   scale the deviance differences; by default the dispersion of the largest
#'   model (fixed at one for binomial and Poisson models), as in
#'   [stats::anova.glm()].
#'
#' @details
#' The models are sorted by their number of estimated parameters, the `df`
#' attribute of [logLik()] (coefficients plus cutpoints, the negative
#' binomial theta, and a dispersion estimated by maximum likelihood), and each
#' model is compared with the one before it. The statistic is
#' \eqn{2\{\ell(m_1) - \ell(m_0)\}}{2 (logLik(m1) - logLik(m0))} with degrees
#' of freedom equal to the difference in the number of parameters. For
#' generalized linear models the table is that of [stats::anova.glm()]
#' (residual degrees of freedom, residual deviance, and the deviance
#' difference scaled by the dispersion of the largest model); with a
#' dispersion estimated by moments the F statistic divides the scaled
#' deviance difference by its degrees of freedom and uses the residual degrees
#' of freedom of the largest model. A clearly negative statistic (the larger
#' model fits worse) gets no p-value and a warning.
#'
#' **Checks.** No p-value is given before these checks pass.
#' * Same data: the models must use the same number of observations, the same
#'   sum of prior weights, the same response, and the same offsets. These
#'   checks do not depend on the order of the rows. The response is compared
#'   through a fingerprint recorded at fit time (the number of observations,
#'   the sum of the weights, and the weighted sums of the response and of its
#'   square, to a relative tolerance of 1e-12); for ordinal and multinomial
#'   fits held in memory, through the weighted frequency of each category.
#'   Offsets are compared through the sums of the offset and of its square
#'   (per part for models with two linear predictors) and, for fits held in
#'   memory with rows confirmed to be in the same order (see Nesting), row by
#'   row. A failed data check is an error.
#' * Same model family: the same class of model, error distribution, and
#'   link, except for the pairs described below. A difference is an error.
#' * Nesting. For fits held in memory, each column of the smaller model's
#'   design (columns of estimated coefficients) must lie in the column space
#'   of the larger model's design, part by part (relative residual norm below
#'   1e-8, from a QR factorization computed over blocks of rows; for ordinal
#'   models the thresholds act as an intercept and a common slope is nested in
#'   the cutpoint-specific slopes). Coefficient names then need not match, so
#'   `y ~ poly(x, 2)` is nested in `y ~ poly(x, 3)` and `y ~ b` in
#'   `y ~ 0 + b + a` for a factor `b`. This needs the rows of the two fits in
#'   the same order: the response and weights must agree row by row, and the
#'   order is confirmed by informative row names of the model frames or
#'   designs, or by design columns of the same name that agree row by row. A
#'   failed check with confirmed order is an error. When the order is not
#'   confirmed and the check fails, or for streamed, `big.matrix`, or sparse
#'   fits, the coefficient names of the smaller model must be a subset of
#'   those of the larger model (with the variables of interactions in any
#'   order), and a warning says that nesting was checked by names only. The
#'   warning is not given for two streamed fits of the identical source, or
#'   of formula sources built from the identical callback and prototype,
#'   where a coefficient name determines its column. A matrix design without
#'   column names cannot be checked by names, and the test is then given with
#'   a warning. A failed nesting check is otherwise an error.
#'
#' **Poisson against negative binomial.** A Poisson model (a GLM fit, matrix,
#' formula, or streamed) can be compared with a negative binomial fit
#' ([fastglm_nb()]) whose mean model contains it, and likewise the Poisson and
#' negative binomial versions of a truncated, hurdle, or zero-inflated model.
#' The Poisson model is the negative binomial model at the boundary
#' 1/theta = 0 of the parameter space. When the larger model adds r ordinary
#' parameters besides theta, the statistic has the 50:50 mixture of
#' chi-square(r) and chi-square(r + 1) as its reference distribution, so the
#' p-value is \eqn{P(\chi^2_r > x)/2 + P(\chi^2_{r+1} > x)/2}{0.5 P(chisq_r > x) + 0.5 P(chisq_(r+1) > x)};
#' with the same mean model (r = 0) it is half the chi-square(1) p-value. The
#' printed table says so in a note. The negative binomial fit must estimate
#' theta and be the larger model.
#'
#' **Zero-inflated models** are not compared with models without zero
#' inflation: the inflation probability is on the boundary and the models are
#' not nested in the usual sense. Such a call is an error; compare these
#' models with [AIC()].
#'
#' **Two-part models** ([fastglm_twopart()]) are compared through
#' [logLik()], which maximizes the Gamma log likelihood of the positive part
#' over the shape given the fitted means (see [fastglm_twopart()]), so the
#' statistic is a likelihood ratio statistic and [AIC()] agrees with it.
#'
#' **Bias-reduced (Firth) fits** are compared by a penalized likelihood ratio
#' test: [fastglm_profile_test()] on the larger model tests jointly that the
#' coefficients absent from the smaller model are zero, keeping the larger
#' model's penalty, which is the nested comparison of
#' `logistf::anova.logistf(method = "nested")`. The difference of two
#' separately penalized fits is not this statistic and is never used. This is
#' available for binomial-logit Firth fits held in memory with a dense design,
#' when every coefficient of the smaller model appears by name in the larger
#' one; other bias-reduced fits have no likelihood ratio comparison (an error
#' points to [fastglm_wald_test()] and, where supported,
#' [fastglm_profile_test()]). A Firth fit is never compared with a maximum
#' likelihood fit.
#'
#' **Robust covariance** does not change a likelihood ratio test: a fit with
#' a robust or clustered `vcov` type gets the same test. For a comparison that
#' uses a robust covariance, use [fastglm_wald_test()].
#'
#' **Formula fits.** Calls with only formula `fastglm()` fits (and other `glm`
#' objects) are handed to [stats::anova.glm()] unchanged, with its tests
#' (including `test = "Rao"`), results, and messages. Bias-reduced formula
#' fits follow the Firth rule above, and a call that mixes a formula fit with
#' another class (for example a Poisson formula fit and a [fastglm_nb()] fit)
#' uses the checks above. R dispatches `anova()` on its first argument, so a
#' `stats::glm()` fit given first hands every model to `stats::anova.glm()`,
#' which drops the models that are not `glm` objects: put a fastglm fit
#' first.
#'
#' @returns An object of class `c("anova", "data.frame")` with a `heading`
#'   attribute. Generalized linear models: columns `Resid. Df`, `Resid. Dev`,
#'   `Df`, `Deviance`, and `Pr(>Chi)` or `F` and `Pr(>F)`. Other models:
#'   `#Df` (number of parameters), `LogLik`, `Df`, `Chisq`, and
#'   `Pr(>Chisq)`.
#' @examples
#' set.seed(1)
#' d <- data.frame(x1 = rnorm(300), x2 = rnorm(300))
#' d$y <- rnbinom(300, mu = exp(0.5 + 0.4 * d$x1), size = 2)
#' p0 <- fastglm(model.matrix(~ x1, d), d$y, family = poisson())
#' p1 <- fastglm(model.matrix(~ x1 + x2, d), d$y, family = poisson())
#' anova(p0, p1)
#' nb1 <- fastglm_nb(y ~ x1 + x2, data = d)
#' anova(p1, nb1)
#' @name anova.fastglm
#' @md
NULL

## Classes whose objects are fitted models for these methods.
.fastglm_model_classes <- c("fastglm", "fastglmFit", "glm", "fastglm_nb", "fastglm_truncated",
                            "fastglm_zi", "fastglm_hurdle", "fastglm_ordinal", "fastglm_vglm")

.fastglm_anova_collect <- function(object, dots, labels)
{
    is.model <- vapply(dots, function(x) inherits(x, .fastglm_model_classes), NA)
    named <- if (is.null(names(dots))) rep_len(FALSE, length(dots)) else names(dots) != ""
    drop <- !is.model | named
    if (any(drop))
        warning("the following arguments to 'anova' are not fitted models and are dropped: ",
                paste(labels[-1L][drop], collapse = ", "), call. = FALSE)
    list(models = c(list(object), dots[!drop]), labels = c(labels[1L], labels[-1L][!drop]))
}

.fastglm_anova_labels <- function(mc)
{
    c(deparse1(mc$object), vapply(as.list(mc$...), deparse1, character(1)))
}

#' @rdname anova.fastglm
#' @exportS3Method stats::anova fastglm
anova.fastglm <- function(object, ..., test = NULL, dispersion = NULL, trace = FALSE)
{
    mc <- match.call(expand.dots = FALSE)
    m <- .fastglm_anova_collect(object, list(...), .fastglm_anova_labels(mc))
    # One model: the sequential table (see ?drop1.fastglm).
    if (length(m$models) < 2L)
        return(.fastglm_anova_sequential(object, test = test, dispersion = dispersion, label = m$labels[1L],
                                         trace = trace))
    .fastglm_lrt(m$models, test = test, labels = m$labels, dispersion = dispersion)
}

#' @rdname anova.fastglm
#' @exportS3Method stats::anova fastglm_stream
anova.fastglm_stream <- anova.fastglm
#' @rdname anova.fastglm
#' @exportS3Method stats::anova fastglm_nb
anova.fastglm_nb <- anova.fastglm
#' @rdname anova.fastglm
#' @exportS3Method stats::anova fastglm_zi
anova.fastglm_zi <- anova.fastglm
#' @rdname anova.fastglm
#' @exportS3Method stats::anova fastglm_hurdle
anova.fastglm_hurdle <- anova.fastglm
#' @rdname anova.fastglm
#' @exportS3Method stats::anova fastglm_truncated
anova.fastglm_truncated <- anova.fastglm
#' @rdname anova.fastglm
#' @exportS3Method stats::anova fastglm_ordinal
anova.fastglm_ordinal <- anova.fastglm
#' @rdname anova.fastglm
#' @exportS3Method stats::anova fastglm_vglm
anova.fastglm_vglm <- anova.fastglm

## TRUE when a formula fit (or another glm) call must use the engine: one of
## the models is not a glm object, or one is a bias-reduced fit.
.fastglm_anova_needs_engine <- function(models)
{
    any(vapply(models, function(x) !inherits(x, "glm") || isTRUE(x$firth) ||
                   isTRUE(x$control$firth), NA))
}

## ---- The engine ---------------------------------------------------------------

## models: a list of fitted models; labels: their names in the call.
## Returns an anova data frame; see ?anova.fastglm.
.fastglm_lrt <- function(models, test = NULL, labels = NULL, dispersion = NULL)
{
    if (length(models) < 2L)
        stop("anova() needs two or more nested models fitted to the same data; ",
             "for tests of single terms use drop1()", call. = FALSE)
    if (is.null(labels)) labels <- paste("Model", seq_along(models))
    stats.glm <- vapply(models, function(m) inherits(m, "glm") && !inherits(m, "fastglmFit"), NA)
    if (any(stats.glm))
        stop(sprintf(paste0("'%s' is a stats::glm() fit, which is compared here only with formula fastglm() ",
                            "fits alone (through stats::anova.glm); refit it with fastglm() to compare it ",
                            "with the other models"), labels[which(stats.glm)[1L]]), call. = FALSE)
    if (!is.null(test)) {
        if (identical(test, "Rao"))
            stop("test = \"Rao\" is available only when formula fastglm() fits are compared among ",
                 "themselves (stats::anova.glm); use fastglm_score_test() for other fits", call. = FALSE)
        if (!is.character(test) || length(test) != 1L)
            stop("'test' must be \"Chisq\", \"LRT\", or \"F\"", call. = FALSE)
        test <- match.arg(test, c("Chisq", "LRT", "F"))
        if (test == "LRT") test <- "Chisq"
    }
    recs <- lapply(models, .fastglm_cmp_record)
    .fastglm_cmp_check_data(recs, labels)
    mode <- .fastglm_cmp_check_family(recs, labels)
    ord <- order(vapply(recs, .fastglm_cmp_npar, numeric(1)))
    models <- models[ord]; recs <- recs[ord]; labels <- labels[ord]
    pairs <- .fastglm_cmp_check_nesting(models, recs, labels, firth = mode == "profile")
    descr <- vapply(seq_along(models), function(i) .fastglm_cmp_describe(models[[i]], labels[i]), "")
    kinds <- vapply(recs, function(r) paste0(.fastglm_cmp_kind(r),
        if (r$class_family %in% c("truncated", "hurdle", "zi")) paste0(", ", r$family)), "")
    if (length(unique(kinds)) > 1L) descr <- paste0(descr, " (", kinds, ")")
    topnote <- paste0("Model ", format(seq_along(models)), ": ", descr, collapse = "\n")
    if (mode == "profile") return(.fastglm_lrt_firth(models, recs, pairs, topnote, test))
    if (mode == "glm") return(.fastglm_lrt_glm(models, recs, topnote, test, dispersion))
    if (!is.null(dispersion)) warning("'dispersion' applies only to generalized linear models and is ignored",
                                      call. = FALSE)
    if (identical(test, "F"))
        stop("an F test is available only for generalized linear models whose dispersion is ",
             "estimated by moments; use test = \"Chisq\"", call. = FALSE)
    .fastglm_lrt_loglik(models, recs, pairs, topnote)
}

## Number of estimated parameters; a negative binomial fit with theta held
## fixed does not estimate theta, although its logLik() df counts it.
.fastglm_cmp_npar <- function(rec)
{
    if (rec$class_family == "nb" && rec$dispersion == "fixed") rec$df - 1L else rec$df
}

## ---- Data checks ----------------------------------------------------------------

## TRUE when two numeric summaries agree to a relative tolerance (scaled by
## at least one, so that sums near zero compare absolutely).
.fastglm_cmp_close <- function(a, b, tol = 1e-12)
{
    length(a) == length(b) && all(abs(a - b) <= tol * pmax(abs(a), abs(b), 1))
}

## The offset fingerprints of two records, by part, with a "main" part
## matched to a "count" part (a single linear predictor against the count
## part of a two-part model).
.fastglm_cmp_offset_parts <- function(fp)
{
    if (is.null(fp)) return(NULL)
    if (!is.null(fp$main)) fp <- list(count = fp$main, zero = NULL)
    lapply(fp[c("count", "zero")], function(x) if (is.null(x)) c(so = 0, soo = 0) else x)
}

## Order-independent checks that the models use the same data: number of
## observations, sum of weights, response (fingerprint, or category
## frequencies by label for categorical responses held in memory), offsets.
.fastglm_cmp_check_data <- function(recs, labels)
{
    r1 <- recs[[1L]]
    for (i in seq_along(recs)[-1L]) {
        ri <- recs[[i]]
        n1 <- as.numeric(r1$nobs); ni <- as.numeric(ri$nobs)
        if (!isTRUE(abs(n1 - ni) <= 1e-12 * max(abs(n1), 1)))
            stop(sprintf(paste0("the models were not fitted to the same data: '%s' uses %s observations ",
                                "and '%s' uses %s"), labels[1L], format(n1), labels[i], format(ni)),
                 call. = FALSE)
        if (!anyNA(c(r1$wsum, ri$wsum)) &&
            abs(r1$wsum - ri$wsum) > 1e-12 * max(abs(r1$wsum), abs(ri$wsum), 1e-300))
            stop(sprintf(paste0("the models were not fitted with the same prior weights: the weights of ",
                                "'%s' sum to %s and those of '%s' to %s"),
                         labels[1L], format(r1$wsum, digits = 15), labels[i], format(ri$wsum, digits = 15)),
                 call. = FALSE)
        different <- function()
            stop(sprintf("the models were not fitted to the same data: '%s' and '%s' have different responses or weights",
                         labels[1L], labels[i]), call. = FALSE)
        if (!is.null(r1$cells) && !is.null(ri$cells)) {
            lev <- union(names(r1$cells), names(ri$cells))
            ca <- r1$cells[lev]; cb <- ri$cells[lev]; ca[is.na(ca)] <- 0; cb[is.na(cb)] <- 0
            if (!.fastglm_cmp_close(unname(ca), unname(cb))) different()
        } else {
            same <- .fastglm_same_fingerprint(r1$fingerprint, ri$fingerprint)
            if (is.na(same))
                warning(sprintf("the responses of '%s' and '%s' could not be compared (a fit did not retain its response)",
                                labels[1L], labels[i]), call. = FALSE)
            else if (!same) different()
        }
        oa <- .fastglm_cmp_offset_parts(r1$offset_fingerprint)
        ob <- .fastglm_cmp_offset_parts(ri$offset_fingerprint)
        if (is.null(oa) || is.null(ob)) {
            warning(sprintf("the offsets of '%s' and '%s' could not be compared (not recorded by the fit)",
                            labels[1L], labels[i]), call. = FALSE)
        } else if (!.fastglm_cmp_close(unlist(oa), unlist(ob))) {
            .fastglm_cmp_offset_error(labels[1L], labels[i])
        }
    }
    invisible(TRUE)
}

.fastglm_cmp_offset_error <- function(a, b)
    stop(sprintf(paste0("'%s' and '%s' have different offsets; an offset fixes part of the linear predictor, ",
                        "so the models are not nested as fitted"), a, b), call. = FALSE)

## ---- Model families -----------------------------------------------------------

## The role of a record in the Poisson / negative binomial rule: the
## distribution and the kind of model, or NA.
.fastglm_cmp_count_role <- function(rec)
{
    cf <- rec$class_family
    if (cf == "glm" && identical(rec$family, "poisson") && identical(rec$link, "log")) return("poisson:glm")
    if (cf == "nb" && identical(rec$link, "log")) return("negbin:glm")
    if (cf %in% c("truncated", "hurdle", "zi")) return(paste0(rec$family, ":", cf))
    NA_character_
}

## TRUE when two records form an allowed Poisson / negative binomial pair.
.fastglm_cmp_boundary_pair <- function(a, b)
{
    ra <- .fastglm_cmp_count_role(a); rb <- .fastglm_cmp_count_role(b)
    if (anyNA(c(ra, rb))) return(FALSE)
    sa <- strsplit(ra, ":", fixed = TRUE)[[1L]]; sb <- strsplit(rb, ":", fixed = TRUE)[[1L]]
    sa[2L] == sb[2L] && setequal(c(sa[1L], sb[1L]), c("poisson", "negbin")) &&
        identical(a$link, b$link)
}

## Checks the classes, families, links, and estimation methods; returns the
## comparison mode: "glm" (the anova.glm table), "loglik", or "profile".
.fastglm_cmp_check_family <- function(recs, labels)
{
    pen <- vapply(recs, function(r) if (is.na(r$penalized)) NA else r$penalized, NA)
    method <- vapply(recs, `[[`, "", "lr_method")
    if (any(method == "none")) {
        i <- which(method == "none")[1L]
        stop(sprintf(paste0("'%s' is a bias-reduced fit without a likelihood ratio comparison (only ",
                            "binomial-logit Firth fits held in memory with a dense design have one); ",
                            "use fastglm_wald_test(), or fastglm_profile_test() where it supports the fit"),
                     labels[i]), call. = FALSE)
    }
    firth <- method == "profile" | (!is.na(pen) & pen)
    if (any(firth) && !all(firth))
        stop(sprintf(paste0("'%s' is a bias-reduced (Firth) fit and '%s' a maximum likelihood fit; a penalized ",
                            "and an unpenalized likelihood cannot be compared"),
                     labels[which(firth)[1L]], labels[which(!firth)[1L]]), call. = FALSE)
    cf <- vapply(recs, `[[`, "", "class_family")
    if (any(cf == "zi") && !all(cf == "zi"))
        stop(sprintf(paste0("'%s' is zero-inflated and '%s' is not: the inflation probability is on the boundary ",
                            "of its space and the models are not nested in the usual sense, so no likelihood ",
                            "ratio test is given; compare them with AIC()"),
                     labels[which(cf == "zi")[1L]], labels[which(cf != "zi")[1L]]), call. = FALSE)
    r1 <- recs[[1L]]
    for (i in seq_along(recs)[-1L]) {
        ri <- recs[[i]]
        if (.fastglm_cmp_boundary_pair(r1, ri)) next
        if (r1$class_family != ri$class_family) {
            extra <- if (any(c(r1$class_family, ri$class_family) == "hurdle"))
                "; to choose between non-nested models, use AIC()" else ""
            stop(sprintf("'%s' (%s) and '%s' (%s) are different kinds of models and are not nested%s",
                         labels[1L], .fastglm_cmp_kind(r1), labels[i], .fastglm_cmp_kind(ri), extra),
                 call. = FALSE)
        }
        if (!identical(r1$family, ri$family))
            stop(sprintf("'%s' and '%s' have different families (%s and %s)",
                         labels[1L], labels[i], r1$family, ri$family), call. = FALSE)
        if (!identical(r1$link, ri$link))
            stop(sprintf("'%s' and '%s' have different links (%s and %s)", labels[1L], labels[i],
                         paste(r1$link, collapse = "/"), paste(ri$link, collapse = "/")), call. = FALSE)
    }
    if (all(firth)) return("profile")
    if (all(cf == "glm")) return("glm")
    "loglik"
}

.fastglm_cmp_kind <- function(rec)
{
    switch(rec$class_family,
           glm = paste(rec$family, "GLM"), nb = "negative binomial", zi = "zero-inflated",
           hurdle = "hurdle", twopart = "two-part", truncated = "truncated",
           ordinal = "ordinal", multinom = "multinomial", rec$class_family)
}

## ---- Nesting ------------------------------------------------------------------

## The response and prior weights of an in-memory fit, row by row, or NULL.
.fastglm_cmp_rows <- function(object)
{
    if (!is.null(object$source)) return(NULL)
    if (.fastglm_two_part(object)) {
        y <- .fastglm_cmp_twopart_y(object); w <- object$weights
    } else if (inherits(object, "fastglm_ordinal")) {
        y <- object$y; w <- object$weights
    } else {
        y <- object$y; w <- object$prior.weights
    }
    if (is.null(y)) return(NULL)
    if (is.factor(y)) y <- as.character(y)
    if (is.null(w)) w <- rep(1, NROW(y))
    list(y = y, w = as.numeric(w))
}

## Row names that identify the rows of an in-memory fit: those of the stored
## model frame, or of the stored design; NULL when absent or automatic
## (1, ..., n), which carry no information about the order of the rows.
.fastglm_cmp_rownames <- function(object)
{
    rn <- NULL
    if (is.data.frame(object$model)) {
        if (.row_names_info(object$model) > 0L) rn <- rownames(object$model)
    } else if (is.matrix(object$x)) rn <- rownames(object$x)
    if (is.null(rn) || identical(rn, as.character(seq_along(rn)))) return(NULL)
    rn
}

## Per-call memo of each model's rows, designs, and row names, so that the
## designs are built once in a call with several models.
.fastglm_cmp_memo <- function(models)
{
    env <- new.env(parent = emptyenv())
    get <- function(what, i, fun) {
        key <- paste0(what, i)
        if (!exists(key, envir = env, inherits = FALSE)) assign(key, list(fun(models[[i]])), envir = env)
        get0(key, envir = env, inherits = FALSE)[[1L]]
    }
    list(rows = function(i) get("rows", i, .fastglm_cmp_rows),
         designs = function(i) get("designs", i, .fastglm_cmp_designs),
         rownames = function(i) get("rownames", i, .fastglm_cmp_rownames))
}

## Whether the rows of two in-memory fits (indices i and j) are known to be
## in the same order: "confirmed" when the response and weights agree row by
## row and the rows are identified by the same informative row names, or the
## design columns with the same name (and more than one value) agree row by
## row; "inferred" when the response and weights agree and nothing else tells;
## "none" otherwise.
.fastglm_cmp_alignment <- function(i, j, memo)
{
    ra <- memo$rows(i); rb <- memo$rows(j)
    if (is.null(ra) || is.null(rb)) return("none")
    if (!identical(dim(ra$y), dim(rb$y)) || length(ra$y) != length(rb$y) || length(ra$w) != length(rb$w))
        return("none")
    close <- function(u, v) all(abs(u - v) <= 1e-12 * pmax(abs(u), abs(v), 1))
    same.y <- if (is.numeric(ra$y) && is.numeric(rb$y)) close(ra$y, rb$y) else
        identical(as.character(ra$y), as.character(rb$y))
    if (!same.y || !close(ra$w, rb$w)) return("none")
    na <- memo$rownames(i); nb <- memo$rownames(j)
    if (!is.null(na) && !is.null(nb)) return(if (identical(na, nb)) "confirmed" else "none")
    pool <- function(d) {
        out <- list()
        for (x in d) for (k in seq_len(ncol(x))) {
            nm <- colnames(x)[k]
            if (!is.null(nm) && is.null(out[[nm]])) out[[nm]] <- x[, k]
        }
        out
    }
    da <- memo$designs(i); db <- memo$designs(j)
    if (is.null(da) || is.null(db)) return("inferred")
    ca <- pool(da); cb <- pool(db)
    shared <- intersect(names(ca), names(cb))
    shared <- shared[vapply(shared, function(nm) { v <- ca[[nm]]; length(v) && max(v) > min(v) }, NA)]
    if (!length(shared)) return("inferred")
    if (all(vapply(shared, function(nm) close(ca[[nm]], cb[[nm]]), NA))) "confirmed" else "none"
}

## The designs of an in-memory fit, by part, as dense matrices; NULL for
## streamed, big.matrix, and sparse designs, or when no design was retained.
## Ordinal designs have no intercept column (the thresholds take its place);
## partial proportional odds designs are split into location and nominal
## (threshold-specific) columns. Columns whose coefficients were not estimated
## (aliased) are left out, so the designs describe the fitted models.
.fastglm_cmp_designs <- function(object)
{
    if (!is.null(object$source)) return(NULL)
    d <- tryCatch(.fastglm_refit_data(object), error = function(e) NULL)
    if (is.null(d)) return(NULL)
    ok <- function(x) is.matrix(x) && is.numeric(x)
    estimated <- function(x, b) {
        if (length(b) == ncol(x) && anyNA(b)) x[, !is.na(b), drop = FALSE] else x
    }
    if (.fastglm_two_part(object)) {
        if (!ok(d$X) || !ok(d$Z)) return(NULL)
        return(list(count = estimated(d$X, object$coefficients$count),
                    zero = estimated(d$Z, object$coefficients$zero)))
    }
    if (!ok(d$X)) return(NULL)
    if (inherits(object, "fastglm_ordinal")) {
        np <- if (inherits(object, "fastglm_ordinal_partial")) as.integer(object$nonparallel) else integer()
        loc <- setdiff(seq_len(ncol(d$X)), np)
        return(list(location = d$X[, loc, drop = FALSE], nominal = d$X[, np, drop = FALSE]))
    }
    if (inherits(object, "fastglm_vglm")) return(list(main = d$X))
    list(main = estimated(d$X, object$coefficients))
}

## The R factor of the columns of cbind(x1, x0) (with a leading column of ones
## when constant = TRUE) over the given rows, computed over blocks of rows
## (TSQR: each step factors the previous factor stacked on the next block), so
## that memory is of the order of a block times the number of columns, not of
## the number of rows. With X = QR and Q orthonormal, linear dependences and
## relative residual norms among the columns of X equal those among the
## columns of R. The column pivoting of each step is undone, so the columns of
## the result stay in the order of X (the result is then not triangular, but
## X = QR still holds with Q orthonormal, which is all that is used).
.fastglm_cmp_tsqr <- function(x1, x0, rows, constant = FALSE)
{
    idx <- which(rows)
    p <- ncol(x1) + ncol(x0) + constant
    size <- max(256L, as.integer(floor(2^20 / max(p, 1L))))
    R <- NULL
    for (start in seq.int(1L, length(idx), by = size)) {
        i <- idx[start:min(length(idx), start + size - 1L)]
        b <- cbind(x1[i, , drop = FALSE], x0[i, , drop = FALSE])
        if (constant) b <- cbind(1, b)
        q <- qr(rbind(R, b), LAPACK = TRUE)
        R <- qr.R(q)[, order(q$pivot), drop = FALSE]
    }
    R
}

## TRUE when every column of x0 lies in the column space of x1 (plus a
## constant when constant = TRUE) over the given rows: relative residual norm
## at most 1e-8, computed from the blockwise R factor. Each column of the
## factor is divided by its largest absolute value first, so the norms stay
## finite whatever the magnitude of the columns.
.fastglm_cmp_in_span <- function(x0, x1, rows, constant = FALSE)
{
    if (!ncol(x0)) return(TRUE)
    if (!ncol(x1) && !constant) return(all(x0[rows, ] == 0))
    R <- .fastglm_cmp_tsqr(x1, x0, rows, constant)
    s <- apply(abs(R), 2L, max)
    s[!(s > 0) | !is.finite(s)] <- 1
    R <- sweep(R, 2L, s, "/")
    k1 <- ncol(x1) + constant
    r1 <- R[, seq_len(k1), drop = FALSE]; r0 <- R[, k1 + seq_len(ncol(x0)), drop = FALSE]
    q <- qr(r1, tol = 1e-10)
    res <- qr.resid(q, r0)
    all(sqrt(colSums(res^2)) <= 1e-8 * pmax(sqrt(colSums(r0^2)), .Machine$double.xmin))
}

## Design-based nesting of model i (smaller, record a) in model j: TRUE,
## FALSE, or NA when a design is not available.
.fastglm_cmp_design_nested <- function(i, j, a, memo)
{
    da <- memo$designs(i); db <- memo$designs(j)
    if (is.null(da) || is.null(db)) return(NA)
    rows <- memo$rows(j)$w > 0
    if (a$class_family == "ordinal") {
        loc1 <- cbind(db$location, db$nominal)
        return(.fastglm_cmp_in_span(da$location, loc1, rows, constant = TRUE) &&
               .fastglm_cmp_in_span(da$nominal, db$nominal, rows, constant = TRUE))
    }
    if (!is.null(da$main) && is.null(db$main)) da <- list(count = da$main)
    if (!is.null(db$main) && is.null(da$main)) db <- list(count = db$main)
    all(vapply(names(da), function(p) !is.null(db[[p]]) && .fastglm_cmp_in_span(da[[p]], db[[p]], rows), NA))
}

## The in-memory offsets of two aligned fits must agree row by row.
.fastglm_cmp_same_offset_rows <- function(ma, mb)
{
    oa <- .fastglm_cmp_offset_values(ma); ob <- .fastglm_cmp_offset_values(mb)
    if (is.null(oa) || is.null(ob)) return(TRUE)
    part <- function(o) { if (!is.null(o$main)) list(count = o$main, zero = NULL) else o }
    oa <- part(oa); ob <- part(ob)
    n <- length(.fastglm_cmp_rows(ma)$w)
    for (p in c("count", "zero")) {
        a <- oa[[p]]; b <- ob[[p]]
        a <- if (is.null(a)) numeric(n) else as.numeric(a)
        b <- if (is.null(b)) numeric(n) else as.numeric(b)
        if (length(a) != length(b) || !all(abs(a - b) <= 1e-12 * pmax(abs(a), abs(b), 1))) return(FALSE)
    }
    TRUE
}

## TRUE when the coefficient names of a fit are the generated defaults of the
## matrix interface (the design had no column names), so they cannot be used
## to check nesting.
.fastglm_cmp_generated_names <- function(object, rec)
{
    if (!is.null(object$terms) || !is.null(object$formula)) return(FALSE)
    all_nm <- unlist(rec$coef_names, use.names = FALSE)
    core <- ifelse(grepl("^nonparallel:", all_nm), sub("^nonparallel:(.*):[^:]*$", "\\1", all_nm),
                   sub("^[^:()]*:", "", all_nm))
    core <- core[!grepl("^\\(Intercept\\)[0-9]*$", core)]
    length(core) > 0L && all(grepl("^[Xx][0-9]+$", core))
}

## Coefficient names with the variables of interactions in a fixed order, so
## that a:c and c:a match (multinomial names keep their category prefix,
## nonparallel ordinal names their threshold suffix).
.fastglm_cmp_norm <- function(x)
    vapply(strsplit(x, ":", fixed = TRUE), function(p) paste(sort(p), collapse = ":"), character(1))

.fastglm_cmp_norm_names <- function(rec)
{
    lapply(rec$coef_names, function(nm) {
        if (!length(nm)) return(character())
        if (rec$class_family == "multinom") {
            pre <- sub(":.*$", "", nm); rest <- sub("^[^:]*:", "", nm)
            return(paste0(pre, ":", .fastglm_cmp_norm(rest)))
        }
        np <- grepl("^nonparallel:", nm)
        out <- .fastglm_cmp_norm(nm)
        if (any(np)) {
            v <- sub("^nonparallel:(.*):[^:]*$", "\\1", nm[np]); thr <- sub("^.*:", "", nm[np])
            out[np] <- paste0("nonparallel:", .fastglm_cmp_norm(v), ":", thr)
        }
        out
    })
}

## The variable of an ordinal coefficient name: a location name, or the
## variable of a nonparallel (cutpoint-specific) coefficient.
.fastglm_cmp_ordinal_var <- function(nm) sub("^nonparallel:(.*):[^:]*$", "\\1", nm)

## The (normalized) coefficient names of the smaller model a missing from the
## larger model b, by part; character(0) when a is nested in b by names.
.fastglm_cmp_missing <- function(a, b)
{
    na <- .fastglm_cmp_norm_names(a); nb <- .fastglm_cmp_norm_names(b)
    if (a$class_family == "ordinal") {
        bvars <- .fastglm_cmp_ordinal_var(nb$nominal)
        return(c(na$location[!(na$location %in% nb$location | na$location %in% bvars)],
                 setdiff(na$nominal, nb$nominal)))
    }
    if (!is.null(na$main) && is.null(nb$main)) na <- list(count = na$main)
    if (!is.null(nb$main) && is.null(na$main)) nb <- list(count = nb$main)
    parts <- names(na)
    unlist(lapply(parts, function(p) {
        m <- setdiff(na[[p]], nb[[p]])
        if (length(m) && length(parts) > 1L) paste0(p, ": ", m) else m
    }), use.names = FALSE)
}

## TRUE when two streamed fits read the same data with the same design
## coding: the identical source object, or formula sources built from the
## identical callback, prototype, number of blocks, and weight and offset
## specifications. A coefficient name then determines its column.
.fastglm_cmp_same_stream <- function(ma, mb)
{
    sa <- ma$source; sb <- mb$source
    if (is.null(sa) || is.null(sb)) return(FALSE)
    if (identical(sa, sb)) return(TRUE)
    pa <- sa$spec; pb <- sb$spec
    !is.null(pa) && !is.null(pb) && !is.null(pa$prototype) &&
        identical(pa$callback, pb$callback) && identical(pa$prototype, pb$prototype) &&
        identical(sa$n_chunks, sb$n_chunks) && identical(pa$weights, pb$weights) &&
        identical(pa$offset, pb$offset)
}

## Nesting of model i (smaller, record a) in model j (record b). Fits held in
## memory are compared by the column spaces of their designs when their rows
## are in the same order; a failure is an error when the order is confirmed,
## and otherwise the check falls back to coefficient names with a warning.
## Returns TRUE or an error.
.fastglm_cmp_nested <- function(i, j, models, a, b, la, lb, memo)
{
    ma <- models[[i]]; mb <- models[[j]]
    alignment <- .fastglm_cmp_alignment(i, j, memo)
    if (alignment == "confirmed" && !.fastglm_cmp_same_offset_rows(ma, mb)) .fastglm_cmp_offset_error(la, lb)
    if (alignment != "none") {
        nested <- .fastglm_cmp_design_nested(i, j, a, memo)
        if (isTRUE(nested)) return(TRUE)
        if (isFALSE(nested) && alignment == "confirmed")
            stop(sprintf(paste0("'%s' is not nested in '%s': columns of its design are not in the column space ",
                                "of the larger model's design"), la, lb), call. = FALSE)
    }
    same.stream <- .fastglm_cmp_same_stream(ma, mb)
    reason <- if (!is.null(ma$source) || !is.null(mb$source)) "a fit reads its data in blocks"
              else if (alignment == "none") "the rows of the fits are not in the same order, or a fit did not retain its response"
              else if (alignment == "inferred") "the rows of the fits could not be confirmed to be in the same order"
              else "a design is not held in memory as a dense matrix"
    if (.fastglm_cmp_generated_names(ma, a) || .fastglm_cmp_generated_names(mb, b)) {
        warning(sprintf(paste0("nesting of '%s' in '%s' cannot be verified: %s, and a design without column ",
                               "names gives no coefficient names to match"), la, lb, reason), call. = FALSE)
        return(TRUE)
    }
    miss <- .fastglm_cmp_missing(a, b)
    if (length(miss))
        stop(sprintf("'%s' is not nested in '%s': coefficients %s are absent from the larger model",
                     la, lb, paste0("'", miss, "'", collapse = ", ")), call. = FALSE)
    if (!same.stream)
        warning(sprintf("nesting of '%s' in '%s' was checked by coefficient names only (%s)", la, lb, reason),
                call. = FALSE)
    TRUE
}

## Nesting of consecutive models (sorted by size). Returns, for each pair,
## whether it is a boundary (Poisson / negative binomial) pair and the number
## r of ordinary parameters the larger model adds besides theta.
.fastglm_cmp_check_nesting <- function(models, recs, labels, firth = FALSE)
{
    k <- length(recs)
    memo <- .fastglm_cmp_memo(models)
    boundary <- logical(k); r <- rep(NA_integer_, k)
    for (i in seq_len(k)[-1L]) {
        a <- recs[[i - 1L]]; b <- recs[[i]]
        la <- labels[i - 1L]; lb <- labels[i]
        dfa <- .fastglm_cmp_npar(a); dfb <- .fastglm_cmp_npar(b)
        if (.fastglm_cmp_boundary_pair(a, b)) {
            if (a$family != "poisson")
                stop(sprintf(paste0("'%s' (negative binomial) is not nested in '%s' (Poisson): the Poisson model ",
                                    "has no theta, so it must be the smaller model"), la, lb), call. = FALSE)
            if (b$dispersion != "ml")
                stop(sprintf(paste0("'%s' holds theta fixed; a Poisson model is nested in a negative binomial model ",
                                    "only when theta is estimated"), lb), call. = FALSE)
            .fastglm_cmp_nested(i - 1L, i, models, a, b, la, lb, memo)
            boundary[i] <- TRUE
            r[i] <- as.integer(dfb - dfa - 1L)
            next
        }
        .fastglm_cmp_nested(i - 1L, i, models, a, b, la, lb, memo)
        if (firth) {
            miss <- setdiff(a$coef_names$main, b$coef_names$main)
            if (length(miss) || .fastglm_cmp_generated_names(models[[i - 1L]], a) ||
                .fastglm_cmp_generated_names(models[[i]], b))
                stop(sprintf(paste0("the penalized likelihood ratio test of '%s' against '%s' sets the larger ",
                                    "model's other coefficients to zero, so every coefficient of the smaller ",
                                    "model must appear by name in the larger model"), la, lb), call. = FALSE)
        }
    }
    list(boundary = boundary, r = r)
}

## A one-line description of a model for the heading: its formula when it
## has one, otherwise its name in the call.
.fastglm_cmp_describe <- function(object, label)
{
    if (is.null(object$terms) && is.null(object$formula)) return(label)
    f <- tryCatch(stats::formula(object), error = function(e) NULL)
    if (!inherits(f, "formula")) return(label)
    paste(deparse(f), collapse = "\n")
}

## ---- Tables -------------------------------------------------------------------

## GLMs: the table of stats::anova.glm.
.fastglm_lrt_glm <- function(models, recs, topnote, test, dispersion)
{
    resdf <- vapply(models, function(m) as.numeric(m$df.residual), numeric(1))
    resdev <- vapply(models, function(m) as.numeric(m$deviance), numeric(1))
    table <- data.frame(resdf, resdev, c(NA, -diff(resdf)), c(NA, -diff(resdev)))
    dimnames(table) <- list(seq_along(models), c("Resid. Df", "Resid. Dev", "Df", "Deviance"))
    big <- models[[length(models)]]
    big.rec <- recs[[length(recs)]]
    moment <- big.rec$dispersion == "moment"
    if (is.null(test)) test <- if (!is.null(dispersion)) "Chisq" else if (moment) "F" else "Chisq"
    if (is.null(dispersion)) {
        fixed <- .fixed_dispersion(big$family)
        dispersion <- if (!moment && !is.na(fixed)) fixed else .fastglm_cmp_glm_dispersion(big)
    }
    df.scale <- if (moment) big$df.residual else Inf
    if (test == "F" && df.scale == Inf) {
        fam <- big$family$family
        if (fam %in% c("binomial", "poisson"))
            warning(sprintf("using F test with a '%s' family is inappropriate", fam), call. = FALSE)
        else warning("using F test with a fixed dispersion is inappropriate", call. = FALSE)
    }
    dfs <- table[, "Df"]
    if (test == "F" && df.scale == 0) {
        out <- table
    } else if (test == "Chisq") {
        vals <- table[, "Deviance"] / dispersion * sign(dfs)
        vals[dfs %in% 0] <- NA
        vals[!is.na(vals) & vals < 0] <- NA
        out <- cbind(table, `Pr(>Chi)` = stats::pchisq(vals, abs(dfs), lower.tail = FALSE))
    } else {
        f <- (table[, "Deviance"] / dfs) / dispersion
        f[dfs %in% 0] <- NA
        f[!is.na(f) & f < 0] <- NA
        out <- cbind(table, F = f, `Pr(>F)` = stats::pf(f, abs(dfs), df.scale, lower.tail = FALSE))
    }
    structure(out, heading = c("Analysis of Deviance Table\n", topnote), class = c("anova", "data.frame"))
}

## The dispersion of a GLM with a dispersion estimated by moments (Pearson),

## The dispersion of a GLM with a dispersion estimated by moments (Pearson),
## as summary.glm() gives it.
.fastglm_cmp_glm_dispersion <- function(object)
{
    if (!is.null(object$dispersion) && is.finite(object$dispersion)) return(as.numeric(object$dispersion))
    summary(object)$dispersion
}

## The p-value of a Poisson / negative binomial step: the larger model adds r
## ordinary parameters and theta, whose Poisson limit 1/theta = 0 is on the
## boundary, so the reference is the 50:50 mixture of chi-square(r) and
## chi-square(r + 1) (with r = 0, half the chi-square(1) p-value).
.fastglm_boundary_pvalue <- function(stat, r)
{
    0.5 * stats::pchisq(stat, r, lower.tail = FALSE) + 0.5 * stats::pchisq(stat, r + 1, lower.tail = FALSE)
}

## Other models: the likelihood ratio table.
.fastglm_lrt_loglik <- function(models, recs, pairs, topnote)
{
    ll <- vapply(recs, function(r) as.numeric(r$loglik), numeric(1))
    npar <- vapply(recs, .fastglm_cmp_npar, numeric(1))
    dfs <- c(NA, diff(npar))
    stat <- c(NA, 2 * diff(ll))
    # A clearly negative statistic means the models are not nested as fitted
    # or a fit did not converge; a slightly negative one is rounding.
    neg <- !is.na(stat) & stat < 0
    bad <- neg & stat < -1e-6 * pmax(1, abs(c(NA, ll[-1L])))
    if (any(bad))
        warning(sprintf(paste0("the likelihood ratio statistic of model %s against the model before it is negative ",
                               "(%s): the larger model has a smaller log likelihood, so the models are not nested ",
                               "as fitted or a fit did not converge; no p-value is given"),
                        paste(which(bad), collapse = ", "), paste(format(stat[bad], digits = 4), collapse = ", ")),
                call. = FALSE)
    stat[neg & !bad] <- 0
    stat[bad] <- NA
    p <- stats::pchisq(stat, dfs, lower.tail = FALSE)
    p[dfs %in% 0] <- NA
    b <- pairs$boundary
    p[b] <- .fastglm_boundary_pvalue(stat[b], pairs$r[b])
    table <- data.frame(npar, ll, dfs, stat, p)
    dimnames(table) <- list(seq_along(recs), c("#Df", "LogLik", "Df", "Chisq", "Pr(>Chisq)"))
    heading <- c("Likelihood ratio test\n", topnote)
    for (i in which(b))
        heading <- c(heading, paste0(
            "\nNote: model ", i, " against model ", i - 1L, " is a negative binomial model against a Poisson ",
            "model with r = ", pairs$r[i], " additional\nmean parameter", if (pairs$r[i] != 1L) "s", ". ",
            "The Poisson limit (1/theta = 0) is on the boundary of the parameter space, so the p-value uses\n",
            "the 50:50 mixture of chi-square(", pairs$r[i], ") and chi-square(", pairs$r[i] + 1L, ")",
            if (pairs$r[i] == 0L) " (half the chi-square(1) p-value)" else "", "."))
    structure(table, heading = heading, class = c("anova", "data.frame"))
}

## Bias-reduced (Firth) fits: the penalized likelihood ratio of each model
## against the next larger one, from fastglm_profile_test() on the larger
## model with the coefficients absent from the smaller one set to zero.
.fastglm_lrt_firth <- function(models, recs, pairs, topnote, test)
{
    if (identical(test, "F"))
        stop("an F test is not available for bias-reduced fits; use test = \"Chisq\"", call. = FALSE)
    k <- length(models)
    stat <- rep(NA_real_, k); dfs <- rep(NA_real_, k)
    for (i in seq_len(k)[-1L]) {
        a <- recs[[i - 1L]]; b <- recs[[i]]
        if (.fastglm_cmp_generated_names(models[[i - 1L]], a) || .fastglm_cmp_generated_names(models[[i]], b))
            stop("a penalized likelihood ratio test of bias-reduced fits needs coefficient names ",
                 "to identify the tested coefficients (fit with a named design)", call. = FALSE)
        parm <- setdiff(b$coef_names$main, a$coef_names$main)
        dfs[i] <- length(parm)
        if (!length(parm)) next
        pt <- fastglm_profile_test(models[[i]], parm = parm)
        stat[i] <- unname(pt$statistic)
    }
    npar <- vapply(recs, `[[`, numeric(1), "df")
    p <- stats::pchisq(stat, dfs, lower.tail = FALSE)
    p[dfs %in% 0] <- NA
    table <- data.frame(npar, vapply(recs, function(r) as.numeric(r$loglik), numeric(1)), dfs, stat, p)
    dimnames(table) <- list(seq_along(models), c("#Df", "Penalized LogLik", "Df", "Chisq", "Pr(>Chisq)"))
    heading <- c("Penalized likelihood ratio test (Firth)\n", topnote,
                 paste0("\nEach statistic is the penalized likelihood ratio of a model against the next larger ",
                        "one,\nwith the absent coefficients set to zero under the larger model's penalty ",
                        "(fastglm_profile_test()).\nIt is not twice the difference of the penalized log ",
                        "likelihoods shown."))
    structure(table, heading = heading, class = c("anova", "data.frame"))
}
