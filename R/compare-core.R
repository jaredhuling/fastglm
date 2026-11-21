## Internal groundwork for model comparisons (anova() and drop1() for every
## fitted class): a per-class comparison record and a refit helper.
##
## .fastglm_cmp_record(object) describes a fit in a common form, so that the
## comparison engine needs no class-specific code:
##   class_family  one of "glm", "nb", "zi", "hurdle", "twopart", "truncated",
##                 "ordinal", "multinom"
##   family, link  the error distribution and link (link is a named vector,
##                 one entry per part, for models with several parts)
##   nobs, wsum    nobs(object) and the fingerprint's sum of weights
##   fingerprint   c(n, wsum, sy, syy) identifying the response and weights
##                 (see .fastglm_fingerprint and .fastglm_category_fingerprint_add)
##   coef_names    the coefficient names of each part (main; count and zero;
##                 location and nominal)
##   loglik, df    the log likelihood (penalized for Firth fits that maximize
##                 a penalized likelihood) and its "df" attribute
##   penalized     TRUE for Firth fits that maximize the Jeffreys-penalized
##                 likelihood, FALSE for maximum likelihood, NA for mean bias
##                 reduction that is not a penalized likelihood
##   lr_method     how two fits of this kind are compared: "lr" (likelihood
##                 ratio of maximum likelihood fits), "profile" (Firth fits
##                 that fastglm_profile_test() supports: a binomial-logit
##                 bias-reduced fit with a retained dense design; the smaller
##                 model is the larger one with the absent coefficients held
##                 at zero under the larger model's penalty, as in logistf),
##                 or "none" (other bias-reduced fits; no likelihood ratio
##                 comparison). The penalized log likelihood in loglik is for
##                 information only: the difference of two separately
##                 penalized fits is not the penalized likelihood ratio.
##   lr_available  lr_method != "none"
##   dispersion    "fixed", "ml" (an extra parameter counted in df), or
##                 "moment" (Pearson estimate, as in stats::anova.glm)
##   streamed      TRUE when the fit reads a source in blocks
##   source_reusable  FALSE when the source was declared one-shot
##                 (fastglm_source(reusable = FALSE))
##
## .fastglm_refit(object, formula, drop_columns, part, ...) refits a model on
## the data the fit stored, never on objects looked up in an environment:
## in-memory fits use the stored model frame (or design), response, prior
## weights, offsets, and fitting controls, as stats::drop1.glm uses the stored
## model matrix; streamed fits read their source again, rebuilding each block
## from the source's formula specification. The refit runs with the fit's
## thread count and with the trace off, and is checked to use the same
## observations and response as the original fit.

## ---- Response fingerprint ---------------------------------------------------

## The fingerprint of a response y with prior weights w: the number of rows
## with positive weight, sum(w), sum(w y), and sum(w y^2), over the rows with
## positive weight. Fits of the same data have the same fingerprint whether
## the data were held in memory or read in blocks (up to rounding).
.fastglm_fingerprint <- function(y, weights = NULL)
{
    .fastglm_fingerprint_add(NULL, y, weights)
}

## Adds one block of rows to a fingerprint (NULL starts a new one).
.fastglm_fingerprint_add <- function(fingerprint, y, weights = NULL)
{
    if (is.null(fingerprint)) fingerprint <- c(n = 0, wsum = 0, sy = 0, syy = 0)
    y <- as.numeric(y)
    if (is.null(weights)) weights <- rep(1, length(y))
    keep <- weights > 0
    w <- weights[keep]; y <- y[keep]
    fingerprint + c(sum(keep), sum(w), sum(w * y), sum(w * y^2))
}

## Categorical responses (ordinal and multinomial fits) are fingerprinted by
## cells: each observation of category j (code 1, ..., K) with frequency f
## (the weight, times the count for a matrix of category counts) adds f to n
## and to wsum, f j to sy, and f j^2 to syy. Individual and grouped forms of
## the same data then have the same fingerprint, and n is the total frequency.
.fastglm_category_fingerprint_add <- function(fingerprint, codes = NULL, counts = NULL, weights = NULL)
{
    if (is.null(fingerprint)) fingerprint <- c(n = 0, wsum = 0, sy = 0, syy = 0)
    if (length(counts) && is.matrix(counts) && nrow(counts)) {
        if (is.null(weights)) weights <- rep(1, nrow(counts))
        f <- counts * weights
        j <- col(f)
        keep <- f > 0
        f <- f[keep]; j <- j[keep]
    } else {
        f <- if (is.null(weights)) rep(1, length(codes)) else weights
        j <- as.numeric(codes)
        keep <- f > 0
        f <- f[keep]; j <- j[keep]
    }
    fingerprint + c(sum(f), sum(f), sum(f * j), sum(f * j^2))
}

.fastglm_fingerprint_na <- c(n = NA_real_, wsum = NA_real_, sy = NA_real_, syy = NA_real_)

## The offset fingerprint: the sums of the offset and of its square over the
## rows with positive weight (all entries of a matrix offset, one column per
## equation). Adds one block of rows (NULL starts a new one); a NULL offset
## adds nothing (a zero offset).
.fastglm_offset_fingerprint_add <- function(fingerprint, offset, weights = NULL)
{
    if (is.null(fingerprint)) fingerprint <- c(so = 0, soo = 0)
    if (is.null(offset) || !length(offset)) return(fingerprint)
    keep <- if (is.null(weights)) rep(TRUE, NROW(offset)) else weights > 0
    o <- if (is.matrix(offset)) offset[keep, , drop = FALSE] else as.numeric(offset)[keep]
    fingerprint + c(sum(o), sum(o^2))
}

## The offsets of an in-memory fit, by part (main; count and zero), as stored;
## NULL entries are zero offsets. NULL when the fit reads a source.
.fastglm_cmp_offset_values <- function(object)
{
    if (!is.null(object$source)) return(NULL)
    if (.fastglm_two_part(object)) {
        off <- tryCatch(.count_design(object)$offset, error = function(e) NULL)
        if (is.null(off)) off <- object$offset
        if (!is.list(off)) off <- list(count = off, zero = NULL)
        return(list(count = off$count, zero = off$zero))
    }
    list(main = object$offset)
}

## The offset fingerprint of a fit, by part: recorded at fit time, or computed
## from the stored offsets; NULL when neither is available.
.fastglm_cmp_offset_fingerprint <- function(object)
{
    if (!is.null(object$fingerprint.offset)) return(object$fingerprint.offset)
    if (!is.null(object$source)) return(NULL)
    off <- .fastglm_cmp_offset_values(object)
    w <- if (.fastglm_two_part(object)) object$weights else
        if (inherits(object, "fastglm_ordinal")) object$weights else object$prior.weights
    lapply(off, function(o) .fastglm_offset_fingerprint_add(NULL, o, if (length(w) == NROW(o)) w))
}

## Weighted category frequencies of an in-memory categorical response, named
## by category label (independent of row order and of the level order); NULL
## when unavailable.
.fastglm_cmp_cells <- function(object)
{
    if (!is.null(object$source)) return(NULL)
    if (inherits(object, "fastglm_ordinal")) {
        r <- object$response; lev <- if (!is.null(object$levels)) object$levels else r$levels
        if (is.null(r) || is.null(lev)) return(NULL)
        w <- if (length(r$weights)) r$weights else NULL
        if (length(r$counts) && is.matrix(r$counts) && nrow(r$counts)) {
            f <- colSums(r$counts * (if (is.null(w)) 1 else w))
            if (length(f) != length(lev)) return(NULL)
            return(stats::setNames(f, lev))
        }
        codes <- if (length(r$original.codes)) r$original.codes else r$codes
        if (is.null(w)) w <- rep(1, length(codes))
        return(stats::setNames(vapply(seq_along(lev), function(j) sum(w[codes == j]), 0), lev))
    }
    if (inherits(object, "fastglm_vglm")) {
        y <- object$y; w <- object$prior.weights; lev <- object$levels
        if (is.null(y) || is.null(lev)) return(NULL)
        if (is.null(w)) w <- rep(1, NROW(y))
        if (is.matrix(y)) return(stats::setNames(colSums(y * w), if (is.null(colnames(y))) lev else colnames(y)))
        y <- as.character(y)
        return(stats::setNames(vapply(lev, function(l) sum(w[y == l]), 0), lev))
    }
    NULL
}

## The response of a two-part fit: retained, or from the retained model frame.
.fastglm_cmp_twopart_y <- function(object)
{
    if (!is.null(object$y)) return(object$y)
    if (!is.null(object$model)) return(stats::model.response(object$model))
    NULL
}

## The fingerprint of any fitted class: the one recorded at fit time
## (streamed, big.matrix, and block-reader fits), otherwise computed from the
## retained response and weights; NA when neither is available.
.fastglm_cmp_fingerprint <- function(object)
{
    if (!is.null(object$fingerprint.response)) return(object$fingerprint.response)
    if (inherits(object, c("fastglm_zi", "fastglm_hurdle"))) {
        y <- .fastglm_cmp_twopart_y(object)
        return(if (is.null(y)) .fastglm_fingerprint_na else .fastglm_fingerprint(y, object$weights))
    }
    if (inherits(object, "fastglm_ordinal")) {
        r <- object$response
        if (is.null(r) || !length(r$weights)) return(.fastglm_fingerprint_na)
        codes <- if (length(r$original.codes)) r$original.codes else r$codes
        return(.fastglm_category_fingerprint_add(NULL, codes, r$counts, r$weights))
    }
    if (inherits(object, "fastglm_vglm")) {
        y <- object$y
        if (is.null(y)) return(.fastglm_fingerprint_na)
        if (is.matrix(y)) return(.fastglm_category_fingerprint_add(NULL, counts = y, weights = object$prior.weights))
        return(.fastglm_category_fingerprint_add(NULL, match(as.character(y), object$levels),
                                                 weights = object$prior.weights))
    }
    if (is.null(object$y)) return(.fastglm_fingerprint_na)
    .fastglm_fingerprint(object$y, object$prior.weights)
}

## TRUE when two fingerprints agree to a relative tolerance (default 1e-12);
## NA when either is unavailable.
.fastglm_same_fingerprint <- function(a, b, tol = 1e-12)
{
    if (anyNA(a) || anyNA(b)) return(NA)
    all(abs(a - b) <= tol * pmax(abs(a), abs(b), 1e-300))
}

## ---- Comparison records -------------------------------------------------------

.fastglm_cmp_record <- function(object, ...) UseMethod(".fastglm_cmp_record")

#' @exportS3Method .fastglm_cmp_record default
.fastglm_cmp_record.default <- function(object, ...)
{
    stop(sprintf("model comparison is not available for objects of class '%s'",
                 class(object)[1L]), call. = FALSE)
}

## The fields every record shares. penalized: FALSE, TRUE, or NA (see the top
## of this file); penalized_loglik: the penalized log likelihood when TRUE.
.fastglm_cmp_new <- function(object, class_family, family, link, coef_names,
                             penalized = FALSE, penalized_loglik = NULL, dispersion = "fixed")
{
    fingerprint <- .fastglm_cmp_fingerprint(object)
    ll <- logLik(object)
    if (isTRUE(penalized)) {
        if (is.null(penalized_loglik) || !is.finite(penalized_loglik))
            penalized <- NA
        else ll[] <- penalized_loglik
    }
    nobs <- stats::nobs(object)
    if (is.numeric(nobs) && length(nobs) == 1L && is.finite(nobs) && nobs == round(nobs))
        nobs <- as.integer(nobs)
    streamed <- !is.null(object$source)
    lr_method <- if (!isTRUE(object$firth)) "lr" else
        if (.fastglm_profile_supported(object)) "profile" else "none"
    list(class_family = class_family,
         family = family,
         link = link,
         nobs = nobs,
         wsum = unname(fingerprint[["wsum"]]),
         fingerprint = fingerprint,
         coef_names = coef_names,
         loglik = ll,
         df = as.integer(attr(ll, "df")),
         penalized = penalized,
         lr_method = lr_method,
         lr_available = lr_method != "none",
         dispersion = dispersion,
         offset_fingerprint = .fastglm_cmp_offset_fingerprint(object),
         cells = .fastglm_cmp_cells(object),
         streamed = streamed,
         source_reusable = !streamed || !isFALSE(object$source$reusable))
}

## TRUE for a bias-reduced fit that fastglm_profile_test() supports (see
## .firth_profile_setup()): binomial logit, in memory, with a retained dense
## design.
.fastglm_profile_supported <- function(object)
{
    fam <- object$family
    is.list(fam) && identical(fam$family, "binomial") && identical(fam$link, "logit") &&
        is.null(object$source) && is.matrix(object$x)
}

## Firth fits of GLMs maximize the Jeffreys-penalized likelihood
## l + log det(X'WX) / 2 when the link is canonical and the dispersion is
## fixed (binomial logit, Poisson log), as logistf does; other bias-reduced
## GLMs solve adjusted score equations that are not a penalized likelihood.
.fastglm_cmp_glm_penalty <- function(object)
{
    if (!isTRUE(object$firth)) return(list(penalized = FALSE))
    fam <- object$family
    canonical <- (fam$family == "binomial" && fam$link == "logit") ||
        (fam$family == "poisson" && fam$link == "log")
    if (!canonical || is.null(object$log.det.XtWX)) return(list(penalized = NA))
    list(penalized = TRUE,
         loglik = as.numeric(logLik(object)) + 0.5 * object$log.det.XtWX)
}

## glm fits: formula (fastglmFit), matrix, sparse, big.matrix, and streamed.
#' @exportS3Method .fastglm_cmp_record fastglm
.fastglm_cmp_record.fastglm <- function(object, ...)
{
    fam <- object$family
    moment <- is.na(.fixed_dispersion(fam)) || startsWith(fam$family, "quasi")
    pen <- .fastglm_cmp_glm_penalty(object)
    .fastglm_cmp_new(object, "glm", fam$family, fam$link,
                     list(main = names(coef(object))),
                     penalized = pen$penalized, penalized_loglik = pen$loglik,
                     dispersion = if (moment) "moment" else "fixed")
}

#' @exportS3Method .fastglm_cmp_record fastglmFit
.fastglm_cmp_record.fastglmFit <- .fastglm_cmp_record.fastglm

#' @exportS3Method .fastglm_cmp_record fastglm_nb
.fastglm_cmp_record.fastglm_nb <- function(object, ...)
{
    .fastglm_cmp_new(object, "nb", "negbin", object$family$link,
                     list(main = names(object$coefficients)),
                     dispersion = if (isFALSE(object$theta.estimated)) "fixed" else "ml")
}

#' @exportS3Method .fastglm_cmp_record fastglm_truncated
.fastglm_cmp_record.fastglm_truncated <- function(object, ...)
{
    .fastglm_cmp_new(object, "truncated", object$dist, "log",
                     list(main = names(object$coefficients)),
                     dispersion = if (isTRUE(object$theta.estimated)) "ml" else "fixed")
}

#' @exportS3Method .fastglm_cmp_record fastglm_zi
.fastglm_cmp_record.fastglm_zi <- function(object, ...)
{
    .fastglm_cmp_new(object, "zi", object$dist, c(count = "log", zero = object$link),
                     lapply(object$coefficients[c("count", "zero")], names),
                     dispersion = if (isTRUE(object$theta.estimated)) "ml" else "fixed")
}

#' @exportS3Method .fastglm_cmp_record fastglm_hurdle
.fastglm_cmp_record.fastglm_hurdle <- function(object, ...)
{
    .fastglm_cmp_new(object, "hurdle", object$dist, c(count = "log", zero = object$link),
                     lapply(object$coefficients[c("count", "zero")], names),
                     dispersion = if (isTRUE(object$theta.estimated)) "ml" else "fixed")
}

## Two-part (Gamma) fits: the Gamma dispersion is a parameter of the log
## likelihood and is counted in its df. The bias-reduced Gamma (log link)
## part is not a penalized likelihood fit, so Firth two-part fits have no
## likelihood ratio test.
#' @exportS3Method .fastglm_cmp_record fastglm_twopart
.fastglm_cmp_record.fastglm_twopart <- function(object, ...)
{
    .fastglm_cmp_new(object, "twopart", "Gamma", c(count = "log", zero = object$link),
                     lapply(object$coefficients[c("count", "zero")], names),
                     penalized = if (isTRUE(object$firth)) NA else FALSE, dispersion = "ml")
}

## Ordinal fits: location coefficients (common slopes) and, for partial
## proportional odds fits, the nominal (cutpoint-specific) coefficients. The
## thresholds are common to every fit of the same response and are left out.
## Bias-reduced ordinal fits solve adjusted score equations that are not a
## penalized likelihood (see ?fastglm_ordinal).
#' @exportS3Method .fastglm_cmp_record fastglm_ordinal
.fastglm_cmp_record.fastglm_ordinal <- function(object, ...)
{
    nm <- names(object$coefficients)[-seq_along(object$alpha)]
    nominal <- startsWith(nm, "nonparallel:")
    parts <- list(location = nm[!nominal])
    if (inherits(object, "fastglm_ordinal_partial")) parts$nominal <- nm[nominal]
    .fastglm_cmp_new(object, "ordinal", "cumulative", object$link, parts,
                     penalized = if (isTRUE(object$firth)) NA else FALSE)
}

## Vector GLM fits; the multinomial family is the one with comparisons. The
## multinomial logit is canonical, so bias reduction maximizes the
## Jeffreys-penalized likelihood (stored as penalized.loglik).
#' @exportS3Method .fastglm_cmp_record fastglm_vglm
.fastglm_cmp_record.fastglm_vglm <- function(object, ...)
{
    if (!identical(object$family, "multinomial"))
        stop(sprintf("model comparison is not available for '%s' vector GLM fits", object$family),
             call. = FALSE)
    .fastglm_cmp_new(object, "multinom", "multinomial", "logit",
                     list(main = names(object$coefficients)),
                     penalized = isTRUE(object$firth), penalized_loglik = object$penalized.loglik)
}

## ---- Refits -------------------------------------------------------------------

.fastglm_refit <- function(object, formula = NULL, drop_columns = NULL, part = NULL, ...)
    UseMethod(".fastglm_refit")

#' @exportS3Method .fastglm_refit default
.fastglm_refit.default <- function(object, formula = NULL, drop_columns = NULL, part = NULL, ...)
{
    stop(sprintf("refitting is not available for objects of class '%s'", class(object)[1L]),
         call. = FALSE)
}

## formula: a change of the model formula, as in update() (for example
## `. ~ . - a`; for one part of a two-part or partial ordinal model, a
## one-sided change such as `~ . - a` with part = "count", "zero",
## "location", or "nominal"). Refits of in-memory fits can only remove terms.
## drop_columns: design columns (indices or names) to leave out, for fits
## without a formula; part selects the design of a two-part model. Further
## named arguments replace fitting arguments (trace = TRUE traces the refit).
## check = FALSE skips the same-data check.
## columns = TRUE allows drop_columns for a fit with a formula: the columns
## are removed from the stored design (or, for a streamed formula source,
## from each block as read), with no recoding of the remaining terms, as
## stats::drop1.glm() and stats::anova.glm() do (GLM and negative binomial
## fits only).
## nb_theta_fixed = TRUE refits a negative binomial fit as a GLM with theta
## held at the fit's estimate (family negbin(theta)), as stats::drop1.glm()
## and MASS's anova.negbin() for one model do.
#' @exportS3Method .fastglm_refit fastglm
.fastglm_refit.fastglm <- function(object, formula = NULL, drop_columns = NULL, part = NULL, ...,
                                   check = TRUE, columns = FALSE, nb_theta_fixed = FALSE)
{
    extras <- list(...)
    part <- .fastglm_part_internal(part)
    if (!is.null(formula) && !is.null(drop_columns))
        stop("supply either 'formula' or 'drop_columns', not both", call. = FALSE)
    streamed <- !is.null(object$source)
    if (streamed && isFALSE(object$source$reusable))
        stop("the source of this fit was declared one-shot (fastglm_source(reusable = FALSE)); ",
             "it cannot be read again for a refit", call. = FALSE)
    trace <- if (is.null(extras$trace)) FALSE else extras$trace
    extras$trace <- NULL
    scope <- .fastglm_scope_threads(object$threads)
    on.exit(scope$restore(), add = TRUE)
    # trace = TRUE: the option is on inside the scope, so the fitting
    # function (called with trace = NULL) traces the refit.
    tr <- .fastglm_scope_trace(trace, nested = isTRUE(trace))
    on.exit(tr$restore(), add = TRUE)
    if (columns && !inherits(object, c("fastglm", "fastglmFit")))
        stop("column refits of a formula fit are available only for GLM and negative binomial fits",
             call. = FALSE)
    if (nb_theta_fixed && !inherits(object, "fastglm_nb"))
        stop("'nb_theta_fixed' applies only to negative binomial fits", call. = FALSE)
    opts <- list(columns = columns, nb_theta_fixed = nb_theta_fixed)
    fit <- if (streamed) .fastglm_refit_streamed(object, formula, drop_columns, part, extras, opts)
           else .fastglm_refit_memory(object, formula, drop_columns, part, extras, opts)
    if (check) .fastglm_refit_check(object, fit)
    fit
}

#' @exportS3Method .fastglm_refit fastglmFit
.fastglm_refit.fastglmFit <- .fastglm_refit.fastglm
#' @exportS3Method .fastglm_refit fastglm_nb
.fastglm_refit.fastglm_nb <- .fastglm_refit.fastglm
#' @exportS3Method .fastglm_refit fastglm_truncated
.fastglm_refit.fastglm_truncated <- .fastglm_refit.fastglm
#' @exportS3Method .fastglm_refit fastglm_zi
.fastglm_refit.fastglm_zi <- .fastglm_refit.fastglm
#' @exportS3Method .fastglm_refit fastglm_hurdle
.fastglm_refit.fastglm_hurdle <- .fastglm_refit.fastglm
#' @exportS3Method .fastglm_refit fastglm_ordinal
.fastglm_refit.fastglm_ordinal <- .fastglm_refit.fastglm
#' @exportS3Method .fastglm_refit fastglm_vglm
.fastglm_refit.fastglm_vglm <- .fastglm_refit.fastglm

.fastglm_two_part <- function(object) inherits(object, c("fastglm_zi", "fastglm_hurdle"))

## The internal name of a part of a model with two linear predictors: the
## positive part of a two-part fit is stored as `count`.
.fastglm_part_internal <- function(part) if (identical(part, "positive")) "count" else part

## A refit must use the same observations and response as the original fit.
.fastglm_refit_check <- function(object, fit)
{
    a <- .fastglm_cmp_fingerprint(object)
    b <- .fastglm_cmp_fingerprint(fit)
    same <- .fastglm_same_fingerprint(a, b)
    if (is.na(same)) {
        warning("the refit could not be checked against the data of the original fit ",
                "(the response was not retained)", call. = FALSE)
    } else if (!same) {
        if (abs(a[["n"]] - b[["n"]]) > 1e-12 * max(abs(a[["n"]]), 1))
            stop(sprintf("the refit uses %s observations and the original fit %s",
                         format(b[["n"]]), format(a[["n"]])), call. = FALSE)
        stop("the refit does not use the same response and weights as the original fit",
             call. = FALSE)
    }
    invisible(TRUE)
}

## ---- Formula changes ----------------------------------------------------------

## The factor sets of the terms of tt, as keys that do not depend on the order
## of the variables in an interaction.
.fastglm_term_keys <- function(tt)
{
    labels <- attr(tt, "term.labels"); factors <- attr(tt, "factors")
    if (!length(labels)) return(character())
    vapply(seq_along(labels), function(j)
        paste(sort(rownames(factors)[factors[, j] != 0]), collapse = "\034"), character(1))
}

## The response and offsets of terms, which a refit may not change.
.fastglm_term_special <- function(tt)
{
    v <- vapply(as.list(attr(tt, "variables"))[-1L], deparse1, character(1))
    list(response = v[attr(tt, "response")], offset = sort(v[attr(tt, "offset")]))
}

## Terms after a formula change that removes terms (or the intercept), with
## the data-dependent bases (predvars) of the kept terms preserved. Returns
## the new terms and the indices of the kept terms of tt.
.fastglm_refit_new_terms <- function(tt, change)
{
    change <- stats::as.formula(change)
    if (length(change) == 3L && attr(tt, "response") == 0L) change <- change[-2L]
    nt <- stats::terms(stats::update.formula(stats::formula(tt), change))
    if (!identical(.fastglm_term_special(tt), .fastglm_term_special(nt)))
        stop("a refit for a model comparison cannot change the response or the offsets", call. = FALSE)
    kept <- match(.fastglm_term_keys(nt), .fastglm_term_keys(tt))
    if (anyNA(kept))
        stop("a refit for a model comparison can only remove terms; use update() to add terms",
             call. = FALSE)
    drop <- setdiff(seq_along(attr(tt, "term.labels")), kept)
    out <- if (length(drop)) stats::drop.terms(tt, drop, keep.response = attr(tt, "response") > 0L) else tt
    attr(out, "intercept") <- attr(nt, "intercept")
    list(terms = out, kept = sort(kept), intercept = attr(nt, "intercept"))
}

## The design of new terms from the stored model frame and contrasts. The
## stored frame holds every variable already evaluated (with predvars), so
## nothing is looked up in an environment; model.matrix() matches the
## variables to the frame's columns by name. Without a frame, the columns of
## the stored design are kept by assign, which is valid only when no coding
## changes (the intercept and marginality are then required to stay).
.fastglm_refit_new_design <- function(tt, X, assign, frame, contrasts, change)
{
    nt <- .fastglm_refit_new_terms(tt, change)
    if (!is.null(frame)) {
        vars <- rownames(attr(nt$terms, "factors"))
        ctr <- contrasts[names(contrasts) %in% vars]
        x <- stats::model.matrix(nt$terms, frame, contrasts.arg = if (length(ctr)) ctr)
        return(list(x = x, terms = nt$terms, assign = attr(x, "assign")))
    }
    if (nt$intercept != attr(tt, "intercept") || attr(tt, "intercept") == 0L)
        stop("refitting without the intercept, or a model without an intercept, needs the model frame ",
             "(fit with model = TRUE)", call. = FALSE)
    keys <- strsplit(.fastglm_term_keys(tt), "\034", fixed = TRUE)
    for (j in setdiff(seq_along(keys), nt$kept)) for (k in nt$kept)
        if (all(keys[[j]] %in% keys[[k]]))
            stop("removing a term contained in a remaining term needs the model frame (fit with model = TRUE)",
                 call. = FALSE)
    cols <- which(assign == 0L | assign %in% nt$kept)
    x <- X[, cols, drop = FALSE]
    list(x = x, terms = nt$terms, assign = match(assign[cols], c(0L, nt$kept)) - 1L)
}

## A one-sided change for one part (drops a left-hand side if given).
.fastglm_part_change <- function(change)
{
    change <- stats::as.formula(change)
    if (length(change) == 3L) change <- change[-2L]
    change
}

## The per-part changes of a two-part model: a list with count and zero.
.fastglm_two_part_changes <- function(old, change, part)
{
    if (is.null(part)) {
        old <- Formula::as.Formula(old)
        new <- Formula::as.Formula(stats::update(old, Formula::as.Formula(change)))
        nr <- length(new)[2L]
        return(list(count = stats::formula(new, lhs = 1L, rhs = 1L),
                    zero = stats::formula(new, lhs = 1L, rhs = if (nr > 1L) 2L else 1L)))
    }
    part <- match.arg(part, c("count", "zero"))
    out <- list(count = ~ ., zero = ~ .)
    out[[part]] <- .fastglm_part_change(change)
    out
}

## The new nonparallel formula of a partial ordinal refit.
.fastglm_ordinal_changes <- function(object, change, part, env)
{
    np <- if (length(object$nonparallel.terms)) stats::reformulate(object$nonparallel.terms, env = env)
    part <- if (is.null(part)) "location" else match.arg(part, c("location", "nominal"))
    if (part == "nominal") {
        np <- stats::update.formula(if (is.null(np)) ~ 0 else np, .fastglm_part_change(change))
        location <- ~ .
    } else location <- change
    list(location = location, nonparallel = np)
}

## Keeps the nonparallel terms that remain in the new terms.
.fastglm_keep_nonparallel <- function(np, terms)
{
    if (is.null(np)) return(NULL)
    keep <- intersect(attr(stats::terms(np), "term.labels"), attr(terms, "term.labels"))
    if (length(keep)) stats::reformulate(keep, env = environment(np)) else NULL
}

## Column indices kept after dropping drop_columns (indices or names) from a
## design with column names cn and p columns.
.fastglm_refit_keep <- function(drop_columns, cn, p)
{
    if (is.character(drop_columns)) {
        idx <- match(drop_columns, cn)
        if (anyNA(idx)) stop("design columns not found: ", paste(drop_columns[is.na(idx)], collapse = ", "),
                             call. = FALSE)
    } else if (is.numeric(drop_columns)) {
        idx <- as.integer(drop_columns)
        if (anyNA(idx) || any(idx < 1L | idx > p)) stop("'drop_columns' must index design columns", call. = FALSE)
    } else stop("'drop_columns' must be column indices or names", call. = FALSE)
    keep <- setdiff(seq_len(p), idx)
    if (!length(keep)) stop("a refit needs at least one design column", call. = FALSE)
    keep
}

## ---- In-memory refits ---------------------------------------------------------

## The data an in-memory fit stored: designs X (and Z for two-part models),
## response, prior weights, offsets, terms, assign, model frame, contrasts.
.fastglm_refit_data <- function(object)
{
    need <- function(x, what) if (is.null(x))
        stop(sprintf("the fit did not retain its %s, which a refit needs", what), call. = FALSE) else x
    if (.fastglm_two_part(object)) {
        if (is.null(object$x) && is.null(object$model))
            stop("the fit did not retain its design or model frame, which a refit needs", call. = FALSE)
        d <- .count_design(object)
        y <- need(.fastglm_cmp_twopart_y(object), "response")
        off <- d$offset
        n <- NROW(y)
        if (is.null(off)) off <- list(count = rep(0, n), zero = rep(0, n))
        return(list(X = d$x$count, Z = d$x$zero, y = y, weights = object$weights, offset = off,
                    terms = object$terms, assign = object$assign, frame = object$model,
                    contrasts = object$contrasts, formula = object$formula))
    }
    if (inherits(object, "fastglm_ordinal")) {
        return(list(X = need(object$x, "design"), y = need(object$y, "response"),
                    weights = object$weights, offset = object$offset, terms = object$terms,
                    assign = object$design.assign, frame = object$model, contrasts = object$contrasts))
    }
    if (inherits(object, "fastglm_vglm")) {
        X <- need(object$x, "design")
        return(list(X = X, y = need(object$y, "response"), weights = object$prior.weights,
                    offset = object$offset, terms = object$terms, assign = attr(X, "assign"),
                    frame = object$model, contrasts = object$contrasts))
    }
    X <- object$x
    if (is.null(X) && !is.null(object$terms) && !is.null(object$model)) X <- model.matrix(object)
    X <- need(X, "design")
    y <- need(object$y, "response")
    off <- object$offset
    if (is.null(off)) off <- rep(0, NROW(y))
    list(X = X, y = y, weights = object$prior.weights, offset = off, terms = object$terms,
         assign = attr(X, "assign"), frame = object$model, contrasts = object$contrasts)
}

.fastglm_refit_memory <- function(object, formula, drop_columns, part, extras, opts = list())
{
    d <- .fastglm_refit_data(object)
    two <- .fastglm_two_part(object)
    ordinal <- inherits(object, "fastglm_ordinal")
    np <- NULL
    if (!is.null(formula)) {
        if (is.null(d$terms))
            stop("this fit uses a design matrix; refit it with 'drop_columns'", call. = FALSE)
        if (two) {
            ch <- .fastglm_two_part_changes(d$formula, formula, part)
            for (k in c("count", "zero")) {
                nd <- .fastglm_refit_new_design(d$terms[[k]], if (k == "count") d$X else d$Z,
                                                d$assign[[k]], d$frame, d$contrasts[[k]], ch[[k]])
                if (k == "count") d$X <- nd$x else d$Z <- nd$x
            }
        } else if (ordinal) {
            ch <- .fastglm_ordinal_changes(object, formula, part, environment(stats::formula(d$terms)))
            # The ordinal design has no intercept column; the stored assign
            # maps its columns to terms.
            nd <- .fastglm_refit_new_design(d$terms, cbind(`(Intercept)` = 1, d$X)[, , drop = FALSE],
                                            c(0L, d$assign), d$frame, d$contrasts, ch$location)
            keep <- nd$assign != 0L
            d$X <- nd$x[, keep, drop = FALSE]
            np <- .fastglm_keep_nonparallel(ch$nonparallel, nd$terms)
            np <- if (is.null(np)) integer() else
                .ordinal_nonparallel_terms(np, nd$terms, nd$assign[keep])$columns
        } else {
            if (!is.null(part)) stop("'part' applies only to models with more than one linear predictor",
                                     call. = FALSE)
            d$X <- .fastglm_refit_new_design(d$terms, d$X, d$assign, d$frame, d$contrasts, formula)$x
        }
    } else if (!is.null(drop_columns)) {
        if (!is.null(d$terms) && !isTRUE(opts$columns))
            stop("this fit uses a formula; refit it with 'formula'", call. = FALSE)
        if (two) {
            part <- if (is.null(part)) "count" else match.arg(part, c("count", "zero"))
            x <- if (part == "count") d$X else d$Z
            keep <- .fastglm_refit_keep(drop_columns, colnames(x), ncol(x))
            if (part == "count") d$keepX <- keep else d$keepZ <- keep
        } else {
            if (!is.null(part)) stop("'part' applies only to models with more than one linear predictor",
                                     call. = FALSE)
            d$keepX <- .fastglm_refit_keep(drop_columns, colnames(d$X), ncol(d$X))
            if (ordinal && inherits(object, "fastglm_ordinal_partial"))
                np <- which(d$keepX %in% object$nonparallel)
        }
    } else if (inherits(object, "fastglm_ordinal_partial")) np <- object$nonparallel
    if (ordinal && is.null(np) && inherits(object, "fastglm_ordinal_partial")) np <- object$nonparallel
    .fastglm_refit_fit(object, d, np, extras, nb_theta_fixed = isTRUE(opts$nb_theta_fixed))
}

## The GLM fitting controls of a refit: family, tolerance, iteration limit,
## bias reduction, covariance rule, and decomposition (an integer method code
## for in-memory designs; the fit's solver name for streamed ones). A
## negative binomial fit with theta held fixed is refitted as a GLM with
## family negbin(theta), with the tolerance, iteration limit, and method of
## its own fit.
.fastglm_refit_glm_controls <- function(object, nb_theta_fixed = FALSE)
{
    if (nb_theta_fixed) {
        rc <- object$refit.controls
        link <- if (!is.null(object$family$link)) object$family$link else "log"
        # An infinite theta (the Poisson limit, where fastglm_nb() returns the
        # Poisson fit) is held fixed as the Poisson family.
        fam <- if (is.finite(object$theta)) negbin(object$theta, link = link) else stats::poisson(link = link)
        return(list(family = fam,
                    tol = if (is.null(rc$tol)) 1e-8 else rc$tol,
                    maxit = if (is.null(rc$maxit)) 100L else rc$maxit,
                    firth = FALSE, covariance = "estimate",
                    method = if (is.null(rc$method)) 2L else rc$method))
    }
    ctl <- if (inherits(object, "fastglmFit")) object$control else object$compare.controls
    method <- if (!is.null(ctl$fastmethod)) ctl$fastmethod else if (!is.null(ctl$method)) ctl$method else 0L
    list(family = object$family, tol = if (is.null(ctl$tol)) 1e-8 else ctl$tol,
         maxit = if (is.null(ctl$maxit)) 100L else ctl$maxit,
         firth = isTRUE(object$firth) || isTRUE(ctl$firth),
         covariance = if (is.null(ctl$covariance)) "estimate" else ctl$covariance,
         method = method)
}

## A row-block source over designs held in memory or in a big.matrix, keeping
## columns keepX of X (and keepZ of Z); nothing is copied but one block.
.fastglm_refit_block_source <- function(d)
{
    X <- d$X; Z <- d$Z; keepX <- d$keepX; keepZ <- d$keepZ
    if (is.null(keepX)) keepX <- seq_len(ncol(X))
    if (!is.null(Z) && is.null(keepZ)) keepZ <- seq_len(ncol(Z))
    # Rows per block as the package's tiled reductions (max(256, 2^20 / p)),
    # so a block holds about 8 MB of design whatever the number of columns.
    block.size <- max(256L, as.integer(floor(2^20 / (length(keepX) + length(keepZ)))))
    y <- d$y; w <- d$weights; off <- d$offset
    n <- NROW(y)
    cols <- function(x, keep) { cn <- colnames(x); if (is.null(cn)) cn <- paste0("X", seq_len(ncol(x))); cn[keep] }
    cx <- cols(X, keepX); cz <- if (!is.null(Z)) cols(Z, keepZ)
    rows <- function(v, i) if (is.null(v)) NULL else if (is.list(v)) lapply(v, rows, i) else
        if (is.matrix(v)) v[i, , drop = FALSE] else v[i]
    block.size <- as.integer(min(block.size, n))
    fastglm_source(function(k) {
        i <- seq.int((k - 1L) * block.size + 1L, min(n, k * block.size))
        xx <- X[i, keepX, drop = FALSE]; colnames(xx) <- cx
        out <- list(X = xx, y = rows(y, i), weights = rows(w, i), offset = rows(off, i))
        if (!is.null(Z)) { zz <- Z[i, keepZ, drop = FALSE]; colnames(zz) <- cz; out$Z <- zz }
        out
    }, ceiling(n / block.size))
}

## The designs of a refit: column subsets of in-memory designs, or a block
## source when a design is a big.matrix.
.fastglm_refit_designs <- function(d)
{
    big <- inherits(d$X, "big.matrix") || inherits(d$Z, "big.matrix")
    if (big) {
        .check_big_alive(d$X)
        return(list(source = .fastglm_refit_block_source(d)))
    }
    X <- if (is.null(d$keepX)) d$X else d$X[, d$keepX, drop = FALSE]
    Z <- if (is.null(d$Z)) NULL else if (is.null(d$keepZ)) d$Z else d$Z[, d$keepZ, drop = FALSE]
    list(X = X, Z = Z)
}

## Fits a refit of an in-memory model from its stored data (d) and controls.
.fastglm_refit_fit <- function(object, d, np, extras, nb_theta_fixed = FALSE)
{
    g <- .fastglm_refit_designs(d)
    src <- g$source
    call.fit <- function(fun, args) do.call(fun, utils::modifyList(args, extras))
    if (inherits(object, "fastglm_vglm")) {
        rc <- object$refit.controls
        ctl <- rc$control; if (is.list(ctl)) ctl$threads <- NULL
        args <- list(reference = object$reference, firth = isTRUE(object$firth), tol = rc$tol,
                     maxit = rc$maxit, block.size = rc$block.size, control = ctl)
        args <- if (is.null(src)) c(list(x = g$X, y = d$y, weights = d$weights, offset = d$offset), args)
                else c(list(x = src, levels = object$levels), args)
        return(call.fit(fastglm_multinom, args))
    }
    if (inherits(object, "fastglm_ordinal")) {
        ctl <- object$control; if (is.list(ctl)) { ctl$threads <- NULL; ctl$trace <- NULL }
        args <- list(x = g$X, y = d$y, weights = d$weights, offset = d$offset, firth = object$firth,
                     tol = object$tol, maxit = object$maxit, block.size = object$block.size,
                     control = ctl, link = object$link, nonparallel = if (length(np)) np)
        return(call.fit(fastglm_ordinal, args))
    }
    if (inherits(object, "fastglm_hurdle") && !inherits(object, "fastglm_twopart") &&
        is.null(src) && !is.null(object$terms) && is.matrix(g$X) && is.matrix(g$Z)) {
        # A hurdle formula fit is refitted through the formula interface (its
        # own fitting path), on its design columns.
        return(.fastglm_refit_hurdle_formula(object, g, d, extras))
    }
    if (.fastglm_two_part(object)) {
        data <- if (is.null(src)) list(formula = g$X, response = d$y, z = g$Z, weights = d$weights,
                                       offset = d$offset) else list(formula = src)
        if (inherits(object, "fastglm_twopart")) {
            fun <- fastglm_twopart
            ctl <- object$refit.controls
        } else if (inherits(object, "fastglm_zi")) {
            fun <- fastglm_zi
            ctl <- c(list(dist = object$dist, link = object$link), object$compare.controls)
        } else {
            fun <- fastglm_hurdle
            ctl <- object$refit.controls
        }
        ctl <- ctl[names(ctl) %in% names(formals(fun))]
        return(call.fit(fun, c(data, ctl)))
    }
    if (inherits(object, "fastglm_truncated")) {
        cc <- object$compare.controls
        args <- c(if (is.null(src)) list(x = g$X, y = d$y, weights = d$weights, offset = d$offset)
                  else list(x = src),
                  list(dist = object$dist), cc[names(cc) %in% names(formals(fastglm_truncated))])
        return(call.fit(fastglm_truncated, args))
    }
    if (inherits(object, "fastglm_nb") && !nb_theta_fixed) {
        rc <- object$refit.controls
        rc <- rc[names(rc) %in% names(formals(fastglm_nb))]
        if (!is.null(object$compare.controls$poisson.limit)) rc$poisson.limit <- object$compare.controls$poisson.limit
        args <- c(if (is.null(src)) list(x = g$X, y = d$y, weights = d$weights, offset = d$offset)
                  else list(x = src), rc)
        return(call.fit(fastglm_nb, args))
    }
    # GLMs: the controls of a formula fit (control) or a matrix fit, or a
    # negative binomial fit with theta held fixed.
    common <- .fastglm_refit_glm_controls(object, nb_theta_fixed)
    method <- common$method; common$method <- NULL
    if (!is.null(src)) {
        # A big.matrix design: read the kept columns in row blocks.
        return(call.fit(fastglm_streaming, c(list(chunk_callback = src,
                                                  method = if (isTRUE(object$solver %in% c("llt", "ldlt"))) object$solver else "llt"),
                                             common)))
    }
    call.fit(fastglm, c(list(x = g$X, y = d$y, weights = d$weights, offset = d$offset, method = method), common))
}

## A hurdle formula fit refitted on its stored design columns through the
## formula interface: the columns become variables of a data frame, so no
## coding or basis is recomputed.
.fastglm_refit_hurdle_formula <- function(object, g, d, extras)
{
    pc <- ncol(g$X); pz <- ncol(g$Z)
    cx <- paste0(".count", seq_len(pc)); cz <- paste0(".zero", seq_len(pz))
    data <- data.frame(.response = d$y, g$X, g$Z, .offc = d$offset$count, .offz = d$offset$zero,
                       check.names = FALSE)
    names(data) <- c(".response", cx, cz, ".offc", ".offz")
    formula <- stats::as.formula(paste(".response ~ 0 +", paste(cx, collapse = " + "),
                                       "+ offset(.offc) | 0 +", paste(cz, collapse = " + "),
                                       "+ offset(.offz)"), env = environment(.fastglm_refit_hurdle_formula))
    ctl <- object$refit.controls
    ctl <- ctl[names(ctl) %in% names(formals(fastglm_hurdle))]
    f <- do.call(fastglm_hurdle, utils::modifyList(c(list(formula = formula, data = data, weights = d$weights),
                                                     ctl), extras))
    names(f$coefficients$count) <- colnames(g$X)
    names(f$coefficients$zero) <- colnames(g$Z)
    nm <- c(paste0("count_", colnames(g$X)), paste0("zero_", colnames(g$Z)))
    if (!is.null(f$vcov)) dimnames(f$vcov) <- list(nm, nm)
    f
}

## ---- Streamed refits ----------------------------------------------------------

## The formula of a source with a formula specification.
.fastglm_source_formula <- function(source)
{
    s <- source$schema
    if (is.null(s) || is.null(s$terms)) return(NULL)
    if (inherits(s$terms, "terms")) stats::formula(s$terms) else Formula::as.Formula(s$formula)
}

## A source with the same blocks and a new formula, built as a new source
## would be (the prototype fixes factor levels, contrasts, and bases).
.fastglm_source_reformulate <- function(source, formula, cache = isTRUE(source$spec$cache),
                                        args = list())
{
    sp <- source$spec
    if (is.null(sp))
        stop("this source has no formula specification to change; build a new source with fastglm_source()",
             call. = FALSE)
    base <- list(callback = sp$callback, n_chunks = source$n_chunks, reset = sp$reset, formula = formula,
                 prototype = sp$prototype, weights = sp$weights, offset = sp$offset,
                 cache = cache, reusable = !isFALSE(source$reusable))
    out <- do.call(fastglm_source, utils::modifyList(base, args, keep.null = TRUE))
    out$origin <- source$origin
    out
}

## A source that reads the original one and keeps some design columns (for
## sources without a formula, whose columns have a fixed coding): keep is a
## list with count (X) and, optionally, zero (Z) column indices. With
## schema = FALSE the new source has no formula schema, so a formula source's
## blocks are read as built (the kept columns keep their coding) and the
## refit is a design-matrix fit.
.fastglm_refit_subset_source <- function(source, keep, two = FALSE, schema = TRUE)
{
    force(source); force(keep); force(two)
    callback <- function(k, ...) {
        z <- source$callback(k, ...)
        if (is.data.frame(z$X)) z$X <- as.matrix(z$X)
        if (two && is.null(z$Z)) z$Z <- z$X
        if (!is.null(keep$count)) z$X <- z$X[, keep$count, drop = FALSE]
        if (!is.null(keep$zero)) z$Z <- z$Z[, keep$zero, drop = FALSE]
        z
    }
    structure(list(callback = callback, n_chunks = source$n_chunks, schema = if (schema) source$schema,
                   cache = NULL, reusable = source$reusable, origin = source$origin),
              class = "fastglm_source")
}

.fastglm_refit_streamed <- function(object, formula, drop_columns, part, extras, opts = list())
{
    source <- object$source
    two <- .fastglm_two_part(object)
    ordinal <- inherits(object, "fastglm_ordinal")
    np <- if (ordinal && length(object$nonparallel.terms))
        stats::reformulate(object$nonparallel.terms) else NULL
    if (!is.null(formula)) {
        old <- .fastglm_source_formula(source)
        if (is.null(old))
            stop("this fit reads design matrices without a formula; refit it with 'drop_columns'", call. = FALSE)
        if (two) {
            ch <- .fastglm_two_part_changes(old, formula, part)
            f1 <- stats::update.formula(stats::formula(old, lhs = 1L, rhs = 1L), ch$count)
            nr <- length(old)[2L]
            f2 <- stats::update.formula(stats::formula(old, lhs = 1L, rhs = if (nr > 1L) 2L else 1L), ch$zero)
            new <- Formula::as.Formula(stats::as.formula(
                call("~", f1[[2L]], call("|", f1[[3L]], f2[[3L]])), env = environment(old)))
        } else if (ordinal) {
            ch <- .fastglm_ordinal_changes(object, formula, part, environment(old))
            new <- stats::update.formula(old, ch$location)
            np <- .fastglm_keep_nonparallel(ch$nonparallel, stats::terms(new))
        } else {
            if (!is.null(part)) stop("'part' applies only to models with more than one linear predictor",
                                     call. = FALSE)
            new <- stats::update.formula(old, formula)
        }
        # No block cache: a refit would otherwise write a copy of its design
        # to disk; it reads the source's blocks directly.
        source <- .fastglm_source_reformulate(source, new, cache = FALSE)
    } else if (!is.null(drop_columns) && isTRUE(opts$columns) && !is.null(source$schema$terms)) {
        # Columns of a formula source's blocks, with no recoding (GLM and
        # negative binomial fits).
        cn <- names(object$coefficients)
        keep <- .fastglm_refit_keep(drop_columns, cn, length(cn))
        source <- .fastglm_refit_subset_source(source, list(count = keep), schema = FALSE)
    } else if (!is.null(drop_columns)) {
        if (!is.null(source$schema$terms))
            stop("this fit reads a formula source; refit it with 'formula'", call. = FALSE)
        if (ordinal) {
            cn <- object$state$columns
        } else if (two) {
            part <- if (is.null(part)) "count" else match.arg(part, c("count", "zero"))
            cn <- names(object$coefficients[[part]])
        } else {
            if (!is.null(part)) stop("'part' applies only to models with more than one linear predictor",
                                     call. = FALSE)
            cn <- if (inherits(object, "fastglm_vglm")) object$design.names else names(object$coefficients)
        }
        keep <- .fastglm_refit_keep(drop_columns, cn, length(cn))
        if (ordinal && inherits(object, "fastglm_ordinal_partial")) {
            np <- intersect(cn[object$nonparallel], cn[keep])
            if (!length(np)) np <- NULL
        }
        source <- .fastglm_refit_subset_source(source,
            if (two && part == "zero") list(zero = keep) else list(count = keep), two = two)
    }
    call.fit <- function(fun, args) do.call(fun, utils::modifyList(args, extras))
    if (inherits(object, "fastglm_vglm")) {
        rc <- object$refit.controls
        ctl <- rc$control; if (is.list(ctl)) ctl$threads <- NULL
        return(call.fit(fastglm_multinom, list(x = source, levels = object$levels,
            reference = object$reference, firth = isTRUE(object$firth), tol = rc$tol,
            maxit = rc$maxit, block.size = rc$block.size, control = ctl)))
    }
    if (ordinal) {
        ctl <- object$control; if (is.list(ctl)) { ctl$threads <- NULL; ctl$trace <- NULL }
        return(call.fit(fastglm_ordinal_streaming, list(source = source, levels = object$levels,
            firth = object$firth, link = object$link, tol = object$tol, maxit = object$maxit,
            block.size = object$block.size, control = ctl, nonparallel = np)))
    }
    if (two) {
        fun <- if (inherits(object, "fastglm_twopart")) fastglm_twopart else
            if (inherits(object, "fastglm_zi")) fastglm_zi else fastglm_hurdle
        ctl <- if (inherits(object, "fastglm_zi"))
            c(list(dist = object$dist, link = object$link), object$compare.controls) else object$refit.controls
        ctl <- ctl[names(ctl) %in% names(formals(fun))]
        return(call.fit(fun, c(list(formula = source), ctl)))
    }
    if (inherits(object, "fastglm_truncated")) {
        cc <- object$compare.controls
        return(call.fit(fastglm_truncated, c(list(x = source, dist = object$dist),
                                             cc[names(cc) %in% names(formals(fastglm_truncated))])))
    }
    if (inherits(object, "fastglm_nb") && !isTRUE(opts$nb_theta_fixed)) {
        rc <- object$refit.controls
        rc <- rc[names(rc) %in% names(formals(fastglm_nb))]
        if (!is.null(object$compare.controls$poisson.limit)) rc$poisson.limit <- object$compare.controls$poisson.limit
        return(call.fit(fastglm_nb, c(list(x = source), rc)))
    }
    common <- .fastglm_refit_glm_controls(object, isTRUE(opts$nb_theta_fixed))
    common$method <- if (!is.null(object$solver) && object$solver %in% c("qr", "llt", "ldlt")) object$solver else "llt"
    call.fit(fastglm_streaming, c(list(chunk_callback = source), common))
}

## ---- Formulas -----------------------------------------------------------------

## A model formula given as a list with elements count and zero (formulas or
## terms objects), as stats::step() stores the terms of a model with two
## linear predictors in its call, becomes the two-part formula
## `y ~ count terms | zero terms`. Other values are returned unchanged.
.two_part_formula_arg <- function(formula)
{
    if (!is.list(formula) || is.data.frame(formula) || inherits(formula, c("formula", "fastglm_source")))
        return(formula)
    nm <- names(formula)
    if (is.null(nm) || !all(c("count", "zero") %in% nm) ||
        !all(vapply(formula[c("count", "zero")], inherits, NA, "formula")))
        return(formula)
    fc <- stats::formula(formula$count); fz <- stats::formula(formula$zero)
    if (length(fc) != 3L || length(fz) < 2L)
        stop("a list formula needs a two-sided count formula and a zero formula", call. = FALSE)
    Formula::as.Formula(stats::as.formula(call("~", fc[[2L]], call("|", fc[[3L]], fz[[length(fz)]])),
                                          env = environment(fc)))
}

## A model formula given next to a data source (as stats::step() writes the
## terms of a streamed fit into its call) rebuilds the source with that
## formula, as update() of the source does; it cannot be given with a design
## matrix.
.fastglm_formula_source <- function(x, formula)
{
    if (!inherits(x, "fastglm_source"))
        stop("'formula' can be given with a model formula or a fastglm_source() with a formula, ",
             "not with a design matrix", call. = FALSE)
    stats::update(x, formula)
}

## formula() of a fitted model: the model formula without the attributes of
## a terms object (stats::formula.default() would return the stored terms
## with all of their attributes). Models with two linear predictors give the
## two-part formula, and streamed fits the formula of their source.
.fastglm_formula <- function(x, ...)
{
    f <- x$formula
    if (is.list(f) && !inherits(f, "formula")) f <- .two_part_formula_arg(f)
    if (inherits(f, "formula")) return(if (inherits(f, "terms")) stats::formula(f) else f)
    tt <- x$terms
    if (inherits(tt, "terms")) return(stats::formula(tt))
    if (!is.null(x$source)) {
        sf <- .fastglm_source_formula(x$source)
        if (!is.null(sf)) return(sf)
    }
    stop("this fit has no formula (it was fitted to a design matrix)", call. = FALSE)
}

#' @exportS3Method stats::formula
formula.fastglm_nb <- .fastglm_formula
#' @exportS3Method stats::formula
formula.fastglm_truncated <- .fastglm_formula
#' @exportS3Method stats::formula
formula.fastglm_zi <- .fastglm_formula
#' @exportS3Method stats::formula
formula.fastglm_hurdle <- .fastglm_formula
#' @exportS3Method stats::formula
formula.fastglm_ordinal <- .fastglm_formula
#' @exportS3Method stats::formula
formula.fastglm_vglm <- .fastglm_formula
#' @exportS3Method stats::formula
formula.fastglm_stream <- .fastglm_formula

## ---- update() -----------------------------------------------------------------

## update() as stats::update.default: the call is changed with the
## unevaluated arguments and evaluated in the caller's frame. Fitting
## functions that take the formula as `x` get it there; streamed fits get
## stats::update(<source>, <formula>), which rebuilds the source.
.fastglm_update <- function(object, formula., extras, evaluate, envir)
{
    call <- object$call
    if (is.null(call)) stop("need an object with call component", call. = FALSE)
    # stats::step() stores the terms as call$formula; these fitting functions
    # take the formula as their first argument, which update() changes.
    call$formula <- NULL
    if (!is.null(formula.)) {
        if (!is.null(object$source)) {
            arg <- if (inherits(object, c("fastglm_nb", "fastglm_truncated"))) "x" else "chunk_callback"
            if (is.null(call[[arg]])) { names(call)[2L] <- arg }
            old <- .fastglm_source_formula(object$source)
            if (is.null(old)) stop("this fit reads design matrices without a formula; build a new source",
                                   call. = FALSE)
            call[[arg]] <- as.call(list(quote(stats::update), call[[arg]], stats::update.formula(old, formula.)))
        } else {
            if (is.null(object$terms) || !inherits(object$terms, "terms"))
                stop("update() with a formula needs a formula fit", call. = FALSE)
            if (is.null(call$x)) names(call)[2L] <- "x"
            call$x <- stats::update.formula(stats::formula(object$terms), formula.)
        }
    }
    if (length(extras)) {
        existing <- !is.na(match(names(extras), names(call)))
        for (a in names(extras)[existing]) call[[a]] <- extras[[a]]
        if (any(!existing)) call <- as.call(c(as.list(call), extras[!existing]))
    }
    if (evaluate) eval(call, envir) else call
}

#' Update and refit a model
#'
#' `update()` methods for negative binomial, zero-truncated, and streamed
#' fits, whose fitting functions take the formula or the data source as their
#' first argument (`x` or `chunk_callback`), and for data sources. They behave
#' as [stats::update()]: `formula.` changes the formula (for example
#' `. ~ . - a`), further named arguments replace those of the original call,
#' and the changed call is evaluated in the calling frame. A streamed fit
#' with a formula source is refitted on the source with the changed formula,
#' which rebuilds each block from the source's prototype; the blocks must
#' still be readable.
#' @param object a fit from [fastglm_nb()], [fastglm_truncated()], or
#'   [fastglm_streaming()], or a [fastglm_source()] made with a formula.
#' @param formula. changes to the formula, as in [stats::update.formula()].
#' @param ... further arguments of the fitting function to change; for a
#'   source, `reset`, `prototype`, `weights`, `offset`, `cache`, or
#'   `reusable` (other names are an error).
#' @param evaluate `TRUE` (default) to refit, `FALSE` to return the call.
#' @returns The refitted model, or its call when `evaluate = FALSE`. For a
#'   source, a new source with the same blocks and the changed formula.
#' @name update-fastglm
#' @md
#' @exportS3Method stats::update
update.fastglm_nb <- function(object, formula., ..., evaluate = TRUE)
{
    .fastglm_update(object, if (missing(formula.)) NULL else formula.,
                    as.list(match.call(expand.dots = FALSE)$...), evaluate, parent.frame())
}

#' @rdname update-fastglm
#' @exportS3Method stats::update
update.fastglm_truncated <- update.fastglm_nb

#' @rdname update-fastglm
#' @exportS3Method stats::update
update.fastglm_stream <- update.fastglm_nb

#' @rdname update-fastglm
#' @exportS3Method stats::update
update.fastglm_source <- function(object, formula., ...)
{
    args <- list(...)
    allowed <- c("reset", "prototype", "weights", "offset", "cache", "reusable")
    if (length(args) && (is.null(names(args)) || any(!nzchar(names(args)))))
        stop("arguments to update() of a source must be named", call. = FALSE)
    bad <- setdiff(names(args), allowed)
    if (length(bad))
        stop("update() of a source cannot change: ", paste(bad, collapse = ", "),
             " (it can change the formula and ", paste(allowed, collapse = ", "), ")", call. = FALSE)
    old <- .fastglm_source_formula(object)
    if (is.null(old)) stop("this source has no formula to change; build a new source", call. = FALSE)
    new <- if (missing(formula.)) old else
        if (inherits(old, "Formula")) Formula::as.Formula(stats::update(old, Formula::as.Formula(formula.)))
        else stats::update.formula(old, formula.)
    .fastglm_source_reformulate(object, new, args = args)
}
