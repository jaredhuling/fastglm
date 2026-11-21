## Term-by-term comparison: drop1() for every fitted class, and anova() of a
## single model (the sequential table) for the classes with one linear
## predictor. Refits go through .fastglm_refit() (compare-core.R) and are
## compared with the full model by the rules of compare-anova.R.

#' Single term deletions for fastglm models
#'
#' `drop1()` refits the model without each term in `scope` (on the data and
#' with the weights, offsets, family, link, method, and controls of the fit,
#' including `threads`, capped at the session setting of
#' [fastglm_threads()]) and compares each refit with the full model. `anova()`
#' of a single model gives the sequential table: the terms are added in the
#' order of the formula, each compared with the model before it.
#'
#' @param object a fitted model.
#' @param scope the terms to drop. For a fit with a formula: a character
#'   vector of term labels or a formula whose terms are taken, as in
#'   [stats::drop1()]; the default is the terms that can be dropped while
#'   respecting marginality ([stats::drop.scope()]), so for `y ~ a * b` only
#'   `a:b`. For a model with two linear predictors (zero-inflated, hurdle,
#'   two-part) or a partial proportional odds model, a list with elements
#'   named by part (`count`, `zero`; `positive`, `zero` for two-part fits,
#'   where `count` is accepted for `positive`; `location`, `nominal`) gives
#'   the scope of each part, and a vector or formula applies to every part
#'   that has the term. For a fit to a design matrix (including `big.matrix` and streamed
#'   design matrices): a list of column groups, each a vector of column
#'   indices or names dropped together (for example `list(g = c(2, 3))`), or a
#'   vector of columns dropped one at a time; the default drops each
#'   non-intercept column on its own. For a model with two linear predictors
#'   fitted to design matrices, a list with elements `count` and `zero` of
#'   such groups.
#' @param scale for generalized linear and negative binomial models, a
#'   positive value replaces the dispersion that scales the deviances (and,
#'   for Gaussian models, the log likelihood in the AIC column), as in
#'   [stats::drop1.glm()]; 0 (the default) uses the fit's dispersion. Other
#'   models accept only 0.
#' @param test `"none"` (the default, as [stats::drop1()]), `"LRT"` or its
#'   synonym `"Chisq"` for the likelihood ratio test, `"Rao"` for the score
#'   test, or `"F"` for the F test of generalized linear and negative
#'   binomial models. The value is matched as [match.arg()] does (case
#'   sensitive). `"Rao"` is available where [fastglm_score_test()] supports
#'   the fit: generalized linear models fitted by maximum likelihood and held
#'   in memory (not `big.matrix`), and ordinal models fitted by maximum
#'   likelihood and held in memory. Other choices give an error naming the
#'   available tests, before any refit.
#' @param k the penalty per parameter in the AIC column (2 for AIC,
#'   `log(n)` for BIC), as in [stats::drop1()].
#' @param trace `TRUE` traces each refit (with the trace of the fitting
#'   function), after a line naming it, such as `drop1 refit 2/6: - b`. Any
#'   other value (including the `trace = 1` that [stats::step()] passes)
#'   leaves the refits silent.
#' @param ... further arguments are not used (a warning names them).
#'
#' @details
#' **Generalized linear and negative binomial models** follow
#' [stats::drop1.glm()]: the columns of a term are removed from the design
#' (by the design's `assign` attribute), without recoding the other terms, and
#' the table has the columns `Df`, `Deviance`, and `AIC` of
#' [stats::drop1.glm()], with `LRT` (or `scaled dev.` when the dispersion is
#' estimated), `Rao score`, or `F value`, and their p-values. For negative
#' binomial fits theta is held at the full model's estimate in every refit,
#' as `drop1()` does for a `MASS::glm.nb()` fit (through
#' [stats::drop1.glm()]); the AIC column then counts the coefficients only,
#' as `MASS`'s `extractAIC()` does, so it differs from [AIC()] by 2. When the
#' estimate of theta is infinite (the Poisson limit, where [fastglm_nb()]
#' returns the Poisson fit), every refit is a Poisson fit.
#'
#' **Other models** (truncated, zero-inflated, hurdle, two-part, ordinal, and
#' multinomial) are refitted without the term, as [stats::drop1()] does
#' through [update()] for `MASS::polr()` and `nnet::multinom()` fits. The table
#' has the columns `Df` (the number of parameters removed), `AIC`, and `LRT`
#' (twice the difference of the log likelihoods) or `Rao score`, and the
#' p-value. For two-part fits both use [logLik()], which maximizes over the
#' Gamma shape (see [fastglm_twopart()]).
#'
#' **Models with two linear predictors** get one row per term per part,
#' named by part (for example `count: a` and `zero: a`, or `positive: a` and
#' `zero: a` for a two-part fit, whose first part models the positive
#' outcomes); each refit drops the term from that part only. A partial proportional odds model with a
#' term `a` that has cutpoint-specific effects lists it twice: `a (all
#' effects)` removes the term (its test has the degrees of freedom of all of
#' its cutpoint-specific coefficients), and `a (nonparallel)` replaces its
#' cutpoint-specific effects by a common effect, which tests the
#' proportional odds restriction for `a`. Other terms keep their label. A
#' list `scope` for these models has elements `location` and `nominal`. A
#' partial proportional odds model fitted to a design matrix (without a
#' formula) gets rows for column groups of the location design only. These
#' models have no sequential `anova()` table, since the order in which terms
#' enter has no clear meaning across parts.
#'
#' **Bias-reduced (Firth) fits** use the penalized likelihood ratio test of
#' [fastglm_profile_test()] on the full model, which tests jointly that the
#' coefficients of the term are zero under the full model's penalty (as
#' `drop1()` for a `logistf` fit). This is available for binomial-logit Firth
#' fits held in memory with a dense design; other bias-reduced fits give an
#' error. The sequential table of a Firth formula fit applies the same test to
#' each model of the sequence.
#'
#' **Streamed and `big.matrix` fits** are refitted by reading the source
#' (or its block cache, or the `big.matrix` in blocks of rows) again, so
#' each refit is a streamed fit, which makes several passes over the data
#' (one or more per iteration). Before the first refit one line says what
#' will be read: the number of refits, and, for a source made with `cache`,
#' the one pass that builds the block cache of the call. A refit by formula
#' (the classes other than generalized linear and negative binomial models)
#' rebuilds each block from the raw data with the reduced formula, so it
#' cannot use the source's cache of built blocks; for a source made with
#' `cache`, `drop1()` and `anova()` therefore read the source once at the
#' start of the call into a block cache of the raw data in [tempdir()],
#' every refit reads that cache, and the cache is removed when the call ends
#' (also on an error or an interrupt). The source's callback is thus read
#' once per call, as the original fit's cache intends. Without `cache`, each
#' refit reads the source on every pass. The per-call cache holds the raw
#' blocks uncompressed, so while the call runs it needs disk space in
#' [tempdir()] about equal to the uncompressed raw data, in addition to the
#' source's own cache. Refits by columns (generalized linear and negative
#' binomial models) read the source's own cache. On a cached streamed fit,
#' each model that `step()` moves to is a new fit with its own source cache,
#' removed when that fit is garbage collected. The refits are not traced unless `trace = TRUE`. A source declared one-shot
#' (`fastglm_source(reusable = FALSE)`) gives an error before any refit.
#'
#' **Model selection.** `extractAIC()` methods for these classes return the
#' number of parameters and the AIC with penalty `k` (the coefficients for
#' generalized linear and negative binomial models, as `extractAIC()` of
#' `glm()` and `MASS::glm.nb()` fits; every parameter of [logLik()]
#' otherwise), the values of the AIC column. With [update()], this lets
#' [stats::step()] and `MASS::stepAIC()` run on fits held in memory, and the
#' fit that `step()` returns can be evaluated again from its call and
#' updated. On a streamed fit each `drop1()` call that `step()` makes
#' announces its refits (and, for a cached source, builds its own block
#' cache), and each model that `step()` moves to is a new streamed fit.
#' `step()` and `MASS::stepAIC()` do not select terms of models with two
#' linear predictors (zero-inflated, hurdle, and two-part), whose terms
#' belong to a part: they return the fit unchanged, with a warning (from
#' `extractAIC()`, which they call) pointing to `drop1()`, which gives the
#' tests for each part.
#' For fits with zero prior weights, the number of parameters is the rank,
#' while `extractAIC()` of `glm()` and `MASS::glm.nb()` fits counts
#' `n - df.residual`, which adds the number of rows with zero weight. The
#' difference is the same for every model of the same data, so `step()`
#' selects the same models and the AIC values differ by a constant (for
#' `glm()` fits, only when `k != 2`).
#'
#' **Sequential tables.** `anova(object)` for a model with a formula and one
#' linear predictor refits the nested sequence: for generalized linear and
#' negative binomial models (theta held fixed, as `MASS`'s `anova()` of one
#' `glm.nb()` fit) by design columns as [stats::anova.glm()], with its table;
#' for truncated, ordinal, and multinomial fits by formula, with the
#' likelihood ratio table of [anova.fastglm()] and rows named by the term
#' added. Formula [fastglm()] fits that are not bias-reduced keep
#' [stats::anova.glm()] and [stats::drop1.glm()].
#'
#' @returns An object of class `c("anova", "data.frame")` with a `heading`
#'   attribute, with a first row `<none>` (the full model) for `drop1()` and
#'   `NULL` (the model without terms) for the sequential `anova()`.
#'   `extractAIC()` returns `c(edf, AIC)`.
#' @seealso [anova.fastglm()] for comparisons of two or more fitted models.
#' @examples
#' set.seed(1)
#' d <- data.frame(a = rnorm(300), b = factor(sample(letters[1:3], 300, TRUE)))
#' d$y <- rnbinom(300, mu = exp(0.5 + 0.4 * d$a), size = 2)
#' nb <- fastglm_nb(y ~ a + b, data = d)
#' drop1(nb, test = "LRT")
#' anova(nb)
#' d$z <- ifelse(runif(300) < 0.2, 0, d$y)
#' zi <- fastglm_zi(z ~ a + b | a, data = d)
#' drop1(zi, test = "LRT")
#' @name drop1.fastglm
#' @md
NULL

## ---- What a fit offers -----------------------------------------------------------

## The kind of term-by-term comparison: "glm" (columns by assign, the
## drop1.glm table; GLM and negative binomial fits), "multi" (one row per
## term per part), or "single" (refits by formula, the likelihood ratio
## table).
.fastglm_drop1_kind <- function(object)
{
    if (.fastglm_two_part(object) || inherits(object, "fastglm_ordinal_partial")) return("multi")
    if (inherits(object, c("fastglm_ordinal", "fastglm_vglm", "fastglm_truncated"))) return("single")
    "glm"
}

## The parts of a model: count and zero (positive and zero for two-part
## fits, whose first part models the positive outcomes); location and
## nominal; or main.
.fastglm_drop1_parts <- function(object)
{
    if (inherits(object, "fastglm_twopart")) return(c("positive", "zero"))
    if (.fastglm_two_part(object)) return(c("count", "zero"))
    if (inherits(object, "fastglm_ordinal_partial")) return(c("location", "nominal"))
    "main"
}

## The terms of each part (a list of terms objects), or NULL for a fit to a
## design matrix.
.fastglm_drop1_terms <- function(object)
{
    tt <- object$terms
    if (is.null(tt) && !is.null(object$source)) {
        f <- .fastglm_source_formula(object$source)
        if (inherits(f, "Formula")) {
            nr <- length(f)[2L]
            tt <- list(count = stats::terms(stats::formula(f, lhs = 1L, rhs = 1L)),
                       zero = stats::terms(stats::formula(f, lhs = 1L, rhs = if (nr > 1L) 2L else 1L)))
        } else if (inherits(f, "formula")) tt <- stats::terms(f)
    }
    if (.fastglm_two_part(object)) {
        if (!is.list(tt) || !inherits(tt$count, "terms") || !inherits(tt$zero, "terms")) return(NULL)
        return(stats::setNames(list(tt$count, tt$zero), .fastglm_drop1_parts(object)))
    }
    if (!inherits(tt, "terms")) return(NULL)
    if (inherits(object, "fastglm_ordinal_partial")) {
        np <- object$nonparallel.terms
        nt <- if (length(np)) stats::terms(stats::reformulate(np)) else
            stats::terms(~ 0)
        return(list(location = tt, nominal = nt))
    }
    list(main = tt)
}

## The term assignment of the columns of a GLM or negative binomial fit.
.fastglm_drop1_assign <- function(object)
{
    a <- object$assign
    if (is.null(a) && !is.null(object$x)) a <- attr(object$x, "assign")
    if (is.null(a) && is.null(object$source)) a <- attr(.fastglm_refit_data(object)$X, "assign")
    if (is.null(a) || length(a) != length(object$coefficients))
        stop("the fit did not retain the term assignment of its design columns, which drop1() needs",
             call. = FALSE)
    as.integer(a)
}

## TRUE when a design of the fit is a big.matrix.
.fastglm_drop1_big <- function(object)
{
    x <- object$x
    if (inherits(x, "big.matrix")) return(TRUE)
    is.list(x) && !is.data.frame(x) && any(vapply(x, function(z) inherits(z, "big.matrix"), NA))
}

## The design column names of a part of a fit without a formula.
.fastglm_drop1_colnames <- function(object, part)
{
    if (.fastglm_two_part(object)) return(names(object$coefficients[[.fastglm_part_internal(part)]]))
    if (inherits(object, "fastglm_ordinal")) {
        cn <- object$design.columns
        if (is.null(cn)) cn <- object$state$columns
        if (is.null(cn)) cn <- names(object$beta)
        return(cn)
    }
    if (inherits(object, "fastglm_vglm")) {
        cn <- object$design.names
        if (is.null(cn)) cn <- colnames(object$x)
        return(cn)
    }
    names(object$coefficients)
}

## Which design columns of a part are intercepts (dropped by no default
## scope): named "(Intercept)", or constant and nonzero.
.fastglm_drop1_intercepts <- function(object, part, cn)
{
    is.int <- cn %in% "(Intercept)"
    if (inherits(object, "fastglm_ordinal")) return(is.int)
    x <- if (.fastglm_two_part(object)) {
        if (is.null(object$source)) tryCatch(.count_design(object)$x[[.fastglm_part_internal(part)]],
                                             error = function(e) NULL)
    } else object$x
    if (!is.null(x) && (is.matrix(x) || inherits(x, c("dgCMatrix", "big.matrix"))) && ncol(x) == length(cn)) {
        is.int <- is.int | as.logical(.intercept_columns(x))
    } else if (length(object$intercept_mask) == length(cn)) {
        is.int <- is.int | as.logical(object$intercept_mask)
    }
    is.int
}

## ---- Scope ---------------------------------------------------------------------

## The terms of a part to drop: drop.scope() by default, otherwise term
## labels or the terms of a formula, which must be terms of the part
## (strict = TRUE) or are kept when they are (strict = FALSE).
## A formula is resolved against the part's formula with update.formula(),
## as stats::drop1() does, so `~ . - c` means every term but c.
.fastglm_drop1_scope_terms <- function(tt, scope, strict = TRUE, part = NULL)
{
    tl <- attr(tt, "term.labels")
    if (is.null(scope)) return(stats::drop.scope(tt))
    if (inherits(scope, "formula"))
        scope <- attr(stats::terms(stats::update.formula(stats::formula(tt), scope)), "term.labels")
    if (!is.character(scope)) stop("'scope' must be term labels or a formula", call. = FALSE)
    if (!strict) return(scope[scope %in% tl])
    if (!all(scope %in% tl)) {
        if (is.null(part)) stop("scope is not a subset of term labels", call. = FALSE)
        stop(sprintf("scope terms of part '%s' are not terms of that part: %s", part,
                     paste(setdiff(scope, tl), collapse = ", ")), call. = FALSE)
    }
    scope
}

## Column groups of a part: list(label = column indices).
.fastglm_drop1_groups <- function(object, part, scope)
{
    cn <- .fastglm_drop1_colnames(object, part)
    p <- length(cn)
    if (is.null(scope)) {
        keep <- which(!.fastglm_drop1_intercepts(object, part, cn))
        return(stats::setNames(as.list(keep), if (is.null(cn)) paste0("X", keep) else cn[keep]))
    }
    if (!is.list(scope)) scope <- as.list(scope)
    idx <- lapply(scope, function(g) {
        if (is.character(g)) {
            i <- match(g, cn)
            if (anyNA(i)) stop("design columns not found: ", paste(g[is.na(i)], collapse = ", "), call. = FALSE)
            i
        } else if (is.numeric(g)) {
            i <- as.integer(g)
            if (anyNA(i) || any(i < 1L | i > p) || any(i != g))
                stop("column groups in 'scope' must index design columns", call. = FALSE)
            i
        } else stop("column groups in 'scope' must be column indices or names", call. = FALSE)
    })
    if (any(lengths(idx) == 0L)) stop("a column group in 'scope' is empty", call. = FALSE)
    nm <- names(scope)
    if (is.null(nm)) nm <- rep("", length(idx))
    auto <- vapply(idx, function(i) paste(if (is.null(cn)) paste0("X", i) else cn[i], collapse = " + "), "")
    nm[!nzchar(nm)] <- auto[!nzchar(nm)]
    stats::setNames(lapply(idx, function(i) sort(unique(i))), nm)
}

## The rows of drop1(): a list of rows, each with label, part, term (for a
## formula refit), and columns (for a column refit).
.fastglm_drop1_rows <- function(object, scope, kind)
{
    parts <- .fastglm_drop1_parts(object)
    tts <- .fastglm_drop1_terms(object)
    # Two-part fits: a list scope may name the positive part `count`.
    if (inherits(object, "fastglm_twopart") && is.list(scope) && !is.null(names(scope)))
        names(scope)[names(scope) == "count"] <- "positive"
    # A partial proportional odds fit to a design matrix: column groups of
    # the location design only (a column refit removes its effects).
    if (is.null(tts) && inherits(object, "fastglm_ordinal_partial")) parts <- "main"
    multi <- length(parts) > 1L
    named.parts <- multi && is.list(scope) && !is.null(names(scope)) && all(names(scope) %in% parts)
    if (multi && is.list(scope) && !named.parts)
        stop(sprintf("a list 'scope' for this model must have elements named %s",
                     paste0("'", parts, "'", collapse = " or ")), call. = FALSE)
    if (multi && is.null(tts) && !is.null(scope) && !named.parts)
        stop(sprintf("for this model 'scope' must be a list with elements named %s",
                     paste0("'", parts, "'", collapse = " or ")), call. = FALSE)
    if (named.parts) parts <- intersect(parts, names(scope))
    part.scope <- function(p) if (named.parts) scope[[p]] else scope
    rows <- list()
    partial <- inherits(object, "fastglm_ordinal_partial") && multi
    nominal.terms <- if (partial) attr(tts$nominal, "term.labels") else character()
    # Partial proportional odds fits: a term with cutpoint-specific effects
    # gets "t (all effects)" (the term removed) and "t (nonparallel)" (its
    # effect made common to the cutpoints); other terms keep their label.
    label <- function(p, x) {
        if (partial) return(if (p == "nominal") paste0(x, " (nonparallel)")
                            else if (x %in% nominal.terms) paste0(x, " (all effects)") else x)
        if (multi) paste0(p, ": ", x) else x
    }
    if (is.null(tts)) {
        for (p in parts) {
            g <- .fastglm_drop1_groups(object, p, part.scope(p))
            for (k in seq_along(g))
                rows[[length(rows) + 1L]] <- list(label = label(p, names(g)[k]), part = if (multi) p,
                                                  term = NULL, columns = g[[k]])
        }
        return(rows)
    }
    if (kind == "glm") assign <- .fastglm_drop1_assign(object)
    found <- character()
    for (p in parts) {
        s <- part.scope(p)
        terms <- .fastglm_drop1_scope_terms(tts[[p]], s, strict = !multi || named.parts,
                                            part = if (named.parts) p)
        found <- c(found, terms)
        tl <- attr(tts[[p]], "term.labels")
        for (t in terms) {
            cols <- if (kind == "glm") which(assign %in% match(t, tl))
            rows[[length(rows) + 1L]] <- list(label = label(p, t), part = if (multi) p, term = t, columns = cols)
        }
    }
    if (multi && !named.parts && !is.null(scope)) {
        if (is.character(scope) && length(miss <- setdiff(scope, found)))
            stop("scope terms are not terms of any part: ", paste(miss, collapse = ", "), call. = FALSE)
        if (!length(found)) stop("the scope selects no term of any part", call. = FALSE)
    }
    rows
}

## ---- Tests -------------------------------------------------------------------

## The tests available for a fit, in the order of the help page.
.fastglm_drop1_tests <- function(object, rec, kind)
{
    if (rec$lr_method == "profile") return(c("none", "LRT", "Chisq"))
    c("none", if (.fastglm_drop1_rao(object)) "Rao", "LRT", "Chisq", if (kind == "glm") "F")
}

## TRUE when fastglm_score_test() supports the fit.
.fastglm_drop1_rao <- function(object)
{
    if (inherits(object, "fastglm_ordinal"))
        return(is.null(object$source) && !isTRUE(object$firth) && !inherits(object, "fastglm_ordinal_stream"))
    inherits(object, c("fastglm", "fastglmFit")) && .glm_ml_inference_model(object) && !.fastglm_drop1_big(object)
}

.fastglm_drop1_test <- function(test, object, rec, kind, what = "drop1")
{
    avail <- .fastglm_drop1_tests(object, rec, kind)
    if (!(test %in% avail)) {
        why <- if (test == "Rao") "the score test of fastglm_score_test() does not support this fit" else
            if (test == "F") "the F test applies to generalized linear and negative binomial models fitted by maximum likelihood" else ""
        stop(sprintf("test = \"%s\" is not available for this fit (%s); available tests: %s",
                     test, why, paste0("\"", avail, "\"", collapse = ", ")), call. = FALSE)
    }
    if (test == "LRT") "Chisq" else test
}

## The bias-reduced fits without a likelihood ratio comparison give an error.
.fastglm_drop1_check_firth <- function(rec, what)
{
    if (rec$lr_method == "none")
        stop(sprintf(paste0("%s() is not available for this bias-reduced fit: it has no likelihood ratio test (only ",
                            "binomial-logit Firth fits held in memory with a dense design have one, through ",
                            "fastglm_profile_test()); use fastglm_wald_test()"), what), call. = FALSE)
}

## ---- Refits --------------------------------------------------------------------

## The refit of the model without a row's term or columns.
.fastglm_drop1_refit <- function(object, row, kind, trace = FALSE)
{
    nb <- inherits(object, "fastglm_nb")
    if (kind == "glm")
        return(.fastglm_refit(object, drop_columns = row$columns, columns = TRUE, nb_theta_fixed = nb,
                              trace = trace))
    if (!is.null(row$term))
        return(.fastglm_refit(object, formula = stats::as.formula(paste("~ . -", row$term)), part = row$part,
                              trace = trace))
    .fastglm_refit(object, drop_columns = row$columns, part = row$part, trace = trace)
}

## Runs the refits, fun(i) for i in 1..k, and keeps only what the table
## needs: keep(fit, i) reduces each refit to a few numbers (deviance, rank,
## log likelihood, AIC, a test statistic) before the next refit starts, so at
## most one refit is held in memory at a time (as stats::drop1.glm() keeps
## scalars only). With trace = TRUE each refit is traced and preceded by a
## header line naming it ("drop1 refit 2/6: - b").
.fastglm_drop1_run <- function(k, fun, headers, what, trace, keep)
{
    out <- vector("list", k)
    for (i in seq_len(k)) {
        if (trace) cat(sprintf("%s refit %d/%d: %s\n", what, i, k, headers[i]))
        fit <- fun(i)
        out[[i]] <- keep(fit, i)
        rm(fit)
    }
    out
}

## What the tables keep of a refit.
.fastglm_drop1_keep_glm <- function(fit, i)
    list(deviance = as.numeric(fit$deviance), rank = as.numeric(fit$rank),
         df.residual = as.numeric(fit$df.residual))

.fastglm_drop1_keep_loglik <- function(k = 2)
{
    force(k)
    function(fit, i) {
        rec <- .fastglm_cmp_record(fit)
        list(npar = as.numeric(.fastglm_cmp_npar(rec)), loglik = .fastglm_drop1_loglik(fit, rec),
             aic = as.numeric(stats::extractAIC(fit, k = k)[2L]))
    }
}

## TRUE only for an explicit trace = TRUE (stats::step() passes trace = 1,
## which leaves the refits silent).
.fastglm_drop1_trace <- function(trace) isTRUE(trace)

## Announces the refits of a streamed or big.matrix fit (one line), after
## checking that the source can be read again. The line states what is read:
## with cache = TRUE (a cached formula source refitted by formula), one pass
## over the source to build the per-call block cache, then the refits, each
## making several passes over the cached blocks; otherwise the refits, each
## making several passes over the source (or the fit's own block cache, or
## the big.matrix in blocks of rows).
.fastglm_drop1_announce <- function(object, k, what, cache = FALSE, columns = FALSE)
{
    streamed <- !is.null(object$source)
    big <- .fastglm_drop1_big(object)
    if (!streamed && !big) return(invisible(FALSE))
    if (streamed && isFALSE(object$source$reusable))
        stop(sprintf(paste0("%s() refits the model by reading its source again, but the source was declared ",
                            "one-shot (fastglm_source(reusable = FALSE)); no refit was run"), what), call. = FALSE)
    if (k > 0L) {
        refits <- sprintf("%d refit%s", k, if (k == 1L) "" else "s")
        each <- if (k == 1L) "a streamed fit" else "each a streamed fit"
        msg <- if (cache)
            sprintf(paste0("%s: one pass over the source to cache its blocks for this call (removed when it ends), ",
                           "then %s, %s making several passes over the cached blocks"), what, refits, each)
        else if (!streamed)
            sprintf("%s: %s, %s making several passes over the big.matrix design in blocks of rows", what, refits, each)
        else if (!is.null(object$source$cache) && (columns || is.null(object$source$spec)))
            sprintf("%s: %s, %s making several passes over the block cache of the source", what, refits, each)
        else
            sprintf("%s: %s, %s making several passes over the source", what, refits, each)
        message(msg)
    }
    invisible(TRUE)
}

## ---- Per-call block cache of a cached formula source -------------------------

## TRUE when refits by formula of this fit would read the raw callback of a
## source made with cache = TRUE. A formula refit rebuilds each block from the
## raw data with the new formula, so it cannot use the fit's own cache (which
## holds the blocks built with the original formula).
.fastglm_refit_cache_needed <- function(object)
{
    src <- object$source
    !is.null(src) && !isFALSE(src$reusable) && !is.null(src$spec) && isTRUE(src$spec$cache)
}

## Reads the raw blocks of the fit's source once (calling reset first, as a
## pass does) into files in directory `dir`, and returns the fit with its
## source specification reading those files, so every formula refit of the
## call reads the cache instead of the callback. The caller creates `dir`
## with tempfile() and removes it with on.exit(), which also runs on an error
## or an interrupt.
.fastglm_refit_cache_build <- function(object, dir)
{
    src <- object$source
    sp <- src$spec
    n <- src$n_chunks
    if (!dir.create(dir, showWarnings = FALSE) && !dir.exists(dir))
        stop("cannot create a directory for the block cache of the refits", call. = FALSE)
    files <- file.path(dir, sprintf("block-%07d.rds", seq_len(n)))
    if (!is.null(sp$reset)) sp$reset()
    for (k in seq_len(n)) saveRDS(sp$callback(k), files[k], compress = FALSE)
    sp$callback <- function(k, ...) readRDS(files[k])
    sp$reset <- NULL
    object$source$spec <- sp
    object
}

## ---- drop1() -------------------------------------------------------------------

.fastglm_drop1 <- function(object, scope, scale, test, k, trace, label, dots)
{
    if (length(dots))
        warning("arguments not used by drop1() for this fit: ",
                paste(if (is.null(names(dots))) rep("<unnamed>", length(dots)) else names(dots), collapse = ", "),
                call. = FALSE)
    if (!is.numeric(k) || length(k) != 1L || !is.finite(k) || k < 0)
        stop("'k' must be a nonnegative number", call. = FALSE)
    if (is.null(scale)) scale <- 0
    if (!is.numeric(scale) || length(scale) != 1L || !is.finite(scale) || scale < 0)
        stop("'scale' must be a nonnegative number", call. = FALSE)
    trace <- .fastglm_drop1_trace(trace)
    rec <- .fastglm_cmp_record(object)
    .fastglm_drop1_check_firth(rec, "drop1")
    kind <- .fastglm_drop1_kind(object)
    if (scale > 0 && (kind != "glm" || rec$lr_method == "profile"))
        stop("'scale' applies to generalized linear and negative binomial models fitted by maximum likelihood ",
             "(as in stats::drop1.glm()); use scale = 0 for this fit", call. = FALSE)
    test <- .fastglm_drop1_test(test, object, rec, kind)
    rows <- .fastglm_drop1_rows(object, scope, kind)
    heading <- c("Single term deletions", "\nModel:", .fastglm_cmp_describe(object, label),
                 if (scale > 0) paste("\nscale: ", format(scale), "\n"))
    if (rec$lr_method == "profile") return(.fastglm_drop1_firth(object, rows, test, heading))
    # Refits by formula of a cached formula source read one per-call cache of
    # the raw blocks (removed on exit); column refits (GLM and negative
    # binomial) read the fit's own block cache.
    by.formula <- kind != "glm" && any(vapply(rows, function(r) !is.null(r$term), NA))
    cache <- by.formula && .fastglm_refit_cache_needed(object)
    .fastglm_drop1_announce(object, length(rows), "drop1", cache = cache, columns = !by.formula)
    if (cache) {
        dir <- tempfile("fastglm-refit-cache-")
        on.exit(unlink(dir, recursive = TRUE), add = TRUE)
        object <- .fastglm_refit_cache_build(object, dir)
    }
    labels <- vapply(rows, `[[`, "", "label")
    keep <- if (kind == "glm") .fastglm_drop1_keep_glm else .fastglm_drop1_keep_loglik(k)
    fits <- .fastglm_drop1_run(length(rows), function(i) .fastglm_drop1_refit(object, rows[[i]], kind, trace),
                               paste("-", labels), "drop1", trace, keep)
    if (kind == "glm") return(.fastglm_drop1_glm_table(object, rows, fits, labels, test, heading, scale, k))
    .fastglm_drop1_loglik_table(object, rec, rows, fits, labels, test, heading, k)
}

## The value of extractAIC(object, k = k) of a GLM fit, or of a
## MASS::glm.nb() fit (the coefficients only; theta is not counted).
.fastglm_drop1_extract_aic <- function(object, k = 2)
{
    if (inherits(object, "fastglm_nb")) return(-2 * as.numeric(logLik(object)) + k * object$rank)
    if (inherits(object, "glm")) return(stats::extractAIC(object, k = k)[2L])
    if (is.null(object$aic)) NA_real_ else as.numeric(object$aic) + (k - 2) * object$rank
}

## The line of a negative binomial table saying how theta is treated in the
## refits: held at the full model's estimate, or, when that estimate is
## infinite (the Poisson limit), every refit is a Poisson fit.
.fastglm_nb_theta_note <- function(object, as)
{
    if (!is.finite(object$theta))
        return(paste0("\nTheta is infinite (the Poisson limit, where fastglm_nb() returns the Poisson fit), ",
                      "so every refit is a Poisson fit"))
    paste0("\nTheta is held at the full model's estimate (", format(object$theta, digits = 6),
           ") in every refit, as ", as)
}

## The dispersion that scales deviance differences, as summary.glm(): fixed
## (one for binomial, Poisson, and negative binomial) or the Pearson estimate.
.fastglm_drop1_dispersion <- function(object)
{
    if (inherits(object, "fastglm_nb")) return(1)
    fam <- object$family
    fixed <- .fixed_dispersion(fam)
    if (!is.na(fixed) && !startsWith(fam$family, "quasi")) fixed else .fastglm_cmp_glm_dispersion(object)
}

## The number of rows of the design (including rows with zero weight), as
## nrow(model.matrix()) in drop1.glm().
.fastglm_drop1_nrow <- function(object)
{
    if (!is.null(object$y)) return(NROW(object$y))
    if (!is.null(object$n)) return(as.numeric(object$n))
    as.numeric(stats::nobs(object))
}

## GLM and negative binomial fits: the table of stats::drop1.glm().
.fastglm_drop1_glm_table <- function(object, rows, fits, labels, test, heading, scale = 0, k = 2)
{
    score <- NULL
    if (test == "Rao") {
        score <- vapply(rows, function(r) {
            st <- fastglm_score_test(object, terms = r$columns)
            unname(st$statistic) * .fastglm_drop1_dispersion(object)
        }, numeric(1))
    }
    dev <- c(as.numeric(object$deviance), vapply(fits, `[[`, numeric(1), "deviance"))
    dfs <- c(object$rank, vapply(fits, `[[`, numeric(1), "rank"))
    # As drop1.glm(): scale > 0 replaces the dispersion.
    dispersion <- if (scale > 0) scale else .fastglm_drop1_dispersion(object)
    fam <- object$family$family
    n <- .fastglm_drop1_nrow(object)
    loglik <- if (fam == "gaussian") { if (scale > 0) dev / scale - n else n * log(dev / n) } else dev / dispersion
    aic <- loglik + k * dfs
    dfs <- dfs[1L] - dfs
    dfs[1L] <- NA
    aic <- aic + (.fastglm_drop1_extract_aic(object, k) - aic[1L])
    aod <- data.frame(Df = dfs, Deviance = dev, AIC = aic, row.names = c("<none>", labels), check.names = FALSE)
    if (all(is.na(aic))) aod <- aod[, -3L]
    if (test == "Chisq") {
        stat <- pmax(0, loglik - loglik[1L])
        stat[1L] <- NA
        nas <- !is.na(stat)
        nm <- if (dispersion == 1) "LRT" else "scaled dev."
        aod[, nm] <- stat
        stat[nas] <- stats::pchisq(stat[nas], aod$Df[nas], lower.tail = FALSE)
        aod[, "Pr(>Chi)"] <- stat
    } else if (test == "Rao") {
        stat <- c(NA, pmax(0, score)) / dispersion
        nas <- !is.na(stat)
        aod[, if (dispersion == 1) "Rao score" else "scaled Rao sc."] <- stat
        stat[nas] <- stats::pchisq(stat[nas], aod$Df[nas], lower.tail = FALSE)
        aod[, "Pr(>Chi)"] <- stat
    } else if (test == "F") {
        if (fam %in% c("binomial", "poisson"))
            warning(sprintf("F test assumes 'quasi%s' family", fam), call. = FALSE)
        d <- aod$Deviance
        rdf <- object$df.residual
        rms <- d[1L] / rdf
        d <- pmax(0, d - d[1L])
        Fs <- (d / aod$Df) / rms
        Fs[aod$Df < 1e-4] <- NA
        P <- Fs
        nas <- !is.na(Fs)
        P[nas] <- stats::pf(Fs[nas], aod$Df[nas], rdf, lower.tail = FALSE)
        aod[, c("F value", "Pr(>F)")] <- list(Fs, P)
    }
    if (inherits(object, "fastglm_nb"))
        heading <- c(heading, paste0(.fastglm_nb_theta_note(object, "drop1() of a MASS::glm.nb() fit"), ";",
                                     "\nAIC counts the coefficients only (as MASS's extractAIC()), 2 less than AIC()."))
    structure(aod, heading = heading, class = c("anova", "data.frame"))
}

## The log likelihood used by the likelihood ratio statistic of a fit.
.fastglm_drop1_loglik <- function(fit, rec = .fastglm_cmp_record(fit)) as.numeric(rec$loglik)

## Likelihood ratio statistics 2 (l1 - l0), with rounding below zero set to
## zero and clearly negative values (a refit better than the full model)
## set to NA with a warning.
.fastglm_drop1_lr <- function(l1, l0, labels)
{
    stat <- 2 * (l1 - l0)
    neg <- !is.na(stat) & stat < 0
    bad <- neg & stat < -1e-6 * pmax(1, abs(l1))
    if (any(bad))
        warning(sprintf(paste0("the likelihood ratio statistic of %s is negative (the refit has a larger log ",
                               "likelihood than the full model, so a fit did not converge); no p-value is given"),
                        paste0("'", labels[bad], "'", collapse = ", ")), call. = FALSE)
    stat[neg & !bad] <- 0
    stat[bad] <- NA
    stat
}

## Score tests of ordinal fits for each row: the term's coefficients are
## zero (a location row), or its cutpoint-specific coefficients are equal (a
## nominal row of a partial proportional odds fit).
.fastglm_drop1_ordinal_score <- function(object, row)
{
    if (is.null(row$term)) {
        cn <- .fastglm_drop1_colnames(object, "location")[row$columns]
        if (inherits(object, "fastglm_ordinal_partial") && identical(row$part, "nominal"))
            return(fastglm_score_test(object, contrast = .fastglm_drop1_nominal_contrast(object, cn)))
        return(fastglm_score_test(object, terms = cn))
    }
    if (identical(row$part, "nominal")) {
        a <- object$design.assign
        cn <- object$design.columns[a %in% match(row$term, attr(object$terms, "term.labels"))]
        return(fastglm_score_test(object, contrast = .fastglm_drop1_nominal_contrast(object, cn)))
    }
    fastglm_score_test(object, terms = row$term)
}

## Equality contrasts of the cutpoint-specific coefficients of design
## columns cn of a partial proportional odds fit.
.fastglm_drop1_nominal_contrast <- function(object, cn)
{
    b <- names(coef(object))
    rows <- list()
    for (c in cn) {
        nm <- b[startsWith(b, paste0("nonparallel:", c, ":"))]
        if (length(nm) < 2L) next
        for (j in seq_len(length(nm) - 1L)) {
            r <- stats::setNames(numeric(length(b)), b)
            r[nm[j]] <- 1; r[nm[j + 1L]] <- -1
            rows[[paste0(nm[j], " - ", nm[j + 1L])]] <- r
        }
    }
    if (!length(rows)) stop("the term has no cutpoint-specific coefficients", call. = FALSE)
    do.call(rbind, rows)
}

## Other models: Df, AIC, and the likelihood ratio (or score) statistic.
.fastglm_drop1_loglik_table <- function(object, rec, rows, fits, labels, test, heading, k = 2)
{
    npar <- c(.fastglm_cmp_npar(rec), vapply(fits, `[[`, numeric(1), "npar"))
    aic <- c(stats::extractAIC(object, k = k)[2L], vapply(fits, `[[`, numeric(1), "aic"))
    dfs <- npar[1L] - npar
    dfs[1L] <- NA
    aod <- data.frame(Df = dfs, AIC = aic, row.names = c("<none>", labels), check.names = FALSE)
    if (test == "Chisq") {
        l1 <- .fastglm_drop1_loglik(object, rec)
        l0 <- vapply(fits, `[[`, numeric(1), "loglik")
        stat <- c(NA, .fastglm_drop1_lr(l1, l0, labels))
        p <- stat
        nas <- !is.na(stat)
        p[nas] <- stats::pchisq(stat[nas], dfs[nas], lower.tail = FALSE)
        aod[, c("LRT", "Pr(>Chi)")] <- list(stat, p)
    } else if (test == "Rao") {
        st <- lapply(rows, function(r) .fastglm_drop1_ordinal_score(object, r))
        stat <- c(NA, vapply(st, function(s) unname(s$statistic), numeric(1)))
        p <- c(NA, vapply(st, function(s) s$p.value, numeric(1)))
        aod[, c("Rao score", "Pr(>Chi)")] <- list(stat, p)
    }
    if (inherits(object, "fastglm_ordinal_partial") && any(grepl(" \\((all effects|nonparallel)\\)$", labels)))
        heading <- c(heading, paste0("\nRows 'all effects' remove the term; rows 'nonparallel' test that its effect is ",
                                     "the same at every cutpoint\n(the proportional odds restriction)."))
    structure(aod, heading = heading, class = c("anova", "data.frame"))
}

## Bias-reduced fits: the penalized likelihood ratio test of each row's
## coefficients on the full model (no refit).
.fastglm_drop1_firth <- function(object, rows, test, heading)
{
    labels <- vapply(rows, `[[`, "", "label")
    tl <- if (!is.null(object$terms)) attr(object$terms, "term.labels")
    assign <- NULL
    cols <- lapply(rows, function(r) {
        if (!is.null(r$columns)) return(r$columns)
        if (is.null(assign)) assign <<- .fastglm_drop1_assign(object)
        which(assign %in% match(r$term, tl))
    })
    res <- if (test == "Chisq") lapply(cols, function(cc) fastglm_profile_test(object, parm = cc))
    dfs <- c(NA, lengths(cols))
    aod <- data.frame(Df = dfs, row.names = c("<none>", labels), check.names = FALSE)
    if (test == "Chisq") {
        aod[, "Penalized LRT"] <- c(NA, vapply(res, function(s) unname(s$statistic), numeric(1)))
        aod[, "Pr(>Chi)"] <- c(NA, vapply(res, function(s) s$p.value, numeric(1)))
    }
    heading <- c(heading, paste0("\nPenalized likelihood ratio tests (Firth): each term's coefficients are set to ",
                                 "zero under the full model's\npenalty (fastglm_profile_test())."))
    structure(aod, heading = heading, class = c("anova", "data.frame"))
}

## ---- Methods -----------------------------------------------------------------

## The arguments of stats::drop1.glm(); test is matched as stats does.
.fastglm_drop1_entry <- function(object, scope, scale = 0, test = c("none", "Rao", "LRT", "Chisq", "F"),
                                 k = 2, trace = FALSE, ..., .label = "object")
{
    test <- match.arg(test)
    .fastglm_drop1(object, if (missing(scope)) NULL else scope, scale, test, k, trace, .label, list(...))
}

.fastglm_drop1_method <- function(object, scope, scale = 0, test = c("none", "Rao", "LRT", "Chisq", "F"),
                                  k = 2, trace = FALSE, ...)
{
    test <- match.arg(test)
    .fastglm_drop1(object, if (missing(scope)) NULL else scope, scale, test, k, trace,
                   deparse1(substitute(object)), list(...))
}

#' @rdname drop1.fastglm
#' @exportS3Method stats::drop1 fastglm
drop1.fastglm <- .fastglm_drop1_method
#' @rdname drop1.fastglm
#' @exportS3Method stats::drop1 fastglm_stream
drop1.fastglm_stream <- .fastglm_drop1_method
#' @rdname drop1.fastglm
#' @exportS3Method stats::drop1 fastglm_nb
drop1.fastglm_nb <- .fastglm_drop1_method
#' @rdname drop1.fastglm
#' @exportS3Method stats::drop1 fastglm_zi
drop1.fastglm_zi <- .fastglm_drop1_method
#' @rdname drop1.fastglm
#' @exportS3Method stats::drop1 fastglm_hurdle
drop1.fastglm_hurdle <- .fastglm_drop1_method
#' @rdname drop1.fastglm
#' @exportS3Method stats::drop1 fastglm_truncated
drop1.fastglm_truncated <- .fastglm_drop1_method
#' @rdname drop1.fastglm
#' @exportS3Method stats::drop1 fastglm_ordinal
drop1.fastglm_ordinal <- .fastglm_drop1_method
#' @rdname drop1.fastglm
#' @exportS3Method stats::drop1 fastglm_vglm
drop1.fastglm_vglm <- .fastglm_drop1_method

## ---- extractAIC() ----------------------------------------------------------------

## c(edf, AIC with penalty k), as stats::extractAIC(), so that stats::step()
## and MASS::stepAIC() work. GLM fits use the number of estimated
## coefficients (as extractAIC.glm()); negative binomial fits do not count
## theta (as MASS's extractAIC() of a glm.nb() fit); other fits count every
## parameter of logLik() (as MASS's extractAIC() of a polr() fit).
.fastglm_extract_aic <- function(fit, scale = 0, k = 2, ...)
{
    if (.fastglm_two_part(fit) && .fastglm_called_by_step(sys.function(sys.parent())))
        warning(paste0("step() does not select terms of models with two linear predictors (zero-inflated, ",
                       "hurdle, and two-part); the fit is returned unchanged. Use drop1() for tests of the ",
                       "terms of each part"), call. = FALSE)
    if (inherits(fit, c("fastglm_nb"))) {
        edf <- fit$rank
        return(c(edf, -2 * as.numeric(logLik(fit)) + k * edf))
    }
    if (inherits(fit, "fastglm") && .fastglm_drop1_kind(fit) == "glm") {
        edf <- fit$rank
        return(c(edf, as.numeric(fit$aic) + (k - 2) * edf))
    }
    ll <- logLik(fit)
    edf <- attr(ll, "df")
    c(edf, -2 * as.numeric(ll) + k * edf)
}

## TRUE when fun is stats::step() or MASS::stepAIC(): the function that
## called the extractAIC() generic. Both call extractAIC() directly from
## their own frame; drop1(), anova(), and AIC() never come through here with
## either of them as the caller.
.fastglm_called_by_step <- function(fun)
{
    if (!is.function(fun)) return(FALSE)
    if (identical(fun, stats::step)) return(TRUE)
    isNamespaceLoaded("MASS") && identical(fun, getExportedValue("MASS", "stepAIC"))
}

#' @rdname drop1.fastglm
#' @param fit a fitted model, for `extractAIC()`.
#' @exportS3Method stats::extractAIC fastglm
extractAIC.fastglm <- .fastglm_extract_aic
#' @rdname drop1.fastglm
#' @exportS3Method stats::extractAIC fastglm_nb
extractAIC.fastglm_nb <- .fastglm_extract_aic
#' @rdname drop1.fastglm
#' @exportS3Method stats::extractAIC fastglm_zi
extractAIC.fastglm_zi <- .fastglm_extract_aic
#' @rdname drop1.fastglm
#' @exportS3Method stats::extractAIC fastglm_hurdle
extractAIC.fastglm_hurdle <- .fastglm_extract_aic
#' @rdname drop1.fastglm
#' @exportS3Method stats::extractAIC fastglm_truncated
extractAIC.fastglm_truncated <- .fastglm_extract_aic
#' @rdname drop1.fastglm
#' @exportS3Method stats::extractAIC fastglm_ordinal
extractAIC.fastglm_ordinal <- .fastglm_extract_aic
#' @rdname drop1.fastglm
#' @exportS3Method stats::extractAIC fastglm_vglm
extractAIC.fastglm_vglm <- .fastglm_extract_aic

## ---- Sequential anova() of one model ---------------------------------------------

## The test of a sequential table, matched as stats::anova.glm() (through
## stat.anova()) does: NULL (the default of the class), FALSE (no test), or
## one of "Rao", "LRT", "Chisq", "F", "Cp".
.fastglm_anova_seq_test <- function(test)
{
    if (is.null(test) || isFALSE(test)) return(test)
    test <- match.arg(test, c("Rao", "LRT", "Chisq", "F", "Cp"))
    if (test == "LRT") "Chisq" else test
}

.fastglm_anova_sequential <- function(object, test = NULL, dispersion = NULL, label = "object", trace = FALSE)
{
    test <- .fastglm_anova_seq_test(test)
    trace <- .fastglm_drop1_trace(trace)
    if (.fastglm_drop1_kind(object) == "multi")
        stop(paste0("anova() of one model (a sequential table) is not available for models with more than one ",
                    "linear predictor, since the order in which terms enter has no clear meaning across parts; ",
                    "use drop1() for tests of single terms in each part, or anova(m0, m1) for nested fits"),
             call. = FALSE)
    tts <- .fastglm_drop1_terms(object)
    if (is.null(tts))
        stop(paste0("anova() of one model gives the sequential table of the terms of a formula, and this fit ",
                    "has a design matrix without terms; use drop1() with a scope of column groups, or ",
                    "anova(m0, m1) for nested fits"), call. = FALSE)
    rec <- .fastglm_cmp_record(object)
    .fastglm_drop1_check_firth(rec, "anova")
    glm.kind <- .fastglm_drop1_kind(object) == "glm" && rec$lr_method != "profile"
    if (is.character(test) && test %in% c("Rao", "Cp", if (!glm.kind) "F"))
        stop(sprintf(paste0("test = \"%s\" is not available in the sequential table of this fit%s; available tests: ",
                            "\"LRT\", \"Chisq\"%s, or FALSE"), test,
                     if (test == "Rao") " (it is for formula fastglm() fits, through stats::anova.glm())" else "",
                     if (glm.kind) ", \"F\"" else ""), call. = FALSE)
    tt <- tts$main
    tl <- attr(tt, "term.labels")
    if (rec$lr_method == "profile") return(.fastglm_anova_seq_firth(object, tl, test, label, trace))
    if (glm.kind) return(.fastglm_anova_seq_glm(object, tt, test, dispersion, trace))
    if (!is.null(dispersion)) warning("'dispersion' applies only to generalized linear models and is ignored",
                                      call. = FALSE)
    .fastglm_anova_seq_loglik(object, rec, tl, label, test, trace)
}

## The response label of a terms object.
.fastglm_anova_response <- function(tt)
{
    v <- as.character(attr(tt, "variables"))[-1L]
    if (attr(tt, "response") > 0L) v[attr(tt, "response")] else ""
}

## GLM and negative binomial fits: the table of stats::anova.glm(), with
## refits by design columns (theta held fixed for negative binomial fits).
.fastglm_anova_seq_glm <- function(object, tt, test, dispersion, trace = FALSE)
{
    assign <- .fastglm_drop1_assign(object)
    nvars <- max(0L, assign)
    tl <- attr(tt, "term.labels")
    nb <- inherits(object, "fastglm_nb")
    k <- max(nvars - 1L, 0L)
    .fastglm_drop1_announce(object, k, "anova", columns = TRUE)
    fits <- .fastglm_drop1_run(k, function(i)
        .fastglm_refit(object, drop_columns = which(assign > i), columns = TRUE, nb_theta_fixed = nb, trace = trace),
        paste("terms up to", tl[seq_len(k)]), "anova", trace, .fastglm_drop1_keep_glm)
    resdev <- vapply(fits, `[[`, numeric(1), "deviance")
    resdf <- vapply(fits, `[[`, numeric(1), "df.residual")
    resdf <- c(as.numeric(object$df.null), resdf, as.numeric(object$df.residual))
    resdev <- c(as.numeric(object$null.deviance), resdev, as.numeric(object$deviance))
    table <- data.frame(c(NA, -diff(resdf)), c(NA, pmax(0, -diff(resdev))), resdf, resdev)
    if (!length(tl)) table <- table[1L, , drop = FALSE]
    dimnames(table) <- list(c("NULL", tl), c("Df", "Deviance", "Resid. Df", "Resid. Dev"))
    fam <- object$family
    title <- paste0("Analysis of Deviance Table", "\n\nModel: ", fam$family, ", link: ", fam$link,
                    "\n\nResponse: ", .fastglm_anova_response(tt),
                    "\n\nTerms added sequentially (first to last)\n\n")
    if (nb) title <- paste0(title, sub("^\n", "", .fastglm_nb_theta_note(object, "anova() of one MASS::glm.nb() fit")),
                            ".\n")
    if (isFALSE(test)) return(structure(table, heading = title, class = c("anova", "data.frame")))
    moment <- !nb && (is.na(.fixed_dispersion(fam)) || startsWith(fam$family, "quasi"))
    if (is.null(test)) test <- if (!is.null(dispersion)) "Chisq" else if (moment) "F" else "Chisq"
    if (is.null(dispersion)) dispersion <- .fastglm_drop1_dispersion(object)
    df.dispersion <- if (moment) object$df.residual else Inf
    if (test == "F" && df.dispersion == 0) {
        return(structure(table, heading = title, class = c("anova", "data.frame")))
    }
    if (test == "F" && df.dispersion == Inf) {
        if (fam$family %in% c("binomial", "poisson"))
            warning(sprintf("using F test with a '%s' family is inappropriate", fam$family), call. = FALSE)
        else warning("using F test with a fixed dispersion is inappropriate", call. = FALSE)
    }
    dfs <- table[, "Df"]
    if (test == "Chisq") {
        vals <- table[, "Deviance"] / dispersion * sign(dfs)
        vals[dfs %in% 0] <- NA
        vals[!is.na(vals) & vals < 0] <- NA
        table <- cbind(table, `Pr(>Chi)` = stats::pchisq(vals, abs(dfs), lower.tail = FALSE))
    } else {
        f <- (table[, "Deviance"] / dfs) / dispersion
        f[dfs %in% 0] <- NA
        f[!is.na(f) & f < 0] <- NA
        table <- cbind(table, F = f, `Pr(>F)` = stats::pf(f, abs(dfs), df.dispersion, lower.tail = FALSE))
    }
    structure(table, heading = title, class = c("anova", "data.frame"))
}

## Truncated, ordinal, and multinomial fits: the likelihood ratio table of
## the nested sequence (refits by formula, starting from the model without
## terms). test = FALSE leaves out the statistic and p-value.
.fastglm_anova_seq_loglik <- function(object, rec, tl, label = "object", test = NULL, trace = FALSE)
{
    nv <- length(tl)
    cache <- nv > 0L && .fastglm_refit_cache_needed(object)
    .fastglm_drop1_announce(object, nv, "anova", cache = cache)
    if (cache) {
        dir <- tempfile("fastglm-refit-cache-")
        on.exit(unlink(dir, recursive = TRUE), add = TRUE)
        object <- .fastglm_refit_cache_build(object, dir)
    }
    fits <- .fastglm_drop1_run(nv, function(i)
        .fastglm_refit(object, formula = stats::as.formula(paste("~ . -", paste(tl[i:nv], collapse = " - "))),
                       trace = trace),
        c("NULL", if (nv > 1L) paste("terms up to", tl[seq_len(nv - 1L)]))[seq_len(nv)], "anova", trace,
        .fastglm_drop1_keep_loglik())
    ll <- c(vapply(fits, `[[`, numeric(1), "loglik"), .fastglm_drop1_loglik(object, rec))
    npar <- c(vapply(fits, `[[`, numeric(1), "npar"), .fastglm_cmp_npar(rec))
    dfs <- c(NA, diff(npar))
    stat <- c(NA, .fastglm_drop1_lr(ll[-1L], ll[-length(ll)], tl))
    p <- stat
    nas <- !is.na(stat)
    p[nas] <- stats::pchisq(stat[nas], dfs[nas], lower.tail = FALSE)
    table <- data.frame(npar, ll, dfs, stat, p)
    dimnames(table) <- list(c("NULL", tl), c("#Df", "LogLik", "Df", "Chisq", "Pr(>Chisq)"))
    if (isFALSE(test)) table <- table[, 1:3]
    heading <- c("Likelihood ratio tests, terms added sequentially (first to last)\n",
                 paste0("Model: ", .fastglm_cmp_describe(object, label), " (", .fastglm_cmp_kind(rec), ")"))
    structure(table, heading = heading, class = c("anova", "data.frame"))
}

## Bias-reduced formula fits: each model of the sequence (refitted by design
## columns) is tested against the one before it by the penalized likelihood
## ratio test of fastglm_profile_test() on the larger model.
.fastglm_anova_seq_firth <- function(object, tl, test, label = "object", trace = FALSE)
{
    assign <- .fastglm_drop1_assign(object)
    nvars <- max(0L, assign)
    # Model i (the terms up to i) tests term i: the statistic is computed
    # from each refit as it is made, and the refit is then discarded.
    profile <- function(fit, i) {
        cols <- which(assign[assign <= i] == i)
        if (!length(cols) || isFALSE(test)) return(NA_real_)
        unname(fastglm_profile_test(fit, parm = cols)$statistic)
    }
    stat <- unlist(.fastglm_drop1_run(max(nvars - 1L, 0L), function(i)
        .fastglm_refit(object, drop_columns = which(assign > i), columns = TRUE, trace = trace),
        paste("terms up to", tl[seq_len(max(nvars - 1L, 0L))]), "anova", trace, profile))
    if (nvars > 0L) stat <- c(stat, profile(object, nvars))
    stat <- as.numeric(stat)
    dfs <- vapply(seq_len(nvars), function(i) as.numeric(sum(assign == i)), numeric(1))
    p <- stats::pchisq(stat, dfs, lower.tail = FALSE)
    table <- data.frame(c(NA, dfs), c(NA, stat), c(NA, p))
    dimnames(table) <- list(c("NULL", tl), c("Df", "Penalized LRT", "Pr(>Chi)"))
    if (isFALSE(test)) table <- table[, 1L, drop = FALSE]
    heading <- c("Penalized likelihood ratio tests (Firth), terms added sequentially (first to last)\n",
                 paste0("Model: ", .fastglm_cmp_describe(object, label)),
                 paste0("\nEach term is tested in the model of the terms up to it, with its coefficients set to zero ",
                        "under that\nmodel's penalty (fastglm_profile_test())."))
    structure(table, heading = heading, class = c("anova", "data.frame"))
}
