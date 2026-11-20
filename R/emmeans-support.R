#' Estimated marginal means with emmeans
#'
#' Methods for the `recover_data()` and `emm_basis()` generics of the
#' 'emmeans' package, so that [emmeans::emmeans()], [emmeans::ref_grid()],
#' and the functions built on them accept fastglm fits. They are registered
#' when 'emmeans' is loaded; 'emmeans' is not required otherwise.
#'
#' @details
#' Formula fits from [fastglm()] are `glm` objects, and 'emmeans' handles
#' them through its own `glm` support. The methods here cover the other
#' classes, with the modes (passed as `mode =` to [emmeans::emmeans()]) of
#' the established support for the corresponding reference packages:
#'
#' * Negative binomial fits ([fastglm_nb()]): as for `MASS::glm.nb()`, the
#'   linear predictor with the log link, so `type = "response"` gives means.
#' * Zero-inflated fits ([fastglm_zi()]): as for `pscl::zeroinfl()`,
#'   `"response"` (the mean, the default), `"count"` (the mean of the count
#'   component), `"zero"` (the probability of a structural zero), and
#'   `"prob0"` (the probability of a zero). With `lin.pred = TRUE`, the
#'   `"count"` and `"zero"` modes are on the scale of the component's linear
#'   predictor.
#' * Hurdle fits ([fastglm_hurdle()]): as for `pscl::hurdle()`, `"response"`
#'   (the mean, the default), `"count"` (the mean of the untruncated count
#'   distribution), `"zero"` (the ratio of the probability of a positive
#'   value to the probability that the untruncated count distribution is
#'   positive), and `"prob0"` (the probability of a zero); `lin.pred = TRUE`
#'   as for zero-inflated fits.
#' * Zero-truncated fits ([fastglm_truncated()]): `"response"` (the mean of
#'   the truncated distribution, the default), `"count"` (the mean of the
#'   untruncated distribution), and `"link"` (the log of the untruncated
#'   mean, so `type = "response"` returns the untruncated mean).
#' * Two-part fits ([fastglm_twopart()]): `"response"` (the mean, the
#'   default), `"zero"` (the probability of a zero), and `"positive"` (the
#'   mean given a positive value).
#'
#' As in 'emmeans', a mode estimates the same quantity as the [predict()]
#' type of the same name, averaged over the reference grid; for example,
#' `mode = "zero"` corresponds to `predict(type = "zero")`.
#' * Cumulative-link fits ([fastglm_ordinal()]): as for `MASS::polr()`,
#'   `"latent"` (the default, with argument `rescale`), `"linear.predictor"`,
#'   `"cum.prob"`, `"exc.prob"`, `"prob"`, and `"mean.class"`.
#' * Cumulative-link fits with threshold-specific slopes (`nonparallel`):
#'   `"prob"` (the default), `"cum.prob"`, `"exc.prob"`, and
#'   `"mean.class"`. The latent variable and a single linear predictor are not
#'   defined when slopes differ across thresholds.
#' * Multinomial fits ([fastglm_multinom()]): as for `nnet::multinom()`,
#'   `"prob"` (the default) and `"latent"` (centered log ratios), whatever
#'   the reference category.
#'
#' Standard errors use the delta method with the covariance from [vcov()]
#' (or the `vcov.` argument of 'emmeans'). For cumulative-link fits, and for
#' the binary component of zero-inflated, hurdle, and two-part fits with a
#' link other than the logit, [vcov()] is the inverse expected information,
#' while `MASS::polr()`, `ordinal::clm()`, and 'pscl' use the observed
#' information; the standard errors then differ from theirs by up to a few
#' percent. Another covariance can be given to [emmeans::emmeans()] as
#' `vcov.`, a matrix (in the order of `coef(object)`) or a function of the
#' fit, for example a robust covariance from [fastglm_vcov()] where it is
#' available. As in the reference packages, the
#' dispersion parameter of negative binomial components is treated as known.
#' The degrees of freedom follow the reference packages: infinite except for
#' hurdle fits (the residual degrees of freedom) and multinomial fits (the
#' number of coefficients, the convention of `nnet::multinom()`).
#'
#' Offsets follow the convention of 'emmeans' for `glm` fits. An offset in
#' the formula is evaluated at the grid values of its variables (their means
#' unless set with `at`), and an offset given through the `offset` argument
#' of the fitting function enters at its mean. The `offset` argument of
#' [emmeans::emmeans()] replaces both, for every mode; on the response scale
#' it enters the linear predictor of the count (or positive) component, not
#' the estimate itself. The 'emmeans' support for `pscl` fits instead
#' ignores an `offset =` argument of the fit in its response-scale modes, so
#' `emmeans(fit, ..., offset = 0)` reproduces those values.
#'
#' The reference grid is built from the model frame stored with the fit when
#' the formula has no function calls (as 'emmeans' does for `glm` fits), so
#' changing the data frame after fitting does not change the grid. Otherwise
#' the data are found again from the call; if they no longer agree with the
#' data the model was fitted to (a different number of rows, other factor
#' levels, or other values), an error asks for the original data through the
#' `data` argument of [emmeans::emmeans()].
#'
#' Streamed fits, fits to a `big.matrix`, and fits from the matrix interface
#' have no data in memory to build a reference grid from, and give an error;
#' [fastglm_marginal_effects()] and [fastglm_average()] compute averaged
#' predictions and marginal effects for them.
#'
#' Two choices differ from the support 'emmeans' 1.11 provides for
#' `pscl` fits, to give correct results. The `"prob0"` mode of zero-inflated
#' fits and the `"zero"` mode of hurdle fits include every factor of the
#' delta-method derivative (the 'emmeans' versions omit the derivative of the
#' mean with respect to the linear predictor, which changes the standard
#' errors but not the estimates). The `"zero"` mode of hurdle fits builds the
#' grid from all predictors, since it involves both components. The hurdle
#' `"response"` mode uses the derivative of the fitted binary link, not of
#' the logit, for links other than the logit.
#' @param object a fitted model.
#' @param data an optional data frame from which to build the reference grid,
#'   as for [emmeans::ref_grid()].
#' @param trms,xlev,grid model terms, factor levels, and reference grid,
#'   supplied by 'emmeans'.
#' @param mode the quantity to estimate; see Details.
#' @param lin.pred for zero-inflated and hurdle fits, `TRUE` gives the
#'   `"count"` and `"zero"` modes on the scale of the linear predictor.
#' @param rescale for the `"latent"` mode of cumulative-link fits, the
#'   location and scale of the latent variable, as for `MASS::polr()`.
#' @param vcov. an optional covariance matrix or a function of the fit that
#'   returns one.
#' @param ... further arguments passed by 'emmeans'.
#' @returns `recover_data()` methods return the data for the reference grid,
#'   or a character string explaining why it cannot be built; `emm_basis()`
#'   methods return the list described in the 'emmeans' vignette "For
#'   developers: extending emmeans".
#' @examples
#' if (requireNamespace("emmeans", quietly = TRUE)) {
#'     set.seed(1)
#'     d <- data.frame(g = factor(rep(c("a", "b", "c"), 40)), x = rnorm(120))
#'     d$y <- rpois(120, exp(0.2 + 0.4 * (d$g == "b") + 0.3 * d$x))
#'     fit <- fastglm_nb(y ~ g + x, data = d)
#'     emmeans::emmeans(fit, "g", type = "response")
#' }
#' @name fastglm-emmeans
#' @rdname fastglm-emmeans
NULL

## ---- shared helpers -------------------------------------------------------

# A message (not an error) when the fit has no in-memory data to build a
# reference grid from; ref_grid() turns a character result of recover_data()
# into an error with exactly this text.
.emm_unavailable <- function(object) {
    big <- function(z) inherits(z, "big.matrix") ||
        (is.list(z) && !is.data.frame(z) && any(vapply(z, inherits, logical(1), "big.matrix")))
    kind <- if (any(grepl("_stream$", class(object))) || !is.null(object$source) ||
                is.function(object$fetch)) "a streamed fit"
            else if (big(object$x)) "a fit to a big.matrix"
            else if (is.null(object$terms) || (is.null(object$model) && is.null(object$call)))
                "a fit from the matrix interface, without a formula"
            else NULL
    if (is.null(kind)) return(NULL)
    paste0("emmeans needs a formula fit whose data are in memory, and this '", class(object)[1L],
           "' object is ", kind, ". Use fastglm_marginal_effects() or fastglm_average() ",
           "for averaged predictions and marginal effects from this fit.")
}

# The fitting call, with a formula given as `x` renamed to `formula`.
.emm_call <- function(object) {
    fcall <- object$call
    nm <- names(fcall)
    if (!is.null(nm) && !("formula" %in% nm) && "x" %in% nm) names(fcall)[nm == "x"] <- "formula"
    fcall
}

.emm_var_names <- function(trms)
    vapply(as.list(attr(trms, "variables"))[-1L], function(e) paste(deparse(e, width.cutoff = 500L), collapse = ""),
           character(1))

# Check data found again from the call against the data of the fit. Returns
# NULL when they agree and a message otherwise.
.emm_check_recovered <- function(object, tbl, trms, frame) {
    fail <- function(why)
        paste0("The data found from the call of this '", class(object)[1L], "' fit differ from the data ",
               "it was fitted to (", why, "); they were probably changed after fitting. ",
               "Supply the original data with emmeans(..., data = ) or refit the model.")
    n.fit <- if (!is.null(frame)) nrow(frame) else NULL
    if (!is.null(n.fit) && nrow(tbl) != n.fit)
        return(fail(sprintf("%d rows instead of %d", nrow(tbl), n.fit)))
    mf <- tryCatch(stats::model.frame(trms, tbl, na.action = stats::na.pass), error = function(e) e)
    if (inherits(mf, "error")) return(fail(conditionMessage(mf)))
    if (!is.null(frame)) {
        for (nm in intersect(names(mf), names(frame))) {
            a <- mf[[nm]]; b <- frame[[nm]]
            same <- if (is.factor(a) || is.factor(b) || is.character(a) || is.character(b)) {
                identical(as.character(a), as.character(b)) &&
                    identical(levels(droplevels(as.factor(a))), levels(droplevels(as.factor(b))))
            } else isTRUE(all.equal(unclass(as.matrix(a)), unclass(as.matrix(b)),
                                    check.attributes = FALSE, tolerance = 1e-12))
            if (!same) return(fail(sprintf("variable '%s' has changed", nm)))
        }
        return(NULL)
    }
    x <- object$x
    if (is.matrix(x) && !is.null(colnames(x)) && nrow(x) == nrow(tbl)) {
        lev <- .emm_levels(object)
        lev <- lev[intersect(names(lev), .emm_var_names(trms))]
        mm <- tryCatch(stats::model.matrix(trms, stats::model.frame(trms, tbl, na.action = stats::na.pass, xlev = lev),
                                           contrasts.arg = .emm_contrasts(object$contrasts, lev)),
                       error = function(e) e)
        if (inherits(mm, "error")) return(fail(conditionMessage(mm)))
        cn <- intersect(colnames(mm), colnames(x))
        if (!isTRUE(all.equal(unname(mm[, cn, drop = FALSE]), unname(x[, cn, drop = FALSE]), tolerance = 1e-12)))
            return(fail("the design matrix has changed"))
    }
    NULL
}

# Factor levels of the fit, under the names used in its model frame.
.emm_levels <- function(object) {
    lev <- object$xlevels
    if (is.null(lev)) lev <- object$levels
    if (is.list(lev)) lev else list()
}

.emm_contrasts <- function(contrasts, keep) {
    if (is.null(contrasts) || !length(contrasts)) return(NULL)
    contrasts[intersect(names(contrasts), names(keep))]
}

# recover_data() for every class: the stored model frame when the terms have
# no function calls (as emmeans does for glm), otherwise the data found again
# from the call, checked against the fit.
.emm_recover <- function(object, trms, data = NULL, pwts = NULL, offset.trick = FALSE, ...) {
    msg <- .emm_unavailable(object)
    if (!is.null(msg)) return(msg)
    if (offset.trick) {
        # As emmeans does for pscl fits: variables inside offset() become
        # predictors of the grid.
        env <- new.env(parent = environment(trms))
        env$offset <- function(x) x
        environment(trms) <- env
    }
    fcall <- .emm_call(object)
    frame <- object$model
    if (!is.null(pwts) && !is.null(frame) && length(pwts) != nrow(frame)) pwts <- NULL
    tbl <- if (is.null(frame))
        emmeans::recover_data(fcall, trms, object$na.action, data = data, pwts = pwts, ...)
    else emmeans::recover_data(fcall, trms, object$na.action, data = data, frame = frame, pwts = pwts, ...)
    if (is.null(data) && is.data.frame(tbl)) {
        msg <- .emm_check_recovered(object, tbl, trms, frame)
        if (!is.null(msg)) return(msg)
    }
    tbl
}

# The model matrix of the grid, with the factor levels and contrasts of the fit.
.emm_design <- function(trms, grid, xlev, object, contrasts) {
    vars <- .emm_var_names(trms)
    lev <- xlev
    fit.lev <- .emm_levels(object)
    if (length(fit.lev)) lev[names(fit.lev)] <- fit.lev
    lev <- lev[intersect(names(lev), vars)]
    m <- stats::model.frame(trms, grid, na.action = stats::na.pass, xlev = lev)
    stats::model.matrix(trms, m, contrasts.arg = .emm_contrasts(contrasts, lev))
}

.emm_offset <- function(trms, grid) {
    off <- rep(0, nrow(grid))
    vars <- attr(trms, "variables")
    for (i in attr(trms, "offset")) off <- off + eval(vars[[i + 1L]], grid)
    off
}

# The value of the `offset` argument of the calling emmeans::ref_grid() or
# emmeans::emmeans(), or NULL when neither was given one. emm_basis() methods
# do not receive it, but 'emmeans' adds it to every estimate, so bases on the
# response scale need it.
.emm_user_offset <- function() {
    for (i in rev(seq_len(sys.nframe()))) {
        fn <- names(formals(sys.function(i)))
        if (all(c("cov.reduce", "offset", "nuisance", "rg.limit") %in% fn) ||
            all(c("specs", "fac.reduce", "offset", "tran") %in% fn)) {
            fr <- sys.frame(i)
            if (!eval(quote(missing(offset)), fr)) return(as.numeric(get("offset", envir = fr))[1L])
        }
    }
    NULL
}

# The offset of a count (or positive) linear predictor at the grid rows. As
# for glm fits: formula offsets at the grid values plus the mean of an
# `offset =` argument of the fit, both replaced by the `offset` argument of
# emmeans when one is given.
.emm_count_offset <- function(trms, grid) {
    user <- .emm_user_offset()
    if (!is.null(user)) return(rep(user, nrow(grid)))
    off <- .emm_offset(trms, grid)
    if (!is.null(grid[[".static.offset."]])) off <- off + grid[[".static.offset."]]
    off
}

# A basis whose estimates are on the response scale (identity X). ref_grid()
# adds a user-supplied `offset` to such estimates; subtracting it here leaves
# the estimates as computed, with the offset already in the linear predictor.
.emm_response_basis <- function(est, delta, V, misc, df = Inf) {
    user <- .emm_user_offset()
    bhat <- if (!is.null(user) && user != 0) est - user else est
    misc$offset.mult <- 0
    .emm_basis_list(diag(1, length(est)), bhat, delta %*% tcrossprod(V, delta), misc, df)
}

.emm_vcov <- function(object, vcov., ...) {
    V <- if (is.null(vcov.)) stats::vcov(object, ...) else if (is.function(vcov.)) vcov.(object, ...) else vcov.
    V <- as.matrix(V)
    if (nrow(V) != ncol(V)) stop("vcov. must be a function or a square matrix", call. = FALSE)
    cf <- stats::coef(object)
    if (is.null(dimnames(V)) && nrow(V) == length(cf)) dimnames(V) <- list(names(cf), names(cf))
    V
}

.emm_basis_list <- function(X, bhat, V, misc, df = Inf, nbasis = estimability::all.estble) {
    list(X = X, bhat = as.numeric(bhat), nbasis = nbasis, V = V,
         dffun = function(k, dfargs) dfargs$df, dfargs = list(df = df), misc = misc)
}

# Probability of zero for a count distribution and its derivative in the mean.
.emm_p0 <- function(mu, dist, theta = NULL) {
    switch(dist,
           poisson = { p <- exp(-mu); rbind(p0 = p, dmu = -p) },
           negbin = { p <- (theta / (mu + theta))^theta; rbind(p0 = p, dmu = -p * theta / (mu + theta)) },
           geometric = rbind(p0 = 1 / (1 + mu), dmu = -1 / (1 + mu)^2),
           binomial = rbind(p0 = 1 - mu, dmu = rep(-1, length(mu))),
           stop("unsupported count distribution '", dist, "'", call. = FALSE))
}

.emm_theta <- function(object) if (identical(object$dist, "negbin")) object$theta else NULL

.emm_stat_vcov <- function(object, vcov., ...) {
    if (is.null(vcov.)) stats::vcov(object) else .emm_vcov(object, vcov.)
}

## ---- GLM-type fits: fastglm (formula, in memory) and fastglm_nb -------------

.emm_glm_basis <- function(object, trms, xlev, grid, vcov. = NULL, ...) {
    bhat <- stats::coef(object)
    X <- .emm_design(trms, grid, xlev, object, object$contrasts)
    X <- X[, names(bhat), drop = FALSE]
    V <- .emm_stat_vcov(object, vcov.)
    nbasis <- estimability::all.estble
    if (anyNA(bhat)) nbasis <- estimability::nonest.basis(stats::model.matrix(object))
    fam <- object$family
    misc <- list()
    if (is.list(fam) && !identical(fam$link, "identity")) {
        misc$tran <- fam$link
        misc$inv.lbl <- if (grepl("binomial", fam$family) && !grepl("Negative", fam$family)) "prob"
                        else if (grepl("poisson", fam$family)) "rate" else "response"
        if (!grepl("gaussian", fam$family)) misc$sigma <- NA
    }
    df <- if (is.list(fam) && fam$family %in% c("gaussian", "Gamma")) object$df.residual else Inf
    .emm_basis_list(X, bhat, V, misc, df, nbasis)
}

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::recover_data
recover_data.fastglm <- function(object, data = NULL, ...) {
    if (is.null(object$model) && is.null(.emm_unavailable(object)))
        return(paste0("emmeans needs a formula fit whose data are in memory, and this '", class(object)[1L],
                      "' object has no stored model frame. Use fastglm_marginal_effects() or fastglm_average() ",
                      "for averaged predictions and marginal effects from this fit."))
    .emm_recover(object, stats::delete.response(object$terms), data = data,
                 pwts = object$prior.weights, ...)
}

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::emm_basis
emm_basis.fastglm <- function(object, trms, xlev, grid, vcov. = NULL, ...)
    .emm_glm_basis(object, trms, xlev, grid, vcov. = vcov.)

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::recover_data
recover_data.fastglm_nb <- function(object, data = NULL, ...)
    .emm_recover(object, stats::delete.response(object$terms), data = data,
                 pwts = object$prior.weights, ...)

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::emm_basis
emm_basis.fastglm_nb <- function(object, trms, xlev, grid, vcov. = NULL, ...)
    .emm_glm_basis(object, trms, xlev, grid, vcov. = vcov.)

## ---- zero-truncated fits ------------------------------------------------------

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::recover_data
recover_data.fastglm_truncated <- function(object, data = NULL, ...)
    .emm_recover(object, stats::delete.response(object$terms), data = data,
                 pwts = object$prior.weights, ...)

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::emm_basis
emm_basis.fastglm_truncated <- function(object, trms, xlev, grid, mode = c("response", "count", "link"),
                                        vcov. = NULL, ...) {
    mode <- match.arg(mode)
    bhat <- stats::coef(object)
    X <- .emm_design(trms, grid, xlev, object, object$contrasts)[, names(bhat), drop = FALSE]
    V <- .emm_stat_vcov(object, vcov.)
    if (mode == "link")
        return(.emm_basis_list(X, bhat, V, list(tran = "log", inv.lbl = "count")))
    eta <- as.numeric(X %*% bhat) + .emm_count_offset(trms, grid)
    mu <- exp(eta)
    if (mode == "count") {
        est <- mu; deta <- mu
    } else {
        p <- .emm_p0(mu, object$dist, .emm_theta(object))
        q <- 1 - p["p0", ]
        est <- .count_moments(mu, object$dist, .emm_theta(object))$mean
        # d/d eta of mu / (1 - p0(mu)) = mu (q + mu p0'(mu)) / q^2.
        deta <- mu * (q + mu * p["dmu", ]) / q^2
    }
    .emm_response_basis(est, X * deta, V, list(estName = mode))
}

## ---- zero-inflated, hurdle, and two-part fits -------------------------------

# match.arg() for recover_data(), whose errors ref_grid() would replace by a
# generic message: a result of length other than 1 carries the message.
.emm_mode <- function(mode) {
    choices <- eval(formals(sys.function(sys.parent()))$mode)
    if (identical(mode, choices)) return(choices[1L])
    if (length(mode) == 1L && mode %in% choices) return(mode)
    structure(character(0), message = paste0("'mode' should be one of ",
                                              paste0('"', choices, '"', collapse = ", ")))
}

.emm_part_terms <- function(object, part) stats::delete.response(object$terms[[part]])

.emm_full_terms <- function(object) stats::delete.response(stats::terms(object$formula))

.emm_part <- function(object, part, trms, xlev, grid) {
    tt <- .emm_part_terms(object, part)
    X <- .emm_design(tt, grid, xlev, object, object$contrasts[[part]])
    b <- stats::coef(object, model = part)
    off <- if (part == "count") .emm_count_offset(tt, grid) else .emm_offset(tt, grid)
    list(X = X[, names(b), drop = FALSE], b = b, eta = as.numeric(X[, names(b), drop = FALSE] %*% b) + off)
}

.emm_count_vcov <- function(object, model, vcov.) {
    if (is.null(vcov.)) return(stats::vcov(object, model = model))
    if (is.function(vcov.)) return(vcov.(object, model = model))
    if (!is.matrix(vcov.)) stop("vcov. must be a function or a square matrix", call. = FALSE)
    # A full covariance supplied for a single-component mode: take its block.
    full <- names(stats::coef(object))
    if (model != "full" && nrow(vcov.) == length(full)) {
        p <- length(stats::coef(object, model = "count"))
        i <- if (model == "count") seq_len(p) else p + seq_along(stats::coef(object, model = "zero"))
        vcov. <- vcov.[i, i, drop = FALSE]
    }
    vcov.
}

# A single component on the response scale, or its linear predictor.
.emm_one_part <- function(object, part, trms, xlev, grid, vcov., lin.pred, tran, inv.lbl, df, p0 = FALSE) {
    pp <- .emm_part(object, part, trms, xlev, grid)
    V <- .emm_count_vcov(object, part, vcov.)
    if (lin.pred) return(.emm_basis_list(pp$X, pp$b, V, list(tran = tran, inv.lbl = inv.lbl), df))
    lnk <- stats::make.link(tran)
    est <- lnk$linkinv(pp$eta)
    mult <- lnk$mu.eta(pp$eta)
    if (p0) {
        # Probability of a zero from the binary component: 1 - pi.
        tmp <- .emm_p0(est, "binomial")
        est <- tmp["p0", ]; mult <- mult * tmp["dmu", ]
    }
    delta <- pp$X * mult
    .emm_response_basis(est, delta, V, list(), df)
}

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::recover_data
recover_data.fastglm_zi <- function(object, data = NULL, mode = c("response", "count", "zero", "prob0"), ...) {
    mode <- .emm_mode(mode)
    if (length(mode) != 1L) return(attr(mode, "message"))
    if (!is.null(msg <- .emm_unavailable(object))) return(msg)
    trms <- if (mode %in% c("count", "zero")) .emm_part_terms(object, mode) else .emm_full_terms(object)
    .emm_recover(object, trms, data = data, pwts = object$weights, offset.trick = TRUE, ...)
}

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::emm_basis
emm_basis.fastglm_zi <- function(object, trms, xlev, grid, mode = c("response", "count", "zero", "prob0"),
                                 lin.pred = FALSE, vcov. = NULL, ...) {
    mode <- match.arg(mode)
    if (mode == "count")
        return(.emm_one_part(object, "count", trms, xlev, grid, vcov., lin.pred, "log", "count", Inf))
    if (mode == "zero")
        return(.emm_one_part(object, "zero", trms, xlev, grid, vcov., lin.pred, object$link, "prob", Inf))
    c1 <- .emm_part(object, "count", trms, xlev, grid)
    z <- .emm_part(object, "zero", trms, xlev, grid)
    mu1 <- exp(c1$eta)
    lnk <- stats::make.link(object$link)
    mu2 <- lnk$linkinv(z$eta); mu2prime <- lnk$mu.eta(z$eta)
    if (mode == "response") {
        est <- (1 - mu2) * mu1
        delta <- cbind(c1$X * (mu1 * (1 - mu2)), z$X * (-mu1 * mu2prime))
    } else {
        tmp <- .emm_p0(mu1, object$dist, .emm_theta(object))
        p0 <- tmp["p0", ]
        est <- (1 - mu2) * p0 + mu2
        # The count derivative includes d mu1 / d eta = mu1.
        delta <- cbind(c1$X * ((1 - mu2) * tmp["dmu", ] * mu1), z$X * (mu2prime * (1 - p0)))
    }
    V <- .emm_count_vcov(object, "full", vcov.)
    .emm_response_basis(est, delta, V, list())
}

# Whether a hurdle mode uses one component only (and so its terms).
.emm_hurdle_single <- function(mode, lin.pred)
    (lin.pred && mode %in% c("count", "zero")) || (!lin.pred && mode %in% c("count", "prob0"))

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::recover_data
recover_data.fastglm_hurdle <- function(object, data = NULL, mode = c("response", "count", "zero", "prob0"),
                                        lin.pred = FALSE, ...) {
    mode <- .emm_mode(mode)
    if (length(mode) != 1L) return(attr(mode, "message"))
    if (!is.null(msg <- .emm_unavailable(object))) return(msg)
    trms <- if (.emm_hurdle_single(mode, lin.pred)) .emm_part_terms(object, if (mode == "count") "count" else "zero")
            else .emm_full_terms(object)
    .emm_recover(object, trms, data = data, pwts = object$weights, offset.trick = TRUE, ...)
}

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::emm_basis
emm_basis.fastglm_hurdle <- function(object, trms, xlev, grid, mode = c("response", "count", "zero", "prob0"),
                                     lin.pred = FALSE, vcov. = NULL, ...) {
    mode <- match.arg(mode)
    df <- object$df.residual
    if (.emm_hurdle_single(mode, lin.pred)) {
        if (mode == "count")
            return(.emm_one_part(object, "count", trms, xlev, grid, vcov., lin.pred, "log", "count", df))
        return(.emm_one_part(object, "zero", trms, xlev, grid, vcov., lin.pred, object$link, "prob", df,
                             p0 = (mode == "prob0")))
    }
    c1 <- .emm_part(object, "count", trms, xlev, grid)
    z <- .emm_part(object, "zero", trms, xlev, grid)
    mu1 <- exp(c1$eta)
    lnk <- stats::make.link(object$link)
    mu2 <- lnk$linkinv(z$eta); mu2prime <- lnk$mu.eta(z$eta)
    tmp <- .emm_p0(mu1, object$dist, .emm_theta(object))
    p1 <- 1 - tmp["p0", ]; dp1 <- -tmp["dmu", ]
    if (mode == "response") {
        # mean = mu2 * mu1 / p1
        est <- mu2 * mu1 / p1
        dmu1 <- mu2 * (p1 - mu1 * dp1) / p1^2
        delta <- cbind(c1$X * (dmu1 * mu1), z$X * ((mu1 / p1) * mu2prime))
    } else {
        # ratio = mu2 / p1
        est <- mu2 / p1
        delta <- cbind(c1$X * (-mu2 * dp1 * mu1 / p1^2), z$X * (mu2prime / p1))
    }
    V <- .emm_count_vcov(object, "full", vcov.)
    .emm_response_basis(est, delta, V, list(estName = mode), df)
}

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::recover_data
recover_data.fastglm_twopart <- function(object, data = NULL, mode = c("response", "zero", "positive"), ...) {
    mode <- .emm_mode(mode)
    if (length(mode) != 1L) return(attr(mode, "message"))
    if (!is.null(msg <- .emm_unavailable(object))) return(msg)
    trms <- switch(mode, response = .emm_full_terms(object), zero = .emm_part_terms(object, "zero"),
                   positive = .emm_part_terms(object, "count"))
    .emm_recover(object, trms, data = data, pwts = object$weights, offset.trick = TRUE, ...)
}

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::emm_basis
emm_basis.fastglm_twopart <- function(object, trms, xlev, grid, mode = c("response", "zero", "positive"),
                                      vcov. = NULL, ...) {
    mode <- match.arg(mode)
    if (mode == "positive") {
        out <- .emm_one_part(object, "count", trms, xlev, grid, vcov., FALSE, "log", "response", Inf)
    } else if (mode == "zero") {
        # P(Y = 0) = 1 - pi, the quantity of predict(type = "zero").
        out <- .emm_one_part(object, "zero", trms, xlev, grid, vcov., FALSE, object$link, "prob", Inf, p0 = TRUE)
    } else {
        c1 <- .emm_part(object, "count", trms, xlev, grid)
        z <- .emm_part(object, "zero", trms, xlev, grid)
        mu1 <- exp(c1$eta)
        lnk <- stats::make.link(object$link)
        p <- lnk$linkinv(z$eta)
        est <- p * mu1
        delta <- cbind(c1$X * (p * mu1), z$X * (mu1 * lnk$mu.eta(z$eta)))
        V <- .emm_count_vcov(object, "full", vcov.)
        out <- .emm_response_basis(est, delta, V, list())
    }
    out$misc$estName <- mode
    out
}

## ---- cumulative-link fits -----------------------------------------------------

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::recover_data
recover_data.fastglm_ordinal <- function(object, data = NULL, ...) {
    if (!is.null(msg <- .emm_unavailable(object))) return(msg)
    w <- object$weights
    .emm_recover(object, stats::delete.response(object$terms), data = data, pwts = w, ...)
}

# The cumulative linear predictors alpha[j] - x beta - z gamma[, j] for every
# cut j (the slowest-varying index) and grid row, with columns in the order
# of coef(object).
.emm_ordinal_X <- function(object, X) {
    cf <- stats::coef(object)
    cuts <- names(object$alpha)
    k <- length(cuts)
    n <- nrow(X)
    out <- matrix(0, n * k, length(cf), dimnames = list(NULL, names(cf)))
    gam <- object$gamma
    for (j in seq_len(k)) {
        rows <- (j - 1L) * n + seq_len(n)
        out[rows, paste0("threshold:", cuts[j])] <- 1
        for (b in names(object$beta)) out[rows, b] <- -X[, b]
        if (!is.null(gam))
            for (r in rownames(gam)) out[rows, paste0("nonparallel:", r, ":", cuts[j])] <- -X[, r]
    }
    out
}

.emm_ordinal_basis <- function(object, trms, xlev, grid, mode, rescale, vcov.) {
    if (!all(paste0("threshold:", names(object$alpha)) %in% names(stats::coef(object))))
        stop("emmeans support needs one coefficient per threshold; this fit has tied or collapsed thresholds",
             call. = FALSE)
    X <- .emm_design(trms, grid, xlev, object, object$contrasts)
    xint <- match("(Intercept)", colnames(X), nomatch = 0L)
    if (xint > 0L) X <- X[, -xint, drop = FALSE]
    V <- if (is.null(vcov.)) stats::vcov(object) else .emm_vcov(object, vcov.)
    k <- length(object$alpha)
    resp <- paste(deparse(object$terms[[2L]], width.cutoff = 500L), collapse = "")
    if (mode == "latent") {
        b <- names(object$beta)
        idx <- c(b, paste0("threshold:", names(object$alpha)))
        Xl <- rescale[2L] * cbind(X[, b, drop = FALSE], matrix(-1 / k, nrow(X), k))
        bhat <- c(object$beta, object$alpha - rescale[1L] / rescale[2L])
        misc <- list(offset.mult = rescale[2L], respName = resp)
        return(.emm_basis_list(Xl, bhat, V[idx, idx, drop = FALSE], misc))
    }
    misc <- list(ylevs = list(cut = names(object$alpha)), tran = object$link, inv.lbl = "cumprob",
                 offset.mult = -1)
    if (mode != "linear.predictor") {
        misc$mode <- mode
        misc$postGridHook <- ".clm.postGrid"
    }
    misc$respName <- resp
    .emm_basis_list(.emm_ordinal_X(object, X), stats::coef(object), V[names(stats::coef(object)),
                    names(stats::coef(object)), drop = FALSE], misc)
}

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::emm_basis
emm_basis.fastglm_ordinal <- function(object, trms, xlev, grid,
                                      mode = c("latent", "linear.predictor", "cum.prob", "exc.prob", "prob",
                                               "mean.class"),
                                      rescale = c(0, 1), vcov. = NULL, ...) {
    mode <- match.arg(mode)
    .emm_ordinal_basis(object, trms, xlev, grid, mode, rescale, vcov.)
}

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::emm_basis
emm_basis.fastglm_ordinal_partial <- function(object, trms, xlev, grid,
                                              mode = c("prob", "cum.prob", "exc.prob", "mean.class"),
                                              vcov. = NULL, ...) {
    mode <- match.arg(mode)
    .emm_ordinal_basis(object, trms, xlev, grid, mode, c(0, 1), vcov.)
}

## ---- multinomial fits ---------------------------------------------------------

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::recover_data
recover_data.fastglm_vglm <- function(object, data = NULL, ...) {
    if (!is.null(msg <- .emm_unavailable(object))) return(msg)
    if (!identical(object$family, "multinomial"))
        return(paste0("emmeans support covers multinomial fits; this fit has family '", object$family, "'."))
    .emm_recover(object, stats::delete.response(object$terms), data = data, pwts = object$prior.weights, ...)
}

#' @rdname fastglm-emmeans
#' @exportS3Method emmeans::emm_basis
emm_basis.fastglm_vglm <- function(object, trms, xlev, grid, mode = c("prob", "latent"), vcov. = NULL, ...) {
    mode <- match.arg(mode)
    lev <- object$levels
    K <- length(lev)
    others <- setdiff(lev, object$reference)
    k <- length(others)
    bhat <- stats::coef(object)
    if (anyNA(bhat)) stop("emmeans support needs a multinomial fit without aliased coefficients", call. = FALSE)
    X <- .emm_design(trms, grid, xlev, object, object$contrasts)
    # Coefficients are term-major: name "<category>:<column>".
    cls <- sub(":.*$", "", names(bhat))
    col <- substring(names(bhat), nchar(cls) + 2L)
    if (!all(cls %in% others) || !all(col %in% colnames(X)))
        stop("unexpected multinomial coefficient names", call. = FALSE)
    # Centered log ratios: psi_l = eta_l - mean(eta), with eta = 0 for the
    # reference category; rows of each category block in the order of levels.
    n <- nrow(X)
    out <- matrix(0, n * K, length(bhat), dimnames = list(NULL, names(bhat)))
    for (l in seq_len(K)) {
        rows <- (l - 1L) * n + seq_len(n)
        w <- (as.numeric(cls == lev[l]) * K - 1) / K
        out[rows, ] <- X[, col, drop = FALSE] * rep(w, each = n)
    }
    V <- if (is.null(vcov.)) stats::vcov(object) else .emm_vcov(object, vcov.)
    ylevs <- list(lev)
    names(ylevs) <- paste(deparse(object$terms[[2L]], width.cutoff = 500L), collapse = "")
    misc <- list(tran = "clr", ylevs = ylevs)
    if (mode == "prob") misc$postGridHook <- ".multinom.postGrid"
    nbasis <- kronecker(rep(1, k), estimability::all.estble)
    .emm_basis_list(out, bhat, V[names(bhat), names(bhat), drop = FALSE], misc, df = length(bhat), nbasis = nbasis)
}
