# broom augment() methods for in-memory fits. Streamed and big.matrix fits do
# not retain their rows, so they are refused with a pointer to predict().

.augment_refuse_streamed <- function(x, newdata = NULL) {
    if (!is.null(newdata)) return(invisible(x))
    streamed <- inherits(x, "fastglm_stream") || inherits(x$x, "big.matrix") ||
        (!is.null(x$source) && (is.function(x$fetch) || inherits(x, "fastglm_vglm")))
    if (streamed)
        stop("augment() needs the rows of the data, which a streamed or big.matrix fit does not retain; ",
             "supply 'newdata' (augmenting new data never reads the source) or use predict() and residuals()", call. = FALSE)
    invisible(x)
}

.augment_check_args <- function(data, newdata, se_fit) {
    if (!is.null(data) && !is.null(newdata)) stop("supply either 'data' or 'newdata', not both", call. = FALSE)
    if (!is.logical(se_fit) || length(se_fit) != 1L || is.na(se_fit))
        stop("'se_fit' must be TRUE or FALSE", call. = FALSE)
}

# Rows to which the prediction columns are attached: supplied data, new data,
# or the data the model was fitted to. A formula fit supplies its model frame.
# A matrix-interface fit supplies its non-constant design columns and the
# response, unless the design is sparse, a big.matrix, or larger than
# 1e7 cells: the design is never densified, and the frame then holds only the
# response (or a row index when none is stored).
.augment_frame <- function(x, data, newdata, n_pred) {
    as_frame <- function(z, what) tryCatch(as.data.frame(z), error = function(e)
        stop("'", what, "' must be a data frame or matrix for augment(); use predict() for other inputs", call. = FALSE))
    omit <- x$na.action
    if (!is.null(newdata)) frame <- as_frame(newdata, "newdata")
    else if (!is.null(data)) frame <- as_frame(data, "data")
    else {
        frame <- NULL
        if (!is.null(x$terms)) frame <- tryCatch(as.data.frame(stats::model.frame(x)), error = function(e) NULL)
        if (is.null(frame)) {
            design <- x$x
            if (is.null(design)) design <- tryCatch(stats::model.matrix(x), error = function(e) NULL)
            if (is.list(design) && !is.data.frame(design) && !is.null(design$count)) design <- design$count
            if (is.null(design))
                stop("the fit does not retain its data; supply 'data' or 'newdata'", call. = FALSE)
            n <- nrow(design)
            y <- x$y
            if (!is.null(y) && is.null(dim(y)) && length(y) == n) y <- unname(y) else y <- NULL
            columns <- !(inherits(design, "sparseMatrix") || inherits(design, "big.matrix") ||
                         as.numeric(n) * ncol(design) > 1e7)
            if (columns) {
                design <- as.matrix(design)
                if (is.null(colnames(design))) colnames(design) <- paste0("x", seq_len(ncol(design)))
                const <- apply(design, 2L, function(v) all(v == v[1L]))
                frame <- as.data.frame(design[, !const, drop = FALSE])
                if (!is.null(y)) {
                    frame <- cbind(data.frame(y = y), frame)
                    names(frame) <- make.unique(names(frame))
                }
            } else frame <- if (!is.null(y)) data.frame(y = y) else data.frame(row = seq_len(n))
        }
    }
    # Row names are kept only when they are not the automatic ones (as in broom).
    rn <- if (.row_names_info(frame) > 0L) rownames(frame) else NULL
    if (identical(rn, as.character(seq_len(nrow(frame))))) rn <- NULL
    rownames(frame) <- NULL
    if (is.null(newdata) && is.null(data)) {
        # na.exclude: predictions and residuals are padded, so restore the omitted rows.
        if (inherits(omit, "exclude") && nrow(frame) == n_pred - length(omit)) {
            keep <- napredict(omit, seq_len(nrow(frame)))
            frame <- frame[keep, , drop = FALSE]
            if (!is.null(rn)) {
                full <- rep(NA_character_, n_pred); full[!is.na(keep)] <- rn
                full[is.na(keep)] <- if (!is.null(names(omit))) names(omit) else as.character(as.integer(omit))
                rn <- full
            }
        }
    }
    if (nrow(frame) != n_pred) {
        if (is.null(newdata) && !is.null(data) && inherits(omit, c("omit", "exclude")) &&
            nrow(frame) == n_pred + length(omit)) {
            frame <- frame[-as.integer(omit), , drop = FALSE]
            if (!is.null(rn)) rn <- rn[-as.integer(omit)]
        } else stop("the rows of the data (", nrow(frame), ") do not match the ", n_pred,
                    " fitted rows; supply the data the model was fitted to", call. = FALSE)
    }
    rownames(frame) <- NULL
    attr(frame, "augment.rownames") <- rn
    frame
}

# Tibble output with .rownames first, as broom's augment methods return.
.augment_finish <- function(frame) {
    rn <- attr(frame, "augment.rownames"); attr(frame, "augment.rownames") <- NULL
    if (!is.null(rn)) frame <- cbind(data.frame(.rownames = rn), frame)
    if (requireNamespace("tibble", quietly = TRUE)) tibble::as_tibble(frame) else frame
}

.augment_reserved <- function(frame, new) {
    if (any(names(frame) %in% new))
        stop("the data already contain augmentation columns: ",
             paste(intersect(names(frame), new), collapse = ", "), call. = FALSE)
}

.augment_col <- function(v) {
    if (is.null(dim(v))) unname(v) else { v <- unname(v); I(v) }
}

# Leverage, leave-one-out sigma, Cook's distance, and standardized residuals
# of a GLM, as stats::influence(), rstandard(), and cooks.distance() give for
# glm fits: the leave-one-out sigma is that of lm.influence() applied to the
# deviance residuals, with n counting the rows of nonzero weight.
.augment_influence <- function(x, type) {
    h <- unname(hatvalues(x)); disp <- x$dispersion
    dev <- unname(stats::residuals(x, type = "deviance"))
    pear <- unname(stats::residuals(x, type = "pearson"))
    res <- if (type == "deviance") dev else pear
    w <- if (!is.null(x$na.action) && inherits(x$na.action, "exclude"))
        napredict(x$na.action, x$prior.weights) else x$prior.weights
    live <- !is.na(dev) & !is.na(h) & w != 0
    n <- sum(live); p <- x$rank
    ss <- sum(dev[live]^2)
    sigma <- sqrt(pmax(0, (ss - dev^2 / (1 - h)) / (n - p - 1)))
    std <- suppressWarnings(res / sqrt(disp * (1 - h)))
    cd <- (pear / (1 - h))^2 * h / (disp * p)
    # A leverage of one gives undefined statistics, as in stats::rstandard.glm.
    std[is.infinite(std)] <- NaN; cd[is.infinite(cd)] <- NaN
    sigma[!is.finite(sigma)] <- NaN
    # Rows of zero weight are not part of the influence computation.
    std[!live] <- NA_real_; cd[!live] <- NA_real_; sigma[!live] <- NA_real_
    list(hat = h, sigma = sigma, cooksd = cd, std.resid = std)
}

#' Augment a fastglm fit with predictions and residuals
#'
#' Methods for the broom generic `augment()`. The result is a tibble (a data
#' frame when tibble is unavailable) with the data columns first, followed by
#' `.fitted` (a prediction of the requested type), `.se.fit` when
#' `se_fit = TRUE`, and, for the training rows, `.resid`. A `.rownames` column
#' comes first when the data have row names other than the automatic ones, as
#' in broom. Formula fits default to the model frame.
#'
#' A matrix-interface fit has no data frame. Its default rows are the
#' non-constant design columns and the response `y`, except that a sparse
#' (`Matrix`) design, a `big.matrix` design, or any design with more than
#' 1e7 cells (rows times columns) is never converted to a dense data frame: the
#' default rows then hold only the response `y`, or a row index `row` when no
#' response is stored. Supply `data =` to get data columns in that case.
#'
#' Streamed and `big.matrix` fits do not retain their rows, so `augment()`
#' needs `newdata` for them (which never reads the source); without it they
#' stop with a message directing the user to [predict()] and [residuals()].
#'
#' @section Generalized linear models:
#' Applies to `fastglm()` and `fastglm_nb()` fits. The defaults follow
#' `broom::augment.glm()`: `type.predict = "link"` and
#' `type.residuals = "deviance"`. For training rows the columns `.hat`
#' (leverage, from [hatvalues()]), `.sigma` (leave-one-out residual standard
#' deviation, computed as `lm.influence()` does from the deviance residuals),
#' `.cooksd` (Cook's distance), and `.std.resid` (standardized residual of
#' type `type.residuals`, `deviance` or `pearson`) follow `.resid` in broom's
#' order. Rows with zero weight give `NA`, and a leverage of one gives `NaN`,
#' as in `rstandard.glm()`. The influence columns are omitted for Firth fits,
#' whose leverage differs from the maximum-likelihood one, and for residual
#' types other than `deviance` and `pearson`. Formula fits are served by this
#' method instead of `broom::augment.glm()`, which needs `influence()` and
#' fails for them.
#'
#' @section Count models:
#' Applies to `fastglm_hurdle()`, `fastglm_zi()`, `fastglm_twopart()` and
#' `fastglm_truncated()` fits. The defaults are `type.predict = "response"`
#' and `type.residuals = "pearson"`, the `pscl` default. A matrix-valued
#' prediction (`"link"` or `"prob"`) is stored as a matrix column.
#'
#' @section Multinomial models:
#' Applies to `fastglm_multinom()` and other `fastglm_vglm` fits.
#' `type.predict = "probability"` adds one `.prob_<level>` column per outcome
#' level and `type.predict = "link"` adds `.link_<level>` columns for the
#' non-reference levels. `.class` is the predicted level in either case.
#' `.se.fit_<level>` and, for training rows, `.resid_<level>` columns hold the
#' standard errors and residuals (`type.residuals = "response"` by default).
#' `augment()` for ordinal fits ([fastglm_ordinal()]) keeps its matrix column
#' `.fitted` of category probabilities and its ordered factor `.class`, and
#' also supplies the same `.prob_<level>` columns, so both shapes are available.
#'
#' @param x a fastglm model.
#' @param data optional data to accompany the training predictions, in the
#'   row order of the fit (with `na.omit`, the omitted rows may be included).
#'   Defaults to the data the model retains.
#' @param newdata optional new data to predict; residuals are then not
#'   computed. Cannot be combined with `data`.
#' @param type.predict the prediction `type` passed to [predict()]. For
#'   generalized linear models `"link"`, `"response"`, or `"terms"`.
#' @param type.residuals the residual `type` passed to [residuals()].
#' @param se_fit include prediction standard errors in `.se.fit`.
#' @param ... additional arguments passed to [predict()] (for example `at` for
#'   probability masses of count models).
#' @returns A tibble with the data columns and the added columns.
#' @examples
#' if (requireNamespace("broom", quietly = TRUE)) {
#'   fit <- fastglm(am ~ wt, data = mtcars, family = binomial())
#'   head(broom::augment(fit, type.predict = "response", se_fit = TRUE))
#' }
#' @exportS3Method broom::augment fastglm
#' @name fastglm-augment
augment.fastglm <- function(x, data = NULL, newdata = NULL,
                            type.predict = c("link", "response", "terms"),
                            type.residuals = c("deviance", "pearson", "working", "response"),
                            se_fit = FALSE, ...) {
    type.predict <- match.arg(type.predict); type.residuals <- match.arg(type.residuals)
    .augment_refuse_streamed(x, newdata); .augment_check_args(data, newdata, se_fit)
    pred <- stats::predict(x, newdata = newdata, type = type.predict, se.fit = se_fit, ...)
    fit <- if (se_fit) pred$fit else pred
    n_pred <- NROW(fit)
    frame <- .augment_frame(x, data, newdata, n_pred)
    reserved <- c(".fitted", ".se.fit", ".resid", ".hat", ".sigma", ".std.resid", ".cooksd", ".rownames")
    .augment_reserved(frame, reserved)
    frame$.fitted <- .augment_col(fit)
    if (se_fit) frame$.se.fit <- .augment_col(pred$se.fit)
    if (is.null(newdata)) {
        res <- unname(stats::residuals(x, type = type.residuals))
        frame$.resid <- res
        if (type.residuals %in% c("deviance", "pearson") && !inherits(x, "fastglm_firth") &&
            !isTRUE(x$firth)) {
            infl <- tryCatch(.augment_influence(x, type.residuals), error = function(e) NULL)
            if (!is.null(infl) && length(infl$hat) == n_pred) {
                miss <- if (inherits(x$na.action, "exclude")) as.integer(x$na.action) else integer()
                for (nm in names(infl)) infl[[nm]][miss] <- NA_real_
                frame$.hat <- infl$hat; frame$.sigma <- infl$sigma
                frame$.cooksd <- infl$cooksd; frame$.std.resid <- infl$std.resid
            }
        }
    }
    .augment_finish(frame)
}

#' @exportS3Method broom::augment fastglmFit
#' @rdname fastglm-augment
augment.fastglmFit <- augment.fastglm

.augment_count_refuse <- function(x, newdata = NULL) {
    if (is.null(newdata) && !is.null(x$source) && is.function(x$fetch))
        stop("augment() needs the rows of the data, which a streamed fit does not retain; ",
             "supply 'newdata' or use predict() instead", call. = FALSE)
}

#' @exportS3Method broom::augment fastglm_hurdle
#' @rdname fastglm-augment
augment.fastglm_hurdle <- function(x, data = NULL, newdata = NULL,
                                   type.predict = "response",
                                   type.residuals = c("pearson", "response"),
                                   se_fit = FALSE, ...) {
    type.residuals <- match.arg(type.residuals)
    .augment_count_refuse(x, newdata); .augment_check_args(data, newdata, se_fit)
    pred <- stats::predict(x, newdata = newdata, type = type.predict, se.fit = se_fit, ...)
    fit <- if (se_fit) pred$fit else pred
    frame <- .augment_frame(x, data, newdata, NROW(fit))
    .augment_reserved(frame, c(".fitted", ".se.fit", ".resid"))
    frame$.fitted <- .augment_col(fit)
    if (se_fit) frame$.se.fit <- .augment_col(pred$se.fit)
    if (is.null(newdata)) frame$.resid <- unname(stats::residuals(x, type = type.residuals))
    .augment_finish(frame)
}

#' @exportS3Method broom::augment fastglm_zi
#' @rdname fastglm-augment
augment.fastglm_zi <- augment.fastglm_hurdle

#' @exportS3Method broom::augment fastglm_truncated
#' @rdname fastglm-augment
augment.fastglm_truncated <- augment.fastglm_hurdle

#' @exportS3Method broom::augment fastglm_vglm
#' @rdname fastglm-augment
augment.fastglm_vglm <- function(x, data = NULL, newdata = NULL,
                                 type.predict = c("probability", "link"),
                                 type.residuals = c("response", "pearson"),
                                 se_fit = FALSE, ...) {
    type.predict <- match.arg(type.predict); type.residuals <- match.arg(type.residuals)
    .augment_refuse_streamed(x, newdata); .augment_check_args(data, newdata, se_fit)
    pred <- stats::predict(x, newdata = newdata, type = type.predict, se.fit = se_fit, ...)
    fit <- if (se_fit) pred$fit else pred
    cls <- stats::predict(x, newdata = newdata, type = "class", ...)
    frame <- .augment_frame(x, data, newdata, nrow(fit))
    prefix <- if (type.predict == "link") ".link_" else ".prob_"
    cols <- paste0(prefix, colnames(fit))
    new <- c(cols, ".class", if (se_fit) paste0(".se.fit_", colnames(fit)),
             if (is.null(newdata)) paste0(".resid_", x$levels))
    .augment_reserved(frame, new)
    for (j in seq_len(ncol(fit))) frame[[cols[j]]] <- unname(fit[, j])
    frame$.class <- unname(cls)
    if (se_fit) for (j in seq_len(ncol(fit))) frame[[paste0(".se.fit_", colnames(fit)[j])]] <- unname(pred$se.fit[, j])
    if (is.null(newdata)) {
        r <- stats::residuals(x, type = type.residuals)
        for (j in seq_along(x$levels)) frame[[paste0(".resid_", x$levels[j])]] <- unname(r[, j])
    }
    .augment_finish(frame)
}
