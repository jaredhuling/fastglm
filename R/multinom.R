#' Fit a multinomial logit model
#'
#' Fits a multinomial (baseline-category) logit model for a categorical
#' response by maximum likelihood or mean bias reduction, with the compiled
#' vector GLM engine. The model has one linear predictor for each non-reference
#' category, `log{Pr(Y = k) / Pr(Y = reference)} = offset_k + x' beta_k`.
#' @param x a formula, a finite numeric matrix of predictors, a sparse
#'   `Matrix::dgCMatrix` design, a `bigmemory::big.matrix`, or a
#'   [fastglm_source()] that supplies blocks of rows. A matrix is used as the design matrix as
#'   given, so it must contain an intercept column of ones if an intercept is
#'   wanted; formula fits include the intercept by default. A `dgCMatrix` is
#'   fitted without densifying it, with results equal to those for the
#'   corresponding dense matrix up to rounding. A `bigmemory::big.matrix` of
#'   type double (one contiguous allocation) is fitted in place: the engine
#'   maps its memory and reads it in blocks of `block.size` rows, and the fit
#'   keeps a reference to it (not a copy) as `x`. If its columns are unnamed,
#'   the coefficients are named as for `x1`, `x2`, ...; the `big.matrix`
#'   itself is never modified.
#' @param y a factor (individual data; a character vector is converted to a
#'   factor), or a matrix with one column of nonnegative counts per category
#'   (grouped data), whose column names are the category labels. In a formula,
#'   the response is a factor or `cbind()` of count columns.
#' @param data,subset,na.action,contrasts formula and model-frame arguments.
#'   Rows with missing values are removed by `na.action` (default `na.omit`),
#'   as in [stats::glm()]. The matrix method rejects missing values.
#' @param weights nonnegative prior weights, one per row. For grouped data
#'   each row's counts are multiplied by its weight.
#' @param offset a vector of length `n`, added to every linear predictor, or an
#'   `n` by `K - 1` matrix with one column per non-reference category, in the
#'   level order of the fit.
#' @param levels for a [fastglm_source()], the category labels in order. By
#'   default they are the levels of the prototype's factor response (a
#'   formula source) or of the first block's factor response or count
#'   columns. For a character or logical response without `levels`, one
#'   extra pass over the source collects the distinct values, sorted as
#'   [factor()] sorts them. A block with a response value outside the
#'   categories stops with an error naming it.
#' @param reference the reference category, as a level name or an index in
#'   `1, ..., K`. The default is the first level, as in `nnet::multinom()`.
#' @param firth use mean bias reduction through adjusted score equations,
#'   which gives finite estimates when the likelihood is maximized at infinity
#'   (separation).
#' @param tol convergence tolerance on the Fisher decrement, scaled by the
#'   deviance as in [fastglm()].
#' @param maxit maximum number of Fisher scoring iterations.
#' @param block.size maximum number of rows processed together by the
#'   numerical kernel. For a [fastglm_source()] the kernel works within each
#'   source block, so it also bounds the rows handled at once inside a block,
#'   and the source's own block size sets how many rows are read at a time.
#' @param control a named list of engine controls (for example `step_bound`).
#'   `solver` chooses how each Fisher scoring step is solved: `"exact"`
#'   solves the full information system, `"cg"` uses a truncated Newton
#'   method (preconditioned conjugate gradients on information-vector
#'   products), and `"auto"` (the default) uses `"cg"` when the design has at
#'   least 8 columns and the number of design columns times `K - 1` is large,
#'   and `"exact"` otherwise. The conjugate gradient solver does not form the
#'   full information during its conjugate gradient iterations. It forms it
#'   to confirm convergence, in the exact steps that follow, and for the
#'   returned `cov.unscaled`, so its peak memory is that of the exact solver.
#'   Convergence is declared by the same rule for both solvers, so the
#'   estimates agree to the convergence tolerance. The conjugate gradient
#'   solver can take more outer iterations than the exact solver, so `iter`
#'   and `maxit` counts may differ. `"auto"` uses the exact solver for Firth
#'   fits (`firth = TRUE`), and `"cg"` with `firth = TRUE` is an error. For a
#'   [fastglm_source()], `"auto"` is `"exact"`, and `"cg"` reads the whole
#'   source for every conjugate gradient step. The solver used is stored in
#'   the fit as `solver`, with the number of information-vector products as
#'   `products`. Under `"cg"`, `passes` also counts the information-vector
#'   products and the passes that confirm convergence. `threads` sets the
#'   number of threads for the compiled computations of this fit (and of its
#'   intercept-only fit for `null.deviance`); the default `NULL` uses the
#'   session setting of [fastglm_threads()]. Results are identical for every
#'   number of threads. The thread count used is stored in the fit as
#'   `threads`. A [fastglm_source()] is read on one thread, and each source
#'   block is split into blocks of `block.size` rows that are processed in
#'   parallel. `trace = TRUE` prints one line per iteration (the iteration,
#'   the passes over the data so far, the deviance, the step halvings of the
#'   iteration, and the elapsed seconds) and `FALSE` prints none; the default
#'   `NULL` uses `getOption("fastglm.trace")`, which defaults to `TRUE` for a
#'   [fastglm_source()] in an interactive session and `FALSE` otherwise.
#' @param ... additional arguments to the default or formula method (not used).
#' @details A [fastglm_source()] is read one block at a time, so the
#'   design is never held in memory as a whole. Each block (a data frame for a
#'   formula source, or a list with `X`, `y`, and optional `weights` and
#'   `offset`) supplies a factor, character, or count-matrix response. Offsets
#'   are one column per non-reference category (`offset` names in
#'   [fastglm_source()]), or one column applied to every linear predictor. The
#'   fit makes one preliminary pass for the category totals and the intercept
#'   column, then the passes of the engine, then a fit of the intercept-only
#'   model for `null.deviance`. `passes` counts the engine's passes for the
#'   model, as for in-memory fits. Under the exact solver there is one pass at
#'   the starting values, which also finds aliased columns, and one per step
#'   length tried, so a fit without step halving makes `iter + 1` passes, and
#'   bias reduction adds one pass per iteration and one at the start
#'   (`2 iter + 2`). Under the conjugate gradient solver, `passes` also counts
#'   the information-vector products and the passes that confirm convergence,
#'   so it exceeds `iter + 1`. `source.passes` counts
#'   every pass over the source, including the preliminary pass and the
#'   intercept-only fit. The fit keeps the source, not its rows: fitted
#'   probabilities are not stored (`fitted.values`, `y`, `x`, `prior.weights`,
#'   and `offset` are `NULL`) and `residuals()` is not available. Use
#'   `predict()` with `newdata`, or without it (optionally with a `sink` that
#'   receives one block at a time) to reread the source. The covariance and
#'   diagnostic functions also reread the source block by block, so it must
#'   return the same rows in the same order.
#'
#'   Coefficients are ordered term-major: all `K - 1` coefficients of the
#'   first design column, then those of the second, and so on, and are named
#'   `"<level>:<term>"`. A design column that is a linear combination of earlier
#'   columns is aliased: its coefficients are `NA` and the rank is reduced.
#'
#'   Every category must have a positive weighted number of observations,
#'   because otherwise no finite maximum exists; an empty category stops the
#'   fit with an error naming it. Rows with zero weight (or, for grouped data,
#'   zero total count) contribute nothing and are not counted in `nobs`.
#'   `df.residual` is `nobs * (K - 1)` minus the rank, and `null.deviance` is
#'   the deviance of the intercept-only model with the same weights and
#'   offsets.
#'
#'   For individual data the deviance is `-2` times the log likelihood. For
#'   grouped (count-matrix) data the saturated log likelihood, the sum over
#'   positive counts `y` of `w * y * log(y / m)` with `m` the row total, is
#'   subtracted, so that `deviance` and `null.deviance` equal those of
#'   [stats::glm()] and [fastglm()] for a grouped binomial response when
#'   `K = 2`. `loglik`, `logLik()`, `AIC()`, and `BIC()` are not changed by this
#'   convention. In a formula fit, unused levels of factor predictors (for
#'   example after `subset`) are dropped, as in [stats::glm()], while the
#'   response keeps all its levels.
#'
#'   Without bias reduction, a fit with separation has coefficients that
#'   diverge. The iteration then stops by the usual rule, the fit warns that
#'   fitted probabilities are numerically 0 or 1 (within 10 times the machine
#'   epsilon, the threshold of [stats::glm.fit()]), and `separation` is `TRUE`.
#'   With `firth = TRUE` the flag is stored but no warning is given.
#'   Bias-reduced fits near separation converge linearly rather than
#'   quadratically, so their coefficients stop within a distance that shrinks
#'   with `tol` (about 2e-4 at the default `tol = 1e-8` on a separated
#'   example, and about 3e-6 at `tol = 1e-12`); lower `tol` for tighter
#'   agreement with other software.
#' @returns An object of class `c("fastglm_multinom", "fastglm_vglm")` with
#'   `coefficients`, `cov.unscaled` (the inverse expected information),
#'   `loglik`, `deviance`, `null.deviance`, `df.residual`, `nobs`, `iter`,
#'   `passes`, `converged`, `convergence.reason`, `levels`, `reference`,
#'   `constraints`, `family`, `terms`, `xlevels`, `contrasts`, `call`,
#'   `fitted.values` (an `n` by `K` matrix of probabilities), `y`,
#'   `prior.weights`, `offset` (an `n` by `K - 1` matrix), `firth`, `rank`,
#'   `aliased`, `separation`, and the design matrix `x`, together with
#'   * `solver`: the solver used, `"exact"` or `"cg"` (see `control`).
#'   * `products`: the number of information-vector products (0 under the
#'     exact solver).
#'   * `threads`: the number of threads the fit used (see `control`).
#'   * `accumulation`: how the information was accumulated, `"lowrank"` (the
#'     low-rank form used for dense designs with `K - 1` of at least 6) or
#'     `"pairwise"` (one weighted cross product per pair of linear
#'     predictors).
#' @seealso `nnet::multinom()` and `VGAM::vglm()` with `multinomial()` for
#'   comparable fits; [fastglm_ordinal()] for ordered outcomes.
#' @examples
#' set.seed(1)
#' n <- 400
#' d <- data.frame(x1 = rnorm(n), g = factor(sample(c("a", "b"), n, TRUE)))
#' eta <- cbind(0, 0.5 * d$x1, -0.5 * d$x1 + 0.3 * (d$g == "b"))
#' p <- exp(eta) / rowSums(exp(eta))
#' d$y <- factor(c("low", "mid", "high")[
#'     apply(p, 1, function(q) sample.int(3, 1, prob = q))],
#'     levels = c("low", "mid", "high"))
#' fit <- fastglm_multinom(y ~ x1 + g, data = d)
#' fit$coefficients
#' head(fit$fitted.values)
#' @md
#' @export
fastglm_multinom <- function(x, ...) {
    if (missing(x)) {
        mc <- match.call(expand.dots = TRUE)
        if (is.null(mc$formula))
            stop("fastglm_multinom: supply a design matrix or model formula", call. = FALSE)
        mc$x <- mc$formula
        mc$formula <- NULL
        mc[[1L]] <- quote(fastglm::fastglm_multinom)
        return(eval(mc, parent.frame()))
    }
    UseMethod("fastglm_multinom")
}

#' @rdname fastglm_multinom
#' @export
fastglm_multinom.default <- function(x, y, weights = NULL, offset = NULL,
                                     reference = NULL, firth = FALSE,
                                     tol = 1e-8, maxit = 100L,
                                     block.size = 8192L, control = list(), ...) {
    if (length(list(...))) stop("fastglm_multinom: unused arguments", call. = FALSE)
    if (missing(y)) stop("fastglm_multinom: a categorical response 'y' is required", call. = FALSE)
    fit <- .multinom_fit(x, y, weights, offset, reference, firth, tol, maxit,
                         block.size, control)
    fit$call <- match.call()
    fit$call[[1L]] <- quote(fastglm::fastglm_multinom)
    fit
}

#' @rdname fastglm_multinom
#' @export
fastglm_multinom.dgCMatrix <- fastglm_multinom.default

#' @rdname fastglm_multinom
#' @export
fastglm_multinom.big.matrix <- fastglm_multinom.default

#' @rdname fastglm_multinom
#' @export
fastglm_multinom.fastglm_source <- function(x, levels = NULL, reference = NULL, firth = FALSE,
                                            tol = 1e-8, maxit = 100L,
                                            block.size = 8192L, control = list(), ...) {
    if (length(list(...))) stop("fastglm_multinom: unused arguments", call. = FALSE)
    err <- function(...) stop("fastglm_multinom: ", ..., call. = FALSE)
    call <- match.call()
    call[[1L]] <- quote(fastglm::fastglm_multinom)
    source <- x
    schema <- source$schema
    if (!is.null(schema) && !inherits(schema$terms, "terms"))
        err("a multinomial source needs a formula with one right-hand side")
    .multinom_check_controls(tol, maxit, block.size, firth, control)
    # The thread count holds for every pass, the intercept-only fit included,
    # and is not kept in the refit controls (bootstrap refits take theirs
    # from the session, one thread per worker process).
    scope <- .fastglm_scope_threads(control[["threads"]])
    on.exit(scope$restore(), add = TRUE)
    control$threads <- NULL
    # The trace applies to this fit only (not its intercept-only fit or refits).
    tr <- .fastglm_scope_trace(control[["trace"]], streamed = TRUE)
    on.exit(tr$restore(), add = TRUE)
    control$trace <- NULL
    # Fail fast on "cg" with bias reduction, before any pass over the source.
    .vglm_control_solver(control, 1L, 1L, firth, streamed = TRUE)
    read <- .vglm_stream_reader(source, levels)
    state <- environment(read)$state

    # Preliminary pass: weighted category totals (for the starting values and
    # the empty-category check), the rows that enter the likelihood, and the
    # intercept, the first design column that is constant and nonzero over
    # every row (the rule of the in-memory fits).
    totals <- NULL
    used <- 0L
    lsat <- 0
    offset.used <- FALSE
    for (k in seq_len(source$n_chunks)) {
        # After a pass that collected the response levels, block 1 starts a
        # new pass.
        z <- if (k == 1L) tryCatch(read(1L), vglm_levels_scanned = function(e) read(1L)) else read(k)
        if (is.null(totals)) {
            totals <- numeric(length(state$levels))
            first <- z$X[1L, ]
            constant <- first != 0
        }
        b <- nrow(z$X)
        if (state$grouped) {
            totals <- totals + colSums(z$counts * z$weights)
            used <- used + sum(z$weights > 0 & rowSums(z$counts) > 0)
            lsat <- lsat + .multinom_saturated(z$counts, z$weights)
        } else {
            s <- rowsum(z$weights, z$codes)
            j <- as.integer(rownames(s))
            totals[j] <- totals[j] + s[, 1L]
            used <- used + sum(z$weights > 0)
        }
        constant <- constant & colSums(z$X != rep(first, each = b)) == 0
        offset.used <- offset.used || any(z$offset != 0)
    }
    lev <- state$levels
    K <- length(lev)
    M <- K - 1L
    empty <- which(!(totals > 0))
    if (length(empty)) err("category '", lev[empty[1L]], "' has no observations")
    ref <- .multinom_reference(reference, lev)
    intercept <- if (any(constant)) which(constant)[1L] else 0L
    value <- if (intercept) first[[intercept]] else 0
    p <- length(state$columns)
    control.null <- .vglm_control_solver(control, 1L, M, FALSE, streamed = TRUE)
    if (identical(control[["solver"]], "cg") && !firth)
        warning("fastglm_multinom: solver = \"cg\" reads the whole source for every conjugate gradient step",
                call. = FALSE)
    control <- .vglm_control_solver(control, p, M, firth, streamed = TRUE)
    engine <- function(fetch, p, intercept, value, constraints, firth, ctl = control, trace = FALSE)
        .multinom_engine_errors(vglm_fit_source(
            fetch, source$n_chunks, as.integer(p), totals, as.integer(intercept), value,
            constraints, "multinomial", as.integer(K), as.integer(ref), firth, tol,
            as.integer(maxit), block.size, trace, ctl))
    constraints <- rep(list(diag(M)), p)
    fit <- engine(read, p, intercept, value, constraints, firth, trace = tr$trace)
    null <- engine(function(k) {
        z <- read(k)
        z$X <- matrix(1, nrow(z$X), 1L)
        z
    }, 1L, 1L, 1, list(diag(M)), FALSE, control.null)

    info <- list(x = NULL, y = NULL, weights = NULL, offset = NULL,
                 null.deviance = null$deviance, nobs = used, lsat = lsat,
                 grouped = state$grouped,
                 constraints = constraints, firth = firth, design.names = state$columns,
                 terms = schema$terms, xlevels = schema$xlevels,
                 contrasts = schema$contrasts,
                 refit = list(tol = tol, maxit = maxit, block.size = block.size,
                              control = control))
    out <- .vglm_fit_object(fit, "multinomial", lev, lev[ref], info, call)
    out$source <- source
    out$fetch <- read
    out$n.rows <- sum(state$rows)
    out$design.assign <- schema$assign
    out$source.offset <- if (is.character(schema$offset)) schema$offset else character(0)
    out$source.offset.used <- offset.used
    out$source.passes <- state$passes
    out$source.reads <- state$reads
    out$threads <- scope$threads
    .multinom_warnings(out)
}

# The reader of a multinomial fit over a fastglm_source. read(k) returns block
# k normalized to list(X, codes, counts, weights, offset): X the b by p dense
# design; codes the 1-based categories (integer(0) for grouped data); counts
# the b by K count matrix (0 by 0 for individual data); weights the b prior
# weights (default 1); and offset the b by M offset matrix (a vector, or a
# source with one offset column, applies to every linear predictor). The
# categories are `levels` if given, otherwise the levels of the prototype's
# response (formula sources), otherwise those of the first block (a factor's
# levels or the count column names, or, for a character or logical response,
# the sorted distinct values over a full pass, after which read(1) signals
# "vglm_levels_scanned" so that the caller restarts the pass). A response
# value outside them stops with the value named. The state (levels, design columns, rows per block, and
# counts of passes and block reads) is kept in environment(read)$state, so the
# fit keeps the reader but not a source block.
.vglm_stream_reader <- function(source, levels = NULL) {
    force(source)
    err <- function(...) stop("fastglm_multinom: ", ..., call. = FALSE)
    if (!is.null(levels)) {
        if (is.factor(levels)) levels <- as.character(levels)
        if (!is.character(levels) || length(levels) < 2L || anyNA(levels) ||
            any(!nzchar(levels)) || anyDuplicated(levels))
            err("'levels' must name at least two distinct categories")
    }
    state <- new.env(parent = emptyenv())
    state$passes <- 0L
    state$reads <- 0L
    state$rows <- integer(source$n_chunks)
    state$levels <- if (!is.null(levels)) levels else source$schema$response.levels
    state$columns <- NULL
    state$grouped <- NULL
    per.predictor <- isTRUE(source$schema$offset.per.predictor)
    scan.levels <- function(y, k) {
        if (k != 1L) err("the response of source block ", k, " is not a factor; supply 'levels'")
        values <- unique(as.character(y))
        for (j in seq_len(source$n_chunks)[-1L]) {
            state$reads <- state$reads + 1L
            yj <- (if (per.predictor) source$callback(j, per.predictor = TRUE) else source$callback(j))$y
            if (is.data.frame(yj)) yj <- as.matrix(yj)
            if (is.matrix(yj) && ncol(yj) == 1L && !is.numeric(yj)) yj <- yj[, 1L]
            if (!is.null(dim(yj)) || !(is.factor(yj) || is.character(yj) || is.logical(yj)))
                err("the response of source block ", j, " is not of the kind (factor or counts) of block 1")
            values <- union(values, unique(as.character(yj)))
        }
        sort(values[!is.na(values)])
    }
    read <- function(k) {
        if (k == 1L) state$passes <- state$passes + 1L
        state$reads <- state$reads + 1L
        z <- if (per.predictor) source$callback(k, per.predictor = TRUE) else source$callback(k)
        if (!is.list(z)) err("source block ", k, " must be a list with X and y")
        x <- z$X
        if (is.data.frame(x)) x <- as.matrix(x)
        if (!is.matrix(x) || !is.numeric(x))
            err("source block ", k, " must supply X as a numeric matrix")
        b <- nrow(x)
        if (b < 1L || ncol(x) < 1L) err("source block ", k, " has no rows or no design columns")
        if (!all(is.finite(x))) err("the design must be finite with no missing values")
        storage.mode(x) <- "double"
        cn <- colnames(x)
        if (is.null(cn)) cn <- paste0("x", seq_len(ncol(x)))
        if (is.null(state$columns)) {
            if (anyNA(cn) || any(!nzchar(cn)) || anyDuplicated(cn))
                err("design column names must be distinct and nonmissing")
            state$columns <- cn
        } else if (!identical(cn, state$columns))
            err("the design columns of source block ", k, " differ from those of block 1")
        if (state$rows[k] != 0L && state$rows[k] != b)
            err("source block ", k, " changed size between passes")
        state$rows[k] <- b

        y <- z$y
        if (is.data.frame(y)) y <- as.matrix(y)
        if (is.matrix(y) && ncol(y) == 1L && !is.numeric(y)) y <- y[, 1L]
        grouped <- is.matrix(y)
        if (is.null(state$grouped)) state$grouped <- grouped
        else if (!identical(grouped, state$grouped))
            err("the response of source block ", k, " is not of the kind (factor or counts) of block 1")
        if (grouped) {
            if (!is.numeric(y) || nrow(y) != b)
                err("the count response of source block ", k, " must be numeric with one row per design row")
            if (is.null(state$levels))
                state$levels <- if (is.null(colnames(y))) as.character(seq_len(ncol(y))) else colnames(y)
            if (ncol(y) != length(state$levels) ||
                (!is.null(colnames(y)) && !identical(colnames(y), state$levels)))
                err("the count columns of source block ", k, " do not match the categories ",
                    paste(state$levels, collapse = ", "))
            if (!all(is.finite(y)) || any(y < 0)) err("counts must be finite and nonnegative")
            counts <- unname(y)
            storage.mode(counts) <- "double"
            codes <- integer(0)
        } else {
            if (!is.null(dim(y)) || length(y) != b)
                err("the response of source block ", k, " must have one value per design row")
            if (!(is.factor(y) || is.character(y) || is.logical(y)))
                err("the response must be a factor, a character vector, or a matrix of counts")
            if (is.null(state$levels)) {
                if (is.factor(y)) state$levels <- base::levels(y)
                else {
                    # A character or logical response without 'levels': the
                    # rest of this pass collects the distinct values, sorted
                    # as factor() sorts them, and the caller starts a new pass.
                    state$levels <- scan.levels(y, k)
                    stop(structure(class = c("vglm_levels_scanned", "error", "condition"),
                                   list(message = "response levels collected", call = NULL)))
                }
            }
            if (anyNA(y)) err("missing values in the response of source block ", k)
            y <- as.character(y)
            codes <- match(y, state$levels)
            if (anyNA(codes))
                err("source block ", k, " has response value(s) that are not categories of the fit: '",
                    paste(unique(y[is.na(codes)]), collapse = "', '"), "' (categories: ",
                    paste(state$levels, collapse = ", "), ")")
            counts <- matrix(0, 0L, 0L)
        }
        if (length(state$levels) < 2L) err("the response must have at least two categories")
        M <- length(state$levels) - 1L

        w <- z$weights
        if (is.null(w)) w <- rep(1, b)
        if (!is.numeric(w) || !is.null(dim(w)) || length(w) != b || anyNA(w) ||
            !all(is.finite(w)) || any(w < 0))
            err("the weights of source block ", k, " must be finite and nonnegative, one per row")
        off <- z$offset
        if (is.null(off)) off <- 0
        if (is.matrix(off) && ncol(off) == 1L) off <- off[, 1L]
        if (!is.numeric(off) || (is.matrix(off) && (nrow(off) != b || ncol(off) != M)) ||
            (!is.matrix(off) && length(off) != 1L && length(off) != b))
            err("the offset of source block ", k, " must be a vector or a ", b, " by K - 1 matrix (K - 1 = ",
                M, "); a source needs one offset column per linear predictor")
        if (!all(is.finite(off))) err("'offset' must be finite with no missing values")
        off <- matrix(as.numeric(off), b, M)
        list(X = x, codes = as.integer(codes), counts = counts, weights = as.numeric(w),
             offset = off)
    }
    read
}

#' @rdname fastglm_multinom
#' @export
fastglm_multinom.formula <- function(x, data, weights, subset, na.action,
                                     offset = NULL, reference = NULL,
                                     firth = FALSE, tol = 1e-8, maxit = 100L,
                                     block.size = 8192L, control = list(),
                                     contrasts = NULL, ...) {
    if (length(list(...))) stop("fastglm_multinom: unused arguments", call. = FALSE)
    call <- match.call()
    mf <- call[c(1L, match(c("x", "data", "weights", "subset", "na.action", "offset"),
                           names(call), 0L))]
    names(mf)[names(mf) == "x"] <- "formula"
    mf[[1L]] <- quote(stats::model.frame)
    # The response keeps all of its levels (an emptied category is an error
    # naming it); unused levels of the predictors are dropped, as in glm().
    # As in model.frame(drop.unused.levels = TRUE), only factors that have
    # unused levels are touched, so contrasts set on the others are kept, and
    # a contrasts attribute lost by dropping levels is reported.
    mf$drop.unused.levels <- FALSE
    mf <- eval(mf, parent.frame())
    terms <- attr(mf, "terms")
    for (j in seq_along(mf)[-1L]) {
        x <- mf[[j]]
        if (is.factor(x) && length(unique(x[!is.na(x)])) < length(levels(x))) {
            ctr <- attr(x, "contrasts")
            mf[[j]] <- x[, drop = TRUE]
            if (!identical(attr(mf[[j]], "contrasts"), ctr))
                warning(gettextf("fastglm_multinom: contrasts dropped from factor %s due to missing levels",
                                 names(mf)[j]), call. = FALSE)
        }
    }
    y <- model.response(mf)
    if (is.null(y)) stop("fastglm_multinom: the formula needs a response", call. = FALSE)
    design <- model.matrix(terms, mf, contrasts.arg = contrasts)
    fit <- .multinom_fit(design, y, model.weights(mf), model.offset(mf), reference,
                         firth, tol, maxit, block.size, control,
                         info = list(terms = terms,
                                     xlevels = stats::.getXlevels(terms, mf),
                                     contrasts = attr(design, "contrasts"),
                                     na.action = attr(mf, "na.action")))
    # Store the formula under the name `formula` so that update() can replace
    # it; the generic maps it back to `x` when the call is evaluated.
    names(call)[names(call) == "x"] <- "formula"
    fit$call <- call
    fit$call[[1L]] <- quote(fastglm::fastglm_multinom)
    fit
}

# Validate the inputs, call the engine, and build the fitted object.
.multinom_fit <- function(x, y, weights, offset, reference, firth, tol, maxit,
                          block.size, control, info = list()) {
    err <- function(...) stop("fastglm_multinom: ", ..., call. = FALSE)
    sparse <- inherits(x, "dgCMatrix")
    big <- inherits(x, "big.matrix")
    if (!sparse && !big) {
        if (is.data.frame(x)) x <- as.matrix(x)
        if (!is.matrix(x) || !is.numeric(x))
            err("'x' must be a formula, a dense numeric matrix, a dgCMatrix, or a big.matrix")
    }
    if (!nrow(x) || !ncol(x)) err("the design must have at least one row and one column")
    # The finiteness of a big.matrix is checked by the engine, block by block.
    if (!big && !all(is.finite(if (sparse) x@x else x)))
        err("the design must be finite with no missing values")
    if (!sparse && !big) storage.mode(x) <- "double"
    n <- nrow(x)
    # bigmemory forbids changing dimnames, so an unnamed big.matrix keeps its
    # NULL names and the default names are supplied by .vglm_colnames().
    if (big) {
        cn <- .vglm_colnames(x)
    } else {
        if (is.null(colnames(x))) colnames(x) <- paste0("x", seq_len(ncol(x)))
        cn <- colnames(x)
    }
    if (anyNA(cn) || any(!nzchar(cn)) || anyDuplicated(cn))
        err("design column names must be distinct and nonmissing")
    .multinom_check_controls(tol, maxit, block.size, firth, control)
    # The thread count holds for the fit and its intercept-only fit, and is
    # not kept in the refit controls (see the source method).
    scope <- .fastglm_scope_threads(control[["threads"]])
    on.exit(scope$restore(), add = TRUE)
    control$threads <- NULL
    tr <- .fastglm_scope_trace(control[["trace"]])
    on.exit(tr$restore(), add = TRUE)
    control$trace <- NULL

    # Response: factor (individual) or count matrix (grouped).
    grouped <- FALSE
    if (is.data.frame(y)) y <- as.matrix(y)
    if (is.matrix(y) && ncol(y) == 1L && !is.numeric(y)) y <- y[, 1L]
    if (is.matrix(y)) {
        if (!is.numeric(y)) err("a count-matrix response must be numeric")
        grouped <- TRUE
        if (nrow(y) != n) err("the response must have the same number of rows as 'x'")
        if (any(is.na(y) & !is.nan(y))) err("missing values in the response")
        if (!all(is.finite(y)) || any(y < 0)) err("counts must be finite and nonnegative")
        lev <- colnames(y)
        if (is.null(lev)) lev <- as.character(seq_len(ncol(y)))
        K <- ncol(y)
        counts <- unname(y)
        storage.mode(counts) <- "double"
        codes <- integer(0)
        yout <- y
    } else {
        if (is.logical(y) || is.character(y)) y <- factor(y)
        if (!is.factor(y)) err("the response must be a factor or a matrix of counts")
        if (length(y) != n) err("the response must have the same number of rows as 'x'")
        if (anyNA(y)) err("missing values in the response")
        lev <- levels(y)
        K <- length(lev)
        codes <- as.integer(y)
        counts <- matrix(0, 0L, 0L)
        yout <- y
    }
    if (K < 2L) err("the response must have at least two categories")
    M <- K - 1L

    # Weights.
    if (is.null(weights)) weights <- rep(1, n)
    if (!is.numeric(weights) || length(weights) != n)
        err("'weights' must be a numeric vector with one entry per row")
    if (anyNA(weights) || !all(is.finite(weights)) || any(weights < 0))
        err("'weights' must be finite and nonnegative, with no missing values")
    weights <- as.numeric(weights)

    # Offsets: a vector for every linear predictor, or an n by M matrix.
    if (is.null(offset)) {
        offset <- matrix(0, n, M)
    } else {
        if (!is.numeric(offset)) err("'offset' must be numeric")
        if (is.matrix(offset) && ncol(offset) == 1L) offset <- offset[, 1L]
        if (is.matrix(offset)) {
            if (nrow(offset) != n || ncol(offset) != M)
                err("'offset' must be a vector of length n or an n by K - 1 matrix (here ",
                    n, " by ", M, ")")
        } else {
            if (length(offset) != n)
                err("'offset' must be a vector of length n or an n by K - 1 matrix (here ",
                    n, " by ", M, ")")
        }
        if (!all(is.finite(offset))) err("'offset' must be finite with no missing values")
        offset <- matrix(as.numeric(offset), n, M)
    }

    ref <- .multinom_reference(reference, lev)

    # Empty categories: no finite maximum exists.
    mass <- if (grouped) colSums(counts * weights) else
        tabulate(codes[weights > 0], nbins = K)
    empty <- which(!(mass > 0))
    if (length(empty))
        err("category '", lev[empty[1L]], "' has no observations")

    used <- if (grouped) weights > 0 & rowSums(counts) > 0 else weights > 0
    lsat <- if (grouped) .multinom_saturated(counts, weights) else 0
    constraints <- rep(list(diag(M)), ncol(x))
    control.null <- .vglm_control_solver(control, 1L, M, FALSE)
    control <- .vglm_control_solver(control, ncol(x), M, firth)
    run <- function(xx, cons, fth, ctl, trace = FALSE) .multinom_engine(
        xx, codes, counts, weights, offset, cons, K, ref, fth, tol, maxit,
        block.size, ctl, trace)
    fit <- run(x, constraints, firth, control, tr$trace)
    null <- run(matrix(1, n, 1L), list(diag(M)), FALSE, control.null)

    info <- c(info, list(x = x, y = yout, weights = weights, offset = offset,
                         null.deviance = null$deviance, nobs = sum(used),
                         lsat = lsat, grouped = grouped, constraints = constraints,
                         firth = firth,
                         refit = list(tol = tol, maxit = maxit, block.size = block.size,
                                      control = control)))
    out <- .vglm_fit_object(fit, "multinomial", lev, lev[ref], info, NULL)
    out$threads <- scope$threads
    .multinom_warnings(out)
}

# The saturated log likelihood of grouped data, sum over y > 0 of
# w * y * log(y / m) with m the row total. It is subtracted from -2 loglik to
# give the deviance of a count-matrix response, as for a grouped binomial
# glm(); for individual data it is zero.
.multinom_saturated <- function(counts, weights) {
    m <- rowSums(counts)
    pos <- counts > 0
    sum((weights * counts * log(counts / ifelse(m > 0, m, 1)))[pos])
}

# Checks of the fitting controls shared by every multinomial method.
.multinom_check_controls <- function(tol, maxit, block.size, firth, control) {
    err <- function(...) stop("fastglm_multinom: ", ..., call. = FALSE)
    if (!is.list(control)) err("'control' must be a list")
    if (!is.numeric(tol) || length(tol) != 1L || !is.finite(tol) || tol <= 0)
        err("'tol' must be a positive number")
    if (!is.numeric(maxit) || length(maxit) != 1L || !is.finite(maxit) || maxit < 1)
        err("'maxit' must be a positive integer")
    if (!is.numeric(block.size) || length(block.size) != 1L || !is.finite(block.size) || block.size < 1)
        err("'block.size' must be a positive integer")
    if (!is.logical(firth) || length(firth) != 1L || is.na(firth))
        err("'firth' must be TRUE or FALSE")
    threads <- control[["threads"]]
    if (!is.null(threads) && (!is.numeric(threads) || is.na(.fastglm_as_count(threads))))
        err("'control$threads' must be a positive integer")
    trace <- control[["trace"]]
    if (!is.null(trace) && (!is.logical(trace) || length(trace) != 1L || is.na(trace)))
        err("'control$trace' must be TRUE, FALSE, or NULL")
    solver <- control[["solver"]]
    if (!is.null(solver) &&
        !(is.character(solver) && length(solver) == 1L && !is.na(solver) &&
          solver %in% c("auto", "exact", "cg")))
        err("'control$solver' must be \"auto\", \"exact\", or \"cg\"")
    invisible(NULL)
}

# The number of columns p of the design times M = K - 1 linear predictors at
# and above which solver = "auto" chooses the conjugate gradient solver
# (timing study, Part 3: cg/exact time ratio 1.18 at p M = 209 and 0.88 at 539
# with 11 design columns; 0.93 at 204 and 0.67 at 459 with 51).
.vglm_cg_threshold <- 400L

# The fewest design columns p (intercept included) at which solver = "auto"
# chooses the conjugate gradient solver. With few columns the per-row work of
# the model type and the extra diagonal passes outweigh the cheaper products
# (n = 20000: p = 5, K = 100, cg 1.64 s against exact 1.40 s; p = 3,
# K = 150, cg 2.18 s against exact 1.76 s; p = 8, K = 60, cg 1.03 s against
# exact 1.20 s). The floor also keeps intercept-only and null fits on the
# exact solver, whose start is already the maximum.
.vglm_cg_min_p <- 8L

# Resolve the user's solver ("auto", "exact", or "cg") to the "exact" or "cg"
# the engine accepts. "auto" is "exact" for bias reduction, for a streamed
# source, when p is below .vglm_cg_min_p, and when p * M is below the
# threshold.
.vglm_resolve_solver <- function(solver, p, M, firth, streamed) {
    if (identical(solver, "cg") && isTRUE(firth))
        stop("fastglm_multinom: the conjugate gradient solver is not available with bias reduction",
             call. = FALSE)
    if (solver != "auto") return(solver)
    if (isTRUE(firth) || isTRUE(streamed) || p < .vglm_cg_min_p ||
        as.double(p) * as.double(M) < .vglm_cg_threshold)
        "exact" else "cg"
}

# Return `control` with its solver resolved for a design with p columns and
# M linear predictors; the user's value defaults to "auto".
.vglm_control_solver <- function(control, p, M, firth, streamed = FALSE) {
    solver <- if (is.null(control[["solver"]])) "auto" else control[["solver"]]
    control$solver <- .vglm_resolve_solver(solver, p, M, firth, streamed)
    control
}

# The 1-based index of the reference category among the levels lev.
.multinom_reference <- function(reference, lev) {
    err <- function(...) stop("fastglm_multinom: ", ..., call. = FALSE)
    K <- length(lev)
    if (is.null(reference)) return(1L)
    if (length(reference) != 1L || is.na(reference)) err("invalid 'reference'")
    if (is.character(reference) || is.factor(reference)) {
        ref <- match(as.character(reference), lev)
        if (is.na(ref)) err("'reference' ('", reference, "') is not a category; levels are ",
                            paste(lev, collapse = ", "))
        return(ref)
    }
    if (is.numeric(reference) && reference == round(reference) &&
        reference >= 1 && reference <= K) return(as.integer(reference))
    err("'reference' must be a level name or an index in 1, ..., ", K)
}

# The warnings of a multinomial fit, as for glm.fit; returns the fit.
.multinom_warnings <- function(out) {
    if (!isTRUE(out$converged)) warning("fastglm: algorithm did not converge", call. = FALSE)
    if (!isTRUE(out$firth) && isTRUE(out$separation))
        warning("fastglm: fitted probabilities numerically 0 or 1 occurred; consider firth = TRUE",
                call. = FALSE)
    out
}

# Call the compiled engine; engine errors are rethrown with a clear prefix.
.multinom_engine <- function(x, codes, counts, weights, offset, constraints, K,
                             reference, firth, tol, maxit, block.size, control,
                             trace = FALSE) {
    # A sparse design is passed transposed, so a block of rows is a
    # contiguous range of columns of the column-compressed matrix.
    sparse <- inherits(x, "dgCMatrix")
    big <- inherits(x, "big.matrix")
    .multinom_engine_errors(
        (if (sparse) vglm_fit_sparse else if (big) vglm_fit_big else vglm_fit_dense)(
            if (sparse) Matrix::t(x) else if (big) x@address else x, codes, counts, weights, offset,
            constraints, "multinomial", as.integer(K), as.integer(reference),
            firth, tol, as.integer(maxit), as.integer(block.size), trace, control))
}

# Evaluate an engine call, rethrowing its errors with a clear prefix.
.multinom_engine_errors <- function(expr) {
    tryCatch(expr,
        error = function(e) {
            msg <- conditionMessage(e)
            if (grepl("adjustment is not finite", msg, fixed = TRUE))
                msg <- paste0("the bias reduction adjustment is not finite; the ",
                              "design may be too close to rank deficient or ",
                              "the fit too close to the boundary")
            else if (grepl("singular at the starting values", msg, fixed = TRUE))
                msg <- paste0("the information is singular at the starting values, ",
                              "so bias reduction cannot start; check the design ",
                              "for aliased or constant columns")
            # Errors of a source reader, called from the engine, carry the
            # prefix already.
            stop(if (startsWith(msg, "fastglm_multinom: ")) "" else "fastglm_multinom: ", msg,
                 call. = FALSE)
        })
}

# Build the fitted object from an engine result. `x_info` holds the data and
# model-frame pieces (x, y, weights, offset, constraints, null.deviance, nobs,
# and optionally terms, xlevels, contrasts, na.action).
.vglm_fit_object <- function(fit, family, levels, reference, x_info, call, ...) {
    K <- length(levels)
    M <- K - 1L
    x <- x_info$x
    lsat <- if (is.null(x_info$lsat)) 0 else x_info$lsat
    ref <- match(reference, levels)
    nonref <- levels[-ref]
    cn <- if (!is.null(x_info$design.names)) x_info$design.names else .vglm_colnames(x)
    nm <- as.vector(outer(nonref, cn, paste, sep = ":"))
    coefficients <- fit$coefficients
    names(coefficients) <- nm
    cov <- fit$cov.unscaled
    dimnames(cov) <- list(nm, nm)
    aliased <- fit$aliased
    names(aliased) <- nm

    # Fitted probabilities, for in-memory fits only (a streamed source is not
    # retained); aliased coefficients are held at zero.
    probs <- NULL
    if (!is.null(x)) {
        B <- coefficients
        B[is.na(B)] <- 0
        eta <- .vglm_linear(x, B, M) + x_info$offset
        full <- matrix(0, nrow(x), K)
        full[, -ref] <- eta
        # Row maxima by columns (pmax), which is exact and much faster than
        # apply() over the rows.
        top <- full[, 1L]
        for (k in seq_len(K)[-1L]) top <- pmax(top, full[, k])
        full <- exp(full - top)
        probs <- full / rowSums(full)
        dimnames(probs) <- list(if (!inherits(x, "big.matrix")) rownames(x), levels)
    }

    out <- list(
        coefficients = coefficients, cov.unscaled = cov,
        loglik = fit$loglik, deviance = fit$deviance + 2 * lsat,
        null.deviance = x_info$null.deviance + 2 * lsat,
        df.residual = x_info$nobs * M - fit$rank,
        nobs = as.integer(x_info$nobs), iter = fit$iter, passes = fit$passes,
        solver = fit$solver, products = fit$products, accumulation = fit$accumulation,
        converged = fit$converged, convergence.reason = fit$convergence.reason,
        levels = levels, reference = reference,
        constraints = x_info$constraints, family = family,
        terms = x_info$terms, xlevels = x_info$xlevels,
        contrasts = x_info$contrasts, na.action = x_info$na.action,
        call = call, fitted.values = probs, y = x_info$y,
        prior.weights = x_info$weights, offset = x_info$offset,
        firth = isTRUE(x_info$firth), refit.controls = x_info$refit,
        rank = fit$rank, aliased = aliased, separation = fit$separation,
        singular = isTRUE(fit$singular),
        information = fit$information, score = fit$score,
        fisher.decrement = fit$fisher.decrement, x = x, design.names = cn)
    if (out$firth) {
        out$penalized.loglik <- fit$penalized.loglik
        out$adjustment <- fit$adjustment
    }
    class(out) <- c(paste0("fastglm_", if (family == "multinomial") "multinom" else family),
                    "fastglm_vglm")
    out
}

# The linear predictors x %*% t(B) (n by M, B the term-major coefficient
# vector with aliased entries set to 0) for a dense, sparse, or big.matrix x.
# A big.matrix, or any x with `cols` (an index into its columns, used to match
# columns by name), is read in row blocks of at most `block` rows, so only one
# block of it is in memory as an R matrix at a time.
.vglm_linear <- function(x, B, M, cols = NULL, block = 8192L) {
    .check_big_alive(x)
    Bm <- t(matrix(B, M))
    if (!inherits(x, "big.matrix") && is.null(cols)) return(as.matrix(x %*% Bm))
    n <- nrow(x)
    out <- matrix(0, n, M)
    for (first in seq.int(1L, max(n, 1L), by = block)) {
        if (n == 0L) break
        i <- first:min(n, first + block - 1L)
        xi <- x[i, if (is.null(cols)) seq_len(ncol(x)) else cols, drop = FALSE]
        out[i, ] <- as.matrix(xi %*% Bm)
    }
    out
}

# Column names of a design, with x1, x2, ... for an unnamed one (a big.matrix
# cannot be renamed in place).
.vglm_colnames <- function(x) {
    cn <- colnames(x)
    if (is.null(cn)) paste0("x", seq_len(ncol(x))) else cn
}

# Stop clearly when a big.matrix's external pointer is null, as after the fit
# was saved and reloaded: dereferencing it would crash R.
.check_big_alive <- function(x) {
    if (inherits(x, "big.matrix") && bigmemory::is.nil(x@address))
        stop("the big.matrix referenced by this fit is no longer available (for example after the fit was saved and reloaded); refit or supply newdata",
             call. = FALSE)
    invisible(x)
}

# Design column names of a fit, from the names cached at fit time (the
# design itself may be a big.matrix that is no longer available).
.vglm_design_names <- function(object) {
    if (!is.null(object$design.names)) object$design.names else .vglm_colnames(object$x)
}
