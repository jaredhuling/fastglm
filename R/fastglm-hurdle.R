#' Hurdle (two-part) Poisson / Negative-Binomial regression
#'
#' Fits a hurdle model for count data with excess zeros: a binary
#' (zero / non-zero) regression on the full sample combined with a zero-
#' truncated count regression on the positive subset. The two parts have
#' independent likelihoods and are estimated jointly. With `dist = "negbin"`
#' the negative-binomial dispersion `theta` is estimated by an inner
#' Brent-1D MLE inside the C++ driver, alternating with the count-side
#' IRLS.
#'
#' Formula fits use the native count-model driver. Sparse designs and data
#' supplied in blocks use the same likelihood with blockwise score updates.
#' The latter permits repeated passes over data that need not fit in memory.
#'
#' @param formula a numeric matrix, `dgCMatrix`, [fastglm_source()], or a
#'   [Formula::Formula] of the form `y ~ x1 + x2 | z1 + z2`
#'   where the right-hand side after `|` specifies the zero-model design.
#'   If `|` is absent, the same RHS is used for both parts.
#' @param response nonnegative integer response for a matrix design. A source
#'   supplies its response in each block; the existing `y` argument controls
#'   whether a fitted model retains its response.
#' @param z optional binary-component matrix for matrix inputs; defaults to
#'   the count design. Source blocks can supply `Z` separately from `X`.
#' @param method decomposition for matrix/source component GLMs, as in [fastglm()].
#' @param block.size maximum number of matrix rows processed together for the
#'   truncated count component. A source determines its own block sizes.
#' @param data optional data frame / environment from which to draw model
#'   matrix variables.
#' @param subset,na.action standard model-frame arguments.
#' @param weights optional non-negative prior weights.
#' @param offset optional vector or matrix.  If a 2-column matrix, columns
#'   are taken as `(count_offset, zero_offset)`.
#' @param dist count component distribution: `"poisson"` or `"negbin"`.
#' @param zero.dist zero/non-zero distribution; only `"binomial"` is
#'   currently supported.
#' @param link character, link for the zero/non-zero binomial.  One of
#'   `"logit"` (default), `"probit"`, `"cloglog"`, `"log"`.
#' @param init.theta optional positive scalar starting value for the NB
#'   dispersion. Formula fits use a moment initializer by default; matrix
#'   and source fits start at one after fitting a truncated Poisson model.
#' @param tol,maxit IRLS-loop convergence tolerance and iteration cap.
#' @param outer.tol,outer.maxit `(beta, theta)` outer-loop convergence
#'   tolerance and iteration cap (only used when `dist = "negbin"`).
#' @param theta.tol,theta.maxit tolerance and iteration cap for the inner
#'   theta update (only used when `dist = "negbin"`).
#' @param model logical; keep the model frame on the result.
#' @param y,x logical; retain the response and, for formula fits, design
#'   matrices. Matrix fits retain their supplied designs for prediction;
#'   source fits retain the reader instead of observation-level data.
#'
#' @returns A list of class `c("fastglm_hurdle", "fastglm")` with elements
#'   `coefficients` (a list with `count` and `zero`), `se` (likewise),
#'   `vcov` (block-diagonal), `loglik`, `loglik_count`, `loglik_zero`,
#'   `theta`, `SE.theta`, `iter_*`, and `converged`.
#'
#' @examples
#' set.seed(1)
#' n <- 300
#' x1 <- rnorm(n); x2 <- rnorm(n)
#' eta_count <- 1.0 + 0.4*x1 - 0.2*x2
#' lam <- exp(eta_count)
#' y_count <- rpois(n, lam)
#' eta_zero  <- -0.5 + 0.6*x1
#' p_pos     <- plogis(eta_zero)
#' is_pos    <- rbinom(n, 1, p_pos)
#' # zero-truncated: resample any rare zero from positive Poisson
#' for (i in seq_len(n)) {
#'   while (y_count[i] == 0) y_count[i] <- rpois(1, lam[i])
#' }
#' y <- ifelse(is_pos == 1, y_count, 0)
#' df <- data.frame(y = y, x1 = x1, x2 = x2)
#' fit <- fastglm_hurdle(y ~ x1 + x2, data = df, dist = "poisson")
#' coef(fit, model = "count")
#' coef(fit, model = "zero")
#'
#' @export
fastglm_hurdle <- function(formula, data, subset, na.action,
                           weights, offset,
                           dist        = c("poisson", "negbin"),
                           zero.dist   = "binomial",
                           link        = c("logit", "probit", "cloglog", "log"),
                           init.theta  = NULL,
                           tol         = 1e-8,
                           maxit       = 100L,
                           outer.tol   = 1e-7,
                           outer.maxit = 50L,
                           theta.tol   = 1e-8,
                           theta.maxit = 100L,
                           model = TRUE, y = TRUE, x = FALSE,
                           response = NULL, z = NULL, method = 2L, block.size = 8192L)
{
    if (!requireNamespace("Formula", quietly = TRUE))
        stop("Package 'Formula' is required for fastglm_hurdle(); install it.",
             call. = FALSE)

    dist      <- match.arg(dist)
    link      <- match.arg(link)
    zero.dist <- match.arg(zero.dist, choices = "binomial")
    if(is.matrix(formula) || inherits(formula,c("dgCMatrix","fastglm_source"))) {
        if(!missing(subset) || !missing(na.action))
            stop("subset and na.action apply only to formula fits; prepare matrix or source rows first",call.=FALSE)
        out <- .count_backend_hurdle(formula,response,z,
            if(missing(weights)) NULL else weights,if(missing(offset)) NULL else offset,
            dist,link,init.theta,method,tol,maxit,outer.tol,outer.maxit,theta.tol,theta.maxit,block.size)
        out$call <- match.call()
        if(!isTRUE(y)) out$y <- NULL
        return(out)
    }

    # ---- 1. Build the model frame using Formula's two-RHS support ------------
    cl <- match.call()
    if (missing(data)) data <- environment(formula)
    f  <- Formula::Formula(formula)
    has_two_rhs <- length(f)[2] >= 2

    mf <- match.call(expand.dots = FALSE)
    mf$formula <- f
    mf$dist <- mf$zero.dist <- mf$link <- mf$init.theta <- NULL
    mf$tol  <- mf$maxit  <- mf$outer.tol <- mf$outer.maxit <- NULL
    mf$theta.tol <- mf$theta.maxit <- NULL
    mf$model <- mf$y <- mf$x <- NULL
    mf$response <- mf$z <- mf$method <- mf$block.size <- NULL
    mf[[1L]] <- as.name("model.frame")
    mf <- eval(mf, parent.frame())

    mt_count <- terms(f, data = data, rhs = 1L)
    mt_zero  <- if (has_two_rhs) terms(f, data = data, rhs = 2L)
                else             mt_count

    Y_resp <- model.response(mf, "numeric")
    if (!is.numeric(Y_resp) || any(Y_resp < 0))
        stop("'y' must be non-negative.", call. = FALSE)
    n <- length(Y_resp)

    X_count <- model.matrix(mt_count, mf)
    Z_zero  <- model.matrix(mt_zero,  mf)

    if (ncol(X_count) == 0L || ncol(Z_zero) == 0L) stop("both components require at least one coefficient", call. = FALSE)

    # weights / offset
    wt <- model.weights(mf)
    if (is.null(wt)) wt <- rep(1, n)
    if (!is.numeric(wt) || any(wt < 0))
        stop("'weights' must be non-negative.", call. = FALSE)

    offsets <- .component_offsets(mf, list(count = mt_count, zero = mt_zero))
    off_count <- offsets$count; off_zero <- offsets$zero
    .validate_glm_data(X_count, Y_resp, wt, off_count)
    .validate_glm_data(Z_zero, Y_resp, wt, off_zero)
    if (any(Y_resp != floor(Y_resp))) stop("count responses must be integers", call. = FALSE)
    .check_iterations(tol, maxit)
    .check_iterations(outer.tol, outer.maxit)
    .check_iterations(theta.tol, theta.maxit)

    # ---- 2. Map link to fam_code & family-fns for the zero/non-zero binomial -
    zero_family <- switch(link,
                          "logit"   = binomial("logit"),
                          "probit"  = binomial("probit"),
                          "cloglog" = binomial("cloglog"),
                          "log"     = binomial("log"))
    zero_fam_code <- switch(link,
                            "logit"   = 3L,
                            "probit"  = 4L,
                            "cloglog" = 5L,
                            "log"     = 6L)

    dist_code <- switch(dist, "poisson" = 0L, "negbin" = 1L)

    init_th <- if (!is.null(init.theta)) {
        if (!is.numeric(init.theta) || length(init.theta) != 1L ||
            !is.finite(init.theta) || init.theta <= 0)
            stop("'init.theta' must be a positive finite scalar.", call. = FALSE)
        init.theta
    } else {
        -1
    }

    # ---- 3. Dispatch into C++ ------------------------------------------------
    res <- fit_glm_hurdle(
        x_count           = X_count,
        z_zero            = Z_zero,
        y                 = as.numeric(Y_resp),
        weights           = wt,
        offset_count      = off_count,
        offset_zero       = off_zero,
        dist_code         = dist_code,
        zero_fam_code     = zero_fam_code,
        init_theta        = as.double(init_th),
        tol               = as.double(tol),
        maxit             = as.integer(maxit),
        outer_tol         = as.double(outer.tol),
        outer_maxit       = as.integer(outer.maxit),
        theta_tol         = as.double(theta.tol),
        theta_maxit       = as.integer(theta.maxit),
        var_fun_zero      = zero_family$variance,
        mu_eta_fun_zero   = zero_family$mu.eta,
        linkinv_fun_zero  = zero_family$linkinv,
        dev_resids_fun_zero = zero_family$dev.resids,
        valideta_fun_zero = zero_family$valideta,
        validmu_fun_zero  = zero_family$validmu)

    # ---- 4. Cosmetic post-processing -----------------------------------------
    cn_count <- colnames(X_count); if (is.null(cn_count)) cn_count <- paste0("xc", seq_len(ncol(X_count)))
    cn_zero  <- colnames(Z_zero);  if (is.null(cn_zero))  cn_zero  <- paste0("xz", seq_len(ncol(Z_zero)))
    names(res$coefficients_count) <- cn_count
    names(res$coefficients_zero)  <- cn_zero
    names(res$se_count) <- cn_count
    names(res$se_zero)  <- cn_zero
    if (!is.null(res$vcov_count)) dimnames(res$vcov_count) <- list(cn_count, cn_count)
    if (!is.null(res$vcov_zero))  dimnames(res$vcov_zero)  <- list(cn_zero,  cn_zero)

    coefficients <- list(count = res$coefficients_count,
                         zero  = res$coefficients_zero)
    se           <- list(count = res$se_count, zero = res$se_zero)

    p_c <- length(coefficients$count); p_z <- length(coefficients$zero)
    pp  <- p_c + p_z
    vcov_full <- matrix(0, pp, pp)
    rownames(vcov_full) <- colnames(vcov_full) <-
        c(paste0("count_", cn_count), paste0("zero_", cn_zero))
    vcov_full[seq_len(p_c), seq_len(p_c)] <- res$vcov_count
    vcov_full[p_c + seq_len(p_z), p_c + seq_len(p_z)] <- res$vcov_zero

    out <- list(
        coefficients   = coefficients,
        se             = se,
        vcov           = vcov_full,
        loglik         = res$loglik,
        loglik_count   = res$loglik_count,
        loglik_zero    = res$loglik_zero,
        theta          = res$theta,
        SE.theta       = res[["SE.theta"]],
        iter_count     = res$iter_count,
        iter_zero      = res$iter_zero,
        outer_iter     = res$outer_iter,
        theta_iter     = res$theta_iter,
        converged      = res$converged,
        n              = res$n,
        n_positive     = res$n_positive,
        df.residual    = sum(wt > 0) - res$df,
        dist           = dist,
        link           = link,
        zero.dist      = zero.dist,
        formula        = f,
        terms          = list(count = mt_count, zero = mt_zero),
        assign         = list(count=attr(X_count,"assign"),zero=attr(Z_zero,"assign")),
        levels         = lapply(mf[vapply(mf, is.factor, logical(1))], levels),
        na.action      = attr(mf, "na.action"),
        contrasts      = list(count = attr(X_count, "contrasts"),
                              zero  = attr(Z_zero,  "contrasts")),
        weights        = wt,
        offset         = list(count = off_count, zero = off_zero),
        eta_count      = res$eta_count,
        mu_count_truncated = res$mu_count_truncated,
        lambda         = res$lambda,
        eta_zero       = res$eta_zero,
        mu_zero        = res$mu_zero,
        call           = cl
    )
    out$vcov.full <- out$vcov
    out$theta.estimated <- dist == "negbin"
    out$theta.boundary <- dist == "negbin" && (out$theta<=1e-6*(1+theta.tol) || out$theta>=1e8/(1+theta.tol))
    out$refit.controls <- list(dist=dist,link=link,tol=tol,maxit=maxit,outer.tol=outer.tol,
        outer.maxit=outer.maxit,theta.tol=theta.tol,theta.maxit=theta.maxit,method=method,block.size=block.size)
    if (dist == "negbin") {
        jj <- c(seq_len(p_c), pp + 1L)
        out$vcov.full <- matrix(0, pp + 1L, pp + 1L)
        out$vcov.full[seq_len(pp), seq_len(pp)] <- out$vcov
        out$vcov.full[jj, jj] <- res$vcov_count_joint
        dimnames(out$vcov.full) <- rep(list(c(names(coef.fastglm_hurdle(out)), "theta")), 2L)
    }
    if(out$theta.boundary) {
        reader <- .count_backend_reader(X_count,Y_resp,Z_zero,wt,offsets,block.size)
        conditional <- .count_backend_joint(reader,out$coefficients$count,out$theta,
                                             truncated=TRUE,theta.estimated=FALSE)
        out$loglik_count <- .count_backend_count_summary(reader,out$coefficients$count,out$theta,TRUE)$loglik
        out$loglik <- out$loglik_count+out$loglik_zero
        out$vcov[seq_len(p_c),seq_len(p_c)] <- conditional
        out$vcov.full[seq_len(pp),seq_len(pp)] <- out$vcov
        out$vcov.full[pp+1L,] <- out$vcov.full[,pp+1L] <- NA_real_
        out$SE.theta <- NA_real_;out$se$count <- sqrt(diag(conditional))
    }
    if (!out$converged) warning("two-component model did not converge", call. = FALSE)
    if (any(!is.finite(diag(out$vcov)))) warning("observed information is not positive definite; covariance is unavailable", call. = FALSE)
    if (model) out$model <- mf
    if (y)     out$y     <- Y_resp
    if (x)     out$x     <- list(count = X_count, zero = Z_zero)
    class(out) <- "fastglm_hurdle"
    out
}

#' @export
coef.fastglm_hurdle <- function(object, model = c("full", "count", "zero"), ...)
{
    model <- match.arg(model)
    if (model == "full") {
        cn <- names(object$coefficients$count)
        zn <- names(object$coefficients$zero)
        out <- c(object$coefficients$count, object$coefficients$zero)
        names(out) <- c(paste0("count_", cn), paste0("zero_", zn))
        return(out)
    }
    object$coefficients[[model]]
}

#' @export
vcov.fastglm_hurdle <- function(object, model = c("full", "count", "zero"), full = FALSE,
                               type=NULL,cluster=NULL,...)
{
    model <- match.arg(model)
    if(!is.null(type) || !is.null(cluster)) {
        v <- fastglm_vcov(object,if(is.null(type)) "HC0" else type,cluster=cluster,full=full,...)
        if(model=="full") return(v)
        p <- length(object$coefficients$count)
        i <- if(model=="count") seq_len(p) else p+seq_along(object$coefficients$zero)
        return(v[i,i,drop=FALSE])
    }
    if (full) return(object$vcov.full)
    if (model == "full") return(object$vcov)
    p_c <- length(object$coefficients$count)
    p_z <- length(object$coefficients$zero)
    if (model == "count") return(object$vcov[seq_len(p_c), seq_len(p_c), drop = FALSE])
    object$vcov[p_c + seq_len(p_z), p_c + seq_len(p_z), drop = FALSE]
}

#' @export
logLik.fastglm_hurdle <- function(object, ...)
{
    df <- length(object$coefficients$count) + length(object$coefficients$zero) +
          (object$dist == "negbin")
    structure(object$loglik, df = df, nobs = nobs(object),
              class = "logLik")
}

#' @exportS3Method print fastglm_hurdle
print.fastglm_hurdle <- function(x, digits = max(3, getOption("digits") - 3), ...)
{
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header("Hurdle regression",x$call)
    .fastglm_print_field("Count distribution",x$dist)
    .fastglm_print_field("Binary link",x$link)
    .fastglm_print_coefficients(x$coefficients$count,"Count coefficients",digits,...)
    .fastglm_print_coefficients(x$coefficients$zero,"Occurrence coefficients",digits,...)
    cat("\n")
    if(x$dist=="negbin") .fastglm_print_table(data.frame(Theta=x$theta,`Std. Error`=x$SE.theta,check.names=FALSE),digits)
    .fastglm_print_field("Log-likelihood",format(x$loglik,digits=digits))
    .fastglm_print_convergence(x$converged,x$em_iter)
    invisible(x)
}
