#' Zero-inflated Poisson / Negative-Binomial regression
#'
#' Fits a zero-inflated count model: a binary inflation component combined
#' with a Poisson or NB count component, where the count component is the
#' *original* (non-truncated) distribution.  Excess zeros above what the
#' count model implies are absorbed by the inflation component.
#'
#' Estimation uses an EM algorithm with all numerical loops in C++.  The
#' E-step computes the posterior `tau_i = P(Z_i = 1 | y_i)` analytically;
#' M-steps fit the inflation logit/probit/cloglog/log and the Poisson/NB
#' count regression via the existing native fastglm IRLS (with
#' weights `w_i (1 - tau_i)` on the count side).  For NB, an inner Brent
#' MLE re-estimates `theta` after the count-side beta step.  Final vcov
#' comes from the numerical Jacobian of the analytical observed score at
#' the EM fixed point (block-structured for `(gamma, beta, theta)`).
#'
#' This is the fastglm analogue of [pscl::zeroinfl()] -- coefficients
#' agree to 1e-5 on standard datasets (slightly looser than hurdle to
#' allow for the EM iteration), with all numerical loops in C++.
#'
#' @param formula a [Formula::Formula] of the form `y ~ x1 + x2 | z1 + z2`
#'   where the right-hand side after `|` specifies the inflation design.
#'   If `|` is absent, the same RHS is used for both parts.
#' @param data optional data frame / environment.
#' @param subset,na.action standard model-frame arguments.
#' @param weights optional non-negative prior weights.
#' @param offset optional vector or matrix.  If a 2-column matrix, columns
#'   are taken as `(count_offset, zero_offset)`.
#' @param dist count component distribution: `"poisson"` or `"negbin"`.
#' @param link character, link for the inflation binomial.  One of
#'   `"logit"` (default), `"probit"`, `"cloglog"`, `"log"`.
#' @param init.theta optional positive scalar starting value for the NB
#'   dispersion.  If `NULL`, a method-of-moments pilot is used.
#' @param em.tol,em.maxit EM convergence tolerance and iteration cap.
#' @param tol,maxit IRLS-loop convergence tolerance and iteration cap
#'   (used inside each M-step).
#' @param theta.tol,theta.maxit Brent-loop tolerance / iteration cap for
#'   the inner theta MLE (only used when `dist = "negbin"`).
#' @param model logical; keep the model frame on the result.
#' @param y,x logical; keep the response / design matrices on the result.
#'
#' @returns A list of class `c("fastglm_zi", "fastglm")` with elements
#'   `coefficients` (a list with `count` and `zero`), `se` (likewise),
#'   `vcov` (full, including theta if NB), `loglik`, `theta`, `SE.theta`,
#'   `tau` (posterior P(Z=1|y)), `em_iter`, `converged`.
#'
#' @examples
#' set.seed(1)
#' n <- 400
#' x  <- rnorm(n)
#' eta_count <- 0.8 + 0.4*x
#' lam <- exp(eta_count)
#' eta_zero <- -0.6 + 0.5*x
#' p_inflate <- plogis(eta_zero)
#' z <- rbinom(n, 1, p_inflate)
#' y <- ifelse(z == 1, 0, rpois(n, lam))
#' df <- data.frame(y = y, x = x)
#' fit <- fastglm_zi(y ~ x, data = df, dist = "poisson")
#' coef(fit, model = "count")
#' coef(fit, model = "zero")
#'
#' @export
fastglm_zi <- function(formula, data, subset, na.action,
                       weights, offset,
                       dist        = c("poisson", "negbin"),
                       link        = c("logit", "probit", "cloglog", "log"),
                       init.theta  = NULL,
                       em.tol      = 1e-8,
                       em.maxit    = 100L,
                       tol         = 1e-9,
                       maxit       = 100L,
                       theta.tol   = 1e-8,
                       theta.maxit = 100L,
                       model = TRUE, y = TRUE, x = FALSE)
{
    if (!requireNamespace("Formula", quietly = TRUE))
        stop("Package 'Formula' is required for fastglm_zi(); install it.",
             call. = FALSE)

    dist <- match.arg(dist)
    link <- match.arg(link)

    cl <- match.call()
    if (missing(data)) data <- environment(formula)
    f  <- Formula::Formula(formula)
    has_two_rhs <- length(f)[2] >= 2

    mf <- match.call(expand.dots = FALSE)
    mf$formula <- f
    mf$dist <- mf$link <- mf$init.theta <- NULL
    mf$em.tol <- mf$em.maxit <- NULL
    mf$tol <- mf$maxit <- mf$theta.tol <- mf$theta.maxit <- NULL
    mf$model <- mf$y <- mf$x <- NULL
    mf[[1L]] <- as.name("model.frame")
    mf <- eval(mf, parent.frame())

    mt_count <- terms(f, data = data, rhs = 1L)
    mt_zero  <- if (has_two_rhs) terms(f, data = data, rhs = 2L) else mt_count

    Y_resp <- model.response(mf, "numeric")
    if (!is.numeric(Y_resp) || any(Y_resp < 0))
        stop("'y' must be non-negative.", call. = FALSE)
    n <- length(Y_resp)

    X_count <- model.matrix(mt_count, mf)
    Z_zero  <- model.matrix(mt_zero,  mf)

    wt <- model.weights(mf)
    if (is.null(wt)) wt <- rep(1, n)
    if (!is.numeric(wt) || any(wt < 0))
        stop("'weights' must be non-negative.", call. = FALSE)

    off <- model.offset(mf)
    if (is.null(off)) {
        off_count <- rep(0, n); off_zero <- rep(0, n)
    } else if (is.matrix(off) && ncol(off) == 2L) {
        off_count <- as.numeric(off[, 1L]); off_zero <- as.numeric(off[, 2L])
    } else {
        off_count <- as.numeric(off);       off_zero <- as.numeric(off)
    }

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

    count_family <- if (dist == "poisson") poisson("log") else
                    # MASS::negative.binomial(theta) provides the family
                    # callbacks we need; use a dummy theta -- the C++ side
                    # honours fam_code/fam_params directly so the value here
                    # only matters as a fallback.
                    {
                        if (!requireNamespace("MASS", quietly = TRUE))
                            stop("Package 'MASS' is required for dist='negbin' fastglm_zi().",
                                 call. = FALSE)
                        MASS::negative.binomial(theta = 1)
                    }

    init_th <- if (!is.null(init.theta)) {
        if (!is.numeric(init.theta) || length(init.theta) != 1L ||
            !is.finite(init.theta) || init.theta <= 0)
            stop("'init.theta' must be a positive finite scalar.", call. = FALSE)
        init.theta
    } else {
        -1
    }

    res <- fit_glm_zi(
        x_count            = X_count,
        z_zero             = Z_zero,
        y                  = as.numeric(Y_resp),
        weights            = wt,
        offset_count       = off_count,
        offset_zero        = off_zero,
        dist_code          = dist_code,
        zero_fam_code      = zero_fam_code,
        init_theta         = as.double(init_th),
        tol                = as.double(tol),
        maxit              = as.integer(maxit),
        em_tol             = as.double(em.tol),
        em_maxit           = as.integer(em.maxit),
        theta_tol          = as.double(theta.tol),
        theta_maxit        = as.integer(theta.maxit),
        var_fun_zero       = zero_family$variance,
        mu_eta_fun_zero    = zero_family$mu.eta,
        linkinv_fun_zero   = zero_family$linkinv,
        dev_resids_fun_zero= zero_family$dev.resids,
        valideta_fun_zero  = zero_family$valideta,
        validmu_fun_zero   = zero_family$validmu,
        var_fun_count      = count_family$variance,
        mu_eta_fun_count   = count_family$mu.eta,
        linkinv_fun_count  = count_family$linkinv,
        dev_resids_fun_count = count_family$dev.resids,
        valideta_fun_count = count_family$valideta,
        validmu_fun_count  = count_family$validmu)

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
    pp  <- p_c + p_z + (dist == "negbin")
    rn <- c(paste0("count_", cn_count), paste0("zero_", cn_zero))
    if (dist == "negbin") rn <- c(rn, "theta")
    if (NROW(res$vcov_full) == pp) {
        rownames(res$vcov_full) <- colnames(res$vcov_full) <- rn
    }

    out <- list(
        coefficients   = coefficients,
        se             = se,
        vcov           = res$vcov_full,
        loglik         = res$loglik,
        theta          = res$theta,
        SE.theta       = res[["SE.theta"]],
        tau            = res$tau,
        em_iter        = res$em_iter,
        converged      = res$converged,
        n              = res$n,
        df.residual    = n - res$df,
        dist           = dist,
        link           = link,
        formula        = f,
        terms          = list(count = mt_count, zero = mt_zero),
        levels         = .getXlevels(mt_count, mf),
        contrasts      = list(count = attr(X_count, "contrasts"),
                              zero  = attr(Z_zero,  "contrasts")),
        weights        = wt,
        offset         = list(count = off_count, zero = off_zero),
        eta_count      = res$eta_count,
        mu_count       = res$mu_count,
        eta_zero       = res$eta_zero,
        call           = cl
    )
    if (model) out$model <- mf
    if (y)     out$y     <- Y_resp
    if (x)     out$x     <- list(count = X_count, zero = Z_zero)
    class(out) <- c("fastglm_zi", "fastglm")
    out
}

#' @export
coef.fastglm_zi <- function(object, model = c("full", "count", "zero"), ...)
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
vcov.fastglm_zi <- function(object, model = c("full", "count", "zero"), ...)
{
    model <- match.arg(model)
    if (model == "full") return(object$vcov)
    p_c <- length(object$coefficients$count)
    p_z <- length(object$coefficients$zero)
    if (model == "count") return(object$vcov[seq_len(p_c), seq_len(p_c), drop = FALSE])
    object$vcov[p_c + seq_len(p_z), p_c + seq_len(p_z), drop = FALSE]
}

#' @export
logLik.fastglm_zi <- function(object, ...)
{
    df <- length(object$coefficients$count) + length(object$coefficients$zero) +
          (object$dist == "negbin")
    structure(object$loglik, df = df, nobs = object$n,
              class = "logLik")
}

#' @export
print.fastglm_zi <- function(x, digits = max(3, getOption("digits") - 3), ...)
{
    cat("\nfastglm_zi:  zero-inflated ", x$dist, " (zero link = ", x$link,
        ")\n", sep = "")
    cat("Call:  "); print(x$call)
    cat("\nCount model coefficients (", x$dist, "):\n", sep = "")
    print(format(x$coefficients$count, digits = digits), quote = FALSE)
    cat("\nZero-inflation coefficients (binomial,", x$link, "):\n")
    print(format(x$coefficients$zero, digits = digits), quote = FALSE)
    if (x$dist == "negbin")
        cat("\nTheta:", format(x$theta, digits = digits),
            "  SE:", format(x[["SE.theta"]], digits = digits), "\n")
    cat("Log-likelihood:", format(x$loglik, digits = digits),
        "on", length(x$coefficients$count) + length(x$coefficients$zero) +
             (x$dist == "negbin"), "df  (EM iter:", x$em_iter, ")\n")
    invisible(x)
}
