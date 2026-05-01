#' Negative binomial family with known dispersion
#'
#' A built-in negative-binomial family for use with [fastglm()] and
#' [fastglmPure()] when `theta` (the NB2 dispersion) is known. Equivalent to
#' `MASS::negative.binomial(theta, link)` but without taking a hard dependency
#' on the **MASS** package. The variance is `mu + mu^2 / theta`; as
#' `theta -> Inf` it reduces to Poisson.
#'
#' For joint estimation of `theta` together with `beta`, use the dedicated
#' (forthcoming) `fastglm_nb()` entry point. The `negbin()` family here is
#' intended for use cases where `theta` has been pre-specified or estimated
#' separately.
#'
#' @param theta the (known) dispersion parameter; must be positive.
#' @param link character; one of `"log"` (default), `"sqrt"`, or `"identity"`.
#'
#' @returns A `family` object with class `"family"` and
#'   `family$family == "Negative Binomial(theta)"`. The object also carries
#'   the slot `family$theta`, which `family_code()` and the dispatch layer
#'   use to detect the native NB fast path.
#'
#' @examples
#' set.seed(1)
#' n <- 500
#' x  <- cbind(1, matrix(rnorm(n * 2), n, 2))
#' mu <- exp(x %*% c(0.3, 0.4, -0.2))
#' y  <- rpois(n, lambda = mu * rgamma(n, shape = 2, rate = 2))   # NB(theta=2)
#'
#' fit <- fastglm(x, y, family = negbin(theta = 2, link = "log"))
#' coef(fit)
#'
#' @export
negbin <- function(theta, link = "log") {
    if (!is.numeric(theta) || length(theta) != 1L || !is.finite(theta) || theta <= 0)
        stop("'theta' must be a positive finite scalar.", call. = FALSE)
    link <- match.arg(link, c("log", "sqrt", "identity"))

    # Reuse MASS's negative.binomial machinery if available — its
    # initialize/aic/dev.resids closures are battle-tested.  Otherwise build
    # the family from scratch.  Either way we attach $theta so the dispatch
    # layer in family_code() / family_params() picks up the NB2 fast path.
    if (requireNamespace("MASS", quietly = TRUE)) {
        fam <- MASS::negative.binomial(theta = theta, link = link)
        fam$theta <- theta
        return(fam)
    }

    # ---- Fallback implementation (no MASS) ---------------------------------
    stats <- make.link(link)

    variance   <- function(mu)  mu + mu^2 / theta
    validmu    <- function(mu)  all(is.finite(mu)) && all(mu > 0)
    dev.resids <- function(y, mu, wt) {
        # 2 * wt * [ y * log(y/mu) - (y + theta) * log((y + theta)/(mu + theta)) ]
        a  <- ifelse(y > 0, y * log(y / mu), 0)
        yt <- y + theta
        mt <- mu + theta
        b  <- yt * log(yt / mt)
        2 * wt * (a - b)
    }
    aic <- function(y, n, mu, wt, dev) {
        # -2 * sum( wt * (lgamma(y + theta) - lgamma(theta) - lgamma(y + 1) +
        #                 theta*log(theta/(theta+mu)) + y*log(mu/(theta+mu))) )
        term1 <- lgamma(y + theta) - lgamma(theta) - lgamma(y + 1)
        term2 <- theta * log(theta / (theta + mu))
        term3 <- ifelse(y > 0, y * log(mu / (theta + mu)), 0)
        -2 * sum(wt * (term1 + term2 + term3))
    }
    initialize <- expression({
        if (any(y < 0)) stop("negative values not allowed for the negative binomial family")
        n       <- rep(1, nobs)
        mustart <- y + (y == 0) / 6
    })
    simfun <- function(object, nsim) {
        ftd <- fitted(object)
        rnbinom(nsim * length(ftd), mu = ftd, size = theta)
    }

    famname <- paste0("Negative Binomial(", format(theta, digits = 4), ")")

    structure(
        list(family     = famname,
             link       = link,
             linkfun    = stats$linkfun,
             linkinv    = stats$linkinv,
             variance   = variance,
             dev.resids = dev.resids,
             aic        = aic,
             mu.eta     = stats$mu.eta,
             initialize = initialize,
             validmu    = validmu,
             valideta   = stats$valideta,
             simulate   = simfun,
             theta      = theta),
        class = "family"
    )
}
