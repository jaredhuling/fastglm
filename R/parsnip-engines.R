#' parsnip engines
#'
#' fastglm provides the engine `"fastglm"` for five model types of the
#' 'parsnip' package, so that fastglm fits can be used in tidymodels
#' workflows, recipes, and resampling. The engines are registered when
#' 'parsnip' is loaded (whether before or after fastglm); 'parsnip' is not
#' required otherwise.
#'
#' @details
#' | Model | Mode | Fitting function | Engine arguments |
#' |---|---|---|---|
#' | `logistic_reg()` | classification | [fastglm()], binomial family | `method`, `firth`, `threads`, `link` |
#' | `linear_reg()` | regression | [fastglm()], gaussian family | `method`, `firth`, `threads`, `link` |
#' | `poisson_reg()` | regression | [fastglm()], poisson family | `method`, `firth`, `threads`, `link` |
#' | `multinom_reg()` | classification | [fastglm_multinom()] | `firth`, `threads` |
#' | `ordinal_reg()` | classification | [fastglm_ordinal()] | `firth`, `threads`; main arguments `ordinal_link`, `odds_link`, `threshold_structure` |
#'
#' Engine arguments are given to `parsnip::set_engine()`, for example
#' `logistic_reg() |> set_engine("fastglm", link = "probit", firth = TRUE)`.
#' `method` is the decomposition of [fastglm()], `firth = TRUE` gives mean
#' bias reduction, and `link` is the link function of the family. Other
#' arguments (for example `tol` or `maxit`) are passed to the fitting
#' function. The `ordinal_link` argument of `ordinal_reg()` accepts
#' `"logistic"` (the default), `"probit"`, and `"cloglog"`; fastglm does not
#' fit the `"loglog"` and `"cauchit"` links. The fits are cumulative-link
#' models with flexible thresholds, so `odds_link` must be
#' `"cumulative_link"` (or unset) and `threshold_structure` must be
#' `"flexible"` (or unset); other values are errors. The penalty and mixture
#' arguments of the model types are not used: the fits are unpenalized (or
#' bias-reduced with `firth = TRUE`).
#'
#' An outcome level with no observations in the training data is an error
#' for `multinom_reg()` and `ordinal_reg()` (the fitting functions do not
#' estimate a category without observations), so in resampling a fold that
#' lacks a level fails with a note. `nnet::multinom()`, in contrast, drops
#' the probability column of such a level.
#'
#' **Threads.** An engine fit uses one thread unless `threads` is given to
#' `set_engine()`. Resampling and tuning ('tune') usually run fits in
#' several worker processes at once, and the default thread count of fastglm
#' in each worker would multiply the number of processes by the number of
#' threads. Pass `threads` explicitly to use more threads for a single fit.
#' Results are identical for every number of threads.
#'
#' **Predictions.** Class probabilities have one column per level of the
#' outcome factor, in the order of its levels, named `.pred_<level>` by
#' 'parsnip'. For `logistic_reg()`, as for the `"glm"` engine, the model is
#' for the probability of the second level, and the predicted class is the
#' second level when that probability is at least 0.5. Regression models
#' predict the mean (`type = "numeric"`), and `type = "raw"` returns the
#' default [predict()] output of the fit. For `linear_reg()` and
#' `logistic_reg()`, `type = "conf_int"` gives Wald intervals for the mean,
#' computed as by the `"glm"` engine: the interval
#' `eta +/- t * se(eta)` on the link scale, with `t` the quantile of the t
#' distribution on the residual degrees of freedom at the requested `level`,
#' transformed by the inverse link (columns `.pred_lower` and `.pred_upper`,
#' or `.pred_lower_<level>` and `.pred_upper_<level>` for classification).
#'
#' The fitted model (the `fit` element of the 'parsnip' model fit, or
#' `parsnip::extract_fit_engine()`) is the fastglm fit, with all of its
#' methods, including [update()], [stats::step()], and [anova()] of an
#' updated fit: the stored call refers to the training data through a
#' private environment (it prints as `data = <environment>$data`), so it can
#' be evaluated again from any frame. The model formula keeps its own
#' environment, so [predict()] with new data that lack a predictor is an
#' error, as for the `"glm"` engine. A family of the 'stats' package is
#' stored in the call as a short call such as
#' `stats::binomial(link = "logit")`. The functions below are the fitting
#' functions called by the
#' engines; they are exported for 'parsnip' and are not meant to be called
#' directly.
#'
#' @param formula,data the model formula and data, supplied by 'parsnip'.
#' @param weights optional case weights, supplied by 'parsnip'.
#' @param family the family name (`"binomial"`, `"gaussian"`, or
#'   `"poisson"`), or a family object.
#' @param link the link function, a name accepted by the family. For
#'   `fastglm_parsnip_ordinal()`, a 'parsnip' ordinal link (`"logistic"`,
#'   `"probit"`, or `"cloglog"`) or a link of [fastglm_ordinal()].
#' @param odds_link,threshold_structure the 'parsnip' arguments of
#'   `ordinal_reg()`; only `"cumulative_link"` and `"flexible"` (or `NULL`)
#'   are supported.
#' @param method the decomposition, as in [fastglm()].
#' @param firth use mean bias reduction.
#' @param threads the number of threads for the fit, 1 by default.
#' @param ... further arguments to the fitting function.
#' @returns The fastglm fit.
#' @examplesIf requireNamespace("parsnip", quietly = TRUE)
#' library(parsnip)
#' set.seed(1)
#' d <- data.frame(x = rnorm(100))
#' d$y <- factor(ifelse(d$x + rnorm(100) > 0, "yes", "no"), levels = c("yes", "no"))
#' fit <- logistic_reg() |> set_engine("fastglm") |> fit(y ~ x, data = d)
#' head(predict(fit, d, type = "prob"))
#' @name fastglm-parsnip
#' @md
#' @keywords internal
NULL

#' @rdname fastglm-parsnip
#' @export
fastglm_parsnip_glm <- function(formula, data, weights = NULL, family = "gaussian",
                                link = NULL, method = 0L, firth = FALSE,
                                threads = 1L, ...) {
    if (is.character(family)) {
        if (length(family) != 1L || !family %in% c("binomial", "gaussian", "poisson"))
            stop("'family' must be \"binomial\", \"gaussian\", or \"poisson\"", call. = FALSE)
        fam <- get(family, envir = asNamespace("stats"), mode = "function")
        family <- if (is.null(link)) fam() else fam(link = link)
    } else if (!is.null(link)) {
        stop("give 'link' only with a family name", call. = FALSE)
    }
    .fastglm_parsnip_eval(quote(fastglm::fastglm), formula, data, weights,
                          c(list(family = .fastglm_parsnip_family_call(family), method = method,
                                 firth = firth, threads = threads), list(...)))
}

# A family of stats as a short call (stats::binomial(link = "probit")), so the
# stored call, and the Call line of a printed workflow, does not hold the
# family object (parsnip's "glm" engine stores `family = stats::binomial`).
# Other family objects are kept as they are.
.fastglm_parsnip_family_call <- function(family) {
    if (!inherits(family, "family")) return(family)
    name <- family$family
    if (!name %in% c("binomial", "gaussian", "poisson", "Gamma", "inverse.gaussian",
                     "quasibinomial", "quasipoisson"))
        return(family)
    cl <- call("::", quote(stats), as.name(name))
    cl <- as.call(c(list(cl), list(link = family$link)))
    rebuilt <- tryCatch(eval(cl), error = function(e) NULL)
    if (is.null(rebuilt) || !identical(rebuilt$family, family$family) || !identical(rebuilt$link, family$link))
        return(family)
    cl
}

#' @rdname fastglm-parsnip
#' @export
fastglm_parsnip_multinom <- function(formula, data, weights = NULL, firth = FALSE,
                                     threads = 1L, ...) {
    args <- .fastglm_parsnip_control(list(...), threads)
    .fastglm_parsnip_eval(quote(fastglm::fastglm_multinom), formula, data, weights,
                          c(list(firth = firth), args))
}

#' @rdname fastglm-parsnip
#' @export
fastglm_parsnip_ordinal <- function(formula, data, weights = NULL, link = "logistic",
                                    odds_link = NULL, threshold_structure = NULL,
                                    firth = FALSE, threads = 1L, ...) {
    if (!is.null(odds_link) && !identical(odds_link, "cumulative_link"))
        stop("the fastglm engine supports only odds_link = \"cumulative_link\"", call. = FALSE)
    if (!is.null(threshold_structure) && !identical(threshold_structure, "flexible"))
        stop("the fastglm engine supports only threshold_structure = \"flexible\"", call. = FALSE)
    if (!is.character(link) || length(link) != 1L || is.na(link))
        stop("'link' must be a single string", call. = FALSE)
    map <- c(logistic = "logit", logit = "logit", probit = "probit", cloglog = "cloglog")
    if (!link %in% names(map))
        stop(sprintf("the fastglm engine does not fit the ordinal link \"%s\"; use \"logistic\", \"probit\", or \"cloglog\"",
                     link), call. = FALSE)
    args <- .fastglm_parsnip_control(list(...), threads)
    .fastglm_parsnip_eval(quote(fastglm::fastglm_ordinal), formula, data, weights,
                          c(list(link = unname(map[link]), firth = firth), args))
}

# The thread count of multinomial and ordinal fits goes in `control`.
.fastglm_parsnip_control <- function(args, threads) {
    control <- if (is.null(args$control)) list() else args$control
    if (!is.list(control)) stop("'control' must be a list", call. = FALSE)
    control$threads <- threads
    args$control <- control
    args
}

# Evaluate a formula fit with `data` bound in a private environment that the
# stored call refers to directly: the call reads
# `fun(formula = y ~ x + z, data = <environment>$data, ...)`, which evaluates
# to the training data from any frame, so update(), step(), and
# anova(update(fit), fit) work (a call `fun(formula, data = data)` would find
# utils::data in the caller's frame). The formula keeps its own environment,
# so, as for a fit made at top level with `data = d`, predict() with new data
# that lack a predictor stops with "object 'x' not found" instead of finding
# the training column. Case weights are a column of the data. A `.` in the
# formula is expanded against the data first.
.fastglm_parsnip_eval <- function(fun, formula, data, weights, args) {
    if ("." %in% all.vars(formula))
        formula <- stats::formula(stats::terms(formula, data = data))
    data <- as.data.frame(data)
    cl <- c(list(fun, formula = formula), args)
    if (!is.null(weights)) {
        data[["..weights.."]] <- as.numeric(weights)
        cl$weights <- quote(..weights..)
    }
    holder <- new.env(parent = emptyenv())
    holder$data <- data
    cl$data <- as.call(list(as.name("$"), holder, as.name("data")))
    parent <- environment(formula)
    if (is.null(parent)) parent <- globalenv()
    eval(as.call(cl), parent)
}

# Prediction post-processing.
.fastglm_parsnip_binary_class <- function(x, object) {
    unname(ifelse(x >= 0.5, object$lvl[2L], object$lvl[1L]))
}

.fastglm_parsnip_binary_prob <- function(x, object) {
    x <- unname(x)
    out <- data.frame(1 - x, x)
    names(out) <- object$lvl
    tibble::as_tibble(out)
}

.fastglm_parsnip_class <- function(x, object) {
    factor(as.character(x), levels = object$lvl)
}

# Wald intervals for the mean, as computed by parsnip's "glm" engine.
.fastglm_parsnip_conf_int_reg <- function(results, object) {
    extras <- object$spec$method$pred$conf_int$extras
    const <- stats::qt((1 - extras$level) / 2, df = object$fit$df.residual, lower.tail = FALSE)
    trans <- object$fit$family$linkinv
    fit <- unname(results$fit); se <- unname(results$se.fit)
    lo <- trans(fit - const * se); hi <- trans(fit + const * se)
    # A decreasing inverse link swaps the endpoints.
    swap <- hi < lo
    out <- data.frame(.pred_lower = ifelse(swap, hi, lo), .pred_upper = ifelse(swap, lo, hi))
    if (isTRUE(extras$std_error)) out$.std_error <- se
    tibble::as_tibble(out)
}

.fastglm_parsnip_conf_int_binary <- function(results, object) {
    extras <- object$spec$method$pred$conf_int$extras
    const <- stats::qt((1 - extras$level) / 2, df = object$fit$df.residual, lower.tail = FALSE)
    trans <- object$fit$family$linkinv
    fit <- unname(results$fit); se <- unname(results$se.fit)
    lo2 <- trans(fit - const * se); hi2 <- trans(fit + const * se)
    out <- data.frame(1 - hi2, 1 - lo2, lo2, hi2)
    names(out) <- c(paste0(".pred_lower_", object$lvl[1L]), paste0(".pred_upper_", object$lvl[1L]),
                    paste0(".pred_lower_", object$lvl[2L]), paste0(".pred_upper_", object$lvl[2L]))
    if (isTRUE(extras$std_error)) out$.std_error <- se
    tibble::as_tibble(out)
}

# Probability matrices (one column per level of the fit) are returned with
# the columns in the order of the outcome levels.
.fastglm_parsnip_prob <- function(x, object) {
    if (is.null(dim(x))) x <- matrix(x, nrow = 1L, dimnames = list(NULL, names(x)))
    if (!all(object$lvl %in% colnames(x)))
        stop("the fitted levels do not match the levels of the outcome", call. = FALSE)
    x <- x[, object$lvl, drop = FALSE]
    out <- as.data.frame(unname(x))
    names(out) <- object$lvl
    tibble::as_tibble(out)
}

.fastglm_parsnip_pred <- function(args, post = NULL) {
    list(pre = NULL, post = post, func = c(fun = "predict"),
         args = c(list(object = quote(object$fit), newdata = quote(new_data)), args))
}

.fastglm_parsnip_encoding <- list(predictor_indicators = "traditional",
                                  compute_intercept = TRUE, remove_intercept = TRUE,
                                  allow_sparse_x = FALSE)

# The engine is registered for a model and mode only when the model is
# defined in parsnip's model environment. Each component (engine, dependency,
# model arguments, fit, encoding, prediction types) is added only when it is
# missing, so a call is idempotent and completes a partial registration.
# Returns FALSE when the model is not defined.
.fastglm_parsnip_engine <- function(model, mode, fit, preds, model.args = NULL) {
    env <- parsnip::get_model_env()
    if (is.null(env[[model]])) return(FALSE)
    has <- function(suffix, type = NULL) {
        tab <- env[[paste0(model, suffix)]]
        if (is.null(tab) || !nrow(tab)) return(FALSE)
        hit <- tab$engine == "fastglm" & tab$mode == mode
        if (!is.null(type)) hit <- hit & tab$type == type
        any(hit)
    }
    if (!has("")) parsnip::set_model_engine(model, mode = mode, eng = "fastglm")
    parsnip::set_dependency(model, eng = "fastglm", pkg = "fastglm", mode = mode)
    for (a in model.args) do.call(parsnip::set_model_arg, c(list(model = model, eng = "fastglm"), a))
    if (!has("_fit"))
        parsnip::set_fit(model = model, eng = "fastglm", mode = mode,
                         value = list(interface = "formula",
                                      protect = c("formula", "data", "weights"),
                                      func = c(pkg = "fastglm", fun = fit$fun),
                                      defaults = fit$defaults))
    if (!has("_encoding"))
        parsnip::set_encoding(model = model, eng = "fastglm", mode = mode,
                              options = .fastglm_parsnip_encoding)
    for (type in names(preds))
        if (!has("_predict", type))
            parsnip::set_pred(model = model, eng = "fastglm", mode = mode, type = type,
                              value = preds[[type]])
    TRUE
}

.fastglm_parsnip_glm_engine <- function(model, family) {
    mode <- if (family == "binomial") "classification" else "regression"
    conf.int <- list(type = "link", se.fit = TRUE)
    preds <- if (family == "binomial") {
        list(class = .fastglm_parsnip_pred(list(type = "response"),
                                           .fastglm_parsnip_binary_class),
             prob = .fastglm_parsnip_pred(list(type = "response"),
                                          .fastglm_parsnip_binary_prob),
             conf_int = .fastglm_parsnip_pred(conf.int, .fastglm_parsnip_conf_int_binary),
             raw = .fastglm_parsnip_pred(list()))
    } else {
        out <- list(numeric = .fastglm_parsnip_pred(list(type = "response"),
                                                    function(x, object) unname(x)))
        # Intervals for linear_reg(), as the "glm" engine provides.
        if (family == "gaussian")
            out$conf_int <- .fastglm_parsnip_pred(conf.int, .fastglm_parsnip_conf_int_reg)
        out$raw <- .fastglm_parsnip_pred(list())
        out
    }
    .fastglm_parsnip_engine(model, mode,
                            fit = list(fun = "fastglm_parsnip_glm",
                                       defaults = list(family = family)),
                            preds = preds)
}

.fastglm_register_multinom <- function() {
    .fastglm_parsnip_engine("multinom_reg", "classification",
        fit = list(fun = "fastglm_parsnip_multinom", defaults = list()),
        preds = list(class = .fastglm_parsnip_pred(list(type = "class"),
                                                   .fastglm_parsnip_class),
                     prob = .fastglm_parsnip_pred(list(type = "probability"),
                                                  .fastglm_parsnip_prob),
                     raw = .fastglm_parsnip_pred(list())))
}

.fastglm_register_ordinal <- function() {
    arg <- function(name, original)
        list(parsnip = name, original = original, func = list(pkg = "dials", fun = name),
             has_submodel = FALSE)
    .fastglm_parsnip_engine("ordinal_reg", "classification",
        fit = list(fun = "fastglm_parsnip_ordinal", defaults = list()),
        preds = list(class = .fastglm_parsnip_pred(list(type = "class"),
                                                   .fastglm_parsnip_class),
                     prob = .fastglm_parsnip_pred(list(type = "probability"),
                                                  .fastglm_parsnip_prob),
                     raw = .fastglm_parsnip_pred(list())),
        model.args = list(arg("ordinal_link", "link"),
                          arg("odds_link", "odds_link"),
                          arg("threshold_structure", "threshold_structure")))
}

.fastglm_register_poisson <- function() .fastglm_parsnip_glm_engine("poisson_reg", "poisson")

# A registration failure is reported as a warning naming the model and never
# stops parsnip (or fastglm) from loading or the other models from
# registering.
.fastglm_parsnip_try <- function(model, fun) {
    tryCatch(fun(), error = function(e) {
        warning(sprintf("fastglm could not register its parsnip engine for %s: %s",
                        model, conditionMessage(e)), call. = FALSE)
        invisible(FALSE)
    })
}

# The model types are defined by parsnip (poisson_reg() since parsnip 1.0.0,
# ordinal_reg() since parsnip 1.5.0). The poissonreg and ordered hooks are
# idempotent fallbacks: ordered requires parsnip 1.5.0 or later, where parsnip
# already defines ordinal_reg().
.fastglm_register_parsnip <- function() {
    if (!isNamespaceLoaded("parsnip")) return(invisible(FALSE))
    .fastglm_parsnip_try("logistic_reg", function() .fastglm_parsnip_glm_engine("logistic_reg", "binomial"))
    .fastglm_parsnip_try("linear_reg", function() .fastglm_parsnip_glm_engine("linear_reg", "gaussian"))
    .fastglm_parsnip_try("poisson_reg", .fastglm_register_poisson)
    .fastglm_parsnip_try("multinom_reg", .fastglm_register_multinom)
    .fastglm_parsnip_try("ordinal_reg", .fastglm_register_ordinal)
    invisible(TRUE)
}

.fastglm_register_poissonreg <- function() {
    if (!isNamespaceLoaded("parsnip")) return(invisible(FALSE))
    .fastglm_parsnip_try("poisson_reg", .fastglm_register_poisson)
    invisible(TRUE)
}

.fastglm_register_ordered <- function() {
    if (!isNamespaceLoaded("parsnip")) return(invisible(FALSE))
    .fastglm_parsnip_try("ordinal_reg", .fastglm_register_ordinal)
    invisible(TRUE)
}

# Register now when the package is loaded, and also whenever it is loaded
# later (a package unloaded and loaded again starts with a fresh model
# environment), as vctrs::s3_register() does.
.fastglm_on_package_load <- function(pkg, fun) {
    setHook(packageEvent(pkg, "onLoad"), function(...) fun())
    if (isNamespaceLoaded(pkg)) fun()
    invisible(NULL)
}
