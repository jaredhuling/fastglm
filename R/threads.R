#' Number of threads used by fastglm
#'
#' Reports or sets the number of OpenMP threads that fastglm uses for its
#' compiled computations in this session. Results do not depend on the
#' number of threads.
#'
#' The default is half the physical cores (half the logical cores when the
#' physical count is unavailable), at most 8 and at least 1. It is reduced to
#' the values of the environment variables \code{OMP_THREAD_LIMIT} and
#' \code{OMP_NUM_THREADS} when they are set, and to at most 2 when
#' \code{_R_CHECK_LIMIT_CORES_} is set to anything other than \code{"false"}, as it is
#' during \code{R CMD check}.
#'
#' A requested count is reduced to the number of logical cores and, when
#' \code{_R_CHECK_LIMIT_CORES_} is set, to at most 2. When the package was built
#' without OpenMP, fastglm always uses 1 thread and a request for more warns
#' once per session.
#'
#' The setting is stored in \code{options(fastglm.threads = )}, which can also be
#' set before the package is loaded.
#'
#' A fit holds its thread setting for the whole call and then restores the
#' session setting, so a call to \code{fastglm_threads()} made during a fit (for
#' example from a \code{\link{fastglm_source}} callback) is undone when the fit
#' returns.
#'
#' @param n \code{NULL} to report the current setting, or a single positive whole
#'   number of threads to use.
#'
#' @return A list with elements \code{threads} (the effective number of threads),
#'   \code{default} (the default number of threads), \code{openmp} (whether the package
#'   was built with OpenMP), and \code{logical.cores} (the number of logical
#'   cores). The list is returned visibly when \code{n} is \code{NULL} and invisibly
#'   when \code{n} sets the count.
#'
#' @examples
#' fastglm_threads()
#' old <- fastglm_threads(1)
#' fastglm_threads(old$threads)
#' @export
fastglm_threads <- function(n = NULL)
{
    if (is.null(n))
        return(.fastglm_threads_info(.fastglm_resolve_threads(NULL)))
    eff <- .fastglm_resolve_threads(n, name = "n")
    options(fastglm.threads = eff)
    fastglm_set_threads_cpp(eff)
    invisible(.fastglm_threads_info(eff))
}

.fastglm_threads_info <- function(threads)
{
    list(threads       = threads,
         default       = if (fastglm_has_openmp_cpp()) .fastglm_default_threads() else 1L,
         openmp        = fastglm_has_openmp_cpp(),
         logical.cores = .fastglm_logical_cores())
}

.fastglm_state <- new.env(parent = emptyenv())

.fastglm_logical_cores <- function()
{
    n <- if (fastglm_has_openmp_cpp()) fastglm_logical_cores_cpp()
         else parallel::detectCores(logical = TRUE)
    n <- suppressWarnings(as.integer(n))
    if (length(n) != 1L || is.na(n) || n < 1L) 1L else n
}

## A single positive whole number, or NA_integer_ when x is not one.
.fastglm_as_count <- function(x)
{
    if (is.character(x)) x <- strsplit(x, ",", fixed = TRUE)[[1L]][1L]
    x <- suppressWarnings(as.numeric(x))
    if (length(x) != 1L || is.na(x) || !is.finite(x) || x < 1 || x != floor(x))
        return(NA_integer_)
    as.integer(min(x, .Machine$integer.max))
}

## Applies the _R_CHECK_LIMIT_CORES_ cap (at most 2 threads) to n.
.fastglm_check_cap <- function(n, limit = Sys.getenv("_R_CHECK_LIMIT_CORES_", ""))
{
    n <- as.integer(n)
    if (length(limit) == 1L && !is.na(limit) && nzchar(limit) &&
        !identical(tolower(limit), "false"))
        n <- min(n, 2L)
    n
}

## parallel::detectCores(), cached for the session: on some platforms it
## starts a subprocess (about 10 ms), which would otherwise be paid by every
## fit that uses the default thread count.
.fastglm_detected_cores <- function(logical = TRUE)
{
    key <- if (logical) "cores_logical" else "cores_physical"
    if (is.null(.fastglm_state[[key]]))
        assign(key, parallel::detectCores(logical = logical), envir = .fastglm_state)
    .fastglm_state[[key]]
}

## The default thread count. The inputs are arguments so that the rule can be
## tested; env is a named list of environment variables, where a missing or
## empty entry means the variable is unset.
.fastglm_default_threads <- function(
    physical = .fastglm_detected_cores(logical = FALSE),
    logical  = .fastglm_detected_cores(logical = TRUE),
    env      = as.list(Sys.getenv(c("OMP_THREAD_LIMIT", "OMP_NUM_THREADS",
                                    "_R_CHECK_LIMIT_CORES_"), unset = "")))
{
    physical <- .fastglm_as_count(physical)
    logical  <- .fastglm_as_count(logical)
    base <- if (!is.na(physical)) physical else logical
    n <- if (is.na(base)) 1L else as.integer(base %/% 2L)
    n <- max(1L, min(8L, n))
    for (var in c("OMP_THREAD_LIMIT", "OMP_NUM_THREADS")) {
        lim <- .fastglm_as_count(if (is.null(env[[var]])) NA else env[[var]])
        if (!is.na(lim)) n <- min(n, lim)
    }
    chk <- env[["_R_CHECK_LIMIT_CORES_"]]
    n <- .fastglm_check_cap(n, if (is.null(chk)) "" else chk)
    if (!is.na(logical)) n <- min(n, logical)
    max(1L, as.integer(n))
}

## The effective thread count for a request. NULL means the session setting
## (options(fastglm.threads), else the default). An explicit request must be a
## single positive whole number; it is reduced to the logical cores and by the
## _R_CHECK_LIMIT_CORES_ cap, and is 1 on a serial build (warning once per
## session when more was requested).
.fastglm_resolve_threads <- function(threads = NULL, name = "threads")
{
    explicit <- !is.null(threads)
    if (explicit) {
        n <- if (is.numeric(threads)) .fastglm_as_count(threads) else NA_integer_
        if (is.na(n))
            stop(sprintf("'%s' must be a positive integer", name), call. = FALSE)
    } else {
        ## A malformed option falls back to the default rather than erroring.
        opt <- getOption("fastglm.threads")
        n <- if (is.null(opt)) NA_integer_ else .fastglm_as_count(opt)
        if (is.na(n)) n <- .fastglm_default_threads()
    }
    if (!fastglm_has_openmp_cpp()) {
        if (explicit && n > 1L && !isTRUE(.fastglm_state$warned_serial)) {
            .fastglm_state$warned_serial <- TRUE
            warning("fastglm was built without OpenMP; using 1 thread", call. = FALSE)
        }
        return(1L)
    }
    n <- min(n, .fastglm_logical_cores())
    max(1L, .fastglm_check_cap(n))
}

## Sets the C++ thread setting for the duration of a call and returns a
## function that restores the previous setting. Callers use
## restore <- .fastglm_with_threads(threads); on.exit(restore(), add = TRUE)
.fastglm_with_threads <- function(threads = NULL)
{
    eff <- .fastglm_resolve_threads(threads)
    prev <- fastglm_set_threads_cpp(eff)
    function() invisible(fastglm_set_threads_cpp(prev))
}

## Sets the thread count for everything an entry point runs: the C++ setting
## and options(fastglm.threads), so that fits it calls with threads = NULL
## (pilot fits, component fits, refits) use the same count. Returns a list
## with the effective count (threads) and a function (restore) that restores
## both settings. Callers use
##   scope <- .fastglm_scope_threads(threads); on.exit(scope$restore(), add = TRUE)
## and keep the scope until every pass of the fit, including any finishing
## pass, has run.
.fastglm_scope_threads <- function(threads = NULL)
{
    eff <- .fastglm_resolve_threads(threads)
    old <- options(fastglm.threads = eff)
    prev <- fastglm_set_threads_cpp(eff)
    list(threads = eff,
         restore = function() {
             fastglm_set_threads_cpp(prev)
             options(old)
             invisible(NULL)
         })
}

## Wraps fun so that it first sets one thread for the session; used for
## forked worker processes (parallel::mclapply), which inherit the parent's
## setting. The parent's setting is not changed.
.fastglm_one_thread <- function(fun)
{
    force(fun)
    function(...) {
        fastglm_threads(1L)
        fun(...)
    }
}
