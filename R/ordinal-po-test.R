# Test selected threshold-specific slope deviations at an ML common-slope null.
# Expected information and ordinary scores use the same frequency normalization.
.ordinal_po_test_columns <- function(object,terms) {
    columns <- names(object$beta)
    if(is.null(terms)) return(list(index=seq_along(columns),terms=if(is.null(object$terms))
        columns else attr(object$terms,"term.labels"),columns=columns))
    if(!is.character(terms) || !length(terms) || anyNA(terms) || any(!nzchar(terms)) || anyDuplicated(terms))
        stop("'po.terms' must contain distinct, nonmissing term or column names",call.=FALSE)
    if(is.null(object$terms)) {
        if(any(!terms %in% columns)) stop("unknown predictor column in 'po.terms'",call.=FALSE)
        index <- which(columns %in% terms)
    } else {
        labels <- attr(object$terms,"term.labels")
        if(any(!terms %in% labels)) stop("unknown formula term in 'po.terms'",call.=FALSE)
        assignment <- object$design.assign
        if(is.null(assignment) || length(assignment)!=length(columns))
            stop("refit this ordinal model to retain the design assignments required by 'po.terms'",call.=FALSE)
        index <- which(assignment %in% match(terms,labels))
    }
    if(!length(index)) stop("'po.terms' selects no location predictors",call.=FALSE)
    list(index=index,terms=terms,columns=columns[index])
}

.ordinal_po_test_control <- function(control) {
    if(!is.list(control) || (length(control) && (is.null(names(control)) || anyNA(names(control)) ||
       any(!nzchar(names(control))) || anyDuplicated(names(control)) || any(!names(control)%in%c("tol","maxit")))))
        stop("'po.test.control' permits only named 'tol' and 'maxit' entries",call.=FALSE)
    out <- utils::modifyList(list(tol=1e-10,maxit=500L),control)
    scalar <- function(z) is.numeric(z) && !is.complex(z) && length(z)==1L && is.finite(z) && z>0
    if(!scalar(out$tol) || !scalar(out$maxit) || out$maxit!=floor(out$maxit) || out$maxit>.Machine$integer.max)
        stop("'po.test.control' requires positive 'tol' and positive integer 'maxit'",call.=FALSE)
    out
}

.ordinal_po_expected_score <- function(x,response,offset,fit,selected,block.size) {
    n <- nrow(x); p <- ncol(x); q <- length(fit$alpha); s <- length(selected)
    d0 <- q+p; d <- d0+s*(q-1L); frequency <- fit$frequency
    alpha <- fit$working$alpha; beta <- fit$working$beta
    gap <- diff(alpha)
    if(any(!is.finite(gap)|gap<=0)) stop("the null fit requires finite, strictly ordered thresholds",call.=FALSE)
    tgap <- exp(-gap)/(-expm1(-gap))
    U <- numeric(d); I <- matrix(0,d,d)
    for(first in seq.int(1L,n,by=block.size)) {
        at <- seq.int(first,min(n,first+block.size-1L))
        xx <- .ordinal_prediction_scale(x[at,,drop=FALSE],fit$working$center,fit$working$scale)
        eta <- outer(-streaming_qr_eta(xx,beta,offset[at]),alpha,"+")
        F <- plogis(eta); S <- plogis(-eta)
        counts <- .ordinal_assessment_counts(response,at,q+1L)/frequency
        row.mass <- response$totals[at]/frequency
        for(k in seq_len(q+1L)) {
            upper <- if(k<=q) S[,k]+if(k>1L) tgap[k-1L] else 0 else numeric(length(at))
            lower <- if(k>1L) -F[,k-1L]-if(k<=q) tgap[k-1L] else 0 else numeric(length(at))
            probability <- if(k==1L) F[,1L] else if(k==q+1L) S[,q] else
                F[,k]*S[,k-1L]*(-expm1(alpha[k-1L]-alpha[k]))
            if(any(!is.finite(probability)|probability<=0))
                stop("the fitted null has a category probability outside the representable positive range",call.=FALSE)
            score <- matrix(0,length(at),d)
            if(k<=q) score[,k] <- upper
            if(k>1L) score[,k-1L] <- lower
            location <- (if(k>1L) F[,k-1L] else 0)-(if(k<=q) S[,k] else 0)
            score[,q+seq_len(p)] <- xx*location
            if(k>=2L && k<=q) score[,d0+(k-2L)*s+seq_len(s)] <- -xx[,selected,drop=FALSE]*upper
            if(k>2L) score[,d0+(k-3L)*s+seq_len(s)] <- -xx[,selected,drop=FALSE]*lower
            U <- U+drop(crossprod(score,counts[,k]))
            I <- I+crossprod(score,(row.mass*probability)*score)
        }
    }
    if(any(!is.finite(U)) || any(!is.finite(I))) stop("score or expected information is nonfinite",call.=FALSE)
    # Normalize parameter coordinates before factoring to avoid unit-dependent
    # numerical rejection. This does not change the quadratic statistic.
    scale <- sqrt(diag(I))
    if(any(!is.finite(scale)|scale<=0)) stop("expected information has a nonpositive diagonal",call.=FALSE)
    I <- I/outer(scale,scale); U <- U/scale
    nuisance <- seq_len(d0); tested <- seq.int(d0+1L,d)
    A <- I[nuisance,nuisance,drop=FALSE]; C <- I[tested,nuisance,drop=FALSE]
    root <- chol(A)
    projection <- t(backsolve(root,forwardsolve(t(root),t(C))))
    e <- drop(U[tested]-projection%*%U[nuisance])
    J <- I[tested,tested,drop=FALSE]-projection%*%t(C); J <- (J+t(J))/2
    js <- sqrt(diag(J))
    if(any(!is.finite(js)|js<=0)) stop("efficient expected information is not positive definite",call.=FALSE)
    J <- J/outer(js,js); e <- e/js
    root <- chol(J); condition <- min(rcond(A),rcond(J))
    if(!is.finite(condition) || condition<1e-14)
        stop("expected information is numerically singular",call.=FALSE)
    statistic <- frequency*sum(forwardsolve(t(root),e)^2)
    if(!is.finite(statistic)) stop("the score statistic is nonfinite",call.=FALSE)
    list(statistic=statistic,condition=condition)
}

.ordinal_po_separation <- function(x,response,K) {
    # A sufficient certificate, not a general separation detector. For a
    # two-valued column, shifting its slope and every threshold can leave one
    # group unchanged while increasing an endpoint-only group's likelihood.
    for(j in seq_len(ncol(x))) {
        values <- unique(x[,j])
        if(length(values)!=2L) next
        for(value in values) {
            at <- which(x[,j]==value)
            counts <- if(response$grouped) colSums(response$y[at,,drop=FALSE]*response$weights[at]) else
                tabulate(response$codes[at],nbins=K)
            present <- which(counts>0)
            if(length(present)==1L && present %in% c(1L,K))
                return(list(column=colnames(x)[j],value=value,category=present))
        }
    }
    NULL
}

.ordinal_assessment_po_test <- function(object,design,response,included,terms,control,
                                       estimation="ML") {
    selected <- .ordinal_po_test_columns(object,terms); ctl <- .ordinal_po_test_control(control)
    K <- length(object$levels); p <- length(object$beta)
    out <- structure(list(method="Expected-information proportional-odds score test",available=FALSE,
        statistic=NA_real_,df=length(selected$index)*(K-2L),p.value=NA_real_,
        null.estimation=estimation,terms=selected$terms,columns=selected$columns,
        observations=sum(included),frequency=sum(response$totals[included]),
        reason="",detail="",null.converged=FALSE,null.iterations=NA_integer_,
        information.rcond=NA_real_,empty.categories=character(),warning=character(),separation=NULL,
        parameter=c(df=length(selected$index)*(K-2L)),data.name="ordinal assessment outcomes",
        note=if(estimation=="ML") "First-order chi-square approximation." else
            "Ordinary efficient score at a mean-bias-reduced null; first-order chi-square approximation."),
        class=c("fastglm_po_test","fastglm_test","htest"))
    unavailable <- function(reason,detail) {out$reason <- reason; out$detail <- detail; out}
    if(K<3L || !p) return(unavailable("no_slope_contrasts","At least three categories and one predictor are required."))
    if(!any(included)) return(unavailable("no_usable_outcomes","No positive-frequency complete assessment rows are available."))
    if(!is.matrix(design$x) || !is.numeric(design$x))
        return(unavailable("unsupported_design","The score test currently requires a dense numeric design."))
    index <- which(included)
    xx <- design$x[index,,drop=FALSE]; off <- design$offset[index]
    yy <- if(response$grouped) response$y[index,,drop=FALSE] else response$codes[index]
    rr <- .ordinal_assessment_response(yy,response$weights[index],object$levels,length(index))
    totals <- numeric(K)
    for(first in seq.int(1L,length(index),by=object$block.size)) {
        at <- seq.int(first,min(length(index),first+object$block.size-1L))
        totals <- totals+colSums(.ordinal_assessment_counts(rr,at,K))
    }
    out$empty.categories <- object$levels[totals==0]
    if(length(out$empty.categories)) return(unavailable("empty_categories",
        paste0("Every declared category must have positive frequency; empty: ",paste(out$empty.categories,collapse=", "),".")))
    constant <- vapply(seq_len(p),function(j) all(xx[,j]==xx[1L,j]),logical(1))
    if(any(constant)) return(unavailable("rank_deficient_design",paste0(
        "No predictor variation in the assessment rows: ",paste(colnames(xx)[constant],collapse=", "),
        ". The requested model is not identifiable.")))
    out$separation <- .ordinal_po_separation(xx,rr,K)
    if(estimation=="ML" && !is.null(out$separation)) {
        sep <- out$separation; out$null.iterations <- 0L
        return(unavailable("null_separation",paste0("Predictor '",sep$column,"' at ",
            format(sep$value,trim=TRUE)," occurs only in endpoint category '",object$levels[sep$category],
            "'. The ML null has no finite solution. A separately requested meanBR null may be used; its test remains an approximation.")))
    }
    warning <- character(); error <- ""
    fit <- tryCatch(withCallingHandlers(.ordinal_fit_core(xx,yy,rr$weights,off,object$levels,
        firth=estimation=="meanBR",tol=ctl$tol,maxit=ctl$maxit,block.size=object$block.size),warning=function(w) {
            warning <<- c(warning,conditionMessage(w)); invokeRestart("muffleWarning")
        }),error=function(e) {error <<- conditionMessage(e); NULL})
    out$warning <- unique(warning)
    if(is.null(fit)) return(unavailable(if(grepl("rank|constant column",error)) "rank_deficient_design" else "null_fit_error",error))
    out$null.converged <- isTRUE(fit$converged); out$null.iterations <- fit$iterations
    if(!out$null.converged) return(unavailable("null_fit_not_converged",
        paste0("The ",estimation," proportional-odds null did not converge: ",gsub("_"," ",fit$convergence.reason,fixed=TRUE),".")))
    result <- tryCatch(.ordinal_po_expected_score(xx,rr,off,fit,selected$index,object$block.size),
        error=function(e) {error <<- conditionMessage(e); NULL})
    if(is.null(result)) return(unavailable("information_unavailable",error))
    out$available <- TRUE; out$statistic <- result$statistic
    out$p.value <- pchisq(out$statistic,out$df,lower.tail=FALSE)
    out$information.rcond <- result$condition; out$reason <- "available"; out
}

#' Test the proportional-odds assumption
#'
#' Computes the expected-information score test directly, without calculating
#' prediction losses or calibration summaries. Print the result for a compact
#' test table, or use `as.data.frame()` or `broom::tidy()` for reporting.
#' @param object a `fastglm_ordinal` fit retaining its design and outcomes.
#'   Its original coefficient estimates are not used in the test. The requested
#'   null is refitted and must converge, including if the original fit did not.
#' @param terms formula terms, or named matrix columns, to test jointly.
#'   The default tests all location predictors.
#' @param null.estimation `"ML"` (default) or `"meanBR"`. The latter evaluates
#'   the ordinary efficient score at a mean-bias-reduced null fit, with explicit
#'   projection of the ordinary nuisance score. It can supply a finite null
#'   under separation, but does not guarantee convergence or small-sample accuracy.
#' @param newdata,y,weights,offset assessment data as in [fastglm_ordinal_assess()].
#' @param control named list containing `tol` and/or `maxit` for the null refit.
#' @details Both choices use expected information and an asymptotic chi-square
#'   reference. Mean bias reduction gives a first-order equivalent score test
#'   under a regular interior null with fixed predictor and category dimensions.
#'   The nuisance projection is essential because the ordinary nuisance score
#'   is nonzero at a bias-reduced root. This is not an adjusted-score test with
#'   established higher-order accuracy. It can be conservative in small samples.
#'   The method never switches null estimators automatically.
#'
#'   Empty categories and deficient predictor rank remain unsupported. A
#'   sufficient separation check identifies two-valued predictors with a level
#'   observed only in an endpoint category. It does not detect all forms of
#'   separation; other unsuccessful null fits retain their convergence details.
#'   See [fastglm_ordinal_assess()] for the hypothesis, assumptions, and frequency
#'   conventions, and [fastglm-test-methods] for result extraction.
#' @references Kosmidis, I. (2014). Improved estimation in cumulative link
#'   models. Journal of the Royal Statistical Society, Series B, 76, 169--196.
#'   \doi{10.1111/rssb.12025}. The present meanBR option uses the ordinary
#'   projected score at a bias-reduced null, rather than the larger-model
#'   adjusted-score statistic discussed in Section 12 of that article.
#' @return A `fastglm_po_test` object inheriting from `fastglm_test` and `htest`,
#'   with numeric statistic, degrees of freedom, p-value, availability, null
#'   estimation method, tested terms, and detailed fitting diagnostics.
#' @examples
#' fit <- fastglm_ordinal(Sat ~ Infl + Type + Cont, MASS::housing,
#'                        weights = Freq, firth = TRUE)
#' fastglm_po_test(fit)
#' as.data.frame(fastglm_po_test(fit, terms = "Type"))
#' @export
fastglm_po_test <- function(object,terms=NULL,null.estimation=c("ML","meanBR"),
                            newdata=NULL,y=NULL,weights=NULL,offset=NULL,control=list()) {
    if(!inherits(object,"fastglm_ordinal")) stop("supply a fastglm_ordinal fit",call.=FALSE)
    null.estimation <- match.arg(null.estimation)
    training <- is.null(newdata) && is.null(y)
    if(!is.null(newdata) && is.null(y)) stop("supply assessment outcomes 'y' with 'newdata'",call.=FALSE)
    if(is.null(y)) y <- object$y
    if(training && is.null(weights)) weights <- object$weights
    design <- .ordinal_prediction_design(object,newdata,offset)
    response <- .ordinal_assessment_response(y,weights,object$levels,nrow(design$x))
    included <- !is.na(response$totals) & response$totals>0 & !response$missing.response & !is.na(response$weights)
    .design_blocks(design$x,function(z,index) {
        off <- design$offset[index]
        if(any(is.infinite(z)) || any(is.infinite(off)))
            stop("ordinal test predictors and offsets must be finite or missing",call.=FALSE)
        included[index] <<- included[index] & rowSums(is.na(z))==0L & !is.na(off)
    },object$block.size)
    result <- .ordinal_assessment_po_test(object,design,response,included,terms,control,null.estimation)
    result$data.name <- paste(deparse(substitute(object)),collapse=" ")
    result
}
