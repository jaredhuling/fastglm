.ordinal_profile_control <- function(control) {
    defaults <- list(maxit=300L,score.tol=1e-7,step.tol=1e-7,root.tol=1e-6,
        max.expand=25L,max.root=100L,max.halves=35L,max.step=4,condition.tol=1e-14)
    if(!is.list(control) || (length(control) && (is.null(names(control)) || anyNA(names(control)) ||
        any(!nzchar(names(control))) || anyDuplicated(names(control)) || any(!names(control)%in%names(defaults)))))
        stop("unknown, unnamed, or duplicated ordinal inference control",call.=FALSE)
    if(any(vapply(control,is.null,logical(1))))
        stop("ordinal inference controls must be positive finite scalars",call.=FALSE)
    ctl <- utils::modifyList(defaults,control)
    for(nm in names(ctl)) if(!is.numeric(ctl[[nm]]) || is.complex(ctl[[nm]]) ||
        length(ctl[[nm]])!=1L || !is.finite(ctl[[nm]]) || ctl[[nm]]<=0)
        stop("ordinal inference controls must be positive finite scalars",call.=FALSE)
    for(nm in c("maxit","max.expand","max.root","max.halves"))
        if(ctl[[nm]]>.Machine$integer.max || ctl[[nm]]!=floor(ctl[[nm]]))
            stop("ordinal inference iteration limits must be positive integers",call.=FALSE)
    ctl
}

.ordinal_profile_setup <- function(object,control,profile=TRUE) {
    if(!inherits(object,"fastglm_ordinal")) stop("ordinal inference requires an ordinal model",call.=FALSE)
    if(profile && isTRUE(object$firth))
        stop("ordinal profile likelihood requires maximum-likelihood estimation; use an ML fit or fastglm_score_test() for adjusted-score inference",call.=FALSE)
    if(inherits(object,"fastglm_ordinal_stream") || !is.matrix(object$x))
        stop("ordinal constrained inference currently requires a retained dense design",call.=FALSE)
    if(!isTRUE(object$converged)) stop("ordinal constrained inference requires a converged original fit",call.=FALSE)
    if(isTRUE(object$boundary) || any(object$category$category.totals<=0))
        stop("ordinal constrained inference requires every declared category to be observed",call.=FALSE)
    ctl <- .ordinal_profile_control(control)
    if(inherits(object,"fastglm_ordinal_partial")) return(.ordinal_partial_profile_setup(object,ctl))
    b <- coef(object);q <- length(object$alpha);p <- length(object$beta);d <- q+p
    if(length(b)!=d || any(!is.finite(b)) || object$rank!=d)
        stop("ordinal constrained inference requires a full-rank identifiable model",call.=FALSE)
    category <- object$category;x <- object$x;offset <- object$offset
    if(ncol(x)!=p || nrow(x)!=category$n || length(offset)!=nrow(x) ||
       length(category$weights)!=nrow(x) || any(!is.finite(x)) || any(!is.finite(offset)))
        stop("invalid retained ordinal data",call.=FALSE)
    # Reuse the fitted linear change of coordinates. Constraints and inference
    # remain on the original threshold/common-slope parameter scale.
    center <- object$working$center;scale <- object$working$scale
    if(length(center)!=p || length(scale)!=p || any(!is.finite(c(center,scale))) || any(scale<=0))
        stop("ordinal fitting coordinates are unavailable",call.=FALSE)
    T <- Ti <- diag(d)
    if(p) {
        x <- .ordinal_prediction_scale(x,center,scale)
        T[seq_len(q),q+seq_len(p)] <- matrix(center/scale,q,p,byrow=TRUE)
        T[q+seq_len(p),q+seq_len(p)] <- diag(1/scale,nrow=p)
        Ti[seq_len(q),q+seq_len(p)] <- matrix(-center,q,p,byrow=TRUE)
        Ti[q+seq_len(p),q+seq_len(p)] <- diag(scale,nrow=p)
    }
    positive <- category$totals>0
    if(any(!positive)) {x[!positive,] <- 0;offset[!positive] <- 0}
    storage.mode(x) <- "double"
    evaluate <- function(gamma,adjusted=FALSE) tryCatch({
        value <- ordinal_native_evaluate(x,category$codes,category$counts,category$weights,
            offset,q+1L,as.numeric(gamma),adjusted,object$block.size,ctl$condition.tol,.ordinal_link_code(.ordinal_link(object)))
        value$valid <- TRUE;value$message <- "";value
    },error=function(e) list(valid=FALSE,message=conditionMessage(e)))
    constrained <- function(gamma,basis,adjusted) ordinal_constrained_native(x,category$codes,
        category$counts,category$weights,offset,q+1L,gamma,basis,positive,adjusted,object$block.size,
        ctl$condition.tol,.ordinal_link_code(.ordinal_link(object)),as.integer(ctl$maxit),ctl$score.tol,
        ctl$step.tol,ctl$max.step,as.integer(ctl$max.halves))
    list(coefficients=b,q=q,p=p,d=d,T=T,Ti=Ti,x=x,positive=positive,
        evaluate=evaluate,constrained=constrained,control=ctl,frequency=object$frequency,observations=object$nobs.rows)
}

.ordinal_constraint_basis <- function(setup,L) {
    if(!nrow(L)) return(list(matrix=matrix(numeric(),0L,setup$d),basis=diag(setup$d),row.scale=numeric()))
    C <- L%*%setup$T;scale <- sqrt(rowSums(C^2))
    if(any(!is.finite(scale)) || any(scale<=0)) stop("invalid or unscalable ordinal constraint",call.=FALSE)
    C <- C/scale
    decomposition <- qr(t(C),tol=64*.Machine$double.eps*max(dim(C)))
    if(nrow(C)>ncol(C) || decomposition$rank!=nrow(C))
        stop("ordinal constraints must be linearly independent",call.=FALSE)
    Q <- qr.Q(decomposition,complete=TRUE)
    B <- Q[,which(seq_len(setup$d)>nrow(C)),drop=FALSE]
    list(matrix=C,basis=B,row.scale=scale)
}

.ordinal_constraint_start <- function(setup,L,null,start,constraint) {
    if(isTRUE(setup$partial)) return(.ordinal_partial_constraint_start(setup,null,start,constraint))
    b <- start
    if(nrow(L)) {
        nonzero <- L!=0
        selection <- rowSums(nonzero)==1L
        if(all(selection)) {
            index <- max.col(abs(L),ties.method="first")
            b[index] <- null/L[cbind(seq_len(nrow(L)),index)]
            fixed <- sort(index[index<=setup$q]);alpha <- b[seq_len(setup$q)]
            if(length(fixed)>1L && any(diff(alpha[fixed])<=0))
                return(list(valid=FALSE,message="fixed thresholds are not strictly ordered"))
            anchors <- c(0L,fixed,setup$q+1L)
            typical <- if(setup$q>1L) median(diff(start[seq_len(setup$q)])) else 1
            for(k in seq_len(length(anchors)-1L)) {
                lo <- anchors[k];hi <- anchors[k+1L];free <- which(seq_len(setup$q)>lo & seq_len(setup$q)<hi)
                if(!length(free)) next
                lower <- if(lo) alpha[lo] else -Inf;upper <- if(hi<=setup$q) alpha[hi] else Inf
                if(all(alpha[free]>lower & alpha[free]<upper) && all(diff(alpha[free])>0)) next
                alpha[free] <- if(is.finite(lower) && is.finite(upper))
                    lower+(upper-lower)*seq_along(free)/(length(free)+1) else if(is.finite(lower))
                    lower+typical*seq_along(free) else upper-typical*rev(seq_along(free))
            }
            b[seq_len(setup$q)] <- alpha
        } else if(all(L[,seq_len(setup$q),drop=FALSE]==0)) {
            # Location contrasts leave the original thresholds unconstrained.
            A <- L[,setup$q+seq_len(setup$p),drop=FALSE]
            scaled <- A/rep(sqrt(rowSums(A^2)),ncol(A))
            rhs <- (null-drop(L%*%b))/sqrt(rowSums(A^2))
            b[setup$q+seq_len(setup$p)] <- b[setup$q+seq_len(setup$p)]+
                drop(t(scaled)%*%solve(tcrossprod(scaled),rhs))
        }
    }
    gamma <- drop(setup$Ti%*%b)
    if(nrow(L)) {
        C <- constraint$matrix;target <- null/constraint$row.scale
        project <- function(g) g+drop(t(C)%*%solve(tcrossprod(C),target-drop(C%*%g)))
        gamma <- project(gamma)
        if(setup$q>1L && any(diff(gamma[seq_len(setup$q)])<=0)) {
            # Feasibility only: alternating projections supply a starting point
            # in the affine constraint and strict threshold-order domains.
            gap <- min(diff(start[seq_len(setup$q)]))/100
            for(iter in seq_len(500L)) {
                alpha <- stats::isoreg(seq_len(setup$q),gamma[seq_len(setup$q)]-gap*seq_len(setup$q))$yf+
                    gap*seq_len(setup$q)
                gamma[seq_len(setup$q)] <- alpha;gamma <- project(gamma)
                if(all(diff(gamma[seq_len(setup$q)])>0)) break
            }
        }
    }
    if(any(!is.finite(gamma)) || (setup$q>1L && any(diff(gamma[seq_len(setup$q)])<=0)))
        return(list(valid=FALSE,message="no ordered starting thresholds were found under the supplied constraints"))
    list(valid=TRUE,coefficients=gamma,message="")
}

.ordinal_constrained_fit <- function(setup,L=matrix(numeric(),0L,setup$d),null=numeric(),
                                     start=setup$coefficients,adjusted=FALSE) {
    ctl <- setup$control;constraint <- .ordinal_constraint_basis(setup,L);B <- constraint$basis
    initial <- .ordinal_constraint_start(setup,L,null,start,constraint)
    finish <- function(gamma,state,code,message,iterations,score=Inf,step=Inf,evaluations=0L) {
        coefficients <- if(length(gamma)) if(isTRUE(setup$partial))
            streaming_qr_eta(setup$T,gamma,numeric(setup$d)) else drop(setup$T%*%gamma) else rep(NA_real_,setup$d)
        if(code==0L && isTRUE(setup$partial) && nrow(L)) {
            residual <- streaming_qr_eta(L,coefficients,numeric(nrow(L)))-null
            if(any(!is.finite(residual)) || any(abs(residual)>512*.Machine$double.eps*(1+abs(null)))) {
                code <- 6L;message <- "linear constraints cannot be represented accurately in the reported coefficients"
            }
        }
        names(coefficients) <- names(setup$coefficients)
        list(coefficients=coefficients,working.coefficients=gamma,
            value=if(isTRUE(state$valid)) state$loglik else NA_real_,converged=code==0L,
            score=score,predictor.step=step,code=code,message=message,iterations=iterations,
            evaluations=evaluations,state=state,basis=B,adjusted=adjusted)
    }
    if(!initial$valid) return(finish(numeric(),list(valid=FALSE),1L,initial$message,0L))
    out <- setup$constrained(as.numeric(initial$coefficients),B,adjusted)
    state <- if(is.null(out$state)) list(valid=FALSE,message=out$message) else
        c(out$state,list(valid=TRUE,message=""))
    finish(out$gamma,state,out$code,out$message,out$iterations,out$score,out$step,out$evaluations)
}

.ordinal_profile_lr <- function(full,restricted,control) {
    if(!isTRUE(full$converged)) stop(paste("unrestricted fit:",full$message),call.=FALSE)
    if(!isTRUE(restricted$converged)) stop(paste("constrained fit:",restricted$message),call.=FALSE)
    value <- 2*(full$value-restricted$value)
    rounding <- max(control$score.tol^2,256*.Machine$double.eps*(1+abs(full$value)))
    if(!is.finite(value) || value < -rounding)
        stop("constrained likelihood exceeds the unrestricted maximum",call.=FALSE)
    max(0,value)
}

.ordinal_profile_test <- function(object,parm=NULL,null=0,control=list(),data.name=deparse(substitute(object)),contrast=NULL) {
    setup <- .ordinal_profile_setup(object,control)
    if(!is.null(contrast)) {
        if(!is.null(parm)) stop("supply either 'parm' or 'contrast', not both",call.=FALSE)
        L <- .fastglm_wald_contrast(object,NULL,contrast)$contrast
    } else {
        parm <- .firth_profile_parameters(parm,setup$coefficients)
        L <- diag(setup$d)[parm,,drop=FALSE]
        dimnames(L) <- list(names(setup$coefficients)[parm],names(setup$coefficients))
    }
    if(length(null)==1L) null <- rep(null,nrow(L))
    if(!is.numeric(null) || is.complex(null) || length(null)!=nrow(L) || any(!is.finite(null)))
        stop("'null' must contain one finite value per constraint",call.=FALSE)
    names(null) <- rownames(L)
    .ordinal_constraint_basis(setup,L)
    full <- .ordinal_constrained_fit(setup);restricted <- NULL;statistic <- NA_real_
    message <- tryCatch({
        if(!full$converged) stop(paste("unrestricted fit:",full$message),call.=FALSE)
        restricted <- .ordinal_constrained_fit(setup,L,null,start=full$coefficients)
        statistic <- .ordinal_profile_lr(full,restricted,setup$control);""
    },error=conditionMessage)
    if(nzchar(message)) warning("ordinal profile test failed: ",message,call.=FALSE)
    # Full native states are useful during computation but are not duplicated
    # in the public diagnostic record.
    strip <- function(fit) {fit$state <- fit$basis <- NULL;fit}
    structure(list(statistic=c(`likelihood ratio`=statistic),parameter=c(df=nrow(L)),
        p.value=if(is.finite(statistic)) pchisq(statistic,nrow(L),lower.tail=FALSE) else NA_real_,
        estimate=setNames(drop(L%*%full$coefficients),rownames(L)),null.value=null,
        contrast=if(!is.null(contrast)) L else NULL,alternative="two.sided",
        method="Ordinal likelihood ratio test (asymptotic chi-square reference)",data.name=data.name,
        loglik=c(full=full$value,restricted=if(is.null(restricted)) NA_real_ else restricted$value),
        null.estimation="ML",frequency=setup$frequency,observations=setup$observations,
        available=!nzchar(message),diagnostics=list(converged=!nzchar(message),message=message,
            full=strip(full),restricted=if(!is.null(restricted)) strip(restricted))),
        class=c("fastglm_test","htest"))
}

.ordinal_profile_confint <- function(object,parm,level,control=list()) {
    setup <- .ordinal_profile_setup(object,control)
    parm <- .firth_profile_parameters(parm,setup$coefficients)
    full <- .ordinal_constrained_fit(setup);target <- qchisq(level,1L)
    scale <- if(full$converged) sqrt(diag(setup$T%*%full$state$cov.unscaled%*%t(setup$T))) else rep(NA_real_,setup$d)
    out <- matrix(NA_real_,length(parm),2L,dimnames=list(names(setup$coefficients)[parm],
        paste0(format(100*c((1-level)/2,(1+level)/2),trim=TRUE),"%")))
    diagnostics <- list()
    for(k in seq_along(parm)) for(side in 1:2) {
        j <- parm[k];latest <- full;lr <- NA_real_;root.iterations <- 0L
        message <- tryCatch({
            if(!full$converged) stop(paste("unrestricted fit:",full$message),call.=FALSE)
            center <- full$coefficients[j];L <- diag(setup$d)[j,,drop=FALSE]
            cache <- new.env(parent=emptyenv())
            evaluate <- function(z) {
                key <- sprintf("%.17g",z)
                if(exists(key,cache,inherits=FALSE)) latest <<- get(key,cache,inherits=FALSE) else {
                    latest <<- .ordinal_constrained_fit(setup,L,center+z*scale[j],start=latest$coefficients)
                    latest$state <<- NULL
                    latest$basis <<- NULL
                    assign(key,latest,cache)
                }
                lr <<- .ordinal_profile_lr(full,latest,setup$control)
                lr-target
            }
            direction <- if(side==1L) -1 else 1
            distance <- sqrt(target);endpoint <- direction*distance;gap <- evaluate(endpoint)
            for(iter in seq_len(setup$control$max.expand)) {
                if(gap>=0) break
                distance <- 2*distance;endpoint <- direction*distance;gap <- evaluate(endpoint)
            }
            if(gap<0) stop("profile endpoint not bracketed; an unbounded limit has not been established",call.=FALSE)
            interval <- sort(c(0,endpoint))
            root <- uniroot(evaluate,interval=interval,f.lower=if(side==1L) gap else -target,
                f.upper=if(side==1L) -target else gap,tol=setup$control$root.tol,
                maxiter=setup$control$max.root,check.conv=TRUE)
            root.iterations <- root$iter;evaluate(root$root)
            out[k,side] <- center+root$root*scale[j];""
        },error=conditionMessage)
        diagnostics[[length(diagnostics)+1L]] <- data.frame(parameter=names(setup$coefficients)[j],
            side=c("lower","upper")[side],converged=!nzchar(message),estimate=out[k,side],
            statistic=lr,target=target,lr.error=lr-target,score=latest$score,
            predictor.step=latest$predictor.step,root.iterations=root.iterations,
            optimizer.code=latest$code,message=message,stringsAsFactors=FALSE)
    }
    diagnostics <- do.call(rbind,diagnostics)
    if(any(!diagnostics$converged)) warning(sum(!diagnostics$converged),
        " ordinal profile endpoints failed; inspect attr(result, 'diagnostics')",call.=FALSE)
    full$state <- full$basis <- NULL
    attr(out,"diagnostics") <- diagnostics;attr(out,"fit.diagnostics") <- full
    attr(out,"method") <- "Ordinal profile likelihood (asymptotic chi-square reference)"
    attr(out,"level") <- level
    class(out) <- c("fastglm_confint","matrix","array")
    out
}

#' Score tests for ordinal model terms and coefficient contrasts
#'
#' Evaluates ordinary or mean-bias-adjusted likelihood scores under linear
#' constraints, after re-estimating the nuisance parameters. The information
#' matrix and, when requested, the bias adjustment refer to the full model.
#' @param object a converged ordinal model, including a nonparallel model,
#'   with retained dense data and every
#'   declared category observed.
#' @param terms formula terms, or location coefficient names for a matrix fit.
#'   The default jointly tests all location slopes.
#' @param contrast numeric vector or matrix of linear coefficient contrasts,
#'   as in [fastglm_wald_test()]. Named columns may select a subset of parameters.
#' @param null hypothesized contrast values; a scalar is repeated across rows.
#' @param null.estimation \code{"ML"} for ordinary constrained likelihood
#'   scores, or \code{"meanBR"} for constrained full-model adjusted scores.
#'   The default is ML even when \code{object} was fitted with bias reduction.
#' @param control named numerical controls, shared with ordinal profile
#'   inference. \code{maxit=300} limits constrained scoring iterations;
#'   \code{score.tol=1e-7} bounds each nuisance score divided by the square
#'   root of its expected information; and \code{step.tol=1e-7} bounds the
#'   largest remaining cumulative-predictor change. \code{max.step=4} caps
#'   each predictor update (the adjusted-score solver caps it at one),
#'   \code{max.halves=35} limits step reductions, and
#'   \code{condition.tol=1e-14} is the native information-conditioning
#'   threshold. For profile intervals, \code{root.tol=1e-6} is measured in
#'   fitted standard-error units, \code{max.root=100} limits root iterations,
#'   and \code{max.expand=25} limits endpoint-bracket doublings beyond the
#'   initial normal-approximation distance. These last three controls do not
#'   affect score tests. Exhausting a profile search yields a missing endpoint,
#'   not an assertion of an unbounded confidence interval.
#' @details Let \eqn{B} span the nuisance directions allowed by the constraints,
#'   \eqn{I} be full-model expected information, and \eqn{U} the selected score
#'   vector. The test removes the nuisance component using
#'   \eqn{U_e=U-IB(B^TIB)^{-1}B^TU}, and reports \eqn{U_e^T I^{-1}U_e}.
#'   With \code{null.estimation="meanBR"}, constrained fitting solves
#'   \eqn{B^TU^*=0} using the adjustment of the full model; it does not recompute
#'   a bias adjustment for a smaller model with the tested columns removed.
#'
#'   For independent ordinal observations or integer frequency counts, fixed
#'   parameter dimension, an interior true parameter, and nonsingular limiting
#'   information, with the usual local smoothness and consistent constrained
#'   estimation, the statistic has a first-order chi-square reference with
#'   one degree of freedom per independent constraint. The mean-bias adjustment
#'   is order one, while the likelihood score is order square root of sample
#'   size. This establishes the same first-order reference, not a higher-order
#'   calibration result or exact small-sample inference. No resampling is used.
#'
#'   Infeasible starts, failed nuisance fits, singular information, or exhausted
#'   iterations return an unavailable test with an explanation. A finite
#'   bias-reduced estimate does not guarantee a successful constrained fit.
#'   For nonparallel models, constraints and all nuisance updates retain ordered
#'   cumulative predictors at each positive-frequency training row. Equality
#'   contrasts between a predictor's threshold-specific slopes test its common
#'   slope restriction while allowing other predictors to remain nonparallel.
#'   Tests of a term's zero effects are different from tests of equal slopes.
#'   Streaming fits and collapsed or empty categories are currently unsupported.
#' @returns A \code{fastglm_test} object with the package's common printed table,
#'   numeric \code{as.data.frame()} extraction, and constrained-fit diagnostics.
#' @references Kosmidis, I. (2014). Improved estimation in cumulative link models.
#'   Journal of the Royal Statistical Society: Series B, 76, 169--196.
#'   \doi{10.1111/rssb.12025}. The nuisance projection and full-model constrained
#'   adjustment used here specify the implementation beyond the paper's general
#'   discussion of adjusted-score tests.
#' @examples
#' fit <- fastglm_ordinal(Sat ~ Infl + Type + Cont, MASS::housing, weights=Freq)
#' fastglm_score_test(fit, terms="Type")
#' fastglm_score_test(fit, terms="Type", null.estimation="meanBR")
#' @md
#' @export
fastglm_score_test <- function(object,terms=NULL,contrast=NULL,null=0,
                               null.estimation=c("ML","meanBR"),control=list()) {
    data.name <- deparse(substitute(object));null.estimation <- match.arg(null.estimation)
    setup <- .ordinal_profile_setup(object,control,profile=FALSE)
    selected <- .fastglm_wald_contrast(object,terms,contrast)
    L <- selected$contrast
    if(length(null)==1L) null <- rep(null,nrow(L))
    if(!is.numeric(null) || is.complex(null) || length(null)!=nrow(L) || any(!is.finite(null)))
        stop("'null' must contain one finite value per constraint",call.=FALSE)
    names(null) <- rownames(L)
    # Reject invalid hypothesis specifications before attempting numerical work.
    .ordinal_constraint_basis(setup,L)
    restricted <- NULL;statistic <- NA_real_
    message <- tryCatch({
        restricted <- .ordinal_constrained_fit(setup,L,null,adjusted=null.estimation=="meanBR")
        if(!restricted$converged) stop(restricted$message,call.=FALSE)
        state <- restricted$state;B <- restricted$basis;efficient <- drop(state$score)
        if(ncol(B)) efficient <- efficient-drop(state$information%*%B%*%
            solve(crossprod(B,state$information%*%B),crossprod(B,efficient)))
        statistic <- drop(crossprod(efficient,state$cov.unscaled%*%efficient))
        if(!is.finite(statistic) || statistic<0) stop("invalid efficient score quadratic",call.=FALSE)
        ""
    },error=conditionMessage)
    if(nzchar(message)) warning("ordinal score test failed: ",message,call.=FALSE)
    if(!is.null(restricted)) restricted$state <- restricted$basis <- NULL
    structure(list(statistic=c(`score statistic`=statistic),parameter=c(df=nrow(L)),
        p.value=if(is.finite(statistic)) pchisq(statistic,nrow(L),lower.tail=FALSE) else NA_real_,
        available=!nzchar(message),estimate=drop(L%*%coef(object)),null.value=null,
        terms=selected$terms,contrast=L,alternative="two.sided",null.estimation=null.estimation,
        estimation=if(isTRUE(object$firth)) "mean bias reduction" else "maximum likelihood",
        method=paste(if(null.estimation=="meanBR") "Ordinal adjusted-score test" else "Ordinal score test",
            "(asymptotic chi-square reference)"),data.name=data.name,
        frequency=setup$frequency,observations=setup$observations,
        diagnostics=list(converged=!nzchar(message),message=message,restricted=restricted)),
        class=c("fastglm_test","htest"))
}
