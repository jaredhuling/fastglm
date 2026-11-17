# Profile-likelihood intervals, likelihood-ratio tests, and score tests for
# maximum-likelihood GLM fits. A restricted fit holds selected coefficients at
# fixed values by moving their columns into the offset, so each evaluation is
# an ordinary fastglm fit on the remaining columns.

.glm_ml_inference_model <- function(object) {
    inherits(object,c("fastglm","fastglmFit")) && !isTRUE(object$firth) &&
        !inherits(object,c("fastglm_nb","fastglm_stream","fastglm_truncated")) &&
        is.null(object$source)
}

.glm_ml_setup <- function(object) {
    if(!isTRUE(object$converged)) stop("likelihood inference requires a converged fit",call.=FALSE)
    b <- coef(object)
    if(anyNA(b)) stop("likelihood inference requires a full-rank fit",call.=FALSE)
    x <- tryCatch(as.matrix(model.matrix(object)),error=function(e) NULL)
    if(is.null(x) || is.null(object$y))
        stop("likelihood inference requires the retained design and response",call.=FALSE)
    n <- nrow(x)
    w <- object$prior.weights; if(is.null(w)) w <- rep(1,n)
    off <- object$offset; if(is.null(off)) off <- rep(0,n)
    fixed <- .fixed_dispersion(object$family)
    list(object=object,x=x,y=object$y,weights=w,offset=off,family=object$family,
         coefficients=b,deviance=object$deviance,
         dispersion=if(is.na(fixed)) summary(object)$dispersion else fixed,
         estimated.dispersion=is.na(fixed),df.residual=object$df.residual,
         se=sqrt(diag(vcov(object))))
}

# Fit with coefficients 'fixed' (indices) held at 'values'.
.glm_ml_restricted <- function(setup,fixed,values,start=NULL) {
    p <- ncol(setup$x); free <- setdiff(seq_len(p),fixed)
    off <- setup$offset+drop(setup$x[,fixed,drop=FALSE]%*%values)
    coefficients <- numeric(p); coefficients[fixed] <- values
    if(!length(free)) {
        mu <- setup$family$linkinv(off)
        return(list(deviance=sum(setup$family$dev.resids(setup$y,mu,setup$weights)),
                    coefficients=coefficients,free=numeric(),converged=TRUE,reason="converged"))
    }
    fit <- suppressWarnings(fastglm(setup$x[,free,drop=FALSE],setup$y,family=setup$family,
        weights=setup$weights,offset=off,start=start))
    coefficients[free] <- coef(fit)
    list(deviance=fit$deviance,coefficients=coefficients,free=coef(fit),
         converged=isTRUE(fit$converged),reason=fit$convergence.reason)
}

.glm_ml_columns <- function(setup,parm) {
    b <- setup$coefficients
    if(is.character(parm)) parm <- match(parm,names(b))
    if(!is.numeric(parm) || !length(parm) || anyNA(parm) || any(parm<1 | parm>length(b)) || anyDuplicated(parm))
        stop("unknown or duplicated coefficient in 'parm'",call.=FALSE)
    as.integer(parm)
}

.glm_ml_profile_confint <- function(object,parm,level,control=list()) {
    if(length(level)!=1L || !is.finite(level) || level<=0 || level>=1)
        stop("'level' must lie between zero and one",call.=FALSE)
    ctrl <- utils::modifyList(list(max.expand=25L,root.tol=1e-9,max.root=100L),control)
    setup <- .glm_ml_setup(object)
    b <- setup$coefficients
    parm <- if(missing(parm) || is.null(parm)) seq_along(b) else .glm_ml_columns(setup,parm)
    z <- stats::qnorm((1+level)/2)
    out <- matrix(NA_real_,length(parm),2L,dimnames=list(names(b)[parm],
        paste0(100*c((1-level)/2,(1+level)/2)," %")))
    diagnostics <- list()
    for(k in seq_along(parm)) for(side in 1:2) {
        j <- parm[k]; direction <- if(side==1L) -1 else 1; start <- b[-j]; last <- NULL
        # Signed root of the scaled deviance difference, as in stats::profile.glm().
        signed <- function(value) {
            r <- .glm_ml_restricted(setup,j,value,start)
            if(!r$converged) stop("restricted fit did not converge (",r$reason,")",call.=FALSE)
            start <<- r$free; last <<- r
            sign(value-b[j])*sqrt(max(0,r$deviance-setup$deviance)/setup$dispersion)
        }
        message <- tryCatch({
            step <- z*setup$se[j]; endpoint <- b[j]+direction*step; value <- signed(endpoint)
            expand <- 0L
            while(abs(value)<z && expand<ctrl$max.expand) {
                step <- 2*step; endpoint <- b[j]+direction*step; value <- signed(endpoint); expand <- expand+1L
            }
            if(abs(value)<z) {
                # The deviance does not reach the cutoff: the profile is
                # unbounded in this direction (for example under separation).
                out[k,side] <- direction*Inf
                "profile does not reach the cutoff; endpoint is infinite"
            } else {
                lower <- min(b[j],endpoint); upper <- max(b[j],endpoint)
                root <- stats::uniroot(function(v) signed(v)-direction*z,c(lower,upper),
                    tol=ctrl$root.tol*setup$se[j],maxiter=ctrl$max.root)
                out[k,side] <- root$root
                ""
            }
        },error=conditionMessage)
        diagnostics[[length(diagnostics)+1L]] <- data.frame(parameter=names(b)[j],
            side=c("lower","upper")[side],estimate=out[k,side],message=message,stringsAsFactors=FALSE)
    }
    diagnostics <- do.call(rbind,diagnostics)
    failed <- !(diagnostics$message %in% c("","profile does not reach the cutoff; endpoint is infinite"))
    if(any(failed)) {
        out[cbind(match(diagnostics$parameter[failed],rownames(out)),
                  ifelse(diagnostics$side[failed]=="lower",1L,2L))] <- NA_real_
        warning(sum(failed)," profile endpoints failed; inspect attr(result, 'diagnostics')",call.=FALSE)
    }
    attr(out,"diagnostics") <- diagnostics
    attr(out,"method") <- "profile likelihood"
    attr(out,"level") <- level
    class(out) <- c("fastglm_confint","matrix","array")
    out
}

.glm_ml_null <- function(null,k) {
    if(length(null)==1L) null <- rep(null,k)
    if(!is.numeric(null) || length(null)!=k || any(!is.finite(null)))
        stop("'null' must contain one finite value per tested coefficient",call.=FALSE)
    null
}

# Joint likelihood-ratio test that the selected coefficients equal 'null'.
.glm_ml_lr_test <- function(object,parm,null,data.name) {
    setup <- .glm_ml_setup(object)
    parm <- .glm_ml_columns(setup,parm); null <- .glm_ml_null(null,length(parm))
    r <- .glm_ml_restricted(setup,parm,null)
    if(!r$converged) stop("restricted fit did not converge (",r$reason,")",call.=FALSE)
    change <- max(0,r$deviance-setup$deviance); k <- length(parm)
    if(setup$estimated.dispersion) {
        statistic <- c(F=change/(k*setup$dispersion))
        parameter <- c(df1=k,df2=setup$df.residual)
        p <- stats::pf(statistic,k,setup$df.residual,lower.tail=FALSE)
        method <- "Likelihood ratio test (F reference, estimated dispersion)"
    } else {
        statistic <- c(`likelihood ratio`=change/setup$dispersion)
        parameter <- c(df=k); p <- stats::pchisq(statistic,k,lower.tail=FALSE)
        method <- "Likelihood ratio test (asymptotic chi-square reference)"
    }
    names(null) <- names(setup$coefficients)[parm]
    structure(list(statistic=statistic,parameter=parameter,p.value=unname(p),
        estimate=setup$coefficients[parm],null.value=null,alternative="two.sided",
        method=method,data.name=data.name,
        deviance=c(full=setup$deviance,restricted=r$deviance),null.estimation="maximum likelihood"),
        class=c("fastglm_test","htest"))
}

.glm_ml_term_columns <- function(setup,terms) {
    object <- setup$object; b <- setup$coefficients
    tt <- object$terms
    if(!is.null(tt) && is.character(terms) && all(terms %in% attr(tt,"term.labels"))) {
        assignment <- attr(setup$x,"assign")
        if(is.null(assignment)) stop("score tests by term require the design's term assignment",call.=FALSE)
        return(which(assignment %in% match(terms,attr(tt,"term.labels"))))
    }
    .glm_ml_columns(setup,terms)
}

# Rao score test that the selected coefficients equal 'null', evaluated at
# the restricted maximum likelihood fit with the expected information.
.glm_ml_score_test <- function(object,terms,null,data.name) {
    setup <- .glm_ml_setup(object)
    if(is.null(terms)) stop("supply the terms or coefficients to test",call.=FALSE)
    parm <- .glm_ml_term_columns(setup,terms); null <- .glm_ml_null(null,length(parm))
    r <- .glm_ml_restricted(setup,parm,null)
    if(!r$converged) stop("restricted fit did not converge (",r$reason,")",call.=FALSE)
    fam <- setup$family
    eta <- drop(setup$x%*%r$coefficients)+setup$offset
    mu <- fam$linkinv(eta); dm <- fam$mu.eta(eta); v <- fam$variance(mu)
    score <- colSums(setup$x*(setup$weights*(setup$y-mu)*dm/v))
    information <- crossprod(setup$x,setup$x*(setup$weights*dm^2/v))
    # The dispersion is that of the full model, as in anova(..., test = "Rao").
    statistic <- drop(crossprod(score,solve(information,score)))/setup$dispersion
    # Chi-square reference, as in anova(..., test = "Rao"), also when the
    # dispersion is estimated.
    k <- length(parm)
    statistic <- c(score=statistic); parameter <- c(df=k)
    p <- stats::pchisq(statistic,k,lower.tail=FALSE)
    method <- "Score test (asymptotic chi-square reference)"
    names(null) <- names(setup$coefficients)[parm]
    structure(list(statistic=statistic,parameter=parameter,p.value=unname(p),
        estimate=setup$coefficients[parm],null.value=null,alternative="two.sided",
        method=method,data.name=data.name,null.estimation="maximum likelihood"),
        class=c("fastglm_test","htest"))
}
