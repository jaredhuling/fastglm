.firth_profile_control <- function(control) {
    defaults<-list(maxit=300L,reltol=1e-12,score.tol=1e-5,root.tol=1e-6,max.expand=25L,max.root=100L)
    if(!is.list(control) || (length(control) && (is.null(names(control)) || any(!names(control)%in%names(defaults)))))
        stop("unknown or unnamed profile control",call.=FALSE)
    out<-utils::modifyList(defaults,control)
    for(nm in names(out)) if(!is.numeric(out[[nm]]) || length(out[[nm]])!=1L || !is.finite(out[[nm]]) || out[[nm]]<=0)
        stop("profile controls must be positive finite scalars",call.=FALSE)
    for(nm in c("maxit","max.expand","max.root")) if(out[[nm]]>.Machine$integer.max || out[[nm]]!=floor(out[[nm]]))
        stop("profile iteration limits must be positive integers",call.=FALSE)
    out
}
.firth_profile_parameters <- function(parm,b) {
    if(is.character(parm) && anyDuplicated(names(b)))
        stop("coefficient names are ambiguous; select 'parm' by index",call.=FALSE)
    if(is.character(parm)) parm<-match(parm,names(b))
    if(!is.numeric(parm) || !length(parm) || anyNA(parm) || any(!is.finite(parm)) ||
       any(parm<1L | parm>length(b)) || any(parm!=floor(parm)) || anyDuplicated(parm))
        stop("'parm' must select distinct known coefficients",call.=FALSE)
    as.integer(parm)
}
.firth_profile_setup <- function(object,control) {
    if(!isTRUE(object$firth) || !identical(object$family$family,"binomial") || !identical(object$family$link,"logit"))
        stop("profile inference requires a binomial-logit Firth fit",call.=FALSE)
    if(!isTRUE(object$converged)) stop("profile inference requires a converged original fit",call.=FALSE)
    b<-coef(object)
    if(any(!is.finite(b)) || object$rank!=length(b))
        stop("profile inference requires a full-rank identifiable design",call.=FALSE)
    if(!is.null(object$source) || !is.matrix(object$x))
        stop("profile inference currently requires a retained dense design matrix",call.=FALSE)
    d<-.prediction_design(object);x<-d$x;y<-object$y;w<-object$prior.weights
    if(is.null(y) || is.matrix(y) || length(y)!=nrow(x) || length(w)!=nrow(x))
        stop("profile inference requires the retained initialized binomial response and prior weights",call.=FALSE)
    storage.mode(x)<-"double"
    if(any(!is.finite(x)) || any(!is.finite(y)) || any(y<0 | y>1) || any(!is.finite(w)) || any(w<0) || any(!is.finite(d$offset)))
        stop("invalid retained binomial data",call.=FALSE)
    successes<-w*y;failures<-w*(1-y)
    counts<-c(successes,failures)
    # Allow only multiplication round-off, capped well below half a count.
    count.tolerance<-pmin(1e-7,64*.Machine$double.eps*pmax(1,abs(counts)))
    if(any(abs(counts-round(counts))>count.tolerance))
        stop("profile inference requires integer effective successes and failures; fractional power or survey weights are unsupported",call.=FALSE)
    y<-as.numeric(y);w<-as.numeric(w);offset<-as.numeric(d$offset)
    evaluate<-function(beta) firth_profile_evaluate(x,y,w,offset,as.numeric(beta))
    value<-evaluate(b)
    if(!value$valid) stop(value$message,call.=FALSE)
    scale<-sqrt(diag(chol2inv(chol(value$information))))
    if(any(!is.finite(scale)) || any(scale<=0)) stop("profile parameter scales are unavailable",call.=FALSE)
    list(coefficients=b,evaluate=evaluate,scale=scale,control=.firth_profile_control(control))
}
.firth_profile_fit <- function(setup,fixed=integer(),values=numeric(),start=setup$coefficients) {
    b<-start;b[fixed]<-values;free<-setdiff(seq_along(b),fixed);ctl<-setup$control
    last.beta<-NULL;last<-NULL;calls<-0L
    evaluate<-function(beta) {
        if(!identical(beta,last.beta)) {last<<-setup$evaluate(beta);last.beta<<-beta;calls<<-calls+1L}
        last
    }
    finish<-function(beta,code=0L,message="") {
        ev<-evaluate(beta)
        score<-if(ev$valid && length(free)) max(abs(ev$gradient[free]*setup$scale[free])) else if(ev$valid) 0 else Inf
        ok<-ev$valid && code==0L && is.finite(score) && score<=ctl$score.tol
        if(!ok && !nzchar(message)) message<-if(!ev$valid) ev$message else if(code!=0L)
            paste("nuisance optimizer did not converge (code",code,")") else "scaled adjusted score exceeds score.tol"
        list(coefficients=beta,value=if(ev$valid) ev$value else NA_real_,converged=ok,
             score=score,code=code,evaluations=calls,message=message)
    }
    initial<-evaluate(b)
    if(!initial$valid) return(finish(b,1L,initial$message))
    if(!length(free) || max(abs(initial$gradient[free]*setup$scale[free]))<=ctl$score.tol)
        return(finish(b))
    unpack<-function(gamma) {bb<-b;bb[free]<-gamma*setup$scale[free];bb}
    fn<-function(gamma) {e<-evaluate(unpack(gamma));if(e$valid) initial$value-e$value else Inf}
    gr<-function(gamma) {
        e<-evaluate(unpack(gamma));if(!e$valid) stop(e$message,call.=FALSE)
        -as.numeric(e$gradient[free])*setup$scale[free]
    }
    remaining<-ctl$maxit
    repeat {
        result<-tryCatch(optim(b[free]/setup$scale[free],fn,gr,method="BFGS",
            control=list(maxit=remaining,reltol=ctl$reltol)),error=function(e)e)
        if(inherits(result,"error")) return(finish(b,1L,conditionMessage(result)))
        b<-unpack(result$par)
        fitted<-finish(b,result$convergence,if(is.null(result$message)) "" else result$message)
        remaining<-remaining-max(1L,result$counts[["function"]])
        if(fitted$converged || result$convergence!=0L || remaining<=0L) return(fitted)
        # Relative objective convergence can precede score convergence when
        # the constraint has already caused a substantial likelihood drop.
        # Restart the same optimizer with the objective centered at its new
        # starting value, retaining a finite remaining evaluation budget.
        initial<-evaluate(b)
    }
}
.firth_profile_lr <- function(full,restricted,control) {
    if(!restricted$converged) stop(restricted$message,call.=FALSE)
    statistic<-2*(full$value-restricted$value)
    error<-max(control$score.tol^2,256*.Machine$double.eps*(1+abs(full$value)))
    if(statistic < -error) stop("constrained objective exceeds the unrestricted maximum",call.=FALSE)
    max(0,statistic)
}
.fastglm_profile_confint <- function(object,parm,level,control=list()) {
    if(length(level)!=1L || !is.finite(level) || level<=0 || level>=1) stop("'level' must lie between zero and one",call.=FALSE)
    setup<-.firth_profile_setup(object,control);parm<-.firth_profile_parameters(parm,setup$coefficients)
    full<-.firth_profile_fit(setup);target<-qchisq(level,1)
    out<-matrix(NA_real_,length(parm),2L,dimnames=list(names(setup$coefficients)[parm],paste0(100*c((1-level)/2,(1+level)/2)," %")))
    diagnostics<-list()
    for(k in seq_along(parm)) for(side in 1:2) {
        j<-parm[k];latest<-full;lr<-NA_real_;value<-NA_real_;root.iterations<-0L
        result<-tryCatch({
            if(!full$converged) stop(paste("unrestricted fit:",full$message),call.=FALSE)
            center<-full$coefficients[j];distance<-sqrt(target)
            evaluate<-function(standardized) {
                theta<-center+standardized*setup$scale[j]
                latest<<-.firth_profile_fit(setup,j,theta,start=latest$coefficients)
                lr<<-.firth_profile_lr(full,latest,setup$control)
                lr-target
            }
            direction<-if(side==1L) -1 else 1
            endpoint<-direction*distance;gap<-evaluate(endpoint)
            for(iter in seq_len(setup$control$max.expand)) {
                if(gap>=0) break
                distance<-2*distance;endpoint<-direction*distance;gap<-evaluate(endpoint)
            }
            if(gap<0) stop("profile endpoint could not be bracketed within max.expand",call.=FALSE)
            interval<-sort(c(0,endpoint))
            root<-uniroot(evaluate,interval=interval,
                f.lower=if(side==1L) gap else -target,f.upper=if(side==1L) -target else gap,
                tol=setup$control$root.tol,maxiter=setup$control$max.root,check.conv=TRUE)
            value<-center+root$root*setup$scale[j];root.iterations<-root$iter;evaluate(root$root)
            out[k,side]<-value
            ""
        },error=conditionMessage)
        diagnostics[[length(diagnostics)+1L]]<-data.frame(parameter=names(setup$coefficients)[j],
            side=c("lower","upper")[side],converged=!nzchar(result),estimate=out[k,side],
            statistic=lr,target=target,lr.error=lr-target,score=latest$score,root.iterations=root.iterations,
            optimizer.code=latest$code,message=result,stringsAsFactors=FALSE)
    }
    diagnostics<-do.call(rbind,diagnostics)
    if(any(!diagnostics$converged)) warning(sum(!diagnostics$converged)," profile endpoints failed; inspect attr(result, 'diagnostics')",call.=FALSE)
    attr(out,"diagnostics")<-diagnostics;attr(out,"fit.diagnostics")<-full
    attr(out,"method")<-"Firth penalized profile likelihood"
    attr(out,"level")<-level
    class(out)<-c("fastglm_confint","matrix","array")
    out
}
#' Profile likelihood inference
#'
#' Tests one or more coefficient values by comparing an unrestricted maximum
#' with the maximum under those constraints. The remaining coefficients are
#' re-estimated. Binomial-logit Firth fits use their penalized likelihood;
#' ordinal maximum-likelihood fits use their ordinary likelihood. For intervals, use
#' `confint(object, method = "profile")`; the default `confint()` method remains
#' Wald inference.
#'
#' @param object a converged binomial-logit Firth model or an ordinal
#'   maximum-likelihood model, with a retained dense design matrix and response
#'   and a full-rank design. Ordinal models require every category to be observed.
#' @param parm coefficient names or indices. The test treats all selected
#'   coefficients jointly; confidence intervals profile each separately.
#'   Omit `parm` when supplying an ordinal `contrast` matrix.
#' @param null hypothesized coefficient values, in the order specified by
#'   `parm`, or one value per row of `contrast`. A scalar is repeated.
#' @param control named list of numerical controls: `maxit = 300` limits each
#'   nuisance optimization; `reltol = 1e-12` is its objective tolerance;
#'   `score.tol = 1e-5` bounds the adjusted score after scaling by coefficient
#'   standard errors; `root.tol = 1e-6` is the endpoint tolerance in these same
#'   standardized units; `max.expand = 25` limits bracket expansions; and
#'   `max.root = 100` limits root iterations.
#'   Ordinal controls are described in [fastglm_score_test()].
#' @param contrast optional numeric vector or full-row-rank numeric matrix
#'   for ordinal tests. Columns correspond to the full coefficient vector;
#'   names may select a subset. Each matrix row specifies a linear hypothesis. For example,
#'   differences between a predictor's threshold-specific slopes test whether
#'   that predictor has a common effect while other effects remain unrestricted.
#' @details For binomial-logit Firth models the objective is the binomial log likelihood plus one half of
#'   the log determinant of the full Fisher information matrix. Tested columns
#'   remain in that determinant when their coefficients are constrained.
#'   Grouped binomial responses, frequency weights, and offsets retain their
#'   fitted meanings. Effective successes and failures must be nonnegative
#'   integer counts, allowing arithmetic round-off. Fractional power or survey
#'   weights are unsupported. Zero-weight observations make no contribution.
#'
#'   For ordinal maximum-likelihood fits, the objective is the joint
#'   cumulative-link log likelihood, including partial proportional-odds
#'   and other nonparallel cumulative-link models. Thresholds and untested slopes are
#'   re-estimated subject to the selected coefficient constraints. Logit,
#'   probit, and complementary log-log links, grouped category counts,
#'   frequency weights, and offsets retain their fitted meanings. Ordinal
#'   mean bias reduction does not generally maximize a penalized likelihood
#'   and is unsupported by this profile method; [fastglm_score_test()] provides
#'   an analytic score procedure with an explicit null-estimation convention.
#'
#'   The likelihood-ratio statistic uses an asymptotic chi-square reference
#'   with one degree of freedom per constraint. Profile intervals invert its
#'   one-parameter version. These are approximations under an independent
#'   binomial or ordinal sampling model, including frequency-weighted observations; they
#'   are not exact finite-sample or cluster-robust procedures.
#'
#'   Sparse, mapped, and streaming designs are currently unsupported rather
#'   than converted to dense matrices. Native evaluation uses additional
#'   memory of order the squared number of coefficients plus a fixed block
#'   of rows. Ordinal constrained inference also creates a standardized dense
#'   design, requiring additional storage proportional to observations times
#'   predictors; it does not construct an observation-by-category-by-parameter
#'   array.
#'
#'   Every constrained optimization checks the ordinary or adjusted score of the free
#'   coefficients. The unrestricted fit is checked and refined when needed.
#'   Failed tests return missing statistics and p-values with a warning;
#'   failed confidence limits are missing, with endpoint diagnostics attached.
#'   No Wald replacement is made.
#'   Optimization starts from the fitted solution; convergence diagnostics do
#'   not establish that a global maximum has been found in a multimodal problem.
#' @returns `fastglm_profile_test()` returns a `fastglm_test` object inheriting
#'   from `htest`, with a compact printed table and `as.data.frame()` and
#'   optional `broom::tidy()` extraction. It retains the statistic,
#'   degrees of freedom, p-value, coefficient estimates, hypothesized values,
#'   the full and constrained log likelihoods (penalized for Firth fits, excluding constants
#'   independent of coefficients), and `diagnostics` containing convergence,
#'   scaled scores, optimizer codes, evaluation counts, and messages.
#'   `confint(..., method = "profile")` returns a two-column `fastglm_confint` matrix with
#'   `diagnostics`, `fit.diagnostics`, and `method` attributes. Each diagnostic
#'   row identifies the coefficient and endpoint, convergence, likelihood-ratio
#'   statistic and target, root error, scaled nuisance score, optimizer code,
#'   and failure message.
#' @references Heinze, G. and Schemper, M. (2002). A solution to the problem of
#'   separation in logistic regression. Statistics in Medicine, 21, 2409--2419.
#'   \doi{10.1002/sim.1047}.
#' @examples
#' d <- data.frame(y = rep(0:1, each = 10), x = rep(0:1, each = 10))
#' fit <- fastglm(y ~ x, d, family = binomial(), firth = TRUE)
#' confint(fit, parm = "x", method = "profile")
#' fastglm_profile_test(fit, parm = "x")
#' @md
#' @export
fastglm_profile_test <- function(object,parm,null=0,control=list(),contrast=NULL) {
    data.name<-deparse(substitute(object))
    if(inherits(object,"fastglm_ordinal"))
        return(.ordinal_profile_test(object,if(missing(parm)) NULL else parm,null,control,data.name,contrast=contrast))
    if(!is.null(contrast))
        stop("linear profile contrasts are currently available only for ordinal fits",call.=FALSE)
    setup<-.firth_profile_setup(object,control);parm<-.firth_profile_parameters(parm,setup$coefficients)
    if(length(null)==1L) null<-rep(null,length(parm))
    if(!is.numeric(null) || length(null)!=length(parm) || any(!is.finite(null)))
        stop("'null' must contain one finite value per tested coefficient",call.=FALSE)
    names(null)<-names(setup$coefficients)[parm]
    full<-.firth_profile_fit(setup);restricted<-NULL;statistic<-NA_real_
    message<-tryCatch({
        if(!full$converged) stop(paste("unrestricted fit:",full$message),call.=FALSE)
        restricted<-.firth_profile_fit(setup,parm,null,start=full$coefficients)
        statistic<-.firth_profile_lr(full,restricted,setup$control)
        ""
    },error=conditionMessage)
    if(nzchar(message)) warning("profile test failed: ",message,call.=FALSE)
    structure(list(statistic=c(`penalized likelihood ratio`=statistic),parameter=c(df=length(parm)),
        p.value=if(is.finite(statistic)) pchisq(statistic,length(parm),lower.tail=FALSE) else NA_real_,
        estimate=full$coefficients[parm],null.value=null,alternative="two.sided",
        method="Firth penalized likelihood ratio test (asymptotic chi-square reference)",data.name=data.name,
        loglik=c(full=full$value,restricted=if(is.null(restricted)) NA_real_ else restricted$value),
        null.estimation="Firth penalized likelihood",
        diagnostics=list(converged=!nzchar(message),message=message,full=full,restricted=restricted)),
        class=c("fastglm_test","htest"))
}
