#' Tidy summaries for fastglm models
#'
#' Optional methods for the broom generics. Formula fits also retain the
#' standard glm model structure for tools that consume coefficients, terms,
#' prediction, and covariance methods.
#' @param x a fastglm model.
#' @param conf.int include Wald confidence limits.
#' @param conf.level confidence level.
#' @param exponentiate exponentiate estimates and confidence limits.
#' @param ... additional arguments.
#' @returns A data frame with one row per coefficient (`tidy`) or one row per model (`glance`).
#' @exportS3Method broom::tidy fastglm
#' @name fastglm-tidy
#' @rdname fastglm-tidy
tidy.fastglm <- function(x,conf.int=FALSE,conf.level=.95,exponentiate=FALSE,...) {
    tab <- summary(x)$coefficients
    out <- data.frame(term=rownames(tab),estimate=tab[,1],std.error=tab[,2],
                       statistic=tab[,3],p.value=tab[,4],row.names=NULL)
    if(conf.int) {
        ci <- confint(x,parm=out$term,level=conf.level)
        out$conf.low <- ci[,1];out$conf.high <- ci[,2]
    }
    if(exponentiate) for(nm in intersect(c("estimate","conf.low","conf.high"),names(out))) out[[nm]] <- exp(out[[nm]])
    out
}
#' @exportS3Method broom::tidy fastglmFit
#' @rdname fastglm-tidy
tidy.fastglmFit <- tidy.fastglm
#' @exportS3Method broom::glance fastglm
#' @rdname fastglm-tidy
glance.fastglm <- function(x,...) {
    data.frame(null.deviance=x$null.deviance,df.null=x$df.null,
        logLik=as.numeric(logLik(x)),AIC=AIC(x),BIC=BIC(x),
        deviance=x$deviance,df.residual=x$df.residual,nobs=nobs(x))
}
#' @exportS3Method broom::glance fastglmFit
#' @rdname fastglm-tidy
glance.fastglmFit <- glance.fastglm
#' @exportS3Method broom::tidy fastglm_hurdle
#' @rdname fastglm-tidy
tidy.fastglm_hurdle <- function(x,conf.int=FALSE,conf.level=.95,exponentiate=FALSE,...) {
    tab <- summary(x)$coefficients
    out <- data.frame(term=rownames(tab),estimate=tab[,1],std.error=tab[,2],
                       statistic=tab[,3],p.value=tab[,4],row.names=NULL)
    # Two-component models prefix each coefficient with its component.
    if(!inherits(x,"fastglm_truncated"))
        out <- cbind(component=sub("_.*$","",out$term),out)
    if(conf.int) {
        z <- stats::qnorm(1-(1-conf.level)/2)
        out$conf.low <- out$estimate-z*out$std.error;out$conf.high <- out$estimate+z*out$std.error
    }
    if(exponentiate) for(nm in intersect(c("estimate","conf.low","conf.high"),names(out))) out[[nm]] <- exp(out[[nm]])
    out
}
#' @exportS3Method broom::tidy fastglm_zi
#' @rdname fastglm-tidy
tidy.fastglm_zi <- tidy.fastglm_hurdle
#' @exportS3Method broom::tidy fastglm_truncated
#' @rdname fastglm-tidy
tidy.fastglm_truncated <- tidy.fastglm_hurdle
#' @exportS3Method broom::glance fastglm_hurdle
#' @rdname fastglm-tidy
glance.fastglm_hurdle <- function(x,...) {
    ll <- logLik(x)
    out <- data.frame(logLik=as.numeric(ll),AIC=AIC(ll),BIC=BIC(ll),nobs=nobs(x),
                      converged=isTRUE(x$converged))
    if(identical(x$dist,"negbin")) out$theta <- x$theta
    out
}
#' @exportS3Method broom::glance fastglm_zi
#' @rdname fastglm-tidy
glance.fastglm_zi <- glance.fastglm_hurdle
#' @exportS3Method broom::glance fastglm_truncated
#' @rdname fastglm-tidy
glance.fastglm_truncated <- glance.fastglm_hurdle
