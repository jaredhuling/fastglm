.ordinal_nonparallel_columns <- function(nonparallel,columns) {
    if(is.null(nonparallel) || !length(nonparallel)) return(integer())
    if(inherits(nonparallel,"formula"))
        stop("matrix fits select nonparallel predictors by column name or index",call.=FALSE)
    if(is.character(nonparallel)) {
        if(anyNA(nonparallel) || any(!nzchar(nonparallel)))
            stop("nonparallel column names must be nonmissing",call.=FALSE)
        selected <- match(nonparallel,columns)
    } else if(is.numeric(nonparallel) && !is.complex(nonparallel) && is.null(dim(nonparallel))) {
        selected <- nonparallel
    } else stop("nonparallel predictors must be column names or indices",call.=FALSE)
    if(anyNA(selected) || any(!is.finite(selected)) || any(selected<1 | selected>length(columns) | selected!=floor(selected)))
        stop("nonparallel predictors must select existing design columns",call.=FALSE)
    if(anyDuplicated(selected)) stop("nonparallel predictors must be distinct",call.=FALSE)
    sort(as.integer(selected))
}

.ordinal_nonparallel_terms <- function(nonparallel,terms,assignment) {
    if(is.null(nonparallel) || !length(nonparallel)) return(list(columns=integer(),terms=character()))
    if(!inherits(nonparallel,"formula") || length(nonparallel)!=2L)
        stop("formula fits require a one-sided nonparallel formula selecting complete model terms",call.=FALSE)
    if(length(attr(stats::terms(nonparallel,allowDotAsName=TRUE),"offset")))
        stop("offset terms cannot have nonparallel coefficients",call.=FALSE)
    selected <- stats::terms(update.formula(formula(terms),nonparallel),keep.order=TRUE)
    key <- function(tt) {
        labels <- attr(tt,"term.labels"); factors <- attr(tt,"factors")
        vapply(seq_along(labels),function(j) paste(sort(rownames(factors)[factors[,j]!=0]),collapse="\034"),character(1))
    }
    indices <- match(key(selected),key(terms))
    if(anyNA(indices)) stop("nonparallel formula must select complete terms already present in the model",call.=FALSE)
    list(columns=which(assignment%in%indices),terms=attr(terms,"term.labels")[indices])
}

.ordinal_partial_fit_matrix <- function(x,y,weights,offset,levels,firth,tol,maxit,
                                         block.size,control,link,nonparallel) {
    if(!is.logical(firth) || length(firth)!=1L || is.na(firth))
        stop("'firth' must be TRUE or FALSE",call.=FALSE)
    link.code <- .ordinal_link_code(link);link <- c("logit","probit","cloglog")[link.code+1L]
    # Check the unreduced response scale before choosing any category mapping.
    response <- .ordinal_response_data(y,weights,levels,firth=TRUE)
    if(any(response$original.category.totals<=0))
        stop("nonparallel ordinal fitting requires every declared category to have positive frequency",call.=FALSE)
    n <- nrow(x);p <- ncol(x)
    if(response$n!=n) stop("ordinal response length does not match the design",call.=FALSE)
    if(is.null(offset)) offset <- numeric(n)
    if(!is.numeric(offset) || is.complex(offset) || !is.null(dim(offset)) ||
       length(offset)!=n || any(!is.finite(offset)))
        stop("ordinal location offsets must be finite and match the observations",call.=FALSE)
    offset <- as.numeric(offset)
    positive <- function(z,integer=FALSE) is.numeric(z) && !is.complex(z) && length(z)==1L &&
        is.finite(z) && z>0 && (!integer || (z<=.Machine$integer.max && z==floor(z)))
    if(!positive(tol)) stop("'tol' must be finite and positive",call.=FALSE)
    if(!positive(maxit,TRUE)) stop("'maxit' must be a positive integer",call.=FALSE)
    if(!positive(block.size,TRUE)) stop("'block.size' must be a positive integer",call.=FALSE)
    defaults <- list(score.tol=tol,step.tol=tol,max.step=4,max.halves=35L,
        condition.tol=1e-14,standardize=TRUE,trace=FALSE)
    if(!is.list(control) || (length(control) && (is.null(names(control)) || anyNA(names(control)) ||
       any(!nzchar(names(control))) || anyDuplicated(names(control)))) ||
       length(setdiff(names(control),names(defaults))))
        stop("unknown or duplicated ordinal optimizer control",call.=FALSE)
    ctl <- utils::modifyList(defaults,control)
    for(nm in c("score.tol","step.tol","max.step","max.halves","condition.tol"))
        if(!positive(ctl[[nm]],nm=="max.halves"))
            stop("invalid ordinal optimizer control: ",nm,call.=FALSE)
    for(nm in c("standardize","trace"))
        if(!is.logical(ctl[[nm]]) || length(ctl[[nm]])!=1L || is.na(ctl[[nm]]))
            stop("invalid ordinal optimizer flag: ",nm,call.=FALSE)
    setup <- .ordinal_design_setup(x,response$totals,ctl$standardize)
    center <- setup$center;scale <- setup$scale;xx <- setup$x;good <- setup$positive
    common <- setdiff(seq_len(p),nonparallel);pc <- length(common);pn <- length(nonparallel)
    K <- length(response$levels);q <- K-1L;d <- q+pc+q*pn
    working.offset <- offset;working.offset[!good] <- 0
    cumulative <- cumsum(response$category.totals)[seq_len(q)]
    remaining <- rev(cumsum(rev(response$category.totals)))[-1L]
    alpha <- .ordinal_start_thresholds(cumulative,remaining,link)+
        sum(offset[good]*(response$totals[good]/setup$frequency))
    fit <- ordinal_partial_native_fit(xx[,common,drop=FALSE],xx[,nonparallel,drop=FALSE],
        response$codes,response$counts,response$weights,working.offset,K,c(alpha,numeric(d-q)),
        firth,ctl$score.tol,ctl$step.tol,as.integer(maxit),as.integer(block.size),
        ctl$max.step,as.integer(ctl$max.halves),ctl$condition.tol,ctl$trace,link.code)
    T <- Ti <- diag(d)
    if(pc) {
        at <- q+seq_len(pc)
        T[seq_len(q),at] <- matrix(center[common]/scale[common],q,pc,byrow=TRUE)
        T[at,at] <- diag(1/scale[common],nrow=pc)
        Ti[seq_len(q),at] <- matrix(-center[common],q,pc,byrow=TRUE)
        Ti[at,at] <- diag(scale[common],nrow=pc)
    }
    for(j in seq_len(q)) {
        at <- q+pc+(j-1L)*pn+seq_len(pn)
        T[j,at] <- center[nonparallel]/scale[nonparallel]
        T[at,at] <- diag(1/scale[nonparallel],nrow=pn)
        Ti[j,at] <- -center[nonparallel]
        Ti[at,at] <- diag(scale[nonparallel],nrow=pn)
    }
    threshold.names <- paste(response$levels[-K],response$levels[-1L],sep="|")
    gamma.names <- unlist(lapply(threshold.names,function(label)
        paste0("nonparallel:",colnames(x)[nonparallel],":",label)),use.names=FALSE)
    nms <- c(paste0("threshold:",threshold.names),colnames(x)[common],gamma.names)
    if(anyDuplicated(nms)) stop("predictor names conflict with generated ordinal parameter names",call.=FALSE)
    effective <- list(coefficients=drop(T%*%fit$coefficients),covariance=T%*%fit$cov.unscaled%*%t(T),
        information=crossprod(Ti,fit$information%*%Ti),score=drop(crossprod(Ti,fit$score)),
        ordinary.score=drop(crossprod(Ti,fit$ordinary.score)),adjustment=drop(crossprod(Ti,fit$adjustment)),
        step=drop(T%*%fit$step),levels=response$levels)
    for(nm in c("coefficients","score","ordinary.score","adjustment","step")) names(effective[[nm]]) <- nms
    for(nm in c("covariance","information")) dimnames(effective[[nm]]) <- list(nms,nms)
    parts <- function(b) list(alpha=setNames(b[seq_len(q)],threshold.names),
        beta=setNames(b[q+seq_len(pc)],colnames(x)[common]),
        gamma=matrix(b[q+pc+seq_len(q*pn)],pn,q,dimnames=list(colnames(x)[nonparallel],threshold.names)))
    effective <- c(effective,parts(effective$coefficients))
    working <- c(list(coefficients=setNames(as.numeric(fit$coefficients),nms),
        center=center,scale=scale,T=T,Ti=Ti,covariance=fit$cov.unscaled,information=fit$information,
        score=as.numeric(fit$score),ordinary.score=as.numeric(fit$ordinary.score),
        adjustment=as.numeric(fit$adjustment),step=as.numeric(fit$step),
        scaled.score=fit$scaled.score,predictor.step=fit$predictor.step,information.rcond=fit$information.rcond),
        parts(as.numeric(fit$coefficients)))
    colnames(fit$history) <- c("loglik","scaled.score","predictor.step","minimum.gap")
    if(!isTRUE(fit$converged)) warning("nonparallel ordinal fitting did not converge: ",fit$convergence.reason,call.=FALSE)
    object <- list(coefficients=effective$coefficients,alpha=effective$alpha,beta=effective$beta,
        gamma=effective$gamma,thresholds=effective$alpha,nonparallel=nonparallel,common=common,
        levels=response$levels,effective.levels=response$levels,threshold.map=seq_len(q),
        retained.categories=seq_len(K),collapsed.categories=integer(),empty.end.categories=integer(),
        boundary=FALSE,tied.thresholds=rep(FALSE,q),effective=effective,working=working,
        mapped.covariance=effective$covariance,cov.unscaled=effective$covariance,
        information=effective$information,score=effective$score,ordinary.score=effective$ordinary.score,
        adjustment=effective$adjustment,scaled.score=fit$scaled.score,predictor.step=fit$predictor.step,
        information.rcond=fit$information.rcond,minimum.gap=fit$minimum.gap,
        minimum.log.probability=fit$minimum.log.probability,converged=fit$converged,
        convergence.reason=fit$convergence.reason,iterations=fit$iterations,evaluations=fit$evaluations,
        fallbacks=fit$fallbacks,last.rejected.evaluation=fit$last.rejected.evaluation,history=fit$history,
        loglik=fit$loglik,rank=d,df.residual=setup$frequency-d,nobs=setup$frequency,
        nobs.rows=sum(good),frequency=setup$frequency,weights=response$weights,offset=offset,
        response=response,category=response,x=x,y=y,design.columns=colnames(x),preparation.reused=FALSE,
        firth=firth,link=link,method=if(firth) "meanBR" else "ML",control=ctl,tol=tol,
        maxit=as.integer(maxit),block.size=as.integer(block.size))
    class(object) <- c("fastglm_ordinal_partial","fastglm_ordinal")
    object
}

.ordinal_partial_indices <- function(object,component) {
    q <- length(object$alpha);pc <- length(object$beta);d <- length(object$coefficients)
    switch(component,all=seq_len(d),location=q+seq_len(d-q),threshold=seq_len(q),
        common=q+seq_len(pc),nonparallel=q+pc+seq_len(d-q-pc))
}

# The working coordinates are an invertible linear transformation of the
# reported coefficients. Category gradients below use those working coordinates.
.ordinal_partial_probability_block <- function(object,x,offset,what="all",derivatives=TRUE,rows=NULL) {
    xx <- .ordinal_prediction_scale(x,object$working$center,object$working$scale)
    n <- nrow(xx);q <- length(object$alpha);pc <- length(object$beta)
    # Add thresholds, offsets, and both slope blocks in one compensated sum.
    # Separate matrix products can discard small terms before large opposing
    # common and threshold-specific contributions cancel.
    augmented <- cbind(1,xx);slopes <- numeric(ncol(xx))
    if(pc) slopes[object$common] <- object$working$beta
    eta <- matrix(0,n,q)
    for(j in seq_len(q)) {
        slopes[object$nonparallel] <- object$working$gamma[,j]
        eta[,j] <- streaming_qr_eta(augmented,c(object$working$alpha[j],-slopes),-offset)
    }
    location <- matrix(object$working$alpha,n,q,byrow=TRUE)-eta
    if(any(!is.finite(eta)))
        stop("nonparallel cumulative predictors cannot be represented at the supplied prediction values",call.=FALSE)
    # Common location terms cancel from adjacent differences. Computing gaps
    # directly avoids losing a valid gap after a large common predictor shift.
    gaps <- matrix(0,n,max(0L,q-1L))
    if(q>1L) {
        # Keep both original coefficients in the sum. Rounding their
        # difference first can change the gap at large predictor values.
        varying <- cbind(1,1,xx[,object$nonparallel,drop=FALSE],xx[,object$nonparallel,drop=FALSE])
        for(j in seq_len(q-1L)) gaps[,j] <- streaming_qr_eta(varying,
            c(object$working$alpha[j+1L],-object$working$alpha[j],
                -object$working$gamma[,j+1L],object$working$gamma[,j]),numeric(n))
    }
    invalid <- if(q>1L) which(rowSums(!is.finite(gaps)|gaps<=0)>0L) else integer()
    if(length(invalid)) {
        labels <- if(is.null(rows)) if(!is.null(rownames(x))) rownames(x) else seq_len(n) else rows
        stop("nonparallel cumulative probabilities are not ordered at prediction row(s): ",
            paste(utils::head(labels[invalid],10L),collapse=", "),
            if(length(invalid)>10L) " (additional rows omitted)" else "",
            "; restrict prediction to a valid predictor domain",call.=FALSE)
    }
    link <- .ordinal_link(object)
    if(link=="logit") {
        lc <- -pmax(-eta,0)-log1p(exp(-abs(eta)))
        ls <- -pmax(eta,0)-log1p(exp(-abs(eta)))
        F <- plogis(eta);S <- plogis(-eta);g <- F*S;curvature <- g*(S-F)
    } else if(link=="probit") {
        lc <- pnorm(eta,log.p=TRUE);ls <- pnorm(eta,lower.tail=FALSE,log.p=TRUE)
        F <- exp(lc);S <- exp(ls);g <- dnorm(eta);curvature <- -eta*g
    } else {
        e <- exp(eta);ls <- -e;lc <- eta
        at <- eta>=-36;lc[at] <- .ordinal_log1mexp(-e[at])
        F <- exp(lc);S <- exp(ls);g <- exp(eta-e);curvature <- g*(1-e)
        curvature[is.infinite(e) & g==0] <- 0
    }
    logp <- matrix(NA_real_,n,q+1L);logp[,1L] <- lc[,1L];logp[,q+1L] <- ls[,q]
    if(q>1L) for(j in 2:q) {
        gap <- gaps[,j-1L]
        if(link=="logit") logp[,j] <- lc[,j]+ls[,j-1L]+log(-expm1(-gap))
        else if(link=="cloglog") {
            ld <- eta[,j]+log(-expm1(-gap));logden <- ld
            at <- ld>=-36;logden[at] <- .ordinal_log1mexp(-exp(ld[at]))
            logp[,j] <- ls[,j-1L]+logden
        } else logp[,j] <- ifelse(eta[,j]<=0,
            .ordinal_log_difference(lc[,j],lc[,j-1L]),
            .ordinal_log_difference(ls[,j-1L],ls[,j]))
    }
    list(x=xx,probability=exp(logp),log.probability=logp,cumulative=F,survival=S,
        derivative=g,curvature=curvature,eta=eta,location=location,
        partial=list(common=object$common,nonparallel=object$nonparallel,d=length(object$coefficients)))
}

.ordinal_partial_gradient <- function(parts,j,type) {
    q <- ncol(parts$derivative);pc <- length(parts$partial$common);pn <- length(parts$partial$nonparallel)
    n <- nrow(parts$x);g <- matrix(0,n,parts$partial$d)
    directions <- if(type=="cumulative") j else c(if(j>1L) j-1L,if(j<=q) j)
    for(k in directions) {
        value <- parts$derivative[,k]*(if(type=="probability" && k==j-1L) -1 else 1)
        g[,k] <- g[,k]+value
        if(pc) g[,q+seq_len(pc)] <- g[,q+seq_len(pc),drop=FALSE]-parts$x[,parts$partial$common,drop=FALSE]*value
        at <- q+pc+(k-1L)*pn+seq_len(pn)
        g[,at] <- g[,at,drop=FALSE]-parts$x[,parts$partial$nonparallel,drop=FALSE]*value
    }
    g
}
