# Fit cumulative links in their threshold/common-slope coordinates. Centering
# and scaling are linear changes of coordinates, under which mean bias reduction
# is equivariant. A log-spacing transformation would change its target.
.ordinal_fit_core <- function(x,y,weights=NULL,offset=NULL,levels=NULL,
                              firth=FALSE,tol=1e-8,maxit=200L,
                              block.size=2048L,control=list(),cache=NULL,link="logit") {
    link.code <- .ordinal_link_code(link); link <- c("logit","probit","cloglog")[link.code+1L]
    if(!is.matrix(x) || !is.numeric(x) || is.complex(x))
        stop("ordinal fitting requires a finite dense numeric matrix without a location intercept",call.=FALSE)
    n<-nrow(x);p<-ncol(x)
    if(!n) stop("ordinal fitting requires observations",call.=FALSE)
    if(p && is.null(colnames(x))) colnames(x)<-paste0("x",seq_len(p))
    response<-.ordinal_response_data(y,weights,levels,firth)
    if(response$n!=n) stop("ordinal response length does not match the design",call.=FALSE)
    if(is.null(offset)) offset<-numeric(n)
    if(!is.numeric(offset) || is.complex(offset) || !is.null(dim(offset)) || length(offset)!=n || any(!is.finite(offset)))
        stop("ordinal location offsets must be finite and match the observations",call.=FALSE)
    offset<-as.numeric(offset)
    positive.scalar<-function(z,integer=FALSE) is.numeric(z) && !is.complex(z) && length(z)==1L &&
        is.finite(z) && z>0 && (!integer || (z<=.Machine$integer.max && z==floor(z)))
    if(!positive.scalar(tol)) stop("'tol' must be finite and positive",call.=FALSE)
    if(!positive.scalar(maxit,TRUE)) stop("'maxit' must be a positive integer",call.=FALSE)
    if(!positive.scalar(block.size,TRUE)) stop("'block.size' must be a positive integer",call.=FALSE)
    defaults<-list(score.tol=tol,step.tol=tol,max.step=4,max.halves=35L,
                   condition.tol=1e-14,standardize=TRUE,trace=FALSE)
    if(!is.list(control) || (length(control) &&
       (is.null(names(control)) || anyNA(names(control)) || any(!nzchar(names(control))) ||
        anyDuplicated(names(control)))) || length(setdiff(names(control),names(defaults))))
        stop("unknown or duplicated ordinal optimizer control",call.=FALSE)
    ctl<-utils::modifyList(defaults,control)
    for(nm in c("score.tol","step.tol","max.step","max.halves","condition.tol"))
        if(!positive.scalar(ctl[[nm]],nm=="max.halves"))
            stop("invalid ordinal optimizer control: ",nm,call.=FALSE)
    for(nm in c("standardize","trace"))
        if(!is.logical(ctl[[nm]]) || length(ctl[[nm]])!=1L || is.na(ctl[[nm]]))
            stop("invalid ordinal optimizer flag: ",nm,call.=FALSE)

    totals<-response$totals
    setup<-.ordinal_design_setup(x,totals,ctl$standardize,cache)
    frequency<-setup$frequency;good<-setup$positive
    center<-setup$center;scale<-setup$scale;xx<-setup$x
    # Native block products can touch every row even though score accumulation
    # skips empty ones. Keep their working inputs benign; retain original offsets
    # and the public design separately for subsequent prediction.
    working.offset<-offset;working.offset[!good]<-0
    K<-length(response$levels);q<-K-1L;d<-q+p
    cumulative<-cumsum(response$category.totals)[seq_len(q)]
    remaining<-rev(cumsum(rev(response$category.totals)))[-1L]
    # Half-count starts are empirical, including when an endpoint is empty.
    # Calculate the two log terms separately to avoid rounding probabilities
    # exactly to one at large total frequencies.
    alpha<-.ordinal_start_thresholds(cumulative,remaining,link)+sum(offset[good]*(totals[good]/frequency))
    fit<-ordinal_native_fit(xx,response$codes,response$counts,response$weights,
        working.offset,K,c(alpha,numeric(p)),firth,ctl$score.tol,ctl$step.tol,as.integer(maxit),
        as.integer(block.size),ctl$max.step,as.integer(ctl$max.halves),ctl$condition.tol,ctl$trace,link.code)

    # Original effective coordinates = T times the centered/scaled coordinates.
    # Form the inverse analytically: solve(T) can reject a valid transformation
    # solely because a predictor has a very large origin or small measurement unit.
    T<-Ti<-diag(d)
    if(p) {
        T[seq_len(q),q+seq_len(p)]<-matrix(center/scale,q,p,byrow=TRUE)
        T[q+seq_len(p),q+seq_len(p)]<-diag(1/scale,nrow=p)
        Ti[seq_len(q),q+seq_len(p)]<-matrix(-center,q,p,byrow=TRUE)
        Ti[q+seq_len(p),q+seq_len(p)]<-diag(scale,nrow=p)
    }
    parameter.names<-c(paste(response$levels[-K],response$levels[-1L],sep="|"),colnames(x))
    effective<-list(coefficients=drop(T%*%fit$coefficients),
        covariance=T%*%fit$cov.unscaled%*%t(T),
        information=crossprod(Ti,fit$information%*%Ti),
        score=drop(crossprod(Ti,fit$score)),
        ordinary.score=drop(crossprod(Ti,fit$ordinary.score)),
        adjustment=drop(crossprod(Ti,fit$adjustment)),step=drop(T%*%fit$step),
        levels=response$levels)
    for(nm in c("coefficients","score","ordinary.score","adjustment","step"))
        names(effective[[nm]])<-parameter.names
    for(nm in c("covariance","information")) dimnames(effective[[nm]])<-list(parameter.names,parameter.names)
    effective$alpha<-effective$coefficients[seq_len(q)]
    effective$beta<-effective$coefficients[q+seq_len(p)]
    # Duplicate reduced cutpoints for an empty interior category. The expanded
    # covariance describes only estimable tangent directions. In particular it
    # does not identify separate tied thresholds or provide their regular tests.
    original.K<-length(response$original.levels)
    expansion<-c(response$threshold.map,q+seq_len(p))
    original.names<-c(paste(response$original.levels[-original.K],
                           response$original.levels[-1L],sep="|"),colnames(x))
    coefficients<-effective$coefficients[expansion];names(coefficients)<-original.names
    mapped.covariance<-effective$covariance[expansion,expansion,drop=FALSE]
    dimnames(mapped.covariance)<-list(original.names,original.names)
    working<-list(coefficients=as.numeric(fit$coefficients),center=center,scale=scale,
                  covariance=fit$cov.unscaled,information=fit$information,
                  score=as.numeric(fit$score),ordinary.score=as.numeric(fit$ordinary.score),
                  adjustment=as.numeric(fit$adjustment),step=as.numeric(fit$step),
                  scaled.score=fit$scaled.score,predictor.step=fit$predictor.step,
                  information.rcond=fit$information.rcond)
    names(working$coefficients)<-parameter.names
    working$alpha<-working$coefficients[seq_len(q)]
    working$beta<-working$coefficients[q+seq_len(p)]
    colnames(fit$history)<-c("loglik","scaled.score","predictor.step","minimum.gap")
    if(response$boundary)
        warning("empty interior ordinal categories were collapsed for fitting; their labels are retained with zero probability and coincident thresholds",call.=FALSE)
    if(!fit$converged)
        warning("ordinal fitting did not converge: ",fit$convergence.reason,call.=FALSE)
    list(coefficients=coefficients,alpha=coefficients[seq_len(original.K-1L)],
         beta=coefficients[original.K-1L+seq_len(p)],
         levels=response$original.levels,effective.levels=response$levels,
         threshold.map=response$threshold.map,retained.categories=response$retained.categories,
         collapsed.categories=response$collapsed.categories,boundary=response$boundary,
         empty.end.categories=response$empty.end.categories,
         effective=effective,working=working,mapped.covariance=mapped.covariance,
         information=effective$information,cov.unscaled=mapped.covariance,
         score=effective$score,ordinary.score=effective$ordinary.score,
         adjustment=effective$adjustment,scaled.score=fit$scaled.score,
         predictor.step=fit$predictor.step,information.rcond=fit$information.rcond,
         minimum.log.probability=fit$minimum.log.probability,
         converged=fit$converged,convergence.reason=fit$convergence.reason,
         iterations=fit$iterations,evaluations=fit$evaluations,fallbacks=fit$fallbacks,
         last.rejected.evaluation=fit$last.rejected.evaluation,history=fit$history,
         loglik=fit$loglik,rank=d,df.residual=frequency-d,nobs=sum(good),
         frequency=frequency,weights=response$weights,offset=offset,response=response,category=response,
         preparation.reused=setup$reused,
         firth=firth,link=link,method=if(firth) "meanBR" else "ML",control=ctl,
         tol=tol,maxit=as.integer(maxit),block.size=as.integer(block.size))
}

.ordinal_design_setup <- function(x,totals,standardize,cache=NULL,nonparallel=NULL) {
    key<-list(x=x,totals=totals,standardize=standardize,nonparallel=nonparallel)
    entry<-if(is.environment(cache)) cache$entry
    if(!is.null(entry) && identical(entry$key,key)) {
        # Refresh references after deserialization or an equal replacement, so
        # future comparisons need not rescan an equivalent copied matrix.
        cache$entry<-list(key=key,value=entry$value)
        value<-entry$value;value$reused<-TRUE
        return(value)
    }
    # Retaining X in the key lets R's copy-on-write rules distinguish a changed
    # matrix. Unchanged references do not require another scan of all entries.
    if(is.null(entry) || !identical(entry$key$x,x)) {
        if(!is.matrix(x) || !is.numeric(x) || is.complex(x) || any(!is.finite(x)))
            stop("ordinal fitting requires a finite dense numeric matrix without a location intercept",call.=FALSE)
    }
    n<-nrow(x);p<-ncol(x);frequency<-sum(totals);good<-totals>0
    center<-numeric(p);scale<-rep(1,p);xx<-x
    if(p) {
        # Rank is assessed after centering and scaling, so a change of units or
        # origin does not by itself imply nonidentifiability. Zero-frequency
        # rows must be excluded before squaring finite extreme values.
        active<-if(all(good)) x else x[good,,drop=FALSE]
        center<-colSums(active*(totals[good]/frequency))
        centered<-sweep(active,2,center,"-")
        scale<-sqrt(colSums(centered^2*(totals[good]/frequency)))
        if(any(!is.finite(scale)|scale<=0))
            stop("the ordinal location design contains a redundant constant column or cannot be scaled",call.=FALSE)
        standardized<-sweep(centered,2,scale,"/")
        if(qr(cbind(1,standardized))$rank!=p+1L)
            stop("the ordinal location design must be full rank with a separate threshold intercept",call.=FALSE)
        if(standardize) {
            if(all(good)) xx<-standardized else {
                xx<-matrix(0,n,p,dimnames=dimnames(x))
                xx[good,]<-standardized
            }
        } else {
            center<-numeric(p);scale<-rep(1,p)
            if(any(!good)) xx[!good,]<-0
        }
    }
    if(is.integer(xx)) storage.mode(xx)<-"double"
    value<-list(x=xx,center=center,scale=scale,positive=good,frequency=frequency,reused=FALSE)
    # Publish key and value together only after every check succeeds. A failed
    # change cannot pair an invalid new key with an older prepared design.
    if(is.environment(cache)) {
        cache$entry<-list(key=key,value=value)
        cache$builds<-if(is.null(cache$builds)) 1L else cache$builds+1L
    }
    value
}
