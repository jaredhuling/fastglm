.ordinal_partial_require_common <- function(object,operation) {
    if(inherits(object,"fastglm_ordinal_partial"))
        stop(operation," is not yet available for nonparallel ordinal models",call.=FALSE)
    invisible(NULL)
}

.ordinal_partial_test_columns <- function(object,terms) {
    columns <- object$design.columns
    if(!is.null(object$terms)) {
        labels <- attr(object$terms,"term.labels")
        if(is.null(terms)) terms <- labels
        if(!is.character(terms) || !length(terms) || anyNA(terms) || anyDuplicated(terms) ||
           any(!terms%in%labels)) stop("unknown or duplicated ordinal formula term",call.=FALSE)
        assignment <- object$design.assign
        if(length(assignment)!=length(columns))
            stop("ordinal term tests require retained design assignments; supply a named contrast",call.=FALSE)
        selected <- which(assignment%in%match(terms,labels))
    } else {
        if(is.null(terms)) terms <- columns
        if(!is.character(terms) || !length(terms) || anyNA(terms) || anyDuplicated(terms) ||
           any(!terms%in%columns)) stop("unknown or duplicated ordinal predictor column",call.=FALSE)
        selected <- match(terms,columns)
    }
    q <- length(object$alpha)
    design.index <- c(rep(NA_integer_,q),object$common,rep(object$nonparallel,q))
    index <- which(design.index%in%selected)
    list(index=index,terms=terms)
}

.ordinal_partial_profile_setup <- function(object,control) {
    b <- coef(object);q <- length(object$alpha);pc <- length(object$common);pn <- length(object$nonparallel)
    d <- q+pc+q*pn;category <- object$category;x <- object$x;offset <- object$offset
    if(length(b)!=d || any(!is.finite(b)) || object$rank!=d)
        stop("ordinal constrained inference requires a full-rank identifiable model",call.=FALSE)
    if(ncol(x)!=pc+pn || nrow(x)!=category$n || length(offset)!=nrow(x) ||
       length(category$weights)!=nrow(x) || any(!is.finite(x)) || any(!is.finite(offset)) ||
       !identical(sort(c(object$common,object$nonparallel)),seq_len(ncol(x))))
        stop("invalid retained nonparallel ordinal data",call.=FALSE)
    center <- object$working$center;scale <- object$working$scale
    T <- object$working$T;Ti <- object$working$Ti
    if(length(center)!=ncol(x) || length(scale)!=ncol(x) || any(!is.finite(c(center,scale))) ||
       any(scale<=0) || !is.matrix(T) || !is.matrix(Ti) || !identical(dim(T),c(d,d)) ||
       !identical(dim(Ti),c(d,d)) || any(!is.finite(c(T,Ti))))
        stop("ordinal fitting coordinates are unavailable",call.=FALSE)
    x <- .ordinal_prediction_scale(x,center,scale);positive <- category$totals>0
    if(any(!positive)) {x[!positive,] <- 0;offset[!positive] <- 0}
    common <- x[,object$common,drop=FALSE];varying <- x[,object$nonparallel,drop=FALSE]
    storage.mode(common) <- storage.mode(varying) <- "double"
    evaluate <- function(gamma,adjusted=FALSE) tryCatch({
        value <- ordinal_partial_native_evaluate(common,varying,category$codes,category$counts,
            category$weights,offset,q+1L,as.numeric(gamma),adjusted,object$block.size,
            control$condition.tol,.ordinal_link_code(.ordinal_link(object)))
        value$valid <- TRUE;value$message <- "";value
    },error=function(e) list(valid=FALSE,message=conditionMessage(e)))
    constrained <- function(gamma,basis,adjusted) ordinal_partial_constrained_native(common,varying,
        category$codes,category$counts,category$weights,offset,q+1L,gamma,basis,positive,adjusted,
        object$block.size,control$condition.tol,.ordinal_link_code(.ordinal_link(object)),
        as.integer(control$maxit),control$score.tol,control$step.tol,control$max.step,
        as.integer(control$max.halves))
    list(coefficients=b,q=q,p=d-q,pc=pc,pn=pn,d=d,T=T,Ti=Ti,x=common,z=varying,
        domain.z=varying[positive,,drop=FALSE],positive=positive,partial=TRUE,
        evaluate=evaluate,constrained=constrained,control=control,frequency=object$frequency,observations=object$nobs.rows)
}

.ordinal_partial_profile_gaps <- function(setup,b) {
    if(setup$q<2L) return(matrix(numeric(),nrow(setup$domain.z),0L))
    slopes <- matrix(b[setup$q+setup$pc+seq_len(setup$q*setup$pn)],setup$pn,setup$q)
    design <- cbind(1,1,setup$domain.z,setup$domain.z)
    gaps <- matrix(0,nrow(design),setup$q-1L)
    for(j in seq_len(setup$q-1L)) gaps[,j] <- streaming_qr_eta(design,
        c(b[j+1L],-b[j],-slopes[,j+1L],slopes[,j]),numeric(nrow(design)))
    gaps
}

.ordinal_partial_constraint_start <- function(setup,null,start,constraint) {
    C <- constraint$matrix;B <- constraint$basis;target <- null/constraint$row.scale
    project <- if(nrow(C)) function(g)
        g+drop(t(C)%*%solve(tcrossprod(C),target-streaming_qr_eta(C,g,numeric(nrow(C))))) else identity
    initial <- streaming_qr_eta(setup$Ti,as.numeric(start),numeric(setup$d))
    for(iter in seq_len(3L)) initial <- project(initial)
    failure <- function(message) list(valid=FALSE,message=message)
    if(any(!is.finite(initial))) return(failure("non-finite constrained starting coefficients"))
    satisfied <- function(g) !nrow(C) || all(abs(streaming_qr_eta(C,g,numeric(nrow(C)))-target)<=
        512*.Machine$double.eps*(1+abs(target)))
    if(!satisfied(initial)) return(failure("linear constraints cannot be represented accurately at the proposed starting coefficients"))
    gap <- .ordinal_partial_profile_gaps(setup,initial)
    if(!length(gap) || all(gap>0)) return(list(valid=TRUE,coefficients=initial,message=""))
    if(!ncol(B)) return(failure("fixed coefficients do not give strictly ordered cumulative predictors at every positive-frequency row"))
    original.gap <- .ordinal_partial_profile_gaps(setup,drop(setup$Ti%*%start))
    positive.gap <- original.gap[is.finite(original.gap)&original.gap>0]
    margin <- if(length(positive.gap)) min(.001,min(positive.gap)/100) else .001
    # Project onto the most violated row halfspace while remaining in the null
    # affine space. Shrinking the starting margin avoids imposing an additional
    # fixed probability-gap restriction on the statistical model.
    for(attempt in 0:2) {
        gamma <- initial;desired <- margin*.01^attempt
        for(iter in seq_len(500L)) {
            gaps <- .ordinal_partial_profile_gaps(setup,gamma)
            if(all(is.finite(gaps)) && all(gaps>0) && satisfied(gamma))
                return(list(valid=TRUE,coefficients=gamma,message=""))
            if(any(!is.finite(gaps))) break
            index <- which.min(gaps);i <- (index-1L)%%nrow(gaps)+1L;j <- (index-1L)%/%nrow(gaps)+1L
            a <- numeric(setup$d);a[j] <- -1;a[j+1L] <- 1
            at <- setup$q+setup$pc+(j-1L)*setup$pn+seq_len(setup$pn)
            a[at] <- setup$domain.z[i,];a[at+setup$pn] <- -setup$domain.z[i,]
            direction <- drop(B%*%crossprod(B,a));squared <- sum(direction^2)
            if(!is.finite(squared) || squared<=256*.Machine$double.eps^2*sum(a^2))
                return(failure("the supplied constraints prevent strictly ordered cumulative predictors at a positive-frequency row"))
            gamma <- project(gamma+(desired-gaps[index])*direction/squared)
            if(!satisfied(gamma)) break
        }
    }
    failure("no feasible starting cumulative predictors were found under the supplied constraints")
}
