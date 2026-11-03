.ordinal_bootstrap_input <- function(object) {
    if(is.null(object$x) || is.null(object$y))
        stop("ordinal bootstrap requires the retained design and response",call.=FALSE)
    list(x=object$x,y=object$y,weights=object$weights,offset=object$offset,
         n=nrow(object$x))
}

.ordinal_bootstrap_refit <- function(object,input,index=NULL,counts=NULL,control=list(),cache=NULL) {
    reuse <- !is.null(counts) && is.null(index) && is.environment(cache)
    if(is.null(index)) index <- seq_len(input$n)
    if(is.null(counts)) {
        y <- if(is.matrix(input$y)) input$y[index,,drop=FALSE] else input$y[index]
        weights <- input$weights[index]
    } else {
        # Simulate the effective frequencies directly. Dividing these counts by
        # the original weights could produce fractional or overflowing values.
        y <- counts
        weights <- rep(1,nrow(counts))
    }
    if(reuse) {
        key <- list(x=input$x,offset=input$offset)
        if(!identical(cache$bootstrap.inputs$key,key)) {
            # Preserve the ordinary subset's matrix/vector attributes, while
            # allocating those unchanged inputs only once in each process.
            cache$bootstrap.inputs <- list(key=key,x=input$x[index,,drop=FALSE],
                                            offset=input$offset[index])
        }
    }
    args <- list(x=if(reuse) cache$bootstrap.inputs$x else input$x[index,,drop=FALSE],y=y,weights=weights,
                 offset=if(reuse) cache$bootstrap.inputs$offset else input$offset[index],levels=object$levels,firth=object$firth,
                 tol=object$tol,maxit=object$maxit,block.size=object$block.size,
                 control=object$control,link=.ordinal_link(object))
    args <- utils::modifyList(args,control)
    fit <- if(reuse) {
        # NULL overrides remove arguments in modifyList(). The public fitting
        # method would then use these defaults, which must remain identical on
        # the internal prepared path.
        args <- utils::modifyList(list(tol=1e-8,maxit=200L,block.size=2048L,control=list()),args)
        args$cache <- cache
        result <- do.call(.ordinal_fit_matrix,args)
        # Preparation is an internal bootstrap optimization. Keep the refit's
        # public metadata identical to an independent matrix fit, including
        # statistics that inspect the original preparation diagnostic.
        result$preparation.reused <- FALSE
        result$call <- object$call
        result
    } else do.call(fastglm_ordinal,args)
    for(key in c("terms","design.assign","xlevels","contrasts","offset.expression","prediction.offset.required"))
        fit[[key]] <- object[[key]]
    # These refits keep the fitted basis, including factor levels and spline
    # transformations. Remove the original missing-row restoration: the refit
    # rows already refer to the retained training rows or their sampled copies.
    fit$na.action <- NULL
    fit$call <- object$call
    if(!is.null(object$model)) {
        fit$model <- object$model[index,,drop=FALSE]
        fit$model[[1L]] <- y
        if("(weights)" %in% names(fit$model)) fit$model[["(weights)"]] <- weights
    }
    fit
}

.ordinal_bootstrap_pattern <- function(object,input,index=NULL,counts=NULL) {
    if(is.null(index)) index <- seq_len(input$n)
    y <- if(!is.null(counts)) counts else if(is.matrix(input$y))
        input$y[index,,drop=FALSE] else input$y[index]
    weights <- if(is.null(counts)) input$weights[index] else rep(1,nrow(counts))
    # Inspect the declared scale before fitting. In particular an ML refit with
    # an empty endpoint must still record which category caused its rejection.
    z <- .ordinal_response_data(y,weights,object$levels,firth=TRUE)
    list(empty=z$empty.categories,collapsed=z$collapsed.categories,
         endpoints=z$empty.end.categories,frequency=sum(z$totals))
}

.ordinal_bootstrap <- function(object,R,statistic,type,cluster,seed,workers,level,control,call) {
    type <- match.arg(type,c("pairs","cluster","parametric"))
    .check_iterations(1e-8,R); .check_iterations(1e-8,workers)
    if(R<2L) stop("'R' must be at least two",call.=FALSE)
    if(!isTRUE(object$converged)) stop("bootstrap requires a converged original fit",call.=FALSE)
    if(length(level)!=1L || !is.numeric(level) || is.complex(level) ||
       !is.finite(level) || level<=0 || level>=1)
        stop("'level' must lie between zero and one",call.=FALSE)
    if(!is.list(control) || (length(control) &&
       (is.null(names(control)) || anyNA(names(control)) || any(!nzchar(names(control))) ||
        anyDuplicated(names(control))))) stop("'control' must be a named list",call.=FALSE)
    if(length(setdiff(names(control),c("tol","maxit","block.size","control"))))
        stop("ordinal bootstrap control may change only tol, maxit, block.size, and optimizer control",call.=FALSE)
    rng.kind <- RNGkind()
    if(!is.null(seed)) {
        if(!is.numeric(seed) || is.complex(seed) || length(seed)!=1L || !is.finite(seed))
            stop("'seed' must be a finite scalar",call.=FALSE)
        had <- exists(".Random.seed",.GlobalEnv,inherits=FALSE)
        if(had) old <- get(".Random.seed",.GlobalEnv)
        on.exit({
            do.call(RNGkind,as.list(rng.kind))
            if(had) assign(".Random.seed",old,.GlobalEnv) else
                if(exists(".Random.seed",.GlobalEnv,inherits=FALSE)) rm(".Random.seed",envir=.GlobalEnv)
        })
        set.seed(seed)
    }
    if(is.null(statistic)) statistic <- function(fit) coef(fit)
    if(!is.function(statistic)) stop("'statistic' must be a function of the refitted model",call.=FALSE)
    estimate <- statistic(object)
    valid.statistic <- function(value) is.numeric(value) && !is.complex(value) &&
        is.null(dim(value)) && length(value)>0L && all(is.finite(value))
    if(!valid.statistic(estimate)) stop("the statistic must return a finite numeric vector",call.=FALSE)
    if(is.null(names(estimate))) names(estimate) <- paste0("statistic",seq_along(estimate))
    input <- .ordinal_bootstrap_input(object)
    original.empty <- object$category$empty.categories
    if(type=="cluster" && is.null(cluster)) stop("cluster resampling requires cluster identifiers",call.=FALSE)
    if(type!="cluster" && !is.null(cluster)) stop("supply cluster identifiers only for type = 'cluster'",call.=FALSE)
    units <- if(type=="cluster") {
        keys <- .ordinal_cluster_vector(cluster,object)
        retained <- if(length(cluster)==input$n) cluster else cluster[-as.integer(object$na.action)]
        if(anyNA(retained)) stop("cluster identifiers are missing for retained sampling rows",call.=FALSE)
        groups <- split(seq_len(input$n),keys)
        if(length(groups)<2L) stop("cluster bootstrap requires at least two clusters",call.=FALSE)
        groups
    }
    seeds <- sample.int(.Machine$integer.max,as.integer(R),replace=FALSE)
    # Parametric draws preserve the effective row totals. Build setup lazily
    # inside the first refit, preserving per-replicate validation/failure
    # behavior. Serialized worker closures each own their ordinary R cache.
    cache <- if(type=="parametric") new.env(parent=emptyenv()) else NULL
    one <- function(j) {
        do.call(RNGkind,as.list(rng.kind))
        set.seed(seeds[j]); warnings <- character(); fit <- NULL; pattern <- NULL
        ans <- tryCatch(withCallingHandlers({
            index <- if(type=="pairs") sample.int(input$n,input$n,replace=TRUE) else
                if(type=="cluster") unlist(units[sample.int(length(units),length(units),replace=TRUE)],use.names=FALSE) else NULL
            counts <- if(type=="parametric") .ordinal_simulate_counts(object) else NULL
            pattern <- .ordinal_bootstrap_pattern(object,input,index,counts)
            fit <- .ordinal_bootstrap_refit(object,input,index,counts,control,cache)
            value <- statistic(fit)
            if(!valid.statistic(value) || length(value)!=length(estimate))
                stop("refit statistic is non-finite or has changed dimension",call.=FALSE)
            if(!is.null(names(value)) && !identical(names(value),names(estimate)))
                stop("refit statistic names changed",call.=FALSE)
            list(value=as.numeric(value),statistic.ok=TRUE,error=if(isTRUE(fit$converged)) "" else
                 paste0("refit did not converge: ",fit$convergence.reason))
        },warning=function(w) {warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")}),
        error=function(e) list(value=rep(NA_real_,length(estimate)),statistic.ok=FALSE,error=conditionMessage(e)))
        ans$converged <- !is.null(fit) && isTRUE(fit$converged)
        ans$iterations <- if(is.null(fit)) NA_integer_ else as.integer(fit$iterations)
        ans$rank <- if(is.null(fit)) NA_integer_ else as.integer(fit$rank)
        ans$convergence.reason <- if(is.null(fit)) "fit_error" else fit$convergence.reason
        ans$pattern <- pattern
        ans$warning <- paste(unique(warnings),collapse="; ")
        ans
    }
    workers <- min(as.integer(workers),as.integer(R))
    if(workers==1L) out <- lapply(seq_len(R),one) else {
        cl <- parallel::makePSOCKcluster(workers)
        on.exit(parallel::stopCluster(cl),add=TRUE)
        initialize <- function(paths) {.libPaths(paths);loadNamespace("fastglm");NULL}
        environment(initialize) <- baseenv()
        parallel::clusterCall(cl,initialize,.libPaths())
        out <- parallel::parLapply(cl,seq_len(R),one)
    }
    values <- do.call(rbind,lapply(out,`[[`,"value")); colnames(values) <- names(estimate)
    has.pattern <- vapply(out,function(z)!is.null(z$pattern),logical(1))
    empties <- lapply(out,function(z)z$pattern$empty)
    collapsed <- lapply(out,function(z)z$pattern$collapsed)
    endpoints <- lapply(out,function(z)z$pattern$endpoints)
    has.empty <- vapply(empties,length,integer(1))>0L
    changed <- vapply(empties,function(z)!setequal(z,original.empty),logical(1))
    changed[!has.pattern] <- NA
    diagnostics <- data.frame(replicate=seq_len(R),seed=seeds,
        converged=vapply(out,`[[`,logical(1),"converged"),
        statistic.ok=vapply(out,`[[`,logical(1),"statistic.ok"),
        boundary=ifelse(has.pattern,lengths(collapsed)>0L,NA),boundary.changed=changed,
        has.empty.categories=ifelse(has.pattern,has.empty,NA),
        iterations=vapply(out,`[[`,integer(1),"iterations"),rank=vapply(out,`[[`,integer(1),"rank"),
        frequency=vapply(out,function(z)if(is.null(z$pattern)) NA_real_ else z$pattern$frequency,numeric(1)),
        convergence.reason=vapply(out,`[[`,character(1),"convergence.reason"),
        error=vapply(out,`[[`,character(1),"error"),warning=vapply(out,`[[`,character(1),"warning"))
    diagnostics$empty.categories <- I(lapply(empties,function(z)object$levels[z]))
    diagnostics$collapsed.categories <- I(lapply(collapsed,function(z)object$levels[z]))
    diagnostics$empty.end.categories <- I(lapply(endpoints,function(z)object$levels[z]))
    failed <- !diagnostics$converged | !diagnostics$statistic.ok
    endpoint.meanbr.parametric <- isTRUE(object$firth) && type=="parametric"
    original.interior.empty <- object$category$collapsed.categories
    has.interior.empty <- vapply(collapsed,length,integer(1))>0L
    category.support <- if(endpoint.meanbr.parametric) {
        !length(original.interior.empty) && !any(has.interior.empty)
    } else !length(original.empty) && !any(has.empty)
    available <- category.support && !any(failed) && all(has.pattern)
    reason <- character()
    if(endpoint.meanbr.parametric) {
        if(length(original.interior.empty))
            reason <- c(reason,"the original fit has empty interior categories")
        if(any(has.interior.empty))
            reason <- c(reason,"one or more refits have empty interior categories")
    } else {
        if(length(original.empty)) reason <- c(reason,"the original fit has empty categories")
        if(any(has.empty)) reason <- c(reason,"one or more refits have empty categories")
        if(any(changed,na.rm=TRUE)) reason <- c(reason,"the empty-category pattern changed")
    }
    if(any(failed)) reason <- c(reason,"one or more refits or statistics failed or did not converge")
    se <- setNames(rep(NA_real_,length(estimate)),names(estimate))
    ci <- matrix(NA_real_,length(estimate),2L,dimnames=list(names(estimate),c("lower","upper")))
    if(available) {
        se <- apply(values,2L,sd)
        ci <- t(apply(values,2L,quantile,probs=c((1-level)/2,(1+level)/2),names=FALSE))
        dimnames(ci) <- list(names(estimate),c("lower","upper"))
    } else warning("ordinal bootstrap uncertainty summaries are unavailable: ",
                   paste(reason,collapse="; "),". Inspect point replicates and diagnostics",call.=FALSE)
    structure(list(estimate=estimate,replicates=values,std.error=se,conf.int=ci,
        diagnostics=diagnostics,seeds=seeds,rng.kind=rng.kind,type=type,level=level,R=as.integer(R),
        successful=sum(!failed),inference.available=available,inference.reason=reason,
        sampling.unit=switch(type,pairs="stored rows",cluster="independent clusters of stored rows",
            parametric="effective categorical observations conditional on the stored design"),
        original.empty.categories=object$levels[original.empty],
        original.frequency=object$frequency,call=call),class="fastglm_bootstrap")
}
