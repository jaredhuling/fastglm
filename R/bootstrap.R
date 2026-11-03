#' Refit bootstrap uncertainty for GLMs, count models, and ordinal models
#'
#' Repeatedly resamples observations or independent clusters, or generates new
#' responses from the fitted model, and refits the regression. A statistic such
#' as a coefficient or average predicted response is then evaluated after each
#' refit. This includes uncertainty from estimating both model components and
#' negative-binomial size when applicable.
#' @param object a converged fastglm model with its design and response retained,
#'   or a streaming model with a repeatable source.
#' @param R number of bootstrap refits, at least two.
#' @param statistic function of the refitted model returning a finite numeric
#'   vector. The default returns coefficients and estimated theta when present.
#' @param type `"pairs"` resamples rows, `"cluster"` resamples whole independent
#'   clusters, and `"parametric"` simulates responses conditional on the design.
#' @param cluster cluster identifiers for cluster resampling. For a streaming
#'   source, supply a function returning the identifiers in each chunk.
#' @param seed optional random seed. Supplying it preserves the caller's random
#'   number state. Each replicate has its own seed, independent of scheduling.
#' @param workers number of independent R processes. Streaming sources use one
#'   process because database connections may not be transferable.
#'   A custom statistic must capture external objects, such as new prediction
#'   data, in its function environment; worker processes do not inherit objects
#'   from the calling session's global environment.
#' @param level confidence level for percentile intervals.
#' @param control named list of arguments passed to each fitting function,
#'   such as `tol`, `maxit`, or `outer.maxit`.
#' @details Pair resampling treats rows as independent sampling units. Cluster
#'   resampling treats the supplied clusters as independent and retains all rows
#'   from each sampled cluster. The original observation weights and offsets are
#'   retained on every sampled copy. Parametric resampling assumes the fitted
#'   response distributions; it does not model unspecified within-cluster
#'   dependence. For a numeric binomial response, positive prior weights are
#'   interpreted as integer trial counts, as in `stats::simulate()`; for a
#'   two-column response, the column sum supplies the trial count and any
#'   additional case weights are held fixed. The formula design is held fixed, including factor levels and
#'   spline bases, so data-driven preprocessing or model selection is not
#'   repeated. These procedures require regular, identifiable fits; they do not
#'   establish valid inference at separation or mixture boundaries.
#'   Streaming parametric refits regenerate responses one block at a time with
#'   fixed block seeds. Pair and cluster resampling also retain row indices or
#'   multiplicities, requiring memory proportional to the number of rows;
#'   neither procedure collects the full streaming design matrix.
#'
#'   For ordinal fits, pair and cluster resampling copy stored rows, including
#'   their frequency weights and grouped counts; they do not resample the
#'   individuals represented by a frequency row separately. Parametric
#'   resampling draws category counts with each row's effective integer total,
#'   then refits those counts with unit weights. The declared outcome levels,
#'   design columns, and offsets are preserved. Formula metadata is retained for
#'   statistics that predict on new data. Ordinal optimizer arguments may be
#'   changed through `control`, but the estimator, design, response, weights,
#'   and category order may not be overridden there.
#'   Stored zero-frequency rows remain sampling units in pairs and cluster
#'   resampling. Ordinal cluster identifiers may match retained rows or rows
#'   before missing-value omission, after any subset; every retained sampling
#'   row must have a cluster identifier. Ordinal results record `rng.kind`
#'   alongside replicate seeds for reproducing the random-number stream.
#'
#'   Ordinal point replicates and category diagnostics are retained when a
#'   refit has empty categories or fails to converge, provided the statistic can
#'   be evaluated. For mean bias reduction with `type = "parametric"`, empty
#'   endpoint categories are allowed in the original fit and its refits because
#'   the fitted thresholds remain finite. The original fit and every refit must
#'   retain all interior categories, and every refit and statistic must
#'   succeed. An empty interior category ties adjacent thresholds and changes
#'   the fitted model, so uncertainty summaries remain unavailable. The same
#'   empty-category restriction applies to maximum likelihood and to pairs or
#'   cluster resampling. These rules also apply to custom statistics; point
#'   replicates remain available for inspection.
#'
#'   Every failed or nonconverged refit is recorded. If any replicate fails,
#'   standard errors and intervals are returned as missing rather than silently
#'   conditioning inference on the successful fits. Inspect `diagnostics` and
#'   resolve the fitting problem before interpreting bootstrap uncertainty.
#' @returns A `fastglm_bootstrap` list with the original `estimate`, a matrix of
#'   `replicates`, `std.error`, percentile `conf.int`, per-refit `diagnostics`,
#'   replicate `seeds`, and resampling details.
#' @export
fastglm_bootstrap <- function(object,R=199L,statistic=NULL,
                              type=c("pairs","cluster","parametric"),
                              cluster=NULL,seed=NULL,workers=1L,level=.95,
                              control=list()) {
    if(inherits(object,"fastglm_ordinal"))
        return(.ordinal_bootstrap(object,R,statistic,type,cluster,seed,workers,level,control,
                                  call=match.call()))
    type<-match.arg(type);.check_iterations(1e-8,R);.check_iterations(1e-8,workers)
    if(R<2L) stop("'R' must be at least two",call.=FALSE)
    if(!isTRUE(object$converged)) stop("bootstrap requires a converged original fit",call.=FALSE)
    if(.count_inference_model(object) && .count_theta_boundary(object))
        stop("regular bootstrap inference is unavailable at a theta boundary",call.=FALSE)
    if(length(level)!=1L || !is.finite(level) || level<=0 || level>=1) stop("'level' must lie between zero and one",call.=FALSE)
    if(!is.list(control) || (length(control) && (is.null(names(control)) || any(!nzchar(names(control))))))
        stop("'control' must be a named list",call.=FALSE)
    if(!is.null(seed)) {
        if(length(seed)!=1L || !is.finite(seed)) stop("'seed' must be a finite scalar",call.=FALSE)
        had<-exists(".Random.seed",.GlobalEnv,inherits=FALSE)
        if(had) old<-get(".Random.seed",.GlobalEnv)
        on.exit(if(had) assign(".Random.seed",old,.GlobalEnv) else
            if(exists(".Random.seed",.GlobalEnv,inherits=FALSE)) rm(".Random.seed",envir=.GlobalEnv))
        set.seed(seed)
    }
    if(is.null(statistic)) statistic<-function(fit) {
        if(.count_inference_model(fit)) .count_parameters(fit,FALSE) else coef(fit)
    }
    if(!is.function(statistic)) stop("'statistic' must be a function of the refitted model",call.=FALSE)
    estimate<-statistic(object)
    if(!is.numeric(estimate) || !is.null(dim(estimate)) || !length(estimate) || any(!is.finite(estimate)))
        stop("the statistic must return a finite numeric vector",call.=FALSE)
    if(is.null(names(estimate))) names(estimate)<-paste0("statistic",seq_along(estimate))
    streaming<-!is.null(object$source) && is.function(object$fetch)
    if(streaming && workers>1L) stop("streaming bootstrap uses workers = 1 so source connections remain local",call.=FALSE)
    if(type=="cluster" && is.null(cluster)) stop("cluster resampling requires cluster identifiers",call.=FALSE)
    if(type!="cluster" && !is.null(cluster)) stop("supply cluster identifiers only for type = 'cluster'",call.=FALSE)
    input<-.bootstrap_input(object,streaming)
    if(type=="cluster") {
        groups<-if(streaming) {
            if(!is.function(cluster)) stop("streaming clusters require a function of chunk index",call.=FALSE)
            unlist(lapply(seq_along(input$rows),function(k) {
                g<-cluster(k);if(length(g)!=input$rows[k]) stop("invalid cluster identifiers",call.=FALSE);as.character(g)
            }),use.names=FALSE)
        } else cluster
        if(length(groups)!=input$n || anyNA(groups)) stop("invalid cluster identifiers",call.=FALSE)
        units<-split(seq_len(input$n),as.character(groups))
        if(length(units)<2L) stop("cluster bootstrap requires at least two clusters",call.=FALSE)
    } else units<-NULL
    seeds<-sample.int(.Machine$integer.max,as.integer(R),replace=FALSE)
    one<-function(j) {
        set.seed(seeds[j]);warnings<-character()
        ans<-tryCatch(withCallingHandlers({
            index<-if(type=="pairs") sample.int(input$n,input$n,replace=TRUE) else
                if(type=="cluster") unlist(units[sample.int(length(units),length(units),replace=TRUE)],use.names=FALSE) else NULL
            fit<-if(streaming) .bootstrap_stream_refit(object,input,index,type,control) else {
                z<-input$data
                if(type=="parametric") z$y<-.bootstrap_response(object,z)
                else z<-.bootstrap_subset(z,index)
                .bootstrap_refit(object,z,control)
            }
            if(!isTRUE(fit$converged)) stop("refit did not converge",call.=FALSE)
            if(.count_inference_model(fit) && .count_theta_boundary(fit))
                stop("refit reached a theta boundary",call.=FALSE)
            value<-statistic(fit)
            if(!is.numeric(value) || length(value)!=length(estimate) || any(!is.finite(value)))
                stop("refit statistic is non-finite or has changed dimension",call.=FALSE)
            if(!is.null(names(value)) && !identical(names(value),names(estimate)))
                stop("refit statistic names changed",call.=FALSE)
            list(value=as.numeric(value),converged=TRUE,error="")
        },warning=function(w) {warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")}),
        error=function(e) list(value=rep(NA_real_,length(estimate)),converged=FALSE,error=conditionMessage(e)))
        ans$warning<-paste(unique(warnings),collapse="; ");ans
    }
    workers<-min(as.integer(workers),as.integer(R))
    if(workers==1L) out<-lapply(seq_len(R),one) else {
        cl<-parallel::makePSOCKcluster(workers);on.exit(parallel::stopCluster(cl),add=TRUE)
        initialize<-function(paths) {.libPaths(paths);loadNamespace("fastglm");NULL}
        # Serializing a function from this namespace would load fastglm before
        # the worker receives the caller's library paths. Use a base closure.
        environment(initialize)<-baseenv()
        parallel::clusterCall(cl,initialize,.libPaths())
        out<-parallel::parLapply(cl,seq_len(R),one)
    }
    values<-do.call(rbind,lapply(out,`[[`,"value"));colnames(values)<-names(estimate)
    diagnostics<-data.frame(replicate=seq_len(R),seed=seeds,
        converged=vapply(out,`[[`,logical(1),"converged"),
        error=vapply(out,`[[`,character(1),"error"),warning=vapply(out,`[[`,character(1),"warning"))
    failed<-!diagnostics$converged
    se<-setNames(rep(NA_real_,length(estimate)),names(estimate))
    ci<-matrix(NA_real_,length(estimate),2L,dimnames=list(names(estimate),c("lower","upper")))
    if(!any(failed)) {
        se<-apply(values,2L,sd)
        ci<-t(apply(values,2L,quantile,probs=c((1-level)/2,(1+level)/2),names=FALSE))
        dimnames(ci)<-list(names(estimate),c("lower","upper"))
    } else warning(sum(failed)," bootstrap refits failed; uncertainty summaries are unavailable",call.=FALSE)
    structure(list(estimate=estimate,replicates=values,std.error=se,conf.int=ci,
        diagnostics=diagnostics,seeds=seeds,type=type,level=level,R=as.integer(R),
        successful=sum(!failed),call=match.call()),class="fastglm_bootstrap")
}

.bootstrap_input <- function(object,streaming) {
    two<-.count_two_component(object)
    if(streaming) {
        rows<-vapply(seq_len(object$source$n_chunks),function(k)nrow(object$fetch(k)$X),integer(1))
        return(list(rows=rows,n=sum(rows)))
    }
    d<-if(two) .count_design(object) else .prediction_design(object)
    y<-object$y
    if(is.null(y)) stop("bootstrap requires a retained response",call.=FALSE)
    w<-if(two) object$weights else object$prior.weights
    x<-if(two) d$x$count else d$x
    z<-list(X=x,Z=if(two) d$x$zero,y=y,weights=w,
            offset=if(two) d$offset else list(count=d$offset,zero=rep(0,nrow(x))))
    # The binomial GLM stores proportions and trial-adjusted weights. Restore
    # the original trial counts so family initialization is performed once.
    if(!.count_inference_model(object) && identical(object$family$family,"binomial") && length(object$n)==length(y) && any(object$n!=1)) {
        trials<-object$n
        z$y<-cbind(y*trials,(1-y)*trials);z$weights<-w/pmax(1,trials)
    }
    list(data=z,n=nrow(x))
}
.bootstrap_subset <- function(z,index) {
    list(X=z$X[index,,drop=FALSE],Z=if(!is.null(z$Z)) z$Z[index,,drop=FALSE],
        y=if(is.matrix(z$y)) z$y[index,,drop=FALSE] else z$y[index],
        weights=z$weights[index],offset=lapply(z$offset,`[`,index))
}
.bootstrap_response <- function(object,z) {
    two<-.count_two_component(object);n<-nrow(z$X)
    if(two) {
        eta<-drop(z$X%*%object$coefficients$count)+z$offset$count
        eta0<-drop(z$Z%*%object$coefficients$zero)+z$offset$zero
        mu<-exp(eta);pi<-make.link(object$link)$linkinv(eta0)
        if(inherits(object,"fastglm_twopart")) {
            # Gamma likelihood weights describe precision, as in glm's family.
            shape<-ifelse(z$weights>0,z$weights,1)/object$positive$dispersion
            return(rbinom(n,1,pi)*rgamma(n,shape=shape,scale=mu/shape))
        }
        if(inherits(object,"fastglm_zi")) {
            y<-if(object$dist=="poisson") rpois(n,mu) else rnbinom(n,mu=mu,size=object$theta)
            return(rbinom(n,1,1-pi)*y)
        }
        q<-.count_moments(mu,object$dist,object$theta)$q;u<-runif(n)*q
        y<-if(object$dist=="poisson") qpois(u,mu,lower.tail=FALSE) else qnbinom(u,mu=mu,size=object$theta,lower.tail=FALSE)
        return(rbinom(n,1,pi)*y)
    }
    eta<-drop(z$X%*%coef(object))+z$offset$count
    if(inherits(object,"fastglm_truncated")) {
        mu<-exp(eta);u<-runif(n)*.count_moments(mu,object$dist,object$theta)$q
        return(if(object$dist=="poisson") qpois(u,mu,lower.tail=FALSE) else qnbinom(u,mu=mu,size=object$theta,lower.tail=FALSE))
    }
    mu<-object$family$linkinv(eta);fam<-object$family$family
    if(inherits(object,"fastglm_nb") || grepl("^Negative Binomial",fam)) {
        theta<-if(inherits(object,"fastglm_nb")) object$theta else family_params(object$family)[1]
        return(rnbinom(n,mu=mu,size=theta))
    }
    if(fam=="poisson") return(rpois(n,mu))
    if(fam=="binomial") {
        trials<-if(is.matrix(z$y)) rowSums(z$y) else z$weights
        if(any(abs(trials-round(trials))>1e-8))
            stop("parametric binomial bootstrap requires integer trial counts",call.=FALSE)
        trials<-pmax(1,round(trials))
        yy<-rbinom(n,trials,mu)
        return(if(is.matrix(z$y)) cbind(yy,trials-yy) else yy/trials)
    }
    w<-ifelse(z$weights>0,z$weights,1)
    if(fam=="gaussian") return(rnorm(n,mu,sqrt(object$dispersion/w)))
    if(fam=="Gamma") return(rgamma(n,shape=w/object$dispersion,scale=mu*object$dispersion/w))
    stop("parametric bootstrap is unavailable for this family",call.=FALSE)
}
.bootstrap_refit <- function(object,z,control) {
    defaults<-object$refit.controls;if(is.null(defaults)) defaults<-list()
    controls<-utils::modifyList(defaults,control)
    fit<-if(.count_two_component(object)) {
        if(inherits(object,"fastglm_zi")) {
            if(inherits(z$X,"sparseMatrix") || inherits(z$Z,"sparseMatrix"))
                stop("zero-inflated bootstrap requires dense component designs",call.=FALSE)
            # The zero-inflated API requires a formula. Reuse its existing
            # dense design columns so the fitted basis remains fixed.
            pc<-ncol(z$X);pz<-ncol(z$Z)
            cx<-paste0(".count",seq_len(pc));cz<-paste0(".zero",seq_len(pz))
            data<-data.frame(.response=z$y,z$X,z$Z,check.names=FALSE)
            names(data)<-c(".response",cx,cz)
            formula<-as.formula(paste(".response ~ 0 +",paste(cx,collapse=" + "),"| 0 +",paste(cz,collapse=" + ")))
            args<-list(formula=formula,data=data,weights=z$weights,
                offset=cbind(z$offset$count,z$offset$zero),link=object$link)
        } else {
            args<-list(formula=z$X,response=z$y,z=z$Z,weights=z$weights,
                offset=z$offset,link=object$link)
        }
        if(inherits(object,"fastglm_twopart")) {
            fn<-fastglm_twopart;args$firth<-.count_adjusted(object)
        } else {
            fn<-if(inherits(object,"fastglm_zi")) fastglm_zi else fastglm_hurdle
            args$dist<-object$dist
            if(object$dist=="negbin") args$init.theta<-object$theta
            args$x<-TRUE
        }
        f<-do.call(fn,utils::modifyList(args,controls))
        names(f$coefficients$count)<-names(object$coefficients$count)
        names(f$coefficients$zero)<-names(object$coefficients$zero)
        colnames(f$x$count)<-colnames(z$X);colnames(f$x$zero)<-colnames(z$Z)
        nm<-names(coef(object));dimnames(f$vcov)<-list(nm,nm)
        if(!is.null(f$vcov.full)) dimnames(f$vcov.full)<-rep(list(c(nm,if(.count_has_theta(object)) "theta")),2L)
        for(key in c("terms","formula","levels","contrasts")) f[[key]]<-object[[key]]
        f$call<-object$call;f$na.action<-NULL;f$model<-NULL
        f
    } else if(inherits(object,"fastglm_truncated")) {
        args<-list(x=z$X,y=z$y,dist=object$dist,weights=z$weights,offset=z$offset$count,
            theta=if(.count_has_theta(object) || object$dist!="negbin") NULL else object$theta,start=coef(object))
        do.call(fastglm_truncated,utils::modifyList(args,controls))
    } else if(inherits(object,"fastglm_nb")) {
        args<-list(x=z$X,y=z$y,weights=z$weights,offset=z$offset$count,start=coef(object),
            init.theta=object$theta,link=object$family$link)
        do.call(fastglm_nb,utils::modifyList(args,controls))
    } else {
        args<-list(x=z$X,y=z$y,family=object$family,weights=z$weights,offset=z$offset$count,
            start=coef(object),firth=isTRUE(object$firth),method="auto")
        do.call(fastglm,utils::modifyList(args,controls))
    }
    if(!.count_two_component(object)) {
        for(key in c("terms","xlevels","contrasts")) fit[[key]]<-object[[key]]
        fit$call<-object$call;fit$na.action<-NULL
    }
    fit
}

.bootstrap_stream_refit <- function(object,input,index,type,control) {
    rows<-input$rows;ends<-cumsum(rows);starts<-c(0,utils::head(ends,-1L))
    two<-.count_two_component(object)
    if(type=="parametric") {
        chunks<-seq_along(rows);within<-NULL
        chunk.seeds<-sample.int(.Machine$integer.max,length(rows),replace=FALSE)
    } else {
        frequencies<-tabulate(index,nbins=input$n)
        within<-lapply(seq_along(rows),function(k) rep(seq_len(rows[k]),frequencies[starts[k]+seq_len(rows[k])]))
        chunks<-which(lengths(within)>0L)
    }
    callback<-function(j) {
        k<-chunks[j]
        first<-if(j==1L) 1L else chunks[j-1L]+1L
        for(original in seq.int(first,k)) raw<-object$fetch(original)
        # Sequential readers must consume unsampled blocks too, including
        # the final blocks before the next pass resets the underlying source.
        if(j==length(chunks) && k<length(rows))
            for(original in seq.int(k+1L,length(rows))) object$fetch(original)
        z<-.count_normalize_block(raw,two)
        if(!.count_inference_model(object) && identical(object$family$family,"binomial") && !is.null(raw$trials) && any(raw$trials!=1)) {
            z$y<-cbind(z$y*raw$trials,(1-z$y)*raw$trials)
            z$weights<-z$weights/pmax(1,raw$trials)
        }
        if(type=="parametric") {
            had<-exists(".Random.seed",.GlobalEnv,inherits=FALSE);if(had) old<-get(".Random.seed",.GlobalEnv)
            on.exit(if(had) assign(".Random.seed",old,.GlobalEnv) else
                if(exists(".Random.seed",.GlobalEnv,inherits=FALSE)) rm(".Random.seed",envir=.GlobalEnv))
            set.seed(chunk.seeds[k]);z$y<-.bootstrap_response(object,z)
        } else z<-.bootstrap_subset(z,within[[k]])
        if(!two) z$offset<-z$offset$count
        z
    }
    src<-fastglm_source(callback,length(chunks));src$schema<-object$source$schema
    controls<-utils::modifyList(if(is.null(object$refit.controls)) list() else object$refit.controls,control)
    if(inherits(object,"fastglm_twopart")) {
        args<-list(formula=src,link=object$link,firth=.count_adjusted(object));fn<-fastglm_twopart
    } else if(inherits(object,"fastglm_hurdle")) {
        args<-list(formula=src,link=object$link,dist=object$dist,init.theta=object$theta);fn<-fastglm_hurdle
    } else if(inherits(object,"fastglm_nb")) {
        args<-list(x=src,link=object$family$link,start=coef(object),init.theta=object$theta);fn<-fastglm_nb
    } else {
        args<-list(chunk_callback=src,family=object$family,start=coef(object),firth=isTRUE(object$firth));fn<-fastglm_streaming
    }
    do.call(fn,utils::modifyList(args,controls))
}

#' @rdname fastglm_bootstrap
#' @method print fastglm_bootstrap
#' @param x a `fastglm_bootstrap` result.
#' @param ... further arguments passed to `print()`.
#' @export
print.fastglm_bootstrap <- function(x,...) {
    cat("Refit bootstrap (",x$type,"): ",x$successful," of ",x$R," successful refits\n",sep="")
    print(cbind(Estimate=x$estimate,`Std. Error`=x$std.error,x$conf.int),...)
    invisible(x)
}
