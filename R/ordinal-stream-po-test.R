# Only two distinct values per predictor are needed for the sufficient
# endpoint-separation certificate. A third value permanently disqualifies it.
.ordinal_stream_po_scan <- function(object) {
    p <- length(object$design.columns);K <- length(object$levels)
    values <- vector("list",p);more <- logical(p)
    counts <- lapply(seq_len(p),function(j) matrix(0,2L,K))
    totals <- numeric(K);observations <- frequency <- 0
    for(k in seq_len(object$source$n_chunks)) {
        z <- object$fetch(k);good <- which(z$totals>0)
        if(!length(good)) next
        X <- z$X[good,,drop=FALSE]
        yy <- if(nrow(z$counts)) z$counts[good,,drop=FALSE]*z$weights[good] else {
            out <- matrix(0,length(good),K)
            out[cbind(seq_along(good),z$codes[good])] <- z$weights[good];out
        }
        totals <- totals+colSums(yy);observations <- observations+length(good)
        frequency <- frequency+sum(z$totals[good])
        for(j in which(!more)) {
            # Check novelty in vectors bounded by the current block. The
            # retained values never grow with rows or source blocks.
            v <- values[[j]]
            if(!length(v)) v <- unname(X[1L,j])
            unseen <- X[!X[,j]%in%v,j]
            if(length(unseen)) {
                if(length(v)==2L) {more[j] <- TRUE;values[j] <- list(NULL);next}
                v <- c(v,unname(unseen[1L]))
                if(any(!unseen%in%v)) {more[j] <- TRUE;values[j] <- list(NULL);next}
            }
            values[[j]] <- v
            for(a in seq_along(v)) counts[[j]][a,] <- counts[[j]][a,]+
                colSums(yy[X[,j]==v[a],,drop=FALSE])
        }
    }
    separation <- NULL
    for(j in which(!more)) if(length(values[[j]])==2L) for(a in 1:2) {
        present <- which(counts[[j]][a,]>0)
        if(length(present)==1L && present%in%c(1L,K) && is.null(separation))
            separation <- list(column=object$design.columns[j],value=values[[j]][a],category=present)
    }
    list(observations=observations,frequency=frequency,totals=totals,separation=separation,
        constant=object$design.columns[!more & lengths(values)==1L])
}

.ordinal_stream_po_source <- function(object) fastglm_source(function(k) {
    z <- object$fetch(k)
    list(X=z$X,y=if(nrow(z$counts)) z$counts else z$codes,
        weights=z$weights,offset=z$offset)
},object$source$n_chunks)

.ordinal_stream_po_diagnostics <- function(fit,nonparallel,estimation) list(
    converged=fit$converged,iterations=fit$iterations,score=fit$scaled.score,
    predictor.step=fit$predictor.step,message=fit$convergence.reason,
    coefficients=coef(fit),nonparallel=nonparallel,estimation=estimation,
    evaluations=fit$evaluations,last.rejected.evaluation=fit$last.rejected.evaluation)

.ordinal_stream_po_moments <- function(object,fit,selected) {
    moments <- NULL
    for(k in seq_len(object$source$n_chunks)) {
        z <- object$fetch(k);at <- which(z$totals>0)
        if(!length(at)) next
        y <- if(nrow(z$counts)) z$counts[at,,drop=FALSE] else z$codes[at]
        response <- .ordinal_assessment_response(y,z$weights[at],object$levels,length(at))
        block <- .ordinal_po_score_moments(z$X[at,,drop=FALSE],response,z$offset[at],
            fit,selected,object$block.size)
        if(is.null(moments)) moments <- block else {
            moments$score <- moments$score+block$score
            moments$information <- moments$information+block$information
        }
    }
    moments
}

.ordinal_stream_partial_po_score <- function(object,fit,L) {
    q <- length(object$alpha);pc <- length(object$common);pn <- length(object$nonparallel)
    varying <- object$design.columns[object$nonparallel];gamma <- matrix(0,pn,q)
    for(j in seq_len(pn)) gamma[j,] <- if(varying[j]%in%names(fit$working$beta))
        fit$working$beta[varying[j]] else fit$working$gamma[varying[j],]
    coefficients <- c(fit$working$alpha,
        fit$working$beta[object$design.columns[object$common]],as.vector(gamma))
    fetch <- function(k) {
        z <- object$fetch(k);good <- z$totals>0
        xx <- .ordinal_prediction_scale(z$X,fit$working$center,fit$working$scale)
        xx[!good,] <- 0;z$offset[!good] <- 0
        z$X <- xx[,object$common,drop=FALSE];z$Z <- xx[,object$nonparallel,drop=FALSE];z
    }
    state <- ordinal_partial_stream_native_evaluate(fetch,object$source$n_chunks,pc,pn,q+1L,
        as.numeric(coefficients),FALSE,object$block.size,fit$control$condition.tol,
        .ordinal_link_code(.ordinal_link(object)))
    state$coefficients <- as.numeric(coefficients)
    state$converged <- TRUE;state$convergence.reason <- "reduced_null_fit"
    state$iterations <- fit$iterations;state$evaluations <- fit$evaluations
    state$history <- matrix(numeric(),0L,4L)
    embedded <- .ordinal_partial_result(state,fit$working$center,fit$working$scale,
        fit$category,fit$frequency,fit$nobs.rows,fit$firth,.ordinal_link(object),
        fit$control,fit$tol,fit$maxit,object$block.size,object$design.columns,object$nonparallel)
    B <- .ordinal_constraint_basis(list(T=embedded$working$T,d=length(coefficients)),L)$basis
    ordinary <- as.numeric(state$ordinary.score)
    nuisance <- drop(crossprod(B,ordinary))
    efficient <- ordinary-state$information%*%B%*%
        solve(crossprod(B,state$information%*%B),nuisance)
    statistic <- drop(crossprod(efficient,state$cov.unscaled%*%efficient))
    if(!is.finite(statistic) || statistic<0) stop("invalid efficient score quadratic",call.=FALSE)
    list(statistic=statistic,condition=state$information.rcond,
        coefficients=coef(embedded),nuisance.score=nuisance)
}

.ordinal_stream_po_test <- function(object,terms=NULL,control=list(),estimation="ML",
                                    by.term=FALSE,method="score") {
    if(!is.function(object$fetch) || is.null(object$source$n_chunks))
        stop("original-source common-slope tests require the recorded block reader",call.=FALSE)
    partial <- inherits(object,"fastglm_ordinal_partial")
    select <- if(partial) .ordinal_partial_po_columns else .ordinal_po_test_columns
    selected <- select(object,terms);ctl <- .ordinal_po_test_control(control)
    K <- length(object$levels)
    title <- if(partial) paste(if(.ordinal_link(object)=="logit") "Partial proportional-odds" else
        paste0("Partial common-slope (",.ordinal_link(object),")"),
        if(method=="lr") "likelihood ratio test" else "expected-information score test") else
        if(.ordinal_link(object)=="logit") "Expected-information proportional-odds score test" else
            paste0("Expected-information common-slope score test (",.ordinal_link(object),")")
    note <- if(method=="lr") "Both alternative and null are refitted by maximum likelihood; first-order chi-square approximation." else
        if(estimation=="ML") "Ordinary efficient score at the ML null; first-order chi-square approximation." else
            "Ordinary efficient score at a reduced-null mean-bias-reduced estimate; this is not the full-model adjusted-score test. First-order chi-square approximation."
    selections <- list(Joint=selected)
    if(by.term) {
        selections <- c(selections,lapply(selected$terms,function(term)select(object,term)))
        names(selections) <- make.unique(c("Joint",selected$terms))
    }
    scan <- list(observations=NA_real_,frequency=NA_real_,separation=NULL)
    template <- function(selection) {
        df <- if(partial) nrow(selection$contrast) else length(selection$index)*max(0L,K-2L)
        structure(list(method=title,available=FALSE,statistic=NA_real_,df=df,
            parameter=c(df=df),p.value=NA_real_,null.estimation=estimation,
            terms=selection$terms,columns=selection$columns,
            already.common.terms=selection$common.terms,contrast=selection$contrast,
            observations=scan$observations,frequency=scan$frequency,reason="",detail="",
            null.converged=FALSE,null.iterations=NA_integer_,information.rcond=NA_real_,
            empty.categories=character(),warning=character(),separation=scan$separation,
            data.name="original ordinal source",source=TRUE,note=paste(note,
                if(length(selection$common.terms)) paste0("Already-common terms add no restrictions: ",
                    paste(selection$common.terms,collapse=", "),".") else "")),
            class=c("fastglm_po_test","fastglm_test","htest"))
    }
    collect <- function(results) if(!by.term) results[[1L]] else
        structure(list(tests=results,method=title,null.estimation=estimation,note=paste(note,
            if(partial) "Each term-specific null restricts only that term; other varying effects remain unrestricted.",
            "Term-specific p-values are unadjusted for multiple testing.")),class="fastglm_test_list")
    unavailable <- function(reason,detail,empty=character()) collect(lapply(selections,function(selection) {
        out <- template(selection);out$reason <- reason;out$detail <- detail;out$empty.categories <- empty;out
    }))
    error <- ""
    scan <- tryCatch(.ordinal_stream_po_scan(object),error=function(e) {error <<- conditionMessage(e);NULL})
    if(is.null(scan)) {
        scan <- list(observations=NA_real_,frequency=NA_real_,separation=NULL)
        return(unavailable("source_read_error",error))
    }
    if(scan$frequency<=0) return(unavailable("no_usable_outcomes","No positive-frequency source rows are available."))
    empty <- object$levels[scan$totals==0]
    if(length(empty)) return(unavailable("empty_categories",paste0(
        "Every declared category must have positive frequency; empty: ",paste(empty,collapse=", "),"."),empty))
    if(length(scan$constant)) return(unavailable("rank_deficient_design",paste0(
        "No predictor variation in the source rows: ",paste(scan$constant,collapse=", "),".")))
    restrictions <- vapply(selections,function(s)if(partial)nrow(s$contrast) else length(s$index)*(K-2L),numeric(1))
    if(estimation=="ML" && !is.null(scan$separation) && any(restrictions>0L)) {
        sep <- scan$separation
        return(unavailable("null_separation",paste0("Predictor '",sep$column,"' at ",
            format(sep$value,trim=TRUE)," occurs only in endpoint category '",object$levels[sep$category],
            "'. The ML null has no finite solution. A separately requested meanBR null remains a first-order approximation.")))
    }
    source <- .ordinal_stream_po_source(object)
    refit <- function(nonparallel,firth) fastglm_ordinal_streaming(source,levels=object$levels,
        nonparallel=nonparallel,firth=firth,link=.ordinal_link(object),tol=ctl$tol,
        maxit=ctl$maxit,block.size=object$block.size)
    warning <- character()
    attempt <- function(fun) {
        error <<- "";warning <<- character()
        tryCatch(withCallingHandlers(fun(),warning=function(w) {
            warning <<- c(warning,conditionMessage(w));invokeRestart("muffleWarning")
        }),error=function(e) {error <<- conditionMessage(e);NULL})
    }
    full <- common <- moments <- NULL;shared.error <- "";shared.warning <- character()
    if(any(restrictions>0L) && (method=="lr" || !partial)) {
        shared <- attempt(function()refit(if(partial)object$nonparallel else integer(),estimation=="meanBR"))
        if(partial) full <- shared else common <- shared
        shared.error <- error;shared.warning <- warning
        if(!partial && !is.null(common) && isTRUE(common$converged)) {
            moments <- attempt(function().ordinal_stream_po_moments(object,common,selected$index))
            shared.error <- error;shared.warning <- c(shared.warning,warning)
        }
    }
    collect(lapply(selections,function(selection) {
        out <- template(selection)
        fail <- function(reason,detail) {out$reason <- reason;out$detail <- detail;out}
        if(!out$df) return(fail("no_slope_contrasts",if(!partial)
            "At least three categories and one predictor are required." else if(K<3L)
            "A binary outcome has only one threshold, so there is no common-slope restriction to test." else
            "The selected terms already have common slopes; no equal-slope restrictions were added."))
        out$warning <- unique(shared.warning)
        if(method=="lr") {
            if(is.null(full)) return(fail("full_fit_error",shared.error))
            out$full.converged <- isTRUE(full$converged);out$full.iterations <- full$iterations
            out$diagnostics <- list(full=.ordinal_stream_po_diagnostics(full,object$nonparallel,"ML"))
            if(!out$full.converged) return(fail("full_fit_not_converged",paste0(
                "The ML alternative did not converge: ",gsub("_"," ",full$convergence.reason,fixed=TRUE),".")))
        }
        remaining <- if(partial) setdiff(object$nonparallel,selection$index) else integer()
        null <- if(partial) attempt(function()refit(remaining,estimation=="meanBR")) else common
        if(partial) out$warning <- unique(c(out$warning,warning))
        null.error <- if(partial) error else shared.error
        if(is.null(null)) return(fail(if(grepl("rank|constant",null.error)) "rank_deficient_design" else "null_fit_error",null.error))
        out$null.converged <- isTRUE(null$converged);out$null.iterations <- null$iterations
        out$diagnostics$null <- .ordinal_stream_po_diagnostics(null,remaining,estimation)
        if(method=="lr") out$diagnostics$restricted <- out$diagnostics$null
        if(!out$null.converged) return(fail("null_fit_not_converged",paste0(
            "The ",estimation," null did not converge: ",gsub("_"," ",null$convergence.reason,fixed=TRUE),".")))
        result <- attempt(function() {
            if(method=="lr") return(list(statistic=.ordinal_profile_lr(
                list(converged=full$converged,value=full$loglik),
                list(converged=null$converged,value=null$loglik),list(score.tol=ctl$tol))))
            if(partial) return(.ordinal_stream_partial_po_score(object,null,selection$contrast))
            if(is.null(moments)) stop(shared.error,call.=FALSE)
            positions <- match(selection$index,selected$index)
            tested <- moments$nuisance+as.vector(outer(positions,
                seq.int(0L,K-3L)*length(selected$index),"+"))
            .ordinal_po_score_statistic(moments,tested)
        })
        out$warning <- unique(c(out$warning,warning))
        if(is.null(result)) return(fail(if(method=="lr")"likelihood_unavailable" else "information_unavailable",error))
        out$statistic <- result$statistic;out$p.value <- pchisq(out$statistic,out$df,lower.tail=FALSE)
        if(method=="lr") out$loglik <- c(full=full$loglik,restricted=null$loglik) else {
            out$information.rcond <- result$condition
            if(partial) {out$null.coefficients <- result$coefficients;out$ordinary.nuisance.score <- result$nuisance.score}
        }
        out$available <- TRUE;out$reason <- "available";out
    }))
}
