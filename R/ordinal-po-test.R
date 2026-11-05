# Ordinary-score common-slope tests. Expected information and ordinary scores
# use the same frequency normalization.
.ordinal_po_test_columns <- function(object,terms) {
    columns <- names(object$beta)
    if(is.null(terms)) return(list(index=seq_along(columns),terms=if(is.null(object$terms))
        columns else attr(object$terms,"term.labels"),columns=columns))
    if(!is.character(terms) || !length(terms) || anyNA(terms) || any(!nzchar(terms)) || anyDuplicated(terms))
        stop("'po.terms' must contain distinct, nonmissing term or column names",call.=FALSE)
    if(is.null(object$terms)) {
        if(any(!terms %in% columns)) stop("unknown predictor column in 'po.terms'",call.=FALSE)
        index <- which(columns %in% terms)
    } else {
        labels <- attr(object$terms,"term.labels")
        if(any(!terms %in% labels)) stop("unknown formula term in 'po.terms'",call.=FALSE)
        assignment <- object$design.assign
        if(is.null(assignment) || length(assignment)!=length(columns))
            stop("refit this ordinal model to retain the design assignments required by 'po.terms'",call.=FALSE)
        index <- which(assignment %in% match(terms,labels))
    }
    if(!length(index)) stop("'po.terms' selects no location predictors",call.=FALSE)
    list(index=index,terms=terms,columns=columns[index])
}

.ordinal_po_test_control <- function(control) {
    if(!is.list(control) || (length(control) && (is.null(names(control)) || anyNA(names(control)) ||
       any(!nzchar(names(control))) || anyDuplicated(names(control)) || any(!names(control)%in%c("tol","maxit")))))
        stop("'po.test.control' permits only named 'tol' and 'maxit' entries",call.=FALSE)
    out <- utils::modifyList(list(tol=1e-10,maxit=500L),control)
    scalar <- function(z) is.numeric(z) && !is.complex(z) && length(z)==1L && is.finite(z) && z>0
    if(!scalar(out$tol) || !scalar(out$maxit) || out$maxit!=floor(out$maxit) || out$maxit>.Machine$integer.max)
        stop("'po.test.control' requires positive 'tol' and positive integer 'maxit'",call.=FALSE)
    out
}

.ordinal_po_score_moments <- function(x,response,offset,fit,selected,block.size) {
    n <- nrow(x); p <- ncol(x); q <- length(fit$alpha); s <- length(selected)
    d0 <- q+p; d <- d0+s*(q-1L); frequency <- fit$frequency
    alpha <- fit$working$alpha; beta <- fit$working$beta
    gap <- diff(alpha)
    if(any(!is.finite(gap)|gap<=0)) stop("the null fit requires finite, strictly ordered thresholds",call.=FALSE)
    tgap <- exp(-gap)/(-expm1(-gap))
    U <- numeric(d); I <- matrix(0,d,d)
    ia <- seq_len(q); ib <- q+seq_len(p)
    ig <- function(j) d0+(j-2L)*s+seq_len(s)
    for(first in seq.int(1L,n,by=block.size)) {
        at <- seq.int(first,min(n,first+block.size-1L))
        xx <- .ordinal_prediction_scale(x[at,,drop=FALSE],fit$working$center,fit$working$scale)
        zz <- xx[,selected,drop=FALSE]
        eta <- outer(-streaming_qr_eta(xx,beta,offset[at]),alpha,"+")
        logistic <- .ordinal_link(fit)=="logit"
        if(logistic) {F <- plogis(eta); S <- plogis(-eta); g <- F*S}
        else {
            kernel <- .ordinal_link_kernel(alpha,streaming_qr_eta(xx,beta,offset[at]),.ordinal_link(fit),scores=TRUE)
            F <- kernel$cumulative; S <- kernel$survival; g <- kernel$derivative
        }
        counts <- .ordinal_assessment_counts(response,at,q+1L)/frequency
        mass <- response$totals[at]/frequency
        threshold <- diagonal <- rowsum <- matrix(0,length(at),q)
        adjacent <- matrix(0,length(at),q-1L)
        location <- if(logistic) numeric(length(at)) else -rowSums(counts*kernel$translation)
        probability <- if(logistic) cbind(F[,1L],F[,-1L,drop=FALSE]*S[,-q,drop=FALSE]*
            rep(-expm1(-gap),each=length(at)),S[,q]) else kernel$probability
        if(any(!is.finite(probability)|probability<=0))
            stop("the fitted null has a category probability outside the representable positive range",call.=FALSE)
        for(j in seq_len(q)) {
            left <- if(logistic) S[,j]+if(j>1L) tgap[j-1L] else 0 else kernel$left[,j]
            right <- if(logistic) F[,j]+if(j<q) tgap[j] else 0 else kernel$right[,j]
            threshold[,j] <- counts[,j]*left-counts[,j+1L]*right
            diagonal[,j] <- mass*g[,j]*(left+right)
            # Direct row sums avoid cancellation when an interior category is narrow.
            rowsum[,j] <- mass*g[,j]*(if(logistic) probability[,j]+probability[,j+1L] else kernel$translation.difference[,j])
            if(j<q) adjacent[,j] <- -mass*g[,j+1L]*right
            if(logistic) location <- location+counts[,j+1L]*F[,j]-counts[,j]*S[,j]
        }
        U[ia] <- U[ia]+colSums(threshold)
        U[ib] <- U[ib]+drop(crossprod(xx,location))
        I[cbind(ia,ia)] <- I[cbind(ia,ia)]+colSums(diagonal)
        I[cbind(ia[-q],ia[-1L])] <- I[cbind(ia[-q],ia[-1L])]+colSums(adjacent)
        I[ia,ib] <- I[ia,ib]-crossprod(rowsum,xx)
        I[ib,ib] <- I[ib,ib]+crossprod(xx,xx*rowSums(rowsum))
        dg <- crossprod(diagonal,zz); ad <- crossprod(adjacent,zz)
        for(j in 2:q) {
            index <- ig(j)
            U[index] <- U[index]-drop(crossprod(zz,threshold[,j]))
            I[j,index] <- I[j,index]-dg[j,]
            I[j-1L,index] <- I[j-1L,index]-ad[j-1L,]
            if(j<q) I[j+1L,index] <- I[j+1L,index]-ad[j,]
            I[ib,index] <- I[ib,index]+crossprod(xx,zz*rowsum[,j])
            I[index,index] <- I[index,index]+crossprod(zz,zz*diagonal[,j])
            if(j<q) I[index,ig(j+1L)] <- I[index,ig(j+1L)]+crossprod(zz,zz*adjacent[,j])
        }
    }
    I[lower.tri(I)] <- t(I)[lower.tri(I)]
    if(any(!is.finite(U)) || any(!is.finite(I))) stop("score or expected information is nonfinite",call.=FALSE)
    list(score=U,information=I,nuisance=d0,frequency=frequency)
}


.ordinal_po_score_statistic <- function(moments,tested=NULL) {
    d0 <- moments$nuisance
    if(is.null(tested)) tested <- seq.int(d0+1L,length(moments$score))
    keep <- c(seq_len(d0),tested)
    U <- moments$score[keep]; I <- moments$information[keep,keep,drop=FALSE]
    d <- length(U); frequency <- moments$frequency
    # Normalize parameter coordinates before factoring to avoid unit-dependent
    # numerical rejection. This does not change the quadratic statistic.
    scale <- sqrt(diag(I))
    if(any(!is.finite(scale)|scale<=0)) stop("expected information has a nonpositive diagonal",call.=FALSE)
    I <- I/outer(scale,scale); U <- U/scale
    nuisance <- seq_len(d0); tested <- seq.int(d0+1L,d)
    A <- I[nuisance,nuisance,drop=FALSE]; C <- I[tested,nuisance,drop=FALSE]
    root <- chol(A)
    projection <- t(backsolve(root,forwardsolve(t(root),t(C))))
    e <- drop(U[tested]-projection%*%U[nuisance])
    J <- I[tested,tested,drop=FALSE]-projection%*%t(C); J <- (J+t(J))/2
    js <- sqrt(diag(J))
    if(any(!is.finite(js)|js<=0)) stop("efficient expected information is not positive definite",call.=FALSE)
    J <- J/outer(js,js); e <- e/js
    root <- chol(J); condition <- min(rcond(A),rcond(J))
    if(!is.finite(condition) || condition<1e-14)
        stop("expected information is numerically singular",call.=FALSE)
    statistic <- frequency*sum(forwardsolve(t(root),e)^2)
    if(!is.finite(statistic)) stop("the score statistic is nonfinite",call.=FALSE)
    list(statistic=statistic,condition=condition)
}

.ordinal_po_expected_score <- function(x,response,offset,fit,selected,block.size) {
    .ordinal_po_score_statistic(.ordinal_po_score_moments(x,response,offset,fit,selected,block.size))
}

.ordinal_po_separation <- function(x,response,K) {
    # A sufficient certificate, not a general separation detector. For a
    # two-valued column, shifting its slope and every threshold can leave one
    # group unchanged while increasing an endpoint-only group's likelihood.
    for(j in seq_len(ncol(x))) {
        values <- unique(x[,j])
        if(length(values)!=2L) next
        for(value in values) {
            at <- which(x[,j]==value)
            counts <- if(response$grouped) colSums(response$y[at,,drop=FALSE]*response$weights[at]) else
                tabulate(response$codes[at],nbins=K)
            present <- which(counts>0)
            if(length(present)==1L && present %in% c(1L,K))
                return(list(column=colnames(x)[j],value=value,category=present))
        }
    }
    NULL
}

.ordinal_assessment_po_test <- function(object,design,response,included,terms,control,
                                       estimation="ML",by.term=FALSE,method="score") {
    if(inherits(object,"fastglm_ordinal_partial"))
        return(.ordinal_partial_po_test(object,design,response,included,terms,control,estimation,by.term,method))
    selected <- .ordinal_po_test_columns(object,terms); ctl <- .ordinal_po_test_control(control)
    K <- length(object$levels); p <- length(object$beta)
    out <- structure(list(method=if(.ordinal_link(object)=="logit") "Expected-information proportional-odds score test" else
        paste0("Expected-information common-slope score test (",.ordinal_link(object),")"),available=FALSE,
        statistic=NA_real_,df=length(selected$index)*(K-2L),p.value=NA_real_,
        null.estimation=estimation,terms=selected$terms,columns=selected$columns,
        observations=sum(included),frequency=sum(response$totals[included]),
        reason="",detail="",null.converged=FALSE,null.iterations=NA_integer_,
        information.rcond=NA_real_,empty.categories=character(),warning=character(),separation=NULL,
        parameter=c(df=length(selected$index)*(K-2L)),data.name="ordinal assessment outcomes",
        note=if(estimation=="ML") "First-order chi-square approximation." else
            "Ordinary efficient score at a mean-bias-reduced null; first-order chi-square approximation."),
        class=c("fastglm_po_test","fastglm_test","htest"))
    selections <- list(Joint=selected)
    if(by.term) for(term in selected$terms)
        selections[[length(selections)+1L]] <- .ordinal_po_test_columns(object,term)
    if(by.term) names(selections) <- make.unique(c("Joint",selected$terms))
    template <- function(selection) {
        z <- out; z$terms <- selection$terms; z$columns <- selection$columns
        z$df <- length(selection$index)*(K-2L); z$parameter <- c(df=z$df); z
    }
    collect <- function(results) if(!by.term) results[[1L]] else
        structure(list(tests=results,method=out$method,null.estimation=estimation,
            note=paste(out$note,"Term-specific p-values are unadjusted for multiple testing.")),
            class="fastglm_test_list")
    unavailable <- function(reason,detail) {
        collect(lapply(selections,function(selection) {
            z <- template(selection); z$reason <- reason; z$detail <- detail; z
        }))
    }
    if(K<3L || !p) return(unavailable("no_slope_contrasts","At least three categories and one predictor are required."))
    if(!any(included)) return(unavailable("no_usable_outcomes","No positive-frequency complete assessment rows are available."))
    if(!is.matrix(design$x) || !is.numeric(design$x))
        return(unavailable("unsupported_design","The score test currently requires a dense numeric design."))
    index <- which(included)
    xx <- design$x[index,,drop=FALSE]; off <- design$offset[index]
    yy <- if(response$grouped) response$y[index,,drop=FALSE] else response$codes[index]
    rr <- .ordinal_assessment_response(yy,response$weights[index],object$levels,length(index))
    totals <- numeric(K)
    for(first in seq.int(1L,length(index),by=object$block.size)) {
        at <- seq.int(first,min(length(index),first+object$block.size-1L))
        totals <- totals+colSums(.ordinal_assessment_counts(rr,at,K))
    }
    out$empty.categories <- object$levels[totals==0]
    if(length(out$empty.categories)) return(unavailable("empty_categories",
        paste0("Every declared category must have positive frequency; empty: ",paste(out$empty.categories,collapse=", "),".")))
    constant <- vapply(seq_len(p),function(j) all(xx[,j]==xx[1L,j]),logical(1))
    if(any(constant)) return(unavailable("rank_deficient_design",paste0(
        "No predictor variation in the assessment rows: ",paste(colnames(xx)[constant],collapse=", "),
        ". The requested model is not identifiable.")))
    out$separation <- .ordinal_po_separation(xx,rr,K)
    if(estimation=="ML" && !is.null(out$separation)) {
        sep <- out$separation; out$null.iterations <- 0L
        return(unavailable("null_separation",paste0("Predictor '",sep$column,"' at ",
            format(sep$value,trim=TRUE)," occurs only in endpoint category '",object$levels[sep$category],
            "'. The ML null has no finite solution. A separately requested meanBR null may be used; its test remains an approximation.")))
    }
    warning <- character(); error <- ""
    fit <- tryCatch(withCallingHandlers(.ordinal_fit_core(xx,yy,rr$weights,off,object$levels,
        firth=estimation=="meanBR",tol=ctl$tol,maxit=ctl$maxit,block.size=object$block.size,link=.ordinal_link(object)),warning=function(w) {
            warning <<- c(warning,conditionMessage(w)); invokeRestart("muffleWarning")
        }),error=function(e) {error <<- conditionMessage(e); NULL})
    out$warning <- unique(warning)
    if(is.null(fit)) return(unavailable(if(grepl("rank|constant column",error)) "rank_deficient_design" else "null_fit_error",error))
    out$null.converged <- isTRUE(fit$converged); out$null.iterations <- fit$iterations
    if(!out$null.converged) return(unavailable("null_fit_not_converged",
        paste0("The ",estimation," common-slope null did not converge: ",gsub("_"," ",fit$convergence.reason,fixed=TRUE),".")))
    moments <- tryCatch(.ordinal_po_score_moments(xx,rr,off,fit,selected$index,object$block.size),
        error=function(e) {error <<- conditionMessage(e); NULL})
    if(is.null(moments)) return(unavailable("information_unavailable",error))
    collect(lapply(selections,function(selection) {
        z <- template(selection)
        positions <- match(selection$index,selected$index)
        tested <- moments$nuisance+as.vector(outer(positions,
            seq.int(0L,K-3L)*length(selected$index),"+"))
        result <- tryCatch(.ordinal_po_score_statistic(moments,tested),
            error=function(e) {error <<- conditionMessage(e); NULL})
        if(is.null(result)) {z$reason <- "information_unavailable"; z$detail <- error; return(z)}
        z$available <- TRUE; z$statistic <- result$statistic
        z$p.value <- pchisq(z$statistic,z$df,lower.tail=FALSE)
        z$information.rcond <- result$condition; z$reason <- "available"; z
    }))
}

.ordinal_partial_po_columns <- function(object,terms) {
    columns <- object$design.columns
    if(!is.null(object$terms)) {
        labels <- attr(object$terms,"term.labels");assignment <- object$design.assign
        if(length(assignment)!=length(columns))
            stop("common-slope term tests require retained design assignments",call.=FALSE)
        if(is.null(terms)) terms <- labels[seq_along(labels)%in%assignment[object$nonparallel]]
        if(!is.character(terms) || !length(terms) || anyNA(terms) || anyDuplicated(terms) || any(!terms%in%labels))
            stop("unknown or duplicated formula term in 'terms'",call.=FALSE)
        selected <- which(assignment%in%match(terms,labels))
        common.terms <- terms[!match(terms,labels)%in%assignment[object$nonparallel]]
    } else {
        if(is.null(terms)) terms <- columns[object$nonparallel]
        if(!is.character(terms) || !length(terms) || anyNA(terms) || anyDuplicated(terms) || any(!terms%in%columns))
            stop("unknown or duplicated predictor column in 'terms'",call.=FALSE)
        selected <- which(columns%in%terms)
        common.terms <- setdiff(terms,columns[object$nonparallel])
    }
    selected <- intersect(selected,object$nonparallel)
    q <- length(object$alpha);pc <- length(object$common);pn <- length(object$nonparallel)
    L <- matrix(0,length(selected)*max(0L,q-1L),length(coef(object)),
        dimnames=list(NULL,names(coef(object))))
    if(nrow(L)) {
        first <- q+pc+match(selected,object$nonparallel)
        labels <- character(nrow(L))
        for(j in 2:q) {
            at <- (j-2L)*length(selected)+seq_along(selected)
            later <- first+(j-1L)*pn
            L[cbind(at,later)] <- 1;L[cbind(at,first)] <- -1
            labels[at] <- paste(names(coef(object))[later],"-",names(coef(object))[first])
        }
        rownames(L) <- labels
    }
    list(index=selected,terms=terms,columns=columns[selected],common.terms=common.terms,contrast=L)
}

# Embed a reduced-null estimate in the full model without fitting the full
# alternative. In particular, meanBR belongs to the reduced null, while the
# ordinary score and expected information belong to the full partial model.
.ordinal_partial_po_null_setup <- function(object,fit,x,offset,control) {
    q <- length(object$alpha);pc <- length(object$common);pn <- length(object$nonparallel)
    common.names <- object$design.columns[object$common]
    varying.names <- object$design.columns[object$nonparallel]
    gamma <- matrix(0,pn,q)
    for(j in seq_len(pn)) gamma[j,] <- if(varying.names[j]%in%names(fit$working$beta))
        fit$working$beta[varying.names[j]] else fit$working$gamma[varying.names[j],]
    coefficients <- c(fit$working$alpha,fit$working$beta[common.names],as.vector(gamma))
    xx <- .ordinal_prediction_scale(x,fit$working$center,fit$working$scale)
    category <- fit$category
    state <- ordinal_partial_native_evaluate(xx[,object$common,drop=FALSE],xx[,object$nonparallel,drop=FALSE],
        category$codes,category$counts,category$weights,offset,q+1L,coefficients,FALSE,
        object$block.size,control$condition.tol,.ordinal_link_code(.ordinal_link(object)))
    state$coefficients <- as.numeric(coefficients)
    state$converged <- TRUE;state$convergence.reason <- "reduced_null_fit"
    state$iterations <- fit$iterations;state$evaluations <- fit$evaluations
    state$history <- matrix(numeric(),0L,4L)
    full <- .ordinal_partial_result(state,fit$working$center,fit$working$scale,category,
        fit$frequency,fit$nobs.rows,fit$firth,.ordinal_link(object),fit$control,fit$tol,
        fit$maxit,object$block.size,object$design.columns,object$nonparallel)
    full$x <- x;full$offset <- offset
    setup <- .ordinal_profile_setup(full,control,profile=FALSE)
    list(setup=setup,state=state,coefficients=coef(full))
}

.ordinal_partial_po_test <- function(object,design,response,included,terms,control,
                                      estimation="ML",by.term=FALSE,method="score") {
    selected <- .ordinal_partial_po_columns(object,terms);ctl <- .ordinal_po_test_control(control)
    inference.control <- .ordinal_profile_control(list(maxit=ctl$maxit,score.tol=ctl$tol,step.tol=ctl$tol))
    title <- if(.ordinal_link(object)=="logit") "Partial proportional-odds" else
        paste0("Partial common-slope (",.ordinal_link(object),")")
    title <- paste(title,if(method=="lr") "likelihood ratio test" else "expected-information score test")
    note <- if(method=="lr") "Both alternative and null are refitted by maximum likelihood; first-order chi-square approximation." else
        if(estimation=="ML") "Ordinary efficient score at the ML null; first-order chi-square approximation." else
        "Ordinary efficient score at a reduced-null mean-bias-reduced estimate; this is not the full-model adjusted-score test. First-order chi-square approximation."
    selections <- list(Joint=selected)
    if(by.term) {
        selections <- c(selections,lapply(selected$terms,function(term) .ordinal_partial_po_columns(object,term)))
        names(selections) <- make.unique(c("Joint",selected$terms))
    }
    separation <- NULL
    template <- function(selection) structure(list(method=title,available=FALSE,statistic=NA_real_,
        df=nrow(selection$contrast),parameter=c(df=nrow(selection$contrast)),p.value=NA_real_,
        null.estimation=estimation,terms=selection$terms,columns=selection$columns,
        already.common.terms=selection$common.terms,contrast=selection$contrast,
        observations=sum(included),frequency=sum(response$totals[included]),reason="",detail="",
        null.converged=FALSE,null.iterations=NA_integer_,information.rcond=NA_real_,
        empty.categories=character(),warning=character(),separation=separation,
        data.name="ordinal assessment outcomes",note=paste(note,
            if(length(selection$common.terms)) paste0("Already-common terms add no restrictions: ",
                paste(selection$common.terms,collapse=", "),".") else "")),
        class=c("fastglm_po_test","fastglm_test","htest"))
    collect <- function(results) if(!by.term) results[[1L]] else
        structure(list(tests=results,method=title,null.estimation=estimation,note=paste(note,
            "Each term-specific null restricts only that term; other varying effects remain unrestricted.",
            "Term-specific p-values are unadjusted for multiple testing.")),class="fastglm_test_list")
    unavailable <- function(reason,detail,empty=character()) collect(lapply(selections,function(selection) {
        out <- template(selection);out$reason <- reason;out$detail <- detail;out$empty.categories <- empty;out
    }))
    if(!any(included)) return(unavailable("no_usable_outcomes","No positive-frequency complete assessment rows are available."))
    if(!is.matrix(design$x) || !is.numeric(design$x))
        return(unavailable("unsupported_design","The test requires a dense numeric evaluation design."))
    index <- which(included);xx <- design$x[index,,drop=FALSE];off <- design$offset[index]
    yy <- if(response$grouped) response$y[index,,drop=FALSE] else response$codes[index]
    ww <- response$weights[index];K <- length(object$levels)
    rr <- .ordinal_assessment_response(yy,ww,object$levels,length(index))
    totals <- numeric(K)
    for(first in seq.int(1L,length(index),by=object$block.size)) {
        at <- seq.int(first,min(length(index),first+object$block.size-1L))
        totals <- totals+colSums(.ordinal_assessment_counts(rr,at,K))
    }
    empty <- object$levels[totals==0]
    if(length(empty)) return(unavailable("empty_categories",paste0(
        "Every declared category must have positive frequency; empty: ",paste(empty,collapse=", "),"."),empty))
    separation <- .ordinal_po_separation(xx,rr,K)
    if(estimation=="ML" && !is.null(separation) &&
       any(vapply(selections,function(s)nrow(s$contrast)>0,logical(1))))
        return(unavailable("null_separation",paste0("Predictor '",separation$column,"' at ",
            format(separation$value,trim=TRUE)," occurs only in endpoint category '",
            object$levels[separation$category],"'. The ML null has no finite solution. ",
            "A separately requested meanBR score-test null remains a first-order approximation.")))
    fit <- function(nonparallel,firth) .ordinal_fit_matrix(xx,yy,ww,off,object$levels,
        firth,ctl$tol,ctl$maxit,object$block.size,list(),link=.ordinal_link(object),nonparallel=nonparallel)
    full <- NULL;full.error <- "";full.warning <- character()
    if(method=="lr" && any(vapply(selections,function(s)nrow(s$contrast)>0,logical(1))))
        full <- tryCatch(withCallingHandlers(fit(object$nonparallel,FALSE),warning=function(w) {
            full.warning <<- c(full.warning,conditionMessage(w));invokeRestart("muffleWarning")
        }),error=function(e) {full.error <<- conditionMessage(e);NULL})
    collect(lapply(selections,function(selection) {
        out <- template(selection);L <- selection$contrast
        fail <- function(reason,detail) {out$reason <- reason;out$detail <- detail;out}
        if(!nrow(L)) return(fail("no_slope_contrasts",if(K<3L)
            "A binary outcome has only one threshold, so there is no common-slope restriction to test." else
            "The selected terms already have common slopes; no equal-slope restrictions were added."))
        warning <- full.warning;error <- ""
        if(method=="lr") {
            out$warning <- unique(warning)
            if(is.null(full)) return(fail("full_fit_error",full.error))
            out$full.converged <- isTRUE(full$converged);out$full.iterations <- full$iterations
            if(!isTRUE(full$converged)) return(fail("full_fit_not_converged",paste0(
                "The ML alternative did not converge: ",gsub("_"," ",full$convergence.reason,fixed=TRUE),".")))
            result <- tryCatch(withCallingHandlers(.ordinal_profile_test(full,contrast=L,
                control=inference.control),warning=function(w) {
                warning <<- c(warning,conditionMessage(w));invokeRestart("muffleWarning")
            }),error=function(e) {error <<- conditionMessage(e);NULL})
            out$warning <- unique(warning)
            if(is.null(result)) return(fail("constrained_fit_error",error))
            out$diagnostics <- result$diagnostics;out$loglik <- result$loglik
            out$full.converged <- isTRUE(result$diagnostics$full$converged)
            out$full.iterations <- result$diagnostics$full$iterations
            restricted <- result$diagnostics$restricted
            out$null.converged <- isTRUE(restricted$converged)
            out$null.iterations <- if(is.null(restricted)) NA_integer_ else restricted$iterations
            if(!isTRUE(result$available)) return(fail("constrained_fit_not_converged",result$diagnostics$message))
            out$statistic <- unname(result$statistic);out$p.value <- result$p.value
        } else {
            remaining <- setdiff(object$nonparallel,selection$index)
            null <- tryCatch(withCallingHandlers(fit(remaining,estimation=="meanBR"),warning=function(w) {
                warning <<- c(warning,conditionMessage(w));invokeRestart("muffleWarning")
            }),error=function(e) {error <<- conditionMessage(e);NULL})
            out$warning <- unique(warning)
            if(is.null(null)) return(fail(if(grepl("rank|constant",error)) "rank_deficient_design" else "null_fit_error",error))
            out$null.converged <- isTRUE(null$converged);out$null.iterations <- null$iterations
            out$diagnostics <- list(null=list(converged=null$converged,iterations=null$iterations,
                score=null$scaled.score,predictor.step=null$predictor.step,message=null$convergence.reason,
                coefficients=coef(null),nonparallel=remaining,estimation=estimation))
            if(!out$null.converged) return(fail("null_fit_not_converged",paste0(
                "The ",estimation," null did not converge: ",gsub("_"," ",null$convergence.reason,fixed=TRUE),".")))
            result <- tryCatch({
                embedded <- .ordinal_partial_po_null_setup(object,null,xx,off,inference.control)
                setup <- embedded$setup;state <- embedded$state
                B <- .ordinal_constraint_basis(setup,L)$basis
                ordinary <- as.numeric(state$ordinary.score)
                efficient <- ordinary-state$information%*%B%*%
                    solve(crossprod(B,state$information%*%B),crossprod(B,ordinary))
                statistic <- drop(crossprod(efficient,state$cov.unscaled%*%efficient))
                if(!is.finite(statistic) || statistic<0) stop("invalid efficient score quadratic",call.=FALSE)
                list(statistic=statistic,condition=state$information.rcond,
                    coefficients=embedded$coefficients,nuisance.score=drop(crossprod(B,ordinary)))
            },error=function(e) {error <<- conditionMessage(e);NULL})
            if(is.null(result)) return(fail("information_unavailable",error))
            out$statistic <- result$statistic;out$p.value <- pchisq(out$statistic,nrow(L),lower.tail=FALSE)
            out$information.rcond <- result$condition
            out$null.coefficients <- result$coefficients;out$ordinary.nuisance.score <- result$nuisance.score
        }
        out$available <- TRUE;out$reason <- "available";out
    }))
}

#' Test common slopes in a cumulative-link model
#'
#' Computes a common-slope test directly, without calculating
#' prediction losses or calibration summaries. Print the result for a compact
#' test table, or use `as.data.frame()` or `broom::tidy()` for reporting.
#' @param object a `fastglm_ordinal` fit retaining its design and outcomes,
#'   or an ordinal source fit with its original block reader.
#'   Its original coefficient estimates are not used in the test. The requested
#'   null is refitted and must converge, including if the original fit did not.
#' @param terms formula terms, or named matrix columns, to test jointly.
#'   For a common-slope fit, the default tests all location predictors. For a
#'   partial fit, the default selects all currently nonparallel terms. Complete
#'   factor and interaction blocks are selected together. Terms that already
#'   have common slopes add no restrictions; they are never tested for zero
#'   effects by this function.
#' @param null.estimation `"ML"` (default) or `"meanBR"`. The latter evaluates
#'   the ordinary efficient score at a mean-bias-reduced null fit, with explicit
#'   projection of the ordinary nuisance score. It can supply a finite null
#'   under separation, but does not guarantee convergence or small-sample accuracy.
#' @param newdata,y,weights,offset assessment data as in [fastglm_ordinal_assess()].
#' @param control named list containing `tol` and/or `maxit` for refitting.
#'   For a partial likelihood ratio test these controls apply to both models
#'   and their constrained convergence checks.
#' @param by.term also test each selected formula term or matrix column
#'   separately. For a common-slope model, tests share one null refit; each
#'   term-specific alternative relaxes only that term. For a partial model,
#'   each term-specific null restricts only that term, leaving other varying
#'   effects unrestricted. These tests require separate null fits.
#' @param method `"score"` (default) for the expected-information ordinary
#'   efficient score, or `"lr"` for a likelihood ratio test of a partial model.
#'   Likelihood ratio tests require `null.estimation="ML"` and refit both
#'   the full partial model and the constrained null by ML, including when
#'   the supplied object was fitted with mean bias reduction.
#' @details Both null-estimation choices for score tests use expected
#'   information and an asymptotic chi-square
#'   reference. Mean bias reduction gives a first-order equivalent score test
#'   under a regular interior null with fixed predictor and category dimensions.
#'   The nuisance projection is essential because the ordinary nuisance score
#'   is nonzero at a bias-reduced root. This is not an adjusted-score test with
#'   established higher-order accuracy. It can be conservative in small samples.
#'   The method never switches null estimators automatically.
#'   The fitted link is retained. For logit models the hypothesis is proportional
#'   odds; for probit and complementary log-log models it is the common-slope
#'   restriction on that link's scale.
#'
#'   For a partial model, a selected predictor with slopes `gamma[,j]` is tested
#'   through `gamma[,j] - gamma[,1] = 0` for every threshold after the first.
#'   Other varying effects remain unrestricted in both the null and alternative.
#'   If `s` selected design columns have varying slopes across `K` categories,
#'   this imposes `s*(K-2)` independent restrictions. Binary outcomes and
#'   selections containing only already-common terms return an unavailable
#'   test with zero degrees of freedom and an explanation.
#'
#'   For partial score tests, `null.estimation="meanBR"` fits bias reduction
#'   in the reduced null model, embeds that estimate in the full partial model,
#'   and projects its ordinary score using full-model expected information.
#'   This preserves the common-slope test convention. It differs from
#'   [fastglm_score_test()] with `null.estimation="meanBR"`, which uses the
#'   full-model adjustment in constrained fitting and in the score statistic.
#'   Likelihood ratio tests use ordinary ML likelihoods and first-order
#'   chi-square references; an unsuccessful full or constrained fit is reported
#'   as unavailable. No finite-sample or higher-order calibration is claimed.
#'
#'   Empty categories and deficient predictor rank remain unsupported. A
#'   sufficient separation check identifies two-valued predictors with a level
#'   observed only in an endpoint category. It does not detect all forms of
#'   separation; other unsuccessful null fits retain their convergence details.
#'   See [fastglm_ordinal_assess()] for the hypothesis, assumptions, and frequency
#'   conventions, and [fastglm-test-methods] for result extraction.
#'   With `by.term=TRUE`, term-specific p-values are unadjusted for multiple
#'   testing. The joint test remains the single test of all selected terms.
#'
#'   An ordinal source fit is tested on its original source when `newdata` and
#'   `y` are omitted. The source is replayed for refitting and information
#'   accumulation without collecting its design or outcomes. Recorded frequency
#'   weights and offsets are used; overrides require explicit evaluation data.
#'   Score tests retain the same null-estimation and term-selection conventions
#'   as dense fits. Partial-model likelihood ratio tests fit the equivalent
#'   reduced common-slope model and the full model by streamed ML. Memory is
#'   bounded by one source block and coefficient-sized matrices, in addition to
#'   the fitted source's existing block metadata. The reader must reproduce the
#'   same observations on every pass, as required for source fitting itself.
#' @references Kosmidis, I. (2014). Improved estimation in cumulative link
#'   models. Journal of the Royal Statistical Society, Series B, 76, 169--196.
#'   \doi{10.1111/rssb.12025}. The present meanBR option uses the ordinary
#'   projected score at a bias-reduced null, rather than the larger-model
#'   adjusted-score statistic discussed in Section 12 of that article.
#' @return A `fastglm_po_test` object inheriting from `fastglm_test` and `htest`,
#'   with numeric statistic, degrees of freedom, p-value, availability, null
#'   estimation method, tested terms, and detailed fitting diagnostics.
#'   With `by.term=TRUE`, a `fastglm_test_list` containing the joint and
#'   term-specific tests in `$tests`, with print and data-frame methods.
#' @examples
#' fit <- fastglm_ordinal(Sat ~ Infl + Type + Cont, MASS::housing,
#'                        weights = Freq, firth = TRUE)
#' fastglm_po_test(fit)
#' as.data.frame(fastglm_po_test(fit, terms = "Type"))
#' @md
#' @export
fastglm_po_test <- function(object,terms=NULL,null.estimation=c("ML","meanBR"),
                            newdata=NULL,y=NULL,weights=NULL,offset=NULL,control=list(),by.term=FALSE,
                            method=c("score","lr")) {
    if(!inherits(object,"fastglm_ordinal")) stop("supply a fastglm_ordinal fit",call.=FALSE)
    null.estimation <- match.arg(null.estimation);method <- match.arg(method)
    if(method=="lr" && null.estimation!="ML")
        stop("likelihood ratio common-slope tests require null.estimation = 'ML'",call.=FALSE)
    if(method=="lr" && !inherits(object,"fastglm_ordinal_partial"))
        stop("method = 'lr' currently requires a partial ordinal model; use method = 'score' for a common-slope fit",call.=FALSE)
    if(!is.logical(by.term) || length(by.term)!=1L || is.na(by.term))
        stop("'by.term' must be TRUE or FALSE",call.=FALSE)
    training <- is.null(newdata) && is.null(y)
    if(training && inherits(object,"fastglm_ordinal_stream")) {
        if(!is.null(weights) || !is.null(offset))
            stop("original-source common-slope tests use recorded weights and offsets; supply explicit evaluation data to override them",call.=FALSE)
        result <- .ordinal_stream_po_test(object,terms,control,null.estimation,by.term,method)
        data.name <- paste(deparse(substitute(object)),collapse=" ")
        if(by.term) result$tests <- lapply(result$tests,function(z) {z$data.name <- data.name;z})
        else result$data.name <- data.name
        return(result)
    }
    if(!is.null(newdata) && is.null(y)) stop("supply assessment outcomes 'y' with 'newdata'",call.=FALSE)
    if(is.null(y)) y <- object$y
    if(training && is.null(weights)) weights <- object$weights
    design <- .ordinal_prediction_design(object,newdata,offset)
    response <- .ordinal_assessment_response(y,weights,object$levels,nrow(design$x))
    included <- !is.na(response$totals) & response$totals>0 & !response$missing.response & !is.na(response$weights)
    .design_blocks(design$x,function(z,index) {
        off <- design$offset[index]
        if(any(is.infinite(z)) || any(is.infinite(off)))
            stop("ordinal test predictors and offsets must be finite or missing",call.=FALSE)
        included[index] <<- included[index] & rowSums(is.na(z))==0L & !is.na(off)
    },object$block.size)
    result <- .ordinal_assessment_po_test(object,design,response,included,terms,control,null.estimation,by.term,method)
    data.name <- paste(deparse(substitute(object)),collapse=" ")
    if(by.term) result$tests <- lapply(result$tests,function(z) {z$data.name <- data.name; z})
    else result$data.name <- data.name
    result
}
