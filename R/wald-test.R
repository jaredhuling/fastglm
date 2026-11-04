.fastglm_wald_component_terms <- function(object,terms,component) {
    selected <- if(component=="all") c("count","zero") else component
    columns <- names(coef(object)); requested <- terms
    formula <- !is.null(object$terms)
    labels <- lapply(selected,function(k) if(formula) attr(object$terms[[k]],"term.labels") else
        setdiff(names(object$coefficients[[k]]),"(Intercept)"))
    if(is.null(terms)) terms <- unique(unlist(labels,use.names=FALSE))
    intercept <- any(vapply(selected,function(k) "(Intercept)" %in% names(object$coefficients[[k]]),logical(1)))
    possible <- c(if(intercept) "(Intercept)",unlist(labels,use.names=FALSE))
    if(!is.character(terms) || !length(terms) || anyNA(terms) || anyDuplicated(terms) ||
       any(!terms %in% possible)) stop("unknown or duplicated term in the selected component(s)",call.=FALSE)
    index <- integer()
    for(j in seq_along(selected)) {
        k <- selected[j]
        local <- terms[terms %in% c(intersect("(Intercept)",names(object$coefficients[[k]])),labels[[j]])]
        if(!length(local)) next
        names <- names(object$coefficients[[k]])
        if(formula) {
            assignment <- object$assign[[k]]
            if(is.null(assignment)) assignment <- attr(object$x[[k]],"assign")
            if(is.null(assignment) && !is.null(object$model)) assignment <- attr(model.matrix(
                object$terms[[k]],object$model[1L,,drop=FALSE],contrasts.arg=object$contrasts[[k]]),"assign")
            if(is.null(assignment) || length(assignment)!=length(names))
                stop("formula term tests require retained design assignments; supply a named 'contrast'",call.=FALSE)
            take <- which(assignment %in% (match(local,c("(Intercept)",labels[[j]]))-1L))
        } else take <- match(local,names)
        index <- c(index,match(paste0(k,"_",names[take]),columns))
    }
    list(index=index,terms=if(is.null(requested)) terms else requested)
}

.fastglm_wald_alias_relation <- function(object) {
    if(!is.null(object$alias.relation) || !.count_two_component(object)) return(object$alias.relation)
    b <- coef(object); good <- which(!is.na(b)); bad <- which(is.na(b))
    relation <- matrix(0,length(good),length(bad)); start <- 0L
    for(component in c("count","zero")) {
        coefficients <- object$coefficients[[component]]
        if(anyNA(coefficients)) {
            fit <- if(component=="count") object$positive else object$binary
            part <- fit$alias.relation
            if(is.null(part)) return(NULL)
            relation[match(start+which(!is.na(coefficients)),good),match(start+which(is.na(coefficients)),bad)] <- part
        }
        start <- start+length(coefficients)
    }
    relation
}

.fastglm_wald_contrast <- function(object,terms,contrast,component="all") {
    b <- coef(object); columns <- names(b); p <- length(b)
    if(!is.numeric(b) || is.null(columns) || anyNA(columns) || any(!nzchar(columns)) || anyDuplicated(columns))
        stop("Wald tests require distinct, named numeric coefficients",call.=FALSE)
    if(!is.null(terms) && !is.null(contrast)) stop("supply either 'terms' or 'contrast', not both",call.=FALSE)
    if(is.null(contrast)) {
        if(.count_two_component(object)) {
            selected <- .fastglm_wald_component_terms(object,terms,component)
            index <- selected$index;terms <- selected$terms
        } else if(inherits(object,"fastglm_ordinal_partial")) {
            selected <- .ordinal_partial_test_columns(object,terms)
            index <- selected$index; terms <- selected$terms
        } else if(inherits(object,"fastglm_ordinal")) {
            selected <- .ordinal_po_test_columns(object,terms)
            index <- match(selected$columns,columns); terms <- selected$terms
        } else if(!is.null(object$terms)) {
            labels <- attr(object$terms,"term.labels")
            if(is.null(terms)) terms <- labels
            if(!is.character(terms) || !length(terms) || anyNA(terms) || anyDuplicated(terms) ||
               any(!terms %in% c("(Intercept)",labels))) stop("unknown or duplicated formula term in 'terms'",call.=FALSE)
            assignment <- attr(object$x,"assign")
            if(is.null(assignment)) assignment <- object$assign
            if(is.null(assignment) || length(assignment)!=p)
                stop("formula term tests require retained design assignments; supply a named 'contrast'",call.=FALSE)
            target <- match(terms,c("(Intercept)",labels))-1L
            index <- which(assignment %in% target)
        } else {
            if(is.null(terms)) terms <- setdiff(columns,"(Intercept)")
            if(!is.character(terms) || !length(terms) || anyNA(terms) || anyDuplicated(terms) ||
               any(!terms %in% columns)) stop("unknown or duplicated coefficient name in 'terms'",call.=FALSE)
            index <- match(terms,columns)
        }
        if(!length(index)) stop("no coefficients were selected",call.=FALSE)
        L <- diag(p)[index,,drop=FALSE]; dimnames(L) <- list(columns[index],columns)
    } else {
        L <- contrast
        if(is.numeric(L) && is.null(dim(L))) L <- matrix(L,nrow=1L,dimnames=list(NULL,names(L)))
        if(!is.matrix(L) || !is.numeric(L) || is.complex(L) || !nrow(L) || !ncol(L) || any(!is.finite(L)))
            stop("'contrast' must be a finite numeric vector or matrix",call.=FALSE)
        if(is.null(colnames(L))) {
            if(ncol(L)!=p) stop("an unnamed contrast must have one column per coefficient",call.=FALSE)
            colnames(L) <- columns
        } else {
            nm <- colnames(L)
            if(anyNA(nm) || any(!nzchar(nm)) || anyDuplicated(nm) || any(!nm %in% columns))
                stop("contrast columns must uniquely name fitted coefficients",call.=FALSE)
            full <- matrix(0,nrow(L),p,dimnames=list(rownames(L),columns))
            full[,match(nm,columns)] <- L; L <- full
        }
        if(any(rowSums(abs(L))==0)) stop("each contrast must select at least one coefficient",call.=FALSE)
        if(is.null(rownames(L))) rownames(L) <- paste("Contrast",seq_len(nrow(L)))
        if(anyNA(rownames(L)) || any(!nzchar(rownames(L))) || anyDuplicated(rownames(L)))
            stop("contrast row names must be distinct and nonmissing",call.=FALSE)
        if(.count_two_component(object) && component!="all") {
            allowed <- paste0(component,"_",names(object$coefficients[[component]]))
            if(any(L[,!columns %in% allowed,drop=FALSE]!=0))
                stop("the contrast includes coefficients outside the selected component",call.=FALSE)
        }
    }
    list(coefficients=b,contrast=L,terms=terms)
}

#' Joint Wald tests for model terms and coefficient contrasts
#'
#' Tests specified linear combinations of regression coefficients in a fitted
#' model. A factor or interaction can be tested as a complete formula
#' term, and a named contrast can compare coefficients without refitting.
#' @param object a fastglm GLM (including negative-binomial and streaming GLMs),
#'   ordinal, truncated-count, hurdle, zero-inflated, or two-part fit.
#' @param terms formula terms, or coefficient names for a matrix fit, to test
#'   jointly. The default selects all formula terms or all matrix coefficients
#'   except a coefficient named `(Intercept)`. For ordinal models it selects
#'   location slopes. A nonparallel ordinal term includes its slopes at all
#'   thresholds. Formula terms refer to their fitted design columns;
#'   a main-effect test with interactions depends on their coding and reference
#'   values and is not a test of every effect involving that variable.
#'   For two-component models, names select complete terms in the selected
#'   component(s), or unprefixed coefficient names for matrix fits. With
#'   `component="all"`, each term is tested wherever it occurs. A term must
#'   occur in at least one selected component.
#' @param contrast numeric vector or matrix specifying linear combinations.
#'   Named columns may select a subset of coefficients in any order; omitted
#'   columns have weight zero. Unnamed contrasts must follow `coef(object)`.
#'   Each matrix row specifies one hypothesis. Use either `terms` or `contrast`.
#'   For two-component models, named contrasts use the `count_` and `zero_`
#'   prefixes returned by `coef(object)`. A contrast spanning components requires
#'   `component="all"`; its interpretation depends on their coefficient scales.
#' @param null hypothesized value for each selected coefficient or contrast;
#'   a scalar is repeated across hypotheses. The default is zero.
#' @param vcov. optional coefficient covariance matrix, or a function of
#'   `object` returning one. The default uses `vcov(object)`. A named matrix
#'   must contain all coefficient names; its order may differ. Robust or
#'   cluster covariance can be supplied from [fastglm_vcov()].
#' @param component `"all"` (default), `"count"`, or `"zero"` for two-component
#'   models. Two-part fits also accept `"positive"` for the positive-amount
#'   component and `"binary"` for the positive-outcome probability. The zero
#'   component describes the probability of a positive outcome for hurdle and
#'   two-part fits, and structural-zero probability for zero-inflated fits.
#'   Single-component and ordinal fits use `"all"`.
#' @details For contrast matrix `L`, coefficient estimate `b`, and covariance
#'   `V`, the statistic is `(L b - null)' solve(L V L', L b - null)`. Its
#'   reference is chi-square with one degree of freedom per independent
#'   contrast. This is a first-order approximation requiring asymptotically
#'   normal, estimable coefficients and a consistent covariance for the sampling
#'   design. It is not a finite-sample F test or a likelihood-ratio test.
#'
#'   Mean-bias-reduced estimates use the same first-order convention. Sparse
#'   outcomes, extreme effects, and few independent clusters can make the
#'   approximation inaccurate. Tests of regression coefficients do not assess
#'   the common-slope restriction; use [fastglm_po_test()] for that hypothesis.
#'
#'   Nonconvergence, collapsed ordinal categories, nonestimable contrasts, and
#'   missing or numerically singular contrast covariance return an unavailable
#'   result with an explanation. Degrees of freedom are not reduced silently.
#'   For aliased GLM coefficients, estimability is checked against the retained
#'   dependence between design columns. No generalized inverse is used.
#'   Tests concern regression coefficients, excluding dispersion or size
#'   parameters. An estimated negative-binomial size at its numerical boundary
#'   returns an unavailable result. Truncated, hurdle, and zero-inflated fits
#'   use regression covariance accounting for estimated size in regular fits.
#'   For [fastglm_nb()], the default retains its conventional fixed-size GLM
#'   covariance; to use joint observed information, supply the regression block
#'   of `vcov(object, full=TRUE)`. Supplied covariance must contain all regression coefficients,
#'   including both components when applicable, even for a selected-component test.
#'   Cross-component covariance is retained; it is not replaced with independent
#'   component calculations. The covariance restrictions of [fastglm_vcov()]
#'   continue to apply, including its restriction on adjusted two-part fits.
#' @return A `fastglm_wald_test` inheriting from `fastglm_test` and `htest`,
#'   with statistic, degrees of freedom, p-value, availability, contrast estimates,
#'   standard errors, null values, and the contrast matrix and its covariance.
#'   Print the object directly or use `as.data.frame()` for numeric extraction.
#' @examples
#' fit <- fastglm_ordinal(Sat ~ Infl + Type + Cont, MASS::housing,
#'                       weights=Freq, firth=TRUE)
#' fastglm_wald_test(fit, terms="Type")
#' fastglm_wald_test(fit, contrast=c(InflHigh=1, InflMedium=-1))
#' @md
#' @export
fastglm_wald_test <- function(object,terms=NULL,contrast=NULL,null=0,vcov.=NULL,component="all") {
    if(!inherits(object,c("fastglm","fastglmFit","fastglm_ordinal","fastglm_nb",
        "fastglm_hurdle","fastglm_zi","fastglm_twopart","fastglm_truncated")))
        stop("supply a supported fastglm fitted model",call.=FALSE)
    component <- match.arg(component,c("all","count","zero","positive","binary"))
    if(component %in% c("positive","binary")) {
        if(!inherits(object,"fastglm_twopart")) stop("'positive' and 'binary' components apply to two-part models",call.=FALSE)
        component <- if(component=="positive") "count" else "zero"
    }
    if(!.count_two_component(object) && component!="all")
        stop("component selection requires a two-component model",call.=FALSE)
    setup <- .fastglm_wald_contrast(object,terms,contrast,component)
    b <- setup$coefficients; L <- setup$contrast; df <- nrow(L)
    if(!is.numeric(null) || is.complex(null) || !is.null(dim(null)) ||
       !length(null)%in%c(1L,df) || any(!is.finite(null)))
        stop("'null' must be a finite number or one value per contrast",call.=FALSE)
    null <- setNames(rep(null,length.out=df),rownames(L))
    out <- structure(list(method="Joint Wald test",statistic=NA_real_,df=df,p.value=NA_real_,
        available=FALSE,parameter=c(df=df),null.value=null,estimate=setNames(rep(NA_real_,df),rownames(L)),
        std.error=setNames(rep(NA_real_,df),rownames(L)),contrast=L,covariance=NULL,
        terms=setup$terms,reason="",detail="",data.name=paste(deparse(substitute(object)),collapse=" "),
        estimation=if(isTRUE(object$firth)) "mean bias reduction" else "fitted model coefficients",
        covariance.type=if(is.null(vcov.)) "model-based" else "supplied coefficient covariance",
        note="First-order chi-square approximation; validity depends on coefficient estimability and the covariance assumptions."),
        class=c("fastglm_wald_test","fastglm_test","htest"))
    unavailable <- function(reason,detail) {out$reason <- reason; out$detail <- detail; out}
    if(.count_two_component(object)) {
        label <- c(count=if(inherits(object,"fastglm_twopart")) "Positive amount" else "Count distribution",
            zero=if(inherits(object,"fastglm_zi")) "Structural-zero probability" else "Positive-outcome probability")
        out$component <- if(component=="all") unname(label) else unname(label[component])
    }
    if(!isTRUE(object$converged)) return(unavailable("fit_not_converged","The fitted model did not converge."))
    if(.count_inference_model(object) && .count_theta_boundary(object))
        return(unavailable("theta_boundary","Regular joint Wald inference is unavailable at the estimated negative-binomial size boundary."))
    if(inherits(object,"fastglm_ordinal") && isTRUE(object$boundary))
        return(unavailable("collapsed_categories","Joint Wald tests require an ordinal fit without collapsed categories."))
    good <- !is.na(b)
    if(any(!good)) {
        relation <- .fastglm_wald_alias_relation(object)
        if(is.null(relation)) return(unavailable("estimability_unavailable","The aliased design relation was not retained."))
        implied <- L[,good,drop=FALSE]%*%relation
        requested <- L[,!good,drop=FALSE]
        if(any(abs(implied-requested)>1e-8*pmax(1,abs(implied),abs(requested))))
            return(unavailable("nonestimable_contrast","A requested contrast is not estimable from the rank-deficient design."))
    }
    active <- which(good & colSums(abs(L))>0)
    if(!length(active) || any(!is.finite(b[active])))
        return(unavailable("nonfinite_estimate","The requested coefficients do not have finite estimates."))
    error <- ""
    V <- tryCatch(if(is.null(vcov.)) vcov(object) else if(is.function(vcov.)) vcov.(object) else vcov.,
        error=function(e) {error <<- conditionMessage(e); NULL})
    if(is.null(V)) return(unavailable("covariance_unavailable",error))
    if(!is.matrix(V) || !is.numeric(V) || is.complex(V) || any(dim(V)!=length(b)))
        stop("'vcov.' must be a numeric square matrix with one row per coefficient",call.=FALSE)
    if(!is.null(rownames(V)) || !is.null(colnames(V))) {
        if(is.null(rownames(V)) || is.null(colnames(V)) || anyDuplicated(rownames(V)) || anyDuplicated(colnames(V)) ||
           !setequal(rownames(V),names(b)) || !setequal(colnames(V),names(b)))
            stop("covariance row and column names must match the fitted coefficients",call.=FALSE)
        V <- V[names(b),names(b),drop=FALSE]
    }
    V <- V[active,active,drop=FALSE]; A <- L[,active,drop=FALSE]
    if(any(!is.finite(V))) return(unavailable("covariance_unavailable","The requested coefficient covariance contains missing or nonfinite entries."))
    if(any(abs(V-t(V))>1e-8*pmax(1,abs(V),abs(t(V))))) stop("the supplied covariance must be symmetric",call.=FALSE)
    out$estimate <- setNames(drop(A%*%b[active]),rownames(L))
    C <- A%*%V%*%t(A); C <- (C+t(C))/2
    out$covariance <- C
    scale <- sqrt(replace(diag(C),diag(C)<0,NA_real_))
    out$std.error <- setNames(scale,rownames(L))
    if(any(!is.finite(C)) || any(!is.finite(scale)|scale<=0))
        return(unavailable("singular_contrast_covariance","The contrast covariance is not finite and positive definite."))
    correlation <- C/outer(scale,scale)
    root <- tryCatch(chol(correlation),error=function(e) NULL)
    if(is.null(root) || rcond(correlation)<1e-14)
        return(unavailable("singular_contrast_covariance","The contrast covariance is singular or numerically unstable; supply independent estimable contrasts."))
    difference <- (out$estimate-null)/scale
    statistic <- sum(forwardsolve(t(root),difference)^2)
    if(!is.finite(statistic)) return(unavailable("nonfinite_statistic","The Wald statistic is nonfinite."))
    out$statistic <- statistic; out$p.value <- pchisq(statistic,df,lower.tail=FALSE)
    out$available <- TRUE; out$reason <- "available"; out
}
