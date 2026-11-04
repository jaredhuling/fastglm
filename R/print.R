# Shared console formatting. Formatting never changes the returned numeric data.
.fastglm_print_digits <- function(digits) {
    if(!is.numeric(digits) || length(digits)!=1L || !is.finite(digits) || digits<1 || digits>22 || digits!=floor(digits))
        stop("'digits' must be an integer from 1 to 22",call.=FALSE)
    as.integer(digits)
}

.fastglm_print_header <- function(title,call=NULL) {
    cat(title,"\n",sep="")
    if(!is.null(call)) cat("\nCall:\n",paste(deparse(call),collapse="\n"),"\n",sep="")
}

.fastglm_print_field <- function(label,value) {
    if(is.null(value) || !length(value)) return(invisible(NULL))
    cat(paste(strwrap(paste0(label,": ",paste(value,collapse=", ")),
        width=max(40L,getOption("width")),exdent=2L),collapse="\n"),"\n",sep="")
}

.fastglm_print_table <- function(x,digits=4L,p.columns=character(),row.names=FALSE) {
    digits <- .fastglm_print_digits(digits)
    x <- as.data.frame(x,check.names=FALSE)
    for(nm in names(x)) {
        if(nm %in% p.columns) x[[nm]] <- format.pval(x[[nm]],digits=digits,eps=1e-4,na.form="NA")
        else if(is.numeric(x[[nm]])) x[[nm]] <- format(x[[nm]],digits=digits,trim=TRUE,na.encode=TRUE)
    }
    print(x,row.names=row.names,right=TRUE,quote=FALSE)
    invisible(NULL)
}

.fastglm_print_coefficients <- function(x,title="Coefficients",digits=4L,...) {
    digits <- .fastglm_print_digits(digits)
    cat("\n",title,":\n",sep="")
    if(is.null(dim(x))) x <- matrix(x,ncol=1L,dimnames=list(names(x),"Estimate"))
    if(!nrow(x)) { cat("  No coefficients\n"); return(invisible(NULL)) }
    do.call(printCoefmat,utils::modifyList(list(x=x,digits=digits,eps.Pvalue=1e-4),list(...)))
    invisible(NULL)
}

.fastglm_print_convergence <- function(converged,iterations=NULL,reason=NULL) {
    if(is.null(converged)) return(invisible(NULL))
    status <- if(isTRUE(converged)) "converged" else "not converged"
    detail <- character()
    if(!is.null(iterations)) detail <- paste(iterations,if(iterations==1) "iteration" else "iterations")
    if(!isTRUE(converged) && length(reason) && nzchar(reason)) detail <- c(detail,gsub("_"," ",reason,fixed=TRUE))
    .fastglm_print_field("Convergence",paste0(status,if(length(detail)) paste0(" (",paste(detail,collapse="; "),")")))
}

#' Display profile confidence intervals
#'
#' Profile intervals print a compact table while retaining endpoint and fit
#' diagnostics as attributes. Matrix indexing and numeric extraction are
#' unchanged. Failed limits remain missing rather than being replaced by Wald
#' limits.
#' @param x a profile confidence-interval matrix returned by `confint()`.
#' @param digits number of significant digits used for display.
#' @param ... additional arguments, currently unused.
#' @return Invisibly returns `x` with all numerical values and attributes intact.
#' @name fastglm-confint-methods
#' @md
#' @export
print.fastglm_confint <- function(x,digits=max(3L,getOption("digits")-3L),...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header("Profile confidence intervals")
    .fastglm_print_field("Method",attr(x,"method"))
    level <- attr(x,"level")
    if(length(level)) .fastglm_print_field("Confidence level",paste0(format(100*level),"%"))
    cat("\n")
    values <- matrix(as.numeric(x),nrow=nrow(x),ncol=ncol(x),dimnames=dimnames(x))
    .fastglm_print_table(values,digits=digits,row.names=TRUE)
    if(anyNA(values)) {
        cat("\n")
        .fastglm_print_field("Unavailable limits",sum(is.na(values)))
        .fastglm_print_field("Details","Inspect attr(x, 'diagnostics') for endpoint convergence and failure messages.")
    }
    invisible(x)
}

#' Display and extract numerical diagnostics
#'
#' Diagnostics use the same headings and numeric formatting as fitted models.
#' Scores and remaining updates assess numerical convergence; they are not
#' hypothesis tests. For bias-reduced GLMs the ordinary likelihood score can
#' remain nonzero even when the adjusted equations have been solved.
#' @param x an object returned by [fastglm_diagnostics()].
#' @param digits number of significant digits used for display.
#' @param row.names optional row names for the extracted data frame.
#' @param optional logical, passed to the data-frame constructor.
#' @param ... additional arguments, currently unused.
#' @return The print method invisibly returns `x`. `as.data.frame()` returns
#'   one row with numeric and logical scalar columns at their original precision.
#'   Coefficient score vectors, aliased coefficients, collapsed categories, and
#'   rejected-evaluation details are retained as list columns.
#' @name fastglm-diagnostics-methods
#' @md
NULL

#' @rdname fastglm-diagnostics-methods
#' @export
as.data.frame.fastglm_diagnostics <- function(x,row.names=NULL,optional=FALSE,...) {
    vectors <- c("score","ordinary.score","adjusted.score","aliased",
                 "collapsed.categories","last.rejected.evaluation")
    out <- lapply(names(x),function(nm) {
        z <- x[[nm]]
        if(nm %in% vectors || length(z)!=1L || !is.atomic(z) || !is.null(dim(z))) I(list(z)) else unname(z)
    })
    names(out) <- names(x)
    as.data.frame(out,row.names=row.names,optional=optional)
}

#' @rdname fastglm-diagnostics-methods
#' @export
print.fastglm_diagnostics <- function(x,digits=max(3L,getOption("digits")-3L),...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header("Numerical diagnostics")
    .fastglm_print_convergence(x$converged,x$iterations,x$convergence.reason)
    if(length(x$solver)) .fastglm_print_field("Solver",switch(x$solver,
        qr="column-pivoted QR",qr_unpivoted="unpivoted QR",qr_full="full-pivoted QR",
        llt="Cholesky (LLT)",cached_llt="Cholesky (reused decomposition)",
        ldlt="Cholesky (LDLT)",svd="singular value decomposition",
        empty="no coefficients to estimate",x$solver))
    .fastglm_print_field("Solver fallback",x[["fallback"]])
    if(length(x$fallbacks) && x$fallbacks>0) .fastglm_print_field("Solver fallbacks",x$fallbacks)
    if(length(x$step.reductions) && x$step.reductions>0) .fastglm_print_field("Reduced scoring steps",x$step.reductions)
    .fastglm_print_field("Estimable coefficients (rank)",x$rank)
    .fastglm_print_field("Aliased coefficients",x$aliased)
    .fastglm_print_field("Boundary flag",if(isTRUE(x$boundary)) "yes" else "no")
    if(isTRUE(x$theta.boundary)) .fastglm_print_field("Negative-binomial size boundary","yes")
    .fastglm_print_field("Collapsed categories",x$collapsed.categories)
    labels <- c(standardized.score=if(identical(x$score.type,"mean-bias-reduced"))
            "Standardized adjusted score" else "Standardized ordinary score",
        standardized.adjusted.score="Standardized adjusted score",
        standardized.dispersion.score="Standardized dispersion score",
        remaining.predictor.step="Remaining change in linear predictor",
        adjusted.score.rms="Joint adjusted score per row",
        remaining.dispersion.step="Remaining relative dispersion change",
        adjusted.dispersion.score.rms="Joint dispersion score per row",
        covariance.condition="Covariance condition number",
        information.rcond="Reciprocal information condition number",
        minimum.gap="Smallest fitted cumulative-predictor gap")
    fields <- intersect(names(labels),names(x))
    values <- vapply(fields,function(nm) {
        z <- x[[nm]]
        if(length(z)!=1L || is.na(z)) "Unavailable" else format(z,digits=digits,trim=TRUE)
    },character(1))
    cat("\n")
    .fastglm_print_table(data.frame(Check=unname(labels[fields]),Value=unname(values)),digits)
    if("adjusted.score.available" %in% names(x)) {
        cat("\n")
        .fastglm_print_field("Note","Use the adjusted scores to assess bias-reduced convergence; the ordinary score need not be zero.")
    }
    invisible(x)
}

.fastglm_prediction <- function(x,type) structure(x,class="fastglm_prediction",prediction.type=type)

#' Display and extract predictions with standard errors
#'
#' Predictions requested with `se.fit=TRUE` retain their `fit` and `se.fit`
#' components and print aligned estimates and standard errors. For matrix
#' predictions, each row and component (such as an outcome category) occupies
#' one row of the displayed or extracted table. Missing standard errors remain
#' missing, including for collapsed ordinal categories.
#' @param x a prediction returned with `se.fit=TRUE` by a fastglm model.
#' @param digits number of significant digits used for display.
#' @param max.rows maximum number of prediction/component rows to print;
#'   use `Inf` to display all rows. Extraction always returns all rows.
#' @param row.names optional row names for the extracted data frame.
#' @param optional logical, passed to the data-frame constructor.
#' @param ... additional arguments, currently unused.
#' @return The print method invisibly returns `x`. `as.data.frame()` returns
#'   row identifiers, component labels for matrix predictions, and numeric
#'   `fit` and `se.fit` columns at full precision. Original matrix dimensions,
#'   names, and any model-specific metadata remain in `x`.
#' @name fastglm-prediction-methods
#' @md
NULL

#' @rdname fastglm-prediction-methods
#' @export
as.data.frame.fastglm_prediction <- function(x,row.names=NULL,optional=FALSE,...) {
    fit <- x$fit
    if(is.null(dim(fit))) {
        rows <- names(fit)
        if(is.null(rows)) rows <- as.character(seq_along(fit))
        out <- list(row=rows,fit=as.numeric(fit),se.fit=as.numeric(x$se.fit))
    } else {
        rows <- rownames(fit); components <- colnames(fit)
        if(is.null(rows)) rows <- as.character(seq_len(nrow(fit)))
        if(is.null(components)) components <- as.character(seq_len(ncol(fit)))
        out <- list(row=rep(rows,each=ncol(fit)),component=rep(components,times=nrow(fit)),
            fit=as.numeric(t(fit)),se.fit=as.numeric(t(x$se.fit)))
    }
    as.data.frame(out,row.names=row.names,optional=optional,stringsAsFactors=FALSE)
}

#' @rdname fastglm-prediction-methods
#' @export
print.fastglm_prediction <- function(x,digits=max(3L,getOption("digits")-3L),max.rows=20L,...) {
    digits <- .fastglm_print_digits(digits)
    if(!is.numeric(max.rows) || length(max.rows)!=1L || is.na(max.rows) ||
       max.rows<1 || max.rows!=floor(max.rows))
        stop("'max.rows' must be a positive integer or Inf",call.=FALSE)
    .fastglm_print_header("Predictions with standard errors")
    .fastglm_print_field("Scale",switch(attr(x,"prediction.type"),
        response="response mean",link="linear predictor",cumulative.link="cumulative linear predictor",
        terms="term contributions to the linear predictor",
        probability="category probabilities",cumulative="cumulative probabilities",
        prob="probability masses",zero="probability of zero",count="component mean"))
    .fastglm_print_field("Covariance",x$covariance.scope)
    if(length(x$residual.scale)) .fastglm_print_field("Residual scale",format(x$residual.scale,digits=digits))
    # Select rows before conversion so console output stays small for large data.
    fit <- x$fit; n <- if(is.null(dim(fit))) length(fit) else nrow(fit)*ncol(fit)
    shown <- min(n,max.rows)
    preview <- x
    if(is.null(dim(fit))) {
        preview$fit <- fit[seq_len(shown)]; preview$se.fit <- x$se.fit[seq_len(shown)]
    } else {
        rows <- seq_len(if(ncol(fit)) ceiling(shown/ncol(fit)) else 0L)
        preview$fit <- fit[rows,,drop=FALSE]; preview$se.fit <- x$se.fit[rows,,drop=FALSE]
    }
    tab <- as.data.frame(preview)[seq_len(shown),,drop=FALSE]
    names(tab) <- if(is.null(dim(fit))) c("Row","Estimate","Std. Error") else
        c("Row","Component","Estimate","Std. Error")
    cat("\n"); .fastglm_print_table(tab,digits)
    if(shown<n) .fastglm_print_field("Display",paste(shown,"of",n,"rows; use as.data.frame() to extract all predictions"))
    invisible(x)
}

#' Display and extract hypothesis tests
#'
#' Test objects print a compact table while retaining full-precision numeric
#' results. Data-frame and optional broom methods provide a single row for
#' reporting, including unavailable tests and their explanations.
#' @param x a test returned by [fastglm_po_test()], the proportional-odds test
#'   component of [fastglm_ordinal_assess()], [fastglm_wald_test()], or
#'   [fastglm_profile_test()]. Collections from `fastglm_po_test(by.term=TRUE)`
#'   print a table and extract one row per hypothesis.
#' @param digits number of significant digits used for display.
#' @param row.names optional row names for the extracted data frame.
#' @param optional logical, passed to the data-frame constructor.
#' @param details include method descriptions and failure explanations in the
#'   extracted table. The default returns the main numeric results and status.
#' @param ... additional arguments, currently unused.
#' @return Print methods invisibly return the original object. Extraction
#'   methods return a data frame with numeric statistics and p-values.
#' @name fastglm-test-methods
#' @md
NULL

#' @rdname fastglm-test-methods
#' @export
as.data.frame.fastglm_test <- function(x,row.names=NULL,optional=FALSE,details=FALSE,...) {
    available <- if(is.null(x$available)) is.finite(x$p.value) else isTRUE(x$available)
    out <- data.frame(statistic=unname(x$statistic),df=if(is.null(x$df)) unname(x$parameter) else x$df,
        p.value=x$p.value,available=available,method=x$method,
        null.estimation=if(is.null(x$null.estimation)) NA_character_ else x$null.estimation,
        terms=paste(if(is.null(x$terms)) names(x$null.value) else x$terms,collapse=", "),
        reason=if(is.null(x$reason)) if(available) "available" else "fit_not_converged" else x$reason,
        detail=if(is.null(x$detail)) if(is.null(x$diagnostics$message)) "" else x$diagnostics$message else x$detail,
        row.names=row.names,check.names=!optional)
    if(!isTRUE(details)) out <- out[c("statistic","df","p.value","null.estimation","available")]
    out
}

#' @rdname fastglm-test-methods
#' @export
print.fastglm_test <- function(x,digits=max(3L,getOption("digits")-3L),...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header(x$method)
    if(length(x$terms)) .fastglm_print_field("Terms",x$terms)
    if(!is.null(x$null.estimation)) .fastglm_print_field("Null fit",switch(x$null.estimation,
        ML="maximum likelihood",meanBR="mean bias reduction",x$null.estimation))
    .fastglm_print_field("Estimation",x$estimation)
    .fastglm_print_field("Covariance",x$covariance.type)
    .fastglm_print_field("Component",x$component)
    if(!is.null(x$frequency)) .fastglm_print_field("Sample",paste0(format(x$frequency,trim=TRUE),
        " observations represented by ",x$observations," rows"))
    tab <- as.data.frame(x,details=TRUE)
    if(tab$available) {
        display <- tab[c("statistic","df","p.value")]
        names(display) <- c("Statistic","df","p-value")
        cat("\n"); .fastglm_print_table(display,digits,p.columns="p-value")
    } else {
        .fastglm_print_field("Status","Unavailable")
        .fastglm_print_field("Reason",tab$detail)
    }
    if(length(x$estimate)) {
        estimates <- cbind(Estimate=x$estimate)
        if(length(x$std.error)) estimates <- cbind(estimates,`Std. Error`=x$std.error)
        estimates <- cbind(estimates,Null=x$null.value)
        .fastglm_print_coefficients(estimates,if(is.null(x$contrast)) "Tested coefficients" else
            "Tested contrasts",digits,P.values=FALSE,has.Pvalue=FALSE)
    }
    if(length(x$note)) {cat("\n"); .fastglm_print_field("Note",x$note)}
    invisible(x)
}

#' @rdname fastglm-test-methods
#' @exportS3Method broom::tidy fastglm_test
tidy.fastglm_test <- function(x,...) as.data.frame(x,...)

#' @rdname fastglm-test-methods
#' @export
as.data.frame.fastglm_test_list <- function(x,row.names=NULL,optional=FALSE,details=FALSE,...) {
    tables <- lapply(seq_along(x$tests),function(j) data.frame(hypothesis=names(x$tests)[j],
        as.data.frame(x$tests[[j]],details=details),check.names=!optional))
    out <- do.call(rbind,tables)
    rownames(out) <- row.names
    out
}

#' @rdname fastglm-test-methods
#' @export
print.fastglm_test_list <- function(x,digits=max(3L,getOption("digits")-3L),...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header(x$method)
    .fastglm_print_field("Null fit",switch(x$null.estimation,ML="maximum likelihood",meanBR="mean bias reduction"))
    tab <- as.data.frame(x,details=TRUE)
    display <- tab[c("hypothesis","statistic","df","p.value","available")]
    names(display) <- c("Hypothesis","Statistic","df","p-value","Available")
    cat("\n"); .fastglm_print_table(display,digits,p.columns="p-value")
    for(j in which(!tab$available)) .fastglm_print_field(tab$hypothesis[j],tab$detail[j])
    cat("\n"); .fastglm_print_field("Note",x$note)
    invisible(x)
}

#' @rdname fastglm-test-methods
#' @exportS3Method broom::tidy fastglm_test_list
tidy.fastglm_test_list <- function(x,...) as.data.frame(x,...)

#' @exportS3Method print summary.fastglm
print.summary.fastglm <- function(x,digits=max(3L,getOption("digits")-3L),
                                 symbolic.cor=x$symbolic.cor,...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header(if(is.null(x$title)) "Generalized linear model" else x$title,x$call)
    .fastglm_print_field("Family",paste0(x$family$family," (",x$family$link," link)"))
    if(isTRUE(x$firth)) .fastglm_print_field("Estimation","mean bias reduction")
    .fastglm_print_coefficients(x$coefficients,digits=digits,...)
    if(any(x$aliased)) .fastglm_print_field("Aliased coefficients",names(x$aliased)[x$aliased])
    cat("\n")
    .fastglm_print_table(data.frame(Dispersion=x$dispersion,`Null deviance`=x$null.deviance,
        `Null df`=x$df.null,Deviance=x$deviance,`Residual df`=x$df.residual,AIC=x$aic,check.names=FALSE),digits)
    if(!is.null(x$theta)) .fastglm_print_table(data.frame(Theta=x$theta,`Std. Error`=x$SE.theta,check.names=FALSE),digits)
    .fastglm_print_convergence(x$converged,x$iter,x$convergence.reason)
    if(!is.null(x$correlation)) {
        cat("\nCoefficient correlations:\n")
        if(isTRUE(symbolic.cor)) print(symnum(x$correlation))
        else .fastglm_print_table(x$correlation,digits,row.names=TRUE)
    }
    invisible(x)
}
