.ordinal_retained_probabilities <- function(object) {
    object$na.action <- NULL
    predict(object,type="probability")
}

# Draw frequencies without expanding a row into its individual observations.
# Reverse sums retain small upper tails that total-minus-cumulative can lose.
.ordinal_draw_counts <- function(probability,totals) {
    if(!is.matrix(probability) || nrow(probability)!=length(totals) ||
       any(!is.finite(totals)|totals<0|totals>.Machine$integer.max) ||
       any(abs(totals-round(totals))>1e-7))
        stop("ordinal simulation requires integer row frequencies between zero and .Machine$integer.max",call.=FALSE)
    n <- nrow(probability); K <- ncol(probability); active <- totals>0
    if(K<2L || any(!is.finite(probability[active,,drop=FALSE])) ||
       any(probability[active,,drop=FALSE]<0) ||
       any(abs(rowSums(probability[active,,drop=FALSE])-1)>1e-8))
        stop("ordinal simulation requires finite category probabilities summing to one",call.=FALSE)
    out <- matrix(0L,n,K,dimnames=dimnames(probability))
    remaining <- as.integer(round(totals))
    tail <- probability
    for(j in seq.int(K-1L,1L)) tail[,j] <- tail[,j]+tail[,j+1L]
    for(j in seq_len(K-1L)) {
        take <- which(remaining>0L)
        if(length(take)) {
            if(any(tail[take,j]<=0)) stop("ordinal simulation has no remaining probability mass",call.=FALSE)
            draw <- as.integer(rbinom(length(take),remaining[take],probability[take,j]/tail[take,j]))
            out[take,j] <- draw
            remaining[take] <- remaining[take]-draw
        }
    }
    out[,K] <- remaining
    out
}

.ordinal_simulate_counts <- function(object) {
    .ordinal_draw_counts(.ordinal_retained_probabilities(object),object$category$totals)
}

#' Simulate categorical responses from an ordinal fit
#'
#' Generates independent categorical observations conditional on the retained
#' predictors and offsets. Frequency rows are simulated as multinomial counts,
#' so a frequency of ten generates ten observations at that predictor row.
#' @param object a converged \code{fastglm_ordinal} model.
#' @param nsim number of simulated response datasets, a positive integer.
#' @param seed optional random seed. When supplied, the caller's random-number
#'   state is restored on exit.
#' @param ... further arguments; currently unused.
#' @details The original category labels and their order are preserved.
#'   A collapsed category has fitted probability zero and therefore simulated
#'   count zero. This is simulation from the fitted distribution; it does not
#'   quantify uncertainty about a category absent from that distribution.
#'   Row totals are effective frequencies, including counts times fitting
#'   weights, and must not exceed \code{.Machine$integer.max}.
#'   Rows omitted under \code{na.exclude} are restored with missing values.
#' @returns For an individual response with unit frequency in every retained
#'   row, a data frame of ordered factors, one column per simulation. Otherwise
#'   a named list of count matrices, one per simulation, with rows matching the
#'   fitted design and columns in the declared category order. Count matrices
#'   already include the fitting frequencies and should be refitted with unit
#'   weights. They are not counts to which the original weights are reapplied.
#' @exportS3Method simulate fastglm_ordinal
simulate.fastglm_ordinal <- function(object,nsim=1L,seed=NULL,...) {
    if(length(list(...))) stop("unused ordinal simulation arguments",call.=FALSE)
    if(!isTRUE(object$converged)) stop("ordinal simulation requires a converged fit",call.=FALSE)
    if(!is.numeric(nsim) || is.complex(nsim) || length(nsim)!=1L ||
       !is.finite(nsim) || nsim<1 || nsim>.Machine$integer.max || nsim!=floor(nsim))
        stop("'nsim' must be a positive integer",call.=FALSE)
    if(!is.null(seed)) {
        if(!is.numeric(seed) || is.complex(seed) || length(seed)!=1L || !is.finite(seed))
            stop("'seed' must be a finite numeric scalar",call.=FALSE)
        had <- exists(".Random.seed",.GlobalEnv,inherits=FALSE)
        if(had) old <- get(".Random.seed",.GlobalEnv)
        on.exit(if(had) assign(".Random.seed",old,.GlobalEnv) else
            if(exists(".Random.seed",.GlobalEnv,inherits=FALSE)) rm(".Random.seed",envir=.GlobalEnv))
        set.seed(seed)
    }
    probability <- .ordinal_retained_probabilities(object)
    totals <- object$category$totals
    individual <- !object$category$grouped && all(totals==1)
    out <- lapply(seq_len(nsim),function(j) {
        counts <- .ordinal_draw_counts(probability,totals)
        value <- if(individual)
            ordered(object$levels[max.col(counts,ties.method="first")],levels=object$levels) else counts
        if(!is.null(object$na.action)) value <- napredict(object$na.action,value)
        value
    })
    names(out) <- paste0("sim_",seq_len(nsim))
    if(individual) out <- as.data.frame(out)
    attr(out,"response") <- if(individual) "ordered categories" else "effective category counts"
    attr(out,"frequency") <- if(is.null(object$na.action)) totals else napredict(object$na.action,totals)
    out
}

#' Probability-based residuals for ordinal models
#'
#' Measures discrepancies between observed categories and the fitted category
#' distribution without assigning numerical distances to category labels.
#' @param object a fitted \code{fastglm_ordinal} model.
#' @param type \code{"probability"} gives probability-scale residuals;
#'   \code{"cumulative"} gives observed minus fitted cumulative proportions;
#'   \code{"pearson"} standardizes cumulative count differences using their
#'   marginal binomial variances.
#' @param ... further arguments; currently unused.
#' @details For an individual outcome \eqn{y}, the probability-scale residual is
#'   \eqn{F(y-)+F(y)-1}, where \eqn{F} is the fitted cumulative distribution.
#'   Positive values indicate outcomes above the fitted distribution. For a
#'   count response, this method returns the mean of the individual residuals
#'   within each row. Probability-scale residuals lie in \eqn{[-1,1]}; for
#'   ordinal data they generally have neither a normal nor a uniform distribution.
#'
#'   Cumulative residuals have one column per threshold. For row frequency
#'   \eqn{m}, the Pearson version divides cumulative observed count minus
#'   \eqn{m F(j)} by \eqn{\sqrt{m F(j)(1-F(j))}}. These columns are dependent
#'   and are descriptive diagnostics, not independent goodness-of-fit tests.
#'   Zero-frequency rows have missing residuals. Rows omitted under
#'   \code{na.exclude} are restored with missing values.
#' @returns A named vector for probability-scale residuals or a matrix for
#'   cumulative or Pearson residuals.
#' @references Li, C. and Shepherd, B. E. (2012). A new residual for ordinal
#'   outcomes. Biometrika, 99, 473--480. \doi{10.1093/biomet/asr073}.
#' @exportS3Method residuals fastglm_ordinal
residuals.fastglm_ordinal <- function(object,type=c("probability","cumulative","pearson"),...) {
    if(length(list(...))) stop("unused ordinal residual arguments",call.=FALSE)
    type <- match.arg(type)
    n <- nrow(object$x); K <- length(object$levels)
    out <- if(type=="probability") rep(NA_real_,n) else
        matrix(NA_real_,n,K-1L,dimnames=list(rownames(object$x),object$levels[-K]))
    .design_blocks(object$x,function(x,i) {
        m <- object$category$totals[i]; positive <- m>0
        if(!any(positive)) return(invisible(NULL))
        ii <- i[positive]; x <- x[positive,,drop=FALSE]; m <- m[positive]
        parts <- .ordinal_probability_block(object,x,object$offset[ii])
        F <- parts$cumulative[,object$threshold.map,drop=FALSE]
        S <- parts$survival[,object$threshold.map,drop=FALSE]
        if(object$category$grouped) {
            proportions <- matrix(0,length(ii),K)
            proportions[,object$retained.categories] <-
                (object$category$counts[ii,,drop=FALSE]*object$weights[ii])/m
            observed <- proportions[,seq_len(K-1L),drop=FALSE]
            if(K>2L) for(j in 2:(K-1L)) observed[,j] <- observed[,j]+observed[,j-1L]
            observed.upper <- proportions[,2:K,drop=FALSE]
            if(K>2L) for(j in seq.int(K-2L,1L))
                observed.upper[,j] <- observed.upper[,j]+observed.upper[,j+1L]
            value <- if(type=="probability") rowSums(proportions*(cbind(0,F)-cbind(S,0))) else {
                difference <- observed-F
                difference[F>.5] <- (S-observed.upper)[F>.5]
                difference
            }
        } else {
            code <- object$category$original.codes[ii]
            if(type=="probability") {
                lower <- cbind(0,F); upper <- cbind(S,0)
                value <- lower[cbind(seq_along(ii),code)]-upper[cbind(seq_along(ii),code)]
            } else {
                below <- outer(code,seq_len(K-1L),"<=")
                value <- -F; value[below] <- S[below]
            }
        }
        if(type=="pearson") {
            denominator <- sqrt(parts$derivative[,object$threshold.map,drop=FALSE])
            value <- sqrt(m)*value/denominator
            value[denominator==0 & is.nan(value)] <- NA_real_
        }
        if(type=="probability") out[ii] <<- value else out[ii,] <<- value
    },object$block.size)
    if(type=="probability") names(out) <- rownames(object$x)
    if(!is.null(object$na.action)) out <- napredict(object$na.action,out)
    out
}

#' Tidy summaries of ordinal fits
#'
#' Optional methods for the broom generics, with thresholds distinguished from
#' location slopes and the estimation method retained in model summaries.
#' @param x a fitted \code{fastglm_ordinal} model.
#' @param conf.int include normal confidence limits.
#' @param conf.level confidence level.
#' @param exponentiate exponentiate location slopes and their confidence
#'   limits to common odds ratios. Thresholds and standard errors remain on
#'   their original scales.
#' @param component include all parameters, location slopes, or thresholds.
#' @param vcov. optional full covariance matrix for a converged fit.
#' @param ... further arguments; currently unused.
#' @details Thresholds have no reported zero-null statistic or p-value.
#'   Nonconverged fits retain point estimates with unavailable uncertainty.
#'   For bias-reduced fits, the likelihood, AIC, and BIC in \code{glance()}
#'   use the ordinary likelihood evaluated at those estimates; this is not
#'   a maximized likelihood or a bias-reduced information criterion.
#' @returns A data frame with one row per parameter for \code{tidy()}, one
#'   row per model for \code{glance()}, or one row per observation for
#'   \code{augment()}.
#' @exportS3Method broom::tidy fastglm_ordinal
#' @rdname ordinal-tidy
tidy.fastglm_ordinal <- function(x,conf.int=FALSE,conf.level=.95,exponentiate=FALSE,
                                 component=c("all","location","threshold"),vcov.=NULL,...) {
    if(length(list(...))) stop("unused ordinal tidy arguments",call.=FALSE)
    for(flag in list(conf.int,exponentiate))
        if(!is.logical(flag) || length(flag)!=1L || is.na(flag)) stop("tidy flags must be TRUE or FALSE",call.=FALSE)
    if(length(conf.level)!=1L || !is.finite(conf.level) || conf.level<=0 || conf.level>=1)
        stop("'conf.level' must lie between zero and one",call.=FALSE)
    component <- match.arg(component)
    tab <- summary(x,vcov.=vcov.)
    b <- coef(x); q <- length(x$alpha); p <- length(x$beta)
    se <- c(tab$thresholds[,2L],tab$location[,2L])
    out <- data.frame(term=names(b),component=c(rep("threshold",q),rep("location",p)),
        estimate=as.numeric(b),std.error=unname(se),
        statistic=c(rep(NA_real_,q),tab$location[,3L]),
        p.value=c(rep(NA_real_,q),tab$location[,4L]),row.names=NULL)
    if(conf.int) {
        half <- qnorm((1+conf.level)/2)*out$std.error
        out$conf.low <- out$estimate-half; out$conf.high <- out$estimate+half
    }
    if(exponentiate) for(nm in intersect(c("estimate","conf.low","conf.high"),names(out)))
        out[out$component=="location",nm] <- exp(out[out$component=="location",nm])
    out[.ordinal_indices(x,component),,drop=FALSE]
}

#' @exportS3Method broom::glance fastglm_ordinal
#' @rdname ordinal-tidy
glance.fastglm_ordinal <- function(x,...) {
    if(length(list(...))) stop("unused ordinal glance arguments",call.=FALSE)
    data.frame(logLik=as.numeric(logLik(x)),AIC=AIC(x),BIC=BIC(x),deviance=deviance(x),
        df=x$rank,df.residual=x$df.residual,nobs=nobs(x),nobs.rows=x$nobs.rows,
        estimation=x$method,converged=x$converged,boundary=x$boundary,
        iterations=x$iterations)
}

#' @param data optional original data to accompany training predictions, in
#'   the same row order. With \code{na.exclude}, include the omitted rows.
#' @param newdata optional new prediction data; cannot be combined with
#'   \code{data}. Residuals are supplied only for training predictions.
#' @param se_fit include model-based category-probability standard errors.
#' @details \code{augment()} appends a matrix column \code{.fitted} of category
#'   probabilities, an ordered-factor column \code{.class}, and, for training
#'   rows, probability-scale residuals in \code{.resid}. The optional matrix
#'   column \code{.se.fit} contains model-based probability standard errors.
#'   When original data are not supplied, retained model-frame or design values
#'   are used; covariates in omitted rows are unavailable and restored as missing.
#' @exportS3Method broom::augment fastglm_ordinal
#' @rdname ordinal-tidy
augment.fastglm_ordinal <- function(x,data=NULL,newdata=NULL,se_fit=FALSE,...) {
    if(length(list(...))) stop("unused ordinal augment arguments",call.=FALSE)
    if(!is.null(data) && !is.null(newdata)) stop("supply either data or newdata",call.=FALSE)
    if(!is.logical(se_fit) || length(se_fit)!=1L || is.na(se_fit))
        stop("'se_fit' must be TRUE or FALSE",call.=FALSE)
    input <- if(!is.null(newdata)) newdata else data
    if(is.null(input)) {
        input <- if(!is.null(x$model)) x$model else x$x
        frame <- if(ncol(input)) as.data.frame(input) else data.frame(row.names=seq_len(nrow(input)))
        if(!is.null(x$na.action)) {
            rows <- napredict(x$na.action,seq_len(nrow(frame)))
            frame <- frame[rows,,drop=FALSE]
        }
    } else frame <- as.data.frame(input)
    reserved <- c(".fitted",".class",".resid",".se.fit")
    if(any(reserved %in% names(frame)))
        stop("augmentation data already contain reserved prediction columns",call.=FALSE)
    prediction <- predict(x,newdata=newdata,se.fit=se_fit)
    probability <- if(se_fit) prediction$fit else prediction
    if(nrow(frame)!=nrow(probability))
        stop("augmentation data must match prediction rows, including rows restored by na.exclude",call.=FALSE)
    frame$.fitted <- I(probability)
    code <- rep(NA_integer_,nrow(probability)); complete <- complete.cases(probability)
    code[complete] <- max.col(probability[complete,,drop=FALSE],ties.method="first")
    frame$.class <- ordered(x$levels[code],levels=x$levels)
    if(is.null(newdata)) frame$.resid <- residuals(x)
    if(se_fit) frame$.se.fit <- I(prediction$se.fit)
    frame
}
