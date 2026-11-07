# Pure-R reference for the native constrained iterations. It is the earlier
# package implementation, retained to check the compiled loop.
reference_partial_step <- function(setup,delta) {
    slopes <- matrix(delta[setup$q+setup$pc+seq_len(setup$q*setup$pn)],setup$pn,setup$q)
    design <- cbind(1,setup$x[setup$positive,,drop=FALSE],setup$domain.z)
    max(vapply(seq_len(setup$q),function(j)
        max(abs(streaming_qr_eta(design,c(delta[j],-delta[setup$q+seq_len(setup$pc)],-slopes[,j]),
            numeric(nrow(design))))),numeric(1)))
}

reference_predictor_step <- function(setup,delta) {
    if(isTRUE(setup$partial)) return(reference_partial_step(setup,delta))
    shift <- if(setup$p) drop(setup$x%*%delta[setup$q+seq_len(setup$p)])[setup$positive] else 0
    alpha <- range(delta[seq_len(setup$q)])
    max(abs(alpha[1L]-shift),abs(alpha[2L]-shift))
}

reference_constrained_fit <- function(setup,L=matrix(numeric(),0L,setup$d),null=numeric(),
                                     start=setup$coefficients,adjusted=FALSE) {
    ctl <- setup$control;constraint <- .ordinal_constraint_basis(setup,L);B <- constraint$basis
    initial <- .ordinal_constraint_start(setup,L,null,start,constraint)
    calls <- 0L
    evaluate <- function(gamma) {calls <<- calls+1L;setup$evaluate(gamma,adjusted)}
    finish <- function(gamma,state,code,message,iterations,score=Inf,step=Inf) {
        coefficients <- if(length(gamma)) if(isTRUE(setup$partial))
            streaming_qr_eta(setup$T,gamma,numeric(setup$d)) else drop(setup$T%*%gamma) else rep(NA_real_,setup$d)
        if(code==0L && isTRUE(setup$partial) && nrow(L)) {
            residual <- streaming_qr_eta(L,coefficients,numeric(nrow(L)))-null
            if(any(!is.finite(residual)) || any(abs(residual)>512*.Machine$double.eps*(1+abs(null)))) {
                code <- 6L;message <- "linear constraints cannot be represented accurately in the reported coefficients"
            }
        }
        names(coefficients) <- names(setup$coefficients)
        list(coefficients=coefficients,working.coefficients=gamma,
            value=if(isTRUE(state$valid)) state$loglik else NA_real_,converged=code==0L,
            score=score,predictor.step=step,code=code,message=message,iterations=iterations,
            evaluations=calls,state=state,basis=B,adjusted=adjusted)
    }
    if(!initial$valid) return(finish(numeric(),list(valid=FALSE),1L,initial$message,0L))
    gamma <- initial$coefficients;current <- evaluate(gamma)
    if(!current$valid) return(finish(gamma,current,2L,current$message,0L))
    for(iter in 0:ctl$maxit) {
        if(ncol(B)) {
            information <- crossprod(B,current$information%*%B)
            score <- drop(crossprod(B,current$score))
            direction <- tryCatch(drop(B%*%solve(information,score)),error=function(e) rep(NA_real_,setup$d))
            scaled <- max(abs(score)/sqrt(diag(information)))
            step <- if(all(is.finite(direction))) reference_predictor_step(setup,direction) else Inf
        } else {direction <- numeric(setup$d);scaled <- step <- 0}
        if(is.finite(scaled) && scaled<=ctl$score.tol && step<=ctl$step.tol)
            return(finish(gamma,current,0L,"",iter,scaled,step))
        if(!is.finite(step) || !is.finite(scaled))
            return(finish(gamma,current,3L,"non-finite constrained score or information",iter,scaled,step))
        if(iter==ctl$maxit)
            return(finish(gamma,current,5L,"constrained iteration limit reached",iter,scaled,step))
        direction <- direction/max(1,step/(if(adjusted) min(1,ctl$max.step) else ctl$max.step))
        fraction <- 1
        if(setup$q>1L) {
            gap <- if(isTRUE(setup$partial)) .ordinal_partial_profile_gaps(setup,gamma) else diff(gamma[seq_len(setup$q)])
            change <- if(isTRUE(setup$partial)) .ordinal_partial_profile_gaps(setup,direction) else diff(direction[seq_len(setup$q)])
            decreasing <- change<0
            if(any(decreasing)) fraction <- min(fraction,.9*min(-gap[decreasing]/change[decreasing]))
        }
        ascent <- sum(current$score*direction);accepted <- FALSE;last.message <- ""
        for(half in 0:ctl$max.halves) {
            candidate <- gamma+fraction*direction;trial <- evaluate(candidate)
            if(trial$valid) {
                rounding <- 64*.Machine$double.eps*(1+abs(current$loglik))
                if(adjusted || trial$loglik-current$loglik>=.1*fraction*ascent-rounding) {
                    accepted <- TRUE;break
                }
                last.message <- "likelihood step did not increase sufficiently"
            } else last.message <- trial$message
            fraction <- fraction/2
        }
        if(!accepted) return(finish(gamma,current,4L,paste("constrained step search failed:",last.message),iter+1L,scaled,step))
        gamma <- candidate;current <- trial
    }
}

expect_constrained_match <- function(setup,L,null,adjusted=FALSE,start=setup$coefficients) {
    native <- .ordinal_constrained_fit(setup,L,null,start=start,adjusted=adjusted)
    reference <- reference_constrained_fit(setup,L,null,start=start,adjusted=adjusted)
    expect_identical(native$code,reference$code)
    # Linear algebra libraries can change the path slightly, so exact iteration
    # counts and the tightest tolerances are checked only in local development.
    local <- identical(Sys.getenv("NOT_CRAN"),"true")
    if(local) {
        expect_identical(native$iterations,reference$iterations)
        expect_identical(native$evaluations,reference$evaluations)
        expect_identical(native$message,reference$message)
    }
    expect_equal(native$coefficients,reference$coefficients,tolerance=if(local) 1e-10 else 1e-5)
    expect_equal(native$value,reference$value,tolerance=if(local) 1e-10 else 1e-6)
    expect_equal(native$state$cov.unscaled,reference$state$cov.unscaled,tolerance=if(local) 1e-8 else 1e-4)
    if(local) expect_equal(native$score,reference$score,tolerance=1e-6)
    native
}

test_that("native constrained iterations match the R reference for common slopes", {
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(Sat~Infl+Type+Cont,MASS::housing,weights=Freq,link=link,tol=1e-10)
        setup <- .ordinal_profile_setup(fit,list())
        d <- setup$d;none <- matrix(numeric(),0L,d)
        unrestricted <- expect_constrained_match(setup,none,numeric())
        expect_identical(unrestricted$code,0L)
        type <- grep("^Type",names(coef(fit)))
        L <- diag(d)[type,,drop=FALSE]
        expect_identical(expect_constrained_match(setup,L,numeric(nrow(L)))$code,0L)
        expect_constrained_match(setup,L,numeric(nrow(L)),adjusted=TRUE)
        expect_constrained_match(setup,L[1L,,drop=FALSE],0.25)
        expect_identical(expect_constrained_match(setup,diag(d),coef(fit))$code,0L)
    }
})

test_that("native constrained iterations match the R reference for threshold-specific slopes", {
    set.seed(123)
    n <- 400;x <- cbind(a=rnorm(n),b=rnorm(n))
    y <- ordered(cut(rlogis(n)+0.5*x[,1]-0.3*x[,2],c(-Inf,-1,0,1,Inf)))
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(x,y,link=link,nonparallel="b",tol=1e-10)
        setup <- .ordinal_profile_setup(fit,list())
        none <- matrix(numeric(),0L,setup$d)
        expect_identical(expect_constrained_match(setup,none,numeric())$code,0L)
        L <- diag(setup$d)[setup$q+1L,,drop=FALSE]
        expect_identical(expect_constrained_match(setup,L,0)$code,0L)
        equal <- matrix(0,setup$q-1L,setup$d)
        for(j in seq_len(setup$q-1L)) equal[j,setup$q+1L+j+0:1] <- c(1,-1)
        expect_identical(expect_constrained_match(setup,equal,numeric(nrow(equal)))$code,0L)
        expect_constrained_match(setup,equal,numeric(nrow(equal)),adjusted=TRUE)
    }
})

test_that("native constrained iterations report the same failures as the R reference", {
    fit <- fastglm_ordinal(Sat~Infl+Type+Cont,MASS::housing,weights=Freq,tol=1e-10)
    setup <- .ordinal_profile_setup(fit,list(maxit=1L))
    L <- diag(setup$d)[setup$q+1L,,drop=FALSE]
    limited <- expect_constrained_match(setup,L,3)
    expect_identical(limited$code,5L)
    tight <- .ordinal_profile_setup(fit,list(max.halves=1L))
    expect_false(isTRUE(expect_constrained_match(tight,L,1e4)$converged))
})
