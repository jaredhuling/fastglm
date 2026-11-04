partial_inference_fixture <- function() {
    i <- seq_len(16L)
    list(x=cbind(x=sin(i)/2,z1=(i%%4-1.5)/2,z2=cos(i*.7)/2),
        counts=cbind(low=2+i%%3,middle=3+i%%4,high=2+i%%5,top=4+i%%2),
        weights=rep(c(1,2,0,1),4L),offset=sin(i/2)/10)
}

partial_inference_equal <- function(fit) {
    L <- matrix(0,2L,length(coef(fit)),dimnames=list(c("second minus first","third minus first"),names(coef(fit))))
    L[,5L] <- -1;L[1L,7L] <- L[2L,9L] <- 1
    L
}

# Differentiate joint category log probabilities; no cumulative adjustment
# formula or native score is used by this independent reference.
partial_inference_numeric <- function(b,d,link) {
    p <- length(b);I <- Hbar <- matrix(0,p,p);U <- numeric(p);PQ <- array(0,c(p,p,p))
    inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(z) -expm1(-exp(z)))
    for(i in which(d$weights>0)) {
        lp <- function(theta) {
            eta <- theta[1:3]-d$x[i,1L]*theta[4L]-
                drop(matrix(theta[5:10],3L,byrow=TRUE)%*%d$x[i,2:3])-d$offset[i]
            log(diff(c(0,inverse(eta),1)))
        }
        pr <- exp(lp(b));u <- numDeriv::jacobian(lp,b);m <- sum(d$counts[i,])*d$weights[i]
        I <- I+m*crossprod(u,pr*u);U <- U+drop(crossprod(u,d$counts[i,]*d$weights[i]))
        for(k in 1:4) {
            H <- numDeriv::hessian(function(theta) lp(theta)[k],b)
            Hbar <- Hbar-m*pr[k]*H
            for(t in seq_len(p)) PQ[,,t] <- PQ[,,t]+m*pr[k]*(tcrossprod(u[k,])+H)*u[k,t]
        }
    }
    V <- solve(I);A <- vapply(seq_len(p),function(t) sum(V*t(PQ[,,t]))/2,numeric(1))
    list(score=U,adjustment=A,information=I,expected.hessian=Hbar)
}

test_that("partial likelihood ratios test equal slopes with other slopes remaining free", {
    d <- partial_inference_fixture()
    for(link in c("logit","probit","cloglog")) {
        full <- fastglm_ordinal(d$x,d$counts,weights=d$weights,offset=d$offset,
            nonparallel=c("z1","z2"),link=link,tol=1e-10)
        reduced <- fastglm_ordinal(d$x,d$counts,weights=d$weights,offset=d$offset,
            nonparallel="z2",link=link,tol=1e-10)
        L <- partial_inference_equal(full)
        result <- fastglm_profile_test(full,contrast=L)
        expect_true(result$available)
        expect_equal(unname(result$statistic),2*as.numeric(logLik(full)-logLik(reduced)),tolerance=2e-8)
        expect_equal(unname(result$parameter),2)
        expect_equal(drop(L%*%result$diagnostics$restricted$coefficients),c(0,0),tolerance=1e-12,ignore_attr=TRUE)
        expect_equal(unname(result$loglik["restricted"]),as.numeric(logLik(reduced)),tolerance=1e-9)
        A <- matrix(c(2,-.3,.4,1.2),2L)
        moved <- fastglm_profile_test(full,contrast=A%*%L)
        expect_true(moved$available)
        expect_equal(unname(moved$statistic),unname(result$statistic),tolerance=2e-8)
        expect_output(print(result),"Tested contrasts")
        expect_true(as.data.frame(result)$available)
        expect_error(fastglm_profile_test(full,parm=1,contrast=L),"either")
        expect_error(fastglm_profile_test(full,contrast=rbind(L[1,],2*L[1,])),"independent")
    }
})

test_that("partial score statistics use full-model nuisance equations and information", {
    skip_if_not_installed("numDeriv")
    d <- partial_inference_fixture()
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(d$x,d$counts,weights=d$weights,offset=d$offset,
            nonparallel=c("z1","z2"),link=link,firth=TRUE,tol=1e-10)
        L <- partial_inference_equal(fit)
        B <- qr.Q(qr(t(L)),complete=TRUE)[,-(1:2),drop=FALSE]
        for(estimation in c("ML","meanBR")) {
            result <- fastglm_score_test(fit,contrast=L,null.estimation=estimation)
            expect_true(result$available)
            theta <- result$diagnostics$restricted$coefficients
            ref <- partial_inference_numeric(theta,d,link)
            expect_equal(ref$information,ref$expected.hessian,tolerance=2e-6)
            U <- ref$score+if(estimation=="meanBR") ref$adjustment else 0
            I <- ref$information;N <- crossprod(B,I%*%B)
            efficient <- U-drop(I%*%B%*%solve(N,crossprod(B,U)))
            statistic <- drop(crossprod(efficient,solve(I,efficient)))
            expect_equal(unname(result$statistic),statistic,tolerance=2e-6)
            expect_lt(max(abs(crossprod(B,U))/sqrt(diag(N))),2e-7)
            expect_equal(drop(L%*%theta),c(0,0),tolerance=1e-12,ignore_attr=TRUE)
        }
        reduced <- fastglm_ordinal(d$x,d$counts,weights=d$weights,offset=d$offset,
            nonparallel="z2",link=link,firth=TRUE,tol=1e-10)
        b <- coef(reduced)
        inserted <- c(b[1:4],b[5],b[6],b[5],b[7],b[5],b[8])
        state <- partial_inference_numeric(inserted,d,link)
        expect_gt(max(abs(crossprod(B,state$score+state$adjustment))),1e-4)
        expect_error(fastglm_profile_test(fit,contrast=L),"maximum-likelihood")
        expect_error(confint(fit,method="profile"),"maximum-likelihood")
    }
})

test_that("partial profiles invert the constrained ordinary likelihood", {
    d <- partial_inference_fixture()
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(d$x,d$counts,weights=d$weights,offset=d$offset,
            nonparallel=c("z1","z2"),link=link,tol=1e-10)
        interval <- confint(fit,parm=names(coef(fit))[7L],method="profile",level=.8)
        expect_s3_class(interval,"fastglm_confint")
        expect_true(all(attr(interval,"diagnostics")$converged))
        expect_lt(max(abs(attr(interval,"diagnostics")$lr.error)),2e-5)
        expect_lt(interval[1L,1L],coef(fit)[7L]);expect_gt(interval[1L,2L],coef(fit)[7L])
        # A separate optimizer works directly in the original coefficients and
        # evaluates ordinary category probabilities, without the native score.
        inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(z) -expm1(-exp(z)))
        for(value in as.numeric(interval)) {
            tested <- fastglm_profile_test(fit,7L,value)
            start <- tested$diagnostics$restricted$coefficients
            objective <- function(free) {
                b <- start;b[-7L] <- free;b[7L] <- value
                eta <- matrix(b[1:3],nrow(d$x),3L,byrow=TRUE)-d$x[,1L]*b[4L]-
                    d$x[,2:3]%*%matrix(b[5:10],2L,3L)-d$offset
                F <- inverse(eta);P <- cbind(F,1)-cbind(0,F)
                if(any(P[d$weights>0,,drop=FALSE]<=0)) return(Inf)
                -sum(d$weights[d$weights>0]*d$counts[d$weights>0,,drop=FALSE]*log(P[d$weights>0,,drop=FALSE]))
            }
            ref <- nlminb(start[-7L],objective,control=list(iter.max=1000L,eval.max=3000L,rel.tol=1e-12))
            expect_equal(ref$convergence,0L)
            expect_equal(-ref$objective,unname(tested$loglik["restricted"]),tolerance=2e-8)
            expect_equal(unname(tested$statistic),qchisq(.8,1),tolerance=2e-5)
        }
    }
    for(level in list(NA_real_,0,1,Inf,c(.8,.9)))
        expect_error(confint(fit,method="profile",level=level),"level")
    binary <- fastglm_ordinal(matrix(seq(-1,1,length.out=12),ncol=1,dimnames=list(NULL,"z")),
        cbind(rep(c(3,4,2),4),rep(c(2,3,4),4)),nonparallel="z",tol=1e-10)
    expect_equal(nrow(confint(binary,method="profile",level=.8)),length(coef(binary)))
})

test_that("partial inference respects the observed domain rather than threshold order at zero", {
    x <- matrix(seq(10,12,length.out=30),ncol=1,dimnames=list(NULL,"x"))
    eta <- cbind(-1,.8)[rep(1L,30),]-cbind((x[,1]-11)*.7,(x[,1]-11)*.15)
    F <- plogis(eta);counts <- floor(100*(cbind(F,1)-cbind(0,F)))+1
    fit <- fastglm_ordinal(x,counts,nonparallel="x",tol=1e-10)
    expect_lt(diff(unname(fit$alpha)),0)
    result <- fastglm_profile_test(fit,parm=1:2,null=unname(fit$alpha))
    expect_true(result$available);expect_lt(unname(result$statistic),1e-9)
    gap <- setNames(c(-1,1,x[1L,1L],-x[1L,1L]),names(coef(fit)))
    expect_warning(failed <- fastglm_profile_test(fit,contrast=gap,null=-.1),"ordered cumulative|feasible starting")
    expect_false(failed$available);expect_true(is.na(failed$p.value))
    expect_warning(failed <- fastglm_score_test(fit,contrast=gap,null=-.1),"ordered cumulative|feasible starting")
    expect_false(failed$available)
    zero <- fastglm_ordinal(rbind(x,0),rbind(counts,c(1,1,1)),weights=c(rep(1,30),0),
        nonparallel="x",tol=1e-10)
    reference <- fastglm_profile_test(fit,parm=3L,null=.4)
    observed <- fastglm_profile_test(zero,parm=3L,null=.4)
    expect_true(reference$available);expect_true(observed$available)
    expect_equal(unname(reference$statistic),unname(observed$statistic),tolerance=1e-8)
    expect_warning(failed <- fastglm_score_test(fit,terms="x",control=list(maxit=1)),"iteration limit")
    expect_false(failed$available)
})

test_that("partial constrained geometry preserves cancellation and rejects lost constraints", {
    x <- rbind(c(-1e17,1),c(1e17,1))
    z <- rbind(c(1e17,-1e17),c(-1e17,1e17))
    setup <- list(q=2L,pc=2L,pn=2L,d=8L,x=x,domain.z=z,positive=c(TRUE,TRUE))
    b <- c(-1.25,.875,.1,.375,.1,0,.2,.1)
    expect_equal(.ordinal_partial_profile_predictor_step(setup,b),1.625,tolerance=1e-14)
    expect_equal(as.numeric(.ordinal_partial_profile_gaps(setup,b)),rep(2.125,2L),tolerance=1e-14)
    b[c(2L,7L,8L)] <- c(5.875,.4,.3)
    expect_equal(as.numeric(.ordinal_partial_profile_gaps(setup,b)),
        7.125-c(1,-1)*(2^-55*1e17),tolerance=1e-14)
    # The exact requested difference is smaller than the spacing of doubles
    # at these coefficient magnitudes. Do not silently test a zero difference.
    setup <- list(q=1L,pc=0L,pn=1L,d=2L,T=diag(2L),Ti=diag(2L),domain.z=matrix(0,1L,1L))
    L <- matrix(c(1,-1),1L)
    result <- .ordinal_partial_constraint_start(setup,1,c(1e17,1e17),.ordinal_constraint_basis(setup,L))
    expect_false(result$valid)
    expect_match(result$message,"cannot be represented accurately")
})
