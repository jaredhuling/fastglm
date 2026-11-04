partial_effect_fixture <- function(link) {
    d <- data.frame(x=seq(.7,1.8,length.out=48L),z=sin(seq_len(48L)))
    d$off <- d$x/10
    eta <- cbind(-.5,1)[rep(1L,nrow(d)),]-(.3*d$x-.1*d$x^2)-d$off-
        cbind(.1*d$x*d$z-.15*d$z,.2*d$x*d$z-.05*d$z)
    inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(x)-expm1(-exp(x)))
    F <- inverse(eta);P <- cbind(F,1)-cbind(0,F)
    d[c("low","middle","high")] <- floor(35*P)+1
    d
}

test_that("partial marginal effects and full-model uncertainty match independent differentiation", {
    skip_if_not_installed("numDeriv")
    for(link in c("logit","probit","cloglog")) for(br in c(FALSE,TRUE)) {
        d <- partial_effect_fixture(link)
        fit <- fastglm_ordinal(cbind(low,middle,high)~x+I(x^2)+x:z+z+offset(off),d,
            nonparallel=~x:z+z,link=link,firth=br,tol=1e-10)
        expect_true(fit$converged)
        new <- d[c(2,5,11,18,27,34),];w <- 1:nrow(new)
        q <- length(fit$alpha);pc <- length(fit$beta);pn <- length(fit$nonparallel)
        inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(x)-expm1(-exp(x)))
        probability <- function(b,data) {
            X <- model.matrix(delete.response(fit$terms),data)[,-1L,drop=FALSE]
            gamma <- matrix(b[q+pc+seq_len(q*pn)],pn,q)
            eta <- matrix(b[seq_len(q)],nrow(X),q,byrow=TRUE)-
                drop(X[,fit$common,drop=FALSE]%*%b[q+seq_len(pc)])-new$off-
                X[,fit$nonparallel,drop=FALSE]%*%gamma
            F <- inverse(eta);cbind(F,1)-cbind(0,F)
        }
        for(variable in c("x","z")) {
            effect <- function(b) {
                high <- low <- new;h <- 1e-4
                high[[variable]] <- high[[variable]]+h;low[[variable]] <- low[[variable]]-h
                colSums((probability(b,high)-probability(b,low))*w)/(2*h*sum(w))
            }
            result <- fastglm_marginal_effects(fit,variable,new,weights=w)
            # A broader Richardson starting step keeps the outer derivative
            # above rounding noise from the inner probability difference.
            G <- numDeriv::jacobian(effect,coef(fit),method.args=list(d=.01))
            expect_equal(result$estimate,effect(coef(fit)),tolerance=2e-8,ignore_attr=TRUE)
            expect_equal(result$std.error,sqrt(diag(G%*%vcov(fit)%*%t(G))),tolerance=3e-6,ignore_attr=TRUE)
            expect_equal(sum(result$estimate),0,tolerance=1e-14)
            cumulative <- fastglm_marginal_effects(fit,variable,new,weights=w,type="cumulative")
            expect_equal(cumulative$estimate,head(cumsum(result$estimate),-1L),tolerance=1e-12)
            expect_equal(fastglm_marginal_effects(fit,variable,new,weights=w,vcov.=vcov(fit)),result,tolerance=1e-9)
            robust <- fastglm_vcov(fit,cluster=rep(1:8,length.out=nrow(d)))
            ordering <- rev(seq_len(nrow(robust)))
            supplied <- fastglm_marginal_effects(fit,variable,new,weights=w,vcov.=robust[ordering,ordering])
            expect_equal(supplied$std.error,sqrt(diag(G%*%robust%*%t(G))),tolerance=3e-6,
                ignore_attr=TRUE,info=paste(link,br,variable))
        }
        before <- fastglm_marginal_effects(fit,"x",new,weights=w)
        fit$block.size <- 2L
        expect_equal(fastglm_marginal_effects(fit,"x",new,weights=w),before,tolerance=1e-12)
    }
})

test_that("partial matrix marginal effects retain scaling and frequency conventions", {
    for(link in c("logit","probit","cloglog")) {
        d <- partial_effect_fixture(link);X <- as.matrix(d[c("x","z")])
        counts <- as.matrix(d[c("low","middle","high")])
        fit <- fastglm_ordinal(X,counts,nonparallel="z",offset=d$off,link=link,firth=TRUE,tol=1e-10)
        w <- rowSums(counts);index <- rep(seq_len(nrow(X)),w)
        for(variable in c("x","z")) {
            effect <- fastglm_marginal_effects(fit,variable,weights=w)
            expect_equal(fastglm_marginal_effects(fit,variable,X[index,,drop=FALSE],
                offset=d$off[index]),effect,tolerance=1e-11)
            high <- low <- X;high[,variable] <- high[,variable]+1e-4;low[,variable] <- low[,variable]-1e-4
            expected <- colSums((predict(fit,high,offset=d$off)-predict(fit,low,offset=d$off))*w)/(2e-4*sum(w))
            expect_equal(effect$estimate,expected,tolerance=1e-8,ignore_attr=TRUE)
        }
        invalid <- X;invalid[1L,] <- NA
        expect_error(fastglm_marginal_effects(fit,"z",invalid),"complete finite")
        expect_true(all(is.finite(fastglm_marginal_effects(fit,"z",invalid,weights=c(0,w[-1L]))$estimate)))
        expect_error(fastglm_marginal_effects(fit,"none"),"design column")
        direction <- sign(diff(fit$gamma[1L,]))
        invalid <- X;invalid[1L,"z"] <- direction*1e6
        expect_error(fastglm_marginal_effects(fit,"z",invalid),"row\\(s\\): 1")
    }
})
