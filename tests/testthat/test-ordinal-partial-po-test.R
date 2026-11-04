partial_po_fixture <- function(link="logit") {
    d <- expand.grid(x=seq(-1,1,length.out=12L),g=factor(c("a","b","c")))
    d$z <- cos(seq_len(nrow(d)));d$off <- .08*sin(seq_len(nrow(d)))
    X <- model.matrix(~x+g+z+x:g,d)[,-1L,drop=FALSE]
    gamma <- matrix(c(.05,.15,.12,.06,-.04,
        .07,.1,.04,.03,.02,.1,.05,-.08,-.02,.06),5L,3L)
    eta <- matrix(c(-1,.2,1.4),nrow(d),3L,byrow=TRUE)-.15*d$x-d$off-X[,-1L]%*%gamma
    inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(x)-expm1(-exp(x)))
    F <- inverse(eta)
    d[c("low","middle","high","top")] <- floor(45*(cbind(F,1)-cbind(0,F)))+1
    d$w <- rep(c(1,2,0),length.out=nrow(d));d
}

partial_po_contrast <- function(fit,columns) {
    q <- length(fit$alpha);pc <- length(fit$common);pn <- nrow(fit$gamma)
    at <- match(columns,rownames(fit$gamma));L <- matrix(0,length(at)*(q-1L),length(coef(fit)))
    for(j in 2:q) for(k in seq_along(at)) {
        row <- (j-2L)*length(at)+k
        L[row,q+pc+at[k]] <- -1;L[row,q+pc+(j-1L)*pn+at[k]] <- 1
    }
    colnames(L) <- names(coef(fit));L
}

test_that("partial common-slope tests match explicit constraints and reduced ML fits", {
    for(link in c("logit","probit","cloglog")) {
        d <- partial_po_fixture(link)
        fit <- fastglm_ordinal(cbind(low,middle,high,top)~x+g+z+x:g+offset(off),d,
            weights=w,nonparallel=~g+z+x:g,link=link,tol=1e-10)
        for(term in c("g","x:g")) {
            cols <- fit$design.columns[fit$design.assign==match(term,attr(fit$terms,"term.labels"))]
            L <- partial_po_contrast(fit,cols)
            out <- fastglm_po_test(fit,terms=term)
            explicit <- fastglm_score_test(fit,contrast=L,
                control=list(score.tol=1e-10,step.tol=1e-10,maxit=500))
            expect_true(out$available);expect_true(explicit$available)
            expect_equal(out$statistic,unname(explicit$statistic),tolerance=2e-7)
            expect_equal(out$df,4L);expect_equal(unname(out$contrast),unname(L))
            remaining <- setdiff(fit$nonparallel,match(cols,fit$design.columns))
            reduced <- fastglm_ordinal(fit$x,fit$y,weights=fit$weights,offset=fit$offset,
                nonparallel=remaining,link=link,tol=1e-10)
            lr <- fastglm_po_test(fit,terms=term,method="lr")
            profile <- fastglm_profile_test(fit,contrast=L,
                control=list(score.tol=1e-10,step.tol=1e-10,maxit=500))
            expect_true(lr$available);expect_true(lr$null.converged)
            expect_equal(lr$statistic,unname(profile$statistic),tolerance=1e-8)
            expect_equal(lr$statistic,2*as.numeric(logLik(fit)-logLik(reduced)),tolerance=1e-8)
            expect_equal(lr$df,4L)
        }
        all <- fastglm_po_test(fit)
        expect_equal(all$df,10L);expect_setequal(all$terms,c("g","z","x:g"))
        expect_s3_class(all,"fastglm_po_test");expect_s3_class(all,"htest")
        expect_equal(as.data.frame(all)$statistic,all$statistic)
        expect_output(print(all),"p-value")
        assessment <- fastglm_ordinal_assess(fit,po.test="score",po.terms="g")
        expect_equal(assessment$proportional.odds.test$statistic,fastglm_po_test(fit,"g")$statistic,tolerance=1e-10)
        expect_error(fastglm_ordinal_assess(fit,proportional.odds=TRUE),"descriptive binary-slope")
    }
})

test_that("partial meanBR common-slope tests use reduced-null estimates and ordinary scores", {
    for(link in c("logit","probit","cloglog")) {
        d <- partial_po_fixture(link)
        fit <- fastglm_ordinal(cbind(low,middle,high,top)~x+g+z+x:g+offset(off),d,
            weights=w,nonparallel=~g+z+x:g,link=link,firth=TRUE,tol=1e-10)
        test <- fastglm_po_test(fit,"g",null.estimation="meanBR")
        expect_true(test$available);expect_match(test$note,"not the full-model adjusted-score")
        reduced <- fastglm_ordinal(fit$x,fit$y,weights=fit$weights,offset=fit$offset,
            nonparallel=which(fit$design.columns%in%c("z","x:gb","x:gc")),
            link=link,firth=TRUE,tol=1e-10)
        gamma <- fit$gamma
        for(name in rownames(gamma)) gamma[name,] <- if(name%in%names(reduced$beta))
            reduced$beta[name] else reduced$gamma[name,]
        expected <- c(reduced$alpha,reduced$beta[names(fit$beta)],as.vector(gamma))
        expect_equal(unname(test$null.coefficients),unname(expected),tolerance=1e-9)
        setup <- fastglm:::.ordinal_profile_setup(fit,list(),profile=FALSE)
        state <- setup$evaluate(drop(setup$Ti%*%expected),adjusted=FALSE)
        B <- fastglm:::.ordinal_constraint_basis(setup,test$contrast)$basis
        U <- as.numeric(state$ordinary.score)
        efficient <- U-state$information%*%B%*%solve(crossprod(B,state$information%*%B),crossprod(B,U))
        statistic <- drop(crossprod(efficient,state$cov.unscaled%*%efficient))
        expect_equal(test$statistic,statistic,tolerance=2e-8)
        expect_gt(max(abs(crossprod(B,U))),1e-4)
        expect_gt(abs(drop(crossprod(U,state$cov.unscaled%*%U))-statistic),1e-6)
        expect_error(fastglm_po_test(fit,method="lr",null.estimation="meanBR"),"require null.estimation")
        lr <- fastglm_po_test(fit,"g",method="lr")
        expect_true(lr$available);expect_equal(lr$null.estimation,"ML")
        expect_match(lr$note,"refitted by maximum likelihood")
    }
})

test_that("partial term-specific tests leave all other varying effects unrestricted", {
    d <- partial_po_fixture()
    fit <- fastglm_ordinal(cbind(low,middle,high,top)~x+g+z+x:g+offset(off),d,
        weights=w,nonparallel=~g+z+x:g,tol=1e-10)
    for(method in c("score","lr")) {
        multiple <- fastglm_po_test(fit,c("g","x:g"),by.term=TRUE,method=method)
        expect_s3_class(multiple,"fastglm_test_list")
        expect_equal(vapply(multiple$tests,function(z)z$df,integer(1)),c(Joint=8L,g=4L,`x:g`=4L))
        expect_equal(nrow(as.data.frame(multiple)),3L)
        expect_match(multiple$note,"other varying effects remain unrestricted")
        for(term in c("g","x:g")) expect_equal(multiple$tests[[term]]$statistic,
            fastglm_po_test(fit,term,method=method)$statistic,tolerance=1e-10)
    }
    common <- fastglm_po_test(fit,"x")
    expect_false(common$available);expect_equal(common$df,0L)
    expect_identical(common$reason,"no_slope_contrasts")
    expect_match(common$detail,"already have common slopes")
    mixed <- fastglm_po_test(fit,c("x","g"),by.term=TRUE)
    expect_equal(mixed$tests$Joint$statistic,fastglm_po_test(fit,"g")$statistic,tolerance=1e-10)
    expect_identical(mixed$tests$Joint$already.common.terms,"x")
    expect_false(mixed$tests$x$available)
    expect_error(fastglm_po_test(fit,c("g","g")),"duplicated")
    expect_error(fastglm_po_test(fit,"gb"),"unknown")
    expect_error(fastglm_po_test(fit,by.term=NA),"by.term")
    expect_error(fastglm_po_test(fit,control=list(score.tol=1e-8)),"tol.*maxit")
})

test_that("partial common-slope tests preserve grouped frequencies and offsets", {
    for(link in c("logit","probit","cloglog")) {
        d <- partial_po_fixture(link);X <- model.matrix(~x+g+z+x:g,d)[,-1L,drop=FALSE]
        counts <- as.matrix(d[c("low","middle","high","top")])
        fit <- fastglm_ordinal(X,counts,weights=d$w,offset=d$off,nonparallel=2:6,link=link,tol=1e-10)
        frequency <- counts*d$w;ii <- rep(seq_len(nrow(d)),rowSums(frequency))
        yy <- unlist(lapply(seq_len(nrow(d)),function(i)rep(1:4,frequency[i,])))
        expanded <- fastglm_ordinal(X[ii,,drop=FALSE],ordered(yy,levels=1:4,labels=colnames(counts)),
            offset=d$off[ii],nonparallel=2:6,link=link,tol=1e-10)
        for(estimation in c("ML","meanBR")) {
            out <- fastglm_po_test(fit,c("gb","gc"),null.estimation=estimation)
            reference <- fastglm_po_test(expanded,c("gb","gc"),null.estimation=estimation)
            expect_true(out$available);expect_true(reference$available)
            expect_equal(out$statistic,reference$statistic,tolerance=2e-8)
            replaced <- fastglm_po_test(fit,c("gb","gc"),null.estimation=estimation,
                newdata=X,y=counts,weights=d$w,offset=d$off+1.7)
            expect_equal(replaced$statistic,out$statistic,tolerance=2e-8)
        }
        ml <- fastglm_po_test(fit,c("gb","gc"))
        tripled <- fastglm_po_test(fit,c("gb","gc"),newdata=X,y=counts,
            weights=3*d$w,offset=d$off)
        expect_equal(tripled$statistic,3*ml$statistic,tolerance=2e-8)
        changed <- sweep(sweep(X,2L,c(1,-2,3,4,-5,6),"+"),2L,c(.01,-2,3,100,.1,-.2),"*")
        recoded <- fastglm_ordinal(changed,counts,weights=d$w,offset=d$off,
            nonparallel=2:6,link=link,tol=1e-10)
        expect_equal(fastglm_po_test(recoded,c("gb","gc"))$statistic,ml$statistic,tolerance=2e-8)
    }
})

test_that("partial common-slope tests retain exclusions and explicit unavailable results", {
    d <- partial_po_fixture();d$x[2L] <- NA
    fit <- fastglm_ordinal(cbind(low,middle,high,top)~x+g+z+x:g+offset(off),d,
        weights=w,nonparallel=~g+z+x:g,na.action=na.exclude,tol=1e-10)
    out <- fastglm_po_test(fit,"g")
    counts <- as.matrix(d[c("low","middle","high","top")])
    explicit <- fastglm_po_test(fit,"g",newdata=d,y=counts,weights=d$w,offset=d$off)
    expect_equal(explicit$statistic,out$statistic,tolerance=1e-10)
    expect_equal(explicit$observations,sum(d$w>0 & !is.na(d$x)))
    fit$converged <- FALSE
    expect_equal(fastglm_po_test(fit,"g")$statistic,out$statistic,tolerance=1e-10)
    expect_true(fastglm_po_test(fit,"g",method="lr")$available)
    fit$converged <- TRUE
    exhausted <- fastglm_po_test(fit,"g",control=list(maxit=1))
    expect_false(exhausted$available);expect_equal(exhausted$reason,"null_fit_not_converged")
    expect_false(fastglm_po_test(fit,"g",method="lr",control=list(maxit=1))$available)
    counts[,2L] <- 0
    empty <- fastglm_po_test(fit,"g",newdata=d,y=counts,weights=d$w)
    expect_false(empty$available);expect_equal(empty$reason,"empty_categories")
    expect_identical(empty$empty.categories,"middle");expect_equal(empty$df,4L)
    none <- fastglm_po_test(fit,"g",newdata=d,y=counts,weights=rep(0,nrow(d)))
    expect_false(none$available);expect_equal(none$reason,"no_usable_outcomes")
    constant <- d;constant$x <- 1
    bad <- fastglm_po_test(fit,"g",newdata=constant,y=as.matrix(d[c("low","middle","high","top")]),weights=d$w)
    expect_false(bad$available);expect_equal(bad$reason,"rank_deficient_design")
    binary <- fastglm_ordinal(cbind(x=seq(-1,1,length.out=12)),
        cbind(low=1:12,high=12:1),nonparallel="x")
    for(method in c("score","lr")) {
        test <- fastglm_po_test(binary,method=method)
        expect_false(test$available);expect_equal(test$df,0L)
        expect_match(test$detail,"binary outcome")
    }
    expect_warning(separated <- fastglm_ordinal(cbind(x=c(0,0,1,1),z=c(-1,1,-1,1)),
        rbind(c(6,3,2),c(2,4,3),c(0,0,8),c(0,0,9)),nonparallel="x",firth=TRUE,maxit=1),
        "did not converge")
    expect_false(separated$converged)
    expect_equal(fastglm_po_test(separated)$reason,"null_separation")
    expect_true(fastglm_po_test(separated,null.estimation="meanBR")$available)
})

test_that("partial source tests require explicit evaluation data and common API stays unchanged", {
    d <- partial_po_fixture();X <- as.matrix(d[c("x","z")])
    counts <- as.matrix(d[c("low","middle","high","top")])
    source <- fastglm_source(function(k) {
        at <- ((k-1L)*12L+1L):(k*12L)
        list(X=X[at,,drop=FALSE],y=counts[at,,drop=FALSE],weights=d$w[at],offset=d$off[at])
    },3L)
    stream <- fastglm_ordinal_streaming(source,nonparallel="z")
    dense <- fastglm_ordinal(X,counts,weights=d$w,offset=d$off,nonparallel="z")
    expect_error(fastglm_po_test(stream),"explicit evaluation data")
    explicit <- fastglm_po_test(stream,newdata=X,y=counts,weights=d$w,offset=d$off)
    expect_equal(explicit$statistic,fastglm_po_test(dense)$statistic,tolerance=1e-10)
    retained <- c(2:10,15:24,29:36)
    evaluation <- counts[retained,c(2,1,4,3),drop=FALSE]
    colnames(evaluation) <- colnames(counts)
    evaluation[,1L] <- evaluation[,1L]+seq_along(retained)%%4L
    new.offset <- d$off[retained]+seq_along(retained)/100
    evaluated <- fastglm_ordinal(X[retained,,drop=FALSE],evaluation,
        weights=d$w[retained],offset=new.offset,nonparallel="z")
    for(method in c("score","lr")) {
        supplied <- fastglm_po_test(stream,newdata=X[retained,,drop=FALSE],y=evaluation,
            weights=d$w[retained],offset=new.offset,method=method)
        expect_true(supplied$available)
        expect_equal(supplied$statistic,fastglm_po_test(evaluated,method=method)$statistic,tolerance=1e-10)
        expect_equal(supplied$observations,sum(d$w[retained]>0))
        expect_equal(supplied$frequency,sum(evaluation*d$w[retained]))
    }
    common <- fastglm_ordinal(X,counts,weights=d$w,offset=d$off)
    expect_equal(fastglm_po_test(common),fastglm_po_test(common,method="score"))
    expect_error(fastglm_po_test(common,method="lr"),"requires a partial ordinal")
    common.source <- fastglm_ordinal_streaming(source)
    expect_error(fastglm_po_test(common.source),"explicit evaluation data")
    common.evaluation <- fastglm_ordinal(X[retained,,drop=FALSE],evaluation,
        weights=d$w[retained],offset=new.offset)
    supplied <- fastglm_po_test(common.source,newdata=X[retained,,drop=FALSE],y=evaluation,
        weights=d$w[retained],offset=new.offset)
    expect_true(supplied$available)
    expect_equal(supplied$statistic,fastglm_po_test(common.evaluation)$statistic,tolerance=1e-10)
})
