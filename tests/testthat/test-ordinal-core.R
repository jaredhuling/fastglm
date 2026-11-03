test_that("ordinal native scores and information agree with probability derivatives", {
    skip_if_not_installed("numDeriv")
    x<-cbind(c(-1,.2,.8,1.5,-.5,.4),c(.2,1,-.3,.4,.8,-.7))
    counts<-rbind(c(2,1,0,1),c(0,2,3,1),c(1,2,1,0),c(0,1,1,4),c(3,1,1,0),c(1,2,1,1))
    w<-c(1,2,0,1,3,1);off<-c(.1,-.2,.3,0,-.1,.2)
    b<-c(-1,.1,1.2,.4,-.3);q<-3L;d<-length(b)
    probability<-function(b,i) diff(c(0,plogis(b[seq_len(q)]-sum(x[i,]*b[q+1:2])-off[i]),1))
    information<-matrix(0,d,d);score<-numeric(d);jacobians<-vector("list",nrow(x))
    for(i in seq_len(nrow(x))) {
        pr<-probability(b,i)
        J<-numDeriv::jacobian(function(z)probability(z,i),b)
        jacobians[[i]]<-J
        information<-information+sum(counts[i,])*w[i]*crossprod(J,J/pr)
        score<-score+drop(crossprod(J,counts[i,]*w[i]/pr))
    }
    V<-solve(information);adjustment<-numeric(d)
    for(i in seq_len(nrow(x))) {
        c<-vapply(seq_len(q),function(s) {
            H<-numDeriv::hessian(function(z)sum(probability(z,i)[seq_len(s)]),b)
            sum(counts[i,])*w[i]*sum(V*t(H))/2
        },numeric(1))
        adjustment<-adjustment+drop(crossprod(jacobians[[i]],diff(c(0,c,0))/probability(b,i)))
    }
    got<-fastglm:::ordinal_native_evaluate(x,integer(),counts,w,off,4L,b,TRUE,2L)
    expect_equal(got$information,information,tolerance=2e-7)
    expect_equal(drop(got$ordinary.score),score,tolerance=2e-7)
    expect_equal(drop(got$adjustment),adjustment,tolerance=2e-6)
    expect_equal(drop(got$score),score+adjustment,tolerance=2e-6)
    expect_equal(got$cov.unscaled,V,tolerance=2e-7)
    expect_equal(got$loglik,sum(vapply(seq_len(nrow(x)),function(i)
        sum(counts[i,]*w[i]*log(probability(b,i))),numeric(1))),tolerance=1e-12)
})

test_that("binary ordinal mean bias reduction equals logistic Firth fitting", {
    set.seed(38215)
    x<-cbind(x=rnorm(180),z=runif(180,-1,1));off<-runif(180,-.3,.3)
    w<-rep(c(0,1,2,3),length.out=nrow(x))
    y<-rbinom(nrow(x),1,plogis(.3+drop(x%*%c(.7,-.4))+off))
    fit<-fastglm:::.ordinal_fit_core(x,ordered(y,levels=0:1),w,off,firth=TRUE,block.size=11L)
    ref<-fastglm(cbind(1,x),y,binomial(),weights=w,offset=off,firth=TRUE,tol=1e-11,maxit=1000)
    transform<-diag(c(-1,1,1))
    expect_true(fit$converged)
    expect_equal(unname(fit$coefficients),drop(transform%*%coef(ref)),tolerance=2e-7)
    expect_equal(unname(fit$effective$covariance),unname(transform%*%vcov(ref)%*%transform),tolerance=2e-7)
    expect_identical(dim(fit$category$counts),c(0L,0L))
    expect_length(fit$category$codes,nrow(x))
    expect_lt(fit$predictor.step,fit$control$step.tol)
})

test_that("ordinal grouped counts and frequency expansion fit the same model", {
    x<-matrix(c(-1,-.4,.2,.9,1.4),ncol=1,dimnames=list(NULL,"x"))
    counts<-rbind(c(4,2,1),c(2,3,2),c(1,3,3),c(1,2,4),c(0,2,5))
    off<-c(.1,-.1,0,.2,-.2);w<-c(2,1,0,3,1)
    ii<-rep(seq_len(nrow(x)),rowSums(counts*w))
    yy<-unlist(lapply(seq_len(nrow(x)),function(i)rep(1:3,counts[i,]*w[i])))
    for(br in c(FALSE,TRUE)) {
        grouped<-fastglm:::.ordinal_fit_core(x,counts,w,off,firth=br,block.size=2L)
        individual<-fastglm:::.ordinal_fit_core(x[ii,,drop=FALSE],ordered(yy,levels=1:3),
                                              offset=off[ii],firth=br,block.size=7L)
        expect_true(grouped$converged)
        expect_true(individual$converged)
        expect_equal(grouped$coefficients,individual$coefficients,tolerance=1e-8)
        expect_equal(grouped$effective$covariance,individual$effective$covariance,tolerance=1e-8)
        expect_equal(grouped$loglik,individual$loglik,tolerance=1e-10)
        reversed<-fastglm:::.ordinal_fit_core(x,counts[,3:1],w,-off,firth=br,block.size=3L)
        expect_equal(unname(reversed$alpha),-rev(unname(grouped$alpha)),tolerance=1e-8)
        expect_equal(unname(reversed$beta),-unname(grouped$beta),tolerance=1e-8)
    }
})

test_that("intercept-only ordinal fits have exact adjusted-count solutions", {
    for(counts in list(c(3,5,2),c(0,5,2),c(0,5,0))) {
        fit<-fastglm:::.ordinal_fit_core(matrix(numeric(),1,0),matrix(counts,1),firth=TRUE)
        target<-qlogis((cumsum(counts)[1:2]+.5)/(sum(counts)+1))
        expect_true(fit$converged)
        expect_equal(unname(fit$alpha),target,tolerance=1e-9)
        cumulative<-plogis(target);g<-cumulative*(1-cumulative)
        expected<-(outer(cumulative,cumulative,pmin)-outer(cumulative,cumulative))/
            (sum(counts)*outer(g,g))
        expect_equal(unname(fit$effective$covariance),expected,tolerance=1e-9)
    }
    expect_warning(fit<-fastglm:::.ordinal_fit_core(matrix(numeric(),1,0),
                    matrix(c(3,0,0,2),1),firth=TRUE),"empty interior")
    expect_true(fit$converged)
    expect_equal(unname(fit$alpha),rep(qlogis(3.5/6),3),tolerance=1e-9)
    expect_equal(fit$rank,1L)
    expect_length(fit$score,1L)
    expect_identical(fit$threshold.map,rep(1L,3))
    expect_equal(qr(fit$mapped.covariance)$rank,1L)
    # Small tail categories must not disappear through total-minus-cumulative
    # cancellation when the first category has a large frequency.
    counts<-c(1e16,1,1)
    large<-fastglm:::.ordinal_fit_core(matrix(numeric(),1,0),matrix(counts,1),firth=TRUE)
    expect_true(large$converged)
    expect_equal(unname(large$alpha),log(c(1e16,1e16)+.5)-log(c(2,1)+.5),tolerance=1e-12)
})

test_that("ordinal convergence detects separation and respects linear coordinates", {
    x<-matrix(seq(-2,2,length.out=90),ncol=1,dimnames=list(NULL,"x"))
    y<-ordered(findInterval(x[,1],c(-.45,.6))+1,levels=1:3)
    br<-fastglm:::.ordinal_fit_core(x,y,firth=TRUE)
    expect_true(br$converged)
    expect_lt(br$predictor.step,1e-8)
    expect_warning(ml<-fastglm:::.ordinal_fit_core(x,y,maxit=40L),"did not converge")
    expect_false(ml$converged)
    expect_gt(ml$predictor.step,ml$control$step.tol)
    moved<-fastglm:::.ordinal_fit_core((x+100)*.002,y,firth=TRUE,block.size=13L)
    expect_true(moved$converged)
    expect_equal(unname(moved$beta),unname(br$beta/.002),tolerance=2e-7)
    expect_equal(unname(moved$alpha),unname(br$alpha+100*br$beta),tolerance=2e-7)
    expect_lt(moved$predictor.step,1e-8)
})

test_that("ordinal core rejects unsupported and unidentified inputs explicitly", {
    x<-matrix(seq(-1,1,length.out=9),ncol=1);y<-ordered(rep(1:3,3))
    expect_error(fastglm:::.ordinal_fit_core(Matrix::Matrix(x,sparse=TRUE),y),"dense")
    expect_error(fastglm:::.ordinal_fit_core(x+1i,y),"dense")
    expect_error(fastglm:::.ordinal_fit_core(cbind(1,x),y),"constant column|full rank")
    expect_error(fastglm:::.ordinal_fit_core(cbind(x,x),y),"full rank")
    expect_error(fastglm:::.ordinal_fit_core(x,y,offset=0),"offsets")
    expect_error(fastglm:::.ordinal_fit_core(x,y,tol=NA_real_),"tol")
    expect_error(fastglm:::.ordinal_fit_core(x,y,block.size=1.5),"block.size")
    expect_error(fastglm:::.ordinal_fit_core(x,y,control=list(other=TRUE)),"optimizer control")
    expect_error(fastglm:::.ordinal_fit_core(x,y,control=list(score.tol=NULL)),"optimizer control")
    expect_error(fastglm:::.ordinal_fit_core(x,y,maxit=1.2),"maxit")
})

test_that("zero-frequency extremes do not affect ordinal fitting or scaling", {
    x<-matrix(c(-1,0,1,2,1e300,-1e300),ncol=1,dimnames=list(NULL,"x"))
    y<-ordered(c(1,2,3,2,1,3),levels=1:3)
    w<-c(1,1,1,1,0,0);off<-c(.1,-.1,0,.2,1e300,-1e300)
    for(standardize in c(TRUE,FALSE)) {
        fit<-fastglm:::.ordinal_fit_core(x,y,w,off,firth=TRUE,
                                        control=list(standardize=standardize))
        deleted<-fastglm:::.ordinal_fit_core(x[1:4,,drop=FALSE],y[1:4],w[1:4],off[1:4],
                                            firth=TRUE,control=list(standardize=standardize))
        expect_true(fit$converged)
        expect_equal(fit$coefficients,deleted$coefficients,tolerance=1e-10)
        expect_equal(fit$effective$covariance,deleted$effective$covariance,tolerance=1e-10)
        expect_equal(fit$working$center,deleted$working$center,tolerance=1e-14)
        expect_equal(fit$working$scale,deleted$working$scale,tolerance=1e-14)
        expect_identical(fit$offset,off)
    }
    counts<-rbind(c(3,2,1),c(2,3,2),c(1,3,3),c(1,2,4),c(0,0,0),c(0,0,0))
    fit<-fastglm:::.ordinal_fit_core(x,counts,offset=off,firth=TRUE)
    deleted<-fastglm:::.ordinal_fit_core(x[1:4,,drop=FALSE],counts[1:4,],offset=off[1:4],firth=TRUE)
    expect_true(fit$converged)
    expect_equal(fit$coefficients,deleted$coefficients,tolerance=1e-10)
    expect_equal(fit$effective$covariance,deleted$effective$covariance,tolerance=1e-10)
})

test_that("ordinal grouped totals are accumulated after applying frequency weights", {
    x<-matrix(numeric(),1,0)
    huge<-matrix(rep(1e308,3),1)
    ordinary<-matrix(c(1,1,1),1)
    for(br in c(FALSE,TRUE)) {
        fit<-fastglm:::.ordinal_fit_core(x,huge,weights=1e-308,firth=br)
        reference<-fastglm:::.ordinal_fit_core(x,ordinary,firth=br)
        expect_true(fit$converged)
        expect_equal(fit$coefficients,reference$coefficients,tolerance=1e-10)
        expect_equal(fit$effective$covariance,reference$effective$covariance,tolerance=1e-10)
        expect_equal(fit$loglik,reference$loglik,tolerance=1e-10)
        expect_equal(fit$frequency,3,tolerance=1e-14)
        ignored<-fastglm:::.ordinal_fit_core(matrix(numeric(),2,0),rbind(ordinary,huge),
                                            weights=c(1,0),firth=br)
        expect_true(ignored$converged)
        expect_equal(ignored$coefficients,reference$coefficients,tolerance=1e-10)
        expect_equal(ignored$effective$covariance,reference$effective$covariance,tolerance=1e-10)
    }
})
