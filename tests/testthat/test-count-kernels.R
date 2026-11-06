# Sparse designs use the R formulas, so they are the reference for the dense kernels.
count_reader_pair <- function(n=120L,seed=123L) {
    set.seed(seed)
    x <- cbind(`(Intercept)`=1,a=rnorm(n),b=rbinom(n,1,0.4))
    mu <- exp(0.3+0.5*x[,2]-0.4*x[,3])
    y <- rnbinom(n,size=1.3,mu=mu)
    weights <- rpois(n,2);offset <- rnorm(n,sd=0.2)
    list(x=x,y=y,beta=c(0.25,0.45,-0.35),
        dense=fastglm:::.count_backend_reader(x,y,weights=weights,offset=offset),
        sparse=fastglm:::.count_backend_reader(as(Matrix::Matrix(x,sparse=TRUE),"generalMatrix"),y,weights=weights,offset=offset,block.size=37L))
}

test_that("compiled count kernels match the R formulas", {
    d <- count_reader_pair()
    for(link in c("log","sqrt","identity")) for(theta in list(NULL,0.7,3,5e7)) for(truncated in c(FALSE,TRUE)) {
        if(truncated && link!="log") next
        beta <- if(link=="log") d$beta else c(1.4,0.15,-0.1)
        args <- list(beta=beta,theta=theta,truncated=truncated,link=link,system=TRUE,theta.derivatives=!is.null(theta))
        a <- do.call(fastglm:::.count_backend_count_summary,c(list(d$dense),args))
        b <- do.call(fastglm:::.count_backend_count_summary,c(list(d$sparse),args))
        expect_equal(a$loglik,b$loglik,tolerance=1e-12)
        expect_equal(a$score,b$score,tolerance=1e-10)
        expect_equal(unname(a$information),unname(b$information),tolerance=1e-10)
        expect_equal(a$theta.score,b$theta.score,tolerance=1e-8)
        expect_equal(a$theta.information,b$theta.information,tolerance=1e-6)
        if(!truncated && !is.null(theta)) expect_equal(
            unname(fastglm:::.count_backend_nb_information(d$dense,setNames(beta,c("a","b","c")),theta,link)),
            unname(fastglm:::.count_backend_nb_information(d$sparse,setNames(beta,c("a","b","c")),theta,link)),tolerance=1e-9)
    }
})

test_that("an invalid mean stops the likelihood in both kernels", {
    d <- count_reader_pair()
    a <- fastglm:::.count_backend_count_summary(d$dense,c(-5,0,0),3,FALSE,"identity")
    b <- fastglm:::.count_backend_count_summary(d$sparse,c(-5,0,0),3,FALSE,"identity")
    expect_identical(a$loglik,-Inf);expect_identical(b$loglik,-Inf)
})

test_that("dense, sparse, and streamed count fits agree", {
    set.seed(123)
    n <- 600L;x <- cbind(`(Intercept)`=1,a=rnorm(n),b=rnorm(n))
    y <- rpois(n,exp(0.5+0.4*x[,2]))*rbinom(n,1,0.7)
    rows <- split(seq_len(n),ceiling(seq_len(n)/150L))
    source <- function() fastglm_source(function(k) list(X=x[rows[[k]],,drop=FALSE],y=y[rows[[k]]]),length(rows))
    sparse <- as(Matrix::Matrix(x,sparse=TRUE),"generalMatrix")
    for(dist in c("poisson","negbin")) {
        dense <- suppressWarnings(fastglm_hurdle(x,response=y,dist=dist))
        for(other in list(suppressWarnings(fastglm_hurdle(sparse,response=y,dist=dist)),
                          suppressWarnings(fastglm_hurdle(source(),dist=dist)))) {
            expect_equal(dense$coefficients,other$coefficients,tolerance=1e-6)
            expect_equal(dense$loglik,other$loglik,tolerance=1e-8)
            expect_equal(unname(dense$vcov),unname(other$vcov),tolerance=1e-5)
        }
    }
    y <- rnbinom(n,size=2,mu=exp(0.5+0.4*x[,2]))
    dense <- fastglm_nb(x,y)
    streamed <- fastglm_nb(source <- fastglm_source(function(k) list(X=x[rows[[k]],,drop=FALSE],y=y[rows[[k]]]),length(rows)))
    expect_equal(unname(coef(dense)),unname(coef(streamed)),tolerance=1e-5)
    expect_equal(dense$theta,streamed$theta,tolerance=1e-5)
    expect_equal(unname(dense$vcov.full),unname(streamed$vcov.full),tolerance=1e-4)
})
