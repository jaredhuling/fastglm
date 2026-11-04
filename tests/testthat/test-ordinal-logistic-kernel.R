test_that("logistic ordinal kernels agree with direct exponential probabilities", {
    for(alpha in list(c(-2,-.3,.8,3),c(-1,0,1e-7,2),c(-40,-10,10,40))) {
        x <- matrix(c(-20,-2,0,2,20),ncol=1);q <- length(alpha);K <- q+1L
        counts <- matrix(seq_len(5*K)%%4,5,K);w <- c(1,2,0,1,3);off <- c(0,.1,0,-.1,0)
        b <- c(alpha,.4);eta <- outer(-drop(x*.4)-off,alpha,"+");e <- exp(eta)
        P <- cbind(e[,1]/(1+e[,1]),
            e[,-q,drop=FALSE]*rep(expm1(diff(alpha)),each=5)/
                ((1+e[,-q,drop=FALSE])*(1+e[,-1,drop=FALSE])),1/(1+e[,q]))
        g <- e/(1+e)^2;I <- matrix(0,q+1,q+1);U <- numeric(q+1)
        for(i in 1:5) {
            J <- matrix(0,K,q+1)
            for(j in seq_len(q)) {J[j,j] <- g[i,j];J[j+1,j] <- -g[i,j]}
            J[,q+1] <- -x[i]*rowSums(J[,seq_len(q),drop=FALSE])
            I <- I+w[i]*sum(counts[i,])*crossprod(J,J/P[i,])
            U <- U+w[i]*drop(crossprod(J,counts[i,]/P[i,]))
        }
        for(block in c(1L,4L)) {
            got <- ordinal_native_evaluate(x,integer(),counts,w,off,K,b,block_size=block)
            expect_equal(got$loglik,sum(w*counts*log(P)),tolerance=1e-12)
            expect_equal(drop(got$score),U,tolerance=2e-8)
            expect_equal(got$information,I,tolerance=2e-8)
        }
    }
})
