# Independent score/information calculations use log probabilities throughout.
tail_reference <- function(x,y,weights,offset,b,link) {
    eta <- drop(x%*%b)+offset
    if(link=="probit") {
        lp <- pnorm(eta,log.p=TRUE);lq <- pnorm(eta,lower.tail=FALSE,log.p=TRUE)
        ld <- dnorm(eta,log=TRUE);ratio <- -eta
    } else {
        t <- exp(eta);lq <- -t
        lp <- ifelse(t>log(2),log1p(-exp(-t)),log(-expm1(-t)))
        ld <- eta-t;ratio <- 1-t
    }
    working <- weights*exp(2*ld-lp-lq)
    u <- numeric(length(y));positive <- weights>0
    i <- positive & y>0;u[i] <- weights[i]*y[i]*exp(ld[i]-lp[i])
    i <- positive & y<1;u[i] <- u[i]-weights[i]*(1-y[i])*exp(ld[i]-lq[i])
    information <- crossprod(x,x*working);V <- solve(information)
    h <- working*rowSums((x%*%V)*x)
    list(score=drop(crossprod(x,u+.5*h*ratio)),ordinary=drop(crossprod(x,u)),
         information=information,covariance=V,working=working,h=h,lp=lp,lq=lq)
}
