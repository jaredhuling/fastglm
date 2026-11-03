confirmation.scenarios <- c("balanced4","endpoint4","narrow_middle4","correlated6",
    "sparse_binary4","leverage4","alternative_balanced4","alternative_endpoint4","grouped4")
confirmation.cells <- expand.grid(scenario=confirmation.scenarios,n=c(100L,400L,1600L),
    KEEP.OUT.ATTRS=FALSE,stringsAsFactors=FALSE)
confirmation.repetitions <- 1000L
confirmation.generate <- function(scenario,n) {
    base <- sub("^alternative_","",scenario)
    grouped <- base=="grouped4"
    m <- if(grouped) 20L else n
    x <- cbind(x1=runif(m,-1.5,1.5),x2=rbinom(m,1,.35))
    alpha <- c(-1.1,.3,1.5); beta <- c(.45,-.65); offset <- .3*sin(x[,1])
    if(base=="endpoint4") {
        alpha <- c(-3.5,-.6,1); beta <- c(.6,-.4); offset <- .25*x[,1]
    } else if(base=="narrow_middle4") {
        x[,2] <- runif(m,-1.5,1.5)
        alpha <- c(-1.3,-.98,1.2); beta <- c(.8,-.5); offset <- .4*cos(x[,2])
    } else if(base=="correlated6") {
        x <- cbind(x1=x[,1],x2=.85*x[,1]+.15*runif(m,-1.5,1.5),
                   x3=rbinom(m,1,.5),x4=runif(m,-1.5,1.5))
        alpha <- c(-2,-1,-.1,.8,1.7); beta <- c(.3,-.4,.5,-.3); offset <- .2*x[,4]
    } else if(base=="sparse_binary4") {
        x[,2] <- rbinom(m,1,.04)
        alpha <- c(-1,0,1); beta <- c(.5,.9); offset <- numeric(m)
    } else if(base=="leverage4") {
        x[,1] <- x[,1]+8*rbinom(m,1,.05); x[,2] <- runif(m,-1.5,1.5)
        alpha <- c(-.6,.4,1.7); beta <- c(.3,-.2); offset <- .2*sin(x[,2])
    } else if(grouped) {
        x <- cbind(x1=seq(-1.4,1.4,length.out=20),x2=rep(c(0,1),10))
        alpha <- c(-1.4,-.2,1.1); beta <- c(.55,-.45); offset <- .2*cos(x[,1])
    }
    eta <- outer(-drop(x%*%beta)-offset,alpha,"+")
    null <- !startsWith(scenario,"alternative_")
    if(!null) eta <- eta-outer(x[,1],if(base=="balanced4") c(-.25,0,.25) else c(-.45,0,.45))
    F <- plogis(eta); probability <- cbind(F,1)-cbind(0,F)
    stopifnot(all(probability>0),max(abs(rowSums(probability)-1))<1e-14)
    K <- ncol(probability)
    if(grouped) {
        counts <- t(vapply(seq_len(m),function(i) drop(rmultinom(1,n/m,probability[i,])),numeric(K)))
        colnames(counts) <- as.character(seq_len(K))
        y <- counts/2; weights <- rep(2,m)
    } else {
        y <- ordered(1L+rowSums(runif(m)>F),levels=seq_len(K)); weights <- rep(1,m)
        counts <- matrix(0,m,K); counts[cbind(seq_len(m),as.integer(y))] <- 1
    }
    list(x=x,y=y,weights=weights,offset=offset,counts=counts,K=K,null=null)
}
