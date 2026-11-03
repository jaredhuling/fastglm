po_scenarios <- c("regular3","regular5","correlated5","rare_lower3","rare_upper3",
                  "rare_interior3","rare_interior5","strong_signal3","alternative3","alternative5")
po_cells <- expand.grid(n=c(80L,250L,1000L),scenario=po_scenarios,stringsAsFactors=FALSE)
po_cells$cell <- seq_len(nrow(po_cells))
po_repetitions <- 1000L
po_methods <- c("ML","meanBR","car")
po_generate <- function(scenario,n) {
    five <- grepl("5$",scenario)
    x <- if(five) cbind(x1=runif(n,-1,1),x2=runif(n,-1,1),x3=rbinom(n,1,.4)) else
        cbind(x1=runif(n,-1,1))
    if(scenario=="correlated5") {
        x[,3] <- runif(n,-1,1)
        x[,2] <- .7*x[,1]+.3*x[,2]; x[,3] <- .7*x[,1]+.3*x[,3]
    }
    alpha <- if(five) c(-1.8,-.6,.6,1.8) else c(-.8,.8)
    if(scenario=="rare_lower3") alpha <- c(-3.8,.4)
    if(scenario=="rare_upper3") alpha <- c(-.4,3.8)
    if(scenario=="rare_interior3") alpha <- c(-.08,.08)
    if(scenario=="rare_interior5") alpha <- c(-1.8,-.08,.08,1.8)
    beta <- if(five) c(.6,-.5,.4) else if(scenario=="strong_signal3") 4 else .7
    delta <- if(scenario=="alternative3") c(-.5,.5) else
        if(scenario=="alternative5") c(-.6,-.2,.2,.6) else rep(0,length(alpha))
    F <- plogis(outer(-drop(x%*%beta),alpha,"+")-outer(x[,1],delta))
    stopifnot(all(F>0 & F<1),all(F[,-1,drop=FALSE]>F[,-ncol(F),drop=FALSE]))
    y <- 1L+rowSums(F<runif(n))
    counts <- matrix(0,n,length(alpha)+1L)
    counts[cbind(seq_len(n),y)] <- 1
    list(x=x,y=y,counts=counts,truth=delta[2]-delta[1],null=all(delta==0))
}
