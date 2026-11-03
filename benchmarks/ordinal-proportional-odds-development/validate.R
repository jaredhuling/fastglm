# Run from the package root with the development version of fastglm installed.
source("benchmarks/ordinal-proportional-odds-development/prototype.R")
set.seed(110305)
x <- cbind(x=rnorm(30),z=runif(30,-1,1))
counts <- matrix(sample(2:8,120,TRUE),30,4)
offset <- runif(30,-.2,.2)
rows <- rep(seq_len(nrow(counts)),rowSums(counts))
y <- unlist(lapply(seq_len(nrow(counts)),function(i) rep(seq_len(ncol(counts)),counts[i,])))
individual <- matrix(0,length(y),ncol(counts))
individual[cbind(seq_along(y),y)] <- 1
errors <- matrix(NA_real_,2,5,dimnames=list(c("ML","meanBR"),
    c("grouped_coefficients","grouped_covariance","direct_meat","reversal_statistic","minimum_eigenvalue")))
for(firth in c(FALSE,TRUE)) {
    a <- ordinal_po_covariance(x,counts,offset,firth)
    b <- ordinal_po_covariance(x[rows,,drop=FALSE],individual,offset[rows],firth)
    z <- cbind(1,x[rows,,drop=FALSE])
    score <- do.call(cbind,lapply(seq_len(ncol(counts)-1L),function(j)
        z*(as.numeric(y>j)-a$probability[rows,j])))
    # Reversing outcomes also reverses the offset sign under the chosen convention.
    reverse <- ordinal_po_covariance(x,counts[,ncol(counts):1],-offset,firth)
    value <- c(max(abs(a$coefficients-b$coefficients)),
        max(abs(a$covariance-b$covariance)),max(abs(a$meat-crossprod(score))),
        abs(a$statistic-reverse$statistic),min(eigen(a$covariance,symmetric=TRUE)$values))
    errors[if(firth) "meanBR" else "ML",] <- value
    stopifnot(max(value[1:4])<1e-7,value[5]>0)
}
print(errors,digits=5)
cat("\nAlgebra and representation checks only; no size or power study.\n")
print(packageVersion("fastglm"))
print(R.version.string)
