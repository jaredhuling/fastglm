# Rscript .../confirm.R NEW_OUTPUT (load current R with the compiled library)
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==1L)
out <- args[1L];if(dir.exists(out)) stop("Use a fresh confirmation directory")
dir.create(out,recursive=TRUE)
pkgload::load_all(".",compile=FALSE,quiet=TRUE)
stopifnot(requireNamespace("ordinal",quietly=TRUE),requireNamespace("numDeriv",quietly=TRUE))
source("benchmarks/partial-proportional-odds-validation/reference.R")
checks <- list();fits <- list()
record <- function(name,error,tolerance) {
    checks[[length(checks)+1L]] <<- data.frame(check=name,error=error,tolerance=tolerance,
        passed=is.finite(error)&&error<tolerance)
    write.csv(do.call(rbind,checks),file.path(out,"checks.csv"),row.names=FALSE)
}
i <- seq_len(18L)
x <- cbind(x=cos(i*.43)/2,z1=(i%%5-2)/3,z2=sin(i*.91)/2)
counts <- cbind(low=3+i%%4,middle=2+i%%5,high=4+i%%3,top=3+i%%2)
w <- rep(c(2,1,0),6L);offset <- cos(i*.27)/8
rows <- rep(i,rowSums(counts)*w)
y <- ordered(unlist(lapply(i,function(j) rep(1:4,counts[j,]*w[j]))),levels=1:4)
data <- data.frame(x[rows,,drop=FALSE],off=offset[rows],y=y)
prediction <- data.frame(x,off=offset)
for(link in c("logit","probit","cloglog")) {
    full <- fastglm_ordinal(x,counts,weights=w,offset=offset,nonparallel=c("z1","z2"),link=link,tol=1e-10)
    reduced <- fastglm_ordinal(x,counts,weights=w,offset=offset,nonparallel="z2",link=link,tol=1e-10)
    external <- ordinal::clm(y~x+offset(off),nominal=~z1+z2,data=data,link=link,
        control=ordinal::clm.control(gradTol=1e-8,maxIter=200L))
    external.reduced <- ordinal::clm(y~x+z1+offset(off),nominal=~z2,data=data,link=link,
        control=ordinal::clm.control(gradTol=1e-8,maxIter=200L))
    L <- matrix(0,2L,length(coef(full)),dimnames=list(c("second minus first","third minus first"),names(coef(full))))
    L[,5L] <- -1;L[1L,7L] <- L[2L,9L] <- 1
    B <- qr.Q(qr(t(L)),complete=TRUE)[,-(1:2),drop=FALSE]
    A <- matrix(c(1.7,-.2,.3,.8),2L)
    lr <- fastglm_profile_test(full,contrast=L)
    record(paste(link,"LR available"),as.numeric(!lr$available),.5)
    record(paste(link,"nested native likelihood ratio"),abs(unname(lr$statistic)-2*as.numeric(logLik(full)-logLik(reduced))),2e-7)
    for(pair in list(list(full,external,"full"),list(reduced,external.reduced,"reduced"))) {
        record(paste(link,pair[[3]],"external convergence"),as.numeric(pair[[2]]$convergence$code!=0L),.5)
        record(paste(link,pair[[3]],"external likelihood"),abs(as.numeric(logLik(pair[[1]])-logLik(pair[[2]]))),2e-7)
        record(paste(link,pair[[3]],"external probabilities"),max(abs(predict(pair[[1]])-
            predict(pair[[2]],newdata=prediction,type="prob")$fit)),2e-6)
    }
    moved <- fastglm_profile_test(full,contrast=A%*%L)
    record(paste(link,"LR row basis"),abs(unname(moved$statistic-lr$statistic)),2e-7)
    scores <- list()
    for(estimation in c("ML","meanBR")) {
        test <- fastglm_score_test(full,contrast=L,null.estimation=estimation)
        record(paste(link,estimation,"score available"),as.numeric(!test$available),.5)
        theta <- test$diagnostics$restricted$coefficients
        ref <- partial_cumulant_reference(theta,x[,1L,drop=FALSE],x[,2:3],counts,w,offset,link)
        U <- if(estimation=="ML") ref$ordinary.score else ref$score
        I <- ref$information;N <- crossprod(B,I%*%B)
        efficient <- U-drop(I%*%B%*%solve(N,crossprod(B,U)))
        statistic <- drop(crossprod(efficient,solve(I,efficient)))
        record(paste(link,estimation,"independent efficient statistic"),abs(unname(test$statistic)-statistic),2e-5)
        record(paste(link,estimation,"independent nuisance equations"),max(abs(crossprod(B,U))/sqrt(diag(N))),5e-7)
        record(paste(link,estimation,"linear constraints"),max(abs(L%*%theta)),2e-11)
        moved <- fastglm_score_test(full,contrast=A%*%L,null.estimation=estimation)
        record(paste(link,estimation,"score row basis"),abs(unname(moved$statistic-test$statistic)),2e-7)
        scores[[estimation]] <- test
    }
    interval <- confint(full,parm=names(coef(full))[8L],method="profile",level=.8)
    record(paste(link,"profile available"),as.numeric(any(!attr(interval,"diagnostics")$converged)),.5)
    inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(t) -expm1(-exp(t)))
    for(side in 1:2) {
        value <- interval[1L,side];test <- fastglm_profile_test(full,parm=8L,null=value)
        start <- test$diagnostics$restricted$coefficients
        objective <- function(free) {
            b <- start;b[-8L] <- free;b[8L] <- value
            eta <- matrix(b[1:3],nrow(x),3L,byrow=TRUE)-x[,1L]*b[4L]-x[,2:3]%*%matrix(b[5:10],2L,3L)-offset
            F <- inverse(eta);P <- cbind(F,1)-cbind(0,F)
            if(any(P[w>0,,drop=FALSE]<=0)) return(Inf)
            -sum(w[w>0]*counts[w>0,,drop=FALSE]*log(P[w>0,,drop=FALSE]))
        }
        reference <- nlminb(start[-8L],objective,control=list(iter.max=1000L,eval.max=3000L,rel.tol=1e-12))
        record(paste(link,side,"profile reference convergence"),reference$convergence,.5)
        record(paste(link,side,"profile independent likelihood"),abs(-reference$objective-test$loglik["restricted"]),2e-7)
        record(paste(link,side,"profile target"),abs(test$statistic-qchisq(.8,1L)),3e-5)
    }
    expanded <- fastglm_ordinal(x[rows,,drop=FALSE],y,offset=offset[rows],nonparallel=c("z1","z2"),link=link,tol=1e-10)
    expanded.lr <- fastglm_profile_test(expanded,contrast=L)
    expanded.score <- fastglm_score_test(expanded,contrast=L,null.estimation="meanBR")
    record(paste(link,"expanded LR"),abs(unname(expanded.lr$statistic-lr$statistic)),2e-7)
    record(paste(link,"expanded adjusted score"),abs(unname(expanded.score$statistic-scores$meanBR$statistic)),2e-7)
    fits[[link]] <- list(lr=lr,scores=scores,interval=interval,external=external,external.reduced=external.reduced)
    saveRDS(fits,file.path(out,"fits.rds"))
}
writeLines(capture.output(sessionInfo()),file.path(out,"session.txt"))
files <- c("R/ordinal-profile.R","R/ordinal-partial-inference.R","R/firth-profile.R",
    "R/ordinal-partial-methods.R","src/fastglm.so",file.path("benchmarks/ordinal-partial-inference",c("protocol.txt","confirm.R")))
write.csv(data.frame(path=files,md5=unname(tools::md5sum(files))),file.path(out,"source-hashes.csv"),row.names=FALSE)
result <- do.call(rbind,checks);print(result[!result$passed,,drop=FALSE])
cat(sum(result$passed),"of",nrow(result),"confirmation checks passed\n")
stopifnot(all(result$passed))
