# Rscript .../run.R INSTALLED_LIBRARY NEW_OUTPUT
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==2L)
.libPaths(c(normalizePath(args[1L]),.libPaths()));library(fastglm)
out <- args[2L];if(dir.exists(out)) stop("Use a new confirmation directory")
dir.create(out,recursive=TRUE)
file.copy("R/ordinal-partial-boundary.R",file.path(out,"boundary-helper.R"))
sys.source(file.path(out,"boundary-helper.R"),.GlobalEnv)
d <- list(z=rbind(c(.2,-.4),c(1.1,.3),c(-.5,.8)),
    counts=rbind(c(4,0,3,2),c(2,5,0,4),c(1,3,2,5)),
    weights=c(2,1,3),offset=c(.13,-.21,.08))
saveRDS(d,file.path(out,"input.rds"))
checks <- list()
record <- function(name,error,tolerance) {
    checks[[length(checks)+1L]] <<- data.frame(check=name,error=error,tolerance=tolerance,
        passed=is.finite(error)&&error<tolerance)
    write.csv(do.call(rbind,checks),file.path(out,"checks.csv"),row.names=FALSE)
}
certify <- function(d,size=nrow(d$z),common=0L,link="logit",firth=TRUE) {
    state <- .ordinal_partial_boundary_start(common,ncol(d$z),ncol(d$counts),firth,link)
    for(at in split(seq_len(nrow(d$z)),ceiling(seq_len(nrow(d$z))/size)))
        .ordinal_partial_boundary_add(state,d$z[at,,drop=FALSE],integer(),d$counts[at,,drop=FALSE],
            d$weights[at],d$offset[at])
    .ordinal_partial_boundary_finish(state)
}
certificate <- certify(d)
record("saturated design certified",as.numeric(is.null(certificate)),.5)
record("one-row blocks identical",as.numeric(!identical(certify(d,1L),certificate)),.5)
changed <- d;changed$z <- sweep(d$z%*%matrix(c(2,.3,-.4,.8),2L),2L,c(1e4,-2e4),"+")
record("invertible predictor recoding",as.numeric(!identical(certify(changed),certificate)),.5)
rows <- rep(1:3,rowSums(d$counts*d$weights))
y <- unlist(lapply(1:3,function(i) rep(1:4,d$counts[i,]*d$weights[i])))
for(size in c(1L,7L)) {
    state <- .ordinal_partial_boundary_start(0L,2L,4L,TRUE,"logit")
    for(at in split(seq_along(rows),ceiling(seq_along(rows)/size)))
        .ordinal_partial_boundary_add(state,d$z[rows[at],,drop=FALSE],as.integer(y[at]),
            matrix(numeric(),0L,0L),rep(1,length(at)),d$offset[rows[at]])
    record(paste("coded expansion blocks",size),as.numeric(!identical(.ordinal_partial_boundary_finish(state),certificate)),.5)
}
zero <- d;zero$z <- rbind(d$z,c(9,9));zero$counts <- rbind(d$counts,rep(1,4))
zero$weights <- c(d$weights,0);zero$offset <- c(d$offset,20)
record("zero-frequency pattern ignored",as.numeric(!identical(certify(zero),certificate)),.5)
zero$counts[4L,] <- 1e-300;zero$weights[4L] <- 1e-300
record("zero-effective underflow ignored",as.numeric(!identical(certify(zero),certificate)),.5)
for(link in c("probit","cloglog")) record(paste(link,"not certified"),as.numeric(!is.null(certify(d,link=link))),.5)
record("ML not certified",as.numeric(!is.null(certify(d,firth=FALSE))),.5)
record("common predictor not certified",as.numeric(!is.null(certify(d,common=1L))),.5)
extra <- d;extra$z <- rbind(d$z,c(.4,.9));extra$counts <- rbind(d$counts,c(1,1,1,1))
extra$weights <- c(d$weights,1);extra$offset <- c(d$offset,0)
state <- .ordinal_partial_boundary_start(0L,2L,4L,TRUE,"logit")
.ordinal_partial_boundary_add(state,extra$z,integer(),extra$counts,extra$weights,extra$offset)
record("extra pattern not certified",as.numeric(!is.null(.ordinal_partial_boundary_finish(state))),.5)
record("extra pattern state released",as.numeric(!(is.null(state$patterns)&&is.null(state$counts)&&is.null(state$offsets))),.5)
changed <- d;changed$z <- rbind(d$z,d$z[1L,]);changed$counts <- rbind(d$counts,d$counts[1L,])
changed$weights <- c(d$weights,1);changed$offset <- c(d$offset,d$offset[1L]+2^-52)
record("different within-pattern offset not certified",as.numeric(!is.null(certify(changed))),.5)
changed <- d;changed$z <- cbind(c(0,2,4),1e16+c(0,2,4))
record("shifted rank-deficient design not certified",as.numeric(!is.null(certify(changed))),.5)
changed <- d;changed$counts <- d$counts[,c(1L,4L)]
record("binary outcome not certified",as.numeric(!is.null(certify(changed))),.5)
changed <- d;changed$counts <- rbind(c(0,2,3,4),c(2,3,4,0),c(1,2,3,4))
record("endpoint-only empty cells not certified",as.numeric(!is.null(certify(changed))),.5)

beta <- rbind(c(-.9,.15,1.2,.1,-.15,.2,.05,-.1,.12),
    c(-1.2,0,1.5,.05,-.075,.1,.025,-.05,.06),
    c(-.55,.35,1.9,-.12,.08,.16,-.2,.07,.11))
states <- list();pseudo <- d$counts*d$weights;pseudo[,c(1L,4L)] <- pseudo[,c(1L,4L)]+.5
for(case in 1:3) {
    b <- beta[case,]
    native <- fastglm:::ordinal_partial_native_evaluate(matrix(numeric(),3L,0L),d$z,
        integer(),d$counts,d$weights,d$offset,4L,b,TRUE)
    eta <- matrix(b[1:3],3L,3L,byrow=TRUE)-d$z%*%matrix(b[4:9],2L,3L)-d$offset
    F <- plogis(eta);P <- cbind(F,1)-cbind(0,F);reference <- numeric(9L)
    for(i in 1:3) for(j in 1:3) {
        direction <- numeric(9L);direction[j] <- 1;direction[3L+(j-1L)*2L+1:2] <- -d$z[i,]
        reference <- reference+direction*F[i,j]*(1-F[i,j])*
            (pseudo[i,j]/P[i,j]-pseudo[i,j+1L]/P[i,j+1L])
    }
    record(paste("affine endpoint-half-count equation",case),max(abs(native$score-reference)),2e-10)
    states[[case]] <- list(coefficients=b,native=native,reference.score=reference)
}
target <- pseudo/rowSums(pseudo);cumulative <- t(apply(target,1L,cumsum))[,1:3]
eta <- qlogis(cumulative);mapped <- solve(cbind(1,-d$z),eta+d$offset)
record("closed target has required zero probabilities",max(abs(target[cbind(c(1L,2L),c(2L,3L))])),1e-14)
record("first pattern adjacent tie",abs(eta[1L,1L]-eta[1L,2L]),1e-14)
record("second pattern adjacent tie",abs(eta[2L,2L]-eta[2L,3L]),1e-14)
record("boundary map remains finite",as.numeric(any(!is.finite(mapped))),.5)
record("boundary affine map",max(abs(cbind(1,-d$z)%*%mapped-d$offset-eta)),2e-12)
positive <- d;positive$counts[positive$counts==0] <- 1;colnames(positive$z) <- c("a","b")
record("positive intermediate counts not certified",as.numeric(!is.null(certify(positive))),.5)
fit <- fastglm_ordinal(positive$z,positive$counts,weights=positive$weights,offset=positive$offset,
    nonparallel=1:2,firth=TRUE,tol=1e-10)
pseudo.positive <- positive$counts*positive$weights
pseudo.positive[,c(1L,4L)] <- pseudo.positive[,c(1L,4L)]+.5
record("positive counterpart converges",as.numeric(!fit$converged),.5)
record("positive counterpart exact probabilities",max(abs(predict(fit)-pseudo.positive/rowSums(pseudo.positive))),2e-9)
saveRDS(list(states=states,boundary.probabilities=target,boundary.predictors=eta,
    boundary.map=mapped,positive.fit=fit),file.path(out,"reference-values.rds"))
writeLines(capture.output(sessionInfo()),file.path(out,"session.txt"))
files <- c("R/ordinal-partial-boundary.R",file.path("benchmarks/ordinal-partial-boundary-confirmation",c("protocol.txt","run.R")),
    list.files(system.file("libs",package="fastglm"),full.names=TRUE))
write.csv(data.frame(path=files,md5=unname(tools::md5sum(files))),file.path(out,"source-hashes.csv"),row.names=FALSE)
result <- do.call(rbind,checks);print(result[!result$passed,,drop=FALSE])
cat(sum(result$passed),"of",nrow(result),"certificate confirmation checks passed\n")
stopifnot(all(result$passed))
