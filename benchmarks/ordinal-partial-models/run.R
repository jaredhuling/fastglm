args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L,!file.exists(args[1L]))
out <- args[1L];dir.create(out,recursive=TRUE)
pkgload::load_all('.',quiet=TRUE,compile=FALSE)
stopifnot(requireNamespace('ordinal',quietly=TRUE))
files <- c('R/ordinal.R','R/ordinal-partial.R','R/ordinal-partial-methods.R',
    'inst/include/ordinal_partial.h','src/fit_ordinal_partial.cpp','src/fastglm.so',
    'benchmarks/ordinal-partial-models/protocol.txt','benchmarks/ordinal-partial-models/run.R')
write.csv(data.frame(path=files,md5=unname(tools::md5sum(files))),file.path(out,'source-md5.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,'session.txt'))
G <- function(eta,link) switch(link,logit=plogis(eta),probit=pnorm(eta),cloglog=-expm1(-exp(eta)))
records <- list();models <- list();inputs <- list()
record <- function(name,role,link,build) {
    warnings <- character();a <- b <- NULL
    result <- tryCatch(withCallingHandlers({
        input <- build();inputs[[name]] <<- input
        input$data$.caseweight <- input$weights
        formula <- input$formula;nominal <- input$nominal
        a <- fastglm_ordinal(formula,input$data,weights=.caseweight,
            nonparallel=nominal,link=link,tol=1e-9,maxit=500)
        b <- ordinal::clm(input$reference.formula,data=input$expanded,nominal=nominal,
            weights=frequency,link=link,
            control=ordinal::clm.control(maxIter=500,gradTol=1e-9))
        prediction <- predict(a,input$prediction)
        reference <- predict(b,newdata=input$prediction,type='prob')$fit
        probability.error <- max(abs(prediction-reference))
        likelihood.error <- abs(as.numeric(logLik(a))-as.numeric(logLik(b)))
        recoded.error <- NA_real_
        if(role=='fresh') {
            recoded <- input$data
            recoded$x1 <- 10+1000*recoded$x1;recoded$z1 <- 2+5*recoded$z1
            cfit <- fastglm_ordinal(formula,recoded,weights=.caseweight,
                nonparallel=nominal,link=link,tol=1e-9,maxit=500)
            recoded.error <- max(abs(predict(cfit)-predict(a)))
            if(!cfit$converged) stop('recoded fit failed to converge')
        }
        clm.code <- b$convergence$code
        passed <- isTRUE(a$converged) && isTRUE(clm.code==0) && likelihood.error<2e-6 &&
            probability.error<2e-6 && (is.na(recoded.error) || recoded.error<2e-7)
        data.frame(name,role,link,converged=a$converged,reference.code=clm.code,
            likelihood.error,probability.error,recoded.error,
            score=a$scaled.score,predictor.step=a$predictor.step,minimum.gap=a$minimum.gap,
            passed,message='')
    },warning=function(w) {warnings <<- c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),
        error=function(e) data.frame(name,role,link,converged=FALSE,reference.code=NA_integer_,
            likelihood.error=NA_real_,probability.error=NA_real_,recoded.error=NA_real_,
            score=NA_real_,predictor.step=NA_real_,minimum.gap=NA_real_,passed=FALSE,message=conditionMessage(e)))
    result$warnings <- paste(unique(warnings),collapse=' | ')
    records[[name]] <<- result;models[[name]] <<- list(fastglm=a,reference=b)
    write.csv(do.call(rbind,records),file.path(out,'comparison.csv'),row.names=FALSE)
    saveRDS(inputs,file.path(out,'inputs.rds'),compress='xz')
    saveRDS(models,file.path(out,'models.rds'),compress='xz')
}
counter <- 0L
for(link in c('logit','probit','cloglog')) for(size in 1:2) {
    counter <- counter+1L;seed <- 115100L+counter
    record(paste(link,size,sep='-'),'fresh',link,function() {
        set.seed(seed);n <- c(36L,90L)[size];K <- c(3L,5L)[size];pc <- pn <- size;q <- K-1L
        x <- matrix(rnorm(n*pc),n,pc);z <- matrix(runif(n*pn,-1,1),n,pn)
        colnames(x) <- paste0('x',seq_len(pc));colnames(z) <- paste0('z',seq_len(pn))
        off <- sin(seq_len(n))/10;beta <- seq(.25,.45,length.out=pc)
        gamma <- outer(seq_len(pn)-1L,seq_len(q),function(i,j) .12*sin(i+j))
        eta <- matrix(seq(-1.1,1.1,length.out=q),n,q,byrow=TRUE)-drop(x%*%beta)-off-z%*%gamma
        F <- G(eta,link);P <- cbind(F,1)-cbind(0,F)
        stopifnot(all(P>0))
        counts <- t(vapply(seq_len(n),function(i) drop(rmultinom(1L,20L,P[i,])),integer(K)))
        colnames(counts) <- paste0('category',seq_len(K))
        data <- data.frame(x,z,off=off);data$counts <- I(counts)
        w <- rep(c(1,2,0),length.out=n)
        expanded <- data[rep(seq_len(n),each=K),,drop=FALSE];expanded$counts <- NULL
        expanded$y <- ordered(rep(colnames(counts),n),levels=colnames(counts))
        expanded$frequency <- as.vector(t(counts))*rep(w,each=K)
        rhs <- paste(c(colnames(x),colnames(z),'offset(off)'),collapse='+')
        list(data=data,weights=w,formula=as.formula(paste('counts~',rhs)),
            nominal=reformulate(colnames(z)),expanded=expanded,
            reference.formula=reformulate(c(colnames(x),'offset(off)'),response='y'),
            prediction=data[,c(colnames(x),colnames(z),'off')])
    })
}
for(link in c('logit','probit','cloglog')) record(paste0('housing-',link),'application',link,function() {
    data <- MASS::housing;expanded <- data;expanded$frequency <- data$Freq
    list(data=data,weights=data$Freq,formula=Sat~Infl+Type+Cont,nominal=~Cont,
        expanded=expanded,reference.formula=Sat~Infl+Type,prediction=data[,c('Infl','Type','Cont')])
})
print(do.call(rbind,records))
if(!all(vapply(records,function(z) isTRUE(z$passed),logical(1)))) stop('Some comparisons failed; inspect all records')
