args<-commandArgs(trailingOnly=TRUE)
modes<-c(build="components-build.R",components="components.R",generate="generate.R",run="run.R",summarize="summarize.R",verify="verify.R")
if(!length(args) || !args[1L]%in%names(modes))
    stop("Modes: build, components, generate, run, summarize, verify; see README.txt for arguments")
status<-system2(file.path(R.home("bin"),"Rscript"),
    c(shQuote(file.path("benchmarks/ordinal-disk-performance",modes[[args[1L]]])),shQuote(args[-1L])))
if(status!=0L) stop("Study command failed with status ",status)
