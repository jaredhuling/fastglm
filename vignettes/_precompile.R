# Regenerate static benchmark vignettes from their executable sources.
# Run with Rscript vignettes/_precompile.R [vignette-name ...].
args <- commandArgs(trailingOnly=TRUE)
inputs <- list.files("vignettes",pattern="\\.Rmd\\.orig$",full.names=TRUE)
if(length(args)) inputs <- inputs[sub("\\.Rmd\\.orig$","",basename(inputs)) %in% args]
if(!length(inputs)) stop("No matching benchmark sources")
for(input in inputs) {
    old <- setwd("vignettes")
    tryCatch(knitr::knit(basename(input),output=sub("\\.orig$","",basename(input)),
                        envir=new.env(parent=globalenv()),quiet=FALSE),
             finally=setwd(old))
}
