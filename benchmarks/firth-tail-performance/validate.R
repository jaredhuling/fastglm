# Validate recorded results without rebuilding or timing fitted models.
# Rscript benchmarks/firth-tail-performance/validate.R RESULTS_DIRECTORY
args<-commandArgs(trailingOnly=TRUE);stopifnot(length(args)==1L);root<-args[1]
for(label in c('baseline','candidate')) {
 d<-read.csv(file.path(root,label,'timing.csv'))
 key<-do.call(paste,c(d[c('case','link','method','operation','repetition')],sep=':'))
 stopifnot(nrow(d)==484L,!anyDuplicated(key),all(d$converged),all(is.finite(d$nanoseconds)),all(d$nanoseconds>0))
 s<-readRDS(file.path(root,label,'solutions.rds'));stopifnot(length(s)==22L)
}
a<-read.csv(file.path(root,'agreement.csv'))
stopifnot(nrow(a)==22L,all(a$iterations_identical),all(a[,2:10]<1e-9))
s<-read.csv(file.path(root,'final-stress.csv'))
stopifnot(nrow(s)==146L,all(s$converged),all(s$exact_score<1e-6),all(s$covariance_relative_error<1e-10))
values<-vapply(s[c('exact_score','covariance_relative_error','reported_score_error')],max,numeric(1))
writeLines(c(paste('Fits:',nrow(s)),paste('Converged:',sum(s$converged)),
 paste(names(values),format(values,digits=16),sep=': ')),file.path(root,'stress-summary.txt'))
if(file.exists(file.path(root,'archive.csv'))) {
 archive<-read.csv(file.path(root,'archive.csv'));stopifnot(nrow(archive)==2L,all(archive$identical_object))
 for(i in seq_len(nrow(archive))) stopifnot(unname(tools::md5sum(file.path(root,archive$label[i],'solutions.rds')))==archive$compressed_md5[i])
}
cat('Complete paired timing grid, numerical agreement, 146 stress fits, and solution archive verified.\n')
