# Lossless recompression of complete numerical records after timing finishes.
# Rscript benchmarks/firth-tail-performance/archive.R RESULTS_DIRECTORY BASELINE_LABEL CANDIDATE_LABEL
args<-commandArgs(trailingOnly=TRUE);stopifnot(length(args)==3L)
record<-file.path(args[1],'archive.csv');stopifnot(!file.exists(record))
records<-list()
for(label in args[2:3]) {
 path<-file.path(args[1],label,'solutions.rds');stopifnot(file.exists(path))
 before<-readRDS(path);original.size<-file.info(path)$size;original.md5<-unname(tools::md5sum(path))
 temporary<-paste0(path,'.xz');stopifnot(!file.exists(temporary))
 saveRDS(before,temporary,compress='xz')
 stopifnot(identical(before,readRDS(temporary)))
 # The numerical object has been verified before replacing its byte encoding.
 stopifnot(file.copy(temporary,path,overwrite=TRUE),identical(before,readRDS(path)))
 unlink(temporary)
 records[[length(records)+1L]]<-data.frame(label,original_bytes=original.size,
  compressed_bytes=file.info(path)$size,original_md5=original.md5,
  compressed_md5=unname(tools::md5sum(path)),identical_object=TRUE)
 write.csv(do.call(rbind,records),record,row.names=FALSE)
}
print(do.call(rbind,records))
