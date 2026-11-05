args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=1L || dir.exists(args[1L])) stop("Supply a new input directory")
source("benchmarks/ordinal-disk-performance/config.R")
config<-disk_configuration();dir.create(args[1L],recursive=TRUE);root<-normalizePath(args[1L])
datasets<-list()
for(i in seq_len(nrow(config$cases))) {
    case<-config$cases[i,];links<-if(case$id=="large") "logit" else config$links
    for(link in links) {
        id<-paste(case$id,link,sep="-");dir.create(file.path(root,id))
        chunks<-ceiling(case$n/case$block);files<-character(chunks)
        for(k in seq_len(chunks)) {
            first<-(k-1L)*case$block+1L;block.rows<-min(case$block,case$n-first+1L)
            z<-disk_fixture(block.rows,case$K,case$r,link,config$seed+i*10000L+k,first)
            z$theta<-NULL
            files[k]<-file.path(id,sprintf("block-%04d.rds",k))
            saveRDS(z,file.path(root,files[k]),compress=FALSE,version=2)
        }
        datasets[[id]]<-list(case=case,link=link,files=files,
            bytes=unname(file.info(file.path(root,files))$size),
            md5=unname(tools::md5sum(file.path(root,files))))
        message("Generated ",id," (",case$n," rows; ",chunks," files)")
    }
}
saveRDS(list(configuration=config,datasets=datasets,session=sessionInfo()),file.path(root,"inputs.rds"))
