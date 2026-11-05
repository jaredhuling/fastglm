args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=1L) stop("Supply the archived study directory")
root<-normalizePath(args[1L])
manifest<-read.csv(file.path(root,"artifact-manifest.csv"),stringsAsFactors=FALSE)
stopifnot(!anyDuplicated(manifest$file),all(nzchar(manifest$file)),
    !any(grepl("(^/|(^|/)\\.\\.(/|$))",manifest$file)))
paths<-file.path(root,manifest$file)
stopifnot(all(file.exists(paths)),all(file.info(paths)$size==manifest$bytes),
    identical(unname(tools::md5sum(paths)),manifest$md5))
cat(nrow(manifest),"archived file sizes and hashes verified\n")
