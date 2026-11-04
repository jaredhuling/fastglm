root <- "benchmarks/ordinal-streaming/results"
records <- list();differences <- list()
for(n in c(5000L,200000L)) {
    for(method in c("dense","source")) {
        prefix <- file.path(root,paste(n,method,sep="-"))
        d <- read.csv(paste0(prefix,".csv"));log <- readLines(paste0(prefix,".log"))
        rss <- log[grepl("maximum resident set size",log,fixed=TRUE)]
        d$peak_process_bytes <- if(length(rss)) as.numeric(strsplit(trimws(rss)," +")[[1L]][1L]) else NA_real_
        records[[paste(n,method)]] <- d
    }
    a <- readRDS(file.path(root,paste0(n,"-dense-solutions.rds")))
    b <- readRDS(file.path(root,paste0(n,"-source-solutions.rds")))
    for(estimation in names(a)) {
        u <- a[[estimation]];v <- b[[estimation]]
        differences[[paste(n,estimation)]] <- data.frame(n,estimation,
            coefficient_error=max(abs(u$coefficients-v$coefficients)),
            covariance_error=max(abs(u$covariance-v$covariance)),
            score_error=max(abs(u$score-v$score)),loglik_error=abs(u$loglik-v$loglik),
            deviance_error=abs(u$deviance-v$deviance),prediction_error=max(abs(u$prediction$fit-v$prediction$fit)),
            prediction_se_error=max(abs(u$prediction$se.fit-v$prediction$se.fit)))
    }
}
write.csv(do.call(rbind,records),file.path(root,"comparison.csv"),row.names=FALSE)
write.csv(do.call(rbind,differences),file.path(root,"agreement.csv"),row.names=FALSE)
print(do.call(rbind,records));print(do.call(rbind,differences))

# The macOS memory reruns preserve both process metrics. Their elapsed times
# were collected alongside other checks and are not speed-comparison evidence.
memory.root <- file.path(root,"peak-memory")
memory <- list();large.agreement <- list();heap <- list()
for(n in c(5000L,200000L,1000000L)) {
    for(method in c("dense","source")) {
        prefix <- file.path(memory.root,paste(n,method,sep="-"))
        if(!file.exists(paste0(prefix,".log"))) next
        log <- readLines(paste0(prefix,".log"))
        extract <- function(label) {
            line <- log[grepl(label,log,fixed=TRUE)]
            if(length(line)) as.numeric(strsplit(trimws(line)," +")[[1L]][1L]) else NA_real_
        }
        memory[[paste(n,method)]] <- data.frame(n,method,
            maximum_resident_bytes=extract("maximum resident set size"),
            peak_footprint_bytes=extract("peak memory footprint"))
        for(estimation in c("ML","meanBR")) {
            path <- paste0(prefix,"-",estimation,"-heap.csv")
            if(!file.exists(path)) next
            h <- read.csv(path)
            # The first 1M run labeled cell counts as maximum_used_mb.
            # Preserve those raw records; use the known 8-byte Vcell unit here.
            cells <- if("maximum_used_cells" %in% names(h)) h$maximum_used_cells else h$maximum_used_mb
            heap[[paste(n,method,estimation)]] <- data.frame(n,method,estimation,
                live_before_mb=sum(h$used_before_mb),live_after_mb=sum(h$used_after_mb),
                maximum_vector_heap_mb=cells[h$kind=="Vcells"]*8/2^20)
        }
    }
    if(n!=1000000L) next
    a <- readRDS(file.path(memory.root,paste0(n,"-dense-solutions.rds")))
    b <- readRDS(file.path(memory.root,paste0(n,"-source-solutions.rds")))
    for(estimation in names(a)) {
        u <- a[[estimation]];v <- b[[estimation]]
        large.agreement[[estimation]] <- data.frame(n,estimation,
            coefficient_error=max(abs(u$coefficients-v$coefficients)),
            covariance_error=max(abs(u$covariance-v$covariance)),score_error=max(abs(u$score-v$score)),
            loglik_error=abs(u$loglik-v$loglik),deviance_error=abs(u$deviance-v$deviance),
            prediction_error=max(abs(u$prediction$fit-v$prediction$fit)),
            prediction_se_error=max(abs(u$prediction$se.fit-v$prediction$se.fit)))
    }
}
if(length(memory)) write.csv(do.call(rbind,memory),file.path(root,"memory.csv"),row.names=FALSE)
if(length(heap)) write.csv(do.call(rbind,heap),file.path(root,"heap.csv"),row.names=FALSE)
if(length(large.agreement)) write.csv(do.call(rbind,large.agreement),file.path(root,"agreement-million.csv"),row.names=FALSE)
