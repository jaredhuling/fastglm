# Serial orchestration; do not run alongside other computational work.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
candidate <- normalizePath(args[1L]); baseline <- normalizePath(args[2L]); out <- args[3L]
if (dir.exists(out)) stop("Use a fresh output directory")
dir.create(out, recursive = TRUE)
Sys.setenv(OMP_NUM_THREADS = 1, OPENBLAS_NUM_THREADS = 1, MKL_NUM_THREADS = 1,
           VECLIB_MAXIMUM_THREADS = 1)
deadline <- as.numeric(Sys.time()) + 300
jobs <- list(c("partial", candidate, "candidate-partial"),
             c("common", baseline, "baseline-common"),
             c("common", candidate, "candidate-common"))
status <- list()
for (job in jobs) {
    target <- file.path(out, job[3L]); dir.create(target)
    code <- system2(file.path(R.home("bin"), "Rscript"),
        args = c("benchmarks/ordinal-partial-performance/worker.R", shQuote(job[1L]),
                 shQuote(job[2L]), shQuote(target), format(deadline, scientific = FALSE, digits = 16)),
        stdout = file.path(target, "console.txt"), stderr = file.path(target, "console.txt"))
    status[[length(status) + 1L]] <- data.frame(worker = job[3L], exit.status = code)
    write.csv(do.call(rbind, status), file.path(out, "worker-status.csv"), row.names = FALSE)
    cat(job[3L], "finished with status", code, "\n"); flush.console()
}
source("benchmarks/ordinal-partial-performance/summarize.R")
summarize_partial_performance(out)
