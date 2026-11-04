# Pointwise equation checks before enabling partial adjusted-score inference.
args <- commandArgs(trailingOnly = TRUE); stopifnot(length(args) == 2L)
.libPaths(c(normalizePath(args[1L]), .libPaths())); library(fastglm)
out <- args[2L]; if (dir.exists(out)) stop("Use a fresh output directory")
dir.create(out, recursive = TRUE)
source("benchmarks/partial-proportional-odds-validation/reference.R")
i <- seq_len(12L)
x <- cbind(sin(i) / 2, cos(i * 1.7) / 2)
z <- cbind((i %% 4L - 1.5) / 2)
counts <- cbind(2 + i %% 3L, 3 + i %% 4L, 2 + i %% 5L, 4 + i %% 2L)
w <- rep(c(1, 2, 0), 4L); off <- sin(i / 2) / 10
theta <- c(-1, .1, 1.2, .25, -.2, .15, .15, .15)
L <- matrix(0, 2L, 8L); L[, 6L] <- -1; L[1L, 7L] <- L[2L, 8L] <- 1
B <- qr.Q(qr(t(L)), complete = TRUE)[, -(1:2), drop = FALSE]
checks <- list()
for (link in 0:2) {
    native <- fastglm:::ordinal_partial_native_evaluate(x, z, integer(), counts,
        w, off, 4L, theta, TRUE, link = link)
    reference <- partial_cumulant_reference(theta, x, z, counts, w, off,
        c("logit", "probit", "cloglog")[link + 1L])
    statistic <- function(state) {
        I <- state$information; U <- drop(state$score)
        M <- diag(8L) - I %*% B %*% solve(crossprod(B, I %*% B), t(B))
        e <- drop(M %*% U)
        c(value = drop(crossprod(e, solve(I, e))),
          annihilation = max(abs(M %*% I %*% B)),
          covariance = max(abs(M %*% I %*% t(M) - M %*% I)))
    }
    a <- statistic(native); b <- statistic(reference)
    for (name in c("ordinary.score", "information", "adjustment", "score"))
        checks[[length(checks) + 1L]] <- data.frame(link = link, check = name,
            error = max(abs(native[[name]] - reference[[name]])), tolerance = 2e-5)
    checks[[length(checks) + 1L]] <- data.frame(link = link, check = "efficient quadratic",
        error = abs(a[1L] - b[1L]), tolerance = 2e-5)
    for (name in c("annihilation", "covariance"))
        checks[[length(checks) + 1L]] <- data.frame(link = link, check = name,
            error = a[name], tolerance = 1e-10)
}
result <- do.call(rbind, checks); result$passed <- result$error < result$tolerance
write.csv(result, file.path(out, "checks.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "session.txt"))
print(result); stopifnot(all(result$passed))
