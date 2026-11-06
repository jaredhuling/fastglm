# Write the dated ordinal section of the benchmark vignette from summary.csv.
# The vignette cannot read benchmarks/, which is excluded from the package build.
# Rscript benchmarks/ordinal-comparison/make-vignette-section.R
here <- "benchmarks/ordinal-comparison"
d <- read.csv(file.path(here, "summary.csv"))
meta <- read.csv(file.path(here, "metadata.csv"))
order_methods <- c("fastglm", "rms::orm", "ordinal::clm", "MASS::polr", "VGAM::vglm", "WeightIt::ordinal_weightit")
cells <- rbind(d[d$panel == "Common slopes" & d$n == 100000L, ], d[d$panel == "Many categories", ])
cells <- cells[order(cells$panel, match(cells$link, c("logit", "probit", "cloglog")), cells$K,
                     match(cells$method, order_methods)), ]
table <- data.frame(Link = cells$link, Observations = format(cells$n, big.mark = ",", trim = TRUE), Categories = cells$K,
    Method = paste0("`", cells$method, "`"), Seconds = signif(cells$median_seconds, 3),
    Ratio = ifelse(cells$agrees, signif(cells$speedup, 3), NA))
print(table, row.names = FALSE)
lines <- c("# Ordinal models", "",
    sprintf("Unlike the historical sections above, these times come from a run on %s (UTC) of *fastglm* commit `%s`, created by `benchmarks/benchmark-ordinal-comparison.R` in the package repository. Each design has 10 predictors, and each time includes coefficient covariance. The ratio is the comparison time divided by the *fastglm* time, shown only when that fit converged and agreed with *fastglm*.",
            substr(meta$value[meta$item == "date_utc"], 1L, 10L), substr(meta$value[meta$item == "commit"], 1L, 7L)),
    "", as.character(knitr::kable(table, col.names = c("Link", "Observations", "Categories", "Method", "Seconds", "Time ratio"))), "",
    "`rms::orm` is designed for outcomes with very many distinct values and is the faster choice at 300 or more categories. `VGAM::vglm` was limited to 30 categories by memory. For cloglog, `VGAM::vglm` warned about intersecting predictors and returned a different fit, so it has no ratio.", "")
writeLines(lines, file.path(here, "vignette-section.md"))
