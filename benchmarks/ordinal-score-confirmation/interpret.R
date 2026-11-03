root <- "benchmarks/ordinal-score-confirmation"
rates <- read.csv(file.path(root,"rates.csv"))
precision <- read.csv(file.path(root,"precision/rates.csv"))
failures <- read.csv(file.path(root,"failures.csv"))
timing <- read.csv(file.path(root,"timing-summary.csv"))
labels <- c(balanced4="Balanced, four categories",endpoint4="Rare endpoint",
    narrow_middle4="Narrow interior category",correlated6="Correlated predictors, six categories",
    sparse_binary4="Uncommon binary predictor",leverage4="High leverage",
    grouped4="Grouped frequencies",alternative_balanced4="Nonproportional, balanced",
    alternative_endpoint4="Nonproportional, rare endpoint")
percent <- function(x) sprintf("%.2f%%",100*x)
entry <- function(z) paste0(percent(z$rate)," (",percent(z$lower)," to ",percent(z$upper),")")
table <- function(null) {
    z <- rates[rates$null==null & rates$level==.05,]
    cells <- unique(z$cell)
    vapply(cells,function(i) {
        g <- z[z$cell==i,]; a <- g[g$test=="joint",]; b <- g[g$test=="x1",]
        paste0("| ",labels[a$scenario]," | ",a$n," | ",a$available,"/",a$total," | ",entry(a)," | ",entry(b)," |")
    },character(1))
}
lines <- c("# Confirmation assessment: analytic proportional-odds score inference", "",
    "The results support integrating the expected-information score test as an",
    "optional part of ordinal assessment. It supplies deterministic joint and",
    "selected-term inference with grouped frequencies and offsets. The default",
    "assessment stays descriptive. The inferential claim remains a regular",
    "first-order chi-square approximation, including when the original fitted",
    "model used mean bias reduction. A usable ML null remains necessary.", "",
    "## Independent confirmation", "",
    "The frozen implementation was evaluated on 27,000 fresh datasets across",
    "nine scenarios and sample sizes 100, 400, and 1600. Each dataset supplies",
    "joint and selected-first-predictor tests, giving 54,000 records. The",
    "parameter values and seed blocks differ from development. The method,",
    "generator, protocol, source snapshot, and installed library were frozen",
    "before generating confirmation outcomes. This evaluates the public package",
    "function, rather than a harness-local substitute.", "",
    "At nominal 5%, null rejection rates among available tests ranged from 3.6%",
    "to 7.0%. The 7.0% joint result for correlated predictors and the 6.6%",
    "selected-predictor result for a narrow interior category occurred at n=1600.",
    "Their pointwise Monte Carlo intervals excluded 5%; they were therefore",
    "examined in a separate precision study below. The original results remain",
    "unchanged. Occasional pointwise departures across many comparisons do not",
    "by themselves establish a persistent size distortion.", "",
    "The table reports available datasets and rejection rates with 95% Wilson",
    "Monte Carlo intervals. These intervals concern simulation precision, not",
    "confidence intervals returned to an applied analyst. Joint and selected",
    "tests had the same availability in this grid.", "",
    "| Null scenario | Individual frequency | Available | Joint test | First predictor |",
    "| --- | ---: | ---: | --- | --- |",table(TRUE),"",
    "At nominal 1%, null rejection rates ranged from 0.10% to 1.72%. The",
    "0.10% selected-predictor rate occurred in the balanced n=100 sample; its",
    "Wilson interval was below 1%, indicating conservatism in that cell. The",
    "1.72% rate occurred for the rare endpoint at n=100. The complete 1% results",
    "and their Monte Carlo intervals are retained in `rates.csv`. The evidence",
    "does not justify an exact-tail-accuracy claim, especially for small samples.", "",
    "## Precision follow-up", "",
    "The two elevated n=1600 comparisons motivated 5000 new replications in",
    "each corresponding scenario, with the same frozen method and generator.",
    "This is a post-confirmation diagnostic selected after seeing the first",
    "results, not a replacement confirmation grid. No method choice changed.", "",
    "| Scenario | Test | Available | Rejection at 5% (95% MC interval) |",
    "| --- | --- | ---: | --- |")
for(i in which(precision$level==.05)) {
    z <- precision[i,]
    lines <- c(lines,paste0("| ",labels[z$scenario]," | ",z$test," | ",z$available,"/",z$total," | ",entry(z)," |"))
}
lines <- c(lines,"",
    "The previously elevated correlated-predictor joint rate was 4.82%",
    "(95% MC interval 4.26% to 5.45%), and the narrow-interior selected rate was",
    "5.08% (4.51% to 5.72%). All 10,000 new datasets yielded both tests. The",
    "four 1% rates were 0.86% to 1.12%. This follow-up is consistent with Monte",
    "Carlo variation in the two original elevated rates; it provides no evidence",
    "of a persistent distortion of their original magnitude. Modest finite-sample",
    "distortion elsewhere remains possible and is not a reason to add an empirical",
    "correction or tune the reference distribution.", "",
    "## Availability and power", "",
    "There were 304 unavailable test records on 152 original datasets. At n=100,",
    "the uncommon binary predictor produced 20 deficient designs and 96 failed",
    "ML null fits out of 1000 datasets. Its tests were available on 884 datasets.",
    "Conditional 5% rejection was 4.98% jointly and 4.64% for the first predictor.",
    "The joint rejection bounds over all datasets were 4.4% to 16.0% when",
    "unavailable decisions are counted as all nonrejections or all rejections.",
    "These are missing-decision bounds, not proposed operating rules. Conditional",
    "size alone would conceal this practical limitation. At n=400, one ML null",
    "fit failed in the same scenario; availability was complete at n=1600.", "",
    "The remaining failures were empty categories: 13 rare-endpoint null samples,",
    "three narrow-interior null samples, and 19 rare-endpoint alternative samples,",
    "all at n=100. No category was removed to manufacture an available test.",
    "No other failure reason occurred in the confirmation grid. All failures",
    "retain their original warnings, details, category totals, seeds, and records.", "",
    "The alternatives change only the first predictor's slopes. Testing that",
    "predictor alone avoids spending degrees of freedom on unchanged slopes and",
    "has higher power in these examples. That is a consequence of the specified",
    "alternative, not evidence that term selection after inspecting the data",
    "would retain the reported error rate.", "",
    "| Alternative | Individual frequency | Available | Joint power | First-predictor power |",
    "| --- | ---: | ---: | --- | --- |",table(FALSE),"",
    "## Numerical validation and computation", "",
    "The 40 targeted test assertions passed. They cover numerical full-likelihood",
    "derivatives and expected information, grouped/expanded equivalence, offsets,",
    "frequency doubling, outcome reversal, predictor scaling, factor and interaction",
    "terms, prepared-design serialization, newdata, row exclusions, and explicit",
    "unsupported cases. Twenty additional global production/prototype comparisons",
    "agreed within 8.9e-15. The complete package test suite passed, with only the",
    "optional Arrow test skipped because Arrow was unavailable.", "",
    "The local source-package check compiled and installed successfully. It found",
    "one existing Rd usage markup note, which was corrected. Rechecking the updated",
    "documentation against the frozen installation returned Status: OK. Tests and",
    "vignettes were excluded from those check invocations because they were run",
    "separately; the updated ordinal vignette rendered successfully to HTML. This",
    "is local validation, not a Windows or release check.", "",
    "The following local medians use five paired repetitions with three predictors",
    "and five categories. The score column times the entire public assessment,",
    "including prediction scores, calibration summaries, the ML null refit, and",
    "score accumulation. Initial model fitting is excluded. The incremental column",
    "is the median paired difference and can differ from the difference of medians.",
    "Timings were collected alongside other local work and are not isolated speed",
    "benchmarks. The implementation does not expand grouped frequencies.", "",
    "| Rows | Descriptive assessment (seconds) | With score test (seconds) | Increment (seconds) |",
    "| ---: | ---: | ---: | ---: |")
for(i in seq_len(nrow(timing))) {
    z <- timing[i,]
    lines <- c(lines,sprintf("| %d | %.3f | %.3f | %.3f |",z$n,z$plain.seconds,z$score.seconds,z$incremental.seconds))
}
lines <- c(lines,"","## Decision and remaining scope", "",
    "Keep the optional analytic score test and its explicit availability diagnostics.",
    "The former inflated stacked-binary Wald tests remain development comparators",
    "and are not exposed as formal inference. The earlier paired development",
    "experiments explain why expected information was selected; the independent",
    "confirmation provides new evidence on the frozen production implementation.", "",
    "The main remaining limitation is inference when an ML null is unavailable",
    "despite finite mean-bias-reduced estimates. A principled analytic test based",
    "on adjusted scores would need its own nuisance treatment and distributional",
    "justification. Simply substituting a bias-reduced null or an empirical",
    "variance multiplier would not supply that justification. It is a separate",
    "methods project, not a prerequisite for this explicitly scoped optional test.", "",
    "Future package work can build on the established ordinal interface and",
    "diagnostics. Partial proportional-odds fitting is a natural extension if its",
    "parameterization and ordered-probability constraints are addressed explicitly.",
    "The present score test does not fit that alternative model.")
writeLines(lines,file.path(root,"ASSESSMENT.md"))
