# Generate the scientific assessment from saved results and diagnostic records.
root <- "benchmarks/ordinal-proportional-odds-development/study"
s <- read.csv(file.path(root,"summary.csv"))
selected <- subset(s,method=="meanBR" & scenario%in%c("regular3","regular5","rare_lower3","rare_upper3","rare_interior3"))
percent <- function(x) sprintf("%.1f%%",100*x)
table.rows <- vapply(seq_len(nrow(selected)),function(i) {
    z <- selected[i,]
    paste0("| ",paste(c(z$scenario,z$n,paste0(z$available,"/",z$datasets),percent(z$rejection),
        paste(percent(c(z$rejection.lower,z$rejection.upper)),collapse=" to ")),collapse=" | ")," |")
},character(1))
rare <- subset(s,method=="meanBR" & scenario=="rare_lower3" & n==80)
power <- subset(s,method=="meanBR" & !null)
power.rows <- vapply(seq_len(nrow(power)),function(i) {
    z <- power[i,]
    paste0("| ",paste(c(z$scenario,z$n,percent(z$rejection),
        paste(percent(c(z$rejection.lower,z$rejection.upper)),collapse=" to ")),collapse=" | ")," |")
},character(1))
null1000 <- subset(s,method=="meanBR" & null & n==1000)
checks <- read.csv(file.path(root,"implementation-checks.csv"))
paired <- read.csv(file.path(root,"paired.csv"))
paired.rare <- subset(paired,first=="ML" & second=="meanBR" & scenario=="rare_lower3" & n==80)
writeLines(c("# Assessment of the proportional-odds test candidate","",
"The current chi-square test should remain a development prototype. It has substantial false-positive inflation in sparse endpoint samples and in some small samples with several slope contrasts. Mean bias reduction improves some coefficient behavior but does not consistently calibrate this test. The public graphical diagnostic should remain descriptive.","",
"## Evidence and scope","",
"The frozen development study completed 30,000 datasets in 30 cells, with 90,000 method records. Every dataset is represented by its seed and method outcomes, including unsuccessful results. The three methods were the empirical-score covariance at ML and meanBR binary fits, and the installed car::poTest() comparator. There were 1,000 datasets per scenario/sample-size cell. Results support decisions within this grid; they do not establish general calibration or a minimum safe sample size.","",
"The table below reports the meanBR candidate's false-positive rate at nominal 5%, conditional on an available test. Intervals quantify Monte Carlo uncertainty. The complete comparison, including correlated predictors and five-category interior sparsity, is in RESULTS.md and summary.csv.","",
"| Scenario | n | Available | False-positive rate | 95% MC interval |",
"|---|---:|---:|---:|---|",table.rows,"",
"The endpoint failure appears for both outcome orders. Inflation also remains at the intermediate sample size, where almost all tests are available, so excluding empty-category datasets cannot explain it. The regular five-category result demonstrates a limitation beyond extreme category sparsity. Sparse interior categories produce a different pattern, including conservative behavior for the three-category case. A single inflation factor would not address these distinct behaviors.","",
sprintf("Across the eight null scenarios at n=1000, meanBR rejection rates ranged from %s to %s. These results are encouraging for regular large-sample operation, but this grid cannot justify a universal n cutoff.",
    percent(min(null1000$rejection)),percent(max(null1000$rejection))),"",
"## Diagnosis","",
sprintf("In the rare-lower-endpoint setting at n=80, the empirical standard deviation of the first x1 slope contrast was %.3f, while the square root of the mean estimated variance was %.3f. The corresponding interval coverage was %s. This supports underestimated sampling variability as a component of the failure; it does not isolate all sources of poor approximation.",
    rare$empirical.sd,rare$root.mean.variance,percent(rare$coverage)),"",
sprintf("MeanBR reduced false-positive rejection relative to ML in that cell by %.1f percentage points (paired Monte Carlo standard error %.1f points), on the same %d available datasets. That improvement leaves substantial inflation. In some other cells meanBR increased rejection slightly; coefficient bias reduction does not imply improvement in test calibration.",
    100*paired.rare$difference,100*paired.rare$mcse,paired.rare$shared),"",
sprintf("Post-study implementation checks replayed 30 meanBR datasets and rebuilt covariance from direct individual scores and a numerical score Jacobian. The maximum relative covariance discrepancy was %.2g; statistic replays agreed within numerical precision. Together with the earlier grouped-frequency and outcome-reversal checks, this gives evidence against a score-product or normalization coding error in these cases.",
    max(checks$covariance.relative.error)),"",
"Some rare-endpoint ML fits returned very large coefficients and fitted-probability warnings despite satisfying the optimizer's convergence flag. Such a flag does not establish the existence of a finite ML solution. These records were retained under the frozen eligibility rule. The meanBR failure is also present with finite estimates, so excluding problematic ML fits would not resolve the main finding.","",
"The observed problem is compatible with poor finite-sample covariance and chi-square approximations when only a few observations identify a binary slope. It does not falsify the conditional asymptotic argument under a fixed, regular interior model. No covariance multiplier, category-count threshold, or nominal level was tuned after seeing these results.","",
"## Power and comparator limits","",
"Power increased with sample size under the prespecified nonparallel alternatives:","",
"| Scenario | n | MeanBR rejection probability | 95% MC interval |",
"|---|---:|---:|---|",power.rows,"",
"The elevated null rejection in the small five-category samples prevents interpreting their power as a calibrated advantage. The alternatives studied here use moderate category frequencies; they do not measure sparse-endpoint power.","",
"The conventional car comparator was conservative in the sparse-endpoint cases, and it sometimes returned finite statistics when an endpoint was unobserved. Its interface does not expose all auxiliary convergence statuses. The stored diagnostics also show nonsymmetric contrast covariance matrices in some multipredictor cases for the installed implementation. Its results were preserved unchanged as an external reference, with these limitations recorded; it is not an infallible numerical reference. See comparator-support-diagnostics.csv and car-source.txt.","",
"## Package decision and next experiment","",
"Keep the public proportional-odds diagnostic descriptive. These findings concern the new stacked-score hypothesis-test prototype; they do not evaluate or invalidate the existing ordinal marginal-effects API or the separate endpoint-bootstrap procedure. No production fitting or inference code changed in this study.","",
"The next bounded investigation is parametric calibration under a fitted proportional-odds null. It needs a discrepancy statistic that remains defined when simulated categories are empty, and an explicit rule for genuine refit failures. Simply bootstrapping the present covariance-inverted statistic can produce singular contrasts and unavailable results in exactly the sparse settings of interest. Derive that statistic and its regularity argument first, then compare a frozen version using fresh seeds. A fitted-null bootstrap would still require empirical calibration checks and would not be an exact finite-sample test.","",
"No test has been promoted. Partial proportional-odds models remain a separate later extension.","",
"## Reproduction and provenance","",
"run.R audits the frozen method and scenario sources and the complete replication grid. summarize.R regenerates the tables and figures. diagnose.R performs the post-study numerical audit, and interpret.R generates this assessment from saved summaries. The raw-data hashes are recorded in results-manifest.txt. This assessment is a development finding, not an independent confirmation study."),file.path(root,"ASSESSMENT.md"))
inputs <- file.path(root,c("summary.csv","paired.csv","implementation-checks.csv","comparator-support-diagnostics.csv",
                         "diagnose.R","interpret.R","ASSESSMENT.md","results-manifest.txt"))
write.csv(data.frame(file=basename(inputs),md5=unname(tools::md5sum(inputs))),
          file.path(root,"assessment-hashes.csv"),row.names=FALSE)
