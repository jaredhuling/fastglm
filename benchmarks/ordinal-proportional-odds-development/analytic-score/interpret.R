root <- "benchmarks/ordinal-proportional-odds-development/analytic-score"
s <- read.csv(file.path(root,"summary.csv")); stress <- read.csv(file.path(root,"stress-summary.csv"))
info <- read.csv(file.path(root,"information-comparison.csv")); published <- read.csv(file.path(root,"published-example-check.csv"))
timing <- read.csv(file.path(root,"benchmark.csv")); paired <- read.csv(file.path(root,"paired.csv"))
pct <- function(x) sprintf("%.1f%%",100*x)
get <- function(scenario,n,method="score_ML",level=.05,tab=s) {
    z <- tab[tab$scenario==scenario & tab$n==n & tab$method==method & tab$level==level,]
    stopifnot(nrow(z)==1L); z
}
interval <- function(z) sprintf("%s (%s–%s)",pct(z$rate),pct(max(0,z$mc.lower)),pct(min(1,z$mc.upper)))
table <- function(tab,scenarios,n) vapply(scenarios,function(scenario) {
    z <- lapply(c("binary_meanBR_Wald","score_ML","score_meanBR"),function(m) get(scenario,n,m,tab=tab))
    sprintf("| %s | %s | %s | %s | %s |",scenario,interval(z[[1]]),interval(z[[2]]),interval(z[[3]]),
        paste(vapply(z,function(a) paste0(a$available,"/",a$attempted),character(1)),collapse="; "))
},character(1))
lines <- c("# Analytic inference for the proportional-odds diagnostic", "",
    "The expected-information score test is the most promising next candidate. It substantially improves false-positive control in the development comparisons, requires one null-model fit, and needs no bootstrap or simulation when analyzing a dataset. A mean-bias-reduced null fit is not uniformly better for this test. Production remains unchanged pending a separate confirmation study and a complete public-interface specification.","",
    "## What changed the diagnosis", "",
    "A score test asks whether the fitted proportional-odds model leaves evidence of threshold-specific slope changes. It evaluates that evidence at the common-slope null model, avoiding the unstable separate binary slope estimates used by the earlier Wald prototype. Its uncertainty is computed from the ordinal likelihood.","",
    "The information matrix is a second consequential choice. Expected information averages likelihood curvature over possible outcomes under the fitted model. Observed information uses the outcomes actually seen. These are both established score-test constructions under regular asymptotics; they need not behave alike in small samples. Our candidate used expected information before any rejection results were inspected.","",
    "A post-study replay held each ML null fit and its ordinary score fixed, changing only the information matrix. Results at nominal 5%, n=80, are below. All comparisons use shared available datasets; they are targeted development diagnoses, not independent confirmation.","",
    "| Scenario | Expected information | Observed information | Shared datasets |","|---|---:|---:|---:|")
for(scenario in unique(info$scenario)) {
    z <- info[info$scenario==scenario & info$level==.05,]
    e <- z[z$method=="expected",]; o <- z[z$method=="observed",]
    lines <- c(lines,sprintf("| %s | %s | %s | %d |",scenario,interval(e),interval(o),e$shared))
}
lines <- c(lines,"",sprintf("An external numerical check used the published SAS Cheese example. Our expected-information statistic is %.6f. An observed-Hessian generalized-inverse calculation gives %.6f, compared with the published %.4f. The observed matrix has rank %d of %d; the expected matrix is nonsingular. This supports the information-choice explanation for that example, without claiming access to SAS internals. The generalized inverse was used only for this external diagnosis, never for a candidate test. INFORMATION-AUDIT.md records the derivation and source locations.",published$expected.statistic,published$observed.generalized.statistic,published$published.SAS.statistic,published$observed.rank,published$expected.rank),"",
    "## Main development comparison", "",
    "The main frozen study completed 30,000 datasets, comparing three methods on identical inputs. There are 1,000 datasets in each of ten scenarios at n=80,250,1000. The ordinary and meanBR-root score calculations use the same expected information and nuisance projection. All methods decline the regular full-dimensional test when an original category is empty. Intervals below are 95% Wilson Monte Carlo intervals; rates condition on availability.","",
    "The n=80 null cases show the mechanism clearly. Availability entries are ordered Wald; score at ML; score at meanBR.","",
    "| Scenario | Earlier binary meanBR Wald | Expected score at ML | Expected score at meanBR | Available / attempted |",
    "|---|---:|---:|---:|---|",table(s,c("regular3","regular5","correlated5","rare_lower3","rare_upper3","rare_interior3","rare_interior5","strong_signal3"),80),"",
    "The score approach improves the sparse-interior case as well as the endpoint cases: it avoids the earlier test's strong conservatism when an interior category is rare. At nominal 1%, some score comparisons remain conservative. MeanBR-root scoring is particularly conservative for rare endpoints, so coefficient bias reduction does not supply an automatic inferential improvement.","",
    "The meanBR-root calculation explicitly projects out the ordinary nuisance score, which is nonzero at a meanBR estimate. Omitting this projection adds a nonnegative nuisance-estimation component that is not evidence of nonparallel slopes. This distinction is algebraically verified. The meanBR-root candidate has a first-order argument only; it is not labeled a higher-order Firth test.","",
    "## Power", "",
    "The alternatives are the two coherent nonparallel models fixed in the earlier study. These power comparisons are developmental. In particular, the earlier Wald test's higher rejection in the small five-category alternative comes with inflated null rejection and cannot be interpreted as a calibrated power advantage.","",
    "| Alternative and sample size | Earlier Wald | Expected score at ML | Expected score at meanBR |","|---|---:|---:|---:|")
for(scenario in c("alternative3","alternative5")) for(n in c(80,250,1000)) {
    z <- vapply(c("binary_meanBR_Wald","score_ML","score_meanBR"),function(method) pct(get(scenario,n,method)$rate),character(1))
    lines <- c(lines,sprintf("| %s, n=%d | %s | %s | %s |",scenario,n,z[1],z[2],z[3]))
}
lines <- c(lines,"","## Separate stress comparison", "",
    "After the main results, a fixed additional comparison added 8,000 datasets with normal predictors, eight predictors, uncommon binary predictor levels, and occasional high-leverage observations. Its protocol was recorded before generating these data. No method was changed, and no correction or eligibility cutoff was tuned. The n=80 results are:","",
    "| Scenario | Earlier binary meanBR Wald | Expected score at ML | Expected score at meanBR | Available / attempted |",
    "|---|---:|---:|---:|---|",table(stress,unique(stress$scenario),80),"",
    "At n=250 the expected-information score rates remained near or below nominal in this panel; the complete tables include both nominal levels. These successes are evidence for further work, not a finite-sample theorem or proof of uniform calibration. The uncommon-binary ML score is available on only 930/1000 n=80 datasets: 22 have deficient predictor rank and 48 fail null convergence. The meanBR fit improves availability there, but fails convergence in 71/1000 high-leverage n=80 datasets. No silent fallback between methods was used.","",
    "These convergence failures need separate investigation. Availability and unconditional rejection bounds are recorded alongside conditional rates. A favorable rate on successful fits cannot establish reliability on the missing cases. The expected-information construction should therefore receive a clear failure result when its required fit is unavailable.","",
    "## Computational evidence", "",
    "The implementation accumulates category scores and information in blocks. It stores no observation-by-observation matrix and fits no unrestricted ordinal alternative. The representative timing exercise below used three predictors and five categories; entries are medians of three repetitions on one dataset per size. These are local prototype timings, not a cross-platform performance guarantee.","",
    "| n | Null estimate | Fit time (s) | Score calculation (s) | Total (s) |","|---:|---|---:|---:|---:|")
for(n in unique(timing$n)) for(firth in c(FALSE,TRUE)) {
    z <- timing[timing$n==n & timing$firth==firth,]
    lines <- c(lines,sprintf("| %d | %s | %.3f | %.3f | %.3f |",n,if(firth) "meanBR" else "ML",
        median(z$fit.seconds),median(z$score.seconds),median(z$total.seconds)))
}
lines <- c(lines,"", "## Decision and remaining work", "",
    "Continue with the expected-information score at an ML null fit as the primary analytic candidate. It is the simplest supported route in these results. Preserve the meanBR-root variant as a labeled development comparison; do not automatically substitute it when ML fails. No new Bartlett factor, adjusted degrees of freedom, or simulation-calibrated threshold is justified by this study.","",
    "Before public integration, freeze an independent confirmation panel that includes sparse-alternative power and fit failures, extend and validate grouped-frequency/offset/formula-term handling, and specify which original fits support testing. An assessment of a meanBR-fitted model must state explicitly whether it refits the ML null for the test. Empty categories and deficient design rank must have explicit outcomes. If integrated, document the test naturally in the ordinal vignette, including its distinction from the descriptive binary-slope plots.","",
    "The exact score and information identities are checked; asymptotic chi-square validity is conditional on the regularity stated in METHODS.md. Finite-sample uniform validity, behavior outside the studied designs, and higher-order accuracy remain open. The score method and information choices are established ingredients; this work contributes implementation and development evidence, without claiming a new statistical test.","",
    "The earlier simulation-calibration proposal remains archived as a diagnostic alternative. It is not the active direction after the user's preference for analytic methods. No public package function, version, vignette or inference default changed in this pass.","",
    "## Audit and reproduction", "",
    "The evidence comprises 38,000 distinct development datasets, 114,000 main/stress method records, and 4,000 post-study replay records. Frozen source/install manifests, complete-grid audits, analytic/numerical identities, outcome reversal, affine predictor invariance, block-size checks and the external-example diagnosis are retained. README.md gives the dispatcher commands. interpret.R generates this assessment directly from saved summaries; provenance.csv records source and result hashes. Rebuilding in a relocated directory is checked separately.")
writeLines(lines,file.path(root,"ASSESSMENT.md"))
# Paired information comparison: one figure, unchanged method definitions.
draw <- function() {
    par(mar=c(5,12,4,1)); scenarios <- unique(info$scenario); yy <- rev(seq_along(scenarios))
    plot(NA,xlim=c(0,45),ylim=c(.5,4.5),axes=FALSE,xlab="False-positive rate (%)",ylab="",
        main="Expected versus observed information")
    axis(1); axis(2,at=yy,labels=c("5 categories; 3 predictors","Rare endpoint","5 categories; 8 predictors","High leverage"),las=1,cex.axis=.9)
    abline(v=5,lty=2,col="gray50")
    for(j in seq_along(scenarios)) for(method in c("expected","observed")) {
        z <- info[info$scenario==scenarios[j] & info$method==method & info$level==.05,]
        y <- yy[j]+if(method=="expected") .12 else -.12; color <- if(method=="expected") "#176b86" else "#ad493a"
        segments(100*z$mc.lower,y,100*z$mc.upper,y,col=color)
        points(100*z$rate,y,col=color,pch=19)
    }
    legend("bottomright",legend=c("Expected information","Observed information"),pch=19,col=c("#176b86","#ad493a"),bty="n")
    mtext("Nominal 5%; n = 80; paired available datasets; 95% Monte Carlo intervals.",side=1,line=3.7,cex=.85); box()
}
pdf(file.path(root,"information-comparison.pdf"),width=10,height=5.5); draw(); dev.off()
png(file.path(root,"information-comparison.png"),width=1500,height=825,res=150); draw(); dev.off()
