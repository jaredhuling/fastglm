# Narrative synthesis after the frozen mechanism and reference-precision studies.
root <- "benchmarks/ordinal-proportional-odds-development/mechanisms"
s <- read.csv(file.path(root,"summary.csv")); m <- read.csv(file.path(root,"moments.csv"))
ref <- read.csv(file.path(root,"reference-check.csv")); counts <- read.csv(file.path(root,"endpoint-count-diagnostics.csv"))
p <- read.csv(file.path(root,"paired.csv")); failures <- read.csv(file.path(root,"failure-counts.csv"))
diagnostics <- read.csv(file.path(root,"fit-diagnostics.csv"))
pct <- function(x) sprintf("%.1f%%",100*x)
rate <- function(cell,method,level=.05,population="observed") {
    z <- s[s$cell==cell & s$method==method & s$level==level & s$population==population,]
    stopifnot(nrow(z)==1L); z
}
form <- function(z) sprintf("%s (%s–%s)",pct(z$rate),pct(z$mc.lower),pct(z$mc.upper))
mm <- function(cell,method) m[m$cell==cell & m$method==method & m$population=="observed",]
methods <- c("empirical_ordinary","empirical_adjusted","independent_mean_empirical","independent_sampling_covariance","independent_sampling_centered")
labels <- c("Original empirical covariance","Exact adjusted-score derivative","Independent mean empirical covariance (oracle)",
    "Independent sampling covariance (oracle)","Independent sampling covariance and mean (oracle)")
table.rows <- vapply(seq_along(methods),function(j) paste0("| ",labels[j]," | ",
    paste(vapply(c(1L,3L,6L,8L),function(i) pct(rate(i,methods[j])$rate),character(1)),collapse=" | ")," |"),character(1))
endpoint <- mm(3,"empirical_ordinary"); adjusted <- mm(3,"empirical_adjusted")
regular <- mm(8,"empirical_ordinary")
rare.paired <- p[p$cell==3 & p$population=="observed" & p$method=="empirical_adjusted" & p$level==.05,]
lines <- c("# What causes the inflated proportional-odds rejection rates?", "",
    "The evidence supports a combination of inaccurate finite-sample covariance, unstable covariance inversion, and a poor chi-square tail approximation. Correcting the derivative of the Firth-adjusted score helps the rare-endpoint case but does not solve it, and does not improve the small five-category joint test. Mean bias reduction of coefficients is not sufficient to calibrate their Wald test.","",
    "## Design and scope", "",
    sprintf("The frozen main experiment contains %s response datasets over nine fixed predictor designs. Each design has independent banks for moments (1,000), reference calibration (1,000), and evaluation (2,000). All %s binary-fit datasets completed; the largest adjusted-score residual was %.2g. Analytic derivatives agree with numerical differentiation, the original covariance is reproduced, and selected extreme and empty-category fits replay correctly.",
        format(sum(failures$attempted),big.mark=","),format(sum(failures$attempted)-sum(failures$binary.failed),big.mark=","),max(diagnostics$score.max)),"",
    "The scenarios reproduce the earlier study's model parameters while holding predictors fixed. They isolate conditional sampling behavior, rather than averaging over changing designs. One predictor design per scenario/sample-size cell limits generalization. This is a mechanism investigation, not a confirmation study or a search for the covariance formula closest to nominal size. No production code changed.","",
    "The table reports rejection at nominal 5%, conditional on every category being observed. The three-category settings have one predictor; the five-category setting has three predictors and nine slope contrasts. All columns use n=80. Oracle rows require information unavailable in an application.","",
    "| Calculation | Regular, 3 categories | Rare endpoint | Rare interior | Regular, 5 categories |",
    "|---|---:|---:|---:|---:|",table.rows,"",
    "Full results, including 95% Monte Carlo intervals, nominal 1%, both selection populations and larger samples, are in summary.csv. For independent-bank procedures the evaluation intervals condition on the realized moment/reference bank; they do not include uncertainty in those banks.","",
    "## Covariance scale is a major part of the endpoint failure", "",
    sprintf("For the rare endpoint with all categories observed, the contrast's independent moment-bank standard deviation is %.3f, compared with %.3f for the square root of the mean original estimated variance. The mean variance estimate is only %.1f%% of the independently estimated sampling variance. Its mean contrast is %.3f (Monte Carlo standard error %.3f), providing little evidence that a large mean shift explains the problem.",endpoint$sd.first,sqrt(endpoint$mean.variance.first),100*endpoint$variance.ratio.first,endpoint$bias.first,endpoint$bias.mcse),"",
    sprintf("Using the exact adjusted-score derivative changes rejection from %s to %s. The paired change is %.1f percentage points (Monte Carlo standard error %.1f points). However, its mean estimated variance becomes %.2f times the independent sampling variance while rejection remains inflated. A correct derivative alone does not make a local covariance approximation accurate; averages can also be dominated by extreme covariance estimates.",form(rate(3,"empirical_ordinary")),form(rate(3,"empirical_adjusted")),100*rare.paired$difference,100*rare.paired$mcse,adjusted$variance.ratio.first),"",
    sprintf("Replacing the empirical score products by the true score variance while retaining the fitted ordinary sensitivity gives %s rejection in the rare-endpoint case and %s in the five-category case. Using a fitted proportional-odds score variance gives %s and %s, respectively. These formulas substantially overcorrect. Knowing the score variance does not remove errors from evaluating nonlinear sensitivities at noisy estimates. This rules out presenting a simple score-variance substitution as the solution.",pct(rate(3,"true_ordinary")$rate),pct(rate(8,"true_ordinary")$rate),pct(rate(3,"null_ordinary")$rate),pct(rate(8,"null_ordinary")$rate)),"",
    "## Covariance randomness and chi-square shape also matter", "",
    sprintf("For the five-category joint test, replacing each dataset's covariance by an independent mean of that same covariance formula reduces rejection from %s to %s. This changes covariance variability and its dependence on the slope contrasts together. It does not separate those two effects. The mean covariance still underestimates variance in some contrast directions: its eigenvalues relative to the independent sampling covariance range from %.2f to %.2f.",pct(rate(8,"empirical_ordinary")$rate),pct(rate(8,"independent_mean_empirical")$rate),regular$normalized.eigen.min,regular$normalized.eigen.max),"",
    sprintf("Even with an independently estimated sampling covariance, the five-category chi-square test rejects %s at nominal 5%% and %s at nominal 1%%. Removing the independently estimated mean changes these to %s and %s. Distributional shape therefore remains relevant after covariance and mean corrections, subject to the finite precision of the moment bank.",form(rate(8,"independent_sampling_covariance")),form(rate(8,"independent_sampling_covariance",.01)),pct(rate(8,"independent_sampling_centered")$rate),pct(rate(8,"independent_sampling_centered",.01)$rate)),"",
    sprintf("For the rare endpoint, the sampling-covariance calculation gives %s rejection at nominal 5%%, but %s at nominal 1%%. Centering the contrast does little. Agreement at one nominal level can conceal an incorrect tail shape; a multiplicative variance adjustment would not establish a valid test.",form(rate(3,"independent_sampling_covariance")),form(rate(3,"independent_sampling_covariance",.01))),"",
    "## Sparsity and empty categories", "",
    "The post-study count breakdown below uses the same rare-endpoint n=80 evaluation sample. It is descriptive and was not used to select an eligibility cutoff. The number of observations supporting a cumulative split is more informative here than total sample size alone.","",
    "| Observations in rare endpoint | Datasets | Original rejection at 5% | Exact-derivative rejection |","|---|---:|---:|---:|",
    vapply(seq_len(nrow(counts)),function(j) sprintf("| %s | %d | %s | %s |",counts$rare.category.count[j],counts$n[j],pct(counts$empirical.ordinary.fpr[j]),pct(counts$empirical.adjusted.fpr[j])),character(1)),"",
    sprintf("Retaining all datasets increases original rare-endpoint rejection from %s to %s. Thus the earlier all-categories-observed restriction did not create the inflation. In the sparse-interior case the original test is conservative (%s among observed-category datasets), so there is no single directional correction across sparsity patterns.",pct(rate(3,"empirical_ordinary")$rate),pct(rate(3,"empirical_ordinary",population="all")$rate),pct(rate(6,"empirical_ordinary")$rate)),"",
    "For an empty interior category, adjacent binary responses are identical. Their fitted slope contrast and its theoretical empirical covariance are zero. In 49 of the 85 such evaluation datasets, floating-point cancellation left a tiny positive covariance (at most 3.0e-33), which passed a Cholesky factorization. These are not genuine nonsingular covariance estimates. The frozen results preserve that behavior and label it here; the observed-category comparisons above avoid it. A future test must handle the mathematical support explicitly, not infer it solely from numerical factorization.","",
    "## Oracle simulation calibration and reference uncertainty", "",
    "The oracle experiment uses the true model to generate independent reference statistics. The rank p-value has a finite-sample exchangeability argument, including conservative treatment of ties. This argument averages over the random reference sample; many evaluation datasets sharing one reference bank do not give independent unconditional p-values. A fitted-null bootstrap does not inherit the exact argument because its nuisance parameters are estimated.","",
    "One original reference bank gave an unusually low endpoint critical value. A separately specified post-study check generated exactly 20,000 new reference datasets for each of the endpoint and five-category n=80 designs, keeping the original moment and evaluation banks fixed. The original results remain available. This adds 40,000 response datasets and measures reference precision; it is not fresh confirmation on new evaluation data.","",
    "| Scenario | Reference bank | Selected reference size | Rejection at 5% | Rejection at 1% |","|---|---|---:|---:|---:|")
for(i in c(3L,8L)) for(b in c(2L,4L)) {
    z <- ref[ref$cell==i & ref$reference.bank==b & ref$population=="observed",]
    lines <- c(lines,sprintf("| %s | %s | %d | %s | %s |",if(i==3) "Rare endpoint" else "Regular, 5 categories",
        if(b==2) "Original" else "Precision check",z$reference.n[1],form(z[z$level==.05,]),form(z[z$level==.01,])))
}
lines <- c(lines,"", "Intervals in this table condition on the reference and moment banks. The additional reference bank reduces its own tail-probability uncertainty substantially, but does not eliminate evaluation uncertainty. The endpoint 1% estimate remains somewhat high. These results support simulation calibration as a mechanism worth pursuing; the rank argument, rather than closeness of one observed rate to nominal size, establishes validity of the known-parameter oracle construction.","",
    "## Consequences for the next method", "",
    "Keep the public proportional-odds assessment descriptive for now. Do not substitute the adjusted sensitivity, multiply the covariance by an empirical constant, or add a universal minimum-n rule. The next development candidate should compare the fixed-dimensional meanBR slope contrasts with a fitted-null simulation distribution and avoid a new covariance inversion for each simulated dataset.","",
    "NEXT-METHOD.md defines a concrete fixed-metric statistic and its regular-interior asymptotic argument, together with the unresolved nuisance-parameter and original-empty-category issues. In particular, simulations with empty categories must not be silently discarded or have their contrast dimension reduced. Unknown null parameters are the principal remaining obstacle to a rigorously calibrated sparse-sample test.","",
    "The present study concerns the development stacked-binary proportional-odds test. It does not invalidate the package's ordinal fitting, marginal-effects API, or existing separate bootstrap procedures.","",
    "## Reproduction", "",
    "PROTOCOL.md, THEORY.md and manifest.rds define the main study. REFERENCE-CHECK.md and reference-manifest.rds define the additional precision check. run.R audits the original 36,000 datasets and regenerates its tables; reference-check.R audits and summarizes the additional 40,000. diagnose.R verifies independent design regeneration and exact replays and records support diagnostics. interpret.R generates this assessment and the figures. README.md gives complete commands. Raw-file hashes and report-input hashes record the complete evidence set.")
writeLines(lines,file.path(root,"ASSESSMENT.md"))

plot.results <- function() {
    old <- par(mfrow=c(2,2),mar=c(5,10,3,1),oma=c(2,0,0,0)); on.exit(par(old))
    short <- c("Original covariance","Exact derivative","Fixed mean covariance*","Sampling covariance*","Sampling cov., centered*")
    for(i in c(1L,3L,6L,8L)) {
        z <- do.call(rbind,lapply(methods,function(method) rate(i,method)))
        yy <- rev(seq_len(nrow(z)))
        plot(100*z$rate,yy,xlim=c(0,30),ylim=c(.5,5.5),axes=FALSE,xlab="False-positive rate (%)",ylab="",pch=19,
            col="black",main=c("Regular, 3 categories","Rare endpoint","Rare interior","Regular, 5 categories")[match(i,c(1L,3L,6L,8L))])
        axis(1); axis(2,at=yy,labels=short,las=1,cex.axis=.8); abline(v=5,lty=2,col="gray50")
        segments(100*z$mc.lower,yy,100*z$mc.upper,yy,col="gray40"); points(100*z$rate,yy,pch=19)
        box()
    }
    mtext("n = 80; every category observed; nominal 5%.  * Oracle diagnostic. Intervals condition on moment bank.",side=1,outer=TRUE,cex=.8)
}
pdf(file.path(root,"mechanism-rejection.pdf"),width=13,height=8); plot.results(); dev.off()
png(file.path(root,"mechanism-rejection.png"),width=1950,height=1200,res=150); plot.results(); dev.off()
inputs <- sort(c("interpret.R","NEXT-METHOD.md","summary.csv","moments.csv","paired.csv","failure-counts.csv","fit-diagnostics.csv",
    "reference-check.csv","endpoint-count-diagnostics.csv","empty-interior-diagnostics.csv","replay-checks.csv"))
write.csv(data.frame(file=inputs,md5=unname(tools::md5sum(file.path(root,inputs)))),file.path(root,"report-input-hashes.csv"),row.names=FALSE)
