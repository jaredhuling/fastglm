# Frozen analytic-score development comparison

Specified November 3, 2026 before the evaluation run. This is development,
not independent confirmation: its scenarios have informed the preceding work.
The new response/predictor seeds are untouched. The decision is whether the
null-score construction merits further analytic refinement and production work.
No per-analysis Monte Carlo calibration or bootstrap is used by any candidate.

Compare three procedures on common datasets:

1. The existing stacked-binary meanBR Wald prototype in ../prototype.R.
2. The ordinary efficient score test at the ML proportional-odds null fit.
3. The ordinary efficient score test at the meanBR proportional-odds null fit,
   with the explicit nuisance projection in METHODS.md.

Use all ten scenarios and n=80,250,1000 from the frozen generator in
../study/config.R, including eight null designs and two coherent alternatives.
Their exact definitions are pinned by the source manifest. There are 1,000
datasets per cell: 30,000 datasets and 90,000 method records. Response and
predictor generation share dataset seed 80361101+10000*cell+replication.
The timing-only pilot uses indices 9001:9002 and is excluded. Use
Mersenne-Twister/Inversion/Rejection, three processes, and one numerical thread
per process. The installed fastglm source baseline is b5e08af, version 0.1.3.

Hold optimizer tolerances at 1e-10 and maxit=500 as in the preceding studies.
Require every declared category to be observed and the expected contrast
dimension to remain p(K-2). Failed fits or factorizations remain unavailable;
no pseudoinverse, redraw, fallback or method switch is allowed.

Record per-method statistics, p-values, elapsed time, availability, reasons,
warnings, null iterations, information conditioning and the ordinary nuisance
score's quadratic norm. The latter documents the difference between projected
and naive full-score calculations; it is not used to choose the test.

Report 5% and 1% null rejection, power under the two alternatives, Wilson
Monte Carlo intervals, availability and unconditional rejection bounds with
unavailable outcomes unresolved. Pairwise rejection differences use shared
available datasets and paired standard errors. Report timings as descriptive
medians and paired ratios. Type-I accuracy and power must be read together;
conservatism alone is not evidence of success. No numerical promotion threshold
or calibration correction is selected from results.

Before freezing, validate analytic scores against numerical derivatives of
the full ordinal likelihood, expected information against a numerical expected
Hessian, exact tiny-sample score moments, block independence, production null
score/information identities, nuisance-score decomposition, outcome reversal
and affine predictor invariance. A numerical Hessian step of .01 avoids
second-difference cancellation at zero deviation parameters; this numerical
verification setting is unrelated to rejection performance.

The main run and summaries preserve source/package hashes and the full seed
grid. A new method or higher-order correction after inspection requires a new
comparison; these results are never relabeled as confirmation.
