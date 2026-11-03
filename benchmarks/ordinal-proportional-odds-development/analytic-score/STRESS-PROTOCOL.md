# Additional analytic-score stress comparison

Specified after reading the main comparison, November 3, 2026, before generating
the additional datasets. Ordinary score inference substantially improved the
main grid. The meanBR-root score was sometimes conservative. This comparison
tests the known warning about ordinary score inference as dimension, leverage
and uncommon predictor levels increase. It is a development stress study, not
confirmation and not a search for a correction factor.

Retain exactly the three methods, solver settings and eligibility rules from
PROTOCOL.md. There are four new null designs at n=80 and n=250, each with 1,000
datasets (8,000 total). Generate predictors independently unless specified.

* normal5: four standard normal predictors, beta=(.8,-.6,.4,-.3),
  thresholds=(-1.8,-.6,.6,1.8).
* many_predictors5: eight standard normal predictors,
  beta=(.5,-.4,.3,-.2,0,0,0,0), the same thresholds.
* uncommon_binary5: x1 standard normal, x2 Bernoulli(.05), x3 Bernoulli(.2),
  x4 uniform[-1,1], beta=(.6,.8,-.5,.4), the same thresholds.
* leverage3: x1=uniform[-1,1]+15*Bernoulli(.03), x2 uniform[-1,1],
  beta=(.25,-.4), thresholds=(-.8,.8).

For each design F_ij=logistic(alpha_j-x_i'beta). Record every empty category,
rank-deficient design and fit/factorization failure; do not redraw. No new
cutoff or switching rule will be selected. Normal predictor designs broaden
the empirical scope beyond the bounded-design example in METHODS.md, without
claiming that the stated bounded-design argument already covers them uniformly.

Cell order uses n=80,250 within the above scenario order. Dataset seeds are
90361101+10000*cell+replication, separate from the main run. RNG and numerical
thread settings match the main study; use three processes. Report null rejection
at 5% and 1%, Wilson intervals, availability, conditional paired differences,
failure reasons and descriptive median runtime. These eight cells are fixed;
there is no automatic expansion or repeated search until a desired rate appears.
