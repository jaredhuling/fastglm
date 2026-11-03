# Additional reference-bank precision check

Specified after the main study, November 3, 2026, and before generating any
additional responses. The original study and its results remain unchanged.

The single oracle reference bank produced 8.25% rejection at nominal 5% in
the rare-lower-endpoint n=80 observed-category population. Its absolute-contrast
95th percentile was 2.701, compared with 3.247 and 3.334 in the independent
moment and evaluation banks. The rank test is valid marginally over reference
randomness, but repeated evaluation against a shared reference has additional
uncertainty. Evaluation-only binomial intervals do not capture that uncertainty.

Generate a new independent bank of exactly 20,000 responses for each of the
rare_lower3 n=80 and regular5 n=80 fixed designs (cells 3 and 8). This gives a
more precise estimate of their reference distributions. Use bank number 4 in
the original seed formula, with replication indices 1:20000, which do not
overlap any original response seeds or pilot seeds. Run the existing frozen
fit routine unchanged. Keep the original moment covariance and original
evaluation samples fixed, and recompute oracle rank p-values for both declared
populations. Also retain and report the original reference results.

This is a post-study precision diagnosis, not independent confirmation of a
candidate or a newly tuned method. No stopping rule depends on the results;
no more reference banks are planned. Record original and new reference sizes,
critical values, rates at 5% and 1%, and evaluation-only Wilson intervals.
The latter still condition on the realized reference bank. For orientation,
sqrt(alpha*(1-alpha)/(B+2)) approximates the standard deviation of the random
tail probability of a continuous-distribution Monte Carlo critical value.
It is not the standard error of the evaluation rate and need not be exact
when there are atoms. Shared reference error is not averaged away by adding
evaluation samples.

The same fixed-dimension contrast and independent moment covariance are used
throughout. This precision check does not establish fitted-null bootstrap
validity or eliminate uncertainty from estimating the sampling covariance.
