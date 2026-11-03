# Sparse ordinal inference and null estimation

The immediate issue is whether failed ML fits are numerical failures or reflect
nonexistence of a finite ML estimate. In the original n=100 uncommon-predictor
scenario, all 96 failed ML fits have a two-valued predictor whose rare group
contains only the highest or lowest outcome. Twenty additional datasets have
no observations in that predictor group. `diagnose.R` retains a replay of all
116 datasets; it is developmental diagnosis, not new confirmation evidence.

## Separation certificate

Suppose a predictor takes values a and b. Changing its coefficient by t/(b-a)
and every threshold by ta/(b-a) leaves all cumulative logits in group a unchanged
and subtracts t from every cumulative logit in group b. If group b contains
only the highest outcome, increasing t strictly increases its likelihood and
does not change the likelihood elsewhere. The supremum is approached as
t tends to infinity, so no finite ML solution exists. Reversing the direction
handles a lowest-outcome group. Fixed offsets and other unchanged predictors
do not alter this argument, and grouping preserves it exactly.

This is an exact sufficient certificate for these configurations. It is not a
general separation detector. Failure to find this certificate does not prove
that a finite ML estimate exists. Constant predictors are reported separately
as lack of identifiability. Increasing an optimizer's iteration limit cannot
repair either absence of a finite solution or absence of predictor variation.

## Explicit meanBR null and its scope

The additional option fits the proportional-odds null by mean bias reduction,
then evaluates the ordinary efficient score and expected information at that
root. It uses exactly the projected-score quadratic statistic specified in
the earlier confirmation METHODS.md, including individual-frequency scaling.
The ordinary nuisance score is not zero at a meanBR root. Its projection is
therefore retained explicitly, rather than adding nuisance estimation error
to the tested slope-deviation statistic.

For a regular interior null with fixed dimension, the mean-bias adjustment is
O(1), the information is O(N), and a consistent adjusted-score root differs
from the ML root by O_p(1/N). Smoothness then gives an O_p(1) change in the
efficient score and an o_p(1) change in its normalized quadratic form. Thus
the same first-order chi-square limit applies under the ordinary independent,
correctly specified, nonsingular-information conditions. Bounded full-rank
predictors, fixed positive category probabilities and separated thresholds
provide a concrete regular regime. The argument does not assert accurate
tails uniformly over rare-level frequencies tending to zero.

This establishes first-order equivalence conditional on regularity. It does not
establish second-order accuracy, finite-sample size, or conditional validity
given that ML failed. The confirmation compares two fixed choices on all
datasets, including unavailable results. No estimator is selected automatically
according to convergence, a p-value, or a simulation scenario. The new option
can improve availability under separation, while retaining unsupported status
for empty declared categories and deficient rank.

Kosmidis (2014), *Improved estimation in cumulative link models*, JRSS B 76(1),
169-196, doi:10.1111/rssb.12025, supplies the bias-reduced null estimator and
its first-order properties (Section 4, published page 175). The published
[institutional PDF](https://wrap.warwick.ac.uk/id/eprint/98828/7/WRAP-improved-estimation-cumulative-link-models-Kosmidis-2014.pdf)
was consulted on 2026-11-03, specifically the model/adjustment definitions,
first-order discussion on PDF page 7, and Sections 11-12 on PDF pages 24-26.
Section 12 also describes a distinct statistic using the adjusted score of the
larger model. That statistic is not implemented here: the meanBR option uses
the ordinary nuisance-projected score. No novelty or higher-order Firth-test
claim is made for this implementation. Developing the larger-model adjustment
would be a separate method change and would require new validation.

The numerical tests independently differentiate the likelihood and its expected
curvature at the meanBR root, and compare grouped and expanded frequencies.
Prediction and coefficient objects are not modified. The direct public test
uses only retained design metadata and supplied outcomes; coefficients from
the model passed as `object` do not enter the test.
