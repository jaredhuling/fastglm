# Original-source common-slope tests

This is a development and implementation validation, based on commit
`2619731`. The change lets common-slope tests replay an ordinal model's original
source. It does not change the statistical procedure or its chi-square
calibration. The defining implementation requirement is to avoid collecting
the full design or outcomes: storage is one source block, coefficient-sized
matrices, and the source object's existing block metadata.

## Protocol and decision

Before evaluating performance, require agreement with the frozen dense
implementation for all three links, ML and reduced-null meanBR score tests,
and partial-model likelihood-ratio tests. Verify complete factor and
interaction selection and both joint and term-specific tests. Check grouped
versus individual outcomes, source partitions, frequency replication, affine
predictor recoding, and offset shifts. Independently differentiate category
probabilities to reconstruct the ordinary score and expected information at
the embedded reduced-null estimate.

Failure cases cover empty categories, zero frequency, deficient design rank,
reader errors, exhausted fitting controls, and the sufficient separation
certificate. Check that a nonconverged supplied fit does not substitute for a
newly fitted null or alternative. Calls using explicit evaluation data must
retain their existing behavior. No calibration experiments, tuning, timing
claims, or new inferential claims are part of this development gate.

The source extension is accepted only when these identities and failure
contracts hold. Native and on-disk performance work is a separate coordinated
comparison; the present validation does not imply a speedup.

## Exact estimator and blockwise identity

Common-model tests fit the common null once. Their ordinary score and expected
information are sums over observations; each source block contributes on the
same global frequency scale. Joint and term-specific alternatives use the
same null, as in the existing dense API. Separate block-local normalizations
or block-specific fitted nulls would change the procedure and are not used.

Partial-model tests move each selected varying design column into the common
design of the reduced null. Other varying columns remain unrestricted.
Formula selectors retain complete factor and interaction terms. The
reduced-null estimate is fitted by the requested ML or meanBR method, then
embedded in the full partial model by repeating newly common slopes across
thresholds. All coordinate transformations use the reduced fit's full-design
centering and scaling. Source blocks then accumulate the full ordinary score
`U` and expected information `I` at that embedded estimate.

If `B` spans the equality-null tangent space in working coordinates, the score
test is

```
e = U - I B (B' I B)^(-1) B' U
statistic = e' I^(-1) e.
```

The nuisance projection is essential at a reduced-null meanBR root. The
ordinary nuisance score is generally nonzero. This remains distinct from
`fastglm_score_test()` using full-model constrained adjusted equations.

For partial-model LR tests, equating a selected column's slopes across all
thresholds gives exactly the model obtained by placing that column in the
common design. Fitting this reduced model by streamed ML therefore maximizes
the same null likelihood as the dense equality-constrained engine, subject
to successful numerical convergence. The alternative is separately refitted
by streamed ML. Both fits must converge. The existing likelihood-difference
check rejects a materially larger null likelihood; only a negative difference
within its declared floating-point tolerance is set to zero. MeanBR LR tests
are rejected. Common-model LR support is unchanged and remains unavailable.

## Assumptions and explicit limits

Additivity of score/information and equality of the reduced and constrained
model spaces are finite-sample algebraic identities. The chi-square reference
remains conditional on a correctly specified identifiable interior null,
fixed parameter dimension, nonsingular limiting information, consistent null
estimation, and an independent ordinal/integer-frequency score central limit
theorem. A fixed bounded design with positive category probabilities and
increasing integer cell counts supplies a concrete regular regime. The
implementation does not establish boundary-null, higher-order, cluster,
survey-weight, or growing-dimension calibration.

Effective category counts must be integer frequencies. Zero-frequency rows
do not contribute to information, category totals, or separation certificates.
The two-valued-column endpoint certificate combines counts across all source
blocks and stops considering a column after a third positive-frequency value
is seen. It is a sufficient certificate, not a complete separation detector.
Every declared category must have positive total frequency. The existing
streaming fit performs global rank checks and retains its native boundary,
domain, information, and convergence checks. Failures remain unavailable
tests with explanations; no alternate estimator or generalized inverse is
substituted.

The source must reproduce the same rows on every pass. Schema validation
checks representation, dimensions, categories, and column names; it does not
prove that an arbitrary user callback returns immutable contents. This is
the same replay requirement as source fitting. Source reads and null/full
fits are never treated as independent observations or independent models.

## Reproduction and primary provenance

`validate.R` generates a deterministic dense reference with the immutable
baseline installation, then compares current source tests with it and with
an independent probability-Jacobian calculation. It reports numerical
differences and fails when a prespecified tolerance is exceeded. Run from
the repository root, using a compiled current package without recompilation:

```sh
Rscript benchmarks/ordinal-source-po-tests/validate.R baseline \
  /path/to/baseline-library /tmp/ordinal-source-po-reference.rds
Rscript benchmarks/ordinal-source-po-tests/validate.R current \
  /tmp/ordinal-source-po-reference.rds
```

The baseline used in this session is the immutable installation at
`/private/tmp/fastglm-ordinal-source-next.0uB7VE/baseline-library`.
The broader deterministic regression suite is
`tests/testthat/test-ordinal-source-po-test.R`, together with the existing
common/partial PO and source tests. These development checks are evidence
of implementation agreement, not empirical evidence of calibration.

On 2026-11-04, the 15 frozen-reference comparisons differed by at most
`1.37e-12` in absolute value. Independently reconstructed score quadratics
differed by at most `3.54e-9`. The combined assessment/common/partial/source
regression run passed 968 expectations without failures, errors, or test
warnings. A final focused run after strengthening information-failure
diagnostics passed all 188 source-PO expectations. These figures were produced
by the commands above and the repository test files; they are numerical
agreement results, not performance or coverage claims.

The inferential conventions and verified primary provenance are recorded in
[`../ordinal-partial-po-tests/README.md`](../ordinal-partial-po-tests/README.md)
and [`../ordinal-partial-inference/methods.txt`](../ordinal-partial-inference/methods.txt).
The nearest primary ordinal reference is Kosmidis (2014), *Improved estimation
in cumulative link models*, JRSS B 76(1), 169–196,
[doi:10.1111/rssb.12025](https://doi.org/10.1111/rssb.12025).
The published version's Sections 11–12 were inspected in the preceding
development pass: equal-slope contrasts and first-order adjusted-score
motivation, respectively. The ordinary projected-score convention retained
here is not that paper's larger-model adjusted-score statistic. The current
change adapts established additive likelihood calculations to source replay;
it makes no statistical novelty claim.
