Replication: ordinal row kernels and file-backed fitting
========================================================

Run from the fastglm repository root. protocol.txt fixes the development design,
candidate, scientific/computational contracts, and limitations. The single
dispatcher is study.R; it never launches an expensive stage implicitly.

Dependencies: R, the package's normal dependencies, Rcpp/RcppEigen and a compiler
for the component wrappers; /usr/bin/time on macOS or Linux for process resource
records. Use installed baseline 2619731 and candidate packages built with the same
compiler/settings. candidate.patch records the native change relative to baseline.
Native component caches are disposable and platform/path specific; build them
from source instead of treating compiled binaries as portable evidence.

Each output/cache/input directory must be new. Replace the illustrative paths:

  Rscript benchmarks/ordinal-disk-performance/study.R build /baseline/library /work/components-baseline
  Rscript benchmarks/ordinal-disk-performance/study.R build /candidate/library /work/components-current

Run the following sequentially in an otherwise idle CPU window, with
OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, and VECLIB_MAXIMUM_THREADS=1:

  Rscript benchmarks/ordinal-disk-performance/study.R components /baseline/library /work/components-baseline /results/components-baseline
  Rscript benchmarks/ordinal-disk-performance/study.R components /candidate/library /work/components-current /results/components-current
  Rscript benchmarks/ordinal-disk-performance/study.R generate /work/disk-inputs
  Rscript benchmarks/ordinal-disk-performance/study.R run /baseline/library /candidate/library /work/disk-inputs /results/disk

Generation needs about 75 MB for uncompressed blocks. Whole-fit execution is 32
cells x 2 versions x 3 repetitions, each in a separate R process; allow several
minutes on the development machine and longer on other hardware. The run command
sets the thread environment for child processes, verifies all input hashes, and
preserves per-worker results, logs, resource output, and an execution ledger.
Statistical failures remain records. Operational failures stop with evidence
intact; corrected reruns use a new directory rather than overwriting results.

The retained disk-v1 attempt stopped after its first successful fit because the
restricted macOS resource collector could not query sysctl kern.clockrate.
Its result, log, and resource error remain available. The same fixed panel was
restarted in disk-v2 with permission for that resource query; no worker,
candidate, input, configuration, or fitting tolerance changed. Only disk-v2 is
used for that complete retry. Its later provenance audit found that both
package libraries contained the identical native binary despite different
installed headers. The candidate's header-only edit had not caused a native
rebuild. disk-v2 and its explicitly invalid report are retained for diagnosis,
not comparative performance claims. The two affected translation units were
then rebuilt; the actual patched-binary ordinal/display suite passed 4075
expectations with two expected worker skips and no failures/errors/warnings.
Its log and result object are retained under validation/. The fresh disk-v3
panel is the valid whole-fit comparison. run.R now rejects identical baseline
and candidate binaries before any timing, verifies candidate headers against
the source, and summarize.R checks every worker's frozen build hashes.

Regenerate the derived report and verify saved numerical/provenance checks:

  Rscript benchmarks/ordinal-disk-performance/study.R summarize /results/disk /results/components-baseline /results/components-current benchmarks/ordinal-disk-performance/report.txt

This last command only reads saved records. It writes complete measurement,
comparison, and component tables and errors if equality/provenance checks fail.
For the retained evidence, use disk-v3, components-baseline-v1, and
components-current-v1 under this directory. All 192 fits converged, and the 96
version pairs have identical recorded numerical state. The separately compiled
component panel also agrees with its installed evaluator and between versions.
The report includes small negative control differences and both failed attempts;
neither historical failed panel supports a candidate performance claim.
Portable reanalysis was also run from a separate temporary copy: all six
derived tables/objects reproduced byte-for-byte, and the report differed only
in its displayed input-directory path. validation/reanalysis.txt records this
check. The report discusses the unchanged common-path controls, warm-cache
limitation, and the distinction between bounded retained data and peak RSS.

Verify the archived file hashes without executing fits or timing code:

  Rscript benchmarks/ordinal-disk-performance/study.R verify benchmarks/ordinal-disk-performance

The input manifest and generator permit regeneration without the original
absolute work directory. Input values/hashes are expected to reproduce under
the recorded R/RNG/platform; numerical behavior on other builds should be
checked with appropriate rounding tolerances. Timing and peak RSS are inherently
platform dependent. These are explicitly warm-cache file-backed measurements,
not cold-device throughput tests or post-fit inference/assessment benchmarks.
