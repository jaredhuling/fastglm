
## CRAN resubmission for 'fastglm' 0.1.2

This is a patch release that fixes a compilation failure reported by
CRAN's clang-trunk checks (clang 23 with libc++):

  https://www.stats.ox.ac.uk/pub/bdr/clang23/fastglm.log

libc++ in clang 23 and newer no longer supplies `<iterator>` and
`<algorithm>` as transitive includes. The `bigmemory` header
`BigMatrix.h`, which the big.matrix backend pulls in, uses
`std::back_inserter` and `std::copy` without including those headers
itself. The two source files that include `BigMatrix.h`
(`fit_glm_dense.cpp` and `bigmemory.cpp`) now include `<iterator>` and
`<algorithm>` explicitly before the `bigmemory` headers. No user-facing
behavior changes.

## Test environments

* local macOS (aarch64, R 4.5.1)
* R-hub:
    - linux, Ubuntu 24.04.4 LTS: R-devel (2026-06-21 r90185)
    - macos, macOS Sequoia 15.7: R-devel (2026-08-25 r90447)
    - macos-arm64, macOS Tahoe 26: R-devel (2026-08-25 r90447)
    - m1-san (ASAN + UBSAN), macOS Sequoia 15.7: R-devel (2026-08-25 r90447)
    - windows, Windows Server 2022: R-devel (2026-08-25 r90447 ucrt)
    - atlas, Fedora Linux 42: R-devel (2026-06-21 r90185)

## R CMD check results

Status: OK (0 errors | 0 warnings | 0 notes) on all environments listed
above, including `checking whether package can be installed`, `checking
compiled code`, and `checking tests`.

