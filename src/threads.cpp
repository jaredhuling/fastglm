#include <RcppEigen.h>
#include "parallel_reduce.h"

// Sets the package thread setting and returns the previous value. A value
// below 1 is stored as 1, and a serial build always stores 1.
// [[Rcpp::export]]
int fastglm_set_threads_cpp(int n)
{
    const int prev = fglm_par::thread_setting();
    if (n == NA_INTEGER || n < 1 || !fglm_par::has_openmp()) n = 1;
    fglm_par::thread_setting() = n;
    return prev;
}

// [[Rcpp::export]]
bool fastglm_has_openmp_cpp()
{
    return fglm_par::has_openmp();
}

// The number of logical processors available to OpenMP, or 1 on a serial
// build.
// [[Rcpp::export]]
int fastglm_logical_cores_cpp()
{
#ifdef _OPENMP
    return omp_get_num_procs();
#else
    return 1;
#endif
}
