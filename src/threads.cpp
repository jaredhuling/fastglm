#include <RcppEigen.h>
#include "parallel_reduce.h"
#include "trace.h"

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

// Sets the package trace flag and returns the previous value. When restart
// is true and the trace is switched on from off, the start time of the
// traced fit is set to now; restoring a previous setting passes
// restart = false, so a nested call does not move the start of the outer
// fit.
// [[Rcpp::export]]
bool fastglm_set_trace_cpp(bool on, bool restart = true)
{
    const bool prev = fglm_trace::trace_setting();
    if (on && !prev && restart) fglm_trace::trace_start() = fglm_trace::clock::now();
    fglm_trace::trace_setting() = on;
    return prev;
}

// [[Rcpp::export]]
bool fastglm_trace_on_cpp()
{
    return fglm_trace::trace_setting();
}

// One trace line in the common format, for iteration loops written in R.
// [[Rcpp::export]]
void fastglm_trace_line_cpp(int iter, int passes, double criterion, int halvings,
                            std::string label)
{
    fglm_trace::Tracer().iteration(iter, passes, criterion, halvings, label.c_str());
}
