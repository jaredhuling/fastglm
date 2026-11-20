#ifndef FASTGLM_TRACE_H
#define FASTGLM_TRACE_H

// Progress trace shared by the engines.
//
// The R side resolves the trace flag at each entry point and stores it here
// through fastglm_set_trace_cpp(), which also records the start time of the
// fit when the trace is switched on. An engine builds a Tracer from the
// setting (or from its own flag) and calls iteration() once per iteration,
// on the calling thread, between parallel regions: Rprintf and
// R_FlushConsole are R API calls and must never run inside a parallel
// region. With the trace off, iteration() returns at its first test.
//
// Line format (one line per iteration):
//   iter   3  passes    5  deviance 1234.567891  halvings 0  0.4s
// with label "deviance" or "loglik". passes counts the passes over the data
// so far, as the engine defines them (source passes for streamed fits,
// evaluations of the criterion over all rows otherwise); halvings counts the
// step halvings (or other step reductions) of this iteration; the last field
// is the time elapsed since the fit started.

#include <R_ext/Print.h>
#include <chrono>

extern "C" void R_FlushConsole(void);

namespace fglm_trace {

using clock = std::chrono::steady_clock;

// The package-wide trace flag. A function-local static in an inline function
// has a single instance across the translation units of the shared library.
inline bool& trace_setting()
{
    static bool on = false;
    return on;
}

// The start time of the traced fit.
inline clock::time_point& trace_start()
{
    static clock::time_point t = clock::now();
    return t;
}

struct Tracer {
    bool on;
    clock::time_point start;

    // From the package setting.
    Tracer() : on(trace_setting()), start(trace_start()) {}
    // From an engine's own flag (an existing trace argument).
    explicit Tracer(bool on_) : on(on_), start(trace_start()) {}

    // One trace line. Must be called from the calling thread, outside any
    // parallel region.
    void iteration(int iter, int passes, double criterion, int halvings,
                   const char* label) const
    {
        if (!on) return;
        const double secs =
            std::chrono::duration<double>(clock::now() - start).count();
        Rprintf("iter %3d  passes %4d  %s %.10g  halvings %d  %.1fs\n",
                iter, passes, label, criterion, halvings, secs);
        R_FlushConsole();
    }
};

}  // namespace fglm_trace

#endif  // FASTGLM_TRACE_H
