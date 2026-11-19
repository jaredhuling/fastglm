#ifndef FASTGLM_PARALLEL_REDUCE_H
#define FASTGLM_PARALLEL_REDUCE_H

// Thread setting and deterministic block reductions shared by the engines.
//
// The R side resolves the thread count at each entry point and stores it
// here through fastglm_set_threads_cpp(). Engines ask threads_for() how many
// threads to use for a given number of row blocks; a parallel region is
// entered only when there are at least 2 row blocks.
//
// The reductions lane_reduce() and tile_reduce() fix the floating-point
// operations and their order for every output entry as a function of the
// block count and the accumulator dimension only, so results are identical
// to the bit for every thread count, including 1. Their only R API call is
// Rcpp::checkUserInterrupt(), made on the calling thread outside any
// parallel region. The callbacks they run inside a parallel region must not
// call the R API.

#include <RcppEigen.h>
#include <algorithm>
#include <atomic>
#include <exception>
#include <vector>

#ifdef _OPENMP
// CRAN's R for macOS ships libomp.dylib in its lib directory but no omp.h.
// When the header is absent, declare the few OpenMP runtime functions the
// package uses; they are part of the OpenMP C API with a stable C ABI.
#  if defined(__has_include)
#    if __has_include(<omp.h>)
#      include <omp.h>
#      define FASTGLM_HAVE_OMP_H 1
#    endif
#  else
#    include <omp.h>
#    define FASTGLM_HAVE_OMP_H 1
#  endif
#  ifndef FASTGLM_HAVE_OMP_H
extern "C" {
int omp_get_num_procs(void);
int omp_get_num_threads(void);
int omp_get_thread_num(void);
int omp_get_max_threads(void);
}
#  endif
#endif

namespace fglm_par {

// The package-wide thread setting. A function-local static in an inline
// function has a single instance across the translation units of the shared
// library. Defaults to 1 until the R side sets it at load.
inline int& thread_setting()
{
    static int n = 1;
    return n;
}

// True when the package was compiled with OpenMP.
inline bool has_openmp()
{
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

// The index of the calling thread within the current parallel region (0
// outside a region and in a serial build).
inline int current_thread()
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// The number of threads in the current team (1 outside a parallel region
// and in a serial build).
inline int team_size()
{
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

// The largest team size seen by the reductions since the last reset, for
// tests that check that threading is in effect. Written only by thread 0
// of a team; read on the calling thread after the region.
inline int& max_team_size_seen()
{
    static int m = 1;
    return m;
}
inline void note_team_size()
{
    if (current_thread() == 0) {
        const int n = team_size();
        if (n > max_team_size_seen()) max_team_size_seen() = n;
    }
}

// The number of threads to use for nblocks row blocks: 1 when there are
// fewer than 2 blocks, otherwise the setting capped at the block count.
inline int threads_for(Eigen::Index nblocks)
{
    if (nblocks < 2) return 1;
    const int n = thread_setting();
    if (n < 1) return 1;
    return static_cast<Eigen::Index>(n) < nblocks ? n : static_cast<int>(nblocks);
}

// Elementwise work over fixed row blocks: f(r, k) for the blocks [r, r + k)
// of n rows with block_rows rows each (the last block may be shorter), in
// parallel with a static schedule when there are at least 2 blocks. f must
// write only to rows of its own block and must not call the R API.
//
// The blocks are the same for every thread count, so work whose result for
// a row depends on how the rows are split (for example Eigen's vectorized
// exp(), whose packet and scalar paths can differ in the last bit) gives the
// same result for every thread count. An exception thrown by f is caught
// inside the region; after the region the exception of the smallest failing
// block is rethrown on the calling thread, which is the exception a serial
// pass would have thrown first.
template <class F>
void parallel_blocks(Eigen::Index n, Eigen::Index block_rows, F f)
{
    if (n <= 0) return;
    if (block_rows < 1) block_rows = 1;
    const Eigen::Index nblocks = (n + block_rows - 1) / block_rows;
    const int nt = threads_for(nblocks);
    if (nt <= 1) {
        for (Eigen::Index r = 0; r < n; r += block_rows)
            f(r, std::min<Eigen::Index>(block_rows, n - r));
        return;
    }
    std::vector<std::exception_ptr> err(static_cast<std::size_t>(nblocks));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (Eigen::Index b = 0; b < nblocks; ++b) {
        const Eigen::Index r = b * block_rows;
        if (b == 0) note_team_size();
        try {
            f(r, std::min<Eigen::Index>(block_rows, n - r));
        } catch (...) {
            err[static_cast<std::size_t>(b)] = std::current_exception();
        }
    }
    for (const auto& e : err)
        if (e) std::rethrow_exception(e);
}

// Fixed reduction parameters. None of them depends on the thread count or on
// a user option, so the summation order of every output entry is the same
// for any thread count.
constexpr int kLanes = 8;                 // accumulation lanes for small d
constexpr Eigen::Index kLaneMaxDim = 1024; // 8 lanes of d^2 doubles <= 64 MiB
constexpr Eigen::Index kTileWidth = 64;   // column tile width for large d
constexpr int kGroupBlocks = 8;           // blocks prepared per tile group

// Rows per block in the tile scheme: max(256, 2^20 / d), so that one weighted
// block of d columns takes at most 8 MB (for d up to 4096).
inline Eigen::Index tile_block_rows(Eigen::Index d)
{
    if (d < 1) d = 1;
    return std::max<Eigen::Index>(256, Eigen::Index(1048576) / d);
}

// True when the accumulator dimension d uses lanes, false for column tiles.
inline bool use_lanes(Eigen::Index d)
{
    return d <= kLaneMaxDim;
}

// The number of column tiles of width kTileWidth covering d columns.
inline Eigen::Index tile_count(Eigen::Index d)
{
    return d <= 0 ? 0 : (d + kTileWidth - 1) / kTileWidth;
}

// The sum over the row blocks [r, r + k) of n rows (block_rows rows each,
// the last block possibly shorter) of f(r, k), a block's sum, in the lane
// order of lane_reduce with a double accumulator: block b's sum is added to
// lane b mod kLanes (8) in increasing b, and the lanes are then added in lane order.
// The block sums are computed by parallel_blocks, so f must write only to
// rows of its own block and must not call the R API; the order of the
// additions depends on the block count only, so the result is the same for
// every thread count. With a single block the result is f(0, n), the serial
// sum. An exception thrown by f is rethrown as parallel_blocks describes.
// With parallel = false the blocks are evaluated serially on the calling
// thread, in the same order of additions, for an f that calls the R API
// (for example R's math library).
template <class F>
double ordered_block_sum(Eigen::Index n, Eigen::Index block_rows, F f, bool parallel = true)
{
    if (n <= 0) return 0.0;
    if (block_rows < 1) block_rows = 1;
    const Eigen::Index nblocks = (n + block_rows - 1) / block_rows;
    std::vector<double> part(static_cast<std::size_t>(nblocks), 0.0);
    auto one = [&](Eigen::Index r, Eigen::Index k) {
        part[static_cast<std::size_t>(r / block_rows)] = f(r, k);
    };
    if (parallel) {
        parallel_blocks(n, block_rows, one);
    } else {
        for (Eigen::Index r = 0; r < n; r += block_rows)
            one(r, std::min<Eigen::Index>(block_rows, n - r));
    }
    const int nlanes = nblocks < kLanes ? static_cast<int>(nblocks) : kLanes;
    double out = 0.0;
    for (int l = 0; l < nlanes; ++l) {
        double lane = 0.0;
        for (Eigen::Index b = l; b < nblocks; b += kLanes)
            lane += part[static_cast<std::size_t>(b)];
        out += lane;
    }
    return out;
}

// Lane reduction for small accumulators, with persistent lane accumulators
// and per-thread workspace.
//
// lanes holds the lane accumulators; it is grown to min(nblocks, 8) entries
// on the calling thread and its entries are kept between calls, so a caller
// that keeps lanes alive (for example across the passes of an iterative
// fit) does not allocate them again. reset(acc) zeroes an accumulator (and
// sizes it on first use); it is called for every lane used, inside the
// region, before the lane's first block. make_work() returns a workspace
// (for example a weighted row-block buffer); it is called once per running
// thread. Workspace must not carry results from one block to the next: add
// must overwrite whatever it reads from it. add(b, acc, work) adds block b's
// contribution to acc and returns false on error.
//
// Lane l receives blocks l, l + 8, l + 16, ... in increasing order. The
// lanes are dealt to up to min(threads_for(nblocks), 8) threads, and each
// thread visits the blocks in increasing order, processing those of its own
// lanes, so a single thread reads the rows once from first to last. Which
// thread owns a lane does not affect the result. combine(out, lane) is then
// called for the lanes in increasing order on the calling thread. Only
// min(nblocks, 8) lanes are used, because a lane with no blocks would add an
// exact zero; this choice depends on nblocks only.
//
// reset, add, and make_work must not call the R API. An exception thrown by
// add is caught inside the region and counts as a failure of its block; an
// exception from reset or make_work counts as a failure of the first block of
// each lane of that thread. Each lane stops at its own first failure and the
// other lanes run on, so the smallest failing block index is the same for
// every thread count. Returns false if any block failed; out is then left
// unchanged and, when failed_block is not null, *failed_block is set to the
// smallest failing block index (0-based). On success *failed_block is -1.
template <class Acc, class Reset, class MakeWork, class Add, class Combine>
bool lane_reduce_into(Eigen::Index nblocks, Acc& out, std::vector<Acc>& lanes, Reset reset,
                      MakeWork make_work, Add add, Combine combine,
                      Eigen::Index* failed_block = nullptr)
{
    if (failed_block) *failed_block = -1;
    if (nblocks <= 0) return true;
    const int nlanes = nblocks < kLanes ? static_cast<int>(nblocks) : kLanes;
    if (static_cast<int>(lanes.size()) < nlanes) lanes.resize(nlanes);

    // First failing block of each lane, or -1. Only the lane's owner writes
    // its entry.
    std::vector<Eigen::Index> lane_fail(nlanes, -1);
    const int nt = std::min(threads_for(nblocks), nlanes);
    (void) nt;
#ifdef _OPENMP
#pragma omp parallel num_threads(nt) if (nt > 1)
#endif
    {
        const int t = current_thread();
        const int T = team_size();
        note_team_size();
        bool ready = true;
        try {
            for (int l = t; l < nlanes; l += T) reset(lanes[l]);
        } catch (...) {
            ready = false;
        }
        if (ready) {
            try {
                auto work = make_work();
                for (Eigen::Index b = 0; b < nblocks; ++b) {
                    const int l = static_cast<int>(b % kLanes);
                    if (l % T != t || lane_fail[l] >= 0) continue;
                    bool ok = false;
                    try {
                        ok = add(b, lanes[l], work);
                    } catch (...) {
                        ok = false;
                    }
                    if (!ok) lane_fail[l] = b;
                }
            } catch (...) {
                ready = false;
            }
        }
        if (!ready)
            for (int l = t; l < nlanes; l += T)
                if (lane_fail[l] < 0) lane_fail[l] = l;
    }

    Rcpp::checkUserInterrupt();
    Eigen::Index first = -1;
    for (int l = 0; l < nlanes; ++l)
        if (lane_fail[l] >= 0 && (first < 0 || lane_fail[l] < first)) first = lane_fail[l];
    if (first >= 0) {
        if (failed_block) *failed_block = first;
        return false;
    }
    for (int l = 0; l < nlanes; ++l) combine(out, static_cast<const Acc&>(lanes[l]));
    return true;
}

// Lane reduction with lane accumulators made for this call: as
// lane_reduce_into, with make() returning a zeroed accumulator.
template <class Acc, class Make, class MakeWork, class Add, class Combine>
bool lane_reduce_ws(Eigen::Index nblocks, Acc& out, Make make, MakeWork make_work,
                    Add add, Combine combine, Eigen::Index* failed_block = nullptr)
{
    std::vector<Acc> lanes;
    return lane_reduce_into(nblocks, out, lanes, [&make](Acc& a) { a = make(); },
                            make_work, add, combine, failed_block);
}

// Lane reduction for accumulators that need no workspace: as lane_reduce_ws,
// with add(b, acc).
template <class Acc, class Make, class Add, class Combine>
bool lane_reduce(Eigen::Index nblocks, Acc& out, Make make, Add add, Combine combine,
                 Eigen::Index* failed_block = nullptr)
{
    struct NoWork {};
    return lane_reduce_ws(
        nblocks, out, make, []() { return NoWork(); },
        [&add](Eigen::Index b, Acc& acc, NoWork&) { return add(b, acc); },
        combine, failed_block);
}

// Column-tile reduction for large accumulators.
//
// Blocks are processed in groups of kGroupBlocks. Within a group starting at
// block g:
//   1. prepare(g + j, work[j]) runs in parallel for the group's slots j and
//      returns false on error;
//   2. add_tile(work[j], t) runs in parallel over the tiles t = 0, ...,
//      ntiles - 1, each tile adding the group's slots j = 0, 1, ... in
//      increasing order, so every entry of a tile receives the blocks in
//      increasing block order whatever thread owns the tile;
//   3. add_ordered(work[j]) runs on the calling thread for the group's slots
//      in increasing order (for vector and scalar sums).
// add_tile for different tiles must write disjoint entries. prepare and
// add_tile must not call the R API; exceptions they throw are caught inside
// the region and count as failures. work must hold at least
// min(kGroupBlocks, nblocks) slots.
//
// Returns false if any prepare or add_tile failed; the remaining groups are
// then skipped. Every slot of a group is prepared even when another fails,
// so when failed_block is not null, *failed_block is set to the smallest
// failing block index (0-based), the same for every thread count. A failure
// in add_tile (an exception, for example std::bad_alloc) reports the group's
// first block. On success *failed_block is set to -1.
template <class Work, class Prepare, class AddTile, class AddOrdered>
bool tile_reduce(Eigen::Index nblocks, Eigen::Index ntiles, std::vector<Work>& work,
                 Prepare prepare, AddTile add_tile, AddOrdered add_ordered,
                 Eigen::Index* failed_block = nullptr)
{
    if (failed_block) *failed_block = -1;
    if (nblocks <= 0) return true;
    if (static_cast<Eigen::Index>(work.size()) <
        std::min<Eigen::Index>(kGroupBlocks, nblocks))
        Rcpp::stop("tile_reduce: work must hold kGroupBlocks slots");

    const int nt = threads_for(nblocks);
    std::vector<char> slot_fail(kGroupBlocks, 0);

    for (Eigen::Index g = 0; g < nblocks; g += kGroupBlocks) {
        const int gsize = static_cast<int>(
            std::min<Eigen::Index>(kGroupBlocks, nblocks - g));

        std::fill(slot_fail.begin(), slot_fail.end(), 0);
        const int ntp = std::min(nt, gsize);
        (void) ntp;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(ntp) if (ntp > 1)
#endif
        for (int j = 0; j < gsize; ++j) {
            if (j == 0) note_team_size();
            bool ok = false;
            try {
                ok = prepare(g + j, work[j]);
            } catch (...) {
                ok = false;
            }
            if (!ok) slot_fail[j] = 1;
        }
        for (int j = 0; j < gsize; ++j) {
            if (slot_fail[j]) {
                Rcpp::checkUserInterrupt();
                if (failed_block) *failed_block = g + j;
                return false;
            }
        }

        std::atomic<bool> tile_failed(false);
        const int ntt = static_cast<int>(
            std::min<Eigen::Index>(nt, std::max<Eigen::Index>(ntiles, 1)));
        (void) ntt;
        // Tiles shrink toward the bottom right of the lower triangle, so they
        // are dealt round robin; the assignment does not affect the result.
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1) num_threads(ntt) if (ntt > 1)
#endif
        for (Eigen::Index t = 0; t < ntiles; ++t) {
            try {
                for (int j = 0; j < gsize; ++j) add_tile(static_cast<const Work&>(work[j]), t);
            } catch (...) {
                tile_failed.store(true, std::memory_order_relaxed);
            }
        }
        if (tile_failed.load()) {
            Rcpp::checkUserInterrupt();
            if (failed_block) *failed_block = g;
            return false;
        }

        for (int j = 0; j < gsize; ++j) add_ordered(static_cast<const Work&>(work[j]));
        Rcpp::checkUserInterrupt();
    }
    return true;
}

} // namespace fglm_par

#endif
