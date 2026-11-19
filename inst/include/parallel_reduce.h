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

// The number of threads to use for nblocks row blocks: 1 when there are
// fewer than 2 blocks, otherwise the setting capped at the block count.
inline int threads_for(Eigen::Index nblocks)
{
    if (nblocks < 2) return 1;
    const int n = thread_setting();
    if (n < 1) return 1;
    return static_cast<Eigen::Index>(n) < nblocks ? n : static_cast<int>(nblocks);
}

// Fixed reduction parameters. None of them depends on the thread count or on
// a user option, so the summation order of every output entry is the same
// for any thread count.
constexpr int kLanes = 8;                 // accumulation lanes for small d
constexpr Eigen::Index kLaneMaxDim = 512; // 8 d^2 doubles <= 16 MB
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

// Lane reduction for small accumulators, with per-lane workspace.
//
// make() returns a zeroed accumulator. make_work() returns a workspace (for
// example a weighted row-block buffer); it is called once at the start of
// each lane, inside the lane's iteration, so at most one workspace per
// running thread is alive at a time. Workspace must not carry results from
// one block to the next: add must overwrite whatever it reads from it.
// add(b, acc, work) adds block b's contribution to acc and returns false on
// error.
//
// Lane l receives blocks l, l + 8, l + 16, ... in increasing order; lanes run
// in parallel on up to min(threads_for(nblocks), 8) threads, and
// combine(out, lane) is then called for the lanes in increasing order on the
// calling thread. Only min(nblocks, 8) lanes are created, because a lane with
// no blocks would add an exact zero; this choice depends on nblocks only.
//
// add and make_work must not call the R API. An exception thrown by either
// is caught inside the region and counts as a failure of the block being
// processed (the lane's first block for make_work). Each lane stops at its
// own first failure and the other lanes run on, so the smallest failing
// block index is the same for every thread count. Returns false if any block
// failed; out is then left unchanged and, when failed_block is not null,
// *failed_block is set to the smallest failing block index (0-based). On
// success *failed_block is set to -1.
template <class Acc, class Make, class MakeWork, class Add, class Combine>
bool lane_reduce_ws(Eigen::Index nblocks, Acc& out, Make make, MakeWork make_work,
                    Add add, Combine combine, Eigen::Index* failed_block = nullptr)
{
    if (failed_block) *failed_block = -1;
    if (nblocks <= 0) return true;
    const int nlanes = nblocks < kLanes ? static_cast<int>(nblocks) : kLanes;
    std::vector<Acc> lanes;
    lanes.reserve(nlanes);
    for (int l = 0; l < nlanes; ++l) lanes.push_back(make());

    // First failing block of each lane, or -1. Each lane writes only its
    // own entry.
    std::vector<Eigen::Index> lane_fail(nlanes, -1);
    const int nt = std::min(threads_for(nblocks), nlanes);
    (void) nt;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
#endif
    for (int l = 0; l < nlanes; ++l) {
        Eigen::Index b = l;
        try {
            auto work = make_work();
            for (; b < nblocks; b += kLanes) {
                if (!add(b, lanes[l], work)) {
                    lane_fail[l] = b;
                    break;
                }
            }
        } catch (...) {
            lane_fail[l] = b;
        }
    }

    Rcpp::checkUserInterrupt();
    Eigen::Index first = -1;
    for (int l = 0; l < nlanes; ++l)
        if (lane_fail[l] >= 0 && (first < 0 || lane_fail[l] < first)) first = lane_fail[l];
    if (first >= 0) {
        if (failed_block) *failed_block = first;
        return false;
    }
    for (int l = 0; l < nlanes; ++l) combine(out, lanes[l]);
    return true;
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
