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
#include <memory>
#include <new>
#include <stdexcept>
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

// Rethrows, on the calling thread after a parallel region, an exception
// captured inside the region. An allocation failure (std::bad_alloc) becomes
// a std::runtime_error saying that memory ran out, with no block number: it
// is not a property of the data, and which block first fails to allocate can
// depend on the thread count. Any other exception is rethrown unchanged.
// The reductions use this for every exception thrown by their callbacks, so
// an exception is never reported as a data failure (a callback returning
// false).
[[noreturn]] inline void rethrow_region_exception(const std::exception_ptr& e)
{
    try {
        std::rethrow_exception(e);
    } catch (const std::bad_alloc&) {
        throw std::runtime_error(
            "not enough memory: an allocation failed inside a parallel computation");
    }
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
// block is rethrown on the calling thread by rethrow_region_exception, which
// is the exception a serial pass would have thrown first.
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
        if (e) rethrow_region_exception(e);
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
// reset, add, and make_work must not call the R API. A block fails when add
// returns false (a data failure) or throws. Each lane stops at its own first
// failure and the other lanes run on, so the smallest failing block index is
// the same for every thread count. If the smallest failing block threw, its
// exception is rethrown on the calling thread after the region by
// rethrow_region_exception (an allocation failure becomes a memory error),
// as a serial pass would have stopped there; an exception from reset or
// make_work (not tied to a block) is always rethrown. Otherwise returns false
// if any block failed; out is then left unchanged and, when failed_block is
// not null, *failed_block is set to the smallest failing block index
// (0-based). On success *failed_block is -1.
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
    // The exception of a lane's first failing block, if it threw, and an
    // exception from reset or make_work of each lane's thread.
    std::vector<std::exception_ptr> lane_exc(nlanes), setup_exc(nlanes);
    const int nt = std::min(threads_for(nblocks), nlanes);
    // One workspace per thread, made by make_work() on that thread in the
    // first round and kept for the later rounds.
    using Work = decltype(make_work());
    std::vector<std::unique_ptr<Work>> works(static_cast<std::size_t>(std::max(nt, 1)));

    // The blocks are processed in rounds of kLanes (8) blocks, one per lane,
    // with Rcpp::checkUserInterrupt() on the calling thread after each round,
    // so a long pass can be interrupted within about one round. Lane l still
    // receives blocks l, l + 8, ... in increasing order, so the result is
    // that of a single pass. A round with a failing block is the last: the
    // blocks of later rounds are all larger than any block of this round.
    for (Eigen::Index first = 0; first < nblocks; first += kLanes) {
        const int count = static_cast<int>(std::min<Eigen::Index>(kLanes, nblocks - first));
        const bool initial = first == 0;
        const int ntr = std::min(nt, count);
        (void) ntr;
#ifdef _OPENMP
#pragma omp parallel num_threads(ntr) if (ntr > 1)
#endif
        {
            const int t = current_thread();
            const int T = team_size();
            note_team_size();
            std::exception_ptr setup;
            if (initial) {
                try {
                    for (int l = t; l < nlanes; l += T) reset(lanes[l]);
                } catch (...) {
                    setup = std::current_exception();
                }
            }
            if (!setup) {
                try {
                    std::unique_ptr<Work>& work = works[static_cast<std::size_t>(t)];
                    if (!work) work.reset(new Work(make_work()));
                    for (int l = t; l < count; l += T) {
                        const Eigen::Index b = first + l;
                        bool ok = false;
                        try {
                            ok = add(b, lanes[l], *work);
                        } catch (...) {
                            lane_exc[l] = std::current_exception();
                            ok = false;
                        }
                        if (!ok) lane_fail[l] = b;
                    }
                } catch (...) {
                    setup = std::current_exception();
                }
            }
            if (setup)
                for (int l = t; l < nlanes; l += T) setup_exc[l] = setup;
        }

        Rcpp::checkUserInterrupt();
        for (int l = 0; l < nlanes; ++l)
            if (setup_exc[l]) rethrow_region_exception(setup_exc[l]);
        Eigen::Index first_fail = -1;
        int first_lane = -1;
        for (int l = 0; l < nlanes; ++l)
            if (lane_fail[l] >= 0 && (first_fail < 0 || lane_fail[l] < first_fail)) {
                first_fail = lane_fail[l];
                first_lane = l;
            }
        if (first_fail >= 0) {
            if (lane_exc[first_lane]) rethrow_region_exception(lane_exc[first_lane]);
            if (failed_block) *failed_block = first_fail;
            return false;
        }
    }
    for (int l = 0; l < nlanes; ++l) combine(out, static_cast<const Acc&>(lanes[l]));
    return true;
}

// Lane reduction with a serial stage: as lane_reduce_into, but the blocks
// are processed in rounds of kLanes (8) blocks, and stage(first, count) is
// called on the calling thread, outside any parallel region, before each
// round, for its blocks [first, first + count) (first is a multiple of 8, so
// block b of a round is its slot b - first = b mod 8). The stage may call the
// R API (for example R's math library) and may itself run parallel_blocks;
// it prepares inputs that add then reads. Each round runs in its own
// parallel region, in which lane l (block first + l) is processed by thread
// l mod T. Lane l receives the blocks l, l + 8, ... in increasing order and
// the lanes are combined in lane order, so the result is that of
// lane_reduce_into with the same callbacks.
//
// make_work() is called once per running thread in each round. Failures are
// handled as in lane_reduce_into; since the blocks of later rounds are all
// larger than any block of the current round, the reduction stops after the
// first round with a failing block, without staging further rounds.
// Rcpp::checkUserInterrupt() is called after each round. An exception thrown
// by stage propagates from this function.
template <class Acc, class Reset, class MakeWork, class Stage, class Add, class Combine>
bool lane_reduce_staged(Eigen::Index nblocks, Acc& out, std::vector<Acc>& lanes, Reset reset,
                        MakeWork make_work, Stage stage, Add add, Combine combine,
                        Eigen::Index* failed_block = nullptr)
{
    if (failed_block) *failed_block = -1;
    if (nblocks <= 0) return true;
    const int nlanes = nblocks < kLanes ? static_cast<int>(nblocks) : kLanes;
    if (static_cast<int>(lanes.size()) < nlanes) lanes.resize(nlanes);
    std::vector<Eigen::Index> lane_fail(nlanes, -1);
    std::vector<std::exception_ptr> lane_exc(nlanes), setup_exc(nlanes);
    const int nt = std::min(threads_for(nblocks), nlanes);
    (void) nt;
    for (Eigen::Index first = 0; first < nblocks; first += kLanes) {
        const int count = static_cast<int>(std::min<Eigen::Index>(kLanes, nblocks - first));
        stage(first, static_cast<Eigen::Index>(count));
        const bool initial = first == 0;
        const int ntr = std::min(nt, count);
        (void) ntr;
#ifdef _OPENMP
#pragma omp parallel num_threads(ntr) if (ntr > 1)
#endif
        {
            const int t = current_thread();
            const int T = team_size();
            note_team_size();
            std::exception_ptr setup;
            if (initial) {
                try {
                    for (int l = t; l < nlanes; l += T) reset(lanes[l]);
                } catch (...) {
                    setup = std::current_exception();
                }
            }
            if (!setup) {
                try {
                    auto work = make_work();
                    for (int l = t; l < count; l += T) {
                        const Eigen::Index b = first + l;
                        bool ok = false;
                        try {
                            ok = add(b, lanes[l], work);
                        } catch (...) {
                            lane_exc[l] = std::current_exception();
                            ok = false;
                        }
                        if (!ok) lane_fail[l] = b;
                    }
                } catch (...) {
                    setup = std::current_exception();
                }
            }
            if (setup)
                for (int l = t; l < nlanes; l += T) setup_exc[l] = setup;
        }
        Rcpp::checkUserInterrupt();
        for (int l = 0; l < nlanes; ++l)
            if (setup_exc[l]) rethrow_region_exception(setup_exc[l]);
        Eigen::Index first_fail = -1;
        int first_lane = -1;
        for (int l = 0; l < nlanes; ++l)
            if (lane_fail[l] >= 0 && (first_fail < 0 || lane_fail[l] < first_fail)) {
                first_fail = lane_fail[l];
                first_lane = l;
            }
        if (first_fail >= 0) {
            if (lane_exc[first_lane]) rethrow_region_exception(lane_exc[first_lane]);
            if (failed_block) *failed_block = first_fail;
            return false;
        }
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
// the region. work must hold at least min(kGroupBlocks, nblocks) slots.
//
// A block fails when prepare returns false (a data failure) or throws. Every
// slot of a group is prepared even when another fails, so the smallest
// failing block is the same for every thread count: if it threw, its
// exception is rethrown on the calling thread after the region by
// rethrow_region_exception (an allocation failure becomes a memory error);
// otherwise tile_reduce returns false, the remaining groups are skipped, and
// when failed_block is not null *failed_block is set to that block index
// (0-based). An exception thrown by add_tile (for example std::bad_alloc
// from a temporary) is rethrown in the same way, that of the smallest tile
// index first; add_tile never reports a data failure. On success
// *failed_block is set to -1.
//
// tile_reduce_staged is tile_reduce with a serial stage: stage(g, gsize) is
// called on the calling thread, outside any parallel region, before the
// group of blocks [g, g + gsize) is prepared (g is a multiple of 8, so block
// b of a group is its slot b - g). The stage may call the R API and may
// itself run parallel_blocks; it prepares inputs that prepare then reads. An
// exception thrown by stage propagates from tile_reduce_staged.
template <class Work, class Stage, class Prepare, class AddTile, class AddOrdered>
bool tile_reduce_staged(Eigen::Index nblocks, Eigen::Index ntiles, std::vector<Work>& work,
                        Stage stage, Prepare prepare, AddTile add_tile, AddOrdered add_ordered,
                        Eigen::Index* failed_block = nullptr)
{
    if (failed_block) *failed_block = -1;
    if (nblocks <= 0) return true;
    if (static_cast<Eigen::Index>(work.size()) <
        std::min<Eigen::Index>(kGroupBlocks, nblocks))
        Rcpp::stop("tile_reduce: work must hold kGroupBlocks slots");

    const int nt = threads_for(nblocks);
    std::vector<char> slot_fail(kGroupBlocks, 0);
    std::vector<std::exception_ptr> slot_exc(kGroupBlocks);
    std::vector<std::exception_ptr> tile_exc(static_cast<std::size_t>(std::max<Eigen::Index>(ntiles, 0)));

    for (Eigen::Index g = 0; g < nblocks; g += kGroupBlocks) {
        const int gsize = static_cast<int>(
            std::min<Eigen::Index>(kGroupBlocks, nblocks - g));

        stage(g, static_cast<Eigen::Index>(gsize));
        std::fill(slot_fail.begin(), slot_fail.end(), 0);
        std::fill(slot_exc.begin(), slot_exc.end(), std::exception_ptr());
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
                slot_exc[j] = std::current_exception();
                ok = false;
            }
            if (!ok) slot_fail[j] = 1;
        }
        for (int j = 0; j < gsize; ++j) {
            if (slot_fail[j]) {
                Rcpp::checkUserInterrupt();
                if (slot_exc[j]) rethrow_region_exception(slot_exc[j]);
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
                tile_exc[static_cast<std::size_t>(t)] = std::current_exception();
                tile_failed.store(true, std::memory_order_relaxed);
            }
        }
        if (tile_failed.load()) {
            Rcpp::checkUserInterrupt();
            for (const auto& e : tile_exc)
                if (e) rethrow_region_exception(e);
        }

        for (int j = 0; j < gsize; ++j) add_ordered(static_cast<const Work&>(work[j]));
        Rcpp::checkUserInterrupt();
    }
    return true;
}

// Column-tile reduction without a stage (see tile_reduce_staged).
template <class Work, class Prepare, class AddTile, class AddOrdered>
bool tile_reduce(Eigen::Index nblocks, Eigen::Index ntiles, std::vector<Work>& work,
                 Prepare prepare, AddTile add_tile, AddOrdered add_ordered,
                 Eigen::Index* failed_block = nullptr)
{
    return tile_reduce_staged(nblocks, ntiles, work, [](Eigen::Index, Eigen::Index) {},
                              prepare, add_tile, add_ordered, failed_block);
}

} // namespace fglm_par

#endif
