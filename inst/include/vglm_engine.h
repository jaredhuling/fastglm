#ifndef FASTGLM_VGLM_ENGINE_H
#define FASTGLM_VGLM_ENGINE_H
#include <RcppEigen.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
#include "vglm_family.h"
#include "vglm_data.h"
#include "parallel_reduce.h"
namespace fglm_vglm {

// Fisher scoring for vector GLMs (design document, Sections 5 and 7).
//
// The linear predictors of row i are eta_i = sum_k H_k beta_k x_ik + o_i,
// where H_k is the M by r_k constraint matrix of design column k. The
// coefficient vector stacks beta_1, ..., beta_p (term-major). Each pass over
// the data accumulates, for every pair a <= b of linear predictors,
//   G_ab = X_a' diag(W[, ab]) X_b   and   g_a = X_a' u[, a],
// and maps them to the information I and score U through the H_k.

// ---------------------------------------------------------------------------
// Per-block kernels on a design view. The engine reaches the design of a
// block only through these, so each data representation supplies one
// overload of each (dense views, Eigen::Ref; sparse views, SparseVglmView)
// and every engine code path is shared.

// Whether two design views refer to the same data, so that the products for
// every linear predictor can be formed at once.
inline bool vglm_same_view(const Eigen::Ref<const MatrixXd>& a,
                           const Eigen::Ref<const MatrixXd>& b) {
    return a.data() == b.data() && a.rows() == b.rows() && a.cols() == b.cols() &&
        a.outerStride() == b.outerStride();
}
inline bool vglm_same_view(const SparseVglmView& a, const SparseVglmView& b) {
    return a.t.outerIndexPtr() == b.t.outerIndexPtr() &&
        a.t.innerIndexPtr() == b.t.innerIndexPtr() && a.t.valuePtr() == b.t.valuePtr() &&
        a.rows() == b.rows() && a.cols() == b.cols();
}

// out = X B, for the b by p design view X.
template <class Rhs, class Out>
inline void vglm_mul(const Eigen::Ref<const MatrixXd>& X, const Rhs& B, Out&& out) {
    out.noalias() = X * B;
}
template <class Rhs, class Out>
inline void vglm_mul(const SparseVglmView& X, const Rhs& B, Out&& out) {
    out.noalias() = X.t.transpose() * B;
}

// out += X' B.
template <class Rhs, class Out>
inline void vglm_tmul_add(const Eigen::Ref<const MatrixXd>& X, const Rhs& B, Out&& out) {
    out.noalias() += X.transpose() * B;
}
template <class Rhs, class Out>
inline void vglm_tmul_add(const SparseVglmView& X, const Rhs& B, Out&& out) {
    out.noalias() += X.t * B;
}

// out[i] = sum_j Y(i, j) X(i, j): the row-wise inner products of a dense
// b by p matrix Y with the design view X.
template <class Out>
inline void vglm_row_dots(const MatrixXd& Y, const Eigen::Ref<const MatrixXd>& X, Out&& out) {
    out = Y.cwiseProduct(X).rowwise().sum();
}
template <class Out>
inline void vglm_row_dots(const MatrixXd& Y, const SparseVglmView& X, Out&& out) {
    for (int i = 0; i < X.rows(); ++i) {
        double s = 0;
        for (SparseVglmView::Map::InnerIterator it(X.t, i); it; ++it) s += Y(i, it.row()) * it.value();
        out[i] = s;
    }
}

// row = row i of the design view, as a dense row (one row densified at a
// time for the structural QR).
template <class Row>
inline void vglm_copy_row(const Eigen::Ref<const MatrixXd>& X, int i, Row&& row) {
    row = X.row(i);
}
template <class Row>
inline void vglm_copy_row(const SparseVglmView& X, int i, Row&& row) {
    row.setZero();
    for (SparseVglmView::Map::InnerIterator it(X.t, i); it; ++it) row[it.row()] = it.value();
}

// Whether every linear predictor of a chunk uses the same design view.
template <class Chunk>
inline bool vglm_shared_view(const Chunk& chunk, int M) {
    bool shared = true;
    for (int a = 1; a < M && shared; ++a) shared = vglm_same_view(chunk.design[0], chunk.design[a]);
    return shared;
}

// The b by M linear predictors of a chunk for the p by M coefficient matrix
// coef (column a holds the coefficients of linear predictor a), plus offsets.
template <class Chunk>
inline void vglm_linear(const Chunk& chunk, bool shared, const MatrixXd& coef, MatrixXd& out) {
    const int M = static_cast<int>(coef.cols());
    if (shared) vglm_mul(chunk.design[0], coef, out);
    else {
        out.resize(chunk.rows(), M);
        for (int a = 0; a < M; ++a) vglm_mul(chunk.design[a], coef.col(a), out.col(a));
    }
    out += chunk.offset;
}

// G += Xa' diag(w) Xb, with work as scratch.
inline void vglm_add_cross(MatrixXd& G, const Eigen::Ref<const MatrixXd>& Xa,
                           const Eigen::Ref<const VectorXd>& w,
                           const Eigen::Ref<const MatrixXd>& Xb, MatrixXd& work) {
    work.noalias() = w.asDiagonal() * Xb;
    G.noalias() += Xa.transpose() * work;
}
// Sparse views: a sparse product Xa' (diag(w) Xb) added to the dense G; the
// scaled copy of Xb' has the nonzeros of the block. work is not used.
inline void vglm_add_cross(MatrixXd& G, const SparseVglmView& Xa,
                           const Eigen::Ref<const VectorXd>& w, const SparseVglmView& Xb,
                           MatrixXd& work) {
    (void)work;
    const Eigen::SparseMatrix<double> scaled = Xb.t * w.asDiagonal();
    G += Xa.t * scaled.transpose();
}

// ---------------------------------------------------------------------------
// Constraint layout: the position of each column's coefficients and the map
// between coefficients and the p by M matrix B with B(k, a) = (H_k beta_k)[a].

struct VglmLayout {
    std::vector<MatrixXd> H;
    std::vector<int> start;   // first coefficient of column k
    int p = 0, M = 0, P = 0;
    bool identity = true;     // every H_k is I_M and the mapping is skipped

    VglmLayout(const std::vector<MatrixXd>& constraints, int M_, bool force_mapping)
        : H(constraints), start(constraints.size()), p(static_cast<int>(constraints.size())), M(M_) {
        for (int k = 0; k < p; ++k) {
            if (H[k].rows() != M || H[k].cols() < 1 || !H[k].allFinite())
                Rcpp::stop("each constraint matrix must be finite with K - 1 rows and at least one column");
            start[k] = P;
            P += static_cast<int>(H[k].cols());
            if (H[k].cols() != M || !H[k].isIdentity(0.0)) identity = false;
        }
        if (force_mapping) identity = false;
    }

    int r(int k) const { return static_cast<int>(H[k].cols()); }

    MatrixXd predictors(const VectorXd& beta) const {
        MatrixXd B(p, M);
        for (int k = 0; k < p; ++k) {
            if (identity) B.row(k) = beta.segment(start[k], M).transpose();
            else B.row(k).noalias() = (H[k] * beta.segment(start[k], r(k))).transpose();
        }
        return B;
    }

    // Packed G (each p by p) and g (p by M) to the information and score.
    // Only blocks (k, l) with k <= l are formed; the rest are mirrored, and
    // diagonal blocks are symmetrized, so both branches read the same
    // entries of G and agree exactly when every H_k is the identity.
    void map(const std::vector<MatrixXd>& G, const MatrixXd& g, MatrixXd& I, VectorXd& U) const {
        I.setZero(P, P);
        U.setZero(P);
        MatrixXd S(M, M), block;
        for (int k = 0; k < p; ++k) {
            for (int l = k; l < p; ++l) {
                for (int b = 0; b < M; ++b)
                    for (int a = 0; a < M; ++a)
                        S(a, b) = a <= b ? G[pack_index(a, b, M)](k, l) : G[pack_index(b, a, M)](l, k);
                if (identity) block = S;
                else block.noalias() = H[k].transpose() * S * H[l];
                if (k == l) block = (0.5 * (block + block.transpose())).eval();
                I.block(start[k], start[l], r(k), r(l)) = block;
                if (k != l) I.block(start[l], start[k], r(l), r(k)) = block.transpose();
            }
        }
        score_map(g, U);
    }

    // map for the diagonal blocks alone: Gd[a] (p by p, a = 0, ..., M - 1)
    // holds G_aa and the blocks with a != b are 0, so block (k, l) of I is
    // H_k' diag(s) H_l with s[a] = Gd[a](k, l). As in map, only k <= l is
    // formed and read, and diagonal blocks are symmetrized.
    void map_diagonal(const std::vector<MatrixXd>& Gd, MatrixXd& I) const {
        I.setZero(P, P);
        VectorXd sd(M);
        MatrixXd block;
        for (int k = 0; k < p; ++k) {
            for (int l = k; l < p; ++l) {
                for (int a = 0; a < M; ++a) sd[a] = Gd[a](k, l);
                if (identity) block = sd.asDiagonal();
                else block.noalias() = H[k].transpose() * sd.asDiagonal() * H[l];
                if (k == l) block = (0.5 * (block + block.transpose())).eval();
                I.block(start[k], start[l], r(k), r(l)) = block;
                if (k != l) I.block(start[l], start[k], r(l), r(k)) = block.transpose();
            }
        }
    }

    // A p by M matrix g with g(k, a) = sum_i x_aik v_i[a] to the coefficient
    // scale, sum_i X_i*' v_i: entry (k, s) is sum_a H_k[a, s] g(k, a).
    void score_map(const MatrixXd& g, VectorXd& U) const {
        U.setZero(P);
        for (int k = 0; k < p; ++k) {
            if (identity) U.segment(start[k], M) = g.row(k).transpose();
            else U.segment(start[k], r(k)).noalias() = H[k].transpose() * g.row(k).transpose();
        }
    }

    // The reverse of map for a P by P matrix V (an inverse information):
    // the packed p by p matrices F with F[pack_index(a, b)](k, l) =
    // (H_k V_kl H_l')[a, b], where V_kl is the block of V for columns k and
    // l. Then X_i* V X_i*' has entry (a, b) equal to x_ai' F_ab x_bi, with
    // x_ai row i of the design view of linear predictor a.
    void eta_inverse(const MatrixXd& V, std::vector<MatrixXd>& F) const {
        F.assign(packed_size(M), MatrixXd(p, p));
        MatrixXd E;
        for (int k = 0; k < p; ++k)
            for (int l = 0; l < p; ++l) {
                if (identity) E = V.block(start[k], start[l], M, M);
                else E.noalias() = H[k] * V.block(start[k], start[l], r(k), r(l)) * H[l].transpose();
                for (int b = 0; b < M; ++b)
                    for (int a = 0; a <= b; ++a) F[pack_index(a, b, M)](k, l) = E(a, b);
            }
    }
};

// ---------------------------------------------------------------------------
// Engine options. force_mapping (test only) applies the constraint mapping
// even when every H_k is the identity. force_pairwise (test only) accumulates
// the information pair by pair even when the model type provides the
// diagonal-minus-low-rank structure; force_lowrank (test only) uses that
// structure even below vglm_lowrank_min_M (force_pairwise takes precedence).
// lowrank_cap is the largest number of entries of the low-rank matrix Z of
// one sub-block of rows (32 MB of doubles by default; set smaller only by
// tests). product_cache_cap is the largest number of doubles the
// conjugate gradient solve keeps for the per-row information terms of the
// current estimate (VglmProductCache; 256 MB by default): within one solve
// beta is fixed, so the model type is evaluated in the first product only.
// Above the cap every product evaluates it again; 0 (test only) never caches.
//
// solver selects the step of each iteration: "exact" (the default) solves
// I delta = U from the factored information; "cg" is the truncated Newton
// solver of vglm_fit (preconditioned conjugate gradients on
// information-vector products, design Section 5). max_products caps the
// products of one conjugate gradient solve; a negative value means
// min(2 rank, 250), with rank the number of coefficients that are not
// aliased.

struct VglmOptions {
    bool force_mapping = false;
    bool force_pairwise = false;
    bool force_lowrank = false;
    std::int64_t lowrank_cap = 4194304;
    std::int64_t product_cache_cap = 33554432;
    std::string solver = "exact";
    int max_products = -1;
    int cg_reject_confirmation = 0;   // test only, see vglm_fit
    int cg_reject_steps = 0;          // test only, see vglm_fit
};

// Fewest linear predictors M at which the low-rank accumulation is used by
// default. Both paths do the same arithmetic, M(M + 1)/2 p^2 multiply-adds
// per row, so the low-rank path gains only from its larger matrix kernels
// and loses the cost of forming Z; below this M the pairwise path is as fast
// or faster. Set from a timing sweep of full fits (dense, n = 20000, M in
// 1..10 and 15, p in 4, 10, 40, 100): the low-rank to pairwise time ratio
// was at most 1.02 from M = 6 on at every p (0.84 to 0.89 at p = 4 and 10),
// and 1.03 to 1.04 at M = 5 for p = 40 and 100, about 1.1 to 1.3 below that.
const int vglm_lowrank_min_M = 6;

// The rank r of the low-rank accumulation in full passes of vglm_pass, or 0
// for the pairwise products. (VGLM_DIAGONAL passes and vglm_info_product
// have their own rule: they never form Z, so they use the low-rank terms
// whenever the model type provides them.)
inline int vglm_lowrank_rank(const VectorFamily& fam, const VglmOptions& opts, int M) {
    return opts.force_pairwise || (M < vglm_lowrank_min_M && !opts.force_lowrank)
        ? 0 : fam.information_rank();
}

// Low-rank accumulation for a block whose linear predictors share the dense
// design view X (b by p), with W_i = diag(d_i) - C_i C_i' and C_i of M by r
// (column s * M + a of C holds C_i[a, s]). Adds to the lower triangle of the
// pM by pM matrix S (index a * p + j) the diagonal terms X' diag(d[, a]) X,
// in the diagonal block of a, and -Z'Z, where row (s, i) of Z is
// C_i[, s]' kron x_i'. Both are symmetric rank updates, so the arithmetic
// equals that of the pairwise products. The diagonal terms are rank updates
// of diag(sqrt(d[, a])) X, so they need d >= 0; if some d of the block is
// negative they are added to the packed G[pack_index(a, a)] by
// vglm_add_cross instead. Rows are taken in sub-blocks so that Z has at most
// cap entries (one row at a time when a row alone exceeds the cap). Z keeps
// its allocation across calls and only grows; vglm_lowrank_unpack moves S
// into G.
inline void vglm_add_lowrank(std::vector<MatrixXd>& G, MatrixXd& S,
                             const Eigen::Ref<const MatrixXd>& X, const MatrixXd& d,
                             const MatrixXd& C, int r, std::int64_t cap, MatrixXd& Z,
                             MatrixXd& work) {
    typedef Eigen::Index Index;
    const Index b = X.rows(), p = X.cols(), M = d.cols(), pM = p * M;
    if ((d.array() >= 0).all()) {
        for (Index a = 0; a < M; ++a) {
            work.noalias() = d.col(a).cwiseSqrt().asDiagonal() * X;
            S.block(a * p, a * p, p, p).template selfadjointView<Eigen::Lower>()
                .rankUpdate(work.transpose(), 1.0);
        }
    } else {
        for (Index a = 0; a < M; ++a)
            vglm_add_cross(G[pack_index(static_cast<int>(a), static_cast<int>(a),
                                        static_cast<int>(M))],
                           X, d.col(a), X, work);
    }
    const std::int64_t width = static_cast<std::int64_t>(r) * pM;
    const Index rows = static_cast<Index>(
        std::max<std::int64_t>(1, std::min<std::int64_t>(b, cap / width)));
    if (Z.rows() < rows * r || Z.cols() != pM) Z.resize(rows * r, pM);
    for (Index first = 0; first < b; first += rows) {
        const Index bs = std::min(rows, b - first);
        for (Index s = 0; s < r; ++s)
            for (Index a = 0; a < M; ++a)
                Z.block(s * bs, a * p, bs, p).noalias() =
                    C.col(s * M + a).segment(first, bs).asDiagonal() * X.middleRows(first, bs);
        S.template selfadjointView<Eigen::Lower>().rankUpdate(Z.topRows(bs * r).transpose(), -1.0);
    }
}

// Adds the pM by pM matrix S of vglm_add_lowrank (lower triangle) to the
// packed G: G[pack_index(a, b)](k, l) += S(b * p + l, a * p + k) for a <= b.
inline void vglm_lowrank_unpack(const MatrixXd& S, int p, int M, std::vector<MatrixXd>& G) {
    for (int bb = 0; bb < M; ++bb)
        for (int a = 0; a <= bb; ++a) {
            MatrixXd& Gq = G[pack_index(a, bb, M)];
            for (int l = 0; l < p; ++l)
                for (int k = 0; k < p; ++k) {
                    const int row = bb * p + l, col = a * p + k;
                    Gq(k, l) += row >= col ? S(row, col) : S(col, row);
                }
        }
}

// ---------------------------------------------------------------------------
// One pass over the data at the coefficients beta.

struct VglmState {
    double loglik = 0;
    MatrixXd I;
    VectorXd U;
    double pmin = 1, pmax = 0;  // extreme fitted probabilities over rows that count
    bool lowrank = false;       // some block used the low-rank accumulation
    bool full = true;           // I is the full information (VGLM_FULL)
    // VGLM_DIAGONAL under an identity layout only (I is then empty): the M
    // symmetric p by p blocks G_aa, with Gdiag[a](k, l) the information entry
    // of coefficients (k, a) and (l, a). Empty otherwise.
    std::vector<MatrixXd> Gdiag;
};

enum VglmPassStatus { VGLM_PASS_OK, VGLM_PASS_INVALID, VGLM_PASS_SHRINK };

// What a pass accumulates for the information. VGLM_FULL gives the mapped
// information I. VGLM_DIAGONAL accumulates only the M blocks G_aa (a == b),
// at order b p^2 M per block instead of b p^2 M(M + 1)/2 and with M p by p
// matrices of memory instead of M(M + 1)/2: the information with the
// off-diagonal predictor blocks dropped, the matrix the preconditioner of
// the truncated Newton solver factors. Under an identity layout that matrix
// is the information with the entries for coefficients (k, a), (l, b),
// a != b, set to 0; it is block diagonal, so it is kept as the M blocks in
// out.Gdiag and out.I is left empty (no P by P matrix). Under other layouts
// it is mapped to the P by P matrix out.I. The log likelihood, score,
// extreme probabilities, and structure are computed exactly as in VGLM_FULL
// (the model type is evaluated through the same hook), so they are identical.
enum VglmPassMode { VGLM_FULL, VGLM_DIAGONAL };

// Whether row i of a chunk enters the likelihood: positive prior weight and,
// for a grouped response, a positive total count.
template <class Chunk>
inline bool vglm_row_counts(const Chunk& chunk, int i) {
    if (!(chunk.weights[i] > 0)) return false;
    return chunk.codes != nullptr || chunk.counts.row(i).sum() > 0;
}

// ---------------------------------------------------------------------------
// Threading (OpenMP, through the deterministic reductions of
// parallel_reduce.h). A pass reaches the data as segments of rows (the whole
// in-memory data, or one block of a streamed source at a time, read on the
// calling thread) and splits each segment into fixed row blocks. The blocks
// are summed by one of two schemes, chosen from the dimensions alone:
//
// * lanes (fglm_par::lane_reduce_into): each of up to 8 lane accumulators
//   holds the information terms, score, log likelihood, extreme
//   probabilities, and design factor of the blocks l, l + 8, ...; the lanes
//   are then added in lane order. The blocks have vglm_lane_rows() rows
//   (block_size unless the dimensions are large).
// * column tiles (fglm_par::tile_reduce), for a full pass when the
//   information has more than kLaneMaxDim rows (P = p M > 1024), or for a
//   diagonal pass when its M blocks of p by p entries exceed kLaneMaxDim^2:
//   blocks of min(block_size, tile_block_rows(P)) rows are prepared in
//   groups of 8; the pM by pM lower triangle S (or each G_aa) is split into
//   column tiles of width 64 within each linear predictor's columns, each
//   tile adding the group's blocks in order; the score, log likelihood,
//   extreme probabilities, and design factor are added on the calling thread
//   in block order.
//
// Every choice is a fixed function of p, M, the block size, and the segment
// lengths, so results are identical for every thread count. Inside a
// parallel region the engine calls only the C++ model type and Eigen (no R
// API). An exception thrown there (for example std::bad_alloc) is captured
// by the reductions of parallel_reduce.h and rethrown on the calling thread
// when its block is the smallest failing one, which is the block a serial
// pass would have stopped at; an allocation failure becomes a memory error.
// It is never reported as an invalid pass (VGLM_PASS_INVALID), which is kept
// for data the model type rejects.

// Largest number of doubles of the low-rank matrix Z of one thread (16 MB),
// whatever lowrank_cap allows, so that up to 8 threads stay within 128 MB of
// such buffers.
const std::int64_t vglm_thread_zcap = 2097152;

// Rows per block of the lane scheme: block_size, capped as the GLM lanes
// are (ruling R17) at max(tile_block_rows(d), 2^21 / d) rows for the
// dimension d = max(P, M(M + 1)/2), so that a thread's scratch of d columns
// (the low-rank Z, of P columns, or the packed A of the bias reduction, of
// M(M + 1)/2) takes at most 16 MB. A fixed function of the dimensions and
// the block size; it is block_size whenever block_size <= 2^21 / d (for
// example at the default block_size of 8192 whenever d <= 256).
inline Eigen::Index vglm_lane_rows(Eigen::Index block_size, int P, int Q) {
    const Eigen::Index d = std::max<Eigen::Index>(1, std::max(P, Q));
    const Eigen::Index cap =
        std::max<Eigen::Index>(fglm_par::tile_block_rows(d), Eigen::Index(2097152) / d);
    return std::max<Eigen::Index>(1, std::min<Eigen::Index>(block_size, cap));
}

// Accumulator of a pass (a lane, or the total of a pass).
struct VglmAcc {
    std::vector<MatrixXd> G;   // packed G_ab (full) or G_aa (diagonal), p by p each
    bool G_used = false;       // G holds this pass's sums (sized and zeroed)
    MatrixXd S;                // pM by pM lower triangle
    bool S_used = false;       // S holds sums
    bool lowrank = false;      // some block used the low-rank accumulation
    MatrixXd g;                // p by M score sums
    double sum = 0, err = 0;   // log likelihood, compensated
    double pmin = 1, pmax = 0;
    MatrixXd R;                // design factor (structure passes)
    bool R_used = false;
};

// Sizes and zeroes the G of a, with count matrices of p by p, the first time
// a pass uses it.
inline void vglm_use_G(VglmAcc& a, int count, int p) {
    if (a.G_used) return;
    if (static_cast<int>(a.G.size()) != count) a.G.assign(count, MatrixXd());
    for (MatrixXd& m : a.G) m.setZero(p, p);
    a.G_used = true;
}

// sum += x with the rounding error of the addition added to err (Neumaier).
inline void vglm_add_compensated(double& sum, double& err, double x) {
    const double next = sum + x;
    err += std::abs(sum) >= std::abs(x) ? (sum - next) + x : (x - next) + sum;
    sum = next;
}

// Scratch of one thread (lanes) or one group slot (tiles). Nothing in it
// carries a result from one block to the next except the fields of a block
// prepared for the tile scheme, which are read before the slot is reused.
struct VglmWork {
    MatrixXd eta, eta_prev, direction, u, W, d, C, P, work, stack, Wdiag, Z, Xd, gb;
    MatrixXd E, two, s, tc, A, t, XF;
    VectorXd proj;
    // A block prepared by the tile scheme.
    int first = 0, rows = 0;
    double ll = 0, pmin = 1, pmax = 0;
    const double* x = nullptr;   // its dense rows (the design view, or Xd)
    Eigen::Index xstride = 0;
    int kind = 0;                // see vglm_pass
};

// Accumulators and scratch kept for a fit, so that its passes do not
// allocate (and page in) them again: the lane accumulators (at most 8 of
// at most 8 MB each in a full pass, P <= 1024), the total, and the thread
// or group-slot scratch.
struct VglmWorkspace {
    std::vector<VglmAcc> lanes;
    VglmAcc out;
    std::vector<VglmWork> slots;
};

// The scratch of the calling thread in a lane reduction: its slot of the
// workspace, or (should the thread index exceed the slots) its own.
struct VglmWorkHandle {
    VglmWork* w = nullptr;
    std::unique_ptr<VglmWork> own;
};

// Sizes the workspace slots for a lane reduction over nblocks blocks (one per
// running thread) on the calling thread; returns the number of slots.
inline Eigen::Index vglm_lane_slots(VglmWorkspace& ws, Eigen::Index nblocks) {
    const Eigen::Index nlanes = std::min<Eigen::Index>(nblocks, fglm_par::kLanes);
    const Eigen::Index ns =
        std::min<Eigen::Index>(nlanes, std::max(1, fglm_par::thread_setting()));
    if (static_cast<Eigen::Index>(ws.slots.size()) < ns) ws.slots.resize(ns);
    if (static_cast<Eigen::Index>(ws.lanes.size()) < nlanes) ws.lanes.resize(nlanes);
    return ns;
}

inline VglmWorkHandle vglm_thread_work(VglmWorkspace& ws, Eigen::Index nslots) {
    VglmWorkHandle h;
    const Eigen::Index id = fglm_par::current_thread();
    if (id < nslots) h.w = &ws.slots[id];
    else {
        h.own.reset(new VglmWork());
        h.w = h.own.get();
    }
    return h;
}

// Stack the rows of a chunk that enter the likelihood under R and replace R
// by the triangular factor of the stack (Householder QR).
template <class Chunk>
inline void vglm_structure_add(const Chunk& chunk, int p, MatrixXd& R, MatrixXd& stack) {
    const int b = chunk.rows();
    int used = 0;
    for (int i = 0; i < b; ++i) used += vglm_row_counts(chunk, i) ? 1 : 0;
    if (used == 0) return;
    stack.resize(p + used, p);
    stack.topRows(p) = R;
    int r = p;
    for (int i = 0; i < b; ++i)
        if (vglm_row_counts(chunk, i)) vglm_copy_row(chunk.design[0], i, stack.row(r++));
    Eigen::HouseholderQR<MatrixXd> qr(stack);
    R = qr.matrixQR().topRows(p).template triangularView<Eigen::Upper>();
}

// R = the triangular factor of [R; R2] (combining design factors).
inline void vglm_structure_combine(MatrixXd& R, const MatrixXd& R2, MatrixXd& stack) {
    const Eigen::Index p = R.cols();
    stack.resize(2 * p, p);
    stack.topRows(p) = R;
    stack.bottomRows(p) = R2;
    Eigen::HouseholderQR<MatrixXd> qr(stack);
    R = qr.matrixQR().topRows(p).template triangularView<Eigen::Upper>();
}

// When previous is given and the model type restricts steps, the pass first
// asks it for the largest valid fraction of the change in eta from previous
// to beta; if that is below 1 the pass stops with VGLM_PASS_SHRINK and
// fraction holds it. When structure is given, the pass also accumulates the
// p by p triangular factor R of the design rows that enter the likelihood
// (R'R = X'X over those rows, by Householder QR of [R; X_block]), from which
// vglm_structural_aliased() finds the structurally aliased coefficients. The
// rows of a block are copied into the dense QR stack by vglm_copy_row, so a
// sparse block is densified one block at a time (memory O(block x p)) and the
// factor is computed from the same numbers as for a dense design. The
// structure uses the shared design view; covariates that differ across
// linear predictors (0.2.0) will need the factor of each view. Under lanes
// each lane keeps its own factor and the lane factors are combined in lane
// order; under tiles the rows are stacked in block order on the calling
// thread.
//
// The outcome is that of a serial pass over the blocks: the first block
// (in row order) at which the step check fails gives VGLM_PASS_SHRINK, with
// fraction the smallest fraction over every block, unless an earlier block
// is invalid (VGLM_PASS_INVALID). The step fractions are computed in a first
// parallel sweep, so only the blocks before the first failing one are
// accumulated.
//
// workspace, when given, holds the accumulators and scratch kept between
// passes (vglm_fit keeps one for the fit).
template <class Data>
VglmPassStatus vglm_pass(const Data& data, const VectorFamily& fam, const VglmLayout& L,
                         const VglmOptions& opts, const VectorXd& beta,
                         const VectorXd* previous, VglmState& out, double& fraction,
                         MatrixXd* structure = nullptr, VglmPassMode mode = VGLM_FULL,
                         VglmWorkspace* workspace = nullptr) {
    typedef typename Data::Segment Segment;
    typedef typename Segment::Chunk Chunk;
    typedef Eigen::Index Index;
    const int p = L.p, M = L.M, Q = packed_size(M), K = fam.K(), PM = p * M;
    const bool full = mode == VGLM_FULL;
    // The low-rank accumulation applies to blocks whose linear predictors
    // share one dense design view; other blocks keep the pairwise products.
    // Below vglm_lowrank_min_M linear predictors the pairwise path is the
    // faster one and is kept unless a test forces the low-rank path.
    const int rank = vglm_lowrank_rank(fam, opts, M);
    // A VGLM_DIAGONAL pass reads only the diagonal W_i[a, a] = d_i[a] -
    // sum_s C_i[a, s]^2, which needs neither Z nor the packed W (of b by
    // M(M + 1)/2 entries), so it uses the low-rank terms for every design
    // view (sparse included) and at every M whenever the model type has them.
    const int diag_rank = opts.force_pairwise ? 0 : fam.information_rank();
    constexpr bool dense_view =
        std::is_same<typename Data::DesignView, Eigen::Ref<const MatrixXd> >::value;
    const bool check_step = previous != nullptr && fam.restricts_step();
    const MatrixXd B = L.predictors(beta);
    MatrixXd Bprev;
    if (check_step) Bprev = L.predictors(*previous);
    const std::int64_t zcap = std::min<std::int64_t>(opts.lowrank_cap, vglm_thread_zcap);

    VglmWorkspace local;
    VglmWorkspace& ws = workspace ? *workspace : local;
    // The accumulation scheme, from the dimensions alone.
    const std::int64_t lane_max = static_cast<std::int64_t>(fglm_par::kLaneMaxDim) *
        fglm_par::kLaneMaxDim;
    const bool lanes = full ? fglm_par::use_lanes(PM)
                            : static_cast<std::int64_t>(M) * p * p <= lane_max;
    // G has M(M + 1)/2 matrices in a full pass and M in a diagonal pass.
    const int Gcount = full ? Q : M;
    // Lanes: S holds the low-rank accumulation (full passes over dense data
    // with rank > 0). Tiles: S holds every dense block of a full pass.
    const bool lowrank_pass = full && dense_view && rank > 0;
    const bool S_pass = full && (lanes ? lowrank_pass : dense_view);
    const Index Rk = lanes ? vglm_lane_rows(data.block_rows(), PM, Q)
                           : std::min<Index>(data.block_rows(), fglm_par::tile_block_rows(PM));
    const Index tp = fglm_par::tile_count(p);

    VglmAcc& acc = ws.out;
    acc.g.setZero(p, M);
    acc.sum = acc.err = 0;
    acc.pmin = 1;
    acc.pmax = 0;
    acc.lowrank = false;
    acc.S_used = false;
    acc.G_used = false;
    vglm_use_G(acc, Gcount, p);
    if (S_pass) {
        acc.S.setZero(PM, PM);
        acc.S_used = true;
    }
    acc.R_used = false;
    if (structure) acc.R.setZero(p, p);

    VglmPassStatus status = VGLM_PASS_OK;
    bool shrinking = false;
    fraction = 1;

    // The per-block work shared by both schemes: eta, validity, the log
    // likelihood ll, the row terms (low-rank d and C when terms, else the
    // packed W), the score sums added to g, the extreme probabilities, and,
    // in a diagonal pass, Wdiag. Returns false for an invalid block.
    auto evaluate = [&](const Chunk& chunk, VglmWork& w, MatrixXd& g, bool& terms,
                        bool& shared, double& ll, double& pmin, double& pmax) -> bool {
        const int b = chunk.rows();
        shared = vglm_shared_view(chunk, M);
        vglm_linear(chunk, shared, B, w.eta);
        const VglmBlock block{w.eta, chunk.codes, chunk.counts, chunk.weights};
        if (!fam.valid(block)) return false;
        ll = fam.loglik(block, nullptr);
        if (!std::isfinite(ll)) return false;
        terms = full ? (dense_view && shared && rank > 0) : diag_rank > 0;
        if (terms) {
            fam.score_information_lowrank(block, w.u, w.d, w.C);
            if (!w.u.allFinite() || !w.d.allFinite() || !w.C.allFinite()) return false;
        } else {
            fam.score_information(block, w.u, w.W);
            if (!w.u.allFinite() || !w.W.allFinite()) return false;
        }
        if (shared) vglm_tmul_add(chunk.design[0], w.u, g);
        else
            for (int a = 0; a < M; ++a) vglm_tmul_add(chunk.design[a], w.u.col(a), g.col(a));
        if (!full) {
            // Only the diagonal entries W_i[a, a] of the row information:
            // d_i[a] - sum_s C_i[a, s]^2 under the low-rank structure.
            if (terms) {
                w.Wdiag = w.d;
                for (int s = 0; s < diag_rank; ++s)
                    w.Wdiag -= w.C.middleCols(s * M, M).cwiseAbs2();
            } else {
                w.Wdiag.resize(b, M);
                for (int a = 0; a < M; ++a) w.Wdiag.col(a) = w.W.col(pack_index(a, a, M));
            }
        }
        // Extreme fitted probabilities over the rows that enter the likelihood.
        fam.probabilities(w.eta, w.P);
        for (int i = 0; i < b; ++i) {
            if (!vglm_row_counts(chunk, i)) continue;
            for (int k = 0; k < K; ++k) {
                pmin = std::min(pmin, w.P(i, k));
                pmax = std::max(pmax, w.P(i, k));
            }
        }
        return true;
    };

    // The information of a block added to the packed G (lanes, and the
    // serial blocks of the tile scheme), as in a serial pass.
    auto add_packed = [&](const Chunk& chunk, VglmWork& w, VglmAcc& a, bool terms) {
        if (!full) {
            for (int q = 0; q < M; ++q)
                vglm_add_cross(a.G[q], chunk.design[q], w.Wdiag.col(q), chunk.design[q], w.work);
            return;
        }
        if (terms) {
            if constexpr (dense_view) {
                // The diagonal terms go to G only if some d is negative.
                if (!(w.d.array() >= 0).all()) vglm_use_G(a, Gcount, p);
                vglm_add_lowrank(a.G, a.S, chunk.design[0], w.d, w.C, rank, zcap, w.Z, w.work);
                a.S_used = true;
                a.lowrank = true;
            }
            return;
        }
        vglm_use_G(a, Gcount, p);
        for (int bb = 0; bb < M; ++bb)
            for (int q0 = 0; q0 <= bb; ++q0) {
                const int q = pack_index(q0, bb, M);
                vglm_add_cross(a.G[q], chunk.design[q0], w.W.col(q), chunk.design[bb], w.work);
            }
    };

    data.segments([&](const Segment& seg) {
        if (status == VGLM_PASS_INVALID) return;
        const int n = seg.n();
        if (n <= 0) return;
        const Index nb = (n + Rk - 1) / Rk;
        Index stop = nb;   // the blocks [0, stop) are accumulated
        if (check_step) {
            std::vector<double> f(static_cast<std::size_t>(nb), 1.0);
            fglm_par::parallel_blocks(n, Rk, [&](Index r, Index k) {
                const Chunk chunk = seg.chunk(static_cast<int>(r), static_cast<int>(k));
                const bool shared = vglm_shared_view(chunk, M);
                MatrixXd eta, eta_prev;
                vglm_linear(chunk, shared, B, eta);
                vglm_linear(chunk, shared, Bprev, eta_prev);
                const MatrixXd direction = eta - eta_prev;
                f[static_cast<std::size_t>(r / Rk)] = fam.max_step_fraction(eta_prev, direction);
            });
            for (Index b = 0; b < nb; ++b) {
                const double fb = f[static_cast<std::size_t>(b)];
                if (!(fb >= 1)) {
                    if (stop == nb) stop = b;
                    fraction = std::min(fraction, std::isfinite(fb) ? std::max(fb, 0.0) : 0.0);
                }
            }
            // After a block that needs a shorter step, a serial pass only
            // checks the step of the remaining blocks.
            if (shrinking) return;
        }
        if (stop > 0) {
            Index failed = -1;
            bool ok;
            if (lanes) {
                const Index nl = std::min<Index>(stop, fglm_par::kLanes);
                const Index ns = vglm_lane_slots(ws, stop);
                // Size the lanes here, so that the resets inside the region
                // do not allocate.
                for (Index l = 0; l < nl; ++l) {
                    VglmAcc& a = ws.lanes[l];
                    a.G_used = false;
                    if (S_pass) a.S.resize(PM, PM);
                    if (!lowrank_pass) vglm_use_G(a, Gcount, p);
                    a.g.resize(p, M);
                    if (structure) a.R.resize(p, p);
                }
                ok = fglm_par::lane_reduce_into(
                    stop, acc, ws.lanes,
                    [&](VglmAcc& a) {
                        a.g.setZero(p, M);
                        a.sum = a.err = 0;
                        a.pmin = 1;
                        a.pmax = 0;
                        a.lowrank = false;
                        a.S_used = false;
                        if (S_pass) a.S.setZero(PM, PM);
                        a.G_used = false;
                        if (!lowrank_pass) vglm_use_G(a, Gcount, p);
                        if (structure) a.R.setZero(p, p);
                        a.R_used = false;
                    },
                    [&]() { return vglm_thread_work(ws, ns); },
                    [&](Index blk, VglmAcc& a, VglmWorkHandle& h) -> bool {
                        VglmWork& w = *h.w;
                        const int first = static_cast<int>(blk * Rk);
                        const Chunk chunk =
                            seg.chunk(first, static_cast<int>(std::min<Index>(Rk, n - first)));
                        bool terms = false, shared = true;
                        double ll = 0;
                        if (!evaluate(chunk, w, a.g, terms, shared, ll, a.pmin, a.pmax))
                            return false;
                        vglm_add_compensated(a.sum, a.err, ll);
                        add_packed(chunk, w, a, terms);
                        if (structure) {
                            vglm_structure_add(chunk, p, a.R, w.stack);
                            a.R_used = true;
                        }
                        return true;
                    },
                    [&](VglmAcc& o, const VglmAcc& a) {
                        if (a.S_used) o.S.template triangularView<Eigen::Lower>() += a.S;
                        o.lowrank = o.lowrank || a.lowrank;
                        if (a.G_used)
                            for (int q = 0; q < Gcount; ++q) o.G[q] += a.G[q];
                        o.g += a.g;
                        vglm_add_compensated(o.sum, o.err, a.sum);
                        o.err += a.err;
                        o.pmin = std::min(o.pmin, a.pmin);
                        o.pmax = std::max(o.pmax, a.pmax);
                        if (a.R_used) {
                            if (!o.R_used) o.R = a.R;
                            else vglm_structure_combine(o.R, a.R, ws.slots[0].stack);
                            o.R_used = true;
                        }
                    },
                    &failed);
            } else {
                // Tile scheme. kind of a prepared block: 0 low-rank (Z and
                // the diagonal terms d), 1 pairwise (W), 2 diagonal pass
                // (Wdiag), all into S or G_aa by tiles; 3 serial (sparse or
                // unshared views), added to G on the calling thread.
                const Index nslots = std::min<Index>(fglm_par::kGroupBlocks, stop);
                if (static_cast<Index>(ws.slots.size()) < nslots) ws.slots.resize(nslots);
                const Index ntiles = static_cast<Index>(M) * tp;
                ok = fglm_par::tile_reduce(
                    stop, ntiles, ws.slots,
                    [&](Index blk, VglmWork& w) -> bool {
                        w.first = static_cast<int>(blk * Rk);
                        w.rows = static_cast<int>(std::min<Index>(Rk, n - w.first));
                        const Chunk chunk = seg.chunk(w.first, w.rows);
                        const int b = w.rows;
                        bool terms = false, shared = true;
                        w.gb.setZero(p, M);
                        w.pmin = 1;
                        w.pmax = 0;
                        if (!evaluate(chunk, w, w.gb, terms, shared, w.ll, w.pmin, w.pmax))
                            return false;
                        w.kind = 3;
                        if constexpr (dense_view) {
                            if (shared) {
                                w.x = chunk.design[0].data();
                                w.xstride = chunk.design[0].outerStride();
                                w.kind = !full ? 2 : (terms ? 0 : 1);
                            }
                        }
                        if (w.kind == 0) {
                            // Row (s, i) of Z is C_i[, s]' kron x_i'.
                            const Eigen::Map<const MatrixXd, 0, Eigen::OuterStride<> > X(
                                w.x, b, p, Eigen::OuterStride<>(w.xstride));
                            if (w.Z.rows() < static_cast<Index>(b) * rank || w.Z.cols() != PM)
                                w.Z.resize(static_cast<Index>(b) * rank, PM);
                            for (int s = 0; s < rank; ++s)
                                for (int a = 0; a < M; ++a)
                                    w.Z.block(static_cast<Index>(s) * b, static_cast<Index>(a) * p, b, p)
                                        .noalias() = w.C.col(s * M + a).asDiagonal() * X;
                        }
                        return true;
                    },
                    [&](const VglmWork& w, Index t) {
                        if (w.kind == 3) return;
                        const int b = w.rows;
                        const Eigen::Map<const MatrixXd, 0, Eigen::OuterStride<> > X(
                            w.x, b, p, Eigen::OuterStride<>(w.xstride));
                        const int a = static_cast<int>(t / tp);
                        const Index k0 = (t % tp) * fglm_par::kTileWidth;
                        const Index wc = std::min<Index>(fglm_par::kTileWidth, p - k0);
                        const Index c = static_cast<Index>(a) * p + k0;
                        if (w.kind == 2) {
                            const MatrixXd T = w.Wdiag.col(a).asDiagonal() * X.middleCols(k0, wc);
                            acc.G[a].block(k0, k0, p - k0, wc).noalias() +=
                                X.rightCols(p - k0).transpose() * T;
                            return;
                        }
                        // The diagonal predictor block (a, a): rows k0, ..., p - 1.
                        {
                            const auto wa = w.kind == 0 ? w.d.col(a) : w.W.col(pack_index(a, a, M));
                            const MatrixXd T = wa.asDiagonal() * X.middleCols(k0, wc);
                            acc.S.block(c, c, p - k0, wc).noalias() +=
                                X.rightCols(p - k0).transpose() * T;
                        }
                        if (w.kind == 1) {
                            // Pairwise: blocks (bb, a), bb > a, of W_{a, bb}.
                            for (int bb = a + 1; bb < M; ++bb) {
                                const MatrixXd T = w.W.col(pack_index(a, bb, M)).asDiagonal() *
                                    X.middleCols(k0, wc);
                                acc.S.block(static_cast<Index>(bb) * p, c, p, wc).noalias() +=
                                    X.transpose() * T;
                            }
                            return;
                        }
                        // Low rank: S -= Z'Z on rows c, ..., pM - 1 of the tile.
                        const auto Zb = w.Z.topRows(static_cast<Index>(b) * rank);
                        acc.S.block(c, c, PM - c, wc).noalias() -=
                            Zb.rightCols(PM - c).transpose() * Zb.middleCols(c, wc);
                    },
                    [&](const VglmWork& w) {
                        vglm_add_compensated(acc.sum, acc.err, w.ll);
                        acc.g += w.gb;
                        acc.pmin = std::min(acc.pmin, w.pmin);
                        acc.pmax = std::max(acc.pmax, w.pmax);
                        if (w.kind == 0) acc.lowrank = true;
                        if (w.kind == 3 || structure) {
                            const Chunk chunk = seg.chunk(w.first, w.rows);
                            // const_cast: the slot's scratch, used on the calling thread.
                            VglmWork& ww = const_cast<VglmWork&>(w);
                            if (w.kind == 3) add_packed(chunk, ww, acc, false);
                            if (structure) {
                                vglm_structure_add(chunk, p, acc.R, ww.stack);
                                acc.R_used = true;
                            }
                        }
                    },
                    &failed);
            }
            if (!ok) {
                status = VGLM_PASS_INVALID;
                return;
            }
        }
        if (stop < nb) {
            status = VGLM_PASS_SHRINK;
            shrinking = true;
        }
    });
    if (status != VGLM_PASS_OK) return status;
    if (structure) *structure = acc.R;
    out.loglik = acc.sum + acc.err;
    out.pmin = acc.pmin;
    out.pmax = acc.pmax;
    out.lowrank = acc.lowrank;
    out.full = full;
    if (full) {
        out.Gdiag.clear();
        if (acc.S_used) vglm_lowrank_unpack(acc.S, p, M, acc.G);
        // The tile scheme's S is pM by pM (more than 8 MB); free it before
        // I is mapped, as is G after.
        if (!lanes) acc.S.resize(0, 0);
        L.map(acc.G, acc.g, out.I, out.U);
        if (!lanes) {
            acc.G.clear();
            acc.G_used = false;
        }
        if (!out.I.allFinite() || !out.U.allFinite()) return VGLM_PASS_INVALID;
        return VGLM_PASS_OK;
    }
    // Each G_aa is made symmetric from one triangle: the upper one, as map
    // reads it, when the blocks were added whole (lanes and serial blocks),
    // the lower one when they were added by column tiles (dense tile scheme).
    const bool from_lower = !lanes && dense_view;
    bool finite = true;
    std::vector<MatrixXd>& G = acc.G;
    for (int a = 0; a < M; ++a) {
        MatrixXd sym = from_lower
            ? MatrixXd(G[a].template selfadjointView<Eigen::Lower>())
            : MatrixXd(G[a].template selfadjointView<Eigen::Upper>());
        G[a].swap(sym);
        finite = finite && G[a].allFinite();
    }
    L.score_map(acc.g, out.U);
    if (L.identity) {
        out.I.resize(0, 0);
        out.Gdiag.swap(G);
    } else {
        out.Gdiag.clear();
        L.map_diagonal(G, out.I);
        finite = finite && out.I.allFinite();
    }
    acc.G_used = false;
    if (!finite || !out.U.allFinite()) return VGLM_PASS_INVALID;
    return VGLM_PASS_OK;
}

// ---------------------------------------------------------------------------
// Information-vector product (design Section 5, B1): out = I(beta) v with no
// P by P matrix. With T the p by M matrix of v (T(k, a) = (H_k v_k)[a]),
// each row gives t_i = X_i* v as row i of X T, then s_i = W_i t_i, and the
// product is sum_i X_i*' s_i, accumulated as g = sum_i x_i s_i' and mapped
// by score_map. Per block the design is read once for both eta and t, as
// X [B T] (or X_a [B_a T_a] per linear predictor when the views differ).
//
// When the model type has the low-rank structure (information_rank() > 0)
// the product uses s_i = d_i * t_i - C_i (C_i' t_i), at order M r per row,
// so the pass costs order b p M per block (the two products with the
// design). Unlike the accumulation in vglm_pass, this form never builds the
// matrix Z, so it applies to every design view (sparse included, shared or
// not) and at every M; only force_pairwise (test only) turns it off. Then,
// as for model types without the structure, s_i is formed from the packed
// W_i at order M^2 per row. Rows that do not enter the likelihood contribute
// nothing. Entries of v at aliased coefficients must be 0 on entry (the
// caller's job), so that the product is that of the kept system; the
// product itself is generally nonzero there. Returns VGLM_PASS_INVALID if
// the model type cannot be evaluated at beta or the product is not finite.
//
// cache (optional) keeps the per-row terms of the model type at beta (d and
// C, or the packed W) block by block for later products at the same beta:
// a product with an empty cache evaluates the model type and stores them
// (unless they would exceed cache->cap doubles, in which case the cache is
// disabled and every product evaluates the model type, as without a cache);
// a product with a filled cache forms only t = X T, s_i = W_i t_i, and the
// sum of X_i*' s_i, reading the terms in block order. The caller must reset
// the cache whenever beta changes.

struct VglmProductCache {
    std::int64_t cap = 0;     // largest number of doubles kept
    std::int64_t size = 0;    // doubles kept so far
    bool filled = false, disabled = false;
    std::vector<MatrixXd> first, second;  // per block: d and C, or W and empty
    void reset(std::int64_t cap_) {
        cap = cap_;
        size = 0;
        filled = false;
        disabled = cap_ <= 0;
        first.clear();
        second.clear();
    }
};

template <class Data>
VglmPassStatus vglm_info_product(const Data& data, const VectorFamily& fam, const VglmLayout& L,
                                 const VglmOptions& opts, const VectorXd& beta,
                                 const VectorXd& v, VectorXd& out,
                                 VglmProductCache* cache = nullptr,
                                 VglmWorkspace* workspace = nullptr) {
    typedef typename Data::Segment Segment;
    typedef typename Segment::Chunk Chunk;
    typedef Eigen::Index Index;
    const int p = L.p, M = L.M, Q = packed_size(M);
    const int rank = opts.force_pairwise ? 0 : fam.information_rank();
    const bool use = cache != nullptr && cache->filled && !cache->disabled;
    const bool fill = cache != nullptr && !cache->filled && !cache->disabled;
    MatrixXd BT(p, 2 * M);
    BT << L.predictors(beta), L.predictors(v);
    const MatrixXd T = BT.rightCols(M);
    // [B_a T_a] for blocks whose views differ across linear predictors.
    std::vector<MatrixXd> pairs(M, MatrixXd(p, 2));
    for (int a = 0; a < M; ++a) {
        pairs[a].col(0) = BT.col(a);
        pairs[a].col(1) = BT.col(M + a);
    }
    VglmWorkspace local;
    VglmWorkspace& ws = workspace ? *workspace : local;
    VglmAcc& acc = ws.out;
    acc.g.setZero(p, M);
    VglmPassStatus status = VGLM_PASS_OK;
    const Index Rk = vglm_lane_rows(data.block_rows(), L.P, Q);
    // Doubles of the cached terms per row.
    const std::int64_t per_row = rank > 0 ? static_cast<std::int64_t>(M) * (1 + rank)
                                          : static_cast<std::int64_t>(Q);
    std::size_t base = 0;   // index of the segment's first block among all blocks

    // s_i = W_i t_i from the low-rank terms or the packed W.
    auto form = [&](VglmWork& w, const auto& t, const MatrixXd& dd, const MatrixXd& CC,
                    const MatrixXd& WW) {
        const Index b = t.rows();
        if (rank > 0) {
            w.s = dd.cwiseProduct(t);
            for (int q = 0; q < rank; ++q) {
                w.proj = CC.middleCols(q * M, M).cwiseProduct(t).rowwise().sum();
                w.s.noalias() -= w.proj.asDiagonal() * CC.middleCols(q * M, M);
            }
        } else {
            w.s.setZero(b, M);
            for (int bb = 0; bb < M; ++bb)
                for (int a = 0; a <= bb; ++a) {
                    const int q = pack_index(a, bb, M);
                    w.s.col(a) += WW.col(q).cwiseProduct(t.col(bb));
                    if (a != bb) w.s.col(bb) += WW.col(q).cwiseProduct(t.col(a));
                }
        }
    };

    data.segments([&](const Segment& seg) {
        if (status != VGLM_PASS_OK) return;
        const int n = seg.n();
        if (n <= 0) return;
        const Index nb = (n + Rk - 1) / Rk;
        // Whether this segment's terms are stored. The cache holds a
        // segment's terms only if all of them fit, which is the outcome of
        // storing block by block until the cap is passed.
        bool fill_seg = false;
        if (fill && !cache->disabled) {
            const std::int64_t add = static_cast<std::int64_t>(n) * per_row;
            if (cache->size + add > cache->cap) {
                cache->disabled = true;
                cache->first.clear();
                cache->second.clear();
                cache->first.shrink_to_fit();
                cache->second.shrink_to_fit();
            } else {
                cache->size += add;
                cache->first.resize(base + nb);
                cache->second.resize(base + nb);
                fill_seg = true;
            }
        }
        if (use && base + nb > cache->first.size()) {
            status = VGLM_PASS_INVALID;
            return;
        }
        const Index nl = std::min<Index>(nb, fglm_par::kLanes);
        const Index ns = vglm_lane_slots(ws, nb);
        for (Index l = 0; l < nl; ++l) ws.lanes[l].g.resize(p, M);
        Index failed = -1;
        const bool ok = fglm_par::lane_reduce_into(
            nb, acc, ws.lanes, [&](VglmAcc& a) { a.g.setZero(p, M); },
            [&]() { return vglm_thread_work(ws, ns); },
            [&](Index blk, VglmAcc& a, VglmWorkHandle& h) -> bool {
                VglmWork& w = *h.w;
                const int first = static_cast<int>(blk * Rk);
                const int b = static_cast<int>(std::min<Index>(Rk, n - first));
                const Chunk chunk = seg.chunk(first, b);
                const bool shared = vglm_shared_view(chunk, M);
                const std::size_t index = base + static_cast<std::size_t>(blk);
                if (use) {
                    // t = X T only; the terms at beta are those of the cache.
                    if (shared) vglm_mul(chunk.design[0], T, w.tc);
                    else {
                        w.tc.resize(b, M);
                        for (int a2 = 0; a2 < M; ++a2)
                            vglm_mul(chunk.design[a2], T.col(a2), w.tc.col(a2));
                    }
                    const MatrixXd& c1 = cache->first[index];
                    const MatrixXd& c2 = cache->second[index];
                    if (c1.rows() != b) return false;
                    form(w, w.tc, c1, c2, c1);
                } else {
                    // E = [eta, t]: eta = X B + offset in the left M columns
                    // and t = X T (no offset) in the right M columns.
                    if (shared) vglm_mul(chunk.design[0], BT, w.E);
                    else {
                        w.E.resize(b, 2 * M);
                        for (int a2 = 0; a2 < M; ++a2) {
                            vglm_mul(chunk.design[a2], pairs[a2], w.two);
                            w.E.col(a2) = w.two.col(0);
                            w.E.col(M + a2) = w.two.col(1);
                        }
                    }
                    w.E.leftCols(M) += chunk.offset;
                    const auto t = w.E.rightCols(M);
                    const VglmBlock block{w.E.leftCols(M), chunk.codes, chunk.counts,
                                          chunk.weights};
                    if (!fam.valid(block)) return false;
                    if (rank > 0) fam.score_information_lowrank(block, w.u, w.d, w.C);
                    else fam.score_information(block, w.u, w.W);
                    form(w, t, w.d, w.C, w.W);
                    if (fill_seg) {
                        cache->first[index] = rank > 0 ? w.d : w.W;
                        cache->second[index] = rank > 0 ? w.C : MatrixXd();
                    }
                }
                for (int i = 0; i < b; ++i)
                    if (!vglm_row_counts(chunk, i)) w.s.row(i).setZero();
                if (!w.s.allFinite()) return false;
                if (shared) vglm_tmul_add(chunk.design[0], w.s, a.g);
                else
                    for (int a2 = 0; a2 < M; ++a2)
                        vglm_tmul_add(chunk.design[a2], w.s.col(a2), a.g.col(a2));
                return true;
            },
            [&](VglmAcc& o, const VglmAcc& a) { o.g += a.g; }, &failed);
        if (!ok) {
            status = VGLM_PASS_INVALID;
            return;
        }
        base += static_cast<std::size_t>(nb);
    });
    if (status != VGLM_PASS_OK) return status;
    if (use && base != cache->first.size()) return VGLM_PASS_INVALID;
    if (fill && !cache->disabled) cache->filled = true;
    L.score_map(acc.g, out);
    if (!out.allFinite()) return VGLM_PASS_INVALID;
    return VGLM_PASS_OK;
}

// ---------------------------------------------------------------------------
// Preconditioner of the truncated Newton solver (design Section 5, B2): the
// inverse of Idiag, the information with the off-diagonal predictor blocks
// dropped (a VGLM_DIAGONAL pass), on the coefficients that are not aliased.
// Under an identity layout Idiag couples only coefficients (k, a) and (l, a)
// with the same linear predictor a, and the pass keeps it as the M blocks
// G_aa (state.Gdiag), so it is factored as M separate Cholesky factors of
// at most p by p (the kept rows and columns k of G_aa), with no P by P
// matrix. Otherwise the constraint mapping mixes the linear predictors and
// the kept part of the P by P matrix state.I (generally not block diagonal)
// is factored by one LLT.
//
// compute() returns false, and leaves the object failed (ok false, no
// factors), if the state is not a VGLM_DIAGONAL state for this layout, the
// sizes do not match, or a factor is not numerically positive definite.
// apply() on a failed object stops with an error.

struct VglmPreconditioner {
    std::vector<std::vector<int> > index;     // coefficients of each factor
    std::vector<Eigen::LLT<MatrixXd> > llt;   // one per entry of index
    int P = 0;
    bool ok = false;

    bool compute(const VglmState& state, const std::vector<bool>& aliased, const VglmLayout& L) {
        clear();
        if (state.full || static_cast<int>(aliased.size()) != L.P) return false;
        std::vector<MatrixXd> factors;
        if (L.identity) {
            if (static_cast<int>(state.Gdiag.size()) != L.M) return false;
            for (int a = 0; a < L.M; ++a) {
                const MatrixXd& Ga = state.Gdiag[a];
                if (Ga.rows() != L.p || Ga.cols() != L.p) return false;
                std::vector<int> ks;
                for (int k = 0; k < L.p; ++k)
                    if (!aliased[L.start[k] + a]) ks.push_back(k);
                if (ks.empty()) continue;
                const int m = static_cast<int>(ks.size());
                MatrixXd A(m, m);
                for (int c = 0; c < m; ++c)
                    for (int r = 0; r < m; ++r) A(r, c) = Ga(ks[r], ks[c]);
                std::vector<int> idx(m);
                for (int r = 0; r < m; ++r) idx[r] = L.start[ks[r]] + a;
                index.push_back(idx);
                factors.push_back(A);
            }
        } else {
            if (state.I.rows() != L.P || state.I.cols() != L.P) return false;
            std::vector<int> idx;
            for (int j = 0; j < L.P; ++j) if (!aliased[j]) idx.push_back(j);
            if (!idx.empty()) {
                const int m = static_cast<int>(idx.size());
                MatrixXd A(m, m);
                for (int c = 0; c < m; ++c)
                    for (int r = 0; r < m; ++r) A(r, c) = state.I(idx[r], idx[c]);
                index.push_back(idx);
                factors.push_back(A);
            }
        }
        llt.resize(factors.size());
        for (std::size_t f = 0; f < factors.size(); ++f) {
            if (!factors[f].allFinite()) { clear(); return false; }
            llt[f].compute(factors[f]);
            if (llt[f].info() != Eigen::Success) { clear(); return false; }
        }
        P = L.P;
        ok = true;
        return true;
    }

    // z = Idiag^{-1} r on the kept coefficients and 0 at the aliased ones;
    // entries of r at aliased coefficients are not read.
    void apply(const VectorXd& r, VectorXd& z) const {
        if (!ok) Rcpp::stop("the preconditioner was not computed successfully");
        if (r.size() != P) Rcpp::stop("the preconditioner was given a vector of the wrong length");
        z.setZero(P);
        VectorXd x;
        for (std::size_t f = 0; f < index.size(); ++f) {
            const std::vector<int>& idx = index[f];
            const int m = static_cast<int>(idx.size());
            x.resize(m);
            for (int a = 0; a < m; ++a) x[a] = r[idx[a]];
            llt[f].solveInPlace(x);
            for (int a = 0; a < m; ++a) z[idx[a]] = x[a];
        }
    }

private:
    void clear() {
        index.clear();
        llt.clear();
        P = 0;
        ok = false;
    }
};

// ---------------------------------------------------------------------------
// The truncated Newton step (design Section 5, B1 to B3): I delta = U solved
// approximately by preconditioned conjugate gradients from delta = 0, on the
// coefficients that are not aliased (the initial residual is U with the
// aliased entries set to 0; the preconditioner and the products keep them
// at 0). Each product I d is one pass over the data (vglm_info_product) and
// is added to products. The solve stops, returning true, when ||r|| <=
// max(forcing, 1e-12) ||U||, or when d' I d <= 0 after at least one step (the
// iterate so far is kept). It returns false, with delta not to be used, when
// max_products products were made without meeting the stop rule, when a
// product pass fails, or when the curvature is not positive at the first
// step. U = 0 gives delta = 0 with no product.
template <class Data>
bool vglm_cg_solve(const Data& data, const VectorFamily& fam, const VglmLayout& L,
                   const VglmOptions& opts, const VectorXd& beta, const VectorXd& U,
                   const std::vector<bool>& aliased, const VglmPreconditioner& pre,
                   double forcing, int max_products, VectorXd& delta, int& products,
                   VglmWorkspace* workspace = nullptr) {
    const int P = L.P;
    delta.setZero(P);
    VectorXd r = U;
    for (int j = 0; j < P; ++j) if (aliased[j]) r[j] = 0;
    const double unorm = r.norm();
    if (!std::isfinite(unorm)) return false;
    if (unorm == 0) return true;
    const double target = std::max(forcing, 1e-12) * unorm;
    VectorXd z, d, q;
    VglmProductCache cache;
    cache.reset(opts.product_cache_cap);
    pre.apply(r, z);
    d = z;
    double rz = r.dot(z);
    for (int k = 0;; ++k) {
        if (k >= max_products) return false;
        ++products;
        if (vglm_info_product(data, fam, L, opts, beta, d, q, &cache, workspace) != VGLM_PASS_OK)
            return false;
        for (int j = 0; j < P; ++j) if (aliased[j]) q[j] = 0;
        const double dq = d.dot(q);
        if (!(dq > 0) || !std::isfinite(dq)) return k > 0;
        const double alpha = rz / dq;
        delta.noalias() += alpha * d;
        r.noalias() -= alpha * q;
        if (r.norm() <= target) return true;
        pre.apply(r, z);
        const double rz_next = r.dot(z);
        // The preconditioner is positive definite, so this is 0 only when r
        // is negligible; the iterate so far is kept.
        if (!(rz_next > 0)) return true;
        d = z + (rz_next / rz) * d;
        rz = rz_next;
    }
}

// ---------------------------------------------------------------------------
// Mean bias reduction (Firth 1993; design Section 6).
//
// The adjusted score is U + sum_i X_i*' t_i, where the model type returns
// t_i from A_i = X_i* V X_i*' with V = I^{-1} at the same coefficients (0
// in the rows and columns of aliased coefficients). This pass computes the
// adjustment sum_i X_i*' t_i at beta. Per block it forms the packed A_i
// through the p by p matrices F_ab of VglmLayout::eta_inverse, at
// M(M + 1)/2 products of b by p by p, the same work as the information.
//
// Second pass (design Section 11, decided here): V is the inverse of the
// information at beta, which is known only once a pass at beta has ended,
// so the exact adjustment at beta needs a second pass over the data. The
// engine makes it for every data representation, streamed sources included:
// each iteration is one trial pass per step length tried plus one
// adjustment pass at the accepted estimate, so a fit without step halving
// makes 2 iter + 2 passes (one at the start, iter + 1 adjustment passes, and
// iter trial passes). The alternatives were rejected. A lagged V (from the
// previous iterate, as the streamed GLM bias reduction does) saves the pass
// but makes the adjusted score, the Fisher decrement, and the reported
// adjustment inexact away from the fixed point, so the convergence check
// would confirm a different quantity than the step solves. Accumulating
// third-order moments of the design in the first pass avoids V but needs
// O(p^3 M^3) memory. Bias reduction is used on problems near separation,
// where an exact convergence check matters more than one extra read.
template <class Data>
void vglm_adjustment(const Data& data, const VectorFamily& fam, const VglmLayout& L,
                     const VectorXd& beta, const MatrixXd& V, VectorXd& adjustment,
                     VglmWorkspace* workspace = nullptr) {
    typedef typename Data::Segment Segment;
    typedef typename Segment::Chunk Chunk;
    typedef Eigen::Index Index;
    const int p = L.p, M = L.M;
    const MatrixXd B = L.predictors(beta);
    std::vector<MatrixXd> F;
    L.eta_inverse(V, F);
    VglmWorkspace local;
    VglmWorkspace& ws = workspace ? *workspace : local;
    VglmAcc& acc = ws.out;
    acc.g.setZero(p, M);
    const Index Rk = vglm_lane_rows(data.block_rows(), L.P, packed_size(M));
    data.segments([&](const Segment& seg) {
        const int n = seg.n();
        if (n <= 0) return;
        const Index nb = (n + Rk - 1) / Rk;
        const Index nl = std::min<Index>(nb, fglm_par::kLanes);
        const Index ns = vglm_lane_slots(ws, nb);
        for (Index l = 0; l < nl; ++l) ws.lanes[l].g.resize(p, M);
        Index failed = -1;
        const bool ok = fglm_par::lane_reduce_into(
            nb, acc, ws.lanes, [&](VglmAcc& a) { a.g.setZero(p, M); },
            [&]() { return vglm_thread_work(ws, ns); },
            [&](Index blk, VglmAcc& a, VglmWorkHandle& h) -> bool {
                VglmWork& w = *h.w;
                const int first = static_cast<int>(blk * Rk);
                const int b = static_cast<int>(std::min<Index>(Rk, n - first));
                const Chunk chunk = seg.chunk(first, b);
                const bool shared = vglm_shared_view(chunk, M);
                vglm_linear(chunk, shared, B, w.eta);
                w.A.resize(b, packed_size(M));
                for (int bb = 0; bb < M; ++bb)
                    for (int a2 = 0; a2 <= bb; ++a2) {
                        const int q = pack_index(a2, bb, M);
                        vglm_mul(chunk.design[a2], F[q], w.XF);
                        vglm_row_dots(w.XF, chunk.design[bb], w.A.col(q));
                    }
                const VglmBlock block{w.eta, chunk.codes, chunk.counts, chunk.weights};
                fam.bias_terms(block, w.A, w.t);
                if (shared) vglm_tmul_add(chunk.design[0], w.t, a.g);
                else
                    for (int a2 = 0; a2 < M; ++a2)
                        vglm_tmul_add(chunk.design[a2], w.t.col(a2), a.g.col(a2));
                return true;
            },
            [&](VglmAcc& o, const VglmAcc& a) { o.g += a.g; }, &failed);
        if (!ok) {
            Rcpp::stop("the bias reduction adjustment could not be evaluated");
        }
    });
    L.score_map(acc.g, adjustment);
    if (!adjustment.allFinite()) Rcpp::stop("the bias reduction adjustment is not finite");
}

// ---------------------------------------------------------------------------
// Structural aliasing, decided once from the design (not from an iterate).
//
// With R'R = X'X over the rows that enter the likelihood and the expanded
// constraint matrix Hbig = blockdiag(H_1, ..., H_p), the information at unit
// working weights is Z'Z with Z = (R (x) I_M) Hbig, so the coefficients are
// identifiable exactly when Z has full column rank; Z is formed from R, not
// from the squared cross products, so collinearity is judged on the scale of
// the design, as glm does. Columns of Z are taken in coefficient order and a
// column is aliased when its residual, after projection on the kept earlier
// columns (Gram-Schmidt, applied twice), is at most tol times its original
// norm: the rule of R's dqrdc2 with glm.fit's default tol of 1e-11, so the
// later of two collinear columns is reported as aliased.
inline std::vector<bool> vglm_structural_aliased(const MatrixXd& R, const VglmLayout& L,
                                                 double tol = 1e-11) {
    const int p = L.p, M = L.M, P = L.P;
    if (L.identity) {
        // Every H_k is I_M, so the column of Z for coefficient (k, a) is
        // R(:, k) (x) e_a. Columns with different a are orthogonal, and the
        // projection of one on the earlier columns with the same a is the
        // projection of R(:, k) on the earlier kept columns of R, so the
        // Gram-Schmidt rule below gives the same decision for every a: the
        // coefficient (k, a) is aliased exactly when column k of R is.
        std::vector<bool> aliased(P, false);
        MatrixXd basis(p, p);
        int kept = 0;
        for (int k = 0; k < p; ++k) {
            VectorXd v = R.col(k);
            const double norm0 = v.norm();
            bool drop = !(norm0 > 0);
            if (!drop) {
                for (int pass = 0; pass < 2; ++pass)
                    for (int j = 0; j < kept; ++j) v -= basis.col(j).dot(v) * basis.col(j);
                const double norm = v.norm();
                if (norm <= tol * norm0) drop = true;
                else basis.col(kept++) = v / norm;
            }
            if (drop)
                for (int a = 0; a < M; ++a) aliased[L.start[k] + a] = true;
        }
        return aliased;
    }
    MatrixXd Z = MatrixXd::Zero(p * M, P);
    for (int k = 0; k < p; ++k)
        for (int s = 0; s < L.r(k); ++s)
            for (int j = 0; j < p; ++j)
                for (int a = 0; a < M; ++a)
                    Z(j * M + a, L.start[k] + s) = R(j, k) * L.H[k](a, s);
    std::vector<bool> aliased(P, false);
    MatrixXd basis(p * M, P);
    int kept = 0;
    for (int c = 0; c < P; ++c) {
        VectorXd v = Z.col(c);
        const double norm0 = v.norm();
        if (!(norm0 > 0)) { aliased[c] = true; continue; }
        for (int pass = 0; pass < 2; ++pass)
            for (int j = 0; j < kept; ++j) v -= basis.col(j).dot(v) * basis.col(j);
        const double norm = v.norm();
        if (norm <= tol * norm0) { aliased[c] = true; continue; }
        basis.col(kept++) = v / norm;
    }
    return aliased;
}

// ---------------------------------------------------------------------------
// The step: solve I delta = U on the coefficients that are not aliased.
//
// I is scaled to unit diagonal, C = D^{-1} I D^{-1}, and factored by LLT. If
// LLT fails, or its reciprocal condition estimate is below rcond_threshold,
// the iterate is numerically singular although the design is not (for
// example, weights that underflow near separation): the step is then the
// minimum-norm least squares solution from a complete orthogonal
// decomposition of C (a column-pivoted QR followed by an orthogonal
// reduction), the covariance is its pseudoinverse, and singular is set so
// the caller can flag both. Nothing is aliased here; aliasing is structural.
// A zero diagonal entry is scaled by 1.
//
// The factor is computed once per iterate and used both for the covariance
// (which bias reduction needs before the right-hand side is known) and for
// the step.

struct VglmFactor {
    std::vector<int> kept;   // coefficients that are not aliased
    VectorXd s;              // diagonal scaling, D^{-1}
    MatrixXd C;              // scaled information on kept
    Eigen::LLT<MatrixXd> llt;
    Eigen::CompleteOrthogonalDecomposition<MatrixXd> cod;
    MatrixXd cov;            // P by P inverse information; filled by vglm_covariance
    bool singular = false;   // the fallback was used
};

inline void vglm_factor(const MatrixXd& I, const std::vector<bool>& aliased,
                        double rcond_threshold, VglmFactor& F) {
    const int P = static_cast<int>(I.rows());
    F.kept.clear();
    for (int j = 0; j < P; ++j) if (!aliased[j]) F.kept.push_back(j);
    const int m = static_cast<int>(F.kept.size());
    F.s.resize(m);
    F.C.resize(m, m);
    for (int a = 0; a < m; ++a) {
        const double d = I(F.kept[a], F.kept[a]);
        F.s[a] = d > 0 && std::isfinite(d) ? 1 / std::sqrt(d) : 1.0;
    }
    for (int bb = 0; bb < m; ++bb)
        for (int a = 0; a < m; ++a) F.C(a, bb) = F.s[a] * I(F.kept[a], F.kept[bb]) * F.s[bb];
    F.singular = false;
    if (m > 0) {
        F.llt.compute(F.C);
        if (!(F.llt.info() == Eigen::Success && F.llt.rcond() >= rcond_threshold)) {
            F.cod.compute(F.C);
            F.singular = true;
        }
    }
}

// Fill F.cov, the P by P inverse information of the iterate F was computed
// at (the pseudoinverse on a singular iterate), NA at aliased coefficients.
// It is not part of vglm_factor because the step needs only a solve; the
// inverse is formed when bias reduction needs it and once for the returned
// estimate.
inline void vglm_covariance(VglmFactor& F, int P) {
    const int m = static_cast<int>(F.kept.size());
    MatrixXd inverse(m, m);
    if (m > 0) {
        inverse = F.singular ? MatrixXd(F.cod.pseudoInverse())
                             : MatrixXd(F.llt.solve(MatrixXd::Identity(m, m)));
        inverse = (0.5 * (inverse + inverse.transpose())).eval();
    }
    F.cov.setConstant(P, P, NA_REAL);
    for (int a = 0; a < m; ++a)
        for (int bb = 0; bb < m; ++bb)
            F.cov(F.kept[a], F.kept[bb]) = F.s[a] * inverse(a, bb) * F.s[bb];
}

// The step for the score U (ordinary or adjusted). decrement is the Fisher
// decrement U' delta. On a singular iterate, the part of the scaled score
// D^{-1} U that lies in the null space of C cannot be removed by any step and
// is not seen by the decrement; residual holds its squared norm, || C x -
// D^{-1} U ||^2 at the least squares solution x. Both are on the same scale
// (in coordinate j, (D^{-1} U)_j^2 = U_j^2 / I_jj is the decrement of that
// coordinate alone), so the convergence check uses their sum. residual is 0
// when LLT was used.
struct VglmStep {
    VectorXd delta;     // length P, 0 at aliased coefficients
    double decrement = std::numeric_limits<double>::quiet_NaN();  // U' I^{-1} U
    double residual = 0;
};

inline void vglm_direction(const VglmFactor& F, const VectorXd& U, VglmStep& step) {
    const int P = static_cast<int>(U.size()), m = static_cast<int>(F.kept.size());
    VectorXd rhs(m), solution(m);
    for (int a = 0; a < m; ++a) rhs[a] = F.s[a] * U[F.kept[a]];
    step.residual = 0;
    if (m > 0) {
        if (!F.singular) solution = F.llt.solve(rhs);
        else {
            solution = F.cod.solve(rhs);
            step.residual = (F.C * solution - rhs).squaredNorm();
        }
    }
    step.delta.setZero(P);
    for (int a = 0; a < m; ++a) step.delta[F.kept[a]] = F.s[a] * solution[a];
    step.decrement = 0;
    for (int a = 0; a < m; ++a) step.decrement += U[F.kept[a]] * step.delta[F.kept[a]];
}

// log det I over the coefficients that are not aliased, the penalty of the
// bias-reduced objective l + (1/2) log det I: log det C + sum_j log I_jj from
// the LLT of the scaled matrix C. It is -Inf when C is not numerically
// positive definite, so a trial step to such a point is halved.
//
// Accuracy near separation: the multinomial row information has diagonal
// c p (1 - p), and 1 - p cancels when p is near 1, so a row with p near 1
// contributes an absolute error of about eps c to I. That is the size of the
// rounding error of the accumulation itself, so log det I, I^{-1}, and the
// A_i built from it keep the accuracy of the summed information unless I is
// nearly singular; the penalty -> -Inf as I approaches singularity, so
// bias-reduced iterates stay away from that region.
inline double vglm_logdet(const MatrixXd& I, const std::vector<bool>& aliased) {
    const int P = static_cast<int>(I.rows());
    std::vector<int> kept;
    for (int j = 0; j < P; ++j) if (!aliased[j]) kept.push_back(j);
    const int m = static_cast<int>(kept.size());
    const double minus_inf = -std::numeric_limits<double>::infinity();
    if (m == 0) return 0;
    VectorXd s(m);
    double sum = 0;
    for (int a = 0; a < m; ++a) {
        const double d = I(kept[a], kept[a]);
        if (!(d > 0) || !std::isfinite(d)) return minus_inf;
        s[a] = 1 / std::sqrt(d);
        sum += std::log(d);
    }
    MatrixXd C(m, m);
    for (int bb = 0; bb < m; ++bb)
        for (int a = 0; a < m; ++a) C(a, bb) = s[a] * I(kept[a], kept[bb]) * s[bb];
    Eigen::LLT<MatrixXd> llt(C);
    if (llt.info() != Eigen::Success) return minus_inf;
    for (int a = 0; a < m; ++a) {
        const double l = llt.matrixLLT()(a, a);
        if (!(l > 0)) return minus_inf;
        sum += 2 * std::log(l);
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Starting values.

// Weighted category totals over all rows.
template <class Data>
VectorXd vglm_category_totals(const Data& data, int K) {
    VectorXd totals = VectorXd::Zero(K);
    data.visit([&](const typename Data::Chunk& chunk) {
        for (int i = 0; i < chunk.rows(); ++i) {
            const double w = chunk.weights[i];
            if (w == 0) continue;
            if (chunk.codes == nullptr) totals.noalias() += w * chunk.counts.row(i).transpose();
            else totals[chunk.codes[i] - 1] += w;
        }
    });
    return totals;
}

// Intercept coefficients from the model type's eta-scale start values, all
// other coefficients 0. intercept is the 0-based design column that is
// constant (-1 when there is none) and value its constant entry; with a
// constraint matrix H the coefficients solve H beta = eta / value by least
// squares. Offsets are not used.
inline VectorXd vglm_start(const VectorFamily& fam, const VectorXd& totals,
                           const std::vector<MatrixXd>& H, int intercept, double value) {
    const VglmLayout L(H, fam.M(), false);
    VectorXd beta = VectorXd::Zero(L.P);
    const VectorXd eta = fam.start_eta(totals);   // also checks for empty categories
    if (intercept >= 0 && intercept < L.p && value != 0 && std::isfinite(value)) {
        const VectorXd target = eta / value;
        beta.segment(L.start[intercept], L.r(intercept)) =
            L.H[intercept].colPivHouseholderQr().solve(target);
    }
    return beta;
}

// ---------------------------------------------------------------------------
// The fit.
//
// The first pass, at the starting values, also gives the design factor from
// which the aliased coefficients are decided once (they are held at 0 and
// reported as NA). Each iteration then uses the state (log likelihood, U, I)
// of one pass at the current coefficients. The step delta = I^{-1} U is
// scaled by the model type's max_step_fraction (only when it restricts
// steps; a fraction that is not positive ends the fit with
// "step_halving_failed", since no step can be taken) and then halved until
// every row is valid and the objective does not decrease beyond the rounding
// allowance of a long sum, 64 eps (1 + |objective|). The pass that accepts a
// step is the state for the next iteration, so a fit without halving makes
// iter + 1 passes.
//
// With bias reduction (firth), the score is the adjusted score U + sum_i
// X_i*' t_i, computed by vglm_adjustment at each iterate with I^{-1} at that
// iterate (a second pass; see vglm_adjustment), and the objective is the
// penalized log likelihood l + (1/2) log det I, whose gradient is the
// adjusted score for the canonical multinomial link (design Section 6). A
// fit without halving then makes 2 iter + 2 passes. Without bias reduction
// the objective is l.
//
// Convergence (the GLM rule, design Section 7), with D = -2 objective: the
// relative change |D - D_old| / (0.1 + |D|) < tol after an accepted step,
// then confirmed at the new estimate by the Fisher decrement,
// lambda^2 / (0.1 + |D|) < tol with lambda^2 = U' I^{-1} U for the (adjusted)
// score U. On a singular iterate the part of the score in the null space of
// I is added to lambda^2 (VglmStep::residual), so a fit is never reported
// converged while that part is not negligible. The confirmed estimate is
// returned without taking its step, and its information gives cov.unscaled
// with no further pass; "singular" reports that this information was
// numerically singular, in which case cov.unscaled is a pseudoinverse.
//
// Truncated Newton solver (opts.solver == "cg", design Section 5; not with
// bias reduction). The iterate state, at the start and in every trial pass
// of the halving, comes from a VGLM_DIAGONAL pass: log likelihood, U, and
// the information with the off-diagonal predictor blocks dropped, which is
// factored as the preconditioner (under an identity layout as M p by p
// factors, so no P by P matrix is formed). The step solves I delta = U by
// vglm_cg_solve with forcing term min(0.1, sqrt(c)), c the decrement U' delta
// of the step taken in the previous iteration divided by (0.1 + |D|) (0.1 in
// the first iteration). If the preconditioner cannot be factored, or the
// solve fails (product cap reached), the iteration takes the exact step from
// a VGLM_FULL pass at the current estimate; the next iteration returns to
// conjugate gradients. The step then goes through the same max_step_fraction,
// halving, and rounding rules. Convergence is never judged from the CG
// decrement (a lower bound on lambda^2): the first time the deviance rule is
// pending, a VGLM_FULL pass at the current estimate gives the exact Fisher
// decrement, and the fit is converged only if the exact rule holds. From
// then on the solver takes exact steps with full trial passes (few
// iterations remain, and the full pass is needed for cov.unscaled in any
// case), so the last iterations are not spent tightening CG toward an exact
// solve. opts.cg_reject_confirmation (test only) treats that many
// confirmations that would succeed as failed, so that the continuation is
// exercised. If a CG step is blocked (max_step_fraction not positive) or no
// halving of it is accepted, the iteration is repeated with the exact step,
// which then goes through the exact solver's rules (a blocked or failed
// exact step ends the fit as there; opts.cg_reject_steps, test only, treats
// the halving of that many CG steps as failed). Every exit returns the
// information, score, and covariance of a VGLM_FULL pass at the returned
// estimate (one more pass when the loop ended on a diagonal state). passes
// counts every pass, products included; products counts the
// information-vector products.

template <class Data>
Rcpp::List vglm_fit(const Data& data, const VectorFamily& fam, const std::vector<MatrixXd>& H,
                    const VectorXd& start, bool firth, double tol, int maxit, bool trace,
                    const VglmOptions& opts = VglmOptions()) {
    if (!(tol > 0) || !std::isfinite(tol) || maxit < 1)
        Rcpp::stop("tol must be positive and finite and maxit at least 1");
    if (opts.lowrank_cap < 1) Rcpp::stop("lowrank_cap must be a positive integer");
    if (opts.solver != "exact" && opts.solver != "cg")
        Rcpp::stop("solver must be \"exact\" or \"cg\"");
    const bool cg = opts.solver == "cg";
    if (cg && firth)
        Rcpp::stop("the conjugate gradient solver is not available with bias reduction");
    if (static_cast<int>(H.size()) != data.p())
        Rcpp::stop("there must be one constraint matrix per design column");
    const VglmLayout L(H, fam.M(), opts.force_mapping);
    const int P = L.P;
    if (start.size() != P || !start.allFinite())
        Rcpp::stop("starting values must be finite with one entry per coefficient");

    const double rcond_threshold = 1e-12;
    const int max_halves = 30;
    const double eps = std::numeric_limits<double>::epsilon();
    // The iterate state of the CG solver is a diagonal pass until it
    // switches to exact steps (exact_phase).
    VglmPassMode iterate_mode = cg ? VGLM_DIAGONAL : VGLM_FULL;

    VectorXd beta = start;
    int passes = 0, updates = 0, products = 0;
    double fraction = 1;
    VglmState current, trial;
    MatrixXd R;
    // Accumulators and scratch kept for every pass of the fit.
    VglmWorkspace ws;
    // Some pass that formed the full information used the low-rank
    // accumulation (diagonal passes never do, so they do not count).
    bool any_lowrank = false;
    auto evaluate_current = [&](MatrixXd* structure) {
        ++passes;
        if (vglm_pass(data, fam, L, opts, beta, nullptr, current, fraction, structure,
                      iterate_mode, &ws) != VGLM_PASS_OK)
            Rcpp::stop("the model cannot be evaluated at the starting values");
        any_lowrank = any_lowrank || (current.full && current.lowrank);
    };
    // CG solver only: replace the diagonal state at beta by the full one.
    auto evaluate_full = [&]() {
        ++passes;
        if (vglm_pass(data, fam, L, opts, beta, nullptr, current, fraction, nullptr,
                      VGLM_FULL, &ws) != VGLM_PASS_OK)
            Rcpp::stop("the model cannot be evaluated at the current estimate");
        any_lowrank = any_lowrank || current.lowrank;
    };
    // A state used as the information (factor, Firth, convergence
    // confirmation, the returned information) must be a full one.
    auto require_full = [](const VglmState& s) {
        if (!s.full) Rcpp::stop("internal error: a diagonal pass was used as the information");
    };
    evaluate_current(&R);
    const std::vector<bool> aliased = vglm_structural_aliased(R, L);
    {
        // Aliased coefficients are reported as NA, so they must not carry a
        // value into eta (a duplicated intercept column, say).
        bool moved = false;
        for (int j = 0; j < P; ++j)
            if (aliased[j] && beta[j] != 0) { beta[j] = 0; moved = true; }
        if (moved) evaluate_current(nullptr);
    }
    int kept = 0;
    for (int j = 0; j < P; ++j) if (!aliased[j]) ++kept;
    const int max_products = opts.max_products < 0 ? std::min(2 * kept, 250) : opts.max_products;
    auto objective = [&](const VglmState& s) {
        if (firth) require_full(s);
        return firth ? s.loglik + 0.5 * vglm_logdet(s.I, aliased) : s.loglik;
    };
    double current_objective = objective(current);
    if (!std::isfinite(current_objective))
        Rcpp::stop("the information is singular at the starting values");
    double D = -2 * current_objective;
    bool pending = false, converged = false;
    std::string reason = "iteration_limit";
    VglmFactor factor;
    VglmStep step;
    VectorXd candidate(P), adjustment = VectorXd::Zero(P), score;
    MatrixXd V;
    VglmPreconditioner pre;
    // CG solver: the decrement of the step taken in the previous iteration
    // (NaN before the first), which sets the forcing term. It is U' delta of
    // the full direction of that step, even when the accepted step was halved
    // (f < 1): the CG decrement after a CG step, and the exact Fisher
    // decrement after an exact step (fallback or retry). And whether to
    // repeat the iteration with the exact step after no halving of a CG step
    // was accepted.
    double previous_decrement = std::numeric_limits<double>::quiet_NaN();
    bool retry_exact = false, exact_phase = false;
    int reject_confirmations = opts.cg_reject_confirmation;
    int reject_steps = opts.cg_reject_steps;
    // Largest forcing term. 0.1 gave fewer products in all than 0.5 over K
    // in 20 and 50 and p in 11 and 51 (dense, n = 20000): 403 against 416.
    const double forcing_max = 0.1;
    // The exact step at the current estimate from a full state.
    auto exact_direction = [&]() {
        require_full(current);
        vglm_factor(current.I, aliased, rcond_threshold, factor);
        vglm_direction(factor, current.U, step);
    };
    auto decrement_small = [&](double measure) {
        return std::isfinite(measure) && measure / (0.1 + std::abs(D)) < tol;
    };

    for (;;) {
        Rcpp::checkUserInterrupt();
        bool small = false;
        bool exact_step = !cg;
        if (!cg) {
            require_full(current);
            vglm_factor(current.I, aliased, rcond_threshold, factor);
            score = current.U;
            if (firth) {
                vglm_covariance(factor, P);
                // I^{-1} at this iterate, with 0 for the aliased coefficients.
                V = factor.cov;
                for (int j = 0; j < P; ++j)
                    if (aliased[j]) { V.row(j).setZero(); V.col(j).setZero(); }
                ++passes;
                vglm_adjustment(data, fam, L, beta, V, adjustment, &ws);
                score += adjustment;
            }
            vglm_direction(factor, score, step);
            const double lambda2 = step.decrement, measure = lambda2 + step.residual;
            small = decrement_small(measure);
            if (trace)
                Rcpp::Rcout << "iteration " << updates << ": deviance " << D
                            << ", Fisher decrement " << lambda2 << "\n";
            if (pending && small) { converged = true; reason = "converged"; break; }
        } else if (exact_phase || pending || retry_exact) {
            // Exact steps from the full information: from the first time the
            // deviance rule is pending on (that full pass also confirms
            // convergence and is needed for cov.unscaled anyway), and to
            // repeat an iteration whose CG step could not be accepted.
            if (pending && !exact_phase) {
                exact_phase = true;
                iterate_mode = VGLM_FULL;
            }
            if (!current.full) evaluate_full();
            exact_direction();
            small = decrement_small(step.decrement + step.residual);
            if (pending && small && reject_confirmations > 0) {
                // Test only: treat this confirmation as failed.
                --reject_confirmations;
                small = false;
            }
            if (trace)
                Rcpp::Rcout << "iteration " << updates << ": deviance " << D
                            << ", solver exact" << (retry_exact ? " (retry)" : "")
                            << ", Fisher decrement " << step.decrement << "\n";
            // Converged as in the exact solver: pending and small. After a CG
            // step that could not be taken (retry_exact), the exact step is
            // tried first, and the "no acceptable step" rule below decides.
            if (pending && small) { converged = true; reason = "converged"; break; }
            exact_step = true;
            retry_exact = false;
        } else {
            // No step is taken at the iteration limit, so no solve is needed.
            if (updates >= maxit) { reason = "iteration_limit"; break; }
            bool solved = false;
            int used = 0;
            if (!current.full && pre.compute(current, aliased, L)) {
                const double forcing = std::isfinite(previous_decrement)
                    ? std::min(forcing_max, std::sqrt(std::max(previous_decrement, 0.0) / (0.1 + std::abs(D))))
                    : forcing_max;
                solved = vglm_cg_solve(data, fam, L, opts, beta, current.U, aliased, pre,
                                       forcing, max_products, step.delta, used, &ws);
                passes += used;
                products += used;
            }
            if (solved) {
                step.residual = 0;
                step.decrement = 0;
                for (int j = 0; j < P; ++j)
                    if (!aliased[j]) step.decrement += current.U[j] * step.delta[j];
                if (trace)
                    Rcpp::Rcout << "iteration " << updates << ": deviance " << D
                                << ", solver cg, products " << used
                                << ", CG decrement " << step.decrement << "\n";
            } else {
                // Fallback: the exact step from the full information. The
                // deviance rule is not pending here, so this is not a
                // convergence check.
                if (!current.full) evaluate_full();
                exact_direction();
                small = decrement_small(step.decrement + step.residual);
                if (trace)
                    Rcpp::Rcout << "iteration " << updates << ": deviance " << D
                                << ", solver exact (fallback), products " << used
                                << ", Fisher decrement " << step.decrement << "\n";
                exact_step = true;
            }
        }
        pending = false;
        if (updates >= maxit) { reason = "iteration_limit"; break; }
        if (!step.delta.allFinite()) { reason = "step_halving_failed"; break; }

        const double rounding = 64 * eps * (1 + std::abs(current_objective));
        double f = 1, trial_objective = current_objective;
        bool accepted = false, blocked = false;
        // Test only: treat the halving of this CG step as failed.
        const bool forced_reject = !exact_step && reject_steps > 0;
        if (forced_reject) --reject_steps;
        for (int h = 0; h <= max_halves && !accepted && !forced_reject; ++h) {
            candidate = beta + f * step.delta;
            double shrink = 1;
            ++passes;
            const VglmPassStatus status = vglm_pass(data, fam, L, opts, candidate, &beta, trial,
                                                    shrink, nullptr, iterate_mode, &ws);
            any_lowrank = any_lowrank || (status == VGLM_PASS_OK && trial.full && trial.lowrank);
            if (status == VGLM_PASS_SHRINK) {
                if (!(shrink > 0)) { blocked = true; break; }
                f *= shrink;
                continue;
            }
            if (status == VGLM_PASS_OK) {
                trial_objective = objective(trial);
                if (trial_objective >= current_objective - rounding) { accepted = true; continue; }
            }
            f /= 2;
        }
        // A CG step that is blocked or has no accepted halving: repeat the
        // iteration with the exact step, whose own block or failed halving
        // then ends the fit as in the exact solver.
        if ((blocked || !accepted) && !exact_step) { retry_exact = true; continue; }
        if (blocked) { reason = "step_halving_failed"; break; }
        if (!accepted) {
            // No acceptable step along the scoring direction: converged only
            // if the decrement at the current estimate is already small.
            if (small) { converged = true; reason = "converged"; }
            else reason = "step_halving_failed";
            break;
        }
        beta = candidate;
        std::swap(current, trial);
        current_objective = trial_objective;
        ++updates;
        previous_decrement = step.decrement;
        const double D_old = D;
        D = -2 * current_objective;
        if (std::abs(D - D_old) / (0.1 + std::abs(D)) < tol) pending = true;
    }

    // The exact solver's factor belongs to the current iterate on every exit
    // from the loop (current changes only after the factor is built and a
    // step accepted, and the loop then returns to rebuild the factor). The
    // CG solver builds a factor only right after a full pass at beta, so its
    // factor belongs to the current iterate whenever that state is full;
    // otherwise the loop ended on a diagonal state and one full pass at the
    // returned estimate gives the information, score, and covariance (and
    // the exact Fisher decrement reported).
    if (!current.full) {
        evaluate_full();
        exact_direction();
    }
    require_full(current);
    vglm_covariance(factor, P);
    int rank = 0;
    Rcpp::LogicalVector aliased_out(P);
    Rcpp::NumericVector coefficients(P);
    for (int j = 0; j < P; ++j) {
        aliased_out[j] = aliased[j];
        coefficients[j] = aliased[j] ? NA_REAL : beta[j];
        if (!aliased[j]) ++rank;
    }
    // Fitted probabilities numerically 0 or 1, with the threshold of glm.fit
    // (10 times the machine epsilon), so that legitimately small probabilities
    // on wide data are not flagged.
    const double edge = 10 * std::numeric_limits<double>::epsilon();
    const bool separation = current.pmin < edge || current.pmax > 1 - edge;
    // accumulation reports the full-information passes only: "lowrank" when
    // some VGLM_FULL pass used the low-rank accumulation.
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = coefficients,
        Rcpp::Named("cov.unscaled") = Rcpp::wrap(factor.cov),
        Rcpp::Named("information") = Rcpp::wrap(current.I),
        Rcpp::Named("score") = Rcpp::wrap(current.U),
        Rcpp::Named("adjustment") = Rcpp::wrap(adjustment),
        Rcpp::Named("loglik") = current.loglik,
        Rcpp::Named("penalized.loglik") = current_objective,
        Rcpp::Named("deviance") = -2 * current.loglik,
        Rcpp::Named("iter") = updates,
        Rcpp::Named("passes") = passes,
        Rcpp::Named("converged") = converged,
        Rcpp::Named("convergence.reason") = reason,
        Rcpp::Named("rank") = rank,
        Rcpp::Named("aliased") = aliased_out,
        Rcpp::Named("fisher.decrement") = step.decrement,
        Rcpp::Named("singular") = factor.singular,
        Rcpp::Named("separation") = separation,
        Rcpp::Named("accumulation") = any_lowrank ? "lowrank" : "pairwise",
        Rcpp::Named("solver") = opts.solver,
        Rcpp::Named("products") = products);
}

} // namespace fglm_vglm
#endif
