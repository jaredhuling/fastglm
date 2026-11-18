#ifndef FASTGLM_VGLM_ENGINE_H
#define FASTGLM_VGLM_ENGINE_H
#include <RcppEigen.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include "vglm_family.h"
#include "vglm_data.h"
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
// One pass over the data at the coefficients beta.

struct VglmState {
    double loglik = 0;
    MatrixXd I;
    VectorXd U;
    double pmin = 1, pmax = 0;  // extreme fitted probabilities over rows that count
};

enum VglmPassStatus { VGLM_PASS_OK, VGLM_PASS_INVALID, VGLM_PASS_SHRINK };

// Whether row i of a chunk enters the likelihood: positive prior weight and,
// for a grouped response, a positive total count.
template <class Chunk>
inline bool vglm_row_counts(const Chunk& chunk, int i) {
    if (!(chunk.weights[i] > 0)) return false;
    return chunk.codes != nullptr || chunk.counts.row(i).sum() > 0;
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
// linear predictors (0.2.0) will need the factor of each view.
template <class Data>
VglmPassStatus vglm_pass(const Data& data, const VectorFamily& fam, const VglmLayout& L,
                         const VectorXd& beta, const VectorXd* previous, VglmState& out,
                         double& fraction, MatrixXd* structure = nullptr) {
    const int p = L.p, M = L.M, Q = packed_size(M), K = fam.K();
    const bool check_step = previous != nullptr && fam.restricts_step();
    const MatrixXd B = L.predictors(beta);
    MatrixXd Bprev;
    if (check_step) Bprev = L.predictors(*previous);
    std::vector<MatrixXd> G(Q, MatrixXd::Zero(p, p));
    MatrixXd g = MatrixXd::Zero(p, M);
    MatrixXd eta, eta_prev, direction, u, W, P, work, stack;
    if (structure) structure->setZero(p, p);
    VglmPassStatus status = VGLM_PASS_OK;
    double sum = 0, error = 0, pmin = 1, pmax = 0;
    fraction = 1;

    data.visit([&](const typename Data::Chunk& chunk) {
        if (status == VGLM_PASS_INVALID) return;
        const int b = chunk.rows();
        if (b == 0) return;
        const bool shared = vglm_shared_view(chunk, M);
        vglm_linear(chunk, shared, B, eta);
        if (check_step) {
            vglm_linear(chunk, shared, Bprev, eta_prev);
            direction.noalias() = eta - eta_prev;
            const double f = fam.max_step_fraction(eta_prev, direction);
            if (!(f >= 1)) {
                status = VGLM_PASS_SHRINK;
                fraction = std::min(fraction, std::isfinite(f) ? std::max(f, 0.0) : 0.0);
            }
        }
        if (status != VGLM_PASS_OK) return;

        const VglmBlock block{eta, chunk.codes, chunk.counts, chunk.weights};
        if (!fam.valid(block)) { status = VGLM_PASS_INVALID; return; }
        const double ll = fam.loglik(block, nullptr);
        if (!std::isfinite(ll)) { status = VGLM_PASS_INVALID; return; }
        const double next = sum + ll;
        error += std::abs(sum) >= std::abs(ll) ? (sum - next) + ll : (ll - next) + sum;
        sum = next;

        fam.score_information(block, u, W);
        if (!u.allFinite() || !W.allFinite()) { status = VGLM_PASS_INVALID; return; }
        if (shared) vglm_tmul_add(chunk.design[0], u, g);
        else
            for (int a = 0; a < M; ++a) vglm_tmul_add(chunk.design[a], u.col(a), g.col(a));
        for (int bb = 0; bb < M; ++bb)
            for (int a = 0; a <= bb; ++a) {
                const int q = pack_index(a, bb, M);
                vglm_add_cross(G[q], chunk.design[a], W.col(q), chunk.design[bb], work);
            }

        // Extreme fitted probabilities over the rows that enter the likelihood.
        fam.probabilities(eta, P);
        int used = 0;
        for (int i = 0; i < b; ++i) {
            if (!vglm_row_counts(chunk, i)) continue;
            ++used;
            for (int k = 0; k < K; ++k) {
                pmin = std::min(pmin, P(i, k));
                pmax = std::max(pmax, P(i, k));
            }
        }
        if (structure && used > 0) {
            stack.resize(p + used, p);
            stack.topRows(p) = *structure;
            int r = p;
            for (int i = 0; i < b; ++i)
                if (vglm_row_counts(chunk, i)) vglm_copy_row(chunk.design[0], i, stack.row(r++));
            Eigen::HouseholderQR<MatrixXd> qr(stack);
            *structure = qr.matrixQR().topRows(p).template triangularView<Eigen::Upper>();
        }
    });
    if (status != VGLM_PASS_OK) return status;
    out.loglik = sum + error;
    out.pmin = pmin;
    out.pmax = pmax;
    L.map(G, g, out.I, out.U);
    if (!out.I.allFinite() || !out.U.allFinite()) return VGLM_PASS_INVALID;
    return VGLM_PASS_OK;
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
                     const VectorXd& beta, const MatrixXd& V, VectorXd& adjustment) {
    const int p = L.p, M = L.M;
    const MatrixXd B = L.predictors(beta);
    std::vector<MatrixXd> F;
    L.eta_inverse(V, F);
    MatrixXd g = MatrixXd::Zero(p, M), eta, A, t, XF;
    data.visit([&](const typename Data::Chunk& chunk) {
        const int b = chunk.rows();
        if (b == 0) return;
        const bool shared = vglm_shared_view(chunk, M);
        vglm_linear(chunk, shared, B, eta);
        A.resize(b, packed_size(M));
        for (int bb = 0; bb < M; ++bb)
            for (int a = 0; a <= bb; ++a) {
                const int q = pack_index(a, bb, M);
                vglm_mul(chunk.design[a], F[q], XF);
                vglm_row_dots(XF, chunk.design[bb], A.col(q));
            }
        const VglmBlock block{eta, chunk.codes, chunk.counts, chunk.weights};
        fam.bias_terms(block, A, t);
        if (shared) vglm_tmul_add(chunk.design[0], t, g);
        else
            for (int a = 0; a < M; ++a) vglm_tmul_add(chunk.design[a], t.col(a), g.col(a));
    });
    L.score_map(g, adjustment);
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
    MatrixXd cov;            // P by P inverse information, NA at aliased coefficients
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
    MatrixXd inverse(m, m);
    if (m > 0) {
        F.llt.compute(F.C);
        if (F.llt.info() == Eigen::Success && F.llt.rcond() >= rcond_threshold) {
            inverse = F.llt.solve(MatrixXd::Identity(m, m));
        } else {
            F.cod.compute(F.C);
            inverse = F.cod.pseudoInverse();
            F.singular = true;
        }
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

template <class Data>
Rcpp::List vglm_fit(const Data& data, const VectorFamily& fam, const std::vector<MatrixXd>& H,
                    const VectorXd& start, bool firth, double tol, int maxit, bool trace,
                    bool force_mapping = false) {
    if (!(tol > 0) || !std::isfinite(tol) || maxit < 1)
        Rcpp::stop("tol must be positive and finite and maxit at least 1");
    if (static_cast<int>(H.size()) != data.p())
        Rcpp::stop("there must be one constraint matrix per design column");
    const VglmLayout L(H, fam.M(), force_mapping);
    const int P = L.P;
    if (start.size() != P || !start.allFinite())
        Rcpp::stop("starting values must be finite with one entry per coefficient");

    const double rcond_threshold = 1e-12;
    const int max_halves = 30;
    const double eps = std::numeric_limits<double>::epsilon();

    VectorXd beta = start;
    int passes = 0, updates = 0;
    double fraction = 1;
    VglmState current, trial;
    MatrixXd R;
    auto evaluate_current = [&](MatrixXd* structure) {
        ++passes;
        if (vglm_pass(data, fam, L, beta, nullptr, current, fraction, structure) != VGLM_PASS_OK)
            Rcpp::stop("the model cannot be evaluated at the starting values");
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
    auto objective = [&](const VglmState& s) {
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

    for (;;) {
        Rcpp::checkUserInterrupt();
        vglm_factor(current.I, aliased, rcond_threshold, factor);
        score = current.U;
        if (firth) {
            // I^{-1} at this iterate, with 0 for the aliased coefficients.
            V = factor.cov;
            for (int j = 0; j < P; ++j)
                if (aliased[j]) { V.row(j).setZero(); V.col(j).setZero(); }
            ++passes;
            vglm_adjustment(data, fam, L, beta, V, adjustment);
            score += adjustment;
        }
        vglm_direction(factor, score, step);
        const double lambda2 = step.decrement, measure = lambda2 + step.residual;
        const bool small = std::isfinite(measure) && measure / (0.1 + std::abs(D)) < tol;
        if (trace)
            Rcpp::Rcout << "iteration " << updates << ": deviance " << D
                        << ", Fisher decrement " << lambda2 << "\n";
        if (pending && small) { converged = true; reason = "converged"; break; }
        pending = false;
        if (updates >= maxit) { reason = "iteration_limit"; break; }
        if (!step.delta.allFinite()) { reason = "step_halving_failed"; break; }

        const double rounding = 64 * eps * (1 + std::abs(current_objective));
        double f = 1, trial_objective = current_objective;
        bool accepted = false, blocked = false;
        for (int h = 0; h <= max_halves && !accepted; ++h) {
            candidate = beta + f * step.delta;
            double shrink = 1;
            ++passes;
            const VglmPassStatus status = vglm_pass(data, fam, L, candidate, &beta, trial, shrink);
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
        const double D_old = D;
        D = -2 * current_objective;
        if (std::abs(D - D_old) / (0.1 + std::abs(D)) < tol) pending = true;
    }

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
        Rcpp::Named("separation") = separation);
}

} // namespace fglm_vglm
#endif
