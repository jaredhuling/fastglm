#ifndef FASTGLM_BOUNDARY_STEP_H
#define FASTGLM_BOUNDARY_STEP_H

// Boundary-aware IRLS step, shared by the dense and streaming solvers.
//
// When the full weighted least squares update leaves the family's valid
// domain, uniform step-halving shrinks the whole step, and a fit whose
// optimum lies on the boundary of that domain (e.g. eta = 0 under a
// square-root link, or a fitted probability of 1 under a log link) stalls
// above the optimal deviance. This step instead holds the row that leaves the
// domain first at a fixed fraction of the way to its bound and re-solves the
// weighted least squares problem over the remaining directions, repeating
// until the update is valid:
//
//   minimize  sum_{i not in A} W_i (z_i - x_i' beta)^2
//   subject to x_i' beta = c_i for i in A,
//
// where c_i = eta0_i + frac * t_i * d_i, eta0 = X beta (no offset) at the
// current, valid estimate, d = X beta_new - eta0 is the unconstrained step,
// and t_i is the largest fraction of d_i that keeps row i valid (bisection on
// the family's own checks). Held rows have weight zero in the objective
// because their residual is constant on the constraint set; the reduced
// problem is accumulated without them, never by subtracting them from X'WX
// (a held row's weight can be large, so subtraction would cancel digits).
//
// Null-space method: with X_A' P = Q R (column-pivoted QR, Q = [Q1 Q2]),
// beta = Q1 R^{-T} P' c + Q2 g, and g solves the weighted least squares
// problem in X Q2, by blocked normal equations (LLT and LDLT fits) or by the
// incremental Householder QR used for streamed QR fits (all other methods).
//
// The fraction keeps a held row off the bound itself, where its working
// weight can degenerate (1 / mu as mu -> 0 under an identity link) and make
// the local quadratic model useless for moving the row back inward. A row
// whose optimum is on the bound approaches it geometrically over iterations.

#include <RcppEigen.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <limits>
#include "families.h"
#include "streaming_qr.h"

namespace fglm {

constexpr double BOUNDARY_STEP_FRACTION = 0.5;

// Families whose valid linear predictors form a proper convex subset of the
// real line, so an optimum can sit on its boundary. Unknown families (R
// callbacks) are included; their validity is checked row by row.
inline bool bounded_domain(int code)
{
    switch (code) {
    case FAM_BINOMIAL_LOG: case FAM_POISSON_IDENTITY: case FAM_POISSON_SQRT:
    case FAM_GAMMA_INVERSE: case FAM_GAMMA_IDENTITY: case FAM_INVGAUSS_INVMU2:
    case FAM_INVGAUSS_IDENTITY: case FAM_INVGAUSS_INVERSE: case FAM_NB_SQRT:
    case FAM_NB_IDENTITY: case FAM_TWEEDIE_IDENTITY: case FAM_TWEEDIE_INVERSE:
    case FAM_UNKNOWN:
        return true;
    default:
        return false;
    }
}

// valid(e, idx): every e[idx[k]] (a linear predictor including any offset)
// gives a valid linear predictor and mean.
typedef std::function<bool(const Eigen::VectorXd&, const std::vector<int>&)> RowValidity;

// A block of rows: start is the global index of its first row; z and sw (the
// working response without offset and the square-root working weights at the
// current estimate) are supplied when the source is asked for them.
typedef std::function<void(long start, const Eigen::Ref<const Eigen::MatrixXd>& X,
                           const Eigen::VectorXd& off, const Eigen::VectorXd* z,
                           const Eigen::VectorXd* sw)> BlockFn;
// Calls fn on every block of rows, in order; need_working requests z and sw.
typedef std::function<void(const BlockFn& fn, bool need_working)> BlockSource;

// Indices of invalid entries of eta among idx, by recursive halving of the
// index set: O(k log n) validity calls for k invalid entries.
inline void find_invalid_rows(const Eigen::VectorXd& eta, const RowValidity& valid,
                              const std::vector<int>& idx, std::vector<int>& out)
{
    if (idx.empty() || valid(eta, idx)) return;
    if (idx.size() == 1) { out.push_back(idx[0]); return; }
    const std::vector<int> left(idx.begin(), idx.begin() + idx.size() / 2);
    const std::vector<int> right(idx.begin() + idx.size() / 2, idx.end());
    find_invalid_rows(eta, valid, left, out);
    find_invalid_rows(eta, valid, right, out);
}

// Largest t in [0, 1] (to 2^-60) with e0 + t d valid; e0 is valid.
inline double last_valid_fraction(double e0, double d, const RowValidity& valid)
{
    Eigen::VectorXd e(1); const std::vector<int> one(1, 0);
    double lo = 0, hi = 1;
    for (int k = 0; k < 60; ++k) {
        const double m = 0.5 * (lo + hi);
        e[0] = e0 + m * d;
        if (valid(e, one)) lo = m; else hi = m;
    }
    return lo;
}

struct BoundaryRow { long id; Eigen::VectorXd x; double off; };

// Rows at which X beta + offset is invalid, with their data.
inline std::vector<BoundaryRow> invalid_rows(const BlockSource& blocks, const Eigen::VectorXd& beta,
                                             const RowValidity& valid)
{
    std::vector<BoundaryRow> out;
    blocks([&](long start, const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::VectorXd& off,
               const Eigen::VectorXd*, const Eigen::VectorXd*) {
        const Eigen::VectorXd en = X * beta + off;
        std::vector<int> idx(en.size()), bad;
        for (int i = 0; i < (int)idx.size(); ++i) idx[i] = i;
        find_invalid_rows(en, valid, idx, bad);
        for (int i : bad) out.push_back(BoundaryRow{start + i, X.row(i).transpose(), off[i]});
    }, false);
    return out;
}

// Minimize sum W_i (z_i - x_i' beta)^2 over rows not held, subject to the held
// rows' equalities. Returns false if the held rows are linearly dependent or
// the reduced problem is singular.
inline bool constrained_wls(const BlockSource& blocks, int p, const std::vector<long>& held,
                            const Eigen::MatrixXd& XAt, const Eigen::VectorXd& c,
                            bool normal_equations, Eigen::VectorXd& beta)
{
    using Eigen::MatrixXd; using Eigen::VectorXd;
    const int k = held.size();
    Eigen::ColPivHouseholderQR<MatrixXd> qa(XAt);
    if (qa.rank() < k) return false;
    const MatrixXd Q = qa.householderQ() * MatrixXd::Identity(p, p);
    const MatrixXd R = qa.matrixQR().topLeftCorner(k, k).template triangularView<Eigen::Upper>();
    const VectorXd Ptc = qa.colsPermutation().transpose() * c;
    const VectorXd u = R.transpose().triangularView<Eigen::Lower>().solve(Ptc);
    const VectorXd bp = Q.leftCols(k) * u;
    const int m = p - k;
    if (m == 0) { beta = bp; return beta.allFinite(); }
    const MatrixXd Q2 = Q.rightCols(m);

    MatrixXd H; VectorXd r; StreamingQR sq;
    if (normal_equations) { H.setZero(m, m); r.setZero(m); } else sq.reset(m);
    bool ok = true;
    try {
        blocks([&](long start, const Eigen::Ref<const MatrixXd>& X, const VectorXd&,
                   const VectorXd* z, const VectorXd* sw) {
            const int nb = X.rows();
            VectorXd W = sw->array().square();
            for (long id : held) if (id >= start && id < start + nb) W[id - start] = 0;
            const MatrixXd Xr = X * Q2;
            const VectorXd res = *z - X * bp;
            if (normal_equations) {
                const MatrixXd Xs = W.array().sqrt().matrix().asDiagonal() * Xr;
                H.selfadjointView<Eigen::Lower>().rankUpdate(Xs.adjoint());
                r.noalias() += Xr.adjoint() * (W.array() * res.array()).matrix();
            } else sq.add(Xr, res, W);
        }, true);
        VectorXd g;
        if (normal_equations) {
            Eigen::LDLT<MatrixXd> ld(H.selfadjointView<Eigen::Lower>());
            if (ld.info() != Eigen::Success || ldlt_unusable(ld)) return false;
            g = ld.solve(r);
        } else { sq.finish(); g = sq.solve(); }
        beta = bp + Q2 * g;
    } catch (const std::exception&) { ok = false; }   // rank deficient reduced design
    return ok && beta.allFinite();
}

// Whether delta is a descent direction for the deviance at beta. The
// deviance gradient is -2 X' W (z - eta0), with W and z the working weights
// and response at beta and eta0 = X beta (the IRLS score identity), so delta
// is a descent direction when sum_i W_i (z_i - eta0_i) x_i' delta > 0.
inline bool descent_direction(const BlockSource& blocks, const Eigen::VectorXd& beta,
                              const Eigen::VectorXd& delta)
{
    double s = 0;
    blocks([&](long, const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::VectorXd&,
               const Eigen::VectorXd* z, const Eigen::VectorXd* sw) {
        const Eigen::ArrayXd W = sw->array().square();
        s += (W * (z->array() - (X * beta).array()) * (X * delta).array()).sum();
    }, true);
    return std::isfinite(s) && s > 0;
}

// The boundary step. beta: current, valid estimate. beta_new: unconstrained
// update; replaced by the constrained update on success, with n_held the
// number of held rows. Returns false (beta_new unchanged) if the update is
// already valid or no valid constrained update is found.
inline bool boundary_step(const BlockSource& blocks, int p, const Eigen::VectorXd& beta,
                          Eigen::VectorXd& beta_new, const RowValidity& valid,
                          bool normal_equations, int& n_held)
{
    using Eigen::VectorXd;
    std::vector<long> held; Eigen::MatrixXd XAt(p, 0); VectorXd c(0);
    VectorXd cand = beta_new;
    for (int round = 0; round < p; ++round) {
        const std::vector<BoundaryRow> bad = invalid_rows(blocks, cand, valid);
        if (bad.empty()) {
            if (round == 0) return false;
            // Use the constrained update only if it is a descent direction for
            // the deviance. Holding rows at fixed values moves the solution
            // off the line through beta, so this is not automatic; when it
            // fails, the caller halves the unconstrained update instead.
            if (!descent_direction(blocks, beta, cand - beta)) return false;
            beta_new = cand; n_held = held.size(); return true;
        }
        if ((int)held.size() >= p - 1) return false;
        int jbest = -1; double tbest = 2, e0best = 0, dbest = 0;
        for (int j = 0; j < (int)bad.size(); ++j) {
            if (std::find(held.begin(), held.end(), bad[j].id) != held.end()) continue;
            const double e0 = bad[j].x.dot(beta), d = bad[j].x.dot(cand) - e0;
            const double t = last_valid_fraction(e0 + bad[j].off, d, valid);
            if (t < tbest) { tbest = t; jbest = j; e0best = e0; dbest = d; }
        }
        if (jbest < 0) return false;   // a held row is invalid
        held.push_back(bad[jbest].id);
        XAt.conservativeResize(p, held.size()); XAt.col(held.size() - 1) = bad[jbest].x;
        c.conservativeResize(held.size());
        c[held.size() - 1] = e0best + BOUNDARY_STEP_FRACTION * tbest * dbest;
        if (!constrained_wls(blocks, p, held, XAt, c, normal_equations, cand)) return false;
    }
    return false;
}

} // namespace fglm

#endif
