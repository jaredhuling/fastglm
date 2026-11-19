#ifndef FASTGLM_VGLM_MULTINOMIAL_H
#define FASTGLM_VGLM_MULTINOMIAL_H
#include "vglm_family.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
namespace fglm_vglm {

// Baseline-category (multinomial) logit. The M = K - 1 linear predictors are
// the log odds of the non-reference categories, in level order, against the
// reference category, whose linear predictor is fixed at 0.
//
// Log likelihood: l_i = w_i sum_k y_ik log pi_ik, without the multinomial
// coefficient, so it matches nnet::multinom for individual data and differs
// from the full grouped likelihood by a constant that does not depend on the
// coefficients.
class MultinomialFamily : public VectorFamily {
public:
    MultinomialFamily(int K, int reference) : K_(K), M_(K - 1), ref_(reference) {
        if (K < 2) throw std::invalid_argument("multinomial requires at least 2 categories");
        if (reference < 0 || reference >= K)
            throw std::invalid_argument("multinomial reference category out of range");
    }

    int M() const override { return M_; }
    int K() const override { return K_; }
    std::string name() const override { return "multinomial"; }

    // 0-based category of linear predictor m.
    int category(int m) const { return m < ref_ ? m : m + 1; }

    bool valid(const VglmBlock& block) const override {
        return block.eta.allFinite();
    }

    double loglik(const VglmBlock& block, VectorXd* rows) const override {
        check(block);
        const int b = block.rows();
        if (rows) rows->setZero(b);
        VectorXd logp(K_);
        double sum = 0, error = 0;
        for (int i = 0; i < b; ++i) {
            const double w = block.weights[i];
            double li = 0;
            if (w != 0) {
                log_probabilities(block.eta, i, logp);
                if (block.grouped()) {
                    for (int k = 0; k < K_; ++k) {
                        const double y = block.counts(i, k);
                        if (y != 0) li += w * y * logp[k];
                    }
                } else {
                    li = w * logp[block.codes[i] - 1];
                }
            }
            if (rows) (*rows)[i] = li;
            const double next = sum + li;
            error += std::abs(sum) >= std::abs(li) ? (sum - next) + li : (li - next) + sum;
            sum = next;
        }
        return sum + error;
    }

    void score_information(const VglmBlock& block, MatrixXd& u,
                           MatrixXd& W) const override {
        check(block);
        const int b = block.rows();
        u.setZero(b, M_);
        W.setZero(b, packed_size(M_));
        VectorXd pk(K_), p(M_);
        for (int i = 0; i < b; ++i) {
            double c;
            if (!row_score(block, i, pk, p, u, c)) continue;
            for (int bb = 0; bb < M_; ++bb) {
                for (int a = 0; a < bb; ++a)
                    W(i, pack_index(a, bb, M_)) = -c * p[a] * p[bb];
                W(i, pack_index(bb, bb, M_)) = c * p[bb] * (1.0 - p[bb]);
            }
        }
    }

    // W_i = c (diag(p) - p p') with c = w n, so d_i = c p and C_i = sqrt(c) p
    // (rank 1). Rows with c = 0 have zero u, d, and C.
    int information_rank() const override { return 1; }

    void score_information_lowrank(const VglmBlock& block, MatrixXd& u, MatrixXd& d,
                                   MatrixXd& C) const override {
        check(block);
        const int b = block.rows();
        u.setZero(b, M_);
        d.setZero(b, M_);
        C.setZero(b, M_);
        VectorXd pk(K_), p(M_);
        for (int i = 0; i < b; ++i) {
            double c;
            if (!row_score(block, i, pk, p, u, c)) continue;
            const double root = std::sqrt(c);
            for (int m = 0; m < M_; ++m) {
                d(i, m) = c * p[m];
                C(i, m) = root * p[m];
            }
        }
    }

    void probabilities(const MatrixXd& eta, MatrixXd& P) const override {
        if (eta.cols() != M_) throw std::invalid_argument("eta must have K - 1 columns");
        const int b = static_cast<int>(eta.rows());
        P.resize(b, K_);
        VectorXd pk(K_);
        for (int i = 0; i < b; ++i) {
            probability_row(eta, i, pk);
            P.row(i) = pk.transpose();
        }
    }

    // With W = c (diag(p) - p p') over the non-reference levels and
    // dp_a / d eta_m = p_a (delta_am - p_m),
    //   sum_ab A_ab dW_ab / d eta_m
    //     = c p_m [A_mm - sum_a A_aa p_a - 2 (A p)_m + 2 p' A p].
    void bias_terms(const VglmBlock& block, const MatrixXd& A,
                    MatrixXd& t) const override {
        check(block);
        const int b = block.rows();
        if (A.rows() != b || A.cols() != packed_size(M_))
            throw std::invalid_argument("A must be b by M(M + 1)/2");
        t.setZero(b, M_);
        VectorXd pk(K_), p(M_), Ap(M_);
        for (int i = 0; i < b; ++i) {
            const double w = block.weights[i];
            if (w == 0) continue;
            const double c = w * total(block, i);
            if (c == 0) continue;
            probability_row(block.eta, i, pk);
            for (int m = 0; m < M_; ++m) p[m] = pk[category(m)];
            double trace = 0;
            for (int a = 0; a < M_; ++a) {
                double s = 0;
                for (int bb = 0; bb < M_; ++bb)
                    s += A(i, a <= bb ? pack_index(a, bb, M_) : pack_index(bb, a, M_)) * p[bb];
                Ap[a] = s;
                trace += A(i, pack_index(a, a, M_)) * p[a];
            }
            const double quad = p.dot(Ap);
            for (int m = 0; m < M_; ++m)
                t(i, m) = 0.5 * c * p[m] *
                    (A(i, pack_index(m, m, M_)) - trace - 2.0 * Ap[m] + 2.0 * quad);
        }
    }

    // Log of the marginal proportions relative to the reference.
    VectorXd start_eta(const VectorXd& category_totals) const override {
        if (category_totals.size() != K_)
            throw std::invalid_argument("category totals must have length K");
        for (int k = 0; k < K_; ++k)
            if (!(category_totals[k] > 0) || !std::isfinite(category_totals[k]))
                throw std::invalid_argument("category " + std::to_string(k + 1) +
                                            " has no positive finite total");
        VectorXd eta(M_);
        const double base = std::log(category_totals[ref_]);
        for (int m = 0; m < M_; ++m) eta[m] = std::log(category_totals[category(m)]) - base;
        return eta;
    }

private:
    int K_, M_, ref_;

    void check(const VglmBlock& block) const {
        const int b = block.rows();
        if (block.eta.cols() != M_) throw std::invalid_argument("eta must have K - 1 columns");
        if (block.weights.size() != b) throw std::invalid_argument("weights must have one entry per row");
        if (block.grouped()) {
            if (block.counts.rows() != b || block.counts.cols() != K_)
                throw std::invalid_argument("counts must be b by K");
        } else {
            for (int i = 0; i < b; ++i)
                if (block.codes[i] < 1 || block.codes[i] > K_)
                    throw std::invalid_argument("response category code out of range");
        }
    }

    // The terms of row i shared by score_information and
    // score_information_lowrank: the probabilities p of the non-reference
    // categories (pk holds all K), row i of the score u, and c = w n, the
    // scale of the row information W_i = c (diag(p) - p p'). Returns false,
    // leaving p, u, and c unset, for a row of zero prior weight.
    bool row_score(const VglmBlock& block, int i, VectorXd& pk, VectorXd& p, MatrixXd& u,
                   double& c) const {
        const double w = block.weights[i];
        if (w == 0) return false;
        probability_row(block.eta, i, pk);
        const double n = total(block, i);
        c = w * n;
        for (int m = 0; m < M_; ++m) {
            p[m] = pk[category(m)];
            u(i, m) = w * (response(block, i, category(m)) - n * p[m]);
        }
        return true;
    }

    double response(const VglmBlock& block, int i, int k) const {
        return block.grouped() ? block.counts(i, k) : (block.codes[i] - 1 == k ? 1.0 : 0.0);
    }

    double total(const VglmBlock& block, int i) const {
        if (!block.grouped()) return 1.0;
        double n = 0;
        for (int k = 0; k < K_; ++k) n += block.counts(i, k);
        return n;
    }

    // Stable log-sum-exp with the reference linear predictor fixed at 0.
    // With z_k = eta_k - max(0, eta) and s = sum_k exp(z_k), log pi_k =
    // z_k - log(s). Forming z_k first avoids subtracting the large number
    // max(0, eta) + log(s), which would cost accuracy when |eta| is large.
    template <class Eta>
    void log_probabilities(const Eta& eta, int i, VectorXd& logp) const {
        double top = 0;
        for (int m = 0; m < M_; ++m) top = std::max(top, eta(i, m));
        logp[ref_] = -top;
        for (int m = 0; m < M_; ++m) logp[category(m)] = eta(i, m) - top;
        double s = 0;
        for (int k = 0; k < K_; ++k) s += std::exp(logp[k]);
        const double log_s = std::log(s);
        for (int k = 0; k < K_; ++k) logp[k] -= log_s;
    }

    // Probabilities from the same shifted values, as exp(z_k) / s, so that
    // each has a small relative error and the row sums to 1 to rounding.
    template <class Eta>
    void probability_row(const Eta& eta, int i, VectorXd& p) const {
        double top = 0;
        for (int m = 0; m < M_; ++m) top = std::max(top, eta(i, m));
        p[ref_] = std::exp(-top);
        double s = p[ref_];
        for (int m = 0; m < M_; ++m) {
            p[category(m)] = std::exp(eta(i, m) - top);
            s += p[category(m)];
        }
        p /= s;
    }
};

} // namespace fglm_vglm
#endif
