#ifndef FASTGLM_VGLM_FAMILY_H
#define FASTGLM_VGLM_FAMILY_H
#include <RcppEigen.h>
#include <memory>
#include <string>
namespace fglm_vglm {
using Eigen::MatrixXd;
using Eigen::VectorXd;

// One block of b rows as a model type sees it. eta holds the M linear
// predictors of each row. An individual response gives 1-based category
// codes (counts is empty); a grouped response gives a b by K matrix of
// counts (codes is nullptr). weights are the prior weights.
struct VglmBlock {
    Eigen::Ref<const MatrixXd> eta;     // b x M
    const int* codes;                   // 1-based categories, nullptr for grouped
    Eigen::Ref<const MatrixXd> counts;  // b x K, empty for individual
    Eigen::Ref<const VectorXd> weights; // b

    int rows() const { return static_cast<int>(eta.rows()); }
    bool grouped() const { return codes == nullptr; }
};

// Position of the pair (a, b), a <= b, in the packed upper triangle of an
// M by M symmetric matrix stored column by column: column b holds the
// entries a = 0, ..., b and is preceded by b(b + 1)/2 entries.
inline int pack_index(int a, int b, int M) {
    (void)M;
    return b * (b + 1) / 2 + a;
}

inline int packed_size(int M) { return M * (M + 1) / 2; }

// A model type for a vector GLM. Every method acts on a block of rows, so the
// engine can stream data through it without per-row virtual calls.
class VectorFamily {
public:
    virtual ~VectorFamily() {}

    // Number of linear predictors per row.
    virtual int M() const = 0;
    // Number of response categories.
    virtual int K() const = 0;
    virtual std::string name() const = 0;

    // Whether every row's linear predictors lie in the valid region.
    virtual bool valid(const VglmBlock& block) const = 0;

    // Sum of the weighted log likelihood contributions; when rows is not
    // null it is resized to b and filled with the per-row contributions.
    virtual double loglik(const VglmBlock& block, VectorXd* rows) const = 0;

    // Row scores u_i = d l_i / d eta_i (b x M) and the packed upper triangle
    // of the row expected information W_i (b x M(M+1)/2), both including the
    // prior weight.
    virtual void score_information(const VglmBlock& block, MatrixXd& u,
                                   MatrixXd& W) const = 0;

    // Fitted category probabilities (b x K).
    virtual void probabilities(const MatrixXd& eta, MatrixXd& P) const = 0;

    // Mean bias reduction terms: given packed A_i (b x M(M+1)/2), returns
    // t_i[m] = (1/2) sum_{a,b} A_i[a,b] dW_i[a,b] / d eta_m (b x M), where
    // the sum runs over all ordered pairs of the symmetric A_i and W_i.
    virtual void bias_terms(const VglmBlock& block, const MatrixXd& A,
                            MatrixXd& t) const = 0;

    // Starting linear predictors (length M) from the K category totals.
    virtual VectorXd start_eta(const VectorXd& category_totals) const = 0;

    // Whether max_step_fraction can return less than 1. When false (every
    // eta is valid, as for multinomial) the engine skips the per-block work
    // of forming the change in eta for max_step_fraction. A model type that
    // overrides max_step_fraction must also override this to return true.
    virtual bool restricts_step() const { return false; }

    // Largest fraction of a proposed change in eta that keeps every row
    // valid; used before step halving.
    virtual double max_step_fraction(const MatrixXd& eta,
                                     const MatrixXd& direction) const {
        (void)eta;
        (void)direction;
        return 1.0;
    }
};

// Registry of model types by name. Defined once, in src/fit_vglm.cpp, so the
// table of names lives in one place; other translation units link to it.
// Throws std::invalid_argument for an unknown name or invalid K/reference.
std::unique_ptr<VectorFamily> make_vector_family(const std::string& name, int K,
                                                 int reference,
                                                 const Rcpp::List& parameters);

} // namespace fglm_vglm
#endif
