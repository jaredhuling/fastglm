#include "ordinal_partial.h"
#include <chrono>
// [[Rcpp::depends(RcppEigen)]]
using namespace fglm_ordinal_partial;

// This study harness adds timers around the unchanged native operations.
// It is not part of the installed package or an alternative estimator.
// [[Rcpp::export]]
Rcpp::List profile_partial_operations(Rcpp::NumericMatrix x,Rcpp::NumericMatrix z,
    Rcpp::NumericMatrix counts,Rcpp::NumericVector weights,Rcpp::NumericVector offset,
    int categories,Rcpp::NumericVector coefficients,bool firth,int link,int repetitions=5) {
    PartialOrdinalData data(x,z,Rcpp::IntegerVector(),counts,weights,offset,categories,2048,link);
    const VectorXd b=Rcpp::as<VectorXd>(coefficients);
    using Clock=std::chrono::steady_clock;
    auto seconds=[](Clock::time_point a,Clock::time_point c) {
        return std::chrono::duration<double>(c-a).count();
    };
    Rcpp::NumericMatrix times(repetitions,6);
    Rcpp::colnames(times)=Rcpp::CharacterVector::create("information","factorization",
        "adjustment","score_update","predictor_scan","feasibility_scan");
    PartialOrdinalState last(data.d);
    double fraction=1;
    for(int repeat=0;repeat<repetitions;++repeat) {
        PartialOrdinalState state(data.d);
        const auto a=Clock::now();partial_add_information(data,b,state);
        const auto c=Clock::now();ordinal_factor_information(state,data.d-data.q,data.q,1e-14);
        const auto e=Clock::now();if(firth) partial_add_adjustment(data,b,state);
        const auto f=Clock::now();state.score=state.ordinary+state.adjustment;
        state.step.noalias()=state.inverse*state.score;
        state.scaled_score=(state.score.array().abs()/state.information.diagonal().array().sqrt()).maxCoeff();
        const auto g=Clock::now();state.predictor_step=data.predictor_step(state.step);
        const auto h=Clock::now();
        VectorXd direction=state.step/std::max(1.0,state.predictor_step/(firth ? 1.0 : 4.0));
        fraction=data.feasible_fraction(b,direction);
        const auto j=Clock::now();
        times(repeat,0)=seconds(a,c);times(repeat,1)=seconds(c,e);
        times(repeat,2)=seconds(e,f);times(repeat,3)=seconds(f,g);
        times(repeat,4)=seconds(g,h);times(repeat,5)=seconds(h,j);
        last=std::move(state);
    }
    return Rcpp::List::create(Rcpp::Named("seconds")=times,
        Rcpp::Named("evaluation")=partial_state_list(last),Rcpp::Named("fraction")=fraction);
}
