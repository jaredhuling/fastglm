#ifndef _bigFastlm_BIGMEMORY_H
#define _bigFastlm_BIGMEMORY_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/SVD>
#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include <cmath>

using namespace Rcpp;
using namespace RcppEigen;


RcppExport SEXP crossprod_big(SEXP);

RcppExport SEXP colsums_big(SEXP);

RcppExport SEXP colmax_big(SEXP);

RcppExport SEXP colmin_big(SEXP);

RcppExport SEXP prod_vec_big(SEXP, SEXP);

RcppExport SEXP prod_vec_right(SEXP, SEXP);

#endif
