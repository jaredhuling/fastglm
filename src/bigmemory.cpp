#include <iostream>
#include "math.h"
#include "bigmemory.h"
#include <bigmemory/MatrixAccessor.hpp>
#include <bigmemory/BigMatrix.h>


using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
using namespace Rcpp;
using namespace RcppEigen;

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> BigEigenColSums(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& bigMat)
{
  return bigMat.colwise().sum();
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> BigEigenCrossprod(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& bigMat)
{
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXTT;
  int p = bigMat.cols();

  MatrixXTT retmat(p, p);
  retmat.setZero();

  return retmat.template selfadjointView<Eigen::Upper>().rankUpdate( bigMat.adjoint() );
}

// Logic for BigColSums.
template <typename T>
NumericVector BigColSums(XPtr<BigMatrix> pMat, MatrixAccessor<T> mat) {

  // Create the vector we'll store the column sums in.
  NumericVector colSums(pMat->ncol());
  for (size_t i=0; i < pMat->ncol(); ++i)
    colSums[i] = std::accumulate(mat[i], mat[i]+pMat->nrow(), 0.0);
  return colSums;
}

RcppExport SEXP crossprod_big(SEXP X_)
{
  using namespace Rcpp;
  using namespace RcppEigen;
  try {
    using Eigen::Map;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    typedef Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> MatrixXchar;
    typedef Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic> MatrixXshort;

    XPtr<BigMatrix> bMPtr(X_);


    unsigned int type = bMPtr->matrix_type();


    if (type == 1)
    {
      Map<MatrixXchar> bM = Map<MatrixXchar>((char *)bMPtr->matrix(), bMPtr->nrow(), bMPtr->ncol()  );
      int p = bM.cols();
      MatrixXchar crossprod = MatrixXchar(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate( bM.adjoint() );
      return wrap(crossprod);
    } else if (type == 2)
    {
      Map<MatrixXshort> bM = Map<MatrixXshort>((short *)bMPtr->matrix(), bMPtr->nrow(), bMPtr->ncol()  );
      int p = bM.cols();
      MatrixXshort crossprod = MatrixXshort(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate( bM.adjoint() );
      return wrap(crossprod);
    } else if (type == 4)
    {
      Map<MatrixXi> bM = Map<MatrixXi>((int *)bMPtr->matrix(), bMPtr->nrow(), bMPtr->ncol()  );
      int p = bM.cols();
      MatrixXi crossprod = MatrixXi(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate( bM.adjoint() );
      return wrap(crossprod);
    } else if (type == 6)
    {
      Map<MatrixXf> bM = Map<MatrixXf>((float *)bMPtr->matrix(), bMPtr->nrow(), bMPtr->ncol()  );
      int p = bM.cols();
      MatrixXf crossprod = MatrixXf(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate( bM.adjoint() );
      return wrap(crossprod);
    } else if (type == 8)
    {
      Map<MatrixXd> bM = Map<MatrixXd>((double *)bMPtr->matrix(), bMPtr->nrow(), bMPtr->ncol()  );
      int p = bM.cols();
      MatrixXd crossprod = MatrixXd(p, p).setZero().selfadjointView<Eigen::Upper>().rankUpdate( bM.adjoint() );
      return wrap(crossprod);
    } else {
      // We should never get here, but it resolves compiler warnings.
      throw Rcpp::exception("Undefined type for provided big.matrix");
    }

  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return R_NilValue; //-Wall
}

RcppExport SEXP colsums_big(SEXP X_)
{
  BEGIN_RCPP
    using Eigen::Map;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    typedef Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> MatrixXchar;
    typedef Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic> MatrixXshort;
    typedef Eigen::Matrix<char, Eigen::Dynamic, 1> Vectorchar;
    typedef Eigen::Matrix<short, Eigen::Dynamic, 1> Vectorshort;

    XPtr<BigMatrix> xpMat(X_);


    unsigned int type = xpMat->matrix_type();
    // The data stored in the big.matrix can either be represent by 1, 2,
    // 4, 6, or 8 bytes. See the "type" argument in `?big.matrix`.
    if (type == 1)
    {
      Map<MatrixXchar> bM = Map<MatrixXchar>((char *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
      Vectorchar colSums = bM.colwise().sum();
      return wrap(colSums);
    } else if (type == 2)
    {
      Map<MatrixXshort> bM = Map<MatrixXshort>((short *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
      Vectorshort colSums = bM.colwise().sum();
      return wrap(colSums);
    } else if (type == 4)
    {
      Map<MatrixXi> bM = Map<MatrixXi>((int *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
      VectorXi colSums = bM.colwise().sum();
      return wrap(colSums);
    } else if (type == 6)
    {
      Map<MatrixXf> bM = Map<MatrixXf>((float *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
      VectorXf colSums = bM.colwise().sum();
      return wrap(colSums);
    } else if (type == 8)
    {
      Map<MatrixXd> bM = Map<MatrixXd>((double *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
      VectorXd colSums = bM.colwise().sum();
      return wrap(colSums);
    } else {
      // We should never get here, but it resolves compiler warnings.
      throw Rcpp::exception("Undefined type for provided big.matrix");
    }

    END_RCPP
}



RcppExport SEXP colmax_big(SEXP X_)
{
  BEGIN_RCPP
  using Eigen::Map;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  typedef Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> MatrixXchar;
  typedef Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic> MatrixXshort;
  typedef Eigen::Matrix<char, Eigen::Dynamic, 1> Vectorchar;
  typedef Eigen::Matrix<short, Eigen::Dynamic, 1> Vectorshort;

  XPtr<BigMatrix> xpMat(X_);


  unsigned int type = xpMat->matrix_type();
  // The data stored in the big.matrix can either be represent by 1, 2,
  // 4, 6, or 8 bytes. See the "type" argument in `?big.matrix`.
  if (type == 1)
  {
    Map<MatrixXchar> bM = Map<MatrixXchar>((char *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
    Vectorchar colSums = bM.colwise().maxCoeff();
    return wrap(colSums);
  } else if (type == 2)
  {
    Map<MatrixXshort> bM = Map<MatrixXshort>((short *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
    Vectorshort colSums = bM.colwise().maxCoeff();
    return wrap(colSums);
  } else if (type == 4)
  {
    Map<MatrixXi> bM = Map<MatrixXi>((int *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
    VectorXi colSums = bM.colwise().maxCoeff();
    return wrap(colSums);
  } else if (type == 6)
  {
    Map<MatrixXf> bM = Map<MatrixXf>((float *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
    VectorXf colSums = bM.colwise().maxCoeff();
    return wrap(colSums);
  } else if (type == 8)
  {
    Map<MatrixXd> bM = Map<MatrixXd>((double *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
    VectorXd colSums = bM.colwise().maxCoeff();
    return wrap(colSums);
  } else {
    // We should never get here, but it resolves compiler warnings.
    throw Rcpp::exception("Undefined type for provided big.matrix");
  }

  END_RCPP
}


RcppExport SEXP colmin_big(SEXP X_)
{
  BEGIN_RCPP
  using Eigen::Map;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  typedef Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> MatrixXchar;
  typedef Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic> MatrixXshort;
  typedef Eigen::Matrix<char, Eigen::Dynamic, 1> Vectorchar;
  typedef Eigen::Matrix<short, Eigen::Dynamic, 1> Vectorshort;

  XPtr<BigMatrix> xpMat(X_);


  unsigned int type = xpMat->matrix_type();
  // The data stored in the big.matrix can either be represent by 1, 2,
  // 4, 6, or 8 bytes. See the "type" argument in `?big.matrix`.
  if (type == 1)
  {
    Map<MatrixXchar> bM = Map<MatrixXchar>((char *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
    Vectorchar colSums = bM.colwise().minCoeff();
    return wrap(colSums);
  } else if (type == 2)
  {
    Map<MatrixXshort> bM = Map<MatrixXshort>((short *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
    Vectorshort colSums = bM.colwise().minCoeff();
    return wrap(colSums);
  } else if (type == 4)
  {
    Map<MatrixXi> bM = Map<MatrixXi>((int *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
    VectorXi colSums = bM.colwise().minCoeff();
    return wrap(colSums);
  } else if (type == 6)
  {
    Map<MatrixXf> bM = Map<MatrixXf>((float *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
    VectorXf colSums = bM.colwise().minCoeff();
    return wrap(colSums);
  } else if (type == 8)
  {
    Map<MatrixXd> bM = Map<MatrixXd>((double *)xpMat->matrix(), xpMat->nrow(), xpMat->ncol()  );
    VectorXd colSums = bM.colwise().minCoeff();
    return wrap(colSums);
  } else {
    // We should never get here, but it resolves compiler warnings.
    throw Rcpp::exception("Undefined type for provided big.matrix");
  }

  END_RCPP
}








RcppExport SEXP prod_vec_big(SEXP A_, SEXP B_)
{
  BEGIN_RCPP
  using Eigen::Map;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using Eigen::VectorXi;
  using Eigen::VectorXf;

  typedef Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> MatrixXchar;
  typedef Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic> MatrixXshort;
  typedef Eigen::Matrix<char, Eigen::Dynamic, 1> Vectorchar;
  typedef Eigen::Matrix<short, Eigen::Dynamic, 1> Vectorshort;
  typedef Map<VectorXd> MapVecd;
  typedef Map<VectorXi> MapVeci;
  typedef Map<VectorXf> MapVecf;

  XPtr<BigMatrix> ApMat(A_);


  unsigned int Atype = ApMat->matrix_type();


  // The data stored in the big.matrix can either be represent by 1, 2,
  // 4, 6, or 8 bytes. See the "type" argument in `?big.matrix`.
  if (Atype == 1)
  {
    throw Rcpp::exception("Unavailable type for provided big.matrix");
  } else if (Atype == 2)
  {
    throw Rcpp::exception("Unavailable type for provided big.matrix");
  } else if (Atype == 4)
  {
    Map<MatrixXi> bA = Map<MatrixXi>((int *)ApMat->matrix(), ApMat->nrow(), ApMat->ncol()  );
    const MapVeci B(as<MapVeci>(B_));

    if (ApMat->ncol() != B.size())
    {
      throw Rcpp::exception("Dimensions imcompatible");
    }


    VectorXi prod = bA * B;
    return wrap(prod);
  } else if (Atype == 6)
  {
    throw Rcpp::exception("Unavailable type for provided big.matrix");
  } else if (Atype == 8)
  {
    Map<MatrixXd> bA = Map<MatrixXd>((double *)ApMat->matrix(), ApMat->nrow(), ApMat->ncol()  );
    const MapVecd B(as<MapVecd>(B_));

    if (ApMat->ncol() != B.size())
    {
      throw Rcpp::exception("Dimensions imcompatible");
    }

    VectorXd prod = bA * B;
    return wrap(prod);
  } else {
    // We should never get here, but it resolves compiler warnings.
    throw Rcpp::exception("Undefined type for provided big.matrix");
  }

  return NULL;

  END_RCPP
}


RcppExport SEXP prod_vec_big_right(SEXP A_, SEXP B_)
{
  BEGIN_RCPP
  using Eigen::Map;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using Eigen::VectorXi;
  using Eigen::VectorXf;

  typedef Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> MatrixXchar;
  typedef Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic> MatrixXshort;
  typedef Eigen::Matrix<char, Eigen::Dynamic, 1> Vectorchar;
  typedef Eigen::Matrix<short, Eigen::Dynamic, 1> Vectorshort;
  typedef Map<VectorXd> MapVecd;
  typedef Map<VectorXi> MapVeci;
  typedef Map<VectorXf> MapVecf;

  XPtr<BigMatrix> ApMat(B_);


  unsigned int Atype = ApMat->matrix_type();


  // The data stored in the big.matrix can either be represent by 1, 2,
  // 4, 6, or 8 bytes. See the "type" argument in `?big.matrix`.
  if (Atype == 1)
  {
    throw Rcpp::exception("Unavailable type for provided big.matrix");
  } else if (Atype == 2)
  {
    throw Rcpp::exception("Unavailable type for provided big.matrix");
  } else if (Atype == 4)
  {
    Map<MatrixXi> bA = Map<MatrixXi>((int *)ApMat->matrix(), ApMat->nrow(), ApMat->ncol()  );
    const MapVeci A(as<MapVeci>(A_));

    if (ApMat->nrow() != A.size())
    {
      throw Rcpp::exception("Dimensions imcompatible");
    }


    VectorXi prod = A * bA;
    return wrap(prod);
  } else if (Atype == 6)
  {
    throw Rcpp::exception("Unavailable type for provided big.matrix");
  } else if (Atype == 8)
  {
    Map<MatrixXd> bA = Map<MatrixXd>((double *)ApMat->matrix(), ApMat->nrow(), ApMat->ncol()  );
    const MapVecd A(as<MapVecd>(A_));

    if (ApMat->nrow() != A.size())
    {
      throw Rcpp::exception("Dimensions imcompatible");
    }

    VectorXd prod = A * bA;
    return wrap(prod);
  } else {
    // We should never get here, but it resolves compiler warnings.
    throw Rcpp::exception("Undefined type for provided big.matrix");
  }

  return NULL;

  END_RCPP
}







