---
title: 'fastglm: An R package for fast and stable generalized linear model fitting'
tags:
  - statistics
  - regression
  - generalized linear models
  - statistical computing
  - R language
authors:
  - name: Jared Davis Huling
    orcid: 0000-0003-0670-4845
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Statistics, The Ohio State University
   index: 1
date: 01 March 2019
bibliography: paper.bib
---

# Summary

The `fastglm` R package contains an efficient and stable implementation of iteratively
reweighted least squares (IRLS) for general purpose fitting of generalized linear models (GLMs). The 
implementation of IRLS utilizes a step-halving approach as was used in the `glm2` package 
[@marschner2011glm2] for mitigating divergence issues due boundary violations and other issues that 
often arise when using non-canonical link functions in GLMs. The `fastglm` package combines this with 
parameter initialization in a manner that makes it more robust to divergence problems than 
the base R function `glm()` and the `glm2()` function of the `glm2` package.
The package is written in C++ using the efficient Eigen numerical linear algebra library [@eigenweb]
and is delivered as a package for the R language and environment for statistical computing [@R].

The `fastglm` package is designed to be used concurrently with the `family` class of objects, enabling
its use for a wide variety of GLMs. Further, the API and returned objects make the `fastglm` package
easily-usable by practitioners already familiar with fitting GLMs in R. The `fastglm` package includes 
thorough documentation and usage vignettes that allow for users to quickly and thoroughly understand 
how to utilize the package for their own data analyses.

The `fastglm` package utilizes an IRLS algorithm where at each iteration the weighted least squares 
subproblem is solved via one of six different linear solvers which range along a spectrum between 
computational speed and stability: 1) the column-pivoted QR decomposition (stable, moderately fast) 
2) the unpivoted QR decomposition (moderately stable, fast) 3) LDLT Cholesky decomposition (reasonably 
stable, very fast) 4) LLT Cholesky decomposition (somewhat stable, fastest) 5) the full pivoted QR 
decomposition (very stable, not so fast) 6) the Bidiagonal divide and conquer SVD (the stablest, 
the slowest).





# References