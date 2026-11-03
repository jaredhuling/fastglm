# Exact finite-sample study of endpoint-empty meanBR bootstrap inference.
# This development script enumerates all three-category count vectors, so
# outer coverage and bootstrap availability have no Monte Carlo error. The
# fitted meanBR threshold estimates reduce to the endpoint half-count form in
# this intercept-only setting; the equations in src/fit_ordinal.cpp motivate
# that reduction. Refit distributions are conditioned on retaining the
# interior category, as required to keep both thresholds in the model.

compositions <- function(n, k=3L) {
  if (k == 1L) return(matrix(n, nrow=1L))
  do.call(rbind, lapply(0:n, function(a) cbind(a, compositions(n-a,k-1L))))
}

meanbr_thresholds <- function(y) {
  n <- sum(y)
  c(qlogis((y[1]+.5)/(n+1)), qlogis((y[1]+y[2]+.5)/(n+1)))
}

category_probabilities <- function(alpha) {
  cumulative <- plogis(alpha)
  c(cumulative[1], diff(cumulative), 1-cumulative[2])
}

multinomial_weights <- function(counts, probabilities) {
  n <- rowSums(counts)[1]
  logw <- lgamma(n+1)-rowSums(lgamma(counts+1)) +
    as.vector(counts %*% log(probabilities))
  w <- exp(logw-max(logw))
  w/sum(w)
}

weighted_quantile <- function(x, w, probability) {
  order <- order(x)
  x <- x[order]
  w <- w[order]/sum(w)
  x[which(cumsum(w) >= probability)[1]]
}

one_scenario <- function(truth, n, B=199L) {
  Y <- compositions(n)
  empty.interior <- Y[,2] == 0
  eligible <- which(!empty.interior)
  estimates <- matrix(NA_real_, nrow(Y), 2L)
  estimates[eligible,] <- t(apply(Y[eligible,,drop=FALSE], 1, meanbr_thresholds))
  outer.weights <- multinomial_weights(Y, category_probabilities(truth))
  full <- rowSums(Y > 0) == 3L

  # Probability that all B bootstrap samples preserve every declared
  # category (current policy), or preserve the interior category (candidate).
  current.availability <- vapply(which(full), function(i) {
    p <- category_probabilities(estimates[i,])
    one <- 1-sum((1-p)^n)+sum(p^n)
    one^B
  }, numeric(1))
  endpoint.allowed.availability <- vapply(eligible, function(i) {
    p <- category_probabilities(estimates[i,])
    (1-(1-p[2])^n)^B
  }, numeric(1))

  # Exact ideal percentile limits (inverse-CDF quantiles), conditional on a
  # bootstrap refit retaining the interior category. This removes finite-B
  # quantile noise but not the conditioning induced by the safeguard.
  covered <- matrix(FALSE, length(eligible), 2L)
  for (j in seq_along(eligible)) {
    i <- eligible[j]
    weights <- multinomial_weights(Y, category_probabilities(estimates[i,]))
    weights <- weights[eligible]/sum(weights[eligible])
    draws <- estimates[eligible,,drop=FALSE]
    lower <- apply(draws, 2, weighted_quantile, w=weights, probability=.025)
    upper <- apply(draws, 2, weighted_quantile, w=weights, probability=.975)
    covered[j,] <- truth >= lower & truth <= upper
  }

  candidate.weights <- outer.weights[eligible]*endpoint.allowed.availability
  eligible.probability <- sum(outer.weights[eligible])
  c(n=n,
    endpoint_empty_probability=sum(outer.weights[
      rowSums(Y[,c(1,3),drop=FALSE] == 0) > 0]),
    interior_empty_probability=sum(outer.weights[empty.interior]),
    current_finite_B_availability=sum(
      outer.weights[which(full)]*current.availability),
    endpoint_allowed_finite_B_availability=sum(candidate.weights),
    threshold1_coverage=sum(outer.weights[eligible]*covered[,1])/
      eligible.probability,
    threshold2_coverage=sum(outer.weights[eligible]*covered[,2])/
      eligible.probability,
    joint_threshold_coverage=sum(
      outer.weights[eligible]*apply(covered,1,all))/eligible.probability,
    joint_coverage_given_available=sum(
      candidate.weights*apply(covered,1,all))/sum(candidate.weights))
}

scenarios <- list(
  rare_endpoint=c(-2.5, 1),
  sparse_interior=c(-2, -1.6),
  moderate=c(-1, 1)
)
results <- do.call(rbind, lapply(names(scenarios), function(name) {
  do.call(rbind, lapply(c(10L,20L,40L), function(n) {
    c(scenario=name, one_scenario(scenarios[[name]],n))
  }))
}))
print(results, digits=5)
