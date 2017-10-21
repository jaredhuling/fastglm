


call_fit_glm <- function(x, y, 
                         weights = rep(1, NROW(y)), 
                         offset = rep(0, NROW(y)), 
                         family = gaussian(),
                         type = 1L,
                         tol = 1e-7,
                         maxit = 100L)
{
    fit_glm(x, y, weights, offset, family$variance, family$mu.eta, family$linkinv, family$dev.resids, 
            type, tol, maxit)
}