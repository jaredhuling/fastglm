# Force the R-callback fallback by wrapping a family so the (family, link)
# string no longer matches a native code.
disguise_family <- function(fam) {
    fam$family <- paste0(fam$family, "-disguised")
    fam
}

test_that("native and R-callback paths agree on tier-1 families", {
    cases <- list(
        list(name = "gaussian:identity", fam = gaussian(),                    resp = "gaussian"),
        list(name = "binomial:logit",    fam = binomial("logit"),             resp = "binomial"),
        list(name = "binomial:probit",   fam = binomial("probit"),            resp = "binomial"),
        list(name = "binomial:cloglog",  fam = binomial("cloglog"),           resp = "binomial"),
        list(name = "poisson:log",       fam = poisson("log"),                resp = "poisson"),
        list(name = "Gamma:log",         fam = Gamma("log"),                  resp = "gamma_log"),
        list(name = "Gamma:inverse",     fam = Gamma("inverse"),              resp = "gamma_log")
    )

    for (cc in cases) {
        d <- make_glm_data(n = 400, p = 4, response = cc$resp)
        f_native <- fastglm(d$X, d$y, family = cc$fam,                method = 2)
        f_callback <- fastglm(d$X, d$y, family = disguise_family(cc$fam), method = 2)
        expect_equal(unname(coef(f_native)), unname(coef(f_callback)),
                     tolerance = 1e-7,
                     info = paste0(cc$name, " coefficients"))
        expect_equal(f_native$deviance, f_callback$deviance, tolerance = 1e-8,
                     info = paste0(cc$name, " deviance"))
    }
})
