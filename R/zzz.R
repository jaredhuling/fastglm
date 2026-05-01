.onLoad <- function(libname, pkgname)
{
    if (requireNamespace("sandwich", quietly = TRUE))
    {
        sw <- asNamespace("sandwich")
        registerS3method("vcovHC", "fastglm",    vcovHC.fastglm,    envir = sw)
        registerS3method("vcovHC", "fastglmFit", vcovHC.fastglmFit, envir = sw)
        registerS3method("estfun", "fastglm",    estfun.fastglm,    envir = sw)
        registerS3method("estfun", "fastglmFit", estfun.fastglmFit, envir = sw)
        registerS3method("bread",  "fastglm",    bread.fastglm,     envir = sw)
        registerS3method("bread",  "fastglmFit", bread.fastglmFit,  envir = sw)
    }
}
