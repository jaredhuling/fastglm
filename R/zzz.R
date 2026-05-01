.onLoad <- function(libname, pkgname)
{
    if (requireNamespace("sandwich", quietly = TRUE))
    {
        registerS3method("vcovHC", "fastglm",    vcovHC.fastglm,
                         envir = asNamespace("sandwich"))
        registerS3method("vcovHC", "fastglmFit", vcovHC.fastglmFit,
                         envir = asNamespace("sandwich"))
        registerS3method("vcovCL", "fastglm",    vcovCL.fastglm,
                         envir = asNamespace("sandwich"))
        registerS3method("vcovCL", "fastglmFit", vcovCL.fastglmFit,
                         envir = asNamespace("sandwich"))
    }
}
