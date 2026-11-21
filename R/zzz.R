.onLoad <- function(libname, pkgname)
{
    ## Thread setting for the compiled code: the fastglm.threads option when
    ## set, otherwise the default. No warning at load.
    ## The process that loaded the package; a process forked from it later
    ## (an mclapply worker) resolves the default to 1 thread.
    .fastglm_state$pid <- Sys.getpid()
    fastglm_set_threads_cpp(tryCatch(.fastglm_resolve_threads(NULL),
                                     error = function(e) 1L))

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

    ## parsnip engines, registered now or when the defining package loads.
    .fastglm_on_package_load("parsnip", .fastglm_register_parsnip)
    .fastglm_on_package_load("poissonreg", .fastglm_register_poissonreg)
    .fastglm_on_package_load("ordered", .fastglm_register_ordered)
}
