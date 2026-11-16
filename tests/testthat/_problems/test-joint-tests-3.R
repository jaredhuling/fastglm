# Extracted from test-joint-tests.R:3

# test -------------------------------------------------------------------------
fit <- fastglm_ordinal(Sat~Infl+Type+Cont,MASS::housing,weights=Freq,firth=TRUE)
count <- 0L
original <- .ordinal_fit_core
