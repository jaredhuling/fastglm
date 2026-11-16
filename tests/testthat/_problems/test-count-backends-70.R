# Extracted from test-count-backends.R:70

# test -------------------------------------------------------------------------
set.seed(746)
n<-400L
d<-data.frame(x=rnorm(n),e=exp(rnorm(n,0,.1)),f=exp(rnorm(n,0,.1)))
d$y<-rpois(n,exp(.4+.3*d$x))*rbinom(n,1,.6)
d$w<-rep(c(0,1,2,1),length.out=n)
d$y[1:100]<-0
d$w[101:200]<-0
callback<-function(k) d[seq.int((k-1L)*100L+1L,k*100L),]
source<-fastglm_source(callback,4,formula=y~x+offset(log(e))|x+offset(log(f)),prototype=d,weights="w")
stream<-fastglm_hurdle(source,tol=1e-10)
dense<-fastglm_hurdle(y~x+offset(log(e))|x+offset(log(f)),d,weights=w,tol=1e-10)
expect_equal(coef(stream),coef(dense),tolerance=1e-7)
expect_equal(vcov(stream),vcov(dense),tolerance=1e-7)
