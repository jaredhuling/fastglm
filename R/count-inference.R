# Joint estimating equations for count and semicontinuous models. Keeping the
# equations together is necessary when observations in a cluster contribute to
# both components, or when negative-binomial size is estimated.
.count_inference_model <- function(object) {
    inherits(object,c("fastglm_nb","fastglm_truncated","fastglm_hurdle","fastglm_zi"))
}
.count_two_component <- function(object) inherits(object,c("fastglm_hurdle","fastglm_zi"))
.count_has_theta <- function(object) {
    if(inherits(object,"fastglm_nb")) return(!identical(object$theta.estimated,FALSE))
    identical(object$dist,"negbin") && !identical(object$theta.estimated,FALSE)
}
.count_theta_boundary <- function(object) {
    .count_has_theta(object) && (isTRUE(object$theta.boundary) ||
        object$theta<=1e-6*(1+1e-8) || object$theta>=1e8*(1-1e-8))
}
.count_adjusted <- function(object) {
    isTRUE(object$firth) || isTRUE(object$positive$firth) || isTRUE(object$binary$firth)
}
.count_parameters <- function(object,logtheta=TRUE) {
    out <- coef(object)
    if(.count_has_theta(object)) out <- c(out,theta=if(logtheta) log(object$theta) else object$theta)
    if(any(!is.finite(out))) stop("count inference requires finite, identifiable coefficients",call.=FALSE)
    out
}
.count_normalize_block <- function(z,two=FALSE,materialize=TRUE) {
    x <- z$X; y <- z$y; n <- nrow(x)
    w <- z$weights; if(is.null(w)) w <- rep(1,n)
    off <- z$offset
    if(is.null(off)) off <- list(count=rep(0,n),zero=rep(0,n))
    if(!is.list(off)) {
        off <- if(is.matrix(off)) list(count=off[,1],zero=off[,2]) else
            list(count=off,zero=rep(0,n))
    }
    if(is.null(off$count)) off$count <- rep(0,n)
    if(is.null(off$zero)) off$zero <- rep(0,n)
    zz <- if(two) { if(is.null(z$Z)) x else z$Z } else NULL
    list(X=if(materialize) as.matrix(x) else x,
         Z=if(two) if(materialize) as.matrix(zz) else zz,y=y,weights=w,offset=off)
}
.count_blocks <- function(object,fun,block.size=8192L) {
    .check_iterations(1e-8,block.size)
    two <- .count_two_component(object)
    if(!is.null(object$source) && is.function(object$fetch)) {
        row <- 0L
        for(k in seq_len(object$source$n_chunks)) {
            z <- object$fetch(k); z <- .count_normalize_block(z,two)
            i <- row+seq_len(nrow(z$X));row <- row+length(i)
            fun(z,i,k)
        }
    } else {
        if(is.null(object$y)) stop("inference requires a retained response",call.=FALSE)
        d <- if(two) .count_design(object) else .prediction_design(object)
        x <- if(two) d$x$count else d$x
        .design_blocks(x,function(xx,i) {
            z <- list(X=xx,Z=if(two) d$x$zero[i,,drop=FALSE],y=object$y[i],
                weights=if(two) object$weights[i] else object$prior.weights[i],
                offset=if(two) lapply(d$offset,`[`,i) else d$offset[i])
            fun(.count_normalize_block(z,two),i,NULL)
        },block.size)
    }
    invisible(NULL)
}
.count_score <- function(object,z,parameters) {
    two <- .count_two_component(object); nb <- inherits(object,"fastglm_nb") || identical(object$dist,"negbin")
    continuous <- inherits(object,"fastglm_twopart")
    theta <- if(.count_has_theta(object)) exp(utils::tail(parameters,1L)) else object$theta
    pc <- if(two) length(object$coefficients$count) else length(coef(object))
    pz <- if(two) length(object$coefficients$zero) else 0L
    eta <- drop(z$X%*%parameters[seq_len(pc)])+z$offset$count
    link <- if(inherits(object,"fastglm_nb")) object$family else make.link("log")
    mu <- link$linkinv(eta); dm <- if(is.function(link$mu.eta)) link$mu.eta(eta) else mu
    y <- z$y; positive <- y>0
    if(continuous) {
        sc <- ifelse(positive,y/mu-1,0)
    } else if(nb) {
        sc <- dm*(y-mu)/(mu+mu^2/theta)
    } else sc <- y-mu
    st <- if(.count_has_theta(object)) theta*.count_backend_theta_score(y,mu,theta) else NULL
    if(inherits(object,"fastglm_truncated") || (inherits(object,"fastglm_hurdle") && !continuous)) {
        mm <- .count_moments(mu,if(nb) "negbin" else "poisson",theta)
        sc <- if(nb) theta/(theta+mu)*(y-mm$mean) else y-mm$mean
        if(!is.null(st)) st <- theta*.count_backend_theta_score(y,mu,theta,truncated=TRUE)
        if(two) { sc[!positive] <- 0;if(!is.null(st)) st[!positive] <- 0 }
    }
    if(two) {
        eta0 <- drop(z$Z%*%parameters[pc+seq_len(pz)])+z$offset$zero
        lk <- make.link(object$link); pi <- lk$linkinv(eta0); dpi <- lk$mu.eta(eta0)
        target <- as.numeric(positive)
        if(inherits(object,"fastglm_zi")) {
            lp0 <- if(nb) -theta*log1p(mu/theta) else -mu
            tau <- numeric(length(y));tau[!positive] <- pi[!positive]/
                (pi[!positive]+(1-pi[!positive])*exp(lp0[!positive]))
            sc <- sc*(1-tau);if(!is.null(st)) st <- st*(1-tau)
            target <- tau
        }
        sz <- (target-pi)*dpi/(pi*(1-pi))
        out <- cbind(z$X*sc,z$Z*sz)
    } else out <- z$X*sc
    if(!is.null(st)) out <- cbind(out,st)
    out <- out*z$weights;out[z$weights==0,] <- 0
    if(any(!is.finite(out))) stop("non-finite estimating scores; inspect boundary probabilities and means",call.=FALSE)
    colnames(out) <- names(parameters)
    out
}
# Derivatives of linkinv and of mu.eta are distinct at the numerical limits in
# stats::make.link(). Retain those limits when differentiating the scores above.
.count_link_slopes <- function(link,eta,mu,dm) {
    eps <- .Machine$double.eps
    switch(link,
        identity=list(mean=rep(1,length(eta)),score=rep(0,length(eta))),
        sqrt=list(mean=dm,score=rep(2,length(eta))),
        inverse=list(mean=dm,score=2/eta^3),
        `1/mu^2`=list(mean=dm,score=3/(4*eta^(5/2))),
        log={ d<-ifelse(exp(eta)>eps,dm,0);list(mean=d,score=d) },
        logit={
            interior<-abs(eta)<=30
            list(mean=ifelse(interior,dm,0),score=ifelse(interior,dm*(1-2*mu),0))
        },
        probit={
            d<-dnorm(eta)
            list(mean=ifelse(abs(eta)< -qnorm(eps),d,0),score=ifelse(d>eps,-eta*d,0))
        },
        cauchit={
            d<-dcauchy(eta)
            list(mean=ifelse(abs(eta)< -qcauchy(eps),d,0),
                score=ifelse(d>eps,-2*(eta/(1+eta^2))*d,0))
        },
        cloglog={
            ex<-exp(pmin(eta,700));d<-ex*exp(-ex);prob<--expm1(-ex)
            list(mean=ifelse(prob>eps & prob<1-eps,d,0),score=ifelse(d>eps,d*(1-ex),0))
        },
        stop("unsupported link for joint count information",call.=FALSE))
}

.count_information_block <- function(object,z,parameters) {
    p<-length(parameters);info<-matrix(0,p,p)
    keep<-z$weights>0
    if(!any(keep)) return(info)
    two<-.count_two_component(object);nb<-inherits(object,"fastglm_nb") || identical(object$dist,"negbin")
    continuous<-inherits(object,"fastglm_twopart");has.theta<-.count_has_theta(object)
    truncated<-inherits(object,"fastglm_truncated") || (inherits(object,"fastglm_hurdle") && !continuous)
    theta<-if(has.theta) exp(utils::tail(parameters,1L)) else object$theta
    pc<-if(two) length(object$coefficients$count) else length(coef(object))
    ic<-seq_len(pc);iz<-if(two) pc+seq_along(object$coefficients$zero)
    x<-z$X[keep,,drop=FALSE];y<-z$y[keep];w<-z$weights[keep];positive<-y>0
    eta<-drop(x%*%parameters[ic])+z$offset$count[keep]
    link<-if(inherits(object,"fastglm_nb")) object$family else make.link("log")
    mu<-link$linkinv(eta);dm<-link$mu.eta(eta)
    slopes<-.count_link_slopes(if(inherits(object,"fastglm_nb")) link$link else link$name,eta,mu,dm)
    dmu<-slopes$mean;ddm<-slopes$score
    ct<-tc<-tt<-st<-NULL
    if(continuous) {
        sc<-ifelse(positive,y/mu-1,0)
        cc<-ifelse(positive,y*dmu/mu^2,0)
    } else if(nb) {
        v<-mu+mu^2/theta;vp<-1+2*mu/theta
        sc<-dm*(y-mu)/v
        cc<-dm*dmu/v+(y-mu)*(dm*dmu*vp/v^2-ddm/v)
        if(has.theta) {
            ct<--dm*theta*(y-mu)/(theta+mu)^2
            tc<--dmu*theta*(y-mu)/(theta+mu)^2
            st<-theta*.count_backend_theta_score(y,mu,theta)
            ds<-trigamma(y+theta)-trigamma(theta)+mu/(theta*(theta+mu))-(mu-y)/(theta+mu)^2
            tt<--st-theta^2*ds
            # Match the stable large-theta score expansion rather than subtract
            # nearly equal digamma/trigamma values in the Poisson limit.
            large<-theta>1e4*pmax(1,y,mu)
            if(any(large)) {
                yy<-y[large];mm<-mu[large]
                c2<-(yy-(yy-mm)^2)/2
                c3<-yy*(yy-1)*(2*yy-1)/6+2*mm^3/3-yy*mm^2
                c4<--yy^2*(yy-1)^2/4-3*mm^4/4+yy*mm^3
                tt[large]<-(c2+(2*c3+3*c4/theta)/theta)/theta
            }
            # The elementary probabilities for zero and one avoid cancellation
            # at small means without requiring theta to be large.
            u<-mu/theta;d<--log1p(u)+u/(1+u);small<-u<1e-4
            d[small]<-u[small]^2*(-.5+u[small]*(2/3+u[small]*(-3/4+u[small]*4/5)))
            a.t<-theta*d;a.tt<-a.t+theta*(u/(1+u))^2
            zero<-y==0;one<-y==1
            st[zero]<-a.t[zero];tt[zero]<--a.tt[zero]
            st[one]<-a.t[one]+u[one]/(1+u[one])
            tt[one]<--a.tt[one]+u[one]/(1+u[one])^2
        }
    } else { sc<-y-mu;cc<-dmu }

    if(truncated || inherits(object,"fastglm_zi")) {
        if(nb) {
            u<-mu/theta;c<-1/(1+u);lp0<--theta*log1p(u)
            a.c<--c*mu
            if(!has.theta) {
                d<--log1p(u)+u/(1+u);small<-u<1e-4
                d[small]<-u[small]^2*(-.5+u[small]*(2/3+u[small]*(-3/4+u[small]*4/5)))
                a.t<-theta*d
            }
            a.ct<--theta*(u/(1+u))^2
        } else { c<-1;lp0<--mu;a.c<--mu;a.t<-a.ct<-a.tt<-rep(0,length(mu)) }
        if(truncated) {
            q<--expm1(lp0);r<-exp(lp0)/q;m<-mu/q
            # 1-A/expm1(A) is the Poisson truncated-mean derivative. Its
            # series also stabilizes NB curvature when positive counts are rare.
            aa<--lp0;one.minus<-1-aa*r;small<-aa<1e-3
            one.minus[small]<-aa[small]*(.5+aa[small]*(-1/12+aa[small]^2*(1/720-aa[small]^2/30240)))
            m.deriv<-m*(one.minus-r*a.t)
            cc<-(c*(1-c)*(y-m)+c*m.deriv)*(dmu/mu)
            if(has.theta) {
                ct<-ct-r*a.ct-r*(1+r)*a.c*a.t
                tc<-ct*(dmu/mu)
                tt<-tt-r*a.tt-r*(1+r)*a.t^2
            }
            if(two) { cc[!positive]<-0;if(has.theta) {ct[!positive]<-tc[!positive]<-tt[!positive]<-0} }
        }
    }

    if(two) {
        zz<-z$Z[keep,,drop=FALSE]
        eta0<-drop(zz%*%parameters[iz])+z$offset$zero[keep]
        lk<-make.link(object$link);pi<-lk$linkinv(eta0);dpi<-lk$mu.eta(eta0)
        slopes0<-.count_link_slopes(object$link,eta0,pi,dpi)
        den<-pi*(1-pi);k<-dpi/den;actual.k<-slopes0$mean/den
        dk<-slopes0$score/den-k*(1-2*pi)*actual.k
        target<-as.numeric(positive)
        if(inherits(object,"fastglm_zi")) {
            tau<-numeric(length(y));zero<-!positive
            tau[zero]<-pi[zero]/(pi[zero]+(1-pi[zero])*exp(lp0[zero]))
            target<-tau;variation<-tau*(1-tau)
            actual.sc<-a.c*(dmu/mu)
            cc<-(1-tau)*cc-variation*sc*actual.sc
            cz<-variation*(sc*actual.k+actual.sc*k)/2
            if(has.theta) {
                ct<-(1-tau)*ct-variation*sc*st
                tc<-(1-tau)*tc-variation*actual.sc*st
                tt<-(1-tau)*tt-variation*st^2
                zt<-variation*st*(k+actual.k)/2
            }
        }
        zz.info<-slopes0$mean*k-(target-pi)*dk
        if(inherits(object,"fastglm_zi")) {
            zz.info<-zz.info-variation*k*actual.k
            info[ic,iz]<-crossprod(x,zz*(w*cz));info[iz,ic]<-t(info[ic,iz,drop=FALSE])
            if(has.theta) info[iz,p]<-info[p,iz]<-drop(crossprod(zz,w*zt))
        }
        info[iz,iz]<-crossprod(zz,zz*(w*zz.info))
    }
    info[ic,ic]<-crossprod(x,x*(w*cc))
    if(has.theta) {
        info[ic,p]<-info[p,ic]<-drop(crossprod(x,w*(ct+tc)/2))
        info[p,p]<-sum(w*tt)
    }
    # Use the existing information guard for nonfinite or indefinite results;
    # fitting callers can then retain estimates with unavailable uncertainty.
    (info+t(info))/2
}
.count_inverse_information <- function(info) {
    force(info)
    r <- tryCatch(chol(info),error=function(e) NULL)
    if(is.null(r) || any(!is.finite(info)) || rcond(info)<1e-12)
        stop("joint information is singular or not positive definite; inference is unavailable",call.=FALSE)
    chol2inv(r)
}
.count_information <- function(object,block.size=8192L) {
    parameters <- .count_parameters(object);p<-length(parameters);info<-matrix(0,p,p)
    .count_blocks(object,function(z,i,k) info <<- info+.count_information_block(object,z,parameters),block.size)
    dimnames(info)<-rep(list(names(parameters)),2L)
    list(information=info,inverse=.count_inverse_information(info),parameters=parameters)
}
.count_natural_covariance <- function(v,object,full=TRUE) {
    nm <- names(.count_parameters(object,FALSE))
    if(.count_has_theta(object)) {
        scale <- c(rep(1,length(nm)-1L),object$theta)
        v <- v*outer(scale,scale)
    }
    dimnames(v) <- list(nm,nm)
    if(!full) v <- v[seq_along(coef(object)),seq_along(coef(object)),drop=FALSE]
    v
}
.count_default_vcov <- function(object) {
    if(inherits(object,"fastglm_nb")) {
        if(!is.null(object$vcov.full)) return(object$vcov.full)
        return(.count_natural_covariance(.count_information(object)$inverse,object))
    }
    if(!is.null(object$vcov.full)) object$vcov.full else vcov(object)
}
.count_robust_vcov <- function(object,type,cluster,cadjust,block.size,full) {
    if(.count_adjusted(object)) stop("sandwich inference for adjusted scores is not implemented; use a refit bootstrap",call.=FALSE)
    if(!type %in% c("HC0","HC1")) stop("count models support HC0 or HC1; leverage corrections HC2/HC3 are not defined here",call.=FALSE)
    if(!isTRUE(object$converged)) stop("robust inference requires a converged model",call.=FALSE)
    if(.count_theta_boundary(object))
        stop("joint sandwich inference is unavailable at a theta boundary",call.=FALSE)
    parameters <- .count_parameters(object);p<-length(parameters)
    info<-meat<-matrix(0,p,p);n<-0L;clusters<-new.env(hash=TRUE,parent=emptyenv())
    streaming <- !is.null(object$source) && is.function(object$fetch)
    if(streaming && !is.null(cluster) && !is.function(cluster)) stop("streaming clusters require a function of chunk index",call.=FALSE)
    .count_blocks(object,function(z,i,k) {
        score <- .count_score(object,z,parameters)
        info <<- info+.count_information_block(object,z,parameters)
        keep <- z$weights>0;n <<- n+sum(keep)
        if(is.null(cluster)) meat <<- meat+crossprod(score)
        else {
            groups <- if(streaming) cluster(k) else cluster[i]
            if(length(groups)!=nrow(score) || anyNA(groups)) stop("invalid cluster identifiers",call.=FALSE)
            if(any(keep)) {
                ss <- rowsum(score[keep,,drop=FALSE],as.character(groups[keep]),reorder=FALSE)
                for(j in seq_len(nrow(ss))) {
                    key<-rownames(ss)[j];value<-ss[j,]
                    if(exists(key,clusters,inherits=FALSE)) value<-value+get(key,clusters)
                    assign(key,value,clusters)
                }
            }
        }
    },block.size)
    if(!is.null(cluster)) {
        keys<-ls(clusters,all.names=TRUE);g<-length(keys)
        if(g<2L) stop("cluster covariance requires at least two positive-weight clusters",call.=FALSE)
        for(key in keys) meat<-meat+tcrossprod(get(key,clusters))
        if(cadjust) meat<-meat*g/(g-1)
    }
    if(type=="HC1") {
        if(n<=p) stop("HC1 requires more positive-weight observations than estimated parameters",call.=FALSE)
        meat<-meat*(if(is.null(cluster)) n else n-1)/(n-p)
    }
    bread<-.count_inverse_information(info)
    .count_natural_covariance(bread%*%meat%*%bread,object,full)
}

.count_prediction_parameters <- function(object,z,parameters,type="response") {
    if(.count_two_component(object)) return(.count_predict(object,
        list(x=list(count=z$X,zero=z$Z),offset=z$offset),parameters,type,NULL))
    eta<-drop(z$X%*%parameters[seq_along(coef(object))])+z$offset$count
    if(inherits(object,"fastglm_nb")) return(object$family$linkinv(eta))
    mu<-exp(eta)
    if(type=="count") return(mu)
    theta<-if(.count_has_theta(object)) utils::tail(parameters,1L) else object$theta
    .count_moments(mu,object$dist,theta)$mean
}
.count_prediction_gradient <- function(object,z,parameters,type="response") {
    out<-matrix(0,nrow(z$X),length(parameters))
    for(j in seq_along(parameters)) {
        h<-.Machine$double.eps^(1/3)*(1+abs(parameters[j]))
        if(.count_has_theta(object) && j==length(parameters)) h<-min(h,parameters[j]/4)
        a<-b<-parameters;a[j]<-a[j]+h;b[j]<-b[j]-h
        out[,j]<-(.count_prediction_parameters(object,z,a,type)-.count_prediction_parameters(object,z,b,type))/(2*h)
    }
    out
}
.count_average <- function(object,newdata,contrast,weights,vcov.,level,type,block.size=8192L) {
    type<-match.arg(type,c("response","count","zero"))
    if(!.count_two_component(object) && type=="zero") stop("zero probabilities are not an averaging target for this model",call.=FALSE)
    pars<-.count_parameters(object,FALSE)
    if(is.null(vcov.)) vcov.<-.count_default_vcov(object)
    if(!identical(dim(vcov.),c(length(pars),length(pars)))) stop("count averages require the full joint covariance, including theta when estimated",call.=FALSE)
    total<-estimate<-0;gradient<-numeric(length(pars))
    add<-function(z,w=NULL) {
        n<-nrow(z$X);if(is.null(w)) w<-rep(1,n)
        if(length(w)!=n || any(!is.finite(w)|w<0)) stop("invalid averaging weights",call.=FALSE)
        mu<-.count_prediction_parameters(object,z,pars,type)
        g<-.count_prediction_gradient(object,z,pars,type)
        if(!is.null(contrast)) {
            zz<-z
            if(.count_two_component(object)) {
                xx<-contrast(list(count=z$X,zero=z$Z))
                if(!is.list(xx) || !identical(dim(xx$count),dim(z$X)) || !identical(dim(xx$zero),dim(z$Z)))
                    stop("contrast must preserve both component design dimensions",call.=FALSE)
                zz$X<-xx$count;zz$Z<-xx$zero
            } else {
                xx<-contrast(z$X)
                if(!identical(dim(xx),dim(z$X))) stop("contrast must preserve design dimensions",call.=FALSE)
                zz$X<-xx
            }
            mu<-.count_prediction_parameters(object,zz,pars,type)-mu
            g<-.count_prediction_gradient(object,zz,pars,type)-g
        }
        total<<-total+sum(w);estimate<<-estimate+sum(w*mu);gradient<<-gradient+colSums(w*g)
    }
    streaming<-!is.null(object$source) && is.function(object$fetch) && is.null(newdata)
    if(is.null(newdata)) {
        if(streaming && !is.null(weights) && !is.function(weights)) stop("streaming averaging weights require a function of chunk index",call.=FALSE)
        .count_blocks(object,function(z,i,k) add(z,if(!is.null(weights)) if(streaming) weights(k) else weights[i]),block.size)
    } else {
        two<-.count_two_component(object)
        d<-if(two) .count_design(object,newdata) else .prediction_design(object,newdata)
        x<-if(two) d$x$count else d$x
        .design_blocks(x,function(xx,i) {
            z<-.count_normalize_block(list(X=xx,Z=if(two) d$x$zero[i,,drop=FALSE],
                offset=if(two) lapply(d$offset,`[`,i) else d$offset[i]),two)
            add(z,if(!is.null(weights)) weights[i])
        },block.size)
    }
    if(total<=0) stop("averaging weights must have positive total",call.=FALSE)
    estimate<-estimate/total;gradient<-gradient/total
    active<-which(gradient!=0)
    variance<-if(length(active)) drop(crossprod(gradient[active],
        vcov.[active,active,drop=FALSE]%*%gradient[active])) else 0
    if(!is.finite(variance) || variance<0) stop("invalid covariance for the requested average",call.=FALSE)
    se<-sqrt(variance);q<-qnorm((1+level)/2)
    data.frame(estimate=estimate,std.error=se,conf.low=estimate-q*se,conf.high=estimate+q*se)
}

#' @export
vcov.fastglm_nb <- function(object,dispersion=1,full=FALSE,type=NULL,cluster=NULL,...) {
    if(!is.null(type) || !is.null(cluster))
        return(fastglm_vcov(object,if(is.null(type)) "HC0" else type,cluster=cluster,full=full,...))
    if(full) return(.count_default_vcov(object))
    v<-dispersion*object$cov.unscaled
    dimnames(v)<-rep(list(names(coef(object))),2L)
    v
}
