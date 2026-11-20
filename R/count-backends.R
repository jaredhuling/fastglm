# Read repeatable row blocks for count and two-part regression.
.count_backend_offsets <- function(offset,n) {
    if(is.null(offset)) offset <- list(count=rep(0,n),zero=rep(0,n))
    else if(is.matrix(offset)) {
        if(nrow(offset)!=n || ncol(offset)!=2L) stop("component offsets must have two columns",call.=FALSE)
        offset <- list(count=offset[,1L],zero=offset[,2L])
    } else if(!is.list(offset)) offset <- list(count=offset,zero=rep(0,n))
    for(k in c("count","zero")) {
        if(is.null(offset[[k]])) offset[[k]] <- rep(0,n)
        if(!is.numeric(offset[[k]]) || length(offset[[k]])!=n || any(!is.finite(offset[[k]])))
            stop("each component offset must contain one finite value per row",call.=FALSE)
        offset[[k]] <- as.numeric(offset[[k]])
    }
    offset[c("count","zero")]
}

.count_backend_reader <- function(x,response=NULL,z=NULL,weights=NULL,offset=NULL,
                                   block.size=8192L,counts=TRUE) {
    .check_iterations(1e-8,block.size)
    streaming <- inherits(x,"fastglm_source")
    original <- NULL
    if(streaming) {
        if(!is.null(response) || !is.null(z) || !is.null(weights) || !is.null(offset))
            stop("a data source must supply its own responses, designs, weights, and offsets",call.=FALSE)
        source <- x
    } else {
        if(!(is.matrix(x) || inherits(x,c("dgCMatrix","big.matrix"))))
            stop("design must be a numeric matrix, dgCMatrix, big.matrix, or fastglm_source",call.=FALSE)
        if(is.null(response)) stop("supply 'response' with a matrix design",call.=FALSE)
        if(is.matrix(x) && is.integer(x)) storage.mode(x)<-"double"
        if(is.matrix(z) && is.integer(z)) storage.mode(z)<-"double"
        if(is.null(z)) z <- x
        n <- nrow(x)
        if(is.null(weights)) weights <- rep(1,n)
        offset <- .count_backend_offsets(offset,n)
        original <- list(x=list(count=x,zero=z),y=response,weights=weights,offset=offset)
        if(n<1L) stop("a design must contain at least one row",call.=FALSE)
        # A dense design is already in memory, so one block avoids copying its rows.
        # Sparse and memory-mapped designs are read in row blocks; a block of a
        # big.matrix is an ordinary dense matrix.
        source <- if(is.matrix(x) && is.matrix(z))
            fastglm_source(function(k) list(X=x,Z=z,y=response,weights=weights,offset=offset),1L) else
            fastglm_source(function(k) {
                i <- seq.int((k-1L)*block.size+1L,min(k*block.size,n))
                list(X=x[i,,drop=FALSE],Z=z[i,,drop=FALSE],y=response[i],weights=weights[i],
                     offset=lapply(offset,`[`,i))
            },ceiling(n/block.size))
        if(nrow(z)!=n || length(response)!=n || length(weights)!=n)
            stop("designs, response, and weights must have the same number of rows",call.=FALSE)
    }
    state <- new.env(parent=emptyenv())
    state$reads <- state$passes <- 0L; state$rows <- integer(source$n_chunks)
    state$positive <- integer(source$n_chunks);state$nobs <- state$npositive <- 0L
    state$columns <- state$p <- NULL
    fetch <- function(k) {
        state$reads <- state$reads+1L
        if(k==1L) state$passes <- state$passes+1L
        # Data held by the reader cannot change, so one validation is enough.
        if(!is.null(state$validated)) return(state$validated)
        a <- source$callback(k)
        if(!is.list(a) || is.null(a$X) || is.null(a$y)) stop("each data block must supply X and y",call.=FALSE)
        if(is.null(a$Z)) a$Z <- a$X
        n <- NROW(a$y)
        if(!is.numeric(a$y) || !is.null(dim(a$y)) || n<1L || any(!is.finite(a$y)) || any(a$y<0) ||
           (counts && any(a$y!=floor(a$y)))) stop("invalid nonnegative response in a data block",call.=FALSE)
        for(nm in c("X","Z")) {
            xx <- a[[nm]]
            if(!(is.matrix(xx) && is.numeric(xx)) && !inherits(xx,"dgCMatrix"))
                stop("each block design must be a numeric matrix or dgCMatrix",call.=FALSE)
            if(is.matrix(xx) && is.integer(xx)) {
                storage.mode(xx)<-"double"
                a[[nm]]<-xx
            }
            finite <- if(inherits(xx,"dgCMatrix")) all(is.finite(xx@x)) else all_finite_dense(xx)
            if(nrow(xx)!=n || ncol(xx)<1L || !finite)
                stop("invalid rows, columns, or nonfinite values in a block design",call.=FALSE)
        }
        if(is.null(a$weights)) a$weights <- rep(1,n)
        if(!is.numeric(a$weights) || length(a$weights)!=n || any(!is.finite(a$weights)) || any(a$weights<0))
            stop("block weights must be finite and nonnegative",call.=FALSE)
        a$offset <- .count_backend_offsets(a$offset,n)
        if(!is.null(a$offset.zero)) {
            zero <- .count_backend_offsets(list(zero=a$offset.zero),n)
            a$offset$zero <- zero$zero
        }
        cols <- list(count=colnames(a$X),zero=colnames(a$Z));p <- c(ncol(a$X),ncol(a$Z))
        if(is.null(state$p)) {state$p <- p;state$columns <- cols}
        else if(!identical(p,state$p) || !identical(cols,state$columns))
            stop("design columns changed between blocks",call.=FALSE)
        if(state$rows[k]!=0L && state$rows[k]!=n) stop("block size changed between passes",call.=FALSE)
        state$rows[k] <- n
        block <- list(X=a$X,Z=a$Z,y=as.numeric(a$y),weights=as.numeric(a$weights),offset=a$offset)
        if(!streaming && source$n_chunks==1L) state$validated <- block
        block
    }
    # The tally has its own frame so the retained reader does not keep a block.
    tally <- function(a,k) {
        positive <- a$weights>0
        state$nobs <- state$nobs+sum(positive)
        state$positive[k] <- sum(positive & a$y>0)
    }
    for(k in seq_len(source$n_chunks)) tally(fetch(k),k)
    state$npositive <- sum(state$positive)
    if(state$nobs<=state$p[1L]) stop("more positive-weight observations than count coefficients are required",call.=FALSE)
    cn <- state$columns
    for(k in 1:2) if(is.null(cn[[k]])) cn[[k]] <- paste0("X",seq_len(state$p[k]))
    list(source=source,fetch=fetch,state=state,streaming=streaming,original=original,columns=cn)
}

.count_backend_component <- function(reader,component=c("count","zero"),positive=FALSE,binary=FALSE) {
    component <- match.arg(component)
    blocks <- if(positive) which(reader$state$positive>0) else seq_len(reader$source$n_chunks)
    fetch <- function(k) {
        first <- if(k==1L) 1L else blocks[k-1L]+1L
        for(j in seq.int(first,blocks[k])) a <- reader$fetch(j)
        if(k==length(blocks) && blocks[k]<reader$source$n_chunks)
            for(j in seq.int(blocks[k]+1L,reader$source$n_chunks)) reader$fetch(j)
        keep <- if(positive) a$y>0 & a$weights>0 else rep(TRUE,length(a$y))
        xx <- if(component=="count") a$X else a$Z
        # Only this row block is converted when the native streaming GLM needs a dense matrix.
        list(X=as.matrix(xx[keep,,drop=FALSE]),y=if(binary) as.numeric(a$y[keep]>0) else a$y[keep],
             weights=a$weights[keep],offset=a$offset[[component]][keep])
    }
    out <- fastglm_source(fetch,length(blocks))
    schema <- reader$source$schema
    if(!is.null(schema)) {
        external.offset <- schema$offset
        if(is.list(schema$terms) && !inherits(schema$terms,"terms"))
            schema <- lapply(schema,function(a) if(is.list(a) && component %in% names(a)) a[[component]] else a)
        schema$offset <- if(component=="count") external.offset[1L] else
            if(length(external.offset)>1L) external.offset[2L] else NULL
        out$schema <- schema
    }
    out
}

.count_backend_glm <- function(reader,component,family,positive=FALSE,binary=FALSE,
                               start=NULL,method=2L,tol=1e-8,maxit=100L,firth=FALSE) {
    if(reader$streaming) {
        source <- .count_backend_component(reader,component,positive,binary)
        fastglm_streaming(source,family=family,start=start,method=method,tol=tol,maxit=maxit,firth=firth)
    } else {
        d <- reader$original
        keep <- if(positive) d$y>0 & d$weights>0 else rep(TRUE,length(d$y))
        fastglm(d$x[[component]][keep,,drop=FALSE],if(binary) as.numeric(d$y[keep]>0) else d$y[keep],
                family=family,weights=d$weights[keep],offset=d$offset[[component]][keep],
                start=start,method=method,tol=tol,maxit=maxit,firth=firth)
    }
}

.count_backend_theta_score <- function(y,mu,theta,truncated=FALSE) {
    a <- digamma(y+theta)-digamma(theta)-log1p(mu/theta)+(mu-y)/(theta+mu)
    large <- theta>1e4*pmax(1,y,mu)
    if(any(large)) {
        yy <- y[large];mm <- mu[large]
        c2 <- (yy-(yy-mm)^2)/2
        c3 <- yy*(yy-1)*(2*yy-1)/6+2*mm^3/3-yy*mm^2
        c4 <- -yy^2*(yy-1)^2/4-3*mm^4/4+yy*mm^3
        a[large] <- (c2+(c3+c4/theta)/theta)/theta^2
    }
    if(truncated) {
        lp0 <- -theta*log1p(mu/theta)
        r <- mu/theta;d <- -log1p(r)+r/(1+r)
        small <- r<1e-4
        d[small] <- r[small]^2*(-.5+r[small]*(2/3+r[small]*(-3/4+r[small]*4/5)))
        a <- a+exp(lp0)/(-expm1(lp0))*d
    }
    a
}

.count_backend_count_summary <- function(reader,beta,theta=NULL,truncated=FALSE,
                                         link="log",system=FALSE,theta.derivatives=FALSE) {
    p <- length(beta);ll <- 0;score <- numeric(p);information <- matrix(0,p,p)
    st <- it <- 0
    lnk <- make.link(link)
    for(k in seq_len(reader$source$n_chunks)) {
        a <- reader$fetch(k)
        # Dense blocks use the compiled kernel. Sparse blocks use the same formulas
        # below, which also serve as its reference in the tests.
        if(is.matrix(a$X)) {
            v <- count_block_summary(a$X,a$y,a$weights,a$offset$count,as.numeric(beta),
                if(is.null(theta)) NA_real_ else theta,truncated,match(link,c("log","sqrt","identity"))-1L,
                system,theta.derivatives)
            if(!v$valid) return(list(loglik=-Inf))
            ll <- ll+v$loglik;score <- score+drop(v$score);information <- information+v$information
            st <- st+v$theta.score;it <- it+v$theta.information
            next
        }
        keep <- a$weights>0 & (!truncated | a$y>0)
        if(!any(keep)) next
        xx <- a$X[keep,,drop=FALSE];y <- a$y[keep];w <- a$weights[keep]
        eta <- as.numeric(xx%*%beta)+a$offset$count[keep];mu <- lnk$linkinv(eta)
        if(any(!is.finite(mu) | mu<=0)) return(list(loglik=-Inf))
        nb <- !is.null(theta)
        density <- if(nb) dnbinom(y,mu=mu,size=theta,log=TRUE) else dpois(y,mu,log=TRUE)
        if(truncated) {
            lp0 <- if(nb) -theta*log1p(mu/theta) else -mu
            q <- -expm1(lp0);density <- density-log(q)
        }
        ll <- ll+sum(w*density)
        if(system) {
            if(truncated) {
                mm <- .count_moments(mu,if(nb) "negbin" else "poisson",theta)
                multiplier <- if(nb) theta/(theta+mu) else rep(1,length(mu))
                u <- multiplier*(y-mm$mean)
                iw <- multiplier^2*mm$variance
            } else {
                dm <- lnk$mu.eta(eta);variance <- if(nb) mu+mu^2/theta else mu
                u <- dm*(y-mu)/variance;iw <- dm^2/variance
            }
            score <- score+as.numeric(crossprod(xx,w*u))
            information <- information+as.matrix(crossprod(xx,xx*(w*iw)))
        }
        if(theta.derivatives) {
            # Differentiate the analytic theta score within the current block,
            # so both derivatives require one read of the observations.
            h <- .Machine$double.eps^(1/3)
            ss <- theta*.count_backend_theta_score(y,mu,theta,truncated)
            plus <- theta*exp(h);minus <- theta*exp(-h)
            d <- (plus*.count_backend_theta_score(y,mu,plus,truncated)-
                  minus*.count_backend_theta_score(y,mu,minus,truncated))/(2*h)
            st <- st+sum(w*ss);it <- it-sum(w*d)
        }
    }
    list(loglik=ll,score=score,information=information,theta.score=st,theta.information=it)
}

.count_backend_theta <- function(reader,beta,theta,truncated,link,tol,maxit) {
    lo <- log(1e-6);hi <- log(1e8);z <- max(lo,min(hi,log(theta)))
    current <- .count_backend_count_summary(reader,beta,exp(z),truncated,link,theta.derivatives=TRUE)
    converged <- FALSE
    for(iter in seq_len(maxit)) {
        s <- current$theta.score;info <- current$theta.information
        if(!is.finite(s) || !is.finite(info)) stop("nonfinite theta score or information",call.=FALSE)
        step <- if(info>0) s/info else sign(s)
        step <- max(-2,min(2,step));candidate <- max(lo,min(hi,z+step))
        # The score vanishes throughout the Poisson tail even when it has
        # no finite root. Convergence therefore requires a small log-size
        # step, rather than an absolute score threshold.
        if(abs(candidate-z)<tol) {converged<-TRUE;break}
        accepted <- FALSE
        for(half in 0:25) {
            candidate <- z+(max(lo,min(hi,z+step))-z)*2^(-half)
            nextvalue <- .count_backend_count_summary(reader,beta,exp(candidate),truncated,link,theta.derivatives=TRUE)
            # In the Poisson tail, evaluating the NB mass can lose several
            # digits even though the analytic score remains accurate. A
            # rounding allowance prevents halving a valid step indefinitely.
            rounding <- sqrt(.Machine$double.eps)*(1+abs(current$loglik))
            if(is.finite(nextvalue$loglik) && nextvalue$loglik>=current$loglik-rounding) {accepted<-TRUE;break}
        }
        if(!accepted) break
        change <- abs(candidate-z);z <- candidate;current <- nextvalue
        if(change<tol) {converged<-TRUE;break}
    }
    list(theta=exp(z),converged=converged,iter=iter,boundary=z<=lo+tol || z>=hi-tol,
         loglik=current$loglik)
}

.count_backend_inverse <- function(a) {
    tryCatch({
        rr <- chol((a+t(a))/2)
        if(rcond(a)<1e-12) stop("ill-conditioned")
        chol2inv(rr)
    },error=function(e) stop("count-model information is singular or not positive definite",call.=FALSE))
}

.count_backend_nb_information <- function(reader,beta,theta,link) {
    p <- length(beta);info <- matrix(0,p+1L,p+1L);lnk <- make.link(link)
    for(k in seq_len(reader$source$n_chunks)) {
        z <- reader$fetch(k)
        if(is.matrix(z$X)) {
            info <- info+count_block_nb_information(z$X,z$y,z$weights,z$offset$count,as.numeric(beta),theta,
                match(link,c("log","sqrt","identity"))-1L)
            next
        }
        keep <- z$weights>0
        if(!any(keep)) next
        x <- z$X[keep,,drop=FALSE];y <- z$y[keep];w <- z$weights[keep]
        eta <- as.numeric(x%*%beta)+z$offset$count[keep]
        mu <- lnk$linkinv(eta);dm <- lnk$mu.eta(eta)
        d2 <- switch(link,log=mu,sqrt=rep(2,length(mu)),identity=rep(0,length(mu)))
        variance <- mu+mu^2/theta;vp <- 1+2*mu/theta
        iw <- w*(dm^2/variance+(y-mu)*(dm^2*vp/variance^2-d2/variance))
        cross <- -w*dm*theta*(y-mu)/(theta+mu)^2
        sc <- .count_backend_theta_score(y,mu,theta)
        ds <- trigamma(y+theta)-trigamma(theta)+mu/(theta*(theta+mu))-(mu-y)/(theta+mu)^2
        ii <- -theta*sc-theta^2*ds
        large <- theta>1e4*pmax(1,y,mu)
        if(any(large)) {
            yy<-y[large];mm<-mu[large]
            c2<-(yy-(yy-mm)^2)/2
            c3<-yy*(yy-1)*(2*yy-1)/6+2*mm^3/3-yy*mm^2
            c4<--yy^2*(yy-1)^2/4-3*mm^4/4+yy*mm^3
            ii[large]<-(c2+(2*c3+3*c4/theta)/theta)/theta
        }
        info[seq_len(p),seq_len(p)] <- info[seq_len(p),seq_len(p)]+as.matrix(crossprod(x,x*iw))
        info[seq_len(p),p+1L] <- info[seq_len(p),p+1L]+as.numeric(crossprod(x,cross))
        info[p+1L,p+1L] <- info[p+1L,p+1L]+sum(w*ii)
    }
    info[p+1L,seq_len(p)] <- info[seq_len(p),p+1L]
    nm<-c(names(beta),"log(theta)");dimnames(info)<-list(nm,nm)
    info
}

.count_backend_object <- function(reader,beta,theta=NULL,truncated=FALSE,link="log",theta.estimated=!is.null(theta)) {
    names(beta) <- reader$columns$count
    object <- list(coefficients=beta,theta=theta,theta.estimated=theta.estimated,
        dist=if(is.null(theta)) "poisson" else "negbin",
        family=if(is.null(theta)) poisson(link) else negbin(theta,link),
        source=reader$source,nobs=reader$state$nobs)
    object$fetch <- function(k) {
        z <- reader$fetch(k)
        if(truncated) {
            keep <- z$y>0
            z$X <- z$X[keep,,drop=FALSE];z$y <- z$y[keep];z$weights <- z$weights[keep]
            z$offset <- lapply(z$offset,`[`,keep)
        }
        z
    }
    class(object) <- if(truncated) "fastglm_truncated" else "fastglm_nb"
    object
}

.count_backend_joint <- function(reader,beta,theta=NULL,truncated=FALSE,link="log",theta.estimated=!is.null(theta)) {
    object <- .count_backend_object(reader,beta,theta,truncated,link,theta.estimated)
    tryCatch(.count_natural_covariance(
        if(!truncated && theta.estimated) .count_inverse_information(.count_backend_nb_information(reader,beta,theta,link))
        else .count_information(object)$inverse,object),
        error=function(e) {
            if(!grepl("^joint information is singular or not positive definite",conditionMessage(e))) stop(e)
            warning("joint count-model covariance is unavailable: ",conditionMessage(e),call.=FALSE)
            p <- length(beta)+as.integer(theta.estimated)
            matrix(NA_real_,p,p)
        })
}

# sum(w ((y - mu)^2 - y)) and sum(w mu^2) at a Poisson fit, in one pass. A
# separate function, so that no source block stays in the environment of
# the fit's fetch closure.
.count_backend_overdispersion <- function(reader,beta,link) {
    lnk <- make.link(link);excess <- musq <- 0
    for(k in seq_len(reader$source$n_chunks)) {
        a <- reader$fetch(k);keep <- a$weights>0
        mu <- lnk$linkinv(as.numeric(a$X%*%beta)+a$offset$count)[keep]
        excess <- excess+sum(a$weights[keep]*((a$y[keep]-mu)^2-a$y[keep]))
        musq <- musq+sum(a$weights[keep]*mu^2)
    }
    c(excess=excess,musq=musq)
}

# Joint score, Newton step, and Newton decrement of the zero-truncated NB2
# log likelihood in (beta, log theta), with the observed joint information
# from .count_information(). At the theta cap, when the theta score points
# beyond it, only the coefficient block is used.
.count_backend_trunc_newton <- function(reader,beta,theta) {
    v <- .count_backend_count_summary(reader,beta,theta,TRUE,"log",system=TRUE,theta.derivatives=TRUE)
    if(!is.finite(v$loglik)) return(list(decrement=Inf,loglik=v$loglik,step=NULL))
    g <- c(v$score,v$theta.score)
    object <- .count_backend_object(reader,beta,theta,TRUE,"log")
    info <- tryCatch(.count_information(object)$information,error=function(e) NULL)
    if(is.null(info)) return(list(decrement=Inf,loglik=v$loglik,step=NULL))
    p <- length(beta)
    at.cap <- (theta>=1e8*(1-1e-12) && g[p+1L]>0) || (theta<=1e-6*(1+1e-12) && g[p+1L]<0)
    m <- if(at.cap) p else p+1L
    step <- tryCatch(solve(info[seq_len(m),seq_len(m),drop=FALSE],g[seq_len(m)]),error=function(e) NULL)
    if(is.null(step)) return(list(decrement=Inf,loglik=v$loglik,step=NULL))
    if(at.cap) step <- c(step,0)
    list(decrement=sum(g[seq_len(m)]*step[seq_len(m)]),loglik=v$loglik,step=step)
}

.count_backend_nb_decrement <- function(reader,beta,theta,link,boundary=FALSE,flat.tol=0) {
    v <- .count_backend_count_summary(reader,beta,theta,FALSE,link,system=TRUE,theta.derivatives=TRUE)
    if(!is.finite(v$loglik)) return(Inf)
    # beta was just fitted to convergence at this theta. When that fit held
    # rows near a bound of the sqrt or identity link, the score is not zero
    # at its (constrained) optimum, and the fit's own convergence certifies
    # the coefficient block; only the theta block remains.
    lb <- if(boundary) 0 else tryCatch(sum(v$score*solve(v$information,v$score)),error=function(e) Inf)
    st <- v$theta.score;it <- v$theta.information
    # Where the log likelihood is flat in log(theta) (large theta) the theta
    # information need not be positive; a theta score below flat.tol means
    # that changing log(theta) by one changes the deviance negligibly.
    lt <- if(it>0) st^2/it else if(abs(st)<=flat.tol) 0 else Inf
    if((theta>=1e8*(1-1e-12) && st>0) || (theta<=1e-6*(1+1e-12) && st<0)) lt <- 0
    lb+lt
}

.count_backend_nb <- function(x,y=NULL,weights=NULL,offset=NULL,start=NULL,init.theta=NULL,
                              link="log",method=2L,tol=1e-8,maxit=100L,outer.maxit=50L,
                              outer.tol=1e-7,theta.tol=1e-8,theta.maxit=100L,block.size=8192L,
                              poisson.limit=TRUE) {
    reader <- .count_backend_reader(x,y,weights=weights,offset=offset,block.size=block.size)
    .check_iterations(tol,maxit);.check_iterations(outer.tol,outer.maxit);.check_iterations(theta.tol,theta.maxit)
    if(!is.null(init.theta) && (length(init.theta)!=1L || !is.finite(init.theta) || init.theta<=0))
        stop("init.theta must be a positive finite scalar",call.=FALSE)
    method <- .solver_method(method)
    if(!method %in% 2:3) stop("sparse and streaming NB fits require LLT or LDLT",call.=FALSE)
    fit <- .count_backend_glm(reader,"count",poisson(link),start=start,method=method,tol=tol,maxit=maxit)
    pilot <- fit
    beta <- coef(fit);theta <- if(is.null(init.theta)) 1 else init.theta
    converged <- FALSE;theta.iter <- 0L;decrement <- Inf
    step <- function() {
        th <- .count_backend_theta(reader,beta,theta,FALSE,link,theta.tol,theta.maxit)
        theta <<- th$theta;theta.iter <<- theta.iter+th$iter
        fit <<- .count_backend_glm(reader,"count",negbin(theta,link),start=beta,method=method,tol=tol,maxit=maxit)
        beta <<- coef(fit)
        th
    }
    for(iter in seq_len(outer.maxit)) {
        th <- step()
        # Joint Fisher decrement at (beta, theta), as in the dense driver
        # (src/fit_glm_nb.cpp): the coefficient block g' I^{-1} g plus the
        # theta block s^2 / I_theta (on the log scale, which leaves it
        # unchanged), omitted at the cap when the score points beyond it.
        decrement <- .count_backend_nb_decrement(reader,beta,theta,link,
                                                 identical(fit$convergence.reason,"converged_boundary"),
                                                 outer.tol*(0.1+abs(fit$deviance)))
        if(fit$converged && th$converged && decrement/(0.1+abs(fit$deviance))<outer.tol) {
            # Two polishing rounds, as in the dense driver.
            for(polish in 1:2) th <- step()
            converged <- fit$converged && th$converged
            break
        }
    }
    final <- .count_backend_count_summary(reader,beta,theta,FALSE,link)
    cn <- reader$state$columns$count
    if(is.null(cn)) cn <- paste0("x",seq_along(beta))
    names(beta) <- cn;reader$columns$count <- cn
    names(fit$coefficients) <- names(fit$se) <- cn
    dimnames(fit$cov.unscaled) <- list(cn,cn)
    fit$theta <- theta;fit$theta.estimated <- TRUE;fit$twologlik <- 2*final$loglik
    fit$converged <- converged;fit$iter.theta <- iter;fit$theta.iterations <- theta.iter
    # The coefficient fit's reason describes only the last inner fit; when the
    # outer (beta, theta) loop did not converge, report that instead.
    if(!converged) fit$convergence.reason <- "iteration_limit"
    fit$theta.boundary <- theta<=1e-6*(1+theta.tol) || theta>=1e8/(1+theta.tol)
    if(fit$theta.boundary) {
        joint <- matrix(NA_real_,length(beta)+1L,length(beta)+1L)
        joint[seq_along(beta),seq_along(beta)] <- fit$cov.unscaled
    } else joint <- .count_backend_joint(reader,beta,theta,FALSE,link)
    dimnames(joint) <- rep(list(c(names(beta),"theta")),2L)
    fit$vcov.full <- joint;fit$SE.theta <- sqrt(joint[length(beta)+1L,length(beta)+1L])
    fit$outer.decrement <- decrement
    fit$aic <- -2*final$loglik+2*(length(beta)+1L)
    fit$refit.controls <- list(link=link,method=method,tol=tol,maxit=maxit,outer.maxit=outer.maxit,
        outer.tol=outer.tol,theta.tol=theta.tol,theta.maxit=theta.maxit,block.size=block.size)
    if(reader$streaming) {
        fit$source <- x;fit$state <- reader$state
        fit$fetch <- function(k) {a<-reader$fetch(k);list(X=a$X,y=a$y,weights=a$weights,offset=a$offset$count)}
        fit$passes <- reader$state$passes;fit$chunk.reads <- reader$state$reads
    }
    class(fit) <- c("fastglm_nb",if(reader$streaming) "fastglm_stream","fastglm")
    if(poisson.limit) {
        # The theta = Inf limit (see ?fastglm_nb), with the Poisson pilot fit.
        names(pilot$coefficients) <- names(pilot$se) <- cn;dimnames(pilot$cov.unscaled) <- list(cn,cn)
        pois.ll <- .count_backend_count_summary(reader,coef(pilot),NULL,FALSE,link)$loglik
        od <- .count_backend_overdispersion(reader,coef(pilot),link)
        refit <- function(theta0) tryCatch(suppressWarnings(.count_backend_nb(x,y,weights,offset,unname(coef(pilot)),
            theta0,link,method,tol,maxit,outer.maxit,outer.tol,theta.tol,theta.maxit,block.size,FALSE)),
            error=function(e) NULL)
        fit <- .nb_poisson_limit(fit,pilot,pois.ll,od[["excess"]],od[["musq"]],link,refit)
        converged <- isTRUE(fit$converged)
    }
    if(!converged) warning("negative-binomial fit did not converge",call.=FALSE)
    fit
}

.count_backend_trunc_beta <- function(reader,theta=NULL,start=NULL,tol=1e-8,maxit=100L) {
    p <- reader$state$p[1L]
    if(is.null(start)) {
        a <- matrix(0,p,p);rhs <- numeric(p)
        for(k in seq_len(reader$source$n_chunks)) {
            z <- reader$fetch(k);keep <- z$y>0 & z$weights>0
            if(!any(keep)) next
            xx <- z$X[keep,,drop=FALSE];w <- z$weights[keep]
            a <- a+as.matrix(crossprod(xx,w*xx))
            rhs <- rhs+as.numeric(crossprod(xx,w*(log(pmax(z$y[keep]-.5,.1))-z$offset$count[keep])))
        }
        start <- drop(.count_backend_inverse(a)%*%rhs)
    }
    if(length(start)!=p || any(!is.finite(start))) stop("invalid truncated-count starting coefficients",call.=FALSE)
    beta <- as.numeric(start)
    current <- .count_backend_count_summary(reader,beta,theta,TRUE,"log",system=TRUE)
    converged <- FALSE
    for(iter in seq_len(maxit)) {
        if(!is.finite(current$loglik)) stop("nonfinite truncated-count likelihood",call.=FALSE)
        step <- drop(.count_backend_inverse(current$information)%*%current$score)
        if(max(abs(step))<tol*(1+max(abs(beta)))) {converged<-TRUE;break}
        accepted <- FALSE
        for(half in 0:30) {
            candidate <- beta+step*2^(-half)
            nextvalue <- .count_backend_count_summary(reader,candidate,theta,TRUE,"log",system=TRUE)
            if(is.finite(nextvalue$loglik) && nextvalue$loglik>=current$loglik-1e-12*(1+abs(current$loglik))) {accepted<-TRUE;break}
        }
        if(!accepted) break
        change <- max(abs(candidate-beta));beta <- candidate;current <- nextvalue
        # A small change certifies convergence only for a full Newton step; a
        # step halved toward zero changes little wherever the iteration stalls.
        if(half==0L && change<tol*(1+max(abs(beta)))) {converged<-TRUE;break}
    }
    list(coefficients=beta,loglik=current$loglik,converged=converged,iter=iter)
}

.count_backend_truncated <- function(reader,dist,tol,maxit,init.theta=NULL,
                                     outer.tol=1e-7,outer.maxit=50L,theta.tol=1e-8,theta.maxit=100L) {
    .check_iterations(tol,maxit);.check_iterations(outer.tol,outer.maxit);.check_iterations(theta.tol,theta.maxit)
    if(!is.null(init.theta) && (length(init.theta)!=1L || !is.finite(init.theta) || init.theta<=0))
        stop("init.theta must be a positive finite scalar",call.=FALSE)
    fit <- .count_backend_trunc_beta(reader,tol=tol,maxit=maxit)
    theta <- NULL;outer <- theta.iter <- 0L
    if(dist=="negbin") {
        theta <- if(is.null(init.theta)) 1 else init.theta
        converged <- FALSE;decrement <- Inf
        for(outer in seq_len(outer.maxit)) {
            th <- .count_backend_theta(reader,fit$coefficients,theta,TRUE,"log",theta.tol,theta.maxit)
            theta <- th$theta;theta.iter <- theta.iter+th$iter
            fit <- .count_backend_trunc_beta(reader,theta,fit$coefficients,tol,maxit)
            # A joint Newton step in (beta, log theta) after each alternation
            # round, then the joint Newton decrement and up to two polishing
            # steps, as in the dense truncated loop (inst/include/trunc_count.h):
            # under truncation beta and theta are coupled, and alternation
            # alone can converge too slowly.
            beta <- fit$coefficients
            jd <- .count_backend_trunc_newton(reader,beta,theta);ll <- jd$loglik
            newton <- function() {
                if(is.null(jd$step)) return(FALSE)
                for(half in 0:29) {
                    fr <- 2^(-half)
                    b <- beta+fr*jd$step[seq_along(beta)]
                    t <- exp(min(log(1e8),max(log(1e-6),log(theta)+fr*jd$step[length(beta)+1L])))
                    lc <- .count_backend_count_summary(reader,b,t,TRUE,"log")$loglik
                    if(is.finite(lc) && lc>=ll-64*.Machine$double.eps*(1+abs(ll))) {
                        beta <<- b;theta <<- t;ll <<- lc
                        jd <<- .count_backend_trunc_newton(reader,beta,theta)
                        return(TRUE)
                    }
                }
                FALSE
            }
            newton()
            decrement <- jd$decrement
            if(fit$converged && th$converged && decrement/(0.1+abs(2*ll))<outer.tol) {
                for(polish in 1:2) if(!newton()) break
                fit <- .count_backend_trunc_beta(reader,theta,beta,tol,maxit)
                converged <- fit$converged
                break
            }
            fit$coefficients <- beta
        }
        fit$converged <- converged;fit$outer.decrement <- decrement
    }
    fit$theta <- theta;fit$theta.estimated <- dist=="negbin"
    fit$theta.boundary <- !is.null(theta) && (theta<=1e-6*(1+theta.tol) || theta>=1e8/(1+theta.tol))
    if(fit$theta.boundary) {
        # At a size boundary only the conditional coefficient covariance is retained.
        beta.v <- .count_backend_joint(reader,fit$coefficients,theta,TRUE,theta.estimated=FALSE)
        full <- matrix(NA_real_,length(fit$coefficients)+1L,length(fit$coefficients)+1L)
        full[seq_along(fit$coefficients),seq_along(fit$coefficients)] <- beta.v
        fit$vcov.full <- full
    } else fit$vcov.full <- .count_backend_joint(reader,fit$coefficients,theta,TRUE)
    fit$outer.iter <- outer;fit$theta.iter <- theta.iter
    fit
}

.count_backend_schema <- function(reader) {
    schema <- reader$source$schema
    if(is.null(schema)) return(list(terms=NULL,levels=NULL,contrasts=NULL,formula=NULL,assign=NULL))
    if(inherits(schema$terms,"terms")) {
        terms <- list(count=schema$terms,zero=schema$terms)
        contrasts <- list(count=schema$contrasts,zero=schema$contrasts)
        levels <- schema$xlevels
        assignment <- list(count=schema$assign,zero=schema$assign)
    } else {
        terms <- schema$terms;contrasts <- schema$contrasts
        levels <- c(schema$xlevels$count,schema$xlevels$zero)
        levels <- levels[!duplicated(names(levels))]
        assignment <- schema$assign
    }
    list(terms=terms,levels=levels,contrasts=contrasts,formula=schema$formula,assign=assignment)
}

.count_backend_model <- function(reader,positive,binary,dist,link,controls,continuous=FALSE,firth=FALSE) {
    bc <- coef(positive);if(is.null(bc)) bc <- positive$coefficients
    bz <- coef(binary);pc <- length(bc);pz <- length(bz);pp <- pc+pz
    if(any(!is.finite(c(bc,bz)))) stop("two-component fits require full-rank identifiable designs",call.=FALSE)
    names(bc) <- reader$columns$count;names(bz) <- reader$columns$zero
    cn <- c(paste0(if(continuous) "positive_" else "count_",names(bc)),paste0("zero_",names(bz)))
    vc <- if(continuous) vcov(positive) else positive$vcov.full[seq_len(pc),seq_len(pc),drop=FALSE]
    vz <- vcov(binary);v <- matrix(0,pp,pp,dimnames=list(cn,cn))
    v[seq_len(pc),seq_len(pc)] <- vc;v[pc+seq_len(pz),pc+seq_len(pz)] <- vz
    llc <- if(continuous) as.numeric(logLik(positive)) else positive$loglik
    llz <- as.numeric(logLik(binary));theta <- positive$theta
    out <- c(list(coefficients=list(count=bc,zero=bz),se=list(count=sqrt(diag(vc)),zero=sqrt(diag(vz))),
        vcov=v,vcov.full=v,positive=positive,binary=binary,loglik=llc+llz,
        loglik_count=llc,loglik_zero=llz,converged=positive$converged && binary$converged,
        dist=dist,link=link,zero.dist="binomial",theta=theta,theta.estimated=dist=="negbin",
        theta.boundary=isTRUE(positive$theta.boundary),firth=firth,
        nobs=reader$state$nobs,n=sum(reader$state$rows),n_positive=reader$state$npositive,rank=pp,
        iter_count=positive$iter,iter_zero=binary$iter,outer_iter=positive$outer.iter,
        theta_iter=positive$theta.iter,refit.controls=controls,
        df.residual=reader$state$nobs-pp-as.integer(continuous || dist=="negbin")),
        .count_backend_schema(reader))
    if(dist=="negbin") {
        full <- matrix(0,pp+1L,pp+1L,dimnames=rep(list(c(cn,"theta")),2L))
        full[seq_len(pp),seq_len(pp)] <- v
        j <- c(seq_len(pc),pp+1L);full[j,j] <- positive$vcov.full
        out$vcov.full <- full;out$SE.theta <- sqrt(full[pp+1L,pp+1L])
    }
    if(reader$streaming) {
        out$source <- reader$source;out$fetch <- reader$fetch;out$state <- reader$state
        out$passes <- reader$state$passes;out$chunk.reads <- reader$state$reads
    } else out <- c(out,reader$original)
    class(out) <- if(continuous) c(if(reader$streaming) "fastglm_twopart_stream","fastglm_twopart",if(reader$streaming) "fastglm_hurdle_stream","fastglm_hurdle") else
        c(if(reader$streaming) "fastglm_hurdle_stream","fastglm_hurdle")
    if(!out$converged) warning("two-component model did not converge",call.=FALSE)
    out
}

.count_backend_hurdle <- function(x,response=NULL,z=NULL,weights=NULL,offset=NULL,
                                  dist="poisson",link="logit",init.theta=NULL,method=2L,
                                  tol=1e-8,maxit=100L,outer.tol=1e-7,outer.maxit=50L,
                                  theta.tol=1e-8,theta.maxit=100L,block.size=8192L) {
    reader <- .count_backend_reader(x,response,z,weights,offset,block.size)
    if(reader$state$npositive<=reader$state$p[1L] || reader$state$npositive==reader$state$nobs)
        stop("hurdle models require zeros and enough positive observations for the count design",call.=FALSE)
    if(reader$state$nobs<=reader$state$p[2L]) stop("not enough observations for the binary design",call.=FALSE)
    # The log link needs a valid start (zero coefficients give probability one).
    bstart <- if(link=="log") .count_backend_log_start(reader,"positive",all.rows=TRUE)
    binary <- .count_backend_glm(reader,"zero",binomial(link),binary=TRUE,start=bstart,
                                 method=method,tol=tol,maxit=maxit)
    positive <- .count_backend_truncated(reader,dist,tol,maxit,init.theta,outer.tol,outer.maxit,theta.tol,theta.maxit)
    controls <- list(dist=dist,link=link,method=method,tol=tol,maxit=maxit,outer.tol=outer.tol,
        outer.maxit=outer.maxit,theta.tol=theta.tol,theta.maxit=theta.maxit,block.size=block.size)
    .count_backend_model(reader,positive,binary,dist,link,controls)
}

.count_backend_twopart <- function(x,response=NULL,z=NULL,weights=NULL,offset=NULL,
                                   link="logit",method="auto",tol=1e-8,maxit=100L,firth=FALSE,block.size=8192L) {
    reader <- .count_backend_reader(x,response,z,weights,offset,block.size,counts=FALSE)
    if(reader$state$npositive<=reader$state$p[1L] || reader$state$npositive==reader$state$nobs)
        stop("two-part models require zeros and enough positive observations for the positive design",call.=FALSE)
    if(reader$state$nobs<=reader$state$p[2L]) stop("not enough observations for the binary design",call.=FALSE)
    bstart <- if(identical(link,"log")) .count_backend_log_start(reader,"positive",all.rows=TRUE)
    binary <- .count_backend_glm(reader,"zero",binomial(link),binary=TRUE,start=bstart,
                                 method=method,tol=tol,maxit=maxit,firth=firth)
    positive <- .count_backend_glm(reader,"count",Gamma("log"),positive=TRUE,method=method,tol=tol,maxit=maxit,firth=firth)
    controls <- list(link=link,method=method,tol=tol,maxit=maxit,firth=firth,block.size=block.size)
    .count_backend_model(reader,positive,binary,"Gamma",link,controls,continuous=TRUE,firth=firth)
}

# Zero-truncated regression from sparse, memory-mapped, or streamed designs,
# using the block-reading count fitter of hurdle models.
.count_backend_truncated_model <- function(x,y,weights,offset,dist,theta,tol,maxit,
                                           outer.tol,outer.maxit,block.size=8192L) {
    reader <- .count_backend_reader(x,y,weights=weights,offset=offset,block.size=block.size)
    if(reader$state$npositive!=reader$state$nobs)
        stop("truncated counts must be positive integers",call.=FALSE)
    if(!is.null(theta) && (length(theta)!=1L || !is.finite(theta) || theta<=0))
        stop("'theta' must be positive",call.=FALSE)
    estimated <- dist=="negbin" && is.null(theta)
    if(estimated) {
        fit <- .count_backend_truncated(reader,dist,tol,maxit,NULL,outer.tol,outer.maxit)
    } else {
        fixed <- if(dist=="negbin") theta else NULL
        fit <- .count_backend_trunc_beta(reader,fixed,tol=tol,maxit=maxit)
        fit$theta.boundary <- FALSE
        fit$vcov.full <- .count_backend_joint(reader,fit$coefficients,fixed,TRUE,theta.estimated=FALSE)
    }
    # Exact indexing: for Poisson there is no theta, and fit$theta would
    # partially match fit$theta.boundary.
    theta.value <- if(estimated) fit[["theta"]] else if(dist=="negbin") theta else NULL
    cn <- reader$columns$count
    beta <- fit$coefficients; names(beta) <- cn
    rn <- c(cn,if(estimated) "theta")
    vfull <- fit$vcov.full; dimnames(vfull) <- list(rn,rn)
    # Positional indexing: design column names can be empty (for example the
    # unnamed first column of cbind(1, x1, x2)), which name indexing rejects.
    jc <- seq_along(cn)
    v <- vfull[jc,jc,drop=FALSE]
    loglik <- .count_backend_count_summary(reader,beta,theta.value,TRUE,"log")$loglik
    out <- list(coefficients=beta,se=sqrt(diag(v)),cov.unscaled=v,vcov.full=vfull,
        loglik=loglik,theta=theta.value,
        SE.theta=if(estimated) sqrt(vfull[length(rn),length(rn)]) else NULL,
        converged=isTRUE(fit[["converged"]]),
        convergence.reason=if(isTRUE(fit[["converged"]])) "converged" else "iteration_limit",
        iter=fit[["iter"]],outer.iter=fit[["outer.iter"]],outer.decrement=fit[["outer.decrement"]],
        dist=dist,theta.estimated=estimated,theta.boundary=isTRUE(fit[["theta.boundary"]]),
        rank=length(cn),nobs=reader$state$nobs,df.residual=reader$state$nobs-length(rn),
        dispersion=1,aic=-2*loglik+2*length(rn))
    if(reader$streaming) {
        out$source <- reader$source; out$state <- reader$state
        out$fetch <- function(k) {a<-reader$fetch(k);list(X=a$X,y=a$y,weights=a$weights,offset=a$offset$count)}
        out$passes <- reader$state$passes; out$chunk.reads <- reader$state$reads
    } else {
        o <- reader$original
        out$x <- o$x$count; out$y <- o$y; out$prior.weights <- o$weights; out$offset <- o$offset$count
    }
    class(out) <- "fastglm_truncated"
    if(out$theta.boundary) { out$SE.theta <- NA_real_; out$vcov.full[,] <- NA_real_
        out$vcov.full[jc,jc] <- v }
    if(!out$converged) warning("truncated-count model did not converge",call.=FALSE)
    out
}

# Zero-inflated Poisson and negative binomial regression from sparse,
# memory-mapped, or streamed designs. Each pass over the blocks either
# evaluates the observed log likelihood or accumulates one EM (expectation
# conditional maximization) step: the E-step posteriors and, in the same pass,
# the complete-data scoring steps for the inflation and count coefficients and
# log theta. EM is followed by Newton steps with the joint observed
# information (.count_score() and .count_information_block()), and the fit is
# reported as converged when the Newton decrement is below em.tol relative to
# the log likelihood and the information is positive definite, the rule of
# the dense fitter (src/fit_glm_zi.cpp).
.zi_block_parts <- function(z,beta,gamma,theta,dist,link) {
    keep <- z$weights>0
    if(!any(keep)) return(NULL)
    x <- as.matrix(z$X[keep,,drop=FALSE]);zz <- as.matrix(z$Z[keep,,drop=FALSE])
    y <- z$y[keep];w <- z$weights[keep]
    mu <- exp(drop(x%*%beta)+z$offset$count[keep])
    lk <- make.link(link);eta0 <- drop(zz%*%gamma)+z$offset$zero[keep]
    pi <- lk$linkinv(eta0)
    nb <- dist=="negbin"
    lp0 <- if(nb) -theta*log1p(mu/theta) else -mu
    list(x=x,zz=zz,y=y,w=w,mu=mu,eta0=eta0,pi=pi,lp0=lp0,nb=nb,lk=lk)
}
# Negative binomial log probability, with lgamma(y + theta) - lgamma(theta)
# expanded in 1/theta when theta is large, where dnbinom() loses precision.
.nb_log_density <- function(y,mu,theta) {
    out <- dnbinom(y,mu=mu,size=theta,log=TRUE)
    big <- theta>1e5 & y<1e-3*theta
    if(any(big)) {
        yy <- y[big];m <- mu[big]
        # sum_{j<y} log1p(j/theta) to third order in 1/theta.
        s1 <- yy*(yy-1)/2;s2 <- (yy-1)*yy*(2*yy-1)/6;s3 <- s1^2
        out[big] <- s1/theta-s2/(2*theta^2)+s3/(3*theta^3)-lgamma(yy+1)-
            (theta+yy)*log1p(m/theta)+yy*log(m)
    }
    out
}
.zi_block_loglik <- function(z,beta,gamma,theta,dist,link) {
    a <- .zi_block_parts(z,beta,gamma,theta,dist,link)
    if(is.null(a)) return(0)
    if(any(!is.finite(a$mu)) || any(!is.finite(a$pi)) || any(a$pi<0 | a$pi>1)) return(-Inf)
    zero <- a$y==0;l <- numeric(length(a$y))
    l[zero] <- log(a$pi[zero]+(1-a$pi[zero])*exp(a$lp0[zero]))
    ld <- if(a$nb) .nb_log_density(a$y[!zero],a$mu[!zero],theta) else dpois(a$y[!zero],a$mu[!zero],log=TRUE)
    l[!zero] <- log1p(-a$pi[!zero])+ld
    sum(a$w*l)
}
.zi_block_em <- function(z,beta,gamma,theta,dist,link,acc) {
    a <- .zi_block_parts(z,beta,gamma,theta,dist,link)
    if(is.null(a)) return(acc)
    zero <- a$y==0;tau <- numeric(length(a$y))
    tau[zero] <- a$pi[zero]/(a$pi[zero]+(1-a$pi[zero])*exp(a$lp0[zero]))
    v <- a$w*(1-tau)
    dpi <- a$lk$mu.eta(a$eta0);den <- pmax(a$pi*(1-a$pi),.Machine$double.xmin)
    acc$sg <- acc$sg+drop(crossprod(a$zz,a$w*(tau-a$pi)*dpi/den))
    acc$ig <- acc$ig+crossprod(a$zz,a$zz*(a$w*dpi^2/den))
    shrink <- if(a$nb) 1+a$mu/theta else 1
    acc$sb <- acc$sb+drop(crossprod(a$x,v*(a$y-a$mu)/shrink))
    acc$ib <- acc$ib+crossprod(a$x,a$x*(v*a$mu/shrink))
    if(a$nb) {
        # Score and curvature in log theta; the curvature by a centered
        # difference of the analytic score within the same pass.
        h <- .Machine$double.eps^(1/3)
        st <- function(t) t*.count_backend_theta_score(a$y,a$mu,t)
        acc$st <- acc$st+sum(v*st(theta))
        acc$it <- acc$it-sum(v*(st(theta*exp(h))-st(theta*exp(-h))))/(2*h)
    }
    acc
}
# Starting coefficients for a binary component with the log link, where
# zero coefficients (eta = offset, a probability of one at a zero offset)
# are not valid. With an intercept column (one nonzero value c in every row
# of every block), the intercept gives
#   eta = log(p / 2) + offset - max(offset) < 0,
# where p is the weighted fraction of zero responses (event = "zero", the
# inflation model of zero-inflated fits, as the dense fitter,
# fglm::zi::zero_model_start in inst/include/zi.h) or of positive responses
# (event = "positive", the binary part of hurdle and two-part fits). The
# maximum offset is over the rows with positive weight, or over all rows when
# all.rows is TRUE (fastglm() checks the fitted probabilities of every row).
# Without an intercept column the start stays zero.
.count_backend_log_start <- function(reader,event=c("zero","positive"),all.rows=FALSE) {
    event <- match.arg(event)
    pz <- reader$state$p[2L];const <- NULL;value <- rep(NA_real_,pz)
    sw <- sz <- 0;maxoff <- -Inf
    for(k in seq_len(reader$source$n_chunks)) {
        a <- reader$fetch(k);zz <- a$Z
        lo <- apply(zz,2L,min);hi <- apply(zz,2L,max)
        same <- lo==hi & lo!=0
        if(is.null(const)) {const <- same;value <- lo} else const <- const & same & lo==value
        hit <- if(event=="zero") a$y==0 else a$y>0
        keep <- if(all.rows) rep(TRUE,length(a$y)) else a$weights>0
        sw <- sw+sum(a$weights);sz <- sz+sum(a$weights*hit)
        if(any(keep)) maxoff <- max(maxoff,a$offset$zero[keep])
    }
    gamma <- numeric(pz);j <- which(const)[1L]
    if(!is.na(j)) {
        p <- min(1-1e-4,max(1e-4,if(sw>0) sz/sw else .5))
        gamma[j] <- (log(p/2)-if(is.finite(maxoff)) maxoff else 0)/value[j]
    }
    gamma
}
.count_backend_zi <- function(x,response=NULL,z=NULL,weights=NULL,offset=NULL,dist="poisson",
                              link="logit",init.theta=NULL,em.tol=1e-8,em.maxit=100L,
                              tol=1e-9,maxit=100L,theta.tol=1e-8,theta.maxit=100L,block.size=8192L) {
    .check_iterations(em.tol,em.maxit);.check_iterations(tol,maxit)
    reader <- .count_backend_reader(x,response,z,weights,offset,block.size)
    if(reader$state$npositive==reader$state$nobs)
        stop("zero-inflated models require zero responses",call.=FALSE)
    if(!is.null(init.theta) && (length(init.theta)!=1L || !is.finite(init.theta) || init.theta<=0))
        stop("'init.theta' must be a positive finite scalar.",call.=FALSE)
    nb <- dist=="negbin";cn <- reader$columns$count;zn <- reader$columns$zero
    pc <- length(cn);pz <- length(zn);cap <- 1e8
    blocks <- function(fun,init) {acc <- init;for(k in seq_len(reader$source$n_chunks)) acc <- fun(reader$fetch(k),acc);acc}
    loglik <- function(b,g,t) blocks(function(a,s) s+.zi_block_loglik(a,b,g,t,dist,link),0)
    # Starting values: a Poisson fit of the count design, an inflation
    # intercept at the fitted zero excess, and theta = 1.
    pilot <- .count_backend_glm(reader,"count",poisson(),method=2L,tol=1e-6,maxit=50L)
    beta <- unname(coef(pilot));if(any(!is.finite(beta))) beta[!is.finite(beta)] <- 0
    gamma <- if(link=="log") .count_backend_log_start(reader,"zero") else numeric(pz)
    theta <- if(!nb) NULL else if(is.null(init.theta)) 1 else init.theta
    ll <- loglik(beta,gamma,theta)
    # EM (one scoring step per conditional maximization), with step halving
    # on the observed log likelihood.
    em.iter <- 0L
    for(iter in seq_len(em.maxit)) {
        acc <- blocks(function(a,s) .zi_block_em(a,beta,gamma,theta,dist,link,s),
            list(sg=numeric(pz),ig=matrix(0,pz,pz),sb=numeric(pc),ib=matrix(0,pc,pc),st=0,it=0))
        dg <- tryCatch(solve(acc$ig,acc$sg),error=function(e) NULL)
        db <- tryCatch(solve(acc$ib,acc$sb),error=function(e) NULL)
        if(is.null(dg) || is.null(db)) break
        dt <- if(nb && acc$it>0) acc$st/acc$it else 0
        accepted <- FALSE
        for(half in 0:30) {
            f <- 2^(-half)
            t1 <- if(nb) exp(min(log(cap),max(log(1e-6),log(theta)+f*dt))) else NULL
            l1 <- loglik(beta+f*db,gamma+f*dg,t1)
            if(is.finite(l1) && l1>=ll-64*.Machine$double.eps*(1+abs(ll))) {
                beta <- beta+f*db;gamma <- gamma+f*dg;theta <- t1;accepted <- TRUE;break
            }
        }
        em.iter <- iter
        if(!accepted) break
        change <- l1-ll;ll <- l1
        if(abs(change)<1e-6*(0.1+abs(2*ll))) break
    }
    # Newton steps with the joint observed information.
    names(beta) <- cn;names(gamma) <- zn
    template <- function(fixed.theta) structure(list(coefficients=list(count=beta,zero=gamma),dist=dist,
        link=link,theta=if(nb) theta else NULL,theta.estimated=nb && !fixed.theta),class="fastglm_zi")
    boundary <- FALSE;newton.iter <- 0L;decrement <- Inf;converged <- FALSE;info <- NULL
    certified <- 0L;rounding <- function(l) 64*.Machine$double.eps*(1+abs(l))
    for(k in seq_len(maxit)) {
        if(nb && !boundary && theta>=cap*(1-1e-8)) {
            # theta at its cap: keep it there if its score points beyond
            # (the zero-inflated Poisson limit), as in the dense fitter.
            obj <- template(FALSE);par <- c(beta,gamma,log(theta))
            u <- blocks(function(a,s) s+colSums(.count_score(obj,.count_normalize_block(a,TRUE),par)),numeric(length(par)))
            if(u[length(u)]>=0) boundary <- TRUE
        }
        obj <- template(boundary);par <- c(beta,gamma,if(nb && !boundary) log(theta))
        u <- blocks(function(a,s) s+colSums(.count_score(obj,.count_normalize_block(a,TRUE),par)),numeric(length(par)))
        info <- blocks(function(a,s) s+.count_information_block(obj,.count_normalize_block(a,TRUE),par),
            matrix(0,length(par),length(par)))
        r <- tryCatch(chol(info),error=function(e) NULL)
        if(is.null(r) || any(!is.finite(info))) break
        step <- drop(chol2inv(r)%*%u);decrement <- sum(u*step)
        if(decrement/(0.1+abs(2*ll))<em.tol) {
            # The maximum can lie in the zero-inflated Poisson limit, where the
            # log-theta profile is flat and increasing: try theta at its cap.
            if(nb && !boundary && theta>1e4 && u[length(u)]>0) {
                l1 <- loglik(beta,gamma,cap)
                if(is.finite(l1) && l1>=ll-rounding(ll)) {theta <- cap;boundary <- TRUE;ll <- l1;certified <- 0L;next}
            }
            # Certified; up to two further Newton steps polish the estimate,
            # because the rule allows a remaining gain of em.tol relative to
            # the log likelihood, which on a flat likelihood is a visible
            # change in the coefficients.
            if(certified>=2L) {converged <- TRUE;break}
            certified <- certified+1L
        }
        accepted <- FALSE
        for(half in 0:30) {
            f <- 2^(-half);p1 <- par+f*step
            b1 <- p1[seq_len(pc)];g1 <- p1[pc+seq_len(pz)]
            t1 <- if(nb) if(boundary) theta else exp(min(log(cap),max(log(1e-6),p1[length(p1)]))) else NULL
            l1 <- loglik(b1,g1,t1)
            if(is.finite(l1) && l1>=ll-64*.Machine$double.eps*(1+abs(ll))) {
                beta[] <- b1;gamma[] <- g1;theta <- t1;ll <- l1;accepted <- TRUE;break
            }
        }
        newton.iter <- k
        if(!accepted) {if(certified>0L) converged <- TRUE;break}
    }
    if(!converged && certified>0L) converged <- TRUE
    obj <- template(boundary)
    p.all <- pc+pz+as.integer(nb)
    rn <- c(paste0("count_",cn),paste0("zero_",zn),if(nb) "theta")
    vfull <- matrix(NA_real_,p.all,p.all,dimnames=list(rn,rn))
    if(converged && !boundary) {
        v <- tryCatch(.count_natural_covariance(.count_inverse_information(info),obj),error=function(e) NULL)
        if(!is.null(v)) vfull[,] <- v
    }
    # At the theta boundary regular mixture covariance is unavailable, as for
    # the dense fit.
    vcov <- vfull[seq_len(pc+pz),seq_len(pc+pz),drop=FALSE]
    se <- sqrt(diag(vcov))
    out <- list(coefficients=list(count=beta,zero=gamma),
        se=list(count=setNames(se[seq_len(pc)],cn),zero=setNames(se[pc+seq_len(pz)],zn)),
        vcov=vcov,vcov.full=vfull,loglik=ll,theta=if(nb) theta else NULL,
        SE.theta=if(nb) sqrt(vfull[p.all,p.all]) else NULL,
        em_iter=em.iter,newton.iter=newton.iter,em.decrement=decrement,converged=converged,
        convergence.reason=if(!converged) "not_converged" else if(boundary) "theta_boundary" else "converged",
        n=sum(reader$state$rows),nobs=reader$state$nobs,df.residual=reader$state$nobs-p.all,
        dist=dist,link=link,theta.estimated=nb,theta.boundary=boundary)
    out <- c(out,.count_backend_schema(reader))
    if(reader$streaming) {
        out$source <- reader$source;out$fetch <- reader$fetch;out$state <- reader$state
        out$passes <- reader$state$passes;out$chunk.reads <- reader$state$reads
    } else {
        o <- reader$original
        out$x <- o$x;out$y <- o$y;out$weights <- o$weights;out$offset <- o$offset
    }
    class(out) <- "fastglm_zi"
    if(!converged) warning("two-component model did not converge",call.=FALSE)
    if(boundary) warning("theta is at its upper bound: the log likelihood is highest in the zero-inflated Poisson limit (theta = Inf); regular mixture covariance is unavailable",call.=FALSE)
    out
}
